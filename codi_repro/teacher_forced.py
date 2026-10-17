"""Teacher-forced verbal-CoT counterparts of the whole-trace ablations.

CODI's teacher pass reads question + BOT + written CoT + EOT and then answers. Here the
written CoT is the GSM8k reference solution in the training format (calculator
annotations stripped, steps joined by ". ", the final step dropped as with
include_last_cot=false), and the same corruptions as for the latent thoughts are applied
at the step level: first step removed, each later step removed, steps shuffled, the next
test problem's CoT, and uniformly random tokens of the same length. With an empty CoT the
pass is identical to the zero-thought decode. Same prompt and masked greedy decoding as
recurrent_controls (question + answer-only instruction, no EOS before BOT).
"""
import argparse
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .analysis import equal, last_number, read, write
from .models import load_organism


@torch.no_grad()
def teacher_rollout(model, questions, cots, max_new=64):
    tok = model.tokenizer
    device = next(model.parameters()).device
    seqs = [tok(q, add_special_tokens=False).input_ids + [tok.bot_id] + tok(c, add_special_tokens=False).input_ids + [tok.eot_id] for q, c in zip(questions, cots)]
    width = max(map(len, seqs))
    ids = torch.tensor([[tok.pad_token_id] * (width - len(s)) + s for s in seqs], device=device)
    mask = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in seqs], device=device)
    out = model.codi(input_ids=ids, attention_mask=mask, use_cache=True, logits_to_keep=1)
    embed = model.codi.get_input_embeddings()
    finished = torch.zeros(len(seqs), dtype=torch.bool, device=device)
    sequences = [[] for _ in seqs]
    for _ in range(max_new):
        nxt = out.logits[:, -1, :model.codi.config.vocab_size - 1].argmax(-1)
        for k in range(len(seqs)):
            if not finished[k]:
                if nxt[k].item() == tok.eos_token_id: finished[k] = True
                else: sequences[k].append(nxt[k].item())
        if finished.all(): break
        mask = torch.cat([mask, torch.ones(len(seqs), 1, dtype=mask.dtype, device=device)], 1)
        out = model.codi(inputs_embeds=embed(nxt[:, None]), attention_mask=mask, past_key_values=out.past_key_values, use_cache=True, logits_to_keep=1)
    return [tok.decode(s, skip_special_tokens=False) for s in sequences]


def cot_text(steps):
    return '. '.join(steps) + '.\n' if steps else ''


def main(args):
    rows = read('artifacts/controls.json')
    steps = [r['steps'] for r in read('artifacts/teacher_cots.json')]
    assert [r['i'] for r in read('artifacts/teacher_cots.json')] == [r['i'] for r in rows]
    suffix = ' Output only the answer and nothing else.'
    questions = [r['question'].strip().replace('  ', ' ') + suffix for r in rows]
    model = load_organism(args)
    tok = model.tokenizer

    g = torch.Generator().manual_seed(11)
    def shuffled(s):
        if len(s) < 2: return s
        while True:
            p = torch.randperm(len(s), generator=g).tolist()
            if p != list(range(len(s))): return [s[j] for j in p]
    vocab = len(tok) - len(tok.get_added_vocab())
    def random_tokens(s):
        n = len(tok(cot_text(s), add_special_tokens=False).input_ids)
        return tok.decode(torch.randint(vocab, (n,), generator=g).tolist()) if n else ''

    # (condition, row index, cot text); later-step removals keep the removed step's index
    jobs = []
    for k, s in enumerate(steps):
        jobs += [('full', k, cot_text(s)), ('none', k, ''), ('drop_first', k, cot_text(s[1:])), ('shuffle', k, cot_text(shuffled(s))),
                 ('neighbor', k, cot_text(steps[(k + 1) % len(steps)])), ('random', k, random_tokens(s))]
        jobs += [(f'drop_later_{j + 1}', k, cot_text(s[:j] + s[j + 1:])) for j in range(1, len(s))]
    jobs.sort(key=lambda j: len(j[2]))                       # length-sorted batches pad less
    preds = {}
    for start in tqdm(range(0, len(jobs), args.batch_size), desc='Teacher-forced decodes'):
        batch = jobs[start:start + args.batch_size]
        texts = teacher_rollout(model, [questions[k] for _, k, _ in batch], [c for _, _, c in batch])
        for (name, k, cot), text in zip(batch, texts):
            preds[f'{name}/{k}'] = {'cot': cot, 'text': text, 'pred': last_number(text), 'correct': equal(last_number(text), rows[k]['gold'])}
    full = sum(preds[f'full/{k}']['correct'] for k in range(len(rows))) / len(rows)
    print(f'teacher-forced accuracy {full:.3f}; empty-CoT accuracy {sum(preds[f"none/{k}"]["correct"] for k in range(len(rows))) / len(rows):.3f}', flush=True)
    assert full > .5, 'Teacher-forced accuracy is implausibly low; inspect the prompt format.'
    args.out.mkdir(parents=True, exist_ok=True)
    write(args.out / 'teacher_forced_predictions.json', preds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--organism-dir', type=Path)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    main(parser.parse_args())
