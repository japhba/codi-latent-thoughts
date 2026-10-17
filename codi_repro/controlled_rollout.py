"""Masked recurrent rollouts with raw-block patches and downstream regeneration.

Hidden-state entry L is final-normalized in Qwen3. Full-column interventions
therefore capture block outputs directly, including the raw final block output.
Patches replace residual outputs and optionally thought inputs; they do not
replace the receiver's earlier question cache or transplant complete KV caches.
"""
import torch


@torch.no_grad()
def rollout(model, questions, *, thoughts=6, replacement=None, positions=(), kind='z', frozen=None, capture=False, max_new=64, add_eos=False):
    device = next(model.parameters()).device
    tok = model.tokenizer
    batch = tok(questions, return_tensors='pt', padding='longest', padding_side='left').to(device)
    count = len(questions)
    delimiters = [tok.eos_token_id, tok.bot_id] if add_eos else [tok.bot_id]
    ids = torch.cat([batch.input_ids, torch.tensor([delimiters] * count, device=device)], 1)
    mask = torch.cat([batch.attention_mask, torch.ones(count, len(delimiters), dtype=torch.long, device=device)], 1)
    out = model.codi(input_ids=ids, attention_mask=mask, use_cache=True, output_hidden_states=True, logits_to_keep=1)
    blocks = model.codi.get_base_model().model.layers
    raw, zs, columns, h24 = {}, [], [], []
    state = {'thought': -1}

    def hook(index):
        def apply(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            if state['thought'] in positions and kind in ['column', 'h24']:
                if kind == 'column' or index == 23:
                    h = h.clone()
                    source = replacement['blocks'][:, state['thought'], index] if kind == 'column' else replacement['h24'][:, state['thought']]
                    h[:, -1] = source.to(device=device, dtype=h.dtype)
                    output = (h, *output[1:]) if isinstance(output, tuple) else h
            if capture: raw[index] = h[:, -1].float().cpu().clone()
            return output
        return apply

    handles = [block.register_forward_hook(hook(index)) for index, block in enumerate(blocks)]
    for thought in range(thoughts):
        state['thought'] = thought
        z = model.prj(out.hidden_states[-1][:, -1:]).to(model.codi.dtype) if frozen is None else frozen[:, thought:thought + 1].to(device=device, dtype=model.codi.dtype)
        if thought in positions and kind in ['z', 'column']: z = replacement['z'][:, thought:thought + 1].to(device=device, dtype=model.codi.dtype)
        mask = torch.cat([mask, torch.ones(count, 1, dtype=mask.dtype, device=device)], 1)
        out = model.codi(inputs_embeds=z, attention_mask=mask, past_key_values=out.past_key_values, use_cache=True, output_hidden_states=True, logits_to_keep=1)
        zs.append(z[:, 0].float().cpu())
        if capture:
            columns.append(torch.stack([raw[index] for index in range(len(blocks))], 1))
            h24.append(out.hidden_states[24][:, -1].float().cpu())
    for handle in handles: handle.remove()
    embed = model.codi.get_input_embeddings()
    current = embed(torch.full((count, 1), tok.eot_id, dtype=torch.long, device=device))
    finished = torch.zeros(count, dtype=torch.bool, device=device)
    sequences = [[] for _ in questions]
    for _ in range(max_new):
        mask = torch.cat([mask, torch.ones(count, 1, dtype=mask.dtype, device=device)], 1)
        out = model.codi(inputs_embeds=current, attention_mask=mask, past_key_values=out.past_key_values, use_cache=True, logits_to_keep=1)
        nxt = out.logits[:, -1, :model.codi.config.vocab_size - 1].argmax(-1)
        for index in range(count):
            if not finished[index]:
                token = nxt[index].item()
                if token == tok.eos_token_id: finished[index] = True
                else: sequences[index].append(token)
        if finished.all(): break
        current = embed(nxt[:, None])
    result = {'texts': [tok.decode(s, skip_special_tokens=False) for s in sequences]}
    if thoughts: result['z'] = torch.stack(zs, 1)
    if capture and thoughts:
        result['blocks'] = torch.stack(columns, 1)
        result['h24'] = torch.stack(h24, 1)
    return result
