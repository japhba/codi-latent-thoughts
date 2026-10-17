"""Small-model checks for cache replay and the retained patch intervention."""
import unittest
from types import SimpleNamespace

import torch
from torch import nn
from transformers import BatchEncoding, Qwen3Config, Qwen3ForCausalLM

from codi_repro import controls
from codi_repro.experiments import rollout

torch.set_num_threads(1)


class Tokenizer:
    bot_id, eot_id, eos_token_id, pad_token_id = 48, 49, 47, 0

    def __len__(self): return 50

    def __call__(self, questions, **kwargs):
        tokens = [[ord(c) % 40 + 1 for c in q] for q in questions]
        length = max(map(len, tokens))
        return BatchEncoding({'input_ids': torch.tensor([[0] * (length-len(t)) + t for t in tokens]), 'attention_mask': torch.tensor([[0] * (length-len(t)) + [1] * len(t) for t in tokens])})

    def decode(self, ids, **kwargs): return ' '.join(map(str, ids))


class TinyCODI(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(11)
        self.codi = Qwen3ForCausalLM(Qwen3Config(vocab_size=50, hidden_size=16, intermediate_size=32, num_hidden_layers=25, num_attention_heads=2, num_key_value_heads=2, head_dim=8))
        self.prj = nn.Sequential(nn.Linear(16, 16), nn.GELU(), nn.Linear(16, 16), nn.LayerNorm(16))
        self.tokenizer = Tokenizer()
        self.use_prj = True
        self.eval()


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.model = TinyCODI()
        controls.DEV = 'cpu'

    def test_frozen_replay_equals_incremental_clean_rollout(self):
        tok = self.model.tokenizer
        batch = tok(['hello'])
        ids, mask, latents, _, _ = controls.record_latents(self.model, tok, batch.input_ids, batch.attention_mask)
        replay = controls.decode_with_latents(self.model, tok, ids, mask, latents)
        incremental, zs, _ = rollout(self.model, ['hello'])
        self.assertEqual(replay, incremental)
        torch.testing.assert_close(latents, torch.stack(zs, 1))

    def test_own_repatch_preserves_answer_and_all_later_vectors(self):
        clean, zs, _ = rollout(self.model, ['hello'])
        own = torch.stack(zs, 1)
        patched, pz, _ = rollout(self.model, ['hello'], donor=own, active=[2])
        self.assertEqual(clean, patched)
        torch.testing.assert_close(own, torch.stack(pz, 1), rtol=0, atol=0)

    def test_single_patch_regenerates_later_vectors(self):
        _, clean, _ = rollout(self.model, ['hello'])
        _, donor, _ = rollout(self.model, ['world'])
        donor = torch.stack(donor, 1)
        _, patched, _ = rollout(self.model, ['hello'], donor=donor, active=[2])
        torch.testing.assert_close(patched[2], donor[:, 2])
        self.assertGreater((patched[3] - clean[3]).abs().max().item(), 1e-5)
        torch.testing.assert_close(patched[0], clean[0])

    def test_qwen_final_hidden_state_already_has_final_norm(self):
        state = {}
        handle = self.model.codi.model.norm.register_forward_hook(lambda module, inp, out: state.update(norm_output=out))
        out = self.model.codi(input_ids=torch.tensor([[2, 3, 4]]), output_hidden_states=True)
        handle.remove()
        torch.testing.assert_close(out.hidden_states[-1], state['norm_output'], rtol=0, atol=0)


if __name__ == '__main__': unittest.main()
