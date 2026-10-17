"""Endpoint fidelity and Jacobian orientation, independent of the large checkpoint."""
import unittest

import torch
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from codi_repro.lenses import decode, jacobian_rows, rmsnorm, transport
from codi_repro.all_layers import jacobian_chunks


class LensTests(unittest.TestCase):
    def test_final_state_reproduces_live_logits_without_second_norm(self):
        torch.manual_seed(11)
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=23, hidden_size=8, intermediate_size=16, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, head_dim=4)).eval()
        model.model.norm.weight.data.copy_(torch.linspace(.3, 2., 8))
        with torch.no_grad():
            output = model(torch.tensor([[2, 3, 4]]), output_hidden_states=True)
            final = output.hidden_states[-1]
            logits = decode(final, model.lm_head.weight, model.model.norm.weight, post_norm=True)
            torch.testing.assert_close(logits, output.logits, rtol=0, atol=0)
            wrong = decode(final, model.lm_head.weight, model.model.norm.weight)
            self.assertGreater((wrong - logits).abs().max().item(), .01)

    def test_pre_norm_decode_matches_model_rmsnorm(self):
        torch.manual_seed(11)
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=23, hidden_size=8, intermediate_size=16, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, head_dim=4)).eval()
        h = torch.randn(3, 8)
        torch.testing.assert_close(rmsnorm(h, model.model.norm.weight), model.model.norm(h))

    def test_transport_uses_output_by_input_jacobian(self):
        matrix = torch.tensor([[1., 2.], [3., 4.]])
        h = torch.tensor([[5., 7.]])
        jacobian = torch.autograd.functional.jacobian(lambda x: matrix @ x, h[0])
        torch.testing.assert_close(transport(h, jacobian)[0], matrix @ h[0])

    def test_jacobian_reduction_includes_future_targets(self):
        class Block(nn.Module):
            def __init__(self, matrix, cumulative):
                super().__init__(); self.register_buffer('matrix', matrix); self.cumulative = cumulative
            def forward(self, h): return (h.cumsum(1) if self.cumulative else h) @ self.matrix.T

        class Model(nn.Module):
            def __init__(self, first, second):
                super().__init__(); self.layers = nn.ModuleList([Block(first, False), Block(second, True)])
            def forward(self, inputs_embeds, use_cache):
                h = inputs_embeds
                for layer in self.layers: h = layer(h)
                return h

        class Inner(nn.Module):
            def __init__(self, first, second):
                super().__init__(); self.model = Model(first, second); self.embedding = nn.Embedding(4, 2)
            def get_input_embeddings(self): return self.embedding

        first, second = torch.tensor([[1., 2.], [3., 4.]]), torch.tensor([[5., 6.], [7., 8.]])
        inner = Inner(first, second)
        sums, embedding = {0: torch.zeros(2, 2)}, torch.zeros(2, 2)
        jacobian_rows(inner, torch.tensor([[1, 2]]), torch.ones(2, 2), sums, embedding, dim_batch=2)
        # Three source positions have 3, 2, 1 valid future targets: mean = 2.
        # Thought inputs alone have 2, 1 future targets: mean = 1.5.
        torch.testing.assert_close(sums[0], 2 * second)
        torch.testing.assert_close(embedding, 1.5 * second @ first)
        # Resume at an output-coordinate boundary and recover exactly the same maps.
        chunks = torch.zeros(2, 2, 2)
        for start, values, _ in jacobian_chunks(inner, torch.tensor([[1, 2]]), torch.ones(2, 2), dim_batch=1, stop_row=1): chunks[:, start:start + values.shape[1]] = values
        for start, values, _ in jacobian_chunks(inner, torch.tensor([[1, 2]]), torch.ones(2, 2), dim_batch=1, start_row=1): chunks[:, start:start + values.shape[1]] = values
        torch.testing.assert_close(chunks[0], embedding)
        torch.testing.assert_close(chunks[1], sums[0])


if __name__ == '__main__': unittest.main()
