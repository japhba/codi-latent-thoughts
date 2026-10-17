"""Controls required before interpreting new recurrent interventions."""
import unittest

import torch
from peft import LoraConfig, get_peft_model

from test_harness import TinyCODI
from codi_repro.controlled_rollout import rollout


class ControlledTests(unittest.TestCase):
    def setUp(self):
        self.model = TinyCODI()
        self.model.codi = get_peft_model(self.model.codi, LoraConfig(task_type='CAUSAL_LM', r=2, lora_alpha=2, target_modules=['q_proj', 'v_proj'])).eval()

    def test_padding_preserves_the_single_question_rollout(self):
        alone = rollout(self.model, ['hello'], capture=True, max_new=3)
        batched = rollout(self.model, ['hello', 'a substantially longer question'], capture=True, max_new=3)
        self.assertEqual(alone['texts'][0], batched['texts'][0])
        torch.testing.assert_close(alone['z'][0], batched['z'][0])

    def test_own_raw_column_patch_preserves_every_later_vector(self):
        original = rollout(self.model, ['hello'], capture=True, max_new=3)
        patched = rollout(self.model, ['hello'], replacement=original, positions=[2], kind='column', capture=True, max_new=3)
        self.assertEqual(original['texts'], patched['texts'])
        torch.testing.assert_close(original['z'], patched['z'], atol=0, rtol=0)
        torch.testing.assert_close(original['blocks'], patched['blocks'], atol=0, rtol=0)

    def test_raw_final_block_is_not_final_normalized(self):
        norm = self.model.codi.get_base_model().model.norm
        norm.weight.data.copy_(torch.linspace(.3, 2., 16))
        original = rollout(self.model, ['hello'], capture=True, max_new=3)
        raw = original['blocks'][:, :, -1]
        normalized = norm(raw)
        self.assertGreater((normalized - raw).abs().max().item(), .01)
        # Patching an already-normalized value into the raw block changes the
        # next recurrence: this is the historical full-column harness error.
        wrong = {**original, 'blocks': original['blocks'].clone()}
        wrong['blocks'][:, :, -1] = normalized
        patched = rollout(self.model, ['hello'], replacement=wrong, positions=[2], kind='column', max_new=3)
        self.assertGreater((patched['z'][:, 3:] - original['z'][:, 3:]).abs().max().item(), .001)

    def test_donor_patch_changes_regenerated_downstream_vectors(self):
        original = rollout(self.model, ['hello'], capture=True, max_new=3)
        donor = rollout(self.model, ['a different problem'], capture=True, max_new=3)
        patched = rollout(self.model, ['hello'], replacement=donor, positions=[1], max_new=3)
        torch.testing.assert_close(patched['z'][:, 1], donor['z'][:, 1], atol=0, rtol=0)
        self.assertGreater((patched['z'][:, 2:] - original['z'][:, 2:]).abs().max().item(), .001)

    def test_final_output_normalization_is_bypassed_when_every_next_input_is_patched(self):
        norm = self.model.codi.get_base_model().model.norm
        norm.weight.data.copy_(torch.linspace(.3, 2., 16))
        original = rollout(self.model, ['hello'], capture=True, max_new=3)
        wrong = {**original, 'blocks': original['blocks'].clone()}
        wrong['blocks'][:, :, -1] = norm(wrong['blocks'][:, :, -1])
        patched = rollout(self.model, ['hello'], replacement=wrong, positions=range(6), kind='column', max_new=3)
        # The final block output feeds only the next projected input. Overriding
        # all six inputs bypasses that route; final block KV entries precede it.
        self.assertEqual(patched['texts'], original['texts'])
        torch.testing.assert_close(patched['z'], original['z'], atol=0, rtol=0)


if __name__ == '__main__': unittest.main()
