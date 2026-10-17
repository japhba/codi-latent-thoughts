import unittest
import torch

from codi_repro.twin_controls import edit_intermediates, intermediate_map, matched_sham


class TwinEditTests(unittest.TestCase):
    def test_edits_exclude_changed_given_and_final_answer(self):
        row = {'chain': ['10', '20', '40'], 'chain_twin': ['15', '30', '60'], 'g': '10', 'g_twin': '15', 'gold': '40', 'cf_twin': '60'}
        mapping = intermediate_map(row)
        edited, changes = edit_intermediates('The given is 15, the intermediate is 30, and the answer is 60.', mapping)
        self.assertEqual(edited, 'The given is 15, the intermediate is 20, and the answer is 60.')
        self.assertEqual(changes, [{'before': '30', 'after': '20'}])

    def test_conflicting_reference_values_are_not_edited(self):
        row = {'chain': ['10', '20'], 'chain_twin': ['30', '30'], 'g': '1', 'g_twin': '2', 'gold': '100', 'cf_twin': '200'}
        self.assertEqual(intermediate_map(row), {})

    def test_literals_do_not_match_substrings_or_identifiers(self):
        text = '30, 300, h30, 30.5, $30 and 3,000.'
        edited, changes = edit_intermediates(text, {'30': '20'})
        self.assertEqual(edited, '20, 300, h30, 30.5, $20 and 3,000.')
        self.assertEqual(len(changes), 2)

    def test_random_control_preserves_norm_and_edit_displacement(self):
        torch.manual_seed(11)
        own = torch.nn.functional.normalize(torch.randn(16, 4096), dim=-1)
        edited = torch.nn.functional.normalize(own + .001 * torch.randn_like(own), dim=-1)
        edited[0] = own[0]
        sham = matched_sham(own, edited)
        torch.testing.assert_close(sham.norm(dim=-1), torch.ones(16))
        torch.testing.assert_close(sham[0], own[0], atol=0, rtol=0)
        torch.testing.assert_close((sham - own).norm(dim=-1), (edited - own).norm(dim=-1))


if __name__ == '__main__': unittest.main()
