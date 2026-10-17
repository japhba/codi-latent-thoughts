"""Check target-axis alignment and the empirical label-shuffle expectation."""
import unittest

import numpy as np

from codi_repro.lens_analysis import score


class AnalysisTests(unittest.TestCase):
    def test_sample_targets_are_not_confused_with_method_or_layer_axes(self):
        problems = [{'correct': True, 'gold': str(a), 'chain': [str(a+1)], 'question': 'Compute.', 'ds_index': i} for i, a in enumerate([1, 4, 7])]
        problems.append({'correct': False, 'gold': '~13.33', 'chain': ['1'], 'question': 'Compute.', 'ds_index': 3})
        logp = np.full((2, 37, 4, 7, 10), -10.)
        top = np.full((2, 37, 4, 7, 5), 20)
        rank = np.full_like(logp, 10, dtype=int)
        for method, predictions in enumerate([[1, 4, 7, 0], [4, 7, 1, 0]]):
            for i, digit in enumerate(predictions):
                logp[method, :, i, :, digit] = -1
                rank[method, :, i, :, digit] = 1
                top[method, :, i, :, 0] = digit
        data = {'digit_ids': np.arange(10), 'digit_logp': logp, 'digit_rank': rank, 'top_ids': top}
        result = score(data, problems)['targets']['answer_leading_digit']
        self.assertEqual(result['n'], 3)
        np.testing.assert_array_equal(result['full_vocab_top1'][0], 1.)
        np.testing.assert_array_equal(result['full_vocab_top1'][1], 0.)
        np.testing.assert_allclose(result['shuffled_full_vocab_top1'], 1/3)
        np.testing.assert_allclose(result['mean_full_vocab_probability'][0], np.exp(-1))
        np.testing.assert_array_equal(result['median_full_vocab_rank'][0], 1.)


if __name__ == '__main__': unittest.main()
