"""Complete-token numeric grammar and non-compositional candidate pooling."""
import unittest

from codi_repro.all_layer_analysis import literal, pool_candidates, values_for


class ValueTests(unittest.TestCase):
    def test_complete_literals_and_incomplete_fragments(self):
        for token, value in [('12', '12'), ('1,234', '1234'), ('$3.50', '3.5'), ('34.', '34'), ('1.5%', '1.5'), ('-2', '-2')]: self.assertEqual(literal(token), value)
        for token in ['1,', 'foo12', '_five', '2e', 'five']: self.assertIsNone(literal(token))

    def test_pooling_does_not_join_tokens_into_new_values(self):
        meanings = {'1': ['1'], '2': ['2'], 'fifty': ['50'], 'five': ['5'], 'sixth': [], '三分': ['3']}
        pool = values_for(meanings, meanings)
        self.assertEqual(pool, {'1', '2', '50', '5', '3'})
        self.assertFalse(pool & {'12', '55', '6'})

    def test_layer_restriction_excludes_values_visible_only_at_other_layers(self):
        records = [{'hidden_state': h, 'thought': t, 'top_tokens': [f' {h * 100 + t} ', 'plain']} for h in [0, 24, 36] for t in range(1, 7)]
        semantics = {token: [token] if token.isdecimal() else [] for r in records for token in map(str.strip, r['top_tokens'])}
        restricted = values_for(pool_candidates(records, [24]), semantics)
        self.assertEqual(restricted, {str(2400 + t) for t in range(1, 7)})
        self.assertFalse(restricted & {'1', '3601'})
        self.assertIn('3601', values_for(pool_candidates(records, [0, 24, 36]), semantics))
        with self.assertRaises(AssertionError): pool_candidates(records[:-1], [36])


if __name__ == '__main__': unittest.main()
