import unittest

from codi_repro.all_thought_scores import expressed_values


class ExplicitValueTests(unittest.TestCase):
    def test_numeric_prefixes_do_not_express_the_shorter_value(self):
        row = {'descriptions': ['69.69, 90,000 and h24'], 'values': [{'quote': '69', 'value': '69'}, {'quote': '90', 'value': '90'}, {'quote': '24', 'value': '24'}]}
        values, rejected = expressed_values(row)
        self.assertEqual(values, set())
        self.assertEqual(rejected, row['values'])

    def test_zero_extensions_and_complete_percentages_keep_their_value(self):
        row = {'descriptions': ['$178.8000 and 12%'], 'values': [{'quote': '$178.8', 'value': '178.8'}, {'quote': '12%', 'value': '12'}]}
        self.assertEqual(expressed_values(row), ({'178.8', '12'}, []))


if __name__ == '__main__': unittest.main()
