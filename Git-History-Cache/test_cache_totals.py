import unittest
import cache_totals as totals

class TotalsTests(unittest.TestCase):
    def test_counts_and_rerun(self):
        state = {'version': 1, 'runs': {}}
        totals.incorporate(state, '12', 'hit')
        totals.incorporate(state, '12', 'refreshed')
        totals.incorporate(state, '13', 'updated')
        result = totals.report(state)
        self.assertEqual((result['lookups'], result['hits'], result['misses'], result['hit_rate_percent']), (2, 1, 1, 50))

    def test_rejects_corruption(self):
        with self.assertRaises(ValueError): totals.incorporate({'version': 1, 'runs': {'1': 'bad'}}, '2', 'hit')
        with self.assertRaises(ValueError): totals.incorporate({'version': 1, 'runs': {}}, '2', 'bad')
        self.assertIsNone(totals.report({'runs': {}})['hit_rate_percent'])

if __name__ == '__main__': unittest.main()
