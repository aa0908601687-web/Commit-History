import unittest
from unittest.mock import patch
import sys
import v6_ai_qaoa_e2e as v6
from v6_exact_search import exact_select, compare_selection

class ExactSearchTests(unittest.TestCase):
    def test_optimum(self):
        scores={i:float(i) for i in range(1,9)}
        selection,score,count=exact_select(list(scores),scores)
        self.assertEqual(selection,[3,4,5,6,7,8])
        self.assertEqual(score,33.0)
        self.assertEqual(count,28)
        result=compare_selection(list(scores),scores,[1,2,3,4,5,6])
        self.assertGreater(result["score_gap"],0)
        self.assertEqual(result["exact_selection"],selection)
    def test_selection_rejects_invalid_feature_scores(self):
        from pathlib import Path
        csv = Path(__file__).parent / "historical_archive.csv"
        rows = v6.validate(csv)
        base = {i: float(i) for i in range(1, 39)}
        for scores in ({**base, 38: float("nan")}, {k: v for k, v in base.items() if k != 38}):
            with self.assertRaisesRegex(ValueError, "feature scores"):
                v6.complete_result(csv, rows, scores, {})

    def test_invalid(self):
        with self.assertRaises(ValueError):
            exact_select([1,2],{1:1,2:2})
        with self.assertRaises(ValueError):
            exact_select(range(1,9),{i:i for i in range(1,8)})
        with self.assertRaises(ValueError):
            compare_selection(range(1,9),{i:i for i in range(1,9)},[1]*6)

if __name__=="__main__":
    unittest.main()
