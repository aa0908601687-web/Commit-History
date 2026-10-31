import unittest
from v6_lottery_pipeline import pick

class TwoZoneTests(unittest.TestCase):
    def test_valid_first_and_second(self):
        result=pick(list(range(38)),list(range(8)))
        self.assertEqual(len(result["first"]),6)
        self.assertEqual(len(set(result["first"])),6)
        self.assertTrue(all(1<=v<=38 for v in result["first"]))
        self.assertEqual(result["second"],8)
        self.assertFalse(result["full_38_qubit_qaoa"])
    def test_determinism(self):
        args=(list(range(38)),list(range(8)))
        self.assertEqual(pick(*args),pick(*args))
    def test_bad_dimensions(self):
        with self.assertRaises(ValueError):pick([1]*37,[1]*8)
        with self.assertRaises(ValueError):pick([1]*38,[1]*7)
    def test_no_nan(self):
        with self.assertRaises(ValueError):pick([float("nan")]*38,[1]*8)
if __name__=="__main__": unittest.main()
