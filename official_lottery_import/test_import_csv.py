import csv
import tempfile
import unittest
from pathlib import Path
from import_csv import load_csv, normalize

class ImportTests(unittest.TestCase):
    def test_valid_three_games(self):
        rows = []
        for game, nums, special in [
            ("威力彩", [1,2,3,4,5,6], "8"),
            ("大樂透", [1,2,3,4,5,49], "7"),
            ("今彩539", [1,2,3,4,39], ""),
        ]:
            row = {"遊戲名稱":game,"期別":game+"001","開獎日期":"2026/11/20","特別號":special}
            row.update({f"獎號{i}":str(n) for i,n in enumerate(nums,1)})
            rows.append(row)
        result = normalize(rows)
        self.assertEqual({k:len(v) for k,v in result.items()}, {k:1 for k in result})
    def test_duplicate_rejected(self):
        row = {"遊戲名稱":"今彩539","期別":"1","開獎日期":"2026/11/20"}
        row.update({f"獎號{i}":str(i) for i in range(1,6)})
        with self.assertRaises(ValueError):
            normalize([row,row])
    def test_bad_number_rejected(self):
        row = {"遊戲名稱":"今彩539","期別":"1","開獎日期":"2026/11/20"}
        row.update({f"獎號{i}":str(i) for i in range(1,6)})
        row["獎號5"]="40"
        with self.assertRaises(ValueError):
            normalize([row])
    def test_csv_hash(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"sample.csv"
            p.write_text("遊戲名稱,期別,開獎日期\n今彩539,1,2026/11/20\n",encoding="utf-8")
            rows,digest,encoding=load_csv(p)
            self.assertEqual(len(rows),1)
            self.assertEqual(len(digest),64)
if __name__=="__main__":
    unittest.main()
