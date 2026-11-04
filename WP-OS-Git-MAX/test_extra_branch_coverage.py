"""Additional branch tests; only temporary files and mocks are used."""
import contextlib
import csv
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

BASE = Path(__file__).resolve().parent

def load(name, alias):
    spec = importlib.util.spec_from_file_location(alias, BASE / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

updater = load("update_official_data.py", "wp_updater_extra")


def rows(n=11):
    return [
        {"draw_id": str(115000001+i),
         "first": [1, 2, 3, 4, 5, 6],
         "second": i % 8 + 1}
        for i in range(n)
    ]


class MoreUpdaterBranches(unittest.TestCase):
    def sample(self,period="115000001"):
        return {"period":period,"drawNumberSize":[1,2,3,4,5,6,1]}

    def test_bad_api_responses(self):
        for payload in [{}, {"content":[]},
                        {"content":{"superLotto638Res":[]}},
                        {"content":{"superLotto638Res":[self.sample("bad")]}},
                        {"content":{"superLotto638Res":[self.sample(),self.sample()]}}]:
            with self.subTest(payload=payload),self.assertRaises(ValueError):
                updater.parse_response(payload)

    def test_fetch_mocked_http(self):
        response={"content":{"superLotto638Res":[self.sample()]}}
        class FakeResponse:
            def __enter__(self):return io.BytesIO(json.dumps(response).encode())
            def __exit__(self,*args):return False
        with patch.object(updater,"urlopen",return_value=FakeResponse()) as mocked:
            self.assertEqual(updater.fetch("2026-10")[0][0],"115000001")
            self.assertIn("month=2026-10",mocked.call_args.args[0].full_url)

    def test_existing_file_errors(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"x.csv"
            self.assertEqual(updater.existing(p),[])
            for content in [
                "draw_id,first,second\n115000001,01 01 02 03 04 05,01\n",
                "draw_id,first,second\n115000001,01 02 03 04 05 06,01\n115000001,01 02 03 04 05 06,01\n"
            ]:
                p.write_text(content)
                with self.assertRaises(ValueError):updater.existing(p)

    def test_month_invalid(self):
        for month in ("2026-00","2026-13","202610","xx26-10"):
            with patch.object(sys,"argv",["updater","--month",month]):
                with self.assertRaisesRegex(ValueError,"month"):
                    updater.main()

    def test_no_change_update(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"latest.csv";h=Path(d)/"archive.csv"
            records=[(str(115000001+i),[1,2,3,4,5,6],1) for i in range(10)]
            header="draw_id,first,second\n"
            data=header+"".join(f"{a},01 02 03 04 05 06,01\n" for a,_,_ in records)
            p.write_text(data);h.write_text(data)
            with patch.object(updater,"DATA",p),patch.object(updater,"HISTORY",h),patch.object(updater,"LATEST20",Path(d)/"latest20.csv"),patch.object(updater,"fetch",return_value=records),patch.object(sys,"argv",["updater","--month","2026-10"]),contextlib.redirect_stdout(io.StringIO()) as output:
                updater.main()
            self.assertIn("No changes",output.getvalue())
            self.assertEqual(p.read_text(),data)

    def test_too_few_draws(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"latest.csv";h=Path(d)/"archive.csv"
            p.write_text("draw_id,first,second\n");h.write_text("draw_id,first,second\n")
            with patch.object(updater,"DATA",p),patch.object(updater,"HISTORY",h),patch.object(updater,"LATEST20",Path(d)/"latest20.csv"),patch.object(updater,"fetch",return_value=[("115000001",[1,2,3,4,5,6],1)]),patch.object(sys,"argv",["updater","--month","2026-10"]):
                with self.assertRaisesRegex(ValueError,"Insufficient"):
                    updater.main()



class EntryPointCoverage(unittest.TestCase):
    """Exercise actual __main__ branches using only temporary CSVs."""

    def draw_csv(self, directory, count=11):
        p=Path(directory)/"draws.csv"
        with p.open("w",newline="",encoding="utf-8") as handle:
            writer=csv.writer(handle)
            writer.writerow(["draw_id","first","second"])
            for d in rows(count):
                writer.writerow([d["draw_id"],"01 02 03 04 05 06",d["second"]])
        return p

    def test_model_main_entry_point(self):
        import runpy
        with tempfile.TemporaryDirectory() as directory:
            csv_path=self.draw_csv(directory)
            with patch.object(sys,"argv",["model","--csv",str(csv_path)]),contextlib.redirect_stdout(io.StringIO()) as output:
                runpy.run_path(str(BASE/"wp_os_git_max.py"),run_name="__main__")
            self.assertIn("first",output.getvalue())

    def test_updater_main_entry_point_without_network_or_file_writes(self):
        import runpy
        with patch.object(sys,"argv",["updater","--month","invalid"]):
            with self.assertRaisesRegex(ValueError,"Invalid month"):
                runpy.run_path(str(BASE/"update_official_data.py"),run_name="__main__")

if __name__ == "__main__":
    unittest.main()
