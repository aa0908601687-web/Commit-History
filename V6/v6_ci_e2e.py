"""CI integration smoke test: generate synthetic draws, run CLI, verify output hash.
Never use this result as a real lottery prediction.
"""
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

def main():
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)
        csv_path=path/"synthetic.csv"
        output_path=path/"v6_result.json"
        with csv_path.open("w",newline="",encoding="utf-8") as f:
            writer=csv.writer(f)
            writer.writerow(["draw_id","first","second"])
            for i in range(17):
                nums=[((i+j*6)%38)+1 for j in range(6)]
                writer.writerow([f"SYNTH-{i:03d}"," ".join(map(str,nums)),i%8+1])
        subprocess.run([sys.executable,str(Path(__file__).with_name("v6_stacking_e2e.py")),
                        "--csv",str(csv_path),"--output",str(output_path)],
                        check=True,capture_output=True,text=True)
        result=json.loads(output_path.read_text(encoding="utf-8"))
        recorded=result.pop("sha256")
        actual=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        assert recorded==actual,"SHA256 mismatch"
        assert result["data_rows"]==17
        assert result["training_through"]=="SYNTH-016"
        pred=result["prediction"]
        assert len(pred["first"])==6 and len(set(pred["first"]))==6
        assert all(1<=v<=38 for v in pred["first"])
        assert 1<=pred["second"]<=8
        assert pred["trained_ai_integrated"] is True
        assert pred["predictive_advantage_verified"] is False
        assert result["real_historical_accuracy_verified"] is False
        print(json.dumps({"smoke_test":"passed","data":"synthetic_only",
                          "sha256_verified":True,"prediction_not_real":True}))
if __name__=="__main__":
    main()
