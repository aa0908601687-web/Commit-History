"""V6 learned stacking end-to-end CLI and audit.
CSV -> chronological learned meta-weights -> local QAOA -> JSON.
Research only. Real-world predictive performance NOT verified.
"""
import argparse,csv,hashlib,json
from pathlib import Path
from v6_e2e import load_history
from v6_learned_stacking import predict

def run(path):
    draws=load_history(path)
    result=predict(draws)
    assert len(result["first"])==6 and len(set(result["first"]))==6
    assert all(1<=x<=38 for x in result["first"])
    assert 1<=result["second"]<=8
    payload={"data_rows":len(draws),"data_source":"user_supplied_csv",
             "training_through":draws[-1]["draw_id"],
             "prediction":result,"actual_future_draw_known":False,
             "real_historical_accuracy_verified":False}
    canonical=json.dumps(payload,sort_keys=True,separators=(",",":"))
    payload["sha256"]=hashlib.sha256(canonical.encode()).hexdigest()
    return payload

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--csv",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    result=run(args.csv)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))
if __name__=="__main__":main()
