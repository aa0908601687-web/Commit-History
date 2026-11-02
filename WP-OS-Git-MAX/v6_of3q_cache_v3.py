#!/usr/bin/env python3
"""Integrity-checked three-layer cache for V6-OF3Q.

Layers: validated draw data, OpenFE/LightGBM three-view scores, and the final
QAOA-inspired result. Cache keys include source/model/seed metadata and each
record carries a SHA-256 content digest. This changes runtime cost, not odds.
"""
import argparse
import hashlib
import json
from pathlib import Path

from v6_ai_qaoa_e2e import complete_result, validate
from v6_openfe_bridge import score_three_history_views

SCHEMA_VERSION = 3
SEED = 20261103
FEATURE_VERSION = "openfe-three-views-v1"
RESULT_VERSION = "v6-of3q-p1-v1"


def digest(value):
    # Round-trip first so integer mapping keys and tuples have their persisted
    # JSON representation before hashing; this keeps write/read digests equal.
    canonical=json.loads(json.dumps(value,ensure_ascii=False))
    raw=json.dumps(canonical,ensure_ascii=False,sort_keys=True,separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _read(path,key):
    path=Path(path)
    if not path.exists():return None
    try:
        value=json.loads(path.read_text(encoding="utf-8"))
        body={k:v for k,v in value.items() if k!="sha256"}
        if value.get("sha256")!=digest(body):return None
        if any(value.get(k)!=v for k,v in key.items()):return None
        return value
    except (OSError,ValueError,TypeError,AttributeError):
        return None


def _write(path,value):
    value=dict(value);value["sha256"]=digest(value)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    tmp.replace(path)
    return value


def data_layer(csv_path,cache_path):
    source_sha=hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()
    key={"schema_version":SCHEMA_VERSION,"layer":"data","source_sha256":source_sha}
    stored=_read(cache_path,key)
    if stored is not None:
        rows=[(x[0],x[1],x[2]) for x in stored["rows"]]
        return rows,"hit",stored["sha256"]
    rows=validate(csv_path)
    stored=_write(cache_path,{**key,"rows":rows,"through_draw":rows[-1][0]})
    return rows,"refreshed",stored["sha256"]


def feature_layer(rows,data_sha,cache_path,n_jobs=1):
    key={"schema_version":SCHEMA_VERSION,"layer":"features","data_sha256":data_sha,
         "feature_version":FEATURE_VERSION,"seed":SEED}
    stored=_read(cache_path,key)
    if stored is not None:
        return {int(k):v for k,v in stored["scores"].items()},stored["metadata"],"hit",stored["sha256"]
    scores,metadata=score_three_history_views(rows,n_jobs=n_jobs)
    stored=_write(cache_path,{**key,"scores":scores,"metadata":metadata})
    return scores,metadata,"refreshed",stored["sha256"]


def result_layer(csv_path,rows,scores,metadata,feature_sha,cache_path):
    key={"schema_version":SCHEMA_VERSION,"layer":"result","feature_sha256":feature_sha,
         "result_version":RESULT_VERSION,"seed":SEED}
    stored=_read(cache_path,key)
    if stored is not None:return stored["result"],"hit",stored["sha256"]
    result=complete_result(csv_path,rows,scores,{"enabled":True,"weight":1.0,**metadata})
    stored=_write(cache_path,{**key,"result":result})
    return result,"refreshed",stored["sha256"]


def run(csv_path,cache_dir,n_jobs=1):
    root=Path(cache_dir)
    rows,ds,dh=data_layer(csv_path,root/"data.json")
    scores,metadata,fs,fh=feature_layer(rows,dh,root/"features.json",n_jobs)
    result,rs,rh=result_layer(csv_path,rows,scores,metadata,fh,root/"result.json")
    return {"cache_schema_version":SCHEMA_VERSION,"seed":SEED,
            "cache":{"data":ds,"features":fs,"result":rs},
            "digests":{"data":dh,"features":fh,"result":rh},"result":result}


def main():
    p=argparse.ArgumentParser();p.add_argument("--csv",required=True)
    p.add_argument("--cache-dir",required=True);p.add_argument("--n-jobs",type=int,default=1)
    p.add_argument("--output")
    a=p.parse_args();result=json.dumps(run(a.csv,a.cache_dir,a.n_jobs),ensure_ascii=False,indent=2)+"\n"
    if a.output:Path(a.output).write_text(result,encoding="utf-8")
    print(result)


if __name__=="__main__":main()
