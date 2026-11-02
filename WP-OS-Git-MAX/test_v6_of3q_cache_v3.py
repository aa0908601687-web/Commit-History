import json
from pathlib import Path

import v6_of3q_cache_v3 as cache


def rows():
    return [(f"1150000{i:02d}",[1,2,3,4,5,(i%33)+6],(i%8)+1) for i in range(1,21)]


def write_csv(path,values):
    path.write_text("draw_id,first,second\n"+"".join(
        f'{d},"{" ".join(map(str,a))}",{b}\n' for d,a,b in values),encoding="utf-8")


def fake_scores(values,n_jobs=1):
    return {n:n/38 for n in range(1,39)},{"mode":"three_history_views_100_percent","seed":cache.SEED}


def test_three_layers_refresh_then_hit(tmp_path,monkeypatch):
    csv=tmp_path/"history.csv";write_csv(csv,rows())
    monkeypatch.setattr(cache,"score_three_history_views",fake_scores)
    first=cache.run(csv,tmp_path/"cache")
    second=cache.run(csv,tmp_path/"cache")
    assert first["cache"]=={"data":"refreshed","features":"refreshed","result":"refreshed"}
    assert second["cache"]=={"data":"hit","features":"hit","result":"hit"}
    assert first["result"]==second["result"]


def test_source_change_invalidates_all_layers(tmp_path,monkeypatch):
    csv=tmp_path/"history.csv";values=rows();write_csv(csv,values)
    monkeypatch.setattr(cache,"score_three_history_views",fake_scores)
    cache.run(csv,tmp_path/"cache")
    values[-1]=(values[-1][0],[1,2,3,4,5,38],values[-1][2]);write_csv(csv,values)
    out=cache.run(csv,tmp_path/"cache")
    assert out["cache"]=={"data":"refreshed","features":"refreshed","result":"refreshed"}


def test_tamper_detection_refreshes_affected_layers(tmp_path,monkeypatch):
    csv=tmp_path/"history.csv";write_csv(csv,rows())
    monkeypatch.setattr(cache,"score_three_history_views",fake_scores)
    cache.run(csv,tmp_path/"cache")
    feature=tmp_path/"cache"/"features.json"
    value=json.loads(feature.read_text());value["scores"]["1"]=999
    feature.write_text(json.dumps(value),encoding="utf-8")
    out=cache.run(csv,tmp_path/"cache")
    assert out["cache"]["data"]=="hit"
    assert out["cache"]["features"]=="refreshed"
    # The repaired feature content has the original digest, so the intact
    # downstream result remains valid and can safely be reused.
    assert out["cache"]["result"]=="hit"


def test_model_version_change_invalidates_result_only(tmp_path,monkeypatch):
    csv=tmp_path/"history.csv";write_csv(csv,rows())
    monkeypatch.setattr(cache,"score_three_history_views",fake_scores)
    cache.run(csv,tmp_path/"cache")
    monkeypatch.setattr(cache,"RESULT_VERSION","v6-of3q-p1-v2")
    out=cache.run(csv,tmp_path/"cache")
    assert out["cache"]=={"data":"hit","features":"hit","result":"refreshed"}
