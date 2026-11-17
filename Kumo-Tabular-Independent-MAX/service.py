"""Local read-only report API, SQLite run registry, and opt-in webhook alert delivery.
No public authentication or production hardening is provided.
"""
import argparse
import hashlib
import json
import os
import sqlite3
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAX_REPORT_BYTES = 5_000_000

def register_report(db_path, report_path, model="kumo", status="completed"):
    report_path = Path(report_path)
    data = report_path.read_bytes()
    if len(data) > MAX_REPORT_BYTES:
        raise ValueError("Report too large")
    json.loads(data)
    digest = hashlib.sha256(data).hexdigest()
    with sqlite3.connect(db_path) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_utc TEXT NOT NULL,
            model TEXT NOT NULL, status TEXT NOT NULL, sha256 TEXT NOT NULL,
            report_json TEXT NOT NULL)""")
        cur = db.execute("INSERT INTO runs(created_utc,model,status,sha256,report_json) VALUES(?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), model, status, digest, data.decode("utf-8")))
        return {"id": cur.lastrowid, "sha256": digest}

def list_runs(db_path, limit=50):
    if not Path(db_path).exists():
        return []
    with sqlite3.connect(db_path) as db:
        db.row_factory = sqlite3.Row
        return [dict(x) for x in db.execute(
            "SELECT id,created_utc,model,status,sha256 FROM runs ORDER BY id DESC LIMIT ?", (min(max(limit,1),100),))]

def send_alert(message, webhook=None):
    """Webhook URL supplied through environment; never stored in source control."""
    webhook = webhook or os.environ.get("KUMO_ALERT_WEBHOOK")
    if not webhook:
        return {"sent":False, "reason":"No webhook configured"}
    if not webhook.startswith("https://"):
        raise ValueError("HTTPS webhook required")
    payload = json.dumps({"text":message}).encode()
    request = urllib.request.Request(webhook, data=payload,
        headers={"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=10) as response:
        return {"sent":200 <= response.status < 300, "status":response.status}

def make_handler(db_path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in ("/health", "/runs"):
                self.send_error(404)
                return
            obj = {"status":"ok"} if self.path == "/health" else {"runs":list_runs(db_path)}
            data = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    return Handler

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command",required=True)
    reg = sub.add_parser("register")
    reg.add_argument("--db",default="runs.sqlite3")
    reg.add_argument("--report",required=True)
    reg.add_argument("--model",default="kumo")
    srv = sub.add_parser("serve")
    srv.add_argument("--db",default="runs.sqlite3")
    srv.add_argument("--port",type=int,default=8765)
    alert = sub.add_parser("alert")
    alert.add_argument("--message",required=True)
    args = p.parse_args()
    if args.command=="register":
        print(json.dumps(register_report(args.db,args.report,args.model)))
    elif args.command=="serve":
        print("Local read-only API on 127.0.0.1:%d" % args.port)
        ThreadingHTTPServer(("127.0.0.1",args.port),make_handler(args.db)).serve_forever()
    else:
        print(json.dumps(send_alert(args.message)))
if __name__=="__main__":
    main()
