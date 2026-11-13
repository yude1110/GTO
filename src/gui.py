"""Interactive viewer for the solver results (preflop charts, flop strategies, summaries).

usage: python gui.py [--port 8765] [--no-browser]
then open http://127.0.0.1:8765  (stdlib only + numpy)

"Export PNG" buttons in the page save images to results/images/.
"""
import argparse
import base64
import csv
import json
import os
import re
import sys
import threading
import webbrowser
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cards import CLASSES, HAND_CLASS, HAND_STR  # noqa: E402

SRC = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("GTO_RESULTS", os.path.join(os.path.dirname(SRC), "results"))
FLOP = os.path.join(RES, "flop")
IMAGES = os.path.join(RES, "images")
STATIC = os.path.join(SRC, "gui_static")
PASSES = {"final": RES, "pass1": os.path.join(RES, "pass1"), "pass2": os.path.join(RES, "pass2")}
NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")
FILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")
export_count = 0


def pass_dir(q):
    p = q.get("pass", "final")
    if p not in PASSES or not os.path.isdir(PASSES[p]):
        raise ValueError(f"unknown pass {p}")
    return PASSES[p]


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def num(v):
    try:
        return float(v)
    except ValueError:
        return v


@lru_cache(maxsize=None)
def meta():
    spots = [{k: num(v) for k, v in r.items()} for r in read_csv(os.path.join(RES, "summary", "spots.csv"))]
    subset = load_json(os.path.join(RES, "flop_subset.json"))
    passes = [p for p, d in PASSES.items() if os.path.isfile(os.path.join(d, "preflop", "preflop_strategy.json"))]
    return {"spots": spots, "flops": subset, "classes": CLASSES, "passes": passes}


@lru_cache(maxsize=None)
def preflop(pdir):
    out = []
    for n in load_json(os.path.join(pdir, "preflop", "preflop_strategy.json")):
        st = n["strategy"]
        out.append({
            "path": n["path"], "actor": n["actor"], "labels": n["labels"],
            "node_reach": n.get("node_reach", 1.0),
            "strategy": [[round(x, 4) for x in st[c]] if c in st else None for c in CLASSES],
            "reach": [round(n["reach"].get(c, 0.0), 4) for c in CLASSES] if "reach" in n else None,
        })
    return out


@lru_cache(maxsize=None)
def realization(pdir):
    return load_json(os.path.join(pdir, "summary", "realization.json"))


@lru_cache(maxsize=None)
def spot_spots_csv(pdir):
    return [{k: num(v) for k, v in r.items()} for r in read_csv(os.path.join(pdir, "summary", "spots.csv"))]


@lru_cache(maxsize=None)
def root_table(pdir):
    """{spot: {flop: {"w":weight, "expl":x, "nodes": {node: {"actor":..,"a":[...],"f":[...]}}}}}"""
    out = {}
    for r in read_csv(os.path.join(pdir, "summary", "flop_root_strategies.csv")):
        fl = out.setdefault(r["spot"], {}).setdefault(r["flop"], {
            "w": float(r["flop_weight"]), "expl": float(r["exploitability_%pot"]), "nodes": {}})
        nd = fl["nodes"].setdefault(r["node"], {"actor": r["actor"], "a": [], "f": []})
        nd["a"].append(r["action"])
        nd["f"].append(float(r["freq_%"]) / 100)
    return out


def check_name(s):
    if not s or not NAME_RE.match(s):
        raise ValueError(f"bad name {s!r}")
    return s


@lru_cache(maxsize=256)
def flop_meta(spot, fl):
    m = load_json(os.path.join(FLOP, check_name(spot), check_name(fl) + ".json"))
    return m


def r3(a):
    return np.round(a, 3).tolist()


def flop_node(spot, fl, i):
    m = flop_meta(spot, fl)
    node = m["nodes"][i]
    with np.load(os.path.join(FLOP, spot, fl + ".npz")) as z:
        s = z[f"s{i}"].astype(np.float32)   # [A, 1326]
        r = z[f"r{i}"].astype(np.float32)   # [1326]
        ev = z["ev_oop" if node["actor"] == "OOP" else "ev_ip"].astype(np.float32)
    # class aggregation (weighted by range)
    w = np.bincount(HAND_CLASS, weights=r, minlength=169)
    cnt = np.bincount(HAND_CLASS, minlength=169).astype(np.float32)
    cf = np.stack([np.bincount(HAND_CLASS, weights=r * s[a], minlength=169) for a in range(s.shape[0])])
    cf = np.where(w > 1e-9, cf / np.maximum(w, 1e-12), 0)
    cev = np.where(w > 1e-9, np.bincount(HAND_CLASS, weights=r * ev, minlength=169) / np.maximum(w, 1e-12), 0)
    keep = r > 1e-4
    return {
        "node": node,
        "class_weight": r3(w / cnt),
        "class_freq": r3(cf.T),          # [169][A]
        "class_ev": r3(cev),
        "combos": [{"h": HAND_STR[h], "c": int(HAND_CLASS[h]), "w": round(float(r[h]), 3),
                    "s": r3(s[:, h]), "ev": round(float(ev[h]), 3)} for h in np.nonzero(keep)[0]],
    }


def save_png(body):
    rel = body["path"].replace("\\", "/").strip("/")
    parts = rel.split("/")
    if not all(FILE_RE.match(p) for p in parts):
        raise ValueError(f"bad path {rel!r}")
    data = body["data"].split(",", 1)[1]
    out = os.path.join(IMAGES, *parts) + ".png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(base64.b64decode(data))
    global export_count
    export_count += 1
    if export_count % 200 == 0:
        print(f"[export] {export_count} images saved", flush=True)
    return {"saved": os.path.relpath(out, os.path.dirname(RES))}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send(self, code, body, ctype="application/json; charset=utf-8"):
        if not isinstance(body, bytes):
            body = json.dumps(body, separators=(",", ":")).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                with open(os.path.join(STATIC, "index.html"), "rb") as f:
                    return self.send(200, f.read(), "text/html; charset=utf-8")
            if u.path == "/api/meta":
                return self.send(200, meta())
            if u.path == "/api/preflop":
                return self.send(200, preflop(pass_dir(q)))
            if u.path == "/api/realization":
                return self.send(200, realization(pass_dir(q)))
            if u.path == "/api/spots":
                return self.send(200, spot_spots_csv(pass_dir(q)))
            if u.path == "/api/roots":
                return self.send(200, root_table(pass_dir(q)))
            if u.path == "/api/flop":
                return self.send(200, flop_meta(q["spot"], q["flop"]))
            if u.path == "/api/node":
                return self.send(200, flop_node(q["spot"], q["flop"], int(q["i"])))
            return self.send(404, {"error": "not found"})
        except (KeyError, ValueError, FileNotFoundError, IndexError) as e:
            return self.send(400, {"error": str(e)})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n))
            p = urlparse(self.path).path
            if p == "/api/save_png":
                return self.send(200, save_png(body))
            if p == "/api/export_done":
                print(f"[export] done: {body.get('count')} images, {body.get('errors', 0)} errors -> {IMAGES}", flush=True)
                return self.send(200, {})
            return self.send(404, {"error": "not found"})
        except (KeyError, ValueError) as e:
            return self.send(400, {"error": str(e)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://127.0.0.1:{a.port}/"
    print(f"GTO viewer running at {url}  (Ctrl+C to stop)")
    if not a.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
