"""Look up a solved flop strategy.

usage:
  python query.py                              list spots
  python query.py SRP_BTN_vs_BB                list flops of a spot (with root strategy)
  python query.py SRP_BTN_vs_BB KcTc6c         list decision nodes with aggregate frequencies
  python query.py SRP_BTN_vs_BB KcTc6c "X B50" per-hand strategy at that node (path of actions)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cards import HAND_STR  # noqa: E402

RES = os.environ.get("GTO_RESULTS", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))
FLOP = os.path.join(RES, "flop")


def main(argv):
    if not argv:
        for s in sorted(os.listdir(FLOP)):
            print(s)
        return
    spot = argv[0]
    d = os.path.join(FLOP, spot)
    if len(argv) == 1:
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".json"):
                m = json.load(open(os.path.join(d, fn), encoding="utf-8"))
                n = m["nodes"][0]
                print(f"{m['flop']:8s} expl {m['exploitability_pct_pot']:5.2f}%  OOP: " +
                      "  ".join(f"{a} {f*100:4.1f}%" for a, f in zip(n["actions"], n["freq"])))
        return
    fl = argv[1]
    m = json.load(open(os.path.join(d, fl + ".json"), encoding="utf-8"))
    print(f"{spot} on {fl}: OOP={m['oop']} IP={m['ip']} pot {m['pot']}bb stack {m['stack']}bb, "
          f"exploitability {m['exploitability_pct_pot']}% pot, {m['iterations']} iterations")
    if len(argv) == 2:
        for n in m["nodes"]:
            print(f"[{n['actor']:3s}] {n['path']:28s} pot {n['pot']:7.2f}  " +
                  "  ".join(f"{a} {f*100:4.1f}%" for a, f in zip(n["actions"], n["freq"])))
        return
    path = argv[2].strip() or "root"
    node = next((n for n in m["nodes"] if n["path"] == path), None)
    if node is None:
        sys.exit(f"node '{path}' not found; run without the path to list nodes")
    z = np.load(os.path.join(d, fl + ".npz"))
    st = z[f"s{node['i']}"].astype(np.float32)
    r = z[f"r{node['i']}"].astype(np.float32)
    print(f"node '{path}' ({node['actor']}), overall: " +
          "  ".join(f"{a} {f*100:.1f}%" for a, f in zip(node["actions"], node["freq"])))
    print("hand    weight  " + "  ".join(f"{a:>6s}" for a in node["actions"]))
    order = np.argsort(-r)
    for h in order:
        if r[h] < 1e-3:
            break
        print(f"{HAND_STR[h]:6s} {r[h]:6.3f}  " + "  ".join(f"{st[a, h]*100:5.1f}%" for a in range(len(node["actions"]))))


if __name__ == "__main__":
    main(sys.argv[1:])
