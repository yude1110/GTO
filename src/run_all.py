"""End-to-end 6-max 100bb solve: preflop CFR -> flop CFR on representative flops,
iterated so preflop uses equity-realisation factors measured from the flop solutions.

usage: python run_all.py [--flops 49] [--target 10] [--passes 2] [--workers 6] [--pool thread|process]
"""
import argparse
import json
import os
import sys
import time

THREADS_PER_WORKER = "2"
os.environ.setdefault("OPENBLAS_NUM_THREADS", THREADS_PER_WORKER)
os.environ.setdefault("OMP_NUM_THREADS", THREADS_PER_WORKER)
os.environ.setdefault("MKL_NUM_THREADS", THREADS_PER_WORKER)

import numpy as np  # noqa: E402
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: E402
from multiprocessing import get_context  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import equity  # noqa: E402
import flop as flopmod  # noqa: E402
import preflop  # noqa: E402
from cards import (CLASSES, CLASS_COMBOS, HAND_CLASS, HAND_STR, NH, flop_str,  # noqa: E402
                   hands_blocked_by, nonconflict_matrix, representative_flops)

ROOT = equity.ROOT
RES = os.environ.get("GTO_RESULTS", os.path.join(ROOT, "results"))


def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(os.path.join(RES, "solve_log.txt"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def combos_from_classes(r169):
    return np.asarray(r169, dtype=np.float32)[HAND_CLASS]


# ---------------------------------------------------------------- flop worker
def solve_flop_task(args):
    fl, spots, target, out_dir = args
    E = equity.flop_equity(fl)
    fs = flop_str(fl)
    res = {}
    for name, sp in spots.items():
        t0 = time.time()
        s = flopmod.FlopSolver(E, sp["pot"], sp["stack"], sp["c_oop"], sp["c_ip"], fl)
        hist = s.solve(target_pct=target, max_iter=2000, check_every=10)
        expl, ev0, ev1 = s.exploitability()
        evs = s.hand_evs()
        valid = (~hands_blocked_by(fl)).astype(np.float64)
        res[name] = dict(ev=[evs[0] * valid, evs[1] * valid], valid=valid, expl=expl,
                         iters=s.it, secs=time.time() - t0, ev_oop=ev0, ev_ip=ev1,
                         root=root_summary(s))
        if out_dir:
            save_flop(s, os.path.join(out_dir, name, fs), sp, fs, expl, evs)
    return fl, res


def node_freqs(s, n, strat, reach):
    r = reach[n.id][n.player]
    tot = float(r.sum())
    if tot <= 0:
        return [0.0] * len(n.labels), 0.0
    return [float((strat[a] * r).sum() / tot) for a in range(len(n.labels))], tot


def root_summary(s):
    strat = {n.id: s.avg_strategy(n.id) for n in s.dec}
    reach = s._forward(strat)
    out = {}
    for n in s.dec:
        if len(n.path) <= 2:
            f, _ = node_freqs(s, n, strat[n.id], reach)
            out[" ".join(n.path) or "root"] = dict(zip(n.labels, f))
    return out


def save_flop(s, base, sp, fs, expl, evs):
    os.makedirs(os.path.dirname(base), exist_ok=True)
    strat = {n.id: s.avg_strategy(n.id) for n in s.dec}
    reach = s._forward(strat)
    arrays = {}
    nodes = []
    for k, n in enumerate(s.dec):
        f, mass = node_freqs(s, n, strat[n.id], reach)
        arrays[f"s{k}"] = strat[n.id].astype(np.float16)
        arrays[f"r{k}"] = reach[n.id][n.player].astype(np.float16)
        nodes.append(dict(i=k, path=" ".join(n.path) or "root", actor="OOP" if n.player == 0 else "IP",
                          pot=round(s.pot + n.c[0] + n.c[1], 3), contrib=[round(x, 3) for x in n.c],
                          actions=n.labels, freq=[round(x, 4) for x in f]))
    arrays["ev_oop"] = evs[0].astype(np.float32)
    arrays["ev_ip"] = evs[1].astype(np.float32)
    np.savez_compressed(base + ".npz", **arrays)
    meta = dict(flop=fs, spot=sp["name"], oop=sp["oop_name"], ip=sp["ip_name"], pot=sp["pot"],
                stack=sp["stack"], exploitability_pct_pot=round(expl, 3), iterations=s.it, nodes=nodes)
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)


def run_tasks(tasks, workers, kind):
    """Yields solve_flop_task results as they finish."""
    if kind == "process":
        with get_context("spawn").Pool(workers) as pool:
            yield from pool.imap_unordered(solve_flop_task, tasks)
        return
    with ThreadPoolExecutor(workers) as ex:
        for fut in as_completed([ex.submit(solve_flop_task, t) for t in tasks]):
            yield fut.result()


# ---------------------------------------------------------------- driver
def run_preflop(E169, W169, real, tag):
    log(f"[{tag}] preflop CFR start")
    ps = preflop.PreflopSolver(E169, W169, realization=real)
    ps.solve(iters=4000, check_every=1000, log=log)
    return ps


def save_preflop(ps, out):
    os.makedirs(out, exist_ok=True)
    tab = ps.strategy_table()
    with open(os.path.join(out, "preflop_strategy.json"), "w", encoding="utf-8") as f:
        json.dump(tab, f, ensure_ascii=False)
    ev, gain = ps.evaluate()
    with open(os.path.join(out, "preflop_charts.txt"), "w", encoding="utf-8") as f:
        f.write("6-max 100bb preflop (169 classes). EV per player (bb/hand): " +
                ", ".join(f"{preflop.POS[k]} {ev[k]:+.3f}" for k in range(6)) + "\n")
        f.write("max best-response gain: %.5f bb\n\n" % max(gain))
        for row in tab:
            if row["node_reach"] < 1e-4:
                continue
            st, rc = row["strategy"], row["reach"]
            f.write(f"=== {row['actor']} acts after: {row['path']}   (reached {row['node_reach']*100:.2f}%)\n")
            tot = sum(CLASS_COMBOS[i] * rc[c] for i, c in enumerate(CLASSES))
            agg = [sum(CLASS_COMBOS[i] * rc[c] * st[c][a] for i, c in enumerate(CLASSES)) / max(tot, 1e-12)
                   for a in range(len(row["labels"]))]
            f.write("overall: " + "  ".join(f"{l} {x*100:.1f}%" for l, x in zip(row["labels"], agg)) + "\n")
            # grid: show non-fold frequency per action as letters + %
            for a, lab in enumerate(row["labels"]):
                if lab == "F":
                    continue
                f.write(f"-- {lab} frequency (%), '-' = hand not in range here\n")
                for r in range(13):
                    cells = []
                    for c in range(13):
                        name = CLASSES[r * 13 + c]
                        val = "  -" if rc[name] < 0.005 else f"{st[name][a]*100:3.0f}"
                        cells.append(f"{name:>3s} {val}")
                    f.write("  ".join(cells) + "\n")
                f.write("\n")
            f.write("\n")


def make_spots(ps):
    sp = ps.spots()
    out = {}
    for name, v in sp.items():
        out[name] = dict(name=name, pot=v["pot"], stack=v["stack"], freq=v["freq"],
                         oop=v["oop"], ip=v["ip"], oop_name=preflop.POS[v["oop"]], ip_name=preflop.POS[v["ip"]],
                         c_oop=combos_from_classes(v["r_oop"]), c_ip=combos_from_classes(v["r_ip"]),
                         r_oop=v["r_oop"], r_ip=v["r_ip"], path=v["path"])
    return out


def realization_from(spots, results, flops_w, E1326):
    """Per spot / player / class: EV_flop / (pot * preflop all-in equity)."""
    M = nonconflict_matrix().astype(np.float64)
    onehot = np.zeros((NH, 169))
    onehot[np.arange(NH), HAND_CLASS] = 1
    real = {}
    for name, sp in spots.items():
        real[name] = {}
        for side, (me, opp_c) in enumerate([(sp["oop"], sp["c_ip"]), (sp["ip"], sp["c_oop"])]):
            num = np.zeros(NH)
            den = np.zeros(NH)
            for fl, w in flops_w:
                r = results[fl][name]
                num += w * r["ev"][side]
                den += w * r["valid"]
            ev_c = np.where(den > 0, num / np.maximum(den, 1e-12), 0)
            oq = opp_c.astype(np.float64)
            eq_c = (E1326.astype(np.float64) @ oq) / np.maximum(M @ oq, 1e-12)
            ev_a = onehot.T @ ev_c
            eq_a = onehot.T @ (eq_c * sp["pot"])
            rf = np.where(eq_a > 1e-9, ev_a / np.maximum(eq_a, 1e-9), 1.0)
            real[name][me] = np.clip(rf, 0.05, 3.0)
    return real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flops", type=int, default=49)
    ap.add_argument("--target", type=float, default=10.0, help="flop exploitability target, %% of pot")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--pool", choices=("thread", "process"), default="thread",
                    help="thread: numpy releases the GIL in the matmuls; process: multiprocessing "
                         "(can hang on Windows when child processes may not duplicate handles)")
    a = ap.parse_args()

    os.makedirs(RES, exist_ok=True)
    log(f"=== 6-max 100bb solve: {a.flops} flops, target {a.target}% pot, {a.passes} passes ===")
    E1326, E169, W169 = equity.preflop_equity()
    reps = representative_flops(a.flops)
    flops_w = [(tuple(f), w / 22100.0) for f, w, _ in reps]
    with open(os.path.join(RES, "flop_subset.json"), "w", encoding="utf-8") as f:
        json.dump([dict(flop=flop_str(fl), weight=round(w, 5),
                        represents=[flop_str(m) for m in mem]) for fl, w, mem in reps], f, indent=1)
    log("flop subset: " + " ".join(flop_str(f) for f, _ in flops_w))
    log("precomputing flop equity matrices ...")
    equity.precompute_flops([f for f, _ in flops_w])

    real = None
    for p in range(1, a.passes + 1):
        final = p == a.passes
        tag = f"pass {p}/{a.passes}"
        ps = run_preflop(E169, W169, real, tag)
        save_preflop(ps, os.path.join(RES, "preflop") if final else os.path.join(RES, f"pass{p}", "preflop"))
        spots = make_spots(ps)
        log(f"[{tag}] {len(spots)} flop spots: " + ", ".join(sorted(spots)))
        out_dir = os.path.join(RES, "flop") if final else None
        light = {k: {kk: vv for kk, vv in v.items() if kk not in ("r_oop", "r_ip")} for k, v in spots.items()}
        tasks = [(fl, light, a.target, out_dir) for fl, _ in flops_w]
        results = {}
        t0 = time.time()
        for k, (fl, res) in enumerate(run_tasks(tasks, a.workers, a.pool)):
            results[fl] = res
            worst = max(r["expl"] for r in res.values())
            secs = sum(r["secs"] for r in res.values())
            log(f"[{tag}] flop {k+1}/{len(tasks)} {flop_str(fl)} done "
                f"({secs:.0f}s cpu, worst exploitability {worst:.2f}% pot, elapsed {time.time()-t0:.0f}s)")
        real = realization_from(spots, results, flops_w, E1326)
        write_summary(spots, results, flops_w, real,
                      os.path.join(RES, "summary") if final else os.path.join(RES, f"pass{p}", "summary"))
    log("=== done ===")


def write_summary(spots, results, flops_w, real, out):
    import csv
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "spots.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["spot", "OOP", "IP", "pot_bb", "stack_bb", "reach_freq_%", "OOP_combos", "IP_combos",
                    "avg_EV_OOP_bb", "avg_EV_IP_bb", "max_exploitability_%pot", "preflop_line"])
        for name in sorted(spots, key=lambda n: -spots[n]["freq"]):
            sp = spots[name]
            ev0 = sum(wt * results[fl][name]["ev_oop"] for fl, wt in flops_w)
            ev1 = sum(wt * results[fl][name]["ev_ip"] for fl, wt in flops_w)
            w.writerow([name, sp["oop_name"], sp["ip_name"], sp["pot"], sp["stack"], round(sp["freq"] * 100, 3),
                        round(float(CLASS_COMBOS @ sp["r_oop"]), 1), round(float(CLASS_COMBOS @ sp["r_ip"]), 1),
                        round(ev0, 3), round(ev1, 3),
                        round(max(results[fl][name]["expl"] for fl, _ in flops_w), 2), sp["path"]])
    with open(os.path.join(out, "flop_root_strategies.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["spot", "flop", "flop_weight", "node", "actor", "action", "freq_%", "exploitability_%pot"])
        for name in sorted(spots):
            for fl, wt in flops_w:
                r = results[fl][name]
                for node, acts in r["root"].items():
                    actor = "OOP" if node == "root" or len(node.split()) == 2 else "IP"
                    for lab, fq in acts.items():
                        w.writerow([name, flop_str(fl), round(wt, 5), node, actor, lab, round(fq * 100, 1),
                                    round(r["expl"], 2)])
    with open(os.path.join(out, "realization.json"), "w", encoding="utf-8") as f:
        json.dump({s: {preflop.POS[p]: {CLASSES[c]: round(float(v[c]), 3) for c in range(169)}
                       for p, v in d.items()} for s, d in real.items()}, f)


if __name__ == "__main__":
    main()
