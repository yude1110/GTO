"""Builds/loads the C evaluator and produces cached equity matrices."""
import ctypes
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from cards import NH, HAND_CLASS, CLASS_COMBOS, flop_str, nonconflict_matrix

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
DLL = os.path.join(ROOT, "build", "eval.dll")
CACHE = os.path.join(ROOT, "cache")
THREADS = os.cpu_count() or 8


def build(force=False):
    os.makedirs(os.path.dirname(DLL), exist_ok=True)
    srcf = os.path.join(SRC, "eval.c")
    if force or not os.path.exists(DLL) or os.path.getmtime(DLL) < os.path.getmtime(srcf):
        subprocess.check_call([sys.executable, "-m", "ziglang", "cc", "-O3", "-march=native",
                               "-shared", "-o", DLL, srcf])


_lib = None
_lib_lock = threading.Lock()


def lib():
    global _lib
    with _lib_lock:
        if _lib is None:
            build()
            L = ctypes.CDLL(DLL)
            P = np.ctypeslib.ndpointer
            L.flop_equity.argtypes = [ctypes.c_int] * 3 + [P(np.float32, flags="C")]
            L.preflop_mc.argtypes = [ctypes.c_uint64, ctypes.c_int, P(np.uint32, flags="C"), P(np.uint32, flags="C")]
            L.eval7_py.argtypes = [P(np.int32, flags="C")]
            L.eval7_py.restype = ctypes.c_uint32
            _lib = L
    return _lib


def eval7(cards):
    return lib().eval7_py(np.ascontiguousarray(cards, dtype=np.int32))


def flop_equity(flop):
    """1326x1326 float32 equity matrix (row hand vs column hand), cached on disk."""
    os.makedirs(os.path.join(CACHE, "flop_eq"), exist_ok=True)
    path = os.path.join(CACHE, "flop_eq", flop_str(flop) + ".npy")
    if os.path.exists(path):
        return np.load(path)
    out = np.zeros((NH, NH), dtype=np.float32)
    lib().flop_equity(int(flop[0]), int(flop[1]), int(flop[2]), out)
    np.save(path, out)
    return out


def precompute_flops(flops, workers=THREADS):
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(flop_equity, flops))


def preflop_equity(boards=240000, seed=12345):
    """Returns (E1326 float32, E169, W169).
    E169[a,b] = equity of class a vs class b, W169[a,b] = # non-conflicting combo pairs."""
    path = os.path.join(CACHE, "preflop_eq.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["E1326"], z["E169"], z["W169"]
    os.makedirs(CACHE, exist_ok=True)
    per = boards // THREADS

    def job(t):
        w = np.zeros(NH * NH, dtype=np.uint32)
        c = np.zeros(NH * NH, dtype=np.uint32)
        lib().preflop_mc(seed + 7919 * t, per, w, c)
        return w, c

    with ThreadPoolExecutor(THREADS) as ex:
        res = list(ex.map(job, range(THREADS)))
    wins = sum(r[0].astype(np.float64) for r in res).reshape(NH, NH)
    cnt = sum(r[1].astype(np.float64) for r in res).reshape(NH, NH)
    cnt = cnt + cnt.T
    nc = nonconflict_matrix().astype(np.float64)
    E = np.where(cnt > 0, wins / (2 * np.maximum(cnt, 1)), 0.5) * nc
    # aggregate to classes
    onehot = np.zeros((NH, 169))
    onehot[np.arange(NH), HAND_CLASS] = 1
    W169 = onehot.T @ nc @ onehot
    E169 = (onehot.T @ E @ onehot) / np.maximum(W169, 1)
    E1326 = E.astype(np.float32)
    np.savez_compressed(path, E1326=E1326, E169=E169, W169=W169)
    return E1326, E169, W169
