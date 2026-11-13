"""Heads-up flop CFR solver (vectorised Discounted CFR over 1326 combos).

The flop is played out with the bet sizes 33/50/75/100% pot + all-in (raises use the
same pot fractions). When flop action closes (check-check or a call), the hand is settled
by all-in equity over every turn/river runout, so turn/river play is not modelled.

Utility convention: chips won relative to the start of the flop (the preflop pot P is
owned by nobody), so OOP + IP utilities always sum to P.
"""
import numpy as np

from cards import NH, hands_blocked_by, nonconflict_matrix

OOP, IP = 0, 1
BET_PCTS = (0.33, 0.50, 0.75, 1.00)

_NC = None


def nc_matrix():
    global _NC
    if _NC is None:
        _NC = nonconflict_matrix()
    return _NC


class Node:
    __slots__ = ("id", "kind", "player", "c", "children", "labels", "winner", "nbets", "path")

    def __init__(self, kind, player, c, nbets, path):
        self.kind = kind          # "dec" | "show" | "fold"
        self.player = player      # actor at decision nodes
        self.c = c                # (oop_contrib, ip_contrib) on the flop
        self.children = []
        self.labels = []
        self.winner = None        # fold terminals
        self.nbets = nbets
        self.path = path


def build_tree(pot, stack, pcts=BET_PCTS, max_full_raises=3):
    """pot = preflop pot, stack = effective remaining stack at flop start."""
    nodes = []

    def add(n):
        n.id = len(nodes)
        nodes.append(n)
        return n

    def pct_label(p):
        return str(int(round(p * 100)))

    def sizes(c, me, nbets):
        opp = 1 - me
        hi = max(c)
        if hi >= stack:
            return []
        out = []
        if nbets < max_full_raises:
            for p in pcts:
                if hi == 0:   # bet
                    tgt = p * (pot + c[0] + c[1])
                else:         # raise to: call + p * pot-after-call
                    tgt = hi + p * (pot + 2 * hi)
                if tgt >= stack * 0.999:
                    continue
                out.append((("B" if hi == 0 else "R") + pct_label(p), tgt))
        out.append(("A", float(stack)))
        return out

    def decide(player, c, nbets, path, checked_back=False):
        n = add(Node("dec", player, c, nbets, path))
        opp = 1 - player
        to_call = c[opp] - c[player]
        if to_call > 0:
            # fold
            f = add(Node("fold", None, c, nbets, path + ["F"]))
            f.winner = opp
            n.children.append(f); n.labels.append("F")
            cc = list(c); cc[player] = c[opp]
            n.children.append(add(Node("show", None, tuple(cc), nbets, path + ["C"]))); n.labels.append("C")
        else:
            if player == OOP:
                n.children.append(decide(IP, c, nbets, path + ["X"])); n.labels.append("X")
            else:
                n.children.append(add(Node("show", None, c, nbets, path + ["X"]))); n.labels.append("X")
        for lab, tgt in sizes(c, player, nbets):
            cc = list(c); cc[player] = tgt
            n.children.append(decide(opp, tuple(cc), nbets + 1, path + [lab])); n.labels.append(lab)
        return n

    decide(OOP, (0.0, 0.0), 0, [])
    return nodes


class FlopSolver:
    def __init__(self, E, pot, stack, range_oop, range_ip, flop, pcts=BET_PCTS):
        """E: 1326x1326 equity matrix; ranges: per-combo weights (1326) before flop removal."""
        self.E = E
        self.M = nc_matrix()
        self.pot = float(pot)
        self.stack = float(stack)
        blocked = hands_blocked_by(flop)
        r0 = np.asarray(range_oop, dtype=np.float32).copy(); r0[blocked] = 0
        r1 = np.asarray(range_ip, dtype=np.float32).copy(); r1[blocked] = 0
        self.ranges = (r0, r1)
        self.nodes = build_tree(pot, stack, pcts)
        self.dec = [n for n in self.nodes if n.kind == "dec"]
        self.term = [n for n in self.nodes if n.kind != "dec"]
        self.tidx = {n.id: k for k, n in enumerate(self.term)}
        T = len(self.term)
        # coefficients: value_p[t] = A[p,t] * (E @ r_q) + B[p,t] * (M @ r_q)
        A = np.zeros((2, T), np.float32)
        B = np.zeros((2, T), np.float32)
        for k, t in enumerate(self.term):
            for p in (0, 1):
                if t.kind == "show":
                    A[p, k] = self.pot + t.c[0] + t.c[1]
                    B[p, k] = -t.c[p]
                else:
                    B[p, k] = (self.pot + t.c[1 - p]) if t.winner == p else -t.c[p]
        self.A, self.B = A, B
        self.regret = {n.id: np.zeros((len(n.children), NH), np.float32) for n in self.dec}
        self.ssum = {n.id: np.zeros((len(n.children), NH), np.float32) for n in self.dec}
        self.it = 0

    # ---------- helpers ----------
    @staticmethod
    def _rm(reg):
        pos = np.maximum(reg, 0)
        s = pos.sum(0, keepdims=True)
        A = reg.shape[0]
        return np.where(s > 0, pos / np.where(s > 0, s, 1), 1.0 / A).astype(np.float32)

    def avg_strategy(self, nid):
        s = self.ssum[nid]
        tot = s.sum(0, keepdims=True)
        A = s.shape[0]
        return np.where(tot > 0, s / np.where(tot > 0, tot, 1), 1.0 / A).astype(np.float32)

    def _forward(self, strat):
        reach = {}
        reach[self.nodes[0].id] = list(self.ranges)
        for n in self.dec:
            r = reach[n.id]
            sg = strat[n.id]
            for a, ch in enumerate(n.children):
                rr = list(r)
                rr[n.player] = r[n.player] * sg[a]
                reach[ch.id] = rr
        return reach

    def _terminal_values(self, reach):
        T = len(self.term)
        vals = {}
        for p in (0, 1):
            q = 1 - p
            R = np.empty((NH, T), np.float32)
            for k, t in enumerate(self.term):
                R[:, k] = reach[t.id][q]
            ER = self.E @ R
            MR = self.M @ R
            V = ER * self.A[p][None, :] + MR * self.B[p][None, :]
            vals[p] = V
        return vals

    def _backward(self, strat, tv, br_player=None, update=False, reach=None):
        v = {}
        for t in self.term:
            k = self.tidx[t.id]
            v[t.id] = (tv[0][:, k], tv[1][:, k])
        for n in reversed(self.dec):
            sg = strat[n.id]
            cv = [v[ch.id] for ch in n.children]
            p = n.player
            q = 1 - p
            cp = np.stack([c[p] for c in cv])
            vq = sum(c[q] for c in cv)
            if br_player == p:
                vp = cp.max(0)
            else:
                vp = (sg * cp).sum(0)
            if update:
                self.regret[n.id] += cp - vp[None, :]
            v[n.id] = (vp, vq) if p == 0 else (vq, vp)
        return v[self.nodes[0].id]

    # ---------- public ----------
    def iterate(self, n_iter, alpha=1.5, beta=0.0, gamma=2.0):
        for _ in range(n_iter):
            self.it += 1
            t = self.it
            strat = {n.id: self._rm(self.regret[n.id]) for n in self.dec}
            reach = self._forward(strat)
            tv = self._terminal_values(reach)
            self._backward(strat, tv, update=True)
            pa = t ** alpha / (t ** alpha + 1)
            nb = t ** beta / (t ** beta + 1)
            g = (t / (t + 1)) ** gamma
            for n in self.dec:
                r = self.regret[n.id]
                r *= np.where(r > 0, pa, nb).astype(np.float32)
                self.ssum[n.id] *= g
                self.ssum[n.id] += strat[n.id] * reach[n.id][n.player][None, :]

    def exploitability(self):
        """Returns (exploitability as % of pot, ev_oop, ev_ip) for the average strategy."""
        strat = {n.id: self.avg_strategy(n.id) for n in self.dec}
        reach = self._forward(strat)
        tv = self._terminal_values(reach)
        norm = float(self.ranges[0] @ self.M @ self.ranges[1])
        br = []
        for p in (0, 1):
            root = self._backward(strat, tv, br_player=p)
            br.append(float(self.ranges[p] @ root[p]) / norm)
        root = self._backward(strat, tv)
        ev = [float(self.ranges[p] @ root[p]) / norm for p in (0, 1)]
        expl = (br[0] + br[1] - self.pot) / 2
        return 100.0 * expl / self.pot, ev[0], ev[1]

    def hand_evs(self):
        """Per-combo EV (chips relative to flop start) under the average strategy, per player."""
        strat = {n.id: self.avg_strategy(n.id) for n in self.dec}
        reach = self._forward(strat)
        tv = self._terminal_values(reach)
        root = self._backward(strat, tv)
        out = []
        for p in (0, 1):
            q = 1 - p
            opp_mass = self.M @ self.ranges[q]
            out.append(np.where(opp_mass > 0, root[p] / np.maximum(opp_mass, 1e-12), 0.0))
        return out

    def solve(self, target_pct=10.0, max_iter=1000, check_every=10, log=None):
        hist = []
        while self.it < max_iter:
            self.iterate(check_every)
            e, ev0, ev1 = self.exploitability()
            hist.append((self.it, e))
            if log:
                log(f"  iter {self.it:4d}  exploitability {e:7.3f}% pot")
            if e <= target_pct:
                break
        return hist
