"""6-max preflop CFR over the 169 starting-hand classes (100bb, blinds 0.5/1, no ante).

Tree (keeps every flop heads-up):
  unopened : fold / raise (2.5bb, SB 3bb)                       -- no limping
  vs open  : fold / 3bet (+ call for the BB only)
  vs 3bet  : opener fold / call / 4bet / all-in  (other players fold)
  vs 4bet  : 3bettor fold / call / all-in
  vs jam   : fold / call
Pots that reach the flop are valued with all-in equity x pot, optionally multiplied by
per-hand equity-realisation factors taken from the flop solutions (second pass).
"""
import numpy as np

from cards import CLASS_COMBOS, CLASSES

POS = ["UTG", "HJ", "CO", "BTN", "SB", "BB"]
NP = 6
STACK = 100.0
SB, BB = 4, 5
POSTFLOP_ORDER = [SB, BB, 0, 1, 2, 3]  # first to act postflop first


def oop_of(a, b):
    return a if POSTFLOP_ORDER.index(a) < POSTFLOP_ORDER.index(b) else b


def open_size(o):
    return 3.0 if o == SB else 2.5


def threebet_size(o, t):
    if t == BB and o == SB:
        return 9.0
    if t in (SB, BB):
        return 11.0
    return 3.0 * open_size(o)


def fourbet_size(o, t, three):
    o_is_oop = oop_of(o, t) == o
    return float(round((2.5 if o_is_oop else 2.2) * three))


def fmt(x):
    return f"{x:g}"


class PNode:
    __slots__ = ("id", "kind", "actor", "inv", "children", "labels", "winner", "pair", "spot", "path")

    def __init__(self, kind, actor, inv, path):
        self.kind, self.actor, self.inv, self.path = kind, actor, tuple(inv), path
        self.children, self.labels = [], []
        self.winner = self.pair = self.spot = None


def build_tree():
    nodes = []

    def add(n):
        n.id = len(nodes)
        nodes.append(n)
        return n

    def fold_term(inv, winner, path):
        n = add(PNode("fold", None, inv, path))
        n.winner = winner
        return n

    def show_term(inv, a, b, spot, path):
        n = add(PNode("show", None, inv, path))
        n.pair = (a, b)
        n.spot = spot
        return n

    def dec(actor, inv, path, opts):
        n = add(PNode("dec", actor, inv, path))
        for lab, fn in opts:
            n.labels.append(lab)
            n.children.append(fn(path + [f"{POS[actor]}:{lab}"]))
        return n

    def with_inv(inv, k, amt):
        v = list(inv)
        v[k] = amt
        return v

    def unopened(i, inv, path):
        if i == BB:
            return fold_term(inv, BB, path)
        o = i
        return dec(i, inv, path, [
            ("F", lambda p: unopened(i + 1, inv, p)),
            (f"R{fmt(open_size(o))}", lambda p: responders(o, o + 1, with_inv(inv, o, open_size(o)), p)),
        ])

    def responders(o, j, inv, path):
        if j > BB:
            return fold_term(inv, o, path)
        three = threebet_size(o, j)
        opts = [("F", lambda p: responders(o, j + 1, inv, p))]
        if j == BB:
            opts.append(("C", lambda p: show_term(with_inv(inv, BB, open_size(o)), o, BB,
                                                  f"SRP_{POS[o]}_vs_BB", p)))
        opts.append((f"R{fmt(three)}", lambda p: vs3bet(o, j, with_inv(inv, j, three), p)))
        return dec(j, inv, path, opts)

    def vs3bet(o, t, inv, path):
        three = inv[t]
        four = fourbet_size(o, t, three)
        spot = f"3BP_{POS[o]}_vs_{POS[t]}"
        return dec(o, inv, path, [
            ("F", lambda p: fold_term(inv, t, p)),
            ("C", lambda p: show_term(with_inv(inv, o, three), o, t, spot, p)),
            (f"R{fmt(four)}", lambda p: vs4bet(o, t, with_inv(inv, o, four), p)),
            ("A", lambda p: vsjam(t, o, with_inv(inv, o, STACK), p)),
        ])

    def vs4bet(o, t, inv, path):
        four = inv[o]
        spot = f"4BP_{POS[o]}_vs_{POS[t]}"
        return dec(t, inv, path, [
            ("F", lambda p: fold_term(inv, o, p)),
            ("C", lambda p: show_term(with_inv(inv, t, four), o, t, spot, p)),
            ("A", lambda p: vsjam(o, t, with_inv(inv, t, STACK), p)),
        ])

    def vsjam(actor, jammer, inv, path):
        return dec(actor, inv, path, [
            ("F", lambda p: fold_term(inv, jammer, p)),
            ("C", lambda p: show_term(with_inv(inv, actor, STACK), actor, jammer, "ALLIN", p)),
        ])

    unopened(0, [0, 0, 0, 0, 0.5, 1.0], [])
    return nodes


class PreflopSolver:
    def __init__(self, E169, W169, realization=None):
        """realization: dict spot -> {player_pos_index: array(169)} multiplying equity share."""
        self.nodes = build_tree()
        self.dec = [n for n in self.nodes if n.kind == "dec"]
        self.term = [n for n in self.nodes if n.kind != "dec"]
        n = CLASS_COMBOS
        self.n = n
        self.Wn = W169 / (n[:, None] * 1225.0)        # conditional pair weights, rows sum to 1
        self.WE = self.Wn * E169
        self.E169 = E169
        self.real = realization or {}
        self._ws = {}
        self.regret = {d.id: np.zeros((len(d.children), 169)) for d in self.dec}
        self.ssum = {d.id: np.zeros((len(d.children), 169)) for d in self.dec}
        self.it = 0

    def _share_matrix(self, spot, k, q):
        """Wn * pot-share of k vs q. With realisation factors R the share of the pot is
        E*Rk / (E*Rk + (1-E)*Rq), which keeps the two players' shares summing to the pot."""
        key = (spot, k)
        if key not in self._ws:
            rk = self.real.get(spot, {}).get(k)
            rq = self.real.get(spot, {}).get(q)
            if rk is None or rq is None:
                self._ws[key] = self.WE
            else:
                E = self.E169
                num = E * rk[:, None]
                den = num + (1 - E) * rq[None, :]
                S = np.where(den > 1e-12, num / np.maximum(den, 1e-12), E)
                self._ws[key] = self.Wn * S
        return self._ws[key]

    def mass(self, r):
        return float(self.n @ r) / 1326.0

    @staticmethod
    def _rm(reg):
        pos = np.maximum(reg, 0)
        s = pos.sum(0, keepdims=True)
        return np.where(s > 0, pos / np.where(s > 0, s, 1), 1.0 / reg.shape[0])

    def avg_strategy(self, nid):
        s = self.ssum[nid]
        tot = s.sum(0, keepdims=True)
        return np.where(tot > 0, s / np.where(tot > 0, tot, 1), 1.0 / s.shape[0])

    def _forward(self, strat):
        reach = {self.nodes[0].id: [np.ones(169) for _ in range(NP)]}
        for d in self.dec:
            r = reach[d.id]
            for a, ch in enumerate(d.children):
                rr = list(r)
                rr[d.actor] = r[d.actor] * strat[d.id][a]
                reach[ch.id] = rr
        return reach

    def _term_values(self, t, r):
        masses = [self.mass(x) for x in r]
        inv = t.inv
        out = []
        if t.kind == "fold":
            pot = sum(inv)
            for k in range(NP):
                others = np.prod([masses[j] for j in range(NP) if j != k])
                pay = (pot - inv[k]) if k == t.winner else -inv[k]
                out.append(np.full(169, pay * others))
            return out
        a, b = t.pair
        pot = sum(inv)
        third = np.prod([masses[j] for j in range(NP) if j not in (a, b)])
        for k in range(NP):
            if k in (a, b):
                q = b if k == a else a
                rq = r[q]
                share = pot * (self._share_matrix(t.spot, k, q) @ rq)
                out.append(third * (share - inv[k] * (self.Wn @ rq)))
            else:
                others = np.prod([masses[j] for j in range(NP) if j != k])
                out.append(np.full(169, -inv[k] * others))
        return out

    def _backward(self, strat, reach, br_player=None, update=False):
        v = {t.id: self._term_values(t, reach[t.id]) for t in self.term}
        for d in reversed(self.dec):
            cv = [v[ch.id] for ch in d.children]
            p = d.actor
            vals = []
            for k in range(NP):
                if k == p:
                    cp = np.stack([c[p] for c in cv])
                    vp = cp.max(0) if br_player == p else (strat[d.id] * cp).sum(0)
                    if update:
                        self.regret[d.id] += cp - vp[None, :]
                    vals.append(vp)
                else:
                    vals.append(sum(c[k] for c in cv))
            v[d.id] = vals
        return v[self.nodes[0].id]

    def iterate(self, n_iter, alpha=1.5, beta=0.0, gamma=2.0):
        for _ in range(n_iter):
            self.it += 1
            t = self.it
            strat = {d.id: self._rm(self.regret[d.id]) for d in self.dec}
            reach = self._forward(strat)
            self._backward(strat, reach, update=True)
            pa = t ** alpha / (t ** alpha + 1)
            nb = t ** beta / (t ** beta + 1)
            g = (t / (t + 1)) ** gamma
            for d in self.dec:
                r = self.regret[d.id]
                r *= np.where(r > 0, pa, nb)
                self.ssum[d.id] *= g
                self.ssum[d.id] += strat[d.id] * reach[d.id][d.actor][None, :]

    def evaluate(self):
        """Returns (ev per player in bb, best-response gain per player in bb)."""
        strat = {d.id: self.avg_strategy(d.id) for d in self.dec}
        reach = self._forward(strat)
        root = self._backward(strat, reach)
        w = self.n / 1326.0
        ev = [float(w @ root[k]) for k in range(NP)]
        gain = []
        for k in range(NP):
            br = self._backward(strat, reach, br_player=k)
            gain.append(float(w @ br[k]) - ev[k])
        return ev, gain

    def solve(self, iters=4000, check_every=500, log=None):
        while self.it < iters:
            self.iterate(check_every)
            if log:
                ev, gain = self.evaluate()
                log(f"  preflop iter {self.it:5d}  max BR gain {max(gain):.4f}bb  "
                    f"EV " + " ".join(f"{POS[k]}={ev[k]:+.3f}" for k in range(NP)))

    def spots(self):
        """Terminal flop spots with per-class reach of both players (average strategy).
        Returns dict spot -> dict(oop, ip, pot, stack, freq, r_oop, r_ip)."""
        strat = {d.id: self.avg_strategy(d.id) for d in self.dec}
        reach = self._forward(strat)
        out = {}
        for t in self.term:
            if t.kind != "show" or t.spot == "ALLIN":
                continue
            a, b = t.pair
            o = oop_of(a, b)
            i = b if o == a else a
            r = reach[t.id]
            freq = np.prod([self.mass(x) for x in r])
            pot = sum(t.inv)
            out[t.spot] = dict(oop=o, ip=i, pot=pot, stack=STACK - t.inv[a], freq=freq,
                               r_oop=r[o].copy(), r_ip=r[i].copy(), path=" ".join(t.path))
        return out

    def strategy_table(self):
        out = []
        reach = self._forward({d.id: self.avg_strategy(d.id) for d in self.dec})
        for d in self.dec:
            st = self.avg_strategy(d.id)
            r = reach[d.id][d.actor]
            out.append(dict(path=" ".join(d.path) or "(start)", actor=POS[d.actor], labels=d.labels,
                            strategy={CLASSES[c]: [float(x) for x in st[:, c]] for c in range(169)},
                            reach={CLASSES[c]: float(r[c]) for c in range(169)},
                            node_reach=float(np.prod([self.mass(x) for j, x in enumerate(reach[d.id]) if j != d.actor]))))
        return out
