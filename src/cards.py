"""Card / hand indexing, 169 preflop classes, flop isomorphism and the representative flop subset."""
import itertools
import numpy as np

RANKS = "23456789TJQKA"
SUITS = "cdhs"
NH = 1326


def card_str(c):
    return RANKS[c >> 2] + SUITS[c & 3]


def parse_card(s):
    return RANKS.index(s[0].upper()) * 4 + SUITS.index(s[1].lower())


# hole-card combos in the same order as eval.c (c1 < c2)
HAND_CARDS = np.array([(a, b) for a in range(52) for b in range(a + 1, 52)], dtype=np.int32)
HAND_STR = [card_str(b) + card_str(a) for a, b in HAND_CARDS]  # high card first


def class_name(a, b):
    ra, rb = a >> 2, b >> 2
    hi, lo = max(ra, rb), min(ra, rb)
    if hi == lo:
        return RANKS[hi] * 2
    return RANKS[hi] + RANKS[lo] + ("s" if (a & 3) == (b & 3) else "o")


# 169 classes in the usual 13x13 grid order (row = first rank high->low)
CLASSES = []
for i in range(12, -1, -1):
    for j in range(12, -1, -1):
        if i == j:
            CLASSES.append(RANKS[i] * 2)
        elif i > j:
            CLASSES.append(RANKS[i] + RANKS[j] + "s")
        else:
            CLASSES.append(RANKS[j] + RANKS[i] + "o")
CLASS_INDEX = {c: k for k, c in enumerate(CLASSES)}
HAND_CLASS = np.array([CLASS_INDEX[class_name(a, b)] for a, b in HAND_CARDS], dtype=np.int32)
CLASS_COMBOS = np.bincount(HAND_CLASS, minlength=169).astype(np.float64)  # 6 / 4 / 12

# hand x hand "no shared card" mask
_hm = np.zeros((NH, 52), dtype=bool)
_hm[np.arange(NH), HAND_CARDS[:, 0]] = True
_hm[np.arange(NH), HAND_CARDS[:, 1]] = True
HAND_CARD_MASK = _hm


def nonconflict_matrix():
    m = _hm.astype(np.float32)
    return (m @ m.T == 0).astype(np.float32)


def hands_blocked_by(cards):
    """bool[1326]: hand uses one of the given cards."""
    return _hm[:, list(cards)].any(axis=1)


# ---------------- flops ----------------
_PERMS = list(itertools.permutations(range(4)))


def canonical_flop(cards):
    best = None
    for p in _PERMS:
        t = tuple(sorted(((c >> 2) * 4 + p[c & 3] for c in cards), reverse=True))
        if best is None or t < best:
            best = t
    return best


def all_canonical_flops():
    """dict canonical_flop -> number of raw flops (sums to 22100)."""
    out = {}
    for f in itertools.combinations(range(52), 3):
        k = canonical_flop(f)
        out[k] = out.get(k, 0) + 1
    return out


def flop_str(f):
    return "".join(card_str(c) for c in sorted(f, key=lambda c: (-(c >> 2), c & 3)))


def flop_features(f):
    rs = sorted((c >> 2 for c in f), reverse=True)
    suits = [c & 3 for c in sorted(f, key=lambda c: -(c >> 2))]
    ns = len(set(suits))
    mono = ns == 1
    two = ns == 2
    paired = len(set(rs)) < 3
    # which two cards share the suit on two-tone boards (high-mid / high-low / mid-low)
    tt = [0.0, 0.0, 0.0]
    if two and not paired:
        if suits[0] == suits[1]:
            tt[0] = 1
        elif suits[0] == suits[2]:
            tt[1] = 1
        else:
            tt[2] = 1
    gap = (rs[0] - rs[2]) / 12.0
    return np.array([rs[0] / 12 * 1.6, rs[1] / 12 * 1.3, rs[2] / 12, gap * 0.5,
                     mono * 3.0, two * 2.0, paired * 3.0, *(np.array(tt) * 0.4)])


def representative_flops(k=49, seed=0, iters=60):
    """Weighted k-medoids on flop texture features. Returns list of (flop, weight, members)."""
    cf = all_canonical_flops()
    flops = list(cf.keys())
    w = np.array([cf[f] for f in flops], dtype=np.float64)
    X = np.stack([flop_features(f) for f in flops])
    rng = np.random.default_rng(seed)
    # k-means++ init (weighted)
    centers = [int(np.argmax(w))]
    d2 = ((X - X[centers[0]]) ** 2).sum(1)
    for _ in range(1, k):
        p = w * d2
        centers.append(int(rng.choice(len(flops), p=p / p.sum())))
        d2 = np.minimum(d2, ((X - X[centers[-1]]) ** 2).sum(1))
    centers = np.array(centers)
    for _ in range(iters):
        D = ((X[:, None, :] - X[None, centers, :]) ** 2).sum(2)
        lab = D.argmin(1)
        new = centers.copy()
        for c in range(k):
            idx = np.where(lab == c)[0]
            if len(idx) == 0:
                continue
            sub = X[idx]
            cost = (((sub[:, None, :] - sub[None, :, :]) ** 2).sum(2) * w[idx][None, :]).sum(1)
            new[c] = idx[cost.argmin()]
        if np.array_equal(new, centers):
            break
        centers = new
    D = ((X[:, None, :] - X[None, centers, :]) ** 2).sum(2)
    lab = D.argmin(1)
    out = []
    for c in range(k):
        idx = np.where(lab == c)[0]
        out.append((flops[centers[c]], float(w[idx].sum()), [flops[i] for i in idx]))
    out.sort(key=lambda t: canonical_sort_key(t[0]))
    return out


def canonical_sort_key(f):
    rs = sorted((c >> 2 for c in f), reverse=True)
    return (-rs[0], -rs[1], -rs[2], len(set(c & 3 for c in f)))
