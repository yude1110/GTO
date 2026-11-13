// Fast 7-card hand evaluator + equity matrix builders for the GTO solver.
// Card encoding: c = rank*4 + suit, rank 0..12 = 2..A, suit 0..3.
// Hand (hole cards) index: enumerate c1 < c2 in lexicographic order -> 0..1325.
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#define EXPORT __declspec(dllexport)
#define NH 1326

static int HC1[NH], HC2[NH];
static uint64_t HMASK[NH];
static int inited = 0;

static void init_hands(void) {
    if (inited) return;
    int k = 0;
    for (int a = 0; a < 52; a++)
        for (int b = a + 1; b < 52; b++) {
            HC1[k] = a; HC2[k] = b;
            HMASK[k] = (1ULL << a) | (1ULL << b);
            k++;
        }
    inited = 1;
}

static inline int highbit(uint32_t x) { return 31 - __builtin_clz(x); }

// returns highest card index (0..12) of a straight within 13-bit mask, or -1
static inline int straight_high(uint32_t m) {
    uint32_t w = (m << 1) | ((m >> 12) & 1);  // bit0 = ace-low
    uint32_t s = w & (w >> 1) & (w >> 2) & (w >> 3) & (w >> 4);
    if (!s) return -1;
    return highbit(s) + 3;  // top bit position in w is highbit(s)+4 -> rank = that-1
}

static inline uint32_t top_n(uint32_t m, int n) {
    uint32_t v = 0;
    for (int i = 0; i < n && m; i++) {
        int h = highbit(m);
        v = (v << 4) | (uint32_t)h;
        m &= ~(1u << h);
    }
    return v;
}

// Evaluate 7 cards; larger is better.
static uint32_t eval7(const int *c) {
    uint32_t sm[4] = {0, 0, 0, 0};
    int cnt[13] = {0};
    for (int i = 0; i < 7; i++) {
        sm[c[i] & 3] |= 1u << (c[i] >> 2);
        cnt[c[i] >> 2]++;
    }
    for (int s = 0; s < 4; s++) {
        if (__builtin_popcount(sm[s]) >= 5) {
            int sh = straight_high(sm[s]);
            if (sh >= 0) return (8u << 20) | (uint32_t)sh;
            // 5+ suited cards are distinct ranks, so the 2 others cannot make quads or a full house
            return (5u << 20) | top_n(sm[s], 5);
        }
    }
    uint32_t rm = sm[0] | sm[1] | sm[2] | sm[3];
    int quad = -1, trips[2] = {-1, -1}, nt = 0, pairs[3] = {-1, -1, -1}, np = 0;
    for (int r = 12; r >= 0; r--) {
        if (cnt[r] == 4) quad = r;
        else if (cnt[r] == 3) { if (nt < 2) trips[nt++] = r; }
        else if (cnt[r] == 2) { if (np < 3) pairs[np++] = r; }
    }
    if (quad >= 0) {
        uint32_t k = rm & ~(1u << quad);
        return (7u << 20) | ((uint32_t)quad << 4) | (uint32_t)highbit(k);
    }
    if (nt >= 1 && (nt >= 2 || np >= 1)) {
        int p = (nt >= 2) ? trips[1] : pairs[0];
        if (nt >= 2 && np >= 1 && pairs[0] > p) p = pairs[0];
        return (6u << 20) | ((uint32_t)trips[0] << 4) | (uint32_t)p;
    }
    int sh = straight_high(rm);
    if (sh >= 0) return (4u << 20) | (uint32_t)sh;
    if (nt == 1) {
        uint32_t k = rm & ~(1u << trips[0]);
        return (3u << 20) | ((uint32_t)trips[0] << 8) | top_n(k, 2);
    }
    if (np >= 2) {
        uint32_t k = rm & ~(1u << pairs[0]) & ~(1u << pairs[1]);
        return (2u << 20) | ((uint32_t)pairs[0] << 8) | ((uint32_t)pairs[1] << 4) | (uint32_t)highbit(k);
    }
    if (np == 1) {
        uint32_t k = rm & ~(1u << pairs[0]);
        return (1u << 20) | ((uint32_t)pairs[0] << 12) | top_n(k, 3);
    }
    return top_n(rm, 5);
}

EXPORT uint32_t eval7_py(const int *c) { return eval7(c); }

EXPORT void hand_cards(int *c1, int *c2) {
    init_hands();
    memcpy(c1, HC1, sizeof(HC1));
    memcpy(c2, HC2, sizeof(HC2));
}

// Pairwise accumulation for one complete 5-card board.
// acc[i*NH+j] += 2 if i beats j, 1 each on tie. cnt (optional) += 1 for valid pairs.
static void accumulate(const int *board, uint64_t bmask, uint32_t *rank, int *valid, int *nvalid_out,
                       uint16_t *acc16, uint32_t *acc32, uint32_t *cnt) {
    int nv = 0;
    int c[7];
    memcpy(c, board, 5 * sizeof(int));
    for (int h = 0; h < NH; h++) {
        if (HMASK[h] & bmask) continue;
        c[5] = HC1[h]; c[6] = HC2[h];
        rank[h] = eval7(c);
        valid[nv++] = h;
    }
    for (int a = 0; a < nv; a++) {
        int i = valid[a];
        uint64_t mi = HMASK[i];
        uint32_t ri = rank[i];
        size_t rowi = (size_t)i * NH;
        for (int b = a + 1; b < nv; b++) {
            int j = valid[b];
            if (HMASK[j] & mi) continue;
            uint32_t rj = rank[j];
            if (acc16) {
                if (ri > rj) acc16[rowi + j] += 2;
                else if (ri < rj) acc16[(size_t)j * NH + i] += 2;
                else { acc16[rowi + j] += 1; acc16[(size_t)j * NH + i] += 1; }
            } else {
                if (ri > rj) acc32[rowi + j] += 2;
                else if (ri < rj) acc32[(size_t)j * NH + i] += 2;
                else { acc32[rowi + j] += 1; acc32[(size_t)j * NH + i] += 1; }
                cnt[rowi + j] += 1;
            }
        }
    }
    *nvalid_out = nv;
}

// Equity matrix for a flop, enumerating all turn/river runouts.
// out[i*NH+j] = equity of hand i vs hand j (0 if conflicting with each other or the flop).
EXPORT void flop_equity(int f0, int f1, int f2, float *out) {
    init_hands();
    uint16_t *acc = (uint16_t *)calloc((size_t)NH * NH, sizeof(uint16_t));
    uint32_t *rank = (uint32_t *)malloc(NH * sizeof(uint32_t));
    int *valid = (int *)malloc(NH * sizeof(int));
    uint64_t fm = (1ULL << f0) | (1ULL << f1) | (1ULL << f2);
    int board[5] = {f0, f1, f2, 0, 0};
    int nv;
    for (int t = 0; t < 52; t++) {
        if (fm >> t & 1) continue;
        for (int r = t + 1; r < 52; r++) {
            if (fm >> r & 1) continue;
            board[3] = t; board[4] = r;
            accumulate(board, fm | (1ULL << t) | (1ULL << r), rank, valid, &nv, acc, NULL, NULL);
        }
    }
    // every non-conflicting pair sees exactly C(45,2)=990 runouts
    const float scale = 1.0f / (2.0f * 990.0f);
    for (size_t k = 0; k < (size_t)NH * NH; k++) out[k] = acc[k] * scale;
    free(acc); free(rank); free(valid);
}

static uint64_t rng_next(uint64_t *s) {  // splitmix64
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Monte Carlo preflop all-in equity accumulation over random boards.
// wins[i*NH+j] (2x scaled) and cnt[i*NH+j] (i<j only) accumulate.
EXPORT void preflop_mc(uint64_t seed, int nboards, uint32_t *wins, uint32_t *cnt) {
    init_hands();
    uint32_t *rank = (uint32_t *)malloc(NH * sizeof(uint32_t));
    int *valid = (int *)malloc(NH * sizeof(int));
    uint64_t s = seed;
    int nv;
    for (int n = 0; n < nboards; n++) {
        int board[5];
        uint64_t bm = 0;
        for (int k = 0; k < 5;) {
            int c = (int)(rng_next(&s) % 52);
            if (bm >> c & 1) continue;
            bm |= 1ULL << c;
            board[k++] = c;
        }
        accumulate(board, bm, rank, valid, &nv, NULL, wins, cnt);
    }
    free(rank); free(valid);
}
