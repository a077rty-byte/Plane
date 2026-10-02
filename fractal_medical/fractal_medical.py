#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural based domain and range pool partitioning using Fractal Coding
for nearly lossless medical image compression.

Implementation of the three coders compared in:
    S. Bhavani, K. Thanushkodi, "Neural based domain and range pool
    partitioning using Fractal Coding for nearly lossless Medical Image
    Compression", WSEAS Trans. on Signal Processing, 9(1), 2013.

  * Standard fractal coding  : fixed range blocks, full domain search.
  * Proposed Algorithm I     : quad-tree decomposition + variance based
                               domain/range separation (threshold tau).
                               Domain blocks are stored LOSSLESSLY (seeds),
                               the rest of the image is generated with IFS.
  * Proposed Algorithm II    : Algorithm I + Self-Organising-Map (Kohonen)
                               grouping of domain and range blocks; a range
                               block is only matched against the domains of
                               its own group (fast fractal coding).

Outputs (same structure as the paper):
    Table 1  PSNR at different thresholds
    Table 2  Encoding time at different thresholds
    Table 3  Compression ratio at different thresholds
    Table 4  PSNR / time / compression ratio of the three algorithms
    Fig. 1/2 partition + separated domain blocks (several thresholds)
    Fig. 3   performance analysis ; Fig. 6/7/8 comparison bar charts
"""
import argparse
import math
import os
import time
import zlib

import numpy as np

# --------------------------------------------------------------------------
# Global coding parameters
# --------------------------------------------------------------------------
S_LEVELS = np.linspace(-0.95, 0.95, 31)   # quantised scaling s_i (5 bits, |s|<1)
S_BITS, O_BITS, ISO_BITS = 5, 9, 3        # bits for s_i, o_i, isometry
TAU_LIST = [1e-3, 1e-4, 1e-5, 1e-6]


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------
def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def isometry(blocks, k):
    """Apply one of the 8 square isometries to an array (..., r, r)."""
    if k >= 4:
        blocks = blocks[..., :, ::-1]
    return np.rot90(blocks, k % 4, axes=(-2, -1))


def pool2(img):
    """P[y, x] = mean of img[y:y+2, x:x+2]  (valid positions only)."""
    return 0.25 * (img[:-1, :-1] + img[1:, :-1] + img[:-1, 1:] + img[1:, 1:])


def load_image(path, size):
    from PIL import Image
    im = Image.open(path).convert("L")
    if im.size != (size, size):
        im = im.resize((size, size), Image.BICUBIC)
    return np.asarray(im, dtype=np.float64)


# --------------------------------------------------------------------------
# Synthetic MRI-like head phantoms (used when no real MRI is supplied)
# --------------------------------------------------------------------------
def make_phantom(idx, size=256):
    from scipy.ndimage import gaussian_filter
    rng = np.random.RandomState(100 + idx)
    yy, xx = np.mgrid[0:size, 0:size] / float(size)
    cy, cx = 0.5 + 0.02 * rng.randn(), 0.5 + 0.02 * rng.randn()
    ry, rx = 0.40 + 0.03 * rng.rand(), 0.33 + 0.03 * rng.rand()
    rr = np.sqrt(((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2)
    img = np.zeros((size, size))
    skull = (rr < 1.0) & (rr >= 0.88)
    brain = rr < 0.88
    tex = gaussian_filter(rng.randn(size, size), 1.2)
    tex2 = gaussian_filter(rng.randn(size, size), 4.0)
    img[skull] = 150 + 40 * tex[skull] / tex.std()
    img[brain] = 95 + 18 * tex2[brain] / tex2.std() + 14 * tex[brain] / tex.std()
    # gyri-like folds: dark sulci
    folds = gaussian_filter(rng.randn(size, size), 2.5)
    sulci = brain & (np.abs(folds) < 0.10 * folds.std()) & (rr > 0.35)
    img[sulci] = 40
    # ventricles / eyes / nasal structures depending on the image
    for _ in range(2 + idx % 3):
        vy = cy + (rng.rand() - 0.5) * 0.25
        vx = cx + (rng.rand() - 0.5) * 0.25
        a, b = 0.03 + 0.05 * rng.rand(), 0.02 + 0.03 * rng.rand()
        m = (((yy - vy) / a) ** 2 + ((xx - vx) / b) ** 2) < 1
        img[m & brain] = 25 + 10 * rng.rand()
    if idx % 2 == 1:   # bright lesion
        ly, lx = cy + 0.15 * rng.randn() * 0.3, cx + 0.15 * rng.randn() * 0.3
        m = np.sqrt((yy - ly) ** 2 + (xx - lx) ** 2) < 0.035
        img[m & brain] = 220
    img = gaussian_filter(img, 0.8)
    # Rician-like noise (low in background, as in real MRI)
    n1, n2 = rng.randn(size, size), rng.randn(size, size)
    img = np.sqrt((img + 4 * n1) ** 2 + (4 * n2) ** 2)
    img = np.where(rr > 1.15, np.abs(3 * n1), img)
    return np.clip(np.rint(img), 0, 255)


# --------------------------------------------------------------------------
# Quad-tree decomposition + domain / range separation  (Section 3.1)
# --------------------------------------------------------------------------
def quadtree(img, max_size, min_size, split_var):
    """Variance driven quad-tree; returns list of leaves (y, x, size)."""
    leaves = []

    def rec(y, x, s):
        blk = img[y:y + s, x:x + s]
        if s > min_size and (s > max_size or blk.var() > split_var):
            h = s // 2
            for dy in (0, h):
                for dx in (0, h):
                    rec(y + dy, x + dx, h)
        else:
            leaves.append((y, x, s))

    n = img.shape[0]
    top = max_size
    for y in range(0, n, top):
        for x in range(0, n, top):
            rec(y, x, top)
    return leaves


def separate_domain_range(img, leaves, dmin, tau, rule="figure"):
    """
    Domain / range block separation algorithm of the paper.

        if  size(b) > dmin                         -> Range
        elif var(b) >= tau_eff * var_max           -> Domain  (feature rich)
        else                                       -> Range

    var_max = maximum variance of the dmin x dmin blocks of the image.

    rule = "literal": tau_eff = tau  (formula exactly as printed; with the
                      printed values 1e-3..1e-6 nearly EVERY dmin block is a
                      domain and tau has almost no effect).
    rule = "figure" : Fig. 2 of the paper shows that a LARGER tau (1e-3) keeps
                      MORE domain blocks and a SMALLER tau (1e-6) FEWER, i.e. the
                      opposite monotonicity of the printed inequality.  To
                      reproduce that behaviour the ratio var/var_max is compared
                      with tau_eff = 0.2 * (tau/1e-6)^(-1/3):
                      1e-3 -> 0.02, 1e-4 -> 0.043, 1e-5 -> 0.093, 1e-6 -> 0.2
                      (monotone and inside (0,1) as the paper requires).
    """
    vmax = max(1e-9, max(img[y:y + s, x:x + s].var()
                         for (y, x, s) in leaves if s == dmin))
    if rule == "literal":
        tau_eff = tau
    else:
        tau_eff = 0.2 * (tau / 1e-6) ** (-1.0 / 3.0)
    dom, rng_ = [], []
    for (y, x, s) in leaves:
        if s == dmin and img[y:y + s, x:x + s].var() >= tau_eff * vmax:
            dom.append((y, x, s))
        else:
            rng_.append((y, x, s))
    return dom, rng_, tau_eff


# --------------------------------------------------------------------------
# Self-Organising Map (Kohonen)  (Section 3.2)
# --------------------------------------------------------------------------
class BSLCStream:
    """Bidirectionally coupled chaotic map (BSLC) used as a deterministic
    pseudo-random source.  Same equations as the chaotic BPNN code:
        x' = 4x(1-x)(1-mu) + mu sin^2(pi y/2)
        y' = 4y(1-y)(1-mu) + mu sin^2(pi x/2)
        Z  = (x * y) mod 1
    """

    def __init__(self, mu=0.10, x0=0.2468, y0=0.3691, burn=200):
        self.mu, self.x, self.y = mu, float(x0), float(y0)
        for _ in range(burn):
            self._step()

    def _step(self):
        mu, x, y = self.mu, self.x, self.y
        xn = 4.0 * x * (1.0 - x) * (1.0 - mu) + mu * math.sin(math.pi * y / 2.0) ** 2
        yn = 4.0 * y * (1.0 - y) * (1.0 - mu) + mu * math.sin(math.pi * x / 2.0) ** 2
        self.x = min(max(xn, 1e-7), 1 - 1e-7)
        self.y = min(max(yn, 1e-7), 1 - 1e-7)

    def next_z(self):
        self._step()
        return (self.x * self.y) % 1.0

    def z_sequence(self, n):
        return np.array([self.next_z() for _ in range(n)])


class SOM:
    """
    1. initialise node weights  2. grab an input vector
    3. traverse every node      4. Euclidean distance to the input
    5. keep the Best Matching Unit (BMU)
    6. Wv(t+1) = Wv(t) + theta(t) * alpha(t) * (D(t) - Wv(t))   (neighbourhood)

    There is NO activation function: learning is competitive (winner = smallest
    Euclidean distance) and the neighbourhood function
        theta(t) = exp(-d^2 / (2 sigma(t)^2))        (Gaussian)
    plays the role that an activation plays in a feed-forward network.

    init  : "random"   weights = random training samples (+ tiny noise)
            "chaotic"  weights = lo + (hi-lo) * Z_n  (BSLC stream), per-feature range
            "logistic" same, with the logistic map r=3.9
    sched : "linear"   alpha(t) = lr0 * (1 - t/T)
            "chaotic"  alpha(t) = lr0 * (1 - t/T) * (a_min + (a_max-a_min) Z_t)
                       (BSLC modulation, a_min=0.5, a_max=1.0, so alpha never
                        exceeds the linear schedule)
    """

    def __init__(self, grid=(3, 3), epochs=15, lr0=0.5, seed=0,
                 init="random", sched="linear", chaos_x0=0.2468, chaos_y0=0.3691):
        self.grid, self.epochs, self.lr0 = grid, epochs, lr0
        self.init, self.sched = init, sched
        self.rs = np.random.RandomState(seed)
        self.chaos = (chaos_x0, chaos_y0)
        gy, gx = np.mgrid[0:grid[0], 0:grid[1]]
        self.pos = np.stack([gy.ravel(), gx.ravel()], 1).astype(float)
        self.n = grid[0] * grid[1]

    def _init_weights(self, X):
        if self.init == "random":
            idx = self.rs.choice(len(X), self.n, replace=len(X) < self.n)
            return X[idx].copy() + 1e-3 * self.rs.randn(self.n, X.shape[1])
        if self.init == "chaotic":
            z = BSLCStream(x0=self.chaos[0], y0=self.chaos[1]).z_sequence(self.n * X.shape[1])
        elif self.init == "logistic":
            x, zs = self.chaos[0], []
            for k in range(200 + self.n * X.shape[1]):
                x = 3.9 * x * (1.0 - x)
                if k >= 200:
                    zs.append(x)
            z = np.array(zs)
        else:
            raise ValueError("unknown SOM init: %s" % self.init)
        lo, hi = X.min(0), X.max(0)
        return lo + (hi - lo) * z.reshape(self.n, X.shape[1])

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.W = self._init_weights(X)
        stream = (BSLCStream(x0=self.chaos[0], y0=self.chaos[1])
                  if self.sched == "chaotic" else None)
        total = self.epochs * len(X)
        t = 0
        sig0 = max(self.grid) / 2.0
        for _ in range(self.epochs):
            for i in self.rs.permutation(len(X)):
                f = 1.0 - t / total
                alpha, sig = self.lr0 * f, max(0.5, sig0 * f)
                if stream is not None:
                    alpha *= 0.5 + 0.5 * stream.next_z()
                d = ((self.W - X[i]) ** 2).sum(1)
                bmu = d.argmin()
                theta = np.exp(-((self.pos - self.pos[bmu]) ** 2).sum(1)
                               / (2 * sig ** 2))
                self.W += (theta * alpha)[:, None] * (X[i] - self.W)
                t += 1
        # quantisation error: mean distance of every sample to its BMU
        dall = np.sqrt(((X[:, None, :] - self.W[None]) ** 2).sum(2))
        self.qe = float(dall.min(1).mean())
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        d = ((X[:, None, :] - self.W[None]) ** 2).sum(2)
        return d.argmin(1)


def block_features(blocks):
    """
    Isometry-invariant feature vector of (N, r, r) blocks:
    mean, std and the sorted, normalised quadrant means.
    """
    n, r, _ = blocks.shape
    h = r // 2
    q = np.stack([blocks[:, :h, :h].mean((1, 2)), blocks[:, :h, h:].mean((1, 2)),
                  blocks[:, h:, :h].mean((1, 2)), blocks[:, h:, h:].mean((1, 2))], 1)
    mu = blocks.mean((1, 2))
    sd = blocks.std((1, 2))
    dev = np.sort(q - mu[:, None], 1) / (sd[:, None] + 1.0)
    return np.column_stack([mu / 255.0, np.minimum(sd, 64) / 64.0, 0.3 * dev])


# --------------------------------------------------------------------------
# Fractal matching machinery
# --------------------------------------------------------------------------
class DomainPool:
    """All domain candidates (2r x 2r windows anchored at the seed blocks,
    down-sampled to r x r) with their 8 isometries, for one range size r."""

    def __init__(self, P, anchors, r, n_img, som_grid=None, seed=0, som_kw=None):
        pos = sorted({(min(y, n_img - 2 * r), min(x, n_img - 2 * r))
                      for (y, x) in anchors})
        self.pos = np.array(pos)
        base = np.stack([P[y:y + 2 * r:2, x:x + 2 * r:2] for (y, x) in pos])
        self.n, self.r = len(pos), r
        self.all = np.stack([isometry(base, k).reshape(self.n, -1)
                             for k in range(8)]).astype(np.float32)  # (8,N,r*r)
        self.sum_d = self.all[0].sum(1)
        self.sum_dd = (self.all[0] ** 2).sum(1)
        self.labels = None
        if som_grid is not None:
            self.feat = block_features(base)
            self.som = SOM(som_grid, seed=seed, **(som_kw or {})).fit(self.feat)
            self.labels = self.som.predict(self.feat)

    def candidates(self, label=None):
        """indices of domain windows to search."""
        if label is None or self.labels is None:
            return np.arange(self.n)
        idx = np.where(self.labels == label)[0]
        if len(idx) == 0:                       # nearest non-empty group
            used = np.unique(self.labels)
            d = ((self.som.W[used] - self.som.W[label]) ** 2).sum(1)
            idx = np.where(self.labels == used[d.argmin()])[0]
        return idx


def best_match(rv, pool, idx):
    """Best (domain, isometry, s, o) for range vector rv among pool[idx]."""
    n = rv.size
    sr, srr = rv.sum(), (rv ** 2).sum()
    D = pool.all[:, idx, :]                       # (8, M, n)
    sd, sdd = pool.sum_d[idx], pool.sum_dd[idx]   # (M,)
    sdr = D @ rv.astype(np.float32)               # (8, M)
    den = n * sdd - sd ** 2
    s = np.where(den > 1e-6, (n * sdr - sd * sr) / np.maximum(den, 1e-6), 0.0)
    si = np.abs(s[..., None] - S_LEVELS).argmin(-1)           # quantise s
    sq = S_LEVELS[si]
    o = np.clip(np.rint((sr - sq * sd) / n), -256, 255)       # quantise o
    sse = (sq ** 2 * sdd + n * o ** 2 + srr + 2 * sq * o * sd
           - 2 * sq * sdr - 2 * o * sr)
    k, m = np.unravel_index(sse.argmin(), sse.shape)
    return int(idx[m]), int(k), int(si[k, m]), int(o[k, m]), float(max(sse[k, m], 0))


# --------------------------------------------------------------------------
# Encoder / decoder
# --------------------------------------------------------------------------
class Encoded:
    pass


def encode(img, algo, tau=1e-5, dmin=8, max_size=32, rmin=4, tol=5.0,
           split_var=60.0, rule="figure", som_grid=(3, 3), std_range=8,
           std_stride=8, som_kw=None, som_seed=0):
    """algo in {'standard','alg1','alg2'}.  Returns an Encoded object."""
    t0 = time.perf_counter()
    n_img = img.shape[0]
    P = pool2(img)
    e = Encoded()
    e.algo, e.shape, e.maps = algo, img.shape, []
    e.rmin = rmin

    # ---------------- standard fractal coding ----------------
    if algo == "standard":
        e.dmin, e.seed_mask = dmin, np.zeros(img.shape, bool)
        anchors = [(y, x) for y in range(0, n_img - 2 * std_range + 1, std_stride)
                   for x in range(0, n_img - 2 * std_range + 1, std_stride)]
        pool = DomainPool(P, anchors, std_range, n_img)
        e.pools = {std_range: pool}
        for y in range(0, n_img, std_range):
            for x in range(0, n_img, std_range):
                rv = img[y:y + std_range, x:x + std_range].ravel()
                d, k, si, o, _ = best_match(rv, pool, pool.candidates())
                e.maps.append((y, x, std_range, d, k, si, o))
        e.seed_bytes = 0
        e.bits = _count_bits(e, n_img, tree_bits=0)
        e.time = time.perf_counter() - t0
        return e

    # ---------------- proposed algorithms ----------------
    leaves = quadtree(img, max_size, dmin, split_var)
    dom, rng_blocks, e.tau_eff = separate_domain_range(img, leaves, dmin, tau, rule)
    e.leaves, e.dom, e.rng_blocks, e.dmin = leaves, dom, rng_blocks, dmin
    e.seed_mask = np.zeros(img.shape, bool)
    for (y, x, s) in dom:
        e.seed_mask[y:y + s, x:x + s] = True
    seeds = img[e.seed_mask].astype(np.uint8).tobytes()
    e.seed_bytes = len(zlib.compress(seeds, 9))          # lossless part
    anchors = [(y, x) for (y, x, s) in dom]

    e.pools = {}
    r = max_size
    while r >= rmin:
        e.pools[r] = DomainPool(P, anchors, r, n_img,
                                som_grid if algo == "alg2" else None,
                                seed=som_seed, som_kw=som_kw)
        r //= 2

    # range block features -> group labels (Algorithm II)
    def label_of(blk, pool):
        if algo != "alg2":
            return None
        f = block_features(blk[None])
        return int(pool.som.predict(f)[0])

    stack = [b for b in rng_blocks]
    while stack:
        y, x, r = stack.pop()
        blk = img[y:y + r, x:x + r]
        # the range is down-sampled to the domain grid only for features
        pool = e.pools[r]
        lab = label_of(blk, pool)
        d, k, si, o, sse = best_match(blk.ravel(), pool, pool.candidates(lab))
        rmse = math.sqrt(sse / (r * r))
        if rmse > tol and r > rmin:
            h = r // 2
            stack += [(y, x, h), (y, x + h, h), (y + h, x, h), (y + h, x + h, h)]
        else:
            e.maps.append((y, x, r, d, k, si, o))
    e.bits = _count_bits(e, n_img, tree_bits=len(leaves) * 4 // 3 + len(dom) // 8)
    e.time = time.perf_counter() - t0
    return e


def _count_bits(e, n_img, tree_bits):
    bits = tree_bits
    for (y, x, r, d, k, si, o) in e.maps:
        nd = max(2, e.pools[r].n)
        bits += math.ceil(math.log2(nd)) + ISO_BITS + S_BITS + O_BITS
        if e.algo != "standard" and r > e.rmin:
            bits += 1                                    # "split" flag
    return bits + 8 * e.seed_bytes


def compression_ratio(e):
    return (e.shape[0] * e.shape[1] * 8.0) / e.bits


def decode(e, img_seed=None, iters=12):
    """IFS decoding; the lossless seed blocks are pasted back at every step."""
    n = e.shape[0]
    cur = np.full(e.shape, 128.0)
    if img_seed is not None and e.seed_mask.any():
        cur[e.seed_mask] = img_seed[e.seed_mask]
    for _ in range(iters):
        P = pool2(cur)
        new = cur.copy()
        for (y, x, r, d, k, si, o) in e.maps:
            py, px = e.pools[r].pos[d]
            w = isometry(P[py:py + 2 * r:2, px:px + 2 * r:2], k)
            new[y:y + r, x:x + r] = S_LEVELS[si] * w + o
        cur = np.clip(new, 0, 255)
        if img_seed is not None and e.seed_mask.any():
            cur[e.seed_mask] = img_seed[e.seed_mask]
    return np.rint(cur)


def run(img, algo, **kw):
    e = encode(img, algo, **kw)
    seeds = None
    if e.seed_mask.any():       # the seeds are what the lossless stream stores
        seeds = img
    rec = decode(e, seeds)
    return e, rec, psnr(img, rec), e.time, compression_ratio(e)


# --------------------------------------------------------------------------
# Table printing (same layout as the paper's tables)
# --------------------------------------------------------------------------
def fmt_table(title, header, rows, avg, num_fmt):
    cols = ["Sample MRI Image"] + header
    body = [[str(i + 1)] + [num_fmt(v) for v in r] for i, r in enumerate(rows)]
    if avg is not None:
        body.append(["Avg"] + [num_fmt(v) for v in avg])
    w = [max(len(c), *(len(b[i]) for b in body)) + 2 for i, c in enumerate(cols)]
    line = "+" + "+".join("-" * x for x in w) + "+"
    out = [title, line,
           "|" + "|".join(c.center(x) for c, x in zip(cols, w)) + "|", line]
    for b in body[:-1] if avg is not None else body:
        out.append("|" + "|".join(c.center(x) for c, x in zip(b, w)) + "|")
    if avg is not None:
        out += [line, "|" + "|".join(c.center(x) for c, x in zip(body[-1], w)) + "|"]
    out.append(line)
    return "\n".join(out)


def to_markdown(title, header, rows, avg, num_fmt):
    cols = ["Sample MRI Image"] + header
    md = [f"**{title}**", "", "| " + " | ".join(cols) + " |",
          "|" + "---|" * len(cols)]
    for i, r in enumerate(rows):
        md.append("| " + " | ".join([str(i + 1)] + [num_fmt(v) for v in r]) + " |")
    if avg is not None:
        md.append("| **Avg** | " + " | ".join(num_fmt(v) for v in avg) + " |")
    return "\n".join(md) + "\n"


def tau_name(t):
    return f"tau = 1e{int(round(math.log10(t)))}"


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def make_figures(out, images, enc_cache, tau_list, t4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # Fig. 1  input / partitioned image / separated domain blocks
    fig, ax = plt.subplots(3, 2, figsize=(8, 12))
    for c in range(2):
        e = enc_cache[(c, 1e-5)]
        ax[0, c].imshow(images[c], cmap="gray"); ax[0, c].set_title(f"Sample MRI Image {c+1}")
        ax[1, c].imshow(images[c], cmap="gray")
        for (y, x, s) in e.leaves:
            ax[1, c].add_patch(Rectangle((x, y), s, s, fill=False, ec="w", lw=0.4))
        m = np.ones(images[c].shape); m[e.seed_mask] = 0
        ax[2, c].imshow(m, cmap="gray", vmin=0, vmax=1)
        if c == 0:
            ax[0, 0].set_ylabel("Input Image"); ax[1, 0].set_ylabel("The Partitioned Image")
            ax[2, 0].set_ylabel("Separated Domain Blocks")
    for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Fig. 1  Feature rich and separated domain blocks")
    fig.savefig(os.path.join(out, "fig1_partition.png"), dpi=130, bbox_inches="tight"); plt.close(fig)

    # Fig. 2  domain blocks for various thresholds
    nimg = min(2, len(images)); k = len(tau_list)
    fig, ax = plt.subplots(k + 1, nimg, figsize=(4 * nimg, 3.6 * (k + 1)), squeeze=False)
    for c in range(nimg):
        ax[0, c].imshow(images[-nimg + c], cmap="gray")
        ax[0, c].set_title(f"Sample MRI Image {len(images) - nimg + c + 1}")
        for r, t in enumerate(tau_list):
            e = enc_cache[(len(images) - nimg + c, t)]
            m = np.ones(images[0].shape); m[e.seed_mask] = 0
            ax[r + 1, c].imshow(m, cmap="gray", vmin=0, vmax=1)
            ax[r + 1, c].set_ylabel(f"Threshold = 1e{int(round(math.log10(t)))}")
    for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Fig. 2  Domain blocks separated for various thresholds")
    fig.savefig(os.path.join(out, "fig2_thresholds.png"), dpi=110, bbox_inches="tight"); plt.close(fig)

    # Fig. 3 performance analysis (averages)
    avg = {m: [np.mean([enc_cache[(i, t)].metrics[m] for i in range(len(images))])
               for t in tau_list] for m in ("psnr", "time", "cr")}
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = [f"1e{int(round(math.log10(t)))}" for t in tau_list]
    ax.plot(xs, avg["psnr"], "bd-", label="PSNR")
    ax.plot(xs, avg["time"], "ms-", label="Encoding Time")
    ax.plot(xs, avg["cr"], "b^-", mfc="none", label="Compression Ratio")
    ax.set_xlabel("The Threshold"); ax.set_ylabel("Performance"); ax.grid(True)
    ax.legend(); ax.set_title("The Performance Analysis")
    fig.savefig(os.path.join(out, "fig3_performance.png"), dpi=130, bbox_inches="tight"); plt.close(fig)

    # Fig. 6/7/8 bar charts
    names = ["Standard\nFractal Encoding", "Proposed\nAlgorithm I", "Proposed\nAlgorithm II"]
    for fn, key, yl, ttl, col in (("fig6_psnr.png", 0, "PSNR(db)", "PSNR", "#9999ff"),
                                  ("fig7_time.png", 1, "Compression Time(sec)", "Time Taken for Different Methods", "#339966"),
                                  ("fig8_cr.png", 2, "Compression Ratio", "Compression Ratio of Different Methods", "#3366ff")):
        fig, ax = plt.subplots(figsize=(5, 4))
        vals = [t4[a][key] for a in ("standard", "alg1", "alg2")]
        b = ax.bar(names, vals, color=col, ec="k")
        for rect, v in zip(b, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom")
        ax.set_ylabel(yl); ax.set_title(ttl); ax.grid(axis="y")
        fig.savefig(os.path.join(out, fn), dpi=130, bbox_inches="tight"); plt.close(fig)

    # Fig. 5 reconstructed images
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.4))
    ax[0].imshow(images[-1], cmap="gray"); ax[0].set_title("Original")
    for a, k_, nm in zip(ax[1:], ("standard", "alg1", "alg2"),
                         ("The Std. Fractal compression Algorithm", "The Proposed Algorithm I", "The Proposed Algorithm II")):
        a.imshow(t4[k_][3], cmap="gray", vmin=0, vmax=255)
        a.set_title(f"{nm}\nPSNR={t4[k_][0]:.2f} dB", fontsize=9)
    for a in ax: a.axis("off")
    fig.suptitle("Fig. 5  PSNR for all three algorithms")
    fig.savefig(os.path.join(out, "fig5_reconstruction.png"), dpi=130, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------
# Main experiment
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="*", help="MRI image files (png/jpg/bmp/tif), any number.")
    ap.add_argument("--dir", help="folder whose images are ALL used (e.g. the 'training' folder)")
    ap.add_argument("--size", type=int, default=256,
                    help="image side (paper: 512). Must be a multiple of 32. Default 256 for speed")
    ap.add_argument("--tol", type=float, default=5.0, help="range RMS-error tolerance (quad-tree split)")
    ap.add_argument("--rule", choices=["figure", "literal"], default="figure",
                    help="how tau is mapped to the variance threshold (see docstring)")
    ap.add_argument("--groups", type=int, default=3, help="SOM grid side (groups = side^2)")
    ap.add_argument("--som-init", choices=["random", "chaotic", "logistic"], default="random",
                    help="SOM weight initialisation")
    ap.add_argument("--som-sched", choices=["linear", "chaotic"], default="linear",
                    help="SOM learning-rate schedule")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    paths = []
    if a.dir:
        import glob
        paths = sorted(f for f in glob.glob(os.path.join(a.dir, "**", "*"), recursive=True)
                       if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")))
        if not paths:
            raise SystemExit("No images found in %s" % a.dir)
    elif a.images:
        paths = list(a.images)
    if paths:
        images = [load_image(p, a.size) for p in paths]
        mapping = ["Sample MRI Image %d = %s" % (i + 1, os.path.basename(p)) for i, p in enumerate(paths)]
    else:
        images = [make_phantom(i, a.size) for i in range(4)]
        mapping = ["Sample MRI Image %d = synthetic phantom %d" % (i + 1, i + 1) for i in range(4)]
    print("\n".join(mapping))
    from PIL import Image
    for i, im in enumerate(images):
        Image.fromarray(im.astype(np.uint8)).save(os.path.join(a.out, f"input_{i+1}.png"))
    kw = dict(tol=a.tol, rule=a.rule, som_grid=(a.groups, a.groups),
              som_kw=dict(init=a.som_init, sched=a.som_sched))

    # ---- Tables 1-3 : Algorithm II (fast fractal coding) at each threshold
    enc_cache = {}
    for i, im in enumerate(images):
        for t in TAU_LIST:
            e, rec, p, tm, cr = run(im, "alg2", tau=t, **kw)
            e.metrics = dict(psnr=p, time=tm, cr=cr)
            enc_cache[(i, t)] = e
            print(f"  image {i+1}  {tau_name(t)}  PSNR={p:6.2f}  time={tm:7.2f}s  CR={cr:6.2f}"
                  f"  domains={len(e.dom)}", flush=True)
    header = [f"τ=10^{int(round(math.log10(t)))}" for t in TAU_LIST]
    tabs = {}
    for key, title, f in (
            ("psnr", "Table 1  PSNR at Different Levels of Compression (dB)", lambda v: f"{v:.2f}"),
            ("time", "Table 2  Time Taken at Different Levels of Compression (sec)", lambda v: f"{v:.2f}"),
            ("cr", "Table 3  Compression Ratio at Different Thresholds", lambda v: f"{v:.2f}")):
        rows = [[enc_cache[(i, t)].metrics[key] for t in TAU_LIST] for i in range(len(images))]
        avg = list(np.mean(rows, 0))
        tabs[key] = (title, header, rows, avg, f)

    # ---- Table 4 : the three algorithms on the last image, tau = 1e-5
    last = images[-1]
    t4 = {}
    for algo in ("standard", "alg1", "alg2"):
        e, rec, p, tm, cr = run(last, algo, tau=1e-5, **kw)
        t4[algo] = (p, tm, cr, rec)
        print(f"  Table4 {algo:9s} PSNR={p:6.2f} time={tm:8.2f}s CR={cr:6.2f}", flush=True)

    # ---- print & save
    text, md = [], []
    for key in ("psnr", "time", "cr"):
        text.append(fmt_table(*tabs[key])); md.append(to_markdown(*tabs[key]))
    h4 = ["Standard Fractal Encoding", "Proposed Algorithm I (τ=1e-5)", "Proposed Algorithm II (τ=1e-5)"]
    rows4 = [["PSNR (dB)"] + [f"{t4[k][0]:.2f}" for k in ("standard", "alg1", "alg2")],
             ["Compression Time (sec)"] + [f"{t4[k][1]:.2f}" for k in ("standard", "alg1", "alg2")],
             ["Compression Ratio"] + [f"{t4[k][2]:.2f}" for k in ("standard", "alg1", "alg2")]]
    cols = ["Metric"] + h4
    w = [max(len(c), *(len(r[i]) for r in rows4)) + 2 for i, c in enumerate(cols)]
    ln = "+" + "+".join("-" * x for x in w) + "+"
    t4txt = ["Table 4  PSNR achieved for Different Algorithms", ln,
             "|" + "|".join(c.center(x) for c, x in zip(cols, w)) + "|", ln]
    for r in rows4:
        t4txt.append("|" + "|".join(c.center(x) for c, x in zip(r, w)) + "|")
    t4txt.append(ln)
    text.append("\n".join(t4txt))
    md.append("**Table 4  PSNR achieved for Different Algorithms**\n\n| Metric | " + " | ".join(h4) +
              " |\n|---|---|---|---|\n" + "\n".join("| " + " | ".join(r) + " |" for r in rows4) + "\n")
    report = "\n".join(mapping) + "\n\n" + "\n\n".join(text)
    print("\n" + report)
    open(os.path.join(a.out, "tables.txt"), "w", encoding="utf-8").write(report + "\n")
    open(os.path.join(a.out, "tables.md"), "w", encoding="utf-8").write("\n".join(md))
    make_figures(a.out, images, enc_cache, TAU_LIST, t4)
    print(f"\nTables and figures saved in: {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
