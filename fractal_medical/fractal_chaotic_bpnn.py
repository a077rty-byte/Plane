#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEVELOPMENT 1 -- chaotic BPNN as the supervised classifier of Algorithm II
==========================================================================
Built ON TOP of fractal_baseline.py (which is imported and never modified).

Idea
  Algorithm II of the paper groups the domain blocks and, for every range block, searches
  only the domain blocks of "its" group (step 4: "supervised classification technique").
  Baseline : a SOM is trained ON EVERY IMAGE; a range block gets the group of the nearest
             SOM node (similar feature vector -> same group).
  Here     : a back-propagation network (the chaotic BPNN of Main_codeChaotic.py) learns,
             OFFLINE on other training images, to predict from the range-block features
             the group of the domain block that the exhaustive search would have chosen.
             At coding time only a forward pass is needed (no per-image SOM training).

What is taken from Main_codeChaotic.py (same equations and constants)
  * BSLC chaotic map  x' = 4x(1-x)(1-mu) + mu sin^2(pi y/2),  y' = ...,  Z = (x*y) mod 1
  * chaotic weight initialisation   W = a + (b-a) Z_n         (weights in [a, b] = [-1, 1])
  * chaotic bias   theta_n = theta_min + (theta_max-theta_min) Z_n
  * chaotic factor alpha_n = alpha_min + (alpha_max-alpha_min) Z_n
  * updates  W2 <- alpha W2 + eta h^T delta_k ,  W1 <- alpha W1 + eta x^T delta_j  (sigmoid, MSE)
  * the 4 initialisations (tanh / sigmoid / BSLC / Logistic) x 2 modes (GBDR / chaotic alpha)

Activation function used here: sigmoid  phi(v) = 1/(1+exp(-v))  in the hidden and output layers
(exactly as in Main_codeChaotic.py).  The SOM of the baseline has none (competitive learning).

Compared (all on the SAME images, tau, machine):
  Alg. I (full search)            Alg. II baseline (per-image SOM)
  Alg. II offline codebook        (control: fixed SOM codebook, nearest node, no NN)
  BPNN plain (random init)        (control: ordinary BPNN, no chaos)
  BPNN GBDR x 4 inits             BPNN chaotic-alpha x 4 inits
"""
import argparse
import math
import os
import random
import time
import zlib
from types import SimpleNamespace

import numpy as np

import fractal_baseline as fb

# ==========================================================================
# 1. Constants of Main_codeChaotic.py
# ==========================================================================
MU_BSLC = 0.10
X0, Y0 = 0.2468, 0.3691
BURN_IN = 200
A_WEIGHT, B_WEIGHT = -1.0, 1.0
THETA_MIN, THETA_MAX = 0.0, 0.1
ALPHA_MIN, ALPHA_MAX = 0.98, 0.99999
ETA = 0.1
EPSILON = 0.005
HIDDEN = 128


# ==========================================================================
# 2. Chaotic streams (same equations as Main_codeChaotic.py)
# ==========================================================================
class BSLCStream:
    def __init__(self, mu=MU_BSLC, x0=X0, y0=Y0, burn=BURN_IN):
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
        return np.array([self.next_z() for _ in range(n)], dtype=np.float64)


class Map1D:
    """1-D map used only for the weight initialisation (tanh / sigmoid / Logistic)."""

    def __init__(self, f, x0, burn, to_z=lambda x: x):
        self.f, self.x, self.to_z = f, float(x0), to_z
        for _ in range(burn):
            self.x = self.f(self.x)

    def next_z(self):
        self.x = self.f(self.x)
        return float(np.clip(self.to_z(self.x), 1e-7, 1 - 1e-7))

    def z_sequence(self, n):
        return np.array([self.next_z() for _ in range(n)], dtype=np.float64)


def make_init_stream(method, bslc):
    if method == "BSLC":
        return bslc                                   # the SAME stream that later gives theta, alpha
    if method == "Logistic":
        return Map1D(lambda x: float(np.clip(3.9 * x * (1.0 - x), 1e-7, 1 - 1e-7)), 0.123456, BURN_IN)
    if method == "tanh":
        return Map1D(lambda x: float(np.tanh(2.0 * x + 0.1)), 0.111, 50, to_z=lambda x: 0.5 * (x + 1.0))
    if method == "sigmoid":
        return Map1D(lambda x: float(1.0 / (1.0 + np.exp(-(4.0 * x - 2.0)))), 0.111, 50)
    raise ValueError(method)


def chaotic_init(stream, A, B, C, a=A_WEIGHT, b=B_WEIGHT):
    Z = stream.z_sequence(A * B + B * C)
    W = a + (b - a) * Z                               # U = 2Z-1 in [-1,1]  ->  [a, b]
    return W[:A * B].reshape(A, B), W[A * B:].reshape(B, C)


def phi(v):
    return 1.0 / (1.0 + np.exp(-np.clip(v, -500, 500)))


# ==========================================================================
# 3. The chaotic BPNN
# ==========================================================================
class ChaoticBPNN:
    """
    init          : "BSLC" | "tanh" | "sigmoid" | "Logistic"  (chaotic initialisation)  or "random"
    use_alpha     : True  -> W <- alpha_n W + eta delta (.)      (chaotic momentum / decay)
                    False -> W <- W + eta delta (.)              (baseline GBDR, alpha = 1)
    chaotic_theta : True  -> theta_n from BSLC  (as in Main_codeChaotic.py);  False -> theta = 0
    alpha_mode    : "step"  -> alpha_n changes and is applied at EVERY update (as in Main_codeChaotic.py)
                    "epoch" -> alpha is applied once per EPOCH (alpha drawn from BSLC at the end of every
                               epoch).  This is OUR variant: with thousands of mini-batch updates the
                               per-update decay alpha^steps drives all weights to zero.
    """

    def __init__(self, A, B=HIDDEN, C=9, init="BSLC", use_alpha=True, chaotic_theta=True,
                 eta=ETA, seed=0, alpha_mode="step"):
        self.A, self.B, self.C = A, B, C
        self.init, self.use_alpha, self.chaotic_theta, self.eta = init, use_alpha, chaotic_theta, eta
        self.seed, self.alpha_mode = seed, alpha_mode

    def fit(self, X, T, epochs=30, batch=64, eps=EPSILON, verbose=False):
        t0 = time.time()
        P = len(X)
        stream = BSLCStream()
        if self.init == "random":
            rs = np.random.RandomState(self.seed)
            self.W1 = rs.uniform(A_WEIGHT, B_WEIGHT, (self.A, self.B))
            self.W2 = rs.uniform(A_WEIGHT, B_WEIGHT, (self.B, self.C))
        else:
            self.W1, self.W2 = chaotic_init(make_init_stream(self.init, stream), self.A, self.B, self.C)
        z = stream.next_z()
        theta = THETA_MIN + (THETA_MAX - THETA_MIN) * z if self.chaotic_theta else 0.0
        alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * z
        rng = np.random.RandomState(self.seed)
        self.history, E, epoch = [], np.inf, 0
        while E > eps and epoch < epochs:
            order = rng.permutation(P)
            sq = 0.0
            for s in range(0, P, batch):
                idx = order[s:s + batch]
                x, t = X[idx], T[idx]
                h = phi(x @ self.W1 + theta)
                a = phi(h @ self.W2)
                err = t - a
                sq += 0.5 * float(np.sum(err ** 2))
                dk = err * a * (1.0 - a)
                dj = h * (1.0 - h) * (dk @ self.W2.T)
                m = len(idx)
                if self.alpha_mode == "epoch":
                    a_n = 1.0                                  # alpha is applied once, after the epoch
                else:
                    a_n = alpha if self.use_alpha else 1.0
                self.W2 = a_n * self.W2 + self.eta * (h.T @ dk) / m
                self.W1 = a_n * self.W1 + self.eta * (x.T @ dj) / m
                z = stream.next_z()
                if self.chaotic_theta:
                    theta = THETA_MIN + (THETA_MAX - THETA_MIN) * z
                if self.alpha_mode == "step":
                    alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * z
            if self.alpha_mode == "epoch" and self.use_alpha:
                alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * stream.next_z()
                self.W1, self.W2 = alpha * self.W1, alpha * self.W2
            E = sq / P
            epoch += 1
            self.history.append(E)
            if verbose:
                print("    epoch %3d  E=%.5f" % (epoch, E), flush=True)
        self.theta = theta                              # the last theta is used at prediction time
        self.final_E, self.epochs_run, self.train_time = E, epoch, time.time() - t0
        return self

    def predict_proba(self, X):
        return phi(phi(X @ self.W1 + self.theta) @ self.W2)


# ==========================================================================
# 4. Groupers: how a range block is mapped to a domain group
# ==========================================================================
def feat_baseline(blocks):
    """The 6 features of the baseline SOM: mean, std, 4 sorted normalised quadrant means."""
    return fb.block_features(blocks)


def feat_shape(blocks):
    """Shape only (4 sorted, normalised quadrant deviations): unaffected by the contrast s and
    brightness o of the fractal map, which is what the match is invariant to."""
    return fb.block_features(blocks)[:, 2:]


FEATS = {"baseline": feat_baseline, "shape": feat_shape}


class FullSearch:
    """Algorithm I: no grouping (all domain windows are searched)."""
    name = "alg1"

    def prepare(self, pool, base, r):
        pool.labels = None

    def range_labels(self, blk, r, pool):
        return [None]


class PerImageSOM:
    """Baseline Algorithm II: one SOM per domain pool, trained on the image being coded.
    (featfn = feat_baseline gives exactly the baseline.)"""
    name = "som"

    def __init__(self, featfn=feat_baseline):
        self.featfn = featfn

    def prepare(self, pool, base, r):
        feat = self.featfn(base)
        pool.som = fb.SOM().fit(feat)
        pool.labels = pool.som.predict(feat)

    def range_labels(self, blk, r, pool):
        return [int(pool.som.predict(self.featfn(blk[None]))[0])]


class Codebook:
    """Offline SOM codebook W (K x 6): the group of a block is its nearest node."""

    def __init__(self, W, featfn=feat_baseline):
        self.W = np.asarray(W)
        self.featfn = featfn

    def label(self, feat):
        return ((feat[:, None, :] - self.W[None]) ** 2).sum(2).argmin(1)


class CodebookGrouper(Codebook):
    """Control: fixed codebook, range block -> nearest node (no neural network)."""
    name = "codebook"

    def prepare(self, pool, base, r):
        pool.som = SimpleNamespace(W=self.W)             # used by pool.candidates() for empty groups
        pool.labels = self.label(self.featfn(base))

    def range_labels(self, blk, r, pool):
        return [int(self.label(self.featfn(blk[None]))[0])]


class ConstantGrouper(CodebookGrouper):
    """Control: ignores the range block and always searches the same group."""
    name = "constant"

    def __init__(self, W, group, featfn=feat_baseline):
        super().__init__(W, featfn)
        self.group = int(group)

    def range_labels(self, blk, r, pool):
        return [self.group]


class NNGrouper(CodebookGrouper):
    """Domain groups from the offline codebook; the group of a RANGE block is predicted by the BPNN."""
    name = "nn"

    def __init__(self, W, net, lo, hi, topk=1, featfn=feat_baseline):
        super().__init__(W, featfn)
        self.net, self.lo, self.hi, self.topk = net, lo, hi, topk

    def vector(self, blk, r):
        f = np.concatenate([self.featfn(blk[None])[0], [math.log2(r) / 5.0]])
        return np.clip((f - self.lo) / self.hi, 0.0, 1.0)[None]

    def range_labels(self, blk, r, pool):
        p = self.net.predict_proba(self.vector(blk, r))[0]
        return [int(g) for g in np.argsort(-p)[:self.topk]]


class Recorder(CodebookGrouper):
    """Full search, but records (features, size, group of the best domain) for training."""
    name = "recorder"

    def __init__(self, W, featfn=feat_baseline):
        super().__init__(W, featfn)
        self.X, self.y = [], []

    def prepare(self, pool, base, r):
        pool.labels = None
        pool.cb_labels = self.label(self.featfn(base))

    def range_labels(self, blk, r, pool):
        return [None]

    def observe(self, blk, r, pool, d):
        f = np.concatenate([self.featfn(blk[None])[0], [math.log2(r) / 5.0]])
        self.X.append(f)
        self.y.append(int(pool.cb_labels[d]))


# ==========================================================================
# 5. Encoder with an injectable grouper (same pipeline as fb.encode, Algorithm I/II)
# ==========================================================================
def encode_grouped(img, tau, grouper, rule=fb.RULE):
    t0 = time.perf_counter()
    n_img = img.shape[0]
    P = fb.pool2(img)
    e = fb.Encoded()
    e.algo, e.shape, e.maps, e.rmin = "alg2", img.shape, [], fb.RMIN
    leaves = fb.quadtree(img, fb.MAX_SIZE, fb.DMIN, fb.SPLIT_VAR)
    dom, rng_blocks, e.tau_eff = fb.separate_domain_range(img, leaves, fb.DMIN, tau, rule)
    e.leaves, e.dom = leaves, dom
    e.seed_mask = np.zeros(img.shape, bool)
    for (y, x, s) in dom:
        e.seed_mask[y:y + s, x:x + s] = True
    e.seed_bytes = len(zlib.compress(img[e.seed_mask].astype(np.uint8).tobytes(), 9))
    anchors = [(y, x) for (y, x, s) in dom]
    e.pools, r = {}, fb.MAX_SIZE
    while r >= fb.RMIN:
        pool = fb.DomainPool(P, anchors, r, n_img, use_som=False)
        base = np.stack([P[y:y + 2 * r:2, x:x + 2 * r:2] for (y, x) in pool.pos])
        grouper.prepare(pool, base, r)
        e.pools[r] = pool
        r //= 2
    e.calls = e.cand = 0
    stack = list(rng_blocks)
    while stack:
        y, x, r = stack.pop()
        blk = img[y:y + r, x:x + r]
        pool = e.pools[r]
        labs = grouper.range_labels(blk, r, pool)
        if len(labs) == 1:
            idx = pool.candidates(labs[0])
        else:
            idx = np.unique(np.concatenate([pool.candidates(l) for l in labs]))
        d, k, si, o, sse = fb.best_match(blk.ravel(), pool, idx)
        e.calls += 1
        e.cand += len(idx)
        if hasattr(grouper, "observe"):
            grouper.observe(blk, r, pool, d)
        if math.sqrt(sse / (r * r)) > fb.TOL and r > fb.RMIN:
            h = r // 2
            stack += [(y, x, h), (y, x + h, h), (y + h, x, h), (y + h, x + h, h)]
        else:
            e.maps.append((y, x, r, d, k, si, o))
    e.bits = fb._count_bits(e, tree_bits=len(leaves) * 4 // 3 + len(dom) // 8)
    e.time = time.perf_counter() - t0
    return e


# ==========================================================================
# 6. Data
# ==========================================================================
def training_names(source, cls, split, exclude, n, seed):
    items = dict(fb.list_images(source))
    cand = [m for m in items if os.path.basename(os.path.dirname(m)).lower() == cls.lower()]
    if split:
        cand = [m for m in cand if ("/" + split.lower() + "/") in ("/" + m.lower())] or cand
    cand = sorted(m for m in cand if m not in set(exclude))
    return sorted(random.Random(seed).sample(cand, min(n, len(cand)))), items


def fit_codebook(imgs, tau, featfn=feat_baseline, k_grid=(3, 3), max_vec=3000, seed=0):
    """Offline SOM codebook trained on the domain-window features of the training images."""
    feats = []
    for img in imgs:
        P = fb.pool2(img)
        leaves = fb.quadtree(img, fb.MAX_SIZE, fb.DMIN, fb.SPLIT_VAR)
        dom, _, _ = fb.separate_domain_range(img, leaves, fb.DMIN, tau)
        anchors = [(y, x) for (y, x, s) in dom]
        r = fb.MAX_SIZE
        while r >= fb.RMIN:
            pos = sorted({(min(y, img.shape[0] - 2 * r), min(x, img.shape[0] - 2 * r)) for (y, x) in anchors})
            base = np.stack([P[y:y + 2 * r:2, x:x + 2 * r:2] for (y, x) in pos])
            feats.append(featfn(base))
            r //= 2
    F = np.concatenate(feats)
    rs = np.random.RandomState(seed)
    if len(F) > max_vec:
        F = F[rs.choice(len(F), max_vec, replace=False)]
    som = fb.SOM(grid=k_grid, seed=seed).fit(F)
    return som.W.copy()


def record_dataset(imgs, tau, W, featfn=feat_baseline):
    rec = Recorder(W, featfn)
    for img in imgs:
        encode_grouped(img, tau, rec)
    return np.array(rec.X), np.array(rec.y)


# ==========================================================================
# 7. Tables
# ==========================================================================
def box(title, head, rows):
    w = [max(len(h), *(len(r[i]) for r in rows)) + 2 for i, h in enumerate(head)]
    ln = "+" + "+".join("-" * x for x in w) + "+"
    out = [title, ln, "|" + "|".join(h.center(x) for h, x in zip(head, w)) + "|", ln]
    out += ["|" + "|".join(c.center(x) for c, x in zip(r, w)) + "|" for r in rows]
    return "\n".join(out + [ln])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="archive.zip (or folder)")
    ap.add_argument("--manifest", required=True, help="selected_images.txt of the baseline run (the 5 evaluation images)")
    ap.add_argument("--cls", default="notumor")
    ap.add_argument("--split", default="Training")
    ap.add_argument("--n-train", type=int, default=12, help="OTHER images used to train the network")
    ap.add_argument("--size", type=int, default=fb.IMG_SIZE)
    ap.add_argument("--tau", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=HIDDEN)
    ap.add_argument("--topk", type=int, default=1, help="search the groups of the k best outputs")
    ap.add_argument("--features", choices=list(FEATS), default="baseline",
                    help="baseline = the 6 features of the baseline SOM; shape = 4 contrast/brightness-free features")
    ap.add_argument("--repeats", type=int, default=1, help="timing = mean of this many encodings")
    ap.add_argument("--seed", type=int, default=7, help="choice of the training images")
    ap.add_argument("--out", default="results_chaotic")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    names = [l.strip() for l in open(a.manifest, encoding="utf-8") if l.strip()]
    tr_names, items = training_names(a.source, a.cls, a.split, names, a.n_train, a.seed)
    ev = [fb.load_image(items[m](), a.size) for m in names]
    tr = [fb.load_image(items[m](), a.size) for m in tr_names]
    log = ["Evaluation images (the baseline's): " + ", ".join(names),
           "Training images (disjoint): " + ", ".join(tr_names),
           "tau=%g  size=%d  hidden=%d  epochs<=%d  batch=%d  eta=%.2f  topk=%d  repeats=%d  features=%s" %
           (a.tau, a.size, a.hidden, a.epochs, a.batch, ETA, a.topk, a.repeats, a.features)]
    ff = FEATS[a.features]
    print("\n".join(log), flush=True)

    # ---- offline stage: codebook, dataset, training
    t0 = time.time()
    W = fit_codebook(tr, a.tau, ff)
    Xtr, ytr = record_dataset(tr, a.tau, W, ff)
    Xev, yev = record_dataset(ev, a.tau, W, ff)
    t_data = time.time() - t0
    lo, hi = Xtr.min(0), np.maximum(Xtr.max(0) - Xtr.min(0), 1e-9)
    sc = lambda X: np.clip((X - lo) / hi, 0.0, 1.0)
    K = len(W)
    Ttr = np.eye(K)[ytr]
    print("offline data: %d training blocks, %d evaluation blocks (%.1f s)" % (len(Xtr), len(Xev), t_data), flush=True)
    prior = np.bincount(yev, minlength=K).max() / len(yev)

    def acc(net, X, y, k=1):
        p = net.predict_proba(sc(X))
        top = np.argsort(-p, 1)[:, :k]
        return float(np.mean([y[i] in top[i] for i in range(len(y))]))

    cb_acc = float(np.mean(Codebook(W, ff).label(Xev[:, :-1]) == yev))
    const_g = int(np.bincount(ytr, minlength=K).argmax())

    variants = [("BPNN plain (random init)", dict(init="random", use_alpha=False, chaotic_theta=False))]
    for mode, ua, am in (("GBDR", False, "step"),
                         ("chaotic alpha/update", True, "step"),     # exactly Main_codeChaotic.py
                         ("chaotic alpha/epoch", True, "epoch")):    # our variant
        for init in ("tanh", "sigmoid", "BSLC", "Logistic"):
            variants.append(("BPNN %s / %s" % (mode, init),
                             dict(init=init, use_alpha=ua, chaotic_theta=True, alpha_mode=am)))

    nets, rows1 = {}, []
    for name, kw in variants:
        net = ChaoticBPNN(Xtr.shape[1], a.hidden, K, **kw).fit(sc(Xtr), Ttr, a.epochs, a.batch)
        nets[name] = net
        rows1.append([name, "%.3f" % acc(net, Xtr, ytr), "%.3f" % acc(net, Xev, yev), "%.3f" % acc(net, Xev, yev, 2),
                      "%.4f" % net.final_E, "%.1f" % net.train_time])
        print("  trained %-34s eval acc=%s  E=%.4f  %.1fs" % (name, rows1[-1][2], net.final_E, net.train_time), flush=True)
    rows1.insert(0, ["nearest codeword (no NN)", "-", "%.3f" % cb_acc, "-", "-", "0"])
    rows1.insert(0, ["majority group (prior)", "-", "%.3f" % prior, "-", "-", "0"])
    t1 = box("Table N1  Prediction of the group of the best-matching domain block (K=%d groups)" % K,
             ["Method", "Train acc", "Eval acc (top-1)", "Eval acc (top-2)", "Final MSE", "Train time (s)"], rows1)

    # ---- coding stage: the same evaluation images, same tau
    methods = [("Alg. I (full search)", FullSearch()),
               ("Alg. II baseline (per-image SOM)", PerImageSOM())]
    if a.features != "baseline":
        methods.append(("Alg. II per-image SOM (%s features)" % a.features, PerImageSOM(ff)))
    methods += [("Alg. II offline codebook (no NN)", CodebookGrouper(W, ff)),
                ("CONTROL: constant group (no input)", ConstantGrouper(W, const_g, ff))]
    for name, _ in variants:
        methods.append((name, NNGrouper(W, nets[name], lo, hi, a.topk, ff)))
    rows2, base_t = [], None
    for name, g in methods:
        ps, ts, cs, cands = [], [], [], []
        for img in ev:
            tt = []
            for _ in range(max(1, a.repeats)):
                e = encode_grouped(img, a.tau, g)
                tt.append(e.time)
            rec = fb.decode(e, img)
            ps.append(fb.psnr(img, rec)); ts.append(float(np.mean(tt))); cs.append(fb.compression_ratio(e))
            cands.append(e.cand / max(e.calls, 1))
        rows2.append([name, "%.2f" % np.mean(ps), "%.2f" % np.mean(ts), "%.2f" % np.mean(cs), "%.0f" % np.mean(cands)])
        print("  coded   %-34s PSNR=%s  time=%s  CR=%s  cand=%s" % tuple(rows2[-1]), flush=True)
    t2 = box("Table N2  Coding of the %d evaluation images, tau=%g (mean over images; training time NOT included)" %
             (len(ev), a.tau), ["Method", "PSNR (dB)", "Time (s)", "Comp. ratio", "Mean candidates"], rows2)
    off = ("Offline cost: codebook + dataset %.1f s (one time); network training times are in Table N1." % t_data)
    text = "\n".join(log) + "\n\n" + t1 + "\n\n" + t2 + "\n\n" + off + "\n"
    print("\n" + text)
    open(os.path.join(a.out, "tables_chaotic.txt"), "w", encoding="utf-8").write(text)
    np.savez(os.path.join(a.out, "bpnn_weights.npz"), codebook=W, lo=lo, hi=hi,
             **{n.replace(" ", "_").replace("/", "-"): np.concatenate([v.W1.ravel(), v.W2.ravel()]) for n, v in nets.items()})
    print("Saved in:", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
