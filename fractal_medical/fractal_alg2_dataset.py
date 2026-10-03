#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fractal coding with SOM-based domain / range pool partitioning  --  ANY-SIZE DATA SET
=====================================================================================
Single, self-contained file (needs only numpy + pillow).  Upload it to Google Drive / Colab.

Implements the paper
    S. Bhavani, K. Thanushkodi, "Neural based domain and range pool partitioning using
    Fractal Coding for nearly lossless Medical Image Compression", WSEAS Trans. SP, 2013
    standard : fixed range blocks, full search          (slow reference)
    alg1     : quad-tree + variance based domain/range separation (threshold tau)
    alg2     : alg1 + Self-Organising-Map groups; a range block is matched only
               with the domain blocks of its own group      <-- the paper's Algorithm II

What "any size" means here
  * ANY NUMBER OF IMAGES  - images are streamed one at a time (constant memory); every result
    is appended to results.csv at once, so a Colab disconnect loses nothing and re-running the
    same command RESUMES where it stopped.  --workers N codes N images in parallel.
  * ANY IMAGE SIZE / SHAPE - default: resized to --size x --size (paper: 512).  --size 0 keeps the
    native size (padded to a multiple of 32; images larger than --tile are coded tile by tile).
  * ANY FOLDER LAYOUT     - a .zip file or a folder, searched recursively; the class is the name of
    the parent folder (training/glioma/x.jpg -> 'glioma').  Works with 1 image or 1,000,000.

PAPER MODE  (--paper)  -  the experiment exactly as in the paper, for a small folder (e.g. 6 images)
    !python fractal_alg2_dataset.py --paper --source /content/drive/MyDrive/training --out /content/drive/MyDrive/paper_results
  Every image of the folder = "Sample MRI Image 1, 2, ...".  Tables 1-3: Algorithm II at tau = 1e-3 .. 1e-6
  (one row per image + Avg).  Table 4: standard / Algorithm I / Algorithm II at tau = 1e-5 on image 4
  (--t4-image N to change).  Figures 1, 2, 3, 5, 6, 7, 8 (needs matplotlib).  Output: tables.txt/.md, fig*.png.

Quick start (Colab)
    from google.colab import drive; drive.mount('/content/drive')
    !python /content/drive/MyDrive/fractal_alg2_dataset.py \
        --source /content/drive/MyDrive/archive.zip --out /content/drive/MyDrive/fractal_results
  Useful options
    --limit 200            use at most 200 images (random but reproducible, --seed)
    --per-class 50         at most 50 images of every class
    --split training       only paths that contain /training/ (default; falls back to all)
    --taus 1e-5            one threshold only (default: 1e-3 1e-4 1e-5 1e-6 as in the paper)
    --algos alg2 alg1      also run Algorithm I (add 'standard' for the full-search reference)
    --workers 4            parallel processes (timings are then slightly pessimistic)
    --size 0 --tile 512    native size, tiles of 512

Outputs (in --out)
    results.csv     one row per (image, algorithm, tau): PSNR, time, compression ratio, ...
    summary.txt/.md Tables 1-3 (mean over ALL images, plus per class) and Table 4 (algorithm comparison)
    settings.json   every parameter of the run (a run is resumed only if they are identical)
    recon/          first --save-recon reconstructed images next to their originals
"""
import argparse
import csv
import glob
import io
import json
import math
import os
import random
import sys
import time
import zipfile
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# ==========================================================================
# 1. Fixed parameters (same values as the paper-faithful reference implementation)
# ==========================================================================
IMG_SIZE = 512
TAU_LIST = [1e-3, 1e-4, 1e-5, 1e-6]
DMIN = 8                    # minimum domain block size d_min
MAX_SIZE = 32               # largest quad-tree block
RMIN = 4                    # smallest range block
SPLIT_VAR = 60.0            # quad-tree: split a block while its variance > SPLIT_VAR
S_LEVELS = np.linspace(-0.95, 0.95, 31)   # quantised contrast s_i  (|s| < 1: contractive)
S_BITS, O_BITS, ISO_BITS = 5, 9, 3
SOM_EPOCHS = 15
SOM_LR0 = 0.5
SOM_SEED = 0
STD_RANGE, STD_STRIDE = 8, 8
DECODE_ITERS = 12
EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
ALGOS = ("standard", "alg1", "alg2")
CSV_FIELDS = ["image", "cls", "algo", "tau", "tau_eff", "height", "width", "tiles", "psnr_db",
              "time_s", "cr", "domain_blocks", "range_maps", "bytes"]


# ==========================================================================
# 2. Utilities
# ==========================================================================
def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def isometry(blocks, k):
    """One of the 8 isometries of a square (4 rotations x optional flip)."""
    if k >= 4:
        blocks = blocks[..., :, ::-1]
    return np.rot90(blocks, k % 4, axes=(-2, -1))


def pool2(img):
    """P[y, x] = mean of img[y:y+2, x:x+2]  (2r x 2r domain -> r x r)."""
    return 0.25 * (img[:-1, :-1] + img[1:, :-1] + img[:-1, 1:] + img[1:, 1:])


def tau_name(t):
    return "tau=1e%d" % int(round(math.log10(t))) if t > 0 else "-"


# ==========================================================================
# 3. Data: any folder / zip, streamed one image at a time
# ==========================================================================
_ZIPS = {}      # one open ZipFile per process


def _zip(path):
    if path not in _ZIPS:
        _ZIPS[path] = zipfile.ZipFile(path)
    return _ZIPS[path]


def list_images(source):
    """[(name, class)] for every image in a .zip or a folder (recursive).  Names are relative paths."""
    out = []
    if os.path.isfile(source) and source.lower().endswith(".zip"):
        for n in sorted(_zip(source).namelist()):
            if n.lower().endswith(EXTS) and not n.endswith("/") and "__MACOSX" not in n \
                    and not os.path.basename(n).startswith("._"):
                out.append(n)
    elif os.path.isdir(source):
        for p in sorted(glob.glob(os.path.join(source, "**", "*"), recursive=True)):
            if p.lower().endswith(EXTS) and os.path.isfile(p):
                out.append(os.path.relpath(p, source).replace(os.sep, "/"))
    else:
        raise SystemExit("--source must be a .zip file or a folder: %s" % source)
    res = []
    for n in out:
        parent = os.path.basename(os.path.dirname(n))
        res.append((n, parent if parent else "(root)"))
    return res


def read_bytes(source, name):
    if os.path.isfile(source):
        return _zip(source).read(name)
    with open(os.path.join(source, name), "rb") as f:
        return f.read()


def select_images(items, split, limit, per_class, seed):
    """Reproducible choice: split filter -> per-class cap -> global cap (random, fixed seed)."""
    if split:
        tag = "/" + split.lower().strip("/") + "/"
        sp = [it for it in items if tag in "/" + it[0].lower()]
        if sp:
            items = sp
        else:
            print("note: no path contains '%s' -> using ALL images" % tag.strip("/"), flush=True)
    rnd = random.Random(seed)
    if per_class and per_class > 0:
        by = {}
        for it in items:
            by.setdefault(it[1], []).append(it)
        items = []
        for k in sorted(by):
            g = sorted(by[k])
            items += sorted(rnd.sample(g, per_class)) if len(g) > per_class else g
    if limit and 0 < limit < len(items):
        items = sorted(rnd.sample(sorted(items), limit))
    return sorted(items)


def decode_gray(data):
    """bytes -> 2-D float64 array in [0, 255] (handles 8/16-bit, RGB, palette, ...)."""
    from PIL import Image
    im = Image.open(io.BytesIO(data))
    if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        a = np.asarray(im, dtype=np.float64)
        lo, hi = a.min(), a.max()
        return (a - lo) * (255.0 / (hi - lo)) if hi > lo else np.zeros_like(a)
    return np.asarray(im.convert("L"), dtype=np.float64)


def prepare(gray, size):
    """Return (image to be coded, original height, original width).
    size > 0 : bicubic resize to size x size (size is rounded up to a multiple of 32).
    size = 0 : native size, padded (edge replication) to a multiple of 32, at least 64."""
    from PIL import Image
    h, w = gray.shape
    if size > 0:
        s = max(64, int(math.ceil(size / MAX_SIZE)) * MAX_SIZE)
        if (h, w) != (s, s):
            im = Image.fromarray(gray.astype(np.float32), mode="F").resize((s, s), Image.BICUBIC)
            gray = np.clip(np.asarray(im, dtype=np.float64), 0, 255)
        return np.rint(gray), s, s
    H = max(64, int(math.ceil(h / MAX_SIZE)) * MAX_SIZE)
    W = max(64, int(math.ceil(w / MAX_SIZE)) * MAX_SIZE)
    return np.rint(np.pad(gray, ((0, H - h), (0, W - w)), mode="edge")), h, w


# ==========================================================================
# 4. Quad-tree + domain / range separation  (rectangular images supported)
# ==========================================================================
def quadtree(img, max_size, min_size, split_var):
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

    H, W = img.shape
    for y in range(0, H, max_size):
        for x in range(0, W, max_size):
            rec(y, x, max_size)
    return leaves


def tau_effective(tau, rule="figure"):
    """rule 'figure': threshold on var/var_max chosen so that larger tau keeps MORE domain blocks
    (as in the paper's Fig. 2):  tau_eff = 0.2 * (tau / 1e-6) ** (-1/3).
    rule 'literal': the inequality exactly as printed (tau itself)."""
    return tau if rule == "literal" else 0.2 * (tau / 1e-6) ** (-1.0 / 3.0)


def separate_domain_range(img, leaves, dmin, tau, rule):
    t_eff = tau_effective(tau, rule)
    small = [(y, x, s) for (y, x, s) in leaves if s == dmin]
    vmax = max([img[y:y + s, x:x + s].var() for (y, x, s) in small] + [1e-9])
    dom, rng_ = [], []
    for (y, x, s) in leaves:
        if s == dmin and img[y:y + s, x:x + s].var() >= t_eff * vmax:
            dom.append((y, x, s))
        else:
            rng_.append((y, x, s))
    if not dom:                      # flat image: keep one seed block so the pool is never empty
        dom.append((0, 0, dmin))
    return dom, rng_, t_eff


# ==========================================================================
# 5. Self-Organising Map (Kohonen)
# ==========================================================================
class SOM:
    """1) random weights 2) take an input vector 3) visit every node 4) Euclidean distance
    5) smallest distance = BMU 6) pull the BMU neighbourhood towards the input:
        Wv(t+1) = Wv(t) + theta(t) * alpha(t) * (D(t) - Wv(t))
    theta = Gaussian neighbourhood; alpha and sigma decrease linearly.  No activation function."""

    def __init__(self, grid, epochs=SOM_EPOCHS, lr0=SOM_LR0, seed=SOM_SEED):
        self.grid, self.epochs, self.lr0 = grid, epochs, lr0
        self.rs = np.random.RandomState(seed)
        gy, gx = np.mgrid[0:grid[0], 0:grid[1]]
        self.pos = np.stack([gy.ravel(), gx.ravel()], 1).astype(float)
        self.n = grid[0] * grid[1]

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        idx = self.rs.choice(len(X), self.n, replace=len(X) < self.n)
        self.W = X[idx].copy() + 1e-3 * self.rs.randn(self.n, X.shape[1])
        total = self.epochs * len(X)
        t, sig0 = 0, max(self.grid) / 2.0
        for _ in range(self.epochs):
            for i in self.rs.permutation(len(X)):
                f = 1.0 - t / total
                alpha, sig = self.lr0 * f, max(0.5, sig0 * f)
                bmu = ((self.W - X[i]) ** 2).sum(1).argmin()
                theta = np.exp(-((self.pos - self.pos[bmu]) ** 2).sum(1) / (2 * sig ** 2))
                self.W += (theta * alpha)[:, None] * (X[i] - self.W)
                t += 1
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return ((X[:, None, :] - self.W[None]) ** 2).sum(2).argmin(1)


def block_features(blocks):
    """Isometry-invariant feature vector of (N, r, r) blocks:
    mean, std and the sorted, normalised quadrant means."""
    h = blocks.shape[1] // 2
    q = np.stack([blocks[:, :h, :h].mean((1, 2)), blocks[:, :h, h:].mean((1, 2)),
                  blocks[:, h:, :h].mean((1, 2)), blocks[:, h:, h:].mean((1, 2))], 1)
    mu, sd = blocks.mean((1, 2)), blocks.std((1, 2))
    dev = np.sort(q - mu[:, None], 1) / (sd[:, None] + 1.0)
    return np.column_stack([mu / 255.0, np.minimum(sd, 64) / 64.0, 0.3 * dev])


# ==========================================================================
# 6. Matching machinery
# ==========================================================================
class DomainPool:
    """Domain candidates for ONE range size r: 2r x 2r windows anchored at the domain (seed)
    blocks, down-sampled to r x r, each with its 8 isometries."""

    def __init__(self, P, anchors, r, H, W, grid=None):
        pos = sorted({(min(y, H - 2 * r), min(x, W - 2 * r)) for (y, x) in anchors})
        self.pos = np.array(pos)
        base = np.stack([P[y:y + 2 * r:2, x:x + 2 * r:2] for (y, x) in pos])
        self.n, self.r = len(pos), r
        self.all = np.stack([isometry(base, k).reshape(self.n, -1) for k in range(8)]).astype(np.float32)
        self.sum_d = self.all[0].sum(1)
        self.sum_dd = (self.all[0] ** 2).sum(1)
        self.labels = None
        if grid:
            feats = block_features(base)
            self.som = SOM(grid).fit(feats)
            self.labels = self.som.predict(feats)

    def candidates(self, label=None):
        if label is None or self.labels is None:
            return np.arange(self.n)
        idx = np.where(self.labels == label)[0]
        if len(idx) == 0:                                   # empty group -> nearest non-empty group
            used = np.unique(self.labels)
            d = ((self.som.W[used] - self.som.W[label]) ** 2).sum(1)
            idx = np.where(self.labels == used[d.argmin()])[0]
        return idx


def best_match(rv, pool, idx):
    """Least-squares (s, o) for every candidate and isometry; best (domain, isometry, s index, o, SSE)."""
    n = rv.size
    sr, srr = rv.sum(), (rv ** 2).sum()
    D = pool.all[:, idx, :]
    sd, sdd = pool.sum_d[idx], pool.sum_dd[idx]
    sdr = D @ rv.astype(np.float32)
    den = n * sdd - sd ** 2
    s = np.where(den > 1e-6, (n * sdr - sd * sr) / np.maximum(den, 1e-6), 0.0)
    si = np.abs(s[..., None] - S_LEVELS).argmin(-1)
    sq = S_LEVELS[si]
    o = np.clip(np.rint((sr - sq * sd) / n), -256, 255)
    sse = sq ** 2 * sdd + n * o ** 2 + srr + 2 * sq * o * sd - 2 * sq * sdr - 2 * o * sr
    k, m = np.unravel_index(sse.argmin(), sse.shape)
    return int(idx[m]), int(k), int(si[k, m]), int(o[k, m]), float(max(sse[k, m], 0))


# ==========================================================================
# 7. Encoder / decoder (one image or one tile)
# ==========================================================================
class Encoded:
    pass


def encode(img, algo, tau, cfg):
    t0 = time.perf_counter()
    H, W = img.shape
    P = pool2(img)
    e = Encoded()
    e.algo, e.shape, e.maps = algo, img.shape, []
    e.tau_eff, e.dom, e.leaves = 0.0, [], []

    if algo == "standard":
        e.seed_mask = np.zeros(img.shape, bool)
        anchors = [(y, x) for y in range(0, H - 2 * STD_RANGE + 1, STD_STRIDE)
                   for x in range(0, W - 2 * STD_RANGE + 1, STD_STRIDE)]
        pool = DomainPool(P, anchors, STD_RANGE, H, W)
        e.pools, e.seed_bytes = {STD_RANGE: pool}, 0
        for y in range(0, H, STD_RANGE):
            for x in range(0, W, STD_RANGE):
                d, k, si, o, _ = best_match(img[y:y + STD_RANGE, x:x + STD_RANGE].ravel(), pool, pool.candidates())
                e.maps.append((y, x, STD_RANGE, d, k, si, o))
        e.bits = _count_bits(e, 0)
        e.time = time.perf_counter() - t0
        return e

    leaves = quadtree(img, MAX_SIZE, DMIN, SPLIT_VAR)
    dom, rng_blocks, e.tau_eff = separate_domain_range(img, leaves, DMIN, tau, cfg["rule"])
    e.leaves, e.dom = leaves, dom
    e.seed_mask = np.zeros(img.shape, bool)
    for (y, x, s) in dom:
        e.seed_mask[y:y + s, x:x + s] = True
    e.seed_bytes = len(zlib.compress(img[e.seed_mask].astype(np.uint8).tobytes(), 9))   # lossless part
    anchors = [(y, x) for (y, x, s) in dom]
    grid = (cfg["groups"], cfg["groups"]) if algo == "alg2" else None
    e.pools, r = {}, MAX_SIZE
    while r >= RMIN:
        e.pools[r] = DomainPool(P, anchors, r, H, W, grid)
        r //= 2

    stack = list(rng_blocks)
    while stack:
        y, x, r = stack.pop()
        blk = img[y:y + r, x:x + r]
        pool = e.pools[r]
        lab = int(pool.som.predict(block_features(blk[None]))[0]) if algo == "alg2" else None
        d, k, si, o, sse = best_match(blk.ravel(), pool, pool.candidates(lab))
        if math.sqrt(sse / (r * r)) > cfg["tol"] and r > RMIN:      # poor match -> 4 smaller range blocks
            h = r // 2
            stack += [(y, x, h), (y, x + h, h), (y + h, x, h), (y + h, x + h, h)]
        else:
            e.maps.append((y, x, r, d, k, si, o))
    e.bits = _count_bits(e, len(leaves) * 4 // 3 + len(dom) // 8)
    e.time = time.perf_counter() - t0
    return e


def _count_bits(e, tree_bits):
    bits = tree_bits
    for (y, x, r, d, k, si, o) in e.maps:
        bits += math.ceil(math.log2(max(2, e.pools[r].n))) + ISO_BITS + S_BITS + O_BITS
        if e.algo != "standard" and r > RMIN:
            bits += 1
    return bits + 8 * e.seed_bytes


def decode(e, original):
    """IFS decoding from a flat grey image; the lossless seed blocks are restored at every iteration."""
    cur = np.full(e.shape, 128.0)
    use_seed = e.seed_mask.any()
    if use_seed:
        cur[e.seed_mask] = original[e.seed_mask]
    for _ in range(DECODE_ITERS):
        P = pool2(cur)
        new = cur.copy()
        for (y, x, r, d, k, si, o) in e.maps:
            py, px = e.pools[r].pos[d]
            new[y:y + r, x:x + r] = S_LEVELS[si] * isometry(P[py:py + 2 * r:2, px:px + 2 * r:2], k) + o
        cur = np.clip(new, 0, 255)
        if use_seed:
            cur[e.seed_mask] = original[e.seed_mask]
    return np.rint(cur)


def code_image(img, algo, tau, cfg, tile):
    """Encode + decode a (padded) image, tile by tile when it is larger than `tile`.
    Returns reconstruction, total bits, encoding time, #domain blocks, #maps, #tiles, tau_eff."""
    H, W = img.shape
    rec = np.zeros_like(img)
    bits = dom = maps = ntile = 0
    tm, teff = 0.0, 0.0
    for y in range(0, H, tile):
        for x in range(0, W, tile):
            th, tw = min(tile, H - y), min(tile, W - x)
            y0, x0 = (y, x) if th >= 64 and tw >= 64 else (y if th >= 64 else H - 64, x if tw >= 64 else W - 64)
            win = img[y0:y0 + max(th, 64), x0:x0 + max(tw, 64)]    # thin border strips borrow pixels from the neighbour
            e = encode(win, algo, tau, cfg)
            full = decode(e, win)
            rec[y:y + th, x:x + tw] = full[y - y0:y - y0 + th, x - x0:x - x0 + tw]
            bits += e.bits
            tm += e.time
            dom += len(e.dom)
            maps += len(e.maps)
            teff = e.tau_eff
            ntile += 1
    return rec, bits, tm, dom, maps, ntile, teff


# ==========================================================================
# 8. One image -> result rows (runs in a worker process)
# ==========================================================================
def process_image(job):
    source, name, cls, todo, cfg = job            # todo = [(algo, tau)]
    t_load = time.perf_counter()
    gray = decode_gray(read_bytes(source, name))
    img, oh, ow = prepare(gray, cfg["size"])
    if cfg["size"] > 0:
        oh, ow = img.shape
    orig = img[:oh, :ow]
    rows, recs = [], {}
    for algo, tau in todo:
        rec, bits, tm, dom, maps, ntile, teff = code_image(img, algo, tau, cfg, cfg["tile"])
        rec = rec[:oh, :ow]
        recs[(algo, tau)] = rec
        rows.append(dict(image=name, cls=cls, algo=algo, tau=tau if algo != "standard" else 0, tau_eff=round(teff, 5),
                         height=oh, width=ow, tiles=ntile, psnr_db=round(psnr(orig, rec), 4),
                         time_s=round(tm, 4), cr=round(oh * ow * 8.0 / bits, 4), domain_blocks=dom,
                         range_maps=maps, bytes=int(math.ceil(bits / 8))))
    return name, rows, (orig, recs) if cfg["save_recon"] else None


# ==========================================================================
# 9. CSV (resumable), summary tables
# ==========================================================================
def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def box(title, head, body):
    w = [max(len(h), *(len(r[i]) for r in body)) + 2 for i, h in enumerate(head)]
    ln = "+" + "+".join("-" * x for x in w) + "+"
    o = [title, ln, "|" + "|".join(h.center(x) for h, x in zip(head, w)) + "|", ln]
    for r in body:
        if r[0] in ("Avg", "Avg (all)"):
            o.append(ln)
        o.append("|" + "|".join(c.center(x) for c, x in zip(r, w)) + "|")
    return "\n".join(o + [ln])


def md(title, head, body):
    return ("**%s**\n\n| " % title + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n" +
            "\n".join("| " + " | ".join(r) + " |" for r in body) + "\n")


def write_summary(out, settings_lines):
    rows = read_rows(os.path.join(out, "results.csv"))
    if not rows:
        return
    for r in rows:
        for k in ("tau", "psnr_db", "time_s", "cr", "domain_blocks"):
            r[k] = float(r[k])
    algos = [a for a in ALGOS if any(r["algo"] == a for r in rows)]
    n_img = len({r["image"] for r in rows})
    classes = sorted({r["cls"] for r in rows})
    blocks, mds = [], []

    for algo in algos:
        sub = [r for r in rows if r["algo"] == algo]
        taus = sorted({r["tau"] for r in sub}, reverse=True)
        head = ["Class (images)"] + [tau_name(t) if t else "full search" for t in taus]
        for key, title, fmt in (("psnr_db", "PSNR (dB)", "%.2f"), ("time_s", "Encoding time per image (s)", "%.2f"),
                                ("cr", "Compression ratio", "%.2f")):
            body = []
            for c in classes:
                n = len({r["image"] for r in sub if r["cls"] == c})
                if n:
                    body.append(["%s (%d)" % (c, n)] + [
                        fmt % np.mean([r[key] for r in sub if r["cls"] == c and r["tau"] == t] or [float("nan")])
                        for t in taus])
            body.append(["Avg (all)"] + [fmt % np.mean([r[key] for r in sub if r["tau"] == t]) for t in taus])
            ttl = "%s  -  %s  (algorithm: %s, mean over images)" % ({"psnr_db": "Table 1", "time_s": "Table 2", "cr": "Table 3"}[key], title, algo)
            blocks.append(box(ttl, head, body))
            mds.append(md(ttl, head, body))
    if len(algos) > 1:                                       # Table 4: algorithms on the same images
        taus = sorted({r["tau"] for r in rows if r["algo"] != "standard"}, reverse=True)
        t4 = 1e-5 if 1e-5 in taus else (taus[len(taus) // 2] if taus else 0)
        keys = [{r["image"] for r in rows if r["algo"] == a and (r["tau"] == t4 or a == "standard")} for a in algos]
        common = set.intersection(*keys) if keys else set()
        if common:
            head = ["Metric (%d common images)" % len(common)] + [a if a == "standard" else "%s (%s)" % (a, tau_name(t4)) for a in algos]
            sel = {a: [r for r in rows if r["algo"] == a and r["image"] in common and (r["tau"] == t4 or a == "standard")] for a in algos}
            body = [["PSNR (dB)"] + ["%.2f" % np.mean([r["psnr_db"] for r in sel[a]]) for a in algos],
                    ["Compression time (s)"] + ["%.2f" % np.mean([r["time_s"] for r in sel[a]]) for a in algos],
                    ["Compression ratio"] + ["%.2f" % np.mean([r["cr"] for r in sel[a]]) for a in algos]]
            ttl = "Table 4  -  Algorithms compared (mean over the common images)"
            blocks.append(box(ttl, head, body))
            mds.append(md(ttl, head, body))
    header = "Images coded: %d   |   rows in results.csv: %d" % (n_img, len(rows))
    text = header + "\n" + "\n".join(settings_lines) + "\n\n" + "\n\n".join(blocks) + "\n"
    open(os.path.join(out, "summary.txt"), "w", encoding="utf-8").write(text)
    open(os.path.join(out, "summary.md"), "w", encoding="utf-8").write(
        "```\n" + header + "\n" + "\n".join(settings_lines) + "\n```\n\n" + "\n".join(mds))
    print("\n" + text)


# ==========================================================================
# 9b. PAPER MODE: the paper's experiment exactly (Tables 1-4, Figures 1,2,3,5,6,7,8)
# ==========================================================================
PAPER_TABLE4 = {"PSNR (dB)": (27.49, 29.67, 29.72), "Compression Time (sec)": (1738, 459.73, 37.17),
                "Compression Ratio": (3.20, 19.6, 19.6)}


def make_figures(out, images, enc, t4, t4_idx, taus):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    n = len(images)
    tl = lambda t: "1e%d" % int(round(math.log10(t)))
    t_fig1 = 1e-5 if 1e-5 in taus else taus[len(taus) // 2]

    sel = [0, min(1, n - 1)]                                   # Fig. 1: images 1 and 2
    fig, ax = plt.subplots(3, 2, figsize=(8, 12))
    for c, i in enumerate(sel):
        e = enc[(i, t_fig1)]
        ax[0, c].imshow(images[i], cmap="gray"); ax[0, c].set_title("Sample MRI Image %d" % (i + 1))
        ax[1, c].imshow(images[i], cmap="gray")
        for (y, x, s) in e.leaves:
            ax[1, c].add_patch(Rectangle((x, y), s, s, fill=False, ec="w", lw=0.4))
        m = np.ones(images[i].shape); m[e.seed_mask] = 0
        ax[2, c].imshow(m, cmap="gray", vmin=0, vmax=1)
    ax[0, 0].set_ylabel("Input Image"); ax[1, 0].set_ylabel("The Partitioned Image"); ax[2, 0].set_ylabel("Separated Domain Blocks")
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Fig. 1  Feature rich and separated domain blocks")
    fig.savefig(os.path.join(out, "fig1_partition.png"), dpi=130, bbox_inches="tight"); plt.close(fig)

    sel = [min(2, n - 1), min(3, n - 1)]                       # Fig. 2: images 3 and 4, every threshold
    fig, ax = plt.subplots(len(taus) + 1, 2, figsize=(8, 3.6 * (len(taus) + 1)), squeeze=False)
    for c, i in enumerate(sel):
        ax[0, c].imshow(images[i], cmap="gray"); ax[0, c].set_title("Sample MRI Image %d" % (i + 1))
        for r, t in enumerate(taus):
            m = np.ones(images[i].shape); m[enc[(i, t)].seed_mask] = 0
            ax[r + 1, c].imshow(m, cmap="gray", vmin=0, vmax=1)
            ax[r + 1, c].set_ylabel("Threshold = %s" % tl(t))
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Fig. 2  Domain blocks separated for various thresholds")
    fig.savefig(os.path.join(out, "fig2_thresholds.png"), dpi=110, bbox_inches="tight"); plt.close(fig)

    avg = {k: [np.mean([enc[(i, t)].metrics[k] for i in range(n)]) for t in taus] for k in ("psnr", "time", "cr")}
    fig, ax = plt.subplots(figsize=(7, 5))                     # Fig. 3
    xs = [tl(t) for t in taus]
    ax.plot(xs, avg["psnr"], "bd-", label="PSNR"); ax.plot(xs, avg["time"], "ms-", label="Encoding Time")
    ax.plot(xs, avg["cr"], "b^-", mfc="none", label="Compression Ratio")
    ax.set_xlabel("The Threshold"); ax.set_ylabel("Performance"); ax.grid(True); ax.legend(); ax.set_title("The Performance Analysis")
    fig.savefig(os.path.join(out, "fig3_performance.png"), dpi=130, bbox_inches="tight"); plt.close(fig)

    nm = ["The Std. Fractal compression Algorithm", "The Proposed Algorithm I", "The Proposed Algorithm II"]
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.4))            # Fig. 5
    ax[0].imshow(images[t4_idx], cmap="gray"); ax[0].set_title("Original (Sample image %d)" % (t4_idx + 1))
    for a, k, name in zip(ax[1:], ALGOS, nm):
        a.imshow(t4[k][3], cmap="gray", vmin=0, vmax=255); a.set_title("%s\nPSNR=%.2f dB" % (name, t4[k][0]), fontsize=9)
    for a in ax:
        a.axis("off")
    fig.suptitle("Fig. 5  PSNR for all three algorithms")
    fig.savefig(os.path.join(out, "fig5_reconstruction.png"), dpi=130, bbox_inches="tight"); plt.close(fig)

    bn = ["Standard\nFractal Encoding", "Proposed\nAlgorithm I", "Proposed\nAlgorithm II"]   # Fig. 6, 7, 8
    for fn, key, yl, ttl, col in (("fig6_psnr.png", 0, "PSNR(db)", "PSNR", "#9999ff"),
                                  ("fig7_time.png", 1, "Compression Time(sec)", "Time Taken for Different Methods", "#339966"),
                                  ("fig8_cr.png", 2, "Compression Ratio", "Compression Ratio of Different Methods", "#3366ff")):
        fig, ax = plt.subplots(figsize=(5, 4))
        vals = [t4[a][key] for a in ALGOS]
        bars = ax.bar(bn, vals, color=col, ec="k")
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v, "%.2f" % v, ha="center", va="bottom")
        ax.set_ylabel(yl); ax.set_title(ttl); ax.grid(axis="y")
        fig.savefig(os.path.join(out, fn), dpi=130, bbox_inches="tight"); plt.close(fig)


def paper_main(a, items, cfg, taus):
    """The paper's experiment on the images of --source (the 'training' folder, e.g. 6 images):
    Tables 1-3 = Algorithm II, one row per image + Avg; Table 4 = standard / Alg. I / Alg. II at
    tau = 1e-5 on image --t4-image (paper: image 4); Figures 1, 2, 3, 5, 6, 7, 8."""
    from PIL import Image
    out, n = a.out, len(items)
    names = [it[0] for it in items]
    images = []
    for nm in names:
        img, h, w = prepare(decode_gray(read_bytes(a.source, nm)), a.size or IMG_SIZE)
        images.append(img)
    for i, im in enumerate(images):
        Image.fromarray(im.astype(np.uint8)).save(os.path.join(out, "input_%d.png" % (i + 1)))
    t4_idx = min(max(a.t4_image, 1), n) - 1
    mapping = ["Sample MRI Image %d = %s" % (i + 1, nm) for i, nm in enumerate(names)]
    settings = ["Settings: PAPER MODE  images=%d  size=%dx%d  SOM grid=%dx%d (%d epochs)  tol=%.1f  rule=%s"
                % (n, images[0].shape[0], images[0].shape[1], a.groups, a.groups, SOM_EPOCHS, a.tol, a.rule),
                "  quad-tree: max block %d, d_min %d, r_min %d, split variance %.0f; tau_eff: %s"
                % (MAX_SIZE, DMIN, RMIN, SPLIT_VAR, ", ".join("%s->%.4f" % (tau_name(t), tau_effective(t, a.rule)) for t in taus)),
                "  Table 4 image: Sample MRI Image %d" % (t4_idx + 1)]
    print("\n".join(mapping + [""] + settings), flush=True)

    def run(img, algo, tau):
        e = encode(img, algo, tau, cfg)
        rec = decode(e, img)
        e.metrics = dict(psnr=psnr(img, rec), time=e.time, cr=img.size * 8.0 / e.bits)
        return e, rec

    enc = {}
    for i, im in enumerate(images):                             # Tables 1-3: Algorithm II
        for t in taus:
            e, _ = run(im, "alg2", t)
            enc[(i, t)] = e
            print("  image %d  %s  PSNR=%6.2f  time=%7.2f s  CR=%6.2f  domain blocks=%d"
                  % (i + 1, tau_name(t), e.metrics["psnr"], e.metrics["time"], e.metrics["cr"], len(e.dom)), flush=True)
    head = ["Sample MRI Image"] + [tau_name(t) for t in taus]
    blocks, mds = [], []
    for key, title in (("psnr", "Table 1  PSNR at Different Levels of Compression (dB)"),
                       ("time", "Table 2  Time Taken at Different Levels of Compression (sec)"),
                       ("cr", "Table 3  Compression Ratio at Different Thresholds")):
        rows = [[str(i + 1)] + ["%.2f" % enc[(i, t)].metrics[key] for t in taus] for i in range(n)]
        rows.append(["Avg"] + ["%.2f" % np.mean([enc[(i, t)].metrics[key] for i in range(n)]) for t in taus])
        blocks.append(box(title, head, rows)); mds.append(md(title, head, rows))

    t4, tau4 = {}, 1e-5                                         # Table 4: the three algorithms
    for algo in ALGOS:
        e, rec = run(images[t4_idx], algo, tau4)
        t4[algo] = (e.metrics["psnr"], e.metrics["time"], e.metrics["cr"], rec)
        print("  Table 4  %-8s PSNR=%6.2f  time=%8.2f s  CR=%6.2f" % (algo, *t4[algo][:3]), flush=True)
    h4 = ["Metric", "Standard Fractal Encoding", "Proposed Algorithm I (tau=1e-5)", "Proposed Algorithm II (tau=1e-5)"]
    r4 = [[m] + ["%.2f" % t4[k][j] for k in ALGOS] for j, m in enumerate(("PSNR (dB)", "Compression Time (sec)", "Compression Ratio"))]
    ttl4 = "Table 4  PSNR achieved for Different Algorithms (Sample MRI Image %d)" % (t4_idx + 1)
    blocks.append(box(ttl4, h4, r4)); mds.append(md(ttl4, h4, r4))
    ref = [[m] + ["%g" % v for v in vals] for m, vals in PAPER_TABLE4.items()]
    ttl_ref = "For reference - values printed in the paper's Table 4 (its own MRI image, its own machine)"
    blocks.append(box(ttl_ref, h4[:1] + ["Standard", "Algorithm I", "Algorithm II"], ref))

    text = "\n".join(mapping) + "\n\n" + "\n".join(settings) + "\n\n" + "\n\n".join(blocks) + "\n"
    print("\n" + text)
    open(os.path.join(out, "tables.txt"), "w", encoding="utf-8").write(text)
    open(os.path.join(out, "tables.md"), "w", encoding="utf-8").write(
        "```\n" + "\n".join(mapping + [""] + settings) + "\n```\n\n" + "\n".join(mds))
    try:
        make_figures(out, images, enc, t4, t4_idx, taus)
    except ImportError:
        print("matplotlib is not installed -> figures skipped (pip install matplotlib)")
    print("Saved in:", os.path.abspath(out))


# ==========================================================================
# 10. Main
# ==========================================================================
def maybe_mount_drive(path):
    if "google.colab" in sys.modules and str(path).startswith("/content/drive") \
            and not os.path.isdir("/content/drive/MyDrive"):
        from google.colab import drive
        drive.mount("/content/drive")


def fmt_time(s):
    s = int(s)
    return "%d:%02d:%02d" % (s // 3600, s % 3600 // 60, s % 60)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help=".zip file or folder with the images (any size, any number)")
    ap.add_argument("--out", default="fractal_results", help="output folder (results.csv is appended -> resumable)")
    ap.add_argument("--algos", nargs="+", default=["alg2"], choices=ALGOS, help="default: alg2 (the paper's Algorithm II)")
    ap.add_argument("--taus", nargs="+", type=float, default=TAU_LIST, help="thresholds (default 1e-3 1e-4 1e-5 1e-6)")
    ap.add_argument("--split", default="training", help="keep paths containing /<split>/ ('' = all; falls back to all)")
    ap.add_argument("--limit", type=int, default=0, help="at most this many images (0 = all)")
    ap.add_argument("--per-class", type=int, default=0, help="at most this many images per class (0 = all)")
    ap.add_argument("--seed", type=int, default=1, help="seed of the reproducible image choice")
    ap.add_argument("--size", type=int, default=IMG_SIZE, help="resize to size x size (paper: 512); 0 = native size")
    ap.add_argument("--tile", type=int, default=512, help="native-size mode: images larger than this are coded tile by tile (multiple of 64)")
    ap.add_argument("--groups", type=int, default=3, help="SOM grid side: groups = groups x groups (default 9)")
    ap.add_argument("--tol", type=float, default=5.0, help="range-block RMS error above which a block is split")
    ap.add_argument("--rule", choices=["figure", "literal"], default="figure", help="how tau maps to the variance threshold")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes (1 = exact timings)")
    ap.add_argument("--save-recon", type=int, default=3, help="save the first N reconstructions to recon/ (0 = none)")
    ap.add_argument("--paper", action="store_true",
                    help="reproduce the paper's experiment on ALL images of --source (e.g. your 6 'training' images): "
                         "Tables 1-4 in the paper's layout + Figures 1,2,3,5,6,7,8")
    ap.add_argument("--t4-image", type=int, default=4, help="--paper: sample image used in Table 4 (paper: 4)")
    ap.add_argument("--force", action="store_true", help="continue even if the settings differ from the earlier run")
    a, _ = ap.parse_known_args(argv)          # parse_known_args: harmless inside Jupyter / Colab

    a.tile = max(64, (a.tile // 64) * 64)
    maybe_mount_drive(a.source)
    maybe_mount_drive(a.out)
    os.makedirs(a.out, exist_ok=True)
    cfg = dict(size=a.size, tile=a.tile, groups=a.groups, tol=a.tol, rule=a.rule, save_recon=a.save_recon > 0)
    taus = sorted(set(a.taus), reverse=True)
    algos = [x for x in ALGOS if x in a.algos]
    if a.paper:
        items = sorted(list_images(a.source))
        if a.limit and 0 < a.limit < len(items):
            items = items[:a.limit]
        if not items:
            raise SystemExit("no images found in %s" % a.source)
        return paper_main(a, items, cfg, taus)
    settings = dict(algos=algos, taus=taus, size=a.size, tile=a.tile, groups=a.groups, tol=a.tol, rule=a.rule,
                    split=a.split, limit=a.limit, per_class=a.per_class, seed=a.seed, source=os.path.abspath(a.source),
                    dmin=DMIN, max_block=MAX_SIZE, rmin=RMIN, som_epochs=SOM_EPOCHS)
    sp = os.path.join(a.out, "settings.json")
    if os.path.exists(sp) and not a.force:
        old = json.load(open(sp))
        keys = ("size", "tile", "groups", "tol", "rule", "dmin", "max_block", "rmin", "som_epochs")
        if any(old.get(k) != settings[k] for k in keys):
            raise SystemExit("--out already holds results made with DIFFERENT settings (%s).\n"
                             "Use another --out folder, or --force to mix them." % ", ".join(k for k in keys if old.get(k) != settings[k]))
    json.dump(settings, open(sp, "w"), indent=1)

    items = select_images(list_images(a.source), a.split, a.limit, a.per_class, a.seed)
    if not items:
        raise SystemExit("no images found in %s" % a.source)
    csv_path = os.path.join(a.out, "results.csv")
    done = {(r["image"], r["algo"], float(r["tau"])) for r in read_rows(csv_path)}
    jobs = []
    for name, cls in items:
        todo = [(al, 0.0 if al == "standard" else t) for al in algos for t in ([0.0] if al == "standard" else taus)]
        todo = [(al, t) for al, t in todo if (name, al, t) not in done]
        if todo:
            jobs.append((os.path.abspath(a.source), name, cls, todo, cfg))
    by_cls = {}
    for _, c in items:
        by_cls[c] = by_cls.get(c, 0) + 1
    print("images selected: %d  (%s)" % (len(items), ", ".join("%s: %d" % kv for kv in sorted(by_cls.items()))))
    print("already finished: %d   to do now: %d   algorithms: %s   taus: %s   size: %s   workers: %d"
          % (len(items) - len(jobs), len(jobs), "+".join(algos), " ".join("%g" % t for t in taus),
             a.size or "native", a.workers), flush=True)

    settings_lines = ["Settings: algos=%s size=%s tile=%d SOM grid=%dx%d (%d epochs) tol=%.1f rule=%s split='%s' seed=%d"
                      % ("+".join(algos), a.size or "native", a.tile, a.groups, a.groups, SOM_EPOCHS, a.tol, a.rule, a.split, a.seed),
                      "  quad-tree: max block %d, d_min %d, r_min %d, split variance %.0f; tau_eff: %s"
                      % (MAX_SIZE, DMIN, RMIN, SPLIT_VAR, ", ".join("%s->%.4f" % (tau_name(t), tau_effective(t, a.rule)) for t in taus))]

    new_file = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    t0, ndone, saved = time.time(), 0, 0
    if jobs:
        fh = open(csv_path, "a", newline="", encoding="utf-8")
        wr = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new_file:
            wr.writeheader()
        recon_dir = os.path.join(a.out, "recon")

        def handle(name, rows, extra):
            nonlocal ndone, saved
            for r in rows:
                wr.writerow(r)
            fh.flush()
            os.fsync(fh.fileno())
            ndone += 1
            if extra and saved < a.save_recon:
                from PIL import Image
                os.makedirs(recon_dir, exist_ok=True)
                orig, recs = extra
                stem = "%03d_%s" % (saved + 1, os.path.splitext(os.path.basename(name))[0])
                Image.fromarray(orig.astype(np.uint8)).save(os.path.join(recon_dir, stem + "_original.png"))
                for (al, t), rc in recs.items():
                    Image.fromarray(rc.astype(np.uint8)).save(os.path.join(recon_dir, "%s_%s_%s.png" % (stem, al, tau_name(t))))
                saved += 1
            el = time.time() - t0
            eta = el / ndone * (len(jobs) - ndone)
            print("[%d/%d] %-45s %s | elapsed %s  ETA %s" % (
                ndone, len(jobs), name[-45:], "  ".join("%s %s %.1fdB %.1fs CR%.1f" % (r["algo"], tau_name(float(r["tau"])), float(r["psnr_db"]), float(r["time_s"]), float(r["cr"])) for r in rows[:3]) + (" ..." if len(rows) > 3 else ""),
                fmt_time(el), fmt_time(eta)), flush=True)

        try:
            if a.workers > 1:
                with ProcessPoolExecutor(max_workers=a.workers) as ex:
                    futs = [ex.submit(process_image, j) for j in jobs]
                    for f in as_completed(futs):
                        try:
                            handle(*f.result())
                        except Exception as err:               # a broken image must not stop the run
                            print("  ! skipped an image: %r" % err, flush=True)
            else:
                for j in jobs:
                    try:
                        handle(*process_image(j))
                    except Exception as err:
                        print("  ! skipped %s: %r" % (j[1], err), flush=True)
        except KeyboardInterrupt:
            print("\ninterrupted - results so far are saved; run the same command again to resume.")
        finally:
            fh.close()
    write_summary(a.out, settings_lines)
    print("Saved in:", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
