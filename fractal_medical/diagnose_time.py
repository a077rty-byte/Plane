#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Where does the encoding time go?   (does NOT modify the baseline: it only times its functions)

    python diagnose_time.py --source /content/drive/MyDrive/archive.zip \
        --manifest /content/drive/MyDrive/results_baseline_final/selected_images.txt --image 4

For Algorithm I and II and every tau it prints: number of domain blocks, number of
matching calls, final number of range blocks, mean number of searched candidates per
call, and the seconds spent in SOM training, SOM labelling, best-match search and the rest.
"""
import argparse, time
import numpy as np
import fractal_baseline as fb

T = {}


def timed(fn, key, count_arg=None):
    def w(*a, **k):
        s = time.perf_counter()
        r = fn(*a, **k)
        T[key] = T.get(key, 0.0) + time.perf_counter() - s
        T[key + "_n"] = T.get(key + "_n", 0) + 1
        if count_arg is not None:
            T[key + "_cand"] = T.get(key + "_cand", 0) + len(a[count_arg])
        return r
    return w


fb.best_match = timed(fb.best_match, "search", count_arg=2)
fb.SOM.fit = timed(fb.SOM.fit, "som_fit")
fb.SOM.predict = timed(fb.SOM.predict, "som_pred")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--manifest", required=True, help="selected_images.txt of the baseline run")
    ap.add_argument("--image", type=int, default=4, help="Sample MRI Image number (1-based)")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    names, items = fb.select_images(a.source, "notumor", 5, "Training", 1, a.manifest)
    img = fb.load_image(items[names[a.image - 1]](), a.size)
    print("image:", names[a.image - 1], "| size", a.size, "| mean of", a.repeats, "runs\n")
    head = ["algo", "tau", "domains", "match calls", "final ranges", "mean cand.", "SOM fit", "SOM label", "search", "other", "TOTAL"]
    rows = []
    for algo in ("alg1", "alg2"):
        for tau in fb.TAU_LIST:
            acc = []
            for _ in range(a.repeats):
                T.clear()
                e = fb.encode(img, algo, tau)
                tot = e.time
                s, sf, sp = T.get("search", 0), T.get("som_fit", 0), T.get("som_pred", 0)
                acc.append((tot, sf, sp, s, T.get("search_n", 0), T.get("search_cand", 0)))
            m = np.mean(acc, 0)
            tot, sf, sp, s, n, cand = m
            rows.append([algo, "1e%d" % round(np.log10(tau)), str(len(e.dom)), "%d" % n, str(len(e.maps)),
                         "%.0f" % (cand / max(n, 1)), "%.2f" % sf, "%.2f" % sp, "%.2f" % s,
                         "%.2f" % (tot - sf - sp - s), "%.2f" % tot])
    w = [max(len(h), *(len(r[i]) for r in rows)) + 2 for i, h in enumerate(head)]
    ln = "+" + "+".join("-" * x for x in w) + "+"
    print(ln); print("|" + "|".join(h.center(x) for h, x in zip(head, w)) + "|"); print(ln)
    for r in rows:
        print("|" + "|".join(c.center(x) for c, x in zip(r, w)) + "|")
    print(ln)
    print("times in seconds; 'other' = quad-tree, domain-pool construction, isometries, bit counting")


if __name__ == "__main__":
    main()
