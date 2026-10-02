#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation study: effect of the SOM weight initialisation and learning-rate
schedule (random / logistic / chaotic-BSLC) on Fractal Algorithm II.

    python ablation.py                                  # 8 synthetic phantoms, 256x256
    python ablation.py --dir /content/mri --n 20 --size 512 --repeats 3

Reports mean +- std over (images x repeats) of
PSNR, encoding time, compression ratio and SOM quantisation error.
"""
import argparse, csv, glob, os, random
import numpy as np
import fractal_medical as fm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", help="folder with MRI images (searched recursively)")
    ap.add_argument("--n", type=int, default=8, help="number of images")
    ap.add_argument("--per-class", type=int, default=0,
                    help="if >0: take this many images from EACH class sub-folder of --dir")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--tau", type=float, default=1e-5)
    ap.add_argument("--repeats", type=int, default=3, help="seeds / chaotic start points per config")
    ap.add_argument("--groups", type=int, default=3)
    ap.add_argument("--out", default="results_ablation")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.dir:
        files = sorted(f for f in glob.glob(os.path.join(a.dir, "**", "*"), recursive=True)
                       if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")))
        random.seed(1)
        if a.per_class > 0:
            by = {}
            for f in files:
                by.setdefault(os.path.basename(os.path.dirname(f)), []).append(f)
            files = [f for k in sorted(by) for f in random.sample(by[k], min(a.per_class, len(by[k])))]
            print({k: min(a.per_class, len(v)) for k, v in sorted(by.items())})
        else:
            files = random.sample(files, min(a.n, len(files)))
        images = [fm.load_image(f, a.size) for f in files]
    else:
        images = [fm.make_phantom(i, a.size) for i in range(a.n)]
    print(f"{len(images)} images, {a.repeats} repeats per configuration")

    configs = [("random", "linear"), ("random", "chaotic"),
               ("logistic", "linear"), ("logistic", "chaotic"),
               ("chaotic", "linear"), ("chaotic", "chaotic")]
    base = dict(tau=a.tau, som_grid=(a.groups, a.groups))
    rows = []

    # reference: Algorithm I (no SOM, search over the whole domain pool)
    res = [fm.run(im, "alg1", **base) for im in images]
    ref = [(r[2], r[3], r[4], float("nan")) for r in res]
    rows.append(("Alg. I (no SOM)", "-", ref))

    for init, sched in configs:
        vals = []
        for im in images:
            for k in range(a.repeats):
                kw = dict(base, som_seed=k,
                          som_kw=dict(init=init, sched=sched,
                                      chaos_x0=0.2468 + 0.013 * k, chaos_y0=0.3691 + 0.009 * k))
                e, rec, p, tm, cr = fm.run(im, "alg2", **kw)
                qe = float(np.mean([pl.som.qe for pl in e.pools.values() if pl.labels is not None]))
                vals.append((p, tm, cr, qe))
        rows.append((init, sched, vals))
        print(f"  done init={init:8s} sched={sched}", flush=True)

    def ms(v, i):
        x = np.array([t[i] for t in v], float)
        return f"{np.nanmean(x):.2f} ± {np.nanstd(x):.2f}" if not np.all(np.isnan(x)) else "-"

    head = ["SOM init", "LR schedule", "PSNR (dB)", "Time (s)", "Comp. ratio", "SOM quant. error"]
    table = [[i, s, ms(v, 0), ms(v, 1), ms(v, 2), (ms(v, 3) if s != "-" else "-")] for i, s, v in rows]
    w = [max(len(h), *(len(r[c]) for r in table)) + 2 for c, h in enumerate(head)]
    ln = "+" + "+".join("-" * x for x in w) + "+"
    out = [f"Table A  Effect of SOM initialisation / schedule (tau={a.tau:g})", ln,
           "|" + "|".join(h.center(x) for h, x in zip(head, w)) + "|", ln]
    out += ["|" + "|".join(c.center(x) for c, x in zip(r, w)) + "|" for r in table] + [ln]
    text = "\n".join(out)
    print("\n" + text)
    open(os.path.join(a.out, "ablation.txt"), "w", encoding="utf-8").write(text + "\n")
    with open(os.path.join(a.out, "ablation.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f); wr.writerow(head); wr.writerows(table)


if __name__ == "__main__":
    main()
