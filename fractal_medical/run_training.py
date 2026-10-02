#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fractal coding of the MRI 'training' folder (one image per class)
==================================================================
Reproduces Tables 1-4 of the paper, with the rows named after the CLASSES of the
data set (glioma / meningioma / notumor / pituitary ...).

    python run_training.py --root /content/drive/MyDrive/training
    python run_training.py --root ... --chaos            # SOM with chaotic init + chaotic learning rate
    python run_training.py --root ... --compare-chaos    # extra Table 5: baseline SOM vs chaotic SOM

The folder must contain one sub-folder per class (training/glioma/*.jpg, ...).
If it contains images directly, 4 random images are used instead.
"""
import argparse, glob, math, os, random
import numpy as np
import fractal_medical as fm

EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def pick_images(root, per_class_seed, max_classes=4):
    files = sorted(f for f in glob.glob(os.path.join(root, "**", "*"), recursive=True)
                   if f.lower().endswith(EXT))
    if not files:
        raise SystemExit("No images found under %s" % root)
    by = {}
    for f in files:
        by.setdefault(os.path.basename(os.path.dirname(f)), []).append(f)
    rnd = random.Random(per_class_seed)
    if len(by) >= 2 and os.path.abspath(os.path.dirname(files[0])) != os.path.abspath(root):
        names = sorted(by)[:max_classes]
        return names, [rnd.choice(by[k]) for k in names], {k: len(v) for k, v in by.items()}
    sel = files                      # flat folder: use ALL the images
    return [os.path.splitext(os.path.basename(f))[0] for f in sel], sel, {"(flat folder)": len(files)}


def box(title, head, rows):
    w = [max(len(h), *(len(r[i]) for r in rows)) + 2 for i, h in enumerate(head)]
    ln = "+" + "+".join("-" * x for x in w) + "+"
    out = [title, ln, "|" + "|".join(h.center(x) for h, x in zip(head, w)) + "|", ln]
    for r in rows:
        if r[0] == "Avg":
            out.append(ln)
        out.append("|" + "|".join(c.center(x) for c, x in zip(r, w)) + "|")
    return "\n".join(out + [ln])


def md(title, head, rows):
    return (f"**{title}**\n\n| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n" +
            "\n".join("| " + " | ".join(r) + " |" for r in rows) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="the 'training' folder")
    ap.add_argument("--size", type=int, default=512, help="multiple of 32 (paper: 512)")
    ap.add_argument("--seed", type=int, default=1, help="seed of the random image choice")
    ap.add_argument("--tol", type=float, default=5.0)
    ap.add_argument("--rule", choices=["figure", "literal"], default="figure")
    ap.add_argument("--groups", type=int, default=3)
    ap.add_argument("--chaos", action="store_true", help="chaotic SOM init + chaotic learning-rate schedule")
    ap.add_argument("--compare-chaos", action="store_true", help="add Table 5 (baseline vs chaotic SOM)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="results_training")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    names, paths, counts = pick_images(a.root, a.seed)
    print("images per class in the folder:", counts)
    for n, p in zip(names, paths):
        print(f"  {n:12s} -> {p}")
    images = [fm.load_image(p, a.size) for p in paths]
    from PIL import Image
    for n, im in zip(names, images):
        Image.fromarray(im.astype(np.uint8)).save(os.path.join(a.out, f"input_{n}.png"))
    som_kw = dict(init="chaotic", sched="chaotic") if a.chaos else dict(init="random", sched="linear")
    kw = dict(tol=a.tol, rule=a.rule, som_grid=(a.groups, a.groups), som_kw=som_kw)
    print("SOM:", som_kw)

    # ---------- Tables 1-3 (Algorithm II, four thresholds) ----------
    enc_cache = {}
    for i, im in enumerate(images):
        for t in fm.TAU_LIST:
            e, rec, p, tm, cr = fm.run(im, "alg2", tau=t, **kw)
            e.metrics = dict(psnr=p, time=tm, cr=cr)
            enc_cache[(i, t)] = e
            print(f"  {names[i]:12s} {fm.tau_name(t)}  PSNR={p:6.2f}  time={tm:7.2f}s  CR={cr:6.2f}  domains={len(e.dom)}", flush=True)
    hdr = ["Class"] + [f"τ=10^{int(round(math.log10(t)))}" for t in fm.TAU_LIST]
    blocks, mds = [], []
    for key, title in (("psnr", "Table 1  PSNR at Different Levels of Compression (dB)"),
                       ("time", "Table 2  Time Taken at Different Levels of Compression (sec)"),
                       ("cr", "Table 3  Compression Ratio at Different Thresholds")):
        rows = [[names[i]] + [f"{enc_cache[(i, t)].metrics[key]:.2f}" for t in fm.TAU_LIST] for i in range(len(images))]
        rows.append(["Avg"] + [f"{np.mean([enc_cache[(i, t)].metrics[key] for i in range(len(images))]):.2f}" for t in fm.TAU_LIST])
        blocks.append(box(title, hdr, rows)); mds.append(md(title, hdr, rows))

    # ---------- Table 4: three algorithms on the LAST class image ----------
    t4 = {}
    for algo in ("standard", "alg1", "alg2"):
        e, rec, p, tm, cr = fm.run(images[-1], algo, tau=1e-5, **kw)
        t4[algo] = (p, tm, cr, rec)
        print(f"  Table4 {algo:9s} PSNR={p:6.2f} time={tm:8.2f}s CR={cr:6.2f}", flush=True)
    h4 = ["Metric", "Standard Fractal Encoding", "Proposed Algorithm I (τ=1e-5)", "Proposed Algorithm II (τ=1e-5)"]
    r4 = [["PSNR (dB)"] + [f"{t4[k][0]:.2f}" for k in ("standard", "alg1", "alg2")],
          ["Compression Time (sec)"] + [f"{t4[k][1]:.2f}" for k in ("standard", "alg1", "alg2")],
          ["Compression Ratio"] + [f"{t4[k][2]:.2f}" for k in ("standard", "alg1", "alg2")]]
    ttl4 = f"Table 4  PSNR achieved for Different Algorithms  (image: {names[-1]})"
    blocks.append(box(ttl4, h4, r4)); mds.append(md(ttl4, h4, r4))

    # ---------- Table 5: baseline SOM vs chaotic SOM ----------
    if a.compare_chaos:
        cfgs = [("random / linear (baseline)", dict(init="random", sched="linear")),
                ("chaotic init", dict(init="chaotic", sched="linear")),
                ("chaotic init + chaotic LR", dict(init="chaotic", sched="chaotic"))]
        rows5 = []
        acc = {c[0]: [] for c in cfgs}
        for i, im in enumerate(images):
            row = [names[i]]
            for label, sk in cfgs:
                v = []
                for k in range(a.repeats):
                    kk = dict(kw, som_seed=k, som_kw=dict(sk, chaos_x0=0.2468 + 0.013 * k, chaos_y0=0.3691 + 0.009 * k))
                    e, rec, p, tm, cr = fm.run(im, "alg2", tau=1e-5, **kk)
                    v.append((p, tm, cr))
                v = np.array(v); acc[label].append(v.mean(0))
                row.append(f"{v[:,0].mean():.2f} / {v[:,1].mean():.2f}s / {v[:,2].mean():.2f}")
            rows5.append(row)
        rows5.append(["Avg"] + [" / ".join([f"{np.mean([x[0] for x in acc[c[0]]]):.2f}",
                                            f"{np.mean([x[1] for x in acc[c[0]]]):.2f}s",
                                            f"{np.mean([x[2] for x in acc[c[0]]]):.2f}"]) for c in cfgs])
        h5 = ["Class"] + [c[0] for c in cfgs]
        ttl5 = f"Table 5  Algorithm II, SOM variants (PSNR dB / time / CR, tau=1e-5, mean of {a.repeats} runs)"
        blocks.append(box(ttl5, h5, rows5)); mds.append(md(ttl5, h5, rows5))

    text = "\n\n".join(blocks)
    print("\n" + text)
    open(os.path.join(a.out, "tables.txt"), "w", encoding="utf-8").write(text + "\n")
    open(os.path.join(a.out, "tables.md"), "w", encoding="utf-8").write("\n".join(mds))
    fm.make_figures(a.out, images, enc_cache, fm.TAU_LIST, t4)
    print("\nSaved in:", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
