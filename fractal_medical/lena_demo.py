#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run the fractal coders on ONE image (e.g. Lena) and show the reconstructed image.

    python lena_demo.py --image lena.png                 # 512x512, tau = 1e-5
    python lena_demo.py --image lena.png --size 256      # faster

Outputs (folder results_lena/):
    lena_tables.txt / .md      Table L1 (3 algorithms) and Table L2 (tau sweep, Algorithm II)
    lena_algorithms.png        original | standard | Algorithm I | Algorithm II  (+ error maps)
    lena_tau_sweep.png         original | reconstructions for each tau
"""
import argparse, os
import numpy as np
from scipy.ndimage import uniform_filter
import fractal_medical as fm


def ssim(a, b, win=7):
    """Mean SSIM (uniform window), 8-bit images."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = uniform_filter(a, win), uniform_filter(b, win)
    saa = uniform_filter(a * a, win) - mu_a ** 2
    sbb = uniform_filter(b * b, win) - mu_b ** 2
    sab = uniform_filter(a * b, win) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))
    return float(s.mean())


def table(title, head, rows):
    w = [max(len(h), *(len(r[i]) for r in rows)) + 2 for i, h in enumerate(head)]
    ln = "+" + "+".join("-" * x for x in w) + "+"
    out = [title, ln, "|" + "|".join(h.center(x) for h, x in zip(head, w)) + "|", ln]
    out += ["|" + "|".join(c.center(x) for c, x in zip(r, w)) + "|" for r in rows] + [ln]
    return "\n".join(out)


def md(title, head, rows):
    return (f"**{title}**\n\n| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n" +
            "\n".join("| " + " | ".join(r) + " |" for r in rows) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="path of the image (colour images are converted to grey)")
    ap.add_argument("--size", type=int, default=512, help="multiple of 32")
    ap.add_argument("--tau", type=float, default=1e-5)
    ap.add_argument("--tol", type=float, default=5.0)
    ap.add_argument("--groups", type=int, default=3)
    ap.add_argument("--som-init", choices=["random", "chaotic", "logistic"], default="random")
    ap.add_argument("--som-sched", choices=["linear", "chaotic"], default="linear")
    ap.add_argument("--out", default="results_lena")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    img = fm.load_image(a.image, a.size)
    kw = dict(tol=a.tol, som_grid=(a.groups, a.groups),
              som_kw=dict(init=a.som_init, sched=a.som_sched))

    # ---- Table L1: three algorithms ----
    recs, rows1 = {}, []
    for algo, name in (("standard", "Standard Fractal Encoding"),
                       ("alg1", "Proposed Algorithm I"), ("alg2", "Proposed Algorithm II")):
        e, rec, p, tm, cr = fm.run(img, algo, tau=a.tau, **kw)
        recs[algo] = (rec, p)
        rows1.append([name, f"{p:.2f}", f"{ssim(img, rec):.4f}", f"{tm:.2f}", f"{cr:.2f}"])
        print(f"  {name:28s} PSNR={p:6.2f}  time={tm:7.2f}s  CR={cr:6.2f}", flush=True)
    h1 = ["Algorithm", "PSNR (dB)", "SSIM", "Time (s)", "Compression ratio"]
    t1 = f"Table L1  Reconstruction quality of the image (tau={a.tau:g})"

    # ---- Table L2: tau sweep with Algorithm II ----
    sweep, rows2 = {}, []
    for t in fm.TAU_LIST:
        e, rec, p, tm, cr = fm.run(img, "alg2", tau=t, **kw)
        sweep[t] = (rec, p)
        rows2.append([f"1e{int(round(np.log10(t)))}", f"{p:.2f}", f"{ssim(img, rec):.4f}",
                      f"{tm:.2f}", f"{cr:.2f}", str(len(e.dom))])
        print(f"  alg2 tau={t:g}  PSNR={p:6.2f}  time={tm:6.2f}s  CR={cr:6.2f}", flush=True)
    h2 = ["tau", "PSNR (dB)", "SSIM", "Time (s)", "Compression ratio", "Lossless domain blocks"]
    t2 = "Table L2  Algorithm II at different thresholds"

    text = table(t1, h1, rows1) + "\n\n" + table(t2, h2, rows2)
    print("\n" + text)
    open(os.path.join(a.out, "lena_tables.txt"), "w", encoding="utf-8").write(text + "\n")
    open(os.path.join(a.out, "lena_tables.md"), "w", encoding="utf-8").write(
        md(t1, h1, rows1) + "\n" + md(t2, h2, rows2))

    # ---- figures ----
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 4, figsize=(18, 9))
    ax[0, 0].imshow(img, cmap="gray", vmin=0, vmax=255); ax[0, 0].set_title("Original")
    ax[1, 0].axis("off")
    for c, (k, nm) in enumerate((("standard", "Standard fractal"), ("alg1", "Proposed Algorithm I"),
                                 ("alg2", "Proposed Algorithm II")), 1):
        rec, p = recs[k]
        ax[0, c].imshow(rec, cmap="gray", vmin=0, vmax=255)
        ax[0, c].set_title(f"{nm}\nPSNR = {p:.2f} dB  SSIM = {ssim(img, rec):.3f}")
        ax[1, c].imshow(np.clip(np.abs(img - rec) * 10, 0, 255), cmap="gray", vmin=0, vmax=255)
        ax[1, c].set_title("|error| x10")
    for x in ax.ravel(): x.set_xticks([]); x.set_yticks([])
    fig.savefig(os.path.join(a.out, "lena_algorithms.png"), dpi=110, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(1, 5, figsize=(22, 4.8))
    ax[0].imshow(img, cmap="gray", vmin=0, vmax=255); ax[0].set_title("Original")
    for x, t in zip(ax[1:], fm.TAU_LIST):
        rec, p = sweep[t]
        x.imshow(rec, cmap="gray", vmin=0, vmax=255)
        x.set_title(f"tau = 1e{int(round(np.log10(t)))}   PSNR = {p:.2f} dB")
    for x in ax: x.axis("off")
    fig.savefig(os.path.join(a.out, "lena_tau_sweep.png"), dpi=110, bbox_inches="tight"); plt.close(fig)
    print("\nSaved in:", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
