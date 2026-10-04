# Fractal coding with SOM-based domain/range pool partitioning (Bhavani & Thanushkodi, 2013)

    pip install numpy scipy pillow matplotlib
    python fractal_medical.py                       # 4 synthetic MRI-like phantoms, 256x256
    python fractal_medical.py --images a.png b.png c.png d.png --size 512   # real MRI (paper setting)

Outputs in `results/`: `tables.txt` / `tables.md` (Tables 1-4), figures 1, 2, 3, 5, 6, 7, 8.

Notes
* No MRI data set ships with the paper; without `--images` synthetic phantoms are used,
  so absolute numbers differ from the paper. Use real MRI images for a true comparison.
* `--rule figure` (default) maps tau so that larger tau keeps more domain blocks (as in the
  paper's Fig. 2); `--rule literal` uses the inequality exactly as printed.
* Tables 1-3: Algorithm II. Table 4: standard / Algorithm I / Algorithm II at tau=1e-5 on image 4.

## Chaotic SOM variants (ablation)
    python fractal_medical.py --som-init chaotic --som-sched chaotic
    python ablation.py --dir /content/mri --n 20 --size 512 --repeats 3
`--som-init` random|chaotic|logistic, `--som-sched` linear|chaotic. The SOM has no activation
function (competitive learning + Gaussian neighbourhood). `ablation.py` writes Table A to `results_ablation/`.

## Any-size data set (single file for Colab / Drive)
    python fractal_alg2_dataset.py --source archive.zip --out fractal_results            # whole Training/ part
    python fractal_alg2_dataset.py --source folder --limit 200 --taus 1e-5 --workers 4
`fractal_alg2_dataset.py` is self-contained (numpy + pillow). Any number of images (streamed, resumable via
`results.csv`), any image size (`--size 0 --tile 512` = native size, tiled), zip or folder. Output: `results.csv`,
`summary.txt/.md` (Tables 1-4 averaged over all images and per class), `recon/`.
