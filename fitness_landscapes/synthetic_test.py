"""Synthetic (fake) landscape tests for the eWHT vs WHT truncation gap.

Setting: ESM wildtype-marginal ps, R^2 evaluated on the ESM-bin sequences (as in r2_truncation.py).
Each fake landscape is ONE fitness vector y that both bases see; each basis only subtracts its own mean
(uniform for WHT, mu_p for eWHT).

Setups ("real" is always run once per dataset as the reference):
  yshuf         fitness values permuted across the entire landscape (all 2^L) -> dense
  coef_global   WHT coefficients (order >= 1) permuted across the whole
                spectrum, random signs                                       -> sparse, high order
  coef_band     WHT coefficients permuted within each interaction order,
                random signs                                                 -> sparse, exact low-order profile
Both coefficient setups shuffle in the WHT basis and invert with the WHT, so one landscape is shared.

Metrics per landscape: k95 = number of top-|c|^2 coefficients needed for R^2 >= 0.95 on the eval set, and the
number of order >= 3 terms among them. Pooled over datasets (per seed), as in the paper sentence:
  % fewer interactions = sum(k_WHT - k_eWHT) / sum(k_WHT), and likewise for the order >= 3 counts.

Run (from fitness_landscapes/):
  conda run -n fshap python synthetic_test.py
  conda run -n fshap python synthetic_test.py --n-seed 3 --jobs 32 --setups coef_band --datasets cov2 his3
Outputs (fitness_landscapes/synthetic/):
  results.csv                       per-job k95 counts (eWHT / WHT, all terms and order >= 3)
  summary.csv, sentences.txt        per setup: pooled % per seed, mean +/- sd over seeds, and the paper sentence
  config.json                       args, seed rule, git commit, sha256 of every input file, package versions
  inputs/{ds}_ps.npy                the ps used (snapshot)
  inputs/{ds}_eval_idx.npy          ESM-bin eval genotype indices used
  inputs/{ds}_y_real.npy            the real fitness vector (missing labels filled as in load_y_raw)
  landscapes/{ds}/{setup}_seed{k}.npy     the fake fitness vector y (shared by eWHT and WHT)
  curves/{ds}/{setup}_seed{k}_{ewht,wht}_r2.npy        R^2 vs number of coefficients (eval set)
  curves/{ds}/{setup}_seed{k}_{ewht,wht}_band.npy      energy fraction per interaction order 1..L
  curves/{ds}/{setup}_seed{k}_{ewht,wht}_topk.npy      top-2048 sorted c^2 / total energy
  plots/{setup}/{ds}_summary.pdf    same spectrum panel + bars: pooled % change of eWHT vs WHT in #coefficients
                                    (All) and #order>=3 terms for R^2 = 0.95 (= the sentence numbers), mean +/- sd
                                    over seeds; one symmetric y-range shared by real, yshuf and coef_band;
                                    coef_global gets its own
  plots/{setup}/{ds}.pdf            spectrum + eWHT/WHT R^2 panels (plot_utils.generate_and_render_r2_plots,
                                    same figure as plots/esm_wildtype_marginal/{ds}.pdf), seed --plot-seed
Finished jobs are reused (rows/{ds}/{setup}_seed{k}.json written last marks a job complete) when the dataset's
input files hash the same and CODE_VERSION is unchanged; --overwrite recomputes everything.
Re-running with the same arguments reproduces every file exactly (seed = SEED0 + 1000*seed + setup index);
--verify re-creates each saved landscape from its seed and checks it matches.
--plots-only redraws the PDFs from saved results; --summarize [--datasets ...] recomputes the sentences from results.csv without rerunning (subset -> summary_<ds>.csv).
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # pfwht uses torch; many GPU workers run out of memory

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import landscapes as ls
from landscapes import interaction_order_array, r2_curve
from pbias import ipfwht, pfwht

FL_DIR = Path(__file__).resolve().parent
OUT_DIR = FL_DIR / "synthetic"
SKIP = ["cr9114_h3_mean_dev"]
# the list index is part of each job's seed: only ever append new setups
SETUPS = ["yshuf", "coef_global", "coef_band"]
LABELS = {
    "real": "Real landscape",
    "yshuf": "y-shuffle (dense)",
    "coef_global": "Global coefficient shuffle (sparse, high order)",
    "coef_band": "Within-band shuffle (sparse, exact order)",
}
SEED0 = 20261103
# Bump when make_landscape / r2_curve change: cached jobs from another CODE_VERSION are recomputed.
CODE_VERSION = 1
# Per-setup overrides. yshuf v2: permute the entire landscape (v1 permuted only the eval genotypes).
SETUP_VERSION = {"yshuf": 2}


def code_version(setup: str) -> int:
    return SETUP_VERSION.get(setup, CODE_VERSION)


# ---------------------------------------------------------------- transforms / R^2 curve
def load_y_raw(name: str):
    """(y, L, N, ps, ESM-bin eval indices, label_mask) for a real landscape."""
    y, L, ps, mask = ls.load_landscape(name)
    return y, L, 2 ** L, ps, ls.bin_indices(name, L, mask, "esm"), mask


def check_basis(name: str = "cov2") -> None:
    _, L, N, ps, ev, _ = load_y_raw(name)
    c = np.random.default_rng(0).standard_normal(N)
    for p in (ps, np.full(L, 0.5)):
        assert np.allclose(ls.basis_block(np.arange(N), ev, p, L) @ c, ipfwht(c, p)[ev]), "basis mismatch"


# ---------------------------------------------------------------- fake landscapes
def make_landscape(setup: str, y: np.ndarray, L: int, ev: np.ndarray, rng) -> np.ndarray:
    if setup == "real":
        return y
    if setup == "yshuf":
        return y[rng.permutation(len(y))]  # shuffle the whole landscape, not just the eval (ESM-bin) genotypes
    h = np.full(L, 0.5)   # coefficient shuffles are built in the WHT basis (uniform-centred y)
    mean = y.mean()
    o = interaction_order_array(L)
    c = pfwht(y - mean, ps=h, norm="expectation")
    if setup == "coef_global":
        groups = [np.where(o >= 1)[0]]
    elif setup == "coef_band":
        groups = [np.where(o == k)[0] for k in range(1, L + 1)]
    else:
        raise ValueError(setup)
    c2 = c.copy()
    for m in groups:
        if m.size:
            c2[m] = c[rng.permutation(m)] * rng.choice([-1.0, 1.0], len(m))
    return ipfwht(c2, h) + mean


def band_energy(c: np.ndarray, L: int) -> np.ndarray:
    o = interaction_order_array(L)
    e = np.array([(c[o == k] ** 2).sum() for k in range(1, L + 1)])
    return e / e.sum()


def job_rng(setup: str, seed: int):
    return np.random.default_rng(SEED0 + 1000 * seed + (SETUPS.index(setup) if setup in SETUPS else 99))


def job_files(out_dir: Path, name: str, setup: str, seed: int) -> dict:
    tag = f"{setup}_seed{seed}"
    f = {"row": out_dir / "rows" / name / f"{tag}.json",
         "landscape": out_dir / "landscapes" / name / f"{tag}.npy"}
    for b in ("ewht", "wht"):
        for kind in ("r2", "band", "topk"):
            f[f"{b}_{kind}"] = out_dir / "curves" / name / f"{tag}_{b}_{kind}.npy"
    return f


def cached_row(out_dir: Path, name: str, setup: str, seed: int, inputs_sha: str) -> dict | None:
    """Saved row for this job if it is complete and was made from the same inputs and code; else None."""
    f = job_files(out_dir, name, setup, seed)
    if not f["row"].exists() or not all(v.exists() for v in f.values()):
        return None
    try:
        row = json.loads(f["row"].read_text())
        if row.get("inputs_sha") != inputs_sha or row.get("code_version") != code_version(setup):
            return None
        x_max = int(row["x_max"])
        for b in ("ewht", "wht"):
            if len(np.load(f[f"{b}_r2"], mmap_mode="r")) < x_max:
                return None
    except Exception:
        return None
    return row


def run_job(job):
    name, setup, seed, out_dir, inputs_sha, overwrite = job
    out_dir = Path(out_dir)
    if not overwrite:
        row = cached_row(out_dir, name, setup, seed, inputs_sha)
        if row is not None:
            print(f"  {name:>18s} {setup:>12s} seed {seed}: cached", flush=True)
            return row
    y, L, N, ps, ev, _ = load_y_raw(name)
    yy = make_landscape(setup, y, L, ev, job_rng(setup, seed))
    ke, ne, cv_e, c_e = r2_curve(yy, ps, L, ev)
    kw, nw, cv_w, c_w = r2_curve(yy, np.full(L, 0.5), L, ev)
    x_max = min(N, ls.plotted_k_span(ke, kw))  # plotted span, same rule as the reference plots
    if len(cv_e) < x_max:
        cv_e = r2_curve(yy, ps, L, ev, kmin=x_max)[2]
    if len(cv_w) < x_max:
        cv_w = r2_curve(yy, np.full(L, 0.5), L, ev, kmin=x_max)[2]
    tag = f"{setup}_seed{seed}"
    (out_dir / "landscapes" / name).mkdir(parents=True, exist_ok=True)
    (out_dir / "curves" / name).mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "landscapes" / name / f"{tag}.npy", yy)
    for b, cv, c in (("ewht", cv_e, c_e), ("wht", cv_w, c_w)):
        np.save(out_dir / "curves" / name / f"{tag}_{b}_r2.npy", cv)
        np.save(out_dir / "curves" / name / f"{tag}_{b}_band.npy", band_energy(c, L))
        np.save(out_dir / "curves" / name / f"{tag}_{b}_topk.npy", np.sort(c ** 2)[::-1][:2048] / (c ** 2).sum())
    row = dict(dataset=name, setup=setup, seed=seed, L=L, eval_n=len(ev),
               k95_ewht=ke, k95_wht=kw, ge3_ewht=ne, ge3_wht=nw, x_max=x_max,
               inputs_sha=inputs_sha, code_version=code_version(setup))
    rf = job_files(out_dir, name, setup, seed)["row"]
    rf.parent.mkdir(parents=True, exist_ok=True)
    rf.write_text(json.dumps({k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}))  # written last
    print(f"  {name:>18s} {setup:>12s} seed {seed}: k95 eWHT {ke:6d}  WHT {kw:6d}", flush=True)
    return row


# ---------------------------------------------------------------- plots
def bar_stats(df: pd.DataFrame, setup: str) -> dict:
    """Pooled % change of eWHT vs WHT, exactly the numbers in the paper sentence (sentences.txt):
    per seed 100*(sum k_eWHT - sum k_WHT)/sum k_WHT over all datasets (negative = eWHT needs fewer),
    then mean and sd over seeds. Same for the order >= 3 counts."""
    r = pooled_summary(df[df.setup == setup]).iloc[0]
    return dict(mean=(-float(r.pct_fewer), -float(r.pct_fewer_ge3)),
                sd=(float(r.pct_fewer_sd), float(r.pct_fewer_ge3_sd)),
                n_dataset=int(r.n_dataset), n_seed=int(r.n_seed))


def shared_ylim(stats: dict) -> float:
    """One symmetric limit for every method: largest |mean| + sd, rounded up to a 'nice' value."""
    hi = max(abs(m) + sd for st in stats.values() for m, sd in zip(st["mean"], st["sd"]))
    hi *= 1.08
    step = 10 ** np.floor(np.log10(hi))
    for mult in (1, 2, 2.5, 5, 10):
        if mult * step >= hi:
            return float(mult * step)
    return float(hi)


BAR_PANEL_WIDTH_IN = 1.25   # width of the bar axes in the _summary figure (the R^2 panels together were ~3.3 in)


def draw_summary_panel(fig, stats: dict, pdf: Path) -> None:
    """Replace the two R^2 panels of a generate_and_render_r2_plots figure by one narrower grouped bar chart.

    The figure was just saved as {dataset}.pdf, so its constrained layout (computed with the PDF renderer) is in
    place. It is frozen without redrawing, every remaining axes is re-placed at the same position in inches, and
    the figure is made narrower: the spectrum panel keeps exactly its size and position, only the right part of
    the figure changes.
    """
    from plot_utils import COLOR_PWHT, COLOR_WHT, LINE_WIDTH

    fig.set_layout_engine("none")
    W, H = fig.get_size_inches()
    r2_axes = [a for a in fig.axes if a.get_xlabel().startswith("Number of largest")]
    left, right = r2_axes[0].get_position(), r2_axes[-1].get_position()
    renderer = fig.canvas.get_renderer()
    deco_left_in = r2_axes[0].get_tightbbox(renderer).x0 / fig.dpi   # left edge of the R^2 panel's labels
    right_margin_in = (1 - right.x1) * W
    for a in r2_axes:
        a.remove()
    keep = {a: a.get_position() for a in fig.axes}
    W_new = left.x0 * W + BAR_PANEL_WIDTH_IN + right_margin_in
    fig.set_size_inches(W_new, H)
    for a, pos in keep.items():                       # same position in inches -> spectrum unchanged
        a.set_position([pos.x0 * W / W_new, pos.y0, pos.width * W / W_new, pos.height])
    ax = fig.add_axes([left.x0 * W / W_new, left.y0, BAR_PANEL_WIDTH_IN / W_new, left.height])

    m, sd, lim = stats["mean"], stats["sd"], stats["ylim"]
    x = np.arange(2) * 1.0
    err_kw = dict(elinewidth=LINE_WIDTH, capthick=LINE_WIDTH, capsize=1.5, ecolor="black")
    colors = [COLOR_PWHT if v < 0 else COLOR_WHT for v in m]   # lower is better: eWHT fewer = blue, more = red
    ax.bar(x, m, width=0.55, color=colors, edgecolor="none", zorder=2,
           yerr=sd if stats["n_seed"] > 1 else None, error_kw=err_kw)
    ax.axhline(0, color="black", linewidth=LINE_WIDTH, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(["All", "3rd+ order"])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, x[-1] + 0.6)
    ax.set_ylim(-lim, lim)                       # symmetric; shared by all setups except coef_global
    ax.yaxis.set_major_locator(plt_ticker().MaxNLocator(nbins=4, symmetric=True, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_formatter(plt_ticker().FuncFormatter(lambda v, _: f"{v:g}".replace("-", "\u2212")))
    ax.set_ylabel("% change relative\nto WHT $\\downarrow$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # wide tick labels (e.g. 4-digit counts) would push the y label into the spectrum panel: move the bar axes
    # right until its labels start where the R^2 panel's labels started, widening the figure by the same amount
    shift_in = deco_left_in - ax.get_tightbbox(renderer).x0 / fig.dpi
    if shift_in > 0:
        W_old = W_new
        W_new = W_old + shift_in
        fig.set_size_inches(W_new, H)
        for a in fig.axes:
            pos = a.get_position()
            x0 = pos.x0 * W_old + (shift_in if a is ax else 0.0)
            a.set_position([x0 / W_new, pos.y0, pos.width * W_old / W_new, pos.height])
    fig.savefig(pdf, dpi=300)


def plt_ticker():
    import matplotlib.ticker as mticker
    return mticker


def plot_job(job):
    """synthetic/plots/{setup}/{dataset}.pdf (same figure as plots/esm_wildtype_marginal/{dataset}.pdf, via
    plot_utils.generate_and_render_r2_plots) and {dataset}_summary.pdf (same spectrum panel; the R^2 panels
    replaced by a narrower eWHT vs WHT bar chart of counts averaged over all datasets), for seed `seed`."""
    name, setup, seed, out_dir, stats = job
    out_dir = Path(out_dir)
    import matplotlib.pyplot as plt
    import plot_utils
    _, L, N, ps, ev, _ = load_y_raw(name)
    tag = f"{setup}_seed{seed}"
    y = np.load(out_dir / "landscapes" / name / f"{tag}.npy")
    r2_e = np.load(out_dir / "curves" / name / f"{tag}_ewht_r2.npy")
    r2_w = np.load(out_dir / "curves" / name / f"{tag}_wht_r2.npy")
    res = pd.read_csv(out_dir / "results.csv")
    r = res[(res.dataset == name) & (res.setup == setup) & (res.seed == seed)].iloc[0]
    k_e, k_w, x_max = int(r.k95_ewht), int(r.k95_wht), int(r.x_max)
    y_wht, y_pwht = ls.center(y, np.full(L, 0.5)), ls.center(y, ps)
    c_w = pfwht(y_wht.copy(), ps=np.full(L, 0.5), norm="expectation")
    c_e = pfwht(y_pwht.copy(), ps=ps, norm="expectation")
    o = interaction_order_array(L)
    pdf = out_dir / "plots" / setup / f"{name}.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)

    captured = []
    real_close = plt.close
    plt.close = lambda fig=None: captured.append(fig)      # keep the figure open after it is saved
    try:
        plot_utils.generate_and_render_r2_plots(
            name, L, r2_e[None, :x_max], r2_w[None, :x_max], c_e, c_w,
            o[np.argsort(c_e ** 2)[::-1]], o[np.argsort(c_w ** 2)[::-1]],
            k_e, k_w, x_max, pdf, y_wht=y_wht, y_pwht=y_pwht, ps=ps)
    finally:
        plt.close = real_close
    fig = captured[0]
    spdf = pdf.with_name(f"{name}_summary.pdf")
    draw_summary_panel(fig, stats, spdf)
    plt.close(fig)
    print(f"  wrote {pdf.relative_to(out_dir)} and {spdf.name}", flush=True)
    return str(pdf)


def make_plots(out_dir: Path, jobs: int, seed: int = 0, overwrite: bool = False) -> None:
    """Draw every PDF pair that is missing or older than the data it is drawn from
    ({dataset}.pdf: its landscape + curves; {dataset}_summary.pdf: also results.csv)."""
    res = pd.read_csv(out_dir / "results.csv")
    res_time = (out_dir / "results.csv").stat().st_mtime
    stats = {s: bar_stats(res, s) for s in res.setup.unique()}
    # one symmetric y-range shared by real, yshuf and coef_band; the global shuffle gets its own (~10x larger)
    own = {"coef_global"}
    lim = shared_ylim({k: v for k, v in stats.items() if k not in own})
    for k, st in stats.items():
        st["ylim"] = shared_ylim({k: st}) if k in own else lim
    todo, n_cached = [], 0
    for r in res[res.seed == seed][["dataset", "setup"]].drop_duplicates().itertuples():
        pdf = out_dir / "plots" / r.setup / f"{r.dataset}.pdf"
        spdf = pdf.with_name(f"{r.dataset}_summary.pdf")
        src = job_files(out_dir, r.dataset, r.setup, seed)
        newest = max(src[k].stat().st_mtime for k in ("landscape", "ewht_r2", "wht_r2"))
        fresh = (pdf.exists() and pdf.stat().st_mtime >= newest and spdf.exists()
                 and spdf.stat().st_mtime >= max(newest, res_time))
        if fresh and not overwrite:
            n_cached += 1
            continue
        todo.append((r.dataset, r.setup, seed, str(out_dir), stats[r.setup]))
    if todo:
        with Pool(min(jobs, len(todo))) as pool:
            pool.map(plot_job, todo, chunksize=1)
    print(f"Plots under {out_dir / 'plots'}: {len(todo)} dataset/method pairs drawn, {n_cached} up to date")


# ---------------------------------------------------------------- reproducibility
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def combined_sha(sources: dict) -> str:
    """One hash over a dataset's source files (processed csv, ps npy, esm_data csv)."""
    return hashlib.sha256("".join(sources[k]["sha256"] for k in sorted(sources)).encode()).hexdigest()


def save_inputs(names, out_dir: Path) -> dict:
    """Snapshot ps / eval indices / real y per dataset and hash the source files they came from."""
    (out_dir / "inputs").mkdir(parents=True, exist_ok=True)
    info = {}
    for n in names:
        y, L, N, ps, ev, mask = load_y_raw(n)
        np.save(out_dir / "inputs" / f"{n}_ps.npy", ps)
        np.save(out_dir / "inputs" / f"{n}_eval_idx.npy", ev)
        np.save(out_dir / "inputs" / f"{n}_y_real.npy", y)
        src = {"processed_csv": ls.PROCESSED_DIR / f"{n}.csv",
               "ps_npy": ls.PS_DIR / f"{n}.npy",
               "esm_data_csv": ls.ESM_DATA_DIR / f"{n}.csv"}
        info[n] = dict(L=L, eval_n=int(len(ev)), n_missing_labels=int((~mask).sum()),
                       ps=[float(p) for p in ps],
                       sources={k: dict(path=str(v.relative_to(ls.ROOT)), sha256=sha256(v)) for k, v in src.items()})
        info[n]["inputs_sha"] = combined_sha(info[n]["sources"])
        if np.allclose(ps, 0.5):
            print(f"WARNING: {n}: all ps == 0.5, so eWHT is identical to WHT on this dataset")
    return info


def git_commit() -> str:
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=FL_DIR, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", Path(__file__).name], cwd=FL_DIR, text=True).strip()
        return rev + (" (synthetic_test.py modified/untracked)" if dirty else "")
    except Exception:
        return "unknown"


def verify(out_dir: Path) -> None:
    """Re-create every saved landscape from (dataset, setup, seed) and compare to the saved .npy."""
    df = pd.read_csv(out_dir / "results.csv")
    bad = 0
    for r in df.itertuples():
        y, L, N, ps, ev, _ = load_y_raw(r.dataset)
        yy = make_landscape(r.setup, y, L, ev, job_rng(r.setup, int(r.seed)))
        saved = np.load(out_dir / "landscapes" / r.dataset / f"{r.setup}_seed{r.seed}.npy")
        ok = np.array_equal(yy, saved)
        bad += not ok
        print(f"  {r.dataset:>18s} {r.setup:>12s} seed {r.seed}: {'ok' if ok else 'MISMATCH'}")
    print("all landscapes reproduced exactly" if bad == 0 else f"{bad} landscapes did not match")


# ---------------------------------------------------------------- summary
def pooled_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per setup: pooled % fewer (all and order>=3) per seed, then mean/sd over seeds; plus win counts."""
    out = []
    for setup, g in df.groupby("setup", sort=False):
        per_seed = g.groupby("seed").agg(k_w=("k95_wht", "sum"), k_e=("k95_ewht", "sum"),
                                         g_w=("ge3_wht", "sum"), g_e=("ge3_ewht", "sum"))
        pct = 100 * (per_seed.k_w - per_seed.k_e) / per_seed.k_w
        pct3 = 100 * (per_seed.g_w - per_seed.g_e) / per_seed.g_w.where(per_seed.g_w > 0)
        out.append(dict(setup=setup, n_seed=len(per_seed), n_dataset=g.dataset.nunique(),
                        pct_fewer=pct.mean(), pct_fewer_sd=pct.std(ddof=0),
                        pct_fewer_ge3=pct3.mean(), pct_fewer_ge3_sd=pct3.std(ddof=0),
                        ewht_better=int((g.k95_ewht < g.k95_wht).sum()),
                        tie=int((g.k95_ewht == g.k95_wht).sum()),
                        wht_better=int((g.k95_ewht > g.k95_wht).sum())))
    order = ["real"] + SETUPS
    s = pd.DataFrame(out)
    return s.set_index("setup").loc[[x for x in order if x in set(s.setup)]].reset_index()


NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
             9: "nine", 10: "ten"}


def paper_sentence(r) -> str:
    """Paper sentence for one setup; mean +/- sd over seeds (no +/- when there is one seed)."""
    def pm(mean, sd):
        return f"{abs(mean):.1f}%" + (f" ± {sd:.1f}%" if r.n_seed > 1 else "")
    n = NUM_WORDS.get(r.n_dataset, str(r.n_dataset))
    k_word = "fewer" if r.pct_fewer >= 0 else "more"
    g_word = "reduced" if r.pct_fewer_ge3 >= 0 else "increased"
    return (f"Across the {n} landscapes, eWHT reached an R² of 0.95 using {pm(r.pct_fewer, r.pct_fewer_sd)} "
            f"{k_word} epistatic interactions and {g_word} the number of third-order and higher interactions "
            f"needed to achieve the same accuracy by {pm(r.pct_fewer_ge3, r.pct_fewer_ge3_sd)}.")


def print_summary(s: pd.DataFrame) -> None:
    print("\nPooled over datasets per seed, then mean ± sd over seeds; win counts are over dataset x seed:")
    for r in s.itertuples():
        print(f"\n[{LABELS.get(r.setup, r.setup)}]  ({r.n_dataset} landscapes, {r.n_seed} seed(s))")
        print(f"  {paper_sentence(r)}")
        print(f"  eWHT better / tie / WHT better: {r.ewht_better} / {r.tie} / {r.wht_better}")


def summarize(out_dir: Path, datasets=None) -> None:
    """Recompute summary.csv + sentences.txt from results.csv (optionally on a subset of datasets)."""
    df = pd.read_csv(out_dir / "results.csv")
    if datasets:
        df = df[df.dataset.isin(datasets)]
    s = pooled_summary(df)
    s["sentence"] = [paper_sentence(r) for r in s.itertuples()]
    tag = "" if not datasets else "_" + "_".join(sorted(datasets))
    s.to_csv(out_dir / f"summary{tag}.csv", index=False)
    (out_dir / f"sentences{tag}.txt").write_text(
        "\n\n".join(f"[{LABELS.get(r.setup, r.setup)}]\n{r.sentence}" for r in s.itertuples()) + "\n")
    print(f"datasets: {sorted(df.dataset.unique())}")
    print_summary(s)
    print(f"\nWrote {out_dir}/summary{tag}.csv and sentences{tag}.txt")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-seed", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--datasets", nargs="+", default=None, help="default: all datasets except --skip")
    ap.add_argument("--skip", nargs="+", default=SKIP)
    ap.add_argument("--setups", nargs="+", default=SETUPS, choices=SETUPS)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--overwrite", action="store_true",
                    help="recompute every job and redraw every PDF (default: reuse finished jobs whose inputs "
                         "and CODE_VERSION are unchanged, and PDFs newer than their data)")
    ap.add_argument("--verify", action="store_true", help="only re-create saved landscapes and check they match")
    ap.add_argument("--plots-only", action="store_true",
                    help="only redraw synthetic/plots/{method}/{dataset}.pdf from saved results")
    ap.add_argument("--plot-seed", type=int, default=0, help="which seed's landscape to plot (default 0)")
    ap.add_argument("--summarize", action="store_true",
                    help="only recompute the paper sentences from results.csv (--datasets picks a subset)")
    args = ap.parse_args()
    out_dir = args.out_dir.resolve()
    if args.verify:
        verify(out_dir)
        return
    if args.summarize:
        summarize(out_dir, args.datasets)
        return
    if args.plots_only:
        make_plots(out_dir, args.jobs, args.plot_seed, args.overwrite)
        return

    names = args.datasets or ls.dataset_names(args.skip)
    print(f"datasets: {names}\nsetups: {args.setups}  seeds: {args.n_seed}\noutput: {out_dir}")
    check_basis()
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = save_inputs(names, out_dir)
    keys = [(n, "real", 0) for n in names] + \
           [(n, s, k) for n in names for s in args.setups for k in range(args.n_seed)]
    jobs = [(n, s, k, str(out_dir), inputs[n]["inputs_sha"], args.overwrite) for n, s, k in keys]
    # cheap jobs first would leave the pool idle at the end; start the slowest (largest L) first
    jobs.sort(key=lambda j: -inputs[j[0]]["L"])
    with Pool(min(args.jobs, len(jobs))) as pool:
        rows = pool.map(run_job, jobs, chunksize=1)

    cols = ["dataset", "setup", "seed", "L", "eval_n", "k95_ewht", "k95_wht", "ge3_ewht", "ge3_wht", "x_max"]
    df = pd.DataFrame(rows)[cols].sort_values(["dataset", "setup", "seed"])
    new_csv = df.to_csv(index=False)
    res_path = out_dir / "results.csv"
    if not res_path.exists() or res_path.read_text() != new_csv:   # untouched if unchanged -> PDFs stay cached
        res_path.write_text(new_csv)
    config = dict(
        command=" ".join([Path(sys.executable).name] + sys.argv),
        time=datetime.now().isoformat(timespec="seconds"),
        git_commit=git_commit(),
        args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        datasets=names, setups=args.setups, seed0=SEED0,
        seed_rule="np.random.default_rng(SEED0 + 1000*seed + SETUPS.index(setup)); real uses +99 (unused)",
        setups_order=SETUPS,
        versions=dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__),
        inputs=inputs,
    )
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    pd.set_option("display.width", 200)
    print("\n" + df.to_string(index=False))
    summarize(out_dir)
    make_plots(out_dir, args.jobs, args.plot_seed, args.overwrite)
    print(f"Wrote results.csv, config.json, inputs/, landscapes/, curves/, plots/ under {out_dir}")


if __name__ == "__main__":
    main()
