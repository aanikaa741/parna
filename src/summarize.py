"""Turn results/*.csv into tables and plots for the week 1 write-up."""
import csv, pathlib, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = pathlib.Path(__file__).resolve().parent.parent / "results"
rows = list(csv.DictReader(open(OUT / "sessions.csv")))
segs = list(csv.DictReader(open(OUT / "segments.csv")))
f = lambda v: float(v) if v not in ("", "nan") else np.nan
PAIRS, SNRS, RTS, METHODS = ["easy", "medium", "hard"], ["5", "15", "30"], ["0.2", "0.6", "1.2"], ["blind", "calibrated"]
lines = []

def mean(sel, key):
    v = [f(r[key]) for r in sel]; v = [x for x in v if not np.isnan(x)]
    return np.mean(v) if v else np.nan

lines.append("## Client hit detection F1 by instrument pair and noise (mean over echo levels and seeds)\n")
lines.append("| method | pair | SNR 5 dB | SNR 15 dB | SNR 30 dB |\n|---|---|---|---|---|")
for m in METHODS:
    for p in PAIRS:
        cells = [f"{mean([r for r in rows if r['method']==m and r['pair']==p and r['snr']==s], 'f1_client'):.2f}" for s in SNRS]
        lines.append(f"| {m} | {p} | " + " | ".join(cells) + " |")

lines.append("\n## Therapist and client F1 by echo (RT60), all pairs and noise levels\n")
lines.append("| method | who | 0.2 s | 0.6 s | 1.2 s |\n|---|---|---|---|---|")
for m in METHODS:
    for who in ("therapist", "client"):
        cells = [f"{mean([r for r in rows if r['method']==m and r['rt60']==t], 'f1_'+who):.2f}" for t in RTS]
        lines.append(f"| {m} | {who} | " + " | ".join(cells) + " |")

lines.append("\n## Segment-level timing: does the estimate track the truth?\n")
lines.append("| method | pair | segments | median abs error, mean async (ms) | correlation true vs est async | in-sync rate abs error |\n|---|---|---|---|---|---|")
for m in METHODS:
    for p in PAIRS:
        sel = [r for r in segs if r["method"] == m and r["pair"] == p]
        t = np.array([f(r["true_async"]) for r in sel]); e = np.array([f(r["est_async"]) for r in sel])
        ti = np.array([f(r["true_insync"]) for r in sel]); ei = np.array([f(r["est_insync"]) for r in sel])
        ok = ~np.isnan(t) & ~np.isnan(e); oki = ~np.isnan(ti) & ~np.isnan(ei)
        corr = np.corrcoef(t[ok], e[ok])[0, 1] if ok.sum() > 2 else np.nan
        lines.append(f"| {m} | {p} | {len(sel)} ({ok.sum()} measurable) | {np.median(np.abs(t[ok]-e[ok])):.0f} | {corr:.2f} | {np.median(np.abs(ti[oki]-ei[oki])):.2f} |")

lines.append("\n## Calibrated method: segment timing error by behaviour\n")
lines.append("| behaviour | segments | median abs error (ms) |\n|---|---|---|")
for k in ["sync", "late", "drift", "catchup", "half"]:
    sel = [r for r in segs if r["method"] == "calibrated" and r["kind"] == k]
    t = np.array([f(r["true_async"]) for r in sel]); e = np.array([f(r["est_async"]) for r in sel]); ok = ~np.isnan(t) & ~np.isnan(e)
    if ok.any():
        lines.append(f"| {k} | {len(sel)} | {np.median(np.abs(t[ok]-e[ok])):.0f} |")

(OUT / "tables.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# plots
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
for ax, p in zip(axes, PAIRS):
    for m, ls in zip(METHODS, ("--", "-")):
        ax.plot([5, 15, 30], [mean([r for r in rows if r['method']==m and r['pair']==p and r['snr']==s], 'f1_client') for s in SNRS], ls, marker="o", label=m)
    ax.set_title(f"{p} pair"); ax.set_xlabel("signal-to-noise (dB)"); ax.set_ylim(0, 1.02); ax.grid(alpha=.3)
axes[0].set_ylabel("client hit F1 (50 ms)"); axes[0].legend()
fig.tight_layout(); fig.savefig(OUT / "f1_client.png", dpi=150)

fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
for ax, m in zip(axes, METHODS):
    for p, c in zip(PAIRS, ("tab:green", "tab:orange", "tab:red")):
        sel = [r for r in segs if r["method"] == m and r["pair"] == p]
        t = np.array([f(r["true_async"]) for r in sel]); e = np.array([f(r["est_async"]) for r in sel]); ok = ~np.isnan(t) & ~np.isnan(e)
        ax.scatter(t[ok], np.clip(e[ok], -295, 395), s=10, alpha=.6, c=c, label=p)
    ax.plot([-200, 250], [-200, 250], "k:", lw=1); ax.set_ylim(-300, 400); ax.set_title(f"{m} (points beyond the frame are clipped)"); ax.set_xlabel("true mean timing (ms, + = late)"); ax.grid(alpha=.3)
axes[0].set_ylabel("estimated mean timing (ms)"); axes[0].legend()
fig.tight_layout(); fig.savefig(OUT / "segment_timing.png", dpi=150)
