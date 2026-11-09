"""Tables (with 95% bootstrap intervals over sessions) and figures for the final report."""
import csv, json, pathlib, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = pathlib.Path(__file__).resolve().parent.parent / "results" / "final"
rd = lambda n: list(csv.DictReader(open(OUT / f"{n}.csv")))
H, TM, TP, CA, SI, AB, ST = (rd(n) for n in ("hits", "timing", "tempo", "calls", "singing", "ablation", "stereo"))
f = lambda v: float(v) if v not in ("", "nan", "None") else np.nan
KINDS, METHODS = ["different", "related", "same"], ["blind", "calibrated", "stereo"]
rng = np.random.default_rng(0)


def boot(groups, stat, n=2000):
    """groups: list of per-session value arrays; resample sessions. Returns (point, lo, hi)."""
    point = stat(np.concatenate(groups)) if groups else np.nan
    bs = []
    for _ in range(n):
        idx = rng.integers(0, len(groups), len(groups))
        bs.append(stat(np.concatenate([groups[i] for i in idx])))
    return point, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)


def by_session(rows, key, filt):
    g = {}
    for r in rows:
        if filt(r):
            v = f(r[key]) if not callable(key) else key(r)
            g.setdefault(r["seed"], []).append(v)
    return [np.array(v, float) for v in g.values()]


fmt = lambda t, d=2: f"{t[0]:.{d}f} [{t[1]:.{d}f}, {t[2]:.{d}f}]"
L, summary = [], {}

L += ["## 1. Hit detection F1 (50 ms), 20 sessions per pairing, mean [95% CI]\n", "| pairing | method | therapist | client |", "|---|---|---|---|"]
for k in KINDS:
    for m in METHODS:
        a = boot(by_session(H, "f1_therapist", lambda r: r["pair_kind"] == k and r["method"] == m), np.nanmean)
        b = boot(by_session(H, "f1_client", lambda r: r["pair_kind"] == k and r["method"] == m), np.nanmean)
        L.append(f"| {k} | {m} | {fmt(a)} | {fmt(b)} |"); summary[f"f1_client_{k}_{m}"] = b

L += ["\n## 2. Exercise timing: median absolute error of the client's mean timing (ms) [95% CI], and in-sync rate error\n",
      "| pairing | method | timing error (ms) | in-sync rate error (points) |", "|---|---|---|---|"]
for k in KINDS:
    for m in METHODS:
        e = boot(by_session(TM, lambda r: abs(f(r["true_async"]) - f(r["est_async"])), lambda r: r["pair_kind"] == k and r["method"] == m), np.nanmedian)
        i = boot(by_session(TM, lambda r: abs(f(r["true_insync"]) - f(r["est_insync"])) * 100, lambda r: r["pair_kind"] == k and r["method"] == m), np.nanmedian)
        L.append(f"| {k} | {m} | {fmt(e, 0)} | {fmt(i, 0)} |"); summary[f"timing_{k}_{m}"] = e

L += ["\n## 3. Call and response (client answered?) and response start error\n",
      "| pairing | method | answers found | silences found | start error (ms, median) | exact hit count |", "|---|---|---|---|---|---|"]
for k in KINDS:
    for m in METHODS:
        rs = [r for r in CA if r["pair_kind"] == k and r["method"] == m]
        ans = [r for r in rs if r["answered_true"] == "1"]; sil = [r for r in rs if r["answered_true"] == "0"]
        lat = [f(r["latency_err_ms"]) for r in ans]; lat = [x for x in lat if not np.isnan(x)]
        L.append(f"| {k} | {m} | {sum(r['answered_est'] == '1' for r in ans)}/{len(ans)} | {sum(r['answered_est'] == '0' for r in sil)}/{len(sil)} | {np.median(lat):.0f} | {np.mean([r['n_true'] == r['n_est'] for r in ans]):.0%} |")

L += ["\n## 4. Tempo following: client lag in beats after a tempo change\n",
      "| pairing | method | measure exact vs script (from true hits) | exact | within 1 beat |", "|---|---|---|---|---|"]
for k in KINDS:
    for m in METHODS:
        rs = [r for r in TP if r["pair_kind"] == k and r["method"] == m]
        s, th, e = np.array([f(r["lag_script"]) for r in rs]), np.array([f(r["lag_from_true_hits"]) for r in rs]), np.array([f(r["lag_est"]) for r in rs])
        L.append(f"| {k} | {m} | {np.mean(th == s):.0%} (n={len(rs)}) | {np.mean(e == s):.0%} | {np.mean(np.abs(e - s) <= 1):.0%} |")

sang = [r for r in SI]
tp = sum(r["sang_true"] == "1" and r["sang_est"] == "1" for r in sang); fn = sum(r["sang_true"] == "1" and r["sang_est"] == "0" for r in sang)
fp = sum(r["sang_true"] == "0" and r["sang_est"] == "1" for r in sang); tn = sum(r["sang_true"] == "0" and r["sang_est"] == "0" for r in sang)
se = [f(r["start_err_ms"]) for r in sang]; se = [x for x in se if not np.isnan(x)]
L += ["\n## 5. Client singing (pitch-based, all pairings)\n",
      f"Sang and detected {tp}/{tp + fn}; stayed quiet and detected as quiet {tn}/{tn + fp}. Start error median {np.median(se):.0f} ms (n={len(se)})."]

L += ["\n## 6. Calibration length (mono calibrated, client F1)\n", "| pairing | 2 taps | 3 taps | 5 taps |", "|---|---|---|---|"]
for k in KINDS:
    c = [fmt(boot(by_session(AB, "f1_client", lambda r: r["pair_kind"] == k and r["taps"] == str(n)), np.nanmean)) for n in (2, 3)]
    c.append(fmt(boot(by_session(H, "f1_client", lambda r: r["pair_kind"] == k and r["method"] == "calibrated"), np.nanmean)))
    L.append(f"| {k} | " + " | ".join(c) + " |")

ang = {r["seed"]: f(r["phone_angle"]) for r in ST}
L += ["\n## 7. Identical drums: client F1 by phone orientation (0 = pointing at the players, 90 = across)\n",
      "| orientation | sessions | mono calibrated | stereo |", "|---|---|---|---|"]
for lo, hi in ((0, 30), (30, 60), (60, 90)):
    sel = lambda r, m: r["pair_kind"] == "same" and r["method"] == m and lo <= min(ang[r["seed"]] % 180, 180 - ang[r["seed"]] % 180) < hi
    mono = [f(r["f1_client"]) for r in H if sel(r, "calibrated")]; ster = [f(r["f1_client"]) for r in H if sel(r, "stereo")]
    L.append(f"| {lo} to {hi} degrees | {len(mono)} | {np.mean(mono):.2f} | {np.mean(ster):.2f} |")

(OUT / "tables.md").write_text("\n".join(L) + "\n")
print("\n".join(L))

# --- figures
colors = {"blind": "#9aa0a6", "calibrated": "#1a73e8", "stereo": "#e8710a"}
fig, ax = plt.subplots(figsize=(6.4, 3.2))
w = 0.26
for j, m in enumerate(METHODS):
    pts = [summary[f"f1_client_{k}_{m}"] for k in KINDS]
    x = np.arange(3) + (j - 1) * w
    ax.bar(x, [p[0] for p in pts], w, color=colors[m], label=m)
    ax.errorbar(x, [p[0] for p in pts], yerr=[[p[0] - p[1] for p in pts], [p[2] - p[0] for p in pts]], fmt="none", ecolor="k", capsize=2, lw=1)
ax.set_xticks(range(3), ["different instruments", "related drums", "identical drums"]); ax.set_ylim(0, 1.05)
ax.set_ylabel("client hit F1"); ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=8); ax.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "fig_f1.png", dpi=200)

fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), sharex=True, sharey=True)
for ax, m in zip(axes, METHODS):
    for k, c in zip(KINDS, ("#188038", "#f9ab00", "#d93025")):
        rs = [r for r in TM if r["method"] == m and r["pair_kind"] == k]
        t = np.array([f(r["true_async"]) for r in rs]); e = np.array([f(r["est_async"]) for r in rs]); ok = ~np.isnan(t) & ~np.isnan(e)
        ax.scatter(t[ok], np.clip(e[ok], -195, 295), s=8, alpha=.6, c=c, label=k)
    ax.plot([-150, 250], [-150, 250], "k:", lw=1); ax.set_title(m); ax.set_xlabel("true mean timing (ms)"); ax.set_ylim(-200, 300); ax.grid(alpha=.3)
axes[0].set_ylabel("estimated (ms; clipped)"); axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig_timing.png", dpi=200)
json.dump({k: list(map(float, v)) for k, v in summary.items()}, open(OUT / "summary.json", "w"), indent=1)
