"""Tables for the v2 (realistic simulated) test."""
import csv, pathlib, numpy as np

OUT = pathlib.Path(__file__).resolve().parent.parent / "results" / "v2"
S = list(csv.DictReader(open(OUT / "sessions.csv")))
E = list(csv.DictReader(open(OUT / "exercises.csv")))
CR = list(csv.DictReader(open(OUT / "call_response.csv")))
f = lambda v: float(v) if v not in ("", "nan") else np.nan
KINDS = ["different", "related", "same"]
L = []
m = lambda rows, k: np.nanmean([f(r[k]) for r in rows]) if rows else np.nan

L += ["## A. Close phones as the answer key (onset F1 within 30 ms, mean over 8 sessions)\n",
      "| pairing | therapist close phone | client close phone |", "|---|---|---|"]
for k in KINDS:
    rs = [r for r in S if r["pair_kind"] == k]
    L.append(f"| {k} | {m(rs, 'closephone_therapist_f1'):.2f} | {m(rs, 'closephone_client_f1'):.2f} |")

L += ["\n## B. Middle phone: hit detection F1 (50 ms) and false client hits during speech, singing or room noise\n",
      "| pairing | method | therapist F1 | client F1 | false client hits per session |", "|---|---|---|---|---|"]
for k in KINDS:
    rs = [r for r in S if r["pair_kind"] == k]
    for meth in ("blind", "calibrated"):
        L.append(f"| {k} | {meth} | {m(rs, meth + '_therapist_f1'):.2f} | {m(rs, meth + '_client_f1'):.2f} | {m(rs, meth + '_client_false_hits_in_voice_or_noise'):.1f} |")

L += ["\n## B. Middle phone: exercise-level timing (steady beat and tempo change)\n",
      "| pairing | method | exercises | median abs error, mean timing (ms) | correlation | in-sync rate abs error |", "|---|---|---|---|---|---|"]
for k in KINDS:
    for meth in ("blind", "calibrated"):
        rs = [r for r in E if r["pair_kind"] == k and r["method"] == meth]
        t = np.array([f(r["true_async"]) for r in rs]); e = np.array([f(r["est_async"]) for r in rs]); ok = ~np.isnan(t) & ~np.isnan(e)
        ti = np.array([f(r["true_insync"]) for r in rs]); ei = np.array([f(r["est_insync"]) for r in rs]); oki = ~np.isnan(ti) & ~np.isnan(ei)
        c = np.corrcoef(t[ok], e[ok])[0, 1] if ok.sum() > 2 else np.nan
        L.append(f"| {k} | {meth} | {len(rs)} ({ok.sum()} measurable) | {np.median(np.abs(t[ok] - e[ok])):.0f} | {c:.2f} | {np.median(np.abs(ti[oki] - ei[oki])):.2f} |")

L += ["\n## C. Call and response (calibrated method)\n",
      "| pairing | calls | answered correctly detected | silent correctly detected | median latency error (ms) | hit count exact |", "|---|---|---|---|---|---|"]
for k in KINDS:
    rs = [r for r in CR if r["pair_kind"] == k and r["method"] == "calibrated"]
    ans = [r for r in rs if r["answered_true"] == "1"]; sil = [r for r in rs if r["answered_true"] == "0"]
    lat = [abs(f(r["latency_true"]) - f(r["latency_est"])) * 1000 for r in ans if r["answered_est"] == "1"]
    L.append(f"| {k} | {len(rs)} | {sum(r['answered_est'] == '1' for r in ans)}/{len(ans)} | {sum(r['answered_est'] == '0' for r in sil)}/{len(sil)} | {np.median(lat):.0f} | {np.mean([r['n_true'] == r['n_est'] for r in ans]):.0%} |")

(OUT / "tables.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
