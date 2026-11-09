"""v3 test: 5 simulated clients x 8 weekly sessions (40 sessions, seeds 95000+), nothing tuned on them.

Per exercise, each measure is computed twice: from the true hit times and from the hits the calibrated
method finds in the single middle-phone recording. Outputs results/v3/*.csv and per-session JSON
(results/v3/sessions/) for session reports.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import csv, json, pathlib, shutil, tempfile, numpy as np
import soundfile as sf
from multiprocessing import Pool
from synth3 import client_profile, make3, SR
from analyze import analyze, sync_measures, match_f1
from measures3 import correction_gains, leader, engagement

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "v3"
TUNED = json.load(open(ROOT / "results" / "tuned.json"))
CLIENTS = [95000 + 100 * c for c in range(5)]
JOBS = [(c, w) for c in CLIENTS for w in range(8)]


def measure(th, cl, ex, tr):
    a, b = ex["start"], ex["end"]
    w = lambda x: np.asarray(x)[(np.asarray(x) >= a) & (np.asarray(x) < b + 0.3)]
    T, C = w(th), w(cl)
    r = {}
    if ex["kind"] in ("steady", "free_play"):
        e = engagement(T, C, a, b) or {}
        r.update({k: e.get(k) for k in ("joined", "join_beat", "played_share", "longest_run")})
        s = sync_measures(T, C)
        r.update({"mean_async_ms": s["mean_async_ms"], "in_sync_rate": s["in_sync_rate"]})
    if ex["kind"] == "free_play":
        g = correction_gains(T, C)
        r.update({"leader": leader(g), "alpha_th": g and g["therapist"], "alpha_cl": g and g["client"]})
    if ex["kind"] == "call_response":
        P = 60 / tr["bpm"]; n_ans = 0
        for cr in tr["call_response"]:
            c0 = min(cr["call"]) + 4 * P
            if c0 < a or c0 > b:
                continue
            n_ans += int(np.sum((C >= c0 - 0.15) & (C < c0 + 4 * P - 0.1)) > 0)
        r["answered"] = n_ans
    return r


def run(job):
    cseed, week = job
    prof = client_profile(cseed)
    d = pathlib.Path(tempfile.mkdtemp(prefix=f"bs3_{cseed}_{week}_"))
    try:
        tr = make3(prof, week, cseed + week, d)
        x = sf.read(d / "middle.wav")[0].astype(np.float32)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    t0 = tr["calibration"]["client"][1]
    cal = {r: x[int(a * SR):int(b * SR)] for r, (a, b) in tr["calibration"].items()}
    est = analyze(x, "calibrated", cal, alpha=TUNED["alpha"], lag=TUNED["lag"])
    eT = [t for t in est["therapist"] if t > t0]; eC = [t for t in est["client"] if t > t0]
    T = [t for t in tr["therapist_hits"] if t > t0]; C = [t for t in tr["client_hits"] if t > t0]
    rows = []
    for i, ex in enumerate(tr["exercises"]):
        truth_m, est_m = measure(T, C, ex, tr), measure(eT, eC, ex, tr)
        script = {k: ex.get(k) for k in ("leader", "joined", "join_beat", "song")}
        if ex["kind"] == "call_response":
            script["answered"] = sum(bool(cr["response"]) for cr in tr["call_response"])
        rows.append({"client": cseed, "week": week, "skill": tr["skill"], "ex": i, "kind": ex["kind"], "start": ex["start"], "end": ex["end"],
                     **{f"script_{k}": v for k, v in script.items()}, **{f"true_{k}": v for k, v in truth_m.items()}, **{f"est_{k}": v for k, v in est_m.items()}})
    f1 = {"f1_therapist": match_f1(np.array(eT), np.array(T))["f1"], "f1_client": match_f1(np.array(eC), np.array(C))["f1"]}
    (OUT / "sessions").mkdir(parents=True, exist_ok=True)
    json.dump({"client": cseed, "week": week, "profile": prof, "skill": tr["skill"], "bpm": tr["bpm"], "instruments": tr["instruments"],
               "duration": tr["duration"], **f1, "exercises": rows}, open(OUT / "sessions" / f"{cseed}_w{week}.json", "w"), indent=1, default=float)
    return rows


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(4) as p:
        res = p.map(run, JOBS)
    rows = [r for x in res for r in x]
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k.split("_")[0] not in ("client", "week", "skill", "ex", "kind", "start", "end"), k))
    with open(OUT / "exercises.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(len(res), "sessions,", len(rows), "exercises")
