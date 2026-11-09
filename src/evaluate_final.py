"""Final test: 60 new realistic simulated sessions (20 per instrument pairing), seeds 90000+.

Settings are fixed in advance: mono calibrated from results/tuned.json (week 1 dev seeds), stereo
theta=0.5 (one v2 session), nothing tuned on these sessions.

Per session and method: hit F1, exercise timing, call and response, tempo following.
Per session (method independent): client singing.
Ablation: number of calibration taps (2, 3, 5) for the mono calibrated method.
Audio is generated into a temp folder and deleted; truth.json files are kept in results/final/truth/.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")  # one thread per worker; the pool provides the parallelism
import csv, json, pathlib, shutil, tempfile, numpy as np
import soundfile as sf
from multiprocessing import Pool
from synth2 import make, SR
from analyze import analyze, stereo_split, match_f1, sync_measures, activations, onsets_from_activation
from measures import tempo_follow_lag, client_singing

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "final"
TUNED = json.load(open(ROOT / "results" / "tuned.json"))
KINDS = ["different", "related", "same"]
JOBS = [(k, 90_000 + 1000 * i + s) for i, k in enumerate(KINDS) for s in range(20)]


def after(x, t0):
    x = np.asarray(x, float)
    return x[x > t0]


def score(tr, est, t0, method, rows):
    T, C = after(tr["therapist_hits"], t0), after(tr["client_hits"], t0)
    eT, eC = after(est["therapist"], t0), after(est["client"], t0)
    base = {"pair_kind": tr["pair_kind"], "seed": tr["seed"], "method": method}
    rows["hits"].append({**base, "f1_therapist": match_f1(eT, T)["f1"], "f1_client": match_f1(eC, C)["f1"],
                         "phone_angle": tr["phone_angle"], "rt60": tr["rt60"], "snr_db": tr["snr_db"]})
    for ex in tr["exercises"]:
        a, b = ex["start"], ex["end"]
        w = lambda x: x[(x >= a) & (x < b)]
        if ex["kind"] in ("steady", "tempo"):
            tm, em = sync_measures(w(T), w(C)), sync_measures(w(eT), w(eC))
            rows["timing"].append({**base, "exercise": ex["kind"], "behaviour": ex.get("behaviour", "tempo"),
                                   "true_async": tm["mean_async_ms"], "est_async": em["mean_async_ms"],
                                   "true_insync": tm["in_sync_rate"], "est_insync": em["in_sync_rate"]})
        if ex["kind"] == "tempo":
            k_true_hits = tempo_follow_lag(w(T), w(C))[0]
            k_est = tempo_follow_lag(w(eT), w(eC))[0]
            rows["tempo"].append({**base, "lag_script": ex["client_lag_beats"], "lag_from_true_hits": k_true_hits, "lag_est": k_est})
    P = 60 / tr["bpm"]
    for cr in tr["call_response"]:
        c0 = min(cr["call"]) + 4 * P
        got = eC[(eC >= c0 - 0.15) & (eC < c0 + 4 * P - 0.1)]
        rows["calls"].append({**base, "kind": cr["kind"], "answered_true": int(bool(cr["response"])), "answered_est": int(len(got) > 0),
                              "n_true": len(cr["response"]), "n_est": len(got),
                              "latency_err_ms": abs((min(cr["response"]) - got.min()) * 1000) if cr["response"] and len(got) else np.nan})


def run(job):
    kind, seed = job
    d = pathlib.Path(tempfile.mkdtemp(prefix=f"bs_{seed}_"))
    try:
        tr = make(seed, d, kind)
        L = sf.read(d / "middle.wav")[0].astype(np.float32)
        R = sf.read(d / "middle_top.wav")[0].astype(np.float32)
    finally:
        (OUT / "truth").mkdir(parents=True, exist_ok=True)
        if (d / "truth.json").exists():
            shutil.copy(d / "truth.json", OUT / "truth" / f"{kind}_{seed}.json")
        shutil.rmtree(d, ignore_errors=True)
    calw = tr["calibration"]
    t0 = calw["client"][1]
    cal = {r: L[int(a * SR):int(b * SR)] for r, (a, b) in calw.items()}
    rows = {k: [] for k in ("hits", "timing", "tempo", "calls", "singing", "ablation", "stereo")}
    truth_lite = {"therapist_hits": tr["therapist_hits"], "client_hits": tr["client_hits"]}
    est = analyze(L, "blind", cal, {k: after(v, t0) for k, v in truth_lite.items()})
    score(tr, est, t0, "blind", rows)
    acts = activations(L, cal)  # shared by the mono calibrated and stereo methods
    est = {r: onsets_from_activation(a, TUNED["alpha"], TUNED["lag"]) for r, a in acts.items()}
    score(tr, est, t0, "calibrated", rows)
    est, info = stereo_split(L, R, cal, calw, alpha=TUNED["alpha"], lag=TUNED["lag"], theta=0.5, acts=acts)
    score(tr, est, t0, "stereo", rows)
    rows["stereo"].append({"pair_kind": kind, "seed": seed, "phone_angle": tr["phone_angle"], **{k: v for k, v in info.items()}})
    # calibration length ablation (mono calibrated): first n taps of each player's calibration
    for n in (2, 3):
        cal_n = {}
        for r, (a, b) in calw.items():
            taps = [t for t in tr[f"{r}_hits"] if a <= t <= b]
            cal_n[r] = L[int(a * SR):int(min(b, taps[n - 1] + 0.5) * SR)]
        est = analyze(L, "calibrated", cal_n, alpha=TUNED["alpha"], lag=TUNED["lag"])
        T, C = after(tr["therapist_hits"], t0), after(tr["client_hits"], t0)
        rows["ablation"].append({"pair_kind": kind, "seed": seed, "taps": n, "f1_therapist": match_f1(after(est["therapist"], t0), T)["f1"],
                                 "f1_client": match_f1(after(est["client"], t0), C)["f1"]})
    # client singing (pitch-based, independent of the drum method)
    for ex in tr["exercises"]:
        if ex["kind"] != "vocal":
            continue
        truth_c = [v for v in tr["voice"] if v["who"] == "client" and ex["start"] <= v["start"] <= ex["end"]]
        th_start = min(v["start"] for v in tr["voice"] if v["who"] == "therapist" and ex["start"] <= v["start"] <= ex["end"])
        r = client_singing(L, SR, th_start, ex["end"])
        rows["singing"].append({"pair_kind": kind, "seed": seed, "sang_true": int(bool(truth_c)), "sang_est": int(r["sang"]),
                                "start_err_ms": abs(r["start"] - truth_c[0]["start"]) * 1000 if truth_c and r["sang"] else np.nan})
    return rows


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(4) as p:
        res = p.map(run, JOBS)
    for k in res[0]:
        rows = [r for x in res for r in x[k]]
        if rows:
            with open(OUT / f"{k}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(len(res), "sessions")
