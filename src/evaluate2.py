"""Test on realistic simulated sessions (v2). Settings come from week 1 (results/tuned.json), not retuned.

Questions:
 A. Do the close phones give a usable answer key? (needed before trusting week 2 real recordings)
 B. Does the calibrated method still work with real-room echo, speech, singing, distractions and a
    phone microphone, using calibration taps cut from the same recording?
 C. Can it tell whether the client answered in call and response, and how late?
"""
import csv, json, pathlib, numpy as np
import soundfile as sf
import librosa
from multiprocessing import Pool
from synth2 import make, SR
from analyze import analyze, match_f1, sync_measures

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "sessions_v2", ROOT / "results" / "v2"
TUNED = json.load(open(ROOT / "results" / "tuned.json"))
JOBS = [(kind, 70_000 + 100 * i + s) for i, kind in enumerate(["different", "related", "same"]) for s in range(8)]


def close_phone_hits(y, own_hits):
    """Answer-key extraction from a close phone: standard onset detection, nothing clever."""
    on = librosa.onset.onset_detect(y=y.astype(float), sr=SR, hop_length=128, units="time")
    return on


def run(job):
    kind, seed = job
    d = DATA / f"{kind}_{seed}"
    tr = make(seed, d, kind) if not (d / "truth.json").exists() else json.load(open(d / "truth.json"))
    mid = sf.read(d / "middle.wav")[0].astype(np.float32)
    T, C = np.array(tr["therapist_hits"]), np.array(tr["client_hits"])
    row = {"pair_kind": kind, "seed": seed, "instruments": f"{tr['instruments']['therapist']}/{tr['instruments']['client']}",
           "rt60": tr["rt60"], "snr_db": tr["snr_db"], "duration": round(tr["duration"], 1)}
    # A. close phones as answer key
    for role, ref in (("therapist", T), ("client", C)):
        est = close_phone_hits(sf.read(d / f"close_{role}.wav")[0], ref)
        m = match_f1(est, ref, tol=0.03)
        row[f"closephone_{role}_f1"] = round(m["f1"], 3); row[f"closephone_{role}_precision"] = round(m["precision"], 3)
    # B. middle phone, calibration cut from the same recording
    cal = {r: mid[int(a * SR):int(b * SR)] for r, (a, b) in tr["calibration"].items()}
    t_end = tr["calibration"]["client"][1]
    T_s, C_s = T[T > t_end], C[C > t_end]  # score only after calibration
    seg_rows, cr_rows = [], []
    for method in ("blind", "calibrated"):
        est = analyze(mid, method, cal, {"therapist_hits": T_s, "client_hits": C_s}, alpha=TUNED["alpha"], lag=TUNED["lag"])
        e_T = np.asarray(est["therapist"]); e_C = np.asarray(est["client"])
        e_T, e_C = e_T[e_T > t_end], e_C[e_C > t_end]
        row[f"{method}_therapist_f1"] = round(match_f1(e_T, T_s)["f1"], 3)
        row[f"{method}_client_f1"] = round(match_f1(e_C, C_s)["f1"], 3)
        # false client hits during speech / singing / distractions
        busy = [(s["start"], s["end"]) for s in tr["speech"]] + [(v["start"], v["end"]) for v in tr["voice"]] + [(x["t"] - 0.05, x["t"] + 0.6) for x in tr["distractions"]]
        fp = [t for t in e_C if np.min(np.abs(C_s - t)) > 0.05 and any(a <= t <= b for a, b in busy)] if len(C_s) else []
        row[f"{method}_client_false_hits_in_voice_or_noise"] = len(fp)
        for ex in tr["exercises"]:
            if ex["kind"] not in ("steady", "tempo"):
                continue
            a, b = ex["start"], ex["end"]
            sel = lambda x: x[(x >= a) & (x < b)]
            tm, em = sync_measures(sel(T), sel(C)), sync_measures(sel(e_T), sel(e_C))
            seg_rows.append({"pair_kind": kind, "seed": seed, "method": method, "exercise": ex["kind"], "behaviour": ex.get("behaviour", f"x{ex.get('ratio')}"),
                             "true_async": tm["mean_async_ms"], "est_async": em["mean_async_ms"], "true_insync": tm["in_sync_rate"], "est_insync": em["in_sync_rate"]})
        # C. call and response: did the client answer, and when did the answer start?
        for cr in tr["call_response"]:
            call_end = max(cr["call"]); P = 60 / tr["bpm"]
            w0, w1 = min(cr["call"]) + 4 * P - 0.15, min(cr["call"]) + 8 * P - 0.1
            got = e_C[(e_C >= w0) & (e_C < w1)]
            true_start = min(cr["response"]) if cr["response"] else np.nan
            cr_rows.append({"pair_kind": kind, "seed": seed, "method": method, "kind": cr["kind"], "answered_true": int(bool(cr["response"])),
                            "answered_est": int(len(got) > 0), "n_true": len(cr["response"]), "n_est": len(got),
                            "latency_true": true_start - (min(cr["call"]) + 4 * P) if cr["response"] else np.nan,
                            "latency_est": (got.min() - (min(cr["call"]) + 4 * P)) if len(got) else np.nan})
    return row, seg_rows, cr_rows


def write(name, rows):
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(4) as p:
        res = p.map(run, JOBS)
    write("sessions.csv", [r for r, _, _ in res])
    write("exercises.csv", [x for _, s, _ in res for x in s])
    write("call_response.csv", [x for _, _, c in res for x in c])
    print(len(res), "sessions done")
