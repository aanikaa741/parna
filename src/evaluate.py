"""Week 1 test: both methods on held-out synthetic sessions across a difficulty grid.

Writes results/sessions.csv, results/segments.csv, results/report.md and plots.
"""
import csv, json, itertools, pathlib, numpy as np
from multiprocessing import Pool
from synth import Session, random_script, make_session, calibration_clip, PAIRS
from analyze import analyze, match_f1, sync_measures

OUT = pathlib.Path(__file__).resolve().parent.parent / "results"
TUNED = json.load(open(OUT / "tuned.json"))
SNRS, RTS, SEEDS = [5, 15, 30], [0.2, 0.6, 1.2], range(3)


def seg_bounds(tr):
    beats = np.array(tr["therapist_beats"]); segs = tr["config"]["segments"]
    period = np.median(np.diff(beats)); out, i = [], 0
    for s in segs:
        out.append((s, beats[i] - period / 2, beats[min(i + s["beats"], len(beats)) - 1] + period / 2)); i += s["beats"]
    return out


def run(job):
    pair, snr, rt, seed = job
    s = 1000 * (list(PAIRS).index(pair)) + 100 * SNRS.index(snr) + 10 * RTS.index(rt) + seed  # unique per job
    s += 50_000  # test seeds start at 50000, far from the dev seeds 100-111 used in tune.py
    rng = np.random.default_rng(s)
    cfg = Session(seed=s, pair=pair, bpm=float(rng.choice([70, 100, 140])), snr_db=snr, rt60=rt,
                  tempo_change=float(rng.choice([1.0, 0.87, 1.15])), segments=random_script(rng))
    x, tr = make_session(cfg)
    cal = calibration_clip(pair, s, rt60=rt)
    T, C = np.array(tr["therapist_hits"]), np.array(tr["client_hits"])
    rows, segrows = [], []
    for method in ("blind", "calibrated"):
        est = analyze(x, method, cal, tr, alpha=TUNED["alpha"], lag=TUNED["lag"])
        ft, fc = match_f1(est["therapist"], T), match_f1(est["client"], C)
        rows.append(dict(pair=pair, snr=snr, rt60=rt, seed=s, bpm=cfg.bpm, method=method,
                         f1_therapist=ft["f1"], f1_client=fc["f1"], p_client=fc["precision"], r_client=fc["recall"]))
        for sg, a, b in seg_bounds(tr):
            if sg["kind"] == "silent":
                continue
            tm = sync_measures(T, C[(C >= a) & (C < b)])
            em = sync_measures(np.asarray(est["therapist"]), np.asarray(est["client"])[(np.asarray(est["client"]) >= a) & (np.asarray(est["client"]) < b)])
            segrows.append(dict(pair=pair, snr=snr, rt60=rt, seed=s, method=method, kind=sg["kind"],
                                true_async=tm["mean_async_ms"], est_async=em["mean_async_ms"],
                                true_insync=tm["in_sync_rate"], est_insync=em["in_sync_rate"]))
    return rows, segrows


if __name__ == "__main__":
    jobs = list(itertools.product(PAIRS, SNRS, RTS, SEEDS))
    with Pool() as p:
        res = p.map(run, jobs)
    rows = [r for a, _ in res for r in a]; segs = [r for _, b in res for r in b]
    for name, data in (("sessions.csv", rows), ("segments.csv", segs)):
        with open(OUT / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    print(f"{len(jobs)} sessions, {len(segs)} segment rows")
