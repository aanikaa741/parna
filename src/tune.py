"""Pick the calibrated method's two settings on DEV seeds (100-111) only; test seeds never touch this."""
import json, itertools, numpy as np
from synth import Session, random_script, make_session, calibration_clip, PAIRS
from analyze import activations, onsets_from_activation, match_f1

cache = []
for seed in range(100, 112):
    rng = np.random.default_rng(seed)
    pair = list(PAIRS)[seed % 3]
    cfg = Session(seed=seed, pair=pair, bpm=float(rng.choice([70, 100, 140])), snr_db=float(rng.choice([10, 20, 30])),
                  rt60=float(rng.choice([0.2, 0.5, 1.0])), tempo_change=float(rng.choice([1.0, 0.87, 1.15])), segments=random_script(rng))
    x, tr = make_session(cfg)
    cache.append((activations(x, calibration_clip(pair, seed, rt60=cfg.rt60)), tr))

best = []
for alpha, lag in itertools.product([0.2, 0.3, 0.4, 0.5, 0.6], [1, 2, 3, 5]):
    f = []
    for acts, tr in cache:
        for role in ("therapist", "client"):
            f.append(match_f1(onsets_from_activation(acts[role], alpha, lag), tr[f"{role}_hits"])["f1"])
    best.append((float(np.mean(f)), alpha, lag))
best.sort(reverse=True)
for row in best[:6]:
    print("mean F1 %.3f  alpha %.1f  lag %d" % row)
json.dump({"alpha": best[0][1], "lag": best[0][2], "dev_f1": best[0][0]}, open("../results/tuned.json", "w"), indent=1)
