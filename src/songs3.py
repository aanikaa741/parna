"""How many weekly sessions until ranking songs by engagement finds a client's favourite?

Uses the v3 session script (who joins, when, when they stop) for 300 simulated clients over 16 weeks, without
rendering audio: the v3 test shows engagement from detected hits matches the true hits closely, so this
asks only how much data the comparison needs. Output: results/v3/songs.json and fig_songs.png.
"""
import json, pathlib, numpy as np
import synth3
from synth3 import client_profile, build3, SONGS

synth3.hit = lambda kind, rng, vel: np.zeros(1)          # no audio needed
synth3.Timeline.say = lambda self, t, text, who="therapist": t + 1.5

OUT = pathlib.Path(__file__).resolve().parent.parent / "results" / "v3"
N, W = 300, 16
found = np.zeros((N, W)); in_top2 = np.zeros((N, W))
for i in range(N):
    prof = client_profile(200_000 + i)
    vals = {s: [] for s in SONGS}
    for w in range(W):
        tl = build3(prof, w, 300_000 + 100 * i + w)[0]
        for ex in tl.truth["exercises"]:
            if ex["kind"] in ("steady", "free_play"):
                share = (min(ex["stop_beat"], ex["beats"]) - ex["join_beat"]) / ex["beats"] if ex["joined"] else 0.0
                vals[ex["song"]].append(share)
        m = {s: np.mean(v) if v else -1 for s, v in vals.items()}
        order = sorted(SONGS, key=lambda s: -m[s])
        found[i, w] = order[0] == prof["favourite"]; in_top2[i, w] = prof["favourite"] in order[:2]
res = {"weeks": list(range(1, W + 1)), "top1": found.mean(0).tolist(), "top2": in_top2.mean(0).tolist(), "clients": N,
       "chance_top1": 1 / len(SONGS), "chance_top2": 2 / len(SONGS)}
json.dump(res, open(OUT / "songs.json", "w"), indent=1)
for w in (1, 2, 4, 8, 12, 16):
    print(f"after {w:2d} sessions: favourite ranked first {res['top1'][w - 1]:.0%}, in top two {res['top2'][w - 1]:.0%}")
