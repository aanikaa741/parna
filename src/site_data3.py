"""Compact JSON for the website's lead and report sections, from results/v3."""
import json, pathlib, sys
from synth3 import SONGS
V3 = pathlib.Path(__file__).resolve().parent.parent / "results" / "v3"
S = json.load(open(V3 / "summary.json")); songs = json.load(open(V3 / "songs.json"))
nz = lambda v: None if v in ("", None) or v != v else v
clients = []
for c in sorted({p.name.split("_")[0] for p in (V3 / "sessions").glob("*.json")}):
    weeks = []
    for w in range(8):
        d = json.load(open(V3 / "sessions" / f"{c}_w{w}.json"))
        ex = []
        for e in d["exercises"]:
            x = {"kind": e["kind"]}
            if e["kind"] == "call_response":
                x["answered"] = int(float(e["est_answered"]))
            else:
                x.update({"song": e["script_song"], "joined": bool(e["est_joined"]), "join_beat": nz(e["est_join_beat"]),
                          "share": round(float(e["est_played_share"] or 0), 3), "insync": nz(e["est_in_sync_rate"]),
                          "leader": nz(e.get("est_leader")) if e["est_joined"] else None})
            ex.append(x)
        p = S["progress"][str(c)] if str(c) in S["progress"] else S["progress"][c]
        weeks.append({"duration": d["duration"], "ex": ex, "score_est": p["est"][w], "score_true": p["true"][w]})
    clients.append({"instruments": d["instruments"], "bpm": d["bpm"], "favourite": d["profile"]["favourite"], "weeks": weeks})
out = {"clients": clients, "songs": SONGS, "leader": {k[6:]: v for k, v in S.items() if k.startswith("leadm_")}, "songCurve": songs}
pathlib.Path(sys.argv[1]).write_text(json.dumps(out, separators=(",", ":"), default=float))
print(len(clients), "clients")
