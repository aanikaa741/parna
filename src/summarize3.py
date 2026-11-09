"""Tables for the v3 test (who leads, engagement, progress, songs) -> results/v3/tables.md, summary.json."""
import csv, json, pathlib, numpy as np
from synth3 import SONGS

OUT = pathlib.Path(__file__).resolve().parent.parent / "results" / "v3"
R = list(csv.DictReader(open(OUT / "exercises.csv")))
f = lambda v: float(v) if v not in ("", "nan", "None") else np.nan
b = lambda v: v == "True"
L, S = [], {}

# 1. who leads (free play where the client joined)
fp = [r for r in R if r["kind"] == "free_play" and b(r["script_joined"])]
L += ["## 1. Who leads in free play (exercises where the client joined)\n",
      "| scripted leader | n | right from true hits | right from detected hits |", "|---|---|---|---|"]
for lead in ("therapist", "client", "mutual"):
    rs = [r for r in fp if r["script_leader"] == lead]
    a = np.mean([r["true_leader"] == lead for r in rs]); e = np.mean([r["est_leader"] == lead for r in rs])
    L.append(f"| {lead} | {len(rs)} | {a:.0%} | {e:.0%} |"); S[f"leader_{lead}"] = [len(rs), a, e]
a = np.mean([r["true_leader"] == r["script_leader"] for r in fp]); e = np.mean([r["est_leader"] == r["script_leader"] for r in fp])
L.append(f"| all | {len(fp)} | {a:.0%} | {e:.0%} |"); S["leader_all"] = [len(fp), a, e]
meas = [r for r in fp if r["est_leader"] not in ("", "None")]
for lead in ("therapist", "client", "mutual"):
    rs = [r for r in meas if r["script_leader"] == lead]
    S[f"leadm_{lead}"] = [len(rs), float(np.mean([r["est_leader"] == lead for r in rs]))]
S["leadm_all"] = [len(meas), float(np.mean([r["est_leader"] == r["script_leader"] for r in meas]))]
L.append(f"\nWhen there were enough shared beats to judge ({len(meas)} of {len(fp)}), detected-hit verdict right in {S['leadm_all'][1]:.0%}. The rest: client played too few beats, and the tool says so instead of guessing.")
# confusion, detected
L += ["\nDetected-hit confusion (rows: scripted, columns: measured):\n", "| | therapist | client | mutual | none |", "|---|---|---|---|---|"]
for lead in ("therapist", "client", "mutual"):
    rs = [r for r in fp if r["script_leader"] == lead]
    L.append(f"| {lead} | " + " | ".join(str(sum(r["est_leader"] == m for r in rs)) for m in ("therapist", "client", "mutual", "")) + " |")

# 2. engagement
pl = [r for r in R if r["kind"] in ("steady", "free_play")]
jn = [r for r in pl if b(r["script_joined"])]
agree = np.mean([b(r["est_joined"]) == b(r["script_joined"]) for r in pl])
fa = sum(b(r["est_joined"]) and not b(r["script_joined"]) for r in pl); mi = sum(not b(r["est_joined"]) and b(r["script_joined"]) for r in pl)
jb = [abs(f(r["est_join_beat"]) - f(r["script_join_beat"])) for r in jn if r["est_join_beat"] not in ("", "None")]
ps = [abs(f(r["est_played_share"]) - f(r["true_played_share"])) for r in pl if r["est_played_share"] not in ("", "None")]
L += ["\n## 2. Engagement (steady and free play)\n",
      f"- Joined or not: right in {agree:.0%} of {len(pl)} exercises ({fa} false 'joined', {mi} missed).",
      f"- Beat the client joined on: exact in {np.mean(np.array(jb) == 0):.0%}, within one beat in {np.mean(np.array(jb) <= 1):.0%} (n={len(jb)}).",
      f"- Share of beats played: median error {np.median(ps) * 100:.1f} points, 90th percentile {np.percentile(ps, 90) * 100:.1f} (n={len(ps)})."]
S.update({"joined_agree": agree, "join_beat_exact": float(np.mean(np.array(jb) == 0)), "join_beat_within1": float(np.mean(np.array(jb) <= 1)),
          "played_share_err_median": float(np.median(ps))})
cr = [r for r in R if r["kind"] == "call_response"]
L.append(f"- Calls answered (of 4 per session): exact count in {np.mean([f(r['est_answered']) == f(r['script_answered']) for r in cr]):.0%} of {len(cr)} sessions.")

# 3. progress over 8 weeks: a weekly engagement score
def score(rows, src):
    p = [r for r in rows if r["kind"] in ("steady", "free_play")]
    share = np.mean([f(r[f"{src}_played_share"]) if r[f"{src}_played_share"] not in ("", "None") else 0 for r in p])
    c = [r for r in rows if r["kind"] == "call_response"]
    ans = np.mean([f(r[f"{src}_answered"]) / 4 for r in c]) if c else np.nan
    return 0.5 * share + 0.5 * ans

L += ["\n## 3. Progress across 8 weekly sessions (engagement score = mean of share of beats played and share of calls answered)\n",
      "| client | favourite song | score week 0 -> 7, true hits | detected | weekly score error (median) | true skill week 0 -> 7 |", "|---|---|---|---|---|---|"]
clients = sorted({r["client"] for r in R}); prog = {}
for c in clients:
    tr_, es_, sk = [], [], []
    for w in range(8):
        rows = [r for r in R if r["client"] == c and int(r["week"]) == w]
        tr_.append(score(rows, "true")); es_.append(score(rows, "est")); sk.append(f(rows[0]["skill"]))
    fav = json.load(open(OUT / "sessions" / f"{c}_w0.json"))["profile"]["favourite"]
    prog[c] = {"true": tr_, "est": es_, "skill": sk, "favourite": fav}
    L.append(f"| {c} | {fav} | {tr_[0]:.2f} -> {tr_[-1]:.2f} | {es_[0]:.2f} -> {es_[-1]:.2f} | {np.median(np.abs(np.array(tr_) - np.array(es_))):.3f} | {sk[0]:.2f} -> {sk[-1]:.2f} |")
S["progress"] = prog

# 4. songs: does ranking songs by engagement find the hidden favourite?
L += ["\n## 4. Which song gets them playing (rank songs by mean share of beats played, detected hits)\n",
      "| client | favourite | top song after 8 sessions | favourite's rank |", "|---|---|---|---|"]
hit = 0
for c in clients:
    rows = [r for r in pl if r["client"] == c]
    m = {s: np.mean([f(r["est_played_share"]) if r["est_played_share"] not in ("", "None") else 0 for r in rows if r["script_song"] == s]) for s in SONGS}
    order = sorted(SONGS, key=lambda s: -m[s]); fav = prog[c]["favourite"]; hit += order[0] == fav
    L.append(f"| {c} | {fav} | {order[0]} | {order.index(fav) + 1} of {len(SONGS)} |")
S["song_top_hits"] = [hit, len(clients)]

(OUT / "tables.md").write_text("\n".join(L) + "\n")
json.dump(S, open(OUT / "summary.json", "w"), indent=1, default=float)
print("\n".join(L))
