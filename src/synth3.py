"""v3 simulated sessions: a course of weekly sessions for one simulated client.

Adds to synth2.py (whose sessions stay unchanged, so the final test is still reproducible):
- free play: improvised playing where someone leads and the other follows. Each player places their next
  hit from their own previous hit plus their own tempo, then corrects by a fraction (alpha) of the gap to
  the other player's last hit. The leader barely corrects (alpha ~0.05) and lets the tempo wander;
  the follower corrects strongly (~0.5) and adopts the leader's tempo. "Mutual" = both correct ~0.25.
  This is the standard linear phase-correction model of sensorimotor synchronisation.
- engagement: whether the client joins an exercise at all, how many beats they wait, and whether they
  drop out partway. These depend on the client's skill (rising over the course) and on the song.
- songs: each steady or free-play exercise is tagged with a song. Every client has a hidden favourite;
  liked songs make joining in more likely and dropping out less likely.

The song is metadata only: the audio is still drums (and speech), as the drums are what the tool measures.
"""
import json, pathlib
import numpy as np
import soundfile as sf
from synth2 import hit, Timeline, render, PAIRS, SR

SONGS = ["Twinkle Twinkle", "Row Your Boat", "Old MacDonald", "Happy and You Know It", "Wheels on the Bus"]
LEADERS = {"therapist": (0.05, 0.5), "client": (0.5, 0.05), "mutual": (0.25, 0.25)}  # (alpha_therapist, alpha_client)


def client_profile(client_seed):
    """Hidden traits of one simulated client: instrument pairing, base tempo, song liking, learning rate."""
    r = np.random.default_rng(client_seed)
    like = r.uniform(0.45, 0.8, len(SONGS)); fav = int(r.integers(len(SONGS))); like[fav] = 1.0
    return {"client_seed": int(client_seed), "pair_kind": str(r.choice(["different", "related"])),
            "bpm": float(r.choice([72, 90, 110])), "song_like": like.tolist(), "favourite": SONGS[fav],
            "skill0": float(r.uniform(0.15, 0.35)), "gain": float(r.uniform(0.05, 0.09))}


def skill_at(prof, week, rng):
    return float(np.clip(prof["skill0"] + prof["gain"] * week + rng.normal(0, 0.06), 0.05, 0.95))


def free_play(n, P, leader, rng, jit_th=0.010, jit_cl=0.020):
    """Hit times for two players under linear phase and period correction. Returns (therapist, client)."""
    a_th, a_cl = LEADERS[leader]
    b_th, b_cl = (0.0 if a_th < 0.1 else 0.3), (0.0 if a_cl < 0.1 else 0.3)  # followers also adopt the tempo
    if leader == "mutual":
        b_th = b_cl = 0.15
    Pth, Pcl = P * (1 + rng.normal(0, 0.03)), P * (1 + rng.normal(0, 0.03))
    th, cl = [0.0], [rng.normal(0.02, 0.03)]
    for k in range(1, n):
        # the leader's tempo wanders (they're improvising); followers only drift a little
        Pth *= np.exp(rng.normal(0, 0.025 if a_th < 0.1 else 0.006 if leader != "mutual" else 0.012))
        Pcl *= np.exp(rng.normal(0, 0.025 if a_cl < 0.1 else 0.006 if leader != "mutual" else 0.012))
        Pth, Pcl = np.clip(Pth, P * 0.7, P * 1.4), np.clip(Pcl, P * 0.7, P * 1.4)
        A_th, A_cl = th[-1] - cl[-1], cl[-1] - th[-1]
        ioi_th, ioi_cl = th[-1] - th[-2] if k > 1 else Pth, cl[-1] - cl[-2] if k > 1 else Pcl
        th.append(th[-1] + Pth - a_th * A_th + rng.normal(0, jit_th))
        cl.append(cl[-1] + Pcl - a_cl * A_cl + rng.normal(0, jit_cl))
        Pth += b_th * (ioi_cl - Pth); Pcl += b_cl * (ioi_th - Pcl)
    return np.array(th), np.array(cl)


def build3(prof, week, session_seed):
    rng = np.random.default_rng(session_seed)
    pair = PAIRS[prof["pair_kind"]][prof["client_seed"] % len(PAIRS[prof["pair_kind"]])]
    bpm = prof["bpm"]; P = 60 / bpm
    skill = skill_at(prof, week, rng)
    tl = Timeline()
    th_i, cl_i = pair

    def th_hit(t, vel=None):
        tl.add("therapist_drum", t, hit(th_i, rng, vel or rng.uniform(0.65, 1.0))); tl.truth["therapist_hits"].append(float(t))

    def cl_hit(t, vel=None):
        tl.add("client_drum", t, hit(cl_i, rng, vel or rng.uniform(0.35, 0.9))); tl.truth["client_hits"].append(float(t))

    t = tl.say(0.8, "First, tap your drum five times on your own.")
    for role, fn in (("therapist", th_hit), ("client", cl_hit)):
        a = t
        for k in range(5):
            fn(t + k * 0.55 + rng.normal(0, 0.03))
        tl.truth["calibration"][role] = [a - 0.1, t + 5 * 0.55]
        tl.script.append((a, f"{role.upper()} taps alone 5 times (calibration)"))
        t += 5 * 0.55 + 0.8

    songs = list(rng.permutation(SONGS))
    exercises = ["steady", "free_play", "call_response", "steady", "free_play"]
    rng.shuffle(exercises)
    jit = 40 - 28 * skill  # ms, the client's timing spread shrinks as skill grows
    for ex in exercises:
        start = t
        if ex in ("steady", "free_play"):
            song = songs.pop(); like = prof["song_like"][SONGS.index(song)]
            join_p = np.clip(0.35 + 0.6 * skill + 0.5 * (like - 0.7), 0.05, 0.98)
            joins = bool(rng.random() < join_p)
            delay = int(rng.integers(0, 2) if rng.random() < skill * like else rng.integers(2, 7))  # beats before joining
            n = int(rng.integers(20, 29))
            stop = n if rng.random() < 0.35 + 0.6 * skill * like else int(rng.integers(n // 3, n - 3))  # drops out?
        if ex == "steady":
            t = tl.say(t, f"Let's play {song}. Ready, go.")
            off = rng.normal(70 * (1 - skill), 15)
            played = []
            for b in range(n):
                bt = t + b * P + rng.normal(0, 0.008); th_hit(bt)
                if joins and delay <= b < stop:
                    ct = bt + (off + rng.normal(0, jit)) / 1000; cl_hit(ct); played.append(b)
            t += n * P
            tl.truth["exercises"].append({"kind": "steady", "song": song, "start": start, "end": t, "bpm": bpm, "joined": joins,
                                          "join_beat": delay if joins else None, "stop_beat": stop if joins else None, "beats": n,
                                          "true_offset_ms": float(off)})
            tl.script.append((start, f"STEADY BEAT to '{song}', {n} beats. Client " + (f"joins at beat {delay + 1}" + (f", stops at beat {stop}" if stop < n else ", plays to the end") if joins else "doesn't join")))
        elif ex == "free_play":
            leader = str(rng.choice(["therapist", "client", "mutual"]))
            t = tl.say(t, f"Now you lead. {song}, any way you like." if leader == "client" else f"Let's play {song} together, free.")
            th, cl = free_play(n, P, leader, rng, jit_cl=jit / 1000)
            if not joins:  # client doesn't join: the therapist just plays steadily
                th, cl = np.arange(n) * P + rng.normal(0, 0.008, n), np.array([])
            for x in th:
                th_hit(t + x)
            for b, x in enumerate(cl):
                if delay <= b < stop:
                    cl_hit(t + x)
            end = t + max(th[-1], cl[-1] if len(cl) else 0) + P
            tl.truth["exercises"].append({"kind": "free_play", "song": song, "leader": leader if joins else None, "start": start, "end": end,
                                          "bpm": bpm, "joined": joins, "join_beat": delay if joins else None, "stop_beat": stop if joins else None,
                                          "beats": n, "alpha_therapist": LEADERS[leader][0], "alpha_client": LEADERS[leader][1]})
            tl.script.append((start, f"FREE PLAY to '{song}', {n} beats. " + (f"Leader: {leader}. Client joins at beat {delay + 1}" if joins else "Client doesn't join")))
            t = end
        elif ex == "call_response":
            t = tl.say(t, "I play, then you copy me.")
            for rep in range(4):
                grid = np.sort(rng.choice(np.arange(1, 8), size=int(rng.integers(2, 5)), replace=False))
                pattern = np.concatenate([[0], grid]) * P / 2
                call = [t + p + rng.normal(0, 0.008) for p in pattern]
                for c in call:
                    th_hit(c)
                answers = rng.random() < 0.3 + 0.65 * skill
                r0 = t + 4 * P + (rng.normal(0.02, 0.03) if rng.random() < skill else rng.uniform(0.2, 0.5))
                resp = []
                if answers:
                    for p in pattern:
                        if rng.random() < 0.5 + 0.5 * skill:
                            tt = r0 + p + rng.normal(0, jit / 1000); cl_hit(tt); resp.append(float(tt))
                tl.truth["call_response"].append({"call": [float(c) for c in call], "response": resp, "kind": "answered" if resp else "none",
                                                  "pattern_beats": (pattern / P).tolist()})
                tl.script.append((t, f"CALL: {len(pattern)} hits; client " + (f"answers with {len(resp)}" if resp else "stays quiet")))
                t += 8 * P + 0.3
            tl.truth["exercises"].append({"kind": "call_response", "start": start, "end": t, "bpm": bpm})
        t += 0.6
    t = tl.say(t, "That's the end. Thank you!")
    dur = t + 1.0
    tl.truth.update({"week": week, "skill": skill, "client": prof})
    return tl, dur, pair, bpm, rng


def make3(prof, week, session_seed, out_dir):
    tl, dur, pair, bpm, rng = build3(prof, week, session_seed)
    audio, meta = render(tl, dur, rng)
    d = pathlib.Path(out_dir); d.mkdir(parents=True, exist_ok=True)
    sf.write(d / "middle.wav", audio["middle"], SR, subtype="PCM_16")
    tl.truth.update({"seed": session_seed, "pair_kind": prof["pair_kind"], "instruments": {"therapist": pair[0], "client": pair[1]},
                     "bpm": bpm, "duration": dur, **{k: meta[k] for k in ("rt60", "snr_db", "phone_angle")}})
    (d / "truth.json").write_text(json.dumps(tl.truth, indent=1))
    lines = [f"{int(t // 60):02d}:{t % 60:05.2f}  {s}" for t, s in sorted(tl.script)]
    (d / "script.txt").write_text("\n".join(lines) + "\n")
    return tl.truth
