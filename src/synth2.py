"""Realistic simulated sessions (v2): a whole session as three phones would hear it.

Compared with synth.py this adds
- a simulated room (pyroomacoustics image-source model): players, voices and phones at real positions
- three recordings per session, matching the week 2 protocol: one phone close to each player and one
  in the middle of the table (the one the tool analyses)
- a session made of therapy-style exercises: calibration taps, steady beat, call and response,
  speeding up / slowing down, singing together
- spoken cues from the therapist and client vocalisations (espeak-ng speech, synthesised "aah")
- phone microphone character (weak bass, light compression) and room distractions (chair, door, cough)

Every event is placed by this code, so truth.json is an exact answer key, and script.txt is a readable
timeline that doubles as an acting script for real recordings.
"""
import json, pathlib, subprocess, tempfile
import numpy as np
import soundfile as sf
import pyroomacoustics as pra
from scipy.signal import butter, sosfilt, fftconvolve, resample_poly
from synth import hit as base_hit, _bandnoise, _env, pink_noise

SR = 22050

# ----------------------------------------------------------------------------- sound sources

def hit(kind, rng, vel):
    j = lambda s: 1 + rng.normal(0, s)
    if kind == "djembe":  # three strokes, like a real hand drum: bass, tone, slap
        stroke = rng.choice(["bass", "tone", "slap"], p=[0.3, 0.45, 0.25])
        n = int(0.4 * SR); t = np.arange(n) / SR
        if stroke == "bass":
            x = np.sin(2 * np.pi * 75 * j(0.03) * t) * _env(n, 0.002, 0.16) + _bandnoise(rng, n, 200, 1500) * _env(n, 0.001, 0.01) * 0.3
        elif stroke == "tone":
            f = 330 * j(0.03)
            x = (np.sin(2 * np.pi * f * t) + 0.4 * np.sin(2 * np.pi * 1.6 * f * t)) * _env(n, 0.001, 0.08) + _bandnoise(rng, n, 500, 4000) * _env(n, 0.001, 0.012) * 0.4
        else:
            x = _bandnoise(rng, n, 1200, 8000) * _env(n, 0.0005, 0.025) + np.sin(2 * np.pi * 420 * t) * _env(n, 0.001, 0.03) * 0.4
    elif kind == "tambourine":
        n = int(0.35 * SR); t = np.arange(n) / SR
        jingles = sum(np.sin(2 * np.pi * f * j(0.01) * t + rng.uniform(0, 6.3)) for f in (5200, 6100, 7400, 8300))
        x = jingles * _env(n, 0.002, 0.12 * j(0.15)) * 0.5 + _bandnoise(rng, n, 2500, 9500) * _env(n, 0.001, 0.05) + np.sin(2 * np.pi * 180 * t) * _env(n, 0.001, 0.03) * 0.3
    else:
        return base_hit(kind, rng, vel)
    x = x / (np.max(np.abs(x)) + 1e-9)
    brightness = 0.6 + 0.4 * vel  # softer hits are duller, as on real drums
    sos = butter(2, min(0.98, (2500 + 7000 * brightness) / (SR / 2)), output="sos")
    return sosfilt(sos, x) * vel


PAIRS = {
    "different": [("frame_drum", "shaker"), ("djembe", "tambourine"), ("frame_drum", "tambourine")],
    "related": [("frame_drum", "hand_drum"), ("djembe", "hand_drum")],
    "same": [("djembe", "djembe"), ("frame_drum", "frame_drum_b")],
}


def speak(text, voice="en-us+f3", pitch=50, speed=150):
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        subprocess.run(["espeak-ng", "-v", voice, "-p", str(pitch), "-s", str(speed), "-w", f.name, text], check=True, capture_output=True)
        x, sr = sf.read(f.name)
    if sr != SR:
        x = resample_poly(x, SR, sr)
    x = np.trim_zeros(x, "fb")
    return x / (np.max(np.abs(x)) + 1e-9)


def sing_aah(rng, dur, f0, formants):
    n = int(dur * SR); t = np.arange(n) / SR
    vib = 1 + 0.012 * np.sin(2 * np.pi * 5.5 * t + rng.uniform(0, 6))
    f = f0 * vib * (1 + 0.03 * np.sin(2 * np.pi * 0.3 * t))
    phase = 2 * np.pi * np.cumsum(f) / SR
    src = sum(np.sin(k * phase) / k for k in range(1, 30) if k * f0 < SR / 2.3)
    out = np.zeros(n)
    for fc, bw, g in formants:
        sos = butter(2, [(fc - bw / 2) / (SR / 2), (fc + bw / 2) / (SR / 2)], btype="band", output="sos")
        out += g * sosfilt(sos, src)
    env = np.minimum(1, t / 0.08) * np.minimum(1, (dur - t) / 0.15)
    out *= env
    return out / (np.max(np.abs(out)) + 1e-9)


ADULT_A = [(750, 150, 1.0), (1200, 200, 0.6), (2600, 300, 0.25)]
CHILD_A = [(1000, 200, 1.0), (1500, 250, 0.6), (3100, 350, 0.25)]


def distraction(rng):
    kind = rng.choice(["chair", "door", "cough", "paper"])
    if kind == "chair":
        n = int(0.5 * SR); x = _bandnoise(rng, n, 300, 2000) * _env(n, 0.05, 0.2)
    elif kind == "door":
        n = int(0.6 * SR); x = (_bandnoise(rng, n, 60, 600) * _env(n, 0.001, 0.08) + _bandnoise(rng, n, 1000, 5000) * _env(n, 0.001, 0.01))
    elif kind == "cough":
        n = int(0.35 * SR); x = _bandnoise(rng, n, 200, 3500) * _env(n, 0.01, 0.07)
    else:
        n = int(0.8 * SR); x = _bandnoise(rng, n, 2000, 9000) * (0.5 + 0.5 * np.abs(np.sin(np.arange(n) / SR * 23))) * _env(n, 0.05, 0.3)
    return str(kind), x / (np.max(np.abs(x)) + 1e-9)


# ----------------------------------------------------------------------------- the session

class Timeline:
    """Collects every sound with its source, start time and label."""

    def __init__(self):
        self.events = []  # (source, t, audio, label)
        self.truth = {"therapist_hits": [], "client_hits": [], "voice": [], "speech": [], "distractions": [],
                      "exercises": [], "calibration": {}, "call_response": []}
        self.script = []

    def add(self, source, t, audio, label=None):
        self.events.append((source, t, audio, label))

    def say(self, t, text, who="therapist"):
        x = speak(text, voice="en-us+f3" if who == "therapist" else "en-us+f5", pitch=45 if who == "therapist" else 85)
        self.add(f"{who}_voice", t, x * 0.5)
        self.truth["speech"].append({"who": who, "start": t, "end": t + len(x) / SR, "text": text})
        self.script.append((t, f"{who.upper()} says: \"{text}\""))
        return t + len(x) / SR + 0.4


def clap(rng):
    n = int(0.12 * SR)
    x = _bandnoise(rng, n, 600, 9000) * _env(n, 0.0005, 0.012) + _bandnoise(rng, n, 900, 2500) * _env(n, 0.002, 0.04) * 0.5
    return x / (np.max(np.abs(x)) + 1e-9)


def build(seed, pair_kind="different", bpm=None, rt60=None, snr_db=None, claps=False):
    rng = np.random.default_rng(seed)
    pair = PAIRS[pair_kind][rng.integers(len(PAIRS[pair_kind]))]
    bpm = bpm or float(rng.choice([72, 90, 110, 130]))
    tl = Timeline()
    P = 60 / bpm
    th_i, cl_i = pair

    def th_hit(t, vel=None):
        tl.add("therapist_drum", t, hit(th_i, rng, vel or rng.uniform(0.65, 1.0))); tl.truth["therapist_hits"].append(t)

    def cl_hit(t, vel=None):
        tl.add("client_drum", t, hit(cl_i, rng, vel or rng.uniform(0.35, 0.9))); tl.truth["client_hits"].append(t)

    t = 0.8
    if claps:  # sync clap at the start, as in the recording protocol
        tl.add("room", 1.0, clap(rng)); tl.truth["claps"] = [1.0]; t = 3.0
    # 1. calibration: each player taps alone, as in the protocol
    t = tl.say(t, "First, tap your drum five times on your own.")
    for role, fn in (("therapist", th_hit), ("client", cl_hit)):
        a = t
        for k in range(5):
            fn(t + k * 0.55 + rng.normal(0, 0.03))
        tl.truth["calibration"][role] = [a - 0.1, t + 5 * 0.55]
        tl.script.append((a, f"{role.upper()} taps alone 5 times (calibration)"))
        t += 5 * 0.55 + 0.8

    exercises = ["steady", "call_response", "tempo", "steady", "vocal", "call_response"]
    rng.shuffle(exercises[1:])
    for ex in exercises:
        start = t
        if ex == "steady":
            t = tl.say(t, rng.choice(["Let's play together. Ready, go.", "Keep the beat with me.", "Play with me."]))
            beh = str(rng.choice(["sync", "late", "drift", "catchup", "half", "stop_start"], p=[0.35, 0.15, 0.15, 0.15, 0.1, 0.1]))
            n_beats = int(rng.integers(16, 25))
            off = {"sync": rng.normal(-15, 8), "late": rng.uniform(60, 140), "half": rng.normal(-10, 8)}.get(beh, 0.0)
            jit = rng.uniform(10, 28); drift_ratio = rng.choice([0.95, 1.05]); c_t = None
            for b in range(n_beats):
                bt = t + b * P + rng.normal(0, 0.008)
                th_hit(bt)
                if beh == "half" and b % 2:
                    continue
                if beh == "stop_start" and n_beats // 3 <= b < 2 * n_beats // 3:
                    continue
                if beh == "drift":
                    c_t = bt if c_t is None else c_t + P / drift_ratio
                    cl_hit(c_t + rng.normal(0, jit / 1000)); continue
                o = off
                if beh == "catchup":
                    o = 180 * (1 - b / (n_beats - 1)) - 10 * b / (n_beats - 1)
                if beh == "stop_start" and b >= 2 * n_beats // 3:
                    o = 150 * max(0, 1 - (b - 2 * n_beats // 3) / 4)  # comes back in late, then settles
                cl_hit(bt + (o + rng.normal(0, jit)) / 1000)
            t += n_beats * P
            tl.truth["exercises"].append({"kind": "steady", "behaviour": beh, "start": start, "end": t, "bpm": bpm})
            tl.script.append((start, f"STEADY BEAT, {n_beats} beats at {bpm:.0f} bpm. Client: {beh.replace('_', ' ')}"))
        elif ex == "call_response":
            t = tl.say(t, "I play, then you copy me.")
            for rep in range(int(rng.integers(3, 5))):
                grid = np.sort(rng.choice(np.arange(1, 8), size=int(rng.integers(2, 5)), replace=False))
                pattern = np.concatenate([[0], grid]) * P / 2  # eighth-note grid in one bar
                call = [t + p + rng.normal(0, 0.008) for p in pattern]
                for c in call:
                    th_hit(c)
                resp_kind = str(rng.choice(["copy", "partial", "late_start", "rushed", "none"], p=[0.45, 0.2, 0.15, 0.1, 0.1]))
                r0 = t + 4 * P + {"late_start": rng.uniform(0.2, 0.5)}.get(resp_kind, rng.normal(0.02, 0.03))
                scale = 0.8 if resp_kind == "rushed" else 1.0
                resp = []
                if resp_kind != "none":
                    keep = np.ones(len(pattern), bool)
                    if resp_kind == "partial":
                        keep[rng.choice(np.arange(1, len(pattern)), size=max(1, len(pattern) // 2), replace=False)] = False
                    for p, k in zip(pattern, keep):
                        if k:
                            tt = r0 + p * scale + rng.normal(0, 0.025); cl_hit(tt); resp.append(tt)
                tl.truth["call_response"].append({"call": call, "response": resp, "kind": resp_kind, "pattern_beats": (pattern / P).tolist()})
                tl.script.append((t, f"CALL: therapist plays {len(pattern)} hits at beats {', '.join(f'{p / P:.1f}' for p in pattern)}; client response: {resp_kind.replace('_', ' ')}"))
                t += 8 * P + 0.3
            tl.truth["exercises"].append({"kind": "call_response", "start": start, "end": t, "bpm": bpm})
        elif ex == "tempo":
            ratio = float(rng.choice([0.75, 1.3]))
            t = tl.say(t, "Now let's go faster." if ratio > 1 else "Now let's slow down.")
            n_beats, lag = 24, int(rng.integers(1, 6))
            periods = [P * (1 + (1 / ratio - 1) * min(1, max(0, (b - 4) / 12))) for b in range(n_beats)]
            th_t, c_t = [t], [t + rng.normal(-0.01, 0.015)]
            for b in range(1, n_beats):
                th_t.append(th_t[-1] + periods[b - 1])
                c_t.append(c_t[-1] + periods[max(0, b - 1 - lag)] + rng.normal(0, 0.012))
            for a, c in zip(th_t, c_t):
                th_hit(a + rng.normal(0, 0.006)); cl_hit(c)
            t = th_t[-1] + P
            tl.truth["exercises"].append({"kind": "tempo", "ratio": ratio, "client_lag_beats": lag, "start": start, "end": t})
            tl.script.append((start, f"TEMPO CHANGE x{ratio} over 12 beats; client follows {lag} beats behind"))
        elif ex == "vocal":
            t = tl.say(t, "Can you sing ah with me?")
            n_beats = 16
            for b in range(n_beats):
                th_hit(t + b * P + rng.normal(0, 0.008), vel=rng.uniform(0.4, 0.6))  # quieter drum under singing
            d = min(3.0, n_beats * P * 0.4)
            tl.add("therapist_voice", t + P, sing_aah(rng, d, 220, ADULT_A) * 0.45)
            tl.truth["voice"].append({"who": "therapist", "start": t + P, "end": t + P + d})
            client_sings = rng.random() < 0.75
            if client_sings:
                cs = t + P + rng.uniform(0.3, 1.5); cd = rng.uniform(0.8, 3.0)
                tl.add("client_voice", cs, sing_aah(rng, cd, rng.uniform(260, 330), CHILD_A) * rng.uniform(0.25, 0.45))
                tl.truth["voice"].append({"who": "client", "start": cs, "end": cs + cd})
            tl.script.append((t, f"SINGING: therapist sings aah over a soft beat; client {'joins in' if client_sings else 'stays quiet'}"))
            t += n_beats * P
            tl.truth["exercises"].append({"kind": "vocal", "start": start, "end": t})
        t += 0.6
        if rng.random() < 0.5:
            t = tl.say(t, str(rng.choice(["Great job.", "Nice!", "Well done.", "Let's try another one."])))
    t = tl.say(t, "That's the end. Thank you!")
    if claps:  # and at the end, so clock drift between phones can be measured
        tl.add("room", t + 1.0, clap(rng)); tl.truth["claps"].append(t + 1.0); t += 2.5
    dur = t + 1.0

    # room distractions at random moments
    for _ in range(int(rng.integers(2, 6))):
        kind, x = distraction(rng); dt = float(rng.uniform(1, dur - 1))
        tl.add("room", dt, x * rng.uniform(0.2, 0.6)); tl.truth["distractions"].append({"kind": kind, "t": dt})

    return tl, dur, pair, bpm, rng, rt60, snr_db


def render(tl, dur, rng, rt60=None, snr_db=None, close_dist=0.3, phone_angle=None):
    rt60 = rt60 if rt60 is not None else float(rng.choice([0.3, 0.5, 0.8]))
    snr_db = snr_db if snr_db is not None else float(rng.choice([10, 20, 30]))
    room_dim = [rng.uniform(4, 7), rng.uniform(3.5, 6), rng.uniform(2.5, 3.0)]
    e_abs, max_order = pra.inverse_sabine(rt60, room_dim)
    room = pra.ShoeBox(room_dim, fs=SR, materials=pra.Material(e_abs), max_order=min(max_order, 12))
    cx, cy = room_dim[0] / 2, room_dim[1] / 2
    sep = rng.uniform(1.0, 1.6)
    th_pos = np.array([cx - sep / 2, cy, 0.6]); cl_pos = np.array([cx + sep / 2, cy + rng.uniform(-0.3, 0.3), 0.55])
    pos = {"therapist_drum": th_pos, "client_drum": cl_pos,
           "therapist_voice": th_pos + [0, 0, 0.6], "client_voice": cl_pos + [0, 0, 0.45],
           "room": np.array([rng.uniform(0.5, room_dim[0] - 0.5), rng.uniform(0.5, room_dim[1] - 0.5), rng.uniform(0.3, 1.5)])}
    phone = np.array([cx + rng.uniform(-0.15, 0.15), cy - 0.4, 0.75])
    # a phone has two microphones about 14 cm apart (bottom and top edge); its angle on the table is
    # random: 0 = pointing along the line between the players, 90 = across it
    phone_angle = float(rng.uniform(0, 180)) if phone_angle is None else float(phone_angle)
    axis = np.array([np.cos(np.radians(phone_angle)), np.sin(np.radians(phone_angle)), 0.0])
    mics = {"middle": phone - 0.07 * axis, "middle_top": phone + 0.07 * axis,
            "close_therapist": th_pos + [0.0, -close_dist, 0.05], "close_client": cl_pos + [0.0, -close_dist, 0.05]}
    for p in pos.values():
        room.add_source(list(p))
    room.add_microphone_array(np.array(list(mics.values())).T)
    room.compute_rir()
    names = list(pos)
    n = int(dur * SR)
    dry = {s: np.zeros(n) for s in names}
    for s, t, x, _ in tl.events:
        a = int(t * SR)
        if 0 <= a < n:
            m = min(len(x), n - a); dry[s][a:a + m] += x[:m]
    raw = {}
    for mi, mname in enumerate(mics):
        y = np.zeros(n)
        for si, s in enumerate(names):
            if np.any(dry[s]):
                y += fftconvolve(dry[s], room.rir[mi][si])[:n]
        # phone microphone: weak below ~150 Hz, room noise
        y = sosfilt(butter(2, 150 / (SR / 2), btype="high", output="sos"), y)
        rms = np.sqrt(np.mean(y ** 2)) + 1e-12
        noise = pink_noise(rng, n) + 0.3 * np.sin(2 * np.pi * 60 * np.arange(n) / SR)
        noise *= rms / (np.sqrt(np.mean(noise ** 2)) * 10 ** (snr_db / 20))
        raw[mname] = y + noise
    out = {}
    # light compression; the phone's two mics share one gain
    groups = [("middle", "middle_top"), ("close_therapist",), ("close_client",)]
    for g in groups:
        peak = max(np.max(np.abs(raw[m])) for m in g) + 1e-9
        for m in g:
            out[m] = (0.9 * np.tanh(2.0 * raw[m] / peak) / np.tanh(2.0)).astype(np.float32)
    meta = {"rt60": rt60, "snr_db": snr_db, "phone_angle": phone_angle, "room_dim": [float(v) for v in room_dim],
            "positions": {k: [float(v) for v in p] for k, p in pos.items()}, "mics": {k: [float(v) for v in p] for k, p in mics.items()}}
    return out, meta


def make(seed, out_dir, pair_kind="different", **kw):
    tl, dur, pair, bpm, rng, rt60, snr = build(seed, pair_kind, **{k: v for k, v in kw.items() if k in ("bpm", "claps")})
    audio, meta = render(tl, dur, rng, kw.get("rt60"), kw.get("snr_db"), kw.get("close_dist", 0.3), kw.get("phone_angle"))
    meta["close_dist"] = kw.get("close_dist", 0.3)
    d = pathlib.Path(out_dir); d.mkdir(parents=True, exist_ok=True)
    for k, y in audio.items():
        sf.write(d / f"{k}.wav", y, SR, subtype="PCM_16")
    tl.truth.update({"seed": seed, "pair_kind": pair_kind, "instruments": {"therapist": pair[0], "client": pair[1]},
                     "bpm": bpm, "duration": dur, **meta})
    (d / "truth.json").write_text(json.dumps(tl.truth, indent=1))
    lines = [f"Session {seed}: therapist plays {pair[0]}, client plays {pair[1]}, {bpm:.0f} bpm, echo {meta['rt60']} s, noise {meta['snr_db']} dB SNR", ""]
    lines += [f"{int(t // 60):02d}:{t % 60:05.2f}  {s}" for t, s in sorted(tl.script)]
    (d / "script.txt").write_text("\n".join(lines) + "\n")
    return tl.truth


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Write one realistic simulated session: 3 phone WAVs, truth.json, script.txt")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pair", default="different", choices=list(PAIRS))
    ap.add_argument("--out", default="../sessions_v2/example")
    a = ap.parse_args()
    tr = make(a.seed, a.out, a.pair)
    print(f"wrote {a.out}: {tr['duration']:.0f} s, {len(tr['therapist_hits'])} therapist + {len(tr['client_hits'])} client hits")
