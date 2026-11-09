"""Synthetic two-player sessions with an exact answer key.

A "therapist" keeps a beat (with an optional tempo change), a "client" plays along
following a script of behaviours (in sync, late, drifting, silent, catching up).
Every hit time is written by this code, so the true answer is known exactly.
"""
from dataclasses import dataclass, field, asdict
import json
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve

SR = 22050

# ----------------------------------------------------------------------------- instruments

def _bandnoise(rng, n, lo, hi):
    x = rng.standard_normal(n)
    sos = butter(4, [lo / (SR / 2), min(hi / (SR / 2), 0.99)], btype="band", output="sos")
    return sosfilt(sos, x)


def _env(n, attack, decay):
    t = np.arange(n) / SR
    a = np.clip(t / max(attack, 1e-4), 0, 1)
    return a * np.exp(-t / decay)


def hit(kind, rng, vel=1.0):
    """One hit of a hand-percussion instrument. Small random variation per hit, like a real player."""
    j = lambda s: 1 + rng.normal(0, s)  # jitter factor
    if kind == "frame_drum":  # low, boomy
        n = int(0.45 * SR); t = np.arange(n) / SR
        f0 = 95 * j(0.03)
        body = np.sin(2 * np.pi * (f0 * t + 30 * (1 - np.exp(-t / 0.02)) * 0.02)) * _env(n, 0.002, 0.18 * j(0.1))
        skin = _bandnoise(rng, n, 300, 2500) * _env(n, 0.001, 0.015) * 0.5
        x = body + skin
    elif kind == "frame_drum_b":  # a second, similar low drum (hard case)
        n = int(0.45 * SR); t = np.arange(n) / SR
        f0 = 118 * j(0.03)
        body = np.sin(2 * np.pi * f0 * t) * _env(n, 0.002, 0.15 * j(0.1))
        skin = _bandnoise(rng, n, 400, 3000) * _env(n, 0.001, 0.015) * 0.55
        x = body + skin
    elif kind == "hand_drum":  # higher, slap-like (medium case)
        n = int(0.3 * SR); t = np.arange(n) / SR
        f0 = 240 * j(0.04)
        body = (np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(2 * np.pi * 2.3 * f0 * t)) * _env(n, 0.001, 0.07 * j(0.1))
        slap = _bandnoise(rng, n, 800, 6000) * _env(n, 0.0005, 0.02) * 0.8
        x = body + slap
    elif kind == "shaker":  # bright noise (easy case)
        n = int(0.12 * SR)
        x = _bandnoise(rng, n, 3500, 9500) * _env(n, 0.008 * j(0.2), 0.03 * j(0.15))
    else:
        raise ValueError(kind)
    x = x / (np.max(np.abs(x)) + 1e-9)
    return x * vel


PAIRS = {  # therapist instrument, client instrument
    "easy": ("frame_drum", "shaker"),
    "medium": ("frame_drum", "hand_drum"),
    "hard": ("frame_drum", "frame_drum_b"),
}

# ----------------------------------------------------------------------------- room

def room_ir(rng, rt60):
    if rt60 <= 0:
        return np.array([1.0])
    n = int(rt60 * SR)
    t = np.arange(n) / SR
    tail = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)  # -60 dB at rt60
    tail[: int(0.005 * SR)] = 0
    ir = tail * 0.25
    ir[0] = 1.0
    return ir


def pink_noise(rng, n):
    w = rng.standard_normal(n)
    f = np.fft.rfft(w)
    k = np.arange(len(f)); k[0] = 1
    return np.fft.irfft(f / np.sqrt(k), n)


# ----------------------------------------------------------------------------- the script

BEHAVIOURS = ["sync", "late", "drift", "silent", "catchup", "half"]


@dataclass
class Segment:
    kind: str
    beats: int
    offset_ms: float = 0.0      # mean timing offset vs the therapist beat (+ = late)
    jitter_ms: float = 15.0     # how loose the timing is
    tempo_ratio: float = 1.0    # for drift: client tempo / therapist tempo


@dataclass
class Session:
    seed: int
    pair: str = "medium"
    bpm: float = 100.0
    snr_db: float = 20.0
    rt60: float = 0.5
    tempo_change: float = 1.0   # therapist tempo multiplier at the change point (1 = none)
    change_beat: int = 32
    follow_beats: int = 4       # how many beats the client needs to adopt a new tempo
    segments: list = field(default_factory=list)


def random_script(rng, total_beats=64):
    segs, left = [], total_beats
    while left > 0:
        b = int(min(left, rng.integers(8, 17)))
        k = rng.choice(BEHAVIOURS, p=[0.35, 0.15, 0.15, 0.1, 0.15, 0.1])
        s = Segment(kind=str(k), beats=b)
        if k == "sync":
            s.offset_ms, s.jitter_ms = rng.normal(-15, 8), rng.uniform(8, 25)  # people tend to hit slightly early
        elif k == "late":
            s.offset_ms, s.jitter_ms = rng.uniform(60, 140), rng.uniform(10, 30)
        elif k == "drift":
            s.tempo_ratio, s.jitter_ms = rng.choice([0.95, 0.96, 1.04, 1.05]), rng.uniform(10, 25)
        elif k == "catchup":
            s.offset_ms, s.jitter_ms = rng.uniform(120, 220), rng.uniform(10, 25)
        elif k == "half":
            s.offset_ms, s.jitter_ms = rng.normal(-10, 8), rng.uniform(10, 25)
        segs.append(s)
        left -= b
    return segs


def make_session(cfg: Session):
    """Returns (audio, truth dict)."""
    rng = np.random.default_rng(cfg.seed)
    total = sum(s.beats for s in cfg.segments)
    lead_in = 1.0
    # therapist beat times, with human jitter (~8 ms) and a tempo change
    period = 60.0 / cfg.bpm
    t_beats, t = [], lead_in
    for i in range(total):
        p = period * (cfg.tempo_change if i >= cfg.change_beat else 1.0)
        t_beats.append(t)
        t += p
    t_beats = np.array(t_beats)
    t_hits = t_beats + rng.normal(0, 0.008, len(t_beats))

    # client: per beat, follows the therapist's *previous* tempo for follow_beats after a change
    c_hits, c_beat_idx, seg_of_beat = [], [], []
    i = 0
    for si, s in enumerate(cfg.segments):
        drift_t = None
        for b in range(s.beats):
            idx = i + b
            seg_of_beat.append(si)
            target = t_beats[idx]
            # lagging tempo adoption after a change: the client extrapolates the old period
            if cfg.tempo_change != 1.0 and cfg.change_beat <= idx < cfg.change_beat + cfg.follow_beats:
                k = idx - cfg.change_beat + 1
                old_p = period
                new_p = period * cfg.tempo_change
                frac = k / (cfg.follow_beats + 1)
                target = t_beats[cfg.change_beat - 1] + sum(old_p * (1 - (m / (cfg.follow_beats + 1))) + new_p * (m / (cfg.follow_beats + 1)) for m in range(1, k + 1))
            if s.kind == "silent":
                continue
            if s.kind == "half" and b % 2 == 1:
                continue
            if s.kind == "drift":
                drift_t = target if drift_t is None else drift_t + (t_beats[idx] - t_beats[idx - 1]) / s.tempo_ratio
                hit_t = drift_t + rng.normal(0, s.jitter_ms / 1000)
            elif s.kind == "catchup":
                off = s.offset_ms * (1 - b / max(s.beats - 1, 1)) + (-10) * (b / max(s.beats - 1, 1))
                hit_t = target + (off + rng.normal(0, s.jitter_ms)) / 1000
            else:
                hit_t = target + (s.offset_ms + rng.normal(0, s.jitter_ms)) / 1000
            c_hits.append(hit_t); c_beat_idx.append(idx)
        i += s.beats
    c_hits = np.array(c_hits)

    th_inst, cl_inst = PAIRS[cfg.pair]
    dur = t_beats[-1] + 2.0
    n = int(dur * SR)
    th, cl = np.zeros(n), np.zeros(n)

    def place(buf, times, kind, vlo, vhi):
        for tt in times:
            h = hit(kind, rng, rng.uniform(vlo, vhi))
            a = int(tt * SR)
            if a < 0 or a >= n:
                continue
            m = min(len(h), n - a)
            buf[a:a + m] += h[:m]

    place(th, t_hits, th_inst, 0.7, 1.0)
    place(cl, c_hits, cl_inst, 0.4, 0.9)  # clients often play softer and less evenly

    ir = room_ir(rng, cfg.rt60)
    th = fftconvolve(th, ir)[:n]; cl = fftconvolve(cl, ir)[:n]
    mix = th + cl
    sig_rms = np.sqrt(np.mean(mix ** 2))
    noise = pink_noise(rng, n)
    noise *= sig_rms / (np.sqrt(np.mean(noise ** 2)) * 10 ** (cfg.snr_db / 20))
    mix = mix + noise
    mix /= np.max(np.abs(mix)) + 1e-9
    mix *= 0.9

    truth = {
        "config": {**{k: v for k, v in asdict(cfg).items() if k != "segments"}, "segments": [asdict(s) for s in cfg.segments]},
        "therapist_hits": t_hits.tolist(),
        "therapist_beats": t_beats.tolist(),
        "client_hits": c_hits.tolist(),
        "client_beat_index": c_beat_idx,
        "segment_of_beat": seg_of_beat,
        "duration": dur,
    }
    return mix.astype(np.float32), truth


def calibration_clip(pair, seed, rt60=0.5, snr_db=30):
    """A few hits of each instrument alone, as a player would do at the start of a session."""
    rng = np.random.default_rng(10_000 + seed)
    out = {}
    for role, inst in zip(("therapist", "client"), PAIRS[pair]):
        n = int(3.0 * SR)
        buf = np.zeros(n)
        for k in range(5):
            h = hit(inst, rng, rng.uniform(0.5, 1.0))
            a = int((0.2 + 0.5 * k) * SR)
            buf[a:a + len(h)] += h[: n - a]
        buf = fftconvolve(buf, room_ir(rng, rt60))[:n]
        noise = pink_noise(rng, n)
        noise *= np.sqrt(np.mean(buf ** 2)) / (np.sqrt(np.mean(noise ** 2)) * 10 ** (snr_db / 20))
        out[role] = (buf + noise).astype(np.float32)
    return out


if __name__ == "__main__":
    import argparse, soundfile as sf, pathlib
    ap = argparse.ArgumentParser(description="Write one synthetic session as WAV + JSON answer key")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pair", default="medium", choices=list(PAIRS))
    ap.add_argument("--bpm", type=float, default=100)
    ap.add_argument("--snr", type=float, default=20)
    ap.add_argument("--rt60", type=float, default=0.5)
    ap.add_argument("--tempo-change", type=float, default=1.15)
    ap.add_argument("--out", default="session")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    cfg = Session(seed=a.seed, pair=a.pair, bpm=a.bpm, snr_db=a.snr, rt60=a.rt60,
                  tempo_change=a.tempo_change, segments=random_script(rng))
    audio, truth = make_session(cfg)
    p = pathlib.Path(a.out)
    sf.write(p.with_suffix(".wav"), audio, SR)
    p.with_suffix(".json").write_text(json.dumps(truth, indent=1))
    print(f"wrote {p.with_suffix('.wav')} ({truth['duration']:.1f} s, {len(truth['client_hits'])} client hits)")
