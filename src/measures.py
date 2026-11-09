"""Measures beyond hit timing: tempo following and client singing.

Both run on one exercise window, which the app knows because it leads the exercise.
"""
import numpy as np


def clean_iois(hits, max_ratio=1.6):
    """Inter-onset intervals, dropping gaps that span a missed hit."""
    d = np.diff(np.sort(hits))
    if len(d) < 3:
        return d
    med = np.median(d)
    return d[(d > med / max_ratio) & (d < med * max_ratio)]


def tempo_follow_lag(therapist, client, max_lag=8):
    """How many beats behind the therapist's tempo change the client's tempo is.

    Compares the two interval series: the lag k that best lines up client interval i with therapist
    interval i - k. Returns (lag in beats, fit error in ms) or (nan, nan) if there is too little data.
    """
    a, b = clean_iois(therapist), clean_iois(client)
    n = min(len(a), len(b))
    if n < max_lag + 6:
        return np.nan, np.nan
    a, b = a[:n], b[:n]
    errs = []
    for k in range(0, max_lag + 1):
        errs.append(np.mean(np.abs(b[k:] - a[:n - k])) * 1000)
    k = int(np.argmin(errs))
    return k, float(errs[k])


def client_singing(y, sr, t0, t1, split_semitones=3.0, min_run_s=0.3):
    """Did the client sing in [t0, t1], and when did they start?

    Tracks the dominant pitch (pYIN). If the voiced frames form two pitch groups at least
    `split_semitones` apart, the higher group is taken as the client. That assumes the client's voice is
    higher than the therapist's (true for children with an adult therapist, not in general).
    Returns dict(sang, start, seconds).
    """
    import librosa
    seg = y[int(t0 * sr):int(t1 * sr)].astype(float)
    hop = 256
    f0, voiced, _ = librosa.pyin(seg, fmin=150, fmax=600, sr=sr, hop_length=hop, frame_length=2048)
    v = voiced & ~np.isnan(f0)
    if v.sum() < 5:
        return {"sang": False, "start": np.nan, "seconds": 0.0}
    st = 12 * np.log2(f0[v] / 440)
    lo, hi = np.percentile(st, 10), np.percentile(st, 90)
    if hi - lo < split_semitones:
        return {"sang": False, "start": np.nan, "seconds": 0.0}  # one voice only: the therapist
    c = np.array([lo, hi])
    for _ in range(20):
        lab = np.abs(st[:, None] - c[None]).argmin(1)
        c = np.array([st[lab == k].mean() if np.any(lab == k) else c[k] for k in range(2)])
    if c[1] - c[0] < split_semitones:
        return {"sang": False, "start": np.nan, "seconds": 0.0}
    # sung notes are sustained; pitched drum strokes (a djembe tone is ~330 Hz) last ~0.1 s.
    # Keep only runs of client-pitch frames lasting at least `min_run_s`.
    hi_frames = np.zeros(len(f0), bool); hi_frames[np.where(v)[0][lab == 1]] = True
    idx, run = [], []
    for i, on in enumerate(np.r_[hi_frames, False]):
        if on:
            run.append(i)
        else:
            if len(run) * hop / sr >= min_run_s:
                idx += run
            run = []
    idx = np.array(idx, int)
    secs = len(idx) * hop / sr
    if secs < min_run_s:
        return {"sang": False, "start": np.nan, "seconds": secs}
    return {"sang": True, "start": t0 + idx[0] * hop / sr, "seconds": secs}
