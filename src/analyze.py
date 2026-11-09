"""Find each player's hits in one mixed recording, then measure how well the client keeps the beat.

Two ways to split the players:
- "blind": find hits in the mix, then sort them into two groups by sound (spectral shape).
  Cannot see two hits that land at the same moment, which is exactly when the client is in sync.
- "calibrated": each player taps their instrument alone for a few seconds first. Those taps give
  a sound template per instrument, and the mix is decomposed into "how much of each template is
  playing right now" (supervised NMF). Overlapping hits stay separate.
"""
import numpy as np
from scipy.signal import stft, find_peaks
from scipy.ndimage import median_filter

SR = 22050
NFFT, HOP = 1024, 128
FPS = SR / HOP


def spec(x):
    _, _, Z = stft(x, fs=SR, nperseg=NFFT, noverlap=NFFT - HOP, boundary=None, padded=False)
    return np.abs(Z) + 1e-9


def frames_to_time(i):
    return (np.asarray(i) * HOP + NFFT / 2) / SR


def pick_onsets(novelty, delta=0.1, min_gap_s=0.07, med_s=0.5):
    nov = novelty / (np.max(novelty) + 1e-9)
    thr = median_filter(nov, size=max(3, int(med_s * FPS))) + delta
    peaks, _ = find_peaks(nov, height=thr, distance=max(1, int(min_gap_s * FPS)))
    return peaks


def flux(S):
    L = np.log1p(100 * S)
    d = np.diff(L, axis=1, prepend=L[:, :1])
    return np.maximum(d, 0).sum(axis=0)


# ----------------------------------------------------------------------------- blind

def blind_split(x):
    """Standard onset detector (librosa) on the mix, then 2-means on each hit's sound."""
    import librosa
    S = spec(x)
    on_t = librosa.onset.onset_detect(y=x.astype(float), sr=SR, hop_length=HOP, units="time")
    on = np.clip(np.round((on_t * SR - NFFT / 2) / HOP).astype(int), 0, S.shape[1] - 1)
    freqs = np.linspace(0, SR / 2, S.shape[0])
    feats = []
    for f in on:
        seg = S[:, f:f + 4].mean(axis=1)
        p = seg / seg.sum()
        centroid = (freqs * p).sum()
        low = p[freqs < 300].sum()
        feats.append([np.log(centroid), low])
    F = np.array(feats)
    if len(F) < 4:
        return on_t, np.zeros(len(on), int)
    F = (F - F.mean(0)) / (F.std(0) + 1e-9)
    # 2-means, started from the extremes of the first feature
    c = np.stack([F[F[:, 0].argmin()], F[F[:, 0].argmax()]])
    for _ in range(50):
        lab = ((F[:, None, :] - c[None]) ** 2).sum(-1).argmin(1)
        c_new = np.stack([F[lab == k].mean(0) if np.any(lab == k) else c[k] for k in range(2)])
        if np.allclose(c_new, c):
            break
        c = c_new
    return on_t, lab


# ----------------------------------------------------------------------------- calibrated

def nmf(V, W=None, k=4, n_iter=150, fixed=0, rng=None):
    """KL-NMF. The first `fixed` columns of W stay as given; the rest are learned."""
    rng = rng or np.random.default_rng(0)
    F, T = V.shape
    if W is None:
        W = rng.random((F, k)) + 0.1
    H = rng.random((W.shape[1], T)) + 0.1
    for _ in range(n_iter):
        WH = W @ H + 1e-9
        H *= (W.T @ (V / WH)) / (W.sum(0)[:, None] + 1e-9)
        if fixed < W.shape[1]:
            WH = W @ H + 1e-9
            Wf = W[:, fixed:]
            Wf *= ((V / WH) @ H[fixed:].T) / (H[fixed:].sum(1)[None] + 1e-9)
            W[:, fixed:] = Wf / (Wf.sum(0, keepdims=True) + 1e-9)
    return W, H


def templates(clip, k=2):
    W, _ = nmf(spec(clip), k=k, n_iter=200)
    return W / W.sum(0, keepdims=True)


def pick_relative(nov, alpha=0.2, min_gap_s=0.12):
    """Peaks of the novelty curve that reach `alpha` x a robust maximum (95th pct of peak heights)."""
    peaks, props = find_peaks(nov, height=0, distance=max(1, int(min_gap_s * FPS)))
    if len(peaks) == 0:
        return peaks
    h = props["peak_heights"]
    return peaks[h >= alpha * np.percentile(h, 95)]


def activations(x, calib, k_per=2, k_free=3):
    """Supervised NMF: per-player activation curves (how much of each instrument sounds per frame)."""
    V = spec(x)
    Wt, Wc = templates(calib["therapist"], k_per), templates(calib["client"], k_per)
    rng = np.random.default_rng(1)
    Wfree = rng.random((V.shape[0], k_free)) + 0.1
    W = np.concatenate([Wt, Wc, Wfree / Wfree.sum(0, keepdims=True)], axis=1)
    W, H = nmf(V, W=W, fixed=2 * k_per, n_iter=150)
    return {"therapist": H[:k_per].sum(0), "client": H[k_per:2 * k_per].sum(0)}


def onsets_from_activation(act, alpha=0.4, lag=3):
    a = np.log1p(50 * act / (act.max() + 1e-9))
    a = np.convolve(a, np.ones(3) / 3, mode="same")
    nov = np.maximum(a - np.concatenate([np.full(lag, a[0]), a[:-lag]]), 0)
    return frames_to_time(pick_relative(nov, alpha))


def calibrated_split(x, calib, alpha=0.4, lag=3):
    acts = activations(x, calib)
    return {r: onsets_from_activation(a, alpha, lag) for r, a in acts.items()}


# ----------------------------------------------------------------------------- measures

def sync_measures(therapist, client, window=0.5):
    """Client timing relative to the therapist's hits.

    window: max distance to the nearest therapist hit, as a fraction of the local beat period.
    """
    therapist, client = np.sort(therapist), np.sort(client)
    if len(therapist) < 3 or len(client) == 0:
        return {"n_client": int(len(client)), "mean_async_ms": np.nan, "sd_async_ms": np.nan, "in_sync_rate": np.nan, "asyncs": []}
    ioi = np.diff(therapist)
    asyncs = []
    for c in client:
        j = np.searchsorted(therapist, c)
        cands = [k for k in (j - 1, j) if 0 <= k < len(therapist)]
        k = min(cands, key=lambda k: abs(therapist[k] - c))
        period = ioi[min(k, len(ioi) - 1)]
        d = c - therapist[k]
        if abs(d) <= window * period:
            asyncs.append(d)
    a = np.array(asyncs) * 1000
    return {
        "n_client": int(len(client)),
        "mean_async_ms": float(np.mean(a)) if len(a) else np.nan,
        "sd_async_ms": float(np.std(a)) if len(a) else np.nan,
        "in_sync_rate": float(np.mean(np.abs(a) <= 50)) if len(a) else np.nan,  # share of hits within 50 ms
        "asyncs": a.tolist(),
    }


def match_f1(est, ref, tol=0.05):
    est, ref = np.sort(est), np.sort(ref)
    used = np.zeros(len(est), bool)
    tp = 0
    for r in ref:
        if len(est) == 0:
            break
        d = np.abs(est - r)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol:
            used[j] = True
            tp += 1
    p = tp / len(est) if len(est) else 0.0
    rc = tp / len(ref) if len(ref) else 0.0
    return {"precision": p, "recall": rc, "f1": 2 * p * rc / (p + rc) if p + rc else 0.0}


def analyze(x, method="calibrated", calib=None, truth=None, alpha=0.4, lag=3):
    """Returns {'therapist': times, 'client': times}. For 'blind', cluster identity is unknown, so
    when `truth` is given the better of the two labellings is used (a best case for blind)."""
    if method == "calibrated":
        return calibrated_split(x, calib, alpha=alpha, lag=lag)
    times, lab = blind_split(x)
    a = {"therapist": times[lab == 0], "client": times[lab == 1]}
    b = {"therapist": times[lab == 1], "client": times[lab == 0]}
    if truth is None:
        return a
    score = lambda s: match_f1(s["therapist"], truth["therapist_hits"])["f1"] + match_f1(s["client"], truth["client_hits"])["f1"]
    return a if score(a) >= score(b) else b


# ----------------------------------------------------------------------------- answer key from close phones

def answer_key_from_close_phones(y_t, y_c, margin_db=0.0, win=0.04, wait_frames=1, delta=0.3):
    """Hit times per player from the two close phones.

    Plain onset detection on one close phone also picks up the other player (precision ~0.4 in the v2
    simulation). So each onset is kept only if it is louder, by `margin_db`, on that player's own phone
    than on the other one, measured over the first `win` seconds after the onset.
    """
    import librosa
    out = {}
    pairs = (("therapist", y_t, y_c), ("client", y_c, y_t))
    for role, own, other in pairs:
        on = librosa.onset.onset_detect(y=own.astype(float), sr=SR, hop_length=128, units="time", wait=wait_frames, delta=delta)
        keep = []
        for t in on:
            a, b = int(max(0, t - 0.01) * SR), int((t + win) * SR)
            e_own = np.sqrt(np.mean(own[a:b] ** 2)) + 1e-9
            e_oth = np.sqrt(np.mean(other[a:b] ** 2)) + 1e-9
            if 20 * np.log10(e_own / e_oth) >= margin_db:
                keep.append(t)
        out[role] = np.array(keep)
    return out


# ----------------------------------------------------------------------------- two-microphone phone

def gcc_phat(a, b, interp=16, max_lag_s=0.0006, band=(300, 7000)):
    """Generalised cross-correlation with phase transform, limited to `band` Hz (outside it the phone
    signal is mostly noise and hum, which whitening would otherwise amplify into a false zero-lag peak).
    Returns (lags in s, correlation)."""
    n = len(a) + len(b)
    nfft = 1 << (n - 1).bit_length()
    A, B = np.fft.rfft(a, nfft), np.fft.rfft(b, nfft)
    R = A * np.conj(B)
    R /= np.abs(R) + 1e-12
    f = np.fft.rfftfreq(nfft, 1 / SR)
    R[(f < band[0]) | (f > band[1])] = 0
    cc = np.fft.irfft(R, nfft * interp)
    m = int(max_lag_s * SR * interp)
    cc = np.concatenate([cc[-m:], cc[:m + 1]])
    lags = np.arange(-m, m + 1) / (SR * interp)
    return lags, cc


def hit_tdoa_profile(left, right, t, pre=0.004, post=0.03):
    a, b = int((t - pre) * SR), int((t + post) * SR)
    if a < 0 or b > len(left):
        return None
    return gcc_phat(left[a:b], right[a:b])


def stereo_split(left, right, calib, cal_windows, alpha=0.4, lag=3, theta=0.6, tol_s=0.00008, acts=None):
    """Calibrated NMF finds hits (either player); the two phone microphones decide whose hit it is.

    Each player's direction is learned from their calibration taps as a time difference between the
    phone's two microphones. A hit is credited to every player whose time difference shows a peak of at
    least `theta` x the strongest peak, so two hits at the same moment can both be kept.
    Returns ({'therapist': times, 'client': times}, info).
    """
    acts = acts if acts is not None else activations(left, calib)
    total = onsets_from_activation(acts["therapist"] + acts["client"], alpha, lag)
    ref = {}
    for role, (a, b) in cal_windows.items():
        ts = [t for t in total if a <= t <= b]
        prof = [hit_tdoa_profile(left, right, t) for t in ts]
        prof = [p for p in prof if p is not None]
        if not prof:
            return {"therapist": np.array([]), "client": np.array([])}, {"separable": False}
        lags = prof[0][0]
        ref[role] = float(np.median([lags[np.argmax(cc)] for _, cc in prof]))
    sep = abs(ref["therapist"] - ref["client"])
    info = {"tdoa_therapist_ms": ref["therapist"] * 1000, "tdoa_client_ms": ref["client"] * 1000, "separation_ms": sep * 1000}
    out = {"therapist": [], "client": []}
    for t in total:
        p = hit_tdoa_profile(left, right, t)
        if p is None:
            continue
        lags, cc = p
        top = cc.max()
        for role in ("therapist", "client"):
            near = np.abs(lags - ref[role]) <= tol_s
            if cc[near].max() >= theta * top:
                out[role].append(t)
    info["separable"] = sep > 2 * tol_s
    return {k: np.array(v) for k, v in out.items()}, info
