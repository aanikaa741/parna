"""v3 measures: who is leading, and how engaged the client is. All work on hit times only, so they run
the same on true hits and on hits found by any detection method."""
import numpy as np


def pair_beats(th, cl, window=0.5):
    """Pairs each therapist hit with the nearest client hit within `window` of a beat. Returns runs of
    consecutive pairs as a list of (therapist_times, client_times) arrays."""
    th, cl = np.sort(th), np.sort(cl)
    if len(th) < 4 or len(cl) < 4:
        return []
    P = np.median(np.diff(th))
    runs, cur_t, cur_c, used = [], [], [], set()
    for t in th:
        j = np.searchsorted(cl, t)
        cands = [k for k in (j - 1, j) if 0 <= k < len(cl) and k not in used]
        k = min(cands, key=lambda k: abs(cl[k] - t)) if cands else None
        if k is not None and abs(cl[k] - t) <= window * P:
            used.add(k); cur_t.append(t); cur_c.append(cl[k])
        else:
            if len(cur_t) >= 4:
                runs.append((np.array(cur_t), np.array(cur_c)))
            cur_t, cur_c = [], []
    if len(cur_t) >= 4:
        runs.append((np.array(cur_t), np.array(cur_c)))
    return runs


def correction_gains(th, cl):
    """How strongly each player corrects toward the other (alpha), by least squares on
    IOI(n) = c + b * IOI(n-1) - alpha * (own - other)(n-1). The player who corrects less is leading."""
    runs = pair_beats(th, cl)
    out = {}
    for who in ("therapist", "client"):
        X, y = [], []
        for T, C in runs:
            own, other = (T, C) if who == "therapist" else (C, T)
            for n in range(2, len(own)):
                X.append([1.0, own[n - 1] - own[n - 2], own[n - 1] - other[n - 1]]); y.append(own[n] - own[n - 1])
        if len(y) < 8:
            return None
        coef = np.linalg.lstsq(np.array(X), np.array(y), rcond=None)[0]
        out[who] = float(-coef[2])
    out["n_pairs"] = int(sum(len(T) for T, _ in runs))
    return out


def leader(gains, margin=0.15):
    """'therapist' leads if the client corrects clearly more than the therapist, and the reverse."""
    if gains is None:
        return None
    d = gains["client"] - gains["therapist"]
    return "therapist" if d > margin else "client" if d < -margin else "mutual"


def engagement(th, cl, start, end):
    """Did the client join, after how many beats, and what share of the therapist's beats did they play along to."""
    th = np.sort(np.asarray(th)); cl = np.sort(np.asarray(cl))
    th = th[(th >= start) & (th < end)]; cl = cl[(cl >= start) & (cl < end + 0.3)]
    if len(th) < 4:
        return None
    P = np.median(np.diff(th))
    hit_beat = np.array([np.any(np.abs(cl - t) <= 0.5 * P) for t in th])
    joined = int(hit_beat.sum()) >= 3  # three or more matched beats, so a stray sound doesn't count
    join_beat = int(np.argmax(hit_beat)) if joined else None
    longest, run = 0, 0
    for h in hit_beat:
        run = run + 1 if h else 0; longest = max(longest, run)
    return {"joined": joined, "join_beat": join_beat, "played_share": float(hit_beat.mean()),
            "longest_run": int(longest), "beats": int(len(th))}
