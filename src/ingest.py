"""Turn three real phone recordings of one session into aligned audio plus draft hit labels.

Put the three recordings in one folder, named middle.*, close_therapist.*, close_client.* (any format
ffmpeg reads: .m4a, .wav, .mp3 ...). Then:

    python ingest.py path/to/session_folder

It writes, into that folder:
  aligned/*.wav                       all three phones on the middle phone's clock
  labels_therapist.txt, labels_client.txt
                                      draft hit labels in Audacity label format, to check by hand
  session.json                        alignment details and a guess of the calibration taps

Alignment uses the loud clap at the start and the one at the end. Two phones' clocks run at slightly
different speeds (tens of parts per million), which over 5 minutes is 10 to 20 ms, as large as the
effect being measured, so the end clap is what corrects it.
"""
import json, pathlib, subprocess, sys, tempfile
import numpy as np
import soundfile as sf
from scipy.signal import correlate, butter, sosfiltfilt
from analyze import answer_key_from_close_phones, SR

PHONES = ("middle", "close_therapist", "close_client")


def load(path):
    path = pathlib.Path(path)
    if path.suffix.lower() in (".wav", ".flac"):
        y, sr = sf.read(path, always_2d=True)
        y = y.mean(1)
        if sr != SR:
            from scipy.signal import resample_poly
            y = resample_poly(y, SR, sr)
        return y.astype(np.float64)
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), f.name], check=True)
        return sf.read(f.name)[0].astype(np.float64)


def find_file(folder, name):
    hits = [p for p in folder.iterdir() if p.stem == name and p.is_file()]
    if not hits:
        sys.exit(f"missing {name}.* in {folder}")
    return hits[0]


def clap_times(y, search_s=20.0):
    """The sharpest isolated broadband transient in the first and in the last `search_s` seconds."""
    hp = sosfiltfilt(butter(4, 1500 / (SR / 2), btype="high", output="sos"), y)
    hop = 64
    e = np.sqrt(np.convolve(hp ** 2, np.ones(hop) / hop, mode="same")[::hop])
    nov = np.maximum(np.diff(e, prepend=e[0]), 0)
    fps = SR / hop
    out = []
    for lo, hi in ((0, search_s), (len(y) / SR - search_s, len(y) / SR)):
        a, b = int(max(0, lo) * fps), int(min(len(e), hi * fps))
        seg = nov[a:b].copy()
        best = None
        for _ in range(5):  # try the 5 strongest, keep the first that stands alone
            i = int(np.argmax(seg))
            around = np.r_[nov[max(0, a + i - int(fps)):a + i - int(0.15 * fps)], nov[a + i + int(0.15 * fps):a + i + int(fps)]]
            if around.size == 0 or around.max() < 0.35 * seg[i]:
                best = (a + i) / fps; break
            seg[max(0, i - int(0.2 * fps)):i + int(0.2 * fps)] = 0
        out.append(best)
    if out[0] is not None and out[1] is not None and out[1] - out[0] < 10:
        out[1] = None  # the same clap twice: recording too short or no end clap
    return out


def refine(ref, y, t_ref, t_y, win=0.15):
    """Sub-frame lag between two phones around a clap, by cross-correlating the high-passed waveform."""
    sos = butter(4, 1000 / (SR / 2), btype="high", output="sos")
    a = sosfiltfilt(sos, ref[int((t_ref - win) * SR):int((t_ref + win) * SR)])
    b = sosfiltfilt(sos, y[int((t_y - win) * SR):int((t_y + win) * SR)])
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    c = correlate(a, b, mode="full")
    lag = int(np.argmax(np.abs(c))) - (n - 1)
    peak = float(np.max(np.abs(c)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return t_y - lag / SR, peak  # where ref's clap lands in y's samples, i.e. y time of the clap


def align(ref, y):
    """Fit t_y = scale * t_ref + offset from the claps; resample y onto ref's clock."""
    cr, cy = clap_times(ref), clap_times(y)
    pts, quality = [], []
    for r, q in zip(cr, cy):
        if r is not None and q is not None:
            t_y, peak = refine(ref, y, r, q)
            pts.append((r, t_y)); quality.append(peak)
    if not pts:
        raise RuntimeError("no clap found in both recordings")
    if len(pts) == 2:
        (r0, y0), (r1, y1) = pts
        scale = (y1 - y0) / (r1 - r0)
    else:
        scale = 1.0
    offset = pts[0][1] - scale * pts[0][0]
    t_ref = np.arange(len(ref)) / SR
    aligned = np.interp(t_ref * scale + offset, np.arange(len(y)) / SR, y, left=0.0, right=0.0)
    return aligned, {"offset_s": float(offset), "drift_ppm": float((scale - 1) * 1e6), "claps_used": len(pts),
                     "clap_match": [round(q, 3) for q in quality]}


def write_labels(path, times, name):
    with open(path, "w") as f:
        for t in times:
            f.write(f"{t:.4f}\t{t:.4f}\t{name}\n")


def read_labels(path):
    return np.array(sorted(float(l.split("\t")[0]) for l in open(path) if l.strip()))


def guess_calibration(T, C, n=5):
    """First run of n therapist hits with no client hit, then the same for the client."""
    out = {}
    for role, own, other in (("therapist", T, C), ("client", C, T)):
        for i in range(len(own) - n + 1):
            a, b = own[i] - 0.2, own[i + n - 1] + 0.5
            if not np.any((other > a) & (other < b)) and own[i + n - 1] - own[i] < 6:
                out[role] = [round(float(a), 3), round(float(b), 3)]; break
    return out


def ingest(folder):
    folder = pathlib.Path(folder)
    y = {p: load(find_file(folder, p)) for p in PHONES}
    (folder / "aligned").mkdir(exist_ok=True)
    info = {"alignment": {}}
    sf.write(folder / "aligned" / "middle.wav", y["middle"].astype(np.float32), SR)
    for p in PHONES[1:]:
        y[p], info["alignment"][p] = align(y["middle"], y[p])
        sf.write(folder / "aligned" / f"{p}.wav", y[p].astype(np.float32), SR)
    key = answer_key_from_close_phones(y["close_therapist"], y["close_client"])
    write_labels(folder / "labels_therapist.txt", key["therapist"], "T")
    write_labels(folder / "labels_client.txt", key["client"], "C")
    info["calibration_guess"] = guess_calibration(key["therapist"], key["client"])
    info["draft_hits"] = {k: int(len(v)) for k, v in key.items()}
    (folder / "session.json").write_text(json.dumps(info, indent=1))
    return info


if __name__ == "__main__":
    for f in sys.argv[1:]:
        print(f, json.dumps(ingest(f), indent=1))
