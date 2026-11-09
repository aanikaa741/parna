# Beat Sync: measuring how well two players keep time, from one phone recording

Research question: can a single phone recording of a session measure how well a client keeps the beat with a therapist (or a student with a teacher), without electronic instruments?

## Run it

```bash
pip install numpy scipy matplotlib librosa soundfile
cd src
python synth.py --pair medium --out ../session     # one example session: session.wav + session.json (answer key)
python tune.py        # picks detection settings on dev sessions (seeds 100-111)
python evaluate.py    # 81 held-out sessions x 2 methods (~15 min on 4 cores)
python summarize.py   # results/tables.md, results/*.png
```

| file | what it does |
|---|---|
| `src/synth.py` | synthetic sessions: therapist beat (with tempo changes), scripted client behaviour (in sync, late, drifting, catching up, half-time, silent), 3 instrument pairs, room echo, noise. Writes exact hit times as the answer key |
| `src/analyze.py` | two ways to split the players (below) and the sync measures |
| `src/tune.py` | tunes the calibrated method's two settings on dev seeds only |
| `src/evaluate.py`, `src/summarize.py` | the test and its tables/plots |
| `results/report.md` | week 1 findings |

Methods:
- **blind**: librosa's standard onset detector on the mix, then 2-means clustering of each hit's sound. Which cluster is the therapist is chosen using the answer key, so this is a best case for blind.
- **calibrated**: each player taps their instrument alone for ~3 s first. Those taps become spectral templates, and supervised NMF splits the mix into one activation curve per player, so hits that land together stay separate.

## Realistic simulated sessions (v2)

```bash
pip install pyroomacoustics   # plus espeak-ng for the spoken cues (apt install espeak-ng / brew install espeak-ng)
python synth2.py --seed 1 --pair different --out ../sessions_v2/example
python evaluate2.py && python summarize2.py   # 24 sessions, results in results/v2/
```

Each session folder has `middle.wav`, `close_therapist.wav`, `close_client.wav` (what three phones hear), `truth.json` (every hit, call, voice and distraction) and `script.txt` (a readable timeline you can also act out). Audio is kept for three example sessions; the rest regenerate from their seed. Findings: `results/v2/report.md`.

## Week 2: recording protocol (you + one friend)

1. Two different hand instruments first (e.g. a drum and a shaker, or a drum and a tambourine), then two similar drums.
2. Three phones: one close to each player (the answer key) and one in the middle of the table, ~1 m away (what the tool hears). Start all three, then clap once loudly to line them up.
3. Calibration: each player taps alone 5 times.
4. "Therapist" plays a steady beat from a metronome in one earbud (70, 100 or 140 bpm). "Client" follows this script, ~16 beats each: in time, deliberately late, slowly drifting, stop for 8 beats, come back in late and catch up, play every other beat. Halfway through, the therapist changes tempo.
5. The simulated `script.txt` files are ready-made scripts: act one out instead of improvising.
6. Answer key: the close phones get you most hits automatically (`answer_key_from_close_phones` in `analyze.py`) but miss about 30% of soft hits in simulation, so check and fix the labels by hand in Sonic Visualiser or an Audacity label track (my guess: roughly 10 minutes per 2-minute session).
7. Record ~20 sessions: different rooms (small room, big room), with and without background noise (fan, music at low volume).

## Final test (60 held-out sessions)

```bash
python evaluate_final.py && python summarize_final.py   # results/final/tables.md, fig_f1.png, fig_timing.png
```

Report: "Who kept the beat?" (Claude Doc) https://claude.ai/artifact/4rSXmgwCubZ12g2utp7uf3

Demo page (one session, both methods, plays the audio): https://claude.ai/artifact/RGQogFJJnd6BMRYcvAnrVc. Source in `demo/`.
