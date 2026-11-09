# Realistic simulated sessions (v2)

24 simulated sessions, about 2 minutes each, 8 per instrument pairing:
- different (drum + shaker or tambourine)
- related (two drums of different pitch)
- same (two djembes or two frame drums)

Each session was simulated in a modelled room (pyroomacoustics) and recorded by three phones: one close to each player, one in the middle of the table.

Each session contains:
- calibration taps
- steady-beat exercises with scripted client behaviour
- call and response
- a tempo change
- singing together
- therapist speech cues (espeak-ng)
- room distractions (chair, door, cough, paper)
- phone-microphone character: weak bass, compression, hum

Detection settings come from week 1 and were **not** retuned. Full tables are in `tables.md`. The answer-key test is in `answer_key_test.txt`.

## Findings

1. **The week 1 result holds in a realistic room, and gets a little better.** Calibrated, with calibration taps cut from the same recording:
   - Different instruments: therapist/client hit F1 0.97/0.98; exercise timing within a median 5 ms; in-sync rate within 0 points.
   - Two related drums: F1 0.84/0.96; timing within 30 ms.
   - Blind detection still cannot measure timing: errors of 102 to 244 ms, and no correlation with the truth.
2. **Speech and singing barely fool the calibrated method,** at 0.1 to 1.1 false client hits per session against 8 to 28 for blind. The exception is two identical drums (10 per session).
3. **Call and response works.** For different and related instruments it detected every answered call (99 of 99) and 12 of 13 silent ones. It timed the start of the response within a median 9 to 10 ms. It counted the exact number of echoed hits in 90% of calls; that drops to 45% for identical drums.
4. **Two identical drums is still the open problem** (client F1 0.59, in-sync rate off by 14 points).
5. **Close phones are not a ready-made answer key.** Plain onset detection on a close phone also catches:
   - the other player's hits
   - speech
   - double triggers on ringing drums

   That gives F1 of 0.53 to 0.65. With a tuned detector (settings picked on 9 sessions, tested on the other 15), it reaches precision 0.95 but recall 0.69: soft client hits get missed. Moving the phone from 30 cm to 5 cm barely helped in the simulation. **For week 2, the real recordings need a human pass to add missed hits** (for example in Sonic Visualiser or Audacity label tracks), with the close phones as the starting point.

## Caveats

- Still simulation. The instruments are synthesised, and real players vary more.
- The middle phone in the simulation sits near both players; a real phone across the room will do worse.
- Blind detection is still given the correct cluster labels (a best case).
- 8 sessions per pairing is small; treat differences under ~0.05 F1 as noise.
