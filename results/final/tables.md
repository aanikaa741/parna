## 1. Hit detection F1 (50 ms), 20 sessions per pairing, mean [95% CI]

| pairing | method | therapist | client |
|---|---|---|---|
| different | blind | 0.51 [0.44, 0.59] | 0.77 [0.72, 0.81] |
| different | calibrated | 0.94 [0.90, 0.98] | 0.96 [0.94, 0.98] |
| different | stereo | 0.75 [0.65, 0.84] | 0.79 [0.73, 0.84] |
| related | blind | 0.51 [0.43, 0.59] | 0.77 [0.73, 0.81] |
| related | calibrated | 0.87 [0.80, 0.93] | 0.98 [0.96, 0.99] |
| related | stereo | 0.71 [0.62, 0.79] | 0.83 [0.78, 0.88] |
| same | blind | 0.59 [0.53, 0.65] | 0.26 [0.17, 0.35] |
| same | calibrated | 0.75 [0.69, 0.81] | 0.64 [0.57, 0.71] |
| same | stereo | 0.78 [0.71, 0.83] | 0.62 [0.54, 0.69] |

## 2. Exercise timing: median absolute error of the client's mean timing (ms) [95% CI], and in-sync rate error

| pairing | method | timing error (ms) | in-sync rate error (points) |
|---|---|---|---|
| different | blind | 104 [65, 157] | 30 [25, 41] |
| different | calibrated | 4 [3, 12] | 0 [0, 2] |
| different | stereo | 81 [57, 93] | 21 [12, 45] |
| related | blind | 172 [71, 255] | 33 [29, 38] |
| related | calibrated | 3 [1, 32] | 2 [0, 10] |
| related | stereo | 90 [57, 135] | 30 [14, 39] |
| same | blind | 132 [119, 161] | 33 [29, 46] |
| same | calibrated | 43 [20, 52] | 14 [10, 17] |
| same | stereo | 62 [40, 81] | 28 [19, 34] |

## 3. Call and response (client answered?) and response start error

| pairing | method | answers found | silences found | start error (ms, median) | exact hit count |
|---|---|---|---|---|---|
| different | blind | 127/127 | 15/16 | 14 | 86% |
| different | calibrated | 127/127 | 16/16 | 9 | 83% |
| different | stereo | 127/127 | 16/16 | 9 | 81% |
| related | blind | 128/128 | 12/13 | 12 | 88% |
| related | calibrated | 128/128 | 12/13 | 8 | 90% |
| related | stereo | 128/128 | 13/13 | 8 | 90% |
| same | blind | 92/127 | 15/17 | 218 | 17% |
| same | calibrated | 124/127 | 15/17 | 11 | 42% |
| same | stereo | 119/127 | 17/17 | 11 | 39% |

## 4. Tempo following: client lag in beats after a tempo change

| pairing | method | measure exact vs script (from true hits) | exact | within 1 beat |
|---|---|---|---|---|
| different | blind | 75% (n=20) | 0% | 0% |
| different | calibrated | 75% (n=20) | 55% | 75% |
| different | stereo | 75% (n=20) | 0% | 10% |
| related | blind | 65% (n=20) | 5% | 15% |
| related | calibrated | 65% (n=20) | 60% | 75% |
| related | stereo | 65% (n=20) | 0% | 5% |
| same | blind | 85% (n=20) | 0% | 0% |
| same | calibrated | 85% (n=20) | 20% | 35% |
| same | stereo | 85% (n=20) | 0% | 10% |

## 5. Client singing (pitch-based, all pairings)

Sang and detected 42/42; stayed quiet and detected as quiet 0/18. Start error median 821 ms (n=42).

## 6. Calibration length (mono calibrated, client F1)

| pairing | 2 taps | 3 taps | 5 taps |
|---|---|---|---|
| different | 0.95 [0.92, 0.97] | 0.95 [0.92, 0.98] | 0.96 [0.94, 0.98] |
| related | 0.96 [0.93, 0.98] | 0.97 [0.95, 0.99] | 0.98 [0.96, 0.99] |
| same | 0.60 [0.54, 0.66] | 0.61 [0.54, 0.68] | 0.64 [0.57, 0.71] |

## 7. Identical drums: client F1 by phone orientation (0 = pointing at the players, 90 = across)

| orientation | sessions | mono calibrated | stereo |
|---|---|---|---|
| 0 to 30 degrees | 4 | 0.58 | 0.64 |
| 30 to 60 degrees | 7 | 0.55 | 0.56 |
| 60 to 90 degrees | 9 | 0.74 | 0.64 |
