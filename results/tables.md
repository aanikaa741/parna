## Client hit detection F1 by instrument pair and noise (mean over echo levels and seeds)

| method | pair | SNR 5 dB | SNR 15 dB | SNR 30 dB |
|---|---|---|---|---|
| blind | easy | 0.78 | 0.94 | 0.91 |
| blind | medium | 0.39 | 0.83 | 0.92 |
| blind | hard | 0.12 | 0.05 | 0.29 |
| calibrated | easy | 0.83 | 0.73 | 0.96 |
| calibrated | medium | 0.77 | 0.98 | 0.97 |
| calibrated | hard | 0.49 | 0.65 | 0.66 |

## Therapist and client F1 by echo (RT60), all pairs and noise levels

| method | who | 0.2 s | 0.6 s | 1.2 s |
|---|---|---|---|---|
| blind | therapist | 0.69 | 0.70 | 0.66 |
| blind | client | 0.56 | 0.59 | 0.59 |
| calibrated | therapist | 0.94 | 0.89 | 0.77 |
| calibrated | client | 0.71 | 0.85 | 0.78 |

## Segment-level timing: does the estimate track the truth?

| method | pair | segments | median abs error, mean async (ms) | correlation true vs est async | in-sync rate abs error |
|---|---|---|---|---|---|
| blind | easy | 140 (131 measurable) | 185 | -0.16 | 0.43 |
| blind | medium | 144 (132 measurable) | 143 | -0.11 | 0.40 |
| blind | hard | 141 (104 measurable) | 115 | -0.08 | 0.61 |
| calibrated | easy | 140 (136 measurable) | 7 | 0.84 | 0.04 |
| calibrated | medium | 144 (144 measurable) | 7 | 0.77 | 0.05 |
| calibrated | hard | 141 (141 measurable) | 55 | 0.07 | 0.32 |

## Calibrated method: segment timing error by behaviour

| behaviour | segments | median abs error (ms) |
|---|---|---|
| sync | 165 | 22 |
| late | 70 | 16 |
| drift | 76 | 20 |
| catchup | 71 | 28 |
| half | 43 | 13 |
