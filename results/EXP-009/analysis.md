# EXP-009 — Can bias adaptation alone enable additive success?

**Status: interrupted after a complete delay-64 cohort. EXPERIMENTAL RESULT for delay 64 only.**
Delay 128 is not a complete cohort. No bit-flip audit was run. These runs are not added to earlier seed counts.

The process used the preregistered code `573bd668d03ead59d0c7723ed1902e893faecec1`, Python 3.11.9, and PyTorch 2.13.0+cpu.
See `interruption_note.md`.

Regime C trained only `b_x` (`recurrent.bias_ih_l0`) plus the readout. Both weight matrices and `b_h` stayed frozen.
Regime D trained both weight matrices plus the readout. Both preactivation biases stayed frozen.
`b_x` and `b_h` still enter the same preactivation. The readout bias is a separate parameter.

## Delay 64

Success means final test accuracy of at least 0.95.

| Seed | Full | Readout only | Bias + readout | Matrices + readout |
|---|---:|---:|---:|---:|
| 101 | 1.000 | 0.871 | 0.979 | 1.000 |
| 113 | 1.000 | 0.854 | 0.963 | 1.000 |
| 127 | 1.000 | 0.871 | 0.990 | 1.000 |
| 139 | 1.000 | 0.926 | 0.971 | 1.000 |
| 151 | 1.000 | 0.896 | 0.945 | 1.000 |
| 163 | 1.000 | 0.865 | 0.977 | 1.000 |
| 179 | 1.000 | 0.873 | 0.969 | 1.000 |
| 191 | 1.000 | 0.801 | 0.846 | 0.998 |
| 211 | 1.000 | 0.885 | 0.994 | 1.000 |
| 223 | 1.000 | 0.832 | 0.977 | 1.000 |

Success counts: full 10/10, readout only 0/10, bias plus readout 8/10, matrices plus readout 10/10.

The bias-plus-readout misses are seed 151 at 0.945 and seed 191 at 0.846.
On the eight successes, both matrices stayed at initialization.
On all ten matrix-plus-readout runs, both preactivation biases stayed at initialization.

## What the saved checks show

Frozen-parameter hashes matched before and after every saved run.
Readout-only hidden states on a fixed batch stayed unchanged.
For bias plus readout, the saved matrix hashes did not change.
The recorded effective-bias change was the same size as the `b_x` change, about 0.24 to 0.85.
A strict bitwise comparison of the summed bias delta with the `b_x` delta was false, with a largest absolute coordinate gap of about 1.5e-8.
That gap is float rounding in the sum. `b_h` itself had change norm 0 on those runs.
A large bias change is not, by itself, the mechanism claim. The training rule is the evidence.

Retention of the original bit was high for full training and for matrix-plus-readout training, about 0.93 to 1.00.
Bias-plus-readout retention was lower, about 0.52 to 0.87.
The two bias misses still had probe R^2 of about 0.69 and 0.52, so they were not empty of bit information.
Readout-only retention stayed about 0.35 to 0.65, in the same range seen before any recurrent learning in earlier additive runs.

Gradient clipping was much more common when the trainable set was small.
Readout-only and bias-plus-readout runs clipped on roughly 70 to 97 percent of updates.
Full and matrix-plus-readout runs clipped on roughly 7 to 22 percent.
Clipping uses only the trainable set, so these regimes are not isolated from that optimizer difference.

## Delay 128

Not complete. Only seed 101 full (accuracy 1.000) and seed 101 readout only (accuracy 0.801) were saved.
The next run, seed 101 bias plus readout, was started and not saved.
The other 37 delay-128 runs were not started.
No delay-128 conclusion is drawn.

## Interpretation

The fresh controls matched the earlier pattern. Full training solved every delay-64 seed. Readout-only training solved none, and its accuracies line up with EXP-008. That is not outcome E.

Matrix learning plus the readout was enough on every delay-64 seed while both preactivation biases stayed fixed.
One learned preactivation bias plus the readout was enough on 8 of 10 seeds while both matrices stayed fixed, and not enough on the other two under this 400-update schedule.
This does not prove that another budget could never close those two misses.
It does not prove that EXP-008 used the bias-only route. EXP-008 trained a matrix together with a bias.
The matrix-only result shows that bias adaptation is not required for success on this cohort.
The two bias misses show that bias adaptation alone was not reliable on every seed.

This is not a new architecture and it does not validate the broader theory.
The bit-flip audit was not run, so these scores do not include that check.
