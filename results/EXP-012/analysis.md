# EXP-012 — Frozen-state linear readout rescue

**Status: complete. DIAGNOSTIC RESULT.** No recurrent weights were trained. This is an exploratory check of the 80 models already saved by EXP-011, not a new seed count.

A new linear classifier was fit to each model's final hidden state. It saw only that model's training examples. Features were standardized with the training mean and the training population standard deviation. A constant coordinate would have used scale 1. The classifier was L2 logistic regression, C = 1.0, solver lbfgs, tolerance 1e-8, at most 2000 iterations. In scikit-learn 1.9.1, that L2 setting is `l1_ratio = 0`. scikit-learn was not already installed, so that package was installed. Python stayed 3.11.9 and PyTorch stayed 2.13.0+cpu.

All 80 classifiers converged, in at most 170 iterations. The diagnostic took 15.6 seconds. There were no unstarted conditions. Diagnostic code `bc4fcdd1e130`.

## Did the new readout rescue the final-16 misses?

A miss means the original readout scored below 0.95. There were 37 final-16 misses. The new readout reached at least 0.95 on 36 of them.

The one that stayed below the line was delay 128, seed 179, final-16 training without clipping. Its original accuracy was 0.594. The new readout reached 0.881. On the bit-flip pairs, both answers were correct 83.2 percent of the time, and the logit changed on every pair. So that state still carries partial information. This fixed classifier did not turn it into a reliable answer.

Every other final-16 miss reached at least 0.984 on the ordinary test. On the bit-flip check, both answers were correct on at least 97.7 percent of pairs. The new readout was not fit on those pairs. The hidden states did not change when the readout changed.

## Full-history models

Every full-history model stayed at or above 0.973 with the new readout, including the three delay-128 runs whose original readout had missed 0.95. Those three were also lifted to at least 0.973. The diagnostic did not fail on the models that had already solved the task.

## What this means

This is outcome C, and the rescue is nearly complete.

For 36 of 37 final-16 misses, the learned final state already contained enough linearly readable task information for this fixed classifier. The original jointly trained readout did not fully use it. That narrows the explanation of those misses. It does not remove the fact that full-history training was more reliable when the readout and the recurrent weights were trained together.

One final-16 miss was only partly improved. A miss by this one regularized classifier does not prove that no linear classifier could succeed, and it does not prove the bit was forgotten.

This is not a new architecture and it does not validate the broader theory.

## Files

Verification, metrics, classifiers, and hashes are in this folder. Frozen EXP-001 through EXP-011 files matched after the run. Recurrent parameter hashes were unchanged by feature extraction.
