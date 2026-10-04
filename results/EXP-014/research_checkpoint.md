# Research checkpoint before EXP-014

This note records the interpretation reached from EXP-010 through EXP-013.
It was written before any EXP-014 result. Earlier saved analyses were not rewritten.

EXP-010 trained the additive model either through the whole sequence or with the training gradient cut off at the last 16 steps. Full-history training succeeded on all 10 seeds at delays 64 and 128. Final-16 training succeeded on 1 of 10 at delay 64 and on none at delay 128, using the original readout. The early training gradient was blocked. The forward pass could still change with the original bit.

EXP-011 crossed that cutoff with clipping turned on or off. The clipped runs matched EXP-010. At delay 64 the full-history advantage remained without clipping. At delay 128, turning clipping off made full-history training less consistent and still did not make final-16 training reliable. The higher clipping rate under the cutoff was a consequence of larger gradients, not evidence that EXP-010 was invalid.

EXP-012 fit a new linear classifier to the frozen final states. It rescued 36 of 37 unsuccessful final-16 models. One stayed below 0.95. Full-history models stayed strong. Many misses were a failure to use information that was already linearly readable, not proof that the state was empty.

EXP-013 applied that same classifier to never-trained additive states. They reached 0.95 on 6 of 10 seeds at delay 64 and 5 of 10 at delay 128. Full-history and final-16 states reached it on all 10 seeds. The earlier readout-only failures were specific to that readout-training schedule. They did not show that an untrained state has no readable answer. Final-16 learning still improved the weaker untrained states.

EXP-014 asks whether this pattern appears again on ten new seeds, new datasets, and delay 128 only. The training rule and the classifier stay fixed.
