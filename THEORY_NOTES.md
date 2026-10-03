# Theory Notes — Motivation Only

These points motivate experiments; they are not experimental findings or a design specification.

1. Exact recurrent credit information can be much larger than the ordinary hidden state.
2. Under finite error, required credit memory depends strongly on the dynamics.
3. Near-critical recurrent dynamics can preserve difficult long-term credit.
4. Stronger contraction can make old credit fade and become compressible.
5. Some theory examples have robust credit-memory dimension growing faster than linearly.
6. One frozen model family has a regime where fixed absolute input energy makes old credit asymptotically negligible.
7. This suggests testing whether a model can allocate extra credit memory only when sequence dynamics appear to require it.

## What the theory does NOT tell us

It does not tell us the best architecture, prove Adaptive Credit Memory works, or establish practical memory savings. Some asymptotic results may begin at unrealistic widths. Experiments must test practical behavior separately, and experimental outcomes do not prove the theory.