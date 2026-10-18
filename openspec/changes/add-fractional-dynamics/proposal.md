# Proposal

## Why

Fractional-order differential equations model memory and anomalous relaxation. Trzaska (2023 §2.3.6) uses them for
viscoelastic and electrochemical oscillators. Their solutions relax like Mittag-Leffler functions instead of
exponentials, which also describes the anomalous diffusion and long memory that Argun et al. (2021 ch. 6) and Jensen
(2023 §10.4) measure. A fractional derivative of order α depends on the whole history of the state, so it needs its own
law kind and solver.

## What Changes

- Add fractional rate laws: the Caputo derivative of order `0 < α ≤ 1` of a state equals an expression, with one order
  per state. An order of 1 reduces to an ordinary rate law.
- Add fractional solvers: Grünwald-Letnikov and the L1 scheme with the full memory, a short-memory variant with a
  declared window, and the fractional Adams-Bashforth-Moulton predictor-corrector.
- Add the Mittag-Leffler function to the symbolic layer, with an exact solution of linear fractional relaxation.
- Add library models: fractional relaxation, fractional damped oscillator, fractional Van der Pol, and fractional
  Lorenz.

## Capabilities

### New Capabilities

- `fractional-dynamics`: fractional rate laws, solvers, and the Mittag-Leffler function.

### Modified Capabilities

- `model-library`: fractional models.

## Impact

- `skcomplex.model` gains `FractionalRate`, `skcomplex.solvers` gains `GrunwaldLetnikov`, `L1Scheme`, and
  `FractionalABM`, and the analysis layer gains `mittag_leffler`.
- A new form `Form.FRACTIONAL` that co-simulates with the other forms.
- Depends on `add-general-model-interface`, and on `add-hybrid-dynamics` for the history buffer it reuses.
