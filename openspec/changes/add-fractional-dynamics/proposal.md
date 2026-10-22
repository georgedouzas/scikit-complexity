# Proposal

## Why

Fractional-order differential equations model memory and anomalous relaxation. Trzaska (2023 §2.3.6) uses them for
viscoelastic and electrochemical oscillators. Their solutions relax like Mittag-Leffler functions instead of
exponentials, which also describes the anomalous diffusion and long memory that Argun et al. (2021 ch. 6) and Jensen
(2023 §10.4) measure. A fractional derivative of order α depends on the whole history of the state, so it needs its own
solvers.

**Rebased on `redesign-system-model`.** The redesign's inventory gate resolved Caputo derivatives as a non-integer
`order` of a state, which rate laws target unchanged, with solvers declaring support for fractional orders. This change
therefore adds no fractional law kind: it accepts non-integer orders on states, adds the order support to the
declarations and validation, and adds the solvers, the Mittag-Leffler function, and the models.

## What Changes

- A state SHALL accept a non-integer order `0 < α < 1`, `state(order=0.6)`, in `skcomplex.core.components`. A `Rate`
  law that targets the state then declares that its Caputo derivative of order `α` equals the law's expression. An
  order of 1 is the ordinary rate law, so no new law kind is needed.
- The declaration vocabulary of `skcomplex.core.base` gains the order terms `'integer'` and `'fractional'`, and
  `Supports` gains an `orders` field. Validation reports a fractional state under a time that is not continuous, an
  order outside `(0, 1)`, and a solver or backend that does not support fractional orders.
- Add fractional solvers to `skcomplex.core.solvers`: Grünwald-Letnikov and the L1 scheme with the full memory, a
  short-memory variant with a declared window, and the fractional Adams-Bashforth-Moulton predictor-corrector. Each
  declares its support, including fractional orders.
- Add the Mittag-Leffler function to `skcomplex.core.analysis`, with an exact solution of linear fractional relaxation.
- Add library models: fractional relaxation, fractional damped oscillator, fractional Van der Pol, and fractional
  Lorenz.

## Capabilities

### New Capabilities

- `fractional-dynamics`: fractional orders of states, their solvers and validation, and the Mittag-Leffler function.

### Modified Capabilities

- `model-library`: fractional models.

## Impact

- `skcomplex.core.components`: `state(order=...)` accepts non-integer orders.
- `skcomplex.core.base`: the order terms and the `orders` field of `Supports`; every existing solver and backend
  declares integer orders.
- `skcomplex.core.system`: the validation issues for fractional orders.
- `skcomplex.core.solvers` gains `GrunwaldLetnikov`, `L1Scheme`, and `FractionalABM`, and `skcomplex.core.backends`
  gains their Numba kernels.
- `skcomplex.core.analysis` gains `mittag_leffler`.
- The compiler lowers fractional states to an internal fractional form that co-simulates with the other forms.
- Reuses the history buffer of delayed states.
