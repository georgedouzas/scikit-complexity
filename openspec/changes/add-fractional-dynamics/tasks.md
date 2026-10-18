# Tasks

## 1. Laws and solvers

- [ ] 1.1 Add `FractionalRate` and the fractional form with co-simulation, and verify the order-one scenario
- [ ] 1.2 Add the Grünwald-Letnikov, L1, and predictor-corrector schemes with windowed memory on NumPy and Numba, and
  verify the Mittag-Leffler and short memory scenarios

## 2. Analysis and library

- [ ] 2.1 Add `mittag_leffler` and the exact relaxation solution, and verify the exponential case scenario
- [ ] 2.2 Add the fractional models as registered factories, and verify the slower-than-exponential scenario and the
  conformance suite

## 3. Documentation and gate

- [ ] 3.1 Document fractional laws, schemes, and memory in the user guide, and verify the documentation test passes
- [ ] 3.2 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
