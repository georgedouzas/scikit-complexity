# Tasks

## 1. Orders, declarations, and solvers

- [ ] 1.1 Accept non-integer orders in `state(order=...)` and lower fractional states to the internal fractional form
  with co-simulation, and verify the order-one and rate-law scenarios
- [ ] 1.2 Add `OrderKind` and the `orders` field of `Supports`, declare integer orders for every existing solver and
  backend, add the `fractional-order`, `fractional-time`, and `order-unsupported` issues, and verify the validation
  scenarios and the declaration test
- [ ] 1.3 Add the Grünwald-Letnikov, L1, and predictor-corrector schemes with windowed memory on NumPy and Numba, with
  their support declarations, and verify the Mittag-Leffler and short memory scenarios

## 2. Analysis and library

- [ ] 2.1 Add `mittag_leffler` to `skcomplex.core.analysis` and the exact relaxation solution, and verify the
  exponential case scenario
- [ ] 2.2 Add the fractional models as registered factories, and verify the slower-than-exponential scenario and the
  conformance suite

## 3. Documentation and gate

- [ ] 3.1 Document fractional orders, schemes, memory, and their validation in the user guide, and verify the
  documentation test passes
- [ ] 3.2 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
