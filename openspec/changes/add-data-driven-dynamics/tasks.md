# Tasks

## 1. Surrogates

- [ ] 1.1 Survey PySINDy, PyDMD, PyKoopman, and ReservoirPy, record the result, and add the chosen ones as runtime
  dependencies
- [ ] 1.2 Add `fit_dmd` with its element in a continuous-time system, and verify the linear system scenario and that
  the surrogate passes `validate()`
- [ ] 1.3 Add `fit_sindy` with symbolic laws, and verify the Lorenz scenario
- [ ] 1.4 Add `fit_reservoir` with its element in a discrete-time system, and verify the chaotic forecast scenario

## 2. Library

- [ ] 2.1 Add `echo_state_network` with a curated gallery example and review row, verify the echo state property
  scenario, and mark its inventory entry implemented

## 3. Gate

- [ ] 3.1 Document data-driven dynamics in the user guide, and run formatting, checks, the docs build, and tests through
  nox
