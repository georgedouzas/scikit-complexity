# Tasks

## 1. Core mobility interaction

- [x] 1.1 Add `MobilityInteraction` in `src/skcomplex/core/components/interactions/_mobility.py` with the Oseen and Rotne-Prager-Yamakawa tensors, rotlet couplings, stresslet and dipole flows, pair potential forces, rate and map forms, and Cholesky and Lanczos noise, exported from the package (verify: doctest of the module passes)
- [x] 1.2 Add `tests/core/components/interactions/test_mobility.py` covering Stokes drag, the Oseen and RPY pair couplings, rotlet couplings, the stresslet flow, orientation rotation, equilibrium variance with both square roots, the Oseen overlap error, and the variant checks (verify: pytest passes)
- [x] 1.3 Make the Jacobian pattern of a vectorized call dense over the members in `_jacobian_kernel.py`, with a test in `tests/core/engine/problems/test_law_kernel_functions.py` (verify: the test and the catalog Jacobian check of `squirmers` pass)

## 2. Catalog systems

- [x] 2.1 Add `squirmers` in `src/skcomplex/systems/physics/active_matter/_squirmers.py` with its element and `SquirmerInteraction`, registered and exported (verify: `create_system('squirmers')` runs)
- [x] 2.2 Add `hydrodynamically_coupled_traps` in `src/skcomplex/systems/physics/soft_matter/_coupled_traps.py`, registered and exported (verify: `create_system('hydrodynamically_coupled_traps')` runs)
- [x] 2.3 Add tests in `tests/systems/physics/active_matter/test_squirmers.py` and `tests/systems/physics/soft_matter/test_coupled_traps.py` (verify: pytest passes)

## 3. Inventory, gallery, and documentation

- [x] 3.1 Add the `hydrodynamic-interactions` primitive and both systems to `docs/overview/inventory.toml` (verify: `PYTHONPATH=docs python docs/inventory_checks.py` passes)
- [x] 3.2 Add gallery examples `docs/examples/physics/active_matter/plot_squirmers.py` and `docs/examples/physics/soft_matter/plot_hydrodynamically_coupled_traps.py` (verify: both run and `python -m gallery_tools coverage --strict` passes)
- [x] 3.3 Document the mobility interaction in the user guide (verify: the code block runs in the docs tests)

## 4. Checks

- [x] 4.1 Run ruff, black, docformatter, mypy, and pydoclint on the changed files, the catalog tests, and the doctests of the changed modules (verify: all pass)
