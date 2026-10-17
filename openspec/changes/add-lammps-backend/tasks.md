# Tasks

## 1. Setup

- [ ] 1.1 Move `lammps` from the extra to the dependencies, add `mpich` for non-Windows platforms, relock, and verify
  the LAMMPS-loads test passes in a fresh environment
- [ ] 1.2 Add a Windows CI job that imports LAMMPS, and verify it passes or record the platform limitation in the
  Project Profile

## 2. Backend contract

- [ ] 2.1 Add `skcomplex.backends` with `ExternalBackend`, the `check` step in `Simulation`, and the named unsupported
  error, and verify the Lotka-Volterra rejection test passes

## 3. LAMMPS backend

- [ ] 3.1 Translate particle groups, masses, spaces, and boxes, and verify a free ideal gas matches the native run
- [ ] 3.2 Add the Lepton printer and pair interaction translation with native styles for known potentials, and verify
  the custom pair potential force test passes
- [ ] 3.3 Translate constant forces, drag, thermostats, and `fix external` forces, and verify the short-horizon
  agreement test passes
- [ ] 3.4 Reissue commands on parameter changes, and verify the thermostat temperature change test passes

## 4. Model library

- [ ] 4.1 Add `LennardJones`, `SoftSphere`, and `LangevinThermostat`, and verify the minimum and equilibrium
  temperature tests pass on both solvers

## 5. Performance and documentation

- [ ] 5.1 Add the real-time 10000-atom benchmark test, and verify it passes
- [ ] 5.2 Document external backends and the LAMMPS translation table in the user guide, update the Project Profile,
  and verify the documentation test passes
- [ ] 5.3 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass

## Workflow follow-up

- Archive the change after review.
