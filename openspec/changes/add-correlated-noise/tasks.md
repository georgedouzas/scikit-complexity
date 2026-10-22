# Tasks

## 1. Modelling

- [x] 1.1 Add `Noise.process` in `skcomplex.core.laws` with local labels, label tuples, and `Wiener` handles, and verify
  that a law with no process compiles exactly as before
- [x] 1.2 Add the `Wiener` element to `skcomplex.core.components` with `dimension`, `covariance`, the `dW` handle, and
  its `Requires(time={TimeKind.CONTINUOUS})` declaration, and verify the declaration test
- [x] 1.3 Add the `wiener-time`, `wiener-covariance`, `wiener-handle`, `noise-convention-mixed`, and
  `noise-correction-scope` issues to `validate()`, and verify the invalid covariance, misused handle, mixed convention,
  and out-of-scope scenarios
- [x] 1.4 Add `NoiseKind` and the `noise` field of `Supports` to `skcomplex.core.base`, compute a system's noise
  structure, add the `noise-unsupported` issue, and verify that the compatibility table lists the noise support

## 2. Compilation

- [x] 2.1 Lower noise to rows, columns, process count, and covariance blocks with factors from `numpy.linalg`, on
  NumPy and Numba, and verify that a system with no named process gets identity columns and no blocks
- [x] 2.2 Replace the per-law convention correction with the general correction over shared and correlated
  processes, using each law's convention, remove the ambiguity error, and verify the Stratonovich scenarios

## 3. Solvers

- [x] 3.1 Apply the scaled scatter in `EulerMaruyama`, the hybrid SDE loop, `Heun`, and the Numba Euler-Maruyama loop of
  `skcomplex.core.backends`, declare their noise support, and verify the shared process, sample covariance, and same
  seed scenarios on both backends
- [x] 3.2 Declare diagonal noise only for `Milstein` and the LAMMPS backend, and verify that validation reports the
  Milstein and LAMMPS scenarios before compilation
- [x] 3.3 Compare a seeded run of every existing SDE model with a trajectory stored before the change, and verify that
  they are equal

## 4. Library

- [x] 4.1 Add the Langevin mode to `Binding` and `Dissociation`, and verify the `A + B -> C` conservation scenario and
  the error for both flags
- [x] 4.2 Add `chemical_langevin_network` to `chemistry/kinetics`, add its new entry to `docs/overview/inventory.toml`,
  and verify the enzyme totals scenario and the inventory check
- [x] 4.3 Add `correlated_geometric_brownian_motion` to `economics/finance`, add its new entry to
  `docs/overview/inventory.toml`, and verify the correlation scenario and the conformance suite

## 5. Documentation and gate

- [x] 5.1 Document named processes, `Wiener` elements, the per-law corrections, and the declared noise support in the user guide and
  the changelog, and verify the documentation test passes
- [ ] 5.2 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
