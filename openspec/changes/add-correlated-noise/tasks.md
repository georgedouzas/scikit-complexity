# Tasks

## 1. Modelling

- [ ] 1.1 Add `Noise.process` with local labels, label tuples, and `Wiener` handles, and verify that a law with no
  process compiles exactly as before
- [ ] 1.2 Add the `Wiener` element with `dimension` and `covariance`, its checks, and the `dW` handle, and verify the
  invalid covariance and misused handle scenarios

## 2. Compilation

- [ ] 2.1 Lower noise to rows, columns, process count, and covariance blocks with factors from `numpy.linalg`, on
  NumPy and Numba, and verify that a system with no named process gets identity columns and no blocks
- [ ] 2.2 Replace the per-law convention correction with the general correction over shared and correlated
  processes, remove the ambiguity error, add the out-of-scope error, and verify the Stratonovich scenarios

## 3. Solvers

- [ ] 3.1 Apply the scaled scatter in `EulerMaruyama`, the hybrid SDE loop, `Heun`, and the Numba Euler-Maruyama loop,
  and verify the shared process, sample covariance, and same seed scenarios on both backends
- [ ] 3.2 Make `Milstein` and the LAMMPS backend reject shared or correlated processes with a named error, and verify
  the Milstein scenario
- [ ] 3.3 Compare a seeded run of every existing SDE model with a trajectory stored before the change, and verify that
  they are equal

## 4. Library

- [ ] 4.1 Add the Langevin mode to `Binding` and `Dissociation`, and verify the `A + B -> C` conservation scenario and
  the error for both flags
- [ ] 4.2 Add `chemical_langevin_network` to `chemistry/kinetics`, add its new entry to `docs/overview/inventory.toml`,
  and verify the enzyme totals scenario and the inventory check
- [ ] 4.3 Add `correlated_geometric_brownian_motion` to `economics/finance`, add its new entry to
  `docs/overview/inventory.toml`, and verify the correlation scenario and the conformance suite

## 5. Documentation and gate

- [ ] 5.1 Document named processes, `Wiener` elements, the corrections, and the Milstein limit in the user guide and
  the changelog, and verify the documentation test passes
- [ ] 5.2 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
