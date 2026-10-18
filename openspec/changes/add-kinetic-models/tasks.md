# Tasks

## 1. Distributions and encounters

- [ ] 1.1 Add `Distribution` with subsystems and network copies, and its moments as observables, and verify the
  normalised start and mean activity scenarios
- [ ] 1.2 Add `Encounter`, `Proliferation`, `Destruction`, and `Relaxation` with the probability check and tensor
  lowering on NumPy and Numba, and verify the conservative and invalid transition scenarios
- [ ] 1.3 Add `ExternalAction`, and verify the welfare policy scenario

## 2. Solvers and levels

- [ ] 2.1 Add `ParticleKinetics`, and verify the convergence scenario with ensembles
- [ ] 2.2 Add the level conversions, and verify the agents-to-distribution scenario

## 3. Library

- [ ] 3.1 Add the kinetic models as registered factories, and verify the immune response scenario and the conformance
  suite

## 4. Documentation and gate

- [ ] 4.1 Document distributions, encounters, actions, the particle solver, and levels in the user guide, and verify
  the documentation test passes
- [ ] 4.2 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
