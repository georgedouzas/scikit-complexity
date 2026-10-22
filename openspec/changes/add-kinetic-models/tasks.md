# Tasks

## 1. Encounters and moments

- [ ] 1.1 Add `Encounter`, `Proliferation`, and `Destruction` to `core.laws` with the law kind `encounter`, the gain and
  loss lowering, and the tensor lowering on NumPy and Numba, and verify the conservative encounters scenario
- [ ] 1.2 Add the `Requires` declaration of the encounter laws and the validation issues for missing time or space and
  for transition probabilities that do not sum to one, and verify the invalid transition and wrong substrate scenarios
- [ ] 1.3 Add the `moment` reduction, and verify the normalised start and mean activity scenarios
- [ ] 1.4 Verify the welfare policy scenario with a micro-scale action written as an encounter with a fixed field group

## 2. Solvers and levels

- [ ] 2.1 Add `ParticleKinetics` with its `Supports` declaration, and verify the convergence scenario with ensembles and
  the unsupported law scenario
- [ ] 2.2 Add `distribution_from` and `members_from`, and verify the agents-to-distribution scenario

## 3. Library

- [ ] 3.1 Add the kinetic models as registered factories with their observables on the catalog entries, and verify the
  immune response scenario and the conformance suite

## 4. Documentation and gate

- [ ] 4.1 Document distributions as site groups, encounters, external actions, moments, the particle solver, and
  levels in the user guide, and verify the documentation test passes
- [ ] 4.2 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
