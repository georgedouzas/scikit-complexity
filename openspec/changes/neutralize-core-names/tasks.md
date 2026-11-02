# Tasks

## 1. Vocabulary

- [x] 1.1 Rename the sampling vocabulary to `objective` and `intensity` in `SamplingInteraction`, `SampleLaw`, `SamplerSpec`, and `ObjectiveSpec`, with a thin `temperature` subclass in `models/physics/common`, moving its users; verify the sampling scenarios and the statistical mechanics models over 20 seeds.
- [x] 1.2 Rename the clearing specs to matching specs, and merge `GreedyMatchingSpec` into `PartnerSpec`, moving their users; verify the greedy scenario and every model using them over 20 seeds.
- [x] 1.3 Move `corotational` and `upper_convected` to `models/physics/common`, rename `HeadingSpec` to `DirectionSpec`, move `LammpsFixSpec` next to `LammpsSolver`, and take `_Offer` out of `core/interactions`; verify the import contracts and the model tests.
- [x] 1.4 Give particle groups explicit `StateSpec(role=...)` and remove `HEADING_NAMES`, `POSITION_STATE`, and `VELOCITY_STATES` from `analysis/_base.py`; verify the analyses of particle models.
- [x] 1.5 Check `KineticSolver` and every other solver and analysis name against the core rule, and rename or record each; verify `tests/core/test_base.py` `FIELD_TERMS` covers the removed words.

## 2. One way

- [x] 2.1 Merge `DestructionLaw` and `ProliferationLaw` into `BilinearLaw` across the interaction, backends, and kinetics solver; verify the encounter outcomes scenario and identical trajectories of the kinetic models.
- [x] 2.2 Fold `CollisionInteraction` and `CollisionLaw` into `EncounterInteraction(cross_section=)`; verify the binary integral within 1e-12 and the Boltzmann, BGK, and Enskog tests.
- [x] 2.3 Move `SpreadingInteraction` out of the core as a thin `TransitionInteraction` subclass in the models using it; verify those models identical or equal over 20 seeds.

## 3. Packages

- [x] 3.1 Move `models/earth/{climate,oceanography,seismology}` to `models/physics/`, removing the earth field from the catalog, inventory, gallery, tests, import-linter contracts, and the open changes; verify the catalog test, inventory check, gallery coverage, and contracts.

## 4. Operator dynamics

- [x] 4.1 Rename `FormKind.QUANTUM` and the `Quantum*` compiler, problem, kernel, lowering, solvers, analysis, and visuals to `Operator*`, with no alias, and move every user; verify the core names scenario and the quantum model tests with identical trajectories.
- [x] 4.2 State the operation in the docstrings of `GeneratorLaw`, `DissipatorLaw`, `HilbertSpec`, and the renamed classes, keeping the quantum words in `models/physics/common/quantum`, and record the Hermitian limit; verify `FIELD_TERMS` covers `Quantum` and the doctests pass.
- [x] 4.3 Read group states as series in analysis, so models with no position keep default data, and give phases their own role so alignment reads them; verify `generalized_lotka_volterra`, `hypercycles`, `replicator_dynamics`, `closed_contour_cell_exchange`, `doubling_map`, `skew_tent_map`, `on_off_intermittency`, `mirollo_strogatz`, and `swarmalators` have analysis data again.
- [ ] 4.4 Restore the results of particle analyses that changed: `cubic_chaotic_map` measures what it did, and scalar positions keep their previous shape where analyses read them; verify against the code at 72e2adde.
- [ ] 4.5 Rename the `temperature` and `force` arguments of `IrreversibilityAnalysis` to neutral names; verify `FIELD_TERMS` and its tests.
- [ ] 4.6 Rename `SnapshotsSpec.headings` and the `heading` locals to `direction`, `hamiltonian` in the operator code to `generator`, merge `StateSpec.position` with `role='position'`, and remove the fallback to a state named `x`; verify the model tests.

## 5. Documentation and gate

- [x] 5.1 Update the user guide and constitution; verify `pytest --markdown-docs docs/overview`.
- [ ] 5.2 Run formatting, checks, and the full suite through nox on one pinned commit with no other agents running, and the gallery examples of changed models; verify all pass.
