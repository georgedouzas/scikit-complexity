# Tasks

## 1. Vocabulary

- [x] 1.1 Rename the sampling vocabulary to `objective` and `intensity` in `SamplingInteraction`, `SampleLaw`, `SamplerSpec`, and `ObjectiveSpec`, with a thin `temperature` subclass in `models/physics/common`, moving its users; verify the sampling scenarios and the statistical mechanics models over 20 seeds.
- [x] 1.2 Rename the clearing specs to matching specs, and merge `GreedyMatchingSpec` into `PartnerSpec`, moving their users; verify the greedy scenario and every model using them over 20 seeds.
- [x] 1.3 Move `corotational` and `upper_convected` to `models/physics/common`, rename `HeadingSpec` to `DirectionSpec`, move `LammpsFixSpec` next to `LammpsSolver`, and take `_Offer` out of `core/interactions`; verify the import contracts and the model tests.
- [x] 1.4 Give particle groups explicit `StateSpec(role=...)` and remove `HEADING_NAMES`, `POSITION_STATE`, and `VELOCITY_STATES` from `analysis/_base.py`; verify the analyses of particle models.
- [x] 1.5 Check `KineticSolver` and every other solver and analysis name against the core rule, and rename or record each; verify `tests/core/test_base.py` `FIELD_TERMS` covers the removed words.

## 2. One way

- [ ] 2.1 Merge `DestructionLaw` and `ProliferationLaw` into `BilinearLaw` across the interaction, backends, and kinetics solver; verify the encounter outcomes scenario and identical trajectories of the kinetic models.
- [ ] 2.2 Fold `CollisionInteraction` and `CollisionLaw` into `EncounterInteraction(cross_section=)`; verify the binary integral within 1e-12 and the Boltzmann, BGK, and Enskog tests.
- [ ] 2.3 Move `SpreadingInteraction` out of the core as a thin `TransitionInteraction` subclass in the models using it; verify those models identical or equal over 20 seeds.

## 3. Packages

- [x] 3.1 Move `models/earth/{climate,oceanography,seismology}` to `models/physics/`, removing the earth field from the catalog, inventory, gallery, tests, import-linter contracts, and the open changes; verify the catalog test, inventory check, gallery coverage, and contracts.

## 4. Quantum

- [ ] 4.1 Replace the quantum names in `engine` and `analysis` by general hooks (default solver by compiler, issue by declared compiler, backend kernels from the plugin); verify the import contracts and the quantum model tests.
- [ ] 4.2 Move the quantum laws, solvers, kernels, lowering, compiler, visuals, and analysis to `models/physics/common/quantum`, and decide `qutip`'s placement by the dependency rule; verify the quantum scenarios and identical trajectories.

## 5. Documentation and gate

- [ ] 5.1 Update the user guide and constitution; verify `pytest --markdown-docs docs/overview`.
- [ ] 5.2 Run formatting, checks, and the full suite through nox on one pinned commit with no other agents running, and the gallery examples of changed models; verify all pass.
