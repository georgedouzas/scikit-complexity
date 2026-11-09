## 1. Preconditions

- [x] 1.1 Confirm that `rename-core-systems` is archived and that `CompositeSystem` and `ReactionNetworkSystem` import
  from `skcomplex.core.components.systems`. Verify with `pdm run python -c` importing both names.
- [x] 1.2 Read "Decisions by the user" and "The placement rule" in design.md. Verify that every path the tasks below
  name follows them.
- [x] 1.3 Before running pytest from several agents, set `NUMBA_NUM_THREADS`, `OMP_NUM_THREADS`, and
  `OPENBLAS_NUM_THREADS` to 1. Verify with `env` in the shell of each agent.

## 2. The cross-domain package

- [x] 2.1 Create `src/skcomplex/systems/general/__init__.py` with a module docstring and an empty `__all__`, and the
  concept subpackages `elements`, `agents`, `matching`, `kinetics`, and `reactions`, each with an `__init__.py`.
  Mirror them under `tests/systems/general/`. Leave `dynamics` to `add-general-systems`. Verify with `pdm run python -c`
  importing every subpackage.
- [x] 2.2 Add to `pyproject.toml` a forbidden contract that `skcomplex.systems.general` imports no field package, and
  keep `skcomplex.systems.general` out of the field independence and field forbidden contracts so every field may import
  it. Verify with `nox -s checks -- quality`, which runs `lint-imports`.
- [x] 2.3 Extend `tests/systems/test_catalog.py` so every module of `systems/general` must have importers in at least
  two fields and `all_systems()` lists no class of it. Verify the new test passes on the empty subpackages.

## 3. The placement rule and its tests

- [x] 3.1 Add to `tests/test_sources.py` the test that no module under `src/skcomplex/core` imports `skcomplex.systems`,
  read from the syntax tree. Verify that it passes now.
- [x] 3.2 Add to `tests/test_sources.py` the test that the roots of `systems/general` and of every
  `systems/<field>/common` hold only `__init__.py` and subpackages. Make it pass by moving the loose modules into
  families: `economics/common/_asset.py` and `_geometric_brownian_motion.py` into `economics/common/assets`,
  `_balance_sheet.py` into `economics/common/banking`, `_exchange.py` and `_exchange_interaction.py` into
  `economics/common/exchange`, `engineering/common/_car.py` and `_optimal_velocity.py` into the existing
  `engineering/common/traffic`, `mathematics/common/_network_node.py`, `_regulated_node.py`, and `_vertex.py` into
  `mathematics/common/networks`, and `physics/common/_sandpile_cell.py` into `physics/common/avalanches`, with their
  tests and imports. Verify with the new test, `tests/systems` of those fields, and `lint-imports`.
- [x] 3.3 Add to `tests/test_sources.py` the test that every public class of `skcomplex.core.components` is a structural
  base, is named in the code of another core module, or is the general form of a law, with the three proxies of
  design.md and no allowlist. Run it and verify that it fails for exactly `MovementInteraction`,
  `BirthDeathInteraction`, `ImitationInteraction`, `EvolutionInteraction`, `FeedbackInteraction`,
  `OptimizationInteraction`, and `PointElement`. It guards every move below from the working tree, and it is committed
  with the move that makes it pass.
- [x] 3.4 Write the placement rule of design.md, word for word, into `openspec/constitution.md` and into the module
  layout section of `CONTRIBUTING.md`. Verify that both texts match the requirement "Every part has one place by the
  placement rule" of `package-layout`.

## 4. Merges inside the core

- [x] 4.1 Merge `PairCouplingInteraction` into `PairInteraction`: `PairInteraction` takes the optional `function`,
  `potential`, `write`, and `divisor` with the checks and laws of `_pair_coupling.py`, and returns no laws with neither.
  Key the LAMMPS lowering on `PairInteraction` with a potential. Move `test_pair_coupling.py` into `test_pair.py`.
  Verify with `tests/core/components/interactions/test_pair.py` and `tests/core/engine/solvers/test_lammps.py`.
- [x] 4.2 Rebase the subclasses of `PairCouplingInteraction` in biology, physics, and social on `PairInteraction`, and
  replace the class in the engine tests (`test_base.py`, `test_numba_neighbours.py`, `test_jacobians.py`,
  `test_pass_compiler.py`, `test_velocity_verlet.py`). Verify with those tests and with `tests/systems` for
  `collective_behaviour`, `active_matter`, `synchronization`, `common/particles`, and `crowds`.
- [x] 4.3 Merge `ConeNeighborhood` into `RadiusNeighborhood(radius, *, angle=pi, direction=None, center=False)`. Key the
  filter of `_metric_search.py`, `_neighborhood_kernel.py`, `_numba_printer.py`, `_pass_compiler.py`,
  `_reduction_lowering.py`, and `laws/utils/_reductions.py` on `angle < pi`. Move `test_cone.py` into
  `test_radius.py`. Migrate `physics/active_matter/_flocks.py` and the networks page of the user guide. Verify with
  `tests/core/components/spaces`, `tests/core/components/laws/utils/test_reductions.py`, and the flocks tests.

## 5. Elements and reaction networks

- [x] 5.1 Move `NonnegativeElement` to `systems/general/elements/_nonnegative.py` and `PointElement` to
  `systems/general/elements/_point.py`, with their tests under `tests/systems/general`. Replace their uses in core
  doctests and docstrings (`_jump.py`, `_rate.py`, `_gradient.py`, `_encounter.py`, `_geodesic.py`, `_gillespie.py`,
  `_semi_lagrangian.py`, `_strang_splitting.py`, `_velocity_verlet.py`, `_yoshida4.py`, `_lammps.py`, `viz/_points.py`)
  by a local `Element` subclass with the same state, and the same in `tests/core`. Verify with
  `pdm run pytest --doctest-modules src/skcomplex/core` and `tests/core`.
- [x] 5.2 Repoint every field realization of the two elements to `skcomplex.systems.general`. Verify with
  `tests/systems` of biology, chemistry, economics, engineering, physics, and social, and with `lint-imports`.
- [x] 5.3 Move `ReactionInteraction` to `systems/general/reactions/_reaction.py` and `ReactionNetworkSystem` to
  `systems/general/reactions/_reaction_network.py`, with `Stoichiometry` and `ReactionTable` in the public `types`
  module of the subpackage. Remove them from `core/types.py`, the core systems package, and its layers contract. Rewrite
  the doctest of `_gillespie.py` and `tests/core/analysis/test_stability.py` with `JumpInteraction` and
  `RateInteraction`. Move `test_reaction.py` and `test_reaction_network.py` under `tests/systems/general/reactions`.
  Verify with those tests and the core doctests.
- [x] 5.4 Repoint the reactions of biology and chemistry, including `biology/common/kinetics` and
  `chemistry/common/kinetics`, and the reaction networks of `chemistry/kinetics/_reactions.py` and
  `biology/epidemiology/_factories.py`. Verify with `tests/systems/biology` and `tests/systems/chemistry`.

## 6. Agents

- [x] 6.1 Move `MovementInteraction` and `BirthDeathInteraction` to `systems/general/agents`, with their tests. Rewrite
  `tests/core/engine/compile/test_validation.py` and `tests/core/engine/solvers/test_mesa_continuous_spaces.py` with a
  core `RuleLaw` interaction, or move the cases that test the movement itself to `tests/systems/general/agents`. Verify
  with those tests.
- [x] 6.2 Repoint the users of movement and birth-death in biology, economics, engineering, mathematics, and social.
  Verify with `tests/systems` of those fields.
- [x] 6.3 Move `ChoiceInteraction` and `ImitationInteraction` to `systems/general/agents` and
  `EvolutionInteraction` to `systems/general/agents` too, with their tests and the cases of `test_variants.py` and
  `conftest.py` that use them. Verify with `tests/systems/general` and `tests/core/components/interactions`.
- [x] 6.4 Repoint their users in biology, computer science, economics, and social. Verify with `tests/systems` of those
  fields.

## 7. Matching and encounters

- [x] 7.1 Move `MatchingInteraction`, `MatchSpec`, `MatchingRule`, and the rule specs `RandomSearchSpec`,
  `PartnerSwitchSpec`, `QueueSpec`, `IterativeMatchingSpec`, `ContinuousMatchingSpec`, `BatchMatchingSpec`, and
  `StableMatchingSpec` to `systems/general/matching`, with `CLEARINGS`. Add `PartnerMatchingSpec(PartnerSpec)` with
  `value` and its check, and remove `value` from the core `PartnerSpec`. Remove the moved names from `core/specs` and
  `core/components/types.py`. Verify with the moved tests under `tests/systems/general/matching` and with
  `tests/core/specs` and `tests/core/components/laws/test_match.py`.
- [x] 7.2 Repoint the matchings of economics and engineering, and replace `PartnerSpec(..., value=...)` by
  `PartnerMatchingSpec` in commons, shops, barter networks, and Metaphoria. Verify with `tests/systems/economics` and
  `tests/systems/engineering/traffic`, and the scenario "Partners with a clearing value".
- [x] 7.3 Move `EncounterInteraction` to `systems/general/kinetics`, with its tests, and repoint its users in biology,
  economics, physics, and social. Verify that the encounter passes still key on `BilinearLaw` with
  `grep -rn EncounterInteraction src/skcomplex/core` returning nothing, and with the tests of the kinetic systems.

## 8. Recipes of one field

- [x] 8.1 Move `CascadeInteraction` to `physics/common/avalanches/_toppling.py` as `TopplingInteraction`, with its
  tests, and repoint criticality, sandpiles, and faults. Replace its `BallNeighborhood` default by the lattice default.
  Verify with `tests/systems/physics` for those modules and the scenario "Sandpile avalanches".
- [x] 8.2 Move `FeedbackInteraction` to `economics/common/policy/_policy_rule.py` as `PolicyRuleInteraction`, with its
  tests and the case of `tests/core/components/test_types.py`, and repoint its users. Verify with
  `tests/systems/economics`.
- [x] 8.3 Move `OptimizationInteraction` and its four algorithm specs into
  `computer_science/evolutionary_computation/_optimisers.py`, with their tests. Verify with
  `tests/systems/computer_science/evolutionary_computation`.
- [ ] 8.4 Move `SatisfiabilityInteraction`, `ClauseSpec`, and `CardinalitySpec` into
  `computer_science/neural_networks/_perceptron.py`, with their tests. Remove `_satisfiability_issues` from
  `_pass_compiler.py`, and turn its test into the scenario "Real state" with `RoleElementError`. Verify with
  `tests/systems/computer_science/neural_networks` and `tests/core/engine/compile/test_pass_compiler.py`.
- [ ] 8.5 Move the zone branch of `CouplingInteraction` (`radii`, `clamp`, their checks and methods) with `RadiusSpec`
  and `ClampSpec` into `ZoneCouplingInteraction` in `biology/collective_behaviour/_zones.py`, and migrate flocks and
  leadership. Verify with the coupling tests and `tests/systems/biology/collective_behaviour`.
- [ ] 8.6 Move the score branches of `SamplingInteraction`: `RegionScoreSpec` with its methods into
  `CellAdhesionInteraction`, `HolonomyScoreSpec` with its methods into `WilsonSamplingInteraction`, and `RowScoreSpec`
  into `SequenceSamplingInteraction` in `biology/common/sequences/_sequence_sampling.py`, each returning an
  `ArrayUpdateLaw`. Migrate RNA walks, early stage, and fuzzy oil drop. Verify with the sampling tests, the scenarios of
  `molecular-structures`, and `tests/systems` for those modules.
- [ ] 8.7 Move `PseudoRiemannianSpace` to `physics/common/relativity/_spacetime.py` as `SpacetimeSpace` and
  `GeodesicInteraction` to `physics/common/relativity/_geodesic.py`, with their tests under
  `tests/systems/physics/common/relativity`. Load EinsteinPy through `import_extra` with the extra `relativity`, move
  `einsteinpy` from the runtime dependencies of `pyproject.toml` to `[project.optional-dependencies]`, and skip the
  relativity tests without it as the other extras do. Repoint celestial mechanics and cosmology. Verify with
  `tests/systems/physics`, a core import without `einsteinpy` installed, and the test of task 3.3.
- [ ] 8.8 Delete `DialectSpec`, its module, its export from `core/specs`, and its test, since nothing in `src` reads it.
  Verify with `grep -rn DialectSpec src tests` returning nothing and `tests/core/specs`.

## 9. Documents and inventory

- [ ] 9.1 Move the sections that teach a moved class out of the core pages of the user guide
  (`general_interactions.md`, `agent_based_systems.md`, `groups_of_many_elements.md`, `networks.md`,
  `constraints_and_search.md`, `kinds_of_dynamics.md`, `phase_space_fields.md`, `choosing_compatible_components.md`,
  `simulate.md`, `system_classes.md`, `extend_the_library.md`) into the new page
  `docs/overview/user_guide/core_components/general.md` on `skcomplex.systems.general` (create it, it is listed in the
  navigation, and `rewrite-docstrings-and-docs` later rewrites it), and write the core pages with the primitives. Update `docs/generate_parts.py`. Verify with
  `nox -s docs -- build` and the doc tests of the guide.
- [ ] 9.2 Make the inventory follow the core: drop the primitives that name a moved recipe, such as `markets`,
  `satisfiability`, `population-search`, and `label-copies`, rewrite `matrix-groups` without the holonomy sampling,
  and let the inventory check recompute the priority of the affected entries. Verify with the inventory test.

- [ ] 9.3 Add the placement rule of design.md, word for word, to `docs/overview/user_guide/extend_the_library.md`, with
  one example per step. Verify with `nox -s docs -- build` and the scenario "One text of the rule".

## 10. Integration

- [ ] 10.1 Grep `src`, `tests`, `docs`, the README, and CONTRIBUTING for every moved name under its old import path and
  for `PairCouplingInteraction`, `ConeNeighborhood`, `FeedbackInteraction`, `PseudoRiemannianSpace`, `DialectSpec`, and
  `PartnerSpec(` with `value`. Verify that only reviewed hits
  remain.
- [ ] 10.2 Run the full test suite and the conformance suite with random state 0. Verify that every trajectory agrees
  within its tolerance.
- [ ] 10.3 Run `nox -s checks` and `nox -s formatting`. Verify that every session passes with no suppression added.
- [ ] 10.4 Run `openspec validate move-recipes-to-systems --strict`. Verify that it passes.

## Workflow follow-up

- Revise `rename-core-systems` and `add-general-systems` as design.md lists, before each starts.
- Update `rewrite-docstrings-and-docs`, `add-torch-surrogates`, and `add-neural-particle-simulators` where they name a
  moved class.
- Archive the change with `openspec archive move-recipes-to-systems` after verification.
