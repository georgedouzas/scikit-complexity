## Why

The user decided that recipes do not belong in the core. A core component is a building block: a structural base where
laws run, a primitive mechanism that the engine lowers in its own way, or an operator generated from one function. A
recipe bundles strategies into a modelling pattern that the remaining primitives already express, such as an agent
rule, a matching on top of a match law, or a reaction on top of a jump. Recipes hide what the core really offers and
make the core grow with every field. An audit of every component of the core, checked against the code, finds the
recipes and proves for each one how the primitives express it. This change moves them to the systems.

## What Changes

- New cross-domain package `skcomplex.systems.common`, approved by the user, for realizations that several fields
  share. It holds family packages only, imports only the core, and every field may import it.
- **BREAKING** These interactions leave the core for `skcomplex.systems.common`: `MovementInteraction` and
  `BirthDeathInteraction` (family `agents`), `ChoiceInteraction` and `ImitationInteraction` (family `decisions`),
  `EvolutionInteraction` (family `evolution`), `MatchingInteraction` with its rule specs (family `matching`),
  `EncounterInteraction` (family `encounters`), and `ReactionInteraction` (family `reactions`).
- **BREAKING** `ReactionNetworkSystem`, with the aliases `Stoichiometry` and `ReactionTable`, leaves
  `skcomplex.core.components.systems` for `skcomplex.systems.common.reactions`. It builds a `CompositeSystem` of
  reactions, which are recipes, so it is a recipe too.
- **BREAKING** `NonnegativeElement` and `PointElement` leave the core for `skcomplex.systems.common` (families
  `amounts` and `points`). Each is `Element` with one declared state, and the engine reads the state spec, never the
  class.
- **BREAKING** Recipes of one field move into that field. `CascadeInteraction` becomes the physics
  `TopplingInteraction` in `physics/common/avalanches`. `FeedbackInteraction` moves to `economics/common/policy`.
  `OptimizationInteraction` and its four algorithm specs move into `computer_science/evolutionary_computation`.
  `SatisfiabilityInteraction`, `ClauseSpec`, and `CardinalitySpec` move into `computer_science/neural_networks`.
- **BREAKING** Strategy branches of kept classes move to their users. The zone terms of `CouplingInteraction`
  (`radii`, `clamp`, `RadiusSpec`, `ClampSpec`) become `ZoneCouplingInteraction` in `biology/collective_behaviour`.
  The three special objectives of `SamplingInteraction` move into subclasses: `RegionScoreSpec` into the cellular Potts
  module of biology, `HolonomyScoreSpec` into the Wilson sampling of physics, and `RowScoreSpec` into a new biology
  family `sequences`. Each subclass returns an `ArrayUpdateLaw`, which is the primitive they used all along.
- **BREAKING** The clearing value of a match leaves `PartnerSpec`. `PartnerSpec` keeps the pairing that transfers and
  match laws read, and the matching family adds `PartnerMatchingSpec(PartnerSpec)` with `value`.
- **BREAKING** Two merges inside the core. `PairCouplingInteraction` merges into `PairInteraction`, which takes the
  optional `function`, `potential`, `write`, and `divisor`, and the LAMMPS lowering keys on `PairInteraction` with a
  potential. `ConeNeighborhood` merges into `RadiusNeighborhood`, which takes `angle` and `direction`.
- The compiler loses its special check of `SatisfiabilityInteraction`. The role domain check that every composite
  system runs, through `RoleSpec(domains=('boolean',))`, reports the same error.
- Engine support never leaves with a recipe. `MatchLaw`, `CascadeLaw`, `ExtremalLaw`, `BilinearLaw` with its encounter
  kernel, `SampleLaw`, `ArrayUpdateLaw`, `RuleLaw`, `JumpLaw`, and every other law stay, with the record specs their
  kernels write.
- The core keeps its structural bases, its primitives, and its operators from one function: `GroupingInteraction`,
  `TrainingInteraction`, `GradientInteraction`, `HamiltonianInteraction`, `EulerLagrangeInteraction`,
  `GeodesicInteraction` with `PseudoRiemannianSpace`, `BallNeighborhood`, and every law utility. The design gives the
  proof for each.
- Nothing is deleted. Code moves to the systems or into the surviving primitive, and tests move with it. No alias and
  no shim remain, as the project makes breaking changes in 0.x.

## Capabilities

### New Capabilities

- `shared-realizations`: the cross-domain package `skcomplex.systems.common`, its families, its import rules, and the
  recipes it holds for agents, decisions, evolution, matching, encounters, reactions, amounts, and points.

### Modified Capabilities

- `system-modeling`: the core holds building blocks only, and recipes live in the systems.
- `general-mechanisms`: the core mechanisms lose the recipes, pairs couple through `PairInteraction`, and the general
  elements and reactions are realizations of `skcomplex.systems.common`.
- `model-library`: realizations shared by several fields live in `skcomplex.systems.common`, and field recipes such as
  toppling, feedback, zone coupling, and sequence sampling live in their field.
- `constraint-satisfaction`: a state that is not Boolean is reported by the role domain check.
- `search-processes`: population search is a realization of computer science.
- `agent-spaces`: cone neighbourhoods are radius neighbourhoods with an angle.

## Impact

- Core code: `core/components/interactions` loses the modules of the moved classes and merges `_pair_coupling.py`
  into `_pair.py`. `elements` loses `_nonnegative.py` and `_point.py`, `spaces` merges `_cone.py` into `_radius.py`,
  and `systems` loses `_reaction_network.py`. `core/specs` loses the matching, optimization,
  satisfiability, zone, and score specs. `core/types.py` and `core/components/types.py` lose `Stoichiometry`,
  `ReactionTable`, and `MatchingRule`. The compiler, the LAMMPS solver, the metric search, the neighbourhood kernel,
  the numba printer, and the reduction lowering change where they name a moved or merged class. Doctests that used a
  moved class use a primitive or a local subclass.
- Systems: a new package `skcomplex.systems.common` with its families, new field families `physics/common/avalanches`,
  `economics/common/policy`, and `biology/common/sequences`, and the imports of every user module in all eight fields.
- Configuration: the import contracts in `pyproject.toml` gain `skcomplex.systems.common`.
- Tests: the tests of every moved class move under `tests/systems/`, mirroring the new modules. Engine tests that used
  a recipe as a fixture use a primitive.
- Documents: the user guide pages that teach a recipe move it to the systems pages, the inventory primitives follow
  the core, and the drafted changes that name a moved class need artifact updates, listed in the design.
- Ordering: lands right after `rename-core-systems`, and before `add-general-systems`,
  `hoist-constants-and-helper-methods`, and `rewrite-docstrings-and-docs`, so later changes see the final core.
- Users: code that imports a moved class imports it from its new package. Code that passes `value` to `PartnerSpec`
  uses `PartnerMatchingSpec`. Code that subclasses `PairCouplingInteraction` subclasses `PairInteraction`. Code that
  builds `ConeNeighborhood(r, angle)` builds `RadiusNeighborhood(r, angle=angle)`.
