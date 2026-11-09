## Context

The user's decision is in proposal.md. This design proves the verdict of every component of the core from the code, on
`development` after `slim-core-interactions` and `slim-core-laws`, with the names that `rename-core-systems` gives
(`CompositeSystem`, `ReactionNetworkSystem`, `_get_product_propensity`).

The test has three steps for each component.

1. Write it with the remaining primitives. If that works, it is a recipe and leaves.
2. If it needs engine support by type, name that support: a compile pass, a kernel, a solver, or a backend lowering
   that only it triggers. Engine support never leaves with a recipe. Either the support is a cross-domain primitive
   that the core keeps under a mathematical name, and the recipe on top of it moves, or the whole class is the
   primitive and stays.
3. Place what leaves by its users. One system module: that module. Several modules of one package: that package.
   Several packages of one field: a family of `systems/<field>/common`. Several fields: a concept subpackage of the
   new `skcomplex.systems.general`. A field with its own term names its realization. `systems/general` uses neutral
   names.

Users per field come from `grep -rlw <Class> src/skcomplex/systems/<field>`. Engine dependents come from the same grep
over `src/skcomplex/core`. Two read-only audits covered elements, spaces, and systems (A) and the specs and types (B).
Their claims were checked against the code, and the corrections are noted below.

## Goals / Non-Goals

**Goals:**

- The core holds building blocks only, and every class that stays has a stated reason in the code.
- Every recipe keeps its arguments, checks, hooks, docstring, and tests in its new home.
- Every catalog trajectory stays the same within the tolerance of its tests.

**Non-Goals:**

- The engine values of `core/specs` (about forty compiler and kernel records). They are not recipes. Making them
  private to the engine is a package move, which needs the user's approval in its own change.
- Writing the general systems that `add-general-systems` drafts. The user decided that general systems are recipes, so
  they will live in `skcomplex.systems.general`. This change lists the edits that change needs, and does not make them.

## Decisions

### Interactions

| Class | Expressed as, or engine feature kept | Users per field | Verdict and destination |
| --- | --- | --- | --- |
| `MovementInteraction` | An agent rule. `_get_laws` returns `RuleLaw(self._move_all)` or a member `RuleLaw`. It uses only the system view (`agents`, `rng`, `torus`, `report`). No engine file names it | biology, economics, mathematics, social | leaves, `systems/general/agents/_movement.py` |
| `BirthDeathInteraction` | An agent rule. `RuleLaw(self._live)` with `system.create` and removal through the view. No engine file names it | biology, economics, engineering, mathematics, social | leaves, `systems/general/agents/_birth_death.py` |
| `ChoiceInteraction` | Symbolic form: `UpdateLaw(state, categorical(...))` and `RuleLaw` for activity. Agent form: `RuleLaw(self.choose)`. No engine file names it | computer science, economics, social | leaves, `systems/general/agents/_choice.py` |
| `ImitationInteraction` | A `NeighborInteraction` that returns `JumpLaw` per pair of options in rate form, `UpdateLaw` with `categorical` in probability form, and `RuleLaw` in agent form. No engine file names it | economics, social | leaves, `systems/general/agents/_imitation.py` |
| `EvolutionInteraction` | An agent rule, `RuleLaw` only. No engine file names it | biology, computer science, economics, social | leaves, `systems/general/agents/_evolution.py` |
| `MatchingInteraction` | `MatchLaw(requesters, providers, rule).get_pairs(...)` inside a `RuleLaw`, with `transfer` for settlement. The engine lowers `MatchLaw`, never the matching | economics, engineering | leaves with its rule specs, `systems/general/matching/` |
| `EncounterInteraction` | `BilinearLaw` and `RateLaw` with `collision_integral`. The encounter passes (`_compile_encounters.py`, `_encounter_problem.py`, the encounter kernel, `TableValuesSpec`) key on `BilinearLaw`, not on the class | biology, economics, physics, social | leaves, `systems/general/kinetics/_encounter.py`. `BilinearLaw`, its kernel, and `collision_integral` stay |
| `ReactionInteraction` | A `JumpInteraction` whose `changes` are the stoichiometric vector and whose rate is `k` times the product of the reactant amounts. Rate mode is one `RateLaw` per species, Langevin mode adds `NoiseLaw`, agent mode is `RuleLaw`. The Gillespie solver reads `JumpLaw` and names the class only in a doctest | biology, chemistry, and the core `ReactionNetworkSystem` | leaves, `systems/general/reactions/_reaction.py` |
| `CascadeInteraction` | A `NeighborInteraction` that returns `CascadeLaw(drive=[ArrayUpdateLaw], relax=[LocalRuleLaw or ArrayUpdateLaw], stable, activity)` with threshold rules. The lattice solver and the cascade kernel key on `CascadeLaw` | physics (criticality, seismology) | leaves as the physics `TopplingInteraction`, `systems/physics/common/avalanches/_toppling.py`. `CascadeLaw` stays |
| `FeedbackInteraction` | `Interaction` whose `_get_laws` returns `RuleLaw(self.act, phase=self.phase)`, with `system.set_parameter`. No engine file names it | economics (econophysics, finance, innovation, macroeconomics, markets, resources) | leaves as the economics `PolicyRuleInteraction`, `systems/economics/common/policy/_policy_rule.py` |
| `OptimizationInteraction` | `RuleLaw(self._advance)` that runs one pymoo generation. No engine file names it | computer science (`evolutionary_computation/_optimisers.py`) | leaves into that module with its four algorithm specs |
| `SatisfiabilityInteraction` | `RuleLaw(self._decide)` around one PySAT solver. The compiler names it only in `_satisfiability_issues`, a check that the general role domain check already makes, since its role declares `domains=('boolean',)` | computer science (`neural_networks/_perceptron.py`) | leaves into that module with `ClauseSpec` and `CardinalitySpec`. The compiler check goes |
| `PairCouplingInteraction` | The only concrete subclass of `PairInteraction`. The LAMMPS solver lowers it by type when it has a potential (`_lammps.py`) | biology, physics, social | merges into `PairInteraction` in the core. The LAMMPS lowering keys on `PairInteraction` with a potential |
| `GroupingInteraction` | Engine support by type: `SpecsPass._grouping_specs` lowers its symbolic form to hyperedge kernels over random groups redrawn every step, with a reduction over the groups that hold a member | economics, social | stays, a primitive: random hypergraphs redrawn each step |
| `TrainingInteraction` | Engine support by type: the learning compiler finds it, unrolls the nested system its role binds, and differentiates it. `GradientStepLaw` holds the optimizer but not the binding of the nested system | computer science | stays, a primitive. The class is exactly the binding and the law, with no strategy of its own |
| `GradientInteraction`, `HamiltonianInteraction`, `EulerLagrangeInteraction` | Operators generated from one function, a potential, a Hamiltonian, or a Lagrangian, by symbolic differentiation. `core/analysis/_invariants.py` keys on the Hamiltonian and Euler-Lagrange classes | physics, engineering, social | stay, operators from one function |
| `GeodesicInteraction` | The rate law `x'' = -Γ(x)(x', x')` with the Christoffel symbols that its space derives. The operator comes from the space, not from a function the interaction takes, and the space is a recipe (see below). No engine file names either | physics (celestial mechanics, cosmology) | leaves with the space, `systems/physics/common/relativity/_geodesic.py` |
| `RateInteraction`, `ConservationInteraction`, `CrossingInteraction`, `JumpInteraction`, `MapInteraction`, `LocalRuleInteraction`, `TransitionInteraction`, `TransferInteraction`, `SamplingInteraction`, `TopologyInteraction`, `ConnectionInteraction`, `CouplingInteraction` | Each is the interaction form of one law the engine lowers (`RateLaw`, `FluxLaw`, `CrossingLaw`, `JumpLaw`, `UpdateLaw`, `LocalRuleLaw`, `TransitionLaw`, `TransferLaw`, `SampleLaw`), or is keyed by type in the specs pass (topology growth, connections with zero port flows, edge coupling) | many fields | stay. `CouplingInteraction` and `SamplingInteraction` lose their strategy branches, see below |
| `BaseInteraction`, `Interaction`, `NeighborInteraction`, `HyperedgeInteraction`, `PairInteraction` | Structural bases where laws run per member, edge, hyperedge, or pair. The specs pass dispatches on them | all | stay |

A recipe in code, as the moved classes will read after the move:

```python
class FeedbackInteraction(Interaction):
    def _get_laws(self, namespace):
        return [RuleLaw(self.act, phase=self.phase)]


class ReactionInteraction(JumpInteraction):
    def _get_laws(self, namespace):
        if self._get_mode() == 'jump':
            return super()._get_laws(namespace)  # JumpLaw with the stoichiometric changes
        propensity = self._get_rate(namespace)
        return [RateLaw(state, change * propensity) for state, change in self._get_changes(namespace).items()]
```

### Strategy branches of kept classes

| Branch | Expressed as | Users | Destination |
| --- | --- | --- | --- |
| `CouplingInteraction(radii=..., clamp=...)` with `RadiusSpec` and `ClampSpec` | The agent form of a coupling, a `RuleLaw` that sums terms over priority zones and clamps the turn. No engine file reads the specs | biology, two modules of `collective_behaviour` | `ZoneCouplingInteraction(CouplingInteraction)` in `systems/biology/collective_behaviour/_zones.py`, with both specs. Audit B placed it in `biology/common`, but both users are in one package, so the package holds it |
| `SamplingInteraction` with `RegionScoreSpec` | Python methods that build an `ArrayUpdateLaw` and bypass the Monte Carlo kernel | biology, `physiology/_cellular_potts.py` | into `CellAdhesionInteraction`, which already subclasses `SamplingInteraction` |
| `SamplingInteraction` with `HolonomyScoreSpec` | The same, with a heat bath on matrix group states | physics, `common/fields/_wilson_sampling.py` | into `WilsonSamplingInteraction`, which already subclasses it |
| `SamplingInteraction` with `RowScoreSpec` | The same, Metropolis on whole rows | biology, `evolution/_rna_walks.py` and two modules of `systems_biology` | `SequenceSamplingInteraction(SamplingInteraction)` in `systems/biology/common/sequences/_sequence_sampling.py`, with the spec |
| `PartnerSpec.value` and its `CLEARINGS` check | The clearing value of a match, read only by `MatchingInteraction`. `CLEARINGS` lives in `_random_search_spec.py`, which leaves | economics | `PartnerMatchingSpec(PartnerSpec)` in `systems/general/matching`. `PartnerSpec` keeps the pairing that `MatchLaw`, `TransferInteraction`, and `GroupingInteraction` read |

`SamplingInteraction` keeps its symbolic objective and `MoveSpec`, which `SampleLaw` and the Monte Carlo kernel lower.
Its three `isinstance` branches go. A subclass returns its own `ArrayUpdateLaw`, the general primitive these branches
already used. Alternative considered: a new block score hook in the core. Rejected, since `ArrayUpdateLaw` already is
that hook.

### Laws and law utilities

Every law is lowered by the engine by type, so every law stays. The engine dependents are counted by
`grep -rlw <Law> core/engine`.

| Law | Engine feature | Verdict |
| --- | --- | --- |
| `MatchLaw` | `passes/_unpack_matches.py`, `passes/_agents.py`, the census, the numpy and numba backends, Mesa, co-simulation | stays. Audit B moved it with matching, but transfers and groupings read it, and the engine lowers it |
| `CascadeLaw` | the cascade kernel, the lattice solver, `CascadeRecordSpec`, `analysis/_events.py` | stays, relaxation to a fixed point within a step |
| `ExtremalLaw` | the extremal kernel, `ExtremalRecordSpec`, the lattice solver | stays, the update of the extremal site |
| `SampleLaw` | the Monte Carlo kernel, the lattices pass | stays |
| `LinearProgramLaw` | `passes/_linear_programs.py` and every solver that holds algebraic states | stays |
| `GradientStepLaw` | the learning compiler, the gradient descent solver, the JAX backend | stays, with `DatasetSpec` |
| `ProductionLaw`, `SubstitutionLaw` | the rewriting compiler, the word and patch kernels | stay |
| `GeneratorLaw`, `DissipatorLaw` | the operator compiler, the unitary, Lindblad, and trajectory solvers | stay |
| `BilinearLaw` | the encounter passes, kernel, and Jacobian | stays |
| every other law | lowered by the backends and passes | stays |

The law utilities are named math vocabulary or read engine state (`matches`, `transfer`, `residence`,
`collision_integral` with its collision kernel). `slim-core-laws` already removed the parametrised copies. None is a
recipe, so all stay.

### Elements, spaces, and systems

| Class | Expressed as, or engine feature kept | Users per field | Verdict and destination |
| --- | --- | --- | --- |
| `NonnegativeElement` | `Element` with `x: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))]`. No engine file checks the class | biology, chemistry, economics, engineering, physics, social | leaves, `systems/general/elements/_nonnegative.py` |
| `PointElement` | `Element` with a vector state of order 2 and the role `'position'`. LAMMPS, the symplectic solvers, and the boundaries read the state spec, not the class | biology, physics, social | leaves, `systems/general/elements/_point.py` |
| `ReactionNetworkSystem` (today `MassActionSystem`) | `BaseSystem` whose `_build_system` returns `CompositeSystem(elements=[NonnegativeElement(...)], interactions=[ReactionInteraction(...)])`. No engine file checks it. The user decided that general systems are recipes, and the core keeps only `BaseSystem` and `CompositeSystem` | biology, chemistry | leaves with `ReactionTable`, `systems/general/reactions/_reaction_network.py` |
| `ConeNeighborhood` | A radius search filtered by the angle to a direction. `RadiusNeighborhood(r)` equals `ConeNeighborhood(r, pi)`. The filter in `_metric_search.py` and `_neighborhood_kernel.py` stays | physics | merges into `RadiusNeighborhood(radius, *, angle=pi, direction=None, center=False)` |
| `BallNeighborhood` | Audit A called it a convenience. The code says otherwise: `LatticeSpace._default_neighborhood` and `LatticeSpace._ball` build it for the metric of the lattice, hexagonal ring order included | biology, chemistry, computer science, physics, social | stays, the metric ball the lattice uses |
| `PseudoRiemannianSpace` | A `ContinuousSpace` with symbolic data that EinsteinPy derives in `_prepare` (metric, Christoffel symbols, curvature). No engine file reads it, the engine sees a plain `ContinuousSpace` | physics (celestial mechanics, cosmology) | leaves as the physics `SpacetimeSpace`, `systems/physics/common/relativity/_spacetime.py`. `einsteinpy` becomes the model-only extra `relativity`, loaded through `import_extra` |
| every other element, space, neighbourhood, time, and `CompositeSystem`, `BaseSystem` | each triggers its own pass, kernel, or solver, as audit A lists | all | stay |

The core doctests and docstrings that build a `NonnegativeElement` or a `PointElement` define a local `Element`
subclass with the same state, so the core never imports the systems.

### Specs and types

| Name | Read by | Destination |
| --- | --- | --- |
| `RandomSearchSpec`, `PartnerSwitchSpec`, `QueueSpec`, `IterativeMatchingSpec`, `ContinuousMatchingSpec`, `BatchMatchingSpec`, `StableMatchingSpec`, `MatchSpec`, `MatchingRule` | `MatchingInteraction` only | `systems/general/matching` |
| `DifferentialEvolutionSpec`, `ParticleSwarmSpec`, `EvolutionStrategySpec`, `CovarianceAdaptationSpec` | `OptimizationInteraction` only | `computer_science/evolutionary_computation/_optimisers.py` |
| `ClauseSpec`, `CardinalitySpec` | `SatisfiabilityInteraction` only | `computer_science/neural_networks/_perceptron.py` |
| `RadiusSpec`, `ClampSpec` | the zone branch of `CouplingInteraction` | `biology/collective_behaviour/_zones.py` |
| `RegionScoreSpec`, `HolonomyScoreSpec`, `RowScoreSpec` | the score branches of `SamplingInteraction` | with their subclasses, as above |
| `Stoichiometry`, `ReactionTable` | `ReactionInteraction`, `ReactionNetworkSystem` | `systems/general/reactions` |
| `PartnerSpec`, `Pairing`, `Feasibility`, `Preference` | `MatchLaw`, transfers, groupings | stay. Audit B moved `Feasibility` with `MatchLaw`, which stays |
| `DatasetSpec` | `GradientStepLaw` | stays |
| `CascadeRecordSpec`, `ExtremalRecordSpec`, `TableValuesSpec` | the kernels of laws that stay | stay |
| `GeometrySpec` | `WordElement`, the word kernel, which computes turtle segments from it every generation, and `viz/_geometry.py` | stays. The engine handles it by type (step 2 of the placement rule), and the turtle reading of a word is cross-domain in principle: plants, fractals, and words of automata |
| `DialectSpec` | nothing in `src` reads it, only `core/specs/__init__.py` and its test | deleted with its test, as dead code |
| every other spec and kind | the engine or a primitive, as audit B lists | stay |

### The placement rule

One ordered test decides where any part goes, now and in the future. It is the single source of truth. The
`package-layout` spec states it as a requirement, and `openspec/constitution.md`, `CONTRIBUTING.md`, and the extension
page of the user guide repeat it word for word.

1. A structural base, such as a `Base` class, the composite system, or a base where laws run per member, edge,
   hyperedge, or pair, goes to the core.
2. A part that the engine handles by type, through a compile pass, a kernel, a solver, a backend lowering, or an
   analysis that checks the class or its spec, goes to the core, and so does an operator generated from one function.
   It is named by its mathematics and must be cross-domain in principle.
3. Every other part is a recipe, expressible with core parts. Its users place it. Several fields:
   `skcomplex.systems.general.<concept>`. One field with several systems: `systems/<field>/common/<family>`. One system:
   that system's module. A field with its own term names it, and `general` uses neutral names.
4. A recipe that needs something the core cannot express reveals a missing core primitive, which enters by step 2.
   Engine support never lives inside a recipe.
5. The core never imports `skcomplex.systems`.

`tests/test_sources.py` checks the rule mechanically.

- Every public class of `skcomplex.core.components` passes one of three proxies of steps 1 and 2. A structural base is
  a class whose name starts with `Base`, or a class that another public class of `skcomplex.core.components`
  subclasses. A class handled by type is named in the code of a core module other than its own and other than the
  modules of its subclasses. Names count in code only, read from the syntax tree, never in docstrings or doctests.
  The general form of a law is an interaction that, for some law in its `law_classes` other than `RuleLaw`, lists the
  fewest law classes among the public interactions of its structural base that list that law. `RuleLaw` is excluded
  because a law of arbitrary code is never a mathematical operation the engine lowers.
- No module under `src/skcomplex/core` imports `skcomplex.systems`.
- The roots of `systems/general` and of every `systems/<field>/common` hold only `__init__.py` and subpackages.

The third test needs the loose modules at the roots of four field `common` packages to move into families named by
concept: the approved `economics/common/assets`, `banking`, and `exchange`, the existing
`engineering/common/traffic`, and the approved `mathematics/common/networks` and `physics/common/avalanches`. The task list names every module.

The proxies were checked against the core this change leaves, by reading the code with `ast`. Every class that stays
passes, with no allowlist. Engine and analysis references cover the laws, the spaces, the neighbourhoods but one, the
elements, `GroupingInteraction`, `TrainingInteraction`, `TopologyInteraction`, `ConnectionInteraction`,
`LocalRuleInteraction`, `PairInteraction`, `HamiltonianInteraction`, and `EulerLagrangeInteraction`. `LatticeSpace`
names `BallNeighborhood`. The general form of a law covers `RateInteraction` (`NoiseLaw`), `ConservationInteraction`
(`FluxLaw`), `CrossingInteraction`, `TransitionInteraction`, `TransferInteraction`, `SamplingInteraction`,
`GradientInteraction` (`RateLaw` alone), `MapInteraction` (`UpdateLaw`), `JumpInteraction` (`JumpLaw`), and
`CouplingInteraction`, the one subclass of `NeighborInteraction` that returns `RateLaw`. On the core before the change,
the same check fails for `MovementInteraction`, `BirthDeathInteraction`, `ImitationInteraction`,
`EvolutionInteraction`, `FeedbackInteraction`, `OptimizationInteraction`, and `PointElement`, which is the guard the
moves need. It passes `SatisfiabilityInteraction` only through the compiler check this change removes, and
`ChoiceInteraction` because it ties with `MapInteraction`, so the proxy is necessary, not sufficient, see the risks.

### The new package and its subpackages

The user named the package `skcomplex.systems.general` and asked for one subpackage per concept, with no loose module at
its root. The final split from the audit is:

| Subpackage | Holds |
| --- | --- |
| `elements` | `NonnegativeElement`, `PointElement` |
| `agents` | `MovementInteraction`, `BirthDeathInteraction`, `ChoiceInteraction`, `ImitationInteraction`, `EvolutionInteraction` |
| `matching` | `MatchingInteraction`, its rule specs, `PartnerMatchingSpec`, `MatchSpec`, `MatchingRule` |
| `kinetics` | `EncounterInteraction` |
| `reactions` | `ReactionInteraction`, `ReactionNetworkSystem`, `Stoichiometry`, `ReactionTable` |
| `dynamics` | reserved for `DynamicalSystem`, `NetworkDynamicalSystem`, `ReactionDiffusionSystem`, and `StockFlowSystem` of `add-general-systems` |

The user's example also placed `BallNeighborhood` in a `neighborhoods` subpackage and grouping among the agents. The
code keeps both in the core. `LatticeSpace` builds `BallNeighborhood` as its default neighbourhood and its ball, and
the specs pass lowers `GroupingInteraction` by type. So there is no `neighborhoods` subpackage.

The package imports only the public modules of the core and no field package. Every field may import it. The core
never imports it. Core features that produce systems keep returning plain composites of core parts: the data-driven
functions, such as `sparse_identification`, return a `CompositeSystem` of `DeclaredElement` and core interactions, and
the system-dynamics translator writes source text that imports from `skcomplex.systems.general.dynamics`, which is text
and not an import. A module of `systems/general` serves at least two fields, and the catalog test checks that as it
checks the field `common` packages.

Names stay neutral, so every class keeps its name. Field terms appear only in field families. Physics names its
toppling (`TopplingInteraction`), and biology names its sequences (`SequenceSamplingInteraction`).

Alternative considered: field copies instead of a cross-domain package. Rejected. `MovementInteraction` alone has
users in four fields, and copies would drift.

### Ordering

This change lands right after `rename-core-systems`, which renames `MassActionSystem` to `ReactionNetworkSystem` and
`_get_mass_action` to `_get_product_propensity`. Moving first would rename the classes twice. It lands before
`add-general-systems`, `hoist-constants-and-helper-methods`, and `rewrite-docstrings-and-docs`, so they measure and
document the final core. `slim-core-interactions` and `slim-core-laws` are already archived.

### Revisions of drafted changes

The user decided that general systems are recipes. The edits below are listed here and made by whoever revises each
change. This change edits neither.

`rename-core-systems` keeps its renames in the core, so this change moves the renamed classes once.

1. `proposal.md`: in the bullet on `MassActionSystem`, add that `ReactionNetworkSystem` stays in the core only until
   `move-recipes-to-systems` moves it, with `ReactionInteraction`, to `skcomplex.systems.general.reactions`. In
   Capabilities, describe `base-interfaces` as "the system hierarchy names `CompositeSystem`" and drop "the general
   system shared by fields is `ReactionNetworkSystem`". In Impact, add that `move-recipes-to-systems` lands right
   after it.
2. `specs/base-interfaces/spec.md`: drop the modified requirement "General systems enter the core only when fields
   share them". This change rewrites it, and two rewrites of one requirement in a row would contradict each other.
3. `specs/general-mechanisms/spec.md`, `specs/package-layout/spec.md`, and `specs/model-library/spec.md`: no edit.
   This change modifies or removes the same requirements after them, so the renamed wording is an intermediate state.
4. `design.md`: add one sentence to the context that the reaction network and the reaction are renamed in the core
   and moved by the next change.
5. `tasks.md`: no edit. Task 2.2 renames `_mass_action.py` to `_reaction_network.py` in the core, and this change moves
   that module.

`add-general-systems` puts its four systems in `skcomplex.systems.general`, not in the core.

1. `proposal.md`: the four systems go to `systems/general/dynamics/` (`_dynamical.py`, `_network_dynamical.py`,
   `_reaction_diffusion.py`, `_stock_flow.py`). Replace "the core should offer each of them once"
   by "the systems should offer each of them once, in `skcomplex.systems.general`". In Impact, replace the four core
   modules and the core import contract by the subpackage `skcomplex.systems.general.dynamics`, and the core
   components guide by the systems guide. Ordering: after `move-recipes-to-systems`, which creates the package.
2. `specs/base-interfaces/spec.md`: drop the modified requirement. This change already states that general systems
   live in `skcomplex.systems.general`. Its admission test moves to `general-systems` as "a general system serves
   systems of two fields, or can, and is named after its mathematics", without "adds a feature no other core part
   offers", which is the test of a core primitive.
3. `specs/general-systems/spec.md`: the Purpose says "general systems of `skcomplex.systems.general`", and every import
   path in the scenarios names `skcomplex.systems.general.dynamics`.
4. `specs/model-library/spec.md`: rebase "Ready-made systems are classes" on the version of this change, and in
   "Ready-made systems use the general system of their shape" import the general systems from
   `skcomplex.systems.general`.
5. `design.md`: every module path and test path moves to `systems/general/dynamics` and
   `tests/systems/general/dynamics`. The layers contract on `skcomplex.core.components.systems` gains nothing. The
   system-dynamics translator writes source text that imports `StockFlowSystem` from
   `skcomplex.systems.general.dynamics`, which the forbidden contract on the core allows, since it is text and not an
   import.
6. `tasks.md`: task 1 imports `ReactionNetworkSystem` from `skcomplex.systems.general.reactions` and
   `NonnegativeElement` from `skcomplex.systems.general.elements`. Tasks 2.1, 5.1, 7.1, and 9.1 write the modules under
   `src/skcomplex/systems/general/dynamics/` and export them from its `__init__.py`, without a layers contract. Tasks
   2.2, 5.2, 7.2, and 9.2 write the tests under `tests/systems/general/dynamics/`. Task 12.1 writes the sections in the
   systems guide, not in `core_components/systems.md`.

Other drafted changes name a moved class and need artifact updates before they start.

- `rewrite-docstrings-and-docs`: its core components pages use `NonnegativeElement`, `PointElement`,
  `BirthDeathInteraction`, and `EncounterInteraction` as core examples. They move to the systems pages, and the core
  pages take `Element` with a declared state.
- `hoist-constants-and-helper-methods`: no class is named, but its scope must be measured after the moves.
- `add-torch-surrogates` and `add-neural-particle-simulators`: their designs name `PairCouplingInteraction`, which
  becomes `PairInteraction`.

## Risks / Trade-offs

- [The compiler check of satisfiability had the code `law-domain`, the role check raises `RoleElementError`.] → The
  constraint-satisfaction spec changes with it. The perceptron tests assert the new error.
- [Engine tests used recipes as fixtures.] → They use a primitive or a local subclass. A test of a recipe itself moves
  to `tests/systems/general`.
- [Merging `PairCouplingInteraction` makes `PairInteraction` concrete.] → With neither `function` nor `potential` it
  returns no laws, as the base does today, so subclasses that override `_get_laws` keep working.
- [`systems/general` grows into a second core.] → A module enters only with users in two fields, the catalog test
  checks it, and no engine code may key on anything in it.
- [Moving `NonnegativeElement` and `PointElement` reverses `slim-core-elements-and-spaces`.] → The user decided it, by
  the placement rule. The engine reads their state specs, so no solver changes.
- [`einsteinpy` becomes an extra.] → `SpacetimeSpace` loads it through `import_extra`, which names the extra
  `relativity` in its error, and the relativity tests skip without it as the other extras do.
- [The reference proxy of the placement rule is a necessary test, not a sufficient one.] → It catches a class with no
  engine counterpart, such as an agent rule. A parametrised copy of a general form, such as a choice that fixes the
  function of a map, passes it, and the existing requirement that core interactions are distinct operations catches
  it in review.

## Migration Plan

Users replace imports. `from skcomplex.core.components.interactions import MatchingInteraction` becomes
`from skcomplex.systems.general.matching import MatchingInteraction`, and the same holds for every moved class and spec.
`PartnerSpec(..., value='provider')` becomes `PartnerMatchingSpec(..., value='provider')`. A subclass of
`PairCouplingInteraction` subclasses `PairInteraction`. `ConeNeighborhood(r, a, direction='theta')` becomes
`RadiusNeighborhood(r, angle=a, direction='theta')`. There is no rollback beyond reverting the commits.

## Decisions by the user

These are final.

- `NonnegativeElement` and `PointElement` leave to `systems/general/elements`.
- `PseudoRiemannianSpace` leaves to physics as `SpacetimeSpace` in `physics/common/relativity`, with
  `GeodesicInteraction`. `einsteinpy` becomes a model-only dependency, the extra `relativity`.
- `FeedbackInteraction` becomes the economics `PolicyRuleInteraction` in `economics/common/policy`.
- The field families `physics/common/avalanches`, `economics/common/policy`, and `biology/common/sequences` are
  approved.
- The field families `economics/common/assets`, `economics/common/banking`, `economics/common/exchange`, and
  `mathematics/common/networks`, which take the loose modules of their `common` packages, are approved.
- The cross-domain package is `skcomplex.systems.general`, with one subpackage per concept.
- General systems are recipes. The core keeps `BaseSystem` and `CompositeSystem` only.
- `DialectSpec` is deleted in this change, and `GeometrySpec` is placed by the placement rule.
- The placement rule below is the single source of truth for where any new part goes.

## Implementation notes

Decisions taken while applying the change, by the placement rule.

- The source test of task 3.3 also flags `ReactionNetworkSystem`, which `rename-core-systems` named after the audit. It
  is a recipe by step 3 and leaves with task 5.3, so the guard needs no allowlist.
- `SCHEDULES`, the agent schedules every interaction checks, is now exported by `skcomplex.core.components.interactions`,
  like `OPTIMIZERS`. The general reactions, choices, and imitations read it, and a realization imports only the public
  modules of the core.
- `PairInteraction` keeps the write default `'x'` of the former pair coupling. Its role declares the states `position`,
  `write`, and `divisor`, as the coupling did, and with neither a function nor a potential it returns no laws.
- The engine keys the cone filter on `isinstance(over, RadiusNeighborhood) and float(over.angle) < math.pi`, in the metric
  search, the neighbourhood kernel, the Numba printer, and the compiler check of directions.
- Tasks 4.1 and 4.2, and tasks 5.1 to 5.4, are committed together, since the systems import the merged or moved names
  and a commit of one task alone would leave the package unimportable.
- The core doctests that built a `NonnegativeElement` or a `PointElement` define `AmountElement` and `BallElement`, local
  `Element` subclasses with the same states. The tests of `tests/core` that built them define the same two classes.
- The user guide may import any public name of a concept subpackage of `skcomplex.systems.general`, besides the system
  classes of a subfield, so `tests/test_sources.py` accepts these imports.
- Tests of `tests/core` that use a general realization only as a fixture of a base behaviour, such as the aliases and
  the grid checks of `test_base.py`, import it from `skcomplex.systems.general`, as they already import field systems.
  Tests of a realization itself move under `tests/systems/general`.
- `test_validation.py` checks the role element error with a local agent rule on `RuleLaw`. The Mesa tests of continuous
  spaces keep `MovementInteraction` as the fixture that moves agents, imported from `skcomplex.systems.general.agents`,
  since they test the solver and not the movement. The variant cases of movement and birth-death moved to their tests.
- `TopplingInteraction` keeps its default `BallNeighborhood(metric='manhattan', center='first')`. The default
  neighbourhood of a lattice leaves out the site itself, which the relaxation reads first, and the toppling builds its
  laws without the space. `BallNeighborhood` stays in the core, so the physics family imports it publicly.
- The tuple of algorithm specs that `OptimizationInteraction` checks is `ALGORITHM_SPECS` in `_optimisers.py`, since
  the module already names its table of algorithms `ALGORITHMS`. `pymoo` stays a runtime dependency, as the change
  decides no extra for it.
- The agent form of `CouplingInteraction` was the zone branch, since it needed radii, so it leaves whole with
  `ZoneCouplingInteraction`. The core coupling keeps the forms `'rate'` and `'map'` and checks one written state and
  a phase of the map form in `_check_form`, which the zone coupling overrides with its own checks. `AgentCombination`
  moves with it.
- `DialectSpec` is not dead code: the expression printer of the engine reads `DIALECTS`, its table of instances, from
  `core/specs/_dialect_spec.py`. By step 2 of the placement rule the class stays as engine support, so the change
  removes only its public re-export from `skcomplex.core.specs`, which nothing outside the package used, and deletes
  no test, since it had none. Making the printer's dialects private to the engine is a package move left to its own
  change.
