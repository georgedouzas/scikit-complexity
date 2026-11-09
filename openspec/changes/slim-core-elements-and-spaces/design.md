## Context

See proposal.md for the motivation. Paths are written as they will be after `add-system-package`, which moves
`MassActionSystem` to `src/skcomplex/core/components/systems/_mass_action.py`. Two sibling changes slim the
interactions and the laws of the core at the same time, and this change touches neither package beyond the defaults and
docstrings that name a class it removes.

The rules that shape the change:

- The core names math operations, never field quantities. A core class that only fixes the arguments, the state name,
  or the order of another core class leaves the core, even under a standard math name.
- A field gets its own named subclass when it has its own term. A pure math convenience gets no copy, and systems use
  the core class directly. A realization with one user lives in the system module. One shared by several packages of a
  field lives in `systems/<field>/common/<family>/`, with no loose module beside the families. No realization is shared
  across fields.
- Nothing meaningful is deleted. Code moves into the surviving core class or into a realization.
- No alias, no shim module, no `TYPE_CHECKING` import, no suppression. Constructors only store their arguments, and
  checks run in named private methods.

## Goals / Non-Goals

**Goals:**

- One core atom for a nonnegative scalar, `NonnegativeElement`, with the state `x`.
- One core ball of lattice offsets, `BallNeighborhood`, under a named metric.
- One core group for every group, `GroupElement`, with lattice values handled on `'sites'`.
- No private spec imported across packages.

**Non-Goals:**

- No change to `PointElement`, the one vector-position atom, nor to `Element`, `DeclaredElement`, `FieldElement`'s
  discretizations, `WordElement`, or `PatchElement`.
- No change to the other spaces and neighbourhoods, which are each a distinct operation: `LatticeNeighborhood`,
  `AdjacencyNeighborhood`, `MemoryNeighborhood`, `StateTranslationNeighborhood`, `RadiusNeighborhood` and
  `NearestNeighborhood` on continuous spaces, `ConeNeighborhood`, `DelaunayNeighborhood`, `CofaceNeighborhood`, and every
  space and time.
- No rebasing of other field elements whose states happen to be nonnegative scalars, such as `SchloglElement` or the
  network node elements of mathematics. They carry several states or a field-specific name and stay as they are.
- No move of `ScenarioSpec` or of the other compile records of `components/specs`, which stay where they are.
- No change to the trajectory of any system on fixed seeds.

## Decisions

### What leaves or changes

| Class or function | From | To | Realizations per field | Core dependents |
|---|---|---|---|---|
| `DensityElement` (state `f`) | `core/components/elements/_density.py` | merged into `NonnegativeElement` (state `x`) in `core/components/elements/_nonnegative.py` | physics `PhaseSpaceDensityElement` in `systems/physics/common/fields/_phase_space_density.py` (fluids `_kinetic_equations`, active_matter `_kinetic_flocks`). biology `PopulationDensityElement` in `systems/biology/common/populations/_population_density.py` (oncology `_immune_competition`, ecology `_forest_size_structure`). social `ActivityDensityElement` in `systems/social/common/kinetics/_activity_density.py` (opinion_dynamics `_kinetic_opinion`, cooperation `_multilevel_selection`, crowds `_criminality`, crowds `_evacuation`). economics `WealthDensityElement` in `systems/economics/econophysics/_welfare.py` | `interactions/_encounter.py` (default state `'x'`, docstring), docstrings of `engine/solvers/_semi_lagrangian.py` and `_strang_splitting.py` |
| `CompartmentElement` (state `n`) | `core/components/elements/_compartment.py` | merged into `NonnegativeElement` | biology `CompartmentElement` in `systems/biology/common/kinetics/_compartment.py`, base of `PopulationElement` (`common/populations/_population.py`), `BranchElement` and `RatsElement` (collective_behaviour `_decisions`), `NeutralSpeciesElement` (ecology `_neutral`), `HabitatPatchElement` (ecology `_clustering`), and the docstrings of epidemiology, pharmacokinetics, excitable, kinetics, populations, and oncology. chemistry `SpeciesElement` in `systems/chemistry/kinetics/_nanoparticles.py` | `systems/_mass_action.py` (species and keys `(species, 'x')`), docstrings of `interactions/_reaction.py`, `_rate.py`, `_relaxation.py`, `_gradient.py`, `_jump.py`, `engine/solvers/_gillespie.py` |
| biology `SpeciesElement` | subclass of `Element` with its own `x` in `systems/biology/common/kinetics/_species.py` | subclass of `NonnegativeElement`, same module | stays biology | none |
| `LatticeElement` | `core/components/elements/_lattice.py` | merged into `GroupElement` on `'sites'` in `core/components/elements/_group.py`. `FieldElement` subclasses `GroupElement` | none, pure convenience. Systems call `GroupElement(..., on='sites')` in automata, evolutionary_computation, kinetics, networks, cooperation, opinion_dynamics, organizations, spreading, criticality, fluids, seismology, soft_matter, statistical_mechanics, synchronization, ecology, epidemiology, evolution, neuroscience, physiology, systems_biology, and traffic | `elements/_field.py`, docstrings of `analysis/_automata.py`, `interactions/_sampling.py`, `engine/viz/_sites.py` |
| `MooreNeighborhood` | `core/components/spaces/_moore.py` | `BallNeighborhood(r, metric='chebyshev')` in `core/components/spaces/_ball.py` | none, pure convenience | `spaces/_lattice.py` |
| `VonNeumannNeighborhood` | `core/components/spaces/_von_neumann.py` | `BallNeighborhood(r, metric='manhattan')` | none | `spaces/_lattice.py`, defaults of `interactions/_sampling.py` and `interactions/_cascade.py`, docstrings of `specs/_objective_spec.py`, `interactions/_local_rule.py`, `laws/_sample.py`, `laws/_cascade.py`, `laws/utils/_reductions.py` |
| `BoxNeighborhood` | `core/components/spaces/_box.py` | `BallNeighborhood(r, metric='chebyshev', n_dim=1, center='inline')` | none | `LatticeSpace._ball`, docstrings of `analysis/_automata.py`, `laws/_local_rule.py`, `laws/_extremal.py` |
| `TriangularNeighborhood` and `TRIANGULAR_OFFSETS` | `core/components/spaces/_triangular.py` | `BallNeighborhood(1, metric='hexagonal')`, the offsets computed by its hexagonal branch | none | `spaces/_lattice.py` |
| `LatticeSpace(neighborhood='moore' or 'von_neumann')` | `core/components/spaces/_lattice.py` | `LatticeSpace(metric=None)`, `'chebyshev'` or `'manhattan'` on a square lattice, `'hexagonal'` on a triangular one | none | `_default_neighborhood`, `interactions/_movement.py` docstring |
| `GeneratorNeighborhood` | `core/components/spaces/_generator.py` | `systems/computer_science/automata/_generator_neighborhood.py`, name kept, subclass of the core `MemoryNeighborhood` | computer science only (`_free_group_automata`, `_group_automata`) | none |
| `_SubsystemSpec` | private in `core/components/specs/_subsystem_spec.py` | public `SubsystemSpec`, same module, exported by `core/components/specs` | none | `engine/problems/_operator_lowering.py`, `engine/compile/passes/_operator_compiler.py` |

### The atom is `NonnegativeElement` with the state `x`

`NonnegativeElement(label, visual=None, *, x=0.0)` holds `x: Annotated[float, StateSpec(domain=RealSet.closed_open(0,
oo))]`. The name says the domain, which is the only thing the two merged classes fix. The state `x` is the plain name of
a scalar, the name the biology `SpeciesElement` already uses, and it reads well for an amount, a fraction, a density, or
a concentration. Every subclass of the atom keeps `x`, so `PopulationElement('rabbits', x=40)` and the observable
`'rabbits.x'` replace `n=40` and `'rabbits.n'`. Elements that do not subclass the atom keep their own state names.

Alternatives considered: `ScalarElement`, which would hide the domain, and keeping `n`, which reads as a count and not
as a density.

### Realizations follow the field's own term

Each field that has a term for the quantity gets a named subclass with no body beyond its docstring and default:

- physics `PhaseSpaceDensityElement`, the one-particle density of kinetic theory, shared by fluids and active matter.
- biology `PopulationDensityElement`, a structured population density, shared by oncology and ecology, and biology
  `CompartmentElement`, the compartments of epidemiology, pharmacokinetics, and populations. `PopulationElement` keeps
  its default size of 1.
- social `ActivityDensityElement`, the density over activity classes of the kinetic theory of active particles of
  Bellomo et al., shared by opinion dynamics, cooperation, and crowds. It opens the family `social/common/kinetics`.
- economics `WealthDensityElement`, the density over wealth classes, with one user.
- chemistry `SpeciesElement`, a chemical species, with one user. It moves to `chemistry/common/kinetics` when a second
  user appears.

Where the audit was wrong, this design plans what the code shows:

- The audit listed epidemiology, pharmacokinetics, excitable, and kinetics as code users of `CompartmentElement`. They
  only name it in docstring examples. The code users are populations, collective_behaviour, ecology `_neutral` and
  `_clustering`, systems_biology `_logical_networks`, and chemistry `_nanoparticles`.
- The audit proposed a biology `PlaceElement` for the Petri net tokens of `_logical_networks`. A place of a Petri net is
  a math term, not a biology one, so `_logical_networks` uses `NonnegativeElement` directly.
- The audit placed `ActivityDensityElement` in a loose module `social/common/_activity_density.py`. The model-library
  spec asks for family packages with no loose module, so it lives in `social/common/kinetics/`.
- The audit missed that `systems/biology/common/kinetics/_species.py` already defines a copy of the atom. It rebases on
  `NonnegativeElement`.
- The crowds `SecurityAgentsElement` is a density at one efficacy class that encounters read by the default state. It
  rebases on `ActivityDensityElement` with the state `x`, so every encounter of the kinetic systems reads `x`.
- `MassActionSystem` and the epidemiology factories, examples, and tests switch to `x` in the same step as the biology
  compartments, since the epidemic interactions serve both.

### `GroupElement` absorbs the lattice group

`GroupElement._flatten_values` turns a value of the shape of the lattice into one row per site, and a mapping from node
to value on a network into node order, when the group lives on `'sites'`. The space it reads is its own or the one it
inherits, which `_prepare` passes down. `GroupElement._prepare` takes the default visual of that space when the group
lives on `'sites'` and its members declare none. `FieldElement` subclasses `GroupElement` and passes `on='sites'`.

Alternative considered: keep `LatticeElement` as a convenience in each field. It is a pure math convenience, so by the
user's decision no field gets a copy.

### `BallNeighborhood` names the metric

`BallNeighborhood(radius=1, *, metric='chebyshev', n_dim=2, center=None, label='neighborhood')` subclasses
`LatticeNeighborhood` and returns its offsets from private methods, one per metric.

- `'chebyshev'` gives every offset with all coordinates within the radius, and `'manhattan'` every offset with the sum of
  absolute coordinates within it. Both come in lexicographic order.
- `'hexagonal'` gives the offsets of a two-dimensional triangular lattice within the hexagonal distance
  `max(|a|, |b|, |a + b|)`, ring by ring, each ring counterclockwise from the direction of the second axis. For radius 1
  this is today's `TRIANGULAR_OFFSETS`, so direction `k + 3` modulo 6 stays opposite to `k`.
- `center=None` leaves the site out, `'first'` puts it first, and `'inline'` keeps it in its lexicographic place, the
  order of Wolfram's rule numbers that `BoxNeighborhood` gave.
- A private check raises `ValidationError` for an unknown metric or centre, a negative radius, a hexagonal ball outside
  two dimensions, or a hexagonal ball with `center='inline'`, which has no lexicographic place.

`LatticeSpace` takes `metric=None` in place of `neighborhood`. `None` means `'hexagonal'` on a triangular lattice and
`'chebyshev'` on a square one, and a check rejects a hexagonal metric on a square lattice or another metric on a
triangular one. `_default_neighborhood` returns `BallNeighborhood(metric=..., n_dim=self.n_dim_)`. `_ball` returns
the Chebyshev ball with the site inline, as `BoxNeighborhood` did, on both geometries, so no analysis changes.

Alternatives considered: a boolean `center` with a second option for its place, which gives two arguments for one
choice, and keeping `BoxNeighborhood` as a computer science realization, which would copy a pure convenience.

### `GeneratorNeighborhood` moves to the automata package

It only fixes the word list of a `MemoryNeighborhood`, so it leaves the core. Its two users are modules of
`computer_science/automata`, so it lives in a private module of that package and is exported from it. The default
neighbourhood of `CayleySpace` keeps its own word list.

### `SubsystemSpec` becomes public

Two engine packages import it, so it cannot sit next to one of them. It names `OperatorViewSpec`, which lives in
`core/components/specs`, so it stays in the lowest package that names its types, public and exported.

### Import contracts

`systems/social/common` is a new shared package. Every forbidden contract of another field in `pyproject.toml` adds
`skcomplex.systems.social.common`, and the social contract forbids the `common` packages of the other fields.

## Risks / Trade-offs

- [The rename of `n` and `f` to `x` reaches observables, tests, gallery examples, and user guide pages] → Grep for
  `'.n'`, `('...', 'n')`, `{'f':`, and `n=` per module of a subclass of the atom, and let the doctests and the
  conformance suite catch what grep misses.
- [A `GroupElement` on `'sites'` with a space now takes the space's default visual] → Groups that relied on the plain
  visual change their drawing only. The viz tests and the gallery build check the result.
- [`BoxNeighborhood()` defaulted to one dimension and `BallNeighborhood` defaults to two] → Every former call passes
  `n_dim=1` explicitly, and the automata tests check the Wolfram rules.
- [Sibling changes edit the same docstrings in interactions and laws] → This change touches only defaults and
  docstrings that name a removed class, and rebases after them.

## Migration Plan

Breaking in 0.x with no shim. `add-system-package` lands first. The new core classes and the field realizations land first,
the users switch field by field, and the old classes go once no user is left. The release notes list every removed class with its replacement, and the change commits as `refactor!`.

## Open Questions

None. The user settled the atom's name and state, the new `social/common` package, the absence of copies for pure
conveniences, the `center` option for the Wolfram order, and that `PointElement` stays.
