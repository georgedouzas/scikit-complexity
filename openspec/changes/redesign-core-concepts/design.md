# Design

## Context

See proposal.md for the motivation. The facts that shape the approach are these.

**Packages and their sizes.** The core has 13 packages and about 64 000 lines:
- `compile`: 20 000 lines in 33 flat modules.
- `analysis`: 15 000 lines.
- `solvers`: 6 100 lines. `_steppers.py` alone is 1 589 lines and writes RK4 and Euler-Maruyama up to four times each.
- `laws`: 5 100 lines.
- `backends`: 4 100 lines.

The packages already import in one direction, `base` → `space` → `laws` → `components` → `interactions` → `system` →
`compile` → `solvers` → `backends`, and then `runtime` → `viz`, `experiments`, `analysis`. There is no cycle. The
disorder is inside the packages.

**Components and laws.**
- `Element` and `Interaction` derive from `Component`. `Group` and `System` derive from `BaseEstimator` directly.
  `Field` derives from `Group`, and `Particle`, `Wiener`, `Word`, `Patch`, and `HostGraph` derive from `Element`.
- Laws have no common base. `Rate`, `Jump`, and the other 15 law classes in `laws/_laws.py` are independent dataclasses.
- The compiler and validation recognise laws through hand-kept tuples: `FORMLESS`, `LATTICE_LAWS`, `KERNEL_FREE`,
  `DECLARED_FREE`, `law_types()`, and the accepted kinds in `_collect_laws`. `_compiler.py` alone holds 80 `isinstance`
  dispatches.
- `Form`, `LawKind`, and `SpaceKind` are closed enums in `base`.

**How the models use the laws.** The 518 registered models use these laws across 40 fields:

| Law | Model files | Fields |
|---|---|---|
| `Rate` | 137 | 40 |
| `Rule` | 77 | 27 |
| `Update` | 40 | 22 |
| `Field` (a component, for comparison) | 38 | 22 |
| `Noise` | 22 | 12 |
| `Jump` | 21 | 10 |
| `Transfer` | 17 | 10 |
| `LocalRule` | 17 | 13 |
| `Auxiliary` | 11 | 6 |
| `Event` | 10 | 6 |
| `Match` | 8 | 6 |

`Equation`, `Energy`, `Boundary`, `Objective`, `Invariant`, `Dissipator`, `Production`, and `FractionalNoise` are used
in fewer than 8 files each. These counts decide which primitives are general and which belong to a formalism.

**Defects to fix in passing.** The reviews found defects whose fix falls out of the restructure, so this change fixes
them:
- two partition planners (`_compiler._plan_partitions` and `_validation._mixed_issues`);
- about 14 flattenings per compile, from `validate` and nine validation helpers;
- a different step count in `_steppers.py` and in the generated Numba loops;
- a silent backend substitution for stepwise forms, at `_compiler.py:3662`;
- a silent Numba fallback, at `_numba.py:1150`;
- a swallowed lattice check, at `_validation.py:509`;
- the duplicate cell grids in `_numba_cells` and `_numba_neighbours`;
- the `Law` alias, which misses six law classes and cannot be evaluated at run time.

## Goals / Non-Goals

**Goals:**
- One concept model for the core, with every module placed by the concept it owns.
- Formalisms that plug in without any edit to the shared core.
- The same numeric results for every registered model, except where a listed defect is fixed.

**Non-Goals:**
- **Performance work:** the compile-time hotspot in `has_complex`, kernel cache keys that include sizes, cache
  eviction, compiled loops for Heun and Milstein, and parallel neighbour searches. A follow-up change takes them on the
  new structure.
- **Test-suite work:** slow markers, the 652-second avalanche test, and unmarked timing tests.
- **Analysis cleanups:** the five hand-written entropies, the entropy units, and the return types of the analyses.
- **Reorganising `analysis`:** it stays organised by topic, because analyses read trajectories, not formalisms.
- **New primitives:**
  - initial values drawn from the simulation's random stream;
  - a random move to a neighbouring cell for agents.

  The models repeat code that suggests both, and each is its own change.

## Decisions

### The concept model

The shared core holds the general concepts, one package per part of a system or step of a run:

| Package | Concept | Holds |
|---|---|---|
| `base` | vocabulary | symbols, domains, variable declarations, errors, and the open registries of kinds, forms, and traits |
| `space` | where and when | `Space` and its general kinds (continuous, lattice, network), neighbourhoods, and time |
| `laws` | how state changes | the `Law` base and the general laws, plus expressions, draws, reductions, and `neighbors` |
| `components` | what has state | `Component`, `Element`, `Group`, `Wiener`, and `Aggregate` |
| `interactions` | what couples components | `Interaction`, roles, and bindings |
| `system` | the whole model | `System`, `Observable`, and ensembles |
| `compile` | from system to problem | the pass pipeline, layout, kernels, printing, problem, planner, and validation |
| `solvers` | advancing a problem | the stepper contract, `step_count`, and the general steppers by form |
| `backends` | evaluating kernels | NumPy and Numba |
| `runtime`, `viz`, `experiments`, `analysis` | running and inspecting | unchanged in scope |
| `formalisms` | plug-ins | one subpackage per formalism |

The general laws are `Rate`, `Noise`, `Jump` with its exponential and waiting-time clocks, `Update`, `Transition`,
`Emission`, `Event`, `Boundary`, `Equation`, `Transfer`, `Invariant`, and `Call`. Each is used across fields, or it
belongs to the general forms: ODE, SDE, DAE, discrete map, jump process, and co-simulation. Delays and events stay in
the core, because any ODE or map can carry them.

Each formalism gathers modules that are scattered today:

| Formalism | Holds | From |
|---|---|---|
| `agents` | `Rule`, `Match`, phases, agent memory, the Mesa backend, the agent stepper | `laws/_laws`, `interactions/_matching`, `space/_phases`, `components/_memory`, `backends/_mesa`, `solvers/_agents` |
| `lattices` | `LocalRule`, `Energy`, drives, Monte Carlo laws, lattice steppers | `laws/_lattices`, `compile/_lattices`, `solvers/_lattices` |
| `groups` | group spaces, Cayley graphs, their visuals | `space/_groups`, `space/_cayley`, `viz/_cayley` |
| `rewriting` | productions, words, patches, host graphs | `laws/_rewriting`, `components/_rewriting`, `interactions/_rewriting`, `compile/_rewriting`, `_rewriting_kernels`, `_words`, `_structures`, `solvers/_rewriting`, `viz/_geometry`, `viz/_words` |
| `quantum` | `Generator`, `Dissipator`, operators, expectations, QuTiP solvers | `laws/_operators`, `components/_quantum`, `system/_expectations`, `compile/_quantum`, `solvers/_quantum`, `viz/_expectations` |
| `fields` | `Field`, differential operators, flux laws, meshes, finite differences, finite volumes, spectral and stiff integrators, conserved noise | `laws/_fields`, `components/_fields`, `space/_meshes`, `compile/_fields`, `_volumes`, `_splitting`, `_conserved`, `solvers/_stiff` |
| `kinetics` | encounter laws, moments, activity levels, DSMC | `laws/_encounters`, `laws/_moments`, `components/_levels`, `compile/_encounters`, `_contractions`, `solvers/_kinetics` |
| `flows` | stocks, flows, auxiliaries, lookups, system-dynamics delays, conservation monitoring | `laws/_flows`, `compile/_flows`, `_quantities`, `runtime/_conservation` |
| `renewals` | renewal clocks of members | `compile/_renewals`, `solvers/_renewals` |
| `fractional` | the fractional form, Caputo schemes, fractional noise | the fractional parts of `_compiler.py` and `laws/_laws`, `solvers/_caputo`, `solvers/_fractional` |
| `topology` | topology requests, capacity layouts | `laws/_topology`, `compile/_topology` |
| `particles` | `Particle`, potentials, pair potentials, the LAMMPS backend, body visuals | `components/_particles`, `interactions/_potentials`, `backends/_lammps`, `_lammps_lepton`, `viz/_bodies` |
| `control` | `Objective`, collocation, Pontryagin, receding-horizon following | `laws/_laws`, `solvers/_collocation`, `_optimal`, `_pontryagin` |

Every formalism has the same internal layout: `_laws`, `_components`, `_spaces`, `_passes`, `_solvers`, `_backends`,
`_visuals`, and `_checks`, each present only when needed, plus the `__init__` that registers the formalism.

- **Alternative: a layered layout that keeps `compile/lowering/<formalism>` and `solvers/<formalism>`.** It was
  rejected because one formalism would still span five packages, and adding one would still edit the shared core.
- **`groups` is a formalism**, not part of `space`. Group spaces are 1 130 lines, and only the group cellular automata
  use them.
- **Dependencies between formalisms:** `fields` declares `lattices`. The other formalisms declare none, and each phase
  that moves one confirms its dependencies with the layout check.

### One component hierarchy, with no generated fields

`Component(BaseEstimator)` lives in `components/_component.py` and imports only `base`. It holds:
- `label`;
- the class-level `declarations()`, which maps the names of the class's `state`, `parameter`, and `role` declarations
  to their declarations, and adds the derived companions: `<name>_d1` up to order `k - 1`, and the lags;
- one constructor, `__init__(self, label, **values)`, which checks `values` against `declarations()` and stores each
  value under its own name;
- `get_params`, which returns `label` and the declared names, so `clone`, `set_params`, and `repr` follow the estimator
  contract;
- `laws(s)`, which returns no laws by default.

A constructor setting that is not a state, parameter, or role, such as `radius`, `dimension`, or `layer`, is declared by
an explicit `__init__` on its class. That `__init__` takes the setting as a keyword-only argument, stores it under its own
name, and passes the remaining values to `super().__init__(label, **values)`. `get_params` returns the keyword arguments
of the `__init__` signatures along the class's method resolution order, plus the declared names, each with its value or
its default. `replicate` and the other readers of all values read `get_params`.
- **Alternative: a fourth declaration, `option(default)`.** It was rejected by the maintainer in favour of the explicit
  constructor.
- **Alternative: read plain annotated class attributes.** It was rejected because the constructor would read
  annotations implicitly.

`Element`, `Group`, `Interaction`, and `System` derive from it.
- **`Group`** holds an element type, a size, values, and a placement: `on='members' | 'sites' | 'edges' |
  'hyperedges'`. It validates its value names and shapes at construction, and its laws come from its element type.
- **`Field`** is a `Group` subclass in `formalisms/fields`, whose extra arguments choose the spatial discretisation.
- **`Aggregate`** stays an `Element` in `components`, because co-simulation is a core form.

Alternatives considered for the constructor:
- **Keep the `dataclass` transform.** It was rejected, because it is the hidden magic the redesign removes, and it
  injects the companion fields as class attributes.
- **Generate a `__signature__` per class.** It was rejected for the same reason.

The cost is that editors no longer complete declared names in constructors. The docstrings' `Args` blocks, which the
pydoclint gate already enforces, document them.

### One law hierarchy with declared traits

`Law` lives in `laws/_law.py` and imports only `base`. Each law is an explicit frozen dataclass deriving from `Law`, and
declares as `ClassVar`s:
- `kind`, a registered `LawKind`;
- `time_class`: continuous, event, discrete, or constraint;
- `traits`, a frozenset of registered `Trait` values such as `FORMLESS`, `KERNEL_FREE`, `LATTICE`, and `DECLARED_FREE`.

The tables of the compiler and validation become queries over the registered law classes, such as "the law classes with
trait `KERNEL_FREE`". `Law` stops being a type alias, so introspection and autodoc work.

`LawKind`, `Form`, `SpaceKind`, and `Trait` become open registries of named value objects in `base`, instead of closed
enums. The shared core registers the general ones, and a formalism registers its own when it is imported. The current
member names stay as module constants, such as `Form.ODE`, so call sites read the same.

### Formalism registration

`formalisms/_formalism.py` defines the frozen dataclass `Formalism(name, depends, laws, components, spaces, passes,
forms, solvers, backends, visuals, checks)` and `register(formalism)`. `register` does three things:
1. It checks that the dependencies are registered and form no cycle.
2. It adds each entry to the registry its concept owns: law kinds and traits to `base`, passes and checks to
   `compile`, forms and solvers to `solvers`, backends to `backends`, and visuals to `viz`.
3. It records the formalism for `registered_formalisms()`.

Each formalism's `__init__` ends with `register(FORMALISM)`. Importing any name of a formalism therefore registers it,
and a system cannot hold a law of a formalism that was never imported.

Alternatives considered:
- **Python entry points.** These are deferred until a formalism lives outside the package.
- **Decorators scattered over modules.** These were rejected, because they hide the contribution list that the
  declaration states in one place.

### Compile as a pipeline of passes

`compile` keeps the shared steps and runs passes over one `Compilation` state, whose fields are produced in order:
`nodes` (flattened once), `census` (laws by kind), `plan` (partitions and their forms), `layout`, `lowered`, `kernels`,
and `problem`.
- A `Pass` declares the fields it `reads` and `writes`, and the pipeline orders passes by these declarations.
- The general passes live in `compile/passes/`, one module each:
  - `flatten`, `census`, `plan`, and `layout`;
  - `lower_rates`, `lower_noise`, `lower_jumps`, `lower_updates`, and `lower_events`;
  - `lower_reductions`, `lower_neighbourhoods`, `lower_members`, `lower_complex`, `kernels`, and `problem`.
- A formalism contributes its own passes, such as the lattice lowering, the fractional form, or the topology capacity
  layouts.
- `validate` runs the pipeline through `plan` and the registered checks, and returns the issues with the `Compilation`
  state. `compile_system` continues from that state, so the system is flattened once.
- The planner is one function, `plan_partitions(census, solvers)`. It reads the registered forms and the solvers'
  `supports.laws`, and both validation and compilation call it.
- The planner, `plan_partitions(census, kinds, solvers)`, needs both a registered form for the lowered law classes and
  a solver for the declared law kinds, because a solver alone can accept a mix, such as a jump with a rate in the
  renewal solver, that no form lowers. Validation therefore lowers every system without errors up to the `kinds` field,
  and a lowering error becomes a validation issue.
- The mutable accumulators `_Hybrid` and `_Fields` become fields of `Compilation` written by declared passes, so the
  order of the passes is explicit.

`_compiler.py` is removed. Its 4 348 lines split into the passes, and each formalism-specific part moves to its
formalism. The law-value walkers `_law_values`, `_values_of`, `_validation._law_values`, and `_complex._law_fields`
become one traversal in `laws`, `values_of(law, *, include=...)`.

### Solvers by form, one step count

`solvers` holds:
- `_stepper.py`, the stepper contract with `step_count(t, until, dt)`;
- `_ode.py`, with `ScipyODE`, `RK4`, and `VelocityVerlet`;
- `_sde.py`, with Euler-Maruyama, Heun, and Milstein;
- `_maps.py`, with `Iteration`;
- `_jumps.py`, with `Gillespie`;
- `_hybrid.py`, with delays and events;
- `_cosimulation.py`.

`step_count` is the one rounding rule, `max(1, ceil((until - t) / dt - 1e-9))`. The steppers, the Numba loop
generator, and the LAMMPS backend all call it.
- The inline loops in the steppers call the same `_step` functions that the generated loops mirror.
- A parity test runs every fixed-step stepper on both backends.
- `backends` keeps NumPy and Numba. `_numba_cells` merges into `_numba_neighbours`, so the pair kernels use the bounded
  grid of the neighbour search.

### Reported substitutions

- **Stepwise forms on NumPy:** the compiled problem records the backend of each partition and each law kind. When the
  requested backend differs from the one used, compilation emits a `BackendWarning`, which names the part and the
  reason.
- **Numba fallback:** a Numba kernel that falls back to Python emits the same warning once, naming the law kind and the
  reduction.
- **Invalid lattices:** validation turns a `Lattice.check()` failure into a `space-invalid` issue.

### Module order and hoisting

Every module follows the constitution's order: imports, constants, private functions, public functions, private
classes, public classes.
- A helper that serves another concept moves to that concept's module. In `components/_components.py`, for example:
  - `part_count` moves to `space`;
  - `probe_laws` and `_FreeSymbols` move to `compile`;
  - `runs_rules` moves to `formalisms/agents`;
  - `_lattice_visual` moves to `viz`.
- An extended layout test checks these with the standard-library `ast` module:
  - the order of definitions;
  - that the modules defining `Component` and `Law` import only `base`;
  - that no import is guarded by `TYPE_CHECKING`;
  - that the shared core never imports a formalism;
  - that a formalism imports only the formalisms it declares;
  - that exports do not leak, per the export rule of the package-layout spec.

### One spelling of neighbour

Public names use the spelling the models already use most: `neighbors`, `Neighborhood`, and the binding
`over='neighbors'`. `neighbor_sum` and `adjacent` fold into `neighbors(..., reduce='sum')`, and `neighbors` with
`reduce=None` on a lattice neighbourhood becomes `reduce='each'`. The `LocalRule` and `Energy` keyword `neighborhood=`
becomes `over=`, as in `neighbors`. Prose keeps British spelling.

## Risks / Trade-offs

- **Every model's imports change.** → A codemod script rewrites the import paths from a table that maps old paths to
  new ones. The full suite, the catalog conformance tests, and the gallery cache comparison prove the numbers did not
  change.
- **A long branch conflicts with ongoing work.** → No other change is applied while this one is in flight. The phases
  land in order on one feature branch, and the suite passes after each phase.
- **Open registries are slower or less type-safe than enums.** → Registered values are interned frozen objects compared
  by identity. Typed constants keep call sites checked by mypy.
- **The 15 active changes describe old paths.** → After this change merges, an update pass rewrites their designs and
  tasks to the new layout. Their requirements are unaffected.
- **Constructor completion is lost in editors.** → Element types document their declarations under `Args`, which the
  docstring gate enforces.

## Migration Plan

The work proceeds in phases, each of which ends with a green suite.
1. Concepts in the shared core: the open registries in `base`, the `Law` base, and the `Component` hierarchy.
2. The `formalisms` package and its registration. The formalisms then move one at a time, smallest first: `topology`,
   `renewals`, `groups`, `control`, `fractional`, `flows`, `kinetics`, `particles`, `quantum`, `rewriting`, `lattices`,
   `fields`, and `agents`.
3. The compile pipeline: passes, one planner, flattening once, and the reported substitutions.
4. Solvers by form, `step_count`, and the merged cell grids.
5. The module order, hoisting, the spelling, export hygiene, and the extended layout test.
6. The documentation, and the amendment of the constitution's Project Profile. Each earlier move has already rewritten
   the imports of the models, examples, tests, and documentation with the codemod, so the suite passes after every
   move. The Profile amendment covers:
   - the layout and the extension points;
   - the ecosystem contract, which no longer generates dataclass constructors.
7. The full gate. Then the active changes are updated.

Because the lowering of topology, renewals, fractional dynamics, control, kinetics, particles, lattices, fields, and
agents is buried in `_compiler.py`, the pass pipeline (phase 3) lands before those formalisms move. Quantum, rewriting,
and groups move in parallel with phase 3, because they already compile through whole-system passes or touch no
compiler code. Phase 4 and the layout checks of task 5.1 run in parallel with both.

Rollback is by branch: `development` is untouched until the gate passes.

## Open Questions

- Are `Transition` and `Emission`, each used by one model, general Markov-chain primitives, or a `markov` formalism?
  Either answer leaves the specs and the phases unchanged.
