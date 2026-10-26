# Design

## Context

- **Where the bases are today:**

  | Base | Module | Derived classes |
  |---|---|---|
  | `Component` | `core/components/_component.py` | `Element`, `Group`, `Interaction`, `System` |
  | `Interaction` | `core/interactions/_interactions.py`, its only module | `Potential`, `PairPotential`, `Rewriting`, and about 210 model and test interactions |
  | `Law` | `core/laws/_law.py` | 34 laws |
  | `Space`, `Time` | `core/space/_spaces.py`, `_time.py` | 6 spaces, 3 times |
  | `Stepper` | `engine/solvers/_stepper.py` | 37 solvers, among them `ExternalBackend` |
  | `Backend` | `engine/compile/_backends.py` | `NumpyBackend`, `NumbaBackend` |
  | `Problem` | `engine/compile/_problem.py` | `KineticProblem`, `QuantumProblem`, `RewritingProblem` |
  | `Observer`, `Visual`, `Clock`, `Design`, `Surrogate`, `Pass` | `engine` packages | the observers, visuals, clocks, designs, surrogates, passes |

- **What `Component` gives each class:**
  - `Element`, `Group`, and `Interaction` use the label, the declared variables, the declaration-based constructor,
    `get_params`, `define`, and `laws`.
  - `System` uses only the label, and takes `components=` and `interactions=` as separate lists.
- **`Element` adds** the parts of a member type: `define`, `group`, `lattice`, `visual`, `remembers`, and
  `requires`. `Group` holds the members of one element type.
- **Subclass names:** about 400 model and test element types and 210 interactions name the thing, not what it is,
  such as `MinSkeleton(Element)`. So do framework classes, such as `Rate(Law)`, `RK4(Stepper)`, and `Mesh(Space)`.
- **Engine bases:** `Observer`, `Visual`, `Pass`, and `Surrogate` are typed over runtime and compiler types:
  frames, trajectories, compiled elements, compilations, and Rerun.

## Goals / Non-Goals

**Goals:**
- `core.base` holds the bases of the framework and the helpers they are declared with, in as few modules as the
  concepts allow.
- Every base is something no other base is, and relates to the others only by having them.
- Every class derives from what it is, and its name ends with the name of what it is.
- The framework defines the fewest classes that express its concepts.

**Non-Goals:**
- Changing what a class does, beyond detaching `System` and `Interaction` from components.
- The classes that the change `fix-class-relations` reworks keep their names here. These are `Aggregate`, `Wiener`,
  `Auxiliary`, `Invariant`, `Objective`, `Energy`, `Observable`, `Rewriting`, `Rewrite`, the word problems, and the
  experiment schedule. That change names them.

## Decisions

- **The bases and their parents.** Inheritance means *is*, and membership means *has*:

  ```text
  BaseEstimator
  ├── BaseComponent      a part of a system            → ElementComponent, GroupComponent
  ├── BaseInteraction    a relation between parts      → PotentialInteraction, PairPotentialInteraction, ...
  ├── BaseSpace          where the parts live          → ContinuousSpace, LatticeSpace, NetworkSpace, ...
  ├── BaseTime           how time advances             → ContinuousTime, DiscreteTime, StaticTime
  ├── BaseSolver         advances a problem            → Solver, then RK4Solver, GillespieSolver, ...
  ├── BaseBackend        writes numeric code           → NumpyBackend, NumbaBackend
  └── System             the whole, the one concrete system
  BaseLaw                a rule of change              → RateLaw, UpdateLaw, JumpLaw, ...
  BaseDomain             the values of a variable      → NumericDomain, CategoricalDomain, TensorDomain, ...
  BaseProblem            a compiled system             → Problem (then KineticProblem), QuantumProblem, RewritingProblem
  ```

  | Has | |
  |---|---|
  | `System` | components, interactions, subsystems, one space, one time |
  | `BaseComponent` | a label, declared states and parameters, each with a `BaseDomain`, and laws |
  | `BaseInteraction` | a label, roles bound to components, declared parameters, and laws |
  | `BaseProblem` | the form, the layouts, and the initial vectors of a compiled system |
  | `BaseSolver` | the `BaseProblem` it advances |

- **The fewest classes.**
  - **No `BaseSystem`.** There is one kind of system, so `System` stays a single concrete class in `core/system`.
  - **No separate `Interaction`.** There is one interaction family, so users subclass `BaseInteraction` itself.
    `core/interactions` then holds nothing and is removed.
  - **`ElementComponent` and `GroupComponent` stay apart.** An element type declares a member, and a group holds
    members of one type. Neither is the other.
  - *Alternative:* `BaseSystem` and an `Interaction` under `BaseInteraction`, for symmetry. Rejected because each
    would be a base with a single class under it.

- **A subclass name ends with what it is.** A class whose ancestry reaches a base, or a named concrete root such as
  `Problem`, `Solver`, `System`, or `Schedule`, ends with that concept's name. This holds in `core`, `engine`,
  `formalisms`, `models`, `tests`, and `docs`, and for the names given to `define`.
  - The name before the suffix says what kind of that thing it is, so it never names another concept. A solver is
    named after its method (`RK4Solver`, `GillespieSolver`), and a problem after what it holds (`QuantumProblem`).
  - Where the old name already ends with the concept, it stays (`LatticeSpace`, `NumbaBackend`).
  - Where it names something else, it is corrected:
    - `ExternalBackend` becomes `ExternalSolver`;
    - `GroupSpace` becomes `CayleySpace`, a space of group elements;
    - `ParticleKinetics` becomes `KineticSolver`;
    - `Rewriter` becomes `RewritingSolver`.

  | Framework rename | |
  |---|---|
  | components | `Element` → `ElementComponent`, `Group` → `GroupComponent`, `Field` → `FieldComponent`, `Particle` → `ParticleComponent`, `Word` → `WordComponent`, `Patch` → `PatchComponent`, `HostGraph` → `HostGraphComponent` |
  | interactions | `Interaction` → `BaseInteraction`, `Potential` → `PotentialInteraction`, `PairPotential` → `PairPotentialInteraction` |
  | laws | `Rate`, `Noise`, `Jump`, `Update`, `Transition`, `Emission`, `Boundary`, `Event`, `Renewal`, `Equation`, `Transfer`, `Rule`, `Match`, `Flux`, `FractionalNoise`, `Encounter`, `Proliferation`, `Destruction`, `LocalRule`, `Drive`, `Relaxation`, `Move`, `Sampler`, `Sample`, `Extremal`, `Generator`, `Dissipator`, `Production`, `Substitution`, `GraphProduction`, each with `Law` appended |
  | spaces | `Mesh` → `MeshSpace`, `GroupSpace` → `CayleySpace` |
  | domains | `Domain` → `NumericDomain`, `Categorical` → `CategoricalDomain`, `Tensor` → `TensorDomain`, `HilbertSpace` → `HilbertDomain`, `HilbertFamily` → `HilbertFamilyDomain`, `Structure` → `StructureDomain`, `WordStructure` → `WordDomain`, `PatchStructure` → `PatchDomain`, `GraphStructure` → `GraphDomain` |
  | solvers | `Stepper` → `Solver`, `SDEScheme` → `SDESolver`, `CaputoStepper` → `CaputoSolver`, `ConvolutionStepper` → `ConvolutionSolver`, `LatticeStepper` → `LatticeSolver`, `RenewalStepper` → `RenewalSolver`, `MesaStepper` → `MesaSolver`, `_SplitStepper` → `_SplitSolver`, `ExternalBackend` → `ExternalSolver`, `LammpsBackend` → `LammpsSolver`, `ParticleKinetics` → `KineticSolver`, `Rewriter` → `RewritingSolver`, `UnitaryEvolution` → `UnitarySolver`, `LindbladEvolution` → `LindbladSolver`, `L1Scheme` → `L1Solver`, and `Solver` appended to `CoSimulation`, `Gillespie`, `Iteration`, `ScipyODE`, `RK4`, `VelocityVerlet`, `EulerMaruyama`, `Milstein`, `Heun`, `Collocation`, `Pontryagin`, `IMEXEuler`, `IMEX2`, `CrankNicolson`, `ETDRK4`, `SSPRK`, `FractionalEulerMaruyama`, `GrunwaldLetnikov`, `FractionalABM`, `MonteCarlo`, `QuantumTrajectories` |
  | engine bases | `Observer` → `BaseObserver`, `Visual` → `BaseVisual`, `Clock` → `BaseClock`, `Design` → `BaseDesign`, `Surrogate` → `BaseSurrogate`, `Pass` → `BasePass` |
  | engine subclasses | `Recorder` → `RecorderObserver`, `ConservationMonitor` → `ConservationObserver`, `InvariantMonitor` → `InvariantObserver`, `AsFastAsPossible` → `AsFastAsPossibleClock`, `RealTime` → `RealTimeClock`, `Grid` → `GridDesign`, `Rows` → `RowsDesign`, `LatinHypercube` → `LatinHypercubeDesign`, `_Sampled` → `_SampledDesign`, `ModeDecomposition` → `ModeDecompositionSurrogate`, `ReservoirModel` → `ReservoirSurrogate`, `SparseModel` → `SparseSurrogate`, `Phases` → `PhasesSchedule` |

  The model library, the tests, and the documents follow the same rule. For example, `MinSkeleton(Element)` becomes
  `MinSkeletonComponent(ElementComponent)`, and a `Predation(Interaction)` becomes
  `PredationInteraction(BaseInteraction)`.

- **`core/base` in few modules, fundamental apart from auxiliary.**

  | Module | Holds | |
  |---|---|---|
  | `_components.py` | `BaseComponent`, `BaseInteraction` | fundamental: the parts and relations of a system |
  | `_laws.py` | `BaseLaw`, `register_laws`, `law_classes`, `is_registered`, `has_trait` | fundamental |
  | `_domains.py` | `BaseDomain`, `NumericDomain`, `CategoricalDomain`, `TensorDomain` | fundamental |
  | `_spaces.py` | `BaseSpace`, `BaseTime` | fundamental: where and when |
  | `_problems.py` | `BaseProblem`, `Layout`, `Entry` | fundamental |
  | `_solvers.py` | `BaseSolver` | fundamental |
  | `_variables.py` | `state`, `parameter`, `role`, `Variable`, `VariableKind`, and the declaration helpers | fundamental: how parts declare |
  | `_symbolic.py` | `symbol`, `vector`, `Expression`, `SymbolicVector`, and the symbol helpers | fundamental: what laws are written in |
  | `_kinds.py` | `Term` and every kind: `LawKind`, `LawClass`, `TimeKind`, `Trait`, `SpaceKind`, `ComponentKind`, `DomainKind`, `Binding`, `Requires`, `Form`, `Supports`, `OrderKind`, `NoiseKind`, `NUMERIC_DOMAINS`, `vocabulary` | vocabulary |
  | `_errors.py` | the error hierarchy | auxiliary |
  | `_utils.py` | `Registry`, `VisualSpec`, `wrap_angle`, `Vector`, `Indices` | auxiliary |

  Today's `_terms.py`, `_support.py`, `_forms.py`, `_registry.py`, `_visuals.py`, and `_domains.py` merge into
  these. The modules import in the order `_errors`, `_utils`, `_kinds`, `_domains`, `_symbolic`, `_variables`,
  `_spaces`, `_laws`, `_components`, `_problems`, and `_solvers`. `core.base` imports no other package of
  `skcomplex`. The touched packages merge too: `core/interactions` is removed, and `core/space` and
  `core/components` keep only their concrete classes.

- **The engine bases stay in the engine.** `BaseObserver`, `BaseVisual`, `BasePass`, and `BaseSurrogate` are typed
  over frames, trajectories, compiled elements, compilations, and Rerun, which the engine's runtime and compiler
  define. Each sits first in its package's `_base.py`, or in the module that already holds it when the package has
  one module of bases. `BaseClock` and `BaseDesign` go with them, because pacing a run and designing an experiment
  are runtime concepts.

- **`BaseProblem` is what every solver and the runtime read:**
  - It holds `form`, `states`, `parameters`, `initial_state`, `initial_parameters`, and `time`.
  - The runtime runs any problem through these fields. It reads the compiler's other fields, such as `elements`,
    `agents`, and `topology`, only of a `Problem`, after checking that the problem is one.
  - The quantum and rewriting problems derive from `BaseProblem` alone, with the fields they use.
  - `KineticProblem` stays a `Problem`, because ordinary compiler passes build it, and it adds the encounters.

- **What a system compiles to is a core package.** The compiler's problem and the representation its numeric
  functions are written from, sit in a new package, `core/problems`, after `system`:
  - `Problem`;
  - the kernels;
  - the reductions;
  - the neighbourhoods;
  - the printing of expressions;
  - the lowering a backend returns;
  - `BaseBackend` with the registry of backends.

  They depend on laws, spaces, components, and systems, so they cannot sit in `core/base`, which imports nothing
  above it. `engine/compile` keeps what works on a system: the passes, the validation, and the planner.
  - Today `_load_backends` imports `skcomplex.engine.backends` at run time. Instead, the engine registers a loader
    on the backend registry, the way the model catalog does on the model registry. Then no core module reaches into
    the engine.
  - *Alternative:* the whole closure in `core/base`. Rejected because the base would then import every core
    package.
  - *Alternative:* `BaseBackend` in `engine/compile`. Rejected by the maintainer, who wants the backend base in the
    core.

- **A system compiler is not a pass.**
  - A pass is a step of one compilation: it reads and writes the fields of the compilation through `run`.
  - A system compiler claims a whole system of a formalism and compiles it to a problem in place of the pipeline,
    through `claims` and `compile`.
  - So `SystemPass` becomes `BaseSystemCompiler`, a plain class unrelated to `BasePass`. Its subclasses become
    `QuantumCompiler` and `RewritingCompiler`, with `DecayCompiler` in the tests and documents.

- **`BaseSolver` is the interface, and `Solver` adds the run machinery.**
  - `BaseSolver[P: BaseProblem]` declares `form`, `forms`, `supports`, and the flags. It defines the capability
    check, and the abstract `initialize`, `advance`, `plan`, and `initial_state`.
  - `engine/solvers` keeps `Solver(BaseSolver[Problem])`, with events, renewals, delays, and the solver registry.
    Every solver of the compiler's problem derives from it, `KineticSolver` among them. The quantum and rewriting
    solvers derive from `BaseSolver` of their own problem.

- **A domain is a base.** `BaseDomain` declares `domain_kind`, `sage_domain`, `contains`, `encode`, and `decode`,
  with methods that raise `NotImplementedError` rather than `ABC`, so the `NumericDomain` enumeration can derive
  from it. It declares no `value`, which would hide the value of an enumeration member. `AnyDomain` is
  removed, and validation reads `domain.domain_kind`.

- **Errors stay in one hierarchy, except those of one formalism.** The shared errors stay in `_errors.py`. An error
  that only one formalism raises moves to that formalism: `HilbertSpaceError` to `quantum`, `RewritingError` to
  `rewriting`, `WordProblemError` to `groups`, and `HorizonError` to `fractional`.

- **No alias keeps an old name.** A codemod in the scratchpad performs every rename and move; it never enters the
  repository.

- **Tests follow their modules.** The tests of every moved or merged module move to the mirror of its new path.

## Risks / Trade-offs

- [About 2,500 class names change across the library, its tests, and its documents.] → One codemod builds the
  rename map from the class graph. It refuses a rename whose new name collides with an existing one, and applies
  the map by whole words. Then `ruff`, mypy, the import contracts, and the tests of the touched packages check the
  result.
- [User code breaks on every name.] → The package is pre-1.0 and keeps no aliases, as after the earlier split.
- [Active changes name the old classes.] → Each one updates its text when it is applied.
