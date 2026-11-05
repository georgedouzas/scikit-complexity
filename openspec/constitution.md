# scikit-complexity Constitution

This file is seeded from a template. Refine it as the project develops. Run `openspec init` to add the OpenSpec
workflow alongside it, and `openspec/config.yaml` names this file as binding.

- A typed Python library MUST follow the engineering principles and code conventions stated here.
- The body MUST stay repo-agnostic, so it can be reused across projects.
- A repository that adopts the constitution MUST instantiate it in a single Project Profile at the end, naming the
  concrete framework contract, toolchain, dependencies, and delivery surfaces.
- A project MUST extend the constitution by growing its Project Profile, and MUST amend the body only where a new rule
  is genuinely general.
- Every rule MUST be stated with MUST or MUST NOT, and every rule binds equally.

## Engineering

This section states the principles the code obeys, the style it and its documentation are written in, the conventions
it follows, the tools it is built with, and the gate it passes before merge. Every rule here is binding, not advisory.
The conventions are generic Python and hold in any project. The automated gate checks the mechanical parts, and the
conventions are the taste it cannot check.

### Contracts & Types

- Public objects MUST conform to the contract of the framework they plug into, and MUST NOT invent a parallel
  convention that silently breaks downstream code. The concrete contract a project conforms to MUST be named in its
  Project Profile.
- Class initialization parameters MUST be stored unmodified under their own names, and state the object learns at
  runtime MUST be exposed only through the framework's convention for derived state.
- Behavior MUST be configured through explicit parameters, and MUST NOT depend on hidden global or ambient state, so
  an object is deterministic and testable in isolation.
- All code, public and internal, MUST carry complete type annotations, and MUST pass the static type checker with no
  new ignored errors.
- A name MUST be typed correctly rather than silenced, and a `# type: ignore` MUST be a last resort governed by the
  suppression rules in Documentation.
- Data that crosses a public boundary MUST be validated against an explicit, declared schema, and data-shape
  assumptions MUST NOT be enforced by ad-hoc runtime checks scattered through the code.

### Design

- The package MUST stay a library, and an application concern such as an interactive loop, a long-lived session, a
  choice of model or policy, or a credential MUST stay with the caller.
- A capability that needs a credential or performs a real-world side effect MUST live behind an optional extra, and
  MUST NOT ship in the default install.
- Where a project exposes several delivery surfaces, the surfaces MUST expose the same underlying capabilities, none
  MUST hold logic the others cannot reach, and a parity test MUST assert this.
- Functions MUST be small and do one thing, and a function that needs a paragraph of docstring body to explain its
  branches MUST be two functions.
- A function MUST return early with guard clauses, deep nesting MUST be avoided, and a helper MUST be extracted
  before the third level of indentation.
- An error message MUST be built in a variable and then raised, and it MUST tell the reader what to do, the variable
  that was missing, or the value that did not match.
- A specific named exception MUST be raised, and a bare `Exception` or `ValueError` MUST NOT stand where a named one
  carries meaning. Every named exception MUST be defined in one errors module of the shared-leaves package and
  re-exported from that package alone.
- An exception MUST NOT be caught and swallowed, and MUST be caught narrowly or left to propagate.
- A fact, a definition, or a derivation MUST live in exactly one place, and the same thing expressed twice MUST be
  collapsed to one. What can be derived from what is already kept MUST NOT be stored.
- A capability the codebase already has MUST be reused rather than reimplemented or copied into another module.
- A capability that a popular, mature, or performant library already provides MUST be taken from that library as a
  dependency rather than implemented in the codebase, and the codebase MUST hold only the adapter that lowers its own
  abstractions to the library. A capability MUST NOT be implemented in house unless no such library fits, and the
  change that adds it MUST record the libraries it surveyed and why each was rejected.
- A class MUST define every method its children share on their common parent, as a public method written once that
  calls a private hook the children override, and the parent MUST give a trivial hook where a child has nothing to do.
- A constructor MUST only store each argument unmodified under its own name. Every other attribute MUST end in an
  underscore and MUST be set by a named method, and checks MUST run in a named method, not at construction.
- A class MUST NOT define magic methods other than `__init__`, except the arithmetic operators of a type the user writes
  as mathematics. A collection MUST be read through named methods and properties, a callable object MUST expose a
  named method, and a value MUST be a frozen dataclass compared by its fields.
- A constant that two modules define the same way MUST be treated as one fact and lifted to the shared leaves. Two
  constants that share a value but not a meaning MUST stay apart, and the same holds for behavior, so only the
  fragment that is identical at every call site MUST be collapsed.

### Modules & Surface

- A module MUST read top to bottom in one order: a one-line imperative module docstring, the licence header, the
  imports grouped standard library, third party, first party, the module constants and type aliases, and then the
  functions. The linter sorts the imports, so they MUST NOT be sorted by hand.
- A module MUST add `from __future__ import annotations` only where a forward reference needs it, and MUST NOT guard
  an import with `if TYPE_CHECKING:`, which papers over a cycle the way an import inside a function body does. The
  cycle MUST be fixed instead.
- Functions MUST come in dependency order, so a name is defined before it is used, the small helpers first and the
  function the module exists for last.
- A module MUST order its definitions by kind first and privacy second, the private functions, the public functions,
  the private classes, then the public classes, and a class MUST order its private methods before its public ones.
  Where that order and dependency order disagree, dependency order MUST win.
- A module constant MUST be `UPPER_CASE`, MUST live in the constants block near the top of the module, and MUST NOT
  carry a leading underscore, whatever its reach.
- A base class MUST NOT carry a leading underscore, and MUST be kept out of the surface by not being re-exported,
  never by its name. A base a client is expected to subclass MUST be re-exported.
- Inheritance MUST mean *is* and membership *has*: a class MUST derive only from a class it is, and MUST hold what it
  merely has. A class MUST be named after what it is, so its name ends with the concept of the base it derives from,
  such as `RateLaw` of `BaseLaw`, and no base MUST derive from another base, except that the bases of the parts of a
  domain model MAY derive from one base of parts that defines what every part does.
- One module MUST be one concern, a file that grows two MUST be split, and small general helpers MUST share a
  `_utils` module. A base package MUST import no other package of the project, and its modules MUST import one
  another downward.
- The top level of a package MUST hold subpackages and its `__init__` only. The shared leaves, the type vocabulary,
  the shared constants, and the shared building primitives, MUST live in a `core` subpackage, and every import MUST
  run downward, from `core` to domain packages to surfaces, so no cycle can form.
- The only lazy import MUST defer an optional dependency, and MUST go through `importlib`, as
  `skcomplex.models.import_extra` does, so it needs no suppression.
- Implementation modules, classes, and helpers MUST be private, named `_name`.
- The package `__init__` MUST re-export the public surface with an explicit `__all__`, MUST carry only its docstring
  and those re-exports, and MUST be the only place a public name leaves a private module. A name MUST be re-exported
  once, where it lives, and a parent package MUST NOT re-export a subpackage's surface a second time.
- A public name used outside the module that defines it MUST be imported from the surface that re-exports it, and a
  module MUST NOT import another package's private name. A package and its subpackages are one package for this rule.
- A name its own package never re-exports MUST be private, a package MUST re-export only names something outside it
  uses, and a name nothing uses at all MUST be removed. The tests and the documentation are client code, so a name
  either of them reaches MUST stay re-exported.
- A surface package, one that exists to serve a runner rather than an importer, MUST re-export its entry point alone.
  The Project Profile names which packages these are.

### Style & Naming

- Every document, docstring, and example MUST follow one style, written for a reader who wants to understand, not to
  be impressed.
- Sentences MUST be of a natural length, in the normal order, subject then verb then object. Word order MUST NOT be
  inverted for effect, and the passive voice MUST NOT be used unless the doer is unknown or does not matter.
- Words MUST be plain, and an idiom, a metaphor, or a literary flourish MUST NOT appear.
- A real risk MUST be stated once, in the right place, plainly, with no hedging and no repeated warning.
- A document MUST carry headings and subsections so a reader can scan it, and a section that is a set of rules MUST
  be written as a bullet list with one rule per bullet.
- A line MUST be at most 120 characters, and a sentence MUST NOT use a semicolon or a dash as punctuation.
- A function name MUST begin with a verb and name what the function actually does or returns. The verb MUST be
  honest, so a function that loads or resolves an object is `load_` or `resolve_`, not `build_`, and an empty verb
  such as `process`, `handle`, or `manage` with no object MUST NOT be used.
- A name an outside contract fixes MUST keep the spelling that contract fixes, and the verb-first rule MUST NOT
  reach it.
- A method MUST be an action and begin with a verb, and a property MUST name a value as a noun phrase.
- State an instance derives at runtime MUST carry the framework's derived-state marker, in this ecosystem a trailing
  underscore. A public instance name MUST be a constructor parameter stored unmodified under its own name, or a
  derivation carrying the marker, and anything else an instance exposes MUST be private. A class-level constant that
  declares what a class is MUST be a `ClassVar`.
- A module MUST be named for the concern it owns, with a descriptive noun, and MUST NOT be named for the data it
  consumes or the surface that happens to call it. Names MUST come from the domain, used consistently, and MUST NOT
  collide with a dependency's concept.
- A constant MUST be named for what it holds, not for a role it happens to play, and every constant name MUST be
  read again and checked that it still describes its value.

### Documentation

- A user-facing behavioral change MUST update the affected documentation, and MUST add or amend a changelog entry
  where it changes public behavior.
- Every code example in the documentation and the docstrings MUST run, and the build MUST prove it. An example MUST
  NOT be a fragment, pseudo-code, or a demo that cannot run, and MUST exercise the thing it documents.
- Every module, class, and function, public or private, MUST carry a docstring. The summary line MUST be one line,
  imperative, saying what the thing does, and MUST NOT be meta narration such as `Implements the ...` or
  `This function ...`.
- A public function and a public class MUST carry an `Args` block documenting every parameter, a `Returns` block,
  and a `Raises` block naming every exception raised, omitting a block that would stand empty. A private function
  and a private class MUST carry the one-line summary alone.
- An abstract method MUST carry the complete docstring, and a method that overrides it MUST repeat it, changing only
  the summary line where the implementation is worth a word of its own.
- A command or a tool in a surface package MUST carry the one-line summary alone, since the runner shows that line
  and documents the parameters itself.
- A constructor parameter or a dataclass field MUST be documented under `Args`, and an `Attributes` block MUST hold
  learned state only.
- A docstring MUST describe what the thing is and what it holds, plainly. It MUST NOT state its virtues, give the
  rationale for its shape, say what downstream code builds from it, describe what the code does not do, or restate a
  self-evident name, and each point MUST be its own sentence.
- Public API MUST carry a runnable example checked by the doctest run, and a network-touching class MUST NOT. The
  top-level package `__init__` is the one exception to brevity, carrying a tagline and a short overview as the
  library's front page.
- Source MUST carry almost no comments, since the names say what and the docstring says why. An inline comment that
  explains the next line MUST be removed by fixing the line or its names. The only comments in source MUST be the
  licence header and, rarely, a suppression.
- A suppression, a `# noqa` or a `# type: ignore`, MUST be a last resort for a genuine one-off, and MUST carry the
  rule code and the reason. A suppression that recurs across the repository MUST be configured once in the project
  configuration as a scoped ignore instead of repeated inline.

### Tests & Gates

- Every behavioral change MUST ship with tests, a bug fix MUST include a regression test that fails before the fix,
  and new logic MUST NOT reduce the coverage of the module it touches.
- The suite MUST run with branch coverage, randomized ordering, and executable docstrings, so every code example in
  a docstring MUST be correct and runnable.
- The test tree MUST mirror the source tree, a test MUST be named `test_<function>_<behavior>`, and its docstring
  MUST be a single line.
- A test MUST use the public API the way a user does, and MUST NOT import a private name from a private module.
  Where it needs an internal, the internal wants to be public.
- A test MUST NOT reach the network, and MUST use a recorded payload, a fake, or a locally served page instead.
  Fixtures MUST be typed, small, and live in the nearest `conftest.py`.
- Code MUST pass the full automated gate before merge, covering formatting, linting, static type checking, docstring
  coverage, a security scan, and a dependency audit.
- The gate MUST run locally through pre-commit and again in CI, a red run MUST block merge, and failures MUST be
  fixed at the source.
- Work MUST happen on feature branches, and the release branch MUST stay green.
- Before opening a PR, a contributor MUST run the full gate in order, formatting, then checks, then the
  documentation build, then tests, and MUST resolve all findings. The documentation build MUST NOT be skipped, since
  it is what runs the examples.
- A release MUST follow semantic versioning, and MUST update the changelog before tagging.

### Toolchain & Credentials

- The project MUST declare its supported language versions, and MUST remain compatible across all of them.
- A new runtime dependency MUST be justified and declared in the project manifest, and MUST NOT be vendored ad hoc.
- A runtime dependency MUST have a release on the package index within the last 24 months, or the Project Profile MUST
  record why it may stay and when the exception is next reviewed. A proposal for a new one MUST state its last release
  date, and the limit MUST be checked before each release.
- The build MUST use a `src`-based layout with SCM-derived versioning, and generated version metadata MUST NOT be
  hand-edited.
- Canonical task-runner sessions for tests, checks, formatting, docs, and release MUST be the entry points the gate
  runs through.
- The style constants, the line length, the docstring convention, and the formatter options, MUST be set once in the
  project configuration, and MUST NOT be overridden by hand. The concrete versions, tools, and dependency list MUST
  live in the Project Profile.
- A credential MUST be named, and MUST NOT be passed. A function, command flag, or tool argument MUST take the name
  of the variable holding the secret, and MUST read it where it is used.
- A secret MUST NOT become an argument value, a log line, or a pickle.

## Governance

- This constitution supersedes ad-hoc conventions and prior undocumented practice, and applies to all code,
  documentation, and tooling changes in the repository that adopts it.
- An amendment MUST be proposed through a PR that edits this file, states the rationale, and updates the version. An
  amendment that adds or removes a principle or governance rule MUST carry the maintainer's approval.
- The constitution MUST carry its own semantic version. MAJOR MUST mark a backward-incompatible removal or
  redefinition of a principle or a structural rewrite, MINOR a new principle or materially expanded guidance, and
  PATCH a clarification or a non-semantic wording fix.
- Every PR and code review MUST verify adherence to this constitution, not just correctness. Every PR MUST state
  which principles it touches and MUST confirm the gates pass.
- A deviation MUST be justified in the PR, and an unjustified violation MUST block merge.
- Contributor-facing operational guidance MUST live alongside the code, in a `CONTRIBUTING.md` and developer docs,
  and MUST stay consistent with this constitution.
## Project Profile: scikit-complexity

This section instantiates the body above for this repository. It is the repo-specific part, and it is a starting point
to fill in as the project takes shape.

- Ecosystem contract: the scikit-learn estimator contract. Every configurable base derives from `BaseConfig` in
  `skcomplex.base`, a child of `sklearn.base.BaseEstimator`, and speaks two verbs: `prepare(context)` validates the
  stored arguments and derives every attribute ending in an underscore, and `run(inputs, final=False)`
  performs the family's one action and, when `final` is true, releases what `prepare` acquired. Every verb returns the
  object, and a run keeps what it produces in attributes ending in an underscore, one name per family, such as
  `laws_` for elements, interactions, and systems, `value_` for laws, `problem_` for compilers, `state_` for solvers,
  and `values_` for kernels. The verbs call the private hooks `_check`, `_prepare`, and `_run`, which `BaseConfig`
  declares abstract and every class MUST implement, returning `None`; `_release` is optional and does nothing by
  default. No family MUST add a verb of its own. Domains, problems, terms, and compiled records are values: frozen specs
  deriving from `BaseSpec`, the only base whose hooks have defaults, whose only hook they usually implement is `_check`. A system is
  defined by elements, interactions, and laws alone, its parts, which derive from `BasePart`. Elements and groups derive from
  `BaseElement`, and interactions from `BaseInteraction`, which both take the label and the values of the declared
  states, parameters, and roles in one explicit constructor and list them in `get_params`, through the declaration
  helpers they share. A system holds its elements, interactions, and nested systems, and is none of them. What a
  run records or checks, such as observables and invariants, is an argument of the run. An element or interaction
  type declares its fields with `state`, `parameter`, and `role`, and a constructor setting
  that is not a declaration is taken by an explicit keyword-only `__init__` of the type. No element class is turned
  into a dataclass or given generated attributes. Every law derives from `BaseLaw` and declares its kind, its time
  class, and its traits as class variables. The only magic methods beyond `__init__` are the arithmetic of the
  local operators.
- Language: Python >=3.12, <3.15.
- Platforms: Linux and macOS 13 or later on Apple silicon. Windows is not supported because passagemath publishes no
  Windows wheels, and Windows users run the package in WSL. macOS on Intel is not supported because `jaxlib` publishes
  no wheels for it.
- Build and tooling: PDM with SCM-derived versioning and a `src`
  layout. The `nox` sessions are `formatting`, `checks`, `tests`, `docs`, `gallery`, `changelog`, and `release`. The
  gate covers `ruff`, `mypy`, `bandit`, `pip-audit`, `deptry`, `pydoclint`, `interrogate`, and `import-linter`, whose
  contracts in `pyproject.toml` enforce the import rules of the layout, with `pytest` under branch coverage and
  randomized ordering. The test session runs the docstring examples through pytest's doctest collection and the Python
  blocks of the README and the user guide through `pytest-markdown-docs`. The documentation build checks the model
  inventory.
- Runtime dependencies: `numpy` 2.1 or later, `scipy` for numeric integration, `passagemath-symbolics` as the Sage
  symbolic layer every law is written in and lowered from, `passagemath-maxima` for exact solutions, `cysignals` at a
  version the passagemath wheels need, `numba` for the compiled backend, `jax` for the JAX backend and automatic differentiation, `optax` for the
  optimizers of the learning solver, `diffrax` for differentiable ODE solves of the learning solver, `highspy` for linear programs, `lammps` for the LAMMPS backend, `mesa` for agent-based models, `networkx`
  for networks, which Mesa 3.5 imports without declaring it, `xgi` for hypergraphs and simplicial complexes, `rerun-sdk` for visualization, `qutip` for operator dynamics, `SALib` for sensitivity
  analysis, `scikit-fem` for finite-element assembly on unstructured meshes, `shapely` 2.2 or later for the geometry of
  rewriting systems and the regions of agent spaces, `python-sat` for incremental satisfiability, `pymoo` for population optimisers, and `scikit-learn` for the
  estimator contract. The analysis layer adds `antropy` for regularity entropies and Lempel-Ziv complexity, `ordpy`
  for ordinal patterns, `nolds` for fluctuation analyses and Rosenstein's exponent, `powerlaw` for power-law fits,
  `ennemi` for nearest-neighbour transfer entropy, `mrestimator` for branching
  ratios, `pydtmc` for Markov state models, `dit` for discrete
  multivariate information and partial information decomposition, `igraph` for network motif counts, `pydmd` for
  dynamic mode decomposition, `pysindy` for sparse identification of nonlinear dynamics, `reservoirpy` for
  reservoir computing, and `passagemath-polyhedra` for exact enumeration of the generators of a cone. `pysd` reads Vensim and
  XMILE files for the translators of the engine. Rerun is a core dependency,
  and spawning a local viewer happens only on request, so it is not a real-world side effect.
- Exceptions to the 24-month limit on runtime dependencies: none. `pysd` 3.14.3, of 2025-03-23, is its last release and
  passes the limit until 2027-03-23, so its next review is due by 2027-03-23.
- Model gallery: the `gallery` dependency group holds `pillow` 11.0 or later for encoding the screenshots and
  `playwright` 1.50 or later for the web capture backend in headless Chromium, next to `mkdocs-gallery` in the `docs`
  group. The gallery tooling lives in `docs/gallery_tools`, and `nox -s gallery` runs the examples, captures their
  screenshots, and builds the documentation with them.
- Optional extras: `structures` holds `biopython` for the structures of the protein models, `rna` holds `ViennaRNA`,
  whose licence is not OSI approved, for RNA folds, and `trajectories` holds `mdtraj`, under the LGPL, for molecular
  dynamics trajectories.
- Dependency placement: a library that `skcomplex.core`, `skcomplex.engine`, or `skcomplex.analysis` uses MUST be a runtime dependency,
  whatever its licence, including copyleft licences such as that of `passagemath-symbolics`. An optional extra MUST
  hold only libraries that a model of `skcomplex.models` alone needs, and a model that needs one MUST raise a named
  error naming the extra. The extra `metabolism` holds `cobra`, `straindesign`, and `micom`.
- Symbolic layer: Sage is imported only through `sage.all__sagemath_symbolics`, `sage.all__sagemath_polyhedra`, and the
  modules they load, and the
  framework prints expressions to NumPy, Numba, and Lepton code itself. SymPy is installed as a dependency of
  passagemath, but importing it is banned by a `ruff` rule, and laws reject SymPy values.
- LAMMPS: `lammps` is a runtime dependency, with `mpich`, because the LAMMPS wheel links an MPI library it does not
  bundle.
- Layout: library. The importable package under `src/skcomplex` is distributed to others and holds five packages.
  `skcomplex.core` holds what a system is, in the order `base`, `spaces`, `laws`, `elements` and `interactions`,
  `system`, and `specs`, and nothing about how it runs. `skcomplex.engine` holds what runs a system, in the order
  `types`, `problems`, `backends`, `compile`, `solvers`, `runtime`, `viz`, and `translators`. `skcomplex.analysis` builds on the
  engine, and `skcomplex.experiments` on the analysis. What every package shares lives in the root modules
  `skcomplex.base` (`BaseConfig`, the root of every configurable class), `skcomplex.types`, `skcomplex.specs`, and
  `skcomplex.errors`, which import no package. Within `core` and within `engine`, a package imports only the packages
  listed before it. Every kind of dynamics, such as agents, lattices, fields, kinetics, operators, or particles, is no
  package of its own: its laws, elements, and spaces are parts of the core, and its passes, solvers, and visuals parts
  of the engine. Every name of `core` and `engine` is a mathematical, modelling, or engineering term, and none of
  them imports `skcomplex.models`. The test tree mirrors the source tree with no exceptions, so every test module
  tests one module of `skcomplex`, and no test module checks repository conventions, documentation tooling, or
  documentation data.
- General core: a core element or interaction class names a mathematical operation, configured by the roles it binds,
  the names of the states it reads and writes, and functions the user passes, and MUST NOT name a field quantity such
  as a force, a position, a price, an infection, or a harvest. A class enters the core only when it is a fundamental
  paradigm, such as a variational principle, a conservation law, a Markov jump process, a reaction network, or
  imitation dynamics, that applies in principle to other domains than the one that first needs it. A class whose
  operation agent models use takes `form='symbolic'` or `form='agent'`, and a matching algorithm is a frozen rule spec
  of `skcomplex.specs`.
- Model library: `skcomplex.models` is organised by scientific field and subfield, as
  `skcomplex.models.<field>.<subfield>`, and every model registers under the field path `<field>/<subfield>`. Models
  are examples built only from the public names of `core`, `engine`, `analysis`, and `experiments`. Every model
  package is self-contained, so no model package imports another, except that the packages of one field may import
  the code they share from `skcomplex.models.<field>.common.<family>`, such as `physics.common.particles` or
  `biology.common.populations`, which never lives in the core. A case of one field of a general core class is a thin
  subclass of it, even a pure renaming, which lives with the one model package that uses it, or in the `common`
  family of its field when several packages of the field use it. A model the core cannot express MUST be recorded as
  blocked by an OpenSpec change for the missing general capability, never worked around inside the model package.
  Models exist first to probe the core: a survey MUST record the primitives each model exercises and MUST implement
  models that exercise new primitives or combinations before models that repeat covered patterns. The import
  contracts enforce the import rules.
- Surface: the top-level package, `skcomplex.core`, and `skcomplex.engine` re-export nothing, so every public name is
  imported from the subpackage that owns it.
- Extension points: classes users subclass are the bases `BaseElement`, `Element`, `BaseInteraction`, `Interaction`,
  `NeighborInteraction`, `PairInteraction`, `HyperedgeInteraction`, and `BaseLaw`; the general interactions, such as
  `RateInteraction`, `GradientInteraction`, `EulerLagrangeInteraction`, `ConservationInteraction`,
  `CouplingInteraction`, `PairCouplingInteraction`, `MapInteraction`, `JumpInteraction`, `ReactionInteraction`,
  `TransitionInteraction`, and `SamplingInteraction`, and the agent interactions `ChoiceInteraction`,
  `ImitationInteraction`, `ReinforcementInteraction`, `EvolutionInteraction`, `MovementInteraction`,
  `TransferInteraction`, `LifeCycleInteraction`, `GroupingInteraction`, `MatchingInteraction`, `FeedbackInteraction`,
  and `TopologyInteraction`, whose field cases override a private hook; and in the engine `BaseBackend`,
  `BaseSolver`, `Solver`, `ExternalSolver`, `BaseCompiler`, `BaseClock`, `BaseObserver`, and `BaseVisual`. Law
  kinds, forms, space kinds, and traits are open vocabularies.

**Version**: 1.13.1 | **Ratified**: 2023-01-01 | **Last Amended**: 2026-11-02
