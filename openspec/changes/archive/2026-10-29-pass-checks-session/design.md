# Design

## Decisions

### A spec lives at the lowest layer that names its types
A spec whose fields are built only from root types stays in `skcomplex.specs`, such as `StateSpec`, `VisualSpec`, and `SupportSpec`. A spec whose fields name a class of a package moves to that package's `specs` module, one module per spec:
- **`skcomplex.core.specs`:** fields that name core classes, such as `NodeSpec` and `QuantumViewSpec`.
- **`skcomplex.engine.problems`:** fields that name engine classes, such as `ProblemSpec`, the reference and kernel specs, and the compilation specs. A separate `engine.specs` package would form a cycle with `engine.problems`, because kernels take problems and problems hold kernels, so these specs join the package of what a system compiles to. `BoundRenewalSpec`, which names a compile law, lives in `skcomplex.engine.compile`.
- **`skcomplex.experiments.specs`:** fields that name experiment classes, such as the task and scenario specs.

Every spec still derives from `BaseSpec`, so the root is the base of all of them. The layering is unchanged: a package imports the specs of its own layer and below.

The alternative, keeping every spec in the root with fields typed by root bases and Protocols, was rejected by the user, because it loses the field types or adds a Protocol per class.

### Cycles are broken by structure, not by hidden imports
Three import cycles appeared once the type-checking imports became real ones. Each is broken by putting the data where the layers allow:
- **Partitions:** `partitions` belongs to `ProblemSpec`, and `PartitionSpec` holds a `BaseProblemSpec`. Code that reads the partitions of any problem uses the reader `partition_problems`, which is empty on a problem of one form.
- **Compile:** the compilation keeps the state of the lowerings between passes as data, not the lowering objects. It keeps the `DrawKernel` that the draw pass builds, and a `TransferStateSpec` that the pass expanding flows leaves for the pass lowering them. A lowering refreshes its own state through the `_replace_specs` hook.
- **Experiments:** the alias of an experiment output lives in the module of the task spec.

### Agent views are typed per group
A rule reads and sets the states of an agent as attributes of its view, which the stepper adds per group at run time. Each agent model declares a view class of its group, such as `class WordView(AgentView)` with an annotation per state and parameter. Its rules annotate their agents with that view, so mypy checks the names without magic methods.

### `BaseConfig` is generic in its context and input
`class BaseConfig[ContextT, InputT]` types its verbs and hooks:
- `prepare(context: ContextT | None = None)` and `_prepare(context: ContextT | None)`;
- `run(inputs: InputT | None = None, final=False)` and `_run(inputs: InputT | None)`.

A base names its types once, as in `class BaseSolver(BaseConfig[ProblemSpec, StepSpec])`, and its children inherit them. PEP 695 syntax keeps the declaration plain. The project supports Python 3.12, so the parameters take no defaults, and a child that names no types is unchecked in its verbs.

### `_get_laws` returns laws
The parts declare `_get_laws(self, namespace: SimpleNamespace) -> Sequence[BasePart]` at `BasePart`, which every law is; the core cannot name `BaseLaw` below the laws. Elements and interactions return their laws as lists, which `Sequence` accepts.

### Suppressions become fixes
Each kind of suppression has a standard fix:

| Suppression | Fix |
| --- | --- |
| `ANN401` (`Any`) | the real type, or `object` where any value is accepted |
| `PLR2004` | a named constant |
| `PLR0917` | keyword-only arguments |
| `type: ignore` | a correct annotation, an `isinstance` narrowing, or a restructuring |
| `nosec` in tests | plain test code that needs none, or a narrowed bandit configuration for the tests directory |

Each package is done by one agent in its own worktree, so the agents never edit the same file.

## Risks / Trade-offs

- **Moved specs break their import paths.** The library has no released users of them, and a script rewrites every import.
- **The work touches every package.** Formatting runs once first, then the packages are worked in parallel and merged one at a time.
