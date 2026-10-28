# Design

## Decisions

### A spec lives at the lowest layer that names its types
A spec whose fields are built only from root types stays in `skcomplex.specs`, such as `StateSpec`, `VisualSpec`, and `SupportSpec`. A spec whose fields name a class of a package moves to that package's `specs` module, one module per spec:
- **`skcomplex.core.specs`:** fields that name core classes, such as `NodeSpec` and `QuantumViewSpec`.
- **`skcomplex.engine.specs`:** fields that name engine classes, such as `ProblemSpec`, the reference and kernel specs, and the compilation specs.
- **`skcomplex.experiments.specs`:** fields that name experiment classes, such as the task and scenario specs.

Every spec still derives from `BaseSpec`, so the root is the base of all of them. The layering is unchanged: a package imports the specs of its own layer and below.

The alternative, keeping every spec in the root with fields typed by root bases and Protocols, was rejected by the user, because it loses the field types or adds a Protocol per class.

### `BaseConfig` is generic in its context and input
`class BaseConfig[ContextT = object, InputT = object]` types its verbs and hooks:
- `prepare(context: ContextT | None = None)` and `_prepare(context: ContextT)`;
- `run(inputs: InputT | None = None, final=False)` and `_run(inputs: InputT)`.

A base names its types once, as in `class BaseSolver(BaseConfig[ProblemSpec, StepSpec])`, and its children inherit them. PEP 695 syntax and PEP 696 defaults keep the declaration plain.

### `_get_laws` returns laws
The parts declare `_get_laws(self, namespace: SimpleNamespace) -> Sequence[BasePart]` at `BasePart`, which every law is. Elements and interactions return their laws as lists, which `Sequence` accepts.

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
