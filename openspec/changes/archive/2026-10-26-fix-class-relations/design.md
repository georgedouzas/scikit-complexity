# Design

## Context

The names below are the ones this change gives the classes; today they lack their suffix, such as `Invariant`,
`Rewriting`, and `Wiener`.


- **Five quantities, three roots:**
  - `ObservableQuantity(BaseEstimator)` lives in `core/system`. Its docstring calls it a "named quantity of a system".
  - `AuxiliaryQuantity`, `InvariantQuantity`, and `ObjectiveQuantity` derive from `BaseLaw`, and are returned from a component's `laws`.
  - `AggregateQuantity(ElementComponent)` holds a count or a reduction of a group in its state `value`. The scheduler of a
    co-simulation computes it.
- **`WienerProcess(ElementComponent)`:** it has no states. An interaction binds it to a role, and noise laws name its processes
  through that role.
- **`RewritingLaw(BaseInteraction)`:** it binds a single `target` role and declares productions, so it acts on one
  component.
- **The group solvers:** `_FreeSolver`, `_RewritingSolver`, `_PermutationSolver`, `_MatrixSolver`, and
  `_AutomatonSolver` derive from `WordProblem`, a protocol for multiplying and comparing group elements.
- **The schedule:** `engine/experiments/_tasks.py` defines `_Schedule(Observer)`, which applies scheduled parameter
  changes.

## Goals / Non-Goals

**Goals:**
- Every class here derives from what it is, and is held by what has it.

**Non-Goals:**
- Changing what any of these classes computes.

## Decisions

- **A quantity is a base.** `BaseQuantity` in `core/base/_quantities.py` holds a label and an expression of states and
  parameters. A quantity takes one of two forms:
  - an expression of the declaring component's symbols;
  - a reduction over the members of a group.

  The five quantities derive from it, and each adds only what the framework does with it:
  - `InvariantQuantity` adds a tolerance;
  - `ObjectiveQuantity` adds a terminal value and a sense;
  - `AggregateQuantity` adds its source group.
  - *Alternative:* keep the four law-like classes as laws with a trait that excludes them from the dynamics.
    Rejected because a trait cannot make a quantity a rule of change.

- **Quantities are declared next to laws.** A component's `laws(s)` returns laws only. Its auxiliaries,
  invariants, and objectives come from a second method, `quantities(s)`, with the same signature. Observables keep
  their home on the run, because a system accepts no argument about how it runs, and aggregates keep theirs on the
  co-simulation.

- **Noise processes are bound through a role.** A system accepts no noise argument, so the correlated Wiener
  processes are bound by the interactions whose noise laws draw them, and several interactions may bind the same one. `WienerProcess` derives from no component base and is
  bound through a role of `BaseInteraction`, the role declaring that it takes noise processes.

- **A rewriting rule is a law.** `RewritingLaw` and `RewriteLaw` derive from `BaseLaw`, with the law kind `rewrite` of the
  rewriting formalism. A word, patch, or host graph returns them from `laws`.

- **A word problem is implemented, not solved by a solver.** The five classes become `FreeWordProblem`,
  `RewritingWordProblem`, `PermutationWordProblem`, `MatrixWordProblem`, and `AutomatonWordProblem`, and stay
  private.

- **An experiment has a schedule.** `_Schedule` becomes a plain class held by the task. The task applies the changes
  due at each frame through the simulation's frame hook, without registering an observer.

- **Decided by the maintainer.**
  - The method that returns a component's quantities is `quantities(s)`.
  - Noise processes are bound through a role, so several interactions can share one process and its correlations.
- **Names and form.** Every quantity ends in `Quantity`, the rewriting rules in `Law`, and the noise holder is
  `WienerProcess`. Quantities are values: frozen dataclasses whose only shared method is `check()`, like laws.

## Risks / Trade-offs

- [Every model that declares an auxiliary, invariant, or objective changes.] → A codemod moves each of them from
  `laws` to `quantities`. The model tests then check that the runs are unchanged.
