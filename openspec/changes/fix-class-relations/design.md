# Design

## Context

- **Five quantities, three roots:**
  - `Observable(BaseEstimator)` lives in `core/system`. Its docstring calls it a "named quantity of a system".
  - `Auxiliary`, `Invariant`, and `Objective` derive from `Law`, and are returned from a component's `laws`.
  - `Aggregate(Element)` holds a count or a reduction of a group in its state `value`. The scheduler of a
    co-simulation computes it.
- **`Wiener(Element)`:** it has no states. An interaction binds it to a role, and noise laws name its processes
  through that role.
- **`Rewriting(Interaction)`:** it binds a single `target` role and declares productions, so it acts on one
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

- **A quantity is a base.** `BaseQuantity` in `core/base/_quantity.py` holds a label and an expression of states and
  parameters. A quantity takes one of two forms:
  - an expression of the declaring component's symbols;
  - a reduction over the members of a group.

  The five quantities derive from it, and each adds only what the framework does with it:
  - `Invariant` adds a tolerance;
  - `Objective` adds a terminal value and a sense;
  - `Aggregate` adds its source group.
  - *Alternative:* keep the four law-like classes as laws with a trait that excludes them from the dynamics.
    Rejected because a trait cannot make a quantity a rule of change.

- **Quantities are declared next to laws.** A component's `laws(s)` returns laws only. Its auxiliaries,
  invariants, and objectives come from a second method, `quantities(s)`, with the same signature. Observables keep
  their home on the run, because a system accepts no argument about how it runs, and aggregates keep theirs on the
  co-simulation.

- **An interaction has its noise processes.** A system accepts no noise argument, so the correlated Wiener
  processes stay with the interaction whose noise laws draw them. `Wiener` derives from no component base and is
  bound through a role of `Interaction`, the role declaring that it takes noise processes.

- **A rewriting rule is a law.** `Rewriting` and `Rewrite` derive from `BaseLaw`, with the law kind `rewrite` of the
  rewriting formalism. A word, patch, or host graph returns them from `laws`.

- **A word problem is implemented, not solved by a solver.** The five classes become `FreeWordProblem`,
  `RewritingWordProblem`, `PermutationWordProblem`, `MatrixWordProblem`, and `AutomatonWordProblem`, and stay
  private.

- **An experiment has a schedule.** `_Schedule` becomes a plain class held by the task. The task applies the changes
  due at each frame through the simulation's frame hook, without registering an observer.

## Open Questions

- **The method name `quantities`.** It could instead be `declares` or `measures`. The maintainer picks the name
  before task 1.2.
- **Where noise processes live.** Bound through a role, they can be shared by several interactions. Owned by one
  interaction, they cannot. The maintainer decides before task 2.1.

## Risks / Trade-offs

- [Every model that declares an auxiliary, invariant, or objective changes.] → A codemod moves each of them from
  `laws` to `quantities`. The model tests then check that the runs are unchanged.
