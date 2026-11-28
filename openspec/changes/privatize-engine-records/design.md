# Design

## Context

An inventory of the 180 spec classes on 2026-11-28 found:
- 120 in `core/engine`, of which 79 in `problems`, 15 in `compile`, 7 in `solvers`, 2 in `backends` and 2 in
  `translators`.
- 81 private and 99 public. 43 public engine specs never appear in the docs or examples. The extend guide names no
  engine record.
- 24 with one reader. About 11 hold three fields or fewer, have no methods and have one reader.
- 85 with methods. Some, such as `OperatorSpec` (14 methods) and `LayoutSpec` (12), carry real behaviour.

The constitution says every dataclass outside the systems is a frozen spec deriving from `BaseSpec`, named with
`Spec`, which is the rule that produced one module per record.

## Goals / Non-Goals

**Goals:** internal records private, close to their users, as few as the data needs, and no checks on trusted data.

**Non-Goals:** changing what users pass or get back, which is `simplify-user-specs`. Changing the laws, which stay
values deriving from `BaseSpec`. Changing behaviour or speed of any run.

## Decisions

- **Which records stay public.** A record stays public when a solver or backend written outside the library must
  read it: `BaseProblemSpec` and the problem forms solvers receive, such as `ProblemSpec`, `OperatorProblemSpec`,
  `KineticProblemSpec`, `LearningProblemSpec` and `LinearProgramSpec`, and what they expose as fields, such as
  `KernelSpec`, `LayoutSpec` and `RunStateSpec`. The first task audits this list against the solvers and backends and
  records the final set in this design before any move. What a trajectory returns to users, such as
  `NetworkSnapshotSpec`, `AgentSnapshotSpec` and `StructureStateSpec`, is left to `simplify-user-specs`.
- **Private records.** A private frozen dataclass with no base, named for what it holds without `Spec`, defined in
  the module that builds it, or of its main reader when the builder is spread out. When two packages read it, it lives
  in the lower one, as the import contracts require.
- **Removal rule.** Three fields or fewer, no methods and one reader: the reader takes the values as arguments or a
  tuple. A record with behaviour is never removed only by this rule.
- **Arrays.** Where a record is one row of many of the same kind and the reader loops over them, the rows become
  arrays or a structured array. Where the records differ in kind, they stay records.
- **Validation.** Internal records are not checked. A check that today guards a user error moves to the `_check`
  of the public owner that receives the user's value, with the same message.
- **Packages.** `core/engine/problems` keeps one module per problem kind with its private records, and the modules
  of other records go to their users. The user approved these moves on 2026-11-28.
- **Constitution.** The rule on dataclasses becomes: a public value is a class only when it has behaviour or is part
  of the solver contract, and internal records are private frozen dataclasses in the module of their user. Its
  version rises by a minor step.

## Public contract (audit of 2026-11-28)

A solver is prepared with a problem form and run with a `StepSpec`, declares its support with a `SupportSpec`, and
records `BaseRecordSpec` events. A backend is run with `LoweredKernelsSpec` and reads `RunStateSpec`. The public set is
these, the forms, and the public records the forms hold as fields:

- `BaseProblemSpec` and the forms `ProblemSpec`, `OperatorProblemSpec`, `KineticProblemSpec`, `LearningProblemSpec`
  and `RewritingProblemSpec`, the last in `systems/general/rewriting`
- `StepSpec`, `SupportSpec`, `BaseRecordSpec`, `LoweredKernelsSpec`, `KernelSpec` and `RunStateSpec`
- the field records `CompiledElementSpec`, `DiffusiveLimitSpec`, `LayoutSpec`, `LinearTermSpec`, `LoopTermSpec`,
  `PartitionSpec`, `ReferenceSpec` and `TransferNetworkSpec`

Every other engine record becomes private. `PhaseSpec` and `PhasesScheduleSpec`, which systems build, and
`NetworkSnapshotSpec`, `AgentSnapshotSpec`, `FrameSpec` and `StructureStateSpec`, which runs return, are user values
and are left to `simplify-user-specs`. The component values the forms hold, such as `EntrySpec`, are also left to it.

## Order of the work

1. `compile`, then `solvers`, `backends` and `translators`, then the records outside the engine, in parallel, each
   only making its own records private and removing its trivial ones.
2. `problems`, which moves its records into the modules of the first wave.
3. The tests of the new requirements, the checks, and the full suite.

## Risks / Trade-offs

- A wide refactor touches many imports. Waves of at most four agents, each on one package, with lint-imports and the
  tests of the package after each wave, keep it reviewable.
- Making records private breaks code that imported them from outside. None is in the docs, so the risk is low, and the
  release notes list them.
- Structured arrays can make code harder to read than records. They are used only where the reader loops over many
  rows of one kind.
