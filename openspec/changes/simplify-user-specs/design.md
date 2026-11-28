# Design

## Context

Outside the engine, 74 spec classes are public. About 24 are built by users in the docs or examples. The rule
"variants are specs, algorithms private methods" and the constitution rule that every dataclass is a `BaseSpec` made
each variant and each group of values its own class. On 2026-11-28 the user replaced that rule with the scikit-learn
convention: variants are string options, records are plain parameters, tuples or dicts, validation lives in the
owner, results are arrays or named results, and a class exists only when it has behaviour.

## Goals / Non-Goals

**Goals:** fewer public classes, options a user can read in one signature, and errors that name the owner.

**Non-Goals:** the internal records, done in `privatize-engine-records`. The laws, which stay values. Any change of
behaviour or of results.

## Decisions

- **Proposed treatment.** The first task checks every public spec against the rule and records the final treatment
  here before any code changes. The starting proposal:

  | Treatment | Specs |
  |---|---|
  | String option on the owner | the matching specs (batch, continuous, iterative, partner, partner switch, queue, random search, stable), `WienerProcessSpec`, `TotalisticSpec`, `OuterTotalisticSpec`, `LinearRuleSpec`, `TableSpec`, `SamplerSpec`, `ObjectiveSpec`, `LinearObjectiveSpec`, `FiniteVolumeSpec`, `ScheduleSpec` |
  | Plain parameters, tuples or dicts | `AxisSpec`, `LayerSpec`, `VisualSpec`, `ScenarioSpec`, `ModuleSpec`, `PrototileSpec`, `GeometrySpec`, `AffineSpec`, `RequirementSpec`, `DrawSpec`, `MoveSpec`, `PartnerSpec`, `MatchSpec`, `DatasetSpec`, `ReactionSpec`, `StructureSpec`, `SymmetrySpec`, `HilbertSpec`, `SectorSpec` |
  | Kept, it has behaviour | `Observable`, `CallSpec`, `RoleSpec`, `TopologyRequestSpec`, `MatrixGroupSpec`, `PowerSeriesSpec`, `TensorSpec`, `SubshiftSpec`, the rewriting structures |
  | Private, no user reads it | `EntrySpec`, `SignatureSpec` |
  | Named results | `NetworkSnapshotSpec`, `AgentSnapshotSpec`, `StructureStateSpec` |

- **Options.** The option takes the name of what it chooses, such as `rule='stable'` or `noise='wiener'`, with the
  values in a module constant beside the owner, and the parameters of every variant are plain keywords of the owner.
  A keyword that does not apply to the chosen option raises in `_check`, as `RandomWalkSystem` does for `absorbing`
  in continuous time.
- **Tuples and dicts.** When an owner takes many records, such as the axes of a phase space or the scenarios of an
  experiment, it takes a list of tuples or a dict keyed by name, in the order of the old fields, as
  `ColumnTransformer` does. A single record becomes keywords of the owner.
- **Results.** A value returned to users is a frozen `*Result` dataclass, like `DefectsResult`, whose docstring
  lists its fields. scipy returns results the same way, and the analyses already do.
- **Values with behaviour keep `Spec`.** They are immutable mathematical values with methods, such as a matrix group
  or a ring of power series, so they keep deriving from `BaseSpec` and keep their names.
- **Constitution.** The rule on public values and the rule "variants are specs" are rewritten to the convention, and
  the version rises by a minor step.

## Risks / Trade-offs

- This is the largest breaking change since the core API redesign. Every example and guide page that builds a
  removed spec changes. The release notes list each old spec and its replacement.
- Many keywords on one owner can make a signature long. Where an owner would take more than about a dozen
  keywords, the first task flags it, and the owner may keep a spec for that group if the user agrees.
- String options lose the static type of a spec class. `Literal` types on the keywords and the checks in `_check`
  keep the errors early and readable.
