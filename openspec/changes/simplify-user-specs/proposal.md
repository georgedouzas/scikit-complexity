# Proposal

## Why

Users pass and receive dozens of spec classes, such as `StableMatchingSpec`, `TotalisticSpec`, `AxisSpec` and
`ScenarioSpec`, most of them a few values with no behaviour. scikit-learn has almost none: a variant is a string option
on the class that owns it, such as `KMeans(algorithm='lloyd')`, a composite input is a tuple or a dict, such as
`ColumnTransformer(transformers=[(name, transformer, columns)])`, the owner validates its parameters when it is
fitted, and a separate class exists only when it has behaviour, such as a kernel or a CV splitter. This change applies
that convention to every public value of the library.

## What Changes

- **Variants become string options.** A spec that picks an algorithm, such as the matching specs, `WienerProcessSpec`,
  the rule specs `TotalisticSpec`, `OuterTotalisticSpec`, `LinearRuleSpec` and `TableSpec`, `SamplerSpec`, the
  objective specs, `FiniteVolumeSpec` and `ScheduleSpec`, becomes a string option plus plain parameters of its owner.
  The algorithm stays a private method of the owner.
- **Records become plain values.** A spec that only groups values, such as `AxisSpec`, `LayerSpec`, `VisualSpec`,
  `ScenarioSpec`, `ModuleSpec` and `PrototileSpec`, becomes plain parameters of its owner, or a tuple or a dict when
  the owner takes many of them.
- **Validation in the owner.** Every check a removed spec held moves to the `_check` of its owner, called by
  `prepare`, and the error names the owner and the parameter.
- **Classes with behaviour stay.** `Observable`, the laws, and the values with methods of their own, such as
  `MatrixGroupSpec`, `PowerSeriesSpec` and the rewriting structures, stay classes. Public specs no user reads, such as
  `EntrySpec`, become private.
- **Results are named results.** What a trajectory returns, such as `NetworkSnapshotSpec`, `AgentSnapshotSpec` and
  `StructureStateSpec`, becomes a frozen `*Result`, like the results of the analyses.
- **BREAKING:** every removed spec is replaced by options or plain values, with no alias.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `base-interfaces`: the rule for public values.

## Impact

- `core/components` (laws, interactions, spaces, elements), `core/experiments`, `core/engine/runtime`,
  `systems/general/matching`, `systems/general/rewriting`, the systems that build these values, their tests, the
  examples and the user guide, and the constitution.
- Lands after `privatize-engine-records`.
