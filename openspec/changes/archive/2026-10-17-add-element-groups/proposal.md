# Proposal

## Why

Every element is compiled on its own, so compile time and step cost grow with the number of elements. An ideal gas of
2000 molecules takes 62 seconds to compile and runs five times slower than real time, and the Rerun observer logs one
entity per element. Gases, crowds, cell populations, and markets need thousands to millions of identical entities.

## What Changes

- Add element groups: N identical elements of one type stored as arrays with a leading dimension of N, created with
  `ElementType.group(label, size, **values)`.
- Let an interaction role bind to a group, so its laws apply to every member with one compiled expression.
- Add pair interactions, whose energy or rate is written once for a pair of members and summed over all pairs within
  a cutoff, inside one group or between two groups.
- Add a `Boundary` law kind that maps states after every solver step, for reflecting walls and periodic boxes, and
  combines with any form.
- Compile a group once, independent of its size, and evaluate its laws with array operations.
- Name group variables by group path, such as `gas.x` with shape `(N, 2)`, in problems, frames, trajectories, and
  parameter updates.
- Add group visuals that log a whole group as one Rerun batch, with optional colors from a state.
- Add a reflecting box, a periodic box, and an ideal gas builder to the physics module.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `system-modeling`: element groups, group roles, pair interactions, and boundary laws.
- `model-compilation`: size-independent compilation of groups and group naming.
- `simulation`: trajectories and parameter updates for group variables.
- `rerun-visualization`: one batch per group.
- `model-library`: reflecting box walls and an ideal gas.

## Impact

- `skcomplex.model`, `skcomplex.compile`, `skcomplex.runtime`, `skcomplex.viz`, and `skcomplex.models.physics`.
- Existing single elements keep working unchanged, as groups of size one.
- The changes `add-numba-acceleration`, `add-lammps-backend`, and `add-mesa-agents` build on groups.
