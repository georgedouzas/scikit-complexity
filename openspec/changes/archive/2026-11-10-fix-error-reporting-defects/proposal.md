# Proposal

## Why

While the documents were rewritten, the writers found errors that mislead the reader or never fire. Each was
reproduced with public calls before this change was written. A user who gives a negative value to a nonnegative state
is told it has an imaginary part. A bad mode is answered with a list that names `BaseSpec`. A law of the wrong order
passes when no other state has a higher order. Spaces leak raw `IndexError` and `KeyError` from deep inside. Two
error classes are documented but never raised where their names promise. Basin stability draws fractional starts for
count states, and `correlation` reports its times in samples, not in the time units of the run.

## What Changes

- A value outside a real domain is reported as outside that domain. The message about an imaginary part is kept for
  values that really have one.
- An unknown variant names only the accepted values, and describes an accepted spec in words.
- The order of a rate law or a noise law is checked against the order of its state in every system, not only in
  systems that hold a state of higher order.
- `Sink` and `Source` become values that print as `Sink(label='environment')` and `Source(label='environment')` and compare by their fields.
- `MeshSpace` checks the shape of its points and cells and the range of the cell indices when prepared, and an
  adjacency neighbourhood that names a missing layer of a network raises the existing `SpaceError` that lists the
  layers.
- `UnregisteredLawError` is raised when a law class has no form the engine knows, naming the class.
- **BREAKING** `ArgumentError` is removed. The core has no place where it applies, and its only users, in the
  ridesharing system, raise `ValidationError` instead.
- Basin stability draws integers for states whose domain holds integers.
- `correlation` of a trajectory reads the step from the times of the trajectory. For an array the step is still
  given by `dt`.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `system-validation`: errors of values, variants, law orders, transfer ends, spaces, and law forms name the part and
  the argument at fault, and the two unused error classes are resolved.
- `experiments`: basin stability starts count states from integers.
- `analysis-interface`: an analysis of a trajectory reports times in the units of the trajectory.

## Impact

- Code in `core/engine/compile/passes` (`_layout.py`, `_orders.py`, `_plan.py`), `core/components/interactions/_base.py`,
  `core/components/laws/_transfer.py`, `core/components/spaces` (`_mesh.py`, `_network.py`), `core/errors.py`,
  `core/experiments/_designs.py`, `core/analysis/_correlation.py`, and
  `systems/engineering/traffic/_ridesharing.py`.
- Tests beside each, one pinned message in `tests/core/components/interactions/test_base.py`, and
  `tests/core/test_errors.py`.
- The troubleshooting page loses the `ArgumentError` section and states that `UnregisteredLawError` is raised.
- Removing `ArgumentError` breaks code that catches it. The release notes of the next release say so.
