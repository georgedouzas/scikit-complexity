# Design

## Context

Every defect was reproduced with public calls. The scripts are kept in the session scratchpad, outside the repository.
The faults are small and local, so each fix stays in the module that owns the check. See proposal.md for the list.

## Goals / Non-Goals

**Goals:**

- Each error names the part, the argument, and the value at fault, with the class the library already uses for that
  kind of failure.
- Each fix comes with a test that reproduces the defect first.

**Non-Goals:**

- The vectorised path of `basin_stability` still rejects systems of groups, and a replicated `ReactionInteraction` still
  fails its role check. Those are separate limits of replication, not of jumps, and they stay out of this change.
- No new error class.

## Decisions

1. **Domain message.** In `_coerce_value` of `passes/_layout.py`, the imaginary part message is used only when the value
   lies in `CC` and not in `RR`. Every real number lies in `CC`, which is why the old test sent all real values down that
   branch.
2. **Variant message.** `_check_variant` in `interactions/_base.py` prints the quoted names and, when spec kinds are
   accepted, adds the words "or a spec that the part handles". The pinned test in `test_base.py` changes with it.
3. **Order check.** The order check of rate and noise laws in `passes/_orders.py` runs before the early return that skips
   systems with no state of higher order and no lags. Moving the check, not copying it, keeps one place for it.
   `prepare` of a law cannot do this, since a law does not know the order of its state before compilation.
4. **Transfer ends.** `Source` and `Sink` become frozen dataclasses with the field `label`. They subclass `BaseSpec` if the
   source rules of `tests/test_sources.py` accept the names. Otherwise they stay plain frozen dataclasses, because
   renaming two public classes for a repr is not worth a break. Nothing in the library compares them by identity.
5. **Spaces.** `MeshSpace._prepare` checks that the points are a two-dimensional array of two or three rows, that the
   cells are a nonempty two-dimensional array of integers, and that every index lies below the number of points, all
   before skfem is called. Each failure is a `ValidationError` naming `points` or `cells`. In `NetworkSpace`, the two
   raw lookups of a layer reuse the existing check that raises `SpaceError` with the list of layers.
6. **UnregisteredLawError.** The plan pass raises it, as an issue with that error class, for a law whose class is in no
   key of the forms table and not among the formless laws. It comes before the `laws-mixed` issue, so an unknown law is
   named instead of being reported as a mix.
7. **ArgumentError.** It is removed. The constructors of the core raise Python's own `TypeError` for an undeclared
   argument, and catching that would need wrapper machinery the class rules forbid. The three raises in the ridesharing
   system are for values of the wrong kind, which is what `ValidationError` means. Its troubleshooting section and its
   uses in `tests/core/test_errors.py` go in the same commit, since the troubleshooting test requires a section for
   every error class.
8. **Integer starts.** The design that `basin_stability` draws from rounds a draw to an integer when the domain of the
   path holds only integers, and keeps it within the bounds. Real paths draw exactly as before, so seeded results of
   real systems do not change.
9. **Correlation step.** `dt` defaults to `None`. For a trajectory the step comes from its times, the way the other
   analyses read it. For an array, `None` means a step of 1. A `dt` given with a trajectory whose step differs raises an
   `AnalysisError`.

## Risks / Trade-offs

- [Removing `ArgumentError` breaks code that catches it] → It is listed as breaking in the proposal and the release notes.
- [Moving the order check could catch systems that ran before] → Those systems held a law of the wrong order and gave
  wrong derivatives. The full test suite shows whether any library system did.
- [Rounding integer draws makes the extreme bounds less likely] → The draw is rounded over the closed integer range, so
  both bounds are reachable.
