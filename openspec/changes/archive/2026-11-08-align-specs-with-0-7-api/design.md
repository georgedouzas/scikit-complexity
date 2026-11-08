# Design

## Context

The 0.7 code is the reference. Each stale requirement is rewritten to match it, keeping requirement and scenario names
so the history of each requirement stays traceable, except where a name itself is wrong.

## Decisions

- Rename only "Ready-made systems are registered factories", whose name states the removed design. Every other modified
  requirement keeps its name and its scenario names, with new steps.
- Remove "Listing systems imports no system" and "Results are fitted attributes" instead of rewriting them, since
  "Systems are discovered with all_systems" and "Analyses are functions of results" already state the current contract.
- The compiler scenarios use the automatic choice of the private compilers, since no ready-made system tunes a compiler
  of its own and `RewritingCompiler` is private.
- The data-driven-dynamics requirements map `DMDSurrogate`, `SINDySurrogate`, and `ReservoirSurrogate` to the functions
  `mode_decomposition`, `sparse_identification`, and `reservoir` of `skcomplex.core.analysis`, and `system_` to the
  `system` field of their frozen results.

## Risks

None beyond specs drifting again. A grep of the main specs for the removed names after archiving checks this change.
