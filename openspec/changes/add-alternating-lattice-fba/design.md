# Design

## Context

The source check of `add-constraint-based-models` recorded the numbers in the proposal. Raman's step (ii) diffuses metabolites
only and alternates with growth; Harcombe also diffuses biomass, at 3e-9 cm²/s, or 3e-10 for E. coli colonies, with
metabolites at 5e-6 cm²/s, steps of 0.1 h, and 3e-7 g at the centre of a 50 by 50 lattice. Only summaries of Harcombe were
read, so the first task reads the full paper if it can be had, and otherwise states that the values come through a summary.

## Goals / Non-Goals

**Goals:** the published scheme and values, within the gallery budget.

**Non-Goals:** a general operator splitting for every law, or changes to `dynamic_fba`.

## Decisions

### Reuse before adding

The first task reads `SplitSolver` and `StrangSplittingSolver`, which split the linear part of field rates from the rest. If
the program's growth can be the remainder, no solver is added. If the linear program pass cannot run inside them, a small
stepper is added next to them, with exact or implicit diffusion, so the step is not bound by the diffusion stability limit.

### Biomass diffusion stays

The spec keeps biomass diffusion, which Harcombe has and Raman's short description leaves out. The docstring says so.

## Risks / Trade-offs

- [The front moves about 0.1 box/h at the published values, so a short run shows little] → The gallery example runs long
  enough to show it, or a smaller lattice, and states which; if the budget cannot hold, the change pauses for a decision.
