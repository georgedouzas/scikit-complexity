# Proposal

## Why

`comets` solves dynamic flux balance on a lattice by the direct approach with fixed steps, where COMETS alternates growth and
diffusion (Raman 2021, §11.1.3.4, p. 258; Harcombe et al. 2014). Its diffusion coefficients, inoculum, and threshold are
therefore not the published ones: Harcombe's values on 0.02 cm boxes come to about 45 boxes²/h for metabolites and 0.0027
for biomass, against 1 and 0.2 in the code, and an explicit solver at those values needs steps below about 0.008 h, which
breaks the gallery budget.

## What Changes

- Let a lattice with linear programs alternate its growth step and its diffusion step.
- Give `comets` Harcombe's coefficients, step of 0.1 h, and inoculum, and remove the unsourced activity threshold if the
  alternation no longer needs it.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `model-library`: `comets` runs with the published scheme and values.

## Impact

- `skcomplex.engine.solvers`, where `_split.py` and `_strang_splitting.py` already treat parts of a rate apart, and may hold
  most of what is needed. `skcomplex.models.biology.ecology._comets`, its tests, gallery example, and inventory entry.
- No new package and no new dependency.
