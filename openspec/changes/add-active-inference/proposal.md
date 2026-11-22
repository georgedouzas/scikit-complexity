## Why

Heins et al. derive schooling, milling, and swarming from agents that minimise the variational free energy of a model
of their neighbours (2307.14804). Each agent senses the mean distance to its neighbours in a few visual sectors and its
rate of change, holds Gaussian beliefs about these distances in generalised coordinates of motion, updates the beliefs
by descending the free energy, and turns its heading by descending the same free energy through how its heading
changes what it senses. The social forces of classic flocking models appear as precision-weighted prediction errors.
The core has gradient flows of potentials, reductions over neighbours with conditions, angles, and draws. What it
lacks is the derivative of a reduction over neighbours with respect to a state of the reading member, so the
free energy, which reads the sector means, cannot be differentiated with respect to the heading.

## What Changes

- The reductions `total`, `mean`, and `variance` of `skcomplex.core.components.laws.utils` gain symbolic derivatives
  with respect to the values of the reading member, with the condition `where` and the set `over` held fixed, so
  Sage's `.diff` and `GradientInteraction` of `skcomplex.systems.general.gradients` differentiate expressions that hold
  them. The derivative lowers like any reduction.
- New system `ActiveInferenceFlockSystem` in `biology/collective_behaviour`: agents with positions, unit headings,
  and beliefs in generalised coordinates of the sector distances, a Laplace-encoded free energy with sensory and
  process precisions, belief and heading updates by its gradient, and the default and milling configurations of the
  paper. The free-energy update is a recipe in the module, `FreeEnergyInteraction`, an `Interaction` whose `_prepare`
  writes the free energy, differentiates it with `.diff`, and sets `UpdateLaw` laws in `laws_`.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `general-mechanisms`: reductions over neighbours differentiate with respect to the reading member.
- `model-library`: the active-inference flock.

## Impact

- Core: the private `_ReductionFunction` of `core/components/laws/utils/_reductions.py`, which gains the `_derivative_`
  hook that Sage reads, and the printers, which print the derived reductions as they print any reduction. No new law
  kind and no new class.
- Systems: a new module in `skcomplex.systems.biology.collective_behaviour`.
- No new dependency. `inferactively-pymdp` 1.0.5, released 2026-11-07, covers active inference over discrete states and
  is not used, since the flock is continuous.
- Docs: a paragraph on derivatives of reductions in the user guide on writing laws, a gallery example, and the
  inventory with the primitive `reduction-derivatives`.
