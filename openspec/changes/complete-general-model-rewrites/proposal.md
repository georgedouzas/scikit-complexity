# Proposal

## Why

`general-elements-and-interactions` rebuilt the core on general operations and moved most model classes onto them, but
about forty model classes stayed on their own laws because a capability was missing, a rebuild failed its 20-seed
comparison, or the work ran out of time. The constitution records a model the core cannot express as blocked by a
change, so this change holds what remains, one task per missing capability.

## What Changes

- **Coupling:** couple a set of separate elements, couple a hub element to a group by any reduction, and combine a map
  with an update-form coupling, for mutual and global inhibition, transmission lines, and the contour cell.
- **Reactions:** stoichiometric coefficients given as parameters, for the Oregonator, and per-member reactions in the
  agent form.
- **Transfers:** symbolic transfers along the edges of a lattice or network in both directions with parameters, and a
  shared draw per edge, for zero-range, exclusion, and reservoir processes and parallel-update TASEP, and composite
  transfers that hold parameters, for the beer game and the stock-flow consistent circuit.
- **Agent visits:** a phase of N visits drawn with replacement, each composing choice, transfer, and reinforcement of
  the chosen option only, for the perishables market, and births whose probability depends on the composition, for
  tangled nature.
- **Agent coupling:** a coupling of agents over several radii with clamps and a priority of repulsion, for boids and
  the Couzin model.
- **History in rules:** agent rules read the recorded values of observables, so reporter-only feedback interactions go.
- **Experiments:** replicate systems that hold interactions, so the named maps and flows used by orbit diagrams can be
  rebuilt on `MapInteraction` and `RateInteraction`.
- **Rebuilds the core already allows:** the meadow, ants, trails, angiogenesis, agent predation, stigmergic plants,
  degree-class mean fields, the rats' nervousness, the commons herders, and the large single-element ODE models of
  oncology, neuroscience, physiology, and systems biology that decompose into rate and reaction interactions.
- **Conventions:** split the model modules that hold several classes into one class per module.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `general-mechanisms`: adds the coupling, reaction, transfer, visit, agent-coupling, and history capabilities above.

## Impact

- **Code:** `skcomplex.core.interactions`, the compile passes and the Mesa solver of `skcomplex.engine`,
  `skcomplex.experiments`, and the model packages named in the tasks.
- **API:** additive. Every new argument defaults to the current behaviour. Model factories keep their names and
  parameters.
- **Order:** archive `general-elements-and-interactions` first, since this change adds requirements to the capability
  it creates. The coupling of fields by a kernel belongs to `add-nonlocal-operators`, matrix-valued weights to
  `add-learning-dynamics`, and laws per hyperedge to `add-higher-order-interactions`.
- **Not in scope:** non-numeric agent states, such as queues and sets, which stay in their models because observables
  and snapshots record numbers only, and the incubation timing of the epidemiology `PersonElement`, which waits on a
  decision by the maintainer.
