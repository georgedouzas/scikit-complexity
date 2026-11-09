## Why

The core has one concrete system, the composite `CompositeSystem`, and `skcomplex.systems.general.reactions` holds the
general `ReactionNetworkSystem` of species and reactions. Every other catalog system writes its own element class,
repeats each model parameter in the system, in the element, and in the call that joins them, and composes the same few
core parts again. An audit of the catalog finds four system shapes that systems of several fields share, built the same
way each time. The systems should offer each of them once, in `skcomplex.systems.general`, as a general system named
after its mathematics, so a field system gives only its parameters and its equations. General systems are recipes, built
from core parts, and the placement rule of `move-recipes-to-systems` places a recipe that several fields use in a
concept subpackage of `skcomplex.systems.general` (step 3). The four systems share one concept, so they live in
`skcomplex.systems.general.dynamics`.

## What Changes

- New general system `DynamicalSystem` in `systems/general/dynamics/_dynamical.py`. It is a set of declared states and
  their evolution equations: rates in continuous time, with optional noise amplitudes, or maps in discrete time. It
  builds one `DeclaredElement` and one `RateInteraction` or `MapInteraction` per state.
- New general system `NetworkDynamicalSystem(DynamicalSystem)` in `_network_dynamical.py`. It puts the states on the
  nodes of a graph and adds a coupling `sum_j A_ij g(x_i, x_j)` per state through a `CouplingInteraction`.
- New general system `ReactionDiffusionSystem(DynamicalSystem)` in `_reaction_diffusion.py`. It puts the states as
  fields on the sites of a lattice, with local kinetics and a diffusivity per field.
- New general system `StockFlowSystem` in `_stock_flow.py`. It is a set of stocks and named flows between them or
  with the environment. It builds one `DeclaredElement` of the stocks and one `TransferInteraction` per flow, so every
  flow is reported in the trajectory.
- General systems gain the admission test the user set: a general system serves systems of two fields, or can, and is
  named after its mathematics, never after a field quantity. The core never imports them.
- Catalog systems of the shapes above rebase onto the new general systems. Their public names, parameters, state
  names, and trajectories stay the same. The design lists them per field.
- **BREAKING** The element classes that only a rebased system uses leave the catalog. Their equations move into the
  hooks of the system, so no model is lost. The design lists the rule, and the tasks list the classes per module.
- The system-dynamics translator writes a subclass of `StockFlowSystem` for a file whose stocks are not subscripted
  and change only through flow variables.
- The page of `skcomplex.systems.general` in the guide, `core_components/general.md`, gets one section per general
  system, with examples from two fields each.
- Shapes the audit rejects stay as they are: replicator dynamics, Markov chains, cellular automata, Hamiltonian and
  Lagrangian systems, agent-based systems, particles in continuous space, flux balance models, and Boolean networks.
  The design gives each reason.

## Capabilities

### New Capabilities

- `general-systems`: the general systems of `skcomplex.systems.general` that fields share, what each takes, what it
  builds, and how a field system subclasses it.

### Modified Capabilities

- `model-library`: ready-made systems subclass the general system of their shape, and a rebase keeps the public
  interface and the trajectories of a system.
- `system-dynamics-import`: translated modules of flows only are subclasses of `StockFlowSystem`.

## Impact

- Systems code: the subpackage `skcomplex.systems.general.dynamics` with four modules and its `__init__.py`. The core
  gains no module and never imports the subpackage. The translator and its module writer in `core/engine/translators`
  change, and write source text that imports `StockFlowSystem` from `skcomplex.systems.general.dynamics`.
- Systems: modules of biology, chemistry, computer science, economics, engineering, mathematics, physics, and social
  that hold a rebased system. The element classes that only those systems use leave them and their `__init__.py`.
- Tests: new mirrored tests under `tests/systems/general/dynamics`, and the tests of every rebased module.
- Documents: `docs/overview/user_guide/system_classes.md`, the page of `skcomplex.systems.general`
  (`core_components/general.md`), the generated parts page of systems, the gallery examples that name a leaving element
  class, and the changelog.
- Inventory: `docs/overview/inventory.toml` entries whose components change.
- Ordering: lands after `slim-core-elements-and-spaces`, `rename-core-systems`, `slim-core-interactions`,
  `slim-core-laws`, and `move-recipes-to-systems`, which creates `skcomplex.systems.general`, and uses their names and
  places: `NonnegativeElement` of `skcomplex.systems.general.elements`, `BallNeighborhood`, `ArrayUpdateLaw`,
  `BirthDeathInteraction` of `skcomplex.systems.general.agents`, `CompositeSystem`, and `ReactionNetworkSystem` of
  `skcomplex.systems.general.reactions`. It coordinates with `rewrite-docstrings-and-docs` on the page of
  `skcomplex.systems.general`.
- Users: code that imports a leaving element class builds the system that used it, or writes the element with
  `DeclaredElement`. The project makes breaking changes in 0.x without shims.
