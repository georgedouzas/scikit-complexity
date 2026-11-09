## Why

The core has two concrete systems, the composite `CompositeSystem` and `ReactionNetworkSystem` of species and reactions.
Every other catalog system writes its own element class, repeats each model parameter in the system, in the element,
and in the call that joins them, and composes the same few core parts again. An audit of the catalog finds four system
shapes that systems of several fields share, built the same way each time. The core should offer each of them once as
a general system named after its mathematics, so a field system gives only its parameters and its equations.

## What Changes

- New general system `DynamicalSystem` in `core/components/systems/_dynamical.py`. It is a set of declared states and
  their evolution equations: rates in continuous time, with optional noise amplitudes, or maps in discrete time. It
  builds one `DeclaredElement` and one `RateInteraction` or `MapInteraction` per state.
- New general system `NetworkDynamicalSystem(DynamicalSystem)` in `_network_dynamical.py`. It puts the states on the
  nodes of a graph and adds a coupling `sum_j A_ij g(x_i, x_j)` per state through a `CouplingInteraction`.
- New general system `ReactionDiffusionSystem(DynamicalSystem)` in `_reaction_diffusion.py`. It puts the states as
  fields on the sites of a lattice, with local kinetics and a diffusivity per field.
- New general system `StockFlowSystem` in `_stock_flow.py`. It is a set of stocks and named flows between them or
  with the environment. It builds one `DeclaredElement` of the stocks and one `TransferInteraction` per flow, so every
  flow is reported in the trajectory.
- The rule for general systems gains the admission test the user set: a general system is cross-domain or can be made
  so, adds a feature no other core part offers, and is named after mathematics, never after a field quantity.
- Catalog systems of the shapes above rebase onto the new general systems. Their public names, parameters, state
  names, and trajectories stay the same. The design lists them per field.
- **BREAKING** The element classes that only a rebased system uses leave the catalog. Their equations move into the
  hooks of the system, so no model is lost. The design lists the rule, and the tasks list the classes per module.
- The core components guide gets one section per general system, with examples from two fields each.
- Shapes the audit rejects stay as they are: replicator dynamics, Markov chains, cellular automata, Hamiltonian and
  Lagrangian systems, agent-based systems, particles in continuous space, flux balance models, and Boolean networks.
  The design gives each reason.

## Capabilities

### New Capabilities

- `general-systems`: the general systems of the core that fields share, what each takes, what it builds, and how a
  field system subclasses it.

### Modified Capabilities

- `base-interfaces`: the rule for general systems states the admission test and names the general systems.
- `model-library`: ready-made systems subclass the general system of their shape, and a rebase keeps the public
  interface and the trajectories of a system.

## Impact

- Core code: four new modules in `core/components/systems`, its `__init__.py`, and the import contract of the package
  in `pyproject.toml`.
- Systems: modules of biology, chemistry, computer science, economics, engineering, mathematics, physics, and social
  that hold a rebased system. The element classes that only those systems use leave them and their `__init__.py`.
- Tests: new mirrored tests under `tests/core/components/systems`, and the tests of every rebased module.
- Documents: `docs/overview/user_guide/system_classes.md`, the systems page of the core components guide, the
  generated parts page of systems, the gallery examples that name a leaving element class, and the changelog.
- Inventory: `docs/overview/inventory.toml` entries whose components change.
- Ordering: lands after `slim-core-elements-and-spaces`, `rename-core-systems`, `slim-core-interactions`, and
  `slim-core-laws`, and uses their names: `NonnegativeElement`, `BallNeighborhood`, `ArrayUpdateLaw`,
  `BirthDeathInteraction`, `CompositeSystem`, and `ReactionNetworkSystem`. It coordinates with
  `rewrite-docstrings-and-docs` on the page `core_components/systems.md`.
- Users: code that imports a leaving element class builds the system that used it, or writes the element with
  `DeclaredElement`. The project makes breaking changes in 0.x without shims.
