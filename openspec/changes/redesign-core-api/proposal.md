# Proposal

## Why

Written from the point of view of a user who wants a particle falling under gravity, the core makes no sense. The
state of an element is half declared (`order=2` adds a hidden velocity `h_d1`), parameters sit on the wrong part (`g`
on the particle), laws can come from an element or from an interaction with no rule for which, a user who wants a
tunable system has to subclass and copy every parameter through a private hook, and running goes through a free
function beside a `prepare` and `run` that mean something else. An audit of the core counts 7 element classes, 22
interaction classes, 27 law classes, 23 spaces, neighbourhoods, and times, and 103 public spec classes, of which 60 are
used by no system and no document. Interactions duplicate laws one to one, interaction constructors take up to 22
arguments, `StateSpec` has 13 fields, and the core holds laws of single fields such as Lindblad dissipators, sandpile
cascades, and L-system productions. The docstrings average 48 lines and follow a fixed template.

Before more systems are written on it, the core has to become a small set of concepts that every field shares, with
one way to do each thing.

## What Changes

The design is in `design.md`. In short:

- **BREAKING** An element is data: its states and its own parameters. Elements return no laws. Every state is
  declared, and a state names the state that is its rate of change, such as `x` with `rate='v'`, in place of `order`.
  One element class covers one member, many members (`size=`), and one member per site of a space (`space=`), which
  replaces `GroupElement` and `FieldElement`.
- **BREAKING** An interaction is all behaviour. It binds roles, owns the parameters of the relation, and returns laws
  from the public hook `get_laws`. Where it applies is a binding argument, not a class: one target, named roles, every
  pair within a neighbourhood, or every edge of a network. `NeighborInteraction`, `PairInteraction`,
  `HyperedgeInteraction`, and `GroupingInteraction` go, and so do the interactions that only wrap one law.
- **BREAKING** Laws are plain values of seven kinds named by their mathematics: `Rate`, `Noise`, `Jump`, `Update`,
  `Constraint`, `Event`, and `Objective`. Contributions to one state add up. The other law classes become one of these
  kinds or move to the field that uses them.
- **BREAKING** `CompositeSystem` becomes `System`. A system exposes the parameters of its parts by nested names, such
  as `gravity__g`, as a scikit-learn `Pipeline` does, and holds its `solver` and `backend` as settings. A system is
  prepared and simulated like an estimator is fitted and used to predict: `system.prepare()` checks and compiles once,
  and `system.simulate(until, ...)` returns a `Trajectory`. The free function `simulate` is removed.
- **BREAKING** Elements, interactions, spaces, and laws are configurable descriptions with parameters and no
  `prepare` or `run`, like the kernels of scikit-learn. Solvers, backends, and observers keep `prepare` and `run` for
  the engine. Experiments stay functions that clone, set parameters, prepare, and simulate.
- **BREAKING** Ready-made systems stay classes with model parameters and catalog metadata, and their build step becomes
  the public hook `build`.
- **BREAKING** The specs shrink to the declarations users write, `StateSpec`, `ParameterSpec`, and `RoleSpec`, with the
  short aliases `State`, `Parameter`, and `Role`, plus the value objects users pass. `StateSpec` keeps its domain, its
  shape, and its rate. Engine records and compile specs become private to the code that uses them.
- **BREAKING** How an element is drawn moves from the element to the visualisation observer, and `visual=` leaves every
  element constructor.
- **BREAKING** The core keeps only components that general systems of several fields are built from. Every
  field-specific element, interaction, law, space, and neighbourhood moves to `skcomplex.systems.<field>.common` or to
  `skcomplex.systems.general`, and the concrete systems of each field are the consumers of the core.
- The code is simplified along the way: fewer classes, short constructors, no hidden conventions, no private hooks for
  users.
- Docstrings follow the scikit-learn layout: one summary line, one plain paragraph with the formula and uses from
  several fields, a link to the guide, arguments, See Also, and short examples. The fixed "Unlike", "Choose it when",
  and "It appears in" template goes.

## Capabilities

### New Capabilities

None. The redesign changes the requirements of existing capabilities.

### Modified Capabilities

- `system-modeling`: elements as data, interactions as behaviour, the seven law kinds, explicit states with rates,
  bindings, nested parameters, the `System` class, and what the core holds.
- `base-interfaces`: which classes have `prepare` and `run`, what `prepare` and `simulate` return, the reduced specs,
  and ready-made systems with a public `build`.
- `package-layout`: the core as the general building blocks, field-specific components in the systems packages,
  private engine records, and the public surface without the free function `simulate`.
- `simulation`: simulating as a method of a prepared system, with solvers and backends as system settings.
- `experiments`: experiments prepare once and simulate many times.
- `model-catalog`: ready-made systems are used through `prepare` and `simulate`.
- `user-documentation`: the scikit-learn docstring layout replaces the fixed template.

## Impact

- Every module of `skcomplex.core.components`, the compilers and lowerings of `skcomplex.core.engine` that key on law
  classes, `skcomplex.core.specs`, and `skcomplex.core.experiments`.
- Every ready-made system of `skcomplex.systems`, which is rewritten on the new core, field by field.
- About 3,500 calls of `simulate(`: about 200 in `src`, 1,600 in `tests`, and 1,700 in `docs`, plus 19 mentions in the
  main specs.
- About 390 system modules that pass `VisualSpec` to their elements.
- The README, the user guide, the gallery, and every docstring of the core.
- The open change `add-component-finder` assumes the current core. It is parked until this change settles the
  components it lists, and then rebased onto them.
- The next release is a breaking release with no aliases for the removed names.
