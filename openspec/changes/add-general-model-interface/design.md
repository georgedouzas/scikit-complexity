# Design

## Context

Components are dataclass-like `BaseEstimator` subclasses whose fields come from `state`, `parameter`, and `role`, and
whose `laws(s)` method returns laws. Library modules export component classes such as `Population` and
`LogisticGrowth`, plus factories such as `ideal_gas`, `wolf_sheep`, and `harvesting` that return a `System`. Nothing
in the core is specific to a science: the only field-specific code outside `skcomplex.models` is the LAMMPS backend,
which sits above the library by design.

The references converge on one description of a system, which this change makes explicit:

- Thurner et al. (2018 §1.5): elements with states, interactions of several types, update laws for states and for
  interactions, and observables of the whole.
- Mobus (2022 §4.3.3): a recursive tuple of components, internal and external flows, boundary, transformations,
  history, and a time step per level.
- Dilão (2023 ch. 1) and Datseris and Parlitz (2022 ch. 1): a state space and an evolution operator, continuous or
  discrete in time, independent of the field.
- Jensen (2023 ch. 1-2): emergence is visible through macro variables declared at a higher level.

## Goals / Non-Goals

**Goals:**

- One interface that any model from any science is a configuration of, with no field-specific concept in `core`,
  `model`, `compile`, `solvers`, `runtime`, `viz`, or `analysis`.
- Every ready-made model reachable through the catalog and proven to run through the same path.

**Non-Goals:**

- New law kinds, spaces, or solvers. Later changes in this series add them on top of this interface.
- Acausal connectors in the style of Modelica (Ryzhov et al. 2021), kinetic models of active particles (Bellomo et al.
  2017), and fractional derivatives (Trzaska 2023 §2.3.6), which `add-acausal-components`, `add-kinetic-models`, and
  `add-fractional-dynamics` cover.

## Decisions

### Components without subclassing

`Element.define(name, *, states=None, parameters=None, laws=None, visual=None)` and
`Interaction.define(name, *, roles=(), parameters=None, laws=None)` build a new component class with `type` and the
existing field machinery, so groups, `get_params`, `clone`, and pickling keep working. `states` and `parameters` map
names to `state(...)` and `parameter(...)` declarations or to plain defaults, and `laws` is a function `laws(s)`, or
`laws(self, s)` when it needs the component's plain fields. Classes stay the way to write reusable components with
methods, and `define` is the way to write a one-off model in a few lines.

- Alternative considered: generic `Element(states={...})` instances. Rejected because groups, cloning, and parameter
  introspection all rely on fields being class attributes.

### State domains

`Domain` gains `BOOLEAN`, `ANGLE`, and `categorical(*values)`, which returns a finite-alphabet domain. Values stay in
the flat float vector: booleans as 0 and 1, categories as their index, and angles wrapped into `[0, 2π)` after every
step by an automatic boundary law. Symbols of boolean and categorical states are integer symbols, and laws of a finite
alphabet that produce a value outside it raise a validation error at run time in checked mode.

- Alternative considered: separate integer arrays. Rejected for now because every solver, backend, and trajectory
  assumes one float vector, and the encoding is exact for alphabets below 2^53.

### Observables

`Observable(label, expression=None, *, group=None, reduce=None, variable=None)` is declared on a system with
`System(observables=[...])`. An expression observable is a function of the system's symbols, such as the total energy
or a Kuramoto order parameter, and a reduction observable summarises a group with `count`, `sum`, `mean`, `min`, `max`,
`var`, or a callable. Observables compile to one function evaluated at every frame, are stored in trajectories under
`<system path>/<label>`, are logged to Rerun as time series, and are readable by laws of other partitions as inputs.
`Aggregate` becomes a reduction observable that also holds its value as a state, and keeps its current behaviour.

### Model catalog

`register_model(name, *, field, description, tags=(), references=(), defaults=None)` decorates a factory. `catalog()`
returns the registered entries as records with their metadata and the formalism inferred by compiling the default
model once, `describe(name)` returns one record with the factory's parameters and their defaults, and
`create(name, **parameters)` calls the factory. `defaults` holds run settings such as `dt`, `until`, and the solver,
and `run(name, **parameters)` creates a model and simulates it with them. Field modules register their factories when
`skcomplex.models` is imported.

### Library structure

Every field module keeps its components and adds a factory per canonical model: for example `logistic_growth`,
`lotka_volterra`, `sir`, `solow`, `cobweb`, `geometric_brownian_motion`, `harmonic_oscillator`, `two_body_orbit`,
`ideal_gas`, `lennard_jones_fluid`, `schelling`, `boltzmann_wealth`, `voter_model`, `wolf_sheep`, and `harvesting`. A
test parses the imports of `skcomplex.models` and fails on any import of a private name from another package.

### Conformance suite

One parametrized test runs over `catalog()`: it creates every model with its defaults, compiles it, simulates it for
its default duration or ten frames, logs the run to a Rerun recording in memory, clones the system, and checks that the
trajectory holds every state and observable.

## Risks / Trade-offs

- [Two ways to define components] → Classes for reusable components with methods, `define` for one-off models. The
  user guide shows both side by side.
- [Catalog import cost] → Registration stores factories without calling them, and the formalism is inferred lazily on
  the first `catalog()` call.
- [Categorical values in floats] → Exact for practical alphabets, and checked mode catches values outside the alphabet.
