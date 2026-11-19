# Check a system

This tutorial checks a [system](glossary.md#system) before it runs. It prepares the predator and prey model of the
[build tutorial](build_a_custom_system/index.md), reads what each component says about itself and what the prepared
system found, and triggers the validation errors a builder meets most, each with its message and its fix.

## The running example

The model is the cycle of the snowshoe hare and the Canada lynx, with the rates that Carpenter (2018) estimated from
the pelt records of the Hudson's Bay Company. The catalog class `LotkaVolterraSystem` writes it in one line, starting
from the records of 1900, 30.0 thousand hares and 4.0 thousand lynx.

```python
from skcomplex.systems.biology.ecology import LotkaVolterraSystem

system = LotkaVolterraSystem(prey=30.0, predators=4.0, r=0.55, a=0.028, b=0.024, d=0.80).prepare()
[(component.label, type(component).__name__) for component in (*system.system_.elements, *system.system_.interactions)]
```

`prepare` builds the components of the system from its [model parameters](glossary.md#model-parameter) and checks them.
A [system class](glossary.md#system-class), a [BaseSystem](core_components/systems.md#basesystem), keeps them in its
attribute `system_`, a [System](core_components/systems.md#system). Here it holds the populations
`prey` and `predators`, and the interactions `growth`, `predation`, and `death`, the three terms of the equations of
Lotka and Volterra.

## Tags

Every component says what it is through its [tags](glossary.md#tag). `get_tags()` of an interaction returns the
kinds of the [laws](glossary.md#law) it emits and the time they run in. `get_tags()` of a system joins those of its
interactions with the [domains](glossary.md#domain) of its [states](glossary.md#state), the set of values a state may
take, such as `real`.

```python continuation
tags = system.get_tags()
sorted(tags.time), sorted(tags.laws), sorted(tags.domains), tags.randomness
```

The system runs in continuous time, its laws are rates, its states are real numbers, and it draws no random numbers. So
it is an [ordinary differential equation](glossary.md#ordinary-differential-equation). The tags of one component show
where each tag comes from.

```python continuation
prey, predators = system.system_.elements
growth, predation, death = system.system_.interactions
sorted(predation.get_tags().laws), sorted(predation.get_tags().time)
```

`predation` gives the [law kind](glossary.md#law-kind) `rate` and the time `continuous`.
The component pages list the tags of every [element](../../components/elements.md),
[interaction](../../components/interactions.md), [law](../../components/laws.md), [solver](../../components/solvers.md), and
[backend](../../components/backends.md) of the core, each with a compatible and an incompatible example. Start there to find
candidates, and check a candidate in code as the sections below do.

## What a prepared system holds

An element is a plain class whose arguments are its values, and the laws decide what each value is. A value that some
law changes is a [state](glossary.md#state), and any other value is a parameter. `prepare` works this out and keeps
the result in `states_` and `parameters_`.

```python continuation
system.states_, system.parameters_
```

The states are the populations `prey.x` and `predators.x`, and the parameters are the rates of the three
interactions. A value of one member is one number, and an element with `size=` holds one value per member.

- An `Element` without `size` or `space` is one [member](glossary.md#member) with its values.
- An `Element` with `size=` is a [group](glossary.md#group) of many identical members.
- An `Element` with `space=` holds one member per site of the [space](glossary.md#space), and with `on='edges'` or
  `on='hyperedges'` one member per link or hyperedge of a network.

## What an interaction needs

An interaction binds elements to its roles, the named slots it reads and changes, such as `predator` and `prey`.
`get_requirements()` returns what it needs for the arguments it was given, with the roles it binds.

```python continuation
list(predation.get_requirements().roles)
```

`predation` has the roles `predator` and `prey`. The time of a system follows from the kinds of its laws: rates flow in
continuous time, updates step in discrete time, and objectives and constraints alone are static. `form_` names the
problem that `prepare` found.

```python continuation
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw, UpdateLaw
from skcomplex.core.systems import System

hares = Element('hares', values={'x': 30.0})
flow = Interaction('births', roles={'target': 'hares'}, laws=lambda s: [RateLaw(s.target.x, 0.55 * s.target.x)])
step = Interaction('births', roles={'target': 'hares'}, laws=lambda s: [UpdateLaw(s.target.x, 1.55 * s.target.x)])
System(elements=[hares], interactions=[flow]).prepare().form_, System(elements=[hares], interactions=[step]).prepare().form_
```

The rate makes an ordinary differential equation, and the update makes a discrete map.

## Validation errors

A component that does not fit raises an error before anything compiles. `prepare` checks the whole system and raises
one error that lists every problem, each with its path, its message, a fix, and its code in brackets. The variants
below rebuild the model with one-off interactions and break one thing at a time.

```python continuation
def births_laws(s):
    return [RateLaw(s.target.x, 0.55 * s.target.x)]


def hunting_laws(s):
    return [RateLaw(s.prey.x, -0.028 * s.prey.x * s.predator.x)]


births = Interaction('births', roles={'target': 'hares'}, laws=births_laws)
hunting = Interaction('hunting', roles={'prey': 'hares', 'predator': 'lynx'}, laws=hunting_laws)
lynx = Element('lynx', values={'x': 4.0})
pair = System(elements=[hares, lynx], interactions=[births, hunting]).prepare()
round(float(pair.simulate(until=1).get_values('hares.x')[-1]), 1)
```

The two populations with births and hunting run without an error. The lynx keep 4.0 thousand without terms of their
own, so the hares grow at the net rate `0.55 - 0.028 * 4.0`, from 30.0 to 46.5 thousand in a year.

### A value the element does not have

The most frequent error binds a role to an element that lacks the value the interaction reads. Here births change the
value `n` of the hares, which have `x`.

```python continuation
from skcomplex.core.errors import RoleElementError

wrong_state = Interaction('births', roles={'target': 'hares'}, laws=lambda s: [RateLaw(s.target.n, 0.55)])
try:
    System(elements=[hares], interactions=[wrong_state]).prepare()
except RoleElementError as error:
    message = str(error)
message
```

The error has the code `role-element`. Its message says that the role `target` of births reads the state `n`, but the
element hares bound to it carries `x`. The fix reads `s.target.x`, or gives the element a value `n`. A function passed
as `laws` is checked in the same way, and an error of another kind inside a law reaches the caller unchanged.

### A role bound to a missing element

A role names an element by its [label](glossary.md#label), so a typing error or a forgotten element leaves it bound to
nothing. Here the lynx are left out of the system.

```python continuation
from skcomplex.core.errors import RoleError

try:
    System(elements=[hares], interactions=[hunting]).prepare()
except RoleError as error:
    message = str(error)
message.splitlines()[-1]
```

The code is `role`. The message says that hunting binds the role `predator` to `lynx`, which is not an element or
group path relative to the root system. The fix adds the element `lynx` to the system, or binds the role to the label
of an element it holds. A [nested system](glossary.md#nested-system) is reached by a path such as `village/farmers`.

### Two components with one label

Every child of a system needs a label of its own, because roles, states, and parameters are named by it.

```python continuation
from skcomplex.core.errors import LabelError

twins = Element('hares', values={'x': 4.0})
try:
    System(elements=[hares, twins], interactions=[births]).prepare()
except LabelError as error:
    message = str(error)
message.splitlines()[-1]
```

The code is `label`, and the message says that the label `hares` is used by more than one child of the system.
Renaming the second element to `lynx` fixes it.

### A name that does not exist

Model parameters, states, and run parameters are named, and a wrong name fails at once. The catalog class names the
birth rate of the prey `r`, not `alpha` as in the build tutorial, and its states `prey.x` and `predators.x`.

```python continuation
from skcomplex.core.errors import ParameterError, ValidationError

messages = []
try:
    system.set_params(alpha=0.6)
except ValueError as error:
    messages.append(str(error).split('. ')[-1])
try:
    system.prepare().simulate(until=20, initial_state={'hares.x': 10.0})
except ValidationError as error:
    messages.append(str(error))
try:
    system.prepare().simulate(until=20, schedule={'predation.alpha': lambda t: 0.0})
except ParameterError as error:
    messages.append(str(error))
messages
```

- `set_params` raises a `ValueError`, as scikit-learn does, and lists the valid parameters: `K`, `a`, `b`, `backend`,
  `d`, `label`, `predators`, `prey`, `r`, and `solver`.
- An `initial_state` with a wrong name raises a `ValidationError` that suggests `prey.x` and `predators.x`.
- A `schedule` names the parameters of the built components by their path, and a wrong one raises a `ParameterError`
  that lists `growth.r`, `predation.a`, `predation.b`, and `death.d`.

## The solver

The [solver](glossary.md#solver) advances the compiled system. It can be left out, because the tags of a system decide its default solver.
Every solver class lists the tags it supports in `get_supported_tags()`, and `prepare` rejects a solver that lacks a
tag of the system with a [TagError][skcomplex.core.errors.TagError] that names it, before it compiles anything.

```python continuation
from skcomplex.core.engine.solvers import IterationSolver, RK4Solver
from skcomplex.core.errors import TagError

fits = tags.laws <= RK4Solver.get_supported_tags().laws
try:
    system.set_params(solver=IterationSolver()).prepare().simulate(until=20)
except TagError as error:
    message = str(error)
fits, message, system.set_params(solver=RK4Solver()).prepare().simulate(until=20, dt=1.0).get_values('prey.x').round(1)[:6].tolist()
```

`RK4Solver`, the classical Runge-Kutta method of order four, integrates the rates of the model, and its run matches the
default solver of the build tutorial: 30.0, 46.4, 68.5, 81.2, 48.8, 18.6 thousand hares from 1900 to 1905.
`IterationSolver`, which iterates maps, does not support the time `continuous`, and the message says so. The
[solvers page](../../components/solvers.md) lists what every solver supports, and the
[backends page](../../components/backends.md) what every [backend](glossary.md#backend) supports. The tutorial [Simulate](simulate.md) runs the
checked system.
