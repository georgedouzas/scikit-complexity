# Check a system

This tutorial checks a [system](glossary.md#system) before it runs. It prepares the predator and prey model of the
[build tutorial](build_a_custom_system/index.md), reads what each part declares about itself, compares what an
[interaction](glossary.md#interaction) needs with what an [element](glossary.md#element) carries, and triggers the validation errors a builder meets most, each with its
message and its fix.

## The running example

The model is the cycle of the snowshoe hare and the Canada lynx, with the rates that Carpenter (2018) estimated from
the pelt records of the Hudson's Bay Company. The catalog class `LotkaVolterraSystem` writes it in one line, starting
from the records of 1900, 30.0 thousand hares and 4.0 thousand lynx.

```python
from skcomplex.systems.biology.ecology import LotkaVolterraSystem

system = LotkaVolterraSystem(prey=30.0, predators=4.0, r=0.55, a=0.028, b=0.024, d=0.80)
system.prepare()
[(part.label, type(part).__name__) for part in (*system.system_.elements, *system.system_.interactions)]
```

`prepare` builds the parts of the system from its [model parameters](glossary.md#model-parameter) and checks them. A [system class](glossary.md#system-class), a
[BaseSystem](core_components/systems.md#basesystem), keeps them in its attribute `system_`, a
[CompositeSystem](core_components/systems.md#compositesystem). Here it holds the populations `prey` and `predators`,
and the interactions `growth`, `predation`, and `death`, the three terms of the equations of Lotka and Volterra.

## Tags

Every part says what it is through its [tags](glossary.md#tag), which come from its declarations and its arguments alone. `get_tags()`
of an element returns the [domains](glossary.md#domain) of its [states](glossary.md#state), the set of values a state may take, such as `real`. `get_tags()` of
an interaction returns the kinds of the [laws](glossary.md#law) it emits and the time they run in. `get_tags()` of a system joins those of
its parts.

```python continuation
tags = system.get_tags()
sorted(tags.time), sorted(tags.laws), sorted(tags.domains), tags.randomness
```

The system runs in continuous time, its laws are rates, its states are real numbers, and it draws no random numbers.
So it is an [ordinary differential equation](glossary.md#ordinary-differential-equation). The tags of one part show where each tag comes from.

```python continuation
prey, predators = system.system_.elements
growth, predation, death = system.system_.interactions
sorted(prey.get_tags().domains), sorted(predation.get_tags().laws), sorted(predation.get_tags().time)
```

The population `prey` gives the domain `real`, and `predation` gives the [law kind](glossary.md#law-kind) `rate` and the time `continuous`.
The parts pages list the tags of every [element](../../parts/elements.md),
[interaction](../../parts/interactions.md), [law](../../parts/laws.md), [solver](../../parts/solvers.md), and
[backend](../../parts/backends.md) of the core, each with a compatible and an incompatible example. Start there to find
candidates, and check a candidate in code as the sections below do.

## What an element carries

An element carries the states it declares, and `get_states()` returns them with their [specs](glossary.md#spec). The spec of a state holds
its domain, here the numbers from zero up, since a population cannot be negative.

```python continuation
{name: str(spec.domain) for name, spec in prey.get_states().items()}
```

The population carries one state `x` with the domain `[0, +oo)`. Every element is also of one kind, and the kind is
what a [role](glossary.md#role) accepts or rejects.

- An `Element` or a `DeclaredElement` is of the kind `element`, one [member](glossary.md#member) with its states.
- A [GroupElement](core_components/elements.md#groupelement) is of the kind `group`, many identical members. With
  `variable=True` it is a `variable-group`, whose members appear and disappear.
- A [group](glossary.md#group) on a [space](glossary.md#space) is a `site-group`, an `edge-group`, or a `hyperedge-group`, by what its members sit on. A
  `GroupElement` on `'sites'` and a [FieldElement](core_components/elements.md#fieldelement) are site groups.

## What an interaction needs

An interaction binds elements to its roles, the named slots it reads and changes, such as `predator` and `prey`.
`get_requirements()` returns what it needs for the arguments it was given: the time and the space its laws run in, the
law classes it may emit, and, for every role, the [kinds of element](glossary.md#kind-of-element) it accepts, the states it reads, and their domains.
An empty tuple means no restriction.

```python continuation
needs = predation.get_requirements()
[time.__name__ for time in needs.time], needs.space, list(needs.roles)
```

`predation` needs [ContinuousTime](core_components/spaces.md#continuoustime), accepts any space, and has the roles
`predator` and `prey`. A [RateInteraction](core_components/interactions.md#rateinteraction) of the build tutorial
states more about its role, because it knows the state it changes.

```python continuation
from skcomplex.core.components.interactions import RateInteraction

births = RateInteraction('births', target='hares', state='x', rate=lambda s: 0.55 * s.target.x)
target = births.get_requirements().roles['target']
target.elements, target.states, target.domains
```

The role `target` accepts an element of any kind, reads its state `x`, and accepts the domains `real`, `integer`,
`complex`, and `vector`. A rate needs a number it can differentiate, so a boolean state does not fit. The method
`accepts` of the role compares it with the kind of an element.

```python continuation
from skcomplex.core.types import ElementKind

target.accepts(ElementKind.ELEMENT), target.accepts(ElementKind.GROUP)
```

Both are `True`, because the kind `any` accepts every kind of element.

### Time and space

The time decides whether a system flows or steps. A `RateInteraction` returns rates, so it needs continuous time. A
[MapInteraction](core_components/interactions.md#mapinteraction) returns the next value of a state, so it needs
[DiscreteTime](core_components/spaces.md#discretetime).

```python continuation
from skcomplex.core.components.interactions import MapInteraction

step = MapInteraction('step', target='hares', state='x', function=lambda s: 1.55 * s.target.x)
[time.__name__ for time in births.get_requirements().time], [time.__name__ for time in step.get_requirements().time]
```

The rate needs `ContinuousTime` and the map needs `DiscreteTime`. The tags say the same: `sorted(step.get_tags().time)`
is `['discrete']`.

### Roles that accept some kinds

A role may accept some kinds of element and reject the others. A census that counts the members of a group declares in
its [RoleSpec][skcomplex.core.specs.RoleSpec] that its `target` accepts a group and not a single element.

```python continuation
from typing import Annotated, ClassVar

from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RuleLaw
from skcomplex.core.specs import RoleSpec


class CensusInteraction(Interaction):
    law_classes: ClassVar[tuple[type, ...]] = (RuleLaw,)

    def __init__(self, label, *, target: Annotated[str, RoleSpec(elements=(ElementKind.GROUP,))]):
        super().__init__(label)
        self.target = target

    def _get_laws(self, s):
        return [RuleLaw(lambda agent, system: None)]


census = CensusInteraction('census', target='hares').get_requirements().roles['target']
census.elements, census.accepts(ElementKind.GROUP), census.accepts(ElementKind.ELEMENT)
```

The role accepts the kind `group` alone, so it rejects the single population of hares. The section on errors below
shows what a run says when the two meet.

## Validation errors

A part that does not fit raises an error before anything compiles. `prepare` checks the arguments of each part and the
domains of the states that each role binds. [simulate][skcomplex.core.engine.runtime.simulate] then checks the whole
system and raises one error that lists every problem, each with its path, its message, a fix, and its code in
brackets. The variants below rebuild the model from the parts of the build tutorial and break one thing at a time.

```python continuation
from skcomplex.core.components.elements import DeclaredElement
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec

hares = DeclaredElement('hares', variables={'x': DeclaredVariableSpec(30.0, StateSpec())})
lynx = DeclaredElement('lynx', variables={'x': DeclaredVariableSpec(4.0, StateSpec())})
hunting = RateInteraction(
    'hunting', target='hares', state='x', sources=['lynx'], rate=lambda s: -0.028 * s.target.x * s.sources[0].x
)
pair = CompositeSystem(elements=[hares, lynx], interactions=[births, hunting])
round(float(simulate(pair, until=1).get_values('hares.x')[-1]), 1)
```

The two populations with births and hunting run without an error. The lynx keep 4.0 thousand without terms of their
own, so the hares grow at the net rate `0.55 - 0.028 * 4.0`, from 30.0 to 46.5 thousand in a year.

### A state the element does not carry

The most frequent error binds a role to an element that lacks the state the interaction reads. Here births change the
state `n` of the hares, which carry `x`.

```python continuation
from skcomplex.core.errors import RoleElementError

wrong_state = RateInteraction('births', target='hares', state='n', rate=lambda s: 0.55)
try:
    simulate(CompositeSystem(elements=[hares], interactions=[wrong_state]), until=1)
except RoleElementError as error:
    message = str(error)
message
```

The error has the code `role-element`. Its message says that the role `target` of births reads the state `n`, but the
element hares bound to it carries `x`. The fix is `state='x'`, or a state `n` declared on the element. The check
covers more than declared reads.

- A `RoleSpec(states=('n',))` states the read in advance, so `get_requirements()` shows it.
- An interaction that declares nothing is still checked, when `simulate` builds its laws and a law reads a state that
  the element does not have.
- A function passed to an interaction, such as the `rate` of `births`, is covered in the same way.
- An error of another kind inside a law reaches the caller unchanged.

### A role bound to a missing element

A role names an element by its [label](glossary.md#label), so a typing error or a forgotten element leaves it bound to nothing. Here the
lynx are left out of the system.

```python continuation
from skcomplex.core.errors import RoleError

try:
    simulate(CompositeSystem(elements=[hares], interactions=[hunting]), until=1)
except RoleError as error:
    message = str(error)
message.splitlines()[-1]
```

The code is `role`. The message says that hunting binds the role `sources` to `lynx`, which is not an element or group
path relative to the root system. The fix adds the element `lynx` to the system, or binds the role to the label of an
element it holds. A [nested system](glossary.md#nested-system) is reached by a path such as `village/farmers`.

### Two parts with one label

Every child of a system needs a label of its own, because roles, states, and parameters are named by it.

```python continuation
from skcomplex.core.errors import LabelError

twins = DeclaredElement('hares', variables={'x': DeclaredVariableSpec(4.0, StateSpec())})
try:
    simulate(CompositeSystem(elements=[hares, twins], interactions=[births]), until=1)
except LabelError as error:
    message = str(error)
message.splitlines()[-1]
```

The code is `label`, and the message asks for a unique label of letters, digits, and underscores. Renaming the second
element to `lynx` fixes it.

### A role that rejects the kind of its element

The census above accepts groups, and the population of hares is a single element.

```python continuation
counted = CompositeSystem(elements=[hares], interactions=[CensusInteraction('census', target='hares')])
try:
    simulate(counted, until=1)
except RoleElementError as error:
    message = str(error)
message.splitlines()[-1]
```

The code is again `role-element`. The message says that the role `target` of census accepts `group`, but hares is of
the kind `element`. The fix binds the role to a `GroupElement`, such as a group of individual hares.

### A state of a domain the role rejects

A rate cannot change a state whose values are only 0 and 1. `prepare` catches it, so the error appears before a run.

```python continuation
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

present = DeclaredElement(
    'hares', variables={'x': DeclaredVariableSpec(1, StateSpec(domain=FiniteEnumeratedSet([0, 1])))}
)
try:
    CompositeSystem(elements=[present], interactions=[births]).prepare()
except RoleElementError as error:
    message = str(error)
message
```

The message says that the role `target` of `births` accepts the domains real, integer, complex, and vector, but the
element `hares` has the state `x` of the domain `boolean`. The fix is a real state, or an interaction for booleans,
such as a `MapInteraction` that flips the state under discrete time.

### A law in the wrong time

A system that declares one time and holds laws of the other cannot run.

```python continuation
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.errors import ValidationError

try:
    simulate(CompositeSystem(elements=[hares], interactions=[births], time=DiscreteTime()), until=1)
except ValidationError as error:
    message = str(error)
message.splitlines()[-1]
```

The code is `law-time`. The message says that the `RateLaw` of births needs `ContinuousTime` while the system declares
`DiscreteTime`, and suggests either `ContinuousTime()` or laws that run in discrete time. Leaving the time out lets the
system take it from its laws.

### A name that does not exist

Model parameters, states, and run parameters are named, and a wrong name fails at once. The catalog class names the
birth rate of the prey `r`, not `alpha` as in the build tutorial, and its states `prey.x` and `predators.x`.

```python continuation
from skcomplex.core.errors import ParameterError

messages = []
try:
    system.set_params(alpha=0.6)
except ValueError as error:
    messages.append(str(error).split('. ')[-1])
try:
    simulate(system, until=20, initial_state={'hares.x': 10.0})
except ValidationError as error:
    messages.append(str(error))
try:
    simulate(system, until=20, schedule={'predation.alpha': lambda t: 0.0})
except ParameterError as error:
    messages.append(str(error))
messages
```

- `set_params` raises a `ValueError`, as scikit-learn does, and lists the valid parameters: `K`, `a`, `b`, `d`,
  `label`, `predators`, `prey`, and `r`.
- An `initial_state` with a wrong name raises a `ValidationError` that suggests `prey.x` and `predators.x`.
- A `schedule` names the parameters of the built parts by their path, and a wrong one raises a `ParameterError` that
  lists `growth.r`, `predation.a`, `predation.b`, and `death.d`.

## The solver

The [solver](glossary.md#solver) advances the compiled system. It can be left out, because the tags of a system decide its default solver.
Every solver class lists the tags it supports in `get_supported_tags()`, and `simulate` rejects a solver that lacks a
tag of the system with a [TagError][skcomplex.core.errors.TagError] that names it, before it compiles anything.

```python continuation
from skcomplex.core.engine.solvers import IterationSolver, RK4Solver
from skcomplex.core.errors import TagError

fits = tags.laws <= RK4Solver.get_supported_tags().laws
try:
    simulate(system, until=20, solver=IterationSolver())
except TagError as error:
    message = str(error)
fits, message, simulate(system, until=20, dt=1.0, solver=RK4Solver()).get_values('prey.x').round(1)[:6].tolist()
```

`RK4Solver`, the classical Runge-Kutta method of order four, integrates the rates of the model, and its run matches the
default solver of the build tutorial: 30.0, 46.4, 68.5, 81.2, 48.8, 18.6 thousand hares from 1900 to 1905.
`IterationSolver`, which iterates maps, does not support the time `continuous`, and the message says so. The
[solvers page](../../parts/solvers.md) lists what every solver supports, and the
[backends page](../../parts/backends.md) what every [backend](glossary.md#backend) supports. The tutorial [Simulate](simulate.md) runs the
checked system.
