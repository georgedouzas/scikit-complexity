# Choosing compatible components

Find the parts that fit together before you build a system, and read why a part does not fit. Choose them in this order:
time and space, then elements, then interactions, then the solver. Each choice narrows the next, and you can ask every
part what it needs before you build anything.

Every part tells what it is through its tags. `get_tags()` of an element returns the domains of its states, such as
`real`, `get_tags()` of an interaction returns the kinds of the laws it returns and their time, and `get_tags()` of a
system joins those of its parts. The Parts pages list the tags of every
[element](../../../parts/elements.md), [interaction](../../../parts/interactions.md), [law](../../../parts/laws.md),
[solver](../../../parts/solvers.md), and [backend](../../../parts/backends.md) of the core, each with a compatible and an
incompatible example, so start there to find candidates. Then check a candidate in code, as this section shows.

## Time and space

An interaction states the time and the space its laws run in, and an empty `space` means any space.
`get_requirements()` returns them.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import MapInteraction, RateInteraction
from skcomplex.core.components.spaces import ContinuousTime, DiscreteTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.specs import StateSpec
from skcomplex.core.types import ElementKind

growth = RateInteraction('growth', target='rabbits', state='n', rate=lambda s: 0.5 * s.target.n)
needs = growth.get_requirements()
assert needs.time == (ContinuousTime,)
assert needs.space == ()
assert MapInteraction('step', target='rabbits', state='n', function=lambda s: s.target.n).get_requirements().time == (
    DiscreteTime,
)
[law.__name__ for law in needs.laws]
```

`RateInteraction` returns rates, so it needs continuous time and accepts any space. `MapInteraction` returns updates
and rules, so it needs discrete time. The tags say the same for the arguments you passed.

```python continuation
sorted(growth.get_tags().laws), sorted(growth.get_tags().time)
```

## Elements

An element carries the states you declare, and `get_states()` returns them with their specs.

```python continuation
class PopulationElement(Element):
    def __init__(self, label, *, n: Annotated[float, StateSpec()] = 1.0):
        super().__init__(label)
        self.n = n


rabbits = PopulationElement('rabbits', n=40)
list(rabbits.get_states()), sorted(rabbits.get_tags().domains)
```

A `GroupElement` returns the states of its members. Every element is of one kind, and the kind is what a role accepts or
rejects.

- `Element` is of the kind `element`, one member with its states.
- `GroupElement` is of the kind `group`, many identical members. With `variable=True` it is a `variable-group`, whose
  members appear and disappear.
- A group on a space is a `site-group`, an `edge-group`, or a `hyperedge-group`, by what its members sit on. A
  `GroupElement` on `'sites'` and a `FieldElement` are site groups.

## Interactions

An interaction names what each of its roles needs. `get_requirements().roles` holds, for every role, the kinds of
element it accepts and the states it reads for the arguments you passed. Compare them with what the element carries.
The kind `any` accepts every kind of element.

```python continuation
target = needs.roles['target']
target.states, target.elements
assert target.accepts(ElementKind.ELEMENT)
assert all(state in rabbits.get_states() for state in target.states)
```

The two agree, so the interaction fits the element. Some roles accept only some kinds. A census that counts the agents
of a group declares in its `RoleSpec` that its `target` accepts a group and not a single element.

```python continuation
from typing import ClassVar

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


census = CensusInteraction('census', target='rabbits').get_requirements().roles['target']
census.elements
assert census.accepts(ElementKind.GROUP)
assert not census.accepts(ElementKind.ELEMENT)
```

## When a part does not fit

`simulate` checks the system before it runs and raises an error that lists every problem, each with its path, its
message, a fix, and its code in brackets. Here the rate reads the state `n` of a tank that carries `level`.

```python continuation
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import RoleElementError, ValidationError


class TankElement(Element):
    def __init__(self, label, *, level: Annotated[float, StateSpec()] = 1.0):
        super().__init__(label)
        self.level = level


wrong = CompositeSystem(
    elements=[TankElement('tank')],
    interactions=[RateInteraction('growth', target='tank', state='n', rate=1.0)],
)
try:
    simulate(wrong, until=1, dt=0.1)
except RoleElementError as error:
    message = str(error)
assert "reads the state 'n', but the element tank bound to it carries level" in message
message
```

A role filled by an unsuitable element raises `RoleElementError` with the code `role-element`. A role that rejects the
kind of its element gets the same code, and a law in the wrong time gets `law-time`.

```python continuation
kinds = CompositeSystem(elements=[TankElement('tank')], interactions=[CensusInteraction('census', target='tank')])
times = CompositeSystem(
    elements=[TankElement('tank')],
    interactions=[RateInteraction('grow', target='tank', state='level', rate=1.0)],
    time=DiscreteTime(),
)
messages = []
for system in (kinds, times):
    try:
        simulate(system, until=1)
    except ValidationError as error:
        messages.append(str(error).splitlines()[-1])
messages
```

An interaction of your own gets the same check.

- A `RoleSpec(states=('n',))` states the read in advance, so `get_requirements()` shows it.
- An interaction that declares nothing is still checked, when `simulate` builds its laws and a law reads a state that
  the element does not have.
- A function you pass, such as the `rate` of `growth`, is covered in the same way.
- An error of another kind inside a law reaches you unchanged.

## The solver

You can leave the solver out. The laws of a system give its tags, and the tags decide the default solver. Every solver
class lists the tags it supports in `get_supported_tags()`, and `simulate` rejects a solver that lacks a tag of the
system with a `TagError` that names it, before it compiles anything.

```python continuation
from skcomplex.core.engine.solvers import IterationSolver, RK4Solver
from skcomplex.core.errors import TagError

system = CompositeSystem(elements=[rabbits], interactions=[growth])
assert system.get_tags().laws <= RK4Solver.get_supported_tags().laws
try:
    simulate(system, until=1, solver=IterationSolver())
except TagError as error:
    message = str(error)
message, simulate(system, until=1, dt=0.1, solver=RK4Solver()).get_values('rabbits.n')[-1].round(1)
```

`RK4Solver` integrates the differential equations of `growth`, and `IterationSolver`, which iterates maps, does not.
The [solvers page](../../../parts/solvers.md) lists what every solver supports, and the
[backends page](../../../parts/backends.md) what every backend supports.
