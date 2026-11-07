# Choosing compatible components

Find the parts that fit together before you build a system, and read why a part does not fit. Choose them in this order:
time and space, then elements, then interactions, then the solver. Each choice narrows the next, and you can ask every
part what it needs before you build anything.

The [parts pages](../../../parts/elements.md) list every element and interaction of the core, with the domains and law kinds
each declares, so start there to find candidates. Then check a candidate in code, as this section shows.

## Time and space

An interaction states the time and the space its laws run in, and an empty `space` means any space.
`get_requirements()` returns them.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import MapInteraction, MovementInteraction, RateInteraction
from skcomplex.core.components.spaces import ContinuousTime, DiscreteTime
from skcomplex.core.components.system import System
from skcomplex.core.engine.compile import PassCompiler
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
and rules, so it needs discrete time.

## Elements

An element carries the states you declare, and `get_states()` returns them with their specs.

```python continuation
class PopulationElement(Element):
    def __init__(self, label, *, n: Annotated[float, StateSpec()] = 1.0):
        super().__init__(label)
        self.n = n


rabbits = PopulationElement('rabbits', n=40)
list(rabbits.get_states())
```

A `GroupElement` returns the states of its members. Every element is of one kind, and the kind is what a role accepts or
rejects.

- `Element` is of the kind `element`, one member with its states.
- `GroupElement` is of the kind `group`, many identical members. With `variable=True` it is a `variable-group`, whose
  members appear and disappear.
- A group on a space is a `site-group`, an `edge-group`, or a `hyperedge-group`, by what its members sit on. A
  `LatticeElement` and a `FieldElement` are site groups.

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

The two agree, so the interaction fits the element. Some roles accept only some kinds. `MovementInteraction` moves the
agents of a group, so its `target` accepts a group and not a single element.

```python continuation
walk = MovementInteraction('walk', target='rabbits').get_requirements().roles['target']
walk.elements
assert walk.accepts(ElementKind.GROUP)
assert not walk.accepts(ElementKind.ELEMENT)
```

## When a part does not fit

Read the validation report before you run. `PassCompiler(...).prepare(system).report_` holds every problem of the
system, each with a path, a message, and a fix. Here the rate reads the state `n` of a tank that carries `level`.

```python continuation
class TankElement(Element):
    def __init__(self, label, *, level: Annotated[float, StateSpec()] = 1.0):
        super().__init__(label)
        self.level = level


wrong = System(
    elements=[TankElement('tank')],
    interactions=[RateInteraction('growth', target='tank', state='n', rate=1.0)],
)
issue = PassCompiler().prepare(wrong).report_.issues[0]
issue.code, issue.path
issue.message
issue.fix
assert issue.code == 'role-element'
assert issue.message == (
    "The role 'target' of growth reads the state 'n', but the element tank bound to it carries level."
)
```

A role filled by an unsuitable element is the issue `role-element`. A role that rejects the kind of its element gets
the same issue, and a law in the wrong time gets `law-time`.

```python continuation
kinds = System(elements=[TankElement('tank')], interactions=[MovementInteraction('walk', target='tank')])
PassCompiler().prepare(kinds).report_.issues[0].message
times = System(
    elements=[TankElement('tank')],
    interactions=[RateInteraction('grow', target='tank', state='level', rate=1.0)],
    time=DiscreteTime(),
)
PassCompiler().prepare(times).report_.issues[0].message
```

If you do not read the report first, `prepare` of a `Simulation` raises it as an error that lists every problem.

```python continuation
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.errors import RoleElementError

try:
    Simulation(wrong, dt=0.1).prepare()
except RoleElementError as error:
    message = str(error)
message
```

An interaction of your own gets the same report.

- A `RoleSpec(states=('n',))` states the read in advance, so `get_requirements()` shows it.
- An interaction that declares nothing is still reported, when validation builds its laws and a law reads a state that
  the element does not have.
- A function you pass, such as the `rate` of `growth`, is covered in the same way.
- An error of another kind inside a law reaches you unchanged.

## The solver

You can leave the solver out. The laws of a system decide its mathematical form, and the form decides the default
solver. To choose one yourself, pass it to the compiler and read the report before you run.

```python continuation
from skcomplex.core.engine.solvers import IterationSolver, RK4Solver

system = System(elements=[rabbits], interactions=[growth])
PassCompiler().prepare(system).run().problem_.form.value
assert PassCompiler(solver=RK4Solver()).prepare(system).report_.ok
assert not PassCompiler(solver=IterationSolver()).prepare(system).report_.ok
Simulation(system, dt=0.1).prepare().run(until=1).trajectory_.values('rabbits.n')[-1].round(1)
```

`RK4Solver` integrates the differential equations of `growth`, and `IterationSolver`, which iterates maps, does not.
The [solvers page](../../../parts/solvers.md) lists what every solver supports.
