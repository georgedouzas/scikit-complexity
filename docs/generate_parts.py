"""Generate one page per kind of part, with a table of the capability tags of its classes and executed examples."""

import inspect
from collections.abc import Iterable
from types import ModuleType

import mkdocs_gen_files

from skcomplex.core.components import elements, interactions, laws
from skcomplex.core.components.base import BasePart
from skcomplex.core.components.elements import BaseElement
from skcomplex.core.components.interactions import BaseInteraction
from skcomplex.core.components.laws import BaseLaw
from skcomplex.core.engine import backends, solvers
from skcomplex.core.engine.backends import BaseBackend
from skcomplex.core.engine.solvers import BaseSolver
from skcomplex.core.specs import LAWS, SPACES, TIMES, RoleSpec, StateSpec

PARTS = 'parts'


def _public_classes[T](package: ModuleType, base: type[T]) -> list[type[T]]:
    """Return the public classes of a package that derive from a base and can be built, sorted by name."""
    named = [getattr(package, name) for name in package.__all__]
    found = [item for item in named if isinstance(item, type) and issubclass(item, base)]
    return sorted((item for item in found if not inspect.isabstract(item)), key=lambda item: item.__name__)


def _values(values: Iterable[str] | None, every: Iterable[str] = (), empty: str = 'any') -> str:
    """Return tag values as a sorted list of code spans, `empty` for none, and `any` for all the values of the field."""
    if values is None:
        return 'unknown until run'
    found = sorted(values)
    if not found:
        return empty
    if every and set(found) == set(every):
        return 'any'
    return ', '.join(f'`{value}`' for value in found)


def _table(header: list[str], rows: Iterable[list[str]]) -> list[str]:
    """Return the Markdown lines of a table."""
    lines = [f'| {" | ".join(header)} |', f'|{"---|" * len(header)}']
    return lines + [f'| {" | ".join(row)} |' for row in rows]


def _page(title: str, intro: str, header: list[str], rows: Iterable[list[str]], examples: str) -> str:
    """Return a page of a kind of part: its title, its introduction, its table, and its examples."""
    lines = [f'# {title}', '', intro.strip(), '', *_table(header, rows), '', examples.strip(), '']
    return '\n'.join(lines)


def _supported_rows(runners: Iterable[type[BaseSolver] | type[BaseBackend]]) -> list[list[str]]:
    """Return the supported time, space, and law kinds of solvers or backends, one row each."""
    rows = []
    for runner in runners:
        tags = runner.get_supported_tags()
        rows.append(
            [
                f'`{runner.__name__}`',
                _values(tags.time, TIMES),
                _values(tags.space, SPACES),
                _values(tags.laws, LAWS),
            ],
        )
    return rows


def _declares_support(runner: type[BaseSolver] | type[BaseBackend]) -> bool:
    """Check whether a solver or a backend class declares what it supports, which generic classes do not."""
    try:
        runner.get_supported_tags()
    except NotImplementedError:
        return False
    return True


def _elements_page() -> str:
    """Return the page of elements, with the states, domains, and law kinds every element class declares."""
    rows = []
    for element in _public_classes(elements, BaseElement):
        states = {name: spec for name, spec in element._get_specs().items() if isinstance(spec, StateSpec)}
        domains = {spec._get_domain_tag() for spec in states.values()} - {None}
        tags = BasePart._get_law_tags(element._get_declared_laws())
        names = ', '.join(f'`{name}`' for name in states) or 'from a subclass'
        whole = 'from a subclass' if not states else 'a whole structure'
        rows.append([f'`{element.__name__}`', names, _values(domains, empty=whole), _values(tags.laws, empty='none')])
    intro = """
An element holds the states of a system, such as an amount, a position, or a cell of a lattice. The domain of each
state is a tag, read from its `StateSpec`, and `get_tags()` of an element returns its domains. Every role of an
interaction lists the domains it accepts, so the domains of an element decide which interactions it can be bound to.
The table lists what each element class of the core declares. A subclass adds the states your interactions read, and
their domains with them.
"""
    examples = """
## A compatible element

A compartment holds a real amount, and the `target` of a `RateInteraction` accepts real states.

```python
from skcomplex.core.components.elements import CompartmentElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate

tank = CompartmentElement('tank', n=10.0)
decay = RateInteraction('decay', target='tank', state='n', rate=lambda s: -0.5 * s.target.n)
sorted(tank.get_tags().domains)
trajectory = simulate(System(elements=[tank], interactions=[decay]), until=2)
round(trajectory.values('tank.n')[-1], 2)
```

## An incompatible element

A switch that is on or off holds a boolean state. A rate does not change a boolean, so the system rejects the binding
and names the domain.

```python
from typing import Annotated

from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import RoleElementError
from skcomplex.core.specs import StateSpec


class SwitchElement(Element):
    def __init__(self, label, *, on: Annotated[int, StateSpec(FiniteEnumeratedSet([0, 1]))] = 0):
        super().__init__(label)
        self.on = on


switch = SwitchElement('switch')
sorted(switch.get_tags().domains)
flip = RateInteraction('flip', target='switch', state='on', rate=1.0)
try:
    simulate(System(elements=[switch], interactions=[flip]), until=1)
except RoleElementError as error:
    message = str(error)
message
assert "domain 'boolean'" in message
```
"""
    return _page('Elements', intro, ['Element', 'States', 'Domains', 'Law kinds'], rows, examples)


def _role_cell(name: str, role: RoleSpec) -> str:
    """Return one role of an interaction with the domains it accepts."""
    if role.noise:
        return f'`{name}`: a `WienerProcessSpec`'
    return f'`{name}`: {_values(role.domains, empty="any domain")}'


def _interactions_page() -> str:
    """Return the page of interactions, with the domains every role accepts and the law kinds of every class."""
    rows = []
    for interaction in _public_classes(interactions, BaseInteraction):
        roles = [(name, spec) for name, spec in interaction._get_specs().items() if isinstance(spec, RoleSpec)]
        cells = '<br>'.join(_role_cell(name, role) for name, role in roles) or 'from a subclass'
        tags = BasePart._get_law_tags(interaction._get_declared_laws())
        rows.append([f'`{interaction.__name__}`', cells, _values(tags.laws, empty='from a subclass')])
    intro = """
An interaction changes the states of the elements bound to its roles. Each role lists the domains of the states it
accepts, and each interaction lists the kinds of the laws it can return, which its arguments choose among. The law
kinds give the time the interaction runs in, so the parts of one system must agree on it. `get_tags()` of an
interaction returns the law kinds for the arguments you passed.
"""
    examples = """
## A compatible interaction

A rate returns a `rate` law of continuous time, and its `target` accepts the real amount of a compartment.

```python
from skcomplex.core.components.elements import CompartmentElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate

tank = CompartmentElement('tank', n=10.0)
decay = RateInteraction('decay', target='tank', state='n', rate=lambda s: -0.5 * s.target.n)
sorted(decay.get_tags().laws), sorted(decay.get_tags().time)
trajectory = simulate(System(elements=[tank], interactions=[decay]), until=2)
round(trajectory.values('tank.n')[-1], 2)
```

## An incompatible interaction

A map returns an `update` law of discrete time. It cannot join the rate in one system without a time that joins them,
so the system names the tag the two parts disagree on.

```python
from skcomplex.core.components.elements import CompartmentElement
from skcomplex.core.components.interactions import MapInteraction, RateInteraction
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import TagError

tank = CompartmentElement('tank', n=10.0)
decay = RateInteraction('decay', target='tank', state='n', rate=lambda s: -0.5 * s.target.n)
halve = MapInteraction('halve', target='tank', state='n', function=lambda s: s.target.n / 2)
sorted(halve.get_tags().laws), sorted(halve.get_tags().time)
try:
    simulate(System(elements=[tank], interactions=[decay, halve]), until=1)
except TagError as error:
    message = str(error)
message
assert "tag 'time'" in message
```
"""
    return _page('Interactions', intro, ['Interaction', 'Roles and accepted domains', 'Law kinds'], rows, examples)


def _laws_page() -> str:
    """Return the page of laws, with the kind, time, space, and randomness every law class declares."""
    rows = []
    for law in _public_classes(laws, BaseLaw):
        tags = BasePart._get_law_tags([law])
        spaces = {tag for space in law.get_requirements().space for tag in getattr(space, 'space_tags', ())}
        cells = [_values(tags.laws), _values(tags.time, TIMES), _values(spaces, SPACES), f'`{tags.randomness}`']
        rows.append([f'`{law.__name__}`', *cells])
    intro = """
A law is the mathematical object an interaction returns, such as a rate, a noise, a jump, or an update. Its kind is a
tag, and so are the times and spaces it runs in and its randomness. The law kinds of a system are those of its
interactions, and a solver or a backend runs a system only when it supports every one of them. The table lists what
each law class declares.
"""
    examples = """
## A compatible law

An arrival that adds one unit at random times returns a `jump` law, which `GillespieSolver` supports.

```python
from skcomplex.core.components.elements import CompartmentElement
from skcomplex.core.components.interactions import JumpInteraction
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import GillespieSolver

queue = CompartmentElement('queue', n=0.0)
arrive = JumpInteraction('arrive', target='queue', changes={'n': 1}, rate=lambda s: 2.0)
sorted(arrive.get_tags().laws)
system = System(elements=[queue], interactions=[arrive])
trajectory = simulate(system, until=5, solver=GillespieSolver(), random_state=0)
trajectory.values('queue.n')[-1]
```

## An incompatible law

`EulerSolver` integrates rates and does not draw jumps, so it rejects the system and names the law kind.

```python
from skcomplex.core.components.elements import CompartmentElement
from skcomplex.core.components.interactions import JumpInteraction
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import EulerSolver
from skcomplex.core.errors import TagError

queue = CompartmentElement('queue', n=0.0)
arrive = JumpInteraction('arrive', target='queue', changes={'n': 1}, rate=lambda s: 2.0)
try:
    simulate(System(elements=[queue], interactions=[arrive]), until=5, solver=EulerSolver())
except TagError as error:
    message = str(error)
message
assert 'laws jump' in message
```
"""
    return _page('Laws', intro, ['Law', 'Kind', 'Time', 'Space', 'Randomness'], rows, examples)


def _solvers_page() -> str:
    """Return the page of solvers, with the form and the supported tags of every solver."""
    found = [solver for solver in _public_classes(solvers, BaseSolver) if _declares_support(solver)]
    ordered = sorted(found, key=lambda solver: (str(solver._get_form()).lower(), solver.__name__))
    rows = [
        [row[0], str(solver._get_form()), *row[1:]]
        for solver, row in zip(ordered, _supported_rows(ordered), strict=True)
    ]
    intro = """
A solver steps a system through time, or solves it at once for a static system. `get_supported_tags()` of a solver
class returns the times, spaces, and law kinds it supports. `simulate` compares them with the tags of the system before
compiling, and rejects a solver that lacks one. Leave the solver out and the tags choose the default solver. The table
lists the supported tags of every solver, grouped by the mathematical form it solves.
"""
    examples = """
## A compatible solver

A rate with a noise term returns a `rate` and a `noise` law, and `EulerMaruyamaSolver` supports both.

```python
from skcomplex.core.components.elements import CompartmentElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import EulerMaruyamaSolver

tank = CompartmentElement('tank', n=10.0)
decay = RateInteraction('decay', target='tank', state='n', rate=lambda s: -0.5 * s.target.n, noise=0.1)
system = System(elements=[tank], interactions=[decay])
sorted(system.get_tags().laws), sorted(EulerMaruyamaSolver.get_supported_tags().laws)
trajectory = simulate(system, until=1, solver=EulerMaruyamaSolver(), random_state=0)
round(trajectory.values('tank.n')[-1], 2)
```

## An incompatible solver

`RK4Solver` integrates deterministic equations and does not support the `noise` law kind.

```python
from skcomplex.core.components.elements import CompartmentElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.core.errors import TagError

tank = CompartmentElement('tank', n=10.0)
decay = RateInteraction('decay', target='tank', state='n', rate=lambda s: -0.5 * s.target.n, noise=0.1)
try:
    simulate(System(elements=[tank], interactions=[decay]), until=1, solver=RK4Solver())
except TagError as error:
    message = str(error)
message
assert 'laws noise' in message
```
"""
    return _page('Solvers', intro, ['Solver', 'Form', 'Time', 'Space', 'Law kinds'], rows, examples)


def _backends_page() -> str:
    """Return the page of backends, with the supported tags of every backend."""
    found = [backend for backend in _public_classes(backends, BaseBackend) if _declares_support(backend)]
    rows = _supported_rows(found)
    intro = """
A backend turns the compiled laws of a system into numeric functions, with NumPy, Numba, or JAX.
`get_supported_tags()` of a backend class returns the times, spaces, and law kinds it can write functions of.
`simulate` rejects a backend that lacks a tag of the system, and uses the NumPy backend when you leave it out.
"""
    examples = """
## A compatible backend

`NumbaBackend` supports every tag of a decay, so it runs the system.

```python
from skcomplex.core.components.elements import CompartmentElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.system import System
from skcomplex.core.engine.backends import NumbaBackend
from skcomplex.core.engine.runtime import simulate

tank = CompartmentElement('tank', n=10.0)
decay = RateInteraction('decay', target='tank', state='n', rate=lambda s: -0.5 * s.target.n)
trajectory = simulate(System(elements=[tank], interactions=[decay]), until=1, backend=NumbaBackend())
round(trajectory.values('tank.n')[-1], 2)
```

## An incompatible backend

`JaxBackend` writes deterministic functions only, so it rejects a noise term and names the law kind.

```python
from skcomplex.core.components.elements import CompartmentElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.system import System
from skcomplex.core.engine.backends import JaxBackend
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import TagError

tank = CompartmentElement('tank', n=10.0)
decay = RateInteraction('decay', target='tank', state='n', rate=lambda s: -0.5 * s.target.n, noise=0.1)
sorted(JaxBackend.get_supported_tags().laws)
try:
    simulate(System(elements=[tank], interactions=[decay]), until=1, backend=JaxBackend())
except TagError as error:
    message = str(error)
message
assert 'laws noise' in message
```
"""
    return _page('Backends', intro, ['Backend', 'Time', 'Space', 'Law kinds'], rows, examples)


def main() -> None:
    """Write the page of every kind of part into the documentation."""
    pages = {
        'elements': _elements_page,
        'interactions': _interactions_page,
        'laws': _laws_page,
        'solvers': _solvers_page,
        'backends': _backends_page,
    }
    for name, page in pages.items():
        with mkdocs_gen_files.open(f'{PARTS}/{name}.md', 'w') as file:
            file.write(page())


main()
