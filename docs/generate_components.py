"""Generate one page per kind of component.

Each page has a table of the capability tags of its classes, and executed examples.
"""

import inspect
import sys
from collections.abc import Callable, Iterable
from types import ModuleType

import griffe
import mkdocs_gen_files
from gallery_tools import library_systems

from skcomplex.core import analysis, experiments
from skcomplex.core.components import elements, interactions, laws, spaces, systems
from skcomplex.core.components.base import BaseComponent
from skcomplex.core.components.elements import BaseElement
from skcomplex.core.components.interactions import BaseInteraction
from skcomplex.core.components.laws import BaseLaw
from skcomplex.core.components.spaces import BaseNeighborhood, BaseSpace, BaseTime
from skcomplex.core.components.systems import BaseSystem, CompositeSystem
from skcomplex.core.engine import backends, solvers
from skcomplex.core.engine.backends import BaseBackend
from skcomplex.core.engine.solvers import BaseSolver
from skcomplex.core.specs import LAWS, SPACES, TIMES, AnalysisTags, ParameterSpec, RoleSpec, StateSpec, Tags
from skcomplex.systems import all_systems
from skcomplex.systems.general import dynamics
from skcomplex.systems.general.reactions import ReactionNetworkSystem

FAMILIES = 'components'
REFERENCE = '../overview/user_guide/core_components'
SHOWN = 10
EXPERIMENT_NEEDS = {
    'ensemble': 'the randomness `stochastic`, or a solver that draws',
}


def _public_classes[T](package: ModuleType, base: type[T]) -> list[type[T]]:
    """Return the public classes of a package that derive from a base.

    Only classes that can be built count, sorted by name.
    """
    named = [getattr(package, name) for name in package.__all__]
    found = [item for item in named if isinstance(item, type) and issubclass(item, base)]
    return sorted((item for item in found if not inspect.isabstract(item)), key=lambda item: item.__name__)


def _values(values: Iterable[str] | None, every: Iterable[str] = (), empty: str = 'any') -> str:
    """Return tag values as a sorted list of code spans.

    The list is `empty` for none, and `any` for all the values of the field.
    """
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
    """Return a page of a kind of component.

    It holds its title, its introduction, its table, and its examples.
    """
    lines = [f'# {title}', '', intro.strip(), '', *_table(header, rows), '', examples.strip(), '']
    return '\n'.join(lines)


def _supported_rows(runners: Iterable[type[BaseSolver] | type[BaseBackend]]) -> list[list[str]]:
    """Return the supported time, space, and law kinds of solvers or backends.

    There is one row each.
    """
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
    """Check whether a solver or a backend class declares what it supports.

    Generic classes do not.
    """
    try:
        runner.get_supported_tags()
    except NotImplementedError:
        return False
    return True


def _paragraphs(item: type) -> list[str]:
    """Return the paragraphs of the docstring of a class before its sections, each on one line."""
    text = inspect.getdoc(item) or ''
    paragraphs = [' '.join(paragraph.split()) for paragraph in text.split('\n\n')]
    return [paragraph for paragraph in paragraphs if paragraph and not paragraph.endswith(':')]


def _comparison(item: type) -> list[str]:
    """Return what a class adds, when to choose it, and its uses, read from its docstring."""
    paragraphs = _paragraphs(item)
    compare = next((paragraph for paragraph in paragraphs if paragraph.startswith('Unlike')), '')
    adds, _, choose = compare.partition(' Choose it when ')
    uses = next((paragraph for paragraph in paragraphs if paragraph.startswith('It appears in')), '')
    uses = uses.removeprefix('It appears in many fields. ')
    return [adds, choose[:1].upper() + choose[1:], uses]


def _sections(item: type) -> dict[str, list[tuple[str, str]]]:
    """Return the arguments and the errors of the docstring of a class, parsed with griffe."""
    docstring = griffe.Docstring(inspect.getdoc(item) or '', parser='google', parser_options={'warnings': False})
    found: dict[str, list[tuple[str, str]]] = {'parameters': [], 'raises': []}
    for section in docstring.parse():
        if section.kind.value in found:
            for entry in section.value:
                name = getattr(entry, 'name', '') or str(getattr(entry, 'annotation', ''))
                sentence = ' '.join(entry.description.split()).split('. ')[0].rstrip('.')
                found[section.kind.value].append((name, f'{sentence}.'))
    return found


def _catalog_classes() -> list[tuple[str, type[BaseSystem], set[type]]]:
    """Return the gallery name, the class, and the classes every catalog system names or subclasses.

    A system names the classes in the globals of its module and of the field `common` modules it imports.
    """
    found = []
    for name, system in sorted(library_systems().items()):
        module = sys.modules[system.__module__]
        modules = [module]
        for value in vars(module).values():
            origin = getattr(value, '__module__', '') or ''
            if isinstance(value, type) and '.common' in origin and origin in sys.modules:
                modules.append(sys.modules[origin])
        classes = {value for item in modules for value in vars(item).values() if isinstance(value, type)}
        found.append((name, system, classes))
    return found


def _links(items: list[str], more: str) -> str:
    """Return the first links of a list, and a link to the rest when there are more."""
    if not items:
        return 'none yet'
    shown = ', '.join(items[:SHOWN])
    return shown if len(items) <= SHOWN else f'{shown}, and more in {more}'


def _fact_sheet(
    item: type[BaseComponent],
    catalog: list[tuple[str, type[BaseSystem], set[type]]],
    family: str,
) -> list[str]:
    """Return the fact sheet of a class of the core: its variables, tags, solvers, errors, links, and uses.

    Elements and interactions list their states and parameters, laws their kind, and spaces and times their tag.
    """
    if family in {'elements', 'interactions'}:
        specs = item._get_specs()
        states = {name: spec for name, spec in specs.items() if isinstance(spec, StateSpec)}
        parameters = [name for name, spec in specs.items() if isinstance(spec, ParameterSpec)]
        laws = BaseComponent._get_law_tags(item._get_declared_laws()).laws
        tags = Tags(domains=frozenset(spec._get_domain_tag() for spec in states.values()) - {None}, laws=laws)
        domain_cells = {
            name: _values({spec._get_domain_tag()} - {None}, empty='a whole structure') for name, spec in states.items()
        }
        state_cell = ', '.join(f'`{name}` ({cell})' for name, cell in domain_cells.items())
        arguments = inspect.signature(item).parameters
        origin = 'given as data' if 'variables' in arguments else 'declared by a subclass'
        origin = 'those of the element class of its members' if 'element' in arguments else origin
        facts = [
            f'- States: {state_cell or origin}',
            f'- Parameters: {", ".join(f"`{name}`" for name in parameters) or "none declared"}',
            f'- Law kinds: {_values(laws, empty="none of its own")}',
        ]
    elif issubclass(item, BaseLaw):
        tags = BaseComponent._get_law_tags([item]) if item.get_requirements() is not None else Tags()
        facts = [f'- Law kind: {_values(tags.laws, empty="from a subclass")}', f'- Randomness: `{tags.randomness}`']
    elif issubclass(item, BaseTime):
        tags = Tags(time=frozenset({item.time_tag}) - {''})
        facts = [f'- Time tag: {_values(tags.time, empty="from a subclass")}']
    elif issubclass(item, BaseSpace):
        tags = Tags(space=frozenset(item.space_tags[:1]))
        facts = [f'- Space tags: {_values(item.space_tags, empty="from a subclass")}']
    else:
        tags = Tags()
        facts = []
    runners = [runner for runner in _public_classes(solvers, BaseSolver) if _declares_support(runner)]
    accepted = [runner for runner in runners if not runner.get_supported_tags().get_missing(tags)]
    times = {time for runner in accepted for time in runner.get_supported_tags().time}
    engines = [runner for runner in _public_classes(backends, BaseBackend) if _declares_support(runner)]
    compiled = [runner for runner in engines if not runner.get_supported_tags().get_missing(tags)]
    if family != 'systems' and not issubclass(item, BaseNeighborhood):
        facts += [
            f'- Times: {_values(times, TIMES)}',
            f'- Solvers: {_values([runner.__name__ for runner in accepted], [runner.__name__ for runner in runners])}',
            f'- Backends: {_values([runner.__name__ for runner in compiled], [runner.__name__ for runner in engines])}',
        ]
    sections = _sections(item)
    users = [f'[{name}](../systems/{system.field}/{name}.md)' for name, system, classes in catalog if item in classes]
    realizations = sorted(
        {
            kind.__name__
            for _, _, classes in catalog
            for kind in classes
            if kind is not item and issubclass(kind, item) and kind.__module__.startswith('skcomplex.systems.')
        },
    )
    api = f'{item.__module__.rsplit(".", 1)[0]}.{item.__name__}'
    reference = f'{REFERENCE}/{family}.md#{item.__name__.lower()}'
    lines = [
        f'### {item.__name__}',
        '',
        (inspect.getdoc(item) or '').splitlines()[0],
        '',
        *facts,
        '- Arguments:' + ('' if sections['parameters'] else ' none of its own'),
        *(f'    - `{name}`: {text}' for name, text in sections['parameters']),
        '- Errors:' + ('' if sections['raises'] else ' none of its own'),
        *(f'    - `{name}`: {text}' for name, text in sections['raises']),
        f'- Catalog systems: {_links(users, "the [catalog](../systems/catalog.md)")}',
        f'- Field realizations: {_links([f"`{name}`" for name in realizations], "the catalog")}',
        f'- See also: [reference]({reference}), [API][{api}]',
        '',
    ]
    return lines


def _family_classes(package: ModuleType) -> list[type[BaseComponent]]:
    """Return the public component classes of a family package, sorted by name."""
    every = (getattr(package, name) for name in package.__all__)
    found = (item for item in every if isinstance(item, type) and issubclass(item, BaseComponent))
    return sorted(found, key=lambda item: item.__name__)


def _facts(items: list[type[BaseComponent]], family: str) -> str:
    """Return the comparison table and the fact sheets of the classes of a family."""
    catalog = _catalog_classes()
    rows = [[f'[`{item.__name__}`](#{item.__name__.lower()})', *_comparison(item)] for item in items]
    lines = ['## Comparison', '', *_table(['Class', 'What it adds', 'Choose it when', 'Uses'], rows), '']
    lines += ['## Fact sheets', '']
    for item in items:
        lines += _fact_sheet(item, catalog, family)
    return '\n'.join(lines)


def _elements_page() -> str:
    """Return the page of elements.

    It shows the states, domains, and law kinds every element class declares.
    """
    rows = []
    for element in _public_classes(elements, BaseElement):
        states = {name: spec for name, spec in element._get_specs().items() if isinstance(spec, StateSpec)}
        domains = {spec._get_domain_tag() for spec in states.values()} - {None}
        tags = BaseComponent._get_law_tags(element._get_declared_laws())
        names = ', '.join(f'`{name}`' for name in states) or 'from a subclass'
        whole = 'from a subclass' if not states else 'a whole structure'
        rows.append([f'`{element.__name__}`', names, _values(domains, empty=whole), _values(tags.laws, empty='none')])
    intro = """
An element holds the states of a system, the numbers that change as the system runs, such as an amount, a position, or
a cell of a lattice. Each state has a domain, a tag read from its `StateSpec` that names the set its values lie in,
such as `real` or `boolean`. `get_tags()` of an element returns its domains. Every role of an interaction lists the
domains it accepts, so the domains of an element decide which interactions it can be bound to. The table lists the
states, domains, and law kinds that each element class of the core declares. A subclass adds the states its
interactions read, and their domains with them.
"""
    examples = """
## A compatible element

A `DeclaredElement` takes its states as data. Here a tank holds a real amount, and the `target` role of a
`RateInteraction` accepts real states.

```python
from skcomplex.core.components.elements import DeclaredElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec

tank = DeclaredElement('tank', variables={'x': DeclaredVariableSpec(10.0, StateSpec())})
decay = RateInteraction('decay', target='tank', state='x', rate=lambda s: -0.5 * s.target.x)
sorted(tank.get_tags().domains)
trajectory = simulate(CompositeSystem(elements=[tank], interactions=[decay]), until=2)
round(trajectory.get_values('tank.x')[-1], 2)
```

The tank has the domain `real`, and its amount decays from 10 to 3.68 after two time units, which is `10 exp(-1)`.

## An incompatible element

A switch that is on or off holds a state of the domain `boolean`. A rate does not change a boolean, so `simulate`
raises a `RoleElementError` that names the domain.

```python
from typing import Annotated

from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.systems import CompositeSystem
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
    simulate(CompositeSystem(elements=[switch], interactions=[flip]), until=1)
except RoleElementError as error:
    message = str(error)
message
assert "domain 'boolean'" in message
```

The message says that the role `target` of `flip` accepts the domains real, integer, complex, and vector, while the
state `on` of `switch` has the domain `boolean`. Give the state a real domain, or bind it to a role that accepts
booleans.
"""
    facts = _facts(_family_classes(elements), 'elements')
    header = ['Element', 'States', 'Domains', 'Law kinds']
    return _page('Elements', intro, header, rows, f'{facts}\n\n{examples.strip()}')


def _role_cell(name: str, role: RoleSpec) -> str:
    """Return one role of an interaction with the domains it accepts."""
    if role.noise:
        return f'`{name}`: a `WienerProcessSpec`'
    return f'`{name}`: {_values(role.domains, empty="any domain")}'


def _interactions_page() -> str:
    """Return the page of interactions.

    It shows the domains every role accepts and the law kinds of every class.
    """
    rows = []
    for interaction in _public_classes(interactions, BaseInteraction):
        roles = [(name, spec) for name, spec in interaction._get_specs().items() if isinstance(spec, RoleSpec)]
        cells = '<br>'.join(_role_cell(name, role) for name, role in roles) or 'from a subclass'
        tags = BaseComponent._get_law_tags(interaction._get_declared_laws())
        rows.append([f'`{interaction.__name__}`', cells, _values(tags.laws, empty='from a subclass')])
    intro = """
An interaction changes the states of the elements bound to its roles. A role is a named slot, such as `target`, and it
lists the domains of the states it accepts. Each interaction also lists the kinds of the laws it can return, and its
arguments choose among them. The law kinds fix the time the interaction runs in, so the components of one system must
agree on it. `get_tags()` of an interaction returns the law kinds for the arguments you passed. The table lists the
roles of each interaction class of the core, with the domains they accept, and its law kinds.
"""
    examples = """
## A compatible interaction

A `RateInteraction` returns a `rate` law of continuous time, and its `target` accepts the real amount of a declared
element.

```python
from skcomplex.core.components.elements import DeclaredElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec

tank = DeclaredElement('tank', variables={'x': DeclaredVariableSpec(10.0, StateSpec())})
decay = RateInteraction('decay', target='tank', state='x', rate=lambda s: -0.5 * s.target.x)
sorted(decay.get_tags().laws), sorted(decay.get_tags().time)
trajectory = simulate(CompositeSystem(elements=[tank], interactions=[decay]), until=2)
round(trajectory.get_values('tank.x')[-1], 2)
```

The decay has the law kind `rate` and the time `continuous`, and the tank falls from 10 to 3.68 after two time units.

## An incompatible interaction

A `MapInteraction` returns an `update` law of discrete time. It cannot join the rate in one system unless the system
has a time that joins the two, so `simulate` raises a `TagError` that names the tag the components disagree on.

```python
from skcomplex.core.components.elements import DeclaredElement
from skcomplex.core.components.interactions import MapInteraction, RateInteraction
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import TagError
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec

tank = DeclaredElement('tank', variables={'x': DeclaredVariableSpec(10.0, StateSpec())})
decay = RateInteraction('decay', target='tank', state='x', rate=lambda s: -0.5 * s.target.x)
halve = MapInteraction('halve', target='tank', state='x', function=lambda s: s.target.x / 2)
sorted(halve.get_tags().laws), sorted(halve.get_tags().time)
try:
    simulate(CompositeSystem(elements=[tank], interactions=[decay, halve]), until=1)
except TagError as error:
    message = str(error)
message
assert "tag 'time'" in message
```

The map has the law kind `update` and the time `discrete`. The message says that `decay` and `halve` disagree on the
tag `time`, and asks for a time that joins them or for components of one time.
"""
    facts = _facts(_family_classes(interactions), 'interactions')
    header = ['Interaction', 'Roles and accepted domains', 'Law kinds']
    return _page('Interactions', intro, header, rows, f'{facts}\n\n{examples.strip()}')


def _laws_page() -> str:
    """Return the page of laws.

    It shows the kind, time, space, and randomness every law class declares.
    """
    rows = []
    for law in _public_classes(laws, BaseLaw):
        tags = BaseComponent._get_law_tags([law])
        spaces = {tag for space in law.get_requirements().space for tag in getattr(space, 'space_tags', ())}
        cells = [_values(tags.laws), _values(tags.time, TIMES), _values(spaces, SPACES), f'`{tags.randomness}`']
        rows.append([f'`{law.__name__}`', *cells])
    intro = """
A law is the mathematical object an interaction returns, such as a rate, a noise, a jump, or an update. Its kind is a
tag, and so are the times and spaces it runs in and its randomness, which says whether it draws random numbers. The
law kinds of a system are those of its interactions, and a solver or a backend runs a system only when it supports
every one of them. The table lists the kind, time, space, and randomness that each law class declares.
"""
    examples = """
## A compatible law

A `JumpInteraction` adds one unit to a queue at random times, at a rate of 2 per time unit. It returns a `jump` law,
which `GillespieSolver` supports.

```python
from skcomplex.core.components.elements import DeclaredElement
from skcomplex.core.components.interactions import JumpInteraction
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import GillespieSolver
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec

queue = DeclaredElement('queue', variables={'x': DeclaredVariableSpec(0.0, StateSpec())})
arrive = JumpInteraction('arrive', target='queue', changes={'x': 1}, rate=lambda s: 2.0)
sorted(arrive.get_tags().laws)
system = CompositeSystem(elements=[queue], interactions=[arrive])
trajectory = simulate(system, until=5, solver=GillespieSolver(), random_state=0)
trajectory.get_values('queue.x')[-1]
```

The law kind is `jump`, and with the seed 0 the queue holds 11 after five time units, near the mean of 10.

## An incompatible law

`EulerSolver` integrates rates and does not draw jumps, so `simulate` raises a `TagError` that names the law kind.

```python
from skcomplex.core.components.elements import DeclaredElement
from skcomplex.core.components.interactions import JumpInteraction
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import EulerSolver
from skcomplex.core.errors import TagError
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec

queue = DeclaredElement('queue', variables={'x': DeclaredVariableSpec(0.0, StateSpec())})
arrive = JumpInteraction('arrive', target='queue', changes={'x': 1}, rate=lambda s: 2.0)
try:
    simulate(CompositeSystem(elements=[queue], interactions=[arrive]), until=5, solver=EulerSolver())
except TagError as error:
    message = str(error)
message
assert 'laws jump' in message
```

The message says that `EulerSolver` does not support the law kind `jump`, and asks for a solver that supports it or
for none, which lets the tags choose the default.
"""
    facts = _facts(_family_classes(laws), 'laws')
    return _page('Laws', intro, ['Law', 'Kind', 'Time', 'Space', 'Randomness'], rows, f'{facts}\n\n{examples.strip()}')


def _spaces_page() -> str:
    """Return the page of spaces, neighbourhoods, and times.

    It shows the tags of every space and time and the fact sheet of every class.
    """
    rows = []
    for item in _family_classes(spaces):
        if issubclass(item, BaseTime):
            kind, tags = 'time', _values({item.time_tag} - {''}, empty='from a subclass')
        elif issubclass(item, BaseSpace):
            kind, tags = 'space', _values(item.space_tags, empty='from a subclass')
        else:
            kind, tags = 'neighbourhood', 'none'
        rows.append([f'`{item.__name__}`', kind, tags])
    intro = """
A space places the members of a group, such as the sites of a lattice or the nodes of a network, and its tag names the
kind of structure. A neighbourhood says which members a law reads around one member. A time says how the system
advances: in continuous time, in discrete steps, or not at all. A solver runs a system only when it supports the tags
of its space and its time. The table lists the tags that each class declares.
"""
    examples = """
## The tags of a space and a time

`get_tags` of a space returns its space tag, and that of a time returns its time tag. A solver compares them with the
tags it supports before a run.

```python
from skcomplex.core.components.spaces import ContinuousTime, LatticeSpace

grid = LatticeSpace((4, 4))
sorted(grid.get_tags().space), sorted(ContinuousTime().get_tags().time)
```

A lattice of 4 by 4 sites has the space tag `lattice`, and `ContinuousTime` has the time tag `continuous`.
"""
    facts = _facts(_family_classes(spaces), 'spaces')
    return _page(
        'Spaces, neighbourhoods, and times',
        intro,
        ['Class', 'Kind', 'Tags'],
        rows,
        f'{facts}\n\n{examples.strip()}',
    )


def _solvers_page() -> str:
    """Return the page of solvers.

    It shows the form and the supported tags of every solver.
    """
    found = [solver for solver in _public_classes(solvers, BaseSolver) if _declares_support(solver)]
    ordered = sorted(found, key=lambda solver: (str(solver._get_form()).lower(), solver.__name__))
    rows = [
        [row[0], str(solver._get_form()), *row[1:]]
        for solver, row in zip(ordered, _supported_rows(ordered), strict=True)
    ]
    intro = """
A solver steps a system through time, or solves a static system at once. `get_supported_tags()` of a solver class
returns the times, spaces, and law kinds it supports. `simulate` compares them with the tags of the system before it
compiles the system, and raises a `TagError` when the solver lacks one. Leave the solver out and the tags choose the
default solver. The table lists the supported tags of every solver, grouped by the mathematical form it solves.
"""
    examples = """
## A compatible solver

A rate with a noise term returns a `rate` and a `noise` law, which together form a stochastic differential equation.
`EulerMaruyamaSolver` supports both.

```python
from skcomplex.core.components.elements import DeclaredElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import EulerMaruyamaSolver
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec

tank = DeclaredElement('tank', variables={'x': DeclaredVariableSpec(10.0, StateSpec())})
decay = RateInteraction('decay', target='tank', state='x', rate=lambda s: -0.5 * s.target.x, noise=0.1)
system = CompositeSystem(elements=[tank], interactions=[decay])
sorted(system.get_tags().laws), sorted(EulerMaruyamaSolver.get_supported_tags().laws)
trajectory = simulate(system, until=1, solver=EulerMaruyamaSolver(), random_state=0)
round(trajectory.get_values('tank.x')[-1], 2)
```

The system has the law kinds `noise` and `rate`, and the solver supports `equation`, `jump`, `noise`, `rate`, and
`update`. With the seed 0 the tank holds 6.15 after one time unit, near the 6.07 of the decay without noise.

## An incompatible solver

`RK4Solver` integrates deterministic equations and does not support the `noise` law kind, so `simulate` raises a
`TagError`.

```python
from skcomplex.core.components.elements import DeclaredElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.core.errors import TagError
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec

tank = DeclaredElement('tank', variables={'x': DeclaredVariableSpec(10.0, StateSpec())})
decay = RateInteraction('decay', target='tank', state='x', rate=lambda s: -0.5 * s.target.x, noise=0.1)
try:
    simulate(CompositeSystem(elements=[tank], interactions=[decay]), until=1, solver=RK4Solver())
except TagError as error:
    message = str(error)
message
assert 'laws noise' in message
```

The message says that `RK4Solver` does not support the law kind `noise`, and asks for a solver that supports it or for
none.
"""
    return _page('Solvers', intro, ['Solver', 'Form', 'Time', 'Space', 'Law kinds'], rows, examples)


def _backends_page() -> str:
    """Return the page of backends, with the supported tags of every backend."""
    found = [backend for backend in _public_classes(backends, BaseBackend) if _declares_support(backend)]
    rows = _supported_rows(found)
    intro = """
A backend turns the compiled laws of a system into numeric functions, with NumPy, Numba, or JAX.
`get_supported_tags()` of a backend class returns the times, spaces, and law kinds it can write functions for.
`simulate` raises a `TagError` for a backend that lacks a tag of the system, and uses the NumPy backend when you leave
it out. The table lists the supported tags of every backend.
"""
    examples = """
## A compatible backend

`NumbaBackend` supports every tag of a decay, so it runs the system.

```python
from skcomplex.core.components.elements import DeclaredElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.backends import NumbaBackend
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec

tank = DeclaredElement('tank', variables={'x': DeclaredVariableSpec(10.0, StateSpec())})
decay = RateInteraction('decay', target='tank', state='x', rate=lambda s: -0.5 * s.target.x)
trajectory = simulate(CompositeSystem(elements=[tank], interactions=[decay]), until=1, backend=NumbaBackend())
round(trajectory.get_values('tank.x')[-1], 2)
```

The tank holds 6.07 after one time unit, which is `10 exp(-0.5)`.

## An incompatible backend

`JaxBackend` writes deterministic functions only, so `simulate` rejects a noise term and names the law kind.

```python
from skcomplex.core.components.elements import DeclaredElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.backends import JaxBackend
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import TagError
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec

tank = DeclaredElement('tank', variables={'x': DeclaredVariableSpec(10.0, StateSpec())})
decay = RateInteraction('decay', target='tank', state='x', rate=lambda s: -0.5 * s.target.x, noise=0.1)
sorted(JaxBackend.get_supported_tags().laws)
try:
    simulate(CompositeSystem(elements=[tank], interactions=[decay]), until=1, backend=JaxBackend())
except TagError as error:
    message = str(error)
message
assert 'laws noise' in message
```

The law kinds of `JaxBackend` are `equation`, `rate`, and `update`. The message says that it does not support the law
kind `noise` of the system.
"""
    return _page('Backends', intro, ['Backend', 'Time', 'Space', 'Law kinds'], rows, examples)


def _summary(item: object) -> str:
    """Return the first line of the docstring of a function or a class."""
    return (inspect.getdoc(item) or '').split('\n', maxsplit=1)[0]


def _analysis_functions() -> list[Callable[..., object]]:
    """Return the analysis functions of the core, sorted by name.

    The binding of settings is left out.
    """
    named = [getattr(analysis, name) for name in analysis.__all__ if name != 'bind_analysis']
    return sorted((item for item in named if inspect.isfunction(item)), key=lambda item: item.__name__)


def _needs(tags: AnalysisTags) -> list[str]:
    """Return the data kind and the time, space, and randomness an analysis needs.

    They are `any` where it needs nothing.
    """
    randomness = 'any' if tags.randomness is None else f'`{tags.randomness}`'
    return [f'`{tags.data}`', _values(tags.time), _values(tags.space), randomness]


def _kind_cell(tags: AnalysisTags, by_kind: dict[str, AnalysisTags]) -> str:
    """Return the kinds of an analysis whose needs differ from the analysis.

    Each comes with what it needs.
    """
    names = ('data', 'time', 'space', 'randomness')
    default = _needs(tags)
    cells = []
    for kind, needed in sorted(by_kind.items()):
        differ = zip(names, _needs(needed), default, strict=True)
        cells.append(f'`{kind}`: ' + ', '.join(f'{name} {value}' for name, value, base in differ if value != base))
    return '<br>'.join(cells) or 'none'


def _analyses_page() -> str:
    """Return the page of analyses.

    It shows the data and the capability tags every analysis function needs.
    """
    rows = []
    for function in _analysis_functions():
        module = sys.modules[function.__module__]
        tags: AnalysisTags | None = getattr(module, 'TAGS', None)
        if tags is None:
            rows.append([f'`{function.__name__}`', 'any source', 'any', 'any', 'any', 'none'])
            continue
        by_kind: dict[str, AnalysisTags] = getattr(module, 'KIND_TAGS', {})
        rows.append([f'`{function.__name__}`', *_needs(tags), _kind_cell(tags, by_kind)])
    intro = """
An analysis is a function of a source, which is a system, a trajectory, or measured data, and it returns a frozen
result. Each analysis reads one kind of data, such as a series, a field, or a network, and may need a capability tag
of the system behind the source. Lyapunov exponents, for instance, need the randomness `deterministic`. The analysis
checks the tags of the system, or of the system behind a trajectory, before it does any work, and raises a `TagError`
that names the tag it lacks. Measured data carries no tags, so only its kind is checked. The table lists what every
analysis needs, and the kinds of an analysis whose needs differ from its own.
"""
    examples = """
## A compatible analysis

The logistic map at `r=4` is deterministic, so `lyapunov` measures its largest exponent, the mean rate at which nearby
runs separate. In theory it is `log 2`.

```python
from skcomplex.core.analysis import lyapunov
from skcomplex.systems.mathematics.dynamical_systems import LogisticMapSystem

system = LogisticMapSystem(r=4.0)
system.get_tags().randomness
round(float(lyapunov(system, until=20000).exponents[0]), 2)
```

The randomness is `deterministic`, and the estimate rounds to 0.69, which is `log 2`.

## An incompatible analysis

The stochastic SIR epidemic has random events, so nearby runs do not separate at a rate of their own. `lyapunov`
raises a `TagError` that names the randomness it needs.

```python
from skcomplex.core.analysis import lyapunov
from skcomplex.core.errors import TagError
from skcomplex.systems.biology.epidemiology import SIRSystem

system = SIRSystem(stochastic=True)
system.get_tags().randomness
try:
    lyapunov(system)
except TagError as error:
    message = str(error)
message
assert 'randomness deterministic' in message
```

The system has the randomness `stochastic`, and the message says that `lyapunov` of kind `spectrum` needs the
randomness deterministic. The deterministic SIR system, built without `stochastic=True`, passes the check.

A network measure needs a system on a network. A trajectory of the SIR epidemic carries the tags of its system, whose
space is `none`.

```python
from skcomplex.core.analysis import network_measures
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import TagError
from skcomplex.systems.biology.epidemiology import SIRSystem

trajectory = simulate(SIRSystem(), until=10)
try:
    network_measures(trajectory)
except TagError as error:
    message = str(error)
message
assert 'space network' in message
```

The message says that `network_measures` needs the space network, while the system has the space none. Simulate a
system on a network to measure its network.
"""
    header = ['Analysis', 'Data', 'Time', 'Space', 'Randomness', 'Kinds that differ']
    return _page('Analyses', intro, header, rows, examples)


def _experiments_page() -> str:
    """Return the page of experiments.

    It shows what every experiment function needs of a system.
    """
    named = [getattr(experiments, name) for name in experiments.__all__]
    found = [item for item in named if inspect.isfunction(item)]
    rows = [
        [f'`{item.__name__}`', _summary(item), EXPERIMENT_NEEDS.get(item.__name__, 'any system')]
        for item in sorted(found, key=lambda item: item.__name__.lower())
    ]
    intro = """
An experiment runs a system many times through `simulate`, over values of its parameters, over random streams, over
scenarios, or over initial states. Experiments are functions of a system that return frozen results. `calibrate` fits
the model parameters of a system to observed series and returns the fitted system in its result. An experiment that
repeats a system needs randomness to repeat, and the analyses an experiment applies to its runs check their own tags,
as the [analyses](analyses.md) page lists. The table lists what every experiment needs of a system.
"""
    examples = """
## A compatible experiment

The stochastic SIR epidemic draws its events from random streams, so `ensemble` runs it once per stream.

```python
from skcomplex.core.experiments import ensemble
from skcomplex.systems.biology.epidemiology import SIRSystem

result = ensemble(SIRSystem(stochastic=True), replicates=3, until=10, random_state=0)
result.get_column('seed').tolist()
```

The result holds one run for each of the seeds 0, 1, and 2.

## An incompatible experiment

The deterministic SIR epidemic repeats one run, so `ensemble` raises a `TagError` that names the randomness it needs.

```python
from skcomplex.core.errors import TagError
from skcomplex.core.experiments import ensemble
from skcomplex.systems.biology.epidemiology import SIRSystem

try:
    ensemble(SIRSystem(), replicates=3, until=10)
except TagError as error:
    message = str(error)
message
assert 'randomness stochastic' in message
```

The message says that `ensemble` needs the randomness stochastic, and advises one run with `simulate`, or a component or
a solver that draws.

## A compatible calibration

`calibrate` fits the model parameters of a logistic growth to a series computed from its closed form with `r = 0.5`
and `K = 100`.

```python
import numpy as np

from skcomplex.core.experiments import calibrate
from skcomplex.systems.biology.ecology import LogisticGrowthSystem

times = np.arange(0, 21.0)
values = 100 / (1 + 9 * np.exp(-0.5 * times))
bounds = {'r': (0.1, 2.0), 'K': (50, 200)}
fitted = calibrate(LogisticGrowthSystem(), {'population.x': (times, values)}, bounds).system
round(fitted.r, 2), round(fitted.K)
```

The fit recovers `r = 0.5` and `K = 100`.

## An incompatible calibration

Calibration fits model parameters, which `set_params` takes. An initial state is a dotted path, not a model
parameter, so `calibrate` raises a `ValidationError`.

```python
import numpy as np

from skcomplex.core.errors import ValidationError
from skcomplex.core.experiments import calibrate
from skcomplex.systems.biology.ecology import LogisticGrowthSystem

times = np.arange(0, 21.0)
values = 100 / (1 + 9 * np.exp(-0.5 * times))
try:
    calibrate(LogisticGrowthSystem(), {'population.x': (times, values)}, {'population.x': (1, 20)})
except ValidationError as error:
    message = str(error)
message
assert 'population.x' in message
```

The message says that calibration fits model parameters, which `set_params` takes, and not the dotted path
`population.x`.
"""
    return _page('Experiments', intro, ['Experiment', 'What it does', 'What it needs'], rows, examples)


def _core_systems() -> list[type[BaseSystem]]:
    """Return the system classes of the core that can be built.

    They are the base and the composite you build from components.
    """
    found = _public_classes(systems, BaseSystem)
    return sorted(found, key=lambda system: (system is not CompositeSystem, system.__name__))


def _general_systems() -> list[str]:
    """Return the Markdown lines of the table of the general systems and the catalog systems on each.

    A catalog system is listed under the nearest general system it subclasses.
    """
    general = [*(getattr(dynamics, name) for name in dynamics.__all__), ReactionNetworkSystem]
    rows = []
    for system in general:
        users = [
            name
            for name, item in all_systems()
            if next((base for base in item.__mro__[1:] if base in general), None) is system
        ]
        rows.append(
            [
                f'`{system.__name__}`',
                f'`from {system.__module__.rsplit(".", 1)[0]} import {system.__name__}`',
                ', '.join(f'`{name}`' for name in users) or 'none yet',
            ],
        )
    intro = (
        'General systems are recipes of `skcomplex.systems.general` that systems of several fields share. The table '
        'lists each with the catalog systems that subclass it.'
    )
    return ['## General systems', '', intro, '', *_table(['General system', 'Import', 'Catalog systems'], rows), '']


def _systems_page() -> str:
    """Return the page of systems.

    It shows the system classes of the core and a link to the catalog of ready-made ones.
    """
    rows = [
        [f'`{system.__name__}`', f'`from {systems.__name__} import {system.__name__}`', _summary(system)]
        for system in _core_systems()
    ]
    intro = """
A system is a component that holds elements, interactions, nested systems, a space, and a time. `BaseSystem` is the base
of every system. A subclass takes the parameters of its model in `__init__`, builds its components from them in
`_build_system`, and derives its capability tags from its components, which `get_tags()` returns. `CompositeSystem`
takes its components as data, so it needs no subclass. A general system that several fields share, such as
`ReactionNetworkSystem` of `skcomplex.systems.general.reactions`, is a recipe built from core components. The ready-made
systems of `skcomplex.systems` subclass these, one class per model. The [catalog](../systems/catalog.md) lists them by
field and by topic, with their import paths, model parameters, and capability tags, and `all_systems` of
`skcomplex.systems` finds them by field, topic, or tag. The table lists the system classes of the core.
"""
    examples = '\n'.join(_general_systems()) + """
## A compatible system

A composite of random arrivals has jump laws, which `GillespieSolver` draws one event at a time.

```python
from skcomplex.core.components.elements import DeclaredElement
from skcomplex.core.components.interactions import JumpInteraction
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import GillespieSolver
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec

queue = DeclaredElement('queue', variables={'x': DeclaredVariableSpec(0.0, StateSpec())})
arrive = JumpInteraction('arrive', target='queue', changes={'x': 1}, rate=lambda s: 2.0)
system = CompositeSystem(elements=[queue], interactions=[arrive])
sorted(system.get_tags().laws), system.get_tags().randomness
trajectory = simulate(system, until=5, solver=GillespieSolver(), random_state=0)
trajectory.get_values('queue.x')[-1]
```

The system has the law kind `jump` and the randomness `stochastic`, and with the seed 0 the queue holds 11 after five
time units.

## An incompatible system

A composite of a decay has rate equations, which `GillespieSolver` does not draw, so `simulate` raises a `TagError`
that names the law kind.

```python
from skcomplex.core.components.elements import DeclaredElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import GillespieSolver
from skcomplex.core.errors import TagError
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec

tank = DeclaredElement('tank', variables={'x': DeclaredVariableSpec(10.0, StateSpec())})
decay = RateInteraction('decay', target='tank', state='x', rate=lambda s: -0.5 * s.target.x)
try:
    simulate(CompositeSystem(elements=[tank], interactions=[decay]), until=5, solver=GillespieSolver())
except TagError as error:
    message = str(error)
message
assert 'laws rate' in message
```

The message says that `GillespieSolver` does not support the law kind `rate`. Leave the solver out and the tags choose
one that integrates rates.

## Find ready-made systems by tag

`all_systems` takes the values of capability tags, so it finds the ready-made systems that a solver or an analysis
accepts.

```python
from skcomplex.systems import all_systems

stochastic = all_systems(field='biology/epidemiology', tags={'randomness': 'stochastic'})
[name for name, _ in stochastic][:3]
```

The list holds the first three stochastic systems of epidemiology, in alphabetical order.
"""
    facts = _facts(_family_classes(systems), 'systems')
    return _page('Systems', intro, ['System', 'Import', 'Description'], rows, f'{facts}\n\n{examples.strip()}')


def main() -> None:
    """Write the page of every kind of component into the documentation."""
    pages = {
        'elements': _elements_page,
        'interactions': _interactions_page,
        'laws': _laws_page,
        'spaces': _spaces_page,
        'solvers': _solvers_page,
        'backends': _backends_page,
        'systems': _systems_page,
        'analyses': _analyses_page,
        'experiments': _experiments_page,
    }
    for name, page in pages.items():
        with mkdocs_gen_files.open(f'{FAMILIES}/{name}.md', 'w') as file:
            file.write(page())


main()
