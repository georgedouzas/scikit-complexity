"""Generate one page per kind of part.

Each page has a table of the capability tags of its classes, and executed examples.
"""

import inspect
import sys
from collections.abc import Callable, Iterable
from types import ModuleType

import mkdocs_gen_files

from skcomplex.core import analysis, experiments
from skcomplex.core.components import elements, interactions, laws, systems
from skcomplex.core.components.base import BasePart
from skcomplex.core.components.elements import BaseElement
from skcomplex.core.components.interactions import BaseInteraction
from skcomplex.core.components.laws import BaseLaw
from skcomplex.core.components.systems import BaseSystem, CompositeSystem
from skcomplex.core.engine import backends, solvers
from skcomplex.core.engine.backends import BaseBackend
from skcomplex.core.engine.solvers import BaseSolver
from skcomplex.core.specs import LAWS, SPACES, TIMES, AnalysisTags, RoleSpec, StateSpec
from skcomplex.systems import all_systems
from skcomplex.systems.general import dynamics
from skcomplex.systems.general.reactions import ReactionNetworkSystem

PARTS = 'parts'
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
    """Return a page of a kind of part.

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


def _elements_page() -> str:
    """Return the page of elements.

    It shows the states, domains, and law kinds every element class declares.
    """
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

A declared element takes its states as data. Here it holds a real amount, and the `target` of a `RateInteraction`
accepts real states.

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
round(trajectory.values('tank.x')[-1], 2)
```

## An incompatible element

A switch that is on or off holds a boolean state. A rate does not change a boolean, so the system rejects the binding
and names the domain.

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
"""
    return _page('Elements', intro, ['Element', 'States', 'Domains', 'Law kinds'], rows, examples)


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

A rate returns a `rate` law of continuous time, and its `target` accepts the real amount of a declared element.

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
round(trajectory.values('tank.x')[-1], 2)
```

## An incompatible interaction

A map returns an `update` law of discrete time. It cannot join the rate in one system without a time that joins them,
so the system names the tag the two parts disagree on.

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
"""
    return _page('Interactions', intro, ['Interaction', 'Roles and accepted domains', 'Law kinds'], rows, examples)


def _laws_page() -> str:
    """Return the page of laws.

    It shows the kind, time, space, and randomness every law class declares.
    """
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
trajectory.values('queue.x')[-1]
```

## An incompatible law

`EulerSolver` integrates rates and does not draw jumps, so it rejects the system and names the law kind.

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
"""
    return _page('Laws', intro, ['Law', 'Kind', 'Time', 'Space', 'Randomness'], rows, examples)


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
A solver steps a system through time, or solves it at once for a static system. `get_supported_tags()` of a solver
class returns the times, spaces, and law kinds it supports. `simulate` compares them with the tags of the system before
compiling, and rejects a solver that lacks one. Leave the solver out and the tags choose the default solver. The table
lists the supported tags of every solver, grouped by the mathematical form it solves.
"""
    examples = """
## A compatible solver

A rate with a noise term returns a `rate` and a `noise` law, and `EulerMaruyamaSolver` supports both.

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
round(trajectory.values('tank.x')[-1], 2)
```

## An incompatible solver

`RK4Solver` integrates deterministic equations and does not support the `noise` law kind.

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
from skcomplex.core.components.elements import DeclaredElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.backends import NumbaBackend
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec

tank = DeclaredElement('tank', variables={'x': DeclaredVariableSpec(10.0, StateSpec())})
decay = RateInteraction('decay', target='tank', state='x', rate=lambda s: -0.5 * s.target.x)
trajectory = simulate(CompositeSystem(elements=[tank], interactions=[decay]), until=1, backend=NumbaBackend())
round(trajectory.values('tank.x')[-1], 2)
```

## An incompatible backend

`JaxBackend` writes deterministic functions only, so it rejects a noise term and names the law kind.

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
An analysis is a function of a source: a system, a trajectory, or measured data. It returns a frozen result. Each
analysis reads one kind of data, such as a series, a field, or a network, and may need a capability tag of the system
behind the source, such as a deterministic randomness for Lyapunov exponents. The analysis checks the tags of the
system, or of the system behind a trajectory, before it does any work, and raises a `TagError` that names the tag it
lacks. Measured data carries no tags, so only its kind is checked. The table lists what every analysis needs, and the
kinds of an analysis whose needs differ from its own.
"""
    examples = """
## A compatible analysis

The logistic map at `r=4` is deterministic, so `lyapunov` measures its largest exponent, close to `log 2`.

```python
from skcomplex.core.analysis import lyapunov
from skcomplex.systems.mathematics.dynamical_systems import LogisticMapSystem

system = LogisticMapSystem(r=4.0)
system.get_tags().randomness
round(float(lyapunov(system, until=20000).exponents[0]), 2)
```

## An incompatible analysis

The stochastic SIR epidemic has random events, so nearby runs do not separate at a rate of their own. `lyapunov`
rejects it and names the randomness it needs.

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
An experiment runs a system many times through `simulate`: over values of its parameters, over random streams, over
scenarios, or over initial states. Experiments are functions of a system that return frozen results, and `calibrate`
fits the model parameters of a system to observed series and returns the fitted system in its result. An experiment that
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
result.column('seed').tolist()
```

## An incompatible experiment

The deterministic SIR epidemic repeats one run, so `ensemble` rejects it and names the randomness it needs.

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

## A compatible calibration

`calibrate` fits the model parameters of a logistic growth to a series it generates.

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

## An incompatible calibration

Calibration fits model parameters, which `set_params` takes. An initial state is a dotted path, not a model
parameter, so `calibrate` rejects it.

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
"""
    return _page('Experiments', intro, ['Experiment', 'What it does', 'What it needs'], rows, examples)


def _core_systems() -> list[type[BaseSystem]]:
    """Return the system classes of the core that can be built.

    They are the base and the composite you build from parts.
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
A system is a part that holds elements, interactions, nested systems, a space, and a time. `BaseSystem` is the base
of every system: a subclass takes the parameters of its model in `__init__`, builds its parts from them in
`_build_system`, and derives its capability tags from its parts, which `get_tags()` returns. `CompositeSystem` is the
composite you build from parts. A general system that several fields share, such as `ReactionNetworkSystem` of
`skcomplex.systems.general.reactions`, is a recipe built from core parts. The ready-made systems of
`skcomplex.systems` are subclasses of these, one class per model. The [catalog](../systems/catalog.md) lists them by
field and by topic, with their import paths, their model parameters, and their capability tags, and `all_systems` of
`skcomplex.systems` lists them by field, topic, or tag. The table lists the system classes of the core.
"""
    examples = '\n'.join(_general_systems()) + """
## A compatible system

A composite of random arrivals has jump laws, which `GillespieSolver` draws.

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
trajectory.values('queue.x')[-1]
```

## An incompatible system

A composite of a decay has rate equations, which `GillespieSolver` does not draw, so `simulate` names the law kind.

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

## Find ready-made systems by tag

`all_systems` takes the values of capability tags, so it finds the ready-made systems that a solver or an analysis
accepts.

```python
from skcomplex.systems import all_systems

stochastic = all_systems(field='biology/epidemiology', tags={'randomness': 'stochastic'})
[name for name, _ in stochastic][:3]
```
"""
    return _page('Systems', intro, ['System', 'Import', 'Description'], rows, examples)


def main() -> None:
    """Write the page of every kind of part into the documentation."""
    pages = {
        'elements': _elements_page,
        'interactions': _interactions_page,
        'laws': _laws_page,
        'solvers': _solvers_page,
        'backends': _backends_page,
        'systems': _systems_page,
        'analyses': _analyses_page,
        'experiments': _experiments_page,
    }
    for name, page in pages.items():
        with mkdocs_gen_files.open(f'{PARTS}/{name}.md', 'w') as file:
            file.write(page())


main()
