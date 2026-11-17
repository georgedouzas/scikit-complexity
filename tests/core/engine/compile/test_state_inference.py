"""Test the `_state_inference` module of `skcomplex.core.engine.compile`."""

from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
import pytest
import skfem
from sage.all__sagemath_symbolics import abs_symbolic, oo, sin, sqrt
from sage.sets.real_set import RealSet
from scipy.linalg import expm
from scipy.special import ive

from skcomplex.core.analysis import alignment, displacement
from skcomplex.core.components.base import BaseComponent
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import (
    BoundaryLaw,
    ConstraintLaw,
    NoiseLaw,
    RateLaw,
    TopologyRequestSpec,
    UpdateLaw,
)
from skcomplex.core.components.laws.utils import laplacian, neighbors
from skcomplex.core.components.spaces import BaseSpace, CellComplexSpace, ContinuousSpace, LatticeSpace, NetworkSpace
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import Role
from skcomplex.core.systems import System


class Particle(Element):
    """Particle with a position and a mass, declared by nothing but its arguments."""

    def __init__(self, label: str, x: float = 0.0, m: float = 1.0) -> None:
        super().__init__(label)
        self.x = x
        self.m = m


class Herd(Element):
    """Herd whose size is nonnegative and whose status is one of two categories."""

    def __init__(
        self,
        label: str,
        n: Annotated[float, RealSet.closed_open(0, oo)] = 40.0,
        status: Annotated[str, {'S', 'I'}] = 'S',
    ) -> None:
        super().__init__(label)
        self.n = n
        self.status = status


class Economy(Element):
    """Economy with an income."""

    def __init__(self, label: str, y: float = 1.0) -> None:
        super().__init__(label)
        self.y = y


class Gravity(Interaction):
    """Interaction that gives a constant acceleration to its target."""

    def __init__(self, label: str, *, target: Role = 'ball', g: float = 9.81) -> None:
        super().__init__(label)
        self.target = target
        self.g = g

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the acceleration as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x, t = context.target.x, context.t
            self.laws_: list[BaseComponent] = [RateLaw(x.diff(t, 2), -context.g)]


class Jerk(Interaction):
    """Interaction of Sprott's jerk equation."""

    def __init__(self, label: str, *, target: Role = 'ball', A: float = 0.6) -> None:
        super().__init__(label)
        self.target = target
        self.A = A

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the third derivative as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x, t = context.target.x, context.t
            jerk = -context.A * x.diff(t, 2) - x.diff(t) + abs_symbolic(x) - 1
            self.laws_ = [RateLaw(x.diff(t, 3), jerk)]


class Mixed(Interaction):
    """Interaction that gives two orders of one state."""

    def __init__(self, label: str, *, target: Role = 'ball') -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set laws of two orders."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x, t = context.target.x, context.t
            self.laws_ = [RateLaw(x, 1.0), RateLaw(x.diff(t, 2), 1.0)]


class Accelerator(Interaction):
    """Interaction of Samuelson's multiplier and accelerator."""

    def __init__(self, label: str, *, target: Role = 'economy', a: float = 1.2, b: float = -0.5) -> None:
        super().__init__(label)
        self.target = target
        self.a = a
        self.b = b

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the next income as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            y, t = context.target.y, context.t
            self.laws_ = [UpdateLaw(y, 1 + context.a * y(t=t - 1) + context.b * y(t=t - 2))]


class Feedback(Interaction):
    """Interaction of a delayed negative feedback."""

    def __init__(self, label: str, *, target: Role = 'ball', tau: float = 1.0) -> None:
        super().__init__(label)
        self.target = target
        self.tau = tau

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the delayed decay as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x, t = context.target.x, context.t
            self.laws_ = [RateLaw(x, -x(t=t - context.tau))]


def test_get_declared_makes_untouched_values_parameters():
    """A particle no law acts on has no states, and its values are parameters."""
    system = System(elements=[Particle('ball', x=2.0)]).prepare()
    assert system.states_ == ()
    assert system.parameters_ == ('ball.m', 'ball.x')


def test_get_declared_reads_the_order_from_the_laws():
    """A second derivative in a law makes the position a state of order two, named with primes."""
    system = System(elements=[Particle('ball', x=10.0)], interactions=[Gravity('gravity')]).prepare()
    assert system.states_ == ('ball.x', "ball.x'")
    assert system.parameters_ == ('ball.m', 'gravity.g')
    assert system.equations_ == {'ball.x': "ball.x'", "ball.x'": '-gravity.g'}
    trajectory = system.simulate(until=1.4, dt=0.01)
    exact = 10.0 - 9.81 * trajectory.times**2 / 2
    assert trajectory.get_values('ball.x') == pytest.approx(exact, abs=1e-9)


def test_get_declared_starts_from_a_given_derivative():
    """The initial value of a derivative is given by its name with a prime."""
    system = System(elements=[Particle('ball')], interactions=[Gravity('gravity')]).prepare()
    trajectory = system.simulate(until=1.0, dt=0.5, initial_state={"ball.x'": 1.5})
    assert trajectory.get_values("ball.x'")[0] == pytest.approx(1.5)


def test_get_declared_reads_a_third_order():
    """Sprott's jerk equation compiles to three first-order states, and its trajectory stays bounded."""
    system = System(elements=[Particle('ball', x=0.02)], interactions=[Jerk('jerk')]).prepare()
    assert system.states_ == ('ball.x', "ball.x'", "ball.x''")
    assert np.abs(system.simulate(until=50.0, dt=1.0).get_values('ball.x')).max() < 10


def test_get_declared_reads_lags():
    """Lags in a law keep the past values of a state, named with indices."""
    system = System(elements=[Economy('economy')], interactions=[Accelerator('samuelson')]).prepare()
    assert system.states_ == ('economy.y', 'economy.y[-1]', 'economy.y[-2]')
    assert system.equations_['economy.y[-1]'] == 'economy.y'


def test_get_declared_starts_a_derivative_from_a_value():
    """Test that a value named with a prime starts the derivative, as a ball thrown upwards at 3 m/s."""
    ball = Element('ball', values={'x': 10.0, "x'": 3.0})
    trajectory = System(elements=[ball], interactions=[Gravity('gravity')]).prepare().simulate(until=1.0, dt=0.5)
    assert trajectory.get_values("ball.x'")[0] == pytest.approx(3.0)
    assert trajectory.get_values('ball.x')[-1] == pytest.approx(10.0 + 3.0 - 9.81 / 2)


def test_get_declared_starts_past_values_from_values():
    """Test that values named with past indices start the lags, and a missing one is the value itself."""
    economy = Element('economy', values={'y': 1.0, 'y[-2]': 0.5})
    system = System(elements=[economy], interactions=[Accelerator('samuelson')]).prepare()
    values = system.simulate(until=1, dt=1).get_values('economy.y')
    assert values.tolist() == pytest.approx([1.0, 1 + 1.2 * 1.0 - 0.5 * 0.5])


@pytest.mark.parametrize(
    ('values', 'message'),
    [({'x': 1.0, "z'": 1.0}, "no single value named 'z'"), ({'x': 1.0, "x''": 1.0}, "of 'x' at index 2")],
)
def test_get_declared_rejects_unread_starting_values(values, message):
    """Test that a starting value of an unknown value, or of a derivative no law reads, raises an error."""
    with pytest.raises(ValidationError, match=message):
        System(elements=[Element('ball', values=values)], interactions=[Gravity('gravity')]).prepare()


def test_get_declared_rejects_two_orders():
    """Laws that give two orders of one state raise an error."""
    with pytest.raises(ValidationError, match='order'):
        System(elements=[Particle('ball')], interactions=[Mixed('mixed')]).prepare()


@pytest.mark.parametrize(('values', 'message'), [({'n': -1.0}, r'herd\.n = -1\.0'), ({'status': 'X'}, 'herd.status')])
def test_get_declared_checks_allowed_values(values, message):
    """A value outside the values its annotation allows raises an error that names it."""
    with pytest.raises(ValidationError, match=message):
        System(elements=[Herd('herd', **values)]).prepare()


def test_get_declared_reads_a_delay():
    """A state read a delay before the current time follows the delay equation."""
    system = System(elements=[Particle('ball', x=1.0)], interactions=[Feedback('feedback')]).prepare()
    assert system.equations_ == {'ball.x': '-ball.x(t - feedback.tau)'}
    values = system.simulate(until=2.0, dt=1.0).get_values('ball.x')
    assert values == pytest.approx([1.0, 0.0, -0.5], abs=1e-6)


def test_get_declared_moves_many_members():
    """Test that 1000 members, each with its own height, fall under gravity with one shared mass."""
    heights = np.linspace(1.0, 10.0, 1000)
    gas = Element('gas', values={'x': heights, 'm': 1.0}, size=1000)
    system = System(elements=[gas], interactions=[Gravity('gravity', target='gas')]).prepare()
    assert system.states_ == ('gas.x[i]', "gas.x'[i]")
    last = system.simulate(until=1.0, dt=0.01).get_values('gas.x')[-1]
    assert last == pytest.approx(heights - 9.81 / 2, rel=1e-6)


def test_get_declared_reads_values_of_an_element_without_a_subclass():
    """Test that values given by name act like the arguments of a subclass."""
    ball = Element('ball', values={'x': 10.0, 'm': 1.0})
    system = System(elements=[ball], interactions=[Gravity('gravity')]).prepare()
    assert system.states_ == ('ball.x', "ball.x'")
    assert system.parameters_ == ('ball.m', 'gravity.g')
    assert system.simulate(until=1.0, dt=0.01).get_values('ball.x')[-1] == pytest.approx(10.0 - 9.81 / 2)


def test_get_declared_rejects_a_wrong_number_of_values():
    """Test that an array whose length differs from the number of members raises an error naming both."""
    gas = Element('gas', values={'x': [1.0, 2.0, 3.0]}, size=4)
    with pytest.raises(ValidationError, match=r"'gas' has 4 members but its value 'x' has 3 entries"):
        System(elements=[gas], interactions=[Gravity('gravity', target='gas')]).prepare()


def test_get_declared_places_one_member_per_site():
    """Test that an element on a lattice has one member on each site, each with its own height."""
    heights = np.array([1.0, 2.0, 3.0, 4.0])
    cells = Element('cells', values={'x': heights}, space=LatticeSpace((4,)))
    system = System(elements=[cells], interactions=[Gravity('gravity', target='cells')]).prepare()
    last = system.simulate(until=1.0, dt=0.01).get_values('cells.x')[-1]
    assert np.ravel(last) == pytest.approx(heights - 9.81 / 2, rel=1e-6)


@pytest.mark.parametrize('name', ['size', 'label', 'not a name'])
def test_get_declared_rejects_a_reserved_name(name):
    """Test that a value named like a setting of the element, or not a Python name, raises an error."""
    ball = Element('ball', values={name: 1.0})
    with pytest.raises(ValidationError, match=f"'ball' cannot take a value named '{name}'"):
        System(elements=[ball]).prepare()


class Diffusion(Interaction):
    """Interaction that spreads a value over the sites of its target."""

    def __init__(self, label: str, *, target: Role = 'plate', D: float = 1.0) -> None:
        super().__init__(label)
        self.target = target
        self.D = D

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the heat equation as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            u = context.target.u
            self.laws_: list[BaseComponent] = [RateLaw(u.diff(context.t), context.D * laplacian(u))]


class Consensus(Interaction):
    """Interaction that moves every node toward its neighbours on a network."""

    def __init__(self, label: str, *, target: Role = 'nodes', k: float = 1.0) -> None:
        super().__init__(label)
        self.target = target
        self.k = k

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the sum of the differences to the neighbours as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x = context.target.x
            self.laws_: list[BaseComponent] = [
                RateLaw(x.diff(context.t), context.k * neighbors(context.target.other.x - x)),
            ]


def test_get_declared_spreads_heat_on_a_lattice():
    """Test that heat from one hot site spreads as the exact solution of the discrete heat equation."""
    heat = np.zeros(32)
    heat[16] = 1.0
    plate = Element('plate', values={'u': heat}, space=LatticeSpace((32,)))
    system = System(elements=[plate], interactions=[Diffusion('heat')]).prepare()
    assert system.equations_ == {'plate.u[i]': 'heat.D*laplacian(plate.u[i])'}
    last = system.simulate(until=0.5, dt=0.01).get_values('plate.u')[-1]
    exact = ive(np.abs(np.arange(32) - 16), 2 * 0.5)
    assert last == pytest.approx(exact, abs=1e-6)


def test_get_declared_spreads_heat_on_a_mesh():
    """Test that the slowest mode of a square plate with insulated edges decays at its rate, at second order."""
    errors = []
    for n in (9, 17):
        grid = skfem.MeshTri.init_tensor(np.linspace(0, 1, n), np.linspace(0, 1, n))
        heat = np.cos(np.pi * grid.p[0]) * np.cos(np.pi * grid.p[1])
        plate = Element('plate', values={'u': heat}, space=CellComplexSpace(points=grid.p, cells=grid.t))
        system = System(elements=[plate], interactions=[Diffusion('heat', D=0.1)]).prepare()
        last = system.simulate(until=0.5, dt=0.005).get_values('plate.u')[-1]
        errors.append(np.abs(last - heat * np.exp(-2 * np.pi**2 * 0.1 * 0.5)).max())
    assert errors[1] < 0.01
    assert errors[0] / errors[1] > 3.0


def test_get_declared_sums_over_network_neighbours():
    """Test that a sum over the neighbours of a path of four nodes follows the exact consensus `exp(-L t) x`."""
    graph = nx.path_graph(4)
    start = np.array([1.0, 0.0, 0.0, 0.0])
    nodes = Element('nodes', values={'x': start}, space=NetworkSpace(graph))
    system = System(elements=[nodes], interactions=[Consensus('consensus')]).prepare()
    last = system.simulate(until=0.5, dt=0.01).get_values('nodes.x')[-1]
    exact = expm(-nx.laplacian_matrix(graph).toarray() * 0.5) @ start
    assert last == pytest.approx(exact, abs=1e-6)


class Resistor(Interaction):
    """Resistor whose current flows from node `a` to node `b` and adds into the balance of both."""

    def __init__(self, label: str, *, a: Role = 'a', b: Role = 'b', R: float = 1.0) -> None:
        super().__init__(label)
        self.a = a
        self.b = b
        self.R = R

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the current out of `a` and into `b` as terms of their balances."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            current = (context.a.v - context.b.v) / context.R
            self.laws_: list[BaseComponent] = [
                ConstraintLaw(-current, 0, target=context.a.v),
                ConstraintLaw(current, 0, target=context.b.v),
            ]


class VoltageSource(Interaction):
    """Source that holds node `plus` at `E (1 + w sin t)` above node `minus`, its current held by `branch`."""

    def __init__(
        self,
        label: str,
        *,
        branch: Role = 'branch',
        plus: Role = 'plus',
        minus: Role = 'minus',
        E: float = 1.0,
        w: float = 0.0,
    ) -> None:
        super().__init__(label)
        self.branch = branch
        self.plus = plus
        self.minus = minus
        self.E = E
        self.w = w

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the voltage as the equation of the current, and the current as terms of both balances."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            current = context.branch.i
            voltage = context.E * (1 + context.w * sin(context.t))
            self.laws_: list[BaseComponent] = [
                ConstraintLaw(context.plus.v - context.minus.v, voltage, target=current),
                ConstraintLaw(current, 0, target=context.plus.v),
                ConstraintLaw(-current, 0, target=context.minus.v),
            ]


class Ground(Interaction):
    """Ground that holds a node at zero potential, its current held by `branch`."""

    def __init__(self, label: str, *, branch: Role = 'branch', node: Role = 'node') -> None:
        super().__init__(label)
        self.branch = branch
        self.node = node

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set zero potential as the equation of the current, and the current as a term of the balance."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseComponent] = [
                ConstraintLaw(context.node.v, 0, target=context.branch.i),
                ConstraintLaw(context.branch.i, 0, target=context.node.v),
            ]


class Loop(Interaction):
    """Series RLC loop whose charge follows `L q'' + R q' + q / C = 0`, written as one equation."""

    def __init__(self, label: str, *, target: Role = 'loop', L: float = 1.0, R: float = 0.5, C: float = 0.25) -> None:
        super().__init__(label)
        self.target = target
        self.L = L
        self.R = R
        self.C = C

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the voltage law of the loop."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            q, t = context.target.q, context.t
            self.laws_: list[BaseComponent] = [
                ConstraintLaw(context.L * q.diff(t, 2) + context.R * q.diff(t) + q / context.C == 0),
            ]


def test_get_declared_reads_the_order_of_a_state_from_the_derivatives_of_a_constraint():
    """Test that a constraint that reads `q''` gives `q` order 2, and the charge decays as `exp(-t / 4) cos(w t)`."""
    loop = Element('loop', values={'q': 1.0, "q'": -0.25})
    system = System(elements=[loop], interactions=[Loop('kvl')]).prepare()
    assert system.states_ == ('loop.q', "loop.q'")
    trajectory = system.simulate(until=5.0, dt=0.5)
    frequency = np.sqrt(4 - 1 / 16)
    expected = np.exp(-trajectory.times / 4) * np.cos(frequency * trajectory.times)
    assert trajectory.get_values('loop.q') == pytest.approx(expected, abs=1e-3)


def test_get_declared_balances_the_currents_of_three_resistors():
    """Test that the currents of three resistors into the node they share sum to zero at every frame."""
    nodes = [Element(name, values={'v': 0.0}) for name in ('n', 'a1', 'a2', 'g')]
    branches = [Element(name, values={'i': 0.0}) for name in ('s1', 's2', 'earth')]
    interactions = [
        Resistor('r1', a='a1', b='n', R=1.0),
        Resistor('r2', a='a2', b='n', R=2.0),
        Resistor('r3', a='g', b='n', R=4.0),
        VoltageSource('e1', branch='s1', plus='a1', minus='g', E=10.0, w=0.5),
        VoltageSource('e2', branch='s2', plus='a2', minus='g', E=5.0),
        Ground('ground', branch='earth', node='g'),
    ]
    system = System(elements=nodes + branches, interactions=interactions).prepare()
    assert set(system.states_) == {'n.v', 'a1.v', 'a2.v', 'g.v', 's1.i', 's2.i', 'earth.i'}
    trajectory = system.simulate(until=3.0, dt=0.5)
    v = {name: trajectory.get_values(f'{name}.v') for name in ('n', 'a1', 'a2', 'g')}
    into_node = (v['a1'] - v['n']) / 1.0 + (v['a2'] - v['n']) / 2.0 + (v['g'] - v['n']) / 4.0
    assert np.abs(into_node).max() < 1e-9
    assert v['a1'] == pytest.approx(10.0 * (1 + 0.5 * np.sin(trajectory.times)))
    assert np.abs(v['g']).max() < 1e-12


class Walk(Interaction):
    """Interaction that moves every coordinate of its target by Brownian steps of diffusion `D`."""

    def __init__(self, label: str, *, target: Role = 'walkers', D: float = 0.5) -> None:
        super().__init__(label)
        self.target = target
        self.D = D

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set a noise of variance `2 D` per unit time on `x`."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseComponent] = [NoiseLaw(context.target.x, sqrt(2 * context.D))]


def test_get_declared_reads_a_value_named_after_an_axis_as_a_position():
    """Test that walkers whose value `x` names the axis of a line diffuse with `msd = 2 D t`."""
    walkers = Element('walkers', values={'x': 0.0}, size=400, space=ContinuousSpace(n_dim=1))
    system = System(elements=[walkers], interactions=[Walk('walk', D=0.5)]).prepare()
    trajectory = system.simulate(until=20.0, dt=0.1, random_state=0)
    assert displacement(trajectory, dt=0.1).diffusion == pytest.approx(0.5, rel=0.15)


def test_get_declared_places_one_walker_in_a_space_without_sites():
    """Test that an element in a continuous space without a size is one walker whose position no law needs."""
    walker = Element('walker', values={'x': 1.0, 'y': 2.0, 'm': 1.0}, space=ContinuousSpace(n_dim=2))
    system = System(elements=[walker]).prepare()
    assert system.states_ == ('walker.x[i]', 'walker.y[i]')
    assert system.parameters_ == ('walker.m[i]',)


class Growth(Interaction):
    """Interaction that adds one node joined to one other node at every step."""

    def __init__(self, label: str, *, among: Role = 'nodes', along: bool | str | None = None) -> None:
        super().__init__(label, among=among, along=along)

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the request of one new node as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseComponent] = [UpdateLaw(TopologyRequestSpec('add-node', edges=1, one=True), 1.0)]


def test_get_declared_records_the_values_of_a_growing_network():
    """Test that the degrees of the nodes of a growing network are a state recorded for every node at every step."""
    graph = nx.path_graph(5)
    degrees = np.array([degree for _, degree in graph.degree()], dtype=float)
    nodes = Element('nodes', values={'k': degrees}, space=NetworkSpace(graph))
    system = System(elements=[nodes], interactions=[Growth('growth')]).prepare()
    assert system.states_ == ('nodes.k[i]',)
    recorded = system.simulate(until=4, dt=1, random_state=0).get_values('nodes.k')
    assert [int(np.count_nonzero(~np.isnan(frame))) for frame in recorded] == [5, 6, 7, 8, 9]
    assert recorded[:, :5] == pytest.approx(np.tile(degrees, (5, 1)))


class Count(Interaction):
    """Interaction that adds one to a count at every step."""

    def __init__(self, label: str, *, target: Role = 'counter') -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the next count as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseComponent] = [UpdateLaw(context.target.n, context.target.n + 1)]


def test_get_declared_reads_an_int_as_an_integer():
    """Test that a count given as an `int` through `values` is an integer state."""
    counter = Element('counter', values={'n': 0})
    system = System(elements=[counter], interactions=[Count('count')])
    assert system.get_tags().domains == {'integer'}
    assert system.prepare().simulate(until=3, dt=1).get_values('counter.n') == pytest.approx([0, 1, 2, 3])


class Julia(Interaction):
    """Interaction of the map `z -> z**2 + c` of a Julia set."""

    def __init__(self, label: str, *, target: Role = 'point', c: complex = -0.4 + 0.6j) -> None:
        super().__init__(label)
        self.target = target
        self.c = c

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the next point as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            z = context.target.z
            self.laws_: list[BaseComponent] = [UpdateLaw(z, z**2 + context.c)]


def test_get_declared_maps_a_complex_value():
    """Test that a complex point of a Julia set follows `z -> z**2 + c` with a complex parameter `c`."""
    point = Element('point', values={'z': 0.1 + 0.1j})
    system = System(elements=[point], interactions=[Julia('julia', c=-0.4 + 0.6j)]).prepare()
    assert system.parameters_ == ('julia.c.real', 'julia.c.imag')
    expected = [0.1 + 0.1j]
    for _ in range(3):
        expected.append(expected[-1] ** 2 - 0.4 + 0.6j)
    assert system.simulate(until=3, dt=1).get_values('point.z') == pytest.approx(expected)


class Wander(Interaction):
    """Interaction whose rule moves every member of its target by its speed."""

    def __init__(self, label: str, *, target: Role = 'walkers') -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rule of the members as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseComponent] = [UpdateLaw(context.target, self.wander)]

    def wander(self, agent: object, system: object) -> None:
        """Move an agent by its speed, or remove it once it passes ten."""
        agent.x += agent.speed
        if agent.x > 10:
            agent.remove()


def test_get_declared_makes_every_value_of_an_agent_group_a_state():
    """Test that a rule may change every value of its agents and remove them, so all are states of a variable group."""
    walkers = Element('walkers', values={'x': 0.0, 'speed': [2.0, 4.0, 6.0]}, size=3)
    system = System(elements=[walkers], interactions=[Wander('wander')]).prepare()
    assert system.states_ == ('walkers.x', 'walkers.speed')
    trajectory = system.simulate(until=3)
    assert trajectory.get_values('walkers.count').tolist() == [3.0, 3.0, 2.0, 1.0]


class Points(Element):
    """Element whose points `z` are annotated as `object`, since they may be one number or an array."""

    def __init__(self, label: str, *, z: object = 0.0, size: int | None = None) -> None:
        super().__init__(label, size=size)
        self.z = z


def test_get_declared_reads_the_kind_of_a_value_annotated_as_object_from_the_value():
    """Test that complex points of an element class whose argument is annotated `object` are complex states."""
    points = Points('point', z=np.array([0.1 + 0.1j, 0.2 - 0.1j]), size=2)
    system = System(elements=[points], interactions=[Julia('julia', target='point', c=0.0)]).prepare()
    assert system.simulate(until=1, dt=1).get_values('point.z')[-1] == pytest.approx([0.02j, 0.03 - 0.04j])


class Drift(Interaction):
    """Interaction that moves every walker by its velocity, `x <- x + v`, and halves the velocity at every step."""

    def __init__(self, label: str, *, target: Role = 'walkers') -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the next position and the next velocity as the laws."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x, v = context.target.x, context.target.v
            self.laws_: list[BaseComponent] = [UpdateLaw(x, x + v), UpdateLaw(v, v / 2)]


def test_get_declared_reads_the_step_of_a_position_as_its_velocity():
    """Test that the value `v` a map adds to a position `x` at every step is the velocity of the walkers."""
    walkers = Element('walkers', values={'x': 0.0, 'v': [1.0, -1.0, 2.0]}, size=3, space=ContinuousSpace(n_dim=1))
    system = System(elements=[walkers], interactions=[Drift('drift')]).prepare()
    trajectory = system.simulate(until=2, dt=1)
    assert alignment(trajectory, kind='mean-vector').order == pytest.approx([1 / 3] * 3)


class Brake(Interaction):
    """Controller that sets the force on a cart to `-2 x'` after every step, while the cart moves as `x'' = force`."""

    def __init__(self, label: str, *, cart: Role = 'cart') -> None:
        super().__init__(label)
        self.cart = cart

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the motion and the force the controller sets."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            cart, t = context.cart, context.t
            speed = cart.x.diff(t)
            self.laws_: list[BaseComponent] = [RateLaw(speed.diff(t), cart.force), BoundaryLaw(cart.force, -2 * speed)]


def test_get_declared_holds_a_value_a_controller_sets_after_every_step():
    """Test that a value of a plain element that a controller sets after every step holds within the step."""
    cart = Element('cart', values={'x': 0.0, 'force': 0.0})
    system = System(elements=[cart], interactions=[Brake('brake')]).prepare()
    trajectory = system.simulate(until=1.0, dt=0.5, initial_state={"cart.x'": 1.0})
    assert 'cart.force' in system.states_
    assert system.equations_['cart.force'] == '0'
    assert trajectory.get_values('cart.force')[1:] == pytest.approx(-2 * trajectory.get_values("cart.x'")[1:])
