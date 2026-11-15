"""Test the `_state_inference` module of `skcomplex.core.engine.compile`."""

from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
import pytest
import skfem
from sage.all__sagemath_symbolics import abs_symbolic, oo, sin
from sage.sets.real_set import RealSet
from scipy.linalg import expm
from scipy.special import ive

from skcomplex.core.components.base import BaseComponent
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import ConstraintLaw, RateLaw, UpdateLaw
from skcomplex.core.components.laws.utils import laplacian, neighbors
from skcomplex.core.components.spaces import LatticeSpace, MeshSpace, NetworkSpace
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

    def _prepare(self, context: SimpleNamespace | None = None) -> None:
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

    def _prepare(self, context: SimpleNamespace | None = None) -> None:
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

    def _prepare(self, context: SimpleNamespace | None = None) -> None:
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

    def _prepare(self, context: SimpleNamespace | None = None) -> None:
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

    def _prepare(self, context: SimpleNamespace | None = None) -> None:
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

    def _prepare(self, context: SimpleNamespace | None = None) -> None:
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

    def _prepare(self, context: SimpleNamespace | None = None) -> None:
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
        plate = Element('plate', values={'u': heat}, space=MeshSpace(grid.p, grid.t))
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

    def _prepare(self, context: SimpleNamespace | None = None) -> None:
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

    def _prepare(self, context: SimpleNamespace | None = None) -> None:
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

    def _prepare(self, context: SimpleNamespace | None = None) -> None:
        """Set zero potential as the equation of the current, and the current as a term of the balance."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseComponent] = [
                ConstraintLaw(context.node.v, 0, target=context.branch.i),
                ConstraintLaw(context.branch.i, 0, target=context.node.v),
            ]


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
