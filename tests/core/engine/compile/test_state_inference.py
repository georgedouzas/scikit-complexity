"""Test the `_state_inference` module of `skcomplex.core.engine.compile`."""

from collections.abc import Sequence
from dataclasses import dataclass
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
from shapely import box

from skcomplex.core.analysis import alignment, displacement
from skcomplex.core.base import BaseSpec
from skcomplex.core.components import Observable
from skcomplex.core.components.base import BaseComponent
from skcomplex.core.components.elements import Element, Member, Period, Quantity, VisualSpec
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import (
    ConstraintLaw,
    NoiseLaw,
    RateLaw,
    TopologyRequestSpec,
    UpdateLaw,
)
from skcomplex.core.components.laws.utils import (
    bernoulli,
    components,
    divergence,
    laplacian,
    neighbors,
    variance,
    where,
)
from skcomplex.core.components.spaces import (
    BaseSpace,
    CellComplexSpace,
    ContinuousSpace,
    FiniteVolumeSpec,
    LatticeSpace,
    LayerSpec,
    NetworkSpace,
    RadiusNeighborhood,
)
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import ConservationObserver
from skcomplex.core.engine.solvers import AgentView, ETDRK4Solver, IMEXEulerSolver, RK4Solver, SSPRKSolver
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System
from skcomplex.core.types import Role
from skcomplex.systems.mathematics.networks import TriadicPercolationSystem

from ....conftest import _queue_run

SITES = 16
WAVE = 1.0 + 0.5 * np.sin(2 * np.pi * np.arange(SITES) / SITES) + np.eye(SITES)[3]
FIELDS = {
    'zero-flux': ({'boundary': {'u': 'zero-flux'}}, False, RK4Solver(dt=0.01)),
    'fixed': ({'boundary': {'u': ('fixed', 2.0)}}, False, RK4Solver(dt=0.01)),
    'periodic': ({'boundary': {'u': 'periodic'}}, False, RK4Solver(dt=0.01)),
    'spectral': ({'discretization': 'spectral'}, False, ETDRK4Solver(dt=0.01)),
    'symmetry': ({'discretization': 'spectral', 'symmetry': {'u': ('odd',)}}, False, ETDRK4Solver(dt=0.01)),
    'finite-volume': ({'discretization': 'finite-volume'}, True, SSPRKSolver()),
    'godunov': (
        {'discretization': 'finite-volume', 'scheme': FiniteVolumeSpec(limiter='van-leer', riemann='godunov')},
        True,
        SSPRKSolver(),
    ),
}


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


class DelayedDamping(Interaction):
    """Interaction of a damping by the velocity a delay ago."""

    def __init__(self, label: str, *, target: Role = 'ball', tau: float = 1.0) -> None:
        super().__init__(label)
        self.target = target
        self.tau = tau

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the delayed damping of the velocity as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x, t = context.target.x, context.t
            self.laws_ = [RateLaw(x.diff(t, 2), -x.diff(t)(t=t - context.tau))]


class Wave(Interaction):
    """Interaction of the wave equation on a lattice."""

    def __init__(self, label: str, *, target: Role = 'string', c: float = 1.0) -> None:
        super().__init__(label)
        self.target = target
        self.c = c

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the acceleration of every site as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            u, t = context.target.u, context.t
            self.laws_ = [RateLaw(u.diff(t, 2), context.c**2 * laplacian(u))]


def test_get_declared_makes_untouched_values_parameters():
    """A particle no law acts on has no states, and its values are parameters."""
    system = System(elements=[Particle('ball', x=2.0)]).prepare()
    assert system.states_ == ()
    assert system.parameters_ == ('ball.x', 'ball.m')


def test_get_declared_keeps_the_order_of_the_values():
    """The values keep the order of the class arguments and then of `values`, not the alphabetical one."""
    stock = Element('stock', values={'z': 1.0, 'a': 2.0})
    system = System(elements=[Particle('ball', x=2.0), stock]).prepare()
    assert system.parameters_ == ('ball.x', 'ball.m', 'stock.z', 'stock.a')


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


def test_get_declared_moves_a_value_with_a_starting_velocity_and_no_law():
    """Test that a point given the velocity `x'` and no law moves at that velocity, as Newton's first law says."""
    ball = Element('ball', values={'x': 1.0, "x'": 3.0})
    system = System(elements=[ball]).prepare()
    assert system.states_ == ('ball.x', "ball.x'")
    assert system.equations_ == {'ball.x': "ball.x'", "ball.x'": '0'}
    trajectory = system.simulate(until=2.0, dt=1.0)
    assert trajectory.get_values('ball.x').tolist() == pytest.approx([1.0, 4.0, 7.0])
    assert trajectory.get_values("ball.x'").tolist() == pytest.approx([3.0, 3.0, 3.0])


def test_get_declared_starts_past_values_from_values():
    """Test that values named with past indices start the lags, and a missing one is the value itself."""
    economy = Element('economy', values={'y': 1.0, 'y[-2]': 0.5})
    system = System(elements=[economy], interactions=[Accelerator('samuelson')]).prepare()
    values = system.simulate(until=1, dt=1).get_values('economy.y')
    assert values.tolist() == pytest.approx([1.0, 1 + 1.2 * 1.0 - 0.5 * 0.5])


@pytest.mark.parametrize(
    ('values', 'message'),
    [({'x': 1.0, "z'": 1.0}, "no value named 'z'"), ({'x': 1.0, "x''": 1.0}, "of 'x' at index 2")],
)
def test_get_declared_rejects_unread_starting_values(values, message):
    """Test that a starting value of an unknown value, or of a derivative no law reads, raises an error."""
    with pytest.raises(ValidationError, match=message):
        System(elements=[Element('ball', values=values)], interactions=[Gravity('gravity')]).prepare()


def test_get_declared_starts_the_derivative_of_every_member():
    """Test that a group of balls thrown at three speeds starts with one derivative per member."""
    balls = Element('ball', values={'x': np.zeros(3), "x'": np.array([1.0, 2.0, 3.0])}, size=3)
    trajectory = System(elements=[balls], interactions=[Gravity('gravity')]).prepare().simulate(until=1.0, dt=0.5)
    assert trajectory.get_values("ball.x'")[0] == pytest.approx([1.0, 2.0, 3.0])
    assert trajectory.get_values('ball.x')[-1] == pytest.approx(np.array([1.0, 2.0, 3.0]) - 9.81 / 2)


def test_get_declared_starts_the_derivative_of_every_site():
    """Test that a string on a lattice starts with one velocity per site, and moves first by that velocity."""
    speeds = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    string = Element('string', values={'u': np.zeros(5), "u'": speeds}, space=LatticeSpace(5))
    trajectory = System(elements=[string], interactions=[Wave('wave')]).prepare().simulate(until=0.01, dt=0.01)
    assert trajectory.get_values("string.u'")[0] == pytest.approx(speeds)
    assert trajectory.get_values('string.u')[-1] == pytest.approx(0.01 * speeds, abs=1e-5)


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


class ElementDelayFeedback(Interaction):
    """Interaction of a delayed negative feedback whose delay is a value of the element."""

    def __init__(self, label: str, *, target: Role = 'ball') -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the decay delayed by the value `tau` of the target as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x, t = context.target.x, context.t
            self.laws_ = [RateLaw(x, -x(t=t - context.target.tau))]


def test_get_declared_reads_a_delay_that_is_a_value_of_an_element():
    """A delay may be a value of an element, such as the incubation period of a population."""
    ball = Element('ball', values={'x': 1.0, 'tau': 1.0})
    system = System(elements=[ball], interactions=[ElementDelayFeedback('feedback')]).prepare()
    assert system.equations_ == {'ball.x': '-ball.x(t - ball.tau)'}
    values = system.simulate(until=2.0, dt=1.0).get_values('ball.x')
    assert values == pytest.approx([1.0, 0.0, -0.5], abs=1e-6)


def test_get_declared_reads_a_delayed_derivative():
    """A derivative read a delay before the current time keeps its delay, as a driver reacts to a past speed."""
    system = System(elements=[Particle('ball')], interactions=[DelayedDamping('damping')]).prepare()
    assert system.equations_ == {'ball.x': "ball.x'", "ball.x'": "-ball.x'(t - damping.tau)"}
    values = system.simulate(until=2.0, dt=1.0, initial_state={"ball.x'": 1.0}).get_values("ball.x'")
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


@pytest.mark.parametrize(('boundary', 'end'), [('periodic', 0.25), ('reflecting', 1.0)])
def test_get_declared_keeps_one_walker_with_a_vector_position_of_one_component_in_the_box(boundary, end):
    """Test that the position `[0.25]` of one walker on a line is a vector of one component the boundary acts on."""
    space = ContinuousSpace((1.0,), boundary=boundary)
    drift = Interaction('drift', roles={'target': 'walker'}, laws=lambda s: [RateLaw(s.target.x, 1.0)])
    walker = Element('walker', values={'x': [0.25]}, space=space)
    system = System(elements=[walker], interactions=[drift], space=space).prepare()
    assert system.states_ == ('walker.x[i, 0]',)
    assert system.simulate(until=1.0, dt=0.5).get_values('walker.x').ravel()[-1] == pytest.approx(end, abs=1e-3)


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


class SpeciesElement(Element):
    """Species with an abundance whose default is that of a new species."""

    def __init__(self, label: str, *, x: float | np.ndarray = 0.5, space: NetworkSpace | None = None) -> None:
        super().__init__(label, space=space)
        self.x = x


def test_get_declared_gives_new_members_the_default_of_their_class():
    """Test that a member added to a group takes the default of its class, not the value of the first member."""
    nodes = SpeciesElement('nodes', x=np.array([2.0, 3.0, 4.0]), space=NetworkSpace(nx.path_graph(3)))
    system = System(elements=[nodes], interactions=[Growth('growth')]).prepare()
    recorded = system.simulate(until=2, dt=1, random_state=0).get_values('nodes.x')
    assert recorded[-1][:5] == pytest.approx([2.0, 3.0, 4.0, 0.5, 0.5])


@dataclass(frozen=True)
class OutlineSpec(BaseSpec):
    """Spec of the outline a cell is drawn with."""

    sides: int = 6


class OutlinedCellElement(Element):
    """Cell with a size and the outline it is drawn with."""

    def __init__(self, label: str, *, size_: float = 1.0, outline: OutlineSpec | None = None) -> None:
        super().__init__(label)
        self.size_ = size_
        self.outline = outline


def test_get_declared_reads_a_spec_argument_as_a_setting():
    """Test that an argument that holds a spec, such as the outline a cell is drawn with, is no value of the element."""
    system = System(elements=[OutlinedCellElement('cell', outline=OutlineSpec())]).prepare()
    assert system.parameters_ == ('cell.size_',)


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


def test_get_declared_lets_agents_hold_a_value_named_kind():
    """Test that agents may hold a value named `kind`, such as the group of a resident of Schelling's city."""
    walkers = Element('walkers', values={'x': 0.0, 'speed': [2.0, 4.0, 6.0], 'kind': [1, 2, 3]}, size=3)
    trajectory = System(elements=[walkers], interactions=[Wander('wander')]).prepare().simulate(until=3)
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
            self.laws_: list[BaseComponent] = [RateLaw(speed.diff(t), cart.force), UpdateLaw(cart.force, -2 * speed)]


def test_get_declared_holds_a_value_a_controller_sets_after_every_step():
    """Test that a value of a plain element that a controller sets after every step holds within the step."""
    cart = Element('cart', values={'x': 0.0, 'force': 0.0})
    system = System(elements=[cart], interactions=[Brake('brake')]).prepare()
    trajectory = system.simulate(until=1.0, dt=0.5, initial_state={"cart.x'": 1.0})
    assert 'cart.force' in system.states_
    assert system.equations_['cart.force'] == '0'
    assert trajectory.get_values('cart.force')[1:] == pytest.approx(-2 * trajectory.get_values("cart.x'")[1:])


class WeightedExchange(Interaction):
    """Exchange along every edge at the rate of the weight of the link on that edge."""

    def __init__(self, label: str, *, among: Role = 'nodes', along: bool | str | None = True) -> None:
        super().__init__(label, among=among, along=along)

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the flow from the source to the target of an edge, weighted by its link."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            flow = s.edge.w * (s.source.x - s.target.x)
            self.laws_: list[BaseComponent] = [
                RateLaw(s.target.x.diff(s.t), flow),
                RateLaw(s.source.x.diff(s.t), -flow),
            ]


def test_get_declared_puts_one_member_on_every_edge():
    """Test that links on the edges of a path hold one weight each, which an exchange along the edges reads."""
    graph = nx.path_graph(3)
    space = NetworkSpace(graph)
    nodes = Element('nodes', values={'x': np.array([1.0, 0.0, 0.0])}, space=space)
    links = Element('links', values={'w': np.array([2.0, 0.5])}, space=space, on='edges')
    system = System(elements=[nodes, links], interactions=[WeightedExchange('exchange')]).prepare()
    assert 'links.w[i]' in system.parameters_
    last = system.simulate(until=0.5, dt=0.01).get_values('nodes.x')[-1]
    nx.set_edge_attributes(graph, {(0, 1): 2.0, (1, 2): 0.5}, 'w')
    laplacian = nx.laplacian_matrix(graph, weight='w').toarray()
    assert last == pytest.approx(expm(-laplacian * 0.5) @ np.array([1.0, 0.0, 0.0]), abs=1e-6)


class Regulation(Interaction):
    """Regulation of every link by the third node of its hyperedge, as in triadic percolation."""

    def __init__(self, label: str, *, among: Role = 'nodes', along: bool | str | None = 'hyperedges', p: float = 0.5):
        super().__init__(label, among=among, along=along)
        self.p = p

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the activity of the link from the activity and the sign of its regulator."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            regulator = s.others[1].active
            allowed = where(s.edge.sign > 0, regulator, 1 - regulator)
            retained = bernoulli('retained', s.p) * allowed
            self.laws_: list[BaseComponent] = [UpdateLaw(s.edge.active, retained, phase='regulation')]


class Percolation(Interaction):
    """Activity of every node in the giant component of the active links."""

    def __init__(self, label: str, *, nodes: Role = 'nodes', links: Role = 'links') -> None:
        super().__init__(label)
        self.nodes = nodes
        self.links = links

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the membership of the giant component."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            giant = components(context.links.active, reduce='giant')
            self.laws_: list[BaseComponent] = [UpdateLaw(context.nodes.active, giant, phase='percolation')]


def test_get_declared_gives_the_signed_links_of_triadic_percolation():
    """Test that signed links on the edges, regulated along the hyperedges, run triadic percolation step for step."""
    ready = TriadicPercolationSystem(500, p=0.5).prepare().system_
    expected = ready.simulate(until=10, random_state=0).get_values('nodes.active')
    space, signs = ready.elements[0].space, ready.elements[1].sign
    nodes = Element('nodes', values={'active': 1}, space=space)
    links = Element('links', values={'active': 1, 'sign': signs}, space=space, on='edges')
    system = System(
        elements=[nodes, links],
        interactions=[Regulation('regulation', p=0.5), Percolation('percolation')],
    )
    trajectory = system.prepare().simulate(until=10, random_state=0)
    np.testing.assert_array_equal(trajectory.get_values('nodes.active'), expected)


class Majority(Interaction):
    """Pull of every member of a hyperedge towards the mean of its members, at the rate of the hyperedge."""

    def __init__(self, label: str, *, among: Role = 'nodes', along: bool | str | None = 'hyperedges') -> None:
        super().__init__(label, among=among, along=along)

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the pull towards the mean of the members."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            center = (s.member.x + sum(other.x for other in s.others)) / (1 + len(s.others))
            pull = s.hyperedge.k * (center - s.member.x)
            self.laws_: list[BaseComponent] = [RateLaw(s.member.x.diff(s.t), pull)]


def test_get_declared_puts_one_member_on_every_hyperedge():
    """Test that a group of three, whose hyperedge holds its rate, pulls its members to their mean, and others not."""
    space = NetworkSpace(nx.empty_graph(4), hyperedges=[(0, 1, 2)])
    nodes = Element('nodes', values={'x': np.array([3.0, 0.0, 0.0, 5.0])}, space=space)
    groups = Element('groups', values={'k': 2.0}, space=space, on='hyperedges')
    system = System(elements=[nodes, groups], interactions=[Majority('majority')]).prepare()
    last = system.simulate(until=0.5, dt=0.01).get_values('nodes.x')[-1]
    assert last == pytest.approx([1.0 + 2.0 * np.exp(-1.0), 1.0 - np.exp(-1.0), 1.0 - np.exp(-1.0), 5.0], abs=1e-5)


class Transport(Interaction):
    """Diffusion of the state `u` of a field, or its transport by the Burgers flux."""

    def __init__(self, label: str, *, target: Role = 'rod', flux: bool = False) -> None:
        super().__init__(label)
        self.target = target
        self.flux = flux

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rate of `u`."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            u = context.target.u
            self.laws_: list[BaseComponent] = [
                RateLaw(u.diff(context.t), -divergence([u**2 / 2]) if self.flux else laplacian(u)),
            ]


def _run_field(name: str) -> np.ndarray:
    """Return the values of `u` on a lattice of the settings of a field, at the times 0, 0.25, and 0.5."""
    settings, flux, solver = FIELDS[name]
    values = WAVE - WAVE[::-1] if 'symmetry' in settings else WAVE
    rod = Element('rod', values={'u': values}, space=LatticeSpace(SITES, spacing=0.5, **settings))
    system = System(elements=[rod], interactions=[Transport('transport', flux=flux)], solver=solver).prepare()
    return system.simulate(until=0.5, dt=0.25).get_values('rod.u')


@pytest.mark.parametrize('name', ['zero-flux', 'periodic', 'spectral', 'finite-volume', 'godunov'])
def test_field_on_a_lattice_conserves_its_total_without_inflow(name):
    """Test that a closed or periodic lattice and a finite-volume scheme keep the total of the field."""
    values = _run_field(name)
    np.testing.assert_allclose(values.sum(axis=1), WAVE.sum(), rtol=1e-12)


def test_field_on_a_lattice_with_fixed_edges_rises_toward_their_value():
    """Test that fixed edges of value 2 raise the field at both ends, where a closed lattice barely moves."""
    fixed, closed = _run_field('fixed'), _run_field('zero-flux')
    assert fixed[-1, 0] > closed[-1, 0] + 0.4
    assert fixed[-1, -1] > closed[-1, -1] + 0.4
    assert fixed[-1].sum() > WAVE.sum() + 2.0


def test_spectral_field_on_a_lattice_decays_as_its_fourier_modes():
    """Test that the spectral heat equation damps every Fourier mode by `exp(-k^2 t)`, as the exact solution does."""
    k = 2 * np.pi * np.fft.fftfreq(SITES, d=0.5)
    exact = np.fft.ifft(np.fft.fft(WAVE) * np.exp(-(k**2) * 0.5)).real
    np.testing.assert_allclose(_run_field('spectral')[-1], exact, atol=1e-12)


def test_odd_spectral_field_on_a_lattice_keeps_a_zero_total():
    """Test that an odd field under the spectral scheme with odd symmetry keeps its zero total as it decays."""
    values = _run_field('symmetry')
    np.testing.assert_allclose(values.sum(axis=1), 0.0, atol=1e-12)
    assert values[-1].max() < values[0].max()


def test_finite_volume_limiter_changes_the_burgers_transport():
    """Test that the van Leer limiter with the Godunov flux gives another Burgers transport than the default one."""
    assert not np.allclose(_run_field('godunov'), _run_field('finite-volume'))


def test_field_on_a_mesh_holds_its_fixed_edges():
    """Test that a fixed edge on a mesh, a function of the coordinates, holds the boundary as the inside relaxes."""
    grid = skfem.MeshTri.init_lshaped().refined(1)
    edge = ('fixed', lambda x, y: x + y)
    rod = Element('rod', values={'u': 0.0}, space=CellComplexSpace(points=grid.p, cells=grid.t, boundary={'u': edge}))
    system = System(elements=[rod], interactions=[Transport('transport')], solver=IMEXEulerSolver(dt=0.05)).prepare()
    values = system.simulate(until=0.5, dt=0.25).get_values('rod.u')
    plane = grid.p[0] + grid.p[1]
    boundary = grid.boundary_nodes()
    inside = np.setdiff1d(np.arange(plane.size), boundary)
    np.testing.assert_allclose(values[:, boundary], np.tile(plane[boundary], (3, 1)), atol=1e-12)
    np.testing.assert_allclose(values[-1, inside], plane[inside], atol=0.05)


class Reserve(Element):
    """Energy reserve whose stock holds the quantity energy."""

    def __init__(
        self,
        label: str,
        stock: Annotated[float, RealSet.closed_open(0, oo), Quantity('energy')] = 10.0,
    ) -> None:
        super().__init__(label)
        self.stock = stock


class Extraction(Interaction):
    """Interaction that draws energy from a reserve into a sink, or burns it outside any flow."""

    def __init__(self, label: str, *, target: Role = 'reserve', r: float = 0.5, flow: bool = True) -> None:
        super().__init__(label)
        self.target = target
        self.r = r
        self.flow = flow

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the extraction as a flow or as a plain rate."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            stock = context.target.stock
            rate = context.r * stock
            law = RateLaw({stock: -1}, rate, label='extracted') if self.flow else RateLaw(stock, -rate)
            self.laws_ = [law]


def test_get_declared_reads_the_quantity_of_a_value():
    """A value annotated with a quantity is checked by the conservation monitor, and only flows may change it."""
    monitor = ConservationObserver()
    system = System(elements=[Reserve('reserve')], interactions=[Extraction('extraction')]).prepare()
    trajectory = system.simulate(until=2.0, dt=1.0, observers=[monitor])
    assert monitor.totals_['energy'] == pytest.approx(10.0 * np.exp(-1.0), rel=1e-4)
    assert trajectory.get_values('reserve.extracted')[-1] == pytest.approx(5.0 * np.exp(-1.0), rel=1e-4)
    burning = System(elements=[Reserve('reserve')], interactions=[Extraction('extraction', flow=False)])
    with pytest.raises(ValidationError, match=r"holds the quantity 'energy'"):
        burning.prepare()


class Worker(Element):
    """Workers, each with the index of its employer among the firms, or -1 for none."""

    def __init__(self, label: str, *, employer: Annotated[int, Member('firms')] = -1, size: int | None = None) -> None:
        super().__init__(label, size=size)
        self.employer = employer


class Closing(Interaction):
    """Rule that closes the firm of the first worker and records what each worker's employer is."""

    def __init__(self, label: str, *, target: Role = 'workers', firms: Role = 'firms') -> None:
        super().__init__(label)
        self.target = target
        self.firms = firms

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rule of the whole system."""
        super()._prepare(context)
        self.laws_ = [UpdateLaw(context, self.close)]

    def close(self, system):
        """Remove the first firm."""
        system.get_agents('firms')[0].remove()


def test_get_declared_reads_a_member_of_another_group_and_clears_it_when_removed():
    """Test that a value annotated with `Member` names a firm by its id, and becomes -1 when the firm closes."""
    workers = Worker('workers', employer=np.array([0, 1, -1, 0]), size=4)
    firms = Element('firms', values={'wage': [1.0, 2.0]}, size=2)
    system = System(elements=[firms, workers], interactions=[Closing('closing')]).prepare()
    trajectory = system.simulate(until=1, random_state=0)
    first, second = trajectory.get_agents('firms', 0).ids.tolist()
    assert trajectory.get_agents('workers', 0).values['employer'].tolist() == [first, second, -1, first]
    assert trajectory.get_agents('workers', 1).values['employer'].tolist() == [-1, second, -1, -1]


def _spread(center, left, right, *gates):
    """Return the larger neighbour on an open site, and the site itself on a closed one."""
    return np.where(gates[0] > 0, np.maximum(left, right), center)


class Spread(Interaction):
    """Interaction whose local rule reads a value of every site that no law changes."""

    def __init__(self, label: str, *, target: Role = 'cells') -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set a local rule that reads the signal and the gate of every site and its neighbours."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            cells = context.target
            around = RadiusNeighborhood(1, center=True)
            self.laws_ = [UpdateLaw(cells.s, _spread, neighborhood=around, reads=(cells.s, cells.gate))]


def test_get_declared_keeps_a_value_a_local_rule_reads():
    """Test that a local rule reads a value of every site that no law changes, the gate of a spreading signal."""
    values = {'s': np.array([0.0, 0.0, 1.0, 0.0, 0.0]), 'gate': np.array([1.0, 0.0, 1.0, 1.0, 1.0])}
    cells = Element('cells', values=values, space=LatticeSpace(5))
    system = System(elements=[cells], interactions=[Spread('spread')]).prepare()
    assert system.states_ == ('cells.s[i]', 'cells.gate[i]')
    signal = system.simulate(until=2).get_values('cells.s')
    assert signal.tolist() == [[0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0, 1.0]]


class Relaxation(Interaction):
    """Interaction that relaxes a matrix value to zero."""

    def __init__(self, label: str, *, target: Role = 'shape', rate: float = 1.0) -> None:
        super().__init__(label)
        self.target = target
        self.rate = rate

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the decay of the matrix as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_ = [RateLaw(context.target.S, -context.rate * context.target.S)]


@pytest.mark.parametrize('members', [{}, {'size': 3}, {'space': LatticeSpace(3)}])
def test_get_declared_reads_a_matrix_value_as_a_matrix_state(members):
    """Test that a square matrix is a state that laws read as a matrix, for one element, a group, or a field."""
    start = np.array([[1.0, 0.5], [0.5, 2.0]])
    shape = Element('shape', values={'S': np.stack([start] * 3) if members else start}, **members)
    system = System(elements=[shape], interactions=[Relaxation('relaxation')]).prepare()
    suffix = '[i]' if members else ''
    assert system.states_ == tuple(f'shape.S_{name}{suffix}' for name in ('xx', 'xy', 'yx', 'yy'))
    final = system.simulate(until=1.0, dt=0.5).get_values('shape.S')[-1]
    assert np.reshape(final, (-1, 4)) == pytest.approx(
        np.tile(start.ravel() * np.exp(-1.0), (3 if members else 1, 1)),
        rel=1e-4,
    )


class Body(Element):
    """Body drawn as a point at its position."""

    visual = VisualSpec('points', {'position': 'x'})

    def __init__(self, label: str, x: float = 0.0, *, size: int | None = None) -> None:
        super().__init__(label, size=size)
        self.x = x


@pytest.mark.parametrize('size', [None, 3])
def test_get_declared_keeps_the_drawing_of_the_class(size):
    """Test that a plain element, alone or in a group, is drawn as the `visual` of its class says."""
    system = System(elements=[Body('body', size=size)], interactions=[Gravity('gravity', target='body')]).prepare()
    problem = system.simulate(until=0.1, dt=0.1).problem
    assert problem.elements[0].visual == Body.visual


class Spring(Interaction):
    """Interaction that pulls a position in the plane back to the origin."""

    def __init__(self, label: str, *, target: Role = 'ball', k: float = 1.0) -> None:
        super().__init__(label)
        self.target = target
        self.k = k

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the acceleration of the vector position as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x, t = context.target.x, context.t
            self.laws_ = [RateLaw(x.diff(t, 2), -context.k * x)]


@pytest.mark.parametrize(
    ('members', 'start', 'names'),
    [
        ({}, [1.0, 0.0], ('ball.x[0]', 'ball.x[1]', "ball.x'[0]", "ball.x'[1]")),
        (
            {'space': ContinuousSpace(n_dim=2)},
            [1.0, 0.0],
            ('ball.x[i, 0]', 'ball.x[i, 1]', "ball.x'[i, 0]", "ball.x'[i, 1]"),
        ),
        ({'size': 2}, [[1.0, 0.0], [0.0, 1.0]], ('ball.x[i, 0]', 'ball.x[i, 1]', "ball.x'[i, 0]", "ball.x'[i, 1]")),
    ],
)
def test_get_declared_reads_the_second_derivative_of_a_vector(members, start, names):
    """Test that a rate law on `x.diff(t, 2)` of a vector gives the vector order two, like a scalar."""
    ball = Element('ball', values={'x': np.array(start)}, **members)
    system = System(elements=[ball], interactions=[Spring('spring')]).prepare()
    assert system.states_ == names
    final = system.simulate(until=np.pi, dt=np.pi / 2).get_values('ball.x')[-1]
    assert np.ravel(final) == pytest.approx(-np.ravel(start), abs=1e-5)


class Charges(Element):
    """Charges whose value is one number or one number per member."""

    def __init__(self, label: str, q: float | Sequence[float] = 1.0, *, size: int | None = None) -> None:
        super().__init__(label, size=size)
        self.q = q


def test_get_declared_accepts_an_array_for_a_hint_of_several_types():
    """Test that a value hinted as `float | Sequence[float]` takes an array of real values, one per member."""
    system = System(elements=[Charges('charges', q=np.array([1.0, -2.0, 0.5]), size=3)]).prepare()
    assert system.parameters_ == ('charges.q[i]',)
    assert system.simulate(until=0).get_values('charges.q')[0] == pytest.approx([1.0, -2.0, 0.5])


class Rotor(Element):
    """Rotors whose angles wrap around the circle, placed on a line when they have a position."""

    def __init__(
        self,
        label: str,
        theta: Annotated[object, Period(2 * np.pi)] = 0.0,
        *,
        size: int | None = None,
        space: BaseSpace | None = None,
        values: dict[str, object] | None = None,
    ) -> None:
        super().__init__(label, size=size, space=space, values=values)
        self.theta = theta


class Spin(Interaction):
    """Turn of the angles of a target at the unit rate."""

    def __init__(self, label: str, *, target: Role = 'rotors') -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the unit rate of the angles."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_ = [RateLaw(context.target.theta.diff(context.t), 1)]


def test_get_declared_wraps_a_periodic_value_and_reads_it_as_a_phase():
    """Test that an angle with a period comes back within one period and is the phase the alignment reads."""
    rotors = Rotor('rotors', theta=np.array([0.0, 0.0]), size=2)
    trajectory = System(elements=[rotors], interactions=[Spin('spin')]).prepare().simulate(until=4.0, dt=1.0)
    assert trajectory.get_values('rotors.theta')[-1] == pytest.approx([4.0 - 2 * np.pi] * 2)
    assert [element.roles for element in trajectory.problem.elements] == [{'theta': 'phase'}]
    assert alignment(trajectory).order == pytest.approx(np.ones(5))


def test_get_declared_reads_a_periodic_value_of_a_mover_as_a_heading():
    """Test that the angle of an element with a position in a space is its heading."""
    rotors = Rotor('rotors', size=2, space=ContinuousSpace(n_dim=1), values={'x': np.array([[0.0], [1.0]])})
    trajectory = System(elements=[rotors], interactions=[Spin('spin')]).prepare().simulate(until=1.0)
    assert trajectory.problem.elements[0].roles == {'theta': 'direction', 'x': 'position'}


class Echo(Interaction):
    """Map that sets every angle to a tenth of the sum of the angles of its neighbours two steps ago."""

    def __init__(self, label: str, *, target: Role = 'rotors') -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the angle to a tenth of the delayed sum of the neighbours."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            past = context.target.other.theta(t=context.t - 2)
            self.laws_ = [UpdateLaw(context.target.theta, neighbors(past, over=RadiusNeighborhood(1.5)) / 10)]


def test_get_declared_keeps_the_past_values_neighbours_read_from_each_member():
    """Test that past values read through neighbours are kept, and start at the value of each member."""
    line = {'space': ContinuousSpace(n_dim=1), 'values': {'x': np.array([[0.0], [1.0], [2.0]])}}
    rotors = Rotor('rotors', theta=np.array([0.1, 0.2, 0.3]), size=3, **line)
    trajectory = System(elements=[rotors], interactions=[Echo('echo')]).prepare().simulate(until=1)
    assert trajectory.get_values('rotors.theta')[-1] == pytest.approx([0.02, 0.04, 0.02])


class CountedView(AgentView):
    """View of a counter, with its count as an attribute."""

    count: float


class CounterElement(Element):
    """Counters whose agents are seen through `CountedView`."""

    def __init__(self, label: str, *, count: float = 0.0, size: int | None = None) -> None:
        super().__init__(label, size=size)
        self.count = count

    @classmethod
    def _get_view(cls) -> type[CountedView]:
        """Return the view of the counters."""
        return CountedView


class TallyInteraction(Interaction):
    """Rule of the whole system that raises the count of every counter."""

    def __init__(self, label: str, *, target: Role = 'counters') -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context=None):
        """Set the tally as a rule of the system."""
        super()._prepare(context)
        self.laws_ = [UpdateLaw(context, self.tally)]

    def tally(self, system):
        """Raise the count of every counter by one."""
        for counter in system.get_agents('target', CountedView):
            counter.count += 1


def test_a_rule_of_the_system_reads_the_typed_agents_of_a_group_it_binds():
    """Test that a group read only by a rule of the system is agents, seen through the view of its element class."""
    counters = CounterElement('counters', count=np.array([0.0, 5.0]), size=2)
    system = System(elements=[counters], interactions=[TallyInteraction('tally')]).prepare()
    trajectory = system.simulate(until=3, random_state=0)
    assert trajectory.get_agents('counters', 3).values['count'].tolist() == [3.0, 8.0]


class WorkInteraction(Interaction):
    """Agent rule of workers with no values that raises the count of every counter."""

    def __init__(self, label: str, *, target: Role = 'workers', counters: Role = 'counters') -> None:
        super().__init__(label)
        self.target = target
        self.counters = counters

    def _prepare(self, context=None):
        """Set the work of every worker as its rule."""
        super()._prepare(context)
        self.laws_ = [UpdateLaw(context.target, self.work)]

    def work(self, worker, system):
        """Raise the count of every counter by one."""
        for counter in system.get_agents('counters', CountedView):
            counter.count += 1


def test_a_group_with_no_values_runs_its_agent_rules():
    """Test that every member of a group with no values, such as workers who change other agents, runs its rules."""
    workers = Element('workers', size=3)
    counters = CounterElement('counters', size=2)
    system = System(elements=[workers, counters], interactions=[WorkInteraction('work')]).prepare()
    trajectory = system.simulate(until=2, random_state=0)
    assert len(trajectory.get_agents('workers', 2).ids) == 3
    assert trajectory.get_agents('counters', 2).values['count'].tolist() == [6.0, 6.0]


class Combination(Interaction):
    """Interaction that adds one node at every step, made from two nodes with which it forms a hyperedge."""

    def __init__(self, label: str, *, among: Role = 'goods', along: bool | str | None = None) -> None:
        super().__init__(label, among=among, along=along)

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the request of one new node with a hyperedge of three members."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            request = TopologyRequestSpec('add-node', hyperedge=3, independent=True, one=True)
            self.laws_: list[BaseComponent] = [UpdateLaw(request, 1.0)]


def test_get_declared_grows_a_group_on_the_hyperedges_the_laws_add():
    """Test that elements with no values on the nodes and hyperedges of a network grow with the hyperedges added."""
    space = NetworkSpace(nx.empty_graph(3))
    goods = Element('goods', space=space)
    functions = Element('functions', space=space, on='hyperedges')
    trajectory = (
        System(elements=[goods, functions], interactions=[Combination('combination')])
        .prepare()
        .simulate(
            until=4,
            dt=1,
            random_state=0,
        )
    )
    assert trajectory.get_values('goods._active').sum(axis=1).tolist() == [3, 4, 5, 6, 7]
    assert trajectory.get_values('functions._active').sum(axis=1).tolist() == [0, 1, 2, 3, 4]
    members = trajectory.get_values('functions._members')[-1][trajectory.get_values('functions._active')[-1] > 0]
    assert sorted(members[:, -1].tolist()) == [3.0, 4.0, 5.0, 6.0]


def tally_count(s):
    """Return the count of the tally."""
    return s.tally.n


def test_law_reading_an_observable_changes_a_plain_value():
    """Test that the running peak of an observable of a nested system is a state of a plain element."""
    counter = System(
        'counter',
        elements=[Element('tally', values={'n': 0.0})],
        interactions=[
            Interaction('tick', roles={'target': 'tally'}, laws=lambda s: [UpdateLaw(s.target.n, 3 - s.target.n)]),
        ],
    )

    def keep(s):
        return [UpdateLaw(s.target.high, where(s.source.value > s.target.high, s.source.value, s.target.high))]

    system = System(
        elements=[Element('peak', values={'high': 0.0})],
        interactions=[Interaction('keep', roles={'target': 'peak', 'source': 'counter/level'}, laws=keep)],
        systems=[counter],
    ).prepare([Observable('counter/level', tally_count)])
    assert 'peak.high' in system.states_
    trajectory = system.simulate(until=4)
    assert trajectory.get_values('peak.high').tolist() == [0.0, 0.0, 3.0, 3.0, 3.0]


def _settle(agent, system):
    """Count one more step of a resident."""
    agent.steps = agent.steps + 1


def _residents(region: object, size: int = 30) -> System:
    """Return residents who start in named regions of a map and count their steps."""
    space = ContinuousSpace(extent=(10.0, 10.0), regions={'west': box(0, 0, 2, 10), 'east': box(8, 0, 10, 10)})
    residents = Element('residents', values={'steps': 0.0, 'region': region}, size=size, space=space)
    settle = Interaction('settle', roles={'target': 'residents'}, laws=lambda s: [UpdateLaw(s.target, _settle)])
    return System(elements=[residents], interactions=[settle])


def test_members_start_at_points_drawn_inside_the_regions_their_value_names():
    """Test that every resident starts inside its region, and that a seed draws the same points again."""
    names = ['west'] * 10 + ['east'] * 20
    system = _residents(names).prepare()
    first, second = (system.simulate(until=1, random_state=0).get_agents('residents', 0) for _ in range(2))
    x = first.locations[:, 0]
    assert (x[:10] <= 2.0).all()
    assert (x[10:] >= 8.0).all()
    np.testing.assert_array_equal(first.locations, second.locations)
    other = system.simulate(until=1, random_state=1).get_agents('residents', 0).locations
    assert not np.array_equal(first.locations, other)


def test_members_share_one_region_their_value_names():
    """Test that one name of a region places every resident inside it."""
    locations = _residents('east').prepare().simulate(until=1, random_state=0).get_agents('residents', 0).locations
    assert (locations[:, 0] >= 8.0).all()


def test_a_region_the_space_does_not_name_is_refused():
    """Test that a resident who starts in an unknown region raises an error that lists the regions of the space."""
    with pytest.raises(ValidationError, match=r'regions are: west, east'):
        _residents('north').prepare()


@pytest.mark.parametrize(
    ('history', 'expected'),
    [(None, 0.01), (0.0, 0.01 * np.exp(-1.0))],
)
def test_the_past_value_of_a_delayed_value_is_its_history_before_the_start(history, expected):
    """Test that `'I[-1]'` gives a delayed value before the start, which is its initial value without one.

    With `I' = -I + I(t - 1)`, a history of zero gives `I = I_0 exp(-t)` up to `t = 1`, and the initial value as
    history keeps `I = I_0`.
    """
    values = {'I': 0.01} if history is None else {'I': 0.01, 'I[-1]': history}
    spread = Interaction(
        'spread',
        roles={'target': 'flu'},
        laws=lambda s: [RateLaw(s.target.I.diff(s.t), -s.target.I + s.target.I(t=s.t - 1.0))],
    )
    system = System(elements=[Element('flu', values=values)], interactions=[spread], solver=RK4Solver(dt=0.01))
    assert system.prepare().simulate(until=1.0).get_values('flu.I')[-1] == pytest.approx(expected, rel=1e-6)


def test_the_history_of_a_delayed_value_may_be_a_function_of_time():
    """Test that a history given as a function of time is read before the start."""
    values = {'I': 0.01, 'I[-1]': lambda t: 0.01 * (1 + t)}
    spread = Interaction(
        'spread',
        roles={'target': 'flu'},
        laws=lambda s: [RateLaw(s.target.I.diff(s.t), s.target.I(t=s.t - 1.0))],
    )
    system = System(elements=[Element('flu', values=values)], interactions=[spread], solver=RK4Solver(dt=0.01))
    assert system.prepare().simulate(until=1.0).get_values('flu.I')[-1] == pytest.approx(0.015, rel=1e-6)


def test_a_past_value_of_a_value_read_without_delay_is_refused():
    """Test that `'I[-1]'` of a value the laws read with no delay or lag is refused."""
    decay = Interaction('decay', roles={'target': 'flu'}, laws=lambda s: [RateLaw(s.target.I.diff(s.t), -s.target.I)])
    system = System(elements=[Element('flu', values={'I': 0.01, 'I[-1]': 0.0})], interactions=[decay])
    with pytest.raises(ValidationError, match='at index 1'):
        system.prepare()


def _fly(agent, system):
    """Slow a bird down and move it by its velocity."""
    agent.set_value("x'", 0.5 * agent.get_value("x'"))
    agent.x += agent.get_value("x'")
    agent.y += agent.get_value("y'")


def test_agents_change_their_velocity_named_with_a_prime():
    """Test that agents given `"x'"` and `"y'"` hold them as velocities, which their rules change and analyses read."""
    birds = Element(
        'birds',
        values={'x': [1.0, 2.0], 'y': [1.0, 2.0], "x'": [1.0, 1.0], "y'": [0.0, 0.0]},
        size=2,
        space=ContinuousSpace((10.0, 10.0), boundary='periodic'),
    )
    flight = Interaction('flight', roles={'target': 'birds'}, laws=lambda s: [UpdateLaw(s.target, _fly)])
    system = System(elements=[birds], interactions=[flight]).prepare()
    assert system.states_ == ('birds.x', 'birds.y', "birds.x'", "birds.y'")
    trajectory = system.simulate(until=3)
    np.testing.assert_allclose(trajectory.agents_by_frame[-1]['birds'].locations[:, 0], [1.875, 2.875])
    np.testing.assert_allclose(alignment(trajectory).vectors[-1], [[0.125, 0.0], [0.125, 0.0]])


def test_an_element_on_the_edges_of_a_layer_holds_one_member_per_edge_of_that_layer():
    """Test that `on=` with the name of a layer puts one member on every edge of that layer, read along it."""
    immune = nx.DiGraph([(0, 2), (2, 0)])
    immune.add_nodes_from(range(3))
    space = NetworkSpace(nx.path_graph(3, create_using=nx.DiGraph), layers=[LayerSpec('immune', immune)])
    cells = Element('cells', values={'x': np.array([1.0, 0.0, 0.0])}, space=space)
    links = Element('links', values={'w': np.array([2.0, 3.0])}, space=space, on='immune')
    flow = Interaction(
        'flow',
        among='cells',
        along='immune',
        laws=lambda s: [RateLaw(s.target.x.diff(s.t), s.edge.w * s.source.x)],
    )
    system = System(elements=[cells, links], interactions=[flow], solver=RK4Solver(dt=0.001)).prepare()
    assert 'links.w[i]' in system.parameters_
    last = system.simulate(until=0.1).get_values('cells.x')[-1]
    exact = expm(np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]) * 0.1) @ [1.0, 0.0, 0.0]
    np.testing.assert_allclose(last, exact, rtol=1e-6)


class _ProbeElement(Element):
    """Element whose position is a vector with a default of its class."""

    def __init__(self, label, x=(0.0, 0.0), values=None, size=None, space=None):
        super().__init__(label, values=values, size=size, space=space)
        self.x = x


def test_a_single_element_keeps_its_vector_value_over_the_default_of_its_class():
    """Test that one element with a vector value starts at that value, not at the default of its class."""
    drift = Interaction('drift', roles={'target': 'probe'}, laws=lambda s: [RateLaw(s.target.x.diff(s.t), [1.0, 1.0])])
    system = System(elements=[_ProbeElement('probe', x=[3.0, 4.0])], interactions=[drift], solver=RK4Solver(dt=0.5))
    assert system.prepare().simulate(until=1.0).get_values('probe.x')[0].tolist() == [3.0, 4.0]


def _mark(agent, system):
    """Deposit one unit into the field at the cell of an agent."""
    system.deposit('ground.level', agent.location_key, 1.0)


def test_a_field_bound_to_rules_keeps_as_states_only_the_values_its_laws_change():
    """Test that a field that a rule deposits into keeps the values no law changes as parameters."""
    grid = LatticeSpace((3, 3))
    ants = Element('ants', values={'x': [0, 2], 'y': [1, 1]}, size=2, space=grid)
    ground = Element('ground', values={'level': 0.0, 'rate': 0.5}, space=LatticeSpace((3, 3)))
    decay = Interaction(
        'decay',
        roles={'target': 'ground'},
        laws=lambda s: [RateLaw(s.target.level, -s.target.rate * s.target.level)],
    )
    marking = Interaction(
        'marking',
        roles={'target': 'colony/ants', 'ground': 'field/ground'},
        laws=lambda s: [UpdateLaw(s.target, _mark)],
    )
    system = System(
        systems=[
            System('colony', elements=[ants], space=grid),
            System('field', elements=[ground], interactions=[decay]),
        ],
        interactions=[marking],
    )
    trajectory = system.prepare().simulate(until=2, random_state=0)
    assert [name for name in trajectory.problem.states.names if name.startswith('field/')] == ['field/ground.level']
    assert 'field/ground.rate' in trajectory.problem.parameters.names
    assert float(np.sum(trajectory.get_values('field/ground.level')[-1])) > 0


def _tend(system: object) -> None:
    """Leave the system as it is."""
    del system


def test_get_declared_keeps_a_value_a_rate_changes_beside_a_rule_of_the_same_interaction():
    """Test that soil a rate grows stays a state of its nested system when the same interaction runs a system rule."""
    tending = Interaction(
        'tend',
        roles={'soil': 'land/soil', 'farmers': 'village/farmers'},
        laws=lambda s: [RateLaw(s.soil.x, 1.0), UpdateLaw(s, _tend)],
    )
    system = System(
        systems=[
            System('land', elements=[Element('soil', values={'x': 0.0})]),
            System('village', elements=[Element('farmers', values={'wealth': 1.0}, size=2)]),
        ],
        interactions=[tending],
    ).prepare()
    assert system.states_ == ('land/soil.x', 'village/farmers.wealth')


def test_rate_laws_of_two_orders_on_one_state_are_rejected():
    """Test that a rate of `x` beside a rate of `x''` is rejected with the state, both orders, and the fix."""
    drift = Interaction('drift', roles={'target': 'ball'}, laws=lambda s: [RateLaw(s.target.x, 1.0)])
    gravity = Interaction('gravity', roles={'target': 'ball'}, laws=lambda s: [RateLaw(s.target.x.diff(s.t, 2), -9.81)])
    system = System(elements=[Element('ball', values={'x': 10.0})], interactions=[drift, gravity])
    with pytest.raises(ValidationError, match=r'ball\.x has rate laws of order 1 and of order 2') as error:
        system.prepare()
    assert 'Write each law on the same derivative' in str(error.value)


def test_rate_laws_of_one_order_on_one_state_add_up():
    """Test that two rates of the same derivative of a state add up."""
    gravity = Interaction('gravity', roles={'target': 'ball'}, laws=lambda s: [RateLaw(s.target.x.diff(s.t, 2), -9.81)])
    lift = Interaction('lift', roles={'target': 'ball'}, laws=lambda s: [RateLaw(s.target.x.diff(s.t, 2), 9.81)])
    system = System(elements=[Element('ball', values={'x': 10.0})], interactions=[gravity, lift]).prepare()
    assert system.simulate(until=1.0, dt=1.0).get_values('ball.x').tolist() == pytest.approx([10.0, 10.0])


def test_get_declared_reads_an_array_of_tuples_as_a_state_of_objects():
    """Test that queues of orders given as an array of tuples stay one tuple per agent in every snapshot."""
    orders = np.empty(2, dtype=object)
    orders[:] = [(), (7,)]
    trajectory = _queue_run()
    first, last = trajectory.get_agents('queues', 0), trajectory.get_agents('queues', 3)
    assert first.values['orders'].dtype == object
    assert first.values['orders'].tolist() == orders.tolist()
    assert last.values['orders'].tolist() == [(2, 3), (2, 3)]
    assert isinstance(trajectory.get_values('queue.longest')[-1], tuple)


def _spend(agent, system):
    """Spend one unit of money."""
    agent.money -= 1


def test_agents_remember_as_many_values_as_the_oldest_past_value_given():
    """Test that an agent value given with the past value of index `-3` keeps its last 3 values."""
    spending = Interaction('spend', roles={'target': 'savers'}, laws=lambda s: [UpdateLaw(s.target, _spend)])
    savers = Element('savers', values={'money': 5.0, 'money[-3]': 5.0}, size=2)
    system = System(elements=[savers], interactions=[spending]).prepare()
    assert system.states_ == ('savers.money', 'savers.money[-1]', 'savers.money[-2]', 'savers.money[-3]')


def test_get_declared_keeps_the_failures_of_the_probe_by_component():
    """Test that the failure of a component while every value is a state is kept with the path of the component."""
    space = NetworkSpace(nx.complete_graph(4), hyperedges=[(0, 1, 2), (1, 2, 3)])
    rule = Interaction(
        'spread',
        among='nodes',
        along='hyperedges',
        laws=lambda s: [RateLaw(s.member.x, variance(s.members.x, over=s.members))],
    )
    system = System(elements=[Element('nodes', values={'x': np.zeros(4)}, space=space)], interactions=[rule])
    compiler = PassCompiler()
    compiler._get_declared_system(system._get_composite())
    assert list(compiler.probed_) == ['spread']


def _nodes_on_a_network(laws):
    """Return the nodes and links of a network that a system holds, with the laws of the network."""
    space = NetworkSpace(nx.complete_graph(2))
    return System(
        elements=[Element('nodes', values={'age': 0.0}), Element('links', on='edges')],
        interactions=[Interaction('network', laws=laws)],
        space=space,
    )


def test_get_declared_puts_an_element_without_space_on_the_growing_network_of_its_system():
    """Test that an element without a space holds one member per node of a network that the laws grow."""
    system = _nodes_on_a_network([UpdateLaw(TopologyRequestSpec('add-node', edges=1), 1)])
    declared = PassCompiler()._get_declared_system(system._get_composite())
    nodes = next(element for element in declared.elements if element.label == 'nodes')
    assert nodes.space is system.space


def test_get_declared_keeps_an_element_without_space_single_beside_a_fixed_network():
    """Test that an element without a space stays one element when no law changes the network of its system."""
    declared = PassCompiler()._get_declared_system(_nodes_on_a_network([])._get_composite())
    nodes = next(element for element in declared.elements if element.label == 'nodes')
    assert nodes.space is None
