"""Test the `_rate` module of `skcomplex.core.interactions`."""

from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.interactions import RateInteraction
from skcomplex.core.spaces import ContinuousSpace
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import EulerMaruyamaSolver, FractionalEulerMaruyamaSolver
from skcomplex.errors import ValidationError
from skcomplex.specs import ParameterSpec, StateSpec, WienerProcessSpec

from ...conftest import HoldingElement

MEMBERS = 20_000


class InertialElement(Element):
    """Element with a vector state of order 2 and a positive inertia."""

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[object, StateSpec(vector=True, order=2)] = 0.0,
        x_d1: object = 0.0,
        m: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label)
        self.x = x
        self.x_d1 = x_d1
        self.m = m


def _final(interactions, n=1.0, until=1.0, size=MEMBERS, solver=None):
    """Return the final values of the state `n` of a group of holding elements driven by interactions."""
    system = System(
        elements=[GroupElement('walkers', element=HoldingElement, size=size, values={'n': n})],
        interactions=interactions,
    )
    solver = EulerMaruyamaSolver(dt=0.002) if solver is None else solver
    simulation = Simulation(system, solver=solver, dt=until, random_state=0)
    return simulation.check().prepare().run(until=until).trajectory_.values('walkers.n')[-1]


def test_run_divides_a_constant_rate_on_a_state_of_order_two():
    """Test that a constant rate `F` with the divisor `m` on a state of order 2 gives `x = F t**2 / (2 m)`."""
    system = System(
        elements=[InertialElement('body', m=2.0)],
        interactions=[RateInteraction('push', target='body', state='x', rate=4.0, divisor='m')],
        space=ContinuousSpace(n_dim=2),
    )
    trajectory = Simulation(system, dt=0.5).check().prepare().run(until=2.0).trajectory_
    np.testing.assert_allclose(trajectory.values('body.x')[-1], [4.0, 4.0])
    np.testing.assert_allclose(trajectory.values('body.x_d1')[-1], [4.0, 4.0])


def test_run_reads_the_sources():
    """Test that a rate reading a source grows the target by the value of the source."""
    system = System(
        elements=[HoldingElement('stock', n=0.0), HoldingElement('inflow', n=3.0)],
        interactions=[
            RateInteraction('feed', target='stock', state='n', rate=lambda s: s.sources[0].n, sources=['inflow']),
        ],
    )
    trajectory = Simulation(system, dt=0.5).check().prepare().run(until=2.0).trajectory_
    assert trajectory.values('stock.n')[-1] == pytest.approx(6.0)
    assert trajectory.values('inflow.n')[-1] == pytest.approx(3.0)


def test_run_gives_the_variance_of_an_additive_noise():
    """Test that a pure noise of amplitude `sigma` gives the variance `sigma**2 t` within 5 percent."""
    final = _final([RateInteraction('walk', target='walkers', state='n', noise=0.5)], n=0.0, until=2.0)
    assert final.var() == pytest.approx(0.5**2 * 2.0, rel=0.05)


def test_run_follows_the_convention_of_a_multiplicative_noise():
    """Test that `x dW` keeps the log mean 0 in the Stratonovich convention and `-t / 2` in the Ito convention."""
    sigma, until = 0.5, 1.0
    for convention, expected in (('stratonovich', 0.0), ('ito', -(sigma**2) * until / 2)):
        noise = RateInteraction(
            'walk',
            target='walkers',
            state='n',
            noise=lambda s: sigma * s.target.n,
            convention=convention,
        )
        assert np.log(_final([noise], until=until)).mean() == pytest.approx(expected, abs=0.01)


def test_run_gives_the_variance_of_a_fractional_noise():
    """Test that a fractional Brownian motion of Hurst index `H` has the variance `t**(2 H)` within 5 percent."""
    noise = RateInteraction('walk', target='walkers', state='n', noise=1.0, hurst=0.7)
    final = _final([noise], n=0.0, until=2.0, size=5_000, solver=FractionalEulerMaruyamaSolver(dt=0.01))
    assert final.var() == pytest.approx(2.0**1.4, rel=0.05)


def test_run_shares_the_processes_of_a_wiener_spec():
    """Test that two states driven by one process move together, and two correlated processes keep their covariance."""
    common = WienerProcessSpec('common')
    pair = WienerProcessSpec('pair', dimension=2, covariance=((1.0, 0.6), (0.6, 1.0)))
    system = System(
        elements=[HoldingElement(label) for label in ('a', 'b', 'c', 'd')],
        interactions=[
            RateInteraction('first', target='a', state='n', noise=1.0, process=common),
            RateInteraction('second', target='b', state='n', noise=0.5, process=common),
            RateInteraction('third', target='c', state='n', noise=1.0, process=pair, index=0),
            RateInteraction('fourth', target='d', state='n', noise=1.0, process=pair, index=1),
        ],
    )
    simulation = Simulation(system, solver=EulerMaruyamaSolver(dt=0.01), dt=0.01, random_state=0)
    trajectory = simulation.check().prepare().run(until=50.0).trajectory_
    np.testing.assert_allclose(trajectory.values('b.n'), trajectory.values('a.n') / 2, atol=1e-12)
    increments = np.diff(np.column_stack([trajectory.values('c.n'), trajectory.values('d.n')]), axis=0)
    assert np.corrcoef(increments.T)[0, 1] == pytest.approx(0.6, abs=0.03)


@pytest.mark.parametrize(
    ('arguments', 'message'),
    [
        ({'state': '1n'}, 'not a name of a state'),
        ({'rate': 'fast'}, 'function of the namespace or a number'),
        ({'divisor': 'm-1'}, 'name of a state or parameter'),
        ({'convention': 'ito-stratonovich'}, 'Unknown noise convention'),
        ({'hurst': 1.5}, 'Hurst index'),
        ({'index': 0}, 'binds no processes'),
        ({'hurst': 0.3, 'conserved': True}, 'at most one'),
    ],
)
def test_check_rejects_invalid_arguments(arguments, message):
    """Test that `check` rejects a bad state, function, divisor, convention, Hurst index, or noise combination."""
    with pytest.raises(ValidationError, match=message):
        RateInteraction('rate', target='walkers', **arguments).check()


def test_run_divides_by_a_divisor_expression():
    """Test that a constant rate `F` divided by the expression `2 m` gives `x = F t**2 / (4 m)`."""
    push = RateInteraction('push', target='body', state='x', rate=4.0, divisor=lambda member: 2 * member.m)
    system = System(elements=[InertialElement('body', m=2.0)], interactions=[push], space=ContinuousSpace(n_dim=2))
    trajectory = Simulation(system, dt=0.5).check().prepare().run(until=2.0).trajectory_
    np.testing.assert_allclose(trajectory.values('body.x')[-1], [2.0, 2.0])


class FractionalElement(Element):
    """Element with a state and a Hurst index as a parameter."""

    def __init__(
        self,
        label: str,
        *,
        n: Annotated[object, StateSpec()] = 0.0,
        H: Annotated[float, ParameterSpec()] = 0.5,
    ) -> None:
        super().__init__(label)
        self.n = n
        self.H = H


def test_run_reads_the_hurst_index_from_a_parameter():
    """Test that a Hurst index read from a parameter gives the variance `t**(2 H)` within 5 percent."""
    noise = RateInteraction('walk', target='walkers', state='n', noise=1.0, hurst=lambda s: s.target.H)
    system = System(
        elements=[GroupElement('walkers', element=FractionalElement, size=5_000, values={'n': 0.0, 'H': 0.7})],
        interactions=[noise],
    )
    solver = FractionalEulerMaruyamaSolver(dt=0.01)
    simulation = Simulation(system, solver=solver, dt=2.0, random_state=0).check().prepare()
    final = simulation.run(until=2.0).trajectory_.values('walkers.n')[-1]
    assert final.var() == pytest.approx(2.0**1.4, rel=0.05)


class SynapsesElement(Element):
    """Neuron with two synaptic weights."""

    def __init__(
        self,
        label: str,
        *,
        w1: Annotated[float, StateSpec()] = 0.3,
        w2: Annotated[float, StateSpec()] = -0.8,
    ) -> None:
        super().__init__(label)
        self.w1 = w1
        self.w2 = w2


def _oja(first, second):
    """Return Oja's rule of one weight, averaged over inputs of covariance [[2, 1], [1, 2]]."""

    def rate(s):
        w = (s.target.w1, s.target.w2)
        drive = (2 * w[0] + w[1], w[0] + 2 * w[1])
        variance = w[0] * drive[0] + w[1] * drive[1]
        return drive[first] - variance * w[second]

    return rate


def test_run_oja_rule_as_rate_laws():
    """Test that Oja's rule as rate laws on weight states converges to the principal direction of the inputs."""
    system = System(
        elements=[SynapsesElement('neuron')],
        interactions=[
            RateInteraction('first', target='neuron', state='w1', rate=_oja(0, 0)),
            RateInteraction('second', target='neuron', state='w2', rate=_oja(1, 1)),
        ],
    )
    trajectory = Simulation(system).check().prepare().run(until=20.0).trajectory_
    weights = np.array([trajectory.values('neuron.w1')[-1], trajectory.values('neuron.w2')[-1]])
    np.testing.assert_allclose(np.abs(weights), np.full(2, 1 / np.sqrt(2)), atol=1e-2)
