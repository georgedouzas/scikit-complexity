"""Test observables: expressions and reductions computed at every frame, recorded, logged, and read by laws."""

import numpy as np
import pytest
from sage.all__sagemath_symbolics import I, exp

from skcomplex.core.base import Domain, ValidationError, parameter, state
from skcomplex.core.model import AgentType, Element, Interaction, Observable, Rate, Rule, System
from skcomplex.core.runtime import Frame, Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models.ecology import ExponentialGrowth, Population

Oscillator = Element.define(
    'Oscillator',
    states={'theta': state(0.0, domain=Domain.ANGLE)},
    parameters={'omega': parameter(1.0)},
    laws=lambda s: [Rate(s.theta, s.omega)],
)


class Saver(AgentType):
    """Agent whose wealth grows by one every step."""

    wealth: float = state(1.0)

    def laws(self, s):
        """Return the saving rule."""
        return [Rule(self.save)]

    def save(self, agent, model):
        """Add one to the wealth."""
        del model
        agent.wealth += 1


Inflow = Interaction.define('Inflow', roles=['target', 'source'], laws=lambda s: [Rate(s.target.n, s.source.value)])


class LoggedPaths:
    """Recording stand-in that keeps the entity paths it receives."""

    def __init__(self):
        """Start with no logs."""
        self.paths = []

    def log(self, path, archetype):
        """Keep a logged entity path."""
        del archetype
        self.paths.append(path)

    def set_time(self, *args, **kwargs):
        """Ignore the time."""


def _oscillators():
    """Return oscillators with an order parameter built from a complex mean."""
    return System(
        elements=[Oscillator.group('osc', 50, theta=np.linspace(0, 1, 50), omega=np.linspace(0.5, 1.5, 50))],
        observables=[
            Observable('z', lambda m: exp(I * m.theta), group='osc'),
            Observable('order', lambda s: abs(s.z)),
        ],
    )


def test_order_parameter_every_frame():
    """Test that the modulus of the mean of exp(iθ) is recorded at every frame between 0 and 1."""
    trajectory = Simulation(_oscillators(), dt=0.5).run(until=10).trajectory_
    order = trajectory['order']
    assert order.shape == (21,)
    assert np.all((order >= 0) & (order <= 1))
    expected = np.abs(np.exp(1j * np.asarray(trajectory['osc.theta'])).mean(axis=1))
    assert np.allclose(order, expected)


def test_expression_observable_of_elements():
    """Test that a system observable combines symbols of single elements and earlier observables."""
    system = System(
        elements=[Population('a', n=2.0), Population('b', n=6.0)],
        interactions=[ExponentialGrowth('g', target='a', r=1.0)],
        observables=[Observable('total', lambda s: s.a.n + s.b.n), Observable('share', lambda s: s.a.n / s.total)],
    )
    trajectory = Simulation(system, dt=0.5).run(until=1).trajectory_
    assert np.allclose(trajectory['total'], trajectory['a.n'] + trajectory['b.n'])
    assert np.allclose(trajectory['share'], trajectory['a.n'] / trajectory['total'])


def test_observable_by_subsystem_path():
    """Test that an observable of a subsystem is recorded under the subsystem's path."""
    system = System(
        systems=[
            System(
                'market',
                elements=[Population('a', n=3.0)],
                observables=[Observable('volume', lambda s: 2 * s.a.n)],
            ),
        ],
    )
    assert Simulation(system, dt=1).run(until=2).trajectory_['market/volume'].tolist() == [6.0, 6.0, 6.0]


def test_agent_reduction_read_by_other_partition():
    """Test that the mean wealth of agents is recorded and read by an ODE in another partition."""
    system = System(
        systems=[
            System(
                'village',
                elements=[Saver.group('savers', 4)],
                observables=[Observable('mean_wealth', group='savers', variable='wealth')],
            ),
            System('land', elements=[Population('soil', n=0.0)]),
        ],
        interactions=[Inflow('inflow', target='land/soil', source='village/mean_wealth')],
    )
    trajectory = Simulation(system, dt=1).run(until=2).trajectory_
    assert trajectory['village/mean_wealth'].tolist() == [1.0, 2.0, 3.0]
    assert trajectory['land/soil.n'][-1] == pytest.approx(2.0 + 3.0)


def test_observable_read_in_one_form_is_rejected():
    """Test that laws reading an observable of a system with one form raise an error naming the observable."""
    system = System(
        elements=[Population.group('herd', 3), Population('soil')],
        interactions=[Inflow('inflow', target='soil', source='mean_n')],
        observables=[Observable('mean_n', group='herd', variable='n')],
    )
    with pytest.raises(ValidationError, match=r'mean_n .*one form'):
        Simulation(system).run(until=1)


def test_unknown_symbol_is_rejected():
    """Test that an observable using a group member's symbol outside a reduction is rejected."""
    system = System(
        elements=[Population.group('herd', 3)],
        observables=[Observable('bad', lambda s: s.herd.n)],
    )
    with pytest.raises(AttributeError, match='herd'):
        Simulation(system).run(until=1)


def test_observables_logged_to_rerun(tmp_path):
    """Test that observables are logged as time series under the path of their system."""
    observer = RerunObserver('test', save_path=tmp_path / 'run.rrd')
    simulation = Simulation(_oscillators(), dt=0.5, observers=[observer]).run(until=1)
    observer.recording_ = LoggedPaths()
    frame = Frame(0, 0.0, simulation.state_, simulation.parameters_, simulation.problem_, observables={'order': 0.5})
    observer.on_frame(frame)
    assert 'order' in observer.recording_.paths
