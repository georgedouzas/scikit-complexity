"""Test observables: expressions and reductions computed at every frame, recorded, logged, and read by laws."""

import pickle  # nosec B403: the tests pickle and load only objects they create

import numpy as np
import pytest
from sage.all__sagemath_symbolics import I, exp

from skcomplex.core.base import Domain, RoleError, ValidationError, parameter, state, symbol
from skcomplex.core.model import AgentType, Element, Interaction, Observable, Rate, Rule, System, mean, total
from skcomplex.core.runtime import Frame, Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models.biology.ecology import ExponentialGrowth, Population

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


def _phase_vector(member):
    """Return the unit complex number of a member's phase."""
    return exp(I * member.theta)


def _modulus(s):
    """Return the modulus of the complex mean field."""
    return abs(s.z)


def _total(s):
    """Return the sum of the two populations."""
    return s.a.n + s.b.n


def _share(s):
    """Return the share of the first population in the total."""
    return s.a.n / s.total


def _volume(s):
    """Return twice the population of the market."""
    return 2 * s.a.n


def _herd(s):
    """Return the symbol of a group member, outside a reduction."""
    return s.herd.n


def _oscillators():
    """Return oscillators with an order parameter built from a complex mean."""
    return System(
        elements=[Oscillator.group('osc', 50, theta=np.linspace(0, 1, 50), omega=np.linspace(0.5, 1.5, 50))],
        observables=[
            Observable('z', _phase_vector, group='osc'),
            Observable('order', _modulus),
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
        observables=[Observable('total', _total), Observable('share', _share)],
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
                observables=[Observable('volume', _volume)],
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


def test_observable_read_in_one_form_is_lowered():
    """Test that a law reading a symbolic observable of a system with one form reads it at the current state."""
    system = System(
        elements=[Population.group('herd', 3, n=[1.0, 2.0, 6.0]), Population('soil')],
        interactions=[Inflow('inflow', target='soil', source='mean_n')],
        observables=[Observable('mean_n', group='herd', variable='n')],
    )
    trajectory = Simulation(system, dt=1.0).run(until=1).trajectory_
    assert trajectory['soil.n'][-1] == pytest.approx(trajectory['soil.n'][0] + 3.0)


def test_unknown_symbol_is_rejected():
    """Test that an observable using a group member's symbol outside a reduction is rejected."""
    system = System(
        elements=[Population.group('herd', 3)],
        observables=[Observable('bad', _herd)],
    )
    with pytest.raises(AttributeError, match='herd'):
        Simulation(system).run(until=1)


def test_observable_lambda_is_rejected():
    """Test that an observable whose expression is a lambda raises a named error that names it at once."""
    with pytest.raises(ValidationError, match="'total'"):
        Observable('total', lambda s: s.a.n + s.b.n)


def test_observable_lambda_reduction_is_rejected():
    """Test that an observable whose reduction is a lambda raises a named error."""
    with pytest.raises(ValidationError, match='reduction'):
        Observable('spread', group='herd', variable='n', reduce=lambda values: values.ptp())


def test_observable_module_function_pickles():
    """Test that a system with a module-level observable pickles and records the same values once restored."""
    system = System(
        elements=[Population('a', n=2.0), Population('b', n=6.0)],
        interactions=[ExponentialGrowth('g', target='a', r=1.0)],
        observables=[Observable('total', _total)],
    )
    restored = pickle.loads(pickle.dumps(system))  # noqa: S301  # nosec B301: loads only what it dumped
    expected = Simulation(system, dt=0.5).run(until=1).trajectory_['total']
    assert np.allclose(Simulation(restored, dt=0.5).run(until=1).trajectory_['total'], expected)


def test_observables_logged_to_rerun(tmp_path):
    """Test that observables are logged as time series under the path of their system."""
    observer = RerunObserver('test', save_path=tmp_path / 'run.rrd')
    simulation = Simulation(_oscillators(), dt=0.5, observers=[observer]).run(until=1)
    observer.recording_ = LoggedPaths()
    frame = Frame(0, 0.0, simulation.state_, simulation.parameters_, simulation.problem_, observables={'order': 0.5})
    observer.on_frame(frame)
    assert 'order' in observer.recording_.paths


def _timed(s):
    """Return the population times the time."""
    return s.a.n * s.t


def _member_timed(member):
    """Return a member's population plus the time."""
    return member.n + member.t


def test_observables_read_time():
    """Test that system observables and member expressions of reductions read the time as `s.t`."""
    system = System(
        elements=[Population('a', n=2.0), Population.group('herd', 2, n=[1.0, 3.0])],
        observables=[Observable('timed', _timed), Observable('member_timed', _member_timed, group='herd')],
    )
    trajectory = Simulation(system, dt=1.0).run(until=2).trajectory_
    assert trajectory['timed'].tolist() == [0.0, 2.0, 4.0]
    assert trajectory['member_timed'].tolist() == [2.0, 3.0, 4.0]


FIELD_WEIGHTS = np.ones((4, 4)) - np.eye(4)
Decaying = Element.define('Decaying', states={'x': 1.0}, laws=lambda s: [Rate(s.x, -s.x)])


def _local_field_energy(member):
    """Return a member's state times the weighted total of the states of the others."""
    return member.x * total(member.x, weights=FIELD_WEIGHTS)


def _deviation(member):
    """Return a member's deviation from the group mean."""
    return member.x - mean(member.x)


def _deviation_from_other(member):
    """Return a member's state minus the mean of a symbol that is not one of its group."""
    return member.x - mean(member.x, over=symbol('elsewhere'))


def _decaying_group(*observables):
    """Return a decaying group of 4 members with observables."""
    return System(elements=[Decaying.group('u', 4, x=np.arange(1.0, 5.0))], observables=list(observables))


def test_observable_reads_weighted_reduction_of_its_members():
    """Test that a member expression with a weighted total gives the local field energy at every frame."""
    system = _decaying_group(Observable('energy', _local_field_energy, group='u', reduce='sum'))
    trajectory = Simulation(system, dt=0.5).run(until=2).trajectory_
    x = np.asarray(trajectory['u.x'])
    np.testing.assert_allclose(trajectory['energy'], np.einsum('fi,ij,fj->f', x, FIELD_WEIGHTS, x), rtol=1e-12)


def test_observable_reads_plain_reduction_of_its_members():
    """Test that a member expression with a plain mean gives the variance of the group at every frame."""
    system = _decaying_group(Observable('spread', _deviation, group='u', reduce='var'))
    trajectory = Simulation(system, dt=0.5).run(until=2).trajectory_
    np.testing.assert_allclose(trajectory['spread'], np.asarray(trajectory['u.x']).var(axis=1), rtol=1e-12)


def test_observable_rejects_reduction_over_another_group():
    """Test that a reduction over a group other than the observable's raises the role error of laws."""
    system = _decaying_group(Observable('other', _deviation_from_other, group='u', reduce='sum'))
    with pytest.raises(RoleError, match=r'observable other reads a group other than u'):
        Simulation(system, dt=0.5).run(until=1)
