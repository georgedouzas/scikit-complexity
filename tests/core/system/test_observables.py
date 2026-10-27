"""Test observables: expressions and reductions computed at every frame, recorded, logged, and read by laws."""

import dataclasses
import pickle  # nosec B403: the tests pickle and load only objects they create
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import I, exp, pi

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.errors import RoleError, ValidationError
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import RateLaw, RuleLaw
from skcomplex.core.laws.utils import distributed_delay, mean, total
from skcomplex.core.specs import ObservableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec
from skcomplex.core.system import System
from skcomplex.engine.problems import symbol
from skcomplex.engine.runtime import Frame, Simulation
from skcomplex.engine.viz import RerunObserver
from skcomplex.models.biology.ecology import ExponentialGrowthInteraction, PopulationElement


class OscillatorElement(Element):
    """Oscillator element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        theta: Annotated[object, StateSpec(period=2 * pi)] = 0.0,
        omega: Annotated[object, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.theta = theta
        self.omega = omega

    def _get_laws(self, s):
        """Return the laws of the oscillator element."""
        return [RateLaw(s.theta, s.omega)]


class SaverElement(Element):
    """Agent whose wealth grows by one every step."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        wealth: Annotated[float, StateSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.wealth = wealth

    def _get_laws(self, s):
        """Return the saving rule."""
        return [RuleLaw(self.save)]

    def save(self, agent, model):
        """Add one to the wealth."""
        del model
        agent.wealth += 1


class InflowInteraction(Interaction):
    """Inflow interaction."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[object, RoleSpec()],
        source: Annotated[object, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.target = target
        self.source = source

    def _get_laws(self, s):
        """Return the laws of the inflow interaction."""
        return [RateLaw(s.target.n, s.source.value)]


class LoggedPaths:
    """Recording stand-in that keeps the entity paths it receives."""

    def __init__(self):
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


ORDER = [ObservableSpec('z', _phase_vector, group='osc'), ObservableSpec('order', _modulus)]


def _oscillators():
    """Return oscillators whose order parameter `ORDER` builds from a complex mean."""
    return System(
        elements=[
            GroupElement(
                'osc',
                element=OscillatorElement,
                size=50,
                values={'theta': np.linspace(0, 1, 50), 'omega': np.linspace(0.5, 1.5, 50)},
            ),
        ],
    )


def test_order_parameter_every_frame():
    """Test that the modulus of the mean of exp(iθ) is recorded at every frame between 0 and 1."""
    trajectory = Simulation(_oscillators(), dt=0.5, observables=ORDER).run(until=10).trajectory_
    order = trajectory.values('order')
    assert order.shape == (21,)
    assert np.all((order >= 0) & (order <= 1))
    expected = np.abs(np.exp(1j * np.asarray(trajectory.values('osc.theta'))).mean(axis=1))
    assert np.allclose(order, expected)


def test_expression_observable_of_elements():
    """Test that a system observable combines symbols of single elements and earlier observables."""
    system = System(
        elements=[PopulationElement('a', n=2.0), PopulationElement('b', n=6.0)],
        interactions=[ExponentialGrowthInteraction('g', target='a', r=1.0)],
    )
    observables = [ObservableSpec('total', _total), ObservableSpec('share', _share)]
    trajectory = Simulation(system, dt=0.5, observables=observables).run(until=1).trajectory_
    assert np.allclose(trajectory.values('total'), trajectory.values('a.n') + trajectory.values('b.n'))
    assert np.allclose(trajectory.values('share'), trajectory.values('a.n') / trajectory.values('total'))


def test_observable_by_subsystem_path():
    """Test that an observable of a subsystem is recorded under the subsystem's path."""
    system = System(
        systems=[System('market', elements=[PopulationElement('a', n=3.0)])],
    )
    simulation = Simulation(system, dt=1, observables=[ObservableSpec('market/volume', _volume)])
    assert simulation.run(until=2).trajectory_.values('market/volume').tolist() == [6.0, 6.0, 6.0]


def test_agent_reduction_read_by_other_partition():
    """Test that the mean wealth of agents is recorded and read by an ODE in another partition."""
    system = System(
        systems=[
            System('village', elements=[GroupElement('savers', element=SaverElement, size=4)]),
            System('land', elements=[PopulationElement('soil', n=0.0)]),
        ],
        interactions=[InflowInteraction('inflow', target='land/soil', source='village/mean_wealth')],
    )
    observables = [ObservableSpec('village/mean_wealth', group='savers', variable='wealth')]
    trajectory = Simulation(system, dt=1, observables=observables).run(until=2).trajectory_
    assert trajectory.values('village/mean_wealth').tolist() == [1.0, 2.0, 3.0]
    assert trajectory.values('land/soil.n')[-1] == pytest.approx(2.0 + 3.0)


def test_observable_read_in_one_form_is_lowered():
    """Test that a law reading a symbolic observable of a system with one form reads it at the current state."""
    system = System(
        elements=[
            GroupElement('herd', element=PopulationElement, size=3, values={'n': [1.0, 2.0, 6.0]}),
            PopulationElement('soil'),
        ],
        interactions=[InflowInteraction('inflow', target='soil', source='mean_n')],
    )
    observables = [ObservableSpec('mean_n', group='herd', variable='n')]
    trajectory = Simulation(system, dt=1.0, observables=observables).run(until=1).trajectory_
    assert trajectory.values('soil.n')[-1] == pytest.approx(trajectory.values('soil.n')[0] + 3.0)


def test_unknown_symbol_is_rejected():
    """Test that an observable using a group member's symbol outside a reduction is rejected."""
    system = System(elements=[GroupElement('herd', element=PopulationElement, size=3)])
    with pytest.raises(AttributeError, match='herd'):
        Simulation(system, observables=[ObservableSpec('bad', _herd)]).run(until=1)


def test_observable_lambda_is_rejected():
    """Test that checking an observable whose expression is a lambda raises a named error that names it."""
    with pytest.raises(ValidationError, match="'total'"):
        ObservableSpec('total', lambda s: s.a.n + s.b.n).check()


def test_observable_lambda_reduction_is_rejected():
    """Test that an observable whose reduction is a lambda raises a named error."""
    with pytest.raises(ValidationError, match='reduction'):
        ObservableSpec('spread', group='herd', variable='n', reduce=lambda values: values.ptp()).check()


def test_observable_module_function_pickles():
    """Test that a module-level observable pickles and records the same values once restored."""
    system = System(
        elements=[PopulationElement('a', n=2.0), PopulationElement('b', n=6.0)],
        interactions=[ExponentialGrowthInteraction('g', target='a', r=1.0)],
    )
    observables = [ObservableSpec('total', _total)]
    restored = pickle.loads(pickle.dumps(observables))  # noqa: S301  # nosec B301: loads only what it dumped
    expected = Simulation(system, dt=0.5, observables=observables).run(until=1).trajectory_.values('total')
    assert np.allclose(
        Simulation(system, dt=0.5, observables=restored).run(until=1).trajectory_.values('total'),
        expected,
    )


def test_observables_logged_to_rerun(tmp_path):
    """Test that observables are logged as time series under the path of their system."""
    observer = RerunObserver('test', save_path=tmp_path / 'run.rrd')
    simulation = Simulation(_oscillators(), dt=0.5, observers=[observer], observables=ORDER).run(until=1)
    observer.recording_ = LoggedPaths()
    frame = Frame(0, 0.0, simulation.state_, simulation.parameters_, simulation.problem_, observables={'order': 0.5})
    observer.run(frame)
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
        elements=[
            PopulationElement('a', n=2.0),
            GroupElement('herd', element=PopulationElement, size=2, values={'n': [1.0, 3.0]}),
        ],
    )
    observables = [ObservableSpec('timed', _timed), ObservableSpec('member_timed', _member_timed, group='herd')]
    trajectory = Simulation(system, dt=1.0, observables=observables).run(until=2).trajectory_
    assert trajectory.values('timed').tolist() == [0.0, 2.0, 4.0]
    assert trajectory.values('member_timed').tolist() == [2.0, 3.0, 4.0]


FIELD_WEIGHTS = np.ones((4, 4)) - np.eye(4)


class DecayingElement(Element):
    """Decaying element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x

    def _get_laws(self, s):
        """Return the laws of the decaying element."""
        return [RateLaw(s.x, -s.x)]


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
    """Return a simulation of a decaying group of 4 members with observables."""
    system = System(elements=[GroupElement('u', element=DecayingElement, size=4, values={'x': np.arange(1.0, 5.0)})])
    return Simulation(system, dt=0.5, observables=list(observables))


def test_observable_reads_weighted_reduction_of_its_members():
    """Test that a member expression with a weighted total gives the local field energy at every frame."""
    simulation = _decaying_group(ObservableSpec('energy', _local_field_energy, group='u', reduce='sum'))
    trajectory = simulation.run(until=2).trajectory_
    x = np.asarray(trajectory.values('u.x'))
    np.testing.assert_allclose(trajectory.values('energy'), np.einsum('fi,ij,fj->f', x, FIELD_WEIGHTS, x), rtol=1e-12)


def test_observable_reads_plain_reduction_of_its_members():
    """Test that a member expression with a plain mean gives the variance of the group at every frame."""
    simulation = _decaying_group(ObservableSpec('spread', _deviation, group='u', reduce='var'))
    trajectory = simulation.run(until=2).trajectory_
    np.testing.assert_allclose(
        trajectory.values('spread'),
        np.asarray(trajectory.values('u.x')).var(axis=1),
        rtol=1e-12,
    )


def test_observable_rejects_reduction_over_another_group():
    """Test that a reduction over a group other than the observable's raises the role error of laws."""
    simulation = _decaying_group(ObservableSpec('other', _deviation_from_other, group='u', reduce='sum'))
    with pytest.raises(RoleError, match=r'observable other reads a group other than u'):
        simulation.run(until=1)


def test_simulation_checks_its_observables():
    """Test that a run rejects an observable whose expression is a lambda before compiling."""
    system = System(elements=[PopulationElement('a')])
    with pytest.raises(ValidationError, match="'total'"):
        Simulation(system, observables=[ObservableSpec('total', lambda s: s.a.n)]).run(until=1)


def test_observable_is_a_frozen_record_of_no_base():
    """Test that an observable is a frozen dataclass of a label and an expression that derives from no base."""
    assert Observable.__bases__ == (object,)
    assert Observable.__dataclass_params__.frozen
    assert [item.name for item in dataclasses.fields(ObservableSpec)][:2] == ['label', 'expression']
    level = ObservableSpec('level', group='herd', variable='n', reduce='sum')
    assert level.check() == level


def _perceived(s):
    """Return the stock of an element smoothed over its perception time."""
    return distributed_delay(s.n, conserved=False, time=s.perception_time)


def _smoothed_population(s):
    """Return the population smoothed over one unit of time."""
    return distributed_delay(s.n, conserved=False, time=1.0)


class SmoothedElement(Element):
    """Smoothed element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n: Annotated[object, StateSpec()] = 1.0,
        perception_time: Annotated[object, ParameterSpec()] = 2.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.n = n
        self.perception_time = perception_time

    def _get_laws(self, s):
        """Return the laws of the smoothed element."""
        return [RateLaw(s.n, 2.0 - _perceived(s))]


def test_observe_element_records_an_expression_of_its_element_with_delays():
    """Test that an observable of an element's expression records it under the element and reads its delays."""
    system = System(elements=[SmoothedElement('tank')])
    observed = ObservableSpec('perceived', _perceived, element='tank')
    assert observed.label == 'tank.perceived'
    trajectory = Simulation(system, dt=0.5, observables=[observed]).run(until=4).trajectory_
    perceived = trajectory.values('tank.perceived')
    np.testing.assert_allclose(perceived, trajectory.values('tank.smoothing_delay1_0')[:, 0])
    assert perceived[0] == pytest.approx(1.0)
    assert perceived[-1] > perceived[0]


def test_observe_element_rejects_a_delay_no_law_uses():
    """Test that observing a delay that no law of the system reads raises an error that names the observable."""
    system = System(elements=[PopulationElement('rabbits', n=2.0)])
    observed = ObservableSpec('smoothed', _smoothed_population, element='rabbits')
    with pytest.raises(ValidationError, match=r'rabbits\.smoothed'):
        Simulation(system, observables=[observed]).run(until=1)
