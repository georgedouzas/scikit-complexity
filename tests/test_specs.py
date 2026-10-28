"""Test the `specs` module of `skcomplex`."""

import dataclasses
import importlib
import inspect
import pickle  # nosec B403: the tests pickle and load only objects they create
import pkgutil
from typing import Annotated

import networkx as nx
import numpy as np
import pytest
from sage.all__sagemath_symbolics import I, exp, pi, var
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from skcomplex.base import BaseConfig
from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import RateLaw, RuleLaw, UpdateLaw
from skcomplex.core.laws.utils import distributed_delay, mean, total
from skcomplex.core.spaces import ContinuousTime, VonNeumannNeighborhood
from skcomplex.core.system import System
from skcomplex.engine._utils import symbol
from skcomplex.engine.runtime import Simulation
from skcomplex.errors import RewritingError, RoleError, ValidationError
from skcomplex.experiments import Experiment
from skcomplex.models.economics.resources import harvesting
from skcomplex.specs import (
    AffineSpec,
    BaseSpec,
    EnergySpec,
    GeometrySpec,
    HilbertSpec,
    LayerSpec,
    ModuleSpec,
    MoveSpec,
    ObservableSpec,
    ParameterSpec,
    PhaseSpec,
    PhasesScheduleSpec,
    PrototileSpec,
    RequirementSpec,
    RoleSpec,
    SamplerSpec,
    ScenarioSpec,
    ScheduleSpec,
    StateSpec,
    TopologyRequestSpec,
    VisualSpec,
    WienerProcessSpec,
)

HEALTH = FiniteEnumeratedSet(['susceptible', 'infected', 'recovered'])
FIELD_WEIGHTS = np.ones((4, 4)) - np.eye(4)


class PersonElement(Element):
    """Person whose health is one of three labels."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        health: Annotated[object, StateSpec(domain=HEALTH)] = 'susceptible',
    ) -> None:
        super().__init__(label, visual=visual)
        self.health = health


class ProgressionInteraction(Interaction):
    """Progression of the health of its target to the next label."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[object, RoleSpec()],
        step: Annotated[object, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.step = step

    def _get_laws(self, s):
        """Return the progression of the health."""
        return [UpdateLaw(s.target.health, s.target.health + s.step)]


class RotorElement(Element):
    """Rotor whose angle turns at a constant rate."""

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
        """Return the turning of the angle."""
        return [RateLaw(s.theta, s.omega)]


class PopulationElement(Element):
    """Population that grows at a rate."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n: Annotated[object, StateSpec()] = 1.0,
        r: Annotated[object, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.n = n
        self.r = r

    def _get_laws(self, s):
        """Return the growth of the population."""
        return [RateLaw(s.n, s.r * s.n)]


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
    """Inflow into the population of a target at the value of a source."""

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
        """Return the inflow."""
        return [RateLaw(s.target.n, s.source.value)]


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
        """Return the decay."""
        return [RateLaw(s.x, -s.x)]


class SmoothedElement(Element):
    """Stock that relaxes toward 2 as it is perceived over a perception time."""

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
        """Return the relaxation toward the perceived stock."""
        return [RateLaw(s.n, 2.0 - _perceived(s))]


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


def _timed(s):
    """Return the population times the time."""
    return s.a.n * s.t


def _member_timed(member):
    """Return a member's population plus the time."""
    return member.n + member.t


def _local_field_energy(member):
    """Return a member's state times the weighted total of the states of the others."""
    return member.x * total(member.x, weights=FIELD_WEIGHTS)


def _deviation(member):
    """Return a member's deviation from the group mean."""
    return member.x - mean(member.x)


def _deviation_from_other(member):
    """Return a member's state minus the mean of a symbol that is not one of its group."""
    return member.x - mean(member.x, over=symbol('elsewhere'))


def _perceived(s):
    """Return the stock of an element smoothed over its perception time."""
    return distributed_delay(s.n, conserved=False, time=s.perception_time)


def _smoothed_population(s):
    """Return the population smoothed over one unit of time."""
    return distributed_delay(s.n, conserved=False, time=1.0)


ORDER = [ObservableSpec('z', _phase_vector, group='osc'), ObservableSpec('order', _modulus)]


def _oscillators():
    """Return oscillators whose order parameter `ORDER` builds from a complex mean."""
    return System(
        elements=[
            GroupElement(
                'osc',
                element=RotorElement,
                size=50,
                values={'theta': np.linspace(0, 1, 50), 'omega': np.linspace(0.5, 1.5, 50)},
            ),
        ],
    )


def _two_populations():
    """Return two populations, the first growing at rate 1."""
    return System(elements=[PopulationElement('a', n=2.0, r=1.0), PopulationElement('b', n=6.0)])


def _decaying_group(*observables):
    """Return a simulation of a decaying group of 4 members with observables."""
    system = System(elements=[GroupElement('u', element=DecayingElement, size=4, values={'x': np.arange(1.0, 5.0)})])
    return Simulation(system, dt=0.5, observables=list(observables)).check()


@pytest.mark.parametrize(
    ('spec', 'message'),
    [
        (StateSpec(order=0), 'order'),
        (StateSpec(order=-0.5), 'order'),
        (StateSpec(order=float('nan')), 'order'),
        (StateSpec(order=1.5), 'order'),
        (StateSpec(lags=-1), 'lags'),
        (StateSpec(lags=1.5), 'lags'),
        (StateSpec(order=2, lags=1), 'not both'),
        (StateSpec(order=2, initial=(0.0, 1.0)), '2 initial values were given for 1'),
    ],
)
def test_state_spec_rejects_invalid_orders_and_lags(spec, message):
    """Test that an order that is not positive or integral, negative lags, or too many initial values are rejected."""
    with pytest.raises(ValidationError, match=message):
        spec.check()


@pytest.mark.parametrize('spec', [StateSpec(order=0.6), StateSpec(order=2.0), StateSpec(lags=2, initial=(1.0, 2.0))])
def test_state_spec_accepts_fractional_orders_and_lags(spec):
    """Test that a fractional order, an integral order given as a float, and lags with their initial values pass."""
    assert spec.check() is spec


@pytest.mark.parametrize(
    ('spec', 'message'),
    [
        (ParameterSpec(per='inputs', vector=True), 'per entry'),
        (ParameterSpec(per='inputs', adaptive=True), 'per entry'),
        (ParameterSpec(bounds=(0, 1)), 'adaptive parameter'),
        (ParameterSpec(adaptive=True, bounds=(1, 0)), 'lower bound at most the upper one'),
    ],
)
def test_parameter_spec_rejects_invalid_declarations(spec, message):
    """Test that a parameter per role that is a vector or adaptive, and invalid bounds are rejected."""
    with pytest.raises(ValidationError, match=message):
        spec.check()


def test_parameter_spec_accepts_bounds_of_an_adaptive_parameter():
    """Test that an adaptive parameter takes bounds."""
    spec = ParameterSpec(adaptive=True, bounds=(0, 1))
    assert spec.check() is spec


def test_role_spec_is_variadic_or_a_noise_role():
    """Test that a role is either variadic or a noise role, not both."""
    assert RoleSpec(noise=True).check().noise
    with pytest.raises(ValidationError, match='not both'):
        RoleSpec(variadic=True, noise=True).check()


def test_specs_are_frozen_records():
    """Test that the specs are frozen dataclasses that derive from the spec base, itself a configurable object."""
    assert issubclass(BaseSpec, BaseConfig)
    for spec in (StateSpec, ParameterSpec, RoleSpec, ObservableSpec, HilbertSpec, TopologyRequestSpec):
        assert spec.__bases__ == (BaseSpec,)
        assert spec.__dataclass_params__.frozen
    assert PhasesScheduleSpec.__bases__ == (ScheduleSpec,)
    with pytest.raises(dataclasses.FrozenInstanceError):
        StateSpec().order = 2


def test_no_package_outside_the_specs_defines_a_dataclass():
    """Test that no module of the core, the engine, analysis, or experiments defines a dataclass."""
    defined = []
    for package in ('core', 'engine', 'analysis', 'experiments'):
        loaded = importlib.import_module(f'skcomplex.{package}')
        names = [loaded.__name__] + [
            module.name for module in pkgutil.walk_packages(loaded.__path__, f'{loaded.__name__}.')
        ]
        for name in names:
            try:
                module = importlib.import_module(name)
            except ImportError:
                continue
            for value in vars(module).values():
                if inspect.isclass(value) and value.__module__ == module.__name__ and dataclasses.is_dataclass(value):
                    defined.append(f'{module.__name__}.{value.__qualname__}')
    assert defined == []


def test_requirement_and_visual_specs_hold_their_fields():
    """Test that an empty requirement accepts any time and space and a visual keeps its options."""
    assert RequirementSpec() == RequirementSpec(time=(), space=())
    assert RequirementSpec(time=(ContinuousTime,)).time == (ContinuousTime,)
    assert VisualSpec('points', {'position': 'x'}).options == {'position': 'x'}
    assert VisualSpec().kind == 'scalars'


def test_layer_spec_needs_a_string_label():
    """Test that a layer is labelled by a string."""
    assert LayerSpec('work', nx.path_graph(3)).check().label == 'work'
    with pytest.raises(ValidationError, match='string'):
        LayerSpec(1, nx.path_graph(3)).check()


@pytest.mark.parametrize(
    ('spec', 'message'),
    [(WienerProcessSpec(1), 'string'), (WienerProcessSpec('market', dimension=0), 'positive whole dimension')],
)
def test_wiener_process_spec_checks_its_label_and_dimension(spec, message):
    """Test that Wiener processes need a string label and a positive whole dimension."""
    with pytest.raises(ValidationError, match=message):
        spec.check()


@pytest.mark.parametrize(
    ('phases', 'message'),
    [
        ((PhaseSpec('trade'), PhaseSpec('trade')), 'repeat a name'),
        ((PhaseSpec('trade', activation='parallel'),), 'Unknown activation'),
        ((PhaseSpec('trade', repeat=0),), 'at least one round'),
    ],
)
def test_phases_schedule_spec_checks_its_phases(phases, message):
    """Test that phases need their own names, a known activation, and at least one round."""
    with pytest.raises(ValidationError, match=message):
        PhasesScheduleSpec(phases).check()


def test_schedule_spec_has_nothing_to_check():
    """Test that the base schedule spec checks into itself."""
    schedule = ScheduleSpec()
    assert schedule.check() is schedule


def test_monte_carlo_specs_check_their_kinds():
    """Test that an energy has nothing to check, and that moves, samplers, and schedules must be known."""
    spin = var('spin')
    assert EnergySpec(-spin, VonNeumannNeighborhood()).check().expression == -spin
    assert MoveSpec(spin, 'rotate', width=0.5).check().width == 0.5
    with pytest.raises(ValidationError, match='Unknown move'):
        MoveSpec(spin, 'swap').check()
    with pytest.raises(ValidationError, match='Unknown sampler'):
        SamplerSpec('wolff').check()
    with pytest.raises(ValidationError, match='Unknown Monte Carlo schedule'):
        SamplerSpec(schedule='parallel').check()


@pytest.mark.parametrize(
    ('spec', 'message'),
    [
        (HilbertSpec(0), 'positive integer'),
        (HilbertSpec(True), 'positive integer'),
        (HilbertSpec(2, {'z': (np.eye(3), True)}), 'shape'),
        (HilbertSpec(2, basis={'up': 2}), 'outside the dimension'),
        (HilbertSpec(2, {'x': (np.eye(2), True)}), "display operator 'z'"),
    ],
)
def test_hilbert_spec_checks_its_space(spec, message):
    """Test that a Hilbert space needs a positive dimension, square operators, basis states, and its display."""
    with pytest.raises(ValidationError, match=message):
        spec.check()


@pytest.mark.parametrize(
    'spec',
    [
        ModuleSpec(''),
        PrototileSpec('point', [(0, 0)]),
        AffineSpec(translation=(1, 0, 0)),
        AffineSpec(matrix=np.eye(2)),
        GeometrySpec(symbols={'X': 'jump'}),
        GeometrySpec(dimension=4),
    ],
)
def test_rewriting_specs_raise_rewriting_errors(spec):
    """Test that modules, prototiles, affine maps, and geometries reject invalid fields with a rewriting error."""
    with pytest.raises(RewritingError):
        spec.check()


@pytest.mark.parametrize(
    ('spec', 'message'),
    [
        (TopologyRequestSpec('split'), 'Unknown topology request'),
        (TopologyRequestSpec('rewire', keep='middle'), 'source or the target'),
        (TopologyRequestSpec('add-node', edges=1.5), 'whole number'),
        (TopologyRequestSpec('add-node', edges=-1), 'whole number'),
    ],
)
def test_topology_request_spec_checks_its_kind_end_and_edges(spec, message):
    """Test that a topology request needs a known kind and kept end and a whole number of edges."""
    with pytest.raises(ValidationError, match=message):
        spec.check()


def test_categorical_values_round_trip_as_labels():
    """Test that categorical states take labels as values and return them as labels."""
    system = System(
        elements=[PersonElement('ann', health='infected')],
        interactions=[ProgressionInteraction('p', target='ann')],
    )
    trajectory = Simulation(system).check().run(until=1).trajectory_
    assert trajectory.values('ann.health').tolist() == ['infected', 'recovered']


def test_categorical_rejects_unknown_label():
    """Test that an initial value outside the alphabet is rejected."""
    with pytest.raises(ValidationError, match=r"ann\.health = 'dead' is not in its domain"):
        Simulation(System(elements=[PersonElement('ann', health='dead')])).check().run(until=1)


def test_checked_mode_catches_values_outside_alphabet():
    """Test that checked mode names the state when a law leaves the alphabet, and unchecked mode does not."""
    system = System(
        elements=[PersonElement('ann', health='recovered')],
        interactions=[ProgressionInteraction('p', target='ann')],
    )
    with pytest.raises(ValidationError, match=r'ann\.health took the values 3\.0 at t = 1\.0, outside its domain'):
        Simulation(system, checked=True).check().run(until=1)
    unchecked = Simulation(system).check().run(until=1).trajectory_
    assert unchecked.states[-1, 0] == 3.0


def test_angle_wraps_into_one_turn():
    """Test that an angle growing past pi lies in (-pi, pi] after every step."""
    system = System(elements=[RotorElement('r', theta=6.0, omega=1.0)])
    values = Simulation(system, dt=0.5).check().run(until=3).trajectory_.values('r.theta')
    assert np.all((values[1:] > -np.pi) & (values[1:] <= np.pi))
    assert values[-1] == pytest.approx(9.0 - 2 * np.pi, abs=1e-6)


def test_angle_swings_around_zero_without_jumping():
    """Test that an angle decreasing through zero stays near zero instead of jumping a full turn."""
    system = System(elements=[RotorElement('r', theta=0.1, omega=-1.0)])
    values = Simulation(system, dt=0.1).check().run(until=1).trajectory_.values('r.theta')
    assert np.all((values > -np.pi) & (values <= np.pi))
    assert values[-1] == pytest.approx(-0.9, abs=1e-6)


def test_order_parameter_every_frame():
    """Test that the modulus of the mean of exp(i theta) is recorded at every frame between 0 and 1."""
    trajectory = Simulation(_oscillators(), dt=0.5, observables=ORDER).check().run(until=10).trajectory_
    order = trajectory.values('order')
    assert order.shape == (21,)
    assert np.all((order >= 0) & (order <= 1))
    expected = np.abs(np.exp(1j * np.asarray(trajectory.values('osc.theta'))).mean(axis=1))
    assert np.allclose(order, expected)


def test_expression_observable_of_elements():
    """Test that a system observable combines symbols of single elements and earlier observables."""
    observables = [ObservableSpec('total', _total), ObservableSpec('share', _share)]
    trajectory = Simulation(_two_populations(), dt=0.5, observables=observables).check().run(until=1).trajectory_
    assert np.allclose(trajectory.values('total'), trajectory.values('a.n') + trajectory.values('b.n'))
    assert np.allclose(trajectory.values('share'), trajectory.values('a.n') / trajectory.values('total'))


def test_observable_by_subsystem_path():
    """Test that an observable of a subsystem is recorded under the subsystem's path."""
    system = System(systems=[System('market', elements=[PopulationElement('a', n=3.0)])])
    simulation = Simulation(system, dt=1, observables=[ObservableSpec('market/volume', _volume)]).check()
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
    trajectory = Simulation(system, dt=1, observables=observables).check().run(until=2).trajectory_
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
    trajectory = Simulation(system, dt=1.0, observables=observables).check().run(until=1).trajectory_
    assert trajectory.values('soil.n')[-1] == pytest.approx(trajectory.values('soil.n')[0] + 3.0)


def test_observable_of_a_member_symbol_outside_a_reduction_is_rejected():
    """Test that an observable using a group member's symbol outside a reduction is rejected."""
    system = System(elements=[GroupElement('herd', element=PopulationElement, size=3)])
    with pytest.raises(AttributeError, match='herd'):
        Simulation(system, observables=[ObservableSpec('bad', _herd)]).check().run(until=1)


def test_observable_lambda_is_rejected():
    """Test that checking an observable whose expression or reduction is a lambda raises a named error."""
    with pytest.raises(ValidationError, match="expression of the observable 'total'"):
        ObservableSpec('total', lambda s: s.a.n + s.b.n).check()
    with pytest.raises(ValidationError, match='reduction'):
        ObservableSpec('spread', group='herd', variable='n', reduce=lambda values: values.ptp()).check()


def test_simulation_checks_its_observables():
    """Test that a run rejects an observable whose expression is a lambda before compiling."""
    system = System(elements=[PopulationElement('a')])
    with pytest.raises(ValidationError, match="'total'"):
        Simulation(system, observables=[ObservableSpec('total', lambda s: s.a.n)]).check().run(until=1)


def test_observable_module_function_pickles():
    """Test that a module-level observable pickles and records the same values once restored."""
    observables = [ObservableSpec('total', _total)]
    restored = pickle.loads(pickle.dumps(observables))  # noqa: S301  # nosec B301: loads only what it dumped
    expected = (
        Simulation(_two_populations(), dt=0.5, observables=observables).check().run(until=1).trajectory_.values('total')
    )
    values = (
        Simulation(_two_populations(), dt=0.5, observables=restored).check().run(until=1).trajectory_.values('total')
    )
    assert np.allclose(values, expected)


def test_observables_read_time():
    """Test that system observables and member expressions of reductions read the time as `s.t`."""
    system = System(
        elements=[
            PopulationElement('a', n=2.0),
            GroupElement('herd', element=PopulationElement, size=2, values={'n': [1.0, 3.0]}),
        ],
    )
    observables = [ObservableSpec('timed', _timed), ObservableSpec('member_timed', _member_timed, group='herd')]
    trajectory = Simulation(system, dt=1.0, observables=observables).check().run(until=2).trajectory_
    assert trajectory.values('timed').tolist() == [0.0, 2.0, 4.0]
    assert trajectory.values('member_timed').tolist() == [2.0, 3.0, 4.0]


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


def test_observable_of_an_element_records_its_expression_with_delays():
    """Test that an observable of an element's expression records it under the element and reads its delays."""
    observed = ObservableSpec('perceived', _perceived, element='tank')
    trajectory = Simulation(System(elements=[SmoothedElement('tank')]), dt=0.5, observables=[observed]).check()
    perceived = trajectory.run(until=4).trajectory_.values('tank.perceived')
    np.testing.assert_allclose(perceived, trajectory.trajectory_.values('tank.smoothing_delay1_0')[:, 0])
    assert perceived[0] == pytest.approx(1.0)
    assert perceived[-1] > perceived[0]


def test_observable_of_an_element_rejects_a_delay_no_law_uses():
    """Test that observing a delay that no law of the system reads raises an error that names the observable."""
    observed = ObservableSpec('smoothed', _smoothed_population, element='rabbits')
    with pytest.raises(ValidationError, match=r'rabbits\.smoothed'):
        Simulation(System(elements=[PopulationElement('rabbits', n=2.0)]), observables=[observed]).check().run(until=1)


def test_scenario_spec_changes_a_parameter_at_its_scheduled_time():
    """Test that halving the effort at time 50 changes the stock only after time 50, with the same seeds."""
    scenarios = [ScenarioSpec('baseline'), ScenarioSpec('quota', schedule=[(50.0, 'harvest.effort', 0.025)])]
    experiment = Experiment(harvesting, scenarios=scenarios, until=80, outputs=['land/stock.n'], random_state=0)
    result = experiment.check().prepare().run().result_
    baseline, quota = result.series('land/stock.n')
    times = result.runs[0].times
    assert np.array_equal(baseline[times <= 50], quota[times <= 50])
    assert quota[-1] > baseline[-1]
    assert result.runs[0].seed == result.runs[1].seed


def test_scenario_spec_keeps_the_initial_value_before_its_first_change():
    """Test that a parameter keeps its initial value before its first change and the last due change after it."""
    schedule = [(50.0, 'harvest.effort', 0.025), (20.0, 'harvest.effort', 0.04)]
    scenarios = [ScenarioSpec('baseline'), ScenarioSpec('late', schedule=schedule)]
    experiment = Experiment(harvesting, scenarios=scenarios, until=80, outputs=['land/stock.n'], random_state=0)
    result = experiment.check().prepare().run().result_
    baseline, late = result.series('land/stock.n')
    times = result.runs[0].times
    assert np.array_equal(baseline[times <= 20], late[times <= 20])
    assert not np.array_equal(baseline[times > 21], late[times > 21])


def test_scenario_spec_is_a_frozen_spec():
    """Test that a scenario is a frozen spec whose default is the baseline."""
    scenario = ScenarioSpec()
    assert isinstance(scenario, BaseSpec)
    assert scenario.check().name == 'baseline'
    with pytest.raises(dataclasses.FrozenInstanceError):
        scenario.name = 'quota'
