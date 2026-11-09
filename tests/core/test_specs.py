"""Test the `specs` module of `skcomplex`."""

import dataclasses
import importlib
import inspect
import pkgutil
from multiprocessing.reduction import ForkingPickler
from typing import Annotated

import networkx as nx
import numpy as np
import pytest
from sage.all__sagemath_symbolics import I, exp, pi, var
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from skcomplex.core.base import BaseConfig
from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw, RuleLaw, UpdateLaw
from skcomplex.core.components.laws.utils import distributed_delay, mean, total
from skcomplex.core.components.spaces import ContinuousTime, VonNeumannNeighborhood
from skcomplex.core.components.specs import ScenarioSpec
from skcomplex.core.components.systems import System
from skcomplex.core.engine._utils import symbol
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import RoleError, ValidationError
from skcomplex.core.experiments import compare_scenarios
from skcomplex.core.specs import (
    BaseSpec,
    BatchMatchingSpec,
    ClampSpec,
    ContinuousMatchingSpec,
    DatasetSpec,
    HilbertSpec,
    IterativeMatchingSpec,
    LayerSpec,
    MatchSpec,
    MoveSpec,
    ObjectiveSpec,
    ObservableSpec,
    ParameterSpec,
    PartnerSpec,
    PartnerSwitchSpec,
    PhaseSpec,
    PhasesScheduleSpec,
    QueueSpec,
    RadiusSpec,
    RandomSearchSpec,
    RequirementSpec,
    RoleSpec,
    SamplerSpec,
    ScheduleSpec,
    StableMatchingSpec,
    StateSpec,
    SymmetrySpec,
    TopologyRequestSpec,
    VisualSpec,
    WienerProcessSpec,
)
from skcomplex.core.types import ElementKind
from skcomplex.systems.economics.resources import HarvestingSystem

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

    def save(self, agent, system):
        """Add one to the wealth."""
        del system
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


def _harvesting() -> System:
    """Return the parts of the harvesting system, the factory of the scenario experiments."""
    return HarvestingSystem().prepare().system_


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


def _decaying_group(*observables, until=0.0):
    """Return the trajectory of a decaying group of 4 members with observables until a time."""
    system = System(elements=[GroupElement('u', element=DecayingElement, size=4, values={'x': np.arange(1.0, 5.0)})])
    return simulate(system, until=until, dt=0.5, observables=list(observables))


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
        (StateSpec(reference='banks', vector=True), 'scalar of order 1'),
        (StateSpec(reference='banks', lags=1), 'scalar of order 1'),
    ],
)
def test_state_spec_rejects_invalid_orders_and_lags(spec, message):
    """Test that an order that is not positive or integral, negative lags, or too many initial values are rejected."""
    with pytest.raises(ValidationError, match=message):
        spec._check()


@pytest.mark.parametrize('spec', [StateSpec(order=0.6), StateSpec(order=2.0), StateSpec(lags=2, initial=(1.0, 2.0))])
def test_state_spec_accepts_fractional_orders_and_lags(spec):
    """Test that a fractional order, an integral order given as a float, and lags with their initial values pass."""
    spec._check()


@pytest.mark.parametrize(
    ('spec', 'message'),
    [
        (ParameterSpec(per='inputs', vector=True), 'per entry'),
        (ParameterSpec(per='inputs', adaptive=True), 'per entry'),
        (ParameterSpec(bounds=(0, 1)), 'adaptive parameter'),
        (ParameterSpec(adaptive=True, bounds=(1, 0)), 'lower bound at most the upper one'),
        (ParameterSpec(shape=()), 'nonempty tuple of role names'),
        (ParameterSpec(shape=('target',), vector=True), 'neither a vector'),
        (ParameterSpec(shape=('target',), per='inputs'), 'per entry'),
    ],
)
def test_parameter_spec_rejects_invalid_declarations(spec, message):
    """Test that a parameter per role that is a vector or adaptive, invalid bounds, and bad shapes are rejected."""
    with pytest.raises(ValidationError, match=message):
        spec._check()


def test_parameter_spec_accepts_bounds_of_an_adaptive_parameter():
    """Test that an adaptive parameter takes bounds."""
    spec = ParameterSpec(adaptive=True, bounds=(0, 1))
    spec._check()


def test_parameter_spec_accepts_a_shape_of_role_names():
    """Test that a parameter takes the roles of its shape, a role repeated for a tensor."""
    assert ParameterSpec(shape=('target',) * 3, adaptive=True).shape == ('target', 'target', 'target')


def test_role_spec_is_variadic_or_a_noise_role():
    """Test that a role is either variadic or a noise role, not both."""
    assert RoleSpec(noise=True).noise
    with pytest.raises(ValidationError, match='not both'):
        RoleSpec(variadic=True, noise=True)._check()


def test_role_spec_may_be_optional():
    """Test that a role may be declared optional, so it may stay unbound."""
    assert RoleSpec(optional=True).optional
    assert not RoleSpec().optional


def test_specs_are_frozen_records():
    """Test that the specs are frozen dataclasses that derive from the spec base, which is not an estimator."""
    assert not issubclass(BaseSpec, BaseConfig)
    assert not hasattr(BaseSpec, 'prepare')
    assert not hasattr(BaseSpec, 'run')
    for spec in (StateSpec, ParameterSpec, RoleSpec, ObservableSpec, HilbertSpec, TopologyRequestSpec):
        assert spec.__bases__ == (BaseSpec,)
        assert spec.__dataclass_params__.frozen
    assert PhasesScheduleSpec.__bases__ == (ScheduleSpec,)
    with pytest.raises(dataclasses.FrozenInstanceError):
        StateSpec().order = 2


def test_every_dataclass_is_a_frozen_spec_or_result():
    """Test that every dataclass of the core is frozen, and a spec unless it is the result a function returns."""
    defined = []
    loaded = importlib.import_module('skcomplex.core')
    names = [loaded.__name__] + [module.name for module in pkgutil.walk_packages(loaded.__path__, 'skcomplex.core.')]
    for name in names:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        for value in vars(module).values():
            own = inspect.isclass(value) and value.__module__ == module.__name__
            if not (own and dataclasses.is_dataclass(value)):
                continue
            kind = issubclass(value, BaseSpec) or (value.__name__.endswith('Result') and value.__bases__ == (object,))
            if not (kind and value.__dataclass_params__.frozen):
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
    assert LayerSpec('work', nx.path_graph(3)).label == 'work'
    with pytest.raises(ValidationError, match='string'):
        LayerSpec(1, nx.path_graph(3))._check()


@pytest.mark.parametrize(
    ('spec', 'message'),
    [(WienerProcessSpec(1), 'string'), (WienerProcessSpec('market', dimension=0), 'positive whole dimension')],
)
def test_wiener_process_spec_checks_its_label_and_dimension(spec, message):
    """Test that Wiener processes need a string label and a positive whole dimension."""
    with pytest.raises(ValidationError, match=message):
        spec._check()


@pytest.mark.parametrize(
    ('phases', 'message'),
    [
        ((PhaseSpec('trade'), PhaseSpec('trade')), 'repeat a name'),
        ((PhaseSpec('trade', activation='parallel'),), 'Unknown activation'),
        ((PhaseSpec('trade', repeat=0),), 'at least one round'),
        ((PhaseSpec('trade', activation='visits', visits=-1.0),), 'at least 0'),
        ((PhaseSpec('trade', period=0),), 'period of at least one step'),
    ],
)
def test_phases_schedule_spec_checks_its_phases(phases, message):
    """Test that phases need their own names, a known activation, and at least one round."""
    with pytest.raises(ValidationError, match=message):
        PhasesScheduleSpec(phases)._check()


def test_schedule_spec_has_nothing_to_check():
    """Test that the base schedule spec checks into itself."""
    schedule = ScheduleSpec()
    schedule._check()


def test_monte_carlo_specs_check_their_kinds():
    """Test that an energy has nothing to check, and that moves, samplers, and schedules must be known."""
    spin = var('spin')
    assert ObjectiveSpec(-spin, VonNeumannNeighborhood()).expression == -spin
    assert MoveSpec(spin, 'rotate', width=0.5).width == 0.5
    assert MoveSpec(spin, 'gaussian', width=0.5).kind == 'gaussian'
    assert MoveSpec(spin, 'function', function=lambda values, random: values).kind == 'function'
    with pytest.raises(ValidationError, match='Unknown move'):
        MoveSpec(spin, 'swap')._check()
    with pytest.raises(ValidationError, match='callable function'):
        MoveSpec(spin, 'function')._check()
    with pytest.raises(ValidationError, match='Unknown sampler'):
        SamplerSpec('wolff')._check()
    with pytest.raises(ValidationError, match='Unknown Monte Carlo schedule'):
        SamplerSpec(schedule='parallel')._check()


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
        spec._check()


@pytest.mark.parametrize(
    ('spec', 'message'),
    [
        (TopologyRequestSpec('split'), 'Unknown topology request'),
        (TopologyRequestSpec('rewire', keep='middle'), 'source or the target'),
        (TopologyRequestSpec('add-node', edges=1.5), 'whole number'),
        (TopologyRequestSpec('add-node', edges=-1), 'whole number'),
        (TopologyRequestSpec('add-node', hyperedge=1), 'at least 2 members'),
        (TopologyRequestSpec('add-edge', hyperedge=3), 'forms no hyperedge'),
        (TopologyRequestSpec('add-edge', candidates=0), 'at least 1 candidates'),
        (TopologyRequestSpec('add-node', candidates=2), 'chooses among candidates'),
        (TopologyRequestSpec('add-edge', candidates=2, independent=True), 'chooses among candidates'),
        (TopologyRequestSpec('add-edge', score=3), 'function of the namespace'),
        (TopologyRequestSpec('duplicate-node', inherit=1.5), r'in \[0, 1\]'),
        (TopologyRequestSpec('duplicate-node', parent=-0.1), r'in \[0, 1\]'),
    ],
)
def test_topology_request_spec_checks_its_kind_end_and_edges(spec, message):
    """Test that a topology request needs a known kind and kept end, whole edges, and a hyperedge of a new node."""
    with pytest.raises(ValidationError, match=message):
        spec._check()


def test_categorical_values_round_trip_as_labels():
    """Test that categorical states take labels as values and return them as labels."""
    system = System(
        elements=[PersonElement('ann', health='infected')],
        interactions=[ProgressionInteraction('p', target='ann')],
    )
    trajectory = simulate(system, until=1)
    assert trajectory.values('ann.health').tolist() == ['infected', 'recovered']


def test_categorical_rejects_unknown_label():
    """Test that an initial value outside the alphabet is rejected."""
    with pytest.raises(ValidationError, match=r"ann\.health = 'dead' is not in its domain"):
        simulate(System(elements=[PersonElement('ann', health='dead')]), until=1)


def test_simulate_does_not_check_values_outside_alphabet():
    """Test that a run leaves a state that a law moves outside its alphabet unchecked."""
    system = System(
        elements=[PersonElement('ann', health='recovered')],
        interactions=[ProgressionInteraction('p', target='ann')],
    )
    unchecked = simulate(system, until=1)
    assert unchecked.states[-1, 0] == 3.0


def test_angle_wraps_into_one_turn():
    """Test that an angle growing past pi lies in (-pi, pi] after every step."""
    system = System(elements=[RotorElement('r', theta=6.0, omega=1.0)])
    values = simulate(system, until=3, dt=0.5).values('r.theta')
    assert np.all((values[1:] > -np.pi) & (values[1:] <= np.pi))
    assert values[-1] == pytest.approx(9.0 - 2 * np.pi, abs=1e-6)


def test_angle_swings_around_zero_without_jumping():
    """Test that an angle decreasing through zero stays near zero instead of jumping a full turn."""
    system = System(elements=[RotorElement('r', theta=0.1, omega=-1.0)])
    values = simulate(system, until=1, dt=0.1).values('r.theta')
    assert np.all((values > -np.pi) & (values <= np.pi))
    assert values[-1] == pytest.approx(-0.9, abs=1e-6)


def test_order_parameter_every_frame():
    """Test that the modulus of the mean of exp(i theta) is recorded at every frame between 0 and 1."""
    trajectory = simulate(_oscillators(), until=10, dt=0.5, observables=ORDER)
    order = trajectory.values('order')
    assert order.shape == (21,)
    assert np.all((order >= 0) & (order <= 1))
    expected = np.abs(np.exp(1j * np.asarray(trajectory.values('osc.theta'))).mean(axis=1))
    assert np.allclose(order, expected)


def test_expression_observable_of_elements():
    """Test that a system observable combines symbols of single elements and earlier observables."""
    observables = [ObservableSpec('total', _total), ObservableSpec('share', _share)]
    trajectory = simulate(_two_populations(), until=1, dt=0.5, observables=observables)
    assert np.allclose(trajectory.values('total'), trajectory.values('a.n') + trajectory.values('b.n'))
    assert np.allclose(trajectory.values('share'), trajectory.values('a.n') / trajectory.values('total'))


def test_observable_by_subsystem_path():
    """Test that an observable of a subsystem is recorded under the subsystem's path."""
    system = System(systems=[System('market', elements=[PopulationElement('a', n=3.0)])])
    trajectory = simulate(system, until=2, dt=1, observables=[ObservableSpec('market/volume', _volume)])
    assert trajectory.values('market/volume').tolist() == [6.0, 6.0, 6.0]


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
    trajectory = simulate(system, until=2, dt=1, observables=observables)
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
    trajectory = simulate(system, until=1, dt=1.0, observables=observables)
    assert trajectory.values('soil.n')[-1] == pytest.approx(trajectory.values('soil.n')[0] + 3.0)


def test_observable_of_a_member_symbol_outside_a_reduction_is_rejected():
    """Test that an observable using a group member's symbol outside a reduction is rejected."""
    system = System(elements=[GroupElement('herd', element=PopulationElement, size=3)])
    with pytest.raises(AttributeError, match='herd'):
        simulate(system, until=1, observables=[ObservableSpec('bad', _herd)])


def test_observable_lambda_is_rejected():
    """Test that checking an observable whose expression or reduction is a lambda raises a named error."""
    with pytest.raises(ValidationError, match="expression of the observable 'total'"):
        ObservableSpec('total', lambda s: s.a.n + s.b.n)._check()
    with pytest.raises(ValidationError, match='reduction'):
        ObservableSpec('spread', group='herd', variable='n', reduce=lambda values: values.ptp())._check()


def test_simulation_checks_its_observables():
    """Test that a run rejects an observable whose expression is a lambda before compiling."""
    system = System(elements=[PopulationElement('a')])
    with pytest.raises(ValidationError, match="'total'"):
        simulate(system, until=1, observables=[ObservableSpec('total', lambda s: s.a.n)])


def test_observable_module_function_pickles():
    """Test that a module-level observable pickles and records the same values once restored."""
    observables = [ObservableSpec('total', _total)]
    restored = ForkingPickler.loads(ForkingPickler.dumps(observables))
    expected = simulate(_two_populations(), until=1, dt=0.5, observables=observables).values('total')
    values = simulate(_two_populations(), until=1, dt=0.5, observables=restored).values('total')
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
    trajectory = simulate(system, until=2, dt=1.0, observables=observables)
    assert trajectory.values('timed').tolist() == [0.0, 2.0, 4.0]
    assert trajectory.values('member_timed').tolist() == [2.0, 3.0, 4.0]


def test_observable_reads_weighted_reduction_of_its_members():
    """Test that a member expression with a weighted total gives the local field energy at every frame."""
    trajectory = _decaying_group(ObservableSpec('energy', _local_field_energy, group='u', reduce='sum'), until=2)
    x = np.asarray(trajectory.values('u.x'))
    np.testing.assert_allclose(trajectory.values('energy'), np.einsum('fi,ij,fj->f', x, FIELD_WEIGHTS, x), rtol=1e-12)


def test_observable_reads_plain_reduction_of_its_members():
    """Test that a member expression with a plain mean gives the variance of the group at every frame."""
    trajectory = _decaying_group(ObservableSpec('spread', _deviation, group='u', reduce='var'), until=2)
    np.testing.assert_allclose(
        trajectory.values('spread'),
        np.asarray(trajectory.values('u.x')).var(axis=1),
        rtol=1e-12,
    )


def test_observable_rejects_reduction_over_another_group():
    """Test that a reduction over a group other than the observable's raises the role error of laws."""
    with pytest.raises(RoleError, match=r'observable other reads a group other than u'):
        _decaying_group(ObservableSpec('other', _deviation_from_other, group='u', reduce='sum'))


def test_observable_of_an_element_records_its_expression_with_delays():
    """Test that an observable of an element's expression records it under the element and reads its delays."""
    observed = ObservableSpec('perceived', _perceived, element='tank')
    trajectory = simulate(System(elements=[SmoothedElement('tank')]), until=4, dt=0.5, observables=[observed])
    perceived = trajectory.values('tank.perceived')
    np.testing.assert_allclose(perceived, trajectory.values('tank.smoothing_delay1_0')[:, 0])
    assert perceived[0] == pytest.approx(1.0)
    assert perceived[-1] > perceived[0]


def test_observable_of_an_element_rejects_a_delay_no_law_uses():
    """Test that observing a delay that no law of the system reads raises an error that names the observable."""
    observed = ObservableSpec('smoothed', _smoothed_population, element='rabbits')
    with pytest.raises(ValidationError, match=r'rabbits\.smoothed'):
        simulate(System(elements=[PopulationElement('rabbits', n=2.0)]), until=1, observables=[observed])


def test_scenario_spec_changes_a_parameter_at_its_scheduled_time():
    """Test that halving the effort at time 50 changes the stock only after time 50, with the same seeds."""
    scenarios = [ScenarioSpec('baseline'), ScenarioSpec('quota', schedule=[(50.0, 'harvest.effort', 0.025)])]
    result = compare_scenarios(_harvesting, scenarios, until=80, outputs=['land/stock.n'], random_state=0)
    baseline, quota = result.series('land/stock.n')
    times = result.runs[0].times
    assert np.array_equal(baseline[times <= 50], quota[times <= 50])
    assert quota[-1] > baseline[-1]
    assert result.runs[0].seed == result.runs[1].seed


def test_scenario_spec_keeps_the_initial_value_before_its_first_change():
    """Test that a parameter keeps its initial value before its first change and the last due change after it."""
    schedule = [(50.0, 'harvest.effort', 0.025), (20.0, 'harvest.effort', 0.04)]
    scenarios = [ScenarioSpec('baseline'), ScenarioSpec('late', schedule=schedule)]
    result = compare_scenarios(_harvesting, scenarios, until=80, outputs=['land/stock.n'], random_state=0)
    baseline, late = result.series('land/stock.n')
    times = result.runs[0].times
    assert np.array_equal(baseline[times <= 20], late[times <= 20])
    assert not np.array_equal(baseline[times > 21], late[times > 21])


def test_scenario_spec_is_a_frozen_spec():
    """Test that a scenario is a frozen spec whose default is the baseline."""
    scenario = ScenarioSpec()
    assert isinstance(scenario, BaseSpec)
    assert scenario.name == 'baseline'
    with pytest.raises(dataclasses.FrozenInstanceError):
        scenario.name = 'quota'


def test_match_spec_is_a_frozen_spec():
    """Test that a match is a frozen spec of a requester, a provider, a quantity, and a price."""
    match = MatchSpec('requester', 'provider', 2.0, 5.0)
    assert (match.requester, match.provider, match.quantity * match.value) == ('requester', 'provider', 10.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        match.value = 1.0


@pytest.mark.parametrize(
    ('spec', 'name', 'value'),
    [
        (RandomSearchSpec(tries=3), 'tries', 3),
        (PartnerSwitchSpec(sample=2), 'sample', 2),
        (QueueSpec(order='limit'), 'order', 'limit'),
        (IterativeMatchingSpec(iterations=10), 'iterations', 10),
        (ContinuousMatchingSpec(arrival='sequential'), 'arrival', 'sequential'),
        (StableMatchingSpec(clearing='requester'), 'clearing', 'requester'),
        (PartnerSpec(candidates=2.0), 'candidates', 2.0),
    ],
)
def test_matching_rule_specs_hold_their_settings(spec, name, value):
    """Test that a matching rule is a frozen spec that keeps the setting it was given."""
    checked = spec
    assert getattr(checked, name) == value
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(checked, name, value)


def test_batch_matching_spec_has_no_settings():
    """Test that batch matching is a spec without settings."""
    assert dataclasses.fields(BatchMatchingSpec()) == ()


@pytest.mark.parametrize(
    ('spec', 'message'),
    [
        (RandomSearchSpec(clearing='bid'), 'midpoint'),
        (QueueSpec(order='bid'), 'order'),
        (ContinuousMatchingSpec(arrival='bid'), 'arrival'),
        (StableMatchingSpec(clearing='bid'), 'clearing'),
        (PartnerSpec(candidates=0.0), 'positive'),
        (PartnerSpec(candidates=True), 'positive'),
        (PartnerSpec(clearing='bid'), 'clearing'),
        (PartnerSpec(choice='score', score=2), 'score'),
    ],
)
def test_matching_rule_specs_reject_invalid_settings(spec, message):
    """Test that an unknown option, a distance that is not positive, and a bad score raise an error naming them."""
    with pytest.raises(ValidationError, match=message):
        spec._check()


@pytest.mark.parametrize(
    ('spec', 'message'),
    [
        (RadiusSpec(1.0, lambda agent, other, system: (0.0,), reduce='max'), 'reduction'),
        (RadiusSpec(1.0, 'term'), 'callable'),
        (RadiusSpec(-1.0, lambda agent, other, system: (0.0,)), 'nonnegative'),
        (ClampSpec(max_angle=-0.1), 'max_angle'),
    ],
)
def test_check_agent_coupling_specs_reject_invalid_settings(spec, message):
    """Test that a radius or a clamp with an invalid setting raises an error naming it."""
    with pytest.raises(ValidationError, match=message):
        spec._check()


@pytest.mark.parametrize(
    ('spec', 'message'),
    [(SymmetrySpec('', lambda s: {}), 'empty label'), (SymmetrySpec('rotation', 'turn'), 'generator')],
)
def test_check_symmetry_spec_rejects_invalid_settings(spec, message):
    """Test that a symmetry with an empty label or a generator that is not callable raises an error naming it."""
    with pytest.raises(ValidationError, match=message):
        spec._check()


def test_dataset_spec_counts_sequences_and_steps():
    """Test that a dataset counts the sequences and steps of its first array."""
    data = DatasetSpec({'inputs.x': np.zeros((5, 12, 1)), 'outputs.target': np.zeros((5, 12, 1))})
    assert (data.sequences, data.steps) == (5, 12)


@pytest.mark.parametrize('values', [{}, {'inputs.x': np.zeros(5)}, {1: np.zeros((5, 12))}])
def test_dataset_spec_rejects_values_it_cannot_take(values):
    """Test that empty values, an array of one dimension, or a path that is not a string raise an error."""
    with pytest.raises(ValidationError):
        DatasetSpec(values)._check()


@pytest.mark.parametrize(
    'arguments',
    [
        {'choice': 'best'},
        {'fallback': 'first'},
        {'candidates': -1.0},
        {'candidates': True},
        {'capacity': 0},
        {'choice': 'score'},
        {'score': print},
    ],
)
def test_partner_spec_check_rejects_unusable_settings(arguments):
    """Test that the check of a partner spec names a choice, fallback, distance, capacity, or score it cannot take."""
    with pytest.raises(ValidationError):
        PartnerSpec(**arguments)._check()


def test_role_spec_defaults_to_any_element_and_no_states():
    """Test that a role that declares nothing accepts every kind of element and reads no state."""
    role = RoleSpec()
    assert role.states == ()
    assert all(role.accepts(kind) for kind in ElementKind)


def test_role_spec_accepts_only_the_kinds_it_names():
    """Test that a role accepts the kinds it names, or every kind when it names any."""
    role = RoleSpec(elements=(ElementKind.GROUP,), states=('state',))
    assert role.accepts(ElementKind.GROUP)
    assert not role.accepts(ElementKind.ELEMENT)
    assert RoleSpec(elements=(ElementKind.ANY,)).accepts(ElementKind.ELEMENT)
