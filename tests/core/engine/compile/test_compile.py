"""Test the `_compile` module of `skcomplex.core.engine.compile`."""

import time
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import RR, Expression, oo
from sage.modules.free_module_element import FreeModuleElement
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import CompartmentElement, Element, GroupElement, LatticeElement
from skcomplex.core.components.interactions import Interaction, PairCouplingInteraction
from skcomplex.core.components.laws import BoundaryLaw, JumpLaw, NoiseLaw, RateLaw, UpdateLaw
from skcomplex.core.components.laws.utils import mod
from skcomplex.core.components.spaces import ContinuousSpace, LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine._utils import domain_of, symbol
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.compile import compile_system
from skcomplex.core.errors import (
    IncompatibleLawsError,
    LabelError,
    RoleElementError,
    RoleError,
    SpaceError,
    ValidationError,
)
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from skcomplex.core.types import FormKind
from skcomplex.systems.biology.common.populations import (
    ExponentialGrowthInteraction,
    LogisticGrowthInteraction,
    PopulationElement,
)
from skcomplex.systems.biology.ecology import PredationInteraction
from skcomplex.systems.biology.epidemiology import InfectionInteraction, RecoveryInteraction
from skcomplex.systems.economics.finance import AssetElement, GeometricBrownianMotionInteraction
from skcomplex.systems.economics.markets import CobwebInteraction, MarketElement
from skcomplex.systems.physics.common.particles import (
    ConstantForceInteraction,
    HarmonicPotentialInteraction,
    MassPointElement,
)
from skcomplex.systems.physics.statistical_mechanics import ParticleSiteElement, ZeroRangeHopInteraction

from ....conftest import CallableGrowthInteraction, NodeElement, _correlated, _summed
from .conftest import PairElement


class UndeclaredInteraction(Interaction):
    """Interaction that uses a symbol it does not declare."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.target = target

    def _get_laws(self, s):
        """Return a law with an undeclared symbol."""
        return [RateLaw(s.target.n, symbol('k') * s.target.n)]


class DoubleUpdateInteraction(Interaction):
    """Interaction that updates the same state twice."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.target = target

    def _get_laws(self, s):
        """Return two updates of one state."""
        return [UpdateLaw(s.target.price, 1), UpdateLaw(s.target.price, 2)]


def _ecology(growth):
    """Return a system of one population under a growth interaction."""
    return System(elements=[PopulationElement('rabbits', n=10)], interactions=[growth])


class HubElement(Element):
    """Hub element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        y: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.y = y


class SiteDriveInteraction(Interaction):
    """Growth of a hub at the rate of the state of one site."""

    def __init__(
        self,
        label: str,
        *,
        hub: Annotated[str, RoleSpec()],
        site: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.hub = hub
        self.site = site

    def _get_laws(self, s):
        """Return the growth of the hub."""
        return [RateLaw(s.hub.y, s.site.x)]


class SitesDriveInteraction(Interaction):
    """Growth of a hub at the rate of the summed states of several sites."""

    def __init__(
        self,
        label: str,
        *,
        hub: Annotated[str, RoleSpec()],
        sites: Annotated[list[str], RoleSpec(variadic=True)],
    ) -> None:
        super().__init__(label)
        self.hub = hub
        self.sites = sites

    def _get_laws(self, s):
        """Return the growth of the hub."""
        return [RateLaw(s.hub.y, sum((item.x for item in s.sites), 0))]


def _hub_and_sites(interaction):
    """Return a hub and a group of five sites with states 0 to 4, coupled by an interaction."""
    return System(
        elements=[
            GroupElement('sites', element=NodeElement, size=5, values={'x': np.arange(5.0)}),
            HubElement('hub'),
        ],
        interactions=[interaction],
    )


def _compile_time(n_links):
    """Return the shortest of two compile times of a zero-range ring written with one interaction per link."""
    sites = LatticeElement(
        'sites',
        element=ParticleSiteElement,
        space=LatticeSpace(n_links),
        values={'n': np.ones(n_links)},
    )
    links = [
        ZeroRangeHopInteraction(f'hop_{index}', link=[f'sites[{index}]', f'sites[{(index + 1) % n_links}]'])
        for index in range(n_links)
    ]
    system = System(elements=[sites], interactions=links)
    durations = []
    for _ in range(2):
        start = time.perf_counter()
        compile_system(system, backend=NumpyBackend())
        durations.append(time.perf_counter() - start)
    return min(durations)


CUTOFF = 2.5


class LennardJonesInteraction(PairCouplingInteraction):
    """Lennard-Jones pair potential."""

    def __init__(
        self,
        label: str,
        first: str,
        second: str,
        cutoff: float = 1.0,
        position: str = 'x',
        *,
        period: float | None = None,
        epsilon: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        sigma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(
            label,
            first,
            second,
            cutoff=cutoff,
            position=position,
            potential=self._get_pair_energy,
            period=period,
        )
        self.epsilon = epsilon
        self.sigma = sigma

    def _get_pair_energy(self, s, r):
        """Return `4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)`."""
        return 4 * s.epsilon * ((s.sigma / r) ** 12 - (s.sigma / r) ** 6)


class DoublingInteraction(Interaction):
    """Interaction with a callable rate over a group."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.target = target

    def _get_laws(self, s):
        """Return a callable rate equal to twice the member counts."""
        name = f'{self.target}.n'
        return [RateLaw(s.target.n, lambda t, values: 2 * values[name])]


class BirthInteraction(Interaction):
    """Stochastic birth of every member of a group."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        rate: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.rate = rate

    def _get_laws(self, s):
        """Return one jump per member."""
        return [JumpLaw(s.rate * s.target.n, {s.target.n: 1})]


class VolatilityInteraction(Interaction):
    """Noise on every member of a group."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.target = target

    def _get_laws(self, s):
        """Return the noise amplitude `n`."""
        return [NoiseLaw(s.target.n, s.target.n)]


class HalvingInteraction(Interaction):
    """Discrete halving of every member of a group."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.target = target

    def _get_laws(self, s):
        """Return the next value `n / 2`."""
        return [UpdateLaw(s.target.n, s.target.n / 2)]


class SingleUpdateFromGroupInteraction(Interaction):
    """Update of a single element from a group, which has no meaning."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        total: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.target = target
        self.total = total

    def _get_laws(self, s):
        """Return an update of the single element."""
        return [UpdateLaw(s.total.n, s.target.n)]


def _lennard_jones_gas(n, seed=0):
    """Return a gas of Lennard-Jones particles on a jittered lattice."""
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(n)))
    grid = np.stack(np.meshgrid(np.arange(side), np.arange(side)), axis=-1).reshape(-1, 2)[:n] * 1.2
    return System(
        elements=[
            GroupElement(
                'gas',
                element=MassPointElement,
                size=n,
                values={'x': grid + rng.normal(0, 0.05, (n, 2)), 'x_d1': rng.normal(0, 0.5, (n, 2))},
            ),
        ],
        interactions=[LennardJonesInteraction('lj', first='gas', second='gas', cutoff=CUTOFF)],
        space=ContinuousSpace(n_dim=2),
    )


RECEIVED: dict[str, object] = {}


class RecorderInteraction(Interaction):
    """Interaction that records the symbols it receives and adds no change."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.target = target

    def _get_laws(self, s):
        """Record the symbols and return a zero rate on the velocity."""
        RECEIVED.update(x=s.target.x, m=s.target.m)
        return [RateLaw(s.target.x, 0)]


@pytest.mark.parametrize(
    ('system', 'form'),
    [
        (_ecology(LogisticGrowthInteraction('growth', target='rabbits')), FormKind.ODE),
        (
            System(
                elements=[AssetElement('stock', price=100)],
                interactions=[GeometricBrownianMotionInteraction('gbm', target='stock')],
            ),
            FormKind.SDE,
        ),
        (
            System(
                elements=[CompartmentElement('I', n=5), CompartmentElement('R')],
                interactions=[RecoveryInteraction('rec', infected='I', recovered='R', stochastic=True)],
            ),
            FormKind.JUMP,
        ),
        (System(elements=[MarketElement('m')], interactions=[CobwebInteraction('c', target='m')]), FormKind.DISCRETE),
        (System(elements=[PopulationElement('a')]), FormKind.ODE),
    ],
)
def test_compile_system_infers_form(system, form):
    """Test that the form follows from the law kinds."""
    assert compile_system(system).form == form


def test_compile_system_mixed_kinds_raises_error():
    """Test that updates and rates in one system are rejected."""
    system = System(
        elements=[MarketElement('m'), PopulationElement('a')],
        interactions=[CobwebInteraction('c', target='m'), ExponentialGrowthInteraction('g', target='a')],
    )
    with pytest.raises(IncompatibleLawsError, match='RateLaw: a; UpdateLaw: m'):
        compile_system(system)


def test_compile_system_names_nested_states_by_path():
    """Test that nested states are named by their path."""
    system = System(systems=[System('meadow', elements=[PopulationElement('rabbits', n=3)])])
    problem = compile_system(system)
    assert problem.states.names == ['meadow/rabbits.n']
    assert problem.initial_state.tolist() == [3.0]


def test_compile_system_couples_subsystems():
    """Test that an interaction couples elements of two subsystems."""
    system = System(
        systems=[
            System('forest', elements=[PopulationElement('foxes', n=2)]),
            System('meadow', elements=[PopulationElement('rabbits', n=5)]),
        ],
        interactions=[PredationInteraction('hunt', predator='forest/foxes', prey='meadow/rabbits', a=1.0, b=0.5)],
    )
    problem = compile_system(system)
    derivative = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    assert derivative.tolist() == [5.0, -10.0]


def test_compile_system_drift_takes_parameters():
    """Test that the drift reflects the parameter vector it receives."""
    problem = compile_system(_ecology(ExponentialGrowthInteraction('growth', target='rabbits', r=1.0)))
    x = problem.initial_state
    assert problem.drift(0.0, x, np.array([1.0])).tolist() == [10.0]
    assert problem.drift(0.0, x, np.array([2.0])).tolist() == [20.0]


def test_compile_system_callable_law_is_evaluated():
    """Test that a callable rate is added to the drift and left out of the equations."""
    problem = compile_system(_ecology(CallableGrowthInteraction('growth', target='rabbits', r=3.0)))
    assert problem.drift(0.0, problem.initial_state, problem.initial_parameters).tolist() == [30.0]
    assert problem.has_callables
    assert problem.equations == {}


def test_compile_system_rates_add_up():
    """Test that two forces on one particle add up."""
    system = System(
        elements=[MassPointElement('p', x=1.0, m=2.0)],
        interactions=[
            ConstantForceInteraction('push', target='p', F=4.0),
            HarmonicPotentialInteraction('trap', target='p', k=6.0),
        ],
        space=ContinuousSpace(n_dim=1),
    )
    problem = compile_system(system)
    assert problem.drift(0.0, problem.initial_state, problem.initial_parameters).tolist() == [0.0, -1.0]


def test_compile_system_keeps_oscillator_equations():
    """Test that the velocity equation of an oscillator is `-k * (x - center) / m`."""
    system = System(
        elements=[MassPointElement('p')],
        interactions=[HarmonicPotentialInteraction('trap', target='p')],
        space=ContinuousSpace(n_dim=1),
    )
    problem = compile_system(system)
    x, v = problem.state_symbols
    k = problem.symbols['trap.k']
    m = problem.symbols['p.m']
    (center,) = problem.symbols['trap.center']
    assert problem.equations['p.x[0]'] == v
    assert (problem.equations['p.x_d1[0]'] + k * (x - center) / m).simplify_full() == 0


def test_compile_system_vector_state_takes_space_dimension():
    """Test that vector states have one component per dimension."""
    problem = compile_system(System(elements=[MassPointElement('p')], space=ContinuousSpace(n_dim=2)))
    assert problem.states.component_names == ['p.x[0]', 'p.x[1]', 'p.x_d1[0]', 'p.x_d1[1]']


def test_compile_system_vector_state_without_space_raises_error():
    """Test that a vector state needs a space."""
    with pytest.raises(SpaceError, match='must have a space'):
        compile_system(System(elements=[MassPointElement('p')]))


@pytest.mark.parametrize('m', [-2.0, 0.0, 'm'])
def test_compile_system_value_outside_domain_raises_error(m):
    """Test that a parameter outside its domain is rejected."""
    system = System(elements=[MassPointElement('p', m=m)], space=ContinuousSpace(n_dim=1))
    with pytest.raises(ValidationError, match=r'p\.m'):
        compile_system(system)


def test_compile_system_wrong_shape_raises_error():
    """Test that a vector value of the wrong size is rejected."""
    system = System(elements=[MassPointElement('p', x=[1, 2, 3])], space=ContinuousSpace(n_dim=2))
    with pytest.raises(ValidationError, match='shape'):
        compile_system(system)


def test_compile_system_unknown_role_raises_error():
    """Test that a role must name an element."""
    system = _ecology(LogisticGrowthInteraction('growth', target='foxes'))
    with pytest.raises(RoleError, match="'target' to 'foxes'"):
        compile_system(system)


def test_compile_system_duplicate_label_raises_error():
    """Test that siblings must have distinct labels."""
    system = System(elements=[PopulationElement('a'), PopulationElement('a')])
    with pytest.raises(LabelError, match="'a'"):
        compile_system(system)


def test_compile_system_invalid_label_raises_error():
    """Test that labels must be words."""
    with pytest.raises(LabelError, match='not valid'):
        compile_system(System(elements=[PopulationElement('a.b')]))


def test_compile_system_undeclared_symbol_raises_error():
    """Test that laws may only use declared symbols."""
    with pytest.raises(ValidationError, match='symbols k'):
        compile_system(_ecology(UndeclaredInteraction('bad', target='rabbits')))


def test_compile_system_double_update_raises_error():
    """Test that a state takes one update law."""
    system = System(elements=[MarketElement('m')], interactions=[DoubleUpdateInteraction('bad', target='m')])
    with pytest.raises(ValidationError, match='second time'):
        compile_system(system)


def test_compile_system_jump_stoichiometry():
    """Test that jump changes become rows of the stoichiometry matrix."""
    system = System(
        elements=[CompartmentElement('S', n=9), CompartmentElement('I', n=1)],
        interactions=[InfectionInteraction('inf', susceptible='S', infected='I', beta=0.5, stochastic=True)],
    )
    problem = compile_system(system)
    assert problem.stoichiometry.tolist() == [[-1.0, 1.0]]
    assert problem.propensities(0.0, problem.initial_state, problem.initial_parameters).tolist() == [4.5]


@pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend])
def test_compile_variadic_role_sums_any_number_of_inputs(backend):
    """Test that a variadic role binds a sequence of elements that laws read as a tuple of namespaces."""
    problem = compile_system(_summed(['a', 'b', 'c']), backend=backend())
    state = problem.update(0.0, problem.initial_state, problem.initial_parameters)
    assert problem.states.read(state, 'target.x') == 7.0
    assert problem.interactions[0].roles['inputs'] == ('a', 'b', 'c')


@pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend])
def test_compile_variadic_member_reference_reads_one_member(backend):
    """Test that a member reference of a variadic role reads the states of that member of a group."""
    problem = compile_system(_summed(['hub', 'nodes[3]'], weights=[0.0, 1.0]), backend=backend())
    state = problem.update(0.0, problem.initial_state, problem.initial_parameters)
    assert problem.states.read(state, 'target.x') == 3.0


@pytest.mark.parametrize(
    ('inputs', 'match'),
    [
        (['a', 'nodes[12]'], r"sum .*'nodes\[12\]'.*'inputs'.*10 members"),
        (['a', 'missing'], "'missing'"),
        ('a', 'Bind a list'),
    ],
)
def test_compile_variadic_role_with_wrong_entry_raises_error(inputs, match):
    """Test that a variadic entry naming a missing member or no element raises a role error."""
    with pytest.raises(RoleError, match=match):
        compile_system(_summed(inputs))


def test_compile_variadic_group_entry_gives_symbolic_laws_no_states():
    """Test that a variadic entry naming a whole group, which binds its agents, gives symbolic laws no states."""
    with pytest.raises(RoleElementError, match="reads the state 'x'"):
        compile_system(_summed(['a', 'nodes']))


def test_compile_plain_role_binds_one_group_member():
    """Test that a plain role bound to a member reference reads that member like a variadic entry does."""
    plain = compile_system(_hub_and_sites(SiteDriveInteraction('link', hub='hub', site='sites[3]')))
    variadic = compile_system(_hub_and_sites(SitesDriveInteraction('link', hub='hub', sites=['sites[3]'])))
    for problem in (plain, variadic):
        drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
        assert problem.states.read(drift, 'hub.y') == 3.0
    assert plain.interactions[0].roles['site'] == 'sites[3]'


def test_compile_plain_role_past_group_end_raises_error():
    """Test that a plain role bound past the end of a group names the interaction, role, reference, and size."""
    with pytest.raises(RoleError, match=r"link binds 'sites\[7\]' to its role 'site', but sites holds 5 members"):
        compile_system(_hub_and_sites(SiteDriveInteraction('link', hub='hub', site='sites[7]')))


def test_compile_per_entry_parameter_with_wrong_length_raises_error():
    """Test that a per-entry parameter of another length than its role names the interaction and both lengths."""
    with pytest.raises(ValidationError, match=r"sum .*'weights' 2 values.*'inputs' binds 3 entries"):
        compile_system(_summed(['a', 'b', 'c'], weights=[1.0, 2.0]))


def test_compile_per_entry_parameter_has_one_component_per_entry():
    """Test that a per-entry parameter is laid out with one addressable component per entry."""
    problem = compile_system(_summed(['a', 'b', 'c'], weights=[1.0, 2.0, 3.0]))
    assert problem.parameters.component_names == ['sum.weights[0]', 'sum.weights[1]', 'sum.weights[2]']


@pytest.mark.benchmark
def test_compile_system_time_grows_linearly_with_interactions():
    """Test that compiling 800 one-link interactions takes at most 2.5 times as long as compiling 400."""
    assert _compile_time(800) <= 2.5 * _compile_time(400)


@pytest.mark.benchmark
def test_compile_group_time_does_not_grow_with_size():
    """Test that compiling 100000 members takes no longer than 10 members within a factor of 2."""

    def compile_time(n):
        start = time.perf_counter()
        compile_system(
            System(
                elements=[GroupElement('gas', element=MassPointElement, size=n)],
                space=ContinuousSpace(extent=(1.0, 1.0), boundary='reflecting'),
            ),
        )
        return time.perf_counter() - start

    compile_time(10)
    assert compile_time(100_000) <= 2 * compile_time(10) + 0.05


def test_compile_group_names_states_by_group_path():
    """Test that a group state is named by its group path with the group shape."""
    problem = compile_system(
        System(elements=[GroupElement('gas', element=MassPointElement, size=500)], space=ContinuousSpace(n_dim=2)),
    )
    assert problem.states.entry('gas.x').shape == (500, 2)
    assert problem.equations['gas.x[i, 0]'] == problem.symbols['gas.x_d1'][0]


def test_compile_pair_forces_match_brute_force_sum():
    """Test that pair forces equal the sum over members closer than the cutoff."""
    problem = compile_system(_lennard_jones_gas(60))
    x, p = problem.initial_state, problem.initial_parameters
    positions = problem.states.read(x, 'gas.x')
    accelerations = problem.states.read(problem.drift(0.0, x, p), 'gas.x_d1')
    expected = np.zeros_like(positions)
    for i in range(len(positions)):
        for j in range(len(positions)):
            separation = positions[i] - positions[j]
            r = np.linalg.norm(separation)
            if i != j and r < CUTOFF:
                expected[i] += 24 * (2 / r**13 - 1 / r**7) * separation / r
    assert np.allclose(accelerations, expected, rtol=1e-10, atol=1e-10)


def test_compile_pair_forces_between_two_groups():
    """Test that a pair interaction between two groups attracts both beyond the potential minimum."""
    system = System(
        elements=[
            GroupElement('a', element=MassPointElement, size=1, values={'x': [[0.0, 0.0]]}),
            GroupElement('b', element=MassPointElement, size=1, values={'x': [[1.5, 0.0]]}),
        ],
        interactions=[LennardJonesInteraction('lj', first='a', second='b', cutoff=CUTOFF)],
        space=ContinuousSpace(n_dim=2),
    )
    problem = compile_system(system)
    drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    a, b = problem.states.read(drift, 'a.x_d1'), problem.states.read(drift, 'b.x_d1')
    assert a[0, 0] > 0
    assert np.allclose(a, -b)


def test_compile_pair_forces_use_minimum_image():
    """Test that periodic pair interactions use minimum image distances."""
    system = System(
        elements=[GroupElement('gas', element=MassPointElement, size=2, values={'x': [[0.1, 0.5], [9.6, 0.5]]})],
        interactions=[LennardJonesInteraction('lj', first='gas', second='gas', cutoff=CUTOFF, period=10.0)],
        space=ContinuousSpace(n_dim=2),
    )
    problem = compile_system(system)
    accelerations = problem.states.read(
        problem.drift(0.0, problem.initial_state, problem.initial_parameters),
        'gas.x_d1',
    )
    assert accelerations[0, 0] > 0
    assert accelerations[1, 0] < 0


def test_compile_reflecting_box_keeps_ode_form():
    """Test that a reflecting box adds boundary laws without changing the form."""
    problem = compile_system(
        System(
            elements=[GroupElement('gas', element=MassPointElement, size=4)],
            space=ContinuousSpace(extent=(1.0, 1.0), boundary='reflecting'),
        ),
    )
    assert problem.form.name == 'ODE'
    assert problem.boundary is not None


@pytest.mark.parametrize(('role', 'wrapped'), [('position', 0.25), ('site', 0.25), (None, 1.25)])
def test_compile_periodic_box_wraps_places_by_their_role(role, wrapped):
    """Test that a periodic box wraps a vector state with the role position or site, and no other."""

    class PlacedElement(Element):
        """Element with a vector state of the role under test."""

        def __init__(self, label, visual=None, *, x: Annotated[object, StateSpec(vector=True, role=role)] = 0.0):
            super().__init__(label, visual=visual)
            self.x = x

        def _get_laws(self, s):
            """Return the rest of the state."""
            return [RateLaw(s.x, 0.0 * s.x)]

    problem = compile_system(
        System(
            elements=[GroupElement('gas', element=PlacedElement, size=1, values={'x': [[1.25, 0.5]]})],
            space=ContinuousSpace(extent=(1.0, 1.0), boundary='periodic'),
        ),
    )
    state = problem.initial_state
    if problem.boundary is not None:
        state = problem.boundary(0.0, state, problem.initial_parameters)
    assert problem.states.read(state, 'gas.x')[0, 0] == pytest.approx(wrapped)


def test_compile_periodic_box_wraps_positions():
    """Test that a periodic box wraps a position that left the box."""
    problem = compile_system(
        System(
            elements=[
                GroupElement(
                    'gas',
                    element=MassPointElement,
                    size=1,
                    values={'x': [[1.25, 0.5]], 'x_d1': [[1.0, 0.0]]},
                ),
            ],
            space=ContinuousSpace(extent=(1.0, 1.0), boundary='periodic'),
        ),
    )
    state = problem.boundary(0.0, problem.initial_state, problem.initial_parameters)
    assert np.allclose(problem.states.read(state, 'gas.x'), [[0.25, 0.5]])
    assert np.allclose(problem.states.read(state, 'gas.x_d1'), [[1.0, 0.0]])


def test_compile_group_callable_rate():
    """Test that a callable law over a group returns one value per member."""
    problem = compile_system(
        System(
            elements=[GroupElement('herds', element=PopulationElement, size=3, values={'n': [1.0, 2.0, 3.0]})],
            interactions=[DoublingInteraction('d', target='herds')],
        ),
    )
    assert problem.drift(0.0, problem.initial_state, problem.initial_parameters).tolist() == [2.0, 4.0, 6.0]
    assert problem.has_callables


def test_compile_group_jumps_have_one_channel_per_member():
    """Test that a jump law over a group has one channel per member."""
    problem = compile_system(
        System(
            elements=[GroupElement('herds', element=PopulationElement, size=3, values={'n': [1.0, 2.0, 3.0]})],
            interactions=[BirthInteraction('birth', target='herds')],
        ),
    )
    assert problem.propensities(0.0, problem.initial_state, problem.initial_parameters).tolist() == [1.0, 2.0, 3.0]
    assert problem.stoichiometry.tolist() == np.eye(3).tolist()


def test_compile_group_noise_has_one_process_per_member():
    """Test that a noise law over a group drives every member with its own process."""
    problem = compile_system(
        System(
            elements=[GroupElement('herds', element=PopulationElement, size=3, values={'n': [1.0, 2.0, 3.0]})],
            interactions=[VolatilityInteraction('vol', target='herds')],
        ),
    )
    assert problem.noise_rows.tolist() == [0, 1, 2]
    assert problem.diffusion(0.0, problem.initial_state, problem.initial_parameters).tolist() == [1.0, 2.0, 3.0]


def test_compile_group_updates_every_member():
    """Test that a discrete update over a group updates every member."""
    problem = compile_system(
        System(
            elements=[GroupElement('herds', element=PopulationElement, size=2, values={'n': [4.0, 8.0]})],
            interactions=[HalvingInteraction('half', target='herds')],
        ),
    )
    assert problem.update(0.0, problem.initial_state, problem.initial_parameters).tolist() == [2.0, 4.0]
    assert problem.equations['herds.n[i]'] == problem.symbols['herds.n'] / 2


def test_compile_update_of_single_element_from_group_raises_error():
    """Test that an update cannot set a single element from a group."""
    system = System(
        elements=[GroupElement('herds', element=PopulationElement, size=2), PopulationElement('total')],
        interactions=[SingleUpdateFromGroupInteraction('sum', target='herds', total='total')],
    )
    with pytest.raises(ValidationError, match='single element from a group'):
        compile_system(system)


def test_compile_boundary_twice_raises_error():
    """Test that a state takes one boundary law."""

    class WrapInteraction(Interaction):
        """Wrap interaction."""

        def __init__(
            self,
            label: str,
            *,
            target: Annotated[object, RoleSpec()],
        ) -> None:
            super().__init__(label)
            self.target = target

        def _get_laws(self, s):
            """Return the laws of the wrap interaction."""
            return [BoundaryLaw(s.target.x[0], mod(s.target.x[0], 1))]

    system = System(
        elements=[GroupElement('gas', element=MassPointElement, size=2)],
        interactions=[WrapInteraction('wrap', target='gas')],
        space=ContinuousSpace(extent=(1.0,), boundary='reflecting'),
    )
    with pytest.raises(ValidationError, match='second time'):
        compile_system(system)


def test_compile_system_without_processes_keeps_identity_columns():
    """Test that noise laws that name no process get one process per amplitude and no correlated block."""

    class WalkerElement(Element):
        """Walker element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            x: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.x = x

        def _get_laws(self, s):
            """Return the laws of the walker element."""
            return [NoiseLaw(s.x, 1.0), NoiseLaw(s.x, 2.0)]

    problem = compile_system(System(elements=[GroupElement('walkers', element=WalkerElement, size=3)]))
    np.testing.assert_array_equal(problem.noise_columns, np.arange(6))
    assert problem.n_processes == 6
    assert problem.noise_blocks == ()


def test_compile_system_shares_local_process():
    """Test that two laws of one element on one local process share one column."""
    problem = compile_system(System(elements=[PairElement('pair')]))
    np.testing.assert_array_equal(problem.noise_columns, [0, 0])
    assert problem.n_processes == 1


def test_compile_system_gives_members_their_own_local_process():
    """Test that a local process of a group law gives every member its own process."""
    problem = compile_system(System(elements=[GroupElement('pairs', element=PairElement, size=3)]))
    np.testing.assert_array_equal(problem.noise_columns, [0, 1, 2, 0, 1, 2])


def test_compile_system_factors_covariance():
    """Test that a `WienerProcessSpec` with a covariance lowers to one block whose factor reproduces it."""
    problem = compile_system(_correlated([[1.0, 0.6], [0.6, 2.0]]))
    (block,) = problem.noise_blocks
    assert (block.start, block.size) == (0, 2)
    np.testing.assert_allclose(block.factor @ block.factor.T, [[1.0, 0.6], [0.6, 2.0]])


def test_compile_system_factors_singular_covariance():
    """Test that perfectly correlated processes compile with an eigenvalue factor."""
    (block,) = compile_system(_correlated([[1.0, 1.0], [1.0, 1.0]])).noise_blocks
    np.testing.assert_allclose(block.factor @ block.factor.T, np.ones((2, 2)), atol=1e-12)


def test_stratonovich_correction_covers_cross_terms():
    """Test that a shared Stratonovich process adds the cross terms `C G_kl dG_ij/dx_k` to the drift."""
    problem = compile_system(System(elements=[PairElement('pair', conventions=('stratonovich', 'stratonovich'))]))
    drift = problem.drift(0.0, np.asarray([2.0, 3.0]), problem.initial_parameters)
    np.testing.assert_allclose(drift, [0.5 * 0.5 * 0.5 * 2.0, 0.5 * 0.5 * 0.5 * 3.0])


def test_stratonovich_correction_of_several_laws_adds_up():
    """Test that two Stratonovich laws with independent processes on one state compile and add their corrections."""

    class TwiceElement(Element):
        """Twice element."""

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
            """Return the laws of the twice element."""
            return [
                NoiseLaw(s.x, s.x, convention='stratonovich'),
                NoiseLaw(s.x, 2 * s.x, convention='stratonovich'),
            ]

    problem = compile_system(System(elements=[TwiceElement('twice')]))
    drift = problem.drift(0.0, np.asarray([2.0]), problem.initial_parameters)
    assert drift[0] == pytest.approx(0.5 * 2.0 + 0.5 * 4 * 2.0)


def test_stratonovich_callable_amplitude_raises_an_error_naming_the_drift_correction():
    """Test that a Stratonovich noise law with a callable amplitude raises an error instead of running as Itô."""

    class CallableNoiseElement(Element):
        """Element whose noise amplitude is a callable of time and values."""

        def __init__(self, label: str, *, x: Annotated[object, StateSpec()] = 1.0) -> None:
            super().__init__(label)
            self.x = x

        def _get_laws(self, s):
            """Return a Stratonovich noise law with a callable amplitude."""
            return [NoiseLaw(s.x, lambda t, values: values['noisy.x'], convention='stratonovich')]

    with pytest.raises(ValidationError, match='drift correction'):
        compile_system(System(elements=[CallableNoiseElement('noisy')]))


def test_components_receive_sage_symbols_and_vectors():
    """Test that scalar symbols are Sage expressions with their domains and vector states are Sage vectors."""
    system = System(
        elements=[MassPointElement('p')],
        interactions=[RecorderInteraction('record', target='p')],
        space=ContinuousSpace(n_dim=3),
    )
    compile_system(system)
    assert isinstance(RECEIVED['x'], FreeModuleElement)
    assert len(RECEIVED['x']) == 3
    assert all(isinstance(component, Expression) and component.is_symbol() for component in RECEIVED['x'])
    assert RECEIVED['m'].is_real()
    assert domain_of(RECEIVED['m']) is RR


@pytest.mark.benchmark
def test_small_system_drift_is_fast():
    """Test that the NumPy drift of 20 logistic populations takes at most 250 microseconds and is correct."""
    system = System(
        elements=[PopulationElement(f'p{index}', n=1.0 + index) for index in range(20)],
        interactions=[LogisticGrowthInteraction(f'g{index}', target=f'p{index}', r=1.0, K=100) for index in range(20)],
    )
    problem = compile_system(system, backend=NumpyBackend())
    x, p = problem.initial_state, problem.initial_parameters
    expected = x * (1 - x / 100)
    assert problem.drift(0.0, x, p) == pytest.approx(expected)
    repeats = 500
    start = time.perf_counter()
    for _ in range(repeats):
        problem.drift(0.0, x, p)
    assert (time.perf_counter() - start) / repeats <= 250e-6
