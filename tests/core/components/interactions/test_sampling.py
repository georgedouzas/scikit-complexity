"""Test the `_sampling` module of `skcomplex.core.components.interactions`."""

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import cos, pi
from scipy.special import iv

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import SamplingInteraction
from skcomplex.core.components.laws.utils import exterior_derivative, neighbors
from skcomplex.core.components.spaces import (
    BallNeighborhood,
    CellComplexSpace,
    CofaceNeighborhood,
    DiscreteTime,
    LatticeSpace,
)
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import MatrixGroupSpec, ParameterSpec, PartnerSpec, StateSpec, VisualSpec
from skcomplex.systems.physics.common.fields import HolonomyScoreSpec, WilsonSamplingInteraction

SIDE = 16
ISOSPIN = MatrixGroupSpec('SU', 2)
COLOR = MatrixGroupSpec('SU', 3)


class SiteElement(Element):
    """Site with a value `s`."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        s: Annotated[float, StateSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.s = s


class AngleLinkElement(Element):
    """Link with an angle `theta` and a coupling `beta`."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        theta: Annotated[float, StateSpec(period=2 * pi)] = 0.0,
        beta: Annotated[float, ParameterSpec()] = 2.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.theta = theta
        self.beta = beta


def _wilson(namespace):
    """Return the local energy of a link with the plaquettes that contain it."""
    theta = namespace.target.theta
    return -namespace.target.beta * neighbors(cos(exterior_derivative(theta)), over=CofaceNeighborhood())


def _link_system(shape, sampling, values):
    """Return a system of a group on the edges of the complex of a lattice, sampled by an interaction."""
    space = CellComplexSpace(LatticeSpace(shape)).prepare()
    links = GroupElement('links', element=AngleLinkElement, space=space, on='edges', values=values)
    return space, CompositeSystem(elements=[links], interactions=[sampling], time=DiscreteTime())


def _ising(namespace):
    """Return the local energy of a spin with its four neighbours."""
    return -namespace.target.s * neighbors(namespace.target.other.s)


def _final(sampling, until, initial, seed=0):
    """Return the values of the sites after some sweeps."""
    sites = GroupElement(
        'sites',
        element=SiteElement,
        space=LatticeSpace((SIDE, SIDE)),
        values={'s': initial},
        on='sites',
    )
    system = CompositeSystem(elements=[sites], interactions=[sampling], time=DiscreteTime())
    return simulate(system, until=until, random_state=seed).states[-1]


def _spins(seed=1):
    """Return random spins of the sites."""
    return np.random.default_rng(seed).choice([-1.0, 1.0], SIDE * SIDE)


def test_ising_magnetization_orders_well_below_the_critical_temperature():
    """Test that the Ising energy sampled at intensity 1 orders with absolute magnetization above 0.9."""
    sampling = SamplingInteraction('ising', target='sites', state='s', objective=_ising, intensity=1.0)
    assert abs(_final(sampling, 300, _spins()).mean()) > 0.9


def test_ising_magnetization_vanishes_well_above_the_critical_temperature():
    """Test that the Ising energy sampled at intensity 1 / 5 stays disordered."""
    sampling = SamplingInteraction(
        'ising',
        target='sites',
        state='s',
        objective=_ising,
        intensity=1 / 5.0,
        rule='glauber',
    )
    assert abs(_final(sampling, 100, np.ones(SIDE * SIDE)).mean()) < 0.3


def test_gaussian_moves_sample_the_boltzmann_distribution():
    """Test that Gaussian moves in a harmonic energy sample values of variance equal to one over the intensity."""
    sampling = SamplingInteraction(
        'harmonic',
        target='sites',
        state='s',
        objective=lambda n: n.target.s**2 / 2,
        move='gaussian',
        width=1.5,
        intensity=1 / 2.0,
    )
    values = np.concatenate([_final(sampling, 200, np.zeros(SIDE * SIDE), seed) for seed in range(4)])
    assert values.var() == pytest.approx(2.0, rel=0.1)


def test_function_moves_propose_the_values_it_returns():
    """Test that a function move that proposes random spins keeps the ordered Ising phase and melts it above Tc."""

    def redraw(values, rng):
        return rng.choice([-1.0, 1.0], values.size)

    cold, hot = (
        _final(
            SamplingInteraction(
                'ising',
                target='sites',
                state='s',
                objective=_ising,
                move=redraw,
                intensity=1 / temperature,
            ),
            100,
            np.ones(SIDE * SIDE),
        )
        for temperature in (1.0, 5.0)
    )
    assert set(np.unique(cold)) | set(np.unique(hot)) <= {-1.0, 1.0}
    assert abs(cold.mean()) > 0.9
    assert abs(hot.mean()) < 0.3


def _magnetization(move, temperature, seed):
    """Return the mean absolute magnetization of the Ising system over sweeps after equilibration."""
    sampling = SamplingInteraction(
        'ising',
        target='sites',
        state='s',
        objective=_ising,
        intensity=1 / temperature,
        move=move,
    )
    sites = GroupElement(
        'sites',
        element=SiteElement,
        space=LatticeSpace((SIDE, SIDE)),
        values={'s': _spins(seed)},
        on='sites',
    )
    system = CompositeSystem(elements=[sites], interactions=[sampling], time=DiscreteTime())
    states = simulate(system, until=300, random_state=seed).states
    return float(np.abs(states[100:].mean(axis=1)).mean())


@pytest.mark.parametrize('temperature', [2.0, 3.0])
def test_cluster_moves_sample_the_magnetization_of_metropolis_moves(temperature):
    """Test that Wolff cluster moves of the Ising system give the magnetization that Metropolis moves give."""
    cluster = np.mean([_magnetization('cluster', temperature, seed) for seed in range(2)])
    metropolis = np.mean([_magnetization('flip', temperature, seed) for seed in range(2)])
    assert cluster == pytest.approx(metropolis, abs=0.04)


def test_cluster_moves_reject_an_energy_that_the_flip_changes():
    """Test that cluster moves of an energy with an external field raise an error."""
    sampling = SamplingInteraction(
        'ising',
        target='sites',
        state='s',
        objective=lambda n: _ising(n) - 0.5 * n.target.s,
        move='cluster',
    )
    with pytest.raises(ValidationError, match='changes when every value flips'):
        _final(sampling, 1, _spins())


@pytest.mark.parametrize(
    ('arguments', 'match'),
    [
        ({'objective': _ising, 'move': 'swap'}, 'Unknown move'),
        ({'objective': 1.0}, 'function of the namespace'),
        ({'objective': _ising, 'rule': 'greedy'}, 'Unknown sampler'),
        ({'objective': _ising, 'schedule': 'synchronous'}, 'Unknown Monte Carlo schedule'),
    ],
)
def test_check_rejects_moves_rules_and_energies_it_cannot_take(arguments, match):
    """Test that an unknown move, rule, or schedule, or an objective that is not a function, raise errors."""
    with pytest.raises(ValidationError, match=match):
        SamplingInteraction('sampling', target='sites', state='s', **arguments).prepare()


class CandidateElement(Element):
    """Candidate with a real value."""

    def __init__(self, label: str, x: Annotated[float, StateSpec()] = 3.0) -> None:
        super().__init__(label)
        self.x = x


def _quadratic_energy(s: SimpleNamespace) -> object:
    """Return the energy of a candidate, which reads no neighbours."""
    return s.target.x**2


def _geometric(t: float) -> float:
    """Return the factor that falls geometrically to 1e-3 of its start in 500 steps."""
    return (1e-3) ** (t / 500)


def test_run_cooling_reduces_accepted_moves() -> None:
    """Test that cooling lowers the share of proposals accepted and takes a quadratic energy to its minimum."""
    anneal = SamplingInteraction(
        'anneal',
        target='candidates',
        state='x',
        objective=_quadratic_energy,
        move='gaussian',
        width=0.5,
        intensity=1 / 5.0,
        schedule='colored',
        neighborhood=BallNeighborhood(metric='manhattan', n_dim=1),
        cooling=_geometric,
    )
    candidates = GroupElement('candidates', element=CandidateElement, space=LatticeSpace((200,)), on='sites')
    system = CompositeSystem(elements=[candidates], interactions=[anneal], time=DiscreteTime())
    frames = simulate(system, until=500, random_state=0).get_values('candidates.x')
    accepted = np.mean(frames[1:] != frames[:-1], axis=1)
    assert accepted[:50].mean() > 0.8
    assert accepted[-50:].mean() < 0.2
    assert np.mean(frames[-1] ** 2) < 0.01


def test_check_cooling_not_callable() -> None:
    """Test that a cooling that is no function is refused."""
    with pytest.raises(ValidationError, match='cooling'):
        SamplingInteraction('a', target='g', state='x', objective=_quadratic_energy, cooling=0.5).prepare()


def test_two_dimensional_u1_reproduces_the_exact_plaquette():
    """Test that 4000 sweeps after 1000 of transient on a 32 by 32 lattice give `I_1(2) / I_0(2)` within 0.01."""
    sampling = SamplingInteraction(
        'wilson',
        target='links',
        state='theta',
        objective=_wilson,
        move='rotate',
        width=1.5,
        schedule='colored',
    )
    initial = np.random.default_rng(0).uniform(-np.pi, np.pi, 2048)
    space, system = _link_system((32, 32), sampling, {'theta': initial})
    trajectory = simulate(system, until=5000, random_state=3)
    states = trajectory.get_values('links.theta')
    plaquettes = np.cos(space.incidences_[1] @ states.T).mean(axis=0)
    assert abs(plaquettes[1001:].mean() - iv(1, 2) / iv(0, 2)) < 0.01


def test_z3_heat_bath_is_reproducible_with_a_seed():
    """Test that a Z_3 link system sampled by heat bath gives identical link values at every frame for one seed."""
    values = [0.0, 2 * np.pi / 3, 4 * np.pi / 3]
    sampling = SamplingInteraction(
        'wilson',
        target='links',
        state='theta',
        objective=_wilson,
        move='random-value',
        values=values,
        rule='heat-bath',
        schedule='colored',
    )
    initial = np.random.default_rng(1).choice(values, 72)
    frames = [
        simulate(_link_system((6, 6), sampling, {'theta': initial})[1], until=20, random_state=seed).get_values(
            'links.theta',
        )
        for seed in (5, 5, 6)
    ]
    assert np.array_equal(frames[0], frames[1])
    assert not np.array_equal(frames[0], frames[2])
    assert set(np.round(np.mod(frames[0][-1], 2 * np.pi), 6)) <= set(np.round(values, 6))


def test_colored_schedule_on_an_odd_lattice_updates_links_that_share_no_face():
    """Test that no two links of a class of the complex of a lattice with 15 sites along one axis share a plaquette."""
    sampling = SamplingInteraction(
        'wilson',
        target='links',
        state='theta',
        objective=_wilson,
        move='rotate',
        schedule='colored',
    )
    space, system = _link_system((15, 8), sampling, {})
    classes = PassCompiler().prepare(system).run().problem_.lattice[0].classes
    assert sorted(np.concatenate(classes).tolist()) == list(range(240))
    faces = abs(space.incidences_[1]).tocsr()
    for links in classes:
        assert faces[:, links].sum(axis=1).max() == 1


def test_sampling_of_links_rejects_a_cluster_move():
    """Test that cluster moves take no objective over cofaces."""
    sampling = SamplingInteraction(
        'wilson',
        target='links',
        state='theta',
        objective=_wilson,
        move='cluster',
        schedule='colored',
    )
    _, system = _link_system((4, 4), sampling, {})
    with pytest.raises(ValidationError, match='cluster moves'):
        PassCompiler().prepare(system).run()


def test_sampling_of_links_reads_only_the_moved_state_around_a_face():
    """Test that an objective over cofaces reads the oriented sum of the moved state alone."""
    sampling = SamplingInteraction(
        'wilson',
        target='links',
        state='theta',
        objective=lambda n: neighbors(cos(exterior_derivative(2 * n.target.theta)), over=CofaceNeighborhood()),
        move='rotate',
        schedule='colored',
    )
    _, system = _link_system((4, 4), sampling, {})
    with pytest.raises(ValidationError, match='only the moved state'):
        PassCompiler().prepare(system).run()


def test_sampling_of_links_only_sums_over_cofaces():
    """Test that another reduction than a sum over the cofaces raises an error naming it."""
    sampling = SamplingInteraction(
        'wilson',
        target='links',
        state='theta',
        objective=lambda n: neighbors(
            cos(exterior_derivative(n.target.theta)),
            over=CofaceNeighborhood(),
            reduce='mean',
        ),
        move='rotate',
        schedule='colored',
    )
    _, system = _link_system((4, 4), sampling, {})
    with pytest.raises(ValidationError, match='only sums'):
        PassCompiler().prepare(system).run()


class IsospinLinkElement(Element):
    """Link with a matrix of SU(2) and a coupling `beta`."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        U: Annotated[object, StateSpec(group=ISOSPIN)] = tuple(ISOSPIN.identity),
        beta: Annotated[float, ParameterSpec()] = 2.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.U = U
        self.beta = beta


class ColorLinkElement(Element):
    """Link with a matrix of SU(3) and a coupling `beta`."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        U: Annotated[object, StateSpec(group=COLOR)] = tuple(COLOR.identity),
        beta: Annotated[float, ParameterSpec()] = 4.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.U = U
        self.beta = beta


class HolonomySamplingInteraction(WilsonSamplingInteraction):
    """Sampling of matrix links by any holonomy score, the sampler of the Wilson action."""

    def __init__(self, label, *, target, state, objective, **settings):
        SamplingInteraction.__init__(self, label, target=target, state=state, objective=objective, **settings)


def _trace_action(holonomies, beta):
    """Return the Wilson action of the holonomies, `-(beta / N) Re tr`."""
    return -beta / holonomies.shape[-1] * np.trace(holonomies, axis1=-2, axis2=-1).real


def _matrix_system(shape, group, sampling, seed=0, element=IsospinLinkElement):
    """Return a system of matrix links of a lattice complex from random links, sampled by an interaction."""
    space = CellComplexSpace(LatticeSpace(shape)).prepare()
    initial = group.get_components_of(group.get_uniform_of(np.random.default_rng(seed), space.incidences_[0].shape[0]))
    links = GroupElement('links', element=element, space=space, on='edges', values={'U': initial})
    return space, CompositeSystem(elements=[links], interactions=[sampling], time=DiscreteTime())


def _matrix_sampling(group, **arguments):
    """Return the sampling of matrix links by the Wilson action."""
    settings = {'move': 'rotate', 'width': 0.6, 'schedule': 'colored', **arguments}
    hits = settings.pop('hits', 4)
    score = HolonomyScoreSpec(_trace_action, group, hits=hits)
    return HolonomySamplingInteraction('wilson', target='links', state='U', objective=score, **settings)


def _plaquettes(space, group, values):
    """Return the mean of `Re tr U_p / N` over the plaquettes at every frame."""
    loops = [
        group.get_holonomy_of(group.get_matrices_of(frame), space.boundaries_, space.boundary_signs_)
        for frame in values
    ]
    return np.trace(np.stack(loops), axis1=-2, axis2=-1).real.mean(axis=1) / group.degree


def test_two_dimensional_su2_reproduces_the_exact_plaquette():
    """Test that 300 sweeps after 100 of transient on a 16 by 16 lattice give `I_2(2) / I_1(2)` within 0.01."""
    space, system = _matrix_system((16, 16), ISOSPIN, _matrix_sampling(ISOSPIN))
    values = simulate(system, until=400, random_state=3).get_values('links.U')
    assert abs(_plaquettes(space, ISOSPIN, values)[101:].mean() - iv(2, 2) / iv(1, 2)) < 0.01


def test_two_dimensional_su3_reproduces_the_one_plaquette_integral():
    """Test that the SU(3) mean plaquette at `beta = 4` is the integral over the Haar measure within 0.01."""
    angles = np.linspace(-np.pi, np.pi, 300, endpoint=False)
    first, second = np.meshgrid(angles, angles)
    phases = np.exp(1j * np.stack([first, second, -first - second]))
    vandermonde = np.prod([np.abs(phases[i] - phases[j]) ** 2 for i, j in ((0, 1), (0, 2), (1, 2))], axis=0)
    traces = phases.sum(axis=0).real / 3
    weights = vandermonde * np.exp(4.0 * traces)
    exact = (weights * traces).sum() / weights.sum()
    sampling = _matrix_sampling(COLOR)
    space, system = _matrix_system((12, 12), COLOR, sampling, element=ColorLinkElement)
    values = simulate(system, until=300, random_state=4).get_values('links.U')
    assert abs(_plaquettes(space, COLOR, values)[101:].mean() - exact) < 0.01


@pytest.mark.parametrize('schedule', ['colored', 'random-sequential'])
def test_matrix_links_stay_in_the_group(schedule):
    """Test that SU(3) links stay unitary of unit determinant within 1e-12 after 20 sweeps."""
    _, system = _matrix_system((4, 4), COLOR, _matrix_sampling(COLOR, schedule=schedule), element=ColorLinkElement)
    values = simulate(system, until=20, random_state=0).get_values('links.U')
    matrices = COLOR.get_matrices_of(values[-1])
    assert np.abs(matrices @ np.conj(np.swapaxes(matrices, -1, -2)) - np.eye(3)).max() < 1e-12
    assert np.abs(np.linalg.det(matrices) - 1.0).max() < 1e-12


def test_infinite_intensity_never_raises_the_action_of_matrix_links():
    """Test that at an infinite intensity the total plaquette never falls, since links of a class share no face."""
    sampling = _matrix_sampling(ISOSPIN, intensity=float('inf'), move='random-value', hits=1)
    space, system = _matrix_system((5, 4), ISOSPIN, sampling)
    values = simulate(system, until=10, random_state=1).get_values('links.U')
    plaquettes = _plaquettes(space, ISOSPIN, values)
    assert np.all(np.diff(plaquettes) >= -1e-12)
    assert plaquettes[-1] > plaquettes[0]


def test_matrix_sampling_is_reproducible_with_a_seed_and_takes_function_moves():
    """Test that one seed repeats a Glauber run with a function move, and another seed does not."""

    def move(matrices, random):
        return ISOSPIN.get_near_identity_of(random, 0.3, len(matrices)) @ matrices

    runs = []
    for seed in (5, 5, 6):
        _, system = _matrix_system((4, 4), ISOSPIN, _matrix_sampling(ISOSPIN, move=move, rule='glauber'))
        runs.append(simulate(system, until=5, random_state=seed).get_values('links.U'))
    assert np.array_equal(runs[0], runs[1])
    assert not np.array_equal(runs[0], runs[2])


@pytest.mark.parametrize(
    ('arguments', 'match'),
    [
        ({'move': 'flip'}, 'moves matrices of a group'),
        ({'rule': 'heat-bath'}, "takes the rule 'metropolis' or 'glauber'"),
        ({'schedule': 'checkerboard'}, 'updates edges of a cell complex'),
    ],
)
def test_holonomy_score_rejects_moves_rules_and_schedules_of_scalars(arguments, match):
    """Test that a holonomy score refuses a flip, the heat-bath rule, and the checkerboard schedule."""
    with pytest.raises(ValidationError, match=match):
        _matrix_sampling(ISOSPIN, **arguments).prepare()


def test_holonomy_score_needs_a_group_on_the_edges_of_a_cell_complex():
    """Test that a matrix state on the sites of a lattice raises an error at the first sweep."""
    sites = GroupElement('links', element=IsospinLinkElement, space=LatticeSpace((4, 4)), on='sites')
    system = CompositeSystem(elements=[sites], interactions=[_matrix_sampling(ISOSPIN)], time=DiscreteTime())
    with pytest.raises(ValidationError, match='edges of a CellComplexSpace'):
        simulate(system, until=1, random_state=0)


def test_variant_refuses_a_spec_and_lists_what_it_takes():
    """Test that a move refuses a spec, and lists its names and functions."""
    interaction = SamplingInteraction('a', target='b', state='x', objective=lambda s: 0.0, move=PartnerSpec())
    with pytest.raises(ValidationError, match=r'Unknown move .*a function'):
        interaction.prepare()
