"""Test the `_sampling` module of `skcomplex.core.components.interactions`."""

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import cos, pi
from scipy.special import iv

from skcomplex.core.components.elements import Element, GroupElement, LatticeElement
from skcomplex.core.components.interactions import SamplingInteraction
from skcomplex.core.components.laws.utils import exterior_derivative, neighbors
from skcomplex.core.components.spaces import (
    CellComplexSpace,
    CofaceNeighborhood,
    DiscreteTime,
    LatticeSpace,
    VonNeumannNeighborhood,
)
from skcomplex.core.components.system import System
from skcomplex.core.engine.compile import compile_system
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import ParameterSpec, StateSpec, VisualSpec

SIDE = 16


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
    return space, System(elements=[links], interactions=[sampling], time=DiscreteTime())


def _ising(namespace):
    """Return the local energy of a spin with its four neighbours."""
    return -namespace.target.s * neighbors(namespace.target.other.s)


def _final(sampling, until, initial, seed=0):
    """Return the values of the sites after some sweeps."""
    sites = LatticeElement('sites', element=SiteElement, space=LatticeSpace((SIDE, SIDE)), values={'s': initial})
    system = System(elements=[sites], interactions=[sampling], time=DiscreteTime())
    return Simulation(system, random_state=seed).prepare().run(until=until).state_


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
    sites = LatticeElement('sites', element=SiteElement, space=LatticeSpace((SIDE, SIDE)), values={'s': _spins(seed)})
    system = System(elements=[sites], interactions=[sampling], time=DiscreteTime())
    states = Simulation(system, random_state=seed).prepare().run(until=300).trajectory_.states
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
        neighborhood=VonNeumannNeighborhood(n_dim=1),
        cooling=_geometric,
    )
    candidates = LatticeElement('candidates', element=CandidateElement, space=LatticeSpace((200,)))
    system = System(elements=[candidates], interactions=[anneal], time=DiscreteTime())
    frames = Simulation(system, random_state=0).prepare().run(until=500).trajectory_.values('candidates.x')
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
    trajectory = Simulation(system, random_state=3).prepare().run(until=5000).trajectory_
    states = trajectory.values('links.theta')
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
    first = Simulation(_link_system((6, 6), sampling, {'theta': initial})[1], random_state=5)
    second = Simulation(_link_system((6, 6), sampling, {'theta': initial})[1], random_state=5)
    other = Simulation(_link_system((6, 6), sampling, {'theta': initial})[1], random_state=6)
    frames = [item.prepare().run(until=20).trajectory_.values('links.theta') for item in (first, second, other)]
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
    classes = compile_system(system).lattice[0].classes
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
        compile_system(system)


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
        compile_system(system)


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
        compile_system(system)
