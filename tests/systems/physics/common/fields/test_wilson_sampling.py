"""Test the `_wilson_sampling` module of `skcomplex.systems.physics.common.fields`."""

from typing import Annotated

import numpy as np
import pytest
from scipy.special import iv

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.spaces import CellComplexSpace, DiscreteTime, LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import MatrixGroupSpec, ParameterSpec, StateSpec
from skcomplex.systems.physics.common.fields import HolonomyScoreSpec, WilsonSamplingInteraction

ISOSPIN = MatrixGroupSpec('SU', 2)
COLOR = MatrixGroupSpec('SU', 3)


def test_holonomy_score_keeps_its_group_and_hits():
    """Test that a valid score keeps its group and its number of hits."""
    score = HolonomyScoreSpec(np.sum, MatrixGroupSpec('SU', 3), hits=3)
    assert (score.group.degree, score.hits) == (3, 3)


@pytest.mark.parametrize(
    ('arguments', 'match'),
    [
        ({'function': 1.0, 'group': MatrixGroupSpec()}, 'callable'),
        ({'function': np.sum, 'group': 'SU(2)'}, 'MatrixGroupSpec'),
        ({'function': np.sum, 'group': MatrixGroupSpec(), 'hits': 0}, 'at least one move'),
    ],
)
def test_holonomy_score_rejects_what_it_cannot_run(arguments, match):
    """Test that a function that is not callable, a group that is no spec, or no hits raise an error."""
    with pytest.raises(ValidationError, match=match):
        HolonomyScoreSpec(**arguments)._check()


class IsospinLinkElement(Element):
    """Link with a matrix of SU(2) and a coupling `beta`."""

    def __init__(
        self,
        label: str,
        *,
        U: Annotated[object, StateSpec(group=ISOSPIN)] = tuple(ISOSPIN.identity),
        beta: Annotated[float, ParameterSpec()] = 2.0,
    ) -> None:
        super().__init__(label)
        self.U = U
        self.beta = beta


class ColorLinkElement(Element):
    """Link with a matrix of SU(3) and a coupling `beta`."""

    def __init__(
        self,
        label: str,
        *,
        U: Annotated[object, StateSpec(group=COLOR)] = tuple(COLOR.identity),
        beta: Annotated[float, ParameterSpec()] = 4.0,
    ) -> None:
        super().__init__(label)
        self.U = U
        self.beta = beta


def _trace_action(holonomies, beta):
    """Return the Wilson action of the holonomies, `-(beta / N) Re tr`."""
    return -beta / holonomies.shape[-1] * np.trace(holonomies, axis1=-2, axis2=-1).real


def _matrix_system(shape, group, sampling, seed=0, element=IsospinLinkElement, *, beta=4.0):
    """Return a system of matrix links of a lattice complex from random links, sampled by an interaction."""
    space = CellComplexSpace(LatticeSpace(shape)).prepare()
    initial = group.get_components_of(group.get_uniform_of(np.random.default_rng(seed), space.incidences_[0].shape[0]))
    links = GroupElement('links', element=element, space=space, on='edges', values={'U': initial, 'beta': beta})
    return space, CompositeSystem(elements=[links], interactions=[sampling], time=DiscreteTime())


def _matrix_sampling(group, **arguments):
    """Return the sampling of matrix links by the Wilson action at the inverse coupling 4."""
    settings = {'move': 'rotate', 'width': 0.6, 'schedule': 'colored', 'hits': 4, 'beta': 4.0, **arguments}
    return WilsonSamplingInteraction('wilson', target='links', state='U', group=group, **settings)


def _plaquettes(space, group, values):
    """Return the mean of `Re tr U_p / N` over the plaquettes at every frame."""
    loops = [
        group.get_holonomy_of(group.get_matrices_of(frame), space.boundaries_, space.boundary_signs_)
        for frame in values
    ]
    return np.trace(np.stack(loops), axis1=-2, axis2=-1).real.mean(axis=1) / group.degree


def test_two_dimensional_su2_reproduces_the_exact_plaquette():
    """Test that the mean SU(2) plaquette at `beta = 2` is `I_2(2) / I_1(2)` to 0.01."""
    space, system = _matrix_system((16, 16), ISOSPIN, _matrix_sampling(ISOSPIN, beta=2.0), beta=2.0)
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


def test_huge_inverse_coupling_never_raises_the_action_of_matrix_links():
    """Test that at a huge inverse coupling the total plaquette never falls, since links of a class share no face."""
    sampling = _matrix_sampling(ISOSPIN, beta=1e12, move='random-value', hits=1)
    space, system = _matrix_system((5, 4), ISOSPIN, sampling, beta=1e12)
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
