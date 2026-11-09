"""Test the `_encounter` module of `skcomplex.core.components.interactions`."""

from typing import Annotated

import numpy as np
import numpy.typing as npt
import pytest

from skcomplex.core.components.elements import FieldElement, GroupElement
from skcomplex.core.components.interactions import EncounterInteraction
from skcomplex.core.components.laws.utils import coordinate
from skcomplex.core.components.spaces import LatticeSpace, PhaseSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import AxisSpec, RoleSpec
from skcomplex.systems.general.elements import NonnegativeElement

CLASSES = 3
SIZE, HALF_WIDTH = 32, 8.0


class MeetingInteraction(EncounterInteraction):
    """Encounter that binds its candidates as `walkers` and its field as `medium`."""

    def __init__(
        self,
        label: str,
        *,
        walkers: Annotated[str, RoleSpec()],
        medium: Annotated[str, RoleSpec()],
        **arguments: object,
    ) -> None:
        super().__init__(label, candidate=walkers, field=medium, **arguments)
        self.walkers = walkers
        self.medium = medium

    def _get_role(self, role: str) -> str:
        """Return `walkers` for the candidate and `medium` for the field."""
        return {'candidate': 'walkers', 'field': 'medium'}.get(role, role)


def _densities(encounter, until, initial=(0.2, 0.3, 0.5)):
    """Return the candidate densities over time, meeting a constant field of density 1."""
    system = CompositeSystem(
        elements=[
            GroupElement(
                'candidates',
                element=NonnegativeElement,
                size=CLASSES,
                on='sites',
                values={'x': list(initial)},
            ),
            GroupElement('field', element=NonnegativeElement, size=1, values={'x': 1.0}),
        ],
        interactions=[encounter],
        space=LatticeSpace(CLASSES, boundary='fixed'),
    )
    trajectory = simulate(system, until=until, dt=0.5)
    return np.asarray(trajectory.times), trajectory.values('candidates.x')


def test_conservative_encounters_move_the_density_and_keep_its_total():
    """Test that encounters whose kernel moves every candidate to the first class keep the total and empty the rest."""
    kernel = np.zeros((CLASSES, 1, CLASSES))
    kernel[:, :, 0] = 1.0
    encounter = EncounterInteraction(
        'gather',
        candidate='candidates',
        field='field',
        rate=lambda s: 1.0,
        kernel=lambda s: kernel,
    )
    times, densities = _densities(encounter, 4.0)
    np.testing.assert_allclose(densities.sum(axis=1), 1.0, rtol=1e-6)
    np.testing.assert_allclose(densities[:, 1:].sum(axis=1), 0.8 * np.exp(-times), rtol=1e-3)


def test_destructive_encounters_remove_the_candidates():
    """Test that destructive encounters at rate 0.5 with a field of density 1 remove the candidates as `exp(-t / 2)`."""
    encounter = EncounterInteraction(
        'removal',
        candidate='candidates',
        field='field',
        rate=lambda s: 0.5,
        outcome='destructive',
    )
    times, densities = _densities(encounter, 4.0)
    np.testing.assert_allclose(densities.sum(axis=1), np.exp(-0.5 * times), rtol=1e-3)


def test_proliferative_encounters_create_candidates_of_their_own_class():
    """Test that proliferative encounters without a kernel grow every class as `exp(0.5 t)`."""
    encounter = EncounterInteraction(
        'growth',
        candidate='candidates',
        field='field',
        rate=lambda s: 0.5,
        outcome='proliferative',
    )
    times, densities = _densities(encounter, 2.0)
    np.testing.assert_allclose(densities, np.outer(np.exp(0.5 * times), [0.2, 0.3, 0.5]), rtol=1e-3)


@pytest.mark.parametrize(
    ('arguments', 'match'),
    [
        ({'rate': lambda s: 1.0, 'outcome': 'merge'}, 'outcome of encounter'),
        ({'rate': 1.0, 'outcome': 'destructive'}, 'functions of the namespace'),
        ({'rate': lambda s: 1.0}, 'Give `kernel`'),
    ],
)
def test_check_rejects_outcomes_and_functions_it_cannot_take(arguments, match):
    """Test that an unknown outcome, a rate that is not callable, or a conservative move without kernel raise errors."""
    with pytest.raises(ValidationError, match=match):
        EncounterInteraction('encounter', candidate='candidates', field='field', **arguments).prepare()


def test_get_laws_reads_renamed_candidate_and_field():
    """Test that an encounter that binds `walkers` and `medium` gives the densities of the plain encounter."""
    kernel = np.zeros((CLASSES, 1, CLASSES))
    kernel[:, :, 0] = 1.0
    arguments = {'rate': lambda s: 1.0, 'kernel': lambda s: kernel}
    _, plain = _densities(EncounterInteraction('gather', candidate='candidates', field='field', **arguments), 2.0)
    _, renamed = _densities(MeetingInteraction('gather', walkers='candidates', medium='field', **arguments), 2.0)
    np.testing.assert_allclose(renamed, plain)


def test_linear_encounter_erases_polarization_at_the_rate_of_the_noise() -> None:
    """Keep the density and decay the polarization at the rate given by the first Fourier coefficient of the noise."""
    axis = AxisSpec('theta', 32, 0.0, 2 * np.pi, periodic=True)
    space = PhaseSpace(LatticeSpace((3,)), [axis]).prepare()
    theta = space.coordinates_['theta']
    start = np.tile(np.exp(2.78 * np.cos(theta)), (3, 1))
    assert (start[0] * np.cos(theta)).sum() / start[0].sum() == pytest.approx(0.8, abs=0.01)
    gaps = theta[:, None] - theta[None, :]
    spread = 0.5
    noise = np.exp(-(np.minimum(np.abs(gaps), 2 * np.pi - np.abs(gaps)) ** 2) / (2 * spread**2))
    noise /= noise.sum(axis=0, keepdims=True)
    gas = FieldElement('gas', element=NonnegativeElement, space=space, values={'x': start})
    diffusion = EncounterInteraction('noise', candidate='gas', rate=lambda _: 2.0, kernel=lambda _: noise.T)
    trajectory = simulate(
        CompositeSystem(elements=[gas], interactions=[diffusion]),
        until=1.0,
        solver=RK4Solver(dt=0.01),
    )
    values = np.asarray(trajectory.values('gas.x')).reshape(-1, 3, 32)
    assert np.abs(values[-1].sum(axis=1) / start.sum(axis=1) - 1.0).max() < 1e-12
    polarization = (values[-1] * np.cos(theta)).sum(axis=1) / (start * np.cos(theta)).sum(axis=1)
    coefficient = np.sum(noise.T[0] * np.cos(theta - theta[0]))
    assert polarization == pytest.approx(np.exp(-2.0 * (1.0 - coefficient)), abs=1e-6)


def test_encounter_pairs_classes_at_the_same_position() -> None:
    """Run a bilinear encounter at every position, the positions apart from one another."""
    space = PhaseSpace(LatticeSpace((2,)), [AxisSpec('c', 2, 0.0, 1.0)]).prepare()
    start = np.array([[1.0, 0.0], [0.0, 1.0]])
    gas = FieldElement('gas', element=NonnegativeElement, space=space, values={'x': start})
    kernel = np.zeros((2, 2, 2))
    kernel[:, :, 1] = 1.0
    move = EncounterInteraction('move', candidate='gas', field='gas', rate=lambda _: 1.0, kernel=lambda _: kernel)
    trajectory = simulate(CompositeSystem(elements=[gas], interactions=[move]), until=1.0, solver=RK4Solver(dt=0.01))
    final = trajectory.values('gas.x')[-1].reshape(2, 2)
    assert final.sum(axis=1) == pytest.approx([1.0, 1.0])
    assert final[1] == pytest.approx([0.0, 1.0])
    assert final[0, 0] == pytest.approx(np.exp(-1.0), rel=1e-6)


def test_encounter_rate_reads_the_coordinate_of_the_candidate_class() -> None:
    """Remove a density at a rate that reads the coordinate of its class."""
    space = PhaseSpace(LatticeSpace((1,)), [AxisSpec('c', 2, 0.0, 4.0)]).prepare()
    gas = FieldElement('gas', element=NonnegativeElement, space=space, values={'x': 1.0})
    level = FieldElement('level', element=NonnegativeElement, space=space, values={'x': 1.0})
    remove = EncounterInteraction(
        'remove',
        candidate='gas',
        field='level',
        rate=lambda _: coordinate('c'),
        outcome='destructive',
    )
    system = CompositeSystem(elements=[gas, level], interactions=[remove])
    final = simulate(system, until=0.1, solver=RK4Solver(dt=0.01))
    expected = np.exp(-0.1 * 2.0 * np.array([1.0, 3.0]))
    assert final.values('gas.x')[-1] == pytest.approx(expected, rel=1e-6)


def test_encounter_without_field_needs_conservative_outcome() -> None:
    """Raise a named error for a destructive encounter without a field."""
    interaction = EncounterInteraction('e', candidate='gas', rate=lambda _: 1.0, outcome='destructive')
    with pytest.raises(ValidationError, match='Give `field`'):
        interaction.prepare()


def maxwell(_speed: object, _cosine: object) -> float:
    """Return the cross-section of Maxwell molecules in two velocity dimensions."""
    return 1 / (2 * np.pi)


def bkw(time: float, vx: npt.NDArray[np.float64], vy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the Bobylev-Krook-Wu solution of Maxwell molecules in two velocity dimensions."""
    scale = 1 - np.exp(-time / 8) / 2
    squared = vx**2 + vy**2
    return (
        np.exp(-squared / (2 * scale)) / (2 * np.pi * scale**2) * (2 * scale - 1 + (1 - scale) / (2 * scale) * squared)
    )


def velocity_space() -> tuple[PhaseSpace, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return a phase space of one position and two velocity axes, with the velocity grids."""
    axes = [AxisSpec('vx', SIZE, -HALF_WIDTH, HALF_WIDTH), AxisSpec('vy', SIZE, -HALF_WIDTH, HALF_WIDTH)]
    space = PhaseSpace(LatticeSpace((1,)), axes).prepare()
    vx, vy = np.meshgrid(space.coordinates_['vx'], space.coordinates_['vy'], indexing='ij')
    return space, vx, vy


def test_check_rejects_values_it_cannot_take() -> None:
    """Raise named errors for a state that is no name and functions that are not callable."""
    with pytest.raises(ValidationError, match='no name'):
        EncounterInteraction('c', candidate='gas', state='1f', cross_section=maxwell).prepare()
    with pytest.raises(ValidationError, match='cross-section'):
        EncounterInteraction('c', candidate='gas', cross_section=1.0).prepare()
    with pytest.raises(ValidationError, match='factor'):
        EncounterInteraction('c', candidate='gas', cross_section=maxwell, factor=2.0).prepare()


def test_check_rejects_a_cross_section_with_other_settings() -> None:
    """Raise named errors for a cross-section with a rate and a factor without a cross-section."""
    with pytest.raises(ValidationError, match='sets the rate'):
        EncounterInteraction('c', candidate='gas', rate=lambda _: 1.0, cross_section=maxwell).prepare()
    with pytest.raises(ValidationError, match='scales a cross-section'):
        EncounterInteraction(
            'c',
            candidate='gas',
            rate=lambda _: 1.0,
            kernel=lambda _: 1.0,
            factor=lambda _: 2.0,
        ).prepare()


def test_bkw_solution_within_1e_4_and_density_constant() -> None:
    """Match the exact BKW solution within 1e-4 at time 5, and keep the density within 1e-12."""
    space, vx, vy = velocity_space()
    start = bkw(0.0, vx, vy)
    gas = FieldElement('gas', element=NonnegativeElement, space=space, values={'x': start[None]})
    collide = EncounterInteraction('c', candidate='gas', cross_section=maxwell)
    trajectory = simulate(CompositeSystem(elements=[gas], interactions=[collide]), until=5.0, solver=RK4Solver(dt=0.1))
    final = trajectory.values('gas.x')[-1].reshape(SIZE, SIZE)
    assert np.abs(final - bkw(5.0, vx, vy)).max() < 1e-4
    assert abs(final.sum() / start.sum() - 1.0) < 1e-12


def test_density_factor_speeds_relaxation_twice() -> None:
    """Reach every intermediate state in half the time, within 1 percent, with a density factor of 2."""
    space, vx, vy = velocity_space()
    start = bkw(0.0, vx, vy)
    gas = FieldElement('gas', element=NonnegativeElement, space=space, values={'x': start[None]})
    plain = EncounterInteraction('c', candidate='gas', cross_section=maxwell)
    faster = EncounterInteraction('c', candidate='gas', cross_section=maxwell, factor=lambda _: 2.0)
    slow = simulate(CompositeSystem(elements=[gas], interactions=[plain]), until=2.0, solver=RK4Solver(dt=0.1))
    first = slow.values('gas.x')[-1]
    fast = simulate(CompositeSystem(elements=[gas], interactions=[faster]), until=1.0, solver=RK4Solver(dt=0.05))
    second = fast.values('gas.x')[-1]
    assert np.abs(first - second).max() / np.abs(first - start.ravel()).max() < 0.01


def test_collision_without_velocity_axes_is_a_law_space_issue() -> None:
    """Report a collision on a lattice without class axes, with the fix of an interval `AxisSpec`."""
    gas = FieldElement('gas', element=NonnegativeElement, space=LatticeSpace((4,)))
    collide = EncounterInteraction('c', candidate='gas', cross_section=maxwell)
    report = PassCompiler().prepare(CompositeSystem(elements=[gas], interactions=[collide])).report_
    issues = [issue for issue in report.issues if issue.code == 'law-space']
    assert issues
    assert 'AxisSpec' in issues[0].fix


def test_collision_on_periodic_axis_is_a_law_space_issue() -> None:
    """Report a collision on a phase space whose class axes are not two intervals."""
    axes = [AxisSpec('vx', 8, -1.0, 1.0), AxisSpec('vy', 8, 0.0, 6.0, periodic=True)]
    gas = FieldElement('gas', element=NonnegativeElement, space=PhaseSpace(LatticeSpace((1,)), axes))
    collide = EncounterInteraction('c', candidate='gas', cross_section=maxwell)
    report = PassCompiler().prepare(CompositeSystem(elements=[gas], interactions=[collide])).report_
    assert any(issue.code == 'law-space' for issue in report.issues)
