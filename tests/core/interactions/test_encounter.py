"""Test the `_encounter` module of `skcomplex.core.interactions`."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.elements import DensityElement, FieldElement, GroupElement
from skcomplex.core.interactions import EncounterInteraction
from skcomplex.core.laws.utils import coordinate
from skcomplex.core.spaces import LatticeSpace, PhaseSpace
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import RK4Solver
from skcomplex.errors import ValidationError
from skcomplex.specs import AxisSpec, RoleSpec

CLASSES = 3


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
    system = System(
        elements=[
            GroupElement('candidates', element=DensityElement, size=CLASSES, on='sites', values={'f': list(initial)}),
            GroupElement('field', element=DensityElement, size=1, values={'f': 1.0}),
        ],
        interactions=[encounter],
        space=LatticeSpace(CLASSES, boundary='fixed'),
    )
    trajectory = Simulation(system, dt=0.5).check().prepare().run(until=until).trajectory_
    return np.asarray(trajectory.times), trajectory.values('candidates.f')


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
        EncounterInteraction('encounter', candidate='candidates', field='field', **arguments).check()


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
    space = PhaseSpace(LatticeSpace((3,)), [axis]).check().prepare()
    theta = space.coordinates_['theta']
    start = np.tile(np.exp(2.78 * np.cos(theta)), (3, 1))
    assert (start[0] * np.cos(theta)).sum() / start[0].sum() == pytest.approx(0.8, abs=0.01)
    gaps = theta[:, None] - theta[None, :]
    spread = 0.5
    noise = np.exp(-(np.minimum(np.abs(gaps), 2 * np.pi - np.abs(gaps)) ** 2) / (2 * spread**2))
    noise /= noise.sum(axis=0, keepdims=True)
    gas = FieldElement('gas', element=DensityElement, space=space, values={'f': start})
    diffusion = EncounterInteraction('noise', candidate='gas', rate=lambda _: 2.0, kernel=lambda _: noise.T)
    simulation = Simulation(System(elements=[gas], interactions=[diffusion]), solver=RK4Solver(dt=0.01)).check()
    trajectory = simulation.prepare().run(until=1.0).trajectory_
    values = np.asarray(trajectory.values('gas.f')).reshape(-1, 3, 32)
    assert np.abs(values[-1].sum(axis=1) / start.sum(axis=1) - 1.0).max() < 1e-12
    polarization = (values[-1] * np.cos(theta)).sum(axis=1) / (start * np.cos(theta)).sum(axis=1)
    coefficient = np.sum(noise.T[0] * np.cos(theta - theta[0]))
    assert polarization == pytest.approx(np.exp(-2.0 * (1.0 - coefficient)), abs=1e-6)


def test_encounter_pairs_classes_at_the_same_position() -> None:
    """Run a bilinear encounter at every position, the positions apart from one another."""
    space = PhaseSpace(LatticeSpace((2,)), [AxisSpec('c', 2, 0.0, 1.0)]).check().prepare()
    start = np.array([[1.0, 0.0], [0.0, 1.0]])
    gas = FieldElement('gas', element=DensityElement, space=space, values={'f': start})
    kernel = np.zeros((2, 2, 2))
    kernel[:, :, 1] = 1.0
    move = EncounterInteraction('move', candidate='gas', field='gas', rate=lambda _: 1.0, kernel=lambda _: kernel)
    simulation = Simulation(System(elements=[gas], interactions=[move]), solver=RK4Solver(dt=0.01)).check().prepare()
    final = simulation.run(until=1.0).trajectory_.values('gas.f')[-1].reshape(2, 2)
    assert final.sum(axis=1) == pytest.approx([1.0, 1.0])
    assert final[1] == pytest.approx([0.0, 1.0])
    assert final[0, 0] == pytest.approx(np.exp(-1.0), rel=1e-6)


def test_encounter_rate_reads_the_coordinate_of_the_candidate_class() -> None:
    """Remove a density at a rate that reads the coordinate of its class."""
    space = PhaseSpace(LatticeSpace((1,)), [AxisSpec('c', 2, 0.0, 4.0)]).check().prepare()
    gas = FieldElement('gas', element=DensityElement, space=space, values={'f': 1.0})
    level = FieldElement('level', element=DensityElement, space=space, values={'f': 1.0})
    remove = EncounterInteraction(
        'remove',
        candidate='gas',
        field='level',
        rate=lambda _: coordinate('c'),
        outcome='destructive',
    )
    system = System(elements=[gas, level], interactions=[remove])
    final = Simulation(system, solver=RK4Solver(dt=0.01)).check().prepare().run(until=0.1).trajectory_
    expected = np.exp(-0.1 * 2.0 * np.array([1.0, 3.0]))
    assert final.values('gas.f')[-1] == pytest.approx(expected, rel=1e-6)


def test_encounter_without_field_needs_conservative_outcome() -> None:
    """Raise a named error for a destructive encounter without a field."""
    interaction = EncounterInteraction('e', candidate='gas', rate=lambda _: 1.0, outcome='destructive')
    with pytest.raises(ValidationError, match='Give `field`'):
        interaction.check()
