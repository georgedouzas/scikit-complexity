"""Test the `_encounter` module of `skcomplex.core.interactions`."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.elements import DensityElement, GroupElement
from skcomplex.core.interactions import EncounterInteraction
from skcomplex.core.spaces import LatticeSpace
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.errors import ValidationError
from skcomplex.specs import RoleSpec

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
