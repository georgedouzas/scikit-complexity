"""Test the `_density` module of `skcomplex.core.elements`."""

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import var

from skcomplex.core.elements import DensityElement, GroupElement
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import EncounterLaw
from skcomplex.core.spaces import LatticeSpace
from skcomplex.core.system import System
from skcomplex.core.times import ContinuousTime
from skcomplex.engine.runtime import Simulation
from skcomplex.specs import ParameterSpec, RoleSpec

CLASSES = 5


class SpreadingEncounterInteraction(Interaction):
    """Conservative encounters that move a candidate one class towards the field particle with probability 1/2."""

    def __init__(
        self,
        label: str,
        *,
        population: Annotated[str, RoleSpec()],
        eta: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label)
        self.population = population
        self.eta = eta

    def _get_laws(self, s):
        """Return the encounter of the population with itself."""
        table = np.zeros((CLASSES, CLASSES, CLASSES))
        for candidate in range(CLASSES):
            for field in range(CLASSES):
                table[candidate, field, candidate + int(np.sign(field - candidate))] += 0.5
                table[candidate, field, candidate] += 0.5
        return [EncounterLaw(s.population.f, s.population.f, s.eta, table)]


def _system():
    """Return a distribution over 5 activity classes, all at the two extreme classes, with encounters."""
    return System(
        elements=[
            GroupElement('population', element=DensityElement, on='sites', values={'f': [0.5, 0.0, 0.0, 0.0, 0.5]}),
        ],
        interactions=[SpreadingEncounterInteraction('encounters', population='population')],
        space=LatticeSpace(CLASSES, boundary='fixed'),
        time=ContinuousTime(),
    )


def test_density_lists_its_density_as_a_parameter():
    """Test that a density lists its label, its visual, and its density `f`."""
    assert DensityElement('wealth', f=0.2).get_params() == {'f': 0.2, 'label': 'wealth', 'visual': None}


def test_density_runs_into_no_laws():
    """Test that a density contributes no laws."""
    assert DensityElement('wealth').check().prepare().run(SimpleNamespace(f=var('f'))).laws_ == []


def test_density_is_moved_by_an_encounter_law():
    """Test that an encounter law moves a distribution of densities to the middle classes and keeps its total."""
    trajectory = Simulation(_system(), dt=0.5).check().prepare().run(until=2).trajectory_
    densities = np.asarray(trajectory.values('population.f'))
    np.testing.assert_allclose(densities.sum(axis=1), 1.0)
    assert densities[-1, 2] > 0
    np.testing.assert_allclose(densities[-1], densities[-1][::-1])
