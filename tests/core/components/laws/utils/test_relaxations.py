"""Test the `_relaxations` module of `skcomplex.core.components.laws.utils`."""

from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import var

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import ArrayUpdateLaw, LocalRuleLaw, UpdateLaw
from skcomplex.core.components.laws.utils import relax
from skcomplex.core.components.laws.utils._relaxations import relaxation_of
from skcomplex.core.components.spaces import LatticeNeighborhood, LatticeSpace, RadiusNeighborhood
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import StateSpec

CROSS = RadiusNeighborhood(1, metric='manhattan', center=True)
LINE = LatticeNeighborhood(((-1,), (0,), (1,)))


class PotentialElement(Element):
    """Membrane potential of a neuron on a line, which fires at one and passes a half to each neighbour."""

    def __init__(self, label: str, *, potential: Annotated[object, StateSpec()] = 0.0) -> None:
        super().__init__(label)
        self.potential = potential

    def _get_laws(self, s):
        """Return the potentials driven at the first neuron and relaxed until no neuron fires."""
        fire = LocalRuleLaw(
            s.potential,
            LINE,
            lambda left, value, right: value * (value < 1) + 0.5 * (left >= 1) + 0.5 * (right >= 1),
        )
        drive = ArrayUpdateLaw(s.potential, lambda values, *, random: values + 0.6 * (np.arange(values.size) == 0))
        relaxation = relax(s.potential, fire, lambda values: bool((values < 1).all()), drive=drive)
        return [UpdateLaw(s.potential, relaxation, label='avalanche')]


def test_relax_keeps_its_rules_drive_and_settings():
    """Test that a relaxation keeps its rules and drive in order, its stability, and its largest iterations."""
    height = var('height')
    topple = LocalRuleLaw(height, CROSS, lambda z, *n: z)
    settle = LocalRuleLaw(height, CROSS, lambda z, *n: z)
    drop = ArrayUpdateLaw(height, lambda values, *, random: values + 1)
    relaxation = relaxation_of(relax(height, [topple, settle], bool, drive=drop, max_iterations=7))
    assert relaxation.rules == (topple, settle)
    assert relaxation.drive == (drop,)
    assert relaxation.max_iterations == 7
    assert relaxation_of(height + 1) is None


def test_relax_needs_laws():
    """Test that a relaxation needs a rule, and takes laws only."""
    height = var('height')
    with pytest.raises(ValidationError, match='no rules'):
        relax(height, [], bool)
    with pytest.raises(ValidationError, match='laws'):
        relax(height, [lambda z: z], bool)


def test_relax_records_a_neuronal_avalanche():
    """Test that a neuron driven past its threshold fires an avalanche that the run records with its size."""
    group = GroupElement('neurons', element=PotentialElement, space=LatticeSpace(4, boundary='fixed'), on='sites')
    trajectory = simulate(CompositeSystem(elements=[group]), until=2, dt=1)
    (record,) = trajectory.events
    assert (record.label, record.time) == ('avalanche', 2.0)
    assert record.size >= 1
    assert np.all(trajectory.states[-1] < 1)


def test_relax_checks_every_law_it_holds():
    """Test that a run checks every rule of a relaxation."""

    class SettleElement(Element):
        """Height that settles with a rule of an unknown schedule."""

        def __init__(self, label: str, *, height: Annotated[object, StateSpec()] = 0.0) -> None:
            super().__init__(label)
            self.height = height

        def _get_laws(self, s):
            """Return the relaxation with the unknown schedule."""
            settle = LocalRuleLaw(s.height, CROSS, lambda z, *n: z, schedule='parallel')
            return [UpdateLaw(s.height, relax(s.height, settle, bool))]

    group = GroupElement('pile', element=SettleElement, space=LatticeSpace((2, 2)), on='sites')
    with pytest.raises(ValidationError, match='Unknown schedule'):
        simulate(CompositeSystem(elements=[group]), until=1, dt=1)
