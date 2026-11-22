"""Test the `_nonnegative` module of `skcomplex.systems.general.elements`."""

import numpy as np
import pytest

from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System
from skcomplex.systems.general.elements import NonnegativeElement
from skcomplex.systems.general.reactions import ReactionInteraction


def test_nonnegative_lists_its_value_and_its_members():
    """Test that a nonnegative element lists its label, its value `x`, and its members."""
    assert NonnegativeElement('S', x=990).get_params() == {'label': 'S', 'x': 990, 'size': None, 'space': None}


def test_nonnegative_value_is_a_state_when_a_law_changes_it():
    """Test that the value `x` is a state under a decay and a parameter without laws."""
    decay = ReactionInteraction('decay', reactants={('S', 'x'): 1}, k=1.0)
    changed = System(elements=[NonnegativeElement('S', x=1.0)], interactions=[decay]).prepare()
    kept = System(elements=[NonnegativeElement('S', x=1.0)]).prepare()
    assert changed.states_ == ('S.x',)
    assert kept.parameters_ == ('S.x',)


def test_nonnegative_rejects_a_negative_value():
    """Test that preparing rejects an element whose value is outside the nonnegative numbers."""
    with pytest.raises(ValidationError):
        System(elements=[NonnegativeElement('S', x=-1.0)]).prepare()


def test_nonnegative_members_form_a_distribution_over_classes():
    """Test that a nonnegative element on a lattice of classes has one value per class."""
    distribution = NonnegativeElement('distribution', x=np.full(5, 0.2), space=LatticeSpace(5, boundary='fixed'))
    decay = ReactionInteraction('decay', target='distribution', reactants={'x': 1}, k=1.0)
    trajectory = System(elements=[distribution], interactions=[decay]).prepare().simulate(until=1, dt=1.0)
    np.testing.assert_allclose(trajectory.get_lattice('distribution.x')[-1], 0.2 * np.exp(-1.0), rtol=1e-4)
