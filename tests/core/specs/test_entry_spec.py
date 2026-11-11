"""Test the entry spec."""

import numpy as np
import pytest
from sage.all__sagemath_symbolics import RR, ZZ, oo
from sage.sets.real_set import RealSet

from skcomplex.core.components.spaces import LatticeNeighborhood
from skcomplex.core.specs import EntrySpec, SubshiftSpec, TensorSpec
from skcomplex.core.types import DomainKind


def test_kind_is_tensor_for_a_tensor_entry():
    """The components of a tensor state report the kind tensor."""
    entry = EntrySpec('film.Q', 0, (6, 2), RR, 6, tensor=TensorSpec('traceless', 2))
    assert entry.kind == DomainKind.TENSOR


def test_component_names_read_the_index_of_every_component():
    """A tensor entry names its components by their index."""
    entry = EntrySpec('film.Q', 0, (2, 2), RR, 2, tensor=TensorSpec('traceless', 2))
    assert entry.component_names == ['film.Q_xx[0]', 'film.Q_xy[0]', 'film.Q_xx[1]', 'film.Q_xy[1]']
    assert entry.generic_names == ['film.Q_xx[i]', 'film.Q_xy[i]']


def test_subshift_domain_reads_as_its_alphabet():
    """A state with a subshift as its domain holds the values of the alphabet, checked cell by cell."""
    domain = SubshiftSpec(LatticeNeighborhood(((0,), (1,))), ((1, 1),))
    entry = EntrySpec('cells.alive', 0, (3,), domain, 3)
    assert entry.kind == DomainKind.BOOLEAN
    assert entry.contains([0, 1, 1])
    assert not entry.contains([0, 2, 1])
    assert EntrySpec('cells.alive', 0, (3,), SubshiftSpec(domain.neighborhood, ((1, 2),), 3), 3).kind == (
        DomainKind.INTEGER
    )


@pytest.mark.parametrize(
    'domain',
    [
        RR,
        RealSet.closed_open(0, oo),
        RealSet.open(0, oo),
        RealSet.closed(0, 1),
        RealSet.open(-3, -1) + RealSet.point(0),
    ],
)
def test_contains_checks_arrays_of_numbers_as_sage_checks_every_component(domain):
    """Test that an array of numbers is in a real domain exactly when Sage finds every component in it."""
    values = [0.0, 1.0, 1, True, -1.5, -3.0, -1.0, np.nan, np.inf, -np.inf]
    for value in values:
        assert EntrySpec('x', 0, (), domain).contains(np.full((3, 3), value)) == (value in domain)
    assert EntrySpec('x', 0, (), domain).contains(np.zeros(2)) == (0 in domain)


def test_contains_checks_other_domains_component_by_component():
    """Test that an integer domain still rejects a fraction among whole numbers."""
    assert EntrySpec('x', 0, (), ZZ).contains([1, 2, 3])
    assert not EntrySpec('x', 0, (), ZZ).contains([1, 2.5])


def test_layout_of_real_systems_is_unchanged():
    """Test that a real variable keeps one float per component and its names."""
    entry = EntrySpec('a.x', 0, (2,), RR)
    assert (entry.size, entry.columns, entry.component_names) == (2, 2, ['a.x[0]', 'a.x[1]'])
