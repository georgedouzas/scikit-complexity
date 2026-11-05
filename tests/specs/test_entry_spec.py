"""Test the entry spec."""

from sage.all__sagemath_symbolics import RR

from skcomplex.core.spaces import LatticeNeighborhood
from skcomplex.specs import EntrySpec, SubshiftSpec, TensorSpec
from skcomplex.types import DomainKind


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
