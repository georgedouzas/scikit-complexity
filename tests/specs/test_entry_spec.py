"""Test the entry spec."""

from sage.all__sagemath_symbolics import RR

from skcomplex.specs import EntrySpec, TensorSpec
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
