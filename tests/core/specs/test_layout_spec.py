"""Test the `_layout_spec` module of `skcomplex.core.specs`."""

import numpy as np
from sage.all__sagemath_symbolics import CC, RR

from skcomplex.core.specs import EntrySpec, LayoutSpec


def test_layout_reads_its_entries_and_names_by_method():
    """Test that a layout gives its entries, its flat size, and its membership through named methods."""
    layout = LayoutSpec((EntrySpec('a.n', 0, (), RR), EntrySpec('b.x', 1, (2,), RR)))
    assert [entry.name for entry in layout.entries] == ['a.n', 'b.x']
    assert layout.size == 3
    assert layout.has('b.x')
    assert not layout.has('b.y')


def test_complex_view_is_zero_copy():
    """Test that the complex values of a layout view the stored floats without copying."""
    layout = LayoutSpec((EntrySpec('a.z', 0, (3,), CC, is_complex=True),))
    vector = np.arange(6.0)
    values = layout.read(vector, 'a.z')
    assert np.shares_memory(values, vector)
    assert values.tolist() == [1j, 2 + 3j, 4 + 5j]


def test_complex_component_is_located_by_its_part():
    """Test that the real and imaginary parts of a complex variable are located as real components."""
    layout = LayoutSpec((EntrySpec('a.z', 0, (), CC, is_complex=True),))
    entry, indices = layout.locate('a.z.imag')
    assert (entry.domain, indices) == (RR, slice(1, 2))
