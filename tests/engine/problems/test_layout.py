"""Test the `LayoutSpec` class."""

from sage.all__sagemath_symbolics import RR

from skcomplex.specs import EntrySpec, LayoutSpec


def test_layout_reads_its_entries_and_names_by_method():
    """Test that a layout gives its entries, its flat size, and its membership through named methods."""
    layout = LayoutSpec((EntrySpec('a.n', 0, (), RR), EntrySpec('b.x', 1, (2,), RR)))
    assert [entry.name for entry in layout.entries] == ['a.n', 'b.x']
    assert layout.size == 3
    assert layout.has('b.x')
    assert not layout.has('b.y')
