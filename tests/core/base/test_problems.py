"""Test the `_problems` module of `skcomplex.core.base`."""

from skcomplex.core.base import Entry, Layout, NumericDomain


def test_layout_reads_its_entries_and_names_by_method():
    """Test that a layout gives its entries, its flat size, and its membership through named methods."""
    layout = Layout([Entry('a.n', 0, (), NumericDomain.REAL), Entry('b.x', 1, (2,), NumericDomain.REAL)])
    assert [entry.name for entry in layout.entries] == ['a.n', 'b.x']
    assert layout.size == 3
    assert layout.has('b.x')
    assert not layout.has('b.y')
