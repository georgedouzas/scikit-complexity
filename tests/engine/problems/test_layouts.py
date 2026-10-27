"""Test the `_layouts` module of `skcomplex.engine.problems`."""

from sage.all__sagemath_symbolics import RR

from skcomplex.engine.problems import Entry, Layout


def test_layout_reads_its_entries_and_names_by_method():
    """Test that a layout gives its entries, its flat size, and its membership through named methods."""
    layout = Layout([Entry('a.n', 0, (), RR), Entry('b.x', 1, (2,), RR)])
    assert [entry.name for entry in layout.entries] == ['a.n', 'b.x']
    assert layout.size == 3
    assert layout.has('b.x')
    assert not layout.has('b.y')
