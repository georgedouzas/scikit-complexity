"""Test the `_layout_spec` module of `skcomplex.core.engine.problems`."""

import numpy as np
from sage.all__sagemath_symbolics import CC, RR

from skcomplex.core.engine.problems import LayoutSpec
from skcomplex.core.specs import EntrySpec


def test_layout_reads_its_entries_and_names_by_method():
    """Test that a layout gives its entries, its flat size, and its membership through named methods."""
    layout = LayoutSpec((EntrySpec('a.n', 0, (), RR), EntrySpec('b.x', 1, (2,), RR)))
    assert [entry.name for entry in layout.entries] == ['a.n', 'b.x']
    assert layout.size == 3
    assert layout.has('b.x')
    assert not layout.has('b.y')


def test_relay_keeps_the_rows_both_layouts_hold():
    """Test that a vector of a group of two members moves into a layout of four, whose new rows keep their values."""
    old = LayoutSpec((EntrySpec('a.n', 0, (), RR), EntrySpec('g.x', 1, (2,), RR, members=2)))
    new = LayoutSpec(
        (
            EntrySpec('a.n', 0, (), RR),
            EntrySpec('g.x', 1, (4,), RR, members=4),
            EntrySpec('g.active', 5, (4,), RR, members=4),
        ),
    )
    relaid = new.get_relaid(np.array([7.0, 1.0, 2.0]), old, np.zeros(9))
    assert relaid.tolist() == [7.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


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


def test_layout_reads_derivatives_and_past_values_by_the_names_users_read():
    """Test that `x'` names the derivative `x_d1` and `y[-1]` the past value `y_lag1`, also for one member."""
    entries = (EntrySpec('a.x', 0, (), RR), EntrySpec('a.x_d1', 1, (), RR), EntrySpec('b.y_lag1', 2, (2,), RR))
    layout = LayoutSpec(entries)
    vector = np.array([1.0, 2.0, 3.0, 4.0])
    assert layout.has("a.x'")
    assert layout.read(vector, "a.x'") == 2.0
    assert layout.get_entry('b.y[-1]').name == 'b.y_lag1'
    assert layout.locate('b.y[-1][1]')[1] == slice(3, 4)


def test_layout_names_derivatives_of_vectors_of_members_with_primes():
    """Test that the derivative of a vector of a member reads with a prime before its indices."""
    assert LayoutSpec._get_display_name('ball.x_d1[i, 0]') == "ball.x'[i, 0]"
    assert LayoutSpec._get_internal_name("ball.x'[i, 0]") == 'ball.x_d1[i, 0]'
