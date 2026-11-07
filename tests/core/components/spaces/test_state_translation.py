"""Test the `_state_translation` module of `skcomplex.core.components.spaces`."""

import numpy as np
import pytest
from sage.all__sagemath_gap import PermutationGroup, libgap

from skcomplex.core.components.spaces import (
    CayleySpace,
    LatticeSpace,
    MemoryNeighborhood,
    StateTranslationNeighborhood,
)
from skcomplex.core.components.spaces import _cayley as cayley
from skcomplex.core.errors import SpaceError, SpaceTooLargeError, ValidationError

SYMMETRIC = PermutationGroup([[(1, 2)], [(1, 2, 3)]])


def test_neighborhood_keeps_its_words():
    """Test that the neighbourhood holds the words read after the translation, the identity by default."""
    assert StateTranslationNeighborhood().prepare().elements_ == ('1',)
    assert StateTranslationNeighborhood(['1', 'a']).prepare().elements_ == ('1', 'a')


def test_translation_table_of_a_finite_group_is_its_cayley_table():
    """Test that the products of a finite group have the identity, are associative, and permute every row."""
    space = CayleySpace(SYMMETRIC).prepare()
    table = space._translation_table(StateTranslationNeighborhood())[:, :, 0]
    sites = np.arange(space.size_)
    np.testing.assert_array_equal(table[:, 0], sites)
    np.testing.assert_array_equal(table[0], sites)
    np.testing.assert_array_equal(np.sort(table, axis=1), np.broadcast_to(sites, table.shape))
    for first in sites:
        for second in sites:
            np.testing.assert_array_equal(table[table[first, second]], table[first][table[second]])
    letter = space.words_.index('a')
    np.testing.assert_array_equal(table[:, letter], space.generator_table_[:, 0])


def test_translation_table_reads_words_after_the_product():
    """Test that a word of the memory set multiplies the product on the right."""
    space = CayleySpace(SYMMETRIC).prepare()
    table = space._translation_table(StateTranslationNeighborhood(('1', 'a')))
    letter = space.generator_table_[:, 0]
    np.testing.assert_array_equal(table[:, :, 1], letter[table[:, :, 0]])


def test_translation_table_of_a_ball_leaves_it_outside():
    """Test that a product outside a ball of the free group is the number of sites."""
    space = CayleySpace(libgap.FreeGroup(['a', 'b']), ball=2).prepare()
    table = space._translation_table(StateTranslationNeighborhood())[:, :, 0]
    lengths = space.lengths_
    inside = lengths[:, None] + lengths[None, :] <= 2
    assert np.all(table[inside] < space.size_)
    assert np.all((table == space.size_) | (lengths[np.minimum(table, space.size_ - 1)] <= 2))
    assert (table == space.size_).any()


def test_translation_table_rejects_other_neighborhoods_spaces_and_large_tables(monkeypatch):
    """Test that the hook takes a translated memory on a group space within the size limit."""
    space = CayleySpace(SYMMETRIC).prepare()
    with pytest.raises(ValidationError, match='translates a memory set'):
        space._translation_table(MemoryNeighborhood(('1',)))
    with pytest.raises(SpaceError, match='no group space'):
        LatticeSpace(5).prepare()._translation_table(StateTranslationNeighborhood())
    monkeypatch.setattr(cayley, 'MAX_PRODUCTS', 20)
    with pytest.raises(SpaceTooLargeError, match='finite quotient'):
        space._translation_table(StateTranslationNeighborhood())
