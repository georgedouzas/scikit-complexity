"""Test the base of every law, its declarations, and the queries over the law classes."""

import dataclasses

import pytest

from skcomplex.core.base import LawClass, LawKind, Trait, ValidationError, symbol
from skcomplex.core.formalisms.lattices import LocalRule
from skcomplex.core.formalisms.quantum import Dissipator, Generator
from skcomplex.core.formalisms.rewriting import GraphProduction, Production, Substitution
from skcomplex.core.interactions import Match
from skcomplex.core.laws import Boundary, Invariant, Jump, Law, Rate, draw, has_trait, law_classes


def test_law_classes_share_the_base():
    """Test that every law class derives from the law base and declares a kind, a time class, and its traits."""
    classes = law_classes()
    for law in (Match, Invariant, Dissipator, Generator, Production, GraphProduction, Substitution):
        assert law in classes
    for law in classes:
        assert issubclass(law, Law)
        assert isinstance(law.law_kind, LawKind), law
        assert isinstance(law.time_class, LawClass), law
        assert all(isinstance(trait, Trait) for trait in law.traits), law
        assert dataclasses.is_dataclass(law)
        assert law.__dataclass_params__.frozen, law


def test_law_classes_filter_by_trait():
    """Test that the law classes with a trait are the classes that declare it."""
    lattice = law_classes(Trait.LATTICE)
    assert LocalRule in lattice
    assert Rate not in lattice
    assert all(Trait.LATTICE in law.traits for law in lattice)


def test_has_trait_reads_laws_and_classes():
    """Test that a trait is read from a law, from a law class, and is absent from other values."""
    x = symbol('x')
    assert has_trait(Boundary(x, 0), Trait.FORMLESS)
    assert has_trait(Boundary, Trait.FORMLESS)
    assert not has_trait(Rate(x, 1), Trait.FORMLESS)
    assert not has_trait(object(), Trait.FORMLESS)


def test_law_is_importable_at_run_time():
    """Test that the law base is a class that introspection can evaluate."""
    assert isinstance(Law, type)
    assert Law.__module__ == 'skcomplex.core.laws._law'


@pytest.mark.parametrize(
    ('arguments', 'message'),
    [
        ({'waiting': draw('wait', 'expon'), 'changes': {symbol('n'): 1}}, 'changes belong to the exponential clock'),
        ({'rate': 1.0, 'changes': {symbol('n'): 1}, 'reset': {symbol('n'): 0}}, 'resets and labels to the waiting'),
        ({'rate': 1.0, 'changes': {symbol('n'): 1}, 'label': 'fires'}, 'takes a label'),
    ],
)
def test_jump_rejects_option_of_other_clock(arguments, message):
    """Test that a jump rejects at construction an option of the clock it does not use."""
    with pytest.raises(ValidationError, match=message):
        Jump(**arguments)
