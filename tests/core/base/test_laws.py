"""Test the base of every law, its declarations, and the queries over the law classes."""

import dataclasses

from skcomplex.core.base import BaseLaw, LawClass, LawKind, Trait, has_trait, law_classes, symbol
from skcomplex.core.laws import BoundaryLaw, Invariant, RateLaw
from skcomplex.formalisms.agents import MatchLaw
from skcomplex.formalisms.lattices import LocalRuleLaw
from skcomplex.formalisms.quantum import DissipatorLaw, GeneratorLaw
from skcomplex.formalisms.rewriting import GraphProductionLaw, ProductionLaw, SubstitutionLaw


def test_law_classes_share_the_base():
    """Test that every law class derives from the law base and declares a kind, a time class, and its traits."""
    classes = law_classes()
    for law in (MatchLaw, Invariant, DissipatorLaw, GeneratorLaw, ProductionLaw, GraphProductionLaw, SubstitutionLaw):
        assert law in classes
    for law in classes:
        assert issubclass(law, BaseLaw)
        assert isinstance(law.law_kind, LawKind), law
        assert isinstance(law.time_class, LawClass), law
        assert all(isinstance(trait, Trait) for trait in law.traits), law
        assert dataclasses.is_dataclass(law)
        assert law.__dataclass_params__.frozen, law


def test_law_classes_filter_by_trait():
    """Test that the law classes with a trait are the classes that declare it."""
    lattice = law_classes(Trait.LATTICE)
    assert LocalRuleLaw in lattice
    assert RateLaw not in lattice
    assert all(Trait.LATTICE in law.traits for law in lattice)


def test_has_trait_reads_laws_and_classes():
    """Test that a trait is read from a law, from a law class, and is absent from other values."""
    x = symbol('x')
    assert has_trait(BoundaryLaw(x, 0), Trait.FORMLESS)
    assert has_trait(BoundaryLaw, Trait.FORMLESS)
    assert not has_trait(RateLaw(x, 1), Trait.FORMLESS)
    assert not has_trait(object(), Trait.FORMLESS)


def test_law_is_importable_at_run_time():
    """Test that the law base is a class that introspection can evaluate."""
    assert isinstance(BaseLaw, type)
    assert BaseLaw.__module__ == 'skcomplex.core.base._laws'
