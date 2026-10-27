"""Test the `_laws` module of `skcomplex.core.base`."""

import dataclasses

from skcomplex.core import base
from skcomplex.core.base import LawClass, LawKind, Trait, has_trait, law_classes
from skcomplex.core.laws import (
    BaseLaw,
    BoundaryLaw,
    CrossingLaw,
    DissipatorLaw,
    GeneratorLaw,
    GraphProductionLaw,
    LocalRuleLaw,
    MatchLaw,
    ProductionLaw,
    RateLaw,
    SubstitutionLaw,
    TransferLaw,
)
from skcomplex.core.laws.utils import Totalistic
from skcomplex.core.spaces import BoxNeighborhood
from skcomplex.engine.problems import symbol


def test_law_classes_share_the_base():
    """Test that every law class derives from the law base and declares a kind, a time class, and its traits."""
    classes = law_classes()
    for law in (MatchLaw, TransferLaw, DissipatorLaw, GeneratorLaw, ProductionLaw, GraphProductionLaw, SubstitutionLaw):
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


def test_targets_are_a_tuple_once_checked():
    """Test that checking a law turns its targets into a tuple, and that laws without targets keep them empty."""
    x, y = symbol('x'), symbol('y')
    assert isinstance(RateLaw(x, 1).targets, type(x))
    checked = RateLaw(x, 1).check()
    assert isinstance(checked.targets, tuple)
    assert checked.targets[0] is x
    assert LocalRuleLaw([x, y], BoxNeighborhood(), Totalistic({2: 1})).check().targets == (x, y)
    assert CrossingLaw(x - 1).check().targets == ()


def test_law_is_importable_at_run_time():
    """Test that the law base is a class that introspection can evaluate."""
    assert isinstance(BaseLaw, type)
    assert BaseLaw.__module__ == 'skcomplex.core.base._laws'


def test_base_holds_no_quantity():
    """Test that the base defines elements, interactions, and laws, and no base of quantities."""
    assert not hasattr(base, 'BaseQuantity')
    assert not any(name.endswith('Quantity') for name in base.__all__)


def test_law_runs_alone_once_prepared():
    """Test that a prepared rate law knows its targets, reads, and owner, and runs into its rate at given values."""
    n, r = symbol('n'), symbol('r')
    law = RateLaw(n, r * n).prepare('meadow/rabbits')
    assert law.run({'n': 10, 'r': 0.5}) == 5.0
    assert law.targets_ == (n,)
    assert law.reads_ == {n, r}
    assert law.owner_ == 'meadow/rabbits'
    assert law == RateLaw(n, r * n).check()
