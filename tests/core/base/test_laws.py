"""Test the `_laws` module of `skcomplex.core.base`."""

import dataclasses

import numpy as np
import pytest

from skcomplex.core.base import (
    BaseElement,
    BaseLaw,
    BaseQuantity,
    LawClass,
    LawKind,
    Trait,
    ValidationError,
    has_trait,
    law_classes,
    symbol,
)
from skcomplex.core.components import AggregateQuantity
from skcomplex.core.laws import (
    AuxiliaryQuantity,
    BoundaryLaw,
    EventLaw,
    InvariantQuantity,
    ObjectiveQuantity,
    RateLaw,
    Totalistic,
    TransferLaw,
)
from skcomplex.core.space import Neighborhood
from skcomplex.core.system import ObservableQuantity
from skcomplex.formalisms.agents import MatchLaw
from skcomplex.formalisms.lattices import LocalRuleLaw
from skcomplex.formalisms.quantum import DissipatorLaw, GeneratorLaw
from skcomplex.formalisms.rewriting import GraphProductionLaw, ProductionLaw, SubstitutionLaw


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
    assert LocalRuleLaw([x, y], Neighborhood.radius(), Totalistic({2: 1})).check().targets == (x, y)
    assert EventLaw(x - 1).check().targets == ()


def test_law_is_importable_at_run_time():
    """Test that the law base is a class that introspection can evaluate."""
    assert isinstance(BaseLaw, type)
    assert BaseLaw.__module__ == 'skcomplex.core.base._laws'


QUANTITIES = (ObservableQuantity, AuxiliaryQuantity, InvariantQuantity, ObjectiveQuantity, AggregateQuantity)


@pytest.mark.parametrize('quantity', QUANTITIES, ids=lambda quantity: quantity.__name__)
def test_quantity_is_a_quantity_only(quantity):
    """Test that every quantity derives from the quantity base and from neither the law nor the component base."""
    assert issubclass(quantity, BaseQuantity)
    assert not issubclass(quantity, BaseLaw)
    assert not issubclass(quantity, BaseElement)


@pytest.mark.parametrize('quantity', QUANTITIES, ids=lambda quantity: quantity.__name__)
def test_quantity_is_a_frozen_value_with_a_label_and_an_expression(quantity):
    """Test that every quantity is a frozen dataclass whose first fields are the label and the expression."""
    assert dataclasses.is_dataclass(quantity)
    assert quantity.__dataclass_params__.frozen
    assert [item.name for item in dataclasses.fields(quantity)][:2] == ['label', 'expression']


def test_check_returns_a_checked_copy():
    """Test that checking a quantity returns a quantity with its expression checked and leaves the original as is."""
    x = symbol('x')
    auxiliary = AuxiliaryQuantity('double', 2 * x)
    checked = auxiliary.check()
    assert checked == auxiliary
    assert BaseQuantity('plain', x).check() == BaseQuantity('plain', x)


def test_check_rejects_an_invalid_expression():
    """Test that checking a quantity whose expression is a NumPy array raises a named error that names it."""
    with pytest.raises(ValidationError, match='auxiliary double'):
        AuxiliaryQuantity('double', np.ones(2)).check()
