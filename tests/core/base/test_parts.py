"""Test the `_parts` module of `skcomplex.core.base`."""

import pkgutil
from importlib import import_module
from types import SimpleNamespace

import pytest

from skcomplex.core.base import BasePart, symbol
from skcomplex.core.elements import BaseElement
from skcomplex.core.laws import BaseLaw
from skcomplex.core.system import System
from skcomplex.models.biology.ecology import ExponentialGrowthInteraction, PopulationElement


def test_base_holds_the_parts_and_shared_modules():
    """Test that the base package holds the part bases and the shared types, errors, and helpers only."""
    base = import_module('skcomplex.core.base')
    modules = {info.name for info in pkgutil.iter_modules(base.__path__)}
    assert modules == {'_parts', '_elements', '_interactions', '_laws', '_types', '_errors', '_utils'}


def test_parts_derive_from_the_part_base():
    """Test that elements, interactions, laws, and systems are parts, and a part cannot be made without its hooks."""
    assert all(issubclass(base, BasePart) for base in (BaseElement, BaseInteraction, BaseLaw, System))
    with pytest.raises(TypeError, match='abstract'):
        BasePart()


def test_system_runs_into_the_laws_of_its_parts():
    """Test that a prepared system runs into the laws of its element and of its interaction, in order."""
    system = System(
        elements=[PopulationElement('rabbits')],
        interactions=[ExponentialGrowthInteraction('growth', target='rabbits')],
    ).prepare()
    rabbits = SimpleNamespace(n=symbol('n'), r=symbol('r'))
    s = SimpleNamespace(rabbits=rabbits, growth=SimpleNamespace(target=rabbits, r=symbol('g')))
    laws = system.run(s)
    assert [type(law).__name__ for law in laws] == ['RateLaw']
    assert system.components[0].declarations_
