"""Test the system vocabulary and the requirement and support declarations of primitives, solvers, and backends."""

import tomllib
from importlib.resources import files
from pathlib import Path

import pytest
import skcomplex.formalisms.fractional  # noqa: F401  # registers the fractional solvers whose orders the tests read

import skcomplex.engine.compile._validation as validation
from skcomplex.core.base import (
    Binding,
    DomainKind,
    ElementKind,
    Form,
    LawClass,
    LawKind,
    OrderKind,
    Registry,
    RequirementSpec,
    SpaceKind,
    SupportSpec,
    TimeKind,
    law_classes,
)
from skcomplex.core.elements import FieldElement
from skcomplex.core.laws import GeneratorLaw, RateLaw, RuleLaw, UpdateLaw
from skcomplex.core.spaces import ContinuousSpace, LatticeSpace, MeshSpace, NetworkSpace
from skcomplex.core.specs import SamplerSpec
from skcomplex.engine.backends import BACKENDS
from skcomplex.engine.compile import SOLVERS, missing_declarations
from skcomplex.engine.solvers import Solver

TEST_TERMS = {str(outside.DECAY), str(outside.DECAYING)}
INVENTORY = Path(str(files('tests'))).parent / 'docs' / 'overview' / 'inventory.toml'


def test_vocabulary_matches_inventory():
    """Test that the core vocabulary and the inventory vocabulary define the same terms in every category."""
    inventory = tomllib.loads(INVENTORY.read_text())['vocabulary']
    vocabulary = {
        'time': tuple(TimeKind),
        'space': tuple(SpaceKind),
        'components': tuple(ElementKind),
        'bindings': tuple(Binding),
        'laws': tuple(LawKind),
        'domains': tuple(DomainKind),
    }
    assert set(inventory) == set(vocabulary)
    for category, terms in vocabulary.items():
        assert set(inventory[category]) == {str(term) for term in terms} - TEST_TERMS, category


def test_inventory_law_classes_match_core():
    """Test that every law of the inventory vocabulary has the time semantics of the core law kind."""
    laws = tomllib.loads(INVENTORY.read_text())['vocabulary']['laws']
    for law in law_classes():
        if str(law.law_kind) in TEST_TERMS:
            continue
        assert laws[str(law.law_kind)]['class'] == str(law.time_class), law


def test_every_primitive_solver_and_backend_declares():
    """Test that every law type, space, registered solver, and registered backend has a declaration."""
    assert missing_declarations() == []


def test_missing_solver_declaration_is_named(monkeypatch):
    """Test that a solver registered without a support declaration is named by the declaration check."""

    class UndeclaredSolver(Solver):
        """Solver without a support declaration."""

        form = Form.ODE

    registry = Registry[type[Solver]]('solver')
    registry.register((Form.ODE, 'undeclared'), UndeclaredSolver)
    monkeypatch.setattr(validation, 'solver_registry', lambda: registry)
    assert missing_declarations() == ["The solver 'undeclared' (UndeclaredSolver) declares no support."]


@pytest.mark.parametrize(
    ('law', 'kind', 'law_class'),
    [
        (RateLaw, LawKind.RATE, LawClass.CONTINUOUS),
        (UpdateLaw, LawKind.UPDATE, LawClass.DISCRETE),
        (RuleLaw, LawKind.RULE, LawClass.DISCRETE),
        (SamplerSpec, LawKind.SAMPLE, LawClass.DISCRETE),
        (GeneratorLaw, LawKind.GENERATOR, LawClass.CONTINUOUS),
    ],
)
def test_law_kinds_and_classes(law, kind, law_class):
    """Test that law types declare their kind and the time semantics of the kind."""
    assert law.law_kind == kind
    assert law.time_class is law_class


def test_law_requirements():
    """Test that rates need continuous time and numeric domains, and Monte Carlo laws need a lattice or network."""
    assert RateLaw.requires.accepts('time', TimeKind.CONTINUOUS)
    assert not RateLaw.requires.accepts('time', TimeKind.DISCRETE)
    assert not RateLaw.requires.accepts('domains', DomainKind.CATEGORICAL)
    assert SamplerSpec.requires.space == {SpaceKind.LATTICE, SpaceKind.NETWORK}
    assert RequirementSpec().accepts('space', SpaceKind.MESH)


@pytest.mark.parametrize(
    ('space', 'kind'),
    [
        (LatticeSpace, SpaceKind.LATTICE),
        (NetworkSpace, SpaceKind.NETWORK),
        (ContinuousSpace, SpaceKind.CONTINUOUS),
        (MeshSpace, SpaceKind.MESH),
    ],
)
def test_space_kinds(space, kind):
    """Test that every space declares its kind."""
    assert space.space_kind == kind


def test_field_is_a_site_group():
    """Test that fields declare one member per site."""
    assert str(FieldElement.component_kind) == 'site-group'


def test_solver_support_names_missing_terms():
    """Test that the declared support of the default ODE solver lacks noise, and the support lists what is missing."""
    support = SOLVERS.resolve((Form.ODE, 'auto')).supports
    assert support.missing('laws', {LawKind.RATE, LawKind.NOISE}) == ['noise']
    assert SupportSpec().missing('time', {TimeKind.STATIC}) == ['static']
    assert len(law_classes()) == len(set(law_classes()))


def test_every_solver_and_backend_declares_orders():
    """Test that every registered solver and backend declares its orders, and the fractional solvers declare both."""
    for key in SOLVERS.keys:
        assert SOLVERS.resolve(key).supports.orders, key
    for key in BACKENDS.keys:
        assert BACKENDS.resolve(key).supports.orders == set(OrderKind), key
    for name in ('grunwald-letnikov', 'l1', 'fractional-abm', 'auto'):
        assert SOLVERS.resolve((Form.FRACTIONAL, name)).supports.orders == set(OrderKind)
    assert SOLVERS.resolve((Form.ODE, 'auto')).supports.orders == {OrderKind.INTEGER}
