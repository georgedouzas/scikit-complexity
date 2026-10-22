"""Test the system vocabulary and the requirement and support declarations of primitives, solvers, and backends."""

import tomllib
from pathlib import Path

import pytest

import skcomplex.core.compile._validation as validation
from skcomplex.core.base import (
    VOCABULARY,
    DomainKind,
    Form,
    LawClass,
    LawKind,
    Registry,
    Requires,
    SpaceKind,
    Supports,
    TimeKind,
)
from skcomplex.core.compile import law_types, missing_declarations
from skcomplex.core.model import (
    EuclideanSpace,
    Field,
    GridSpace,
    Hamiltonian,
    LatticeSpace,
    Mesh,
    NetworkSpace,
    Rate,
    Rule,
    Sampler,
    Update,
)
from skcomplex.core.solvers import SOLVERS, Stepper

INVENTORY = Path(__file__).parents[3] / 'docs' / 'overview' / 'inventory.toml'


def test_vocabulary_matches_inventory():
    """Test that the core vocabulary and the inventory vocabulary define the same terms in every category."""
    inventory = tomllib.loads(INVENTORY.read_text())['vocabulary']
    assert set(inventory) == set(VOCABULARY)
    for category, terms in VOCABULARY.items():
        assert set(inventory[category]) == {str(term) for term in terms}, category


def test_inventory_law_classes_match_core():
    """Test that every law of the inventory vocabulary has the time semantics of the core law kind."""
    laws = tomllib.loads(INVENTORY.read_text())['vocabulary']['laws']
    for kind in LawKind:
        assert laws[str(kind)]['class'] == str(kind.law_class), kind


def test_every_primitive_solver_and_backend_declares():
    """Test that every law type, space, registered solver, and registered backend has a declaration."""
    assert missing_declarations() == []


def test_missing_solver_declaration_is_named(monkeypatch):
    """Test that a solver registered without a support declaration is named by the declaration check."""

    class Undeclared(Stepper):
        """Stepper without a support declaration."""

        form = Form.ODE

    registry = Registry[type[Stepper]]('solver')
    registry.register((Form.ODE, 'undeclared'), Undeclared)
    monkeypatch.setattr(validation, '_solver_registry', lambda: registry)
    assert missing_declarations() == ["The solver 'undeclared' (Undeclared) declares no support."]


@pytest.mark.parametrize(
    ('law', 'kind', 'law_class'),
    [
        (Rate, LawKind.RATE, LawClass.CONTINUOUS),
        (Update, LawKind.UPDATE, LawClass.DISCRETE),
        (Rule, LawKind.RULE, LawClass.DISCRETE),
        (Sampler, LawKind.SAMPLE, LawClass.DISCRETE),
        (Hamiltonian, LawKind.GENERATOR, LawClass.CONTINUOUS),
    ],
)
def test_law_kinds_and_classes(law, kind, law_class):
    """Test that law types declare their kind and the time semantics of the kind."""
    assert law.law_kind is kind
    assert law.law_kind.law_class is law_class


def test_law_requirements():
    """Test that rates need continuous time and numeric domains, and Monte Carlo laws need a lattice or network."""
    assert Rate.requires.accepts('time', TimeKind.CONTINUOUS)
    assert not Rate.requires.accepts('time', TimeKind.DISCRETE)
    assert not Rate.requires.accepts('domains', DomainKind.CATEGORICAL)
    assert Sampler.requires.space == {SpaceKind.LATTICE, SpaceKind.NETWORK}
    assert Requires().accepts('space', SpaceKind.MESH)


@pytest.mark.parametrize(
    ('space', 'kind'),
    [
        (GridSpace, SpaceKind.LATTICE),
        (LatticeSpace, SpaceKind.LATTICE),
        (NetworkSpace, SpaceKind.NETWORK),
        (EuclideanSpace, SpaceKind.CONTINUOUS),
        (Mesh, SpaceKind.MESH),
    ],
)
def test_space_kinds(space, kind):
    """Test that every space declares its kind."""
    assert space.space_kind is kind


def test_field_is_a_site_group():
    """Test that fields declare one member per site."""
    assert str(Field.component_kind) == 'site-group'


def test_solver_support_names_missing_terms():
    """Test that the declared support of the default ODE solver lacks noise, and the support lists what is missing."""
    support = SOLVERS.resolve((Form.ODE, 'auto')).supports
    assert support.missing('laws', {LawKind.RATE, LawKind.NOISE}) == ['noise']
    assert Supports().missing('time', {TimeKind.STATIC}) == ['static']
    assert len(law_types()) == len(set(law_types()))
