"""Test the `_kinds` module of `skcomplex.core.base`."""

import copy
import pickle

import pytest

from skcomplex.core.base import Form, LawKind, NoiseKind, RegistryError, SpaceKind, Term, Trait
from skcomplex.engine.solvers import EulerMaruyamaSolver, HeunSolver, MilsteinSolver


class Colour(Term):
    """Vocabulary of colours, defined only for these tests."""


def test_term_registers_by_name():
    """Test that registering a registered name with its value returns a term equal to the registered one."""
    assert Form.register('ODE', 'ordinary differential equations') == Form.ODE


def test_term_rejects_other_value():
    """Test that a registered name with another value raises a named error that names both values."""
    with pytest.raises(RegistryError, match="'ordinary differential equations'"):
        Form.register('ODE', 'other equations')


def test_term_registers_new_terms_per_vocabulary():
    """Test that a new term joins its own vocabulary only, after the registered ones."""
    red = Colour.register('RED', 'red')
    assert Colour.registered()[-1] == red
    assert red not in Form.registered()
    assert Colour.register('RED', 'red') == red


def test_term_survives_copy_and_pickle():
    """Test that copying and pickling a term return a term equal to the registered one."""
    assert copy.deepcopy(LawKind.RATE) == LawKind.RATE
    assert copy.copy(SpaceKind.MESH) == SpaceKind.MESH
    assert pickle.loads(pickle.dumps(Form.SDE)) == Form.SDE  # noqa: S301  # nosec B301: loads only what it dumped


def test_term_value_cannot_be_set():
    """Test that the value of a term is read only."""
    with pytest.raises(AttributeError, match='value'):
        Trait.FORMLESS.value = 'other'


def test_term_text_and_name():
    """Test that a term prints its value and knows its registered name."""
    assert str(LawKind.LOCAL_RULE) == 'local-rule'
    assert Form.SDE.name == 'SDE'
    assert Form.MONTE_CARLO.value == 'Monte Carlo sampling'


def test_declared_noise_support():
    """Test that Euler-Maruyama and Heun declare every noise structure and Milstein only diagonal noise."""
    assert set(EulerMaruyamaSolver.supports.noise) == set(NoiseKind)
    assert set(HeunSolver.supports.noise) == set(NoiseKind)
    assert set(MilsteinSolver.supports.noise) == {NoiseKind.DIAGONAL}
