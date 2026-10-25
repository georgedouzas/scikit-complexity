"""Test the `_kinds` module of `skcomplex.core.base`."""

import copy
import pickle  # nosec B403: the test loads only the pickles it makes

import pytest

from skcomplex.core.base import Form, LawKind, NoiseKind, RegistryError, SpaceKind, Term, Trait
from skcomplex.engine.solvers import EulerMaruyamaSolver, HeunSolver, MilsteinSolver


class Colour(Term):
    """Vocabulary of colours, defined only for these tests."""


def test_term_interns_by_name():
    """Test that creating a term of a registered name returns the registered term."""
    assert Form('ODE', 'ordinary differential equations') is Form.ODE


def test_term_rejects_other_value():
    """Test that a registered name with another value raises a named error that names both values."""
    with pytest.raises(RegistryError, match="'ordinary differential equations'"):
        Form('ODE', 'other equations')


def test_term_registers_new_terms_per_vocabulary():
    """Test that a new term joins its own vocabulary only, after the registered ones."""
    red = Colour('RED', 'red')
    assert Colour.registered()[-1] is red
    assert red not in Form.registered()
    assert Colour('RED', 'red') is red


def test_term_survives_copy_and_pickle():
    """Test that copying and pickling a term return the registered term."""
    assert copy.deepcopy(LawKind.RATE) is LawKind.RATE
    assert copy.copy(SpaceKind.MESH) is SpaceKind.MESH
    assert pickle.loads(pickle.dumps(Form.SDE)) is Form.SDE  # noqa: S301  # nosec B301: loads only what it dumped


def test_term_is_frozen():
    """Test that a term cannot be changed."""
    with pytest.raises(AttributeError, match='frozen'):
        Trait.FORMLESS.value = 'other'


def test_term_text_and_representation():
    """Test that a term prints its value and represents itself by vocabulary and name."""
    assert str(LawKind.LOCAL_RULE) == 'local-rule'
    assert repr(Form.MONTE_CARLO) == 'Form.MONTE_CARLO'
    assert Form.SDE.name == 'SDE'


def test_declared_noise_support():
    """Test that Euler-Maruyama and Heun declare every noise structure and Milstein only diagonal noise."""
    assert set(EulerMaruyamaSolver.supports.noise) == set(NoiseKind)
    assert set(HeunSolver.supports.noise) == set(NoiseKind)
    assert set(MilsteinSolver.supports.noise) == {NoiseKind.DIAGONAL}
