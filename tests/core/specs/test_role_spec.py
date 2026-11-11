"""Test the `_role_spec` module of `skcomplex.core.specs`."""

import pytest

from skcomplex.core.enums import ElementKind
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import RoleSpec


def test_role_spec_is_variadic_or_a_noise_role():
    """Test that a role is either variadic or a noise role, not both."""
    assert RoleSpec(noise=True).noise
    with pytest.raises(ValidationError, match='not both'):
        RoleSpec(variadic=True, noise=True)._check()


def test_role_spec_may_be_optional():
    """Test that a role may be declared optional, so it may stay unbound."""
    assert RoleSpec(optional=True).optional
    assert not RoleSpec().optional


def test_role_spec_defaults_to_any_element_and_no_states():
    """Test that a role that declares nothing accepts every kind of element and reads no state."""
    role = RoleSpec()
    assert role.states == ()
    assert all(role.accepts(kind) for kind in ElementKind)


def test_role_spec_accepts_only_the_kinds_it_names():
    """Test that a role accepts the kinds it names, or every kind when it names any."""
    role = RoleSpec(elements=(ElementKind.GROUP,), states=('state',))
    assert role.accepts(ElementKind.GROUP)
    assert not role.accepts(ElementKind.ELEMENT)
    assert RoleSpec(elements=(ElementKind.ANY,)).accepts(ElementKind.ELEMENT)
