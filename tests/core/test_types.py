"""Test the `types` module of `skcomplex.core`."""

import typing
from types import SimpleNamespace
from typing import Annotated, Any

import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.spaces import BaseSpace
from skcomplex.core.errors import RoleElementError, RoleError
from skcomplex.core.systems import System
from skcomplex.core.types import Preference, Role


def test_preferences_score_two_agents():
    """Test that a preference is a function of two agents that returns a score."""
    assert typing.get_args(Preference.__value__)[-1] is float


def test_role_is_the_path_of_an_element():
    """A role holds the path of an element as a string."""
    assert Role.__value__ is str


class PullInteraction(Interaction):
    """Interaction with roles declared by type hints, which pulls the value `x` of its target down."""

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'ball',
        source: Role | None = None,
        switch: Annotated[Role, 'boolean'] | None = None,
        inputs: tuple[Role, ...] = (),
    ) -> None:
        super().__init__(label)
        self.target = target
        self.source = source
        self.switch = switch
        self.inputs = inputs

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the pull on the target, the sum of the inputs when there are any, minus the level of the switch."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            pull = sum((member.x for member in s.inputs), 0) if self.inputs else -1
            self.laws_ = [RateLaw(s.target.x, pull)]
            if self.switch is not None:
                self.laws_.append(RateLaw(s.target.x, -s.switch.level))


def _prepare(*elements: Element, **roles: Any) -> System:
    """Return the prepared system of a ball, the given elements, and one pull with the given roles."""
    return System(
        elements=[Element('ball', values={'x': 1.0}), *elements],
        interactions=[PullInteraction('pull', **roles)],
    ).prepare()


def test_role_declares_a_role():
    """`Role` declares an argument that names an element, whose values the laws read and change."""
    assert _prepare().equations_ == {'ball.x': '-1'}
    with pytest.raises(RoleError, match="binds the role 'target' to 'nowhere'"):
        _prepare(target='nowhere')


def test_optional_role_may_bind_nothing():
    """`Role | None` declares a role that may be `None`, and a role bound to a path still checks the path."""
    assert _prepare(source=None).states_ == ('ball.x',)
    with pytest.raises(RoleError, match="binds the role 'source' to 'nowhere'"):
        _prepare(source='nowhere')


def test_role_with_words_accepts_these_domains():
    """`Annotated[Role, 'boolean']` declares a role that accepts boolean states only."""
    assert _prepare(Element('lamp', values={'level': True}), switch='lamp').states_ == ('ball.x',)
    with pytest.raises(RoleElementError, match="accepts the domains boolean, but the element 'lamp'"):
        _prepare(Element('lamp', values={'level': 0.5}), switch='lamp')


def test_tuple_of_roles_declares_a_role_that_binds_several_elements():
    """`tuple[Role, ...]` declares an argument that names several elements."""
    a, b = Element('a', values={'x': 2.0}), Element('b', values={'x': 3.0})
    assert _prepare(a, b, inputs=('a', 'b')).equations_ == {'ball.x': 'a.x + b.x'}
