"""Test the `types` module of `skcomplex.core`."""

import typing
from types import SimpleNamespace
from typing import Annotated

from skcomplex.core.components.base import BaseComponent
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.specs import RoleSpec
from skcomplex.core.types import Preference, Role


def test_preferences_score_two_agents():
    """Test that a preference is a function of two agents that returns a score."""
    assert typing.get_args(Preference.__value__)[-1] is float


def test_role_is_the_path_of_an_element():
    """A role holds the path of an element as a string."""
    assert Role.__value__ is str


class PullInteraction(Interaction):
    """Interaction with roles declared by type hints."""

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'ball',
        source: Role | None = None,
        switch: Annotated[Role, 'boolean'] = 'lamp',
        inputs: tuple[Role, ...] = (),
    ) -> None:
        super().__init__(label)
        self.target = target
        self.source = source
        self.switch = switch
        self.inputs = inputs

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseComponent]:
        """Return no laws."""
        return []


def test_role_declares_a_role():
    """`Role` declares an argument that names an element."""
    assert PullInteraction('pull')._get_specs()['target'] == RoleSpec()


def test_optional_role_may_bind_nothing():
    """`Role | None` declares a role that may be `None`."""
    assert PullInteraction('pull')._get_specs()['source'] == RoleSpec(optional=True)


def test_role_with_words_accepts_these_domains():
    """`Annotated[Role, 'boolean']` declares a role that accepts boolean states only."""
    assert PullInteraction('pull')._get_specs()['switch'] == RoleSpec(domains=('boolean',))


def test_tuple_of_roles_declares_a_role_that_binds_several_elements():
    """`tuple[Role, ...]` declares an argument that names several elements."""
    assert PullInteraction('pull', inputs=('a', 'b'))._get_specs()['inputs'] == RoleSpec(variadic=True)
