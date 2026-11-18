"""Test the `types` module of `skcomplex.core.specs`."""

from types import SimpleNamespace

from skcomplex.core.components.base import BaseComponent
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.specs import Role, RoleSpec


class PullInteraction(Interaction):
    """Interaction with one role declared by the short annotation."""

    def __init__(self, label: str, *, target: Role = 'ball') -> None:
        super().__init__(label)
        self.target = target

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseComponent]:
        """Return no laws."""
        return []


def test_role_declares_a_role():
    """`Role` declares an argument that names an element."""
    assert PullInteraction('pull')._get_specs()['target'] == RoleSpec()


class SumInteraction(Interaction):
    """Interaction that reads any number of inputs, declared as a tuple of roles."""

    def __init__(self, label: str, *, inputs: tuple[Role, ...] = ()) -> None:
        super().__init__(label)
        self.inputs = inputs


def test_tuple_of_roles_declares_a_role_that_binds_several_elements():
    """`tuple[Role, ...]` declares an argument that names several elements."""
    assert SumInteraction('sum', inputs=('a', 'b'))._get_specs()['inputs'] == RoleSpec(variadic=True)
