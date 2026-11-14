"""Test the `types` module of `skcomplex.core.specs`."""

from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.specs import Parameter, ParameterSpec, Role, RoleSpec, State, StateSpec


class ParticleElement(Element):
    """Element of a particle with a position, a velocity, and a mass."""

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[float, StateSpec(rate='v')] = 0.0,
        v: State[float] = 0.0,
        m: Parameter[float] = 1.0,
    ) -> None:
        super().__init__(label)
        self.x = x
        self.v = v
        self.m = m


def test_state_declares_a_state():
    """`State[float]` declares a real state with the default spec."""
    assert ParticleElement('ball').get_states()['v'] == StateSpec()


def test_parameter_declares_a_parameter():
    """`Parameter[float]` declares a parameter with the default spec."""
    assert ParticleElement('ball')._get_specs()['m'] == ParameterSpec()


def test_role_stands_for_a_role_spec():
    """`Role` stands for a string annotated with the default role spec."""
    assert Role.__value__.__metadata__ == (RoleSpec(),)
