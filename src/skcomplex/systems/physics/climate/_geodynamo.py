"""Dynamo of Gissinger's geodynamo, its quadrupolar and dipolar fields and the velocity of the flow."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class GeodynamoElement(Element):
    """Dynamo of Gissinger's geodynamo, its quadrupolar and dipolar fields and the velocity of the flow.

    The quadrupolar field `Q`, the dipolar field `D`, and the velocity `V` follow `dQ/dt = mu Q - V D`,
    `dD/dt = -nu D + V Q`, and `dV/dt = Gamma - V + Q D`.

    Args:
        label:
            The label of the dynamo.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        Q:
            The initial quadrupolar field.

        D:
            The initial dipolar field.

        V:
            The initial velocity.

        mu:
            The growth rate of the quadrupole.

        nu:
            The damping rate of the dipole.

        Gamma:
            The forcing of the flow.

    Examples:
        >>> from skcomplex.systems.physics.climate import GeodynamoElement
        >>> GeodynamoElement('dynamo').Gamma
        0.9
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        Q: Annotated[float, StateSpec()] = 0.1,
        D: Annotated[float, StateSpec()] = 0.1,
        V: Annotated[float, StateSpec()] = 0.1,
        mu: Annotated[float, ParameterSpec()] = 0.119,
        nu: Annotated[float, ParameterSpec()] = 0.1,
        Gamma: Annotated[float, ParameterSpec()] = 0.9,
    ) -> None:
        super().__init__(label, visual=visual)
        self.Q = Q
        self.D = D
        self.V = V
        self.mu = mu
        self.nu = nu
        self.Gamma = Gamma

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the growth and damping of both fields and the forcing of the flow.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate laws of the quadrupolar field, the dipolar field, and the velocity.
        """
        return [
            RateLaw(s.Q, s.mu * s.Q - s.V * s.D),
            RateLaw(s.D, -s.nu * s.D + s.V * s.Q),
            RateLaw(s.V, s.Gamma - s.V + s.Q * s.D),
        ]
