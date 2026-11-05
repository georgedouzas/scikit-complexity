"""State of a discrete master equation on a ring, with a probability `P` and a potential `U`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, LocalRuleLaw
from ....core.components.spaces import BoxNeighborhood
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ....core.types import Array


class MicrostateElement(Element):
    """State of a discrete master equation on a ring, with a probability `P` and a potential `U`.

    Every step the probability moves to each neighbouring state `B` from a state `A` with the Glauber probability
    `w / (1 + exp(beta (U_B - U_A)))`, so `P_A(t + 1) = P_A + sum_B T_AB P_B - sum_B T_BA P_A`, and the stationary
    distribution is `exp(-beta U) / Z`.

    Args:
        label:
            The label of the state or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        P:
            The initial probability.

        U:
            The potential.

        beta:
            The inverse temperature.

        w:
            The attempt probability of a move to each neighbour, at most 1/2.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import MicrostateElement
        >>> MicrostateElement('state').w
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        P: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        U: Annotated[float, StateSpec()] = 0.0,
        beta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        w: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.P = P
        self.U = U
        self.beta = beta
        self.w = w

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the master equation step.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [LocalRuleLaw(s.P, BoxNeighborhood(), MicrostateElement._master_step, reads=(s.P, s.U))]

    @staticmethod
    def _glauber(source: Array, destination: Array, beta: float | Array, w: float | Array) -> Array:
        """Return the probability `w / (1 + exp(beta (U_destination - U_source)))` of a move between two states."""
        return w / (1 + np.exp(beta * (destination - source)))

    @staticmethod
    def _master_step(*around: Array, beta: float | Array, w: float | Array) -> Array:
        """Return the probabilities after one step of the master equation with moves to the two neighbouring states.

        The neighbourhood columns are the probabilities and then the potentials of the left state, the state, and the
        right state.
        """
        left, center, right, energy_left, energy, energy_right = around
        gain = (
            MicrostateElement._glauber(energy_left, energy, beta, w) * left
            + MicrostateElement._glauber(energy_right, energy, beta, w) * right
        )
        loss = (
            MicrostateElement._glauber(energy, energy_left, beta, w)
            + MicrostateElement._glauber(energy, energy_right, beta, w)
        ) * center
        return center + gain - loss
