"""Harmonic optical trap on overdamped particles, the potential `sum_d k_d (x_d - c_d(t))^2 / 2` of every axis."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression, oo, sin
from sage.sets.real_set import RealSet

from ....core.components.interactions import GradientInteraction
from ....core.components.laws.utils import pulse_train
from ....core.specs import ParameterSpec, RoleSpec


class OpticalTrapInteraction(GradientInteraction):
    """Harmonic optical trap on overdamped particles, the potential `sum_d k_d (x_d - c_d(t))^2 / 2` of every axis.

    The center oscillates as `c(t) = amplitude sin(2 pi frequency t)`, and a blinking trap is on for the first
    `duty` fraction of every `blink_period` and off for the rest. The force divides by the friction `gamma` of the
    target.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle or the group.

        k:
            The stiffness along every axis.

        amplitude:
            The amplitude of the oscillation of the center along every axis.

        frequency:
            The frequency of the oscillation of the center.

        blink_period:
            The period of a blinking trap, or `None` for a trap that is always on.

        duty:
            The fraction of every blinking period during which the trap is on.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import OpticalTrapInteraction
        >>> OpticalTrapInteraction('tweezers', target='bead', k=[1.0, 0.25]).k
        [1.0, 0.25]
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        k: Annotated[npt.ArrayLike, ParameterSpec(vector=True)] = 1.0,
        amplitude: Annotated[npt.ArrayLike, ParameterSpec(vector=True)] = 0.0,
        frequency: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        blink_period: float | None = None,
        duty: float = 0.5,
    ) -> None:
        super().__init__(label, target=target, states='x', divisor='gamma')
        self.k = k
        self.amplitude = amplitude
        self.frequency = frequency
        self.blink_period = blink_period
        self.duty = duty

    def _get_potential(self, namespace: SimpleNamespace) -> Expression:
        """Return the potential of the trap, switched by the blinking.

        Args:
            namespace:
                The namespace of the symbols the potential can use.

        Returns:
            potential:
                The potential.
        """
        s = namespace
        center = s.amplitude * sin(2 * math.pi * s.frequency * s.t)
        on = 1 if self.blink_period is None else pulse_train(s.t, self.blink_period, self.duty * self.blink_period)
        energy = sum(
            (
                stiffness * (item - middle) ** 2 / 2
                for stiffness, item, middle in zip(s.k, s.target.x, center, strict=True)
            ),
            SR(0),
        )
        return on * energy
