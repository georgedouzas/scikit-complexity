"""Theta forcing of a population by the current `I0 (1 - cos(2 pi nu t)) / 2`, as in optogenetic stimulation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import cos, oo, pi
from sage.sets.real_set import RealSet

from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.specs import ParameterSpec, RoleSpec


class ThetaForcingInteraction(Interaction):
    """Theta forcing of a population by the current `I0 (1 - cos(2 pi nu t)) / 2`, as in optogenetic stimulation.

    The current enters the potential of the population as its excitabilities do, `dv/dt += I(t) / tau`. Time is in
    milliseconds and the frequency `nu` in hertz.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the forced population.

        I0:
            The peak current.

        nu:
            The frequency of the forcing, in hertz.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import ThetaForcingInteraction
        >>> ThetaForcingInteraction('theta', target='e', nu=8.0).nu
        8.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        I0: Annotated[float, ParameterSpec()] = 10.0,
        nu: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.I0 = I0
        self.nu = nu

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the forcing current of the potential.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        current = s.I0 / 2 * (1 - cos(2 * pi * s.nu * s.t / 1000))
        return [RateLaw(s.target.v, current / s.target.tau)]
