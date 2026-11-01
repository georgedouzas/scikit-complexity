"""Rosenzweig-MacArthur predation of fast prey by slow predators, whose prey crowding grows at a constant rate."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.interactions import Interaction
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import clip
from ....specs import ParameterSpec, RoleSpec


class RosenzweigMacArthurInteraction(Interaction):
    """Rosenzweig-MacArthur predation of fast prey by slow predators, whose prey crowding grows at a constant rate.

    In units of the predators' time scale, `kappa dx/dt = x (1 - phi x) - x y / (1 + eta x)` and
    `dy/dt = x y / (1 + eta x) - y`. The crowding `phi`, the inverse of the prey's carrying capacity, rises at the rate
    `rate` from `phi` to `phi_end` and stays there.

    Args:
        label:
            The label of the interaction.

        prey:
            The path of the prey population `x`.

        predators:
            The path of the predator population `y`.

        kappa:
            The ratio of the time scales of the prey and the predators.

        eta:
            The handling time of the predators.

        phi:
            The initial crowding of the prey.

        phi_end:
            The final crowding of the prey.

        rate:
            The rate at which the crowding rises.

    Examples:
        >>> from skcomplex.models.biology.ecology import RosenzweigMacArthurInteraction
        >>> RosenzweigMacArthurInteraction('predation', prey='prey', predators='predators', rate=0.01).rate
        0.01
    """

    def __init__(
        self,
        label: str,
        *,
        prey: Annotated[str, RoleSpec()],
        predators: Annotated[str, RoleSpec()],
        kappa: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.01,
        eta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.8,
        phi: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        phi_end: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        rate: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label)
        self.prey = prey
        self.predators = predators
        self.kappa = kappa
        self.eta = eta
        self.phi = phi
        self.phi_end = phi_end
        self.rate = rate

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the prey and the predators under the ramp of the crowding.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        x, y = s.prey.n, s.predators.n
        crowding = clip(s.phi + s.rate * s.t, high=s.phi_end)
        feeding = x * y / (1 + s.eta * x)
        return [RateLaw(x, (x * (1 - crowding * x) - feeding) / s.kappa), RateLaw(y, feeding - y)]
