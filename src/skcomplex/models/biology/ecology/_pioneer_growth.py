"""Michaelis-Menten regrowth of pioneer trees, whose input from seeds the other groups suppress."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.interactions import RateInteraction
from ....core.laws.utils import clip
from ....specs import ParameterSpec, RoleSpec


class PioneerGrowthInteraction(RateInteraction):
    """Michaelis-Menten regrowth of pioneer trees, whose input from seeds the other groups suppress.

    Alone, the volume follows `V(t) = V_mat t / (K_M + t)` after a clear cut, which is
    `dV/dt = V (V / V_mat - 2) / K_M + C V_mat / K_M` with `C = 1`. The other groups reduce the factor
    `C = 1 - w_I V_I / V_mat,I - w_C V_C / V_mat,C`, with weights normalized to sum to one, and no lower than 0.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the pioneer stand.

        intermediate:
            The path of the intermediate stand.

        climax:
            The path of the climax stand.

        half_time:
            The time `K_M` the pioneers take to reach half their mature volume.

        w_intermediate:
            The weight of the intermediate group in the competition.

        w_climax:
            The weight of the climax group in the competition.

    Examples:
        >>> from skcomplex.models.biology.ecology import PioneerGrowthInteraction
        >>> PioneerGrowthInteraction('growth', target='keora', intermediate='gewa', climax='sundari').half_time
        3.65
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        intermediate: Annotated[str, RoleSpec()],
        climax: Annotated[str, RoleSpec()],
        half_time: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 3.65,
        w_intermediate: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.32,
        w_climax: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.68,
    ) -> None:
        super().__init__(label, target=target, state='volume')
        self.intermediate = intermediate
        self.climax = climax
        self.half_time = half_time
        self.w_intermediate = w_intermediate
        self.w_climax = w_climax

    def _get_rate(self, namespace: SimpleNamespace) -> Expression:
        """Return the regrowth with the suppressed input from seeds.

        Args:
            namespace:
                The namespace of the symbols the rate can use, with the stands.

        Returns:
            rate:
                The rate of the volume of the pioneers.
        """
        s = namespace
        volume, mature = s.target.volume, s.target.mature
        suppression = clip(
            1 - PioneerGrowthInteraction._crowding(s.intermediate, s.climax, s.w_intermediate, s.w_climax),
            low=0,
        )
        rate: Expression = (volume * (volume / mature - 2) + suppression * mature) / s.half_time
        return rate

    @staticmethod
    def _crowding(
        first: SimpleNamespace,
        second: SimpleNamespace,
        weight_first: Expression,
        weight_second: Expression,
    ) -> Expression:
        """Return the weighted share of the mature volumes that two stands fill."""
        total = weight_first + weight_second
        return (
            weight_first / total * first.volume / first.mature + weight_second / total * second.volume / second.mature
        )
