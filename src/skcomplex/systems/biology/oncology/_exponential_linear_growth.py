"""Growth that is exponential, `dn/dt = r0 n`, until it reaches the linear rate `r`, and linear after."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import RateInteraction
from ....core.components.laws.utils import minimum
from ....core.specs import ParameterSpec, RoleSpec


class ExponentialLinearGrowthInteraction(RateInteraction):
    """Growth that is exponential, `dn/dt = r0 n`, until it reaches the linear rate `r`, and linear after.

    The switch happens at `tau = log(r / (r0 n0)) / r0`, where the two rates meet, so the solution is continuously
    differentiable.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the tumour.

        r0:
            The exponential growth rate.

        r:
            The linear growth rate.

    Examples:
        >>> from skcomplex.systems.biology.oncology import ExponentialLinearGrowthInteraction
        >>> ExponentialLinearGrowthInteraction('growth', target='tumour', r=67.8).r
        67.8
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        r0: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.31,
        r: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 67.8,
    ) -> None:
        super().__init__(label, target=target, state='n')
        self.r0 = r0
        self.r = r

    def _get_rate(self, s: SimpleNamespace) -> Expression:
        """Return the rate `min(r0 n, r)`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            rate:
                The rate of the size of the tumour.
        """
        exponential = s.r0 * s.target.n
        return minimum(exponential, s.r)
