"""Growth of consumers on the resources they take up, minus their maintenance."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

import numpy as np

from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import total
from ._consumption import ConsumptionInteraction


class ConsumerGrowthInteraction(ConsumptionInteraction):
    """Growth of consumers on the resources they take up, minus their maintenance.

    On substitutable resources, `dN_i/dt = e_i N_i (sum_b w_b c_ib R_b - m_i)`. On essential resources, `dN_i/dt =
    N_i (g_i - m_i)` with `g_i = (sum_b (c_ib R_b)^n)^(1/n)`.

    Args:
        label:
            The label of the interaction.

        consumers:
            The path of the group of consumers.

        resources:
            The path of the group of resources.

        c:
            The uptake rates, one row per consumer and one column per resource.

        kind:
            `'substitutable'` or `'essential'` resources.

        n:
            The exponent of the growth on essential resources, below -1.

    Examples:
        >>> from skcomplex.systems.biology.ecology import ConsumerGrowthInteraction
        >>> ConsumerGrowthInteraction('growth', consumers='consumers', resources='resources').c
        ((1.0,),)
    """

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the growth of the consumers.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        consumers, resources = s.consumers, s.resources
        if self.kind == 'essential':
            growth = self._essential_growth(resources.R) - consumers.m
        else:
            intake = total(resources.w * resources.R, weights=np.asarray(self.c, dtype=float))
            growth = consumers.e * (intake - consumers.m)
        return [RateLaw(consumers.N, consumers.N * growth)]
