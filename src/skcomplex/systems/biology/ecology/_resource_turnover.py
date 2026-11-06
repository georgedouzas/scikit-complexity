"""Supply of resources, by logistic renewal or a chemostat, and their depletion by the consumers."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np

from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import total
from ....core.errors import ValidationError
from ....core.specs import RoleSpec
from ._consumption import ConsumptionInteraction

SUPPLIES = ('logistic', 'chemostat')


class ResourceTurnoverInteraction(ConsumptionInteraction):
    """Supply of resources, by logistic renewal or a chemostat, and their depletion by the consumers.

    A `'logistic'` supply renews resource `a` at `r_a R_a (K_a - R_a) / K_a`, and a `'chemostat'` supply at `kappa_a -
    omega_a R_a`. Substitutable resources lose `sum_j c_ja N_j R_a`, and essential resources lose `sum_j v_ja N_j g_j`
    by the fixed stoichiometry `v` of the consumers.

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

        supply:
            `'logistic'` for self-renewing resources or `'chemostat'` for resources supplied from outside.

        v:
            The stoichiometry of essential resources in the consumers, one row per consumer, or `None` for `c`.

    Raises:
        ValidationError:
            The supply or the kind is unknown, or the exponent of essential resources is not below -1.

    Examples:
        >>> from skcomplex.systems.biology.ecology import ResourceTurnoverInteraction
        >>> ResourceTurnoverInteraction('turnover', consumers='consumers', resources='resources').supply
        'logistic'
    """

    def __init__(
        self,
        label: str,
        *,
        consumers: Annotated[str, RoleSpec()],
        resources: Annotated[str, RoleSpec()],
        c: Sequence[Sequence[float]] = ((1.0,),),
        kind: str = 'substitutable',
        n: float = -10.0,
        supply: str = 'logistic',
        v: Sequence[Sequence[float]] | None = None,
    ) -> None:
        super().__init__(label, consumers=consumers, resources=resources, c=c, kind=kind, n=n)
        self.supply = supply
        self.v = v

    def _check(self) -> None:
        """Check the supply, the kind, and the exponent.

        Raises:
            ValidationError:
                The supply or the kind is unknown, or the exponent of essential resources is not below -1.
        """
        super()._check()
        if self.supply not in SUPPLIES:
            error_msg = f'The supply must be one of {", ".join(SUPPLIES)}, not {self.supply!r}.'
            raise ValidationError(error_msg)

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the supply and the depletion of the resources.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        consumers, resources = s.consumers, s.resources
        R = resources.R
        if self.supply == 'logistic':
            renewal = resources.r * R * (resources.K - R) / resources.K
        else:
            renewal = resources.kappa - resources.omega * R
        if self.kind == 'essential':
            stoichiometry = np.asarray(self.c if self.v is None else self.v, dtype=float)
            depletion = total(consumers.N * self._essential_growth(R), weights=stoichiometry.T)
        else:
            depletion = R * total(consumers.N, weights=np.asarray(self.c, dtype=float).T)
        return [RateLaw(R, renewal - depletion)]
