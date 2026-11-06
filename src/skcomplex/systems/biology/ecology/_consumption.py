"""Consumption of substitutable or essential resources by consumer species, the only way the consumers interact."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression

from ....core.components.interactions import Interaction
from ....core.components.laws.utils import total
from ....core.errors import ValidationError
from ....core.specs import RoleSpec

KINDS = ('substitutable', 'essential')


class ConsumptionInteraction(Interaction):
    """Consumption of substitutable or essential resources by consumer species, the only way the consumers interact.

    Consumer `i` takes up resource `a` at the rate `c_ia R_a` per individual. Substitutable resources add their
    energies `w_b c_ib R_b`, and essential resources all limit the growth, `g_i = (sum_b (c_ib R_b)^n)^(1/n)` for
    `n < -1`, which tends to Liebig's law `g_i = min_b c_ib R_b` as `n` goes to minus infinity. Its subclasses give the
    laws of the consumers and of the resources, which read the other group through reductions.

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

    Raises:
        ValidationError:
            The kind is unknown, or the exponent of essential resources is not below -1.

    Examples:
        >>> from skcomplex.systems.biology.ecology import ConsumptionInteraction
        >>> uptake = ((0.5, 0.3), (0.4, 0.6))
        >>> ConsumptionInteraction('consumption', consumers='consumers', resources='resources', c=uptake).kind
        'substitutable'
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
    ) -> None:
        super().__init__(label)
        self.consumers = consumers
        self.resources = resources
        self.c = c
        self.kind = kind
        self.n = n

    def _check(self) -> None:
        """Check the kind and the exponent.

        Raises:
            ValidationError:
                The kind is unknown, or the exponent of essential resources is not below -1.
        """
        super()._check()
        if self.kind not in KINDS:
            error_msg = f'The kind must be one of {", ".join(KINDS)}, not {self.kind!r}.'
            raise ValidationError(error_msg)
        if self.kind == 'essential' and self.n >= -1:
            error_msg = f'The exponent of essential resources must be below -1, not {self.n}.'
            raise ValidationError(error_msg)

    def _essential_growth(self, R: Expression) -> Expression:
        """Return the growth `(sum_b (c_ib R_b)^n)^(1/n)` of a consumer on essential resources."""
        uptake = np.asarray(self.c, dtype=float)
        return total(R**self.n, weights=uptake**self.n) ** (1 / self.n)
