"""Synchronous update of a node by the sign of the weighted sum of its active inputs, after Li et al."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import SR, Expression

from ....core.interactions import MapInteraction
from ....core.laws.utils import where
from ....specs import ParameterSpec, RoleSpec


class ThresholdFunctionInteraction(MapInteraction):
    """Synchronous update of a node by the sign of the weighted sum of its active inputs, after Li et al.

    The node turns on when the sum `w_0 x_0 + w_1 x_1 + ...` over its inputs is positive and off when it is negative.
    When the sum is zero, a node that decays turns off and any other node keeps its value.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the updated node.

        inputs:
            The paths of the input nodes.

        weights:
            The weight of every input, positive for activation and negative for inhibition.

        decays:
            Whether the node turns off without net input.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import ThresholdFunctionInteraction
        >>> ThresholdFunctionInteraction('sic1', target='Sic1', inputs=['Swi5', 'Cln1_2'], weights=[1.0, -1.0]).weights
        [1.0, -1.0]
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        inputs: Annotated[list[str], RoleSpec(variadic=True)],
        weights: Annotated[Sequence[float], ParameterSpec(per='inputs')] = (),
        decays: bool = False,
    ) -> None:
        super().__init__(label, target=target, state='x', function=self._get_next)
        self.inputs = inputs
        self.weights = weights
        self.decays = decays

    def _get_next(self, s: SimpleNamespace) -> Expression:
        """Return 1 for a positive weighted input, 0 for a negative one, and the held or decayed value for none."""
        total = sum((weight * item.x for weight, item in zip(s.weights, s.inputs, strict=True)), start=SR(0))
        unchanged = 0 if self.decays else s.target.x
        return where(total > 0, 1, where(total < 0, 0, unchanged))
