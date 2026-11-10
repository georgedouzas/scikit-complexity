"""Define bilinear laws.

They are gains and losses of a density at the rate of the product of two densities.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

import numpy as np
from sage.all__sagemath_symbolics import Expression

from ...errors import ValidationError
from ...specs import RequirementSpec
from ..spaces import ContinuousTime, LatticeSpace
from ._base import BaseLaw


class BilinearLaw(BaseLaw):
    """Gain or loss of a target density at a signed rate of a product of densities.

    The product is of two densities. The amount of the first density `f` in class `h` that meets the amount of the
    second density `g` in class `k` adds to class `i` of the target at the rate `r[h, k]` with probability `B[h, k, i]`,
    so the law adds `sum_hk r[h, k] B[h, k, i] f_h g_k` to the rate of the target at `i`. A positive rate gives a gain
    that leaves the first density in place, and a negative rate with the first density as the target gives a loss.

    A conserved law also takes the matching loss from the first density, so it is the bilinear transfer of the first
    density between its classes. It adds the gain and the loss

    `df_i/dt = sum_hk r[h, k] B[h, k, i] f_h g_k - f_i sum_k r[i, k] g_k`,

    where the loss acts on the first density and the gain on the target, the first density itself by default, so the
    total over the classes is conserved. The densities are states of groups on the sites of lattices of classes, one
    member per class. A second density that no law changes is a fixed external input, and a conserved law without one
    is a linear redistribution, with one class of value one.

    Args:
        target:
            The symbol of the density the law changes.

        first:
            The symbol of the first density.

        second:
            The symbol of the second density, or `None` for a linear redistribution of a conserved law.

        rate:
            The signed rate, a number or a parameter expression, or an array `r[h, k]` of them.

        outcomes:
            The array `B[h, k, i]` of the probabilities of the class in the target, numbers or parameter expressions
            that sum to one over `i`, or `None` for the class of the first density, which needs a target of its size. A
            conserved law needs the array, since the identity makes its gain and loss cancel.

        conserved:
            Whether the law also takes the matching loss from the first density, so the total is conserved.

        label:
            The name of the law, or the empty string for a law without one.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws import BilinearLaw
        >>> f, g = var('f'), var('g')
        >>> BilinearLaw(f, f, g, -0.5).prepare().run({}).value_
        (-0.5, None)
        >>> import numpy as np
        >>> rate, outcomes = BilinearLaw(f, f, g, 1.0, np.full((2, 2, 2), 0.5), conserved=True).prepare().run({}).value_
        >>> rate, np.shape(outcomes)
        (1.0, (2, 2, 2))
    """

    law_tag: ClassVar[str] = 'rate'

    def __init__(
        self,
        target: Expression,
        first: Expression,
        second: Expression | None,
        rate: object,
        outcomes: object = None,
        *,
        conserved: bool = False,
        label: str = '',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.first = first
        self.second = second
        self.rate = rate
        self.outcomes = outcomes
        self.conserved = conserved

    def _get_targets(self) -> tuple[object, ...]:
        """Return the densities the encounter changes, the first density and the target."""
        return (self.first,) if self.target is self.first else (self.first, self.target)

    @classmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what the law requires: continuous time on lattices of classes."""
        return RequirementSpec(time=(ContinuousTime,), space=(LatticeSpace,))

    def _prepare(self, context: str | None = None) -> None:
        """Raise an error for a rate or outcome table of the wrong dimensions.

        It also raises it for a conserved law without outcomes.

        It also raises it when their class counts disagree, or for a law without a second density that is not conserved.

        Args:
            context:
                The path of the element or interaction that returns the law, or `None`.
        """
        super()._prepare(context)
        if self.conserved and self.outcomes is None:
            error_msg = (
                f'The conserved bilinear law of {self.target} has no outcome table, but without one its gain and loss '
                'cancel. Give the outcome probabilities `B[h, k, i]`.'
            )
            raise ValidationError(error_msg)
        if self.second is None and not self.conserved:
            error_msg = (
                f'The bilinear law of {self.target} has no second density, which only a conserved law takes. Give the '
                'second density, or set `conserved=True`.'
            )
            raise ValidationError(error_msg)
        self._check_table('rate', (0, 2))
        if self.outcomes is not None:
            self._check_table('outcomes', (3,))
        self._check_outcomes('rate', 'outcomes')

    def _run(self, inputs: object) -> None:
        """Set `value_` to the rate and the outcome table at given symbol values.

        The values are those of the symbols they read.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value.
        """
        outcomes = None if self.outcomes is None else np.asarray(self.outcomes, dtype=object).tolist()
        self.value_ = self._evaluate((np.asarray(self.rate, dtype=object).tolist(), outcomes), inputs)
