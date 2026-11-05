"""Define bilinear laws, gains and losses of a density at the rate of the product of two densities."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np
from sage.all__sagemath_symbolics import Expression

from ...specs import RequirementSpec
from ..spaces import ContinuousTime, LatticeSpace
from ._base import BaseLaw


class BilinearLaw(BaseLaw):
    """Gain or loss of a target density at a signed rate of the product of two densities.

    The amount of the first density `f` in class `h` that meets the amount of the second density `g` in class `k` adds
    to class `i` of the target at the rate `r[h, k]` with probability `B[h, k, i]`, so the law adds
    `sum_hk r[h, k] B[h, k, i] f_h g_k` to the rate of the target at `i`. A positive rate gives a gain that leaves the
    first density in place, and a negative rate with the first density as the target gives a loss.

    Args:
        target:
            The symbol of the density the law changes.

        first:
            The symbol of the first density.

        second:
            The symbol of the second density.

        rate:
            The signed rate, a number or a parameter expression, or an array `r[h, k]` of them.

        outcomes:
            The array `B[h, k, i]` of the probabilities of the class in the target, or `None` for the class of the first
            density, which needs a target of its size.

        label:
            The name of the law, or the empty string for a law without one.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws import BilinearLaw
        >>> f, g = var('f'), var('g')
        >>> BilinearLaw(f, f, g, -0.5).prepare().run({}).value_
        (-0.5, None)
    """

    def __init__(
        self,
        target: Expression,
        first: Expression,
        second: Expression,
        rate: object,
        outcomes: object = None,
        *,
        label: str = '',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.first = first
        self.second = second
        self.rate = rate
        self.outcomes = outcomes

    def _get_targets(self) -> tuple[object, ...]:
        """Return the densities the encounter changes, the first density and the target."""
        return (self.first,) if self.target is self.first else (self.first, self.target)

    @classmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what the law requires of a system: continuous time on lattices of classes."""
        return RequirementSpec(time=(ContinuousTime,), space=(LatticeSpace,))

    def _check(self) -> None:
        """Raise an error for a rate or outcome table of the wrong dimensions, or whose class counts disagree."""
        self._check_table('rate', (0, 2))
        if self.outcomes is not None:
            self._check_table('outcomes', (3,))
        self._check_outcomes('rate', 'outcomes')

    def _run(self, inputs: object) -> None:
        """Set `value_` to the rate and the outcome table at given values of the symbols they read.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value.
        """
        outcomes = None if self.outcomes is None else np.asarray(self.outcomes, dtype=object).tolist()
        self.value_ = self._evaluate((np.asarray(self.rate, dtype=object).tolist(), outcomes), inputs)
