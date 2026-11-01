"""Define destruction laws, bilinear losses of a density at the rate of its product with a second density."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np
from sage.all__sagemath_symbolics import Expression

from ...specs import RequirementSpec
from ..spaces import ContinuousTime, LatticeSpace
from ._base import BaseLaw


class DestructionLaw(BaseLaw):
    """Bilinear loss of a density, removed in every class at the rate of its product with a second density.

    The amount of the candidate density `f` in class `h` that meets the amount of the field density `g` in class `k`
    is removed at the rate `mu[h, k]`, so the law adds `-f_h sum_k mu[h, k] g_k` to the rate of the candidate at `h`.

    Args:
        candidate:
            The symbol of the candidate density.

        field:
            The symbol of the field density.

        rate:
            The destruction rate, a number or a parameter expression, or an array `mu[h, k]` of them.

        label:
            The name of the law, or the empty string for a law without one.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.laws import DestructionLaw
        >>> f, g, mu = var('f'), var('g'), var('mu')
        >>> DestructionLaw(f, g, 2 * mu).check().prepare().run({'mu': 0.1}).value_
        0.2
    """

    def __init__(self, candidate: Expression, field: Expression, rate: object, *, label: str = '') -> None:
        super().__init__(label)
        self.candidate = candidate
        self.field = field
        self.rate = rate

    def _get_targets(self) -> tuple[object, ...]:
        """Return the candidate density, which loses the removed amount."""
        return (self.candidate,)

    def _get_requirements(self) -> RequirementSpec:
        """Return what the law requires of a system: continuous time on lattices of classes."""
        return RequirementSpec(time=(ContinuousTime,), space=(LatticeSpace,))

    def _check(self) -> None:
        """Raise an error for a rate table of the wrong dimensions."""
        self._check_table('rate', (0, 2))

    def _run(self, inputs: object) -> None:
        """Set `value_` to the rate at given values of the symbols it reads.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value.
        """
        self.value_ = self._evaluate(np.asarray(self.rate, dtype=object).tolist(), inputs)
