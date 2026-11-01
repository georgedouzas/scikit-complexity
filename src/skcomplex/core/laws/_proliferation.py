"""Define proliferation laws, bilinear gains of a density at the rate of the product of two densities."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np
from sage.all__sagemath_symbolics import Expression

from ...specs import RequirementSpec
from ..spaces import ContinuousTime, LatticeSpace
from ._base import BaseLaw


class ProliferationLaw(BaseLaw):
    """Bilinear gain of the test density at the rate of the product of two densities, which leaves both unchanged.

    The amount of the candidate density `f` in class `h` that meets the amount of the field density `g` in class `k`
    adds to class `i` of the test density at the rate `mu[h, k]` with probability `B[h, k, i]`, so the law adds
    `sum_hk mu[h, k] B[h, k, i] f_h g_k` to the rate of the test density at `i`.

    Args:
        candidate:
            The symbol of the candidate density.

        field:
            The symbol of the field density.

        rate:
            The proliferation rate, a number or a parameter expression, or an array `mu[h, k]` of them.

        outcomes:
            The array `B[h, k, i]` of the probabilities of the class of the new amount, or `None` for the
            candidate's own class, which needs a test group of the candidate's size.

        test:
            The symbol of the density that gains the new amount, or `None` for the candidate's own density.

        label:
            The name of the law, or the empty string for a law without one.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.laws import ProliferationLaw
        >>> f, g = var('f'), var('g')
        >>> ProliferationLaw(f, g, 0.5).check().prepare().run({}).value_
        (0.5, None)
    """

    def __init__(
        self,
        candidate: Expression,
        field: Expression,
        rate: object,
        outcomes: object = None,
        test: Expression | None = None,
        *,
        label: str = '',
    ) -> None:
        super().__init__(label)
        self.candidate = candidate
        self.field = field
        self.rate = rate
        self.outcomes = outcomes
        self.test = test

    def _get_targets(self) -> tuple[object, ...]:
        """Return the densities the encounter changes, the candidate and the test."""
        return (self.candidate,) if self.test is None or self.test is self.candidate else (self.candidate, self.test)

    def _get_requirements(self) -> RequirementSpec:
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
