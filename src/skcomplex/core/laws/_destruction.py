"""Define destruction laws, encounters of active particles that remove the candidate particle."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Self

import numpy as np
from sage.all__sagemath_symbolics import Expression

from ...specs import RequirementSpec
from ..spaces import ContinuousTime, LatticeSpace
from ._base import BaseLaw


class DestructionLaw(BaseLaw):
    """Destructive encounter, which removes the candidate particle.

    A candidate particle at activity `h` that meets a field particle at activity `k` is removed at the rate
    `mu[h, k]`, so the law adds `-f_h sum_k mu[h, k] g_k` to the rate of the candidate density at `h`.

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
        >>> DestructionLaw(f, g, 2 * mu).check().run({'mu': 0.1})
        0.2
    """

    def __init__(self, candidate: Expression, field: Expression, rate: object, *, label: str = '') -> None:
        super().__init__(label)
        self.candidate = candidate
        self.field = field
        self.rate = rate

    def _get_targets(self) -> tuple[object, ...]:
        """Return the candidate density, which loses the removed particles."""
        return (self.candidate,)

    def _get_requirements(self) -> RequirementSpec:
        """Return what the law requires of a system: continuous time on activity lattices."""
        return RequirementSpec(time=(ContinuousTime,), space=(LatticeSpace,))

    def _check(self) -> Self:
        """Raise an error for a rate table of the wrong dimensions.

        Returns:
            law:
                The law itself.
        """
        self._check_table('rate', (0, 2))
        return self

    def _run(self, inputs: object) -> object:
        """Return the rate at given values of the symbols it reads.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value.

        Returns:
            rate:
                The rate, a number or nested lists of numbers when nothing symbolic is left.
        """
        return self._evaluate(np.asarray(self.rate, dtype=object).tolist(), inputs)
