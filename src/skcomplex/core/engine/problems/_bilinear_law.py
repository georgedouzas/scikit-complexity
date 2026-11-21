"""Define the law the compiler lowers a rate of a bilinear form to."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

import numpy as np
from sage.all__sagemath_symbolics import Expression

from ...components.laws import BaseLaw, RateLaw, RequirementSpec
from ...components.laws.utils._bilinear import bilinears_in, unpack_bilinear
from ...components.spaces import LatticeSpace
from ...components.spaces._continuous_time import _ContinuousTime
from ...errors import ValidationError


class BilinearLaw(BaseLaw):
    """A law where members of two densities meet and change class.

    The compiler runs it for a `RateLaw` whose whole rate is a `bilinear` form, so the encounters lower to sparse
    contractions instead of an expression per class. It reads the densities, the tables, and the conservation of the
    form, and it changes the target of the rate law and, for a conserved form, the first density.

    Args:
        target:
            The symbol of the density the law changes.

        first:
            The symbol of the first density.

        second:
            The symbol of the second density, or `None` for a linear redistribution of a conserved form.

        rate:
            The signed rate, a number or a parameter expression, or an array `r[h, k]` of them.

        outcomes:
            The array `B[h, k, i]` of the probabilities of the class in the target, or `None` for the class of the
            first density.

        conserved:
            Whether the law also takes the matching loss from the first density.

        label:
            The name of the law, or the empty string for a law without one.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.engine.problems._bilinear_law import BilinearLaw
        >>> f, g = var('f g')
        >>> BilinearLaw(f, f, g, -0.5).prepare().rate
        -0.5
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

    @classmethod
    def _get_name(cls) -> str:
        """Return the name of the law in messages, the class users write it with."""
        return 'RateLaw'

    def _get_targets(self) -> tuple[object, ...]:
        """Return the densities the encounter changes, the first density and the target."""
        return (self.first,) if self.target is self.first else (self.first, self.target)

    @classmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what the law requires: continuous time on lattices of classes."""
        return RequirementSpec(time=(_ContinuousTime,), space=(LatticeSpace,))

    def _get_value(self, inputs: object) -> object:
        """Return the rate and the outcome table at given symbol values.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value.
        """
        outcomes = None if self.outcomes is None else np.asarray(self.outcomes, dtype=object).tolist()
        return self._evaluate((np.asarray(self.rate, dtype=object).tolist(), outcomes), inputs)

    @staticmethod
    def _get_lowered(law: BaseLaw) -> BaseLaw:
        """Return the bilinear law of a rate of a bilinear form, or the law itself.

        Args:
            law:
                A law a component returns.

        Returns:
            law:
                The bilinear law of the form, with the target and the label of the rate law, or the law as it is.

        Raises:
            ValidationError:
                A rate holds a bilinear form next to other terms, or a conserved form changes a target other than its
                first density.
        """
        if not isinstance(law, RateLaw):
            return law
        forms = bilinears_in(law.value)
        if not forms:
            return law
        if not isinstance(law.value, Expression) or len(forms) > 1 or str(forms[0]) != str(law.value):
            error_msg = (
                f'The rate of {law.target} holds a bilinear form next to other terms. Give the form as the whole rate '
                'of its own rate law.'
            )
            raise ValidationError(error_msg)
        first, second, rate, outcomes, conserved = unpack_bilinear(forms[0])
        if conserved and str(first) != str(law.target):
            error_msg = (
                f'The conserved bilinear form of {first} changes {law.target}, but its loss leaves the first density. '
                'Target the first density.'
            )
            raise ValidationError(error_msg)
        target = first if str(first) == str(law.target) else law.target
        return BilinearLaw(target, first, second, rate, outcomes, conserved=conserved, label=law.label)
