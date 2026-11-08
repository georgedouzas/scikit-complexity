"""Define encounter laws.

They are bilinear transfers between the classes of a density that conserve its total.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

import numpy as np
from sage.all__sagemath_symbolics import Expression

from ...specs import RequirementSpec
from ..spaces import ContinuousTime, LatticeSpace
from ._base import BaseLaw


class EncounterLaw(BaseLaw):
    """Bilinear transfer of a density between its classes.

    The rate is that of its product with a second density. The amount of the candidate density `f` in class `h` meets
    the amount of the field density `g` in class `k` at the rate `eta[h, k]`, and moves to class `i` with the
    probability `A[h, k, i]`. The law adds the gain and the loss

    `df_i/dt = sum_hk eta[h, k] A[h, k, i] f_h g_k - f_i sum_k eta[i, k] g_k`,

    where the loss acts on the candidate and the gain on the test density, the candidate itself by default, so the
    total over the classes is conserved. The candidate, the field, and the test are states of groups on the sites of
    lattices of classes, one member per class. A field density that no law changes is a fixed external input.

    Args:
        candidate:
            The symbol of the candidate density, a state of a group read member by member.

        field:
            The symbol of the field density, or `None` for a linear redistribution, with one field class of value one.

        rate:
            The encounter rate, a number or a parameter expression, or an array `eta[h, k]` of them.

        transition:
            The array `A[h, k, i]` of transition probabilities, numbers or parameter expressions, which sum to one
            over `i`.

        test:
            The symbol of the density the candidate moves into, or `None` for the candidate's own density.

        label:
            The name of the law, or the empty string for a law without one.

    Examples:
        >>> import numpy as np
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws import EncounterLaw
        >>> f, g = var('f'), var('g')
        >>> rate, transition = EncounterLaw(f, g, 1.0, np.full((2, 2, 2), 0.5)).prepare().run({}).value_
        >>> rate, np.shape(transition)
        (1.0, (2, 2, 2))
    """

    law_tag: ClassVar[str] = 'rate'

    def __init__(
        self,
        candidate: Expression,
        field: Expression | None,
        rate: object,
        transition: object,
        test: Expression | None = None,
        *,
        label: str = '',
    ) -> None:
        super().__init__(label)
        self.candidate = candidate
        self.field = field
        self.rate = rate
        self.transition = transition
        self.test = test

    def _get_targets(self) -> tuple[object, ...]:
        """Return the densities the encounter changes, the candidate and the test."""
        return (self.candidate,) if self.test is None or self.test is self.candidate else (self.candidate, self.test)

    @classmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what the law requires: continuous time on lattices of classes."""
        return RequirementSpec(time=(ContinuousTime,), space=(LatticeSpace,))

    def _check(self) -> None:
        """Raise an error for a rate or transition table of the wrong dimensions.

        It also raises it when their class counts disagree.
        """
        self._check_table('rate', (0, 2))
        self._check_table('transition', (3,))
        self._check_outcomes('rate', 'transition')

    def _run(self, inputs: object) -> None:
        """Set `value_` to the rate and the transition table at given symbol values.

        The symbols are those they read.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value.
        """
        tables = (np.asarray(self.rate, dtype=object).tolist(), np.asarray(self.transition, dtype=object).tolist())
        self.value_ = self._evaluate(tables, inputs)
