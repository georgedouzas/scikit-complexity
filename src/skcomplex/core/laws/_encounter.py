"""Define encounter laws, conservative encounters of active particles that move them to new activities."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Self

import numpy as np
from sage.all__sagemath_symbolics import Expression

from ...specs import RequirementSpec
from ..spaces import ContinuousTime, LatticeSpace
from ._base import BaseLaw


class EncounterLaw(BaseLaw):
    """Conservative encounter of active particles, which moves a candidate particle to a new activity.

    A candidate particle at activity `h` meets a field particle at activity `k` at the encounter rate `eta[h, k]`, and
    ends at activity `i` with the transition probability `A[h, k, i]`. With candidate density `f` and field density
    `g` on activity grids, the law adds Bellomo's gain and loss terms to the rate of the test density,

    `df_i/dt = sum_hk eta[h, k] A[h, k, i] f_h g_k - f_i sum_k eta[i, k] g_k`,

    where the loss acts on the candidate and the gain on the test group, the candidate's own group by default. The
    candidate, the field, and the test are states of groups on the sites of activity lattices, one member per
    activity. A field role bound to a group whose states no law changes is an external action at the micro scale.

    Args:
        candidate:
            The symbol of the candidate density, a state of a group read member by member.

        field:
            The symbol of the field density.

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
        >>> from skcomplex.core.laws import EncounterLaw
        >>> f, g = var('f'), var('g')
        >>> rate, transition = EncounterLaw(f, g, 1.0, np.full((2, 2, 2), 0.5)).check().run({})
        >>> rate, np.shape(transition)
        (1.0, (2, 2, 2))
    """

    def __init__(
        self,
        candidate: Expression,
        field: Expression,
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

    def _get_requirements(self) -> RequirementSpec:
        """Return what the law requires of a system: continuous time on activity lattices."""
        return RequirementSpec(time=(ContinuousTime,), space=(LatticeSpace,))

    def _check(self) -> Self:
        """Raise an error for a rate or transition table of the wrong dimensions, or whose activities disagree.

        Returns:
            law:
                The law itself.
        """
        self._check_table('rate', (0, 2))
        self._check_table('transition', (3,))
        self._check_outcomes('rate', 'transition')
        return self

    def _run(self, inputs: object) -> object:
        """Return the rate and the transition table at given values of the symbols they read.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value.

        Returns:
            tables:
                The rate and the transition table, as nested lists of numbers when nothing symbolic is left.
        """
        tables = (np.asarray(self.rate, dtype=object).tolist(), np.asarray(self.transition, dtype=object).tolist())
        return self._evaluate(tables, inputs)
