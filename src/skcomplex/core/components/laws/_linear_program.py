"""Define linear program laws.

They set algebraic states to an optimal solution of a linear program.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import operator
from collections.abc import Sequence
from typing import ClassVar

from sage.all__sagemath_symbolics import Expression

from ...errors import ValidationError
from ...specs import LinearObjectiveSpec, RequirementSpec
from ..spaces import ContinuousTime, DiscreteTime, StaticTime
from ._base import BaseLaw

INFEASIBLE_POLICIES = ('raise', 'zero', 'hold')
RELATIONS = (operator.eq, operator.le, operator.ge)


class LinearProgramLaw(BaseLaw):
    """Optimal solution of a linear program.

    It sets the algebraic states that are its variables. The laws whose variables overlap, from any element or
    interaction, form one program: their variables unite, their constraints add, and at most one of them holds
    objectives. The objectives are solved in order, and each optimum is kept within a relative 1e-9 while the next one
    is solved.

    Args:
        variables:
            The algebraic states the program sets.

        constraints:
            Sage relations, `==`, `<=`, or `>=`, linear in the variables. A coefficient of a variable reads no state,
            and the other terms may read states, parameters, and the time. A relation in one variable is a bound of
            that variable.

        objectives:
            The objectives, solved in order, or nothing for any feasible point.

        active:
            A condition, or `None` for always. Where it is false the solve is skipped and the variables are zero.

        infeasible:
            `'raise'` to raise `InfeasibleError`, `'zero'` to set the variables and objective values to zero, or
            `'hold'` to keep the last values.

        duals:
            Whether to record the dual of every constraint of the law.

        label:
            The name of the law, or the empty string for a law without one.

    Raises:
        ValidationError:
            A constraint is not a relation, an objective is not a `LinearObjectiveSpec`, or the policy is unknown,
            raised by `prepare`.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws import LinearProgramLaw
        >>> from skcomplex.core.specs import LinearObjectiveSpec
        >>> v, w = var('v w')
        >>> law = LinearProgramLaw([v, w], [v + w <= 1], [LinearObjectiveSpec(v)]).prepare('flows')
        >>> law.targets_
        (v, w)
    """

    law_tag: ClassVar[str] = 'equation'

    def __init__(
        self,
        variables: Sequence[Expression],
        constraints: Sequence[Expression],
        objectives: Sequence[LinearObjectiveSpec] = (),
        *,
        active: Expression | None = None,
        infeasible: str = 'raise',
        duals: bool = False,
        label: str = '',
    ) -> None:
        super().__init__(label)
        self.variables = variables
        self.constraints = constraints
        self.objectives = objectives
        self.active = active
        self.infeasible = infeasible
        self.duals = duals

    def _get_targets(self) -> tuple[object, ...]:
        """Return the algebraic states the program sets."""
        return tuple(self.variables)

    @classmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what the law requires of a system: continuous, discrete, or static time."""
        return RequirementSpec(time=(ContinuousTime, DiscreteTime, StaticTime))

    def _check(self) -> None:
        """Raise an error for no variables, a bad constraint or objective, or a bad policy.

        Raises:
            ValidationError:
                The program has no variables, a constraint is not a relation, an objective is not a
                `LinearObjectiveSpec`, or the policy is unknown.
        """
        if not len(self.variables):
            error_msg = 'A linear program law sets at least one variable, but it has none.'
            raise ValidationError(error_msg)
        for constraint in self.constraints:
            if not isinstance(constraint, Expression) or constraint.operator() not in RELATIONS:
                error_msg = (
                    f'A constraint of a linear program is a Sage relation `==`, `<=`, or `>=`, not {constraint}.'
                )
                raise ValidationError(error_msg)
        for objective in self.objectives:
            if not isinstance(objective, LinearObjectiveSpec):
                error_msg = f'An objective of a linear program is a `LinearObjectiveSpec`, not {objective!r}.'
                raise ValidationError(error_msg)
            objective._validated()
        if self.infeasible not in INFEASIBLE_POLICIES:
            error_msg = (
                f'Unknown infeasibility policy {self.infeasible!r}. Use one of {", ".join(INFEASIBLE_POLICIES)}.'
            )
            raise ValidationError(error_msg)
