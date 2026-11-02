"""Employee of the insider sabotage model, whose risk is the sum of five factors."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ._employee_view import EmployeeView


class EmployeeElement(Element):
    """Employee of the insider sabotage model, whose risk is the sum of five factors.

    The risk is `R = P + D + S + K + A`: the predisposition `P`, the disgruntlement `D` of predisposed employees
    whose daily feeling about the organization falls short of their expectation, the stress `S` raised by seeing a
    supervisor, the skill `K` that grows every day, and the fixed affect `A`. The expectation weighs the first, the
    mean, and the latest feeling. Every day an employee wanders to an adjacent cell and leaves with probability
    `turnover / 365`, replaced by a new hire. In `insider_sabotage`, a movement, a map of every state, and a life cycle
    that renews the employees who leave run these steps in turn in the activation of every employee.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        predisposed:
            Whether the employee is predisposed to sabotage, 1, or not, 0.

        disgruntlement:
            The initial disgruntlement `D`.

        stress:
            The initial stress `S`.

        skill:
            The initial skill `K`.

        affect:
            The affect `A`.

        risk:
            The initial risk `R`.

        days:
            The number of days the employee has felt about the organization.

        first:
            The first feeling.

        mean_feeling:
            The mean feeling.

        expectation:
            The expected feeling.

        feeling:
            The latest feeling.

        turnover:
            The yearly share of employees who leave.

        culture:
            The mean daily feeling about the organization.

        change:
            The variance of the daily feeling.

        primacy:
            The weight of the first feeling in the expectation.

        consistency:
            The weight of the mean feeling in the expectation.

        recency:
            The weight of the latest feeling in the expectation.

        affect_weight:
            The factor of the shortfall of a feeling in the disgruntlement.

        alpha_seen:
            The learning rate of stress when a supervisor is in sight.

        lambda_seen:
            The level stress approaches when a supervisor is in sight.

        alpha_unseen:
            The learning rate of stress otherwise.

        lambda_unseen:
            The level stress approaches otherwise.

        skill_growth:
            The mean daily growth of skill.

        skill_variance:
            The variance of the daily growth of skill.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.social.organizations import EmployeeElement
        >>> group = GroupElement('employees', element=EmployeeElement, size=10, values={'affect': 0.5})
        >>> group.label, group.element.__name__, group.size
        ('employees', 'EmployeeElement', 10)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'risk', 'color_range': (0.0, 5.0)}),
        *,
        predisposed: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
        disgruntlement: Annotated[float, StateSpec()] = 0.0,
        stress: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        skill: Annotated[float, StateSpec()] = 0.0,
        affect: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        risk: Annotated[float, StateSpec()] = 0.5,
        days: Annotated[float, StateSpec(domain=NN)] = 0.0,
        first: Annotated[float, StateSpec()] = 0.0,
        mean_feeling: Annotated[float, StateSpec()] = 0.0,
        expectation: Annotated[float, StateSpec()] = 0.0,
        feeling: Annotated[float, StateSpec()] = 0.0,
        turnover: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.15,
        culture: Annotated[float, ParameterSpec()] = 0.8,
        change: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        primacy: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        consistency: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        recency: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        affect_weight: Annotated[float, ParameterSpec()] = 1.0,
        alpha_seen: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        lambda_seen: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        alpha_unseen: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.25,
        lambda_unseen: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        skill_growth: Annotated[float, ParameterSpec()] = 0.001,
        skill_variance: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0005,
    ) -> None:
        super().__init__(label, visual=visual)
        self.predisposed = predisposed
        self.disgruntlement = disgruntlement
        self.stress = stress
        self.skill = skill
        self.affect = affect
        self.risk = risk
        self.days = days
        self.first = first
        self.mean_feeling = mean_feeling
        self.expectation = expectation
        self.feeling = feeling
        self.turnover = turnover
        self.culture = culture
        self.change = change
        self.primacy = primacy
        self.consistency = consistency
        self.recency = recency
        self.affect_weight = affect_weight
        self.alpha_seen = alpha_seen
        self.lambda_seen = lambda_seen
        self.alpha_unseen = alpha_unseen
        self.lambda_unseen = lambda_unseen
        self.skill_growth = skill_growth
        self.skill_variance = skill_variance

    @classmethod
    def _get_view(cls) -> type[EmployeeView]:
        """Return the class of the views of the agents, whose attributes are the states and parameters."""
        return EmployeeView
