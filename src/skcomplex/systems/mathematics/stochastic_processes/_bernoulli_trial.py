"""Sequence of independent trials, each a success with the same probability, and the count of its successes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ._bernoulli_trial_view import BernoulliTrialView


class BernoulliTrialElement(Element):
    """Sequence of independent trials, each a success with the same probability, and the count of its successes.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        outcome:
            The initial outcome, 1 for a success and 0 for a failure.

        successes:
            The initial number of successes.

        p:
            The probability of a success.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.mathematics.stochastic_processes import BernoulliTrialElement
        >>> group = GroupElement('trials', element=BernoulliTrialElement, size=4, values={'p': 0.2})
        >>> group.label, group.element.__name__, group.size
        ('trials', 'BernoulliTrialElement', 4)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('histograms', {'color_by': 'successes'}),
        *,
        outcome: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
        successes: Annotated[float, StateSpec(domain=NN)] = 0.0,
        p: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.outcome = outcome
        self.successes = successes
        self.p = p

    @classmethod
    def _get_view(cls) -> type[BernoulliTrialView]:
        """Return the view of the agents, which types their states and parameters."""
        return BernoulliTrialView
