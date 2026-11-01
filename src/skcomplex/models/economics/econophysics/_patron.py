"""Patron of the El Farol bar, who forecasts the attendance with its most accurate predictor."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated, ClassVar

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RuleLaw
from ....engine.solvers import ModelView
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ._patron_view import PatronView
from ._predictor_pool import PREDICTORS
from ._predictor_pool_view import PredictorPoolView


class PatronElement(Element):
    """Patron of the El Farol bar, who forecasts the attendance with its most accurate predictor.

    Every patron holds a few predictors of the pool, the set bits of `predictors`, and forecasts with the one of
    smallest error in the pool. A choice then sends the patron to the bar when the forecast is below the capacity.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        predictors:
            The bit mask of the predictors of the pool that the patron holds.

        forecast:
            The last forecast of the patron's most accurate predictor.

        attends:
            Whether the patron went to the bar this week, 1 or 0.

        capacity:
            The attendance above which the bar is too crowded to enjoy.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.economics.econophysics import PatronElement
        >>> GroupElement('patrons', element=PatronElement, size=100, values={'predictors': 7}).size
        100
    """

    pool: ClassVar[str] = 'pool'

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        predictors: Annotated[float, StateSpec(domain=NN)] = 1,
        forecast: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        attends: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
        capacity: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 60.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.predictors = predictors
        self.forecast = forecast
        self.attends = attends
        self.capacity = capacity

    @classmethod
    def _get_view(cls) -> type[PatronView]:
        """Return the view of the agents of the element."""
        return PatronView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the predicting rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.predict, phase='predict')]

    def predict(self, agent: PatronView, model: ModelView) -> None:
        """Forecast the attendance with the most accurate predictor held.

        Args:
            agent:
                The patron.

            model:
                The model.
        """
        pool = model.agents(self.pool, PredictorPoolView)[0]
        errors, predicted = np.ravel(pool.errors), np.ravel(pool.forecasts)
        mask = int(agent.predictors)
        held = [index for index in range(len(PREDICTORS)) if mask >> index & 1]
        agent.forecast = float(predicted[held[int(np.argmin(errors[held]))]])
