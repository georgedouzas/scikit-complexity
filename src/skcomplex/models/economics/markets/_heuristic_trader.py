"""Trader who holds the index of the forecasting heuristic in use, 0 for the fundamentalist and 1 for the trend."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN

from ....core.elements import Element
from ....engine.solvers import ModelView
from ....specs import StateSpec, VisualSpec
from ._heuristic_trader_view import HeuristicTraderView

PREDICTORS = ('fundamentalist', 'trend')


PRICING = 'pricing'


class HeuristicTraderElement(Element):
    """Trader who holds the index of the forecasting heuristic in use, 0 for the fundamentalist and 1 for the trend.

    The trader scores every heuristic by its last realised fitness, `fundamentalist_fitness` and `trend_fitness`, which
    a reinforcement of rate 1 sets every step and a logit choice reads.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        predictor:
            The index of the heuristic in use.

        fundamentalist_fitness:
            The last realised fitness of the fundamentalist heuristic.

        trend_fitness:
            The last realised fitness of the trend heuristic.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.economics.markets import HeuristicTraderElement
        >>> GroupElement('traders', element=HeuristicTraderElement, size=10, values={'predictor': 1.0}).size
        10
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        predictor: Annotated[float, StateSpec(domain=NN)] = 0.0,
        fundamentalist_fitness: Annotated[float, StateSpec()] = 0.0,
        trend_fitness: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.predictor = predictor
        self.fundamentalist_fitness = fundamentalist_fitness
        self.trend_fitness = trend_fitness

    @classmethod
    def _get_view(cls) -> type[HeuristicTraderView]:
        """Return the view of a speculator, with its heuristic."""
        return HeuristicTraderView

    @staticmethod
    def get_fitness(agent: HeuristicTraderView, model: ModelView, option: float) -> float:
        """Return the score of a heuristic of a trader, its last realised fitness.

        Args:
            agent:
                The trader.

            model:
                The model view of the choice.

            option:
                The index of the heuristic.

        Returns:
            fitness:
                The last realised fitness of the heuristic.
        """
        return float(getattr(agent, f'{PREDICTORS[int(option)]}_fitness'))

    @staticmethod
    def get_realised_fitness(agent: HeuristicTraderView, model: ModelView, index: int) -> float:
        """Return the fitness of a heuristic that the market maker reported last, zero before the first report.

        Args:
            agent:
                The trader.

            model:
                The model view of the reinforcement.

            index:
                The index of the heuristic.

        Returns:
            fitness:
                The last reported fitness of the heuristic.
        """
        reported = model.history(f'{PRICING}.fitness_{PREDICTORS[index]}', 1)
        return float(reported[-1]) if len(reported) else 0.0
