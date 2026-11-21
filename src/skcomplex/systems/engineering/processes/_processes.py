"""Process systems: stocks of material that flows between stages.

The material flows in, out, and between stages.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import step, where
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role
from ...general.dynamics import StockFlowSystem
from ...general.dynamics.types import FlowTable

INFLOW_START = 5.0


class StageElement(Element):
    """Stage of a process, a stock of material `m`.

    Args:
        label:
            The label of the stage.

        m:
            The initial material of the stage.

    Examples:
        >>> from skcomplex.systems.engineering.processes import StageElement
        >>> StageElement('stage', m=2.0).m
        2.0
    """

    def __init__(self, label: str, *, m: Annotated[object, RealSet.closed_open(0, oo)] = 0.0) -> None:
        super().__init__(label)
        self.m = m


class FeedInteraction(Interaction):
    """Constant feed of material into a stage, at the rate `feed`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the stage that receives the feed.

        feed:
            The feed rate.

    Examples:
        >>> from skcomplex.systems.engineering.processes import FeedInteraction
        >>> FeedInteraction('feed', target='stage_1', feed=2.0).feed
        2.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        feed: Annotated[float, RealSet.closed_open(0, oo)] = 0.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.feed = feed

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the constant feed of the stage.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        self.laws_: list[BaseLaw] = [RateLaw(context.target.m.diff(context.t), context.feed)]


class StageSplitInteraction(Interaction):
    """Material that leaves a stage at unit rate, a fraction `k` upwards and `1 - k` downwards.

    The flow `up` carries `k m` to the stage above and the flow `down` carries `(1 - k) m` to the stage below.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the stage the material leaves.

        above:
            The path of the stage above.

        below:
            The path of the stage below.

        k:
            The distribution coefficient, the fraction of material moving up.

    Examples:
        >>> from skcomplex.systems.engineering.processes import StageSplitInteraction
        >>> StageSplitInteraction('split', target='stage_2', above='stage_1', below='stage_3').k
        0.6
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        above: Role,
        below: Role,
        k: Annotated[float, RealSet.closed(0, 1)] = 0.6,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.above = above
        self.below = below
        self.k = k

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the flows up and down out of the stage.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        s = context
        m = s.target.m
        self.laws_: list[BaseLaw] = [
            RateLaw({m: -1, s.above.m: +1}, s.k * m, label='up'),
            RateLaw({m: -1, s.below.m: +1}, (1 - s.k) * m, label='down'),
        ]


class BathtubSystem(StockFlowSystem):
    """A bathtub, one stock of water with an inflow and an outflow.

    The inflow is from the faucet and the outflow is through the drain. The drain empties the tub at `outflow` gallons
    per minute while there is water, and the faucet opens to `inflow` gallons per minute after five minutes. With equal
    flows the level falls for five minutes and then stays constant. The system has the water `tub.water` in gallons and
    the flows `tub.faucet` and `tub.drain` in gallons per minute, over minutes.

    Args:
        water:
            The initial water in the tub, in gallons.

        inflow:
            The inflow after five minutes, in gallons per minute.

        outflow:
            The outflow, in gallons per minute.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.engineering.processes import BathtubSystem
        >>> BathtubSystem(water=20.0).prepare().system_.elements[0].values['water']
        20.0
    """

    field = 'engineering/processes'
    topics = ('stocks and flows', 'system dynamics', 'ode')
    references = ('Meadows 2008',)
    until = 10.0
    dt = 0.5

    def __init__(
        self,
        water: float = 50.0,
        *,
        inflow: float = 5.0,
        outflow: float = 5.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label=label, solver=solver, backend=backend)
        self.water = water
        self.inflow = inflow
        self.outflow = outflow

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the stocks."""
        return 'tub'

    def _get_stocks(self) -> dict[str, object]:
        """Return the initial value of every stock by name."""
        return {
            'water': self.water,
        }

    def _get_allowed(self) -> dict[str, object]:
        """Return the allowed values of the stocks and parameters that have them, by name."""
        return {
            'water': RealSet.closed_open(0, oo),
            'inflow': RealSet.closed_open(0, oo),
            'outflow': RealSet.closed_open(0, oo),
        }

    def _get_quantities(self) -> dict[str, str]:
        """Return the quantity every stock holds, by name."""
        return {
            'water': 'water',
        }

    def _get_parameters(self) -> dict[str, object]:
        """Return the value of every parameter by name."""
        return {
            'inflow': self.inflow,
            'outflow': self.outflow,
        }

    def _get_flows(self) -> FlowTable:
        """Return the faucet and drain flows of the tub."""
        return {
            'faucet': (None, 'water', lambda s: s.target.inflow * step(s.t, INFLOW_START)),
            'drain': ('water', None, lambda s: where(s.target.water > 0, s.target.outflow, 0)),
        }


class CascadeClassifierSystem(BaseSystem):
    """A cascade classifier, a column of stages that separates particles.

    An air flow separates them. The particles separated are of a narrow size class. Material leaves every stage at unit
    rate, a fraction `k` upwards and `1 - k` downwards. The top stage releases into the fine product and the bottom
    stage into the coarse product. Material of one size class is fed into one stage at a constant rate. At steady state
    the share of the feed that ends in the fine product is
    `F_f = (1 - ((1 - k) / k)^(z + 1 - i)) / (1 - ((1 - k) / k)^(z + 1))` for `z` stages and feeding at stage `i`
    counted from the top, Barsky's fractional extraction.

    The system has the material `stage_<i>.m` of each stage from the top, and the accumulated products `fine.m` and
    `coarse.m`.

    Args:
        n_stages:
            The number of stages `z`.

        k:
            The distribution coefficient, the fraction of material moving up from a stage.

        feed_stage:
            The stage that receives the feed, counted from 1 at the top, or `None` for the middle stage.

        feed:
            The feed rate.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.engineering.processes import CascadeClassifierSystem
        >>> len(CascadeClassifierSystem(n_stages=4).prepare().system_.interactions)
        5
    """

    field = 'engineering/processes'
    topics = ('separation', 'cascade', 'stocks and flows', 'ode')
    references = ('Barsky 2020',)
    until = 50.0
    dt = 0.5

    def __init__(
        self,
        n_stages: int = 10,
        *,
        k: float = 0.6,
        feed_stage: int | None = None,
        feed: float = 1.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_stages = n_stages
        self.k = k
        self.feed_stage = feed_stage
        self.feed = feed

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components of the system built from its model parameters."""
        super()._prepare(context)
        entry = (self.n_stages + 1) // 2 if self.feed_stage is None else self.feed_stage
        stages = [StageElement(f'stage_{index}') for index in range(1, self.n_stages + 1)]
        labels = ['fine', *(stage.label for stage in stages), 'coarse']
        interactions = [
            FeedInteraction('feed', target=f'stage_{entry}', feed=self.feed),
            *(
                StageSplitInteraction(
                    f'split_{index}',
                    target=labels[index],
                    above=labels[index - 1],
                    below=labels[index + 1],
                    k=self.k,
                )
                for index in range(1, self.n_stages + 1)
            ),
        ]
        self.system_ = System(
            self.label,
            elements=[StageElement('fine'), *stages, StageElement('coarse')],
            interactions=interactions,
        )
