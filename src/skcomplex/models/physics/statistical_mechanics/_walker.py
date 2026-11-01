"""Random walker of diffusion-limited aggregation, which sticks when it reaches a stuck walker."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RuleLaw
from ....engine.solvers import ModelView
from ....specs import StateSpec, VisualSpec
from ._walker_view import WalkerView


class WalkerElement(Element):
    """Random walker of diffusion-limited aggregation, which sticks when it reaches a stuck walker.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        stuck:
            Whether the walker belongs to the aggregate.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import WalkerElement
        >>> WalkerElement('walker').stuck
        0.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'stuck', 'color_range': (0.0, 1.0)}),
        *,
        stuck: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.stuck = stuck

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the walk-and-stick rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.walk)]

    @classmethod
    def _get_view(cls) -> type:
        """Return the class of the views of the element's agents."""
        return WalkerView

    def walk(self, agent: WalkerView, model: ModelView) -> None:
        """Stick next to the aggregate, or step to a random neighbouring cell.

        Args:
            agent:
                The walker.

            model:
                The model.
        """
        if agent.stuck:
            return
        if any(isinstance(other, WalkerView) and other.stuck for other in agent.neighbors()):
            agent.stuck = 1.0
            return
        cells = agent.neighbor_cells()
        agent.move_to(cells[int(model.rng.integers(len(cells)))])
