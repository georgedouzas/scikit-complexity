"""Fragment of integer size, which the breaking of the system cuts at a uniformly random point into two fragments."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.engine.solvers import SystemView
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ._fragment_view import FragmentView


class FragmentElement(Element):
    """Fragment of integer size, which the breaking of the system cuts at a uniformly random point into two fragments.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        length:
            The initial size.

        rate:
            The probability that a fragment larger than 1 breaks in a step.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.mathematics.stochastic_processes import FragmentElement
        >>> group = GroupElement('fragments', element=FragmentElement, size=1, values={'length': 100.0})
        >>> group.label, group.element.__name__, group.size
        ('fragments', 'FragmentElement', 1)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'length'}),
        *,
        length: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 1000.0,
        rate: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.length = length
        self.rate = rate

    @classmethod
    def _get_view(cls) -> type[FragmentView]:
        """Return the view of the agents, which types their states and parameters."""
        return FragmentView

    @staticmethod
    def breaks(agent: FragmentView, system: SystemView) -> float:
        """Return the probability that a fragment breaks in a step, zero at size 1.

        Args:
            agent:
                The fragment.

            system:
                The system.

        Returns:
            probability:
                The breaking probability, or zero for a fragment of size 1.
        """
        return agent.rate if round(agent.length) > 1 else 0.0

    @staticmethod
    def cut(agent: FragmentView, system: SystemView) -> float:
        """Return the share of a breaking fragment that stays with it, cut at a uniformly random whole point.

        Args:
            agent:
                The fragment.

            system:
                The system.

        Returns:
            share:
                The share of the size left of the cut.
        """
        length = round(agent.length)
        return int(system.rng.integers(1, length)) / length
