"""Segment of a road network between two points, with the pheromone ants left on it and whether it is open."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec
from ._trail_segment_view import TrailSegmentView


class TrailSegmentElement(Element):
    """Segment of a road network between two points, with the pheromone ants left on it and whether it is open.

    Args:
        label:
            The label of the segment or of the group of segments.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        tau:
            The pheromone level.

        passable:
            1 while the segment can be walked and 0 once it is cut.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import TrailSegmentElement
        >>> TrailSegmentElement('segment').tau
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        tau: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        passable: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.tau = tau
        self.passable = passable

    @classmethod
    def _get_view(cls) -> type[TrailSegmentView]:
        """Return the view of a segment, `TrailSegmentView`."""
        return TrailSegmentView
