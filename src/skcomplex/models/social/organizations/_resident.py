"""Resident of the Kauai population model, who lives, works, and spends time at points of a map."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from typing import Annotated

import numpy.typing as npt
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ._resident_view import ResidentView


class ResidentElement(Element):
    """Resident of the Kauai population model, who lives, works, and spends time at points of a map.

    A resident starts at its home, a point of its census region. It holds a sentiment between -1 and 1, one weight per
    topic of the events that move the sentiment, and the demographic scalars of Carmichael et al. (2019 ch. 3 §4.2).

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        sentiment:
            The initial sentiment.

        weights:
            The weight of every topic.

        home:
            The point of the home, which the first commute sets to the initial location.

        census:
            The index of the census region of the home.

        work:
            The index of the facility of the work.

        favourite:
            The index of the favourite place.

        visits:
            The number of steps the resident stood in its favourite place.

        race:
            The index of the race.

        gender:
            The index of the gender.

        occupation:
            The index of the occupation.

        age:
            The age in years.

        convergence:
            The immutable convergence parameter `CR` of the interactions between agents.

        event_rate:
            The convergence rate `mu` of the response to an event.

        cell_size:
            The side of the grid cells of the space within which residents meet.

        partners:
            The largest number of residents drawn from the grid cell as possible partners.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.social.organizations import ResidentElement
        >>> GroupElement('residents', element=ResidentElement, size=10).size
        10
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'sentiment', 'color_range': (-1.0, 1.0)}),
        *,
        sentiment: Annotated[float, StateSpec(domain=RealSet.closed(-1, 1))] = 0.0,
        weights: Annotated[npt.ArrayLike, StateSpec(vector=True)] = 0.0,
        home: Annotated[npt.ArrayLike, StateSpec(vector=True)] = (math.nan, math.nan),
        census: Annotated[float, StateSpec()] = 0.0,
        work: Annotated[float, StateSpec()] = 0.0,
        favourite: Annotated[float, StateSpec()] = 0.0,
        visits: Annotated[float, StateSpec()] = 0.0,
        race: Annotated[float, StateSpec()] = 0.0,
        gender: Annotated[float, StateSpec()] = 0.0,
        occupation: Annotated[float, StateSpec()] = 0.0,
        age: Annotated[float, StateSpec()] = 0.0,
        convergence: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 1.0,
        event_rate: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.05,
        cell_size: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1000.0,
        partners: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 15.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.sentiment = sentiment
        self.weights = weights
        self.home = home
        self.census = census
        self.work = work
        self.favourite = favourite
        self.visits = visits
        self.race = race
        self.gender = gender
        self.occupation = occupation
        self.age = age
        self.convergence = convergence
        self.event_rate = event_rate
        self.cell_size = cell_size
        self.partners = partners

    @classmethod
    def _get_view(cls) -> type[ResidentView]:
        """Return the view of a resident, `ResidentView`."""
        return ResidentView
