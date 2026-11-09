"""Share the participant element and the clearing of the tests of the matching interactions."""

from collections.abc import Callable, Mapping
from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import MatchingInteraction
from skcomplex.core.components.systems import System
from skcomplex.core.components.types import MatchingRule, SystemViewProtocol
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import AgentView
from skcomplex.core.specs import MatchSpec, StateSpec, VisualSpec

type Clearing = Callable[..., list[tuple[float, float, float, float, float]]]


class ParticipantView(AgentView):
    """View of a participant of a matching, with its states as attributes."""

    limit: float
    quantity: float
    allowance: float
    partner: float
    money: float
    goods: float
    value: float


class ParticipantElement(Element):
    """Participant of a matching with a limit, a quantity, a allowance, a partner, holdings, and a value."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        limit: Annotated[float, StateSpec()] = 0.0,
        quantity: Annotated[float, StateSpec()] = 1.0,
        allowance: Annotated[float, StateSpec()] = np.inf,
        partner: Annotated[float, StateSpec()] = -1.0,
        money: Annotated[float, StateSpec()] = 0.0,
        goods: Annotated[float, StateSpec()] = 0.0,
        value: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.limit = limit
        self.quantity = quantity
        self.allowance = allowance
        self.partner = partner
        self.money = money
        self.goods = goods
        self.value = value

    @classmethod
    def _get_view(cls) -> type[ParticipantView]:
        """Return the view of the participants, which types their states."""
        return ParticipantView


def _clear(
    rule: MatchingRule,
    requesters: Mapping[str, object],
    providers: Mapping[str, object],
    steps: int = 1,
) -> list[tuple[float, float, float, float, float]]:
    """Return the limits of both sides, the provider's value, the quantity, and the price of every match of a run."""
    records: list[tuple[float, float, float, float, float]] = []

    def record(match: MatchSpec, system: SystemViewProtocol) -> None:
        """Record the limits of both sides, the provider's value, the quantity, and the price of a match."""
        del system
        requester, provider = match.requester, match.provider
        if not isinstance(requester, ParticipantView) or not isinstance(provider, ParticipantView):
            return
        records.append((requester.limit, provider.limit, provider.value, match.quantity, match.value))

    sizes = [len(np.atleast_1d(np.asarray(values['limit']))) for values in (providers, requesters)]
    system = System(
        elements=[
            GroupElement('providers', element=ParticipantElement, size=sizes[0], values=dict(providers)),
            GroupElement('requesters', element=ParticipantElement, size=sizes[1], values=dict(requesters)),
        ],
        interactions=[
            MatchingInteraction(
                'matching',
                requesters='requesters',
                providers='providers',
                rule=rule,
                on_match=record,
            ),
        ],
    )
    simulate(system, until=steps, random_state=0)
    return records


@pytest.fixture
def clear() -> Clearing:
    """Return the function that runs a matching of providers created first and requesters created second."""
    return _clear
