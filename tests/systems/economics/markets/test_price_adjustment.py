"""Test the price adjustment of the `_shops` module of `skcomplex.systems.economics.markets`."""

from typing import Annotated

import pytest

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.components.types import AgentViewProtocol, SystemViewProtocol
from skcomplex.core.engine.runtime import Trajectory, simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import StateSpec, VisualSpec
from skcomplex.systems.economics.markets import PriceAdjustmentInteraction


class QuoteElement(Element):
    """Holder of a quote `x` and a goal `goal`."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 1.0,
        goal: Annotated[float, StateSpec()] = 2.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.goal = goal


def _run(element: Element, adjustment: PriceAdjustmentInteraction, until: int) -> Trajectory:
    """Return the trajectory of an element whose quote a price adjustment steps."""
    system = CompositeSystem(elements=[element], interactions=[adjustment], time=DiscreteTime())
    return simulate(system, until=until, random_state=0)


def test_price_adjustment_steps_multiplicatively_up_to_the_bound():
    """Test that the multiplicative stepping moves by the relative step towards the sign of the signal, in bounds."""
    search = PriceAdjustmentInteraction(
        'search',
        target='quote',
        state='x',
        signal=lambda s: s.target.goal - s.target.x,
        step=0.5,
        upper=2.0,
    )
    assert _run(QuoteElement('quote'), search, 3).values('quote.x').tolist() == [1.0, 1.5, 2.0, 2.0]


def test_price_adjustment_steps_additively_and_stops_at_a_zero_signal():
    """Test that the additive stepping steps by the step and stays where the signal is zero."""
    search = PriceAdjustmentInteraction(
        'search',
        target='quote',
        state='x',
        signal=lambda s: s.target.goal - s.target.x,
        step=0.5,
        stepping='additive',
    )
    assert _run(QuoteElement('quote', x=3.0), search, 3).values('quote.x').tolist() == [3.0, 2.5, 2.0, 2.0]


def test_price_adjustment_agent_form_steps_every_agent_within_its_bound():
    """Test that the agent form steps every agent by its own step and keeps it above its lower bound."""
    quotes = GroupElement('quotes', element=QuoteElement, size=2, values={'x': [1.0, 4.0], 'goal': [0.0, 8.0]})
    search = PriceAdjustmentInteraction(
        'search',
        target='quotes',
        state='x',
        signal=lambda agent, system: agent.goal - agent.x,
        step=lambda agent, system: 0.5,
        lower=lambda agent, system: 0.8,
        form='agent',
    )
    trajectory = _run(quotes, search, 2)
    assert trajectory.get_agents('quotes', 1).values['x'].tolist() == pytest.approx([0.8, 6.0])
    assert trajectory.get_agents('quotes', 2).values['x'].tolist() == pytest.approx([0.8, 9.0])


def _crowded(agent: AgentViewProtocol, system: SystemViewProtocol) -> float:
    """Return a signal that is positive while the quotes of the group sum to less than 2.5."""
    return 2.5 - sum(other.x for other in system.get_agents('target'))


@pytest.mark.parametrize(
    ('schedule', 'expected'),
    [('synchronous', [2.0, 2.0]), ('sequential', [2.0, 0.0]), ('member', [0.0, 2.0])],
)
def test_price_adjustment_agent_schedules_read_the_states_set_before(schedule, expected):
    """Test that sequential and member steps read the quotes earlier agents set, while synchronous steps do not."""
    quotes = GroupElement('quotes', element=QuoteElement, size=2, values={'x': [1.0, 1.0]})
    search = PriceAdjustmentInteraction(
        'search',
        target='quotes',
        state='x',
        signal=_crowded,
        step=1.0,
        stepping='additive',
        form='agent',
        schedule=schedule,
    )
    values = _run(quotes, search, 1).get_agents('quotes', 1).values['x'].tolist()
    assert sorted(values) == sorted(expected)
    if schedule == 'sequential':
        assert values == expected


@pytest.mark.parametrize(
    'arguments',
    [{'stepping': 'power'}, {'form': 'rule'}, {'signal': 1.0}, {'schedule': 'member'}, {'schedule': 'parallel'}],
)
def test_check_rejects_invalid_arguments(arguments):
    """Test that an unknown stepping or form, or a signal that is not callable, fails."""
    settings = {'signal': lambda s: 1, **arguments}
    with pytest.raises(ValidationError):
        PriceAdjustmentInteraction('search', target='quote', state='x', **settings).prepare()
