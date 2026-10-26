"""Organizations: hierarchies, queues, workplaces, and institutions."""

from ._employment import WorkplaceElement, employment_takeover
from ._hierarchies import ContenderElement, bonabeau_hierarchies
from ._obesity import EaterElement, RestaurantElement, basal_metabolic_rate, obesity
from ._queues import ExecutionInteraction, TaskElement, priority_queues
from ._sabotage import EmployeeElement, SupervisorElement, insider_sabotage
from ._trafficking import DailyMovementsInteraction, PremisesElement, TravellerElement, trafficking
from ._traps import (
    ArmsRaceElement,
    BurdenElement,
    ContestElement,
    escalation,
    shifting_the_burden,
    success_to_the_successful,
)

__all__: list[str] = [
    'ArmsRaceElement',
    'BurdenElement',
    'ContenderElement',
    'ContestElement',
    'DailyMovementsInteraction',
    'EaterElement',
    'EmployeeElement',
    'ExecutionInteraction',
    'PremisesElement',
    'RestaurantElement',
    'SupervisorElement',
    'TaskElement',
    'TravellerElement',
    'WorkplaceElement',
    'basal_metabolic_rate',
    'bonabeau_hierarchies',
    'employment_takeover',
    'escalation',
    'insider_sabotage',
    'obesity',
    'priority_queues',
    'shifting_the_burden',
    'success_to_the_successful',
    'trafficking',
]
