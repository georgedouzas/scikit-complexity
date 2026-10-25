"""Organizations: hierarchies, queues, workplaces, and institutions."""

from ._employment import WorkplaceComponent, employment_takeover
from ._hierarchies import ContenderComponent, bonabeau_hierarchies
from ._obesity import EaterComponent, RestaurantComponent, basal_metabolic_rate, obesity
from ._queues import ExecutionInteraction, TaskComponent, priority_queues
from ._sabotage import EmployeeComponent, SupervisorComponent, insider_sabotage
from ._trafficking import DailyMovementsInteraction, PremisesComponent, TravellerComponent, trafficking
from ._traps import (
    ArmsRaceComponent,
    BurdenComponent,
    ContestComponent,
    escalation,
    shifting_the_burden,
    success_to_the_successful,
)

__all__: list[str] = [
    'ArmsRaceComponent',
    'BurdenComponent',
    'ContenderComponent',
    'ContestComponent',
    'DailyMovementsInteraction',
    'EaterComponent',
    'EmployeeComponent',
    'ExecutionInteraction',
    'PremisesComponent',
    'RestaurantComponent',
    'SupervisorComponent',
    'TaskComponent',
    'TravellerComponent',
    'WorkplaceComponent',
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
