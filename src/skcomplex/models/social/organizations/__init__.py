"""Organizations: hierarchies, queues, workplaces, and institutions."""

from ._arms_race import ArmsRaceElement
from ._burden import BurdenElement
from ._contender_view import ContenderView
from ._contest import ContestElement
from ._daily_movements import DailyMovementsInteraction
from ._eater import EaterElement, basal_metabolic_rate
from ._eater_view import EaterView
from ._employee import EmployeeElement
from ._employee_view import EmployeeView
from ._employment import WorkplaceElement, employment_takeover
from ._execution import ExecutionInteraction
from ._hierarchies import ContenderElement, bonabeau_hierarchies
from ._obesity import obesity
from ._premises import PremisesElement
from ._premises_view import PremisesView
from ._queues import priority_queues
from ._restaurant import RestaurantElement
from ._restaurant_view import RestaurantView
from ._sabotage import insider_sabotage
from ._supervisor import SupervisorElement
from ._task import TaskElement
from ._task_view import TaskView
from ._trafficking import trafficking
from ._traps import escalation, shifting_the_burden, success_to_the_successful
from ._traveller import TravellerElement
from ._traveller_view import TravellerView

__all__: list[str] = [
    'ArmsRaceElement',
    'BurdenElement',
    'ContenderElement',
    'ContenderView',
    'ContestElement',
    'DailyMovementsInteraction',
    'EaterElement',
    'EaterView',
    'EmployeeElement',
    'EmployeeView',
    'ExecutionInteraction',
    'PremisesElement',
    'PremisesView',
    'RestaurantElement',
    'RestaurantView',
    'SupervisorElement',
    'TaskElement',
    'TaskView',
    'TravellerElement',
    'TravellerView',
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
