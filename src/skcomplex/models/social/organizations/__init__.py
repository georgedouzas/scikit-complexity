"""Organizations: hierarchies, queues, workplaces, and institutions."""

from ._employment import Workplace, employment_takeover
from ._hierarchies import Contender, bonabeau_hierarchies
from ._obesity import Eater, Restaurant, basal_metabolic_rate, obesity
from ._queues import Execution, Task, priority_queues
from ._sabotage import Employee, Supervisor, insider_sabotage
from ._trafficking import DailyMovements, Premises, Traveller, trafficking
from ._traps import ArmsRace, Burden, Contest, escalation, shifting_the_burden, success_to_the_successful

__all__: list[str] = [
    'ArmsRace',
    'Burden',
    'Contender',
    'Contest',
    'DailyMovements',
    'Eater',
    'Employee',
    'Execution',
    'Premises',
    'Restaurant',
    'Supervisor',
    'Task',
    'Traveller',
    'Workplace',
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
