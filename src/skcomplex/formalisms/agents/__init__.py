"""Agent-based models: rules written as code, matchings, phases of a step, agent memories, and the Mesa backend."""

from ...core.base import Form
from .._formalism import Formalism, register
from ._backends import AgentView, MesaSolver, ModelView
from ._components import memory
from ._laws import MatchLaw, RuleLaw, matches, random_pairing, runs_rules
from ._passes import AgentsPass, AgentsProblemPass, SetRulesApartPass, UnpackMatchesPass, _agent_owner
from ._spaces import Phase, PhasesSchedule

__all__: list[str] = [
    'AgentView',
    'MatchLaw',
    'MesaSolver',
    'ModelView',
    'Phase',
    'PhasesSchedule',
    'RuleLaw',
    'matches',
    'memory',
    'random_pairing',
    'runs_rules',
]

register(
    Formalism(
        'agents',
        laws=(RuleLaw, MatchLaw),
        passes=(AgentsPass(), UnpackMatchesPass(), SetRulesApartPass(), AgentsProblemPass()),
        forms=(Form.AGENT,),
        solvers={'mesa': MesaSolver},
        defaults=(MesaSolver,),
        stepwise_forms={frozenset({RuleLaw}): Form.AGENT},
        owners=(_agent_owner,),
    ),
)
