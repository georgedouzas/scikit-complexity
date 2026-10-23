"""Agent-based models: rules written as code, matchings, phases of a step, agent memories, and the Mesa backend."""

from ...base import Form
from .._formalism import Formalism, register
from ._backends import AgentView, MesaStepper, ModelView
from ._components import memory
from ._laws import Match, Rule, matches, random_pairing, runs_rules
from ._passes import AgentsPass, AgentsProblemPass, SetRulesApartPass, UnpackMatchesPass, _agent_owner
from ._spaces import Phase, Phases

__all__: list[str] = [
    'AgentView',
    'Match',
    'MesaStepper',
    'ModelView',
    'Phase',
    'Phases',
    'Rule',
    'matches',
    'memory',
    'random_pairing',
    'runs_rules',
]

register(
    Formalism(
        'agents',
        laws=(Rule, Match),
        passes=(AgentsPass(), UnpackMatchesPass(), SetRulesApartPass(), AgentsProblemPass()),
        forms=(Form.AGENT,),
        solvers={'mesa': MesaStepper},
        defaults=(MesaStepper,),
        stepwise_forms={frozenset({Rule}): Form.AGENT},
        owners=(_agent_owner,),
    ),
)
