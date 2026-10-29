"""Define the view of a person of Metaphoria, as the rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class SettlerView(AgentView):
    """Person as the rules see it: its holdings, age, family, job, fate, propensities, and needs."""

    food: float
    money: float
    loans: float
    age: float
    parent: float
    employer: float
    harvested: float
    born: float
    fate: float
    fertility: float
    exploration: float
    enterprise: float
    restlessness: float
    labour: float
    need: float
    maturity: float
    lifespan: float
    spacing: float
    endowment: float
    mutation: float
    sight: float
    reserve: float
    monetary: float
