"""Define the view of a person of Metaphoria, as the rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

from ....engine.solvers import AgentView


class SettlerView(AgentView):
    """Person as the rules see it: its holdings, age, family, job, fate, propensities, and needs.

    Besides its states and parameters it reads two quantities of the markets: `hireable`, the highest wage limit of a
    job it accepts, which admits every job for an unemployed adult and none otherwise, and `shortfall`, the food it
    lacks for `reserve` steps, whose decrease by a purchase is the food the person receives.
    """

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

    @property
    def hireable(self) -> float:
        """Return the limit of the jobs the person accepts, unlimited for an unemployed adult and none otherwise."""
        return math.inf if self.employer < 0 and self.age >= self.maturity else -math.inf

    @property
    def shortfall(self) -> float:
        """Return the food the person lacks for `reserve` steps."""
        return self.reserve * self.need - self.food

    @shortfall.setter
    def shortfall(self, value: float) -> None:
        """Set the food so that the person lacks `value` for `reserve` steps."""
        self.food = self.reserve * self.need - value
