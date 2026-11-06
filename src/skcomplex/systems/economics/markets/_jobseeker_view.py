"""Define the view of a worker of the labour market, as its rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class JobseekerView(AgentView):
    """Worker as the rules see it: its search and job states, and the parameters of separation and search.

    The states are the reservation wage `ask`, the search flag `supply`, the `employer`, the `wage`, the `duration` of
    search, and the steps `hired` and `offer` of the last hire. The parameters are the `separation` rate, the initial
    `reservation` wage, and its `decline` per step of search.
    """

    ask: float
    supply: float
    employer: float
    wage: float
    duration: float
    hired: float
    offer: float
    separation: float
    reservation: float
    decline: float
