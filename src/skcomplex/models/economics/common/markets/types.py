"""Define the types of the markets: the preferences of the participants."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable

from .....engine.solvers import AgentView

type Preference = Callable[[AgentView, AgentView], float]
