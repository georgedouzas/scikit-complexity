"""Define the types of cooperation: the scores agents imitate."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable

from ....engine.solvers import AgentView

type Score = str | Callable[[AgentView], float]
