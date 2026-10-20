"""Define the mathematical forms a system compiles to."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from enum import Enum


class Form(Enum):
    """Mathematical form of a compiled system."""

    ODE = 'ordinary differential equations'
    SDE = 'stochastic differential equations'
    JUMP = 'jump process'
    DISCRETE = 'discrete map'
    AGENT = 'agent-based model'
    LATTICE = 'lattice dynamics'
    MONTE_CARLO = 'Monte Carlo sampling'
    REWRITING = 'rewriting system'
    COUPLED = 'co-simulation'
