"""Mathematical form of a compiled system, which decides the solvers that advance it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from enum import StrEnum

from ._domain_kind import DomainKind


class Form(StrEnum):
    """Mathematical form of a compiled system, which decides the solvers that advance it.

    Examples:
        >>> from skcomplex.engine.problems import Form
        >>> Form.SDE.value
        'stochastic differential equations'
    """

    ODE = 'ordinary differential equations'
    FRACTIONAL = 'fractional differential equations'
    SDE = 'stochastic differential equations'
    JUMP = 'jump process'
    DISCRETE = 'discrete map'
    AGENT = 'agent-based model'
    LATTICE = 'lattice dynamics'
    MONTE_CARLO = 'Monte Carlo sampling'
    REWRITING = 'rewriting system'
    RENEWAL = 'renewal process'
    COUPLED = 'co-simulation'
    QUANTUM = 'quantum dynamics'


_NUMERIC_DOMAINS = frozenset(
    {DomainKind.REAL, DomainKind.INTEGER, DomainKind.BOOLEAN, DomainKind.ANGLE, DomainKind.COMPLEX, DomainKind.VECTOR},
)
"""Kinds of value that numeric solvers and backends hold."""
