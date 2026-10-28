"""Name the forms of compiled problems and the kinds of what solvers and backends support."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from enum import StrEnum


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


class ElementKind(StrEnum):
    """Kind of element or system that carries states, as solvers and backends support them."""

    ELEMENT = 'element'
    GROUP = 'group'
    VARIABLE_GROUP = 'variable-group'
    SITE_GROUP = 'site-group'
    EDGE_GROUP = 'edge-group'
    HYPEREDGE_GROUP = 'hyperedge-group'
    SYSTEM = 'system'


class DomainKind(StrEnum):
    """Kind of value a state takes, which decides the laws that may target it."""

    REAL = 'real'
    INTEGER = 'integer'
    BOOLEAN = 'boolean'
    CATEGORICAL = 'categorical'
    ANGLE = 'angle'
    COMPLEX = 'complex'
    VECTOR = 'vector'
    TENSOR = 'tensor'
    HILBERT = 'hilbert'
    SEQUENCE = 'sequence'
    GRAPH = 'graph'
    TILING = 'tiling'


class OrderKind(StrEnum):
    """Kind of order of the time derivative that the rate laws on a state give."""

    INTEGER = 'integer'
    FRACTIONAL = 'fractional'


class NoiseKind(StrEnum):
    """Structure of the Wiener processes that the noise laws of a system draw."""

    DIAGONAL = 'diagonal'
    SHARED = 'shared'
    CORRELATED = 'correlated'
    CONSERVED = 'conserved'


_NUMERIC_DOMAINS = frozenset(
    {DomainKind.REAL, DomainKind.INTEGER, DomainKind.BOOLEAN, DomainKind.ANGLE, DomainKind.COMPLEX, DomainKind.VECTOR},
)
"""Kinds of value that numeric solvers and backends hold."""
