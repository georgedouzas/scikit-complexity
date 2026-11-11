"""Define the kinds that every layer of the library reads."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from enum import StrEnum


class DataKind(StrEnum):
    """A kind of data that an analysis reads, whatever system or measurement made it.

    A kind equals its value, so an analysis may return the plain string, such as `'series'`.

    Examples:
        >>> from skcomplex.core.enums import DataKind
        >>> DataKind('field').reading
        'a grid of values, or frames of grids'
        >>> DataKind.SERIES == 'series'
        True
    """

    SERIES = 'series'
    PATH = 'path'
    FIELD = 'field'
    AGENTS = 'agents'
    NETWORK = 'network'
    ENSEMBLE = 'ensemble'
    SYSTEM = 'system'

    @property
    def reading(self) -> str:
        """Plain arrays or graphs the analyses of the kind read."""
        readings = {
            DataKind.SERIES: 'a series of values, or the columns of an array as series',
            DataKind.PATH: 'the jump times and the states after every jump',
            DataKind.FIELD: 'a grid of values, or frames of grids',
            DataKind.AGENTS: 'positions, velocities, directions, or states of agents, one row per agent',
            DataKind.NETWORK: 'a NetworkX graph, or one graph per frame',
            DataKind.ENSEMBLE: 'one row per run and one column per common time',
            DataKind.SYSTEM: (
                'a system, a problem compiled from one, or a trajectory of one, whose equations are analysed'
            ),
        }
        return readings[self]


class DomainKind(StrEnum):
    """A kind of value that a state takes, such as real or integer.

    It decides the laws that may target the state.
    """

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


class VariableKind(StrEnum):
    """A kind of role that a declared argument plays in a component, such as a state."""

    STATE = 'state'
    PARAMETER = 'parameter'
    ROLE = 'role'


class OrderKind(StrEnum):
    """A kind of order of the time derivative that the rate laws on a state give."""

    INTEGER = 'integer'
    FRACTIONAL = 'fractional'


class NoiseKind(StrEnum):
    """A kind of structure of the noise processes that the noise laws of a system draw.

    The processes are Wiener processes, or fractional Brownian motions for noise laws with a Hurst index.

    Examples:
        >>> from skcomplex.core.enums import NoiseKind
        >>> NoiseKind.FRACTIONAL.value
        'fractional'
    """

    DIAGONAL = 'diagonal'
    SHARED = 'shared'
    CORRELATED = 'correlated'
    CONSERVED = 'conserved'
    FRACTIONAL = 'fractional'


class FormKind(StrEnum):
    """A kind of form of a compiled system, such as differential equations.

    It decides the solvers that can advance the system.

    Examples:
        >>> from skcomplex.core.enums import FormKind
        >>> FormKind.SDE.value
        'stochastic differential equations'
    """

    ODE = 'ordinary differential equations'
    FRACTIONAL = 'fractional differential equations'
    SDE = 'stochastic differential equations'
    JUMP = 'jump process'
    DISCRETE = 'discrete map'
    AGENT = 'agent-based system'
    LATTICE = 'lattice dynamics'
    MONTE_CARLO = 'Monte Carlo sampling'
    RENEWAL = 'renewal process'
    COUPLED = 'co-simulation'
    OPERATOR = 'operator dynamics'
    LEARNING = 'learning by gradient steps'
    STATIC = 'static solution'
    REWRITING = 'rewriting system'


class ElementKind(StrEnum):
    """A kind of element or system that carries states.

    Solvers, backends, and the roles of interactions name these kinds. `ANY` names every kind in a role that accepts
    them all, and no element is of this kind.
    """

    ANY = 'any'
    ELEMENT = 'element'
    GROUP = 'group'
    VARIABLE_GROUP = 'variable-group'
    SITE_GROUP = 'site-group'
    EDGE_GROUP = 'edge-group'
    HYPEREDGE_GROUP = 'hyperedge-group'
    SYSTEM = 'system'
