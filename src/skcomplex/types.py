"""Define the types every layer of the package reads: arrays, tuples, functions, keys, and kinds."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping
from enum import StrEnum

import numpy as np
import numpy.typing as npt

# arrays and tuples
type Array = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]
type Indices = npt.NDArray[np.intp]
type ComplexArray = npt.NDArray[np.complex128]
type AnyArray = npt.NDArray[np.generic]
type IntTuple = tuple[int, ...]
type Pair = tuple[int, int]

# functions and keys
type VectorFunction = Callable[[float, Array, Array], Array]
type ProcessKey = tuple[str, str, str | int]
type Preference = Callable[[object, object], float]
type Stoichiometry = Mapping[tuple[str, str] | str, float | Callable[..., object]]


class DataKind(StrEnum):
    """Kind of data an analysis consumes, whatever model or measurement produced it.

    A kind equals its value, so an analysis may return the plain string, such as `'series'`.

    Examples:
        >>> from skcomplex.types import DataKind
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
            DataKind.AGENTS: 'positions, velocities, headings, or states of agents, one row per agent',
            DataKind.NETWORK: 'a NetworkX graph, or one graph per frame',
            DataKind.ENSEMBLE: 'one row per run and one column per common time',
            DataKind.SYSTEM: (
                'a system, a problem compiled from one, or a trajectory of one, whose equations are analysed'
            ),
        }
        return readings[self]


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


class VariableKind(StrEnum):
    """Role a declared argument plays in a part."""

    STATE = 'state'
    PARAMETER = 'parameter'
    ROLE = 'role'


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


class FormKind(StrEnum):
    """Mathematical form of a compiled system, which decides the solvers that advance it.

    Examples:
        >>> from skcomplex.types import FormKind
        >>> FormKind.SDE.value
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
