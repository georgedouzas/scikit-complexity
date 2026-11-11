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


class TieKind(StrEnum):
    """A kind of break of ties between equal values, such as by their order."""

    ORDER = 'order'
    RANDOM = 'random'


class ScheduleKind(StrEnum):
    """A kind of schedule by which an interaction updates its agents in a step.

    Examples:
        >>> from skcomplex.core.enums import ScheduleKind
        >>> 'member' in ScheduleKind
        True
    """

    SYNCHRONOUS = 'synchronous'
    SEQUENTIAL = 'sequential'
    RANDOM_SEQUENTIAL = 'random-sequential'
    MEMBER = 'member'


class OptimizerKind(StrEnum):
    """A kind of optimizer that trains parameters by gradient steps, such as Adam."""

    SGD = 'sgd'
    MOMENTUM = 'momentum'
    RMSPROP = 'rmsprop'
    ADAM = 'adam'


class BoundaryKind(StrEnum):
    """A kind of boundary of a space or of a solution, such as periodic.

    Every context accepts some of the kinds, such as a lattice or the inversion of a field.

    Examples:
        >>> from skcomplex.core.enums import BoundaryKind
        >>> BoundaryKind('zero-flux').name
        'ZERO_FLUX'
    """

    PERIODIC = 'periodic'
    FIXED = 'fixed'
    REFLECTING = 'reflecting'
    ZERO_FLUX = 'zero-flux'


class ComponentReductionKind(StrEnum):
    """A kind of reduction of the connected components of active members."""

    LABEL = 'label'
    SIZE = 'size'
    LARGEST = 'largest'
    GIANT = 'giant'


class PathReductionKind(StrEnum):
    """A kind of reduction of the shortest paths between active members."""

    BETWEENNESS = 'betweenness'
    CLOSENESS = 'closeness'


class DrawKind(StrEnum):
    """A kind of random draw of a law, such as normal.

    `DRAW` names a draw from a distribution of `scipy.stats`.

    Examples:
        >>> from skcomplex.core.enums import DrawKind
        >>> DrawKind.BERNOULLI == 'bernoulli'
        True
    """

    UNIFORM = 'uniform'
    NORMAL = 'normal'
    BERNOULLI = 'bernoulli'
    CATEGORICAL = 'categorical'
    DRAW = 'draw'


class ExecutorKind(StrEnum):
    """A kind of executor that runs the calls of an experiment, such as threads."""

    SERIAL = 'serial'
    THREAD = 'thread'
    PROCESS = 'process'


class CalibrationMethodKind(StrEnum):
    """A kind of method that calibrates the parameters of a system to data."""

    LOCAL = 'local'
    GLOBAL = 'global'
    ABC_SMC = 'abc_smc'
    MCMC = 'mcmc'
    VARIATIONAL = 'variational'


class LossKind(StrEnum):
    """A kind of loss that a calibration minimizes, such as the sum of squares."""

    SQUARES = 'squares'
    LIKELIHOOD = 'likelihood'
    EVENTS = 'events'


class SeverityKind(StrEnum):
    """A kind of severity of a problem that validation finds.

    An error stops compilation and a warning does not.
    """

    ERROR = 'error'
    WARNING = 'warning'


class MoveKind(StrEnum):
    """A kind of change of the value of a lattice site that a sweep proposes."""

    FLIP = 'flip'
    RANDOM_VALUE = 'random-value'
    ROTATE = 'rotate'
    GAUSSIAN = 'gaussian'
    FUNCTION = 'function'
    CLUSTER = 'cluster'


class ActivationKind(StrEnum):
    """A kind of activation of the agents in a phase of a step, such as random."""

    RANDOM = 'random'
    SEQUENTIAL = 'sequential'
    SIMULTANEOUS = 'simultaneous'
    BY_TYPE = 'by-type'
    VISITS = 'visits'


class SamplerKind(StrEnum):
    """A kind of acceptance of Monte Carlo moves, such as the Metropolis rule."""

    METROPOLIS = 'metropolis'
    GLAUBER = 'glauber'
    HEAT_BATH = 'heat-bath'


class SweepKind(StrEnum):
    """A kind of order in which a Monte Carlo sweep visits the sites of a lattice."""

    CHECKERBOARD = 'checkerboard'
    RANDOM_SEQUENTIAL = 'random-sequential'
    COLORED = 'colored'


class TimeTag(StrEnum):
    """A tag of the time a component, a solver, or a backend runs in."""

    CONTINUOUS = 'continuous'
    DISCRETE = 'discrete'
    STATIC = 'static'


class DomainTag(StrEnum):
    """A tag of the domain of the states of a component, a solver, or a backend."""

    REAL = 'real'
    INTEGER = 'integer'
    BOOLEAN = 'boolean'
    CATEGORICAL = 'categorical'
    COMPLEX = 'complex'
    VECTOR = 'vector'
    MATRIX_GROUP = 'matrix group'
    POWER_SERIES = 'power series'


class SpaceTag(StrEnum):
    """A tag of the space a component, a solver, or a backend lives in."""

    NONE = 'none'
    CONTINUOUS = 'continuous'
    LATTICE = 'lattice'
    NETWORK = 'network'
    HYPERGRAPH = 'hypergraph'
    CELL_COMPLEX = 'cell complex'
    PHASE_SPACE = 'phase space'
    CURVED_SPACETIME = 'curved spacetime'


class LawTag(StrEnum):
    """A tag of the kind of the laws of a component, a solver, or a backend.

    Examples:
        >>> from skcomplex.core.enums import LawTag
        >>> 'noise' in LawTag
        True
    """

    RATE = 'rate'
    NOISE = 'noise'
    JUMP = 'jump'
    UPDATE = 'update'
    EQUATION = 'equation'
    SAMPLING = 'sampling'


class RandomnessTag(StrEnum):
    """A tag of the randomness of a component, a solver, or a backend."""

    DETERMINISTIC = 'deterministic'
    STOCHASTIC = 'stochastic'


class StructureTag(StrEnum):
    """A tag of the way the structure of a system changes, such as its topology."""

    VARIABLE_MEMBERSHIP = 'variable membership'
    CHANGING_TOPOLOGY = 'changing topology'


class SymmetryKind(StrEnum):
    """A kind of symmetry of a rank-2 tensor, which decides its independent components."""

    FULL = 'full'
    SYMMETRIC = 'symmetric'
    ANTISYMMETRIC = 'antisymmetric'
    TRACELESS = 'traceless'


class TopologyKind(StrEnum):
    """A kind of request that changes the topology of a network, such as rewire."""

    ADD_EDGE = 'add-edge'
    REMOVE_EDGE = 'remove-edge'
    REWIRE = 'rewire'
    ADD_NODE = 'add-node'
    REMOVE_NODE = 'remove-node'
    ADD_HYPEREDGE = 'add-hyperedge'
    REMOVE_HYPEREDGE = 'remove-hyperedge'
    REPLACE_NODE = 'replace-node'
    CLOSE_TRIANGLE = 'close-triangle'
    CONTRACT_EDGE = 'contract-edge'
    DUPLICATE_NODE = 'duplicate-node'
