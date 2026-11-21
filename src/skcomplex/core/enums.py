"""Define the named choices that every layer of the library reads."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from enum import StrEnum


class DataKind(StrEnum):
    """A kind of data that an analysis reads, whatever system or measurement made it.

    It names what an analysis needs, such as `'series'` for a list of values over time, and `reading` says it in words.
    A kind equals its value, so an analysis may return the plain string. A series holds the price of a stock, a field
    the temperature over a map, and a network the friendships of a school.

    Read more in the user guide, under Analyse.

    See Also:
        AnalysisTags: Lists the kinds of data an analysis reads.

    Examples:
        >>> from skcomplex.core.enums import DataKind
        >>> DataKind('field').reading
        'a grid of values, or frames of grids'
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

    It says what numbers or objects a value of an element can hold, which decides the laws that may change it, such as
    `'integer'` for a count that a jump law raises by one. The number of molecules is an integer, the phase of an
    oscillator an angle, and the opinion of a voter a category.

    Read more in the user guide, under Elements.

    See Also:
        DomainTag: Names the kinds of value a solver or a backend runs.

    Examples:
        >>> from skcomplex.core.enums import DomainKind
        >>> DomainKind('angle') == 'angle'
        True
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
    """A kind of role that a declared argument plays in a component, such as a state.

    It sorts the arguments of a component into states that laws change, parameters that stay fixed, and roles that
    name the elements an interaction binds, such as `'role'` for the `target` of an interaction. The position of a
    ball is a state, the gravity a parameter, and the ball the role a force acts on.

    Read more in the user guide, under Elements.

    See Also:
        DomainKind: Names the kind of value a state takes.

    Examples:
        >>> from skcomplex.core.enums import VariableKind
        >>> [kind.value for kind in VariableKind]
        ['state', 'parameter', 'role']
    """

    STATE = 'state'
    PARAMETER = 'parameter'
    ROLE = 'role'


class OrderKind(StrEnum):
    """A kind of order of the time derivative that the rate laws on a state give.

    It says whether a rate law changes a state through a whole derivative, such as `x'` or `x''`, or through a
    fractional one, which remembers the whole past. A falling ball has an integer order, and the creep of a polymer, the
    spread of a drug in tissue, and the memory of a market have fractional orders.

    Read more in the user guide, under Solvers.

    See Also:
        FormKind: Names the kind of problem the laws make.

    Examples:
        >>> from skcomplex.core.enums import OrderKind
        >>> [kind.value for kind in OrderKind]
        ['integer', 'fractional']
    """

    INTEGER = 'integer'
    FRACTIONAL = 'fractional'


class NoiseKind(StrEnum):
    """A kind of structure of the noise that the noise laws of a system draw.

    It says how the random kicks of noise laws relate, such as `'diagonal'` when every state gets a kick of its own and
    `'shared'` when they share one. Fractional noise has a memory set by its Hurst index. Each stock of a market, each
    particle of a gas, and each neuron of a network may get noise of its own or share a common one.

    Read more in the user guide, under Solvers.

    See Also:
        FormKind: Names the kind of problem the laws make.

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
    """A kind of problem that the laws of a system make, such as rate equations.

    A system compiles its laws into one problem, and its form decides the solvers that can run it, such as
    `'ordinary differential equations'` for rate laws alone. The orbit of a planet is a set of ordinary differential
    equations, a chemical reaction in a cell a jump process, and the Game of Life a set of lattice updates.

    Read more in the user guide, under Solvers.

    See Also:
        OrderKind: Names the order of the derivatives of rate laws.
        NoiseKind: Names how the noise of a system is shared.

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

    Solvers, backends, and the roles of interactions name these kinds to say what they accept, such as `'group'` for a
    group of many members. `'any'` accepts every kind, and no element is of that kind. A single ball is an element, a
    flock a group, and the links of a network a group on edges.

    Read more in the user guide, under Elements.

    See Also:
        VariableKind: Names the role an argument plays in a component.

    Examples:
        >>> from skcomplex.core.enums import ElementKind
        >>> ElementKind('edge-group').name
        'EDGE_GROUP'
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
    """A kind of break of ties between equal values, such as by their order.

    An analysis that ranks values, such as one that turns a series into the order of its values, breaks a tie either by
    position or at random, as in `complexity(x, ties='random')`. Ties arise in rounded prices, counts of cases, and
    readings of a coarse sensor.

    Read more in the user guide, under Analyse.

    See Also:
        DataKind: Names the kind of data an analysis reads.

    Examples:
        >>> from skcomplex.core.enums import TieKind
        >>> [kind.value for kind in TieKind]
        ['order', 'random']
    """

    ORDER = 'order'
    RANDOM = 'random'


class ScheduleKind(StrEnum):
    """A kind of schedule by which an interaction updates its agents in a step.

    It says whether all agents update at once, one after the other in a fixed or random order, or each on its own
    clock, such as `'random-sequential'` for agents picked one by one in a new random order every step. Voters copy a
    neighbour one at a time, traders all act at the close, and cells divide on their own clocks.

    Read more in the user guide, under Interactions.

    See Also:
        ActivationKind: Names the order of agents within a phase of a step.

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
    """A kind of optimizer that trains parameters by gradient steps, such as Adam.

    It names the rule that moves the parameters against the gradient of a loss at every step, from plain steps,
    `'sgd'`, to steps that adapt their size, `'adam'`. It trains the weights of a neural network, fits the rates of a
    model of growth, and tunes the gains of a controller.

    Read more in the user guide, under Solvers.

    See Also:
        CalibrationMethodKind: Names the method that fits parameters to data.

    Examples:
        >>> from skcomplex.core.enums import OptimizerKind
        >>> [kind.value for kind in OptimizerKind]
        ['sgd', 'momentum', 'rmsprop', 'adam']
    """

    SGD = 'sgd'
    MOMENTUM = 'momentum'
    RMSPROP = 'rmsprop'
    ADAM = 'adam'


class BoundaryKind(StrEnum):
    """A kind of edge of a space or of a solution, such as periodic.

    It says what lies beyond the edge, such as `'periodic'` when opposite edges join as on a ring, or `'zero-flux'`
    for an insulated wall. Each place accepts some of the kinds, such as a lattice or the solution of a field. A ring of
    cells is periodic, a rod held in ice has fixed ends, and an insulated wall lets no heat through.

    Read more in the user guide, under Spaces.

    See Also:
        SpaceTag: Names the kind of space a component lives in.

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
    """A kind of summary of the connected clusters of active members.

    A law that groups the active members joined by links into clusters reads one number per member or per system, such
    as `'size'` for the size of the cluster of each member or `'giant'` for the share in the largest one. It measures
    the outbreak of an epidemic, the spanning cluster of a porous rock, and the blackout of a power grid.

    Read more in the user guide, under Laws.

    See Also:
        PathReductionKind: Names the summary of the shortest paths between members.

    Examples:
        >>> from skcomplex.core.enums import ComponentReductionKind
        >>> [kind.value for kind in ComponentReductionKind]
        ['label', 'size', 'largest', 'giant']
    """

    LABEL = 'label'
    SIZE = 'size'
    LARGEST = 'largest'
    GIANT = 'giant'


class PathReductionKind(StrEnum):
    """A kind of summary of the shortest paths between active members.

    A law that reads the shortest paths of a network gets one number per member, `'betweenness'` for how many paths
    pass through it or `'closeness'` for how near it is to all others. It finds the busy junctions of a road network,
    the brokers of a social network, and the weak points of a power grid.

    Read more in the user guide, under Laws.

    See Also:
        ComponentReductionKind: Names the summary of the clusters of members.

    Examples:
        >>> from skcomplex.core.enums import PathReductionKind
        >>> [kind.value for kind in PathReductionKind]
        ['betweenness', 'closeness']
    """

    BETWEENNESS = 'betweenness'
    CLOSENESS = 'closeness'


class DrawKind(StrEnum):
    """A kind of random draw of a law, such as normal.

    It names the distribution a law draws a random number from, such as `'bernoulli'` for a coin that lands heads with
    a given chance, and `'draw'` for any distribution of `scipy.stats`. A person catches a disease with a chance, a
    price moves by a normal step, and a seed lands in a random cell.

    Read more in the user guide, under Laws.

    See Also:
        NoiseKind: Names how the noise of a system is shared.

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
    """A kind of executor that runs the calls of an experiment, such as threads.

    An experiment that runs a system many times runs the calls one after the other, in threads, or in processes, as in
    `ensemble(system, replicates=100, executor='process')`. It runs the seeds of an epidemic, the grid of parameters
    of a climate model, and the trials of a market at once.

    Read more in the user guide, under Run experiments.

    See Also:
        CalibrationMethodKind: Names the method that fits parameters to data.

    Examples:
        >>> from skcomplex.core.enums import ExecutorKind
        >>> [kind.value for kind in ExecutorKind]
        ['serial', 'thread', 'process']
    """

    SERIAL = 'serial'
    THREAD = 'thread'
    PROCESS = 'process'


class CalibrationMethodKind(StrEnum):
    """A kind of method that calibrates the parameters of a system to data.

    It names how `calibrate` searches for the parameters that make a run fit the data, from a local search, `'local'`,
    to a sample of the whole range of likely values, `'mcmc'`. It fits the rates of an epidemic to case counts, the
    constants of a reaction to measured concentrations, and the growth of a firm to its sales.

    Read more in the user guide, under Calibrate.

    See Also:
        LossKind: Names what a calibration minimizes.

    Examples:
        >>> from skcomplex.core.enums import CalibrationMethodKind
        >>> CalibrationMethodKind('abc_smc').name
        'ABC_SMC'
    """

    LOCAL = 'local'
    GLOBAL = 'global'
    ABC_SMC = 'abc_smc'
    MCMC = 'mcmc'
    VARIATIONAL = 'variational'


class LossKind(StrEnum):
    """A kind of loss that a calibration minimizes, such as the sum of squares.

    It names how far a run lies from the data: `'squares'` sums the squared gaps, `'likelihood'` scores how likely the
    data are under the run, and `'events'` compares the times of events. It scores the fit of an epidemic to case
    counts, of a reaction to measured concentrations, and of a model of earthquakes to their dates.

    Read more in the user guide, under Calibrate.

    See Also:
        CalibrationMethodKind: Names the method that searches for the parameters.

    Examples:
        >>> from skcomplex.core.enums import LossKind
        >>> [kind.value for kind in LossKind]
        ['squares', 'likelihood', 'events']
    """

    SQUARES = 'squares'
    LIKELIHOOD = 'likelihood'
    EVENTS = 'events'


class SeverityKind(StrEnum):
    """A kind of severity of a problem that the check of a system finds.

    An `'error'` stops the system from compiling, and a `'warning'` lets it compile and reports the problem. A missing
    value of a predator is an error, and a state that no law changes in a model of a market is a warning.

    Read more in the user guide, under Check.

    See Also:
        ValidationError: Raised when a check finds an error.

    Examples:
        >>> from skcomplex.core.enums import SeverityKind
        >>> [kind.value for kind in SeverityKind]
        ['error', 'warning']
    """

    ERROR = 'error'
    WARNING = 'warning'


class MoveKind(StrEnum):
    """A kind of change of the value of a lattice site that a sampling sweep proposes.

    A sweep proposes a new value for a site and accepts it or not, such as `'flip'` to turn a spin over or `'rotate'`
    to turn an angle by a step. It flips the spins of a magnet, mutates the letters of a genome, and turns the rods of
    a liquid crystal.

    Read more in the user guide, under Laws.

    See Also:
        SamplerKind: Names the rule that accepts or rejects a move.
        SweepKind: Names the order in which a sweep visits the sites.

    Examples:
        >>> from skcomplex.core.enums import MoveKind
        >>> MoveKind('random-value').name
        'RANDOM_VALUE'
    """

    FLIP = 'flip'
    RANDOM_VALUE = 'random-value'
    ROTATE = 'rotate'
    GAUSSIAN = 'gaussian'
    FUNCTION = 'function'
    CLUSTER = 'cluster'


class ActivationKind(StrEnum):
    """A kind of order in which the agents of a phase of a step act, such as random.

    It says which agents act in a phase and in what order, such as `'random'` for all agents in a new random order and
    `'by-type'` for one type of agent after another. Wolves hunt after sheep graze, traders act in random order, and
    voters all decide at once.

    Read more in the user guide, under Agents.

    See Also:
        ScheduleKind: Names how an interaction updates its agents in a step.

    Examples:
        >>> from skcomplex.core.enums import ActivationKind
        >>> ActivationKind('by-type').name
        'BY_TYPE'
    """

    RANDOM = 'random'
    SEQUENTIAL = 'sequential'
    SIMULTANEOUS = 'simultaneous'
    BY_TYPE = 'by-type'
    VISITS = 'visits'


class SamplerKind(StrEnum):
    """A kind of rule that accepts or rejects a proposed move, such as Metropolis.

    A sampling law proposes a change and keeps it with a chance that favours lower energy, such as `'metropolis'`,
    which always keeps a move downhill and sometimes one uphill. It samples the spins of a magnet, the folds of a
    protein, and the opinions of a population at a given noise.

    Read more in the user guide, under Laws.

    See Also:
        MoveKind: Names the change a sweep proposes.
        SweepKind: Names the order in which a sweep visits the sites.

    Examples:
        >>> from skcomplex.core.enums import SamplerKind
        >>> [kind.value for kind in SamplerKind]
        ['metropolis', 'glauber', 'heat-bath']
    """

    METROPOLIS = 'metropolis'
    GLAUBER = 'glauber'
    HEAT_BATH = 'heat-bath'


class SweepKind(StrEnum):
    """A kind of order in which a sampling sweep visits the sites of a lattice.

    A sweep visits every site once, in two colours like a chessboard, `'checkerboard'`, in a random order, or in more
    colours so no two neighbours update together. It samples the spins of a magnet, the gas on a grid, and the cells of
    a tissue.

    Read more in the user guide, under Laws.

    See Also:
        SamplerKind: Names the rule that accepts or rejects a move.

    Examples:
        >>> from skcomplex.core.enums import SweepKind
        >>> [kind.value for kind in SweepKind]
        ['checkerboard', 'random-sequential', 'colored']
    """

    CHECKERBOARD = 'checkerboard'
    RANDOM_SEQUENTIAL = 'random-sequential'
    COLORED = 'colored'


class TimeTag(StrEnum):
    """A tag of the time a component, a solver, or a backend runs in.

    It says whether time flows, `'continuous'`, steps, `'discrete'`, or does not pass, `'static'`, so a system picks a
    solver that fits its laws. A planet moves in continuous time, a cellular automaton in steps, and a market clears in
    a static balance.

    Read more in the user guide, under Solvers.

    See Also:
        SpaceTag: Names the space a component lives in.
        LawTag: Names the kinds of law a component holds.

    Examples:
        >>> from skcomplex.core.enums import TimeTag
        >>> [tag.value for tag in TimeTag]
        ['continuous', 'discrete', 'static']
    """

    CONTINUOUS = 'continuous'
    DISCRETE = 'discrete'
    STATIC = 'static'


class DomainTag(StrEnum):
    """A tag of the kind of value of the states of a component, a solver, or a backend.

    It says what values a solver or a backend can run, such as `'integer'` for counts, so a system picks one that fits
    its states. Counts of molecules are integers, the amplitudes of a wave complex numbers, and the velocities of a
    flock vectors.

    Read more in the user guide, under Solvers.

    See Also:
        DomainKind: Names the kind of value of one state.
        TimeTag: Names the time a component runs in.

    Examples:
        >>> from skcomplex.core.enums import DomainTag
        >>> DomainTag('matrix group').name
        'MATRIX_GROUP'
    """

    REAL = 'real'
    INTEGER = 'integer'
    BOOLEAN = 'boolean'
    CATEGORICAL = 'categorical'
    COMPLEX = 'complex'
    VECTOR = 'vector'
    MATRIX_GROUP = 'matrix group'
    POWER_SERIES = 'power series'


class SpaceTag(StrEnum):
    """A tag of the space a component, a solver, or a backend lives in.

    It names the structure the members sit on, such as `'lattice'` for a grid of sites, so a system picks a solver that
    can run it. The cells of a forest sit on a lattice, people on a contact network, and atoms in a continuous box.

    Read more in the user guide, under Spaces.

    See Also:
        TimeTag: Names the time a component runs in.
        BoundaryKind: Names what lies beyond the edge of a space.

    Examples:
        >>> from skcomplex.core.enums import SpaceTag
        >>> SpaceTag('cell complex').name
        'CELL_COMPLEX'
    """

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

    It names a kind of law, such as `'rate'` for a law of how fast a value changes, so a system picks a solver that can
    run all its laws. Gravity is a rate law, the toss of a coin a jump law, and the rule of the Game of Life an update
    law.

    Read more in the user guide, under Laws.

    See Also:
        TimeTag: Names the time a component runs in.
        RandomnessTag: Names whether a component draws random numbers.

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
    OBJECTIVE = 'objective'
    SAMPLING = 'sampling'


class RandomnessTag(StrEnum):
    """A tag of whether a component, a solver, or a backend draws random numbers.

    A `'deterministic'` component gives the same run every time, and a `'stochastic'` one draws random numbers, so its
    runs differ by seed. A swinging pendulum is deterministic, and the spread of a rumour and the price of a stock are
    stochastic.

    Read more in the user guide, under Solvers.

    See Also:
        LawTag: Names the kinds of law a component holds.

    Examples:
        >>> from skcomplex.core.enums import RandomnessTag
        >>> [tag.value for tag in RandomnessTag]
        ['deterministic', 'stochastic']
    """

    DETERMINISTIC = 'deterministic'
    STOCHASTIC = 'stochastic'


class StructureTag(StrEnum):
    """A tag of the way the structure of a system changes, such as its links.

    It says whether members come and go, `'variable membership'`, or links form and break, `'changing topology'`, so a
    system picks a solver that can follow it. Animals are born and die in an ecosystem, and friendships form and break
    in a social network.

    Read more in the user guide, under Solvers.

    See Also:
        SpaceTag: Names the space a component lives in.
        TopologyKind: Names a change of the links of a network.

    Examples:
        >>> from skcomplex.core.enums import StructureTag
        >>> StructureTag('changing topology').name
        'CHANGING_TOPOLOGY'
    """

    VARIABLE_MEMBERSHIP = 'variable membership'
    CHANGING_TOPOLOGY = 'changing topology'


class SymmetryKind(StrEnum):
    """A kind of symmetry of a matrix value, which decides how many entries are free.

    It says which entries of a 2 by 2, 3 by 3, or larger matrix value are free, such as `'symmetric'` when the entry at
    `(i, j)` equals the one at `(j, i)`. Stress in a solid is symmetric, the rotation of a fluid antisymmetric, and the
    order of a liquid crystal symmetric with a trace of zero.

    Read more in the user guide, under Elements.

    See Also:
        DomainKind: Names the kind of value of a state.

    Examples:
        >>> from skcomplex.core.enums import SymmetryKind
        >>> [kind.value for kind in SymmetryKind]
        ['full', 'symmetric', 'antisymmetric', 'traceless']
    """

    FULL = 'full'
    SYMMETRIC = 'symmetric'
    ANTISYMMETRIC = 'antisymmetric'
    TRACELESS = 'traceless'


class TopologyKind(StrEnum):
    """A kind of request that changes the links or nodes of a network, such as rewire.

    A law asks the network to change, such as `'add-edge'` to join two nodes or `'rewire'` to move one end of a link.
    Friends meet and drift off in a network, genes are copied in the network of a cell, and routers join and leave the
    internet.

    Read more in the user guide, under Laws.

    See Also:
        StructureTag: Names the way the structure of a system changes.

    Examples:
        >>> from skcomplex.core.enums import TopologyKind
        >>> TopologyKind('close-triangle').name
        'CLOSE_TRIANGLE'
    """

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
