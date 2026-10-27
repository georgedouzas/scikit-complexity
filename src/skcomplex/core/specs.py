"""Define the specs that configure parts and runs: states, parameters, roles, layers, schedules, noise, observables.

A part declares which arguments of its constructor are states, parameters, or roles by annotating them with
`typing.Annotated` and a `StateSpec`, a `ParameterSpec`, or a `RoleSpec`. The domain of a state or a parameter is a Sage
set: a ring such as `RR`, `ZZ`, or `CC`, an interval such as `RealSet.closed_open(0, oo)`, or a finite alphabet such as
`FiniteEnumeratedSet(['S', 'I', 'R'])`.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
import pickle  # nosec B403: pickles the model's own functions to check that they pickle, and never loads
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Self

import networkx as nx
import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import RR

from .errors import RewritingError, ValidationError

ACTIVATIONS = ('random', 'sequential', 'simultaneous', 'by-type')
MOVES = ('flip', 'random-value', 'rotate')
SAMPLERS = ('metropolis', 'glauber', 'heat-bath')
MONTE_CARLO_SCHEDULES = ('checkerboard', 'random-sequential', 'colored')
TOPOLOGY_KINDS = ('add-edge', 'remove-edge', 'rewire', 'add-node', 'remove-node')
EDGE_KINDS = ('add-edge', 'remove-edge', 'rewire')
ENDS = ('source', 'target')
TURTLE_SYMBOLS = {
    'F': 'draw',
    'f': 'move',
    '+': 'left',
    '-': 'right',
    '|': 'reverse',
    '[': 'push',
    ']': 'pop',
    '&': 'pitch_down',
    '^': 'pitch_up',
    '\\': 'roll_left',
    '/': 'roll_right',
}


@dataclass(frozen=True)
class StateSpec:
    """Spec of a state of an element, in the annotation of an argument of its constructor.

    The default of the argument is the initial value of the state. A state of order `k` has its `k`-th time derivative
    given by its rate laws, and a state with `m` lags keeps its past values for difference equations.

    Args:
        domain:
            The Sage set of the values of the state, such as `RR`, `ZZ`, `RealSet.closed_open(0, oo)`, or
            `FiniteEnumeratedSet(['S', 'I', 'R'])`, or `None` for a structure rewritten as a whole, such as a word,
            whose element checks it.

        vector:
            Whether the state has one component per dimension of the enclosing space.

        order:
            The order of the highest time derivative that laws give, an integer of at least 1, or a real number in
            `(0, 1)` for the Caputo derivative of a fractional order.

        lags:
            The number of past values the state keeps for difference equations, 0 for none.

        initial:
            The initial values of the derivatives of orders 1 to `order - 1`, or of the past values at steps -1 to
            `-lags`.

        period:
            The period that values of the state wrap around, such as `2 * pi` for an angle, or `None` for none.

        quantity:
            The conserved quantity the state holds, such as `'money'`, or `None`. Only transfers change a tagged state.

        history:
            The values of the state before the start, which delayed laws read, a number or a function of time, or
            `None` to keep the initial value before the start.

        position:
            Whether a vector state is the position of its element in a continuous space, on which the boundary of the
            space acts.

        velocity_of:
            The position state whose rate a vector state is, which a reflecting boundary reverses, or `None`.

    Examples:
        >>> from sage.all__sagemath_symbolics import oo
        >>> from sage.sets.real_set import RealSet
        >>> from skcomplex.core.specs import StateSpec
        >>> spec = StateSpec(domain=RealSet.closed_open(0, oo)).check()
        >>> 2.5 in spec.domain, -1 in spec.domain
        (True, False)
    """

    domain: object = RR
    vector: bool = False
    order: float = 1
    lags: int = 0
    initial: Sequence[object] = ()
    period: float | None = None
    quantity: str | None = None
    history: object = None
    position: bool = False
    velocity_of: str | None = None

    def check(self) -> Self:
        """Raise an error for an order that is not positive, negative lags, both, or too many initial values.

        Returns:
            spec:
                The spec itself.

        Raises:
            ValidationError:
                The order or the lags are invalid, or there are more initial values than derivatives or lags.
        """
        integral = float(self.order).is_integer()
        if not (math.isfinite(self.order) and self.order > 0 and (integral or self.order < 1)):
            error_msg = f'The order {self.order!r} is neither an integer of at least 1 nor a real number in (0, 1).'
            raise ValidationError(error_msg)
        if not (isinstance(self.lags, int) and self.lags >= 0):
            error_msg = f'The lags {self.lags!r} are not a nonnegative integer.'
            raise ValidationError(error_msg)
        if self.order != 1 and self.lags:
            error_msg = 'A state takes either an order other than 1 or lags, not both.'
            raise ValidationError(error_msg)
        companions = max(int(self.order) - 1 if integral else 0, self.lags)
        if len(self.initial) > companions:
            error_msg = f'{len(self.initial)} initial values were given for {companions} derivatives or lags.'
            raise ValidationError(error_msg)
        return self


@dataclass(frozen=True)
class ParameterSpec:
    """Spec of a parameter of an element or an interaction, in the annotation of an argument of its constructor.

    The default of the argument is the default of the parameter. A parameter per entry of a variadic role holds one
    value per entry of the role, and an adaptive parameter is a state that laws may change, such as a policy rate.

    Args:
        domain:
            The Sage set of the values of the parameter, such as `RR` or `RealSet.open(0, oo)`.

        vector:
            Whether the parameter has one component per dimension of the enclosing space.

        per:
            The name of a variadic role whose entries the parameter holds one value for, or `None`.

        adaptive:
            Whether laws may change the parameter.

        bounds:
            The lower and upper bound of an adaptive parameter that is a control input, or `None` for none.

    Examples:
        >>> from sage.all__sagemath_symbolics import oo
        >>> from sage.sets.real_set import RealSet
        >>> from skcomplex.core.specs import ParameterSpec
        >>> 0.5 in ParameterSpec(domain=RealSet.open(0, oo)).check().domain
        True
    """

    domain: object = RR
    vector: bool = False
    per: str | None = None
    adaptive: bool = False
    bounds: tuple[float, float] | None = None

    def check(self) -> Self:
        """Raise an error for a parameter per role that is a vector or adaptive, or for invalid bounds.

        Returns:
            spec:
                The spec itself.

        Raises:
            ValidationError:
                The parameter is per entry of a role and a vector or adaptive, or has bounds without being adaptive or
                with a lower bound above the upper one.
        """
        if self.per is not None and (self.vector or self.adaptive):
            error_msg = f'A parameter per entry of the role {self.per!r} is neither a vector nor adaptive.'
            raise ValidationError(error_msg)
        if self.bounds is not None and not (self.adaptive and float(self.bounds[0]) <= float(self.bounds[1])):
            error_msg = (
                f'The bounds {self.bounds!r} need an adaptive parameter and a lower bound at most the upper one.'
            )
            raise ValidationError(error_msg)
        return self


@dataclass(frozen=True)
class RoleSpec:
    """Spec of a role of an interaction, in the annotation of an argument of its constructor.

    A plain role binds one element path, one group path, or one group member reference `<group>[<index>]`. A variadic
    role binds a sequence of them, and a noise role binds a `WienerProcessSpec`.

    Args:
        variadic:
            Whether the role binds a sequence of entries instead of one path.

        noise:
            Whether the role binds a set of Wiener processes instead of a path.

    Examples:
        >>> from skcomplex.core.specs import RoleSpec
        >>> RoleSpec(variadic=True).check().variadic
        True
    """

    variadic: bool = False
    noise: bool = False

    def check(self) -> Self:
        """Raise an error for a role that is both variadic and a noise role.

        Returns:
            spec:
                The spec itself.

        Raises:
            ValidationError:
                The role is both variadic and a noise role.
        """
        if self.variadic and self.noise:
            error_msg = 'A role is either variadic or a noise role, not both.'
            raise ValidationError(error_msg)
        return self


@dataclass(frozen=True)
class RequirementSpec:
    """What a part requires of the system it is part of: the classes of time and of space it runs in.

    An empty field accepts any class.

    Args:
        time:
            The time classes the part runs in, such as `(ContinuousTime,)`.

        space:
            The space classes the part runs in, such as `(LatticeSpace,)`.

    Examples:
        >>> from skcomplex.core.spaces import ContinuousTime
        >>> from skcomplex.core.specs import RequirementSpec
        >>> RequirementSpec(time=(ContinuousTime,)).time
        (<class 'skcomplex.core.spaces._continuous_time.ContinuousTime'>,)
    """

    time: tuple[type, ...] = ()
    space: tuple[type, ...] = ()


@dataclass(frozen=True)
class VisualSpec:
    """Name of a registered visual and the options it is built with.

    Args:
        kind:
            The name of a visual registered in the visualization layer.

        options:
            The keyword arguments of the visual.

    Examples:
        >>> from skcomplex.core.specs import VisualSpec
        >>> VisualSpec('points', {'position': 'x'}).kind
        'points'
    """

    kind: str = 'scalars'
    options: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class LayerSpec:
    """Spec of a relation over the nodes of a network space, such as contacts or employment, held as a graph of its own.

    A network space holds several layers over the same nodes. Edge groups and interactions over neighbours name the
    layer they bind to, and reductions reduce over its edges with `AdjacencyNeighborhood(layer=label)`.

    Args:
        label:
            The name of the layer, unique in its network space.

        graph:
            The NetworkX graph of the relation, over the nodes of the network space.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.core.specs import LayerSpec
        >>> LayerSpec('work', nx.path_graph(3)).check().label
        'work'
    """

    label: str
    graph: nx.Graph

    def check(self) -> Self:
        """Raise an error for a label that is not a string.

        Returns:
            spec:
                The spec itself.

        Raises:
            ValidationError:
                The label is not a string.
        """
        if not isinstance(self.label, str):
            error_msg = f'The label of a layer must be a string, but it is {self.label!r}.'
            raise ValidationError(error_msg)
        return self


class ScheduleSpec:
    """Base of the specs of schedules of discrete time that a formalism defines, such as the phases of agents.

    A subclass is a frozen dataclass of its own fields that overrides `check`.
    """

    def check(self) -> Self:
        """Return the schedule, which has nothing to check by default.

        Returns:
            spec:
                The spec itself.
        """
        return self


@dataclass(frozen=True)
class WienerProcessSpec:
    """Spec of a set of Wiener processes whose increments over a step `h` have the covariance `h C`.

    An interaction binds the processes through a role declared with `RoleSpec(noise=True)`, and its noise laws name
    them through the handle `namespace.<role>.dW`, a symbol for one process and a vector of symbols for several, so
    `namespace.<role>.dW[k]` names process `k`. Interactions of one system that bind processes of one label share them,
    so laws that name one process share its increment.

    Args:
        label:
            The label of the processes, unique among the parts of the system of the interactions that bind them.

        dimension:
            The number of processes.

        covariance:
            The constant covariance matrix `C` of the processes, symmetric and positive semidefinite of size
            `dimension`, or `None` for the identity.

    Examples:
        >>> from skcomplex.core.specs import WienerProcessSpec
        >>> WienerProcessSpec('market', dimension=2, covariance=((1.0, 0.5), (0.5, 1.0))).check().dimension
        2
    """

    label: str
    dimension: int = 1
    covariance: object = None

    def check(self) -> Self:
        """Raise an error for a label that is not a string or a dimension below 1.

        Returns:
            spec:
                The spec itself.

        Raises:
            ValidationError:
                The label is not a string, or the dimension is not a positive integer.
        """
        if not isinstance(self.label, str):
            error_msg = f'The label of Wiener processes must be a string, but it is {self.label!r}.'
            raise ValidationError(error_msg)
        if not (isinstance(self.dimension, int) and self.dimension >= 1):
            error_msg = f'Wiener processes have a positive whole dimension, but it is {self.dimension!r}.'
            raise ValidationError(error_msg)
        return self


@dataclass(frozen=True)
class PhaseSpec:
    """Named part of a step, in which agents run the rules of the phase under an activation policy.

    The policies are `'random'`, every agent in a new random order, `'sequential'`, every agent in the order of
    creation, `'simultaneous'`, every agent in random order reading the states of the start of the phase while its
    writes apply at the end of the phase, and `'by-type'`, group after group in the order of `groups` or of
    declaration, each group in random order. The rules of interactions and institutions in the phase run once after
    the agents. A phase with `repeat` above one runs that many rounds, such as intraday rounds within a day.

    Args:
        name:
            The name of the phase, which rules name in `RuleLaw(function, phase=name)`.

        activation:
            The activation policy.

        repeat:
            The number of rounds of the phase per step.

        groups:
            The paths of the agent groups in their order of activation for `'by-type'`, or `None` for the order of
            declaration.

    Examples:
        >>> from skcomplex.core.specs import PhaseSpec
        >>> PhaseSpec('trade', activation='simultaneous', repeat=3)
        PhaseSpec(name='trade', activation='simultaneous', repeat=3, groups=None)
    """

    name: str
    activation: str = 'random'
    repeat: int = 1
    groups: Sequence[str] | None = None


@dataclass(frozen=True)
class PhasesScheduleSpec(ScheduleSpec):
    """Spec of a schedule of discrete time that runs named phases in order in every step.

    Args:
        phases:
            The phases, each with its rules and activation policy.

    Examples:
        >>> from skcomplex.core.specs import PhaseSpec, PhasesScheduleSpec
        >>> [phase.name for phase in PhasesScheduleSpec((PhaseSpec('produce'), PhaseSpec('trade'))).check().phases]
        ['produce', 'trade']
    """

    phases: Sequence[PhaseSpec]

    def check(self) -> Self:
        """Raise an error for a phase with an unknown activation or no round, or for a repeated name.

        Returns:
            spec:
                The spec itself.

        Raises:
            ValidationError:
                A phase repeats the name of another, has an unknown activation, or runs no round.
        """
        names = [phase.name for phase in self.phases]
        if len(names) != len(set(names)):
            error_msg = f'The phases {names} repeat a name. Give every phase of a step its own name.'
            raise ValidationError(error_msg)
        for phase in self.phases:
            if phase.activation not in ACTIVATIONS:
                error_msg = (
                    f'Unknown activation {phase.activation!r} of phase {phase.name!r}. Use one of {ACTIVATIONS}.'
                )
                raise ValidationError(error_msg)
            if phase.repeat < 1:
                error_msg = f'Phase {phase.name!r} repeats {phase.repeat} times. Use at least one round.'
                raise ValidationError(error_msg)
        return self


@dataclass(frozen=True)
class ObservableSpec:
    """Spec of a named value of a system that a run computes from its states and parameters at every frame.

    An observable is an expression of the namespace of the system, of one element when `element` is given, or a
    reduction over the members of a group of an expression or a variable of each member. A run records it in its
    trajectory under `<system path>/<label>`, or `<element>.<label>` for an element, and laws of other partitions read
    it through a role bound to its path.

    Args:
        label:
            The label of the observable, unique among the observables of its system.

        expression:
            A function of the namespace of the system, of the element, or of one member, or a symbolic expression.

        element:
            The label of an element or an interaction of the root system whose namespace the expression reads, or
            `None`.

        group:
            The path of the group to reduce over, relative to the system, or `None` for no reduction.

        reduce:
            `'count'`, `'sum'`, `'mean'`, `'min'`, `'max'`, `'var'`, or a function of the array of member values.

        variable:
            The name of the member variable to reduce, instead of an expression.

    Examples:
        >>> from skcomplex.core.specs import ObservableSpec
        >>> ObservableSpec('mean_wealth', group='traders', variable='wealth').check().reduce
        'mean'
    """

    label: str
    expression: object = None
    element: str | None = None
    group: str | None = None
    reduce: str | Callable[[object], object] = 'mean'
    variable: str | None = None

    def check(self) -> Self:
        """Raise an error for an expression or a function reduction that cannot be pickled, such as a lambda.

        Returns:
            spec:
                The spec itself.

        Raises:
            ValidationError:
                The expression or the reduction cannot be pickled.
        """
        for name, value in (('expression', self.expression), ('reduction', self.reduce)):
            try:
                pickle.dumps(value)
            except (pickle.PicklingError, AttributeError, TypeError) as error:
                error_msg = (
                    f'The {name} of the observable {self.label!r} cannot be pickled: {error}. Use a module-level '
                    'function or a symbolic expression instead of a lambda or a local function.'
                )
                raise ValidationError(error_msg) from error
        return self


@dataclass(frozen=True)
class EnergySpec:
    """Spec of the local energy of a lattice site, every term of the total energy that involves the value of the site.

    The energy change of a move is the difference of the local energy at the proposed and the current value. Neighbour
    values are read with `neighbors(..., reduce='each')` in neighbourhood order, or summed with
    `neighbors(..., reduce='sum')`.

    Args:
        expression:
            The local energy, a symbolic expression of the states of the site, its neighbours, and parameters.

        over:
            The neighbourhood that `neighbors` reads in the expression.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.spaces import VonNeumannNeighborhood
        >>> from skcomplex.core.specs import EnergySpec
        >>> spin, J = var('spin J')
        >>> EnergySpec(-J * spin, VonNeumannNeighborhood()).check().expression
        -J*spin
    """

    expression: object
    over: object

    def check(self) -> Self:
        """Return the spec, which has nothing to check.

        Returns:
            spec:
                The spec itself.
        """
        return self


@dataclass(frozen=True)
class MoveSpec:
    """Spec of the proposed change of the value of a lattice site in a Monte Carlo sweep.

    Args:
        target:
            The symbol of the lattice state the move changes.

        kind:
            `'flip'` for the negative value, the spin flip of Ising models, `'random-value'` for a value drawn from
            `values`, as in Potts models, or `'rotate'` for the value plus a uniform number in `[-width, width]`, as in
            XY models.

        values:
            The values a random-value move draws from.

        width:
            The largest change of a rotation.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.specs import MoveSpec
        >>> MoveSpec(var('color'), 'random-value', values=(0.0, 1.0, 2.0)).check().values
        (0.0, 1.0, 2.0)
    """

    target: object
    kind: str = 'flip'
    values: tuple[float, ...] = ()
    width: float = 0.0

    def check(self) -> Self:
        """Raise an error for an unknown kind of move.

        Returns:
            spec:
                The spec itself.

        Raises:
            ValidationError:
                The kind of move is unknown.
        """
        if self.kind not in MOVES:
            error_msg = f'Unknown move {self.kind!r}. Use one of {", ".join(MOVES)}.'
            raise ValidationError(error_msg)
        return self


@dataclass(frozen=True)
class SamplerSpec:
    """Spec of the acceptance of Monte Carlo moves at a temperature, with one sweep of the lattice per unit of time.

    Args:
        kind:
            `'metropolis'` to accept with probability `min(1, exp(-dE / T))`, `'glauber'` with `1 / (1 + exp(dE / T))`,
            or `'heat-bath'` to draw the new value among all values of a discrete move with Boltzmann weights.

        temperature:
            The temperature, usually a parameter symbol so that it changes live.

        schedule:
            `'checkerboard'` to update the two sublattices of a square lattice in turn, `'colored'` to update the
            classes of a proper coloring of the neighbour graph in turn, or `'random-sequential'` to visit the sites one
            at a time in a random order.

    Examples:
        >>> from skcomplex.core.specs import SamplerSpec
        >>> SamplerSpec('glauber', 2.0).check().schedule
        'checkerboard'
    """

    kind: str = 'metropolis'
    temperature: object = 1.0
    schedule: str = 'checkerboard'

    def check(self) -> Self:
        """Raise an error for an unknown kind of sampler or schedule.

        Returns:
            spec:
                The spec itself.

        Raises:
            ValidationError:
                The kind of sampler or the schedule is unknown.
        """
        if self.kind not in SAMPLERS:
            error_msg = f'Unknown sampler {self.kind!r}. Use one of {", ".join(SAMPLERS)}.'
            raise ValidationError(error_msg)
        if self.schedule not in MONTE_CARLO_SCHEDULES:
            error_msg = (
                f'Unknown Monte Carlo schedule {self.schedule!r}. Use one of {", ".join(MONTE_CARLO_SCHEDULES)}.'
            )
            raise ValidationError(error_msg)
        return self


@dataclass(frozen=True)
class HilbertSpec:
    """Spec of the Hilbert space of a quantum element, with the local operators its laws build Hamiltonians from.

    An element whose state is annotated with a `StateSpec` over `CC^dimension` and that takes a `HilbertSpec` is a
    quantum subsystem. Its laws read every local operator by name, and a group of such elements is the tensor product
    of its members' spaces, in member order.

    Args:
        dimension:
            The dimension of the space.

        operators:
            A mapping from operator name to its matrix and whether it is Hermitian, such as `{'z': (matrix, True)}`.

        basis:
            A mapping from the label of a basis state to its index, such as `{'up': 0, 'down': 1}`.

        display:
            The name of the operator whose expectation visuals show for a group of such elements, or `None` for the
            probabilities of the basis states.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.specs import HilbertSpec
        >>> spec = HilbertSpec(2, {'z': (np.diag([1.0, -1.0]), True)}, basis={'up': 0, 'down': 1}).check()
        >>> spec.basis['down']
        1
    """

    dimension: int
    operators: Mapping[str, tuple[npt.ArrayLike, bool]] = field(default_factory=dict)
    basis: Mapping[str, int] = field(default_factory=dict)
    display: str | None = 'z'

    def check(self) -> Self:
        """Raise an error for a dimension, an operator, a basis state, or a display that does not fit the space.

        Returns:
            spec:
                The spec itself.

        Raises:
            ValidationError:
                The dimension is not a positive integer, an operator is not a square matrix of the dimension, a basis
                index is out of range, or the display operator is not one of the operators.
        """
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int) or self.dimension < 1:
            error_msg = f'The dimension of a Hilbert space must be a positive integer, but it is {self.dimension!r}.'
            raise ValidationError(error_msg)
        for name, (matrix, _) in self.operators.items():
            if np.shape(matrix) != (self.dimension, self.dimension):
                error_msg = (
                    f'The operator {name!r} of a Hilbert space of dimension {self.dimension} has the shape '
                    f'{np.shape(matrix)}. Give a square matrix of the dimension.'
                )
                raise ValidationError(error_msg)
        for label, index in self.basis.items():
            if not 0 <= index < self.dimension:
                error_msg = f'The basis state {label!r} has the index {index}, outside the dimension {self.dimension}.'
                raise ValidationError(error_msg)
        if self.display is not None and self.operators and self.display not in self.operators:
            error_msg = f'The display operator {self.display!r} is not one of {", ".join(self.operators)}.'
            raise ValidationError(error_msg)
        return self


@dataclass(frozen=True)
class ModuleSpec:
    """Spec of a symbol of a word with numeric or symbolic parameters, such as `A(x)` or `F(1.0)`.

    Args:
        name:
            The name of the symbol.

        parameters:
            The parameters, numbers in a word and symbols or expressions in a production.

    Examples:
        >>> from skcomplex.core.specs import ModuleSpec
        >>> ModuleSpec('F', (0.5,)).check().parameters
        (0.5,)
    """

    name: str
    parameters: tuple[object, ...] = ()

    def check(self) -> Self:
        """Raise an error for a name that is not a nonempty string.

        Returns:
            spec:
                The spec itself.

        Raises:
            RewritingError:
                The name is not a nonempty string.
        """
        if not isinstance(self.name, str) or not self.name:
            error_msg = f'The name of a module must be a nonempty string, but it is {self.name!r}.'
            raise RewritingError(error_msg)
        return self


@dataclass(frozen=True)
class PrototileSpec:
    """Spec of a shape that tiles of a patch are placed copies of, a segment or a polygon in its own frame.

    Args:
        name:
            The name of the prototile, which names its count state.

        vertices:
            The vertices in the plane, two for a segment and at least three for a polygon.

    Examples:
        >>> from skcomplex.core.specs import PrototileSpec
        >>> len(PrototileSpec('interval', [(0, 0), (1, 0)]).check().vertices)
        2
    """

    name: str
    vertices: Sequence[Sequence[float]]

    def check(self) -> Self:
        """Raise an error for vertices that are not at least two points of the plane.

        Returns:
            spec:
                The spec itself.

        Raises:
            RewritingError:
                The vertices are not points of the plane, or fewer than two.
        """
        points = np.asarray(self.vertices, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:  # noqa: PLR2004
            error_msg = f'Prototile {self.name!r} needs at least two vertices in the plane, but got {self.vertices!r}.'
            raise RewritingError(error_msg)
        return self


@dataclass(frozen=True)
class AffineSpec:
    """Spec of an affine map of the plane: a reflection across the x axis, a rotation, a scale, and a translation.

    The scale, the rotation, and the translation can be expressions of the parameters of the element whose laws hold
    the substitution, which are evaluated at every generation.

    Args:
        scale:
            The scale factor.

        rotation:
            The rotation in degrees.

        translation:
            The translation.

        reflect:
            Whether to reflect across the x axis first.

        matrix:
            A homogeneous 3 by 3 matrix that replaces the other arguments.

    Examples:
        >>> from skcomplex.core.specs import AffineSpec
        >>> AffineSpec(scale=0.5, translation=(1, 0)).check().scale
        0.5
    """

    scale: object = 1.0
    rotation: object = 0.0
    translation: Sequence[object] = (0.0, 0.0)
    reflect: bool = False
    matrix: Sequence[Sequence[float]] | None = None

    def check(self) -> Self:
        """Raise an error for a translation that is not a point of the plane, or a matrix that is not 3 by 3.

        Returns:
            spec:
                The spec itself.

        Raises:
            RewritingError:
                The translation or the matrix has the wrong shape.
        """
        if len(self.translation) != 2:  # noqa: PLR2004
            error_msg = f'The translation of an affine map is a point of the plane, not {self.translation!r}.'
            raise RewritingError(error_msg)
        if self.matrix is not None and np.shape(self.matrix) != (3, 3):
            error_msg = (
                f'The matrix of an affine map is homogeneous 3 by 3, but it has the shape {np.shape(self.matrix)}.'
            )
            raise RewritingError(error_msg)
        return self


@dataclass(frozen=True)
class GeometrySpec:
    r"""Spec of the geometric interpretation of a word as line segments in two or three dimensions.

    A cursor walks the word: symbols mean `draw` (forward with a segment), `move` (forward without one), `left` and
    `right` (turn by the angle), `reverse` (turn by 180 degrees), `push` and `pop` (save and restore the position and
    heading), and in three dimensions `pitch_down`, `pitch_up`, `roll_left`, and `roll_right`. This is the turtle
    interpretation of Prusinkiewicz and Lindenmayer (1990), whose symbols are the default meanings: `F`, `f`, `+`, `-`,
    `|`, `[`, `]`, `&`, `^`, `\`, and `/`. A module's first parameter overrides the step of a move or the angle of a
    turn.

    Args:
        angle:
            The turning angle in degrees.

        step:
            The step at generation 0.

        contraction:
            The factor the step is multiplied by at every generation.

        symbols:
            Meanings of symbols that replace or extend the default ones of `TURTLE_SYMBOLS`.

        dimension:
            2 or 3.

        heading:
            The initial heading in degrees in two dimensions, measured from the x axis.

    Examples:
        >>> from skcomplex.core.specs import GeometrySpec
        >>> GeometrySpec(angle=60.0, contraction=1 / 3).check().angle
        60.0
    """

    angle: float = 90.0
    step: float = 1.0
    contraction: float = 1.0
    symbols: Mapping[str, str] | None = None
    dimension: int = 2
    heading: float = 0.0

    def check(self) -> Self:
        """Raise an error for a meaning that is not an action, or a dimension that is neither 2 nor 3.

        Returns:
            spec:
                The spec itself.

        Raises:
            RewritingError:
                A meaning is not an action of the geometry, or the dimension is neither 2 nor 3.
        """
        actions = set(TURTLE_SYMBOLS.values())
        unknown = sorted(set((self.symbols or {}).values()) - actions)
        if unknown:
            error_msg = f'The word geometry has no action {unknown[0]!r}. Use one of {", ".join(sorted(actions))}.'
            raise RewritingError(error_msg)
        if self.dimension not in {2, 3}:
            error_msg = f'The word geometry has dimension {self.dimension}. Use 2 or 3.'
            raise RewritingError(error_msg)
        return self


@dataclass(frozen=True, eq=False)
class TopologyRequestSpec:
    """Spec of a change of the topology of a network that a law requests for every member or edge it runs for.

    A request is the target of an `UpdateLaw`, whose value is the probability that it fires at every step for the
    member or edge the law runs for, or of a `JumpLaw`, as `JumpLaw(rate, {request: 1})`, which fires it at the rate.
    The stepper applies the requests that fire after the step, in the order of the members and edges. A request on
    edges, `'add-edge'`, `'remove-edge'`, or `'rewire'`, adds or removes members of the variable group on the edges of
    its layer, and a request on nodes, `'add-node'` or `'remove-node'`, adds or removes members of every variable group
    on the nodes of the network. New ends are drawn among the nodes present that are not yet joined to the node, with
    probability proportional to `toward`.

    Args:
        kind:
            `'add-edge'`, `'remove-edge'`, `'rewire'`, `'add-node'`, or `'remove-node'`.

        edges:
            The number of edges a new node or a node adding edges joins to other nodes.

        toward:
            A function of the namespace of a candidate node that returns its weight for a new end, such as
            `lambda node: neighbors(1)` for the degree of preferential attachment, or `None` for equal weights.

        keep:
            The end of an edge that a rewiring keeps, `'source'` or `'target'`.

        layer:
            The layer whose edges the request changes, or `None` for the graph of the network.

        values:
            The initial states of a new node or edge as pairs of a state name and a value, which default to the
            defaults of the element type.

    Examples:
        >>> from skcomplex.core.laws import UpdateLaw
        >>> from skcomplex.core.specs import TopologyRequestSpec
        >>> UpdateLaw(TopologyRequestSpec('rewire', keep='target').check(), 0.1).check().target.keep
        'target'
    """

    kind: str
    edges: int = 1
    toward: Callable[[object], object] | None = None
    keep: str = 'source'
    layer: str | None = None
    values: tuple[tuple[str, object], ...] = ()

    def check(self) -> Self:
        """Raise an error for an unknown kind or kept end, or a number of edges that is not a whole number.

        Returns:
            spec:
                The spec itself.

        Raises:
            ValidationError:
                The kind or the kept end is unknown, or the number of edges is negative or not whole.
        """
        if self.kind not in TOPOLOGY_KINDS:
            error_msg = f'Unknown topology request {self.kind!r}. Use one of {", ".join(TOPOLOGY_KINDS)}.'
            raise ValidationError(error_msg)
        if self.keep not in ENDS:
            error_msg = f'A rewiring keeps the source or the target of its edge, not {self.keep!r}.'
            raise ValidationError(error_msg)
        if int(self.edges) != self.edges or self.edges < 0:
            error_msg = f'A topology request adds a whole number of edges of at least 0, not {self.edges!r}.'
            raise ValidationError(error_msg)
        return self
