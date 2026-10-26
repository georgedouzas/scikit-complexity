"""Define the shared types: terms, domains of values, declarations of variables, symbols, and visuals."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import copy
import dataclasses
import inspect
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Any, ClassVar, Self, cast

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR
from sage.all__sagemath_symbolics import vector as sage_vector
from sage.modules.free_module_element import FreeModuleElement as SymbolicVector
from sage.symbolic.expression import Expression

from ._errors import ArgumentError, LabelError, RegistryError

TERMS: dict[type, dict[str, Any]] = {}


def vocabulary() -> dict[str, tuple['StrEnum | Term', ...]]:
    """Return the terms of every category of the system vocabulary, with the registered space and law kinds.

    Returns:
        vocabulary:
            A mapping from category, such as `'laws'`, to its terms.

    Examples:
        >>> from skcomplex.core.base import vocabulary
        >>> [str(term) for term in vocabulary()['time']]
        ['continuous', 'discrete', 'static']
    """
    return {
        'time': tuple(TimeKind),
        'space': SpaceKind.registered(),
        'components': tuple(ComponentKind),
        'bindings': tuple(Binding),
        'laws': LawKind.registered(),
        'domains': tuple(DomainKind),
    }


class Term(str):
    """Named value of an open vocabulary, a string compared by its value.

    Each subclass is a vocabulary of its own. An extension adds a term with `register`, which keeps the name of every
    value, and a term prints as its value.

    Examples:
        >>> from skcomplex.core.base import Form
        >>> Form.ODE == Form.register('ODE', 'ordinary differential equations')
        True
        >>> Form.ODE.name, Form.ODE.value
        ('ODE', 'ordinary differential equations')
    """

    @classmethod
    def register(cls, name: str, value: str) -> Self:
        """Return the term of a value, registered under a name in the vocabulary.

        Args:
            name:
                The name of the term in upper case, such as `'ODE'`.

            value:
                The text of the term, such as `'ordinary differential equations'`.

        Returns:
            term:
                The registered term.

        Raises:
            RegistryError:
                The name is registered in the vocabulary with another value.
        """
        members: dict[str, Self] = TERMS.setdefault(cls, {})
        registered = members.get(name)
        if registered is not None and registered != value:
            error_msg = (
                f'{cls.__name__} {name} is registered with the value {str(registered)!r}, not {value!r}. Give the '
                'registered value, or choose another name.'
            )
            raise RegistryError(error_msg)
        if registered is None:
            members[name] = cls(value)
        return members[name]

    @classmethod
    def registered(cls) -> tuple[Self, ...]:
        """Return the terms of the vocabulary in registration order.

        Returns:
            terms:
                The registered terms.

        Examples:
            >>> from skcomplex.core.base import SpaceKind
            >>> SpaceKind.LATTICE in SpaceKind.registered()
            True
        """
        return tuple(cast('dict[str, Self]', TERMS.get(cls, {})).values())

    @property
    def name(self) -> str:
        """Name of the term in its vocabulary, or the value itself for a term that was never registered."""
        names = {str(term): name for name, term in TERMS.get(type(self), {}).items()}
        return names.get(str(self), str(self))

    @property
    def value(self) -> str:
        """Text of the term."""
        return str(self)


class Form(Term):
    """Mathematical form of a compiled system, a term of an open vocabulary that extensions add forms to.

    Args:
        name:
            The name of the form in upper case.

        value:
            The description of the form.

    Examples:
        >>> from skcomplex.core.base import Form
        >>> Form.SDE.value
        'stochastic differential equations'
    """

    ODE: ClassVar['Form']
    FRACTIONAL: ClassVar['Form']
    SDE: ClassVar['Form']
    JUMP: ClassVar['Form']
    DISCRETE: ClassVar['Form']
    AGENT: ClassVar['Form']
    LATTICE: ClassVar['Form']
    MONTE_CARLO: ClassVar['Form']
    REWRITING: ClassVar['Form']
    RENEWAL: ClassVar['Form']
    COUPLED: ClassVar['Form']
    QUANTUM: ClassVar['Form']


Form.ODE = Form.register('ODE', 'ordinary differential equations')
Form.FRACTIONAL = Form.register('FRACTIONAL', 'fractional differential equations')
Form.SDE = Form.register('SDE', 'stochastic differential equations')
Form.JUMP = Form.register('JUMP', 'jump process')
Form.DISCRETE = Form.register('DISCRETE', 'discrete map')
Form.AGENT = Form.register('AGENT', 'agent-based model')
Form.LATTICE = Form.register('LATTICE', 'lattice dynamics')
Form.MONTE_CARLO = Form.register('MONTE_CARLO', 'Monte Carlo sampling')
Form.REWRITING = Form.register('REWRITING', 'rewriting system')
Form.RENEWAL = Form.register('RENEWAL', 'renewal process')
Form.COUPLED = Form.register('COUPLED', 'co-simulation')
Form.QUANTUM = Form.register('QUANTUM', 'quantum dynamics')


class TimeKind(StrEnum):
    """Kind of time a system evolves in."""

    CONTINUOUS = 'continuous'
    DISCRETE = 'discrete'
    STATIC = 'static'


class ComponentKind(StrEnum):
    """Kind of component that carries state."""

    ELEMENT = 'element'
    GROUP = 'group'
    VARIABLE_GROUP = 'variable-group'
    SITE_GROUP = 'site-group'
    EDGE_GROUP = 'edge-group'
    HYPEREDGE_GROUP = 'hyperedge-group'
    SYSTEM = 'system'


class Binding(StrEnum):
    """How an interaction binds the members it runs its laws over."""

    MEMBERS = 'members'
    PAIRS = 'pairs'
    NEIGHBORS = 'neighbors'
    HYPEREDGES = 'hyperedges'


class LawClass(StrEnum):
    """Time semantics of a law, which decides the time it is valid under."""

    CONTINUOUS = 'continuous'
    EVENT = 'event'
    DISCRETE = 'discrete'
    CONSTRAINT = 'constraint'
    ANY = 'any'


class SpaceKind(Term):
    """Kind of substrate the components of a system live on, a term of an open vocabulary.

    Args:
        name:
            The name of the kind in upper case.

        value:
            The text of the kind.

    Examples:
        >>> from skcomplex.core.base import SpaceKind
        >>> str(SpaceKind.LATTICE)
        'lattice'
    """

    NONE: ClassVar['SpaceKind']
    LATTICE: ClassVar['SpaceKind']
    NETWORK: ClassVar['SpaceKind']
    CONTINUOUS: ClassVar['SpaceKind']
    MESH: ClassVar['SpaceKind']


class LawKind(Term):
    """Kind of law, a term of an open vocabulary.

    Args:
        name:
            The name of the kind in upper case.

        value:
            The text of the kind.

    Examples:
        >>> from skcomplex.core.base import LawKind
        >>> str(LawKind.LOCAL_RULE)
        'local-rule'
    """

    RATE: ClassVar['LawKind']
    NOISE: ClassVar['LawKind']
    EQUATION: ClassVar['LawKind']
    GENERATOR: ClassVar['LawKind']
    DISSIPATOR: ClassVar['LawKind']
    JUMP: ClassVar['LawKind']
    EVENT: ClassVar['LawKind']
    UPDATE: ClassVar['LawKind']
    TRANSITION: ClassVar['LawKind']
    RULE: ClassVar['LawKind']
    LOCAL_RULE: ClassVar['LawKind']
    REWRITE: ClassVar['LawKind']
    SAMPLE: ClassVar['LawKind']
    MATCH: ClassVar['LawKind']
    TRANSFER: ClassVar['LawKind']
    BOUNDARY: ClassVar['LawKind']
    INVARIANT: ClassVar['LawKind']
    OBJECTIVE: ClassVar['LawKind']
    ENCOUNTER: ClassVar['LawKind']


class Trait(Term):
    """Fact about a law class that the compiler and validation read, a term of an open vocabulary.

    Args:
        name:
            The name of the trait in upper case.

        value:
            The text of the trait.

    Examples:
        >>> from skcomplex.core.base import Trait
        >>> str(Trait.FORMLESS)
        'formless'
    """

    FORMLESS: ClassVar['Trait']
    KERNEL_FREE: ClassVar['Trait']
    LATTICE: ClassVar['Trait']
    DECLARABLE: ClassVar['Trait']
    AGENT: ClassVar['Trait']


SpaceKind.NONE = SpaceKind.register('NONE', 'none')
SpaceKind.LATTICE = SpaceKind.register('LATTICE', 'lattice')
SpaceKind.NETWORK = SpaceKind.register('NETWORK', 'network')
SpaceKind.CONTINUOUS = SpaceKind.register('CONTINUOUS', 'continuous')
SpaceKind.MESH = SpaceKind.register('MESH', 'mesh')
LawKind.RATE = LawKind.register('RATE', 'rate')
LawKind.NOISE = LawKind.register('NOISE', 'noise')
LawKind.EQUATION = LawKind.register('EQUATION', 'equation')
LawKind.GENERATOR = LawKind.register('GENERATOR', 'generator')
LawKind.DISSIPATOR = LawKind.register('DISSIPATOR', 'dissipator')
LawKind.JUMP = LawKind.register('JUMP', 'jump')
LawKind.EVENT = LawKind.register('EVENT', 'event')
LawKind.UPDATE = LawKind.register('UPDATE', 'update')
LawKind.TRANSITION = LawKind.register('TRANSITION', 'transition')
LawKind.RULE = LawKind.register('RULE', 'rule')
LawKind.LOCAL_RULE = LawKind.register('LOCAL_RULE', 'local-rule')
LawKind.REWRITE = LawKind.register('REWRITE', 'rewrite')
LawKind.SAMPLE = LawKind.register('SAMPLE', 'sample')
LawKind.MATCH = LawKind.register('MATCH', 'match')
LawKind.TRANSFER = LawKind.register('TRANSFER', 'transfer')
LawKind.BOUNDARY = LawKind.register('BOUNDARY', 'boundary')
LawKind.INVARIANT = LawKind.register('INVARIANT', 'invariant')
LawKind.OBJECTIVE = LawKind.register('OBJECTIVE', 'objective')
LawKind.ENCOUNTER = LawKind.register('ENCOUNTER', 'encounter')
Trait.FORMLESS = Trait.register('FORMLESS', 'formless')
Trait.KERNEL_FREE = Trait.register('KERNEL_FREE', 'kernel-free')
Trait.LATTICE = Trait.register('LATTICE', 'lattice')
Trait.DECLARABLE = Trait.register('DECLARABLE', 'declarable')
Trait.AGENT = Trait.register('AGENT', 'agent')


class OrderKind(StrEnum):
    """Kind of order of the time derivative that the rate laws on a state give."""

    INTEGER = 'integer'
    FRACTIONAL = 'fractional'


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


class NoiseKind(StrEnum):
    """Structure of the Wiener processes that the noise laws of a system draw."""

    DIAGONAL = 'diagonal'
    SHARED = 'shared'
    CORRELATED = 'correlated'
    CONSERVED = 'conserved'


NUMERIC_DOMAINS = frozenset(
    {DomainKind.REAL, DomainKind.INTEGER, DomainKind.BOOLEAN, DomainKind.ANGLE, DomainKind.COMPLEX, DomainKind.VECTOR},
)


@dataclass(frozen=True)
class Requires:
    """What a primitive needs from the system it is part of, each field the set of acceptable terms.

    An empty field accepts any term. Validation compares the requirements of every primitive of a system with the
    system's time, space, and state domains, and with the support of the chosen solver and backend.

    Args:
        time:
            The time kinds the primitive is valid under.

        space:
            The space kinds the primitive is valid on.

        domains:
            The state domains the primitive may target.

        bindings:
            The interaction bindings the primitive is valid in.

    Examples:
        >>> from skcomplex.core.base import Requires, TimeKind
        >>> Requires(time={TimeKind.CONTINUOUS}).accepts('time', TimeKind.DISCRETE)
        False
    """

    time: AbstractSet[TimeKind] = field(default_factory=frozenset)
    space: AbstractSet[SpaceKind] = field(default_factory=frozenset)
    domains: AbstractSet[DomainKind] = field(default_factory=frozenset)
    bindings: AbstractSet[Binding] = field(default_factory=frozenset)

    def accepts(self, category: str, term: StrEnum | Term) -> bool:
        """Check whether the requirement of a category accepts a term.

        Args:
            category:
                The name of the field, such as `'time'`.

            term:
                The term of the system, such as `TimeKind.DISCRETE`.

        Returns:
            accepted:
                Whether the field is empty or holds the term.
        """
        terms: AbstractSet[StrEnum | Term] = set(getattr(self, category))
        return not terms or term in terms


@dataclass(frozen=True)
class Supports:
    """What a solver or backend can run, each field the set of supported terms.

    An empty field supports nothing of its category, so a declaration states its support explicitly.

    Args:
        time:
            The time kinds it advances.

        space:
            The space kinds it runs on.

        laws:
            The law kinds it runs.

        domains:
            The state domains it holds.

        components:
            The component kinds it holds.

        noise:
            The structures of Wiener processes it draws: a process per state, processes shared by several states,
            and correlated processes.
        orders:
            The kinds of order of the time derivatives of states it integrates, integer or fractional.

    Examples:
        >>> from skcomplex.core.base import LawKind, NoiseKind, Supports
        >>> Supports(laws={LawKind.RATE}).missing('laws', {LawKind.RATE, LawKind.NOISE})
        ['noise']
        >>> Supports(noise={NoiseKind.DIAGONAL}).missing('noise', {NoiseKind.SHARED})
        ['shared']
    """

    time: AbstractSet[TimeKind] = field(default_factory=frozenset)
    space: AbstractSet[SpaceKind] = field(default_factory=frozenset)
    laws: AbstractSet[LawKind] = field(default_factory=frozenset)
    domains: AbstractSet[DomainKind] = field(default_factory=frozenset)
    components: AbstractSet[ComponentKind] = field(default_factory=frozenset)
    noise: AbstractSet[NoiseKind] = field(default_factory=frozenset)
    orders: AbstractSet[OrderKind] = field(default_factory=frozenset)

    def missing(self, category: str, terms: Iterable[StrEnum | Term]) -> list[str]:
        """Return the terms of a category that the declaration does not support.

        Args:
            category:
                The name of the field, such as `'laws'`.

            terms:
                The terms a system needs.

        Returns:
            missing:
                The unsupported terms, sorted.
        """
        supported: AbstractSet[StrEnum | Term] = set(getattr(self, category))
        return sorted(str(term) for term in set(terms) - supported)


type Vector = npt.NDArray[np.float64]
type Indices = npt.NDArray[np.intp]


def wrap_angle(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return angles wrapped into one turn centred on zero, the half-open interval `(-pi, pi]`.

    Args:
        values:
            An angle or an array of angles in radians.

    Returns:
        angles:
            The angles in `(-pi, pi]`, with `-pi` moved to `pi` and angles already inside left exactly as they are.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.base import wrap_angle
        >>> wrap_angle(np.array([3 * np.pi / 2, -np.pi, 0.5])).round(4).tolist()
        [-1.5708, 3.1416, 0.5]
    """
    angles = np.asarray(values, dtype=float)
    inside = (angles > -np.pi) & (angles <= np.pi)
    return np.where(inside, angles, np.pi - np.mod(np.pi - angles, 2 * np.pi))


@dataclass(frozen=True)
class VisualSpec:
    """Name of a registered visual and the options it is built with.

    Args:
        kind:
            The name of a visual registered in the visualization layer.

        options:
            The keyword arguments of the visual.

    Examples:
        >>> from skcomplex.core.base import VisualSpec
        >>> VisualSpec('points', {'position': 'x'}).kind
        'points'
    """

    kind: str = 'scalars'
    options: dict[str, Any] = field(default_factory=dict)


class BaseDomain:
    """Base of the domains, the sets of values a variable takes.

    A domain names its kind, which decides the laws that may target a variable of it, checks that a value belongs to it,
    and turns values to and from the stored numbers of a state vector. Its methods raise `NotImplementedError` rather
    than being abstract, so an enumeration of domains can derive from it. Its `value`, the name or parameters of the
    domain, is declared and not defined, so it does not hide the value of an enumeration member.
    """

    value: Any

    def _check(self) -> Self:
        """Return the domain with its values checked, the domain itself by default."""
        return self

    def check(self) -> Self:
        """Return the domain with its values checked.

        Domains are values, so checking returns a domain rather than changing this one. Declaring a state or a
        parameter checks its domain.

        Returns:
            domain:
                The checked domain.

        Raises:
            ValidationError:
                A value of the domain is invalid, such as an unknown kind of structure.
        """
        return self._check()

    @property
    def domain_kind(self) -> DomainKind:
        """Kind of value the domain holds."""
        raise NotImplementedError

    @property
    def sage_domain(self) -> str:
        """Sage domain of the symbols of the domain: real, positive, integer, or complex."""
        raise NotImplementedError

    def encode(self, value: object) -> npt.NDArray[Any]:
        """Return a value as the stored numbers of a state vector.

        Args:
            value:
                A value of the domain.

        Returns:
            array:
                The stored numbers.
        """
        raise NotImplementedError

    def decode(self, values: npt.NDArray[np.float64]) -> Any:
        """Return stored numbers as values of the domain.

        Args:
            values:
                The stored numbers.

        Returns:
            values:
                The values of the domain.
        """
        raise NotImplementedError

    def contains(self, value: object) -> bool:
        """Check that a value belongs to the domain.

        Args:
            value:
                A value.

        Returns:
            is_contained:
                Whether the value belongs to the domain.
        """
        raise NotImplementedError


class NumericDomain(BaseDomain, Enum):
    """Set of values a variable can take.

    Examples:
        >>> from skcomplex.core.base import NumericDomain
        >>> NumericDomain.POSITIVE.contains(2.0)
        True
        >>> NumericDomain.NONNEGATIVE_INTEGER.contains(1.5)
        False
        >>> NumericDomain.COMPLEX.contains(1 + 2j), NumericDomain.REAL.contains(1 + 2j), NumericDomain.REAL.contains(
        ...     3 + 0j
        ... )
        (True, False, True)
    """

    REAL = 'real'
    POSITIVE = 'positive'
    NONNEGATIVE = 'nonnegative'
    INTEGER = 'integer'
    NONNEGATIVE_INTEGER = 'nonnegative integer'
    BOOLEAN = 'boolean'
    ANGLE = 'angle'
    COMPLEX = 'complex'

    @staticmethod
    def categorical(*labels: Any) -> 'CategoricalDomain':
        """Return the domain of a finite alphabet, whose values are stored as the indices of its labels.

        Args:
            *labels:
                The distinct labels of the alphabet, in order.

        Returns:
            domain:
                The categorical domain.

        Examples:
            >>> from skcomplex.core.base import NumericDomain
            >>> health = NumericDomain.categorical('susceptible', 'infected', 'recovered')
            >>> health.encode(['infected', 'recovered']).tolist()
            [1.0, 2.0]
        """
        return CategoricalDomain(tuple(labels))

    @property
    def domain_kind(self) -> DomainKind:
        """Kind of value the domain holds."""
        kinds = {
            NumericDomain.INTEGER: DomainKind.INTEGER,
            NumericDomain.NONNEGATIVE_INTEGER: DomainKind.INTEGER,
            NumericDomain.BOOLEAN: DomainKind.BOOLEAN,
            NumericDomain.ANGLE: DomainKind.ANGLE,
            NumericDomain.COMPLEX: DomainKind.COMPLEX,
        }
        return kinds.get(self, DomainKind.REAL)

    @property
    def sage_domain(self) -> str:
        """Sage domain of the symbols of the domain, which has no nonnegative domains."""
        return {
            NumericDomain.REAL: 'real',
            NumericDomain.POSITIVE: 'positive',
            NumericDomain.NONNEGATIVE: 'real',
            NumericDomain.INTEGER: 'integer',
            NumericDomain.NONNEGATIVE_INTEGER: 'integer',
            NumericDomain.BOOLEAN: 'integer',
            NumericDomain.ANGLE: 'real',
            NumericDomain.COMPLEX: 'complex',
        }[self]

    def encode(self, value: object) -> npt.NDArray[Any]:
        """Return a value as floats, with booleans as 0 and 1 and angles in `(-pi, pi]`, or as complex numbers.

        Args:
            value:
                A number, a boolean, or an array of them. A real domain takes the real part of a complex number.

        Returns:
            array:
                The float values, or the complex values of the complex domain.

        Examples:
            >>> from skcomplex.core.base import NumericDomain
            >>> NumericDomain.COMPLEX.encode([1, 2j]).tolist()
            [(1+0j), 2j]
        """
        array = np.asarray(value)
        if self is NumericDomain.COMPLEX:
            return np.asarray(array, dtype=complex)
        array = np.asarray(array.real if np.iscomplexobj(array) else array, dtype=float)
        return wrap_angle(array) if self is NumericDomain.ANGLE else array

    def decode(self, values: npt.NDArray[np.float64]) -> npt.NDArray[Any]:
        """Return stored values in the domain's own type, booleans for the boolean domain and numbers otherwise.

        Args:
            values:
                The stored float values.

        Returns:
            values:
                The decoded values.
        """
        return np.asarray(values) > 0.5 if self is NumericDomain.BOOLEAN else np.asarray(values)  # noqa: PLR2004

    def contains(self, value: object) -> bool:
        """Check that every component of a value belongs to the domain.

        A real domain holds a complex number whose imaginary part is zero.

        Args:
            value:
                A number or an array of numbers.

        Returns:
            is_contained:
                Whether every component is finite and belongs to the domain.
        """
        try:
            numbers = np.asarray(value, dtype=complex)
        except (TypeError, ValueError):
            return False
        if not np.all(np.isfinite(numbers)):
            return False
        if self is NumericDomain.COMPLEX:
            return True
        if np.any(numbers.imag != 0):
            return False
        array = numbers.real
        integral = bool(np.all(array == np.round(array)))
        checks = {
            NumericDomain.BOOLEAN: bool(np.all((array == 0) | (array == 1))),
            NumericDomain.ANGLE: True,
            NumericDomain.REAL: True,
            NumericDomain.POSITIVE: bool(np.all(array > 0)),
            NumericDomain.NONNEGATIVE: bool(np.all(array >= 0)),
            NumericDomain.INTEGER: integral,
            NumericDomain.NONNEGATIVE_INTEGER: integral and bool(np.all(array >= 0)),
        }
        return checks[self]


@dataclass(frozen=True)
class CategoricalDomain(BaseDomain):
    """Domain of a finite alphabet, whose values are stored as the indices of its labels.

    Args:
        labels:
            The distinct labels of the alphabet, in order.

    Examples:
        >>> from skcomplex.core.base import NumericDomain
        >>> NumericDomain.categorical('low', 'high').contains(['high', 1])
        True
    """

    labels: tuple[Any, ...]

    @property
    def value(self) -> str:
        """Name of the domain, used in error messages."""
        return f'categorical {{{", ".join(map(str, self.labels))}}}'

    @property
    def domain_kind(self) -> DomainKind:
        """Kind of value the domain holds."""
        return DomainKind.CATEGORICAL

    @property
    def sage_domain(self) -> str:
        """Sage domain of the symbols of the domain."""
        return 'integer'

    def _index(self, item: Any) -> float:
        """Return the index of a label, or an index given as an integer."""
        if item in self.labels:
            return float(self.labels.index(item))
        numeric = isinstance(item, int | float | np.integer | np.floating)
        if numeric and float(item).is_integer() and 0 <= item < len(self.labels):
            return float(item)
        error_msg = f'{item!r} is neither a label nor an index of the alphabet {self.labels}.'
        raise ValueError(error_msg)

    def encode(self, value: object) -> npt.NDArray[np.float64]:
        """Return labels or indices as float indices.

        Args:
            value:
                A label, an index, or an array of them.

        Returns:
            array:
                The float indices.
        """
        array = np.asarray(value, dtype=object)
        return np.vectorize(self._index, otypes=[float])(array) if array.size else np.zeros(array.shape)

    def decode(self, values: npt.NDArray[np.float64]) -> npt.NDArray[Any]:
        """Return stored indices as labels.

        Args:
            values:
                The stored float indices.

        Returns:
            labels:
                The labels, as an object array of the same shape.
        """
        indices = np.rint(np.asarray(values)).astype(int)
        return np.asarray(self.labels, dtype=object)[np.clip(indices, 0, len(self.labels) - 1)]

    def contains(self, value: object) -> bool:
        """Check that every component of a value is a label or an index of the alphabet.

        Args:
            value:
                A label, an index, or an array of them.

        Returns:
            is_contained:
                Whether every component belongs to the alphabet.
        """
        try:
            self.encode(value)
        except (TypeError, ValueError):
            return False
        return True


@dataclass(frozen=True)
class TensorDomain(BaseDomain):
    """Domain of a state that is a tensor of a rank over the dimensions of the space, with a declared symmetry.

    Fields of tensors, such as the nematic order of active matter, are specified by `add-tensor-fields`.

    Args:
        rank:
            The number of indices.

        symmetry:
            `'symmetric'`, `'antisymmetric'`, `'traceless'` for symmetric and traceless, or `None` for none.

    Examples:
        >>> from skcomplex.core.base import TensorDomain
        >>> TensorDomain(2, 'traceless').value
        'traceless tensor of rank 2'
    """

    rank: int
    symmetry: str | None = None

    @property
    def value(self) -> str:
        """Name of the domain, used in error messages."""
        kind = f'{self.symmetry} tensor' if self.symmetry else 'tensor'
        return f'{kind} of rank {self.rank}'

    @property
    def domain_kind(self) -> DomainKind:
        """Kind of value the domain holds."""
        return DomainKind.TENSOR

    @property
    def sage_domain(self) -> str:
        """Sage domain of the symbols of the components, which are real."""
        return 'real'

    def encode(self, value: object) -> npt.NDArray[Any]:
        """Return the components of a tensor as floats.

        Args:
            value:
                The components.

        Returns:
            array:
                The float values.
        """
        return np.asarray(value, dtype=float)

    def decode(self, values: npt.NDArray[np.float64]) -> npt.NDArray[Any]:
        """Return stored values unchanged.

        Args:
            values:
                The stored values.

        Returns:
            values:
                The values.
        """
        return np.asarray(values)

    def contains(self, value: object) -> bool:
        """Check that every component of a tensor is a finite real number.

        Args:
            value:
                The components.

        Returns:
            is_contained:
                Whether every component is finite and real.
        """
        try:
            numbers = np.asarray(value, dtype=complex)
        except (TypeError, ValueError):
            return False
        return bool(np.all(np.isfinite(numbers)) and np.all(numbers.imag == 0))


METADATA_KEY = 'skcomplex'


class VariableKind(Enum):
    """Role a declared field plays in a component."""

    STATE = 'state'
    PARAMETER = 'parameter'
    ROLE = 'role'


@dataclass(frozen=True)
class Variable:
    """Declaration of a field of a component.

    Args:
        kind:
            Whether the field is a state variable, a parameter, or a role.

        domain:
            The set of values the field takes.

        vector:
            Whether the field has one component per dimension of the enclosing space.

        order:
            The order of the highest time derivative that laws give for a state, an integer or a real number in
            `(0, 1)` for the Caputo derivative of a fractional order.

        lags:
            The number of past values a state of a discrete map keeps.

        initial:
            The default initial values of the derivatives of orders 1 to `order - 1`, or of the lags 1 to `lags`.

        derivative_of:
            The state whose derivative a companion state holds, or `None`.

        lag_of:
            The state whose past value a companion state holds, or `None`.

        index:
            The order of the derivative or the lag that a companion state holds.

        variadic:
            Whether a role binds a sequence of element paths and group member references instead of one path.

        noise:
            Whether a role binds a set of Wiener processes, which the noise laws of the interaction draw, instead of
            a path.

        per:
            The variadic role whose entries a parameter holds one value for, or `None`.

        quantity:
            The conserved quantity a state holds, such as `'money'` or `'mass'`, or `None`.

        history:
            The values of a state before the start, which delayed laws read, a number or a function of time, or `None`
            to keep the initial value before the start.

        position:
            Whether a vector state is the position of its component in a continuous space, on which the boundary of
            the space acts.

        velocity_of:
            The position state whose rate a vector state is, which a reflecting boundary reverses, or `None`.

        adaptive:
            Whether a state was declared as an adaptive parameter, which laws may change.

        bounds:
            The lower and upper bound of an adaptive parameter that is a control input, or `None` for an unbounded one.
    """

    kind: VariableKind
    domain: BaseDomain = NumericDomain.REAL
    vector: bool = False
    order: float = 1
    lags: int = 0
    initial: tuple[Any, ...] = ()
    derivative_of: str | None = None
    lag_of: str | None = None
    index: int = 0
    variadic: bool = False
    noise: bool = False
    per: str | None = None
    quantity: str | None = None
    history: Any = None
    position: bool = False
    velocity_of: str | None = None
    adaptive: bool = False
    bounds: tuple[float, float] | None = None

    @property
    def is_fractional(self) -> bool:
        """Whether the order of the state is not an integer, so its rate laws give a Caputo derivative."""
        return not isinstance(self.order, int)

    @property
    def derivatives(self) -> int:
        """Number of companion states that hold the derivatives of orders 1 to `order - 1`."""
        return 0 if self.is_fractional else int(self.order) - 1


def _normalized_order(order: object) -> float:
    """Return an order as an integer when it has an integer value, or raise an error when it is not positive."""
    if isinstance(order, bool) or not isinstance(order, int | float | np.integer | np.floating):
        error_msg = f'order = {order!r} must be a number: an integer of at least 1 or a real number in (0, 1).'
        raise TypeError(error_msg)
    if float(order).is_integer():
        return int(order)
    if not (math.isfinite(order) and order > 0):
        error_msg = f'order = {order!r} must be an integer of at least 1 or a real number in (0, 1).'
        raise ValueError(error_msg)
    return float(order)


def _declare_field(default: Any, declaration: Variable) -> Any:
    metadata = {METADATA_KEY: declaration}
    if default is dataclasses.MISSING:
        return dataclasses.field(metadata=metadata)
    if not isinstance(default, int | float | complex | str | bytes | Enum | np.generic | type(None)):
        return dataclasses.field(default_factory=lambda: copy.deepcopy(default), metadata=metadata)
    return dataclasses.field(default=default, metadata=metadata)


def _check_bounds(bounds: tuple[float, float], *, adaptive: bool) -> None:
    """Raise an error for bounds of a parameter that is not adaptive, or whose lower bound is above the upper one."""
    if not adaptive:
        error_msg = f'Only an adaptive parameter takes bounds, but bounds={bounds!r} were given. Add `adaptive=True`.'
        raise ValueError(error_msg)
    lower, upper = (float(value) for value in bounds)
    if not lower <= upper:
        error_msg = f'bounds={bounds!r} must be a pair (lower, upper) with lower at most upper.'
        raise ValueError(error_msg)


def _companions_of(name: str, declaration: Variable, default: Any) -> dict[str, Any]:
    """Return the companion state fields of a state of higher order or with lags, by field name.

    The derivatives of a complex state are complex, and those of any other state are real.

    Args:
        name:
            The name of the state.

        declaration:
            The declaration of the state.

        default:
            The default value of the state, which missing past values take.

    Returns:
        fields:
            The companion fields, empty for a first-order state without lags.
    """
    fields = {}
    domain = NumericDomain.COMPLEX if declaration.domain is NumericDomain.COMPLEX else NumericDomain.REAL
    for index in range(1, declaration.derivatives + 1):
        value = declaration.initial[index - 1] if index <= len(declaration.initial) else 0.0
        companion = Variable(VariableKind.STATE, domain, declaration.vector, derivative_of=name, index=index)
        fields[f'{name}_d{index}'] = _declare_field(value, companion)
    for index in range(1, declaration.lags + 1):
        value = declaration.initial[index - 1] if index <= len(declaration.initial) else default
        companion = Variable(VariableKind.STATE, declaration.domain, declaration.vector, lag_of=name, index=index)
        fields[f'{name}_lag{index}'] = _declare_field(value, companion)
    return fields


def _default_of(field: Any) -> Any:
    """Return the default value a declared field carries, a fresh copy for a mutable default."""
    factory = field.default_factory
    return factory() if callable(factory) else field.default


def _is_declaration(value: object) -> bool:
    """Check whether a value is a field declared with `state`, `parameter`, or `role`."""
    return isinstance(value, dataclasses.Field)


def _namespace_method(function: Callable[..., Sequence[Any]], name: str) -> Callable[..., Sequence[Any]]:
    """Return a `laws` or `quantities` method from a function of the namespace, or of the instance and the namespace."""
    if len(inspect.signature(function).parameters) >= 2:  # noqa: PLR2004
        return function

    def method(self: object, s: Any) -> Sequence[Any]:
        """Return the list of the function."""
        return function(s)

    method.__doc__ = function.__doc__ or f'Return the {name} of the type.'
    return method


def _caller_module() -> str:
    """Return the module of the code that called `define`, three calls above this one, so the type pickles there."""
    frame = inspect.currentframe()
    for _ in range(3):
        frame = frame.f_back if frame is not None else None
    return frame.f_globals.get('__name__', __name__) if frame is not None else __name__


def _fields(kind: type) -> dict[str, Any]:
    """Return the declared fields of a type, after rejecting the names it reserves."""
    fields = declared_fields(kind)
    reserved = sorted(set(fields) & kind.reserved_names)  # type: ignore[attr-defined]
    if reserved:
        error_msg = (
            f'{kind.__name__} declares the variable {reserved[0]!r}, whose name is reserved. Rename it, since '
            f'{", ".join(sorted(kind.reserved_names))} are reserved names.'  # type: ignore[attr-defined]
        )
        raise LabelError(error_msg)
    return fields


def _settings(kind: type, root: type) -> dict[str, inspect.Parameter]:
    """Return the keyword arguments of the constructors of a type up to its base, without `**values`."""
    found: dict[str, inspect.Parameter] = {}
    for klass in kind.__mro__:
        init = vars(klass).get('__init__')
        if init is None:
            continue
        parameters = list(inspect.signature(init).parameters.values())[1:]
        for setting in parameters:
            if setting.kind not in {setting.VAR_KEYWORD, setting.VAR_POSITIONAL}:
                found.setdefault(setting.name, setting)
        if klass is root or not any(item.kind == item.VAR_KEYWORD for item in parameters):
            break
    return found


def _declare(instance: Any, label: str, values: Mapping[str, Any]) -> None:
    """Store the label and every given value of an instance, the default for a declared name not given.

    A declared name without a default that is not given is stored as `dataclasses.MISSING`, and every given name is
    stored, also one the type does not declare, so that `_check_declared` reports both.
    """
    instance.label = label
    for name, declared in declared_fields(type(instance)).items():
        setattr(instance, name, values[name] if name in values else _default_of(declared))
    for name, value in values.items():
        setattr(instance, name, value)


def _constructor_names(kind: type, root: type) -> set[str]:
    """Return the names the constructors of a type up to its base take, without `**values`."""
    names: set[str] = set()
    for klass in kind.__mro__:
        init = vars(klass).get('__init__')
        if init is not None:
            parameters = list(inspect.signature(init).parameters.values())[1:]
            names |= {item.name for item in parameters if item.kind not in {item.VAR_KEYWORD, item.VAR_POSITIONAL}}
        if klass is root:
            break
    return names


def _check_declared(instance: Any, root: type) -> None:
    """Raise an error for a stored value whose name the type does not declare, or a declared name not given."""
    kind = type(instance)
    fields = _fields(kind)
    known = {'label', *fields, *_constructor_names(kind, root)}
    unknown = sorted(name for name in vars(instance) if name not in known and not name.endswith('_'))
    if unknown:
        declared = ', '.join(fields) or 'none'
        error_msg = (
            f'{kind.__name__} got {", ".join(map(repr, unknown))}, which it does not declare. Its declared '
            f'variables are {declared}.'
        )
        raise ArgumentError(error_msg)
    missing = [name for name in fields if getattr(instance, name, dataclasses.MISSING) is dataclasses.MISSING]
    if missing:
        error_msg = f'{kind.__name__} needs the arguments {", ".join(map(repr, missing))}.'
        raise ArgumentError(error_msg)


def _define_type(
    kind: type,
    name: str,
    module: str,
    *,
    roles: Sequence[str] = (),
    states: Mapping[str, Any] | None = None,
    parameters: Mapping[str, Any] | None = None,
    laws: Callable[..., list[Any]] | None = None,
    quantities: Callable[..., Sequence[Any]] | None = None,
    attributes: Mapping[str, Any] | None = None,
) -> type:
    """Return a subclass of a type built from declarations, laws and quantities functions, and class attributes.

    A state or a parameter given as a plain value is declared with that value as its default.
    """
    fields = {
        **{key: role() for key in roles},
        **{key: value if _is_declaration(value) else state(value) for key, value in (states or {}).items()},
        **{key: value if _is_declaration(value) else parameter(value) for key, value in (parameters or {}).items()},
    }
    namespace: dict[str, Any] = {'__module__': module, **(attributes or {}), **fields}
    namespace['__qualname__'] = name
    if laws is not None:
        namespace['laws'] = _namespace_method(laws, 'laws')
    if quantities is not None:
        namespace['quantities'] = _namespace_method(quantities, 'quantities')
    new_type = type(name, (kind,), namespace)
    new_type._declarations()  # type: ignore[attr-defined]
    return new_type


def _declarations(kind: type) -> dict[str, 'Variable']:
    """Return the declarations of the states, parameters, and roles of a type, with the derived companions."""
    return {name: cast('Variable', declaration_of(field)) for name, field in _fields(kind).items()}


def _get_default(kind: type, root: type, name: str) -> Any:
    """Return the default value of a declared name or a setting of a type."""
    fields = _fields(kind)
    if name in fields:
        return _default_of(fields[name])
    setting = _settings(kind, root).get(name)
    if setting is None or setting.default is inspect.Parameter.empty:
        error_msg = f'{kind.__name__} has no default for {name!r}.'
        raise KeyError(error_msg)
    return setting.default


def _argument_names(kind: type, root: type) -> list[str]:
    """Return the names the constructor of a type takes besides the label: its settings and the declared names."""
    return [*(name for name in _settings(kind, root) if name != 'label'), *_fields(kind)]


def _get_params(instance: Any, root: type, *, deep: bool) -> dict[str, Any]:
    """Return the label, the settings, and the values of every declared name of an instance."""
    params: dict[str, Any] = {}
    for name in sorted(['label', *_argument_names(type(instance), root)]):
        value = getattr(instance, name)
        if deep and hasattr(value, 'get_params') and not isinstance(value, type):
            params.update({f'{name}__{key}': item for key, item in value.get_params().items()})
        params[name] = value
    return params


def state(
    default: Any = 0.0,
    *,
    domain: BaseDomain = NumericDomain.REAL,
    vector: bool = False,
    order: float = 1,
    lags: int = 0,
    initial: Sequence[Any] = (),
    quantity: str | None = None,
    history: Any = None,
    position: bool = False,
    velocity_of: str | None = None,
) -> Any:
    """Declare a state variable of an element, with its initial value as default.

    A state of order `k` gets companion states `<name>_d1` to `<name>_d<k-1>` that hold its time derivatives, and laws
    on it give its `k`-th derivative. A state of a discrete map with `m` lags gets companion states `<name>_lag1` to
    `<name>_lag<m>` that hold its past values.

    Args:
        default:
            The default initial value. A scalar default of a vector state applies to every component.

        domain:
            The set of values the state takes.

        vector:
            Whether the state has one component per dimension of the enclosing space.

        order:
            The order of the highest time derivative that laws give, 1 for first-order equations, or a real number in
            `(0, 1)` for the Caputo derivative of a fractional order. An order that is neither is reported by
            validation.

        lags:
            The number of past values the state keeps for difference equations, 0 for none.

        initial:
            The default initial values of the derivatives of orders 1 to `order - 1`, or of the past values at steps
            -1 to `-lags`. Missing derivatives default to 0, and missing past values to `default`.

        quantity:
            The conserved quantity the state holds, such as `'money'`, or `None`. Only flows change a tagged state, so
            the total of every quantity changes only through flows with the environment.

        history:
            The values of the state before the start, which delayed laws read, a number or a function of time, or
            `None` to keep the initial value before the start.

        position:
            Whether a vector state is the position of its component in a continuous space, on which the boundary of
            the space acts.

        velocity_of:
            The position state whose rate a vector state is, which a reflecting boundary of the space reverses.

    Returns:
        field:
            A dataclass field that carries the declaration.

    Raises:
        TypeError:
            The order is not a number.

        ValueError:
            The order is not positive, the lags are negative, an order other than 1 and lags are both given, or there
            are more initial values than companions.

    Examples:
        >>> from skcomplex.core.base import NumericDomain, state
        >>> from skcomplex.core.components import ElementComponent
        >>> class PopulationComponent(ElementComponent):
        ...     n: float = state(1.0, domain=NumericDomain.NONNEGATIVE)
        >>> PopulationComponent('rabbits', n=40).n
        40
    """
    order = _normalized_order(order)
    if not (order > 0 and isinstance(lags, int) and lags >= 0):
        error_msg = f'order = {order!r} must be an integer of at least 1 and lags = {lags!r} a nonnegative integer.'
        raise ValueError(error_msg)
    if order != 1 and lags:
        error_msg = (
            'A state takes either an order other than 1 for differential equations or lags for difference equations.'
        )
        raise ValueError(error_msg)
    companions = max(0 if isinstance(order, float) else order - 1, lags)
    if len(initial) > companions:
        error_msg = f'{len(initial)} initial values were given for {companions} companion states.'
        raise ValueError(error_msg)
    declaration = Variable(
        VariableKind.STATE,
        domain.check(),
        vector,
        order,
        lags,
        tuple(initial),
        quantity=quantity,
        history=history,
        position=position,
        velocity_of=velocity_of,
    )
    return _declare_field(default, declaration)


def declaration_of(value: object) -> Variable | None:
    """Return the declaration a dataclass field carries, or `None` for any other value.

    Args:
        value:
            A class attribute of a component.

    Returns:
        declaration:
            The declaration of a field made by `state`, `parameter`, or `role`, or `None`.
    """
    if isinstance(value, dataclasses.Field):
        return value.metadata.get(METADATA_KEY)
    return None


def parameter(
    default: Any = dataclasses.MISSING,
    *,
    domain: BaseDomain = NumericDomain.REAL,
    vector: bool = False,
    per: str | None = None,
    adaptive: bool = False,
    bounds: tuple[float, float] | None = None,
) -> Any:
    """Declare a parameter of an element or an interaction.

    A parameter declared per entry of a variadic role holds a sequence with one value per entry of the role. Laws read
    it as a tuple of symbols, and the component `i` of the parameter `name` of the interaction at `path` is addressed
    as `<path>.<name>[<i>]`.

    Args:
        default:
            The default value. A parameter without a default must be given to the constructor.

        domain:
            The set of values the parameter takes.

        vector:
            Whether the parameter has one component per dimension of the enclosing space.

        per:
            The name of a variadic role of the interaction whose entries the parameter holds one value for, or `None`.

        adaptive:
            Whether laws may change the parameter, such as a policy rate an institution sets. An adaptive parameter of
            an element is compiled and recorded as a state whose initial value is the one the constructor gets, so
            laws target it like any state.

        bounds:
            The lower and upper bound of an adaptive parameter, such as `(0, 1)`, which makes it a control input that a
            schedule, an optimal control solver, or a controller supplies, or `None` for no bounds. Validation reports
            an initial value outside the bounds.

    Returns:
        field:
            A dataclass field that carries the declaration.

    Raises:
        ValueError:
            The parameter is both a vector and declared per entry of a role, or has bounds without being adaptive or
            with a lower bound above the upper one.

    Examples:
        >>> from skcomplex.core.base import NumericDomain, parameter, role
        >>> from skcomplex.core.base import BaseInteraction
        >>> class GrowthInteraction(BaseInteraction):
        ...     r: float = parameter(1.0, domain=NumericDomain.POSITIVE)
        >>> GrowthInteraction('growth', r=0.5).r
        0.5
        >>> class RegulationInteraction(BaseInteraction):
        ...     inputs: list[str] = role(variadic=True)
        ...     weights: list[float] = parameter(per='inputs')
        >>> RegulationInteraction('regulation', inputs=['a', 'b'], weights=[1.0, -1.0]).weights
        [1.0, -1.0]
        >>> from skcomplex.core.components import ElementComponent
        >>> class FisheryComponent(ElementComponent):
        ...     effort: float = parameter(0.5, adaptive=True, bounds=(0.0, 1.0))
        >>> FisheryComponent('fishery').effort
        0.5
    """
    if bounds is not None:
        _check_bounds(bounds, adaptive=adaptive)
    if adaptive:
        if per is not None:
            error_msg = f'A parameter takes either `adaptive=True` or `per={per!r}`, not both.'
            raise ValueError(error_msg)
        checked = None if bounds is None else (float(bounds[0]), float(bounds[1]))
        declaration = Variable(VariableKind.STATE, domain.check(), vector, adaptive=True, bounds=checked)
        return _declare_field(default, declaration)
    if vector and per is not None:
        error_msg = f'A parameter takes either `vector=True` or `per={per!r}`, not both.'
        raise ValueError(error_msg)
    return _declare_field(default, Variable(VariableKind.PARAMETER, domain.check(), vector, per=per))


def role(*, variadic: bool = False, noise: bool = False) -> Any:
    """Declare a role of an interaction, bound to an element path or to a set of Wiener processes in the constructor.

    A plain role binds one element path, one group path, or one group member reference `<group>[<index>]`, and laws
    read a bound member like a single element whose states stay in the group. A variadic role binds a sequence of
    element paths and group member references, and laws read it as a tuple of namespaces, one per entry in the given
    order. A noise role binds a `WienerProcess`, whose processes the noise laws of the interaction name through the
    handle `s.<role>.dW`. Interactions of one system that bind processes of one label share them and their
    correlations.

    Args:
        variadic:
            Whether the role binds a sequence of entries instead of one path.

        noise:
            Whether the role binds a set of Wiener processes instead of a path.

    Returns:
        field:
            A required dataclass field that carries the declaration.

    Raises:
        ValueError:
            The role is both variadic and a noise role.

    Examples:
        >>> from skcomplex.core.base import role
        >>> from skcomplex.core.base import BaseInteraction
        >>> class MortalityInteraction(BaseInteraction):
        ...     target: str = role()
        >>> MortalityInteraction('death', target='foxes').target
        'foxes'
        >>> MortalityInteraction('death', target='foxes[3]').target
        'foxes[3]'
        >>> class SumInteraction(BaseInteraction):
        ...     target: str = role()
        ...     inputs: list[str] = role(variadic=True)
        >>> SumInteraction('sum', target='total', inputs=['a', 'nodes[3]']).inputs
        ['a', 'nodes[3]']
        >>> from skcomplex.core.components import WienerProcess
        >>> class ShockInteraction(BaseInteraction):
        ...     target: str = role()
        ...     market: WienerProcess = role(noise=True)
        >>> ShockInteraction('shock', target='stock', market=WienerProcess('market', dimension=2)).market.dimension
        2
    """
    if variadic and noise:
        error_msg = 'A role takes either `variadic=True` or `noise=True`, not both.'
        raise ValueError(error_msg)
    return _declare_field(dataclasses.MISSING, Variable(VariableKind.ROLE, variadic=variadic, noise=noise))


def declared_fields(component: Any) -> dict[str, Any]:
    """Return the fields a component class declares with `state`, `parameter`, and `role`, with their companions.

    The fields come in the order of the classes from the base to the class, each class in definition order followed by
    the companion states of its states of higher order or with lags, `<name>_d1` to `<name>_d<k-1>` and
    `<name>_lag1` to `<name>_lag<m>`. A field declared again in a subclass keeps its place.

    Args:
        component:
            A component class or instance.

    Returns:
        fields:
            A mapping from name to the field that carries the declaration and the default.

    Examples:
        >>> from skcomplex.core.base import declared_fields, state
        >>> class PendulumComponent:
        ...     theta = state(0.1, order=2)
        >>> list(declared_fields(PendulumComponent))
        ['theta', 'theta_d1']
    """
    cls = component if isinstance(component, type) else type(component)
    fields: dict[str, Any] = {}
    for klass in reversed(cls.__mro__):
        own = {name: value for name, value in vars(klass).items() if declaration_of(value) is not None}
        fields.update(own)
        for name, value in own.items():
            declaration = cast('Variable', declaration_of(value))
            default = value.default if value.default is not dataclasses.MISSING else 0.0
            companions = _companions_of(name, declaration, default)
            fields.update({key: item for key, item in companions.items() if key not in vars(klass)})
    return fields


def declarations_of(component: Any, kind: VariableKind) -> dict[str, Variable]:
    """Return the declared fields of one kind of a component, in declaration order.

    Args:
        component:
            A component class or instance.

        kind:
            The kind of field to return.

    Returns:
        declarations:
            A mapping from field name to declaration.
    """
    return {
        name: field.metadata[METADATA_KEY]
        for name, field in declared_fields(component).items()
        if field.metadata[METADATA_KEY].kind == kind
    }


_DOMAINS: dict[str, BaseDomain] = {}


def symbol(name: str, domain: BaseDomain = NumericDomain.REAL) -> Expression:
    """Return the Sage symbol of a name, real, integer, or complex, and record its variable domain.

    The symbol is declared real or integer without Maxima, which Sage would otherwise load to record the domain, so
    compiling stays fast. A symbol of the complex domain is left undeclared, which Sage treats as complex. Sage keeps
    one symbol per name, so a name declared before with another domain is redeclared.
    The symbolic analysis gives Maxima the full domains before an exact solution.

    Args:
        name:
            The name of the symbol.

        domain:
            The domain of the variable the symbol stands for.

    Returns:
        symbol:
            The Sage symbol.

    Examples:
        >>> from skcomplex.core.base import NumericDomain, domain_of, symbol
        >>> n = symbol('n', NumericDomain.POSITIVE)
        >>> n.is_real(), domain_of(n)
        (True, <NumericDomain.POSITIVE: 'positive'>)
        >>> symbol('z', NumericDomain.COMPLEX).is_real()
        False
        >>> symbol('z').is_real(), symbol('z', NumericDomain.COMPLEX).is_real()
        (True, False)
    """
    item = SR.symbol(name)
    for previous in ('integer', 'real'):
        item.decl_forget(previous)
    if domain.sage_domain != 'complex':
        item.decl_assume('integer' if domain.sage_domain == 'integer' else 'real')
    _DOMAINS[name] = domain
    return item


def vector(components: Iterable[Any]) -> SymbolicVector:
    """Return a Sage vector of symbolic components.

    Args:
        components:
            The components.

    Returns:
        vector:
            The vector over the symbolic ring.

    Examples:
        >>> from skcomplex.core.base import symbol, vector
        >>> x, y = symbol('x'), symbol('y')
        >>> vector([x, y]).dot_product(vector([x, y]))
        x^2 + y^2
    """
    return sage_vector(SR, list(components))


def components_of(value: Any) -> list[Any]:
    """Return the components of a vector, or a one-element list of any other value.

    Args:
        value:
            A vector, an expression, or a number.

    Returns:
        components:
            The components.
    """
    return list(value) if isinstance(value, SymbolicVector) else [value]


def symbols_of(expressions: Iterable[Expression]) -> set[Expression]:
    """Return the symbols that occur in expressions.

    Args:
        expressions:
            The expressions.

    Returns:
        symbols:
            The symbols.
    """
    return {item for expression in expressions for item in SR(expression).variables()}


def domain_of(item: Expression) -> BaseDomain:
    """Return the variable domain of a symbol.

    Args:
        item:
            The symbol.

    Returns:
        domain:
            The domain the symbol was created with, or `REAL` for a symbol the framework did not create.
    """
    return _DOMAINS.get(str(item), NumericDomain.REAL)
