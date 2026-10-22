"""Define the system vocabulary and the requirement and support declarations that validation compares."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Iterable
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field, fields
from enum import StrEnum


class TimeKind(StrEnum):
    """Kind of time a system evolves in."""

    CONTINUOUS = 'continuous'
    DISCRETE = 'discrete'
    STATIC = 'static'


class SpaceKind(StrEnum):
    """Kind of substrate the components of a system live on."""

    NONE = 'none'
    LATTICE = 'lattice'
    NETWORK = 'network'
    CONTINUOUS = 'continuous'
    MESH = 'mesh'


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
    NEIGHBOURS = 'neighbours'
    HYPEREDGES = 'hyperedges'


class LawClass(StrEnum):
    """Time semantics of a law, which decides the time it is valid under."""

    CONTINUOUS = 'continuous'
    EVENT = 'event'
    DISCRETE = 'discrete'
    CONSTRAINT = 'constraint'
    ANY = 'any'


class LawKind(StrEnum):
    """Kind of law, with the class of its time semantics as `law_class`."""

    RATE = 'rate'
    NOISE = 'noise'
    EQUATION = 'equation'
    GENERATOR = 'generator'
    DISSIPATOR = 'dissipator'
    JUMP = 'jump'
    EVENT = 'event'
    UPDATE = 'update'
    TRANSITION = 'transition'
    RULE = 'rule'
    LOCAL_RULE = 'local-rule'
    REWRITE = 'rewrite'
    SAMPLE = 'sample'
    MATCH = 'match'
    TRANSFER = 'transfer'
    BOUNDARY = 'boundary'
    INVARIANT = 'invariant'
    OBJECTIVE = 'objective'

    @property
    def law_class(self) -> LawClass:
        """Return the time semantics of the law kind."""
        return LAW_CLASSES[self]


LAW_CLASSES: dict[LawKind, LawClass] = {
    LawKind.RATE: LawClass.CONTINUOUS,
    LawKind.NOISE: LawClass.CONTINUOUS,
    LawKind.EQUATION: LawClass.CONTINUOUS,
    LawKind.GENERATOR: LawClass.CONTINUOUS,
    LawKind.DISSIPATOR: LawClass.CONTINUOUS,
    LawKind.JUMP: LawClass.EVENT,
    LawKind.EVENT: LawClass.EVENT,
    LawKind.UPDATE: LawClass.DISCRETE,
    LawKind.TRANSITION: LawClass.DISCRETE,
    LawKind.RULE: LawClass.DISCRETE,
    LawKind.LOCAL_RULE: LawClass.DISCRETE,
    LawKind.REWRITE: LawClass.DISCRETE,
    LawKind.SAMPLE: LawClass.DISCRETE,
    LawKind.MATCH: LawClass.DISCRETE,
    LawKind.TRANSFER: LawClass.ANY,
    LawKind.BOUNDARY: LawClass.CONSTRAINT,
    LawKind.INVARIANT: LawClass.CONSTRAINT,
    LawKind.OBJECTIVE: LawClass.CONSTRAINT,
}


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


VOCABULARY: dict[str, type[StrEnum]] = {
    'time': TimeKind,
    'space': SpaceKind,
    'components': ComponentKind,
    'bindings': Binding,
    'laws': LawKind,
    'domains': DomainKind,
}
NUMERIC_DOMAINS = frozenset(
    {DomainKind.REAL, DomainKind.INTEGER, DomainKind.BOOLEAN, DomainKind.ANGLE, DomainKind.COMPLEX, DomainKind.VECTOR},
)


def _terms(values: Iterable[StrEnum]) -> frozenset[StrEnum]:
    """Return the terms of a declaration as a frozen set."""
    return frozenset(values)


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

    def __post_init__(self) -> None:
        """Freeze every field into a set of terms."""
        for item in fields(self):
            object.__setattr__(self, item.name, _terms(getattr(self, item.name)))

    def accepts(self, category: str, term: StrEnum) -> bool:
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
        terms: frozenset[StrEnum] = getattr(self, category)
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

    def __post_init__(self) -> None:
        """Freeze every field into a set of terms."""
        for item in fields(self):
            object.__setattr__(self, item.name, _terms(getattr(self, item.name)))

    def missing(self, category: str, terms: Iterable[StrEnum]) -> list[str]:
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
        supported: frozenset[StrEnum] = getattr(self, category)
        return sorted(str(term) for term in set(terms) - supported)


EVERY_SPACE = frozenset(SpaceKind)
