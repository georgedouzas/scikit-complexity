"""Define the system vocabulary and the requirement and support declarations that validation compares."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Iterable
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field, fields
from enum import StrEnum
from typing import ClassVar

from ._terms import Term


def _terms(values: Iterable[StrEnum | Term]) -> frozenset[StrEnum | Term]:
    """Return the terms of a declaration as a frozen set."""
    return frozenset(values)


def vocabulary() -> dict[str, tuple[StrEnum | Term, ...]]:
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


SpaceKind.NONE = SpaceKind('NONE', 'none')
SpaceKind.LATTICE = SpaceKind('LATTICE', 'lattice')
SpaceKind.NETWORK = SpaceKind('NETWORK', 'network')
SpaceKind.CONTINUOUS = SpaceKind('CONTINUOUS', 'continuous')
SpaceKind.MESH = SpaceKind('MESH', 'mesh')
LawKind.RATE = LawKind('RATE', 'rate')
LawKind.NOISE = LawKind('NOISE', 'noise')
LawKind.EQUATION = LawKind('EQUATION', 'equation')
LawKind.GENERATOR = LawKind('GENERATOR', 'generator')
LawKind.DISSIPATOR = LawKind('DISSIPATOR', 'dissipator')
LawKind.JUMP = LawKind('JUMP', 'jump')
LawKind.EVENT = LawKind('EVENT', 'event')
LawKind.UPDATE = LawKind('UPDATE', 'update')
LawKind.TRANSITION = LawKind('TRANSITION', 'transition')
LawKind.RULE = LawKind('RULE', 'rule')
LawKind.LOCAL_RULE = LawKind('LOCAL_RULE', 'local-rule')
LawKind.REWRITE = LawKind('REWRITE', 'rewrite')
LawKind.SAMPLE = LawKind('SAMPLE', 'sample')
LawKind.MATCH = LawKind('MATCH', 'match')
LawKind.TRANSFER = LawKind('TRANSFER', 'transfer')
LawKind.BOUNDARY = LawKind('BOUNDARY', 'boundary')
LawKind.INVARIANT = LawKind('INVARIANT', 'invariant')
LawKind.OBJECTIVE = LawKind('OBJECTIVE', 'objective')
LawKind.ENCOUNTER = LawKind('ENCOUNTER', 'encounter')
Trait.FORMLESS = Trait('FORMLESS', 'formless')
Trait.KERNEL_FREE = Trait('KERNEL_FREE', 'kernel-free')
Trait.LATTICE = Trait('LATTICE', 'lattice')
Trait.DECLARABLE = Trait('DECLARABLE', 'declarable')
Trait.AGENT = Trait('AGENT', 'agent')


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

    def __post_init__(self) -> None:
        """Freeze every field into a set of terms."""
        for item in fields(self):
            object.__setattr__(self, item.name, _terms(getattr(self, item.name)))

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
        terms: frozenset[StrEnum | Term] = getattr(self, category)
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

    def __post_init__(self) -> None:
        """Freeze every field into a set of terms."""
        for item in fields(self):
            object.__setattr__(self, item.name, _terms(getattr(self, item.name)))

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
        supported: frozenset[StrEnum | Term] = getattr(self, category)
        return sorted(str(term) for term in set(terms) - supported)
