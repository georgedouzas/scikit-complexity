"""Define the kinds of the framework: open vocabularies of terms, forms, and the declarations validation compares."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Iterable
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field, fields
from enum import StrEnum
from typing import Any, ClassVar, Self, cast

from ._errors import RegistryError

TERMS: dict[type, dict[str, Any]] = {}


def _terms(values: Iterable['StrEnum | Term']) -> frozenset['StrEnum | Term']:
    """Return the terms of a declaration as a frozen set."""
    return frozenset(values)


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
