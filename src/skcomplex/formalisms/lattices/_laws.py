"""Define the laws of lattice models: local rules, driving, relaxation, extremal updates, and Monte Carlo samples."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Self

from ...core.base import (
    BaseLaw,
    Expression,
    LawClass,
    LawKind,
    Requires,
    SpaceKind,
    TimeKind,
    Trait,
)
from ...core.laws import OuterTotalistic, Table, Totalistic
from ...core.space import Adjacency, Memory, Neighborhood

SCHEDULES = ('synchronous', 'random-sequential', 'random-site', 'checkerboard', 'colored')
MONTE_CARLO_SCHEDULES = ('checkerboard', 'random-sequential', 'colored')
SAMPLERS = ('metropolis', 'glauber', 'heat-bath')
MOVES = ('flip', 'random-value', 'rotate')
EXTREMA = ('min', 'max')


type RuleSpec = Table | Totalistic | OuterTotalistic | Callable[..., Any] | Any


def _targets(target: object) -> tuple[Expression, ...]:
    """Return the target symbols of a lattice law as a tuple."""
    return tuple(target) if isinstance(target, tuple | list) else (target,)


@dataclass(frozen=True)
class LocalRuleLaw(BaseLaw):
    """Law that computes the next value of lattice cells from the ordered values of their neighbourhoods.

    A function rule receives one array per neighbour of every read state, in neighbourhood order, with one value per
    site, and keyword arguments for the parameters of the component that it names, plus `random`, an array of uniform
    numbers per site, when the rule is stochastic, and `degree`, the number of neighbours of every site besides the
    site itself, when it names it. On a graph the values are padded to the largest degree with the fill value. It
    returns the next values of the target, or a tuple with one array per target.

    Args:
        target:
            The symbol of a lattice state, or a tuple of them that the rule updates together.

        over:
            The neighbourhood whose values the rule reads, a `Neighborhood` of offsets on a lattice, an `Adjacency`
            on a network, or a `Memory` of group elements on a group space.

        rule:
            A function, a `Table`, a `Totalistic` rule, an `OuterTotalistic` rule, or an object whose `table()`
                returns a `Table`, such as a rule code.

        reads:
            The lattice states whose neighbourhoods a function rule reads, by default the targets.

        schedule:
            `'synchronous'` to update every site from the previous configuration, `'random-sequential'` to update the
            sites one at a time in a random order, `'random-site'` to update as many randomly chosen sites,
            `'checkerboard'` to update the two sublattices of even and odd coordinate sums of a square lattice in
            turn, or `'colored'` to update the classes of a proper coloring of the neighbour graph in turn.

        stochastic:
            Whether a function rule receives the uniform random array `random`.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.formalisms.lattices import LocalRuleLaw
        >>> from skcomplex.core.laws import Totalistic
        >>> from skcomplex.core.space import Neighborhood
        >>> LocalRuleLaw(symbol('cell'), Neighborhood.radius(), Totalistic({2: 1})).schedule
        'synchronous'
    """

    law_kind: ClassVar[LawKind] = LawKind.LOCAL_RULE
    time_class: ClassVar[LawClass] = LawClass.DISCRETE
    traits: ClassVar[frozenset[Trait]] = frozenset({Trait.LATTICE, Trait.KERNEL_FREE, Trait.DECLARABLE})
    requires: ClassVar[Requires] = Requires(time={TimeKind.DISCRETE}, space={SpaceKind.LATTICE, SpaceKind.NETWORK})

    target: Expression | tuple[Expression, ...]
    over: Neighborhood | Adjacency | Memory
    rule: RuleSpec
    reads: tuple[Expression, ...] | None = None
    schedule: str = 'synchronous'
    stochastic: bool = False

    def _check(self) -> Self:
        """Return the law with its values checked and normalized, raising a named error for an invalid one."""
        changes: dict[str, Any] = {}
        if self.schedule not in SCHEDULES:
            error_msg = f'Unknown schedule {self.schedule!r}. Use one of {", ".join(SCHEDULES)}.'
            raise ValueError(error_msg)
        return replace(self, **changes) if changes else self

    @property
    def targets(self) -> tuple[Expression, ...]:
        """Target symbols."""
        return _targets(self.target)


@dataclass(frozen=True)
class DriveLaw(BaseLaw):
    """Law that changes a whole lattice state at once, such as dropping a grain on a random site.

    The function receives the state as an array of the lattice shape, keyword arguments for the parameters of the
    component that it names, and `random`, the simulation's random generator, and returns the new array.

    Args:
        target:
            The symbol of a lattice state.

        function:
            The function `f(values, *, random, **parameters)`.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.formalisms.lattices import DriveLaw
        >>> DriveLaw(symbol('height'), lambda values, *, random: values + 1).target
        height
    """

    law_kind: ClassVar[LawKind] = LawKind.UPDATE
    time_class: ClassVar[LawClass] = LawClass.DISCRETE
    traits: ClassVar[frozenset[Trait]] = frozenset({Trait.LATTICE, Trait.KERNEL_FREE, Trait.DECLARABLE})
    requires: ClassVar[Requires] = Requires(time={TimeKind.DISCRETE}, space={SpaceKind.LATTICE, SpaceKind.NETWORK})

    target: Expression
    function: Callable[..., Any]

    @property
    def targets(self) -> tuple[Expression, ...]:
        """Target symbols."""
        return (self.target,)


@dataclass(frozen=True)
class RelaxationLaw(BaseLaw):
    """Slow drive and fast relaxation: drive laws once per step, then relaxation laws until the lattice is stable.

    Every relaxation that changes the lattice is recorded as an event with its size, the number of site updates, its
    duration, the number of iterations, and its area, the number of distinct sites updated. A site update is a site
    that changes in an iteration, or a site that `activity` marks as active before it, such as a toppling site.

    Args:
        drive:
            The laws applied once per step, local rules or drives.

        relax:
            The laws applied in turn until the lattice is stable.

        stable:
            A function of the arrays of the relaxation laws' targets, in lattice shape, that returns whether the
            lattice is stable.

        max_iterations:
            The largest number of relaxation iterations in one step.

        label:
            The name of the relaxation events, or `None` for the component's path.

        activity:
            A function of the targets' arrays that returns a boolean array of the active sites of an iteration, or
            `None` to count the sites that change. With an activity, every iteration updates its synchronous local
            rules only at the active sites and the sites that read them, so a site that is not active and reads no
            active site must keep its value.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.formalisms.lattices import LocalRuleLaw, RelaxationLaw
        >>> from skcomplex.core.space import Neighborhood
        >>> height = symbol('height')
        >>> topple = LocalRuleLaw(height, Neighborhood.von_neumann(center=True), lambda z, *n: z - 4 * (z >= 4))
        >>> RelaxationLaw([], [topple], lambda z: bool((z < 4).all())).max_iterations
        100000
    """

    law_kind: ClassVar[LawKind] = LawKind.UPDATE
    time_class: ClassVar[LawClass] = LawClass.DISCRETE
    traits: ClassVar[frozenset[Trait]] = frozenset({Trait.LATTICE, Trait.KERNEL_FREE, Trait.DECLARABLE})
    requires: ClassVar[Requires] = Requires(time={TimeKind.DISCRETE}, space={SpaceKind.LATTICE, SpaceKind.NETWORK})

    drive: Sequence[LocalRuleLaw | DriveLaw]
    relax: Sequence[LocalRuleLaw | DriveLaw]
    stable: Callable[..., bool]
    max_iterations: int = 100_000
    label: str | None = None
    activity: Callable[..., Any] | None = None

    @property
    def targets(self) -> tuple[Expression, ...]:
        """Target symbols of the relaxation laws, in order of first appearance."""
        return tuple(dict.fromkeys(item for law in self.relax for item in law.targets))


@dataclass(frozen=True)
class Energy(BaseLaw):
    """Local energy of a lattice site: every term of the total energy that involves the site's value.

    The energy change of a move is the difference of the local energy at the proposed and the current value, derived
    symbolically. Neighbour values are read with `neighbors(..., reduce='each')` in neighbourhood order, or summed over
    the real neighbours of the site with `neighbors(..., reduce='sum')`, which reads one neighbour's value through
    `s.other`. A neighbour sum leaves out the padding of a graph and the sites outside fixed edges.

    Args:
        expression:
            The local energy, a symbolic expression of the site's states, its neighbours' values, and parameters.

        over:
            The neighbourhood that `neighbors` reads in the expression, an `Adjacency` on a network.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.formalisms.lattices import Energy
        >>> from skcomplex.core.laws import neighbors
        >>> from skcomplex.core.space import Neighborhood
        >>> spin, J = symbol('spin'), symbol('J')
        >>> nearest = Neighborhood.von_neumann()
        >>> len(Energy(-J * spin * sum(neighbors(spin, nearest, reduce='each')), nearest).over.offsets)
        4
    """

    law_kind: ClassVar[LawKind] = LawKind.SAMPLE
    time_class: ClassVar[LawClass] = LawClass.DISCRETE
    traits: ClassVar[frozenset[Trait]] = frozenset({Trait.LATTICE, Trait.KERNEL_FREE})
    requires: ClassVar[Requires] = Requires(time={TimeKind.DISCRETE}, space={SpaceKind.LATTICE, SpaceKind.NETWORK})

    expression: object
    over: Neighborhood | Adjacency | Memory


@dataclass(frozen=True)
class MoveLaw(BaseLaw):
    """Proposed change of the value of a lattice site.

    Args:
        target:
            The symbol of the lattice state the move changes.

        kind:
            `'flip'` for the negative value, `'random-value'` for a value drawn from `values`, or `'rotate'` for the
            value plus a uniform number in `[-width, width]`.

        values:
            The values a random-value move draws from.

        width:
            The largest change of a rotation.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.formalisms.lattices import MoveLaw
        >>> MoveLaw.random_value(symbol('color'), range(3)).values
        (0.0, 1.0, 2.0)
    """

    law_kind: ClassVar[LawKind] = LawKind.SAMPLE
    time_class: ClassVar[LawClass] = LawClass.DISCRETE
    traits: ClassVar[frozenset[Trait]] = frozenset({Trait.LATTICE, Trait.KERNEL_FREE})
    requires: ClassVar[Requires] = Requires(time={TimeKind.DISCRETE}, space={SpaceKind.LATTICE, SpaceKind.NETWORK})

    target: Expression
    kind: str = 'flip'
    values: tuple[float, ...] = ()
    width: float = 0.0

    def _check(self) -> Self:
        """Return the law with its values checked and normalized, raising a named error for an invalid one."""
        changes: dict[str, Any] = {}
        if self.kind not in MOVES:
            error_msg = f'Unknown move {self.kind!r}. Use one of {", ".join(MOVES)}.'
            raise ValueError(error_msg)
        return replace(self, **changes) if changes else self

    @classmethod
    def flip(cls, target: Expression) -> 'MoveLaw':
        """Return the move that changes a value to its negative, the spin flip of Ising models.

        Args:
            target:
                The symbol of the lattice state.

        Returns:
            move:
                The move.
        """
        return cls(target, 'flip')

    @classmethod
    def random_value(cls, target: Expression, values: Iterable[float]) -> 'MoveLaw':
        """Return the move that draws a new value uniformly from a set, as Potts models do.

        Args:
            target:
                The symbol of the lattice state.

            values:
                The values to draw from.

        Returns:
            move:
                The move.
        """
        return cls(target, 'random-value', tuple(float(value) for value in values))

    @classmethod
    def rotate(cls, target: Expression, width: float) -> 'MoveLaw':
        """Return the move that adds a uniform number in `[-width, width]`, as XY models do with angles.

        Args:
            target:
                The symbol of the lattice state.

            width:
                The largest change.

        Returns:
            move:
                The move.
        """
        return cls(target, 'rotate', width=float(width))


@dataclass(frozen=True)
class SamplerLaw(BaseLaw):
    """Acceptance of Monte Carlo moves at a temperature, with one sweep of the lattice per unit of time.

    Args:
        kind:
            `'metropolis'` to accept with probability `min(1, exp(-dE / T))`, `'glauber'` with `1 / (1 + exp(dE / T))`,
            or `'heat-bath'` to draw the new value among all values of a discrete move with Boltzmann weights.

        temperature:
            The temperature, usually a parameter symbol so that it changes live.

        schedule:
            `'checkerboard'` to update the two sublattices of a nearest-neighbour energy on a square lattice in turn,
            vectorised, `'colored'` to update the classes of a proper coloring of the neighbour graph in turn, or
            `'random-sequential'` to visit the sites one at a time in a random order.

    Examples:
        >>> from skcomplex.formalisms.lattices import SamplerLaw
        >>> SamplerLaw('glauber', 2.0).schedule
        'checkerboard'
    """

    law_kind: ClassVar[LawKind] = LawKind.SAMPLE
    time_class: ClassVar[LawClass] = LawClass.DISCRETE
    traits: ClassVar[frozenset[Trait]] = frozenset({Trait.LATTICE, Trait.KERNEL_FREE})
    requires: ClassVar[Requires] = Requires(time={TimeKind.DISCRETE}, space={SpaceKind.LATTICE, SpaceKind.NETWORK})

    kind: str = 'metropolis'
    temperature: object = 1.0
    schedule: str = 'checkerboard'

    def _check(self) -> Self:
        """Return the law with its values checked and normalized, raising a named error for an invalid one."""
        changes: dict[str, Any] = {}
        if self.kind not in SAMPLERS:
            error_msg = f'Unknown sampler {self.kind!r}. Use one of {", ".join(SAMPLERS)}.'
            raise ValueError(error_msg)
        if self.schedule not in MONTE_CARLO_SCHEDULES:
            error_msg = (
                f'Unknown Monte Carlo schedule {self.schedule!r}. Use one of {", ".join(MONTE_CARLO_SCHEDULES)}.'
            )
            raise ValueError(error_msg)
        return replace(self, **changes) if changes else self


@dataclass(frozen=True)
class SampleLaw(BaseLaw):
    """Monte Carlo law that samples states by moves accepted from the change of an energy, one sweep per unit of time.

    Args:
        energy:
            The energy of a site and its neighbourhood.

        move:
            The move proposed at a site.

        sampler:
            The acceptance of moves, with its temperature and schedule, or `None` for Metropolis acceptance at the
            parameter `temperature` of the component.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.formalisms.lattices import Energy, MoveLaw, SampleLaw, SamplerLaw
        >>> from skcomplex.core.space import Neighborhood
        >>> spin = symbol('spin')
        >>> law = SampleLaw(Energy(-spin, Neighborhood.von_neumann()), MoveLaw.flip(spin), SamplerLaw('glauber', 2.0))
        >>> law.sampler.kind
        'glauber'
    """

    law_kind: ClassVar[LawKind] = LawKind.SAMPLE
    time_class: ClassVar[LawClass] = LawClass.DISCRETE
    traits: ClassVar[frozenset[Trait]] = frozenset({Trait.DECLARABLE})
    requires: ClassVar[Requires] = Requires(time={TimeKind.DISCRETE}, space={SpaceKind.LATTICE, SpaceKind.NETWORK})

    energy: Energy
    move: MoveLaw
    sampler: SamplerLaw | None = None

    @property
    def parts(self) -> list[Energy | MoveLaw | SamplerLaw]:
        """Energy, move, and sampler, as the compiler reads them."""
        return [self.energy, self.move] if self.sampler is None else [self.energy, self.move, self.sampler]


@dataclass(frozen=True)
class ExtremalLaw(BaseLaw):
    """Law that applies a rule to the site with the smallest or largest value of a key state and to its neighbours.

    Every application finds the extremal site, with ties broken by the simulation's random stream, gathers the site
    and its real neighbours without duplicates, the site first, and calls `f(*values, random, **parameters)` with one
    array per target and a uniform random array of the same length. The rule returns the new values, or a tuple with
    one array per target. Every application is recorded as an event with its time, its site, and the extremal value.

    Args:
        target:
            The symbol of a lattice state, or a tuple of them that the rule updates together.

        over:
            The neighbourhood of the extremal site that the rule also updates.

        rule:
            The function `f(*values, random, **parameters)`.

        key:
            The lattice state whose extremum is searched, or `None` for the first target.

        extremum:
            `'min'` for the smallest value or `'max'` for the largest.

        repeat:
            The number of applications per step.

        label:
            The name of the events, or `None` for the component's path.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.formalisms.lattices import ExtremalLaw
        >>> from skcomplex.core.space import Neighborhood
        >>> law = ExtremalLaw(symbol('fitness'), Neighborhood.radius(), lambda values, *, random: random)
        >>> law.extremum, law.repeat
        ('min', 1)
    """

    law_kind: ClassVar[LawKind] = LawKind.UPDATE
    time_class: ClassVar[LawClass] = LawClass.DISCRETE
    traits: ClassVar[frozenset[Trait]] = frozenset({Trait.LATTICE, Trait.KERNEL_FREE, Trait.DECLARABLE})
    requires: ClassVar[Requires] = Requires(time={TimeKind.DISCRETE}, space={SpaceKind.LATTICE, SpaceKind.NETWORK})

    target: Expression | tuple[Expression, ...]
    over: Neighborhood | Adjacency | Memory
    rule: Callable[..., Any]
    key: Expression | None = None
    extremum: str = 'min'
    repeat: int = 1
    label: str | None = None

    def _check(self) -> Self:
        """Return the law with its values checked and normalized, raising a named error for an invalid one."""
        changes: dict[str, Any] = {}
        if self.extremum not in EXTREMA:
            error_msg = f'Unknown extremum {self.extremum!r}. Use one of {", ".join(EXTREMA)}.'
            raise ValueError(error_msg)
        if int(self.repeat) < 1:
            error_msg = f'An extremal law applies at least once per step, but its repeat is {self.repeat}.'
            raise ValueError(error_msg)
        return replace(self, **changes) if changes else self

    @property
    def targets(self) -> tuple[Expression, ...]:
        """Target symbols."""
        return _targets(self.target)

    @property
    def key_state(self) -> Expression:
        """Symbol of the lattice state whose extremum is searched."""
        return self.targets[0] if self.key is None else self.key
