"""Define the law that gives the next value of a state after one discrete step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from dataclasses import KW_ONLY, dataclass
from types import SimpleNamespace
from typing import ClassVar

from sage.all__sagemath_symbolics import Expression
from sage.modules.free_module_element import FreeModuleElement

from ...errors import ValidationError
from ..spaces import BaseNeighborhood
from ..spaces._discrete_time import _DiscreteTime
from ..types import Target
from ._base import BaseLaw
from ._call_spec import CallSpec
from ._local_rule import SCHEDULES, _LocalRuleLaw
from ._requirement_spec import RequirementSpec
from ._topology_request_spec import TopologyRequestSpec


@dataclass(frozen=True)
class UpdateLaw(BaseLaw):
    """A law that gives what a value becomes after one step of discrete time.

    It says what a value becomes at the next step, such as `UpdateLaw(s.insects.x, s.insects.r * s.insects.x * (1 -
    s.insects.x))` inside the interaction of a logistic map, where a population of insects grows and crowds itself out
    from year to year. Every update reads the values and the time at the start of the step, so the order of the laws
    does not matter. A value takes one update law, unless each of its laws says how it combines with the others, by `increment`
    or by `source`. With `neighborhood`, the law is a local rule that reads the neighbours of every cell of a lattice or
    node of a network, such as the Game of Life, and with `arrays`, a function of the whole array changes every member
    at once. Its target may also be a group, whose members each run a rule `rule(agent, system)` that may add and
    remove members, or a `TopologyRequestSpec`, whose value is the chance that the network changes. A population breeds
    from year to year, a cellular automaton steps its cells, an account earns interest every month, and a player
    revises a strategy every round.

    Read more in the user guide, under Laws.

    Args:
        target:
            The symbol of a scalar value, the column of symbols of a vector value, a `TopologyRequestSpec`, whose
            value is the chance that the request fires, the group of a role, whose members run a rule, or a tuple of
            values that a local rule sets together.

        value:
            A symbolic expression, a `CallSpec`, or a function `f(t, values)` that gives the next value, the rule
            `rule(agent, system)` of the members of a group, the rule of the cells of a local rule, or with `arrays`
            a NumPy function `f(values, **kwargs)` of the whole array of the value. The function of a local rule or of
            a whole array, and its score, receive the same keywords: every parameter of the element by name, `time`,
            the time of the step, `space`, the space of the value, `degree`, the number of neighbours of every cell of
            a local rule and `None` otherwise, and `random`. They take `**kwargs` for the keywords they ignore.

        label:
            The name of the law, or the empty string for a law without one.

        increment:
            Whether the value is the value plus a change, such as `x + 1`, whose change adds to the changes of the
            other update laws of the same value.

        source:
            The value the law moves members from, such as `0.0` for the healthy people of an epidemic, keeping every
            other value as it is, or `None`. Laws of one value with different sources add their changes.

        order:
            A symbolic expression of the values, parameters, and draws of a member, such as a random number, whose
            values order the members that update one after the other, each reading the values set before it, or
            `None` for members that update at once.

        phase:
            The name of the phase of the step the law runs in, which a `PhasesScheduleSpec` of the discrete time
            orders, so each phase reads the values the phases before it set, or `None` for the default phase.

        neighborhood:
            The ordered neighbours of every cell that a local rule reads, such as
            `RadiusNeighborhood(1, metric='chebyshev')`, or `None` for a law of one member. The value is then a
            `TableSpec`, a `TotalisticSpec`, an `OuterTotalisticSpec`, a `LinearRuleSpec`, an object whose
            `get_table()` returns a `TableSpec`, or a function `f(*neighbours, **kwargs)`. A function receives one
            array per neighbour of every value it reads, in the order of the neighbourhood, and returns the next
            values of the target, or one array per value when the target is a tuple of values.

        reads:
            The values whose neighbours a local rule reads, or `None` for its targets.

        schedule:
            The order in which a local rule updates the cells, or `None` for all at once. `'synchronous'` updates
            every cell from the previous values, `'random-sequential'` one cell at a time in a random order,
            `'random-site'` as many cells chosen at random, `'checkerboard'` the two halves of a square lattice in
            turn, like the squares of a chessboard, and `'colored'` groups of cells with no neighbours in common, in
            turn.

        score:
            A function `score(values, **kwargs)` of the whole array of a lattice value, which returns one number per
            row and which a run records at every frame as `<owner>.score`, or `None`. Only an update of the whole
            array takes it, and it never draws numbers.

        arrays:
            Whether the value is a NumPy function of the whole array of the value, shaped like the lattice or with
            one row per member, which replaces the value of every member at once.

        stochastic:
            Whether the function of a local rule or of a whole array draws numbers. Its keyword `random` is then a
            uniform number per cell of a local rule, or the random generator of the run for a whole array, and `None`
            otherwise.

    Raises:
        ValidationError:
            The value is a NumPy array, an increment law has a source, a local rule has a neighbourhood that is no
            neighbourhood, an unknown schedule, or a setting of a law of one member, a law has a score but no function
            of the whole array, `arrays` is given with a value that is no function or a target that is no value, or
            `stochastic` is given with a value that is no function of a local rule or of a whole array.

    See Also:
        RateLaw: Adds a rate to a derivative of a value in continuous time.
        TopologyRequestSpec: A change of a network that an update law may request.
        TotalisticSpec: A local rule given by the sum of the neighbours.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws import UpdateLaw
        >>> x, r = var('x r')
        >>> law = UpdateLaw(x, r * x * (1 - x))
        >>> float(law.value.subs(x=0.5, r=3))
        0.75
    """

    law_tag: ClassVar[str] = 'update'

    target: Target | tuple[Target, ...] | TopologyRequestSpec | SimpleNamespace
    value: object
    _: KW_ONLY
    label: str = ''
    increment: bool = False
    source: float | None = None
    order: object = None
    phase: str | None = None
    neighborhood: BaseNeighborhood | None = None
    reads: Sequence[Target] | None = None
    schedule: str | None = None
    score: Callable[..., object] | None = None
    arrays: bool = False
    stochastic: bool = False

    def _get_targets(self) -> tuple[object, ...]:
        """Return the state the update sets, or the states a local rule sets together."""
        if self.neighborhood is not None and isinstance(self.target, tuple):
            return self.target
        return (self.target,)

    @classmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what the law requires of a system: discrete time."""
        return RequirementSpec(time=(_DiscreteTime,))

    def _get_local_rule(self) -> _LocalRuleLaw:
        """Return the local rule that the engine runs for a law with a neighbourhood.

        A stochastic law gives a function value a uniform number per cell as `random`.

        Returns:
            rule:
                The local rule of the law.

        Raises:
            ValidationError:
                The law has no neighbourhood.
        """
        if self.neighborhood is None:
            error_msg = f'The update law {self.label!r} has no neighbourhood, so it is no local rule.'
            raise ValidationError(error_msg)
        return _LocalRuleLaw(
            self.target,
            self.neighborhood,
            self.value,
            reads=None if self.reads is None else tuple(self.reads),
            schedule=self.schedule or 'synchronous',
            stochastic=self.stochastic,
            label=self.label,
        )

    def _check_local_rule(self) -> None:
        """Raise an error for a bad neighbourhood or schedule of a local rule.

        A setting of a law of one member, such as `phase`, is an error too.
        """
        if self.neighborhood is None:
            if self.reads is not None or self.schedule is not None:
                error_msg = (
                    f'The update law {self.label!r} has `reads` or `schedule` but no neighbourhood. Only a local rule '
                    "takes them, such as UpdateLaw(s.target.x, rule, neighborhood=MOORE, schedule='checkerboard')."
                )
                raise ValidationError(error_msg)
            return
        if not isinstance(self.neighborhood, BaseNeighborhood):
            error_msg = (
                f'The neighbourhood of the update law {self.label!r} is {self.neighborhood!r}. Give a neighbourhood, '
                'such as RadiusNeighborhood(1).'
            )
            raise ValidationError(error_msg)
        if self.schedule is not None and self.schedule not in SCHEDULES:
            error_msg = f'Unknown schedule {self.schedule!r}. Use one of {", ".join(SCHEDULES)}.'
            raise ValidationError(error_msg)
        settings = {'increment': self.increment, 'source': self.source, 'order': self.order, 'phase': self.phase}
        given = [name for name, item in settings.items() if item is not None and item is not False]
        if given:
            error_msg = (
                f'The update law {self.label!r} has a neighbourhood and {", ".join(given)}, which only a law of one '
                'member takes. A local rule sets every cell from its neighbours at once or by its schedule.'
            )
            raise ValidationError(error_msg)

    def _check(self) -> None:
        """Raise an error for an array value, an increment with a source, or a bad rule.

        Raises:
            ValidationError:
                The value is a NumPy array, an increment law has a source, or a local rule has a neighbourhood that
                is no neighbourhood, an unknown schedule, or a setting of a law of one member.
        """
        super()._check()
        self._check_array('value')
        if self.increment and self.source is not None:
            error_msg = (
                f'The update law {self.label!r} is an increment with the source {self.source!r}. An increment changes '
                'every value, so give it no source.'
            )
            raise ValidationError(error_msg)
        self._check_local_rule()
        self._check_arrays()

    def _check_arrays(self) -> None:
        """Raise an error for a score, `arrays`, or `stochastic` on a law of one value."""
        function = callable(self.value) and not isinstance(self.value, Expression | CallSpec)
        if self.stochastic and (not function or (not self.arrays and self.neighborhood is None)):
            error_msg = (
                f'The update law {self.label!r} is stochastic, which only a function of a local rule or of a whole '
                f'array takes, but its value is {self.value!r}. Give one, such as '
                'UpdateLaw(s.target.height, drop, arrays=True, stochastic=True).'
            )
            raise ValidationError(error_msg)
        if self.score is not None and not self.arrays:
            error_msg = (
                f'The update law {self.label!r} has a score, which only an update of a whole array by a function '
                'takes, such as UpdateLaw(s.target.s, accept, arrays=True, score=energy).'
            )
            raise ValidationError(error_msg)
        if not self.arrays:
            return
        state = isinstance(self.target, Expression | FreeModuleElement)
        if not state or not function or self.neighborhood is not None:
            error_msg = (
                f'The update law {self.label!r} has `arrays=True`, which takes a function of the whole array of a '
                f'value and no neighbourhood, but its target is {self.target!r} and its value {self.value!r}. Give '
                'one, such as UpdateLaw(s.target.height, drop, arrays=True) with def drop(values, **kwargs): ...'
            )
            raise ValidationError(error_msg)

    def _get_value(self, inputs: object) -> object:
        """Return the next value at given values of the symbols it reads.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value, and for a local rule `space`, the
                space of the cells.

        Returns:
            value:
                The next value.
        """
        if self.neighborhood is not None:
            rule = self._get_local_rule()
            rule._check()
            return rule._get_value(inputs)
        return self._evaluate(self.value, inputs)
