"""Define the law that gives the next value of a state after one discrete step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import inspect
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
    """A law that gives the next value of a state after one discrete time step.

    A state takes one update law, unless each of its update laws says how it combines with the others. An `increment`
    law gives the state plus a change, and a law with a `source` changes only a state that holds the source value. Such
    laws of one state add their changes, `x + sum_k (f_k - x)`, all computed from the states at the start of the step.
    Two laws with the same source, or a law that declares neither, are an error.

    With `order`, the members of a group update one after the other, in increasing order of `order`, and each reads the
    values set before it. Without it every member updates at once. With `phase`, the law runs in that phase of the step,
    which a `PhasesScheduleSpec` of the discrete time orders, and each phase reads the states the phases before it set.

    An update law whose target is a group bound to a role and whose value is a function `rule(agent, system)` is an
    agent rule, such as `UpdateLaw(s.target, move)` with `def move(agent, system): agent.x += system.rng.normal()`.
    Every member of the group runs the rule at each step, and the rule may add and remove members.

    With `neighborhood`, the law is a local rule of the cells of a lattice or the nodes of a network. Its value reads
    the ordered values of the neighbours of every cell, such as `UpdateLaw(s.target.alive, life(), neighborhood=MOORE)`
    for the Game of Life. The value is then a `TableSpec`, a `TotalisticSpec`, an `OuterTotalisticSpec`, a
    `LinearRuleSpec`, an object whose `get_table()` returns a `TableSpec`, or a function. A function receives one array
    per neighbour of every read state, in neighbourhood order, and the parameters of the element by keyword. It also
    receives `random`, a uniform number per cell, and `degree`, the number of neighbours of every cell, when it names
    them. It returns the next values of the target, or one array per state when the target is a tuple of states.

    With `arrays=True`, the value is a NumPy function of the whole array that replaces the state of every member at
    once, such as `UpdateLaw(s.target.height, drop, arrays=True)` with `def drop(values, *, random): ...` adding a
    grain at one random site of a sandpile. The function receives the array, shaped like the lattice or with one row
    per member, and by keyword the parameters of the element and `random`, the generator, `time`, and `space`, when it
    names them. Without it, a function value is `f(t, values)` and gives the next value of one state.

    Unlike `RateLaw`, which gives a derivative in continuous time, it gives the next value in discrete time. Choose it
    when the model steps, such as a map, a generation, or a round.

    It appears in many fields. The logistic map follows a population from year to year, a cellular automaton steps its
    cells, and an account compounds interest monthly.

    Args:
        target:
            The symbol of a scalar state, the column of symbols of a vector state, a `TopologyRequestSpec`, whose
            value is the probability that the request fires, the group of a role, whose members run a rule, or a tuple
            of states that a local rule sets together.

        value:
            A symbolic expression, a `Call`, or a callable `f(t, values)` that gives the next value, the rule
            `rule(agent, system)` of the members of a group, the rule of the cells of a local rule, or with `arrays`
            a NumPy function `f(values, *, random)` of the whole array of the state.

        label:
            The name of the law, or the empty string for a law without one.

        increment:
            Whether the value is the state plus a change that adds to the changes of the other update laws of the
            state.

        source:
            The value the law moves the state from, keeping every other value, or `None`.

        order:
            A symbolic expression of the states, parameters, and draws of a member, whose values order the members
            that update one after the other, or `None` for members that update at once.

        phase:
            The name of the phase of the step the law runs in, or `None` for the default phase.

        neighborhood:
            The ordered neighbours of every cell that a local rule reads, such as
            `RadiusNeighborhood(1, metric='chebyshev')`, or `None` for a law of one member.

        reads:
            The states whose neighbours a local rule reads, or `None` for its targets.

        schedule:
            The order in which a local rule updates the cells, or `None` for all at once. `'synchronous'` updates
            every cell from the previous values, `'random-sequential'` one cell at a time in a random order,
            `'random-site'` as many cells chosen at random, `'checkerboard'` the two sublattices of a square lattice in
            turn, and `'colored'` the classes of a proper coloring of the neighbour graph in turn.

        score:
            A function `score(values, **parameters)` of the whole array of a lattice state, which returns one number
            per row and which a run records at every frame as `<owner>.score`, or `None`. Only an update of the
            whole array takes it.

        arrays:
            Whether the value is a NumPy function of the whole array of the state, which replaces the state of every
            member at once.

    Raises:
        ValidationError:
            The value is a NumPy array, an increment law has a source, a local rule has a neighbourhood that is no
            neighbourhood, an unknown schedule, or a setting of a law of one member, a law has a score but no function
            of the whole array, or `arrays` is given with a value that is no function or a target that is no state.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws import UpdateLaw
        >>> x = var('x')
        >>> UpdateLaw(x, 4 * x * (1 - x)).target
        x
        >>> UpdateLaw(x, x + 1, increment=True).increment
        True
        >>> UpdateLaw(x, x + 1, phase='imitate').phase
        'imitate'

        A local rule reads the ordered neighbours of every cell:

        >>> from skcomplex.core.components.laws import TotalisticSpec
        >>> from skcomplex.core.components.spaces import RadiusNeighborhood
        >>> around = RadiusNeighborhood(1, center=True)
        >>> parity = UpdateLaw(x, TotalisticSpec(lambda total: total % 2), neighborhood=around)
        >>> parity.neighborhood.radius
        1

        A function of the whole array changes every member at once:

        >>> UpdateLaw(x, lambda values, *, random: values + 1, arrays=True).arrays
        True
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

    def _get_targets(self) -> tuple[object, ...]:
        """Return the state the update sets, or the states a local rule sets together."""
        if self.neighborhood is not None and isinstance(self.target, tuple):
            return self.target
        return (self.target,)

    @classmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what the law requires of a system: discrete time."""
        return RequirementSpec(time=(_DiscreteTime,))

    @staticmethod
    def _is_stochastic(value: object) -> bool:
        """Return whether a function names the keyword `random` in its signature."""
        if not callable(value):
            return False
        try:
            parameter = inspect.signature(value).parameters.get('random')
        except (TypeError, ValueError):
            return False
        return parameter is not None and parameter.kind in {parameter.KEYWORD_ONLY, parameter.POSITIONAL_OR_KEYWORD}

    def _get_local_rule(self) -> _LocalRuleLaw:
        """Return the local rule that the engine runs for a law with a neighbourhood.

        A function value that names `random` receives a uniform number per cell.

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
            stochastic=self._is_stochastic(self.value),
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
            error_msg = f'The neighbourhood of the update law {self.label!r} is {self.neighborhood!r}. Give one.'
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
        """Raise an error for a score or `arrays` on a law of one value."""
        if self.score is not None and not self.arrays:
            error_msg = (
                f'The update law {self.label!r} has a score, which only an update of a whole array by a function '
                'takes, such as UpdateLaw(s.target.s, accept, arrays=True, score=energy).'
            )
            raise ValidationError(error_msg)
        if not self.arrays:
            return
        state = isinstance(self.target, Expression | FreeModuleElement)
        function = callable(self.value) and not isinstance(self.value, Expression | CallSpec)
        if not state or not function or self.neighborhood is not None:
            error_msg = (
                f'The update law {self.label!r} has `arrays=True`, which takes a function of the whole array of a '
                f'state and no neighbourhood, but its target is {self.target!r} and its value {self.value!r}. Give '
                'one, such as UpdateLaw(s.target.height, drop, arrays=True) with def drop(values, *, random): ...'
            )
            raise ValidationError(error_msg)

    def _get_value(self, inputs: object) -> object:
        """Return the next value at given values of the symbols it reads.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value, and for a local rule `space`, the
                space of the cells.
        """
        if self.neighborhood is not None:
            rule = self._get_local_rule()
            rule._check()
            return rule._get_value(inputs)
        return self._evaluate(self.value, inputs)
