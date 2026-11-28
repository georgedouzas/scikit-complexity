"""Define the relaxation of a state, driven once per step and relaxed until stable."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from sage.all__sagemath_symbolics import SR, Expression, var

from ....errors import ValidationError
from .._base import BaseLaw
from ._operators import _OperatorFunction

RELAXATIONS: dict[str, _Relaxation] = {}


@dataclass(frozen=True)
class _Relaxation:
    """Relaxation that a call of `relax` names.

    Args:
        rules:
            The laws applied in turn until the state is stable.

        stable:
            The function of the arrays of the targets of the rules that returns whether the state is stable.

        drive:
            The laws applied once per step before the relaxation.

        activity:
            The function of the arrays of the targets that marks the active members of an iteration, or `None` to count
            the members that change.

        max_iterations:
            The largest number of iterations in one step.
    """

    rules: tuple[BaseLaw, ...]
    stable: Callable[..., bool]
    drive: tuple[BaseLaw, ...] = ()
    activity: Callable[..., object] | None = None
    max_iterations: int = 100_000


class _RelaxFunction(_OperatorFunction):
    """Relaxation of a state, printed with the state alone."""

    def _print_(self, state: Expression, relaxation: Expression) -> str:
        """Return the relaxation call with the state it relaxes."""
        del relaxation
        return f'relax({state})'


RELAX = _RelaxFunction('relax', nargs=2)


def relax(
    state: object,
    rules: BaseLaw | Sequence[BaseLaw],
    stable: Callable[..., bool],
    *,
    drive: BaseLaw | Sequence[BaseLaw] = (),
    activity: Callable[..., object] | None = None,
    max_iterations: int = 100_000,
) -> Expression:
    """Return a state after a push and the rules that settle it until it is stable.

    In one step of an `UpdateLaw`, the `drive` laws run once, then the `rules` run in turn again and again until
    `stable` holds, so a slow push and the fast cascade it starts fit in one step, such as
    `relax(height, topple, stable, drive=grain)` for a sandpile that gets one grain per step. Every cascade that changes
    the state is recorded as an event under the label of the update law, with its size, the number of member updates,
    its duration, the number of rounds, its area, the number of distinct members updated, and its profile, the member
    updates of every round, which sums to the size. A member update is a member that changes in a round, or one that
    `activity` marks as active. The profiles are the shapes of the avalanches that `events(records, kind='excursions')`
    returns, and `events(records, kind='activity')` joins them into the activity of every round in fast time. It gives
    the avalanches of a sandpile, the spread of a forest fire after lightning, and the slips along a fault in an
    earthquake.

    Read more in the user guide, under Laws.

    Args:
        state:
            The symbol of the state on a lattice that the update law sets.

        rules:
            The law, or the laws applied in turn, that relax the state, such as local rules or array updates.

        stable:
            A function of the arrays of the targets of the rules, in lattice shape, that returns whether the state is
            stable.

        drive:
            The law, or the laws, applied once per step before the relaxation, such as local rules or array updates.

        activity:
            A function of the arrays of the targets that returns a boolean array of the active members of a round, or
            `None` to count the members that change.

        max_iterations:
            The largest number of rounds in one step.

    Returns:
        relaxation:
            The unevaluated relaxation, the value of an `UpdateLaw` of the state.

    Raises:
        ValidationError:
            No rule settles the state, or a rule or a drive is not a law.

    See Also:
        UpdateLaw: Sets the state to the relaxation once per step.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws import UpdateLaw
        >>> from skcomplex.core.components.laws.utils import relax
        >>> burn = UpdateLaw(var('fire'), lambda fire, **kwargs: fire, arrays=True)
        >>> relax(var('fire'), burn, lambda fire: True)
        relax(fire)
    """
    rules = (rules,) if isinstance(rules, BaseLaw) else tuple(rules)
    drive = (drive,) if isinstance(drive, BaseLaw) else tuple(drive)
    if not rules:
        error_msg = 'A relaxation has no rules to settle the state with. Give it an UpdateLaw.'
        raise ValidationError(error_msg)
    if not all(isinstance(law, BaseLaw) for law in (*rules, *drive)):
        error_msg = 'The rules and the drive of a relaxation must be laws, such as an UpdateLaw.'
        raise ValidationError(error_msg)
    relaxation = _Relaxation(rules, stable, drive, activity, int(max_iterations))
    name = f'_relaxation_{id(relaxation)}'
    RELAXATIONS[name] = relaxation
    return RELAX(SR(state), var(name))


def relaxation_of(value: object) -> _Relaxation | None:
    """Return the settings of the relaxation an update value calls, or `None`.

    It reads back what `relax` stored, the rules, the stability test, the drive, the activity, and the largest number of
    rounds, so the compiler can run the cascade of a step, and gives `None` for a value that is no relaxation, such as
    `fire + 1`. It finds the cascades of a sandpile, a forest fire, and an earthquake.

    Read more in the user guide, under Extend the library.

    Args:
        value:
            The value of an update law.

    Returns:
        relaxation:
            The settings of the relaxation, or `None`.

    See Also:
        relax: Builds the relaxations it reads.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils._relaxations import relaxation_of
        >>> relaxation_of(var('fire') + 1) is None
        True
    """
    if not isinstance(value, Expression) or value.operator() != RELAX:
        return None
    return RELAXATIONS[str(value.operands()[1])]
