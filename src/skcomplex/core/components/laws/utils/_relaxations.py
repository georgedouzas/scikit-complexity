"""Define the relaxation of a state, driven once per step and relaxed until stable."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence

from sage.all__sagemath_symbolics import SR, Expression, var

from ....errors import ValidationError
from .._base import BaseLaw
from .._relaxation_spec import _RelaxationSpec
from ._operators import _OperatorFunction

RELAXATIONS: dict[str, _RelaxationSpec] = {}


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
    """Return a state after a drive and the rules that relax it until it is stable.

    It is the fixed point of a relaxation map `R` after a drive `D`, `R(R(...R(D(x))))`, which an `UpdateLaw` of the
    state applies once per step. The drive laws run once, then the rules run in turn again and again until `stable`
    holds, so one step holds a whole cascade and the slow drive is apart from the fast relaxation. Every cascade that
    changes the state is recorded as an event, named by the label of the update law, with its size, the number of
    member updates, its duration, the number of iterations, and its area, the number of distinct members updated. A
    member update is a member that changes in an iteration, or a member that `activity` marks as active before it.

    It appears in many fields. Grains topple in the sandpile of Bak, Tang, and Wiesenfeld, fire spreads through a
    forest after a lightning strike, slips spread along a fault in the earthquake model of Olami, Feder, and
    Christensen, and neurons fire in avalanches when their potentials cross a threshold.

    Args:
        state:
            The symbol of the lattice state the update law sets.

        rules:
            The law, or the laws applied in turn, that relax the state, such as local rules or array updates.

        stable:
            A function of the arrays of the targets of the rules, in lattice shape, that returns whether the state is
            stable.

        drive:
            The law, or the laws, applied once per step before the relaxation, such as local rules or array updates.

        activity:
            A function of the arrays of the targets that returns a boolean array of the active members of an
            iteration, or `None` to count the members that change.

        max_iterations:
            The largest number of iterations in one step.

    Returns:
        relaxation:
            The unevaluated relaxation, the value of an `UpdateLaw` of the state.

    Raises:
        ValidationError:
            No rule relaxes the state, or a rule or a drive is not a law.

    Examples:
        >>> import numpy as np
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws import LocalRuleLaw, UpdateLaw
        >>> from skcomplex.core.components.laws.utils import relax
        >>> from skcomplex.core.components.spaces import RadiusNeighborhood
        >>> cross = RadiusNeighborhood(1, metric='manhattan', center=True)

        Grains topple in a sandpile, one grain added per step:

        >>> height = var('height')
        >>> topple = LocalRuleLaw(height, cross, lambda z, *n: z - 4 * (z >= 4) + sum(m >= 4 for m in n))
        >>> grain = UpdateLaw(height, lambda z, *, random: z + np.eye(1, z.size).reshape(z.shape), arrays=True)
        >>> pile = relax(height, topple, lambda z: bool((z < 4).all()), drive=grain)
        >>> UpdateLaw(height, pile, label='avalanche').value
        relax(height)

        Fire spreads to the burning neighbours of trees until no tree catches fire:

        >>> fire = var('fire')
        >>> burn = LocalRuleLaw(fire, cross, lambda f, *n: np.maximum(f, np.max(n, axis=0)))
        >>> relax(fire, burn, lambda f: True)
        relax(fire)
    """
    rules = (rules,) if isinstance(rules, BaseLaw) else tuple(rules)
    drive = (drive,) if isinstance(drive, BaseLaw) else tuple(drive)
    if not rules:
        error_msg = 'A relaxation has no rules to relax the state with. Give it a local rule or an array update.'
        raise ValidationError(error_msg)
    if not all(isinstance(law, BaseLaw) for law in (*rules, *drive)):
        error_msg = 'The rules and the drive of a relaxation are laws, such as local rules or array updates.'
        raise ValidationError(error_msg)
    spec = _RelaxationSpec(rules, stable, drive, activity, int(max_iterations))
    name = f'_relaxation_{id(spec)}'
    RELAXATIONS[name] = spec
    return RELAX(SR(state), var(name))


def relaxation_of(value: object) -> _RelaxationSpec | None:
    """Return the relaxation an update value calls, or `None` for another value."""
    if not isinstance(value, Expression) or value.operator() != RELAX:
        return None
    return RELAXATIONS[str(value.operands()[1])]
