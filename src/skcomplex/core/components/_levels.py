"""Convert between the members of a group and a distribution over a grid of activity values held by a site group."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import dataclasses

import numpy as np
import numpy.typing as npt

from ..base import ValidationError, VariableKind, declarations_of
from ._components import Element, Group

type Values = npt.NDArray[np.float64]


def _grid(activities: object) -> Values:
    """Return the activity values as an increasing float vector, after checking that they are one."""
    grid = np.asarray(activities, dtype=float)
    if grid.ndim != 1 or not grid.size or np.any(np.diff(grid) <= 0):
        error_msg = f'The activities of a distribution are a nonempty increasing vector, not {activities!r}.'
        raise ValidationError(error_msg)
    return grid


def _state_name(element: type[Element], state: str | None) -> str:
    """Return the name of a state of an element, the only state when no name is given."""
    states = declarations_of(element, VariableKind.STATE)
    if state is None and len(states) == 1:
        return next(iter(states))
    if state in states:
        return state
    error_msg = (
        f'{element.__name__} declares the states {", ".join(states) or "none"}. Name the state that holds the values, '
        f'not {state!r}.'
    )
    raise ValidationError(error_msg)


def _member_values(group: Group | npt.ArrayLike, state: str) -> Values:
    """Return the values of a state of every member of a group, or the values themselves."""
    if not isinstance(group, Group):
        return np.ravel(np.asarray(group, dtype=float))
    name = _state_name(group.element, state)
    value = (group.values or {}).get(name)
    if value is None:
        value = next(item.default for item in dataclasses.fields(group.element) if item.name == name)
    return np.broadcast_to(np.asarray(value, dtype=float), (group.size or 0,)).copy()


def distribution_from(
    group: Group | npt.ArrayLike,
    state: str,
    activities: npt.ArrayLike,
    *,
    element: type[Element],
    label: str | None = None,
    density: str | None = None,
) -> Group:
    """Return a distribution site group that bins the values of a state of the members of a group over activities.

    Every member counts at the activity value nearest to its value, and the densities are the fractions of members at
    every activity, so the density of the distribution is 1. The values may come from the initial values of a group, or
    from a frame of a trajectory, such as `trajectory['traders.wealth'][-1]`.

    Args:
        group:
            The group whose members are binned, or the values of their state.

        state:
            The name of the members' state to bin.

        activities:
            The increasing activity values of the distribution, one per site.

        element:
            The element type of the distribution's members, which declares the density state.

        label:
            The label of the distribution group, or `None` for the group's label followed by `_distribution`.

        density:
            The name of the density state of `element`, or `None` for its only state.

    Returns:
        distribution:
            The group on the sites of an activity lattice, one member per activity, with the fractions as densities.

    Raises:
        ValidationError:
            The activities are not an increasing vector, the group has no member, or a state is not declared.

    Examples:
        >>> from skcomplex.core.base import state
        >>> from skcomplex.core.components import Element, distribution_from
        >>> class Density(Element):
        ...     f: float = state(default=0.0)
        >>> distribution = distribution_from([0.0, 1.0, 1.0, 2.0], 'wealth', [0.0, 1.0, 2.0], element=Density)
        >>> distribution.values['f'].tolist()
        [0.25, 0.5, 0.25]
    """
    grid = _grid(activities)
    values = _member_values(group, state)
    if not values.size:
        error_msg = 'A distribution bins the members of a group, but the group has no member.'
        raise ValidationError(error_msg)
    nearest = np.clip(np.searchsorted(grid, values), 1, grid.size - 1) if grid.size > 1 else np.zeros(values.size)
    nearest = np.asarray(nearest, dtype=np.intp)
    if grid.size > 1:
        nearest -= np.abs(values - grid[nearest - 1]) <= np.abs(values - grid[nearest])
    fractions = np.bincount(nearest, minlength=grid.size) / values.size
    name = label if label is not None else f'{getattr(group, "label", "members")}_distribution'
    return Group(name, element=element, size=grid.size, on='sites', values={_state_name(element, density): fractions})


def members_from(
    distribution: Group | npt.ArrayLike,
    element: type[Element],
    n: int,
    *,
    activities: npt.ArrayLike,
    state: str | None = None,
    label: str | None = None,
    density: str | None = None,
    random_state: int | np.random.Generator | None = None,
) -> Group:
    """Return a variable group of members whose state values are drawn from a distribution over activities.

    Every member takes the activity of a site drawn with probability proportional to the site's density.

    Args:
        distribution:
            The distribution group, or its densities, one per activity.

        element:
            The element type of the members.

        n:
            The number of members.

        activities:
            The activity values of the distribution, one per site.

        state:
            The name of the members' state that takes the activity, or `None` for the only state of `element`.

        label:
            The label of the group, or `None` for the distribution's label followed by `_members`.

        density:
            The name of the density state of the distribution's element, or `None` for its only state.

        random_state:
            The seed or the generator of the draws.

    Returns:
        group:
            The variable group of `n` members.

    Raises:
        ValidationError:
            The activities are not an increasing vector, the densities do not match them, or are negative or all zero.

    Examples:
        >>> from skcomplex.core.base import state
        >>> from skcomplex.core.components import Element, members_from
        >>> class Trader(Element):
        ...     wealth: float = state(default=0.0)
        >>> traders = members_from([0.0, 1.0, 0.0], Trader, 4, activities=[0.0, 1.0, 2.0], random_state=0)
        >>> traders.values['wealth'].tolist(), traders.variable
        ([1.0, 1.0, 1.0, 1.0], True)
    """
    grid = _grid(activities)
    if isinstance(distribution, Group):
        weights = _member_values(distribution, _state_name(distribution.element, density))
    else:
        weights = np.ravel(np.asarray(distribution, dtype=float))
    if weights.shape != grid.shape or np.any(weights < 0) or not weights.sum() > 0:
        error_msg = (
            f'A distribution gives one nonnegative density per activity, not all zero, but got {weights.tolist()} '
            f'for {grid.size} activities.'
        )
        raise ValidationError(error_msg)
    rng = np.random.default_rng(random_state)
    drawn = grid[rng.choice(grid.size, size=n, p=weights / weights.sum())]
    name = label if label is not None else f'{getattr(distribution, "label", "distribution")}_members'
    return Group(name, element=element, size=n, values={_state_name(element, state): drawn}, variable=True)
