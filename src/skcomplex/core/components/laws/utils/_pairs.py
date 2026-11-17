"""Define the pairs of agents that rules act on."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Iterator, Sequence
from typing import cast

import numpy as np

from ....errors import ValidationError
from ...types import AgentViewProtocol, Feasibility, Pairing
from .._partner_spec import PartnerSpec


def _read_state(agent: AgentViewProtocol, name: str, setting: str) -> float:
    """Return the value of a state that a setting of a selection of partners names."""
    if not hasattr(agent, name):
        error_msg = (
            f'The partners read the {setting} from the state {name!r}, which the agents of {agent.group!r} do not '
            'have. Declare the state on their element, or give a number.'
        )
        raise ValidationError(error_msg)
    return float(getattr(agent, name))


def _get_room(spec: PartnerSpec, agent: AgentViewProtocol) -> float:
    """Return the largest number of pairs of an agent by the capacity of a selection."""
    if spec.capacity is None:
        return np.inf
    if not isinstance(spec.capacity, str):
        return float(spec.capacity)
    return _read_state(agent, spec.capacity, 'capacity')


def _has_room(spec: PartnerSpec, agent: AgentViewProtocol, taken: dict[int, int]) -> bool:
    """Return whether an agent holds fewer pairs of the step than its capacity."""
    return taken.get(agent.id, 0) < _get_room(spec, agent)


def _is_free(
    spec: PartnerSpec,
    agent: AgentViewProtocol,
    target: AgentViewProtocol,
    taken: dict[int, int],
    feasible: Feasibility | None,
) -> bool:
    """Return whether a target has room and the feasibility admits it for an agent."""
    return _has_room(spec, target, taken) and (feasible is None or feasible(agent, target))


def _locate(targets: Sequence[AgentViewProtocol]) -> dict[object, list[AgentViewProtocol]]:
    """Return the agents by the key of the cell or node they stand in."""
    located: dict[object, list[AgentViewProtocol]] = {}
    for target in targets:
        located.setdefault(target.location_key, []).append(target)
    return located


def _get_candidates(
    spec: PartnerSpec,
    agent: AgentViewProtocol,
    targets: Sequence[AgentViewProtocol],
    located: dict[object, list[AgentViewProtocol]],
) -> list[AgentViewProtocol]:
    """Return the agents an agent considers, by the candidates of a selection."""
    if spec.candidates == 'all':
        return [target for target in targets if target is not agent]
    if spec.candidates == 'local':
        around = {agent.location_key, *agent.get_neighbor_cells()}
        return [target for target in targets if target is not agent and target.location_key in around]
    if spec.candidates in ('cell', 'neighbors'):
        keys = [agent.location_key] if spec.candidates == 'cell' else agent.get_neighbor_cells()
        return [target for key in keys for target in located.get(key, []) if target is not agent]
    radius = spec.candidates
    if isinstance(radius, str):
        radius = _read_state(agent, radius, 'radius')
    return [target for target in targets if target is not agent and agent.get_distance_to(target) <= radius]


def _choose(
    spec: PartnerSpec,
    agent: AgentViewProtocol,
    free: list[AgentViewProtocol],
    rng: np.random.Generator,
) -> list[AgentViewProtocol]:
    """Return the partners an agent picks among its candidates with room."""
    if spec.choice == 'all':
        return free
    if spec.choice == 'uniform':
        return [free[int(rng.integers(len(free)))]]
    if spec.choice == 'nearest':
        return [min(free, key=agent.get_distance_to)]
    if spec.score is None:
        error_msg = 'The choice `score` of partners has no score. Give a function of the agent and the candidate.'
        raise ValidationError(error_msg)
    scores = np.asarray([float(spec.score(agent, target)) for target in free], dtype=float)
    best = np.flatnonzero(scores == scores.max())
    return [free[int(best[int(rng.integers(len(best)))])]]


def _get_free(
    spec: PartnerSpec,
    agent: AgentViewProtocol,
    right: Sequence[AgentViewProtocol],
    located: dict[object, list[AgentViewProtocol]],
    taken: dict[int, int],
    *,
    feasible: Feasibility | None,
) -> list[AgentViewProtocol]:
    """Return the feasible candidates of an agent with room.

    Without any, it returns the nearest such agent under the fallback.
    """
    candidates = _get_candidates(spec, agent, right, located)
    free = [item for item in candidates if _is_free(spec, agent, item, taken, feasible)]
    if free or spec.fallback != 'nearest':
        return free
    others = [item for item in right if item is not agent and _is_free(spec, agent, item, taken, feasible)]
    return [min(others, key=agent.get_distance_to)] if others else []


def _get_partners(
    spec: PartnerSpec,
    left: Sequence[AgentViewProtocol],
    right: Sequence[AgentViewProtocol],
    rng: np.random.Generator,
    feasible: Feasibility | None,
) -> Iterator[tuple[AgentViewProtocol, AgentViewProtocol]]:
    """Yield the pairs of every agent of one side with the partners it picks."""
    within = left is right
    order = range(len(left)) if spec.capacity is None and feasible is None else rng.permutation(len(left))
    located = _locate(right) if spec.candidates in ('cell', 'neighbors') else {}
    taken: dict[int, int] = {}
    for index in order:
        agent = left[int(index)]
        if within and not _has_room(spec, agent, taken):
            continue
        free = _get_free(spec, agent, right, located, taken, feasible=feasible)
        for target in _choose(spec, agent, free, rng) if free else []:
            if within and not _has_room(spec, agent, taken):
                break
            taken[target.id] = taken.get(target.id, 0) + 1
            if within:
                taken[agent.id] = taken.get(agent.id, 0) + 1
            yield agent, target


def _get_disjoint_pairs(
    left: Sequence[AgentViewProtocol],
    right: Sequence[AgentViewProtocol],
    rng: np.random.Generator,
) -> list[tuple[AgentViewProtocol, AgentViewProtocol]]:
    """Return disjoint random pairs within one group or across two groups."""
    first = [left[int(index)] for index in rng.permutation(len(left))]
    if left is right:
        return list(zip(first[::2], first[1::2], strict=False))
    second = [right[int(index)] for index in rng.permutation(len(right))]
    return list(zip(first, second, strict=False))


def _get_random_pairs(
    left: Sequence[AgentViewProtocol],
    right: Sequence[AgentViewProtocol],
    rng: np.random.Generator,
    capacity: int,
) -> list[tuple[AgentViewProtocol, AgentViewProtocol]]:
    """Return pairs drawn uniformly at random.

    Each member is in at most `capacity` pairs and never paired with itself.
    """
    found: list[tuple[AgentViewProtocol, AgentViewProtocol]] = []
    seen: set[tuple[int, int]] = set()
    if left is right:
        for _ in range(capacity):
            order = rng.permutation(len(left))
            for first, second in zip(order[::2], order[1::2], strict=False):
                key = (min(first, second), max(first, second))
                if key not in seen:
                    seen.add(key)
                    found.append((left[first], right[second]))
        return found
    firsts = rng.permutation(np.repeat(np.arange(len(left)), capacity))
    seconds = rng.permutation(np.repeat(np.arange(len(right)), capacity))
    for first, second in zip(firsts, seconds, strict=False):
        if (first, second) not in seen:
            seen.add((first, second))
            found.append((left[first], right[second]))
    return found


def pairs(
    left: Sequence[AgentViewProtocol],
    right: Sequence[AgentViewProtocol] | None,
    rng: np.random.Generator,
    pairing: Pairing | PartnerSpec | None = None,
    *,
    capacity: int = 1,
    feasible: Feasibility | None = None,
) -> Iterator[tuple[AgentViewProtocol, AgentViewProtocol]]:
    """Pair the agents of one or two groups, inside a rule.

    Workers meet firms in a labour market, buyers meet sellers in a bazaar, birds pair up to mate, and players meet
    opponents in a tournament. By default the members pair uniformly at random. A `PartnerSpec` picks partners within
    reach, by score, or among neighbours, and any other pairing, such as a stable matching or an auction, is a function
    `pairing(left, right, rng, capacity)` that returns the pairs.

    With `feasible`, the pairs come one at a time, so a rule can settle each before the next, and the agents act in
    random order, since the first to act take what the feasibility reads.

    Args:
        left:
            The agents of the first group, as a rule reads them with `system.get_agents(path)`.

        right:
            The agents of the second group, or `None` for pairs within the first.

        rng:
            The random generator, such as `system.rng`.

        pairing:
            A `PartnerSpec`, a function of both groups that returns their pairs, or `None` for uniform random pairs.

        capacity:
            The largest number of pairs of every member, for random pairs and for a function.

        feasible:
            The function `feasible(agent, candidate)` that says whether an agent can pair with a candidate now, or
            `None`. It needs a `PartnerSpec`.

    Returns:
        pairs:
            The pairs of agents, the first of every pair from the first group.

    Raises:
        ValidationError:
            The capacity is not a positive integer, the selection of partners is invalid or names a state the agents
            do not have, or a feasibility comes without a `PartnerSpec`.

    Examples:
        >>> import numpy as np
        >>> from types import SimpleNamespace
        >>> from skcomplex.core.components.laws.utils import pairs
        >>> rng = np.random.default_rng(0)
        >>> buyers = [SimpleNamespace(id=index) for index in range(3)]
        >>> sellers = [SimpleNamespace(id=index) for index in range(3, 5)]
        >>> len(list(pairs(buyers, sellers, rng)))
        2
        >>> birds = [SimpleNamespace(id=index) for index in range(6)]
        >>> len(list(pairs(birds, None, rng)))
        3
    """
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
        error_msg = f'The pairs have capacity {capacity!r}. Give a positive integer.'
        raise ValidationError(error_msg)
    right = left if right is None else right
    if feasible is not None:
        if not isinstance(pairing, PartnerSpec):
            error_msg = f'Pairs under a feasibility need a `PartnerSpec`, not {pairing!r}.'
            raise ValidationError(error_msg)
        pairing._check()
        return _get_partners(pairing, left, right, rng, feasible)
    if pairing is None:
        return iter(_get_random_pairs(left, right, rng, capacity))
    if not isinstance(pairing, PartnerSpec):
        return iter(cast(list[tuple[AgentViewProtocol, AgentViewProtocol]], pairing(left, right, rng, capacity)))
    pairing._check()
    if (pairing.candidates, pairing.choice, pairing.capacity, pairing.fallback) == ('all', 'uniform', 1, None):
        return iter(_get_disjoint_pairs(left, right, rng))
    return iter(list(_get_partners(pairing, left, right, rng, None)))
