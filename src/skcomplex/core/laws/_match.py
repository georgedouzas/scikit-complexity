"""Define match laws, the pairings of the members of agent groups that the rules of a step read."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from collections.abc import Sequence

import numpy as np

from ...errors import ValidationError
from ...specs import PartnerSpec, RequirementSpec
from ..spaces import DiscreteTime
from ..types import AgentViewProtocol, ModelViewProtocol, Pairing
from ._base import BaseLaw
from ._rule import RuleLaw
from .utils import random_pairing


class MatchLaw(BaseLaw):
    """Pairing of the members of one agent group, or of two, every step, which the rules of the step read.

    The members are paired uniformly at random by default. Any other pairing, such as stable matching, an order book,
    or an auction, is a function given as `pairing`, and a selection of partners within reach or among neighbours is
    a `PartnerSpec`. Rules read the pairs of the step with `matches(model, label)`. Transfers, matchings, and groupings
    read their pairs from `get_pairs`.

    Args:
        left:
            The path of the first agent group.

        right:
            The path of the second agent group, or `None` for pairs within the first.

        pairing:
            A function `pairing(left, right, rng, capacity)` of the agents of both groups that returns their pairs, a
            `PartnerSpec`, or `None` for uniform random pairing.

        capacity:
            The largest number of pairs of every member.

        phase:
            The phase of the step the pairing runs in, before the rules of the phase that read it.

        label:
            The name of the pairs.

    Examples:
        >>> from skcomplex.core.laws import MatchLaw
        >>> MatchLaw('buyers', 'sellers', label='trades').check().capacity
        1
    """

    def __init__(
        self,
        left: str,
        right: str | None = None,
        pairing: Pairing | PartnerSpec | None = None,
        capacity: int = 1,
        phase: str | None = None,
        *,
        label: str = 'match',
    ) -> None:
        super().__init__(label)
        self.left = left
        self.right = right
        self.pairing = pairing
        self.capacity = capacity
        self.phase = phase

    def _get_targets(self) -> tuple[object, ...]:
        """Return no states, since a pairing stores pairs for the rules of the step."""
        return ()

    def _get_requirements(self) -> RequirementSpec:
        """Return what the pairing requires of a system: discrete time."""
        return RequirementSpec(time=(DiscreteTime,))

    def _get_rule(self) -> RuleLaw:
        """Return the rule the agent stepper runs, which stores the pairs of the step in the model."""
        return RuleLaw(self.run, phase=self.phase)

    def _check(self) -> None:
        """Raise an error for a capacity that is not a positive integer, or an invalid selection of partners."""
        if isinstance(self.pairing, PartnerSpec):
            self.pairing.check()
        if isinstance(self.capacity, bool) or not isinstance(self.capacity, int) or self.capacity < 1:
            error_msg = f'The match {self.label!r} has capacity {self.capacity!r}. Give a positive integer.'
            raise ValidationError(error_msg)

    @staticmethod
    def _get_room(spec: PartnerSpec, agent: AgentViewProtocol) -> float:
        """Return the largest number of pairs of an agent by the capacity of a selection of partners."""
        if spec.capacity is None:
            return np.inf
        if not isinstance(spec.capacity, str):
            return float(spec.capacity)
        return MatchLaw._read_state(agent, spec.capacity, 'capacity')

    @staticmethod
    def _read_state(agent: AgentViewProtocol, name: str, setting: str) -> float:
        """Return the value of a state that a setting of a selection of partners names."""
        if not hasattr(agent, name):
            error_msg = (
                f'The partners read the {setting} from the state {name!r}, which the agents of {agent.group!r} do not '
                'have. Declare the state on their element, or give a number.'
            )
            raise ValidationError(error_msg)
        return float(getattr(agent, name))

    @staticmethod
    def _locate(targets: Sequence[AgentViewProtocol]) -> dict[object, list[AgentViewProtocol]]:
        """Return the agents by the key of the cell or node they stand in."""
        located: dict[object, list[AgentViewProtocol]] = {}
        for target in targets:
            located.setdefault(target.location_key, []).append(target)
        return located

    @staticmethod
    def _get_candidates(
        spec: PartnerSpec,
        agent: AgentViewProtocol,
        targets: Sequence[AgentViewProtocol],
        located: dict[object, list[AgentViewProtocol]],
    ) -> list[AgentViewProtocol]:
        """Return the agents an agent considers, by the candidates of a selection of partners."""
        if spec.candidates == 'all':
            return [target for target in targets if target is not agent]
        if spec.candidates in ('cell', 'neighbors'):
            keys = [agent.location_key] if spec.candidates == 'cell' else agent.neighbor_cells()
            return [target for key in keys for target in located.get(key, []) if target is not agent]
        radius = spec.candidates
        if isinstance(radius, str):
            radius = MatchLaw._read_state(agent, radius, 'radius')
        return [target for target in targets if target is not agent and agent.distance_to(target) <= radius]

    @staticmethod
    def _choose(
        spec: PartnerSpec,
        agent: AgentViewProtocol,
        free: list[AgentViewProtocol],
        rng: np.random.Generator,
    ) -> list[AgentViewProtocol]:
        """Return the partners an agent picks among its candidates with room, by the choice."""
        if spec.choice == 'all':
            return free
        if spec.choice == 'uniform':
            return [free[int(rng.integers(len(free)))]]
        if spec.choice == 'nearest':
            return [min(free, key=agent.distance_to)]
        if spec.score is None:
            error_msg = 'The choice `score` of partners has no score. Give a function of the agent and the candidate.'
            raise ValidationError(error_msg)
        scores = np.asarray([float(spec.score(agent, target)) for target in free], dtype=float)
        best = np.flatnonzero(scores == scores.max())
        return [free[int(best[int(rng.integers(len(best)))])]]

    @staticmethod
    def _get_disjoint_pairs(
        left: Sequence[object],
        right: Sequence[object],
        rng: np.random.Generator,
    ) -> list[tuple[object, object]]:
        """Return disjoint random pairs, of two halves of one group or of a member of each of two groups."""
        first = [left[int(index)] for index in rng.permutation(len(left))]
        if left is right:
            return list(zip(first[::2], first[1::2], strict=False))
        second = [right[int(index)] for index in rng.permutation(len(right))]
        return list(zip(first, second, strict=False))

    def _has_room(self, spec: PartnerSpec, agent: AgentViewProtocol, taken: dict[int, int]) -> bool:
        """Return whether an agent holds fewer pairs of the step than its capacity."""
        return taken.get(agent.id, 0) < self._get_room(spec, agent)

    def _get_free(
        self,
        spec: PartnerSpec,
        agent: AgentViewProtocol,
        right: Sequence[AgentViewProtocol],
        located: dict[object, list[AgentViewProtocol]],
        taken: dict[int, int],
    ) -> list[AgentViewProtocol]:
        """Return the candidates of an agent with room, or else the nearest agent with room under the fallback."""
        candidates = self._get_candidates(spec, agent, right, located)
        free = [target for target in candidates if self._has_room(spec, target, taken)]
        if free or spec.fallback != 'nearest':
            return free
        others = [target for target in right if target is not agent and self._has_room(spec, target, taken)]
        return [min(others, key=agent.distance_to)] if others else []

    def _get_partners(
        self,
        spec: PartnerSpec,
        left: Sequence[AgentViewProtocol],
        right: Sequence[AgentViewProtocol],
        rng: np.random.Generator,
    ) -> list[tuple[object, object]]:
        """Return the pairs of every agent of one side with the partners it picks among the agents of the other."""
        within = left is right
        order = range(len(left)) if spec.capacity is None else rng.permutation(len(left))
        located = self._locate(right) if spec.candidates in ('cell', 'neighbors') else {}
        taken: dict[int, int] = {}
        pairs: list[tuple[object, object]] = []
        for index in order:
            agent = left[int(index)]
            if within and not self._has_room(spec, agent, taken):
                continue
            free = self._get_free(spec, agent, right, located, taken)
            for target in self._choose(spec, agent, free, rng) if free else []:
                if within and not self._has_room(spec, agent, taken):
                    break
                pairs.append((agent, target))
                taken[target.id] = taken.get(target.id, 0) + 1
                if within:
                    taken[agent.id] = taken.get(agent.id, 0) + 1
        return pairs

    def _run(self, inputs: object) -> None:
        """Store the pairs of the step in the storage of the model under `match:<label>`.

        Args:
            inputs:
                The model view the rules receive, with `agents(path)`, `storage`, and `rng`.

        Raises:
            ValidationError:
                The input is not a model view.
        """
        if not isinstance(inputs, ModelViewProtocol):
            error_msg = f'The match {self.label!r} runs on the model view of a step, but got {inputs!r}.'
            raise ValidationError(error_msg)
        left = inputs.agents(self.left)
        right = left if self.right is None else inputs.agents(self.right)
        pairs = self.get_pairs(left, right, inputs.rng)
        inputs.storage[f'match:{self.label}'] = pairs
        self.value_ = pairs

    def get_pairs(
        self,
        left: Sequence[AgentViewProtocol],
        right: Sequence[AgentViewProtocol],
        rng: np.random.Generator,
    ) -> list[tuple[object, object]]:
        """Return the pairs of the agents of both sides by the pairing.

        Args:
            left:
                The agents of the first side.

            right:
                The agents of the second side, the same sequence for pairs within one group.

            rng:
                The random generator.

        Returns:
            pairs:
                The pairs of agents, the first of every pair from the first side.

        Raises:
            ValidationError:
                A selection of partners names a state the agents do not have.
        """
        spec = self.pairing
        if not isinstance(spec, PartnerSpec):
            return (spec or random_pairing)(left, right, rng, self.capacity)
        if (spec.candidates, spec.choice, spec.capacity, spec.fallback) == ('all', 'uniform', 1, None):
            return self._get_disjoint_pairs(left, right, rng)
        return self._get_partners(spec, left, right, rng)
