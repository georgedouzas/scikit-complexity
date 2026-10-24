"""Define the laws of agents: rules written as code, and matchings that pair the members of groups every step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from ...core.base import LawClass, LawKind, Requires, TimeKind, Trait, ValidationError
from ...core.components import Element, Group, probe_laws
from ...core.laws import Law

type Pairing = Callable[[Sequence[Any], Sequence[Any], np.random.Generator, int], list[tuple[Any, Any]]]


def random_pairing(left: Sequence[Any], right: Sequence[Any], rng: np.random.Generator, capacity: int) -> list:
    """Return pairs drawn uniformly at random, each member in at most `capacity` pairs and never paired with itself.

    Args:
        left:
            The members of the first group.

        right:
            The members of the second group, the same sequence as `left` for pairs within one group.

        rng:
            The random generator.

        capacity:
            The largest number of pairs of every member.

    Returns:
        pairs:
            The pairs of members.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.formalisms.agents import random_pairing
        >>> pairs = random_pairing(list(range(5)), list('abc'), np.random.default_rng(0), 1)
        >>> len(pairs)
        3
    """
    pairs: list[tuple[Any, Any]] = []
    seen: set[tuple[int, int]] = set()
    if left is right:
        for _ in range(capacity):
            order = rng.permutation(len(left))
            for first, second in zip(order[::2], order[1::2], strict=False):
                key = (min(first, second), max(first, second))
                if key not in seen:
                    seen.add(key)
                    pairs.append((left[first], right[second]))
        return pairs
    firsts = rng.permutation(np.repeat(np.arange(len(left)), capacity))
    seconds = rng.permutation(np.repeat(np.arange(len(right)), capacity))
    for first, second in zip(firsts, seconds, strict=False):
        if (first, second) not in seen:
            seen.add((first, second))
            pairs.append((left[first], right[second]))
    return pairs


def matches(model: Any, label: str = 'match') -> list[tuple[Any, Any]]:  # noqa: ANN401
    """Return the pairs a matching law stored for the current step.

    Args:
        model:
            The model view a rule receives.

        label:
            The label of the matching law.

    Returns:
        pairs:
            The pairs of agents, or none before the law runs.
    """
    return list(model.storage.get(f'match:{label}', ()))


def runs_rules(component: object) -> bool:
    """Check whether a component runs rules of its own: a variable group, or an element or group with rule laws.

    Such a component is an agent. The compiler also makes agents of the groups whose element type has no laws of its
    own in a system with rules, unless an interaction whose laws are not rules binds them.

    Args:
        component:
            The element, group, interaction, or system.

    Returns:
        runs_rules:
            Whether the component is a variable group, or an element or group whose own laws include a `Rule`.

    Examples:
        >>> from skcomplex.core.components import Element
        >>> from skcomplex.formalisms.agents import Rule, runs_rules
        >>> Walker = Element.define('Walker', laws=lambda s: [Rule(lambda agent, model: None)])
        >>> runs_rules(Walker('walker')), runs_rules(Element.define('Plain')('plain'))
        (True, False)
    """
    if isinstance(component, Group) and component.variable:
        return True
    if not isinstance(component, Element | Group):
        return False
    return any(isinstance(law, Rule) for law in probe_laws(component) or [])


@dataclass(frozen=True)
class Rule(Law):
    """Behaviour written as code, run once per step for every member of an agent group, or once per step for a model.

    An element whose laws include rules is an agent, alone or in a group. A rule of an element receives the agent and
    the model, `function(agent, model)`. A rule of an interaction receives the model alone, `function(model)`. A system
    with rules is agent-based, needs discrete time, and runs on the Mesa backend.

    Args:
        function:
            The function that runs the rule.

        phase:
            The name of the phase of the step the rule runs in, one of the system's `phases`, or `None` for the
            default phase.

    Examples:
        >>> from skcomplex.formalisms.agents import Rule
        >>> rule = Rule(lambda agent, model: None)
        >>> callable(rule.function)
        True
    """

    law_kind: ClassVar[LawKind] = LawKind.RULE
    time_class: ClassVar[LawClass] = LawClass.DISCRETE
    traits: ClassVar[frozenset[Trait]] = frozenset({Trait.DECLARABLE, Trait.AGENT})
    requires: ClassVar[Requires] = Requires(time={TimeKind.DISCRETE})

    function: Callable[..., None]
    phase: str | None = None


@dataclass(frozen=True)
class Match(Law):
    """Pairing of the members of one agent group, or of two, every step, which the rules of the step read.

    The core pairs members uniformly at random. Any other pairing, such as stable matching, an order book, or an
    auction, is a function given as `pairing`, so its realization lives with the models that use it. Rules read the
    pairs of the step with `model.storage['match:<label>']`, or with `matches(model, label)`.

    Args:
        left:
            The path of the first agent group.

        right:
            The path of the second agent group, or `None` for pairs within the first.

        pairing:
            A function `pairing(left, right, rng, capacity)` of the agents of both groups that returns their pairs, or
            `None` for uniform random pairing.

        capacity:
            The largest number of pairs of every member.

        label:
            The name of the pairs.

        phase:
            The phase of the step the pairing runs in, before the rules of the phase that read it.

    Raises:
        ValidationError:
            The capacity is not a positive integer.

    Examples:
        >>> from skcomplex.formalisms.agents import Match
        >>> Match('buyers', 'sellers', label='trades').capacity
        1
    """

    law_kind: ClassVar[LawKind] = LawKind.MATCH
    time_class: ClassVar[LawClass] = LawClass.DISCRETE
    traits: ClassVar[frozenset[Trait]] = frozenset({Trait.DECLARABLE})
    requires: ClassVar[Requires] = Requires(time={TimeKind.DISCRETE})

    left: str
    right: str | None = None
    pairing: Pairing | None = None
    capacity: int = 1
    label: str = 'match'
    phase: str | None = None

    def __post_init__(self) -> None:
        """Check the capacity."""
        if isinstance(self.capacity, bool) or not isinstance(self.capacity, int) or self.capacity < 1:
            error_msg = f'The match {self.label!r} has capacity {self.capacity!r}. Give a positive integer.'
            raise ValidationError(error_msg)

    def _pair(self, model: Any) -> None:  # noqa: ANN401
        """Store the pairs of the step in the storage of the model."""
        left = model.agents(self.left)
        right = left if self.right is None else model.agents(self.right)
        pairing = self.pairing or random_pairing
        model.storage[f'match:{self.label}'] = pairing(left, right, model.rng, self.capacity)

    @property
    def rule(self) -> Rule:
        """Rule the agent stepper runs, which stores the pairs of the step."""
        return Rule(self._pair, phase=self.phase)
