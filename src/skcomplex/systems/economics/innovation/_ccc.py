"""Innovation systems: the co-evolving combinatorial critical model.

Its entities are produced and destroyed by pairs.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated, ClassVar

import numpy as np
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.laws import BaseLaw
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.solvers import AgentView, SystemView
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ....core.types import Array
from ..common.policy import PolicyRuleInteraction


class CombinatorialEntityView(AgentView):
    """Entity as the update sees it: whether it exists and its fitness."""

    exists: float
    fitness: float


class CombinatorialEntityElement(Element):
    """Entity of a combinatorial evolution, which exists or not.

    It is a species, a good, or a technology.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        exists:
            1 when the entity exists, 0 otherwise.

        fitness:
            The active productive minus the active destructive sets of the entity at its last update.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.innovation import CombinatorialEntityElement
        >>> GroupElement('entities', element=CombinatorialEntityElement, size=3).size
        3
    """

    law_classes = ()

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'exists'}),
        *,
        exists: Annotated[float, StateSpec(domain=RealSet.closed(0, 1))] = 0.0,
        fitness: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.exists = exists
        self.fitness = fitness

    @classmethod
    def _get_view(cls) -> type[CombinatorialEntityView]:
        """Return the view of the agents of the element."""
        return CombinatorialEntityView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return no laws, since the update of the system moves every entity.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return []


class CombinatorialUpdateInteraction(PolicyRuleInteraction[SystemView]):
    """Random sequential update of the CCC model by the sign of the fitness.

    Spontaneous flips are added. Every step every entity `i`, in random order, computes its fitness
    `f_i = sum_jk (M+_ijk - M-_ijk) sigma_j sigma_k` from the current states, exists when it is positive, disappears
    when it is negative, and keeps its state when it is zero (Eq. 5.61 of Thurner et al. 2018). Then it switches its
    state with the probability `p`.

    Args:
        label:
            The label of the update.

        rule:
            Unused, since the update overrides `act`.

        phase:
            The phase the update acts in, or `None` for the default phase.

        entities:
            The path of the group of entities.

        production:
            The productive sets, triples `(i, j, k)` of the nonzero entries of `M+`.

        destruction:
            The destructive sets, triples `(i, j, k)` of the nonzero entries of `M-`.

        p:
            The probability per step that an entity switches its state spontaneously.

    Examples:
        >>> from skcomplex.systems.economics.innovation import CombinatorialUpdateInteraction
        >>> CombinatorialUpdateInteraction('update', entities='entities').reports
        ('diversity', 'changes')
    """

    draws = True

    reports: ClassVar[tuple[str, ...]] = ('diversity', 'changes')

    def __init__(
        self,
        label: str,
        *,
        rule: None = None,
        phase: str | None = None,
        entities: Annotated[str, RoleSpec()],
        production: tuple[tuple[int, int, int], ...] = (),
        destruction: tuple[tuple[int, int, int], ...] = (),
        p: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.0002,
    ) -> None:
        super().__init__(label, rule=rule, phase=phase)
        self.entities = entities
        self.production = production
        self.destruction = destruction
        self.p = p

    @staticmethod
    def _sets(triples: tuple[tuple[int, int, int], ...], size: int) -> list[Array]:
        """Return the pairs `(j, k)` of the sets of every entity."""
        array = np.asarray(triples, dtype=int).reshape(-1, 3)
        return [array[array[:, 0] == entity, 1:] for entity in range(size)]

    def act(self, system: SystemView) -> None:
        """Update every entity once in random order.

        Args:
            system:
                The system view of the update.
        """
        entities = sorted(system.agents('entities', CombinatorialEntityView), key=lambda entity: entity.id)
        size, rng = len(entities), system.rng
        states = np.array([entity.exists for entity in entities])
        before = states.copy()
        plus, minus = self._sets(self.production, size), self._sets(self.destruction, size)
        fitness = np.array([entity.fitness for entity in entities])
        flips = rng.random(size) < system.parameter('p')
        for entity in rng.permutation(size):
            gain = states[plus[entity][:, 0]] @ states[plus[entity][:, 1]]
            loss = states[minus[entity][:, 0]] @ states[minus[entity][:, 1]]
            fitness[entity] = gain - loss
            if fitness[entity] != 0:
                states[entity] = float(fitness[entity] > 0)
            if flips[entity]:
                states[entity] = 1 - states[entity]
        for entity, state, value in zip(entities, states, fitness, strict=True):
            entity.exists, entity.fitness = float(state), float(value)
        system.report('diversity', float(states.sum()))
        system.report('changes', float(np.sum(states != before)))


class CccModelSystem(BaseSystem):
    """The co-evolving combinatorial critical model.

    Its diversity shows punctuated equilibria. Every entity exists or not. Pairs of existing entities produce other
    entities through the production tensor `M+` and destroy them through the destruction tensor `M-`, random tensors
    with on average `r+` productive and `r-` destructive sets per entity, fixed during the run. Every step, in random
    sequential order, an entity exists when its active productive sets outnumber its active destructive ones, disappears
    in the opposite case, and keeps its state on a tie, and then switches spontaneously with a small probability `p`
    (Thurner et al. 2018, Section 5.6.4). The diversity stays nearly constant for long quasi-equilibria, whose lifetimes
    and the sizes of the changes between them have approximate power laws, and restructures in bursts. Mostly productive
    tensors fill the system, mostly destructive ones keep it nearly empty, and the punctuated regime lies in between
    without fine-tuning. The source leaves the initial state open, so every entity exists at the start with the
    probability `initial`. With the tensors of this implementation the setting `r+ = 10` and `r- = 15` of Fig. 5.14
    restructures almost every step, so the defaults take `r+ = 5` and `r- = 10`, which is punctuated.

    The system has the group `entities` and the update `update`, whose reporters are the number of existing entities
    `update.diversity` and the entities that changed state in the step `update.changes`.

    Args:
        n_entities:
            The number of entities.

        r_plus:
            The mean number of productive sets per entity.

        r_minus:
            The mean number of destructive sets per entity.

        p:
            The probability per step that an entity switches its state spontaneously.

        initial:
            The probability that an entity exists at the start.

        random_state:
            The seed or generator of the tensors and the initial states.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.economics.innovation import CccModelSystem
        >>> CccModelSystem(10).prepare().system_.elements[0].prepare().size_
        10
    """

    field = 'economics/innovation'
    topics = ('innovation', 'combinatorial evolution', 'punctuated equilibria', 'self-organized criticality', 'agents')
    references = ('Thurner et al. 2018',)
    until = 1000

    def __init__(
        self,
        n_entities: int = 100,
        *,
        r_plus: float = 5.0,
        r_minus: float = 10.0,
        p: float = 0.0002,
        initial: float = 0.5,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_entities = n_entities
        self.r_plus = r_plus
        self.r_minus = r_minus
        self.p = p
        self.initial = initial
        self.random_state = random_state

    @staticmethod
    def _random_sets(size: int, rate: float, rng: np.random.Generator) -> tuple[tuple[int, int, int], ...]:
        """Return random sets of tensor entries.

        Every entry `(i, j, k)` of the tensor is present with probability `rate / size^2`.
        """
        count = rng.binomial(size**3, min(1.0, rate / size**2))
        triples = rng.integers(size, size=(count, 3))
        return tuple((int(i), int(j), int(k)) for i, j, k in triples)

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        rng = np.random.default_rng(self.random_state)
        entities = GroupElement(
            'entities',
            element=CombinatorialEntityElement,
            size=self.n_entities,
            values={'exists': (rng.random(self.n_entities) < self.initial).astype(float)},
        )
        update = CombinatorialUpdateInteraction(
            'update',
            entities='entities',
            production=self._random_sets(self.n_entities, self.r_plus, rng),
            destruction=self._random_sets(self.n_entities, self.r_minus, rng),
            p=self.p,
        )
        return CompositeSystem(self.label, elements=[entities], interactions=[update])
