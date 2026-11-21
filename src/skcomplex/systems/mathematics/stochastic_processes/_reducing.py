"""Stochastic process systems of sample-space reduction.

The module covers sample-space-reducing processes, their cascades, and fragmentation.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, Protocol

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element, VisualSpec
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.spaces import BaseSpace
from ....core.components.types import AgentViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, SystemView
from ....core.systems import BaseSystem, System
from ....core.types import Role


class BallView(AgentViewProtocol, Protocol):
    """View of a ball of a sample-space-reducing process, with its values as attributes."""

    level: float


class BallElement(Element):
    """Balls of a sample-space-reducing process, each at its state `level`.

    Args:
        label:
            The label of the balls.

        level:
            The initial state, one per ball or shared.

        size:
            The number of balls, or `None` for one.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import BallElement
        >>> BallElement('balls', level=10.0, size=3).level
        10.0
    """

    visual = VisualSpec('agents', {'color_by': 'level'})

    def __init__(
        self,
        label: str,
        *,
        level: Annotated[object, NN] = 10.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.level = level


class CascadeInteraction(Interaction):
    """Cascade of balls that multiply by `mu` as they jump to lower states.

    At every step a ball at state 1 leaves the cascade, and any other ball becomes `floor(mu)` balls, and one more with
    the probability of the rest, each at a state drawn uniformly below its own.

    Args:
        label:
            The label of the interaction.

        target:
            The balls of the cascade.

        mu:
            The mean number of balls a ball becomes at every jump, the multiplication factor.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import CascadeInteraction
        >>> CascadeInteraction('cascade', mu=2.0).mu
        2.0
    """

    law_classes = (UpdateLaw,)
    draws = True

    def __init__(self, label: str, *, target: Role = 'balls', mu: float = 1.5) -> None:
        super().__init__(label)
        self.target = target
        self.mu = mu

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the jump of every ball as the rule."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseLaw] = [UpdateLaw(context.target, self._jump)]

    def _jump(self, agent: BallView, system: SystemView) -> None:
        """Replace a ball by its offspring at lower states, or remove it at state 1."""
        if agent.level > 1:
            whole = math.floor(self.mu)
            for _ in range(whole + int(system.rng.random() < self.mu - whole)):
                agent.create(level=float(system.rng.integers(1, round(agent.level))))
        agent.remove()


class FragmentView(AgentViewProtocol, Protocol):
    """View of a fragment, with its values as attributes."""

    length: float


class FragmentElement(Element):
    """Fragments of integer sizes, each with its size `length`.

    Args:
        label:
            The label of the fragments.

        length:
            The initial size, one per fragment or shared.

        size:
            The number of fragments, or `None` for one.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import FragmentElement
        >>> FragmentElement('fragments', length=100.0).length
        100.0
    """

    visual = VisualSpec('agents', {'color_by': 'length'})

    def __init__(
        self,
        label: str,
        *,
        length: Annotated[object, RealSet.open(0, oo)] = 1000.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.length = length


class BreakingInteraction(Interaction):
    """Breaking of every fragment larger than 1 with the probability `rate` at a uniformly random whole point.

    The fragment keeps the size left of the cut, and a new fragment takes the rest, so the total size stays the same.

    Args:
        label:
            The label of the interaction.

        target:
            The fragments that break.

        rate:
            The probability that a fragment larger than 1 breaks in a step.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import BreakingInteraction
        >>> BreakingInteraction('breaking', rate=0.2).rate
        0.2
    """

    law_classes = (UpdateLaw,)
    draws = True

    def __init__(self, label: str, *, target: Role = 'fragments', rate: float = 0.5) -> None:
        super().__init__(label)
        self.target = target
        self.rate = rate

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the breaking of every fragment as the rule."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseLaw] = [UpdateLaw(context.target, self._split)]

    def _split(self, agent: FragmentView, system: SystemView) -> None:
        """Cut a fragment in two with the probability `rate`, unless it has size 1."""
        length = round(agent.length)
        if length > 1 and system.rng.random() < self.rate:
            kept = int(system.rng.integers(1, length))
            agent.length = float(kept)
            agent.create(length=float(length - kept))


class ReducingJumpInteraction(Interaction):
    """Jump of a ball of the sample-space-reducing process to a lower state, to any state, or back to the top state.

    A ball at state 1 restarts from the top state `n_states`. Otherwise it jumps with probability `lam` uniformly to a
    state below its own, and else uniformly to any state, so the states are visited with probabilities proportional to
    `i^(-lam)`.

    Args:
        label:
            The label of the interaction.

        target:
            The balls that jump.

        n_states:
            The number of states.

        lam:
            The probability of a reducing jump.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import ReducingJumpInteraction
        >>> ReducingJumpInteraction('jump', n_states=20).n_states
        20
    """

    law_classes = (UpdateLaw,)
    draws = True

    def __init__(self, label: str, *, target: Role = 'balls', n_states: int = 10, lam: float = 1.0) -> None:
        super().__init__(label)
        self.target = target
        self.n_states = n_states
        self.lam = lam

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the jump of every ball as the rule."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseLaw] = [UpdateLaw(context.target, self._jump)]

    def _jump(self, agent: BallView, system: SystemView) -> None:
        """Move a ball to its next state."""
        top = round(self.n_states)
        if agent.level <= 1:
            agent.level = float(top)
        elif system.rng.random() < self.lam:
            agent.level = float(system.rng.integers(1, round(agent.level)))
        else:
            agent.level = float(system.rng.integers(1, top + 1))


class SampleSpaceReducingSystem(BaseSystem):
    """Sample-space-reducing processes.

    Their state visits follow Zipf's law `1 / i` for `lam = 1`.

    The system has the group `balls` of state `level`.

    Args:
        n_balls:
            The number of independent balls, all started at the top state.

        n_states:
            The number of states.

        lam:
            The probability of a reducing jump, the exponent of the visiting distribution.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import SampleSpaceReducingSystem
        >>> SampleSpaceReducingSystem(4).prepare().system_.elements[0].size
        4
    """

    field = 'mathematics/stochastic_processes'
    topics = ('sample space reduction', 'agents', 'zipf', 'power law')
    references = ('Corominas-Murtra, Hanel, and Thurner 2015', 'Thurner et al. 2018')
    until = 100

    def __init__(
        self,
        n_balls: int = 20,
        *,
        n_states: int = 10,
        lam: float = 1.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_balls = n_balls
        self.n_states = n_states
        self.lam = lam

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the balls at the top state and their jumps."""
        super()._prepare(context)
        jump = ReducingJumpInteraction('jump', target='balls', n_states=self.n_states, lam=self.lam)
        self.system_ = System(
            self.label,
            elements=[BallElement('balls', level=float(self.n_states), size=self.n_balls)],
            interactions=[jump],
        )


class FragmentationSystem(BaseSystem):
    """The random fragmentation of an object of integer size.

    The fragments keep breaking into smaller ones.

    The total size is conserved, and every fragment ends at size 1. The system has the group `fragments` of state
    `length`.

    Args:
        size:
            The size of the initial object.

        rate:
            The probability that a fragment larger than 1 breaks in a step.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import FragmentationSystem
        >>> FragmentationSystem(50).prepare().system_.elements[0].size
        1
    """

    field = 'mathematics/stochastic_processes'
    topics = ('fragmentation', 'agents', 'sample space reduction')
    references = ('Thurner et al. 2018',)
    until = 30

    def __init__(
        self,
        size: int = 1000,
        *,
        rate: float = 0.5,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.size = size
        self.rate = rate

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the initial object and its breaking."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[FragmentElement('fragments', length=float(self.size), size=1)],
            interactions=[BreakingInteraction('breaking', target='fragments', rate=self.rate)],
        )


class CascadingSampleSpaceReducingSystem(BaseSystem):
    """Cascading sample-space-reducing processes.

    Their visits follow the power law `i^(-mu)` for any `mu`.

    Every cascade starts with one ball at the top state `n_states`. At every step each ball above state 1 jumps to a
    state drawn uniformly below its own and becomes `mu` balls there, `floor(mu)` and one more with the probability of
    the rest, and a ball at state 1 leaves the cascade. Every ball stays one step, so the states of the balls of all
    frames are the visits, distributed as `i^(-mu)` (Thurner et al. 2018, section 3.3.5.4). For `mu < 1` balls are lost
    on the way, `1 - mu` being the probability of a removal. With a conservation law that makes the states of the new
    balls sum to the state of their parent, the exponent is 2 whatever `mu`, the case of `FragmentationSystem`. The
    system has the group `balls` of state `level`.

    Args:
        n_cascades:
            The number of independent cascades, run together.

        n_states:
            The number of states.

        mu:
            The multiplication factor, the mean number of balls a ball becomes at every jump.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import CascadingSampleSpaceReducingSystem
        >>> CascadingSampleSpaceReducingSystem(4).prepare().system_.elements[0].size
        4
    """

    field = 'mathematics/stochastic_processes'
    topics = ('sample space reduction', 'agents', 'cascade', 'power law')
    references = ('Corominas-Murtra, Hanel, and Thurner 2017', 'Thurner et al. 2018')
    until = 100

    def __init__(
        self,
        n_cascades: int = 100,
        *,
        n_states: int = 100,
        mu: float = 1.5,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_cascades = n_cascades
        self.n_states = n_states
        self.mu = mu

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of one ball at the top state per cascade and their jumps."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[BallElement('balls', level=float(self.n_states), size=self.n_cascades)],
            interactions=[CascadeInteraction('cascade', target='balls', mu=self.mu)],
        )
