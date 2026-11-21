"""Watchmaker systems: Simon's fable of Hora and Tempus.

They assemble watches with and without stable subassemblies.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, Protocol, cast

from sage.all__sagemath_symbolics import NN
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.types import AgentViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, SystemView
from ....core.systems import BaseSystem, System
from ....core.types import Role


class WatchmakerView(AgentViewProtocol, Protocol):
    """View of a watchmaker of the Hora and Tempus fable.

    Its states and parameters are attributes.
    """

    progress: float
    done: float
    watches: float
    task_size: float
    tasks: float


class WatchmakerElement(Element):
    """Watchmaker who assembles a watch.

    The watch is a number of tasks of a number of actions each.

    Args:
        label:
            The label of the watchmaker or the group.

        progress:
            The number of actions done on the task in hand, which an interruption undoes.

        done:
            The number of stable tasks done on the watch in hand.

        watches:
            The number of finished watches.

        task_size:
            The number of actions of a task, the parts or subassemblies put together into a stable assembly.

        tasks:
            The number of tasks of a watch.

        size:
            The number of members, or `None` for one.

    Examples:
        >>> from skcomplex.systems.social.organizations import WatchmakerElement
        >>> WatchmakerElement('tempus', task_size=1000.0, tasks=1.0).task_size
        1000.0
    """

    def __init__(
        self,
        label: str,
        *,
        progress: Annotated[object, NN] = 0.0,
        done: Annotated[object, NN] = 0.0,
        watches: Annotated[object, NN] = 0.0,
        task_size: Annotated[object, NN] = 10.0,
        tasks: Annotated[object, NN] = 111.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.progress = progress
        self.done = done
        self.watches = watches
        self.task_size = task_size
        self.tasks = tasks


class AssemblyInteraction(Interaction):
    """Assembly of the Hora and Tempus fable.

    There is one action per step, which an interruption can undo. Every step every watchmaker takes one action on the
    task in hand, unless a customer calls, with the probability `interruption`, and the unfinished task falls to pieces.
    A finished task is stable, and a watchmaker who finishes the last task of a watch sells it and starts the next one.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of watchmakers.

        interruption:
            The probability that a watchmaker is interrupted at an action.

    Examples:
        >>> from skcomplex.systems.social.organizations import AssemblyInteraction
        >>> AssemblyInteraction('assembly', target='watchmakers').interruption
        0.01
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        interruption: Annotated[float, RealSet.closed(0, 1)] = 0.01,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.interruption = interruption

    def _prepare(self, s: SimpleNamespace) -> None:
        """Set the assembly rule.

        Args:
            s:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(s)
        self.laws_: list[BaseComponent] = [UpdateLaw(s, self.assemble)]

    def assemble(self, system: SystemView) -> None:
        """Take one action of every watchmaker.

        An interruption undoes the task in hand instead.

        Args:
            system:
                The system.
        """
        for watchmaker in cast(list[WatchmakerView], system.get_agents(self.target)):
            if system.rng.random() < self.interruption:
                watchmaker.progress = 0.0
                continue
            watchmaker.progress += 1.0
            if watchmaker.progress < watchmaker.task_size:
                continue
            watchmaker.progress = 0.0
            watchmaker.done += 1.0
            if watchmaker.done >= watchmaker.tasks:
                watchmaker.done = 0.0
                watchmaker.watches += 1.0


class WatchmakersSystem(BaseSystem):
    """Simon's fable of the watchmakers Hora and Tempus, told by Meadows.

    The watchmakers work with and without stable subassemblies. Both watchmakers build watches of `parts` parts. Tempus
    puts all of them together at once, so an interruption, a customer's call, makes the unfinished watch fall to pieces.
    Hora puts together stable subassemblies of `subassembly` elements, then assemblies of those, up to the whole watch,
    so an interruption loses only the assembly in hand. Hora has more actions to take, `parts - 1` divided by
    `subassembly - 1` tasks of `subassembly` actions each, 111 tasks of 10 actions for 1000 parts. A task of `k` actions
    interrupted with the probability `p` at every action takes `((1 - p)^-k - 1) / p` actions on average, so at
    `p = 0.01` Hora finishes a watch every 1173 steps and Tempus one every 2.3 million steps. The book tells the fable
    as a narrative, so the step of one action and the interruption at every action follow Simon's original telling.

    The system has the group `watchmakers`, Hora first and Tempus second.

    Args:
        parts:
            The number of parts of a watch, a power of `subassembly`.

        subassembly:
            The number of elements of a stable assembly of Hora.

        interruption:
            The probability that a watchmaker is interrupted at an action.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.social.organizations import WatchmakersSystem
        >>> WatchmakersSystem().prepare().system_.elements[0].size
        2
    """

    field = 'social/organizations'
    topics = ('hierarchy', 'agents', 'stable intermediate forms', 'interruptions')
    references = ('Simon 1962', 'Meadows 2008')
    until = 5000

    def __init__(
        self,
        parts: int = 1000,
        *,
        subassembly: int = 10,
        interruption: float = 0.01,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.parts = parts
        self.subassembly = subassembly
        self.interruption = interruption

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the elements and interactions of the system."""
        super()._prepare(context)
        hora_tasks = (self.parts - 1) // (self.subassembly - 1)
        group = WatchmakerElement(
            'watchmakers',
            task_size=[float(self.subassembly), float(self.parts)],
            tasks=[float(hora_tasks), 1.0],
            size=2,
        )
        self.system_ = System(
            self.label,
            elements=[group],
            interactions=[AssemblyInteraction('assembly', target='watchmakers', interruption=self.interruption)],
        )
