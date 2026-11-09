"""Watchmaker systems: Simon's fable of Hora and Tempus.

They assemble watches with and without stable subassemblies.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import NN
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.components.systems import BaseSystem, System
from ....core.engine.solvers import AgentView, SystemView
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec


class WatchmakerView(AgentView):
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

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

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

    Examples:
        >>> from skcomplex.systems.social.organizations import WatchmakerElement
        >>> WatchmakerElement('tempus', task_size=1000.0, tasks=1.0).task_size
        1000.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        progress: Annotated[float, StateSpec(domain=NN)] = 0.0,
        done: Annotated[float, StateSpec(domain=NN)] = 0.0,
        watches: Annotated[float, StateSpec(domain=NN)] = 0.0,
        task_size: Annotated[float, StateSpec(domain=NN)] = 10.0,
        tasks: Annotated[float, StateSpec(domain=NN)] = 111.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.progress = progress
        self.done = done
        self.watches = watches
        self.task_size = task_size
        self.tasks = tasks

    @classmethod
    def _get_view(cls) -> type[WatchmakerView]:
        """Return the class of the views of the agents.

        Their attributes are the states and parameters.
        """
        return WatchmakerView


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

    law_classes = (RuleLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        interruption: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.01,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.interruption = interruption

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the assembly rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.assemble)]

    def assemble(self, system: SystemView) -> None:
        """Take one action of every watchmaker.

        An interruption undoes the task in hand instead.

        Args:
            system:
                The system.
        """
        for watchmaker in system.agents(self.target, WatchmakerView):
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

    Examples:
        >>> from skcomplex.systems.social.organizations import WatchmakersSystem
        >>> WatchmakersSystem().prepare().system_.elements[0].prepare().size_
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
    ) -> None:
        super().__init__(label)
        self.parts = parts
        self.subassembly = subassembly
        self.interruption = interruption

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        hora_tasks = (self.parts - 1) // (self.subassembly - 1)
        group = GroupElement(
            'watchmakers',
            element=WatchmakerElement,
            size=2,
            values={'task_size': [float(self.subassembly), float(self.parts)], 'tasks': [float(hora_tasks), 1.0]},
        )
        return System(
            self.label,
            elements=[group],
            interactions=[AssemblyInteraction('assembly', target='watchmakers', interruption=self.interruption)],
        )
