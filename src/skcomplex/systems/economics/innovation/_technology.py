"""Innovation systems: technologies that improve by redesign or combination.

They redesign their components or combine logic circuits.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import AgentView, BaseSolver, SystemView
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role

NAND = (2, 0b0111)
FALSE = (0, 0)
TRUE = (0, 1)


class TechnologyView(AgentView):
    """Technology as the invention sees it.

    It holds its `arity`, its truth `table`, and its `cost` in NAND gates.
    """

    arity: float
    table: float
    cost: float


class PartView(AgentView):
    """Part of a technology as the redesign sees it: its unit `cost`."""

    cost: float


class PartElement(Element):
    """PartElement of a technology, described by its unit cost.

    Args:
        label:
            The label of the agent or group.

        cost:
            The initial unit cost.

        size:
            The number of members, or `None` for one.

    Examples:
        >>> from skcomplex.systems.economics.innovation import PartElement
        >>> PartElement('components', cost=0.5, size=5).size
        5
    """

    def __init__(
        self,
        label: str,
        *,
        cost: Annotated[object, RealSet.closed_open(0, oo)] = 1.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.cost = cost

    @classmethod
    def _get_view(cls) -> type[PartView]:
        """Return the view of the agents of the element."""
        return PartView


def _truth_table(arity: int, function: Callable[..., bool]) -> int:
    """Return the truth table of a Boolean function.

    Bit `r` holds its value on the input bits of row `r`.
    """
    rows = range(2**arity)
    return sum(int(function(*(bool(row >> pin & 1) for pin in range(arity)))) << row for row in rows)


GOALS = frozenset(
    (arity, _truth_table(arity, function))
    for arity, function in (
        (1, lambda a: not a),
        (2, lambda a, b: a and b),
        (2, lambda a, b: a or b),
        (2, lambda a, b: a != b),
        (2, lambda a, b: not (a or b)),
        (2, lambda a, b: a == b),
        (2, lambda a, b: (not a) or b),
        (3, lambda a, b, c: a and b and c),
        (3, lambda a, b, c: a or b or c),
        (3, lambda a, b, c: (a != b) != c),
        (3, lambda a, b, c: (a and b) or (a and c) or (b and c)),
    )
)


class InventionInteraction(Interaction):
    """Random wiring of technologies of the pool into a circuit.

    The circuit is kept if it computes a wanted function more cheaply. A circuit has between one and `max_inputs` inputs
    and wires between two and `max_components` technologies in a row, each pin to an input or to the output of an
    earlier technology. The output of the last one is the circuit's output. A circuit that computes a wanted function
    joins the pool, or replaces the technology of that function when it uses fewer NAND gates.

    Args:
        label:
            The label of the interaction.

        technologies:
            The path of the group of technologies.

        max_components:
            The largest number of technologies in a circuit.

        max_inputs:
            The largest number of inputs of a circuit.

    Examples:
        >>> from skcomplex.systems.economics.innovation import InventionInteraction
        >>> InventionInteraction('invention', technologies='technologies', max_components=6).max_components
        6
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        technologies: Role,
        max_components: Annotated[float, RealSet.open(0, oo)] = 12,
        max_inputs: Annotated[float, RealSet.open(0, oo)] = 3,
    ) -> None:
        super().__init__(label)
        self.technologies = technologies
        self.max_components = max_components
        self.max_inputs = max_inputs

    def _wire(self, pool: list[TechnologyView], system: SystemView) -> tuple[int, int, int]:
        """Return the arity, the truth table, and the cost of a random circuit.

        The circuit wires technologies of the pool.
        """
        rng = system.rng
        n_inputs = int(rng.integers(1, int(system.get_parameter('max_inputs')) + 1))
        n_parts = int(rng.integers(2, int(system.get_parameter('max_components')) + 1))
        signals, cost = InventionInteraction._get_input_signals(n_inputs), 0
        for _ in range(n_parts):
            part = pool[int(rng.integers(len(pool)))]
            pins = [signals[int(rng.integers(len(signals)))] for _ in range(int(part.arity))]
            signals.append(InventionInteraction._apply(int(part.table), int(part.arity), pins, n_inputs))
            cost += int(part.cost)
        return n_inputs, signals[-1], cost

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rule of the system."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseComponent] = [UpdateLaw(context, self.invent)]

    def invent(self, system: SystemView) -> None:
        """Wire a random circuit and add it to the pool if it is better.

        It is better when it computes a wanted function better.

        Args:
            system:
                The system.
        """
        pool = system.get_agents('technologies', TechnologyView)
        arity, table, cost = self._wire(pool, system)
        if (arity, table) not in GOALS:
            return
        known = [technology for technology in pool if (technology.arity, technology.table) == (arity, table)]
        if not known:
            system.create('technologies', arity=arity, table=table, cost=cost)
        elif cost < known[0].cost:
            known[0].cost = cost

    @staticmethod
    def _get_input_signals(n_inputs: int) -> list[int]:
        """Return the truth tables of the primary inputs of a circuit."""
        rows = range(2**n_inputs)
        return [sum(1 << row for row in rows if row >> pin & 1) for pin in range(n_inputs)]

    @staticmethod
    def _apply(table: int, arity: int, signals: Sequence[int], n_inputs: int) -> int:
        """Return the truth table of a gate with a table and an arity.

        The gate is wired to the signals of a circuit.
        """
        output = 0
        for row in range(2**n_inputs):
            index = sum((signal >> row & 1) << pin for pin, signal in enumerate(signals[:arity]))
            output |= (table >> index & 1) << row
        return output


class RedesignInteraction(Interaction):
    """Attempt to cut the cost of a technology by redesigning its components.

    It redesigns a random component and the components that depend on it. The redesigned components get new costs drawn
    uniformly from `[0, 1]`, and the redesign is kept only if the total cost falls. `dependencies[i][j]` is 1 when a
    change of component `j` requires a change of component `i`.

    Args:
        label:
            The label of the interaction.

        components:
            The path of the group of components.

        dependencies:
            The design structure matrix.

    Examples:
        >>> from skcomplex.systems.economics.innovation import RedesignInteraction
        >>> RedesignInteraction('redesign', components='components', dependencies=[[1.0]]).dependencies
        [[1.0]]
    """

    draws = True

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        components: Role,
        dependencies: Sequence[Sequence[float]] = (),
    ) -> None:
        super().__init__(label)
        self.components = components
        self.dependencies = dependencies

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rule of the system."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseComponent] = [UpdateLaw(context, self.redesign)]

    def redesign(self, system: SystemView) -> None:
        """Propose new costs for a random component and its dependents.

        They are kept if the total cost falls.

        Args:
            system:
                The system.
        """
        components = sorted(system.get_agents('components', PartView), key=lambda component: component.id)
        cost = np.array([component.cost for component in components])
        proposal, changed = RedesignInteraction._get_components(
            cost,
            np.asarray(self.dependencies, dtype=float),
            system.rng,
        )
        if proposal.sum() >= cost.sum():
            return
        for index in changed:
            components[index].cost = float(proposal[index])

    @staticmethod
    def _get_components(
        cost: Array,
        dependencies: Array,
        rng: np.random.Generator,
    ) -> tuple[Array, npt.NDArray[np.intp]]:
        """Return the proposed costs of a redesign of a random component.

        They include the components that depend on it.
        """
        changed = np.flatnonzero(dependencies[:, int(rng.integers(len(cost)))])
        proposal = cost.copy()
        proposal[changed] = rng.random(len(changed))
        return proposal, changed


class TechnologyElement(Element):
    """Logic circuit in the pool of technologies.

    It is described by the truth table of the function it computes. Bit `r` of the table is the output on the inputs
    whose bits form `r`, with the first input as the lowest bit.

    Args:
        label:
            The label of the agent or group.

        arity:
            The number of inputs.

        table:
            The truth table.

        cost:
            The number of NAND gates in the circuit.

        size:
            The number of members, or `None` for one.

    Examples:
        >>> from skcomplex.systems.economics.innovation import TechnologyElement
        >>> TechnologyElement('technologies', arity=[2, 0, 0], table=[7, 0, 1], size=3).size
        3
    """

    def __init__(
        self,
        label: str,
        *,
        arity: Annotated[object, NN] = 2,
        table: Annotated[object, NN] = NAND[1],
        cost: Annotated[object, NN] = 1,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.arity = arity
        self.table = table
        self.cost = cost

    @classmethod
    def _get_view(cls) -> type[TechnologyView]:
        """Return the view of the agents of the element."""
        return TechnologyView


class NkTechnologySystem(BaseSystem):
    """McNerney's system of technological improvement.

    Its cost falls as a power of the cumulative production. Every step is one unit of production and one attempt to
    redesign a random component together with the components that depend on it. Redesigns are kept only when they cut
    the total cost, which falls like Wright's law, more slowly for designs whose components depend on more others. The
    system has the group `components` and the total cost `total_cost`.

    Args:
        n_components:
            The number of components.

        d:
            The number of components that a redesign of one component changes, itself included.

        random_state:
            The seed or generator of the design structure matrix and the initial costs.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.economics.innovation import NkTechnologySystem
        >>> NkTechnologySystem(5, 1).prepare().system_.interactions[0].dependencies[0]
        (1.0, 0.0, 0.0, 0.0, 0.0)
    """

    field = 'economics/innovation'
    topics = ('innovation', 'technology', 'learning curve', 'agents')
    references = ('Thurner et al. 2018', 'McNerney et al. 2011')
    until = 200
    observables = (Observable('total_cost', group='components', reduce='sum', variable='cost'),)

    def __init__(
        self,
        n_components: int = 10,
        d: int = 2,
        *,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_components = n_components
        self.d = d
        self.random_state = random_state

    @staticmethod
    def _get_design_structure(n_components: int, d: int, rng: np.random.Generator) -> Array:
        """Return a design structure matrix.

        A change of a component changes it and `d - 1` other ones.
        """
        matrix = np.eye(n_components)
        for column in range(n_components):
            others = rng.choice([row for row in range(n_components) if row != column], d - 1, replace=False)
            matrix[others, column] = 1.0
        return matrix

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the components built from the model parameters."""
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        dependencies = self._get_design_structure(self.n_components, self.d, rng)
        components = PartElement('components', cost=rng.random(self.n_components), size=self.n_components)
        self.system_ = System(
            self.label,
            elements=[components],
            interactions=[
                RedesignInteraction(
                    'redesign',
                    components='components',
                    dependencies=tuple(map(tuple, dependencies.tolist())),
                ),
            ],
        )


class ArthurPolakSystem(BaseSystem):
    """Arthur and Polak's evolution of technology.

    It builds logic functions from NAND gates. The pool starts with the NAND gate and the constants 0 and 1. Every step
    wires a random circuit from the pool and keeps it when it computes a wanted function, negation, two-input and, or,
    exclusive or, nor, equivalence, and implication, or three-input and, or, parity, and majority, so simple inventions
    become the building blocks of complex ones. The system has the pool `technologies`, whose size is the reporter
    `technologies.count`.

    Args:
        max_components:
            The largest number of technologies in a circuit.

        max_inputs:
            The largest number of inputs of a circuit.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.economics.innovation import ArthurPolakSystem
        >>> ArthurPolakSystem().prepare().system_.elements[0].size
        3
    """

    field = 'economics/innovation'
    topics = ('innovation', 'technology', 'combinatorial evolution', 'agents')
    references = ('Thurner et al. 2018', 'Arthur and Polak 2006')
    until = 200

    def __init__(
        self,
        *,
        max_components: int = 12,
        max_inputs: int = 3,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.max_components = max_components
        self.max_inputs = max_inputs

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the components built from the model parameters."""
        super()._prepare(context)
        primitives = (NAND, FALSE, TRUE)
        technologies = TechnologyElement(
            'technologies',
            arity=[float(arity) for arity, _ in primitives],
            table=[float(table) for _, table in primitives],
            cost=[1.0, 0.0, 0.0],
            size=len(primitives),
        )
        invention = InventionInteraction(
            'invention',
            technologies='technologies',
            max_components=float(self.max_components),
            max_inputs=float(self.max_inputs),
        )
        self.system_ = System(self.label, elements=[technologies], interactions=[invention])
