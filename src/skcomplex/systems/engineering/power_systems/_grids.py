"""Power grid systems: generators and consumers that synchronise.

They synchronise through transmission lines.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar

from sage.all__sagemath_symbolics import oo, sin
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role


class SwingInteraction(Interaction):
    """Swing equation of the phase of a node of a power grid, `phi'' = P - gamma phi'`.

    The node holds its phase `phi` and its power `P`, positive for a generator and negative for a consumer.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the node.

        gamma:
            The damping.

    Examples:
        >>> from skcomplex.systems.engineering.power_systems import SwingInteraction
        >>> SwingInteraction('swing', target='node_0', gamma=0.2).gamma
        0.2
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        gamma: Annotated[float, RealSet.closed_open(0, oo)] = 0.1,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.gamma = gamma

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the swing equation of the phase.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        node, t = context.target, context.t
        self.laws_: list[BaseComponent] = [RateLaw(node.phi.diff(t, 2), node.P - context.gamma * node.phi.diff(t))]


class TransmissionLineInteraction(Interaction):
    """Transmission line of a capacity that couples the phases of two nodes.

    Each node gets the power flow `K sin(phi_j - phi_i)` from the other, so the flows of a line are opposite.

    Args:
        label:
            The label of the interaction.

        a:
            The path of the first node the line joins.

        b:
            The path of the second node the line joins.

        K:
            The capacity of the line.

    Examples:
        >>> from skcomplex.systems.engineering.power_systems import TransmissionLineInteraction
        >>> TransmissionLineInteraction('line', a='node_0', b='node_1', K=2.0).K
        2.0
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        a: Role,
        b: Role,
        K: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label)
        self.a = a
        self.b = b
        self.K = K

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the power flows into both nodes.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        first, second, t = context.a.phi, context.b.phi, context.t
        self.laws_: list[BaseComponent] = [
            RateLaw(first.diff(t, 2), context.K * sin(second - first)),
            RateLaw(second.diff(t, 2), context.K * sin(first - second)),
        ]


class PowerGridSystem(BaseSystem):
    """A ring power grid in the swing-equation system.

    More transmission capacity can break synchrony. The phase of node `i` relative to the grid frequency follows
    `phi_i'' = P_i - gamma phi_i' + Σ_j K_ij sin(phi_j - phi_i)`. The first half of the ring are generators with
    `P = power` and the second half consumers with `P = -power`, joined by lines of capacity `K`. The default grid
    synchronises, but adding the line `(0, 2)` or doubling the capacity of the line `(0, 1)` makes the flows overload a
    line, and synchrony is lost, which is Braess's paradox. The system has the phases `node_<i>.phi` and frequency
    deviations `node_<i>.phi'`.

    Args:
        n_nodes:
            The number of nodes of the ring.

        K:
            The capacity of every line.

        power:
            The power of every generator, and its negative the power of every consumer.

        gamma:
            The damping.

        added_line:
            A pair of nodes to join by an extra line of capacity `K`, or `None`.

        doubled_line:
            A pair of neighbouring nodes whose line has capacity `2 K`, or `None`.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.engineering.power_systems import PowerGridSystem
        >>> len(PowerGridSystem(added_line=(0, 2)).prepare().system_.interactions)
        17
    """

    field = 'engineering/power_systems'
    topics = ('power grid', 'synchronization', 'network', 'braess paradox', 'ode')
    references = ('Witthaut and Timme 2012', 'Datseris and Parlitz 2022')
    until = 300.0
    dt = 1.0

    def __init__(
        self,
        n_nodes: int = 8,
        *,
        K: float = 2.75,
        power: float = 1.0,
        gamma: float = 0.1,
        added_line: tuple[int, int] | None = None,
        doubled_line: tuple[int, int] | None = None,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_nodes = n_nodes
        self.K = K
        self.power = power
        self.gamma = gamma
        self.added_line = added_line
        self.doubled_line = doubled_line

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the nodes, their swing equations, and the lines of the ring."""
        super()._prepare(context)
        powers = [self.power if index < self.n_nodes // 2 else -self.power for index in range(self.n_nodes)]
        elements = [Element(f'node_{index}', values={'phi': 0.0, 'P': power}) for index, power in enumerate(powers)]
        swings = [
            SwingInteraction(f'swing_{index}', target=f'node_{index}', gamma=self.gamma)
            for index in range(self.n_nodes)
        ]
        pairs = [(index, (index + 1) % self.n_nodes) for index in range(self.n_nodes)]
        if self.added_line is not None:
            pairs.append((self.added_line[0], self.added_line[1]))
        doubled = None if self.doubled_line is None else frozenset(self.doubled_line)
        lines = [
            TransmissionLineInteraction(
                f'line_{first}_{second}',
                a=f'node_{first}',
                b=f'node_{second}',
                K=2 * self.K if frozenset((first, second)) == doubled and index < self.n_nodes else self.K,
            )
            for index, (first, second) in enumerate(pairs)
        ]
        self.system_ = System(self.label, elements=elements, interactions=swings + lines)
