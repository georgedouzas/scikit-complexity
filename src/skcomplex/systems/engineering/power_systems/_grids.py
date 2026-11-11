"""Power grid systems: generators and consumers that synchronise.

They synchronise through transmission lines.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo, sin
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import CouplingInteraction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import derivative
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.components.types import Target
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec


class GridNodeElement(Element):
    """Node of a power grid whose phase follows the swing equation.

    The swing equation has a power and a damping.
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        phi: Annotated[object, StateSpec(order=2)] = 0.0,
        phi_d1: float = 0.0,
        P: Annotated[object, ParameterSpec()] = 1.0,
        gamma: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.phi = phi
        self.phi_d1 = phi_d1
        self.P = P
        self.gamma = gamma

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the swing equation of the phase."""
        return [RateLaw(s.phi, s.P - s.gamma * derivative(s.phi, 1))]


class TransmissionLineInteraction(CouplingInteraction):
    """Transmission line of a capacity that couples the phases of two nodes.

    The coupling is the sine of their difference. It is the coupling of the two nodes as separate elements, each getting
    `K sin(phi_j - phi_i)` from the other.

    Args:
        label:
            The label of the interaction.

        ends:
            The paths of the two nodes the line joins.

        K:
            The capacity of the line.
    """

    def __init__(
        self,
        label: str,
        *,
        ends: Annotated[Sequence[str], RoleSpec()],
        K: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, target=ends, read='phi', write='phi', function=self._get_flow, mode='group')
        self.ends = ends
        self.K = K

    @staticmethod
    def _get_flow(namespace: SimpleNamespace) -> Target:
        """Return the power flow from the other end of the line, `K sin(phi_j - phi_i)`."""
        return namespace.K * sin(namespace.other.phi - namespace.member.phi)


class PowerGridSystem(BaseSystem):
    """A ring power grid in the swing-equation system.

    More transmission capacity can break synchrony. The phase of node `i` relative to the grid frequency follows
    `phi_i'' = P_i - gamma phi_i' + Σ_j K_ij sin(phi_j - phi_i)`. The first half of the ring are generators with
    `P = power` and the second half consumers with `P = -power`, joined by lines of capacity `K`. The default grid
    synchronises, but adding the line `(0, 2)` or doubling the capacity of the line `(0, 1)` makes the flows overload a
    line, and synchrony is lost, which is Braess's paradox. The system has the phases `node_<i>.phi` and frequency
    deviations `node_<i>.phi_d1`.

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

    Examples:
        >>> from skcomplex.systems.engineering.power_systems import PowerGridSystem
        >>> len(PowerGridSystem(added_line=(0, 2)).prepare().system_.interactions)
        9
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
    ) -> None:
        super().__init__(label)
        self.n_nodes = n_nodes
        self.K = K
        self.power = power
        self.gamma = gamma
        self.added_line = added_line
        self.doubled_line = doubled_line

    def _build_system(self) -> CompositeSystem:
        """Return the components of the system built from its model parameters."""
        elements = [
            GridNodeElement(
                f'node_{index}',
                phi=0.0,
                phi_d1=0.0,
                P=self.power if index < self.n_nodes // 2 else -self.power,
                gamma=self.gamma,
            )
            for index in range(self.n_nodes)
        ]
        pairs = [(index, (index + 1) % self.n_nodes) for index in range(self.n_nodes)]
        doubled = None if self.doubled_line is None else frozenset(self.doubled_line)
        interactions = [
            TransmissionLineInteraction(
                f'line_{first}_{second}',
                ends=[f'node_{first}', f'node_{second}'],
                K=2 * self.K if frozenset((first, second)) == doubled else self.K,
            )
            for first, second in pairs
        ]
        if self.added_line is not None:
            first, second = self.added_line
            interactions.append(
                TransmissionLineInteraction(
                    f'line_{first}_{second}',
                    ends=[f'node_{first}', f'node_{second}'],
                    K=self.K,
                ),
            )
        return CompositeSystem(self.label, elements=elements, interactions=interactions)
