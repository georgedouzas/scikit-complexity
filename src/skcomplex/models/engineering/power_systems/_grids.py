"""Power grid models: generators and consumers that synchronise through transmission lines."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo, sin
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.interactions import Interaction
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import derivative
from ....core.system import System
from ....specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_model


class GridNodeElement(Element):
    """Grid node element."""

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
        """Return the laws of the grid node element."""
        return [RateLaw(s.phi, s.P - s.gamma * derivative(s.phi, 1))]


class TransmissionLineInteraction(Interaction):
    """Transmission line interaction."""

    def __init__(
        self,
        label: str,
        *,
        first: Annotated[object, RoleSpec()],
        second: Annotated[object, RoleSpec()],
        K: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label)
        self.first = first
        self.second = second
        self.K = K

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the transmission line interaction."""
        return [
            RateLaw(s.first.phi, s.K * sin(s.second.phi - s.first.phi)),
            RateLaw(s.second.phi, s.K * sin(s.first.phi - s.second.phi)),
        ]


@register_model(
    field='engineering/power_systems',
    tags=('power grid', 'synchronization', 'network', 'braess paradox', 'ode'),
    references=('Witthaut and Timme 2012', 'Datseris and Parlitz 2022'),
    defaults={'until': 300.0, 'dt': 1.0},
)
def power_grid(
    n_nodes: int = 8,
    *,
    K: float = 2.75,
    power: float = 1.0,
    gamma: float = 0.1,
    added_line: tuple[int, int] | None = None,
    doubled_line: tuple[int, int] | None = None,
) -> System:
    """Return a ring power grid in the swing-equation model, where more transmission capacity can break synchrony.

    The phase of node `i` relative to the grid frequency follows
    `phi_i'' = P_i - gamma phi_i' + Σ_j K_ij sin(phi_j - phi_i)`.
    The first half of the ring are generators with `P = power` and the second half consumers with `P = -power`, joined
    by lines of capacity `K`. The default grid synchronises, but adding the line `(0, 2)` or doubling the capacity of
    the line `(0, 1)` makes the flows overload a line, and synchrony is lost, which is Braess's paradox.

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

    Returns:
        system:
            The model, with the phases `node_<i>.phi` and frequency deviations `node_<i>.phi_d1`.

    Examples:
        >>> from skcomplex.models.engineering.power_systems import power_grid
        >>> len(power_grid(added_line=(0, 2)).interactions)
        9
    """
    elements = [
        GridNodeElement(
            f'node_{index}',
            phi=0.0,
            phi_d1=0.0,
            P=power if index < n_nodes // 2 else -power,
            gamma=gamma,
        )
        for index in range(n_nodes)
    ]
    pairs = [(index, (index + 1) % n_nodes) for index in range(n_nodes)]
    doubled = None if doubled_line is None else frozenset(doubled_line)
    interactions = [
        TransmissionLineInteraction(
            f'line_{first}_{second}',
            first=f'node_{first}',
            second=f'node_{second}',
            K=2 * K if frozenset((first, second)) == doubled else K,
        )
        for first, second in pairs
    ]
    if added_line is not None:
        first, second = added_line
        interactions.append(
            TransmissionLineInteraction(f'line_{first}_{second}', first=f'node_{first}', second=f'node_{second}', K=K),
        )
    return System(elements=elements, interactions=interactions)
