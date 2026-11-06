"""Quantum spin chains: the quantum Heisenberg XXZ chain with a field and dephasing."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression

from ....core.components.elements import LatticeElement
from ....core.components.interactions import BaseInteraction
from ....core.components.laws.utils import local_operator
from ....core.components.spaces import LatticeSpace
from ....core.components.system import System
from ....core.engine.compile import OperatorCompiler
from ....core.errors import ValidationError
from ....core.specs import ObservableSpec
from ..._catalog import register_system
from ..common.quantum import SpinElement
from ._dephasing import DephasingInteraction
from ._xxz_exchange import XXZExchangeInteraction
from ._zeeman_field import ZeemanFieldInteraction

INITIAL_STATES = ('neel', 'flip', 'domain_wall', 'ground')


BOUNDARIES = {'periodic': 'periodic', 'open': 'fixed'}


def _initial_labels(n_sites: int, initial: str) -> list[str]:
    """Return the basis label of every spin of a product initial state."""
    if initial == 'neel':
        return ['up' if site % 2 == 0 else 'down' for site in range(n_sites)]
    if initial == 'flip':
        return ['down' if site == n_sites // 2 else 'up' for site in range(n_sites)]
    return ['up' if site < n_sites // 2 else 'down' for site in range(n_sites)]


def _sites(values: object) -> object:
    """Return the values of every site, without reducing them."""
    return values


def local_magnetization(member: SimpleNamespace) -> Expression:
    """Return the `z` component of one spin.

    Args:
        member:
            The namespace of one spin.

    Returns:
        operator:
            The local magnetisation.
    """
    return local_operator(member.psi, 'z')


def total_magnetization(s: SimpleNamespace) -> Expression:
    """Return the sum of the `z` components of the spins of the chain.

    Args:
        s:
            The namespace of the system.

    Returns:
        operator:
            The total magnetisation.
    """
    return sum(local_operator(member.psi, 'z') for member in s.chain)


@register_system(
    field='physics/statistical_mechanics',
    tags=('quantum', 'spins', 'lattice', 'magnetism', 'open system'),
    references=('Heisenberg 1928', 'Mihailović et al. 2023'),
    defaults={'until': 8.0, 'dt': 0.05, 'compiler': OperatorCompiler()},
    initial_state=lambda initial, **_: {'quantum_heisenberg.state': 'ground'} if initial == 'ground' else None,
    observables=lambda system: [
        ObservableSpec('sz', local_magnetization, group='chain', reduce=_sites),
        ObservableSpec('magnetization', total_magnetization),
    ],
)
def quantum_heisenberg(
    n_sites: int = 12,
    *,
    J: float = 1.0,
    delta: float = 1.0,
    h: float = 0.0,
    gamma: float = 0.0,
    boundary: str = 'periodic',
    initial: str = 'flip',
) -> System:
    """Return a chain of spins one half with the XXZ Hamiltonian, a field, and optional dephasing.

    The Hamiltonian is `J Σ (S^x_i S^x_j + S^y_i S^y_j + Δ S^z_i S^z_j) - h Σ S^z_i` over neighbouring pairs, with
    `ħ = 1`. A dephasing rate `gamma > 0` adds the dissipator `S^z_i` on every site, so the state is a density matrix
    that loses its coherence. One flipped spin in a ferromagnetic background spreads as a magnon, whose probability at
    distance `d` is `J_d(J t)²` in the XX chain, a light cone in the space-time image of `S^z`. The total magnetisation
    is conserved.

    Args:
        n_sites:
            The number of spins.

        J:
            The exchange coupling, antiferromagnetic when positive.

        delta:
            The anisotropy `Δ`.

        h:
            The field.

        gamma:
            The dephasing rate.

        boundary:
            `'periodic'` for a ring, or `'open'` for a chain with two ends.

        initial:
            `'flip'` for one flipped spin at site `n_sites // 2`, `'neel'` for alternating spins, `'domain_wall'` for
            the left half up and the right half down, or `'ground'` for the ground state, which `run` gives the
            simulation as `initial_state={'quantum_heisenberg.state': 'ground'}`.

    Returns:
        system:
            The system, with the observables `sz` of every site, `magnetization`, and the energy.

    Raises:
        ValidationError:
            The boundary or the initial state is unknown.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import quantum_heisenberg
        >>> quantum_heisenberg(6).elements[0].prepare().size_
        6
    """
    if boundary not in BOUNDARIES or initial not in INITIAL_STATES:
        error_msg = (
            f'Unknown boundary {boundary!r} or initial state {initial!r}. Use a boundary of {", ".join(BOUNDARIES)} '
            f'and an initial state of {", ".join(INITIAL_STATES)}.'
        )
        raise ValidationError(error_msg)
    space = LatticeSpace(n_sites, boundary=BOUNDARIES[boundary])
    labels = _initial_labels(n_sites, initial)
    chain = LatticeElement('chain', element=SpinElement, space=space, values={'psi': labels})
    interactions: list[BaseInteraction] = [
        XXZExchangeInteraction('exchange', target='chain', J=J, delta=delta),
        ZeemanFieldInteraction('field', target='chain', h=h),
    ]
    if gamma > 0:
        interactions.append(DephasingInteraction('dephasing', target='chain', gamma=gamma))
    return System('quantum_heisenberg', elements=[chain], interactions=interactions)
