"""Quantum spin chains: the quantum Heisenberg XXZ chain with a field and dephasing."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any

from ....core.base import Domain, ValidationError, parameter, role
from ....core.model import (
    Dissipator,
    Exchange,
    Generator,
    Interaction,
    LatticeSpace,
    Operator,
    QuantumSystem,
    Spin,
    Symbols,
    expectation,
)
from ....core.runtime import register_model

INITIAL_STATES = ('neel', 'flip', 'domain_wall', 'ground')
BOUNDARIES = {'periodic': 'periodic', 'open': 'fixed'}


def _initial_labels(n_sites: int, initial: str) -> list[str]:
    """Return the basis label of every spin of a product initial state."""
    if initial == 'neel':
        return ['up' if site % 2 == 0 else 'down' for site in range(n_sites)]
    if initial == 'flip':
        return ['down' if site == n_sites // 2 else 'up' for site in range(n_sites)]
    return ['up' if site < n_sites // 2 else 'down' for site in range(n_sites)]


def local_magnetization(member: Symbols) -> Operator:
    """Return the `z` component of one spin.

    Args:
        member:
            The namespace of one spin.

    Returns:
        operator:
            The local magnetisation.
    """
    return member.z


def total_magnetization(s: Symbols) -> Operator:
    """Return the sum of the `z` components of the spins of the chain.

    Args:
        s:
            The namespace of the system.

    Returns:
        operator:
            The total magnetisation.
    """
    return sum((member.z for member in s.chain), start=Operator({}))


class XXZExchange(Exchange):
    """Exchange `J (S^x_i S^x_j + S^y_i S^y_j + Δ S^z_i S^z_j)` between neighbouring spins.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the spin chain.

        J:
            The exchange coupling, antiferromagnetic when positive.

        delta:
            The anisotropy `Δ`, 1 for the isotropic chain and 0 for the XX chain.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import XXZExchange
        >>> XXZExchange('exchange', target='chain', delta=0.5).delta
        0.5
    """

    J: float = parameter(default=1.0, domain=Domain.REAL)
    delta: float = parameter(default=1.0, domain=Domain.REAL)

    def laws(self, s: Symbols) -> list[Any]:
        """Return the exchange Hamiltonian of one bond.

        Args:
            s:
                The namespace of the bond, with `s.first` and `s.second`.

        Returns:
            laws:
                The Hamiltonian of the bond.
        """
        first, second = s.first, s.second
        return [Generator(s.J * (first.x * second.x + first.y * second.y + s.delta * first.z * second.z))]


class ZeemanField(Interaction):
    """Uniform field `-h Σ S^z_i` on the spins of a chain.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the spin chain.

        h:
            The field.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import ZeemanField
        >>> ZeemanField('field', target='chain', h=0.2).h
        0.2
    """

    target: str = role()
    h: float = parameter(default=0.0, domain=Domain.REAL)

    def laws(self, s: Symbols) -> list[Any]:
        """Return the Hamiltonian of the field.

        Args:
            s:
                The namespace of the interaction.

        Returns:
            laws:
                The field Hamiltonian.
        """
        return [Generator(-s.h * sum((member.z for member in s.target), start=Operator({})))]


class Dephasing(Interaction):
    """Dephasing of every spin of a chain, a dissipator with the collapse operator `S^z_i` at the rate `gamma`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the spin chain.

        gamma:
            The dephasing rate.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import Dephasing
        >>> Dephasing('dephasing', target='chain', gamma=0.1).gamma
        0.1
    """

    target: str = role()
    gamma: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Any]:
        """Return one dissipator per spin.

        Args:
            s:
                The namespace of the interaction.

        Returns:
            laws:
                The dissipators.
        """
        return [Dissipator(member.z, s.gamma) for member in s.target]


@register_model(
    field='physics/statistical_mechanics',
    tags=('quantum', 'spins', 'lattice', 'magnetism', 'open system'),
    references=('Heisenberg 1928', 'Mihailović et al. 2023'),
    defaults={'until': 8.0, 'dt': 0.05},
    observables=lambda system: [
        expectation('sz', local_magnetization, group='chain'),
        expectation('magnetization', total_magnetization),
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
) -> QuantumSystem:
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
            the left half up and the right half down, or `'ground'` for the ground state.

    Returns:
        system:
            The model, with the observables `sz` of every site, `magnetization`, and the energy.

    Raises:
        ValidationError:
            The boundary or the initial state is unknown.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import quantum_heisenberg
        >>> quantum_heisenberg(6).elements[0].size
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
    chain = Spin.lattice('chain', space, initial=labels)
    interactions: list[Interaction] = [
        XXZExchange('exchange', target='chain', J=J, delta=delta),
        ZeemanField('field', target='chain', h=h),
    ]
    if gamma > 0:
        interactions.append(Dephasing('dephasing', target='chain', gamma=gamma))
    return QuantumSystem(
        'quantum_heisenberg',
        components=[chain],
        interactions=interactions,
        initial='ground' if initial == 'ground' else None,
    )
