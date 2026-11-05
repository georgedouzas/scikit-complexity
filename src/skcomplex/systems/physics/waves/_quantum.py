"""Quantum wave systems: the Schrödinger equation as two real fields or as a particle on the sites of a lattice."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ....core.components.elements import FieldElement
from ....core.components.interactions import BaseInteraction
from ....core.components.spaces import LatticeSpace
from ....core.components.system import System
from ....core.engine.solvers import ETDRK4Solver
from ....core.errors import ValidationError
from ..._catalog import register_system
from ..common.quantum import SiteElement, site_space
from ._lattice_hamiltonian import LatticeHamiltonianInteraction
from ._position_dephasing import PositionDephasingInteraction
from ._wave_function import WaveFunctionElement

FORMULATIONS = ('fields', 'operators')


@register_system(
    field='physics/waves',
    tags=('waves', 'field', 'pde', 'quantum', 'spectral'),
    references=('Schrödinger 1926', 'Schwichtenberg 2020'),
    defaults={'until': 4.0, 'dt': 0.1, 'solver': ETDRK4Solver(dt=0.01)},
)
def schrodinger(
    n_sites: int = 256,
    *,
    length: float = 40.0,
    D: float = 0.5,
    momentum: float = 2.0,
    width: float = 1.0,
    omega: float = 0.0,
    formulation: str = 'fields',
    gamma: float = 0.0,
) -> System:
    """Return a Gaussian wave packet that follows the Schrödinger equation, as its real and imaginary parts.

    The wave function `psi = u + i w` follows `psi_t = i D laplacian(psi) - i V psi`, written as `u_t = -D laplacian(w)
    + V w` and `w_t = D laplacian(u) - V u`, on a periodic lattice with spectral derivatives. In units with `ħ = 1`,
    `D = 1 / (2 m)`. The potential is the harmonic well `V = omega² x² / (4 D)`, or zero for a free packet. A free
    packet of momentum `k` moves at the group velocity `2 D k` and spreads, and its norm `sum(u² + w²)` is conserved.

    The operator formulation builds the same particle as one quantum subsystem whose basis states are the sites, with
    the Hamiltonian `-D laplacian + V` of the same spectral Laplacian, normalised so that `|psi|²` is the probability of
    every site. It runs with the quantum solvers, and it accepts position dephasing at the rate `gamma`, which turns
    the state into a density matrix.

    Args:
        n_sites:
            The number of sites of the periodic lattice.

        length:
            The length of the periodic box, centered on 0.

        D:
            The coefficient of the Laplacian, `ħ / (2 m)`.

        momentum:
            The mean wave number of the packet.

        width:
            The width of the packet.

        omega:
            The angular frequency of the harmonic well.

        formulation:
            `'fields'` for two real fields, or `'operators'` for a particle on the sites of the lattice.

        gamma:
            The rate of position dephasing of the operator formulation.

    Returns:
        system:
            The system, with the real part `packet.u` and the imaginary part `packet.w` of the wave function, or the
            quantum state `schrodinger.state` of the operator formulation.

    Raises:
        ValidationError:
            The formulation is unknown, or the field formulation has a dephasing rate.

    Examples:
        >>> from skcomplex.systems.physics.waves import schrodinger
        >>> schrodinger(64).elements[0].prepare().size_
        64
    """
    spacing = length / n_sites
    positions = spacing * np.arange(n_sites) - length / 2
    envelope = np.exp(-(positions**2) / (4 * width**2)) / (2 * np.pi * width**2) ** 0.25
    psi = envelope * np.exp(1j * momentum * positions)
    potential = omega**2 * positions**2 / (4 * D)
    space = LatticeSpace(n_sites, spacing=spacing)
    if formulation not in FORMULATIONS or (formulation == 'fields' and gamma):
        error_msg = (
            f'Unknown formulation {formulation!r}, or dephasing in the field formulation. Use one of '
            f'{", ".join(FORMULATIONS)}, and gamma > 0 only with the operator formulation.'
        )
        raise ValidationError(error_msg)
    if formulation == 'operators':
        amplitudes = psi / np.linalg.norm(psi)
        interactions: list[BaseInteraction] = [
            LatticeHamiltonianInteraction('kinetic', target='packet', D=D),
        ]
        if gamma > 0:
            interactions.append(
                PositionDephasingInteraction('dephasing', target='packet', gamma=gamma, n_sites=n_sites),
            )
        return System(
            'schrodinger',
            elements=[SiteElement('packet', amplitudes, hilbert=site_space(space, potential, projectors=gamma > 0))],
            interactions=interactions,
        )
    packet = FieldElement(
        'packet',
        element=WaveFunctionElement,
        space=space,
        values={'u': psi.real, 'w': psi.imag, 'D': D, 'V': potential},
        discretization='spectral',
    )
    return System(elements=[packet])
