"""Quantum wave models: the Schrödinger equation as two real fields or as a particle on the sites of a lattice."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any

import numpy as np

from ....core.base import Domain, ValidationError, parameter, role
from ....core.components import Element, Field
from ....core.interactions import Interaction
from ....core.laws import Dissipator, Generator, Rate, Symbols, laplacian
from ....core.runtime import register_model
from ....core.solvers import ETDRK4
from ....core.space import LatticeSpace
from ....core.system import System
from ..realizations import Site

WaveFunction = Element.define(
    'WaveFunction',
    states={'u': 0.0, 'w': 0.0},
    parameters={'D': parameter(default=0.5, domain=Domain.POSITIVE), 'V': 0.0},
    laws=lambda s: [
        Rate(s.u, -s.D * laplacian(s.w) + s.V * s.w),
        Rate(s.w, s.D * laplacian(s.u) - s.V * s.u),
    ],
)

FORMULATIONS = ('fields', 'operators')


class LatticeHamiltonian(Interaction):
    """Hamiltonian `-D laplacian + V` of a particle on the sites of a lattice.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle.

        D:
            The coefficient of the Laplacian, `ħ / (2 m)`.

        potential:
            The potential at every site.

    Examples:
        >>> from skcomplex.models.physics.waves import LatticeHamiltonian
        >>> LatticeHamiltonian('kinetic', target='packet').D
        0.5
    """

    target: str = role()
    D: float = parameter(default=0.5, domain=Domain.POSITIVE)
    potential: Any = 0.0

    def laws(self, s: Symbols) -> list[Any]:
        """Return the Hamiltonian of the particle.

        Args:
            s:
                The namespace of the interaction.

        Returns:
            laws:
                The Hamiltonian.
        """
        return [Generator(-s.D * s.target.laplacian + s.target.diagonal(self.potential))]


class PositionDephasing(Interaction):
    """Dephasing of a particle in the position basis, one dissipator per site projector at the rate `gamma`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle.

        gamma:
            The dephasing rate.

    Examples:
        >>> from skcomplex.models.physics.waves import PositionDephasing
        >>> PositionDephasing('dephasing', target='packet', gamma=0.5).gamma
        0.5
    """

    target: str = role()
    gamma: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Any]:
        """Return the dissipators of the site projectors.

        Args:
            s:
                The namespace of the interaction.

        Returns:
            laws:
                The dissipators.
        """
        return [Dissipator(s.target.projector(site), s.gamma) for site in s.target.sites]


@register_model(
    field='physics/waves',
    tags=('waves', 'field', 'pde', 'quantum', 'spectral'),
    references=('Schrödinger 1926', 'Schwichtenberg 2020'),
    defaults={'until': 4.0, 'dt': 0.1, 'solver': ETDRK4(dt=0.01)},
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
    every site. It runs with the quantum solvers, so it takes `settings={'solver': 'auto'}`, and it accepts position
    dephasing at the rate `gamma`, which turns the state into a density matrix.

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
            The model, with the real part `packet.u` and the imaginary part `packet.w` of the wave function, or the
            quantum state `schrodinger.state` of the operator formulation.

    Raises:
        ValidationError:
            The formulation is unknown, or the field formulation has a dephasing rate.

    Examples:
        >>> from skcomplex.models.physics.waves import schrodinger
        >>> schrodinger(64).elements[0].size
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
        interactions: list[Interaction] = [LatticeHamiltonian('kinetic', target='packet', D=D, potential=potential)]
        if gamma > 0:
            interactions.append(PositionDephasing('dephasing', target='packet', gamma=gamma))
        return System(
            'schrodinger',
            components=[Site('packet', space=space, initial=amplitudes)],
            interactions=interactions,
        )
    packet = Field(
        'packet',
        WaveFunction,
        space,
        values={'u': psi.real, 'w': psi.imag, 'D': D, 'V': potential},
        discretization='spectral',
    )
    return System(components=[packet])
