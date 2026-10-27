"""Quantum wave models: the Schrödinger equation as two real fields or as a particle on the sites of a lattice."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any

import numpy as np

from skcomplex.core.declarations import state

from ....core.base import BaseInteraction, NumericDomain, ValidationError, parameter, role
from ....core.components import Element
from ....core.elements import FieldElement
from ....core.laws import DissipatorLaw, GeneratorLaw, RateLaw, Symbols
from ....core.laws.utils import laplacian
from ....core.spaces import LatticeSpace
from ....core.system import System
from ....engine.runtime import register_model
from ....engine.solvers import ETDRK4Solver
from ...core.base import BaseLaw
from ..realizations.quantum import SiteElement


class WaveFunctionElement(Element):
    """Wave function element."""

    u = state(0.0)
    w = state(0.0)
    D = parameter(default=0.5, domain=NumericDomain.POSITIVE)
    V = parameter(0.0)

    def laws(self, s: Symbols) -> list[BaseLaw]:
        """Return the laws of the wave function element."""
        return [
            RateLaw(s.u, -s.D * laplacian(s.w) + s.V * s.w),
            RateLaw(s.w, s.D * laplacian(s.u) - s.V * s.u),
        ]


FORMULATIONS = ('fields', 'operators')


class LatticeHamiltonianInteraction(BaseInteraction):
    """Hamiltonian `-D laplacian + V` of a particle on the sites of a lattice.

    Args:
        label:
            The label of the interaction.

        potential:
            The potential at every site.

        **values:
            The values of the declared variables, by name:

            target:
                The path of the particle.

            D:
                The coefficient of the Laplacian, `ħ / (2 m)`.

    Examples:
        >>> from skcomplex.models.physics.waves import LatticeHamiltonianInteraction
        >>> LatticeHamiltonianInteraction('kinetic', target='packet').D
        0.5
    """

    target: str = role()
    D: float = parameter(default=0.5, domain=NumericDomain.POSITIVE)

    def __init__(
        self,
        label: str,
        *,
        potential: Any = 0.0,  # noqa: ANN401
        **values: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(label, **values)
        self.potential = potential

    def laws(self, s: Symbols) -> list[Any]:
        """Return the Hamiltonian of the particle.

        Args:
            s:
                The namespace of the interaction.

        Returns:
            laws:
                The Hamiltonian.
        """
        return [GeneratorLaw(-s.D * s.target.laplacian + s.target.diagonal(self.potential))]


class PositionDephasingInteraction(BaseInteraction):
    """Dephasing of a particle in the position basis, one dissipator per site projector at the rate `gamma`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle.

        gamma:
            The dephasing rate.

    Examples:
        >>> from skcomplex.models.physics.waves import PositionDephasingInteraction
        >>> PositionDephasingInteraction('dephasing', target='packet', gamma=0.5).gamma
        0.5
    """

    target: str = role()
    gamma: float = parameter(default=0.1, domain=NumericDomain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Any]:
        """Return the dissipators of the site projectors.

        Args:
            s:
                The namespace of the interaction.

        Returns:
            laws:
                The dissipators.
        """
        return [DissipatorLaw(s.target.projector(site), s.gamma) for site in s.target.sites]


@register_model(
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
            The model, with the real part `packet.u` and the imaginary part `packet.w` of the wave function, or the
            quantum state `schrodinger.state` of the operator formulation.

    Raises:
        ValidationError:
            The formulation is unknown, or the field formulation has a dephasing rate.

    Examples:
        >>> from skcomplex.models.physics.waves import schrodinger
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
            LatticeHamiltonianInteraction('kinetic', target='packet', D=D, potential=potential),
        ]
        if gamma > 0:
            interactions.append(PositionDephasingInteraction('dephasing', target='packet', gamma=gamma))
        return System(
            'schrodinger',
            elements=[SiteElement('packet', space=space, initial=amplitudes)],
            interactions=interactions,
        )
    packet = FieldElement(
        'packet',
        WaveFunctionElement,
        space,
        values={'u': psi.real, 'w': psi.imag, 'D': D, 'V': potential},
        discretization='spectral',
    )
    return System(elements=[packet])
