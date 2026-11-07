"""Quantum wave systems: the Schrödinger equation as two real fields or as a particle on the sites of a lattice."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, FieldElement
from ....core.components.interactions import BaseInteraction, Interaction
from ....core.components.laws import BaseLaw, DissipatorLaw, GeneratorLaw, RateLaw
from ....core.components.laws.utils import laplacian, local_operator
from ....core.components.spaces import LatticeSpace
from ....core.components.system import System
from ....core.engine.solvers import ETDRK4Solver
from ....core.errors import ValidationError
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_system
from ..common.quantum import SiteElement, site_space


class WaveFunctionElement(Element):
    """Site of the wave function `psi = u + i w` of a particle with the potential `V`, by its real and imaginary parts.

    The parts follow the Schrödinger equation `i psi' = -D laplacian(psi) + V psi`, so
    `u' = -D laplacian(w) + V w` and `w' = D laplacian(u) - V u`.

    Args:
        label:
            The label of the site.

        visual:
            How the site is drawn, or `None` for none.

        u:
            The initial real part.

        w:
            The initial imaginary part.

        D:
            The coefficient of the kinetic energy, `hbar / (2 m)`.

        V:
            The potential at the site.

    Examples:
        >>> from skcomplex.systems.physics.waves import WaveFunctionElement
        >>> WaveFunctionElement('site', V=1.0).V
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[float, StateSpec()] = 0.0,
        w: Annotated[float, StateSpec()] = 0.0,
        D: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        V: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u
        self.w = w
        self.D = D
        self.V = V

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the Schrödinger equation of the real and imaginary parts.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate laws of the two parts.
        """
        return [
            RateLaw(s.u, -s.D * laplacian(s.w) + s.V * s.w),
            RateLaw(s.w, s.D * laplacian(s.u) - s.V * s.u),
        ]


class LatticeHamiltonianInteraction(Interaction):
    """Hamiltonian `-D laplacian + V` of a particle on the sites of a lattice.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle, whose Hilbert space holds the operators `laplacian` and `potential`.

        D:
            The coefficient of the Laplacian, `ħ / (2 m)`.

    Examples:
        >>> from skcomplex.systems.physics.waves import LatticeHamiltonianInteraction
        >>> LatticeHamiltonianInteraction('kinetic', target='packet').D
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        D: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.D = D

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the Hamiltonian of the particle.

        Args:
            s:
                The namespace of the interaction.

        Returns:
            laws:
                The Hamiltonian.
        """
        psi = s.target.psi
        return [GeneratorLaw(-s.D * local_operator(psi, 'laplacian') + local_operator(psi, 'potential'))]


class PositionDephasingInteraction(Interaction):
    """Dephasing of a particle in the position basis, one dissipator per site projector at the rate `gamma`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle, whose Hilbert space holds the site projectors `site_<k>`.

        gamma:
            The dephasing rate.

        n_sites:
            The number of sites.

    Examples:
        >>> from skcomplex.systems.physics.waves import PositionDephasingInteraction
        >>> PositionDephasingInteraction('dephasing', target='packet', gamma=0.5).gamma
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        gamma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        n_sites: int = 1,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.gamma = gamma
        self.n_sites = n_sites

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the dissipators of the site projectors.

        Args:
            s:
                The namespace of the interaction.

        Returns:
            laws:
                The dissipators.
        """
        return [DissipatorLaw(local_operator(s.target.psi, f'site_{site}'), s.gamma) for site in range(self.n_sites)]


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
