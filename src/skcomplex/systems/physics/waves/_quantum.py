"""Quantum wave systems: the Schrödinger equation as two real fields or as a particle on the sites of a lattice."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from sage.all__sagemath_symbolics import CC, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, FieldElement
from ....core.components.interactions import BaseInteraction, Interaction
from ....core.components.laws import BaseLaw, DissipatorLaw, GeneratorLaw, RateLaw
from ....core.components.laws.utils import laplacian, local_operator
from ....core.components.spaces import BaseSpace, LatticeSpace, NetworkSpace
from ....core.components.system import BaseSystem, RunSettings, System
from ....core.engine.compile import OperatorCompiler
from ....core.engine.solvers import ETDRK4Solver
from ....core.errors import ValidationError
from ....core.specs import HilbertSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec


class SiteElement(Element):
    """Single particle on the sites of a lattice or a network, whose basis state `k` is the particle at site `k`.

    Its laws read the operators of `site_space`: `laplacian`, `adjacency`, and `potential`.

    Args:
        label:
            The label of the particle.

        psi:
            The state in the Hilbert space: the initial site, or a vector of amplitudes over the sites.

        hilbert:
            The Hilbert space, `site_space(space, potential)`.

        visual:
            How the particle is drawn.

    Examples:
        >>> from skcomplex.core.components.spaces import LatticeSpace
        >>> from skcomplex.systems.physics.waves._quantum import SiteElement, site_space
        >>> SiteElement('particle', hilbert=site_space(LatticeSpace(16))).hilbert.dimension
        16
    """

    def __init__(
        self,
        label: str,
        psi: Annotated[object, StateSpec(domain=CC)] = 0,
        *,
        hilbert: HilbertSpec | None = None,
        visual: VisualSpec | None = VisualSpec('expectations'),
    ) -> None:
        super().__init__(label, visual=visual)
        self.psi = psi
        self.hilbert = hilbert


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

    law_classes = (RateLaw,)

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


def _lattice_laplacian(lattice: LatticeSpace) -> npt.NDArray[np.float64]:
    """Return the Laplacian of a prepared lattice, spectral when it is periodic and of finite differences otherwise."""
    size = int(lattice.size_)
    laplacian = np.zeros((size, size))
    for axis, (length, spacing) in enumerate(zip(lattice.shape_, lattice.spacings_, strict=True)):
        if lattice.periodic_:
            waves = 2 * np.pi * np.fft.fftfreq(length, d=spacing)
            line = np.fft.ifft(-(waves[:, None] ** 2) * np.fft.fft(np.eye(length), axis=0), axis=0).real
        else:
            line = (np.eye(length, k=-1) - 2 * np.eye(length) + np.eye(length, k=1)) / spacing**2
        term: npt.NDArray[np.floating] = np.ones((1, 1))
        for index, item in enumerate(lattice.shape_):
            term = np.kron(term, line if index == axis else np.eye(item))
        laplacian += term
    return laplacian


def site_space(space: BaseSpace, potential: npt.ArrayLike | None = None, *, projectors: bool = False) -> HilbertSpec:
    """Return the Hilbert space of a particle on the sites of a lattice or a network, whose basis state `k` is site `k`.

    Its operators are the Laplacian, `laplacian`, the adjacency of the neighbouring sites, `adjacency`, with a
    potential the operator `potential` with the potential of every site on its diagonal, and with projectors the
    projector `site_<k>` on every site `k`. The Laplacian of a lattice is spectral when the lattice is periodic and
    of finite differences otherwise, with the spacing of the lattice, and the Laplacian of a network is its graph
    Laplacian.

    Args:
        space:
            The lattice or the network.

        potential:
            The potential of every site, or `None` for no potential.

        projectors:
            Whether the space holds the projector on every site.

    Returns:
        space:
            The Hilbert space, of the dimension of the number of sites.

    Raises:
        ValidationError:
            The space is neither a lattice nor a network.

    Examples:
        >>> from skcomplex.core.components.spaces import LatticeSpace
        >>> from skcomplex.systems.physics.waves._quantum import site_space
        >>> float(site_space(LatticeSpace(16, boundary='fixed')).operators['laplacian'][0].trace())
        -32.0
    """
    if not isinstance(space, LatticeSpace | NetworkSpace):
        error_msg = f'A particle on sites needs a LatticeSpace or a NetworkSpace, not {space!r}.'
        raise ValidationError(error_msg)
    prepared = space.prepare()
    size = int(prepared.size_)
    adjacency = np.zeros((size, size))
    for first, second in prepared._neighbor_pairs():
        adjacency[first, second] = adjacency[second, first] = 1.0
    laplacian = (
        _lattice_laplacian(prepared)
        if isinstance(prepared, LatticeSpace)
        else adjacency - np.diag(adjacency.sum(axis=1))
    )
    operators = {'laplacian': (laplacian, True), 'adjacency': (adjacency, True)}
    if potential is not None:
        operators['potential'] = (np.diag(np.broadcast_to(np.asarray(potential, dtype=float), (size,))), True)
    for site in range(size if projectors else 0):
        operators[f'site_{site}'] = (sp.csr_array(([1.0], ([site], [site])), shape=(size, size)), True)
    return HilbertSpec(size, operators, display=None)


class SchrodingerSystem(BaseSystem):
    """A Gaussian wave packet that follows the Schrödinger equation, as its real and imaginary parts.

    The wave function `psi = u + i w` follows `psi_t = i D laplacian(psi) - i V psi`, written as `u_t = -D laplacian(w)
    + V w` and `w_t = D laplacian(u) - V u`, on a periodic lattice with spectral derivatives. In units with `ħ = 1`,
    `D = 1 / (2 m)`. The potential is the harmonic well `V = omega² x² / (4 D)`, or zero for a free packet. A free
    packet of momentum `k` moves at the group velocity `2 D k` and spreads, and its norm `sum(u² + w²)` is conserved.

    The operator formulation builds the same particle as one quantum subsystem whose basis states are the sites, with
    the Hamiltonian `-D laplacian + V` of the same spectral Laplacian, normalised so that `|psi|²` is the probability of
    every site. It runs with the quantum solvers, and it accepts position dephasing at the rate `gamma`, which turns the
    state into a density matrix. The system has the real part `packet.u` and the imaginary part `packet.w` of the wave
    function, or the quantum state `<label>.state` of the operator formulation.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ValidationError:
            The formulation is unknown, or the field formulation has a dephasing rate.

    Examples:
        >>> from skcomplex.systems.physics.waves import SchrodingerSystem
        >>> SchrodingerSystem(64).prepare().system_.elements[0].prepare().size_
        64
    """

    field = 'physics/waves'
    topics = ('waves', 'field', 'pde', 'quantum', 'spectral')
    references = ('Schrödinger 1926', 'Schwichtenberg 2020')
    until = 4.0
    dt = 0.1
    solver = ETDRK4Solver(dt=0.01)

    def __init__(
        self,
        n_sites: int = 256,
        *,
        length: float = 40.0,
        D: float = 0.5,
        momentum: float = 2.0,
        width: float = 1.0,
        omega: float = 0.0,
        formulation: str = 'fields',
        gamma: float = 0.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_sites = n_sites
        self.length = length
        self.D = D
        self.momentum = momentum
        self.width = width
        self.omega = omega
        self.formulation = formulation
        self.gamma = gamma

    def get_run_settings(self) -> RunSettings:
        """Return the tuned run settings, with the operator compiler and its own solver for the operator formulation.

        Returns:
            settings:
                The tuned run settings by name.
        """
        settings = super().get_run_settings()
        if self.formulation == 'operators':
            settings['compiler'] = OperatorCompiler()
            settings['solver'] = None
        return settings

    def _build_system(self) -> System:
        """Return a Gaussian wave packet that follows the Schrödinger equation, as its real and imaginary parts."""
        spacing = self.length / self.n_sites
        positions = spacing * np.arange(self.n_sites) - self.length / 2
        envelope = np.exp(-(positions**2) / (4 * self.width**2)) / (2 * np.pi * self.width**2) ** 0.25
        psi = envelope * np.exp(1j * self.momentum * positions)
        potential = self.omega**2 * positions**2 / (4 * self.D)
        space = LatticeSpace(self.n_sites, spacing=spacing)
        if self.formulation not in FORMULATIONS or (self.formulation == 'fields' and self.gamma):
            error_msg = (
                f'Unknown formulation {self.formulation!r}, or dephasing in the field formulation. Use one of '
                f'{", ".join(FORMULATIONS)}, and gamma > 0 only with the operator formulation.'
            )
            raise ValidationError(error_msg)
        if self.formulation == 'operators':
            amplitudes = psi / np.linalg.norm(psi)
            interactions: list[BaseInteraction] = [
                LatticeHamiltonianInteraction('kinetic', target='packet', D=self.D),
            ]
            if self.gamma > 0:
                interactions.append(
                    PositionDephasingInteraction('dephasing', target='packet', gamma=self.gamma, n_sites=self.n_sites),
                )
            return System(
                self.label,
                elements=[
                    SiteElement('packet', amplitudes, hilbert=site_space(space, potential, projectors=self.gamma > 0)),
                ],
                interactions=interactions,
            )
        packet = FieldElement(
            'packet',
            element=WaveFunctionElement,
            space=space,
            values={'u': psi.real, 'w': psi.imag, 'D': self.D, 'V': potential},
            discretization='spectral',
        )
        return System(self.label, elements=[packet])
