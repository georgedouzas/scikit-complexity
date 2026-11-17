"""Quantum wave systems.

The Schrödinger equation runs as two real fields or as a particle on the sites of a lattice.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from sage.all__sagemath_symbolics import CC, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import BaseInteraction, Interaction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import laplacian, local_operator
from ....core.components.spaces import BaseSpace, LatticeSpace, NetworkSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, ETDRK4Solver
from ....core.errors import ValidationError
from ....core.specs import ParameterSpec, Role, RoleSpec, StateSpec, VisualSpec
from ....core.systems import BaseSystem, System
from ..common.quantum import DissipatorLaw, GeneratorLaw, HilbertSpec

FORMULATIONS = ('fields', 'operators')


class SiteElement(Element):
    """Single particle on the sites of a lattice or a network.

    Its basis state `k` is the particle at site `k`.

    Its laws read the operators `laplacian`, `adjacency`, and `potential` of its Hilbert space.

    Args:
        label:
            The label of the particle.

        psi:
            The state in the Hilbert space: the initial site, or a vector of amplitudes over the sites.

        hilbert:
            The Hilbert space of the sites, whose basis state `k` is site `k`.

    Examples:
        >>> from skcomplex.systems.physics.waves._quantum import SiteElement
        >>> SiteElement('particle', 3).psi
        3
    """

    visual = VisualSpec('expectations')

    def __init__(
        self,
        label: str,
        psi: Annotated[object, StateSpec(domain=CC)] = 0,
        *,
        hilbert: HilbertSpec | None = None,
    ) -> None:
        super().__init__(label)
        self.psi = psi
        self.hilbert = hilbert


class WaveFunctionElement(Element):
    """Sites of the wave function `psi = u + i w` of a particle with the potential `V`.

    Every site holds the real and imaginary parts of the wave function and the potential there.

    Args:
        label:
            The label of the wave function.

        u:
            The initial real part, one value per site or one for all.

        w:
            The initial imaginary part, one value per site or one for all.

        V:
            The potential, one value per site or one for all.

        space:
            The lattice of the sites, or `None` for one site.

    Examples:
        >>> from skcomplex.systems.physics.waves import WaveFunctionElement
        >>> WaveFunctionElement('site', V=1.0).V
        1.0
    """

    def __init__(
        self,
        label: str,
        *,
        u: object = 0.0,
        w: object = 0.0,
        V: object = 0.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.u = u
        self.w = w
        self.V = V


class SchrodingerInteraction(Interaction):
    """The Schrödinger equation `i psi' = -D laplacian(psi) + V psi` of a wave function `psi = u + i w`.

    The real and imaginary parts of the target follow `u' = -D laplacian(w) + V w` and `w' = D laplacian(u) - V u`,
    where `V` is the potential of the target.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the wave function, with the real part `u`, the imaginary part `w`, and the potential `V`.

        D:
            The coefficient of the kinetic energy, `hbar / (2 m)`.

    Examples:
        >>> from skcomplex.systems.physics.waves import SchrodingerInteraction
        >>> SchrodingerInteraction('evolution', target='packet').D
        0.5
    """

    law_classes = (RateLaw,)

    def __init__(self, label: str, *, target: Role, D: Annotated[float, RealSet.open(0, oo)] = 0.5) -> None:
        super().__init__(label)
        self.target = target
        self.D = D

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rates of the real and imaginary parts of the wave function of the target.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s, t = context, context.t
            u, w, V = s.target.u, s.target.w, s.target.V
            self.laws_ = [
                RateLaw(u.diff(t), -s.D * laplacian(w) + V * w),
                RateLaw(w.diff(t), s.D * laplacian(u) - V * u),
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
    """Dephasing of a particle in the position basis.

    There is one dissipator per site projector at the rate `gamma`.

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


class SchrodingerSystem(BaseSystem):
    """A Gaussian wave packet that follows the Schrödinger equation.

    The packet is written as its real and imaginary parts.

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

        solver:
            The solver of the problem, or `'auto'` for exponential steps of 0.01 of the fields and the quantum solver
            of the operators.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ValidationError:
            The formulation is unknown, or the field formulation has a dephasing rate.

    Examples:
        >>> from skcomplex.systems.physics.waves import SchrodingerSystem
        >>> SchrodingerSystem(64).prepare().system_.elements[0].space.discretization
        'spectral'
    """

    field = 'physics/waves'
    topics = ('waves', 'field', 'pde', 'quantum', 'spectral')
    references = ('Schrödinger 1926', 'Schwichtenberg 2020')
    until = 4.0
    dt = 0.1

    @staticmethod
    def _get_lattice_laplacian(lattice: LatticeSpace) -> npt.NDArray[np.float64]:
        """Return the Laplacian of a prepared lattice.

        The Laplacian is spectral when the lattice is periodic and of finite differences otherwise.
        """
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

    @staticmethod
    def _get_site_space(
        space: BaseSpace,
        potential: npt.ArrayLike | None = None,
        *,
        projectors: bool = False,
    ) -> HilbertSpec:
        """Return the Hilbert space of a particle on the sites of a lattice or a network.

        Its basis state `k` is site `k`, and its dimension is the number of sites. Its operators are the Laplacian,
        `laplacian`, the adjacency of the neighbouring sites, `adjacency`, with a potential of every site the operator
        `potential` with the potential on its diagonal, and with projectors the projector `site_<k>` on every site `k`.
        The Laplacian of a lattice is spectral when the lattice is periodic and of finite differences otherwise, with
        the spacing of the lattice, and the Laplacian of a network is its graph Laplacian. Another space raises a
        `ValidationError`.
        """
        if not isinstance(space, LatticeSpace | NetworkSpace):
            error_msg = f'A particle on sites needs a LatticeSpace or a NetworkSpace, not {space!r}.'
            raise ValidationError(error_msg)
        prepared = space.prepare()
        size = int(prepared.size_)
        adjacency = np.zeros((size, size))
        for first, second in prepared._get_neighbor_pairs():
            adjacency[first, second] = adjacency[second, first] = 1.0
        laplacian = (
            SchrodingerSystem._get_lattice_laplacian(prepared)
            if isinstance(prepared, LatticeSpace)
            else adjacency - np.diag(adjacency.sum(axis=1))
        )
        operators = {'laplacian': (laplacian, True), 'adjacency': (adjacency, True)}
        if potential is not None:
            operators['potential'] = (np.diag(np.broadcast_to(np.asarray(potential, dtype=float), (size,))), True)
        for site in range(size if projectors else 0):
            operators[f'site_{site}'] = (sp.csr_array(([1.0], ([site], [site])), shape=(size, size)), True)
        return HilbertSpec(size, operators, display=None)

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_sites = n_sites
        self.length = length
        self.D = D
        self.momentum = momentum
        self.width = width
        self.omega = omega
        self.formulation = formulation
        self.gamma = gamma

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a Gaussian wave packet that follows the Schrödinger equation.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.

        Raises:
            ValidationError:
                The formulation is unknown, or the field formulation has a dephasing rate.
        """
        super()._prepare(context)
        spacing = self.length / self.n_sites
        positions = spacing * np.arange(self.n_sites) - self.length / 2
        envelope = np.exp(-(positions**2) / (4 * self.width**2)) / (2 * np.pi * self.width**2) ** 0.25
        psi = envelope * np.exp(1j * self.momentum * positions)
        potential = self.omega**2 * positions**2 / (4 * self.D)
        space = LatticeSpace(self.n_sites, spacing=spacing, discretization='spectral')
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
            self.system_ = System(
                self.label,
                elements=[
                    SiteElement(
                        'packet',
                        amplitudes,
                        hilbert=self._get_site_space(space, potential, projectors=self.gamma > 0),
                    ),
                ],
                interactions=interactions,
            )
            return
        packet = WaveFunctionElement('packet', u=psi.real, w=psi.imag, V=potential, space=space)
        evolution = SchrodingerInteraction('evolution', target='packet', D=self.D)
        self.system_ = System(self.label, elements=[packet], interactions=[evolution], solver=ETDRK4Solver(dt=0.01))
