"""Charged quantum waves.

The Schrödinger equation is minimally coupled to a prescribed electromagnetic potential.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import RateLaw
from ....core.components.laws.utils import gradient, laplacian, ramp
from ....core.components.spaces import BaseSpace, LatticeSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role


class ChargedWaveFunctionElement(Element):
    """Sites of the wave function `psi = u + i w` of a charge, held as its real part `u` and its imaginary part `w`.

    Args:
        label:
            The label of the wave function.

        u:
            The initial real part, one value per site or one for all.

        w:
            The initial imaginary part, one value per site or one for all.

        space:
            The lattice of the sites, or `None` for one site.

    Examples:
        >>> from skcomplex.systems.physics.waves import ChargedWaveFunctionElement
        >>> ChargedWaveFunctionElement('site', u=1.0).u
        1.0
    """

    def __init__(self, label: str, *, u: object = 0.0, w: object = 0.0, space: BaseSpace | None = None) -> None:
        super().__init__(label, space=space)
        self.u = u
        self.w = w


class MinimalCouplingInteraction(Interaction):
    """The Schrödinger equation of a charge `q` minimally coupled to a uniform electromagnetic potential.

    The wave function `psi = u + i w` of the target follows `(d_t - i q A_t) psi = i D (d_x - i q A_x)^2 psi`. The
    vector potential `A_x(t) = A_x - E min(t, duration)` holds a uniform electric field `E` from time 0 to `duration`,
    and the scalar potential `A_t` is constant. Expanded, the equation is
    `psi_t = i D psi_xx + 2 D q A_x psi_x + i (q A_t - D q^2 A_x^2) psi`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the wave function, with the real part `u` and the imaginary part `w`.

        D:
            The coefficient of the Laplacian, `ħ / (2 m)`.

        q:
            The electric charge.

        A_x:
            The vector potential at time 0.

        A_t:
            The scalar potential.

        E:
            The electric field.

        duration:
            The time the electric field is on.

    Examples:
        >>> from skcomplex.systems.physics.waves import MinimalCouplingInteraction
        >>> MinimalCouplingInteraction('coupling', target='packet').q
        1.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        D: Annotated[float, RealSet.open(0, oo)] = 0.5,
        q: float = 1.0,
        A_x: float = 0.0,
        A_t: float = 0.0,
        E: float = 0.0,
        duration: Annotated[float, RealSet.open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.D = D
        self.q = q
        self.A_x = A_x
        self.A_t = A_t
        self.E = E
        self.duration = duration

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the minimally coupled Schrödinger equation of the real and imaginary parts of the target.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        s, t = context, context.t
        u, w = s.target.u, s.target.w
        potential = s.A_x - s.E * ramp(t, 0, s.duration, s.duration)
        drift = 2 * s.D * s.q * potential
        phase = s.q * s.A_t - s.D * s.q**2 * potential**2
        self.laws_ = [
            RateLaw(u.diff(t), -s.D * laplacian(w) + drift * gradient(u, 1)[0] - phase * w),
            RateLaw(w.diff(t), s.D * laplacian(u) + drift * gradient(w, 1)[0] + phase * u),
        ]


class ChargedSchrodingerSystem(BaseSystem):
    """A Gaussian wave packet of a charge in a prescribed electromagnetic potential.

    The system follows Schwichtenberg 4.3.

    The wave function follows the Schrödinger equation with the covariant derivatives `d - i q A`, `(d_t - i q A_t) psi
    = i D (d_x - i q A_x)^2 psi`, on a periodic line with spectral derivatives. A mode of wave number `k` turns at the
    frequency `D (k - q A_x)^2 - q A_t`, so the packet moves at the group velocity `2 D (k - q A_x)`. A constant `A_x`
    is a pure gauge that only shifts the canonical momentum, while the electric field `E` switched on for `duration`
    changes `A_x` and accelerates the packet at the rate `2 D q E`. The system has the real part `packet.u` and the
    imaginary part `packet.w` of the wave function.

    Args:
        n_sites:
            The number of sites of the periodic lattice.

        length:
            The length of the periodic line, centered on 0.

        D:
            The coefficient of the Laplacian, `ħ / (2 m)`.

        q:
            The electric charge.

        momentum:
            The mean wave number `k` of the packet.

        width:
            The width of the packet.

        A_x:
            The vector potential at time 0.

        A_t:
            The scalar potential.

        E:
            The electric field.

        duration:
            The time the electric field is on.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.waves import ChargedSchrodingerSystem
        >>> ChargedSchrodingerSystem(64, E=0.5).prepare().system_.interactions[0].E
        0.5
    """

    field = 'physics/waves'
    topics = ('quantum', 'gauge field', 'electromagnetism', 'wave packet', 'spectral', 'pde')
    references = ('Schwichtenberg 2020',)
    until = 8.0
    dt = 0.1

    def __init__(
        self,
        n_sites: int = 256,
        *,
        length: float = 80.0,
        D: float = 0.5,
        q: float = 1.0,
        momentum: float = 0.0,
        width: float = 2.0,
        A_x: float = 0.0,
        A_t: float = 0.0,
        E: float = 1.0,
        duration: float = 2.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_sites = n_sites
        self.length = length
        self.D = D
        self.q = q
        self.momentum = momentum
        self.width = width
        self.A_x = A_x
        self.A_t = A_t
        self.E = E
        self.duration = duration

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a Gaussian wave packet of a charge in the electromagnetic potential.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        spacing = self.length / self.n_sites
        positions = spacing * np.arange(self.n_sites) - self.length / 2
        envelope = np.exp(-(positions**2) / (4 * self.width**2)) / (2 * np.pi * self.width**2) ** 0.25
        psi = envelope * np.exp(1j * self.momentum * positions)
        space = LatticeSpace(self.n_sites, spacing=spacing, discretization='spectral')
        packet = ChargedWaveFunctionElement('packet', u=psi.real, w=psi.imag, space=space)
        coupling = MinimalCouplingInteraction(
            'coupling',
            target='packet',
            D=self.D,
            q=self.q,
            A_x=self.A_x,
            A_t=self.A_t,
            E=self.E,
            duration=self.duration,
        )
        self.system_ = System(self.label, elements=[packet], interactions=[coupling])
