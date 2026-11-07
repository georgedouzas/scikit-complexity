"""Charged quantum waves: the Schrödinger equation minimally coupled to a prescribed electromagnetic potential."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, FieldElement
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import gradient, laplacian, ramp
from ....core.components.spaces import LatticeSpace
from ....core.components.system import BaseSystem, System
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class ChargedWaveFunctionElement(Element):
    """Wave function `psi = u + i w` of a charge `q` in a uniform electromagnetic potential.

    The wave function follows `(d_t - i q A_t) psi = i D (d_x - i q A_x)^2 psi`. The vector potential
    `A_x(t) = A_x - E min(t, duration)` holds a uniform electric field `E` from time 0 to `duration`, and the scalar
    potential `A_t` is constant. Expanded, the equation is
    `psi_t = i D psi_xx + 2 D q A_x psi_x + i (q A_t - D q^2 A_x^2) psi`.

    Args:
        label:
            The label of the site or the field.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        u:
            The initial real part.

        w:
            The initial imaginary part.

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
        >>> from skcomplex.systems.physics.waves import ChargedWaveFunctionElement
        >>> ChargedWaveFunctionElement('site').q
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
        q: Annotated[float, ParameterSpec()] = 1.0,
        A_x: Annotated[float, ParameterSpec()] = 0.0,
        A_t: Annotated[float, ParameterSpec()] = 0.0,
        E: Annotated[float, ParameterSpec()] = 0.0,
        duration: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u
        self.w = w
        self.D = D
        self.q = q
        self.A_x = A_x
        self.A_t = A_t
        self.E = E
        self.duration = duration

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the minimally coupled Schrödinger equation for the real and imaginary parts.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        potential = s.A_x - s.E * ramp(s.t, 0, s.duration, s.duration)
        drift = 2 * s.D * s.q * potential
        phase = s.q * s.A_t - s.D * s.q**2 * potential**2
        return [
            RateLaw(s.u, -s.D * laplacian(s.w) + drift * gradient(s.u, 1)[0] - phase * s.w),
            RateLaw(s.w, s.D * laplacian(s.u) + drift * gradient(s.w, 1)[0] + phase * s.u),
        ]


class ChargedSchrodingerSystem(BaseSystem):
    """A Gaussian wave packet of a charge in a prescribed electromagnetic potential, after Schwichtenberg 4.3.

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

    Examples:
        >>> from skcomplex.systems.physics.waves import ChargedSchrodingerSystem
        >>> ChargedSchrodingerSystem(64, E=0.5).prepare().system_.elements[0].values['E']
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
    ) -> None:
        super().__init__(label)
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

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        spacing = self.length / self.n_sites
        positions = spacing * np.arange(self.n_sites) - self.length / 2
        envelope = np.exp(-(positions**2) / (4 * self.width**2)) / (2 * np.pi * self.width**2) ** 0.25
        psi = envelope * np.exp(1j * self.momentum * positions)
        values = {
            'u': psi.real,
            'w': psi.imag,
            'D': self.D,
            'q': self.q,
            'A_x': self.A_x,
            'A_t': self.A_t,
            'E': self.E,
            'duration': self.duration,
        }
        space = LatticeSpace(self.n_sites, spacing=spacing)
        return System(
            self.label,
            elements=[
                FieldElement(
                    'packet',
                    element=ChargedWaveFunctionElement,
                    space=space,
                    values=values,
                    discretization='spectral',
                ),
            ],
        )
