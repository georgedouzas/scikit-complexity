"""Charged quantum waves: the Schrödinger equation minimally coupled to a prescribed electromagnetic potential."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any

import numpy as np

from ....core.base import Domain, parameter, state
from ....core.components import Element, Field
from ....core.laws import Law, Rate, Symbols, gradient, laplacian, ramp
from ....core.runtime import register_model
from ....core.space import LatticeSpace
from ....core.system import System


class ChargedWaveFunction(Element):
    """Wave function `psi = u + i w` of a charge `q` in a uniform electromagnetic potential.

    The wave function follows `(d_t - i q A_t) psi = i D (d_x - i q A_x)^2 psi`. The vector potential
    `A_x(t) = A_x - E min(t, duration)` holds a uniform electric field `E` from time 0 to `duration`, and the scalar
    potential `A_t` is constant. Expanded, the equation is
    `psi_t = i D psi_xx + 2 D q A_x psi_x + i (q A_t - D q^2 A_x^2) psi`.

    Args:
        label:
            The label of the site or the field.

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
        >>> from skcomplex.models.physics.waves import ChargedWaveFunction
        >>> ChargedWaveFunction('site').q
        1.0
    """

    u: float = state(default=0.0)
    w: float = state(default=0.0)
    D: float = parameter(default=0.5, domain=Domain.POSITIVE)
    q: float = parameter(default=1.0)
    A_x: float = parameter(default=0.0)
    A_t: float = parameter(default=0.0)
    E: float = parameter(default=0.0)
    duration: float = parameter(default=1.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the minimally coupled Schrödinger equation for the real and imaginary parts.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        potential: Any = s.A_x - s.E * ramp(s.t, 0, s.duration, s.duration)
        drift = 2 * s.D * s.q * potential
        phase = s.q * s.A_t - s.D * s.q**2 * potential**2
        return [
            Rate(s.u, -s.D * laplacian(s.w) + drift * gradient(s.u, 1)[0] - phase * s.w),
            Rate(s.w, s.D * laplacian(s.u) + drift * gradient(s.w, 1)[0] + phase * s.u),
        ]


@register_model(
    field='physics/waves',
    tags=('quantum', 'gauge field', 'electromagnetism', 'wave packet', 'spectral', 'pde'),
    references=('Schwichtenberg 2020',),
    defaults={'until': 8.0, 'dt': 0.1},
)
def charged_schrodinger(
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
) -> System:
    """Return a Gaussian wave packet of a charge in a prescribed electromagnetic potential, after Schwichtenberg 4.3.

    The wave function follows the Schrödinger equation with the covariant derivatives `d - i q A`,
    `(d_t - i q A_t) psi = i D (d_x - i q A_x)^2 psi`, on a periodic line with spectral derivatives. A mode of wave
    number `k` turns at the frequency `D (k - q A_x)^2 - q A_t`, so the packet moves at the group velocity
    `2 D (k - q A_x)`. A constant `A_x` is a pure gauge that only shifts the canonical momentum, while the electric
    field `E` switched on for `duration` changes `A_x` and accelerates the packet at the rate `2 D q E`.

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

    Returns:
        system:
            The model, with the real part `packet.u` and the imaginary part `packet.w` of the wave function.

    Examples:
        >>> from skcomplex.models.physics.waves import charged_schrodinger
        >>> charged_schrodinger(64, E=0.5).elements[0].values['E']
        0.5
    """
    spacing = length / n_sites
    positions = spacing * np.arange(n_sites) - length / 2
    envelope = np.exp(-(positions**2) / (4 * width**2)) / (2 * np.pi * width**2) ** 0.25
    psi = envelope * np.exp(1j * momentum * positions)
    values = {
        'u': psi.real,
        'w': psi.imag,
        'D': D,
        'q': q,
        'A_x': A_x,
        'A_t': A_t,
        'E': E,
        'duration': duration,
    }
    space = LatticeSpace(n_sites, spacing=spacing)
    return System(components=[Field('packet', ChargedWaveFunction, space, values=values, discretization='spectral')])
