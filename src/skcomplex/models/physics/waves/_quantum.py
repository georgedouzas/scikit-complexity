"""Quantum wave models: the Schrödinger equation written as two real fields."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ....core.base import Domain, parameter
from ....core.model import Element, Field, LatticeSpace, Rate, System, laplacian
from ....core.runtime import register_model
from ....core.solvers import ETDRK4

WaveFunction = Element.define(
    'WaveFunction',
    states={'u': 0.0, 'w': 0.0},
    parameters={'D': parameter(default=0.5, domain=Domain.POSITIVE), 'V': 0.0},
    laws=lambda s: [
        Rate(s.u, -s.D * laplacian(s.w) + s.V * s.w),
        Rate(s.w, s.D * laplacian(s.u) - s.V * s.u),
    ],
)


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
) -> System:
    """Return a Gaussian wave packet that follows the Schrödinger equation, as its real and imaginary parts.

    The wave function `psi = u + i w` follows `psi_t = i D laplacian(psi) - i V psi`, written as `u_t = -D laplacian(w)
    + V w` and `w_t = D laplacian(u) - V u`, on a periodic lattice with spectral derivatives. In units with `ħ = 1`,
    `D = 1 / (2 m)`. The potential is the harmonic well `V = omega² x² / (4 D)`, or zero for a free packet. A free
    packet of momentum `k` moves at the group velocity `2 D k` and spreads, and its norm `sum(u² + w²)` is conserved.

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

    Returns:
        system:
            The model, with the real part `packet.u` and the imaginary part `packet.w` of the wave function.

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
    packet = Field(
        'packet',
        WaveFunction,
        space,
        values={'u': psi.real, 'w': psi.imag, 'D': D, 'V': potential},
        discretization='spectral',
    )
    return System(elements=[packet])
