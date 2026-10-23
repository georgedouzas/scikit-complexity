"""Chemotaxis: cells that climb the gradient of a chemical they secrete, the Keller-Segel model."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np

from ....core.base import Domain, parameter
from ....core.components import Element, Field
from ....core.laws import Rate, divergence, gradient, laplacian
from ....core.runtime import register_model
from ....core.space import LatticeSpace
from ....core.system import System

Chemotaxis = Element.define(
    'Chemotaxis',
    states={'rho': 1.0, 'c': 1.0},
    parameters={
        'D': parameter(default=1.0, domain=Domain.POSITIVE),
        'chi': parameter(default=5.0, domain=Domain.NONNEGATIVE),
        'Dc': parameter(default=1.0, domain=Domain.POSITIVE),
        'k': parameter(default=1.0, domain=Domain.POSITIVE),
    },
    laws=lambda s: [
        Rate(s.rho, s.D * laplacian(s.rho) - s.chi * divergence(s.rho * gradient(s.c))),
        Rate(s.c, s.Dc * laplacian(s.c) + s.rho - s.k * s.c),
    ],
)


@register_model(
    field='physics/active_matter',
    tags=('field', 'chemotaxis', 'aggregation', 'collapse', 'pde'),
    references=('te Vrugt and Wittkowski 2024', 'Keller and Segel 1970', 'Horstmann 2003'),
    defaults={'until': 10.5, 'dt': 0.25},
)
def keller_segel(
    shape: Sequence[int] = (32, 32),
    *,
    length: float = 32.0,
    D: float = 1.0,
    chi: float = 5.0,
    Dc: float = 1.0,
    k: float = 1.0,
    density: float = 1.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Keller-Segel model of cells that diffuse and climb the gradient of a chemical they secrete.

    The cell density follows `rho_t = D laplacian(rho) - chi div(rho grad(c))` and the chemical
    `c_t = Dc laplacian(c) + rho - k c`, on a periodic square. The total number of cells is conserved. The uniform state
    of density `rho_0` is unstable when the chemotactic drive `chi rho_0` exceeds `D (k + Dc q^2)` for the longest
    wave `q = 2 pi / length`. The cells then aggregate into dense spots, which in two dimensions collapse in finite
    time, so a run above the threshold ends before the density blows up.

    Args:
        shape:
            The number of sites along both axes of the periodic lattice.

        length:
            The side of the square.

        D:
            The diffusivity of the cells.

        chi:
            The chemotactic sensitivity.

        Dc:
            The diffusivity of the chemical.

        k:
            The decay rate of the chemical.

        density:
            The mean cell density `rho_0`, perturbed by 1% noise, with the chemical at its uniform value `rho_0 / k`.

        random_state:
            The seed or generator of the noise.

    Returns:
        system:
            The model, with the cell density `colony.rho` and the chemical `colony.c`.

    Examples:
        >>> from skcomplex.models.physics.active_matter import keller_segel
        >>> keller_segel((8, 8), chi=0.5).elements[0].values['chi']
        0.5
    """
    shape = tuple(shape)
    rng = np.random.default_rng(random_state)
    values = {
        'rho': density * (1 + 0.01 * rng.uniform(-1.0, 1.0, shape)),
        'c': np.full(shape, density / k),
        'D': D,
        'chi': chi,
        'Dc': Dc,
        'k': k,
    }
    space = LatticeSpace(shape, spacing=length / shape[0])
    return System(components=[Field('colony', Chemotaxis, space, values=values)])
