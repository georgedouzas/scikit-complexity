"""Ecology models: population growth, predation, and invasion fronts."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ....core.base import Domain, parameter, state
from ....core.model import Element, Field, LatticeSpace, Rate, System, laplacian
from ....core.runtime import register_model

FisherKPP = Element.define(
    'FisherKPP',
    states={'u': state(default=0.0, domain=Domain.NONNEGATIVE)},
    parameters={'D': parameter(default=1.0, domain=Domain.NONNEGATIVE), 'r': 1.0},
    laws=lambda s: [Rate(s.u, s.D * laplacian(s.u) + s.r * s.u * (1 - s.u))],
)


@register_model(
    field='biology/ecology',
    tags=('field', 'fronts', 'reaction-diffusion', 'pde'),
    references=('Fisher 1937', 'Kolmogorov, Petrovsky, and Piskunov 1937'),
    defaults={'until': 40.0, 'dt': 1.0},
)
def fisher_kpp(
    length: int = 800,
    *,
    D: float = 1.0,
    r: float = 1.0,
    spacing: float = 0.25,
    front: float = 0.05,
) -> System:
    """Return the Fisher-KPP equation, whose step invades the empty line as a front of speed `2 sqrt(r D)`.

    Args:
        length:
            The number of sites of the line.

        D:
            The diffusion coefficient.

        r:
            The growth rate.

        spacing:
            The distance between sites.

        front:
            The fraction of the line that starts occupied, on its left.

    Returns:
        system:
            The model, with the density `line.u`.

    Examples:
        >>> from skcomplex.models.biology.ecology import fisher_kpp
        >>> int(fisher_kpp(100, front=0.1).elements[0].values['u'].sum())
        10
    """
    u = (np.arange(length) < front * length).astype(float)
    space = LatticeSpace(length, spacing=spacing)
    return System(
        elements=[Field('line', FisherKPP, space, boundary={'u': 'zero-flux'}, values={'u': u, 'D': D, 'r': r})],
    )
