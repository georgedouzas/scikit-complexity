"""Neural fields: the potential of a continuum of neurons that excite near and inhibit distant neighbours."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import Expression, exp

from ....core.components.elements import FieldElement
from ....core.components.interactions import RateInteraction
from ....core.components.laws.utils import convolve, displacement, where
from ....core.components.spaces import LatticeSpace
from ....core.components.system import System
from ..._catalog import register_system
from ._neural_field_site import NeuralFieldSiteElement


def _amari_rate(s: SimpleNamespace) -> Expression:
    """Return the rate of the Amari field, the leak, the input of the firing sites through the kernel, and the rest."""
    site = s.target
    distance = abs(displacement(0)) / site.sigma
    kernel = site.A * (1 - distance) * exp(-distance)
    rate: Expression = (-site.u + convolve(where(site.u > 0, 1, 0), kernel) + site.h) / site.tau
    return rate


@register_system(
    field='biology/neuroscience',
    tags=('neural field', 'field', 'convolution', 'bump', 'working memory', 'pde'),
    references=('Amari 1977', 'arXiv:2112.12147'),
    defaults={'until': 40.0, 'dt': 0.5},
)
def amari_neural_field(
    n_sites: int = 800,
    *,
    length: float = 20.0,
    tau: float = 1.0,
    h: float = -0.2,
    A: float = 1.0,
    sigma: float = 1.0,
    width: float = 1.0,
) -> System:
    """Return the Amari neural field, whose lateral inhibition holds a stationary bump of activity.

    The potential follows `tau u_t = -u + integral w(x - y) H(u(y)) dy + h` on a periodic line, where `H` is the
    Heaviside firing rate, `h < 0` the resting level, and `w(x) = A (1 - |x| / sigma) exp(-|x| / sigma)` the
    lateral-inhibition kernel. Amari showed that a bump of width `a` is stationary when `W(a) + h = 0`, with
    `W(a) = A a exp(-a / sigma)` the integral of `w` from 0 to `a`, and stable when `w(a) < 0`. A localized excitation
    then relaxes to the wide root, `a = 2.54` for the defaults.

    Args:
        n_sites:
            The number of sites of the line.

        length:
            The length of the line.

        tau:
            The time constant of the potential.

        h:
            The resting level.

        A:
            The amplitude of the kernel.

        sigma:
            The range of the kernel.

        width:
            The width of the initial excitation, a potential of 1 around the middle of the line and `h` elsewhere.

    Returns:
        system:
            The system, with the potential `field.u`.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import amari_neural_field
        >>> amari_neural_field(100).elements[0].values['h']
        -0.2
    """
    spacing = length / n_sites
    positions = (np.arange(n_sites) + 0.5) * spacing - length / 2
    values = {'u': np.where(np.abs(positions) < width / 2, 1.0, h), 'tau': tau, 'h': h, 'A': A, 'sigma': sigma}
    field = FieldElement(
        'field',
        element=NeuralFieldSiteElement,
        space=LatticeSpace(n_sites, spacing=spacing),
        values=values,
    )
    rate = RateInteraction('rate', target='field', state='u', rate=_amari_rate)
    return System(elements=[field], interactions=[rate])
