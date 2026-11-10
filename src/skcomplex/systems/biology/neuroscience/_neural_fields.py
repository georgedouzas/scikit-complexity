"""Neural fields: the potential of a continuum of neurons.

The neurons excite near and inhibit distant neighbours.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, exp, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, FieldElement
from ....core.components.interactions import RateInteraction
from ....core.components.laws.utils import convolve, displacement, where
from ....core.components.spaces import LatticeSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class NeuralFieldSiteElement(Element):
    """Site of a neural field.

    The site holds the mean membrane potential of the neurons at one position.

    The site holds the potential `u`, its time constant `tau`, the resting level `h`, and the amplitude `A` and the
    range `sigma` of the lateral-inhibition kernel `w(x) = A (1 - |x| / sigma) exp(-|x| / sigma)`, which excites near
    neighbours and inhibits distant ones.

    Args:
        label:
            The label of the site or the field.

        visual:
            How the site is drawn, or `None` for `default_visual_`.

        u:
            The initial membrane potential.

        tau:
            The time constant of the potential.

        h:
            The resting level.

        A:
            The amplitude of the kernel.

        sigma:
            The range of the kernel.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import NeuralFieldSiteElement
        >>> NeuralFieldSiteElement('site').h
        -0.2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[float, StateSpec()] = 0.0,
        tau: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        h: Annotated[float, ParameterSpec()] = -0.2,
        A: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        sigma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u
        self.tau = tau
        self.h = h
        self.A = A
        self.sigma = sigma


class AmariNeuralFieldSystem(BaseSystem):
    """The Amari neural field.

    Its lateral inhibition holds a stationary bump of activity.

    The potential follows `tau u_t = -u + integral w(x - y) H(u(y)) dy + h` on a periodic line, where `H` is the
    Heaviside firing rate, `h < 0` the resting level, and `w(x) = A (1 - |x| / sigma) exp(-|x| / sigma)` the
    lateral-inhibition kernel. Amari showed that a bump of width `a` is stationary when `W(a) + h = 0`, with `W(a) = A a
    exp(-a / sigma)` the integral of `w` from 0 to `a`, and stable when `w(a) < 0`. A localized excitation then relaxes
    to the wide root, `a = 2.54` for the defaults. The system has the potential `field.u`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import AmariNeuralFieldSystem
        >>> AmariNeuralFieldSystem(100).prepare().system_.elements[0].values['h']
        -0.2
    """

    field = 'biology/neuroscience'
    topics = ('neural field', 'field', 'convolution', 'bump', 'working memory', 'pde')
    references = ('Amari 1977', 'arXiv:2112.12147')
    until = 40.0
    dt = 0.5

    def __init__(
        self,
        n_sites: int = 800,
        *,
        length: float = 20.0,
        tau: float = 1.0,
        h: float = -0.2,
        A: float = 1.0,
        sigma: float = 1.0,
        width: float = 1.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_sites = n_sites
        self.length = length
        self.tau = tau
        self.h = h
        self.A = A
        self.sigma = sigma
        self.width = width

    @staticmethod
    def _get_amari_rate(s: SimpleNamespace) -> Expression:
        """Return the rate of the Amari field.

        The rate is the leak, the input of firing sites through the kernel, and the rest.
        """
        site = s.target
        distance = abs(displacement(0)) / site.sigma
        kernel = site.A * (1 - distance) * exp(-distance)
        rate: Expression = (-site.u + convolve(where(site.u > 0, 1, 0), kernel) + site.h) / site.tau
        return rate

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the Amari neural field.

        Its lateral inhibition holds a stationary bump of activity.
        """
        spacing = self.length / self.n_sites
        positions = (np.arange(self.n_sites) + 0.5) * spacing - self.length / 2
        values = {
            'u': np.where(np.abs(positions) < self.width / 2, 1.0, self.h),
            'tau': self.tau,
            'h': self.h,
            'A': self.A,
            'sigma': self.sigma,
        }
        field = FieldElement(
            'field',
            element=NeuralFieldSiteElement,
            space=LatticeSpace(self.n_sites, spacing=spacing),
            values=values,
        )
        rate = RateInteraction('rate', target='field', state='u', rate=self._get_amari_rate)
        return CompositeSystem(self.label, elements=[field], interactions=[rate])
