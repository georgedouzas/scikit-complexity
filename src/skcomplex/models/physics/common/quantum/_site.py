"""Single particle on the sites of a lattice or a network, whose basis state `k` is the particle at site `k`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import CC

from .....core.elements import Element
from .....specs import HilbertSpec, StateSpec, VisualSpec


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
        >>> from skcomplex.core.spaces import LatticeSpace
        >>> from skcomplex.models.physics.common.quantum import SiteElement, site_space
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
