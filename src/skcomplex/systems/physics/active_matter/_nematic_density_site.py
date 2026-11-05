"""Define the site of a dry active nematic, a density coupled to a symmetric traceless alignment tensor."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, identity_matrix

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import gradient, symmetric_part, vector_gradient
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class NematicDensitySiteElement(Element):
    """Site of a dry active nematic, whose alignment tensor grows with the traceless curvature of the density.

    The rate of the tensor from the density is `beta (partial_i partial_j rho - delta_ij laplacian(rho) / 2)`. The
    Landau-de Gennes energy and the current of the density are interactions of the system.

    Args:
        label:
            The label of the site or the field.

        visual:
            How the site is drawn, or `None` for `default_visual_`.

        rho:
            The initial density.

        Q:
            The initial alignment tensor, its independent components `Q_xx` and `Q_xy`.

        beta:
            The coupling of the tensor to the curvature of the density.

    Attributes:
        default_visual_:
            The visual of the element when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import NematicDensitySiteElement
        >>> NematicDensitySiteElement('site').beta
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        rho: Annotated[npt.ArrayLike, StateSpec()] = 1.0,
        Q: Annotated[npt.ArrayLike, StateSpec(tensor='traceless')] = 0.0,
        beta: Annotated[float, ParameterSpec()] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.rho = rho
        self.Q = Q
        self.beta = beta

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the rate of the tensor from the curvature of the density.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate of `Q`.
        """
        s = namespace
        curvature = symmetric_part(vector_gradient(gradient(s.rho)))
        traceless = curvature - curvature.trace() / 2 * identity_matrix(SR, 2)
        return [RateLaw(s.Q, s.beta * traceless)]
