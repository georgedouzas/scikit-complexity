"""Site of a flame front or a thin film, `du/dt = -d(u^2 / 2)/dx - laplacian(u) - laplacian(laplacian(u))`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.elements import Element
from ....specs import StateSpec, VisualSpec


class KuramotoSivashinskyElement(Element):
    """Site of a flame front or a thin film, `du/dt = -d(u^2 / 2)/dx - laplacian(u) - laplacian(laplacian(u))`.

    The conservation of the flux `u^2 / 2` and the rate of the linear operator are interactions of the model.

    Args:
        label:
            The label of the site.

        visual:
            How the site is drawn, or `None` for none.

        u:
            The initial height of the front.

    Examples:
        >>> from skcomplex.models.physics.fluids import KuramotoSivashinskyElement
        >>> KuramotoSivashinskyElement('site', u=0.5).u
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u
