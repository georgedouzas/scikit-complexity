"""Chemotaxis, the Keller-Segel system.

Cells climb the gradient of a chemical they secrete.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, FieldElement
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import divergence, gradient, laplacian
from ....core.components.spaces import LatticeSpace
from ....core.components.systems import BaseSystem, System
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class ChemotaxisElement(Element):
    """Site of the Keller-Segel system.

    The site holds cells of density `rho` that climb the gradient of a chemical `c` they secrete.

    The cells diffuse and drift up the chemical, `rho_t = D laplacian(rho) - chi div(rho grad c)`, and the chemical
    diffuses, is secreted by the cells, and decays, `c_t = Dc laplacian(c) + rho - k c`. Above a critical mass the cells
    aggregate and the density blows up in finite time.

    Args:
        label:
            The label of the site or the field.

        visual:
            How the site is drawn, or `None` for `default_visual_`.

        rho:
            The initial density of cells.

        c:
            The initial concentration of the chemical.

        D:
            The diffusion coefficient of the cells.

        chi:
            The chemotactic sensitivity.

        Dc:
            The diffusion coefficient of the chemical.

        k:
            The decay rate of the chemical.

    Attributes:
        default_visual_:
            The visual of the element when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import ChemotaxisElement
        >>> ChemotaxisElement('site').chi
        5.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        rho: Annotated[float, StateSpec()] = 1.0,
        c: Annotated[float, StateSpec()] = 1.0,
        D: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        chi: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
        Dc: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        k: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.rho = rho
        self.c = c
        self.D = D
        self.chi = chi
        self.Dc = Dc
        self.k = k

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the drift and diffusion of the cells and the balance of the chemical.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rates of `rho` and `c`.
        """
        s = namespace
        return [
            RateLaw(s.rho, s.D * laplacian(s.rho) - s.chi * divergence(s.rho * gradient(s.c))),
            RateLaw(s.c, s.Dc * laplacian(s.c) + s.rho - s.k * s.c),
        ]


class KellerSegelSystem(BaseSystem):
    """The Keller-Segel system of diffusing cells that climb a chemical gradient.

    The cells secrete the chemical.

    The cell density follows `rho_t = D laplacian(rho) - chi div(rho grad(c))` and the chemical `c_t = Dc laplacian(c) +
    rho - k c`, on a periodic square. The total number of cells is conserved. The uniform state of density `rho_0` is
    unstable when the chemotactic drive `chi rho_0` exceeds `D (k + Dc q^2)` for the longest wave `q = 2 pi / length`.
    The cells then aggregate into dense spots, which in two dimensions collapse in finite time, so a run above the
    threshold ends before the density blows up. The system has the cell density `colony.rho` and the chemical
    `colony.c`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import KellerSegelSystem
        >>> KellerSegelSystem((8, 8), chi=0.5).prepare().system_.elements[0].values['chi']
        0.5
    """

    field = "physics/active_matter"
    topics = ("field", "chemotaxis", "aggregation", "collapse", "pde")
    references = ("te Vrugt and Wittkowski 2024", "Keller and Segel 1970", "Horstmann 2003")
    until = 10.5
    dt = 0.25

    def __init__(
        self,
        shape: Sequence[int] = (32, 32),
        *,
        length: float = 32.0,
        D: float = 1.0,
        chi: float = 5.0,
        Dc: float = 1.0,
        k: float = 1.0,
        density: float = 1.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = "system",
    ) -> None:
        super().__init__(label)
        self.shape = shape
        self.length = length
        self.D = D
        self.chi = chi
        self.Dc = Dc
        self.k = k
        self.density = density
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return the Keller-Segel system of diffusing cells climbing a chemical gradient.

        The cells secrete the chemical.
        """
        shape = tuple(self.shape)
        rng = np.random.default_rng(self.random_state)
        values = {
            "rho": self.density * (1 + 0.01 * rng.uniform(-1.0, 1.0, shape)),
            "c": np.full(shape, self.density / self.k),
            "D": self.D,
            "chi": self.chi,
            "Dc": self.Dc,
            "k": self.k,
        }
        space = LatticeSpace(shape, spacing=self.length / shape[0])
        return System(
            self.label,
            elements=[FieldElement("colony", element=ChemotaxisElement, space=space, values=values)],
        )
