"""Chemotaxis: cells that climb the gradient of a chemical they secrete, the Keller-Segel model."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, FieldElement
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import divergence, gradient, laplacian
from ....core.spaces import LatticeSpace
from ....core.system import System
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_model


class ChemotaxisElement(Element):
    """Chemotaxis element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        rho: Annotated[object, StateSpec()] = 1.0,
        c: Annotated[object, StateSpec()] = 1.0,
        D: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        chi: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
        Dc: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        k: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.rho = rho
        self.c = c
        self.D = D
        self.chi = chi
        self.Dc = Dc
        self.k = k

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the chemotaxis element."""
        return [
            RateLaw(s.rho, s.D * laplacian(s.rho) - s.chi * divergence(s.rho * gradient(s.c))),
            RateLaw(s.c, s.Dc * laplacian(s.c) + s.rho - s.k * s.c),
        ]


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
    return System(elements=[FieldElement('colony', element=ChemotaxisElement, space=space, values=values)])
