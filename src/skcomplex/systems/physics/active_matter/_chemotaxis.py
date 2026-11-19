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

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import RateLaw
from ....core.components.laws.utils import divergence, gradient, laplacian
from ....core.components.spaces import BaseSpace, LatticeSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role


class ChemotaxisElement(Element):
    """Sites with the density `rho` of cells and the concentration `c` of a chemical.

    Args:
        label:
            The label of the sites.

        rho:
            The initial density of cells, one per site or shared.

        c:
            The initial concentration of the chemical, one per site or shared.

        space:
            The lattice of the sites, or `None` for one site.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import ChemotaxisElement
        >>> ChemotaxisElement('site', c=2.0).c
        2.0
    """

    def __init__(self, label: str, *, rho: object = 1.0, c: object = 1.0, space: BaseSpace | None = None) -> None:
        super().__init__(label, space=space)
        self.rho = rho
        self.c = c


class ChemotaxisInteraction(Interaction):
    """Keller-Segel chemotaxis of cells that climb the gradient of a chemical they secrete.

    The cells diffuse and drift up the chemical, `rho_t = D laplacian(rho) - chi div(rho grad c)`, and the chemical
    diffuses, is secreted by the cells, and decays, `c_t = Dc laplacian(c) + rho - k c`. Above a critical mass the cells
    aggregate and the density blows up in finite time. It is a colony of slime mould amoebae or of bacteria that
    gather.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the sites, with the density `rho` and the chemical `c`.

        D:
            The diffusion coefficient of the cells.

        chi:
            The chemotactic sensitivity.

        Dc:
            The diffusion coefficient of the chemical.

        k:
            The decay rate of the chemical.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import ChemotaxisInteraction
        >>> ChemotaxisInteraction('chemotaxis', target='colony').chi
        5.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        D: Annotated[float, RealSet.open(0, oo)] = 1.0,
        chi: Annotated[float, RealSet.closed_open(0, oo)] = 5.0,
        Dc: Annotated[float, RealSet.open(0, oo)] = 1.0,
        k: Annotated[float, RealSet.open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.D = D
        self.chi = chi
        self.Dc = Dc
        self.k = k

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the drift and diffusion of the cells and the balance of the chemical of the target.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s, t = context, context.t
            rho, c = s.target.rho, s.target.c
            self.laws_ = [
                RateLaw(rho.diff(t), s.D * laplacian(rho) - s.chi * divergence(rho * gradient(c))),
                RateLaw(c.diff(t), s.Dc * laplacian(c) + rho - s.k * c),
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import KellerSegelSystem
        >>> KellerSegelSystem((8, 8), chi=0.5).prepare().system_.interactions[0].chi
        0.5
    """

    field = 'physics/active_matter'
    topics = ('field', 'chemotaxis', 'aggregation', 'collapse', 'pde')
    references = ('te Vrugt and Wittkowski 2024', 'Keller and Segel 1970', 'Horstmann 2003')
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
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.shape = shape
        self.length = length
        self.D = D
        self.chi = chi
        self.Dc = Dc
        self.k = k
        self.density = density
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the Keller-Segel system of diffusing cells that climb a chemical gradient.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        shape = tuple(self.shape)
        rng = np.random.default_rng(self.random_state)
        colony = ChemotaxisElement(
            'colony',
            rho=(self.density * (1 + 0.01 * rng.uniform(-1.0, 1.0, shape))).ravel(),
            c=np.full(shape, self.density / self.k).ravel(),
            space=LatticeSpace(shape, spacing=self.length / shape[0]),
        )
        chemotaxis = ChemotaxisInteraction('chemotaxis', target='colony', D=self.D, chi=self.chi, Dc=self.Dc, k=self.k)
        self.system_ = System(self.label, elements=[colony], interactions=[chemotaxis])
