"""Pattern formation of conserved fields.

The module holds the nonreciprocal Cahn-Hilliard equations.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import RateLaw
from ....core.components.laws.utils import laplacian
from ....core.components.spaces import BaseSpace, LatticeSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role


class NonreciprocalMixtureElement(Element):
    """Site of a mixture with a demixing field `phi` and a diffusing field `psi`.

    Args:
        label:
            The label of the site or the field.

        phi:
            The initial demixing field.

        psi:
            The initial diffusing field.

        space:
            The lattice whose sites hold one site each, or `None` for one site.

    Examples:
        >>> from skcomplex.systems.physics.pattern_formation import NonreciprocalMixtureElement
        >>> NonreciprocalMixtureElement('site', phi=0.1).phi
        0.1
    """

    def __init__(
        self,
        label: str,
        *,
        phi: object = 0.0,
        psi: object = 0.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.phi = phi
        self.psi = psi


class NonreciprocalCahnHilliardInteraction(Interaction):
    """Cross-diffusion of two conserved fields that couples them nonreciprocally.

    The fields follow `phi_t = laplacian(-phi + phi^3 - kappa laplacian(phi) + (chi + alpha) psi)` and
    `psi_t = laplacian(D psi + (chi - alpha) phi)`. The reciprocal coupling `chi` derives from a free energy, and the
    nonreciprocal coupling `alpha` does not, so above a threshold the demixed domains travel. It is a mixture of active
    colloids, two chemical species that chase each other, or predator and prey that diffuse.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the mixture.

        chi:
            The reciprocal coupling.

        alpha:
            The nonreciprocal coupling.

        D:
            The diffusion coefficient of `psi`.

        kappa:
            The stiffness of the interfaces of `phi`.

    Examples:
        >>> from skcomplex.systems.physics.pattern_formation import NonreciprocalCahnHilliardInteraction
        >>> NonreciprocalCahnHilliardInteraction('coupling', target='mixture').alpha
        0.3
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        chi: float = 0.0,
        alpha: float = 0.3,
        D: float = 0.1,
        kappa: float = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.chi = chi
        self.alpha = alpha
        self.D = D
        self.kappa = kappa

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the conserved rates of both fields of the target.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        s, t = context, context.t
        phi, psi = s.target.phi, s.target.psi
        potential = -phi + phi**3 - s.kappa * laplacian(phi) + (s.chi + s.alpha) * psi
        self.laws_ = [
            RateLaw(phi.diff(t), laplacian(potential)),
            RateLaw(psi.diff(t), laplacian(s.D * psi + (s.chi - s.alpha) * phi)),
        ]


class NonreciprocalCahnHilliardSystem(BaseSystem):
    """The nonreciprocal Cahn-Hilliard equations.

    A demixing field is coupled to a diffusing one.

    The conserved field `phi` phase separates and the conserved field `psi` diffuses, and they are coupled through
    cross-diffusion, `phi_t = laplacian(-phi + phi^3 - kappa laplacian(phi) + (chi + alpha) psi)` and `psi_t =
    laplacian(D psi + (chi - alpha) phi)`, the minimal system of Brauns and Marchetti. The coupling `chi` is reciprocal
    and `alpha` is not. When `alpha^2 - chi^2` exceeds `D^2`, the unstable band of the dispersion relation of the mixed
    state reaches the propagating modes, and the demixed domains travel instead of coarsening. The system has the
    demixing field `mixture.phi` and the diffusing field `mixture.psi`.

    Args:
        shape:
            The number of sites along every axis of the periodic lattice.

        length:
            The side of the domain.

        chi:
            The reciprocal part of the cross-diffusion.

        alpha:
            The nonreciprocal part of the cross-diffusion.

        D:
            The diffusivity of `psi`.

        kappa:
            The stiffness of the interfaces of `phi`.

        random_state:
            The seed or generator of the small random initial fields around the mixed state.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.pattern_formation import NonreciprocalCahnHilliardSystem
        >>> NonreciprocalCahnHilliardSystem((16,), alpha=0.5).prepare().system_.interactions[0].alpha
        0.5
    """

    field = 'physics/pattern_formation'
    topics = ('field', 'conserved', 'nonreciprocal', 'traveling waves', 'phase separation', 'pde')
    references = ('Brauns and Marchetti 2024', 'Saha, Agudo-Canalejo, and Golestanian 2020', 'Frohoff-Hülsmann 2021')
    until = 400.0
    dt = 2.0

    def __init__(
        self,
        shape: Sequence[int] = (128,),
        *,
        length: float = 40.0,
        chi: float = 0.0,
        alpha: float = 0.3,
        D: float = 0.1,
        kappa: float = 1.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.shape = shape
        self.length = length
        self.chi = chi
        self.alpha = alpha
        self.D = D
        self.kappa = kappa
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the mixture with small random fields around the mixed state.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        shape = tuple(self.shape)
        rng = np.random.default_rng(self.random_state)
        phi = 0.01 * rng.uniform(-1.0, 1.0, shape)
        psi = 0.01 * rng.uniform(-1.0, 1.0, shape)
        space = LatticeSpace(shape, spacing=self.length / shape[0])
        coupling = NonreciprocalCahnHilliardInteraction(
            'coupling',
            target='mixture',
            chi=self.chi,
            alpha=self.alpha,
            D=self.D,
            kappa=self.kappa,
        )
        mixture = NonreciprocalMixtureElement('mixture', phi=phi.ravel(), psi=psi.ravel(), space=space)
        self.system_ = System(self.label, elements=[mixture], interactions=[coupling])
