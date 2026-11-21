"""Koopman embeddings.

A finite set of observables makes a quadratic flow with a slow manifold linear.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role


class KoopmanSlowManifoldInteraction(Interaction):
    """The laws of the quadratic flow `x1' = mu x1`, `x2' = lam (x2 - x1^2)`.

    Args:
        label:
            The label of the interaction.

        target:
            The element whose states `x1` and `x2` the flow moves.

        mu:
            The slow eigenvalue.

        lam:
            The fast eigenvalue.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import KoopmanSlowManifoldInteraction
        >>> KoopmanSlowManifoldInteraction('koopman_slow_manifold', target='flow', mu=0.5).mu
        0.5
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'flow',
        mu: float = -0.05,
        lam: float = -1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.mu = mu
        self.lam = lam

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rate of every state as the laws."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x1, x2 = context.target.x1, context.target.x2
            mu, lam = context.mu, context.lam
            self.laws_: list[BaseLaw] = [
                RateLaw(x1, mu * x1),
                RateLaw(x2, lam * (x2 - x1**2)),
            ]


class KoopmanSlowManifoldSystem(BaseSystem):
    """The quadratic flow `x1' = mu x1`, `x2' = lam (x2 - x1^2)`.

    The flow has a slow manifold and a linear embedding.

    For `lam < mu < 0` the fast direction contracts onto the slow manifold `x2 = x1^2`, and the state then slides along
    it to the origin. With the observable `x1^2`, the coordinates `(x1, x2, x1^2)` span a Koopman-invariant subspace in
    which the flow is linear, with the eigenvalues `mu`, `lam`, and `2 mu`. The Koopman eigenfunctions `phi_mu = x1` and
    `phi_lam = x2 - b x1^2` with `b = lam / (lam - 2 mu)` decay exactly as `exp(mu t)` and `exp(lam t)`, so trajectories
    follow the parabolas `x2 = b x1^2 + C |x1|^(lam / mu)`. The system has the states `flow.x1` and `flow.x2`.

    Args:
        x1:
            The initial slow coordinate.

        x2:
            The initial fast coordinate.

        mu:
            The slow eigenvalue.

        lam:
            The fast eigenvalue.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import KoopmanSlowManifoldSystem
        >>> KoopmanSlowManifoldSystem(lam=-2.0).prepare().system_.interactions[0].lam
        -2.0
    """

    field = 'mathematics/dynamical_systems'
    topics = ('Koopman operator', 'slow manifold', 'eigenfunctions', 'ode')
    references = ('Brunton, Budišić, Kaiser, and Kutz 2022', 'Tu et al. 2014', 'Brunton et al. 2016')
    until = 40.0
    dt = 0.1

    def __init__(
        self,
        x1: float = 1.0,
        x2: float = -1.0,
        *,
        mu: float = -0.05,
        lam: float = -1.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x1 = x1
        self.x2 = x2
        self.mu = mu
        self.lam = lam

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the flow and its laws."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[Element('flow', values={'x1': self.x1, 'x2': self.x2})],
            interactions=[
                KoopmanSlowManifoldInteraction('koopman_slow_manifold', target='flow', mu=self.mu, lam=self.lam),
            ],
        )
