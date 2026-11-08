"""Koopman embeddings.

A finite set of observables makes a quadratic flow with a slow manifold linear.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.system import BaseSystem, System
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class SlowManifoldElement(Element):
    """Quadratic flow `x1' = mu x1`, `x2' = lam (x2 - x1^2)`.

    The flow has the slow manifold `x2 = x1^2`.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x1:
            The initial slow coordinate.

        x2:
            The initial fast coordinate.

        mu:
            The slow eigenvalue.

        lam:
            The fast eigenvalue.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import SlowManifoldElement
        >>> SlowManifoldElement('flow', lam=-2.0).lam
        -2.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x1: Annotated[object, StateSpec()] = 1.0,
        x2: Annotated[object, StateSpec()] = 1.0,
        mu: Annotated[object, ParameterSpec()] = -0.05,
        lam: Annotated[object, ParameterSpec()] = -1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x1 = x1
        self.x2 = x2
        self.mu = mu
        self.lam = lam

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the slow and the fast coordinate.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        return [RateLaw(s.x1, s.mu * s.x1), RateLaw(s.x2, s.lam * (s.x2 - s.x1**2))]


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

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import KoopmanSlowManifoldSystem
        >>> KoopmanSlowManifoldSystem(lam=-2.0).prepare().system_.elements[0].lam
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
    ) -> None:
        super().__init__(label)
        self.x1 = x1
        self.x2 = x2
        self.mu = mu
        self.lam = lam

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        return System(
            self.label,
            elements=[SlowManifoldElement('flow', x1=self.x1, x2=self.x2, mu=self.mu, lam=self.lam)],
        )
