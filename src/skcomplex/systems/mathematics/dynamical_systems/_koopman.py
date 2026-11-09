"""Koopman embeddings.

A finite set of observables makes a quadratic flow with a slow manifold linear.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations


class KoopmanSlowManifoldSystem(DynamicalSystem):
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
        >>> KoopmanSlowManifoldSystem(lam=-2.0).prepare().system_.elements[0].get_value('lam')
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
        super().__init__(label=label)
        self.x1 = x1
        self.x2 = x2
        self.mu = mu
        self.lam = lam

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'flow'

    def _get_states(self) -> dict[str, object]:
        """Return the initial value of every state."""
        return {
            'x1': self.x1,
            'x2': self.x2,
        }

    def _get_parameters(self) -> dict[str, object]:
        """Return the parameters of the equations."""
        return {
            'mu': self.mu,
            'lam': self.lam,
        }

    def _get_equations(self) -> Equations:
        """Return the rates of the slow and the fast coordinate."""
        return {
            'x1': lambda s: s.target.mu * s.target.x1,
            'x2': lambda s: s.target.lam * (s.target.x2 - s.target.x1**2),
        }
