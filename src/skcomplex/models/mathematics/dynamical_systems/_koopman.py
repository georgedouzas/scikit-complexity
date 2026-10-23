"""Koopman embeddings: a quadratic flow with a slow manifold that a finite set of observables makes linear."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.components import Element
from ....core.laws import Rate
from ....core.runtime import register_model
from ....core.system import System

SlowManifold = Element.define(
    'SlowManifold',
    states={'x1': 1.0, 'x2': 1.0},
    parameters={'mu': -0.05, 'lam': -1.0},
    laws=lambda s: [Rate(s.x1, s.mu * s.x1), Rate(s.x2, s.lam * (s.x2 - s.x1**2))],
)


@register_model(
    field='mathematics/dynamical_systems',
    tags=('Koopman operator', 'slow manifold', 'eigenfunctions', 'ode'),
    references=('Brunton, Budišić, Kaiser, and Kutz 2022', 'Tu et al. 2014', 'Brunton et al. 2016'),
    defaults={'until': 40.0, 'dt': 0.1},
)
def koopman_slow_manifold(
    x1: float = 1.0,
    x2: float = -1.0,
    *,
    mu: float = -0.05,
    lam: float = -1.0,
) -> System:
    """Return the quadratic flow `x1' = mu x1`, `x2' = lam (x2 - x1^2)`, with a slow manifold and a linear embedding.

    For `lam < mu < 0` the fast direction contracts onto the slow manifold `x2 = x1^2`, and the state then slides
    along it to the origin. With the observable `x1^2`, the coordinates `(x1, x2, x1^2)` span a Koopman-invariant
    subspace in which the flow is linear, with the eigenvalues `mu`, `lam`, and `2 mu`. The Koopman eigenfunctions
    `phi_mu = x1` and `phi_lam = x2 - b x1^2` with `b = lam / (lam - 2 mu)` decay exactly as `exp(mu t)` and
    `exp(lam t)`, so trajectories follow the parabolas `x2 = b x1^2 + C |x1|^(lam / mu)`.

    Args:
        x1:
            The initial slow coordinate.

        x2:
            The initial fast coordinate.

        mu:
            The slow eigenvalue.

        lam:
            The fast eigenvalue.

    Returns:
        system:
            The model, with the states `flow.x1` and `flow.x2`.

    Examples:
        >>> from skcomplex.models.mathematics.dynamical_systems import koopman_slow_manifold
        >>> koopman_slow_manifold(lam=-2.0).elements[0].lam
        -2.0
    """
    return System(components=[SlowManifold('flow', x1=x1, x2=x2, mu=mu, lam=lam)])
