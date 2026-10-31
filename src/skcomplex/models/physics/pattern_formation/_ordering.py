"""Phase ordering: relaxation of a scalar order parameter in a double well, non-conserved and conserved."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence

import numpy as np

from ....core.elements import FieldElement
from ....core.interactions import BaseInteraction, RateInteraction
from ....core.spaces import LatticeSpace
from ....core.system import System
from ....engine.solvers import ETDRK4Solver
from ..._catalog import register_model
from ..common.fields import LandauEnergyInteraction, PhaseFieldElement

REFERENCES = ('Brauns and Marchetti 2024', 'Allen and Cahn 1979', 'Cahn and Hilliard 1958', 'Bray 1994')


def _ordering(
    *,
    shape: Sequence[int],
    length: float,
    mean: float,
    kappa: float,
    random_state: int | np.random.Generator | None,
    conserved: bool,
    temperature: float = 0.0,
) -> System:
    """Return a phase field that starts from small noise around a mean and relaxes in the Landau free energy.

    A positive temperature adds the conserved thermal noise, on finite differences instead of the spectral lattice.
    """
    shape = tuple(shape)
    rng = np.random.default_rng(random_state)
    values = {'phi': mean + 0.05 * rng.uniform(-1.0, 1.0, shape)}
    space = LatticeSpace(shape, spacing=length / shape[0])
    interactions: list[BaseInteraction] = [
        LandauEnergyInteraction('energy', target='mixture', kappa=kappa, conserved=conserved),
    ]
    if temperature:
        noise = RateInteraction(
            'noise',
            target='mixture',
            state='phi',
            noise=math.sqrt(2 * temperature),
            conserved=True,
        )
        interactions.append(noise)
    discretization = 'finite-difference' if temperature else 'spectral'
    field = FieldElement(
        'mixture',
        element=PhaseFieldElement,
        space=space,
        values=values,
        discretization=discretization,
    )
    return System(elements=[field], interactions=interactions)


@register_model(
    field='physics/pattern_formation',
    tags=('field', 'phase ordering', 'coarsening', 'model a', 'spectral', 'pde'),
    references=REFERENCES,
    defaults={'until': 100.0, 'dt': 1.0, 'solver': ETDRK4Solver(dt=0.5)},
)
def allen_cahn(
    shape: Sequence[int] = (128, 128),
    *,
    length: float = 128.0,
    mean: float = 0.0,
    kappa: float = 1.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Allen-Cahn equation, Model A, the non-conserved relaxation of an order parameter in a double well.

    The order parameter follows `phi_t = phi - phi^3 + kappa laplacian(phi)`, the gradient descent of the free energy
    density `-phi^2 / 2 + phi^4 / 4 + kappa |grad(phi)|^2 / 2`. From small noise, domains of `phi = 1` and `phi = -1`
    form and coarsen as their walls move by curvature, so their size grows as `t^(1/2)`. The mean of `phi` is not
    conserved, and a domain shrinks away instead of exchanging material with its neighbours.

    Args:
        shape:
            The number of sites along every axis of the periodic lattice.

        length:
            The side of the domain.

        mean:
            The mean of the initial noise.

        kappa:
            The stiffness of the domain walls.

        random_state:
            The seed or generator of the initial noise of amplitude 0.05.

    Returns:
        system:
            The model, with the field `mixture.phi`.

    Examples:
        >>> from skcomplex.models.physics.pattern_formation import allen_cahn
        >>> allen_cahn((8, 8), kappa=2.0).interactions[0].kappa
        2.0
    """
    return _ordering(
        shape=shape,
        length=length,
        mean=mean,
        kappa=kappa,
        random_state=random_state,
        conserved=False,
    )


@register_model(
    field='physics/pattern_formation',
    tags=('field', 'phase separation', 'coarsening', 'conserved', 'model b', 'spectral', 'pde', 'conserved noise'),
    references=REFERENCES,
    defaults={'until': 1000.0, 'dt': 10.0, 'solver': ETDRK4Solver(dt=0.25)},
)
def cahn_hilliard(
    shape: Sequence[int] = (64, 64),
    *,
    length: float = 64.0,
    mean: float = 0.0,
    kappa: float = 1.0,
    temperature: float = 0.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Cahn-Hilliard equation, Model B, the conserved relaxation of an order parameter in a double well.

    The order parameter follows `phi_t = laplacian(-phi + phi^3 - kappa laplacian(phi))`, a diffusion down the
    gradient of the chemical potential of the same free energy as the Allen-Cahn equation. Inside the spinodal,
    `|mean| < 1 / sqrt(3)`, small noise decomposes into domains of `phi` near 1 and -1, which coarsen by diffusion
    through the bulk so that their size grows as `t^(1/3)`, while the mean of `phi` stays fixed.

    A positive temperature adds the thermal noise of Model B, `div(sqrt(2 temperature) eta)`, which conserves the mean
    of `phi` as well. The field then uses finite differences, and an SDE solver such as `EulerMaruyamaSolver` integrates
    it, with a step below `spacing^4 / (32 kappa)`.

    Args:
        shape:
            The number of sites along every axis of the periodic lattice.

        length:
            The side of the domain.

        mean:
            The mean of the order parameter, which the dynamics conserve.

        kappa:
            The stiffness of the interfaces.

        temperature:
            The temperature of the conserved thermal noise, or 0 for the deterministic equation.

        random_state:
            The seed or generator of the initial noise of amplitude 0.05.

    Returns:
        system:
            The model, with the field `mixture.phi`.

    Examples:
        >>> from skcomplex.models.physics.pattern_formation import cahn_hilliard
        >>> cahn_hilliard((8, 8), mean=0.3).interactions[0].conserved
        True
        >>> [interaction.label for interaction in cahn_hilliard((8, 8), temperature=0.1).interactions]
        ['energy', 'noise']
    """
    return _ordering(
        shape=shape,
        length=length,
        mean=mean,
        kappa=kappa,
        random_state=random_state,
        conserved=True,
        temperature=temperature,
    )
