"""Phase ordering: relaxation of a scalar order parameter in a double well, non-conserved and conserved."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, FieldElement
from ....core.components.interactions import BaseInteraction, GradientInteraction, RateInteraction
from ....core.components.laws.utils import laplacian
from ....core.components.spaces import LatticeSpace
from ....core.components.system import System
from ....core.engine.solvers import ETDRK4Solver
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_system


class PhaseFieldElement(Element):
    """Site of a phase field, which holds the order parameter `phi` that free energies and noise move.

    Args:
        label:
            The label of the site or the field.

        visual:
            How the site is drawn, or `None` for `default_visual_`.

        phi:
            The initial order parameter.

    Attributes:
        default_visual_:
            The visual of the site when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.pattern_formation._ordering import PhaseFieldElement
        >>> PhaseFieldElement('site', phi=0.5).phi
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        phi: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.phi = phi


class LandauEnergyInteraction(GradientInteraction):
    """Gradient flow of a field `phi` in the Landau free energy of a double well with a cost of gradients.

    The free energy is `F = integral -phi^2 / 2 + phi^4 / 4 + kappa |grad phi|^2 / 2`. Its functional derivative is
    the chemical potential `mu = -phi + phi^3 - kappa laplacian(phi)`. A non-conserved field relaxes as `phi_t = -mu`,
    the Allen-Cahn equation of system A, and a conserved field as `phi_t = laplacian(mu)`, the Cahn-Hilliard equation of
    system B, whose total stays constant.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the field.

        state:
            The name of the state of the field that holds the order parameter.

        kappa:
            The cost of gradients, the square of the width of the interfaces.

        conserved:
            Whether the order parameter is conserved.

    Examples:
        >>> from types import SimpleNamespace
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.systems.physics.pattern_formation._ordering import LandauEnergyInteraction
        >>> energy = LandauEnergyInteraction('energy', target='mixture').prepare()
        >>> namespace = SimpleNamespace(target=SimpleNamespace(phi=var('phi')), kappa=var('kappa'))
        >>> [law.value for law in energy.run(namespace).laws_]
        [-phi^3 + kappa*laplacian(phi) + phi]
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        state: str = 'phi',
        kappa: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        conserved: bool = False,
    ) -> None:
        super().__init__(label, target=target, states=state, variational=True, conserved=conserved)
        self.state = state
        self.kappa = kappa

    def _get_potential(self, namespace: SimpleNamespace) -> Expression:
        """Return the density `-phi^2 / 2 + phi^4 / 4 - kappa phi laplacian(phi) / 2`, the gradient term by parts.

        Args:
            namespace:
                The namespace of the symbols the density can use.

        Returns:
            density:
                The density of the free energy.
        """
        phi = getattr(namespace.target, self.state)
        return -(phi**2) / 2 + phi**4 / 4 - namespace.kappa * phi * laplacian(phi) / 2


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


@register_system(
    field='physics/pattern_formation',
    tags=('field', 'phase ordering', 'coarsening', 'system a', 'spectral', 'pde'),
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
    """Return the Allen-Cahn equation, System A, the non-conserved relaxation of an order parameter in a double well.

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
            The system, with the field `mixture.phi`.

    Examples:
        >>> from skcomplex.systems.physics.pattern_formation import allen_cahn
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


@register_system(
    field='physics/pattern_formation',
    tags=('field', 'phase separation', 'coarsening', 'conserved', 'system b', 'spectral', 'pde', 'conserved noise'),
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
    """Return the Cahn-Hilliard equation, System B, the conserved relaxation of an order parameter in a double well.

    The order parameter follows `phi_t = laplacian(-phi + phi^3 - kappa laplacian(phi))`, a diffusion down the
    gradient of the chemical potential of the same free energy as the Allen-Cahn equation. Inside the spinodal,
    `|mean| < 1 / sqrt(3)`, small noise decomposes into domains of `phi` near 1 and -1, which coarsen by diffusion
    through the bulk so that their size grows as `t^(1/3)`, while the mean of `phi` stays fixed.

    A positive temperature adds the thermal noise of System B, `div(sqrt(2 temperature) eta)`, which conserves the mean
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
            The system, with the field `mixture.phi`.

    Examples:
        >>> from skcomplex.systems.physics.pattern_formation import cahn_hilliard
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
