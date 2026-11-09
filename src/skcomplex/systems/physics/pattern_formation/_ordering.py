"""Phase ordering.

A scalar order parameter relaxes in a double well, non-conserved and conserved.
"""

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
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.solvers import ETDRK4Solver
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec


class PhaseFieldElement(Element):
    """Site of a phase field.

    The site holds the order parameter `phi` that free energies and noise move.

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
    """Gradient flow of a field `phi` in the Landau free energy of a double well.

    The energy has a cost of gradients.

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
        """Return the density `-phi^2 / 2 + phi^4 / 4 - kappa phi laplacian(phi) / 2`.

        The gradient term is taken by parts.

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
    label: str,
    *,
    shape: Sequence[int],
    length: float,
    mean: float,
    kappa: float,
    random_state: int | np.random.Generator | None,
    conserved: bool,
    temperature: float = 0.0,
) -> CompositeSystem:
    """Return a system of a label with a phase field that relaxes.

    The field starts from small noise around a mean.

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
    return CompositeSystem(label, elements=[field], interactions=interactions)


class AllenCahnSystem(BaseSystem):
    """The Allen-Cahn equation, System A.

    It is the non-conserved relaxation of an order parameter in a double well.

    The order parameter follows `phi_t = phi - phi^3 + kappa laplacian(phi)`, the gradient descent of the free energy
    density `-phi^2 / 2 + phi^4 / 4 + kappa |grad(phi)|^2 / 2`. From small noise, domains of `phi = 1` and `phi = -1`
    form and coarsen as their walls move by curvature, so their size grows as `t^(1/2)`. The mean of `phi` is not
    conserved, and a domain shrinks away instead of exchanging material with its neighbours. The system has the field
    `mixture.phi`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.pattern_formation import AllenCahnSystem
        >>> AllenCahnSystem((8, 8), kappa=2.0).prepare().system_.interactions[0].kappa
        2.0
    """

    field = 'physics/pattern_formation'
    topics = ('field', 'phase ordering', 'coarsening', 'system a', 'spectral', 'pde')
    references = REFERENCES
    until = 100.0
    dt = 1.0
    solver = ETDRK4Solver(dt=0.5)

    def __init__(
        self,
        shape: Sequence[int] = (128, 128),
        *,
        length: float = 128.0,
        mean: float = 0.0,
        kappa: float = 1.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.shape = shape
        self.length = length
        self.mean = mean
        self.kappa = kappa
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system, built from its model parameters."""
        return _ordering(
            self.label,
            shape=self.shape,
            length=self.length,
            mean=self.mean,
            kappa=self.kappa,
            random_state=self.random_state,
            conserved=False,
        )


class CahnHilliardSystem(BaseSystem):
    """The Cahn-Hilliard equation, System B.

    It is the conserved relaxation of an order parameter in a double well.

    The order parameter follows `phi_t = laplacian(-phi + phi^3 - kappa laplacian(phi))`, a diffusion down the
    gradient of the chemical potential of the same free energy as the Allen-Cahn equation. Inside the spinodal,
    `|mean| < 1 / sqrt(3)`, small noise decomposes into domains of `phi` near 1 and -1, which coarsen by diffusion
    through the bulk so that their size grows as `t^(1/3)`, while the mean of `phi` stays fixed.

    A positive temperature adds the thermal noise of System B, `div(sqrt(2 temperature) eta)`, which conserves the mean
    of `phi` as well. The field then uses finite differences, and an SDE solver such as `EulerMaruyamaSolver` integrates
    it, with a step below `spacing^4 / (32 kappa)`. The system has the field `mixture.phi`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.pattern_formation import CahnHilliardSystem
        >>> CahnHilliardSystem((8, 8), mean=0.3).prepare().system_.interactions[0].conserved
        True
        >>> system = CahnHilliardSystem((8, 8), temperature=0.1).prepare().system_
        >>> [interaction.label for interaction in system.interactions]
        ['energy', 'noise']
    """

    field = 'physics/pattern_formation'
    topics = ('field', 'phase separation', 'coarsening', 'conserved', 'system b', 'spectral', 'pde', 'conserved noise')
    references = REFERENCES
    until = 1000.0
    dt = 10.0
    solver = ETDRK4Solver(dt=0.25)

    def __init__(
        self,
        shape: Sequence[int] = (64, 64),
        *,
        length: float = 64.0,
        mean: float = 0.0,
        kappa: float = 1.0,
        temperature: float = 0.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.shape = shape
        self.length = length
        self.mean = mean
        self.kappa = kappa
        self.temperature = temperature
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system, built from its model parameters."""
        return _ordering(
            self.label,
            shape=self.shape,
            length=self.length,
            mean=self.mean,
            kappa=self.kappa,
            random_state=self.random_state,
            conserved=True,
            temperature=self.temperature,
        )
