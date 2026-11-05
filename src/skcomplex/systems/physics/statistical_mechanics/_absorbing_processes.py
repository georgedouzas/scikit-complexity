"""Fully connected reaction processes with an absorbing state: branching, decay, and pair annihilation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, JumpLaw
from ....core.components.laws.utils import where
from ....core.components.system import System
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_system


class ReactingParticlesElement(Element):
    """Number `n` of particles in a volume `V`, with the reactions `A -> 2A`, `A -> 0`, and `2A -> A`.

    A particle branches at rate `kappa`, decays at rate `nu`, and every pair annihilates at rate `lambda / V`, so
    `W(n + 1 | n) = kappa n + epsilon delta_n0` and `W(n - 1 | n) = nu n + lambda n (n - 1) / V`. The empty state
    absorbs when `epsilon = 0`.

    Args:
        label:
            The label of the particles.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        n:
            The initial number of particles.

        kappa:
            The branching rate.

        nu:
            The decay rate.

        lam:
            The annihilation rate.

        volume:
            The volume `V`.

        epsilon:
            The rate at which a particle appears in the empty state.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import ReactingParticlesElement
        >>> ReactingParticlesElement('particles').kappa
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n: Annotated[float, StateSpec(domain=NN)] = 100.0,
        kappa: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        nu: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        lam: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        volume: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 100.0,
        epsilon: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.n = n
        self.kappa = kappa
        self.nu = nu
        self.lam = lam
        self.volume = volume
        self.epsilon = epsilon

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the creation and the removal of one particle.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        n = s.n
        return [
            JumpLaw(s.kappa * n + s.epsilon * where(n < 1, 1, 0), {n: 1}),
            JumpLaw(s.nu * n + s.lam * n * (n - 1) / s.volume, {n: -1}),
        ]


@register_system(
    field='physics/statistical_mechanics',
    tags=('absorbing state', 'phase transition', 'birth-death process', 'metastability', 'jump process'),
    references=('Bertin 2021', 'Hinrichsen 2000'),
    defaults={'until': 100.0, 'dt': 0.1},
)
def absorbing_birth_death(
    *,
    kappa: float = 2.0,
    nu: float = 1.0,
    lam: float = 1.0,
    volume: float = 100.0,
    density: float = 1.0,
    epsilon: float = 0.0,
) -> System:
    """Return the fully connected branching and annihilation process, active above `kappa = nu` until it dies out.

    The deterministic density obeys `d rho / dt = (kappa - nu) rho - lambda rho^2`, with the active fixed point
    `rho_0 = (kappa - nu) / lambda`. With fluctuations the empty state absorbs every run, but above the transition the
    active state is metastable, with relative fluctuations `sigma / rho_0 = (lambda / (kappa - nu)) sqrt(kappa /
    (lambda V))` and a lifetime that grows like `exp(V phi(0)) / nu`. Below the transition the particles die out in a
    time of order `1 / (nu - kappa)`. A small `epsilon` makes the empty state leave, as in exercise 3.3 of Bertin.

    Args:
        kappa:
            The branching rate.

        nu:
            The decay rate.

        lam:
            The annihilation rate.

        volume:
            The volume `V`.

        density:
            The initial density, rounded to a whole number of particles.

        epsilon:
            The rate at which a particle appears in the empty state.

    Returns:
        system:
            The system, with the number of particles `particles.n`.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import absorbing_birth_death
        >>> absorbing_birth_death(volume=50.0).elements[0].n
        50.0
    """
    particles = ReactingParticlesElement(
        'particles',
        n=float(round(density * volume)),
        kappa=kappa,
        nu=nu,
        lam=lam,
        volume=volume,
        epsilon=epsilon,
    )
    return System(elements=[particles])
