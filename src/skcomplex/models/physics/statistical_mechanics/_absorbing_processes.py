"""Fully connected reaction processes with an absorbing state: branching, decay, and pair annihilation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import Domain, parameter, state
from ....core.model import Element, Jump, Law, Symbols, System, where
from ....core.runtime import register_model


class ReactingParticles(Element):
    """Number `n` of particles in a volume `V`, with the reactions `A -> 2A`, `A -> 0`, and `2A -> A`.

    A particle branches at rate `kappa`, decays at rate `nu`, and every pair annihilates at rate `lambda / V`, so
    `W(n + 1 | n) = kappa n + epsilon delta_n0` and `W(n - 1 | n) = nu n + lambda n (n - 1) / V`. The empty state
    absorbs when `epsilon = 0`.

    Args:
        label:
            The label of the particles.

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
        >>> from skcomplex.models.physics.statistical_mechanics import ReactingParticles
        >>> ReactingParticles('particles').kappa
        2.0
    """

    n: float = state(default=100.0, domain=Domain.NONNEGATIVE_INTEGER)
    kappa: float = parameter(default=2.0, domain=Domain.NONNEGATIVE)
    nu: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    lam: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    volume: float = parameter(default=100.0, domain=Domain.POSITIVE)
    epsilon: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
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
            Jump(s.kappa * n + s.epsilon * where(n < 1, 1, 0), {n: 1}),
            Jump(s.nu * n + s.lam * n * (n - 1) / s.volume, {n: -1}),
        ]


@register_model(
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
            The model, with the number of particles `particles.n`.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import absorbing_birth_death
        >>> absorbing_birth_death(volume=50.0).elements[0].n
        50.0
    """
    particles = ReactingParticles(
        'particles',
        n=float(round(density * volume)),
        kappa=kappa,
        nu=nu,
        lam=lam,
        volume=volume,
        epsilon=epsilon,
    )
    return System(elements=[particles])
