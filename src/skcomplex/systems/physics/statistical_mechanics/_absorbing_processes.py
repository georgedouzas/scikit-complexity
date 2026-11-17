"""Fully connected reaction processes with an absorbing state.

The reactions are branching, decay, and pair annihilation.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import JumpLaw
from ....core.components.laws.utils import where
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.specs import Role
from ....core.systems import BaseSystem, System


class ReactingParticlesElement(Element):
    """Number `n` of particles in a volume.

    Args:
        label:
            The label of the particles.

        n:
            The initial number of particles.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import ReactingParticlesElement
        >>> ReactingParticlesElement('particles').n
        100.0
    """

    def __init__(self, label: str, *, n: Annotated[float, NN] = 100.0) -> None:
        super().__init__(label)
        self.n = n


class BranchingAnnihilationInteraction(Interaction):
    """Reactions `A -> 2A`, `A -> 0`, and `2A -> A` of the particles in a volume `V`.

    A particle branches at rate `kappa`, decays at rate `nu`, and every pair annihilates at rate `lambda / V`, so
    `W(n + 1 | n) = kappa n + epsilon delta_n0` and `W(n - 1 | n) = nu n + lambda n (n - 1) / V`. The empty state
    absorbs when `epsilon = 0`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particles, whose value `n` counts them.

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
        >>> from skcomplex.systems.physics.statistical_mechanics import BranchingAnnihilationInteraction
        >>> BranchingAnnihilationInteraction('reactions', target='particles').kappa
        2.0
    """

    law_classes = (JumpLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        kappa: Annotated[float, RealSet.closed_open(0, oo)] = 2.0,
        nu: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        lam: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        volume: Annotated[float, RealSet.open(0, oo)] = 100.0,
        epsilon: Annotated[float, RealSet.closed_open(0, oo)] = 0.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.kappa = kappa
        self.nu = nu
        self.lam = lam
        self.volume = volume
        self.epsilon = epsilon

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the creation and the removal of one particle.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s, n = context, context.target.n
            self.laws_ = [
                JumpLaw(s.kappa * n + s.epsilon * where(n < 1, 1, 0), {n: 1}),
                JumpLaw(s.nu * n + s.lam * n * (n - 1) / s.volume, {n: -1}),
            ]


class AbsorbingBirthDeathSystem(BaseSystem):
    """The fully connected branching and annihilation process.

    The process is active above `kappa = nu` until it dies out.

    The deterministic density obeys `d rho / dt = (kappa - nu) rho - lambda rho^2`, with the active fixed point `rho_0 =
    (kappa - nu) / lambda`. With fluctuations the empty state absorbs every run, but above the transition the active
    state is metastable, with relative fluctuations `sigma / rho_0 = (lambda / (kappa - nu)) sqrt(kappa / (lambda V))`
    and a lifetime that grows like `exp(V phi(0)) / nu`. Below the transition the particles die out in a time of order
    `1 / (nu - kappa)`. A small `epsilon` makes the empty state leave, as in exercise 3.3 of Bertin. The system has the
    number of particles `particles.n`.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import AbsorbingBirthDeathSystem
        >>> AbsorbingBirthDeathSystem(volume=50.0).prepare().system_.elements[0].n
        50.0
    """

    field = 'physics/statistical_mechanics'
    topics = ('absorbing state', 'phase transition', 'birth-death process', 'metastability', 'jump process')
    references = ('Bertin 2021', 'Hinrichsen 2000')
    until = 100.0
    dt = 0.1

    def __init__(
        self,
        *,
        kappa: float = 2.0,
        nu: float = 1.0,
        lam: float = 1.0,
        volume: float = 100.0,
        density: float = 1.0,
        epsilon: float = 0.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.kappa = kappa
        self.nu = nu
        self.lam = lam
        self.volume = volume
        self.density = density
        self.epsilon = epsilon

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the particles and their reactions.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        reactions = BranchingAnnihilationInteraction(
            'reactions',
            target='particles',
            kappa=self.kappa,
            nu=self.nu,
            lam=self.lam,
            volume=self.volume,
            epsilon=self.epsilon,
        )
        particles = ReactingParticlesElement('particles', n=float(round(self.density * self.volume)))
        self.system_ = System(self.label, elements=[particles], interactions=[reactions])
