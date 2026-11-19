"""Thermohaline circulation between ocean boxes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace

from sage.all__sagemath_symbolics import abs_symbolic

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import RateLaw
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role


class StommelInteraction(Interaction):
    """Relaxation and mixing of the temperature and salinity differences of two ocean boxes.

    The differences `T` and `S` of the target follow `T' = eta1 - T (1 + |T - S|)` and `S' = eta2 - S (eta3 + |T -
    S|)`, a relaxation to their forcing and a mixing by a flow proportional to the density difference.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the boxes.

        eta1:
            The thermal forcing.

        eta2:
            The freshwater forcing.

        eta3:
            The ratio of the relaxation times of temperature and salinity.

    Examples:
        >>> from skcomplex.systems.physics.oceanography import StommelInteraction
        >>> StommelInteraction('circulation', target='ocean', eta2=1.0).eta2
        1.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        eta1: float = 3.0,
        eta2: float = 1.2,
        eta3: float = 0.3,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.eta1 = eta1
        self.eta2 = eta2
        self.eta3 = eta3

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rates of the temperature and salinity differences.

        Args:
            context:
                The namespace of the boxes, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            T, S, t = context.target.T, context.target.S, context.t
            flow = abs_symbolic(T - S)
            self.laws_ = [
                RateLaw(T.diff(t), context.eta1 - T * (1 + flow)),
                RateLaw(S.diff(t), context.eta2 - S * (context.eta3 + flow)),
            ]


class StommelSystem(BaseSystem):
    """Stommel's two-box thermohaline circulation.

    The circulation is bistable between a thermal and a haline mode.

    The dimensionless differences of temperature `T` and salinity `S` between an equatorial and a polar box relax to
    their forcing and are mixed by a flow proportional to the density difference `|T - S|`: `dT/dt = eta1 - T (1 + |T -
    S|)` and `dS/dt = eta2 - S (eta3 + |T - S|)`. For the default parameters a strong circulation driven by temperature
    and a weak one driven by salinity are both stable. The system has the differences `ocean.T` and `ocean.S`.

    Args:
        T:
            The initial temperature difference.

        S:
            The initial salinity difference.

        eta1:
            The thermal forcing.

        eta2:
            The freshwater forcing.

        eta3:
            The ratio of the relaxation times of temperature and salinity.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.oceanography import StommelSystem
        >>> StommelSystem(eta2=1.0).prepare().system_.interactions[0].eta2
        1.0
    """

    field = 'physics/oceanography'
    topics = ('ode', 'bistability', 'thermohaline circulation', 'tipping')
    references = ('Stommel 1961', 'Cessi 1994', 'Ghil and Lucarini 2020')
    until = 50.0
    dt = 0.1

    def __init__(
        self,
        T: float = 3.0,
        S: float = 0.0,
        *,
        eta1: float = 3.0,
        eta2: float = 1.2,
        eta3: float = 0.3,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.T = T
        self.S = S
        self.eta1 = eta1
        self.eta2 = eta2
        self.eta3 = eta3

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the two ocean boxes and their circulation.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[Element('ocean', values={'T': self.T, 'S': self.S})],
            interactions=[
                StommelInteraction('circulation', target='ocean', eta1=self.eta1, eta2=self.eta2, eta3=self.eta3),
            ],
        )
