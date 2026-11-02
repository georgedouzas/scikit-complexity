"""Linear-quadratic survival of cells after a dose of radiation given in fractions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import exp, oo
from sage.sets.real_set import RealSet

from ....core.base import BasePart
from ....core.elements import Element
from ....core.laws import EquationLaw
from ....specs import ParameterSpec, StateSpec


class LinearQuadraticSurvivalElement(Element):
    """Surviving fraction of cells, Lea-Catcheside factor, and biologically effective dose of a total dose.

    The survival is `S = exp(-alpha D - beta G D^2)`, where `D` is the total dose. The Lea-Catcheside factor of `n`
    fractions, each delivered at a constant rate in a time `T` with repair at the rate `mu`, far enough apart to repair
    between them, is `G = 2 (mu T - 1 + exp(-mu T)) / (n (mu T)^2)`, which tends to `1 / n` as `T` tends to 0. The
    biologically effective dose is `BED = D (1 + G D beta / alpha)`.

    Args:
        label:
            The label of the cells.

        survival:
            The initial guess of the surviving fraction, which the equations replace.

        protraction:
            The initial guess of the Lea-Catcheside factor, which the equations replace.

        effective_dose:
            The initial guess of the biologically effective dose, which the equations replace.

        alpha:
            The linear coefficient, per Gy.

        beta:
            The quadratic coefficient, per Gy squared.

        dose:
            The total dose, in Gy.

        fractions:
            The number of fractions.

        repair_rate:
            The rate of repair of sublethal damage, per hour.

        delivery_time:
            The time to deliver one fraction, in hours.

    Examples:
        >>> from skcomplex.models.biology.oncology import LinearQuadraticSurvivalElement
        >>> LinearQuadraticSurvivalElement('cells', dose=60.0, fractions=30.0).fractions
        30.0
    """

    def __init__(
        self,
        label: str,
        *,
        survival: Annotated[float, StateSpec()] = 1.0,
        protraction: Annotated[float, StateSpec()] = 1.0,
        effective_dose: Annotated[float, StateSpec()] = 0.0,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.3,
        beta: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.03,
        dose: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        fractions: Annotated[float, ParameterSpec(domain=RealSet.closed_open(1, oo))] = 1.0,
        repair_rate: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.46,
        delivery_time: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1e-3,
    ) -> None:
        super().__init__(label)
        self.survival = survival
        self.protraction = protraction
        self.effective_dose = effective_dose
        self.alpha = alpha
        self.beta = beta
        self.dose = dose
        self.fractions = fractions
        self.repair_rate = repair_rate
        self.delivery_time = delivery_time

    def _get_laws(self, s: SimpleNamespace) -> Sequence[BasePart]:
        """Return the equations of the Lea-Catcheside factor, the survival, and the biologically effective dose.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The three equations.
        """
        repaired = s.repair_rate * s.delivery_time
        protraction = 2 * (repaired - 1 + exp(-repaired)) / (s.fractions * repaired**2)
        return [
            EquationLaw(s.protraction == protraction, target=s.protraction),
            EquationLaw(
                s.survival == exp(-s.alpha * s.dose - s.beta * protraction * s.dose**2),
                target=s.survival,
            ),
            EquationLaw(
                s.effective_dose == s.dose * (1 + protraction * s.dose * s.beta / s.alpha),
                target=s.effective_dose,
            ),
        ]
