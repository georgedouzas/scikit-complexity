"""Define irradiation, impulses that kill a linear-quadratic fraction of a tumour at every fraction of a dose."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, exp, oo
from sage.sets.real_set import RealSet

from ....core.interactions import ImpulseInteraction
from ....specs import ParameterSpec, RoleSpec


class IrradiationInteraction(ImpulseInteraction):
    """Radiation fractions of a dose `d`, which move the lethally irradiated part of the proliferating cells.

    A fraction at the time `start + i period` moves the part `M = 1 - exp(-(1 + ap Us)(alpha d + beta d^2))` of the
    proliferating cells `A1` to the lethally irradiated cells `A5` and raises the accumulated dose `u_tot` by
    `(1 + au Us) d`, where `Us` is the radiosensitizer. Every fraction is an event labelled by the interaction.

    Args:
        label:
            The label of the interaction and of its events.

        target:
            The path of the tumour, with the states `A1`, `A5`, `Us`, and `u_tot`.

        alpha:
            The linear coefficient of the linear-quadratic model.

        beta:
            The quadratic coefficient of the linear-quadratic model.

        ap:
            The potency of the radiosensitizer in raising the cell kill.

        au:
            The potency of the radiosensitizer in raising the accumulated dose.

        dose:
            The dose of a fraction.

        period:
            The time between fractions.

        start:
            The time of the first fraction.

        fractions:
            The number of fractions.

    Raises:
        ValidationError:
            The times, the period, or the count of the fractions cannot be taken.

    Examples:
        >>> from skcomplex.models.biology.oncology import IrradiationInteraction
        >>> IrradiationInteraction('fraction', target='tumour', fractions=5).check().count
        5
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        alpha: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.054,
        beta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0054,
        ap: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.42,
        au: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.15,
        dose: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        period: float = 1.0,
        start: float = 0.125,
        fractions: int = 10,
    ) -> None:
        super().__init__(label, target=target, reset=self._get_resets(), period=period, start=start, count=fractions)
        self.alpha = alpha
        self.beta = beta
        self.ap = ap
        self.au = au
        self.dose = dose
        self.fractions = fractions

    def _get_damage(self, s: SimpleNamespace) -> Expression:
        """Return the linear-quadratic damage `(1 + ap Us)(alpha d + beta d^2)` of a fraction."""
        return (1 + s.ap * s.target.Us) * (s.alpha * s.dose + s.beta * s.dose**2)

    def _get_resets(self) -> dict[str, Callable[[SimpleNamespace], object]]:
        """Return the surviving proliferating cells, the lethally irradiated cells, and the accumulated dose."""
        return {
            'A1': lambda s: s.target.A1 * exp(-self._get_damage(s)),
            'A5': lambda s: s.target.A5 + (1 - exp(-self._get_damage(s))) * s.target.A1,
            'u_tot': lambda s: s.target.u_tot + (1 + s.au * s.target.Us) * s.dose,
        }
