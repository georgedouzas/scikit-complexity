"""Cell kill proportional to the tumour and to a drug exposure given in a train of rectangular doses."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import RateInteraction
from ....core.components.laws.utils import pulse_train
from ....core.specs import ParameterSpec, RoleSpec


class LogKillInteraction(RateInteraction):
    """Cell kill proportional to the tumour and to a drug exposure given in a train of rectangular doses.

    The kill is `dn/dt = -a n U(t)`, where the exposure `U` is `dose` for `width` time units every `period`, from
    `start` on. Without growth, every dose lowers `log n` by `a dose width`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the tumour.

        a:
            The kill rate per unit of exposure.

        dose:
            The exposure during a dose.

        period:
            The time between the starts of doses.

        width:
            The duration of a dose.

        start:
            The time of the first dose.

    Examples:
        >>> from skcomplex.systems.biology.oncology import LogKillInteraction
        >>> LogKillInteraction('kill', target='tumour', period=21.0).period
        21.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        a: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        dose: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        period: float = 7.0,
        width: float = 1.0,
        start: float = 0.0,
    ) -> None:
        super().__init__(label, target=target, state='n')
        self.a = a
        self.dose = dose
        self.period = period
        self.width = width
        self.start = start

    def _get_rate(self, s: SimpleNamespace) -> Expression:
        """Return the rate `-a n U(t)`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            rate:
                The rate of the size of the tumour.
        """
        exposure = pulse_train(s.t, self.period, self.width, start=self.start)
        return -s.a * s.dose * exposure * s.target.n
