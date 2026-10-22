"""Systems biology models: circadian clocks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import Domain, Expression, parameter, state
from ....core.components import Element
from ....core.laws import Law, Rate, Symbols
from ....core.runtime import register_model
from ....core.system import System


def _saturating(rate: Expression, constant: Expression, amount: Expression) -> Expression:
    """Return the Michaelis-Menten rate `rate * amount / (constant + amount)`."""
    return rate * amount / (constant + amount)


class PerClock(Element):
    """PER protein clock of Drosophila, after Goldbeter: per mRNA, three phosphorylation forms, and nuclear PER.

    The mRNA `M` is transcribed under repression by nuclear PER and degraded. Cytosolic PER is translated as `P0`,
    phosphorylated reversibly to `P1` and `P2`, and `P2` is degraded or enters the nucleus as `PN`. Concentrations are
    in µM and time in hours.

    Args:
        label:
            The label of the clock.

        M:
            The initial per mRNA concentration.

        P0:
            The initial unphosphorylated PER.

        P1:
            The initial monophosphorylated PER.

        P2:
            The initial bisphosphorylated PER.

        PN:
            The initial nuclear PER.

        v_s:
            The maximal transcription rate.

        v_m:
            The maximal mRNA degradation rate.

        K_m:
            The Michaelis constant of mRNA degradation.

        k_s:
            The translation rate.

        v_d:
            The maximal degradation rate of `P2`.

        K_d:
            The Michaelis constant of the degradation of `P2`.

        k1:
            The rate of nuclear entry.

        k2:
            The rate of nuclear exit.

        K_I:
            The threshold of repression by nuclear PER.

        n:
            The Hill coefficient of repression.

        V1:
            The maximal rate of the first phosphorylation.

        V2:
            The maximal rate of the first dephosphorylation.

        V3:
            The maximal rate of the second phosphorylation.

        V4:
            The maximal rate of the second dephosphorylation.

        K:
            The Michaelis constant of the phosphorylations and dephosphorylations.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import PerClock
        >>> PerClock('clock').v_s
        0.76
    """

    M: float = state(default=0.5, domain=Domain.NONNEGATIVE)
    P0: float = state(default=0.5, domain=Domain.NONNEGATIVE)
    P1: float = state(default=0.5, domain=Domain.NONNEGATIVE)
    P2: float = state(default=0.5, domain=Domain.NONNEGATIVE)
    PN: float = state(default=0.5, domain=Domain.NONNEGATIVE)
    v_s: float = parameter(default=0.76, domain=Domain.POSITIVE)
    v_m: float = parameter(default=0.65, domain=Domain.POSITIVE)
    K_m: float = parameter(default=0.5, domain=Domain.POSITIVE)
    k_s: float = parameter(default=0.38, domain=Domain.POSITIVE)
    v_d: float = parameter(default=0.95, domain=Domain.POSITIVE)
    K_d: float = parameter(default=0.2, domain=Domain.POSITIVE)
    k1: float = parameter(default=1.9, domain=Domain.POSITIVE)
    k2: float = parameter(default=1.3, domain=Domain.POSITIVE)
    K_I: float = parameter(default=1.0, domain=Domain.POSITIVE)
    n: float = parameter(default=4.0, domain=Domain.POSITIVE)
    V1: float = parameter(default=3.2, domain=Domain.POSITIVE)
    V2: float = parameter(default=1.58, domain=Domain.POSITIVE)
    V3: float = parameter(default=5.0, domain=Domain.POSITIVE)
    V4: float = parameter(default=2.5, domain=Domain.POSITIVE)
    K: float = parameter(default=2.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return transcription, translation, phosphorylation, degradation, and nuclear transport.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        first = _saturating(s.V1, s.K, s.P0) - _saturating(s.V2, s.K, s.P1)
        second = _saturating(s.V3, s.K, s.P1) - _saturating(s.V4, s.K, s.P2)
        transport = s.k1 * s.P2 - s.k2 * s.PN
        return [
            Rate(s.M, s.v_s * s.K_I**s.n / (s.K_I**s.n + s.PN**s.n) - _saturating(s.v_m, s.K_m, s.M)),
            Rate(s.P0, s.k_s * s.M - first),
            Rate(s.P1, first - second),
            Rate(s.P2, second - transport - _saturating(s.v_d, s.K_d, s.P2)),
            Rate(s.PN, transport),
        ]


@register_model(
    field='biology/systems_biology',
    tags=('circadian rhythm', 'oscillations', 'gene regulation'),
    references=('Goldbeter 1995', 'Trzaska 2023'),
    defaults={'until': 240.0, 'dt': 0.5},
)
def circadian_clock(
    *,
    v_s: float = 0.76,
    v_d: float = 0.95,
    k_s: float = 0.38,
    n: float = 4.0,
) -> System:
    """Return Goldbeter's model of the Drosophila circadian clock, PER that represses its own transcription.

    Multiple phosphorylation and nuclear entry delay the negative feedback, which yields a limit cycle with a period
    near 24 hours for the published parameters. Larger degradation `v_d` lengthens the period, from 22.6 hours at
    `v_d = 0.8` to 29.3 hours at `v_d = 1.6`.

    Args:
        v_s:
            The maximal transcription rate in µM/h.

        v_d:
            The maximal degradation rate of bisphosphorylated PER in µM/h.

        k_s:
            The translation rate per hour.

        n:
            The Hill coefficient of repression.

    Returns:
        system:
            The model, with the concentrations `clock.M`, `clock.P0`, `clock.P1`, `clock.P2`, and `clock.PN`.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import circadian_clock
        >>> circadian_clock(v_d=1.2).elements[0].v_d
        1.2
    """
    return System(components=[PerClock('clock', v_s=v_s, v_d=v_d, k_s=k_s, n=n)])
