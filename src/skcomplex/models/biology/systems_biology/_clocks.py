"""Systems biology models: circadian clocks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....core.system import System
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_model


def _saturating(rate: Expression, constant: Expression, amount: Expression) -> Expression:
    """Return the Michaelis-Menten rate `rate * amount / (constant + amount)`."""
    return rate * amount / (constant + amount)


class PerClockElement(Element):
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
        >>> from skcomplex.models.biology.systems_biology import PerClockElement
        >>> PerClockElement('clock').v_s
        0.76
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        M: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        P0: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        P1: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        P2: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        PN: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        v_s: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.76,
        v_m: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.65,
        K_m: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        k_s: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.38,
        v_d: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.95,
        K_d: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.2,
        k1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.9,
        k2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.3,
        K_I: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        n: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 4.0,
        V1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 3.2,
        V2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.58,
        V3: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 5.0,
        V4: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.5,
        K: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.M = M
        self.P0 = P0
        self.P1 = P1
        self.P2 = P2
        self.PN = PN
        self.v_s = v_s
        self.v_m = v_m
        self.K_m = K_m
        self.k_s = k_s
        self.v_d = v_d
        self.K_d = K_d
        self.k1 = k1
        self.k2 = k2
        self.K_I = K_I
        self.n = n
        self.V1 = V1
        self.V2 = V2
        self.V3 = V3
        self.V4 = V4
        self.K = K

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
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
            RateLaw(s.M, s.v_s * s.K_I**s.n / (s.K_I**s.n + s.PN**s.n) - _saturating(s.v_m, s.K_m, s.M)),
            RateLaw(s.P0, s.k_s * s.M - first),
            RateLaw(s.P1, first - second),
            RateLaw(s.P2, second - transport - _saturating(s.v_d, s.K_d, s.P2)),
            RateLaw(s.PN, transport),
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
    return System(elements=[PerClockElement('clock', v_s=v_s, v_d=v_d, k_s=k_s, n=n)])
