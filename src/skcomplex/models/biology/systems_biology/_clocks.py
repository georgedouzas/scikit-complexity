"""Systems biology models: circadian clocks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.interactions import ReactionInteraction
from ....core.laws.utils import hill
from ....core.system import System
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_model
from ..common.kinetics import MichaelisMentenInteraction


class PerClockElement(Element):
    """PER protein clock of Drosophila, after Goldbeter: per mRNA, three phosphorylation forms, and nuclear PER.

    The mRNA `M` is transcribed under repression by nuclear PER and degraded. Cytosolic PER is translated as `P0`,
    phosphorylated reversibly to `P1` and `P2`, and `P2` is degraded or enters the nucleus as `PN`. Concentrations are
    in µM and time in hours.

    The element holds the states and parameters and has no laws of its own: the interactions of `circadian_clock` read
    its parameters.

    Args:
        label:
            The label of the clock.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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
    clock = PerClockElement('clock', v_s=v_s, v_d=v_d, k_s=k_s, n=n)
    reactions = [
        ReactionInteraction(
            'transcription',
            target='clock',
            products={'M': 1},
            propensity=lambda s: s.target.v_s * (1 - hill(s.target.PN, s.target.K_I, s.target.n)),
        ),
        MichaelisMentenInteraction(
            'mrna_degradation',
            target='clock',
            reactants={'M': 1},
            substrate='M',
            vmax=lambda s: s.target.v_m,
            km=lambda s: s.target.K_m,
        ),
        ReactionInteraction(
            'translation',
            target='clock',
            products={'P0': 1},
            propensity=lambda s: s.target.k_s * s.target.M,
        ),
        MichaelisMentenInteraction(
            'first_phosphorylation',
            target='clock',
            reactants={'P0': 1},
            products={'P1': 1},
            substrate='P0',
            vmax=lambda s: s.target.V1,
            km=lambda s: s.target.K,
        ),
        MichaelisMentenInteraction(
            'first_dephosphorylation',
            target='clock',
            reactants={'P1': 1},
            products={'P0': 1},
            substrate='P1',
            vmax=lambda s: s.target.V2,
            km=lambda s: s.target.K,
        ),
        MichaelisMentenInteraction(
            'second_phosphorylation',
            target='clock',
            reactants={'P1': 1},
            products={'P2': 1},
            substrate='P1',
            vmax=lambda s: s.target.V3,
            km=lambda s: s.target.K,
        ),
        MichaelisMentenInteraction(
            'second_dephosphorylation',
            target='clock',
            reactants={'P2': 1},
            products={'P1': 1},
            substrate='P2',
            vmax=lambda s: s.target.V4,
            km=lambda s: s.target.K,
        ),
        ReactionInteraction(
            'nuclear_entry',
            target='clock',
            reactants={'P2': 1},
            products={'PN': 1},
            propensity=lambda s: s.target.k1 * s.target.P2,
        ),
        ReactionInteraction(
            'nuclear_exit',
            target='clock',
            reactants={'PN': 1},
            products={'P2': 1},
            propensity=lambda s: s.target.k2 * s.target.PN,
        ),
        MichaelisMentenInteraction(
            'per_degradation',
            target='clock',
            reactants={'P2': 1},
            substrate='P2',
            vmax=lambda s: s.target.v_d,
            km=lambda s: s.target.K_d,
        ),
    ]
    return System(elements=[clock], interactions=reactions)
