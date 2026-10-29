"""Ecology models: succession of pioneer, intermediate, and climax tree species."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.interactions import Interaction
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import where
from ....core.system import System
from ....specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_model


def _crowding(
    first: SimpleNamespace,
    second: SimpleNamespace,
    weight_first: Expression,
    weight_second: Expression,
) -> Expression:
    """Return the weighted share of the mature volumes that two stands fill."""
    total = weight_first + weight_second
    return weight_first / total * first.volume / first.mature + weight_second / total * second.volume / second.mature


class StandElement(Element):
    """Stand of trees of one successional group, measured by its stem volume.

    Args:
        label:
            The label of the stand.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        volume:
            The initial stem volume.

        mature:
            The stem volume of the group in a mature forest.

    Examples:
        >>> from skcomplex.models.biology.ecology import StandElement
        >>> StandElement('keora', mature=196.11).mature
        196.11
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        volume: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        mature: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 100.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.volume = volume
        self.mature = mature


class PioneerGrowthInteraction(Interaction):
    """Michaelis-Menten regrowth of pioneer trees, whose input from seeds the other groups suppress.

    Alone, the volume follows `V(t) = V_mat t / (K_M + t)` after a clear cut, which is
    `dV/dt = V (V / V_mat - 2) / K_M + C V_mat / K_M` with `C = 1`. The other groups reduce the factor
    `C = 1 - w_I V_I / V_mat,I - w_C V_C / V_mat,C`, with weights normalized to sum to one, and no lower than 0.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the pioneer stand.

        intermediate:
            The path of the intermediate stand.

        climax:
            The path of the climax stand.

        half_time:
            The time `K_M` the pioneers take to reach half their mature volume.

        w_intermediate:
            The weight of the intermediate group in the competition.

        w_climax:
            The weight of the climax group in the competition.

    Examples:
        >>> from skcomplex.models.biology.ecology import PioneerGrowthInteraction
        >>> PioneerGrowthInteraction('growth', target='keora', intermediate='gewa', climax='sundari').half_time
        3.65
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        intermediate: Annotated[str, RoleSpec()],
        climax: Annotated[str, RoleSpec()],
        half_time: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 3.65,
        w_intermediate: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.32,
        w_climax: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.68,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.intermediate = intermediate
        self.climax = climax
        self.half_time = half_time
        self.w_intermediate = w_intermediate
        self.w_climax = w_climax

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the regrowth with the suppressed input from seeds.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        volume, mature = s.target.volume, s.target.mature
        free = 1 - _crowding(s.intermediate, s.climax, s.w_intermediate, s.w_climax)
        suppression = where(free > 0, free, 0)
        return [RateLaw(volume, (volume * (volume / mature - 2) + suppression * mature) / s.half_time)]


class SuccessionalGrowthInteraction(Interaction):
    """Logistic growth of intermediate or climax trees, crowded by their own group and another, with seedling input.

    The rate is `dV/dt = a + g V (1 - w_1 V_1 / V_mat,1 - w_2 V_2 / V_mat,2)`, with weights normalized to sum to one.
    One of the two crowding stands is usually the target itself.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the growing stand.

        first:
            The path of the first crowding stand.

        second:
            The path of the second crowding stand.

        g:
            The specific growth rate.

        w_first:
            The weight of the first crowding stand.

        w_second:
            The weight of the second crowding stand.

        seedlings:
            The constant input `a` of seedlings that averts extinction.

    Examples:
        >>> from skcomplex.models.biology.ecology import SuccessionalGrowthInteraction
        >>> SuccessionalGrowthInteraction('growth', target='gewa', first='gewa', second='sundari', g=0.1).g
        0.1
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        first: Annotated[str, RoleSpec()],
        second: Annotated[str, RoleSpec()],
        g: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        w_first: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        w_second: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        seedlings: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.first = first
        self.second = second
        self.g = g
        self.w_first = w_first
        self.w_second = w_second
        self.seedlings = seedlings

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the crowded logistic growth with the seedling input.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        volume = s.target.volume
        crowding = _crowding(s.first, s.second, s.w_first, s.w_second)
        return [RateLaw(volume, s.seedlings + s.g * volume * (1 - crowding))]


@register_model(
    field='biology/ecology',
    tags=('succession', 'forest', 'ode', 'system dynamics', 'competition'),
    references=('Bala et al. 2017', 'Tietjen and Huth 2006'),
    defaults={'until': 300.0, 'dt': 1.0},
)
def mangrove_succession(
    keora: float = 0.0,
    gewa: float = 0.0,
    sundari: float = 0.0,
    *,
    g_gewa: float = 0.05,
    g_sundari: float = 0.03,
    seedlings: float = 0.01,
) -> System:
    """Return the regrowth of a mangrove forest of the Sundarbans after a clear cut, from pioneers to climax trees.

    Keora, the pioneer, regrows fastest, Gewa, the intermediate group, follows, and Sundari, the climax group, grows
    slowest but comes to dominate the mature forest and suppresses the others. Volumes, mature volumes, and competition
    weights follow Bala et al. (2017), and the growth rates of Gewa and Sundari are illustrative.

    Args:
        keora:
            The initial stem volume of Keora, the pioneer.

        gewa:
            The initial stem volume of Gewa, the intermediate group.

        sundari:
            The initial stem volume of Sundari, the climax group.

        g_gewa:
            The specific growth rate of Gewa.

        g_sundari:
            The specific growth rate of Sundari.

        seedlings:
            The constant input of Gewa and Sundari seedlings.

    Returns:
        system:
            The model, with the stands `keora`, `gewa`, and `sundari`.

    Examples:
        >>> from skcomplex.models.biology.ecology import mangrove_succession
        >>> [element.mature for element in mangrove_succession().elements]
        [196.11, 80.0, 320.0]
    """
    return System(
        elements=[
            StandElement('keora', volume=keora, mature=196.11),
            StandElement('gewa', volume=gewa, mature=80.0),
            StandElement('sundari', volume=sundari, mature=320.0),
        ],
        interactions=[
            PioneerGrowthInteraction('keora_growth', target='keora', intermediate='gewa', climax='sundari'),
            SuccessionalGrowthInteraction(
                'gewa_growth',
                target='gewa',
                first='gewa',
                second='sundari',
                g=g_gewa,
                w_first=0.32,
                w_second=0.68,
                seedlings=seedlings,
            ),
            SuccessionalGrowthInteraction(
                'sundari_growth',
                target='sundari',
                first='gewa',
                second='sundari',
                g=g_sundari,
                w_first=0.01,
                w_second=0.99,
                seedlings=seedlings,
            ),
        ],
    )
