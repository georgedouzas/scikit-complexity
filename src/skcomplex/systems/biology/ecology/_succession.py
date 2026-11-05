"""Ecology systems: succession of pioneer, intermediate, and climax tree species."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ....core.components.system import System
from ..._catalog import register_system
from ._pioneer_growth import PioneerGrowthInteraction
from ._stand import StandElement
from ._successional_growth import SuccessionalGrowthInteraction


@register_system(
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
            The system, with the stands `keora`, `gewa`, and `sundari`.

    Examples:
        >>> from skcomplex.systems.biology.ecology import mangrove_succession
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
