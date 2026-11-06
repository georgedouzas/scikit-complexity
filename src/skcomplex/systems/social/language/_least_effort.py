"""The word probabilities that minimise the cost of a language over its entropy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import exp

from ....core.components.base import BasePart
from ....core.components.interactions import Interaction
from ....core.components.laws import EquationLaw
from ....core.components.laws.utils import log_sum_exp
from ....core.specs import RoleSpec


class LeastEffortInteraction(Interaction):
    """Probabilities of words that minimise the ratio of the mean cost `C` to the entropy `H`.

    The probabilities that are stationary for `C / H` on the simplex are `p_i = exp(-lambda c_i) / Z`, for which
    `H = lambda C + ln Z`. The ratio is stationary in `lambda` where `H = lambda C`, that is where `ln Z = 0`, which
    solves the exponent. The sum reduces in log space.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the element that holds `exponent`.

        words:
            The path of the group of words, which hold `cost` and `probability`.

    Examples:
        >>> from skcomplex.systems.social.language import LeastEffortInteraction
        >>> LeastEffortInteraction('effort', target='language', words='words').words
        'words'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        words: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.target = target
        self.words = words

    def _get_laws(self, s: SimpleNamespace) -> Sequence[BasePart]:
        """Return the equations of the exponent and of the probabilities.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The stationarity of the ratio and the probability of a word.
        """
        weight = -s.target.exponent * s.words.cost
        log_partition = log_sum_exp(weight, over=s.words)
        return [
            EquationLaw(log_partition == 0, target=s.target.exponent),
            EquationLaw(s.words.probability == exp(weight - log_partition), target=s.words.probability),
        ]
