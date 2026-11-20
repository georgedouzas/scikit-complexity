"""Mandelbrot's least-effort word frequencies, the minimum of cost over entropy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import exp

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element, VisualSpec
from ....core.components.interactions import Interaction
from ....core.components.laws import ConstraintLaw
from ....core.components.laws.utils import log_sum_exp
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role


class LanguageElement(Element):
    """Language with the exponent `lambda` of its word frequencies.

    Args:
        label:
            The label of the language.

        exponent:
            The initial guess of the exponent, from which it is solved.

    Examples:
        >>> from skcomplex.systems.social.language import LanguageElement
        >>> LanguageElement('language').exponent
        1.0
    """

    def __init__(self, label: str, exponent: float = 1.0) -> None:
        super().__init__(label)
        self.exponent = exponent


class WordElement(Element):
    """Word with a quenched cost and the probability of its use.

    Args:
        label:
            The label of the word or the group.

        cost:
            The cost of the word.

        probability:
            The initial guess of the probability, which the equation replaces.

        size:
            The number of words, or `None` for one.

    Examples:
        >>> from skcomplex.systems.social.language import WordElement
        >>> WordElement('word', cost=1.0).cost
        1.0
    """

    visual = VisualSpec('histograms', {'color_by': 'probability'})

    def __init__(
        self,
        label: str,
        cost: object = 1.0,
        probability: object = 0.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.cost = cost
        self.probability = probability


class LeastEffortInteraction(Interaction):
    """Probabilities of words that minimise the ratio of the mean cost to the entropy.

    The mean cost is `C` and the entropy `H`. The probabilities that are stationary for `C / H` on the simplex are
    `p_i = exp(-lambda c_i) / Z`, for which `H = lambda C + ln Z`. The ratio is stationary in `lambda` where
    `H = lambda C`, that is where `ln Z = 0`, which solves the exponent. The sum reduces in log space.

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

    law_classes = (ConstraintLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        words: Role,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.words = words

    def _prepare(self, s: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the equations of the exponent and of the probabilities."""
        super()._prepare(s)
        if isinstance(s, SimpleNamespace):
            weight = -s.target.exponent * s.words.cost
            log_partition = log_sum_exp(weight, over=s.words)
            self.laws_: list[BaseComponent] = [
                ConstraintLaw(log_partition == 0, target=s.target.exponent),
                ConstraintLaw(s.words.probability == exp(weight - log_partition), target=s.words.probability),
            ]


class MandelbrotLeastEffortSystem(BaseSystem):
    """The word frequencies that minimise the mean cost of a language over its entropy.

    They are a power law of rank. A word of rank `i` costs `log(i + 1)`, the length of its code, so the probabilities
    are `(i + 1)^-lambda`, where `lambda` makes them sum to 1 (Giordano 2024 §6, after Mandelbrot 1953). The system has
    the states `language.exponent`, `words.cost`, and `words.probability`.

    Args:
        n_words:
            The number of words.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.social.language import MandelbrotLeastEffortSystem
        >>> trajectory = MandelbrotLeastEffortSystem(50).prepare().simulate(until=1.0)
        >>> round(float(trajectory.get_values('words.probability')[0].sum()), 6)
        1.0
    """

    field = 'social/language'
    topics = ('power law', 'least effort', 'entropy', 'static')
    references = ('Giordano 2024, arXiv 2407.02181', 'Mandelbrot 1953')

    def __init__(
        self,
        n_words: int = 200,
        *,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_words = n_words

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components built from the model parameters."""
        super()._prepare(context)
        costs = np.log(np.arange(1, self.n_words + 1) + 1.0)
        words = WordElement('words', cost=costs, size=self.n_words)
        effort = LeastEffortInteraction('effort', target='language', words='words')
        self.system_ = System(
            self.label,
            elements=[words, LanguageElement('language')],
            interactions=[effort],
        )
