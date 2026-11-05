"""Mandelbrot's least-effort word frequencies, the minimum of cost over entropy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ....core.elements import GroupElement
from ....core.spaces import StaticTime
from ....core.system import System
from ....specs import VisualSpec
from ..._catalog import register_model
from ._language import LanguageElement
from ._least_effort import LeastEffortInteraction
from ._word import WordElement


@register_model(
    field='social/language',
    tags=('power law', 'least effort', 'entropy', 'static'),
    references=('Giordano 2024, arXiv 2407.02181', 'Mandelbrot 1953'),
)
def mandelbrot_least_effort(n_words: int = 200) -> System:
    """Return the word frequencies that minimise the mean cost of a language over its entropy, a power law of rank.

    A word of rank `i` costs `log(i + 1)`, the length of its code, so the probabilities are `(i + 1)^-lambda`, where
    `lambda` makes them sum to 1 (Giordano 2024 §6, after Mandelbrot 1953).

    Args:
        n_words:
            The number of words.

    Returns:
        system:
            The model, with the states `language.exponent`, `words.cost`, and `words.probability`.

    Examples:
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.models.social.language import mandelbrot_least_effort
        >>> trajectory = Simulation(mandelbrot_least_effort(50)).prepare().run(1.0).trajectory_
        >>> round(float(trajectory.values('words.probability')[0].sum()), 6)
        1.0
    """
    costs = np.log(np.arange(1, n_words + 1) + 1.0)
    visual = VisualSpec('histograms', {'color_by': 'probability'})
    words = GroupElement('words', element=WordElement, size=n_words, values={'cost': costs}, visual=visual)
    effort = LeastEffortInteraction('effort', target='language', words='words')
    return System(elements=[words, LanguageElement('language')], interactions=[effort], time=StaticTime())
