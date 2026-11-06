"""Language: the statistics of words from the economy of their use."""

from ._language import LanguageElement
from ._least_effort import LeastEffortInteraction
from ._word import WordElement
from ._word_frequencies import mandelbrot_least_effort

__all__: list[str] = [
    'LanguageElement',
    'LeastEffortInteraction',
    'WordElement',
    'mandelbrot_least_effort',
]
