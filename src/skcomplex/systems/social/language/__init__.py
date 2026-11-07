"""Language: the statistics of words from the economy of their use."""

from ._language import LanguageElement
from ._least_effort import LeastEffortInteraction
from ._word import WordElement
from ._typed_word_view import TypedWordView
from ._typewriting import TypedWordElement, TypingInteraction, random_typewriting
from ._word_frequencies import mandelbrot_least_effort

__all__: list[str] = [
    'LanguageElement',
    'LeastEffortInteraction',
    'TypedWordElement',
    'TypedWordView',
    'TypingInteraction',
    'WordElement',
    'mandelbrot_least_effort',
    'random_typewriting',
]
