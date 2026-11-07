"""Language: the statistics of words from the economy of their use."""

from ._typewriting import TypedWordElement, TypedWordView, TypingInteraction, random_typewriting
from ._word_frequencies import LanguageElement, LeastEffortInteraction, WordElement, mandelbrot_least_effort

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
