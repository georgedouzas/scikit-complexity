"""Language: the statistics of words from the economy of their use."""

from ._typewriting import RandomTypewritingSystem, TypedWordElement, TypedWordView, TypingInteraction
from ._word_frequencies import LanguageElement, LeastEffortInteraction, MandelbrotLeastEffortSystem, WordElement

__all__: list[str] = [
    'LanguageElement',
    'LeastEffortInteraction',
    'MandelbrotLeastEffortSystem',
    'RandomTypewritingSystem',
    'TypedWordElement',
    'TypedWordView',
    'TypingInteraction',
    'WordElement',
]
