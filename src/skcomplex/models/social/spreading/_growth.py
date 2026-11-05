"""Growth models: new words and repeated words in a growing text."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ....core.elements import GroupElement
from ....core.system import System
from ..._catalog import register_model
from ._growing_word import GrowingWordElement
from ._writing import WritingInteraction


@register_model(
    field='social/spreading',
    tags=('agents', 'growth', 'power law', 'language'),
    references=('Simon 1955', 'Yule 1925', 'Thurner, Hanel, and Klimek 2018'),
    defaults={'until': 20},
)
def simon_yule(*, innovation: float = 0.1, words_per_step: int = 100) -> System:
    """Return the Simon model of a text that grows from one word, whose word counts follow a power law.

    The rank-frequency exponent is `1 / (1 - innovation)`, and the number of distinct words grows as `innovation`
    times the length of the text.

    Args:
        innovation:
            The probability that the next word is new.

        words_per_step:
            The number of words written per step.

    Returns:
        system:
            The model, with the group `words`.

    Examples:
        >>> from skcomplex.models.social.spreading import simon_yule
        >>> simon_yule().elements[0].prepare().size_
        1
    """
    words = GroupElement(
        'words',
        element=GrowingWordElement,
        size=1,
        values={'count': 1.0, 'innovation': innovation},
        variable=True,
    )
    return System(
        elements=[words],
        interactions=[WritingInteraction('writing', target='words', words_per_step=words_per_step)],
    )
