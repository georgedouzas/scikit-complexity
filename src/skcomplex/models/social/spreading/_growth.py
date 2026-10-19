"""Growth models: new words and repeated words in a growing text."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ....core.base import Domain, parameter, role, state
from ....core.model import AgentType, Interaction, Law, Rule, Symbols, System
from ....core.runtime import register_model
from ....core.solvers import ModelView


class Word(AgentType):
    """Distinct word of the Simon model, with the number of times it occurs in the text.

    Args:
        label:
            The label of the agent type or group.

        location:
            The initial location, unused without a space.

        count:
            The number of occurrences.

        innovation:
            The probability that the next word of the text is new.

    Examples:
        >>> from skcomplex.models.social.spreading import Word
        >>> Word.group('words', 1, count=1.0)
        Group('words', element=Word, size=1)
    """

    count: float = state(default=1.0, domain=Domain.NONNEGATIVE_INTEGER)
    innovation: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)


class Writing(Interaction):
    """Writing of the Simon model, a number of words per step.

    Every next word is new with probability `innovation`, and otherwise repeats a random earlier occurrence, so a word
    is repeated with probability proportional to its count.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of words.

        words_per_step:
            The number of words written per step.

    Examples:
        >>> from skcomplex.models.social.spreading import Writing
        >>> Writing('writing', target='words').words_per_step
        100
    """

    target: str = role()
    words_per_step: int = 100

    def laws(self, s: Symbols) -> list[Law]:
        """Return the writing rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.write)]

    def write(self, model: ModelView) -> None:
        """Write the words of one step.

        Args:
            model:
                The model.
        """
        words = model.agents(self.target)
        if not words:
            return
        innovation = words[0].innovation
        counts = [int(word.count) for word in words]
        tokens = np.repeat(np.arange(len(counts)), counts).tolist()
        for _ in range(self.words_per_step):
            index = len(counts) if model.rng.random() < innovation else tokens[int(model.rng.integers(len(tokens)))]
            if index == len(counts):
                counts.append(0)
            counts[index] += 1
            tokens.append(index)
        for word, count in zip(words, counts, strict=False):
            word.count = float(count)
        for count in counts[len(words) :]:
            model.create(self.target, count=float(count))


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
        >>> simon_yule().elements[0].size
        1
    """
    words = Word.group('words', 1, count=1.0, innovation=innovation)
    return System(elements=[words], interactions=[Writing('writing', target='words', words_per_step=words_per_step)])
