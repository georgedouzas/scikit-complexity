"""Growth models: new words and repeated words in a growing text."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.interactions import Interaction
from ....core.laws import BaseLaw, RuleLaw
from ....core.system import System
from ....engine.solvers import ModelView
from ....specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_model


class GrowingWordElement(Element):
    """Distinct word of the Simon model, with the number of times it occurs in the text.

    Args:
        label:
            The label of the agent or group.

        location:
            The initial location, unused without a space.

        count:
            The number of occurrences.

        innovation:
            The probability that the next word of the text is new.

    Examples:
        >>> from skcomplex.models.social.spreading import GrowingWordElement
        >>> group = GroupElement('words', element=GrowingWordElement, size=1, values={'count': 1.0})
        >>> group.label, group.element.__name__, group.size
        ('words', 'GrowingWordElement', 1)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        count: Annotated[float, StateSpec(domain=NN)] = 1.0,
        innovation: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.count = count
        self.innovation = innovation


class WritingInteraction(Interaction):
    """Writing of the Simon model, a number of words per step.

    Every next word is new with probability `innovation`, and otherwise repeats a random earlier occurrence, so a word
    is repeated with probability proportional to its count.

    Args:
        label:
            The label of the interaction.

        words_per_step:
            The number of words written per step.

        **values:
            The values of the declared variables, by name:

            target:
                The path of the group of words.

    Examples:
        >>> from skcomplex.models.social.spreading import WritingInteraction
        >>> WritingInteraction('writing', target='words').words_per_step
        100
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        words_per_step: int = 100,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.words_per_step = words_per_step

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the writing rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.write)]

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
        >>> simon_yule().elements[0].check().prepare().size_
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
