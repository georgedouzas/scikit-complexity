"""Writing of the Simon system, a number of words per step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np

from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.engine.solvers import SystemView
from ....core.specs import RoleSpec
from ._growing_word_view import GrowingWordView


class WritingInteraction(Interaction):
    """Writing of the Simon system, a number of words per step.

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
        >>> from skcomplex.systems.social.spreading import WritingInteraction
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

    def write(self, system: SystemView) -> None:
        """Write the words of one step.

        Args:
            system:
                The system.
        """
        words = system.agents(self.target, GrowingWordView)
        if not words:
            return
        innovation = words[0].innovation
        counts = [int(word.count) for word in words]
        tokens = np.repeat(np.arange(len(counts)), counts).tolist()
        for _ in range(self.words_per_step):
            index = len(counts) if system.rng.random() < innovation else tokens[int(system.rng.integers(len(tokens)))]
            if index == len(counts):
                counts.append(0)
            counts[index] += 1
            tokens.append(index)
        for word, count in zip(words, counts, strict=False):
            word.count = float(count)
        for count in counts[len(words) :]:
            system.create(self.target, count=float(count))
