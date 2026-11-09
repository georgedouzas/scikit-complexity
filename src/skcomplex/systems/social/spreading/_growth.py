"""Growth systems: new words and repeated words in a growing text."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.components.systems import BaseSystem, System
from ....core.engine.solvers import AgentView, SystemView
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec


class GrowingWordView(AgentView):
    """View of a word of the Simon system.

    Its states and parameters are attributes.
    """

    count: float
    innovation: float


class GrowingWordElement(Element):
    """Distinct word of the Simon system.

    It holds the number of times it occurs in the text.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        count:
            The number of occurrences.

        innovation:
            The probability that the next word of the text is new.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.social.spreading import GrowingWordElement
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

    @classmethod
    def _get_view(cls) -> type[GrowingWordView]:
        """Return the class of the views of the agents.

        Their attributes are the states and parameters.
        """
        return GrowingWordView


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

    draws = True

    law_classes = (RuleLaw,)

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


class SimonYuleSystem(BaseSystem):
    """The Simon system of a text that grows from one word.

    Its word counts follow a power law. The rank-frequency exponent is `1 / (1 - innovation)`, and the number of
    distinct words grows as `innovation` times the length of the text. The system has the group `words`.

    Args:
        innovation:
            The probability that the next word is new.

        words_per_step:
            The number of words written per step.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.social.spreading import SimonYuleSystem
        >>> SimonYuleSystem().prepare().system_.elements[0].prepare().size_
        1
    """

    field = 'social/spreading'
    topics = ('agents', 'growth', 'power law', 'language')
    references = ('Simon 1955', 'Yule 1925', 'Thurner, Hanel, and Klimek 2018')
    until = 20

    def __init__(
        self,
        *,
        innovation: float = 0.1,
        words_per_step: int = 100,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.innovation = innovation
        self.words_per_step = words_per_step

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        words = GroupElement(
            'words',
            element=GrowingWordElement,
            size=1,
            values={'count': 1.0, 'innovation': self.innovation},
            variable=True,
        )
        return System(
            self.label,
            elements=[words],
            interactions=[WritingInteraction('writing', target='words', words_per_step=self.words_per_step)],
        )
