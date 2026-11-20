"""Growth systems: new words and repeated words in a growing text."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, Protocol, cast

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.spaces import BaseSpace
from ....core.components.types import AgentViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, SystemView
from ....core.systems import BaseSystem, System
from ....core.types import Role


class GrowingWordView(AgentViewProtocol, Protocol):
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

        count:
            The number of occurrences, one per word or shared.

        innovation:
            The probability that the next word of the text is new.

        size:
            The number of words, or `None` for one.

    Examples:
        >>> from skcomplex.systems.social.spreading import GrowingWordElement
        >>> group = GrowingWordElement('words', count=1.0, size=1)
        >>> group.label, group.count, group.size
        ('words', 1.0, 1)
    """

    def __init__(
        self,
        label: str,
        *,
        count: Annotated[object, NN] = 1.0,
        innovation: Annotated[float, RealSet.closed_open(0, oo)] = 0.1,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.count = count
        self.innovation = innovation


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

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        words_per_step: int = 100,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.words_per_step = words_per_step

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the writing as the rule of the system."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseComponent] = [UpdateLaw(context, self.write)]

    def write(self, system: SystemView) -> None:
        """Write the words of one step.

        Args:
            system:
                The system.
        """
        words = cast(list[GrowingWordView], system.get_agents('target'))
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
            system.create('target', count=float(count))


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.social.spreading import SimonYuleSystem
        >>> SimonYuleSystem().prepare().system_.elements[0].size
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.innovation = innovation
        self.words_per_step = words_per_step

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components built from the model parameters."""
        super()._prepare(context)
        words = GrowingWordElement('words', count=1.0, innovation=self.innovation, size=1)
        self.system_ = System(
            self.label,
            elements=[words],
            interactions=[WritingInteraction('writing', target='words', words_per_step=self.words_per_step)],
        )
