"""Typewriting systems: Miller's monkeys.

Their random keystrokes give a power law of word frequencies.
"""

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


class TypedWordView(AgentView):
    """View of a typed word of the random typewriting system.

    Its states are attributes.
    """

    count: float
    length: float
    code: float


class TypedWordElement(Element):
    """Distinct word of a randomly typed text.

    It holds its number of occurrences and its number of letters.

    Args:
        label:
            The label of the word or the group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        count:
            The number of occurrences of the word.

        length:
            The number of letters of the word.

        code:
            The word as a number, its letters `k_1, ..., k_L` from 0 read as the digits `k_i + 1` of base `letters`.

    Examples:
        >>> from skcomplex.systems.social.language import TypedWordElement
        >>> TypedWordElement('word', length=3.0).length
        3.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        count: Annotated[float, StateSpec(domain=NN)] = 1.0,
        length: Annotated[float, StateSpec(domain=NN)] = 1.0,
        code: Annotated[float, StateSpec(domain=NN)] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.count = count
        self.length = length
        self.code = code

    @classmethod
    def _get_view(cls) -> type[TypedWordView]:
        """Return the class of the views of the agents, whose attributes are the states."""
        return TypedWordView


class TypingInteraction(Interaction):
    """Typing of Miller's monkeys.

    The keystrokes are random, on a typewriter with a number of letters and a space bar. Every keystroke hits the space
    bar with the probability `space_bar`, and otherwise one of the `letters` letters, uniformly. The space bar ends the
    word typed since the last space, a new distinct word or one more occurrence of an earlier one, and a space after a
    space types no word. A word is held as its code, its letters `k_1, ..., k_L` from 0 read as the digits `k_i + 1` of
    base `letters`, so distinct words have distinct codes.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the variable group of distinct words.

        letters:
            The number of letters of the typewriter.

        space_bar:
            The probability that a keystroke hits the space bar.

        keystrokes:
            The number of keystrokes per step.

    Examples:
        >>> from skcomplex.systems.social.language import TypingInteraction
        >>> TypingInteraction('typing', target='words').letters
        5
    """

    draws = True

    law_classes = (RuleLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        letters: Annotated[int, ParameterSpec(domain=RealSet.closed_open(1, oo))] = 5,
        space_bar: Annotated[float, ParameterSpec(domain=RealSet.open(0, 1))] = 0.2,
        keystrokes: Annotated[int, ParameterSpec(domain=NN)] = 1000,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.letters = letters
        self.space_bar = space_bar
        self.keystrokes = keystrokes

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the typing rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.type_keys)]

    def type_keys(self, system: SystemView) -> None:
        """Type the keystrokes of one step and count the words they end.

        Args:
            system:
                The system.
        """
        storage = system.storage
        if 'words' not in storage:
            storage['words'] = {int(word.code): word for word in system.agents(self.target, TypedWordView)}
        words, code, length = storage['words'], storage.get('code', 0), storage.get('length', 0)
        if not isinstance(words, dict) or not isinstance(code, int) or not isinstance(length, int):
            return
        spaces = system.rng.random(int(self.keystrokes)) < self.space_bar
        keys = system.rng.integers(int(self.letters), size=int(self.keystrokes))
        new: dict[tuple[int, int], int] = {}
        for is_space, key in zip(spaces.tolist(), keys.tolist(), strict=True):
            if not is_space:
                code, length = code * int(self.letters) + key + 1, length + 1
                continue
            if length:
                new[code, length] = new.get((code, length), 0) + 1
            code, length = 0, 0
        storage['code'], storage['length'] = code, length
        for (word_code, word_length), count in new.items():
            word = words.get(word_code)
            if isinstance(word, TypedWordView):
                word.count += count
                continue
            created = system.create(self.target, count=float(count), length=float(word_length), code=float(word_code))
            words[word_code] = created


class RandomTypewritingSystem(BaseSystem):
    """Miller's random typewriting.

    The random keystrokes of monkeys give Zipf's law of word frequencies. A typewriter has `letters` letters and a space
    bar hit with the probability `q = space`, and every space ends a word. A word of `L` letters has the probability
    `q ((1 - q) / letters)^L`, so its frequency falls exponentially with its length, while the number of distinct words
    of length `L` grows exponentially as `letters^L`. The two exponentials give a power law of the frequency of the word
    of rank `r`, `r^(-alpha)` with `alpha = 1 - log(1 - q) / log(letters)` (Thurner et al. 2018 section 3.3.6.2). The
    system has the variable group `words`, which starts with the first word of the text.

    Args:
        letters:
            The number of letters of the typewriter.

        space:
            The probability that a keystroke hits the space bar.

        keystrokes:
            The number of keystrokes per step.

        random_state:
            The seed or generator of the first word of the text.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.social.language import RandomTypewritingSystem
        >>> RandomTypewritingSystem().prepare().system_.elements[0].prepare().size_
        1
    """

    field = 'social/language'
    topics = ('agents', 'power law', 'zipf', 'language')
    references = ('Miller 1957', 'Thurner, Hanel, and Klimek 2018')
    until = 50

    def __init__(
        self,
        letters: int = 5,
        *,
        space: float = 0.2,
        keystrokes: int = 1000,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.letters = letters
        self.space = space
        self.keystrokes = keystrokes
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        rng = np.random.default_rng(self.random_state)
        code, length = 0, 0
        while length == 0 or rng.random() >= self.space:
            code, length = code * self.letters + int(rng.integers(self.letters)) + 1, length + 1
        words = GroupElement(
            'words',
            element=TypedWordElement,
            size=1,
            values={'count': 1.0, 'length': float(length), 'code': float(code)},
            variable=True,
        )
        typing = TypingInteraction(
            'typing',
            target='words',
            letters=self.letters,
            space_bar=self.space,
            keystrokes=self.keystrokes,
        )
        return System(self.label, elements=[words], interactions=[typing])
