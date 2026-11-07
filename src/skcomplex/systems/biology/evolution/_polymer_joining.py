"""Joining of two polymers of an assembly pool into a new one, at random or by reuse of the newest polymer."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.engine.solvers import SystemView
from ....core.errors import ValidationError
from ....core.specs import RoleSpec
from ._polymer_view import PolymerView

MODES = ('undirected', 'directed')


class PolymerJoiningInteraction(Interaction):
    """Joining of two polymers of an assembly pool into a new one, at random or by reuse of the newest polymer.

    At every step, undirected exploration joins two polymers drawn at random from the pool, and directed exploration
    joins the polymer made most recently with one drawn at random. The joined polymer, of the summed length, is added
    to the pool, and its parts stay in it.

    Args:
        label:
            The label of the interaction.

        polymers:
            The path of the variable group of polymers.

        mode:
            `'undirected'` or `'directed'`.

    Raises:
        ValidationError:
            The mode is unknown.

    Examples:
        >>> from skcomplex.systems.biology.evolution import PolymerJoiningInteraction
        >>> PolymerJoiningInteraction('joining', polymers='pool', mode='directed').mode
        'directed'
    """

    def __init__(self, label: str, *, polymers: Annotated[str, RoleSpec()], mode: str = 'undirected') -> None:
        super().__init__(label)
        self.polymers = polymers
        self.mode = mode

    def _check(self) -> None:
        """Check the mode.

        Raises:
            ValidationError:
                The mode is unknown.
        """
        super()._check()
        if self.mode not in MODES:
            error_msg = f'The mode must be one of {", ".join(MODES)}, not {self.mode!r}.'
            raise ValidationError(error_msg)

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the joining rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.join)]

    def join(self, system: SystemView) -> None:
        """Join two polymers and add the product to the pool.

        Args:
            system:
                The system.
        """
        pool = system.agents(self.polymers, PolymerView)
        partner = pool[int(system.rng.integers(len(pool)))]
        if self.mode == 'directed':
            first = max(pool, key=lambda polymer: polymer.born)
        else:
            first = pool[int(system.rng.integers(len(pool)))]
        born = max(polymer.born for polymer in pool) + 1
        system.create(self.polymers, length=first.length + partner.length, born=born)
