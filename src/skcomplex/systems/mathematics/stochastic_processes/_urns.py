"""Stochastic process systems of urns with reinforcement."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import NN

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, JumpLaw
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role


class PolyaUrnElement(Element):
    """Urns of red and blue balls, counted by color.

    Args:
        label:
            The label of the urns.

        red:
            The initial number of red balls, one per urn or shared.

        blue:
            The initial number of blue balls, one per urn or shared.

        size:
            The number of urns, or `None` for one.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import PolyaUrnElement
        >>> PolyaUrnElement('urn', red=2.0).red
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        red: Annotated[float, NN] = 1.0,
        blue: Annotated[float, NN] = 1.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.red = red
        self.blue = blue


class PolyaDrawInteraction(Interaction):
    """The draws of Pólya's urn, where a drawn ball returns with `added` extra balls of its color.

    The draws happen at rate 1, and a draw picks a color with probability proportional to its number of balls, so the
    sequence of draws is that of Pólya's urn in discrete time.

    Args:
        label:
            The label of the interaction.

        target:
            The urns whose balls the draws add to.

        added:
            The number of balls of the drawn color added after every draw.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import PolyaDrawInteraction
        >>> PolyaDrawInteraction('draws', target='urns', added=2).added
        2
    """

    law_classes = (JumpLaw,)

    def __init__(self, label: str, *, target: Role = 'urns', added: int = 1) -> None:
        super().__init__(label)
        self.target = target
        self.added = added

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the draws of a red and of a blue ball as the laws.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        red, blue = context.target.red, context.target.blue
        self.laws_: list[BaseLaw] = [
            JumpLaw(red / (red + blue), {red: self.added}),
            JumpLaw(blue / (red + blue), {blue: self.added}),
        ]


class PolyaUrnSystem(BaseSystem):
    """Independent Pólya urns.

    The share of red balls converges to a random limit with a beta distribution.

    The limit follows `Beta(red / added, blue / added)`, uniform for one ball of each color and one added ball. The
    system has the states `urns.red` and `urns.blue`.

    Args:
        n_urns:
            The number of independent urns.

        red:
            The initial number of red balls.

        blue:
            The initial number of blue balls.

        added:
            The number of balls of the drawn color added after every draw.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import PolyaUrnSystem
        >>> PolyaUrnSystem(3).prepare().system_.elements[0].size
        3
    """

    field = 'mathematics/stochastic_processes'
    topics = ('urn', 'reinforcement', 'jump process', 'history dependence')
    references = ('Eggenberger and Pólya 1923', 'Thurner et al. 2018')
    until = 100.0
    dt = 1.0

    def __init__(
        self,
        n_urns: int = 10,
        *,
        red: int = 1,
        blue: int = 1,
        added: int = 1,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_urns = n_urns
        self.red = red
        self.blue = blue
        self.added = added

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the urns and their draws."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[PolyaUrnElement('urns', red=float(self.red), blue=float(self.blue), size=self.n_urns)],
            interactions=[PolyaDrawInteraction('draws', target='urns', added=self.added)],
        )
