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
from ....core.components.laws import BaseLaw, JumpLaw, UpdateLaw
from ....core.components.laws.utils import bernoulli
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.errors import ValidationError
from ....core.systems import BaseSystem, System
from ....core.types import Role

TIMES = ('continuous', 'discrete')


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

    The draws of every urn happen at the times of a Poisson process of rate 1, so the number of draws by time `t` is a
    Poisson number of mean `t`. A draw picks a color with probability proportional to its number of balls, so the
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


class PolyaStepInteraction(Interaction):
    """The draws of Pólya's urn in discrete time, one draw in every urn per step.

    A draw picks a color with probability proportional to its number of balls and adds `added` balls of that color.

    Args:
        label:
            The label of the interaction.

        target:
            The urns whose balls the draws add to.

        added:
            The number of balls of the drawn color added after every draw.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import PolyaStepInteraction
        >>> PolyaStepInteraction('draws', target='urns', added=2).added
        2
    """

    law_classes = (UpdateLaw,)

    def __init__(self, label: str, *, target: Role = 'urns', added: int = 1) -> None:
        super().__init__(label)
        self.target = target
        self.added = added

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the draw of one ball in every urn as the rule.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        red, blue = context.target.red, context.target.blue
        drawn = bernoulli('red_drawn', red / (red + blue))
        self.laws_: list[BaseLaw] = [
            UpdateLaw(red, red + self.added * drawn),
            UpdateLaw(blue, blue + self.added * (1 - drawn)),
        ]


class PolyaUrnSystem(BaseSystem):
    """Independent Pólya urns.

    The share of red balls converges to a random limit with a beta distribution.

    The limit follows `Beta(red / added, blue / added)`, uniform for one ball of each color and one added ball. In
    continuous time the draws of every urn happen at the times of a Poisson process of rate 1, so a frame is not a
    draw, and in discrete time every step draws one ball in every urn. The system has the states `urns.red` and
    `urns.blue`.

    Args:
        n_urns:
            The number of independent urns.

        red:
            The initial number of red balls.

        blue:
            The initial number of blue balls.

        added:
            The number of balls of the drawn color added after every draw.

        time:
            The time of the draws, `'continuous'` for draws at the times of a Poisson process of rate 1 or
            `'discrete'` for one draw per step.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ValidationError:
            The time is unknown.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import PolyaUrnSystem
        >>> PolyaUrnSystem(3).prepare().system_.elements[0].size
        3
        >>> trajectory = PolyaUrnSystem(3, time='discrete').prepare().simulate(2, random_state=0)
        >>> (trajectory.get_values('urns.red')[-1] + trajectory.get_values('urns.blue')[-1]).tolist()
        [4.0, 4.0, 4.0]
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
        time: str = 'continuous',
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_urns = n_urns
        self.red = red
        self.blue = blue
        self.added = added
        self.time = time

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the urns and their draws in the chosen time."""
        super()._prepare(context)
        if self.time not in TIMES:
            error_msg = f'Unknown time {self.time!r}. Use one of {", ".join(TIMES)}.'
            raise ValidationError(error_msg)
        draws = PolyaDrawInteraction if self.time == 'continuous' else PolyaStepInteraction
        self.system_ = System(
            self.label,
            elements=[PolyaUrnElement('urns', red=float(self.red), blue=float(self.blue), size=self.n_urns)],
            interactions=[draws('draws', target='urns', added=self.added)],
        )
