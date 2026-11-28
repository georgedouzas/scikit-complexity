"""Stochastic process systems of urns with reinforcement."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import NN, vector

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, JumpLaw, UpdateLaw
from ....core.components.laws.utils import categorical, where
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.errors import ValidationError
from ....core.systems import BaseSystem, System
from ....core.types import Role

TIMES = ('continuous', 'discrete')
MIN_COLOURS = 2


class PolyaUrnElement(Element):
    """Urns of balls of several colours, counted by colour.

    Args:
        label:
            The label of the urns.

        balls:
            The initial number of balls of every colour, one row per urn for a group of urns.

        size:
            The number of urns, or `None` for one.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import PolyaUrnElement
        >>> PolyaUrnElement('urn', balls=(1.0, 1.0, 2.0)).balls
        (1.0, 1.0, 2.0)
    """

    def __init__(
        self,
        label: str,
        *,
        balls: Annotated[object, NN] = (1.0, 1.0),
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.balls = balls


class PolyaDrawInteraction(Interaction):
    """The draws of Pólya's urn, where a drawn ball returns with `added` extra balls of its colour.

    The draws of every urn happen at the times of a Poisson process of rate 1, so the number of draws by time `t` is a
    Poisson number of mean `t`. A draw picks a colour with probability proportional to its number of balls, so the
    sequence of draws is that of Pólya's urn in discrete time.

    Args:
        label:
            The label of the interaction.

        target:
            The urns whose balls the draws add to.

        added:
            The number of balls of the drawn colour added after every draw.

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
        """Set the draw of a ball of every colour as the laws.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        balls = context.target.balls
        total = sum(balls)
        self.laws_: list[BaseLaw] = [JumpLaw(count / total, {count: self.added}) for count in balls]


class PolyaStepInteraction(Interaction):
    """The draws of Pólya's urn in discrete time, one draw in every urn per step.

    A draw picks a colour with probability proportional to its number of balls and adds `added` balls of that colour.

    Args:
        label:
            The label of the interaction.

        target:
            The urns whose balls the draws add to.

        added:
            The number of balls of the drawn colour added after every draw.

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
        balls = context.target.balls
        drawn = categorical('colour', list(balls))
        added = [count + self.added * where(drawn == colour, 1, 0) for colour, count in enumerate(balls)]
        self.laws_: list[BaseLaw] = [UpdateLaw(balls, vector(added))]


class PolyaUrnSystem(BaseSystem):
    """Independent Pólya urns of balls of two or more colours.

    The shares of the colours converge to a random limit with a Dirichlet law.

    The limit of the shares follows `Dirichlet(balls / added)`, so the share of one colour follows a beta law, uniform
    for one ball of each of two colours and one added ball. In continuous time the draws of every urn happen at the
    times of a Poisson process of rate 1, so a frame is not a draw, and in discrete time every step draws one ball in
    every urn. The system has the state `urns.balls`, the number of balls of every colour, of shape
    `(n_urns, n_colours)`.

    Args:
        n_urns:
            The number of independent urns.

        balls:
            The initial number of balls of every colour, at least two colours.

        added:
            The number of balls of the drawn colour added after every draw.

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
            The time is unknown, or there are fewer than two colours.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import PolyaUrnSystem
        >>> PolyaUrnSystem(3).prepare().system_.elements[0].size
        3
        >>> trajectory = PolyaUrnSystem(3, balls=(1, 1, 2), time='discrete').prepare().simulate(2, random_state=0)
        >>> trajectory.get_values('urns.balls')[-1].sum(axis=1).tolist()
        [6.0, 6.0, 6.0]
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
        balls: npt.ArrayLike = (1, 1),
        added: int = 1,
        time: str = 'continuous',
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_urns = n_urns
        self.balls = balls
        self.added = added
        self.time = time

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the urns and their draws in the chosen time."""
        super()._prepare(context)
        if self.time not in TIMES:
            error_msg = f'Unknown time {self.time!r}. Use one of {", ".join(TIMES)}.'
            raise ValidationError(error_msg)
        balls = np.asarray(self.balls, dtype=float)
        if balls.ndim != 1 or balls.size < MIN_COLOURS:
            error_msg = f'The urns of {self.label!r} hold balls={self.balls!r}. Give the counts of two or more colours.'
            raise ValidationError(error_msg)
        draws = PolyaDrawInteraction if self.time == 'continuous' else PolyaStepInteraction
        self.system_ = System(
            self.label,
            elements=[PolyaUrnElement('urns', balls=np.tile(balls, (self.n_urns, 1)), size=self.n_urns)],
            interactions=[draws('draws', target='urns', added=self.added)],
        )
