"""Define the sampling of sequences.

A sequence is a row of a state on a lattice, and the sampling accepts moves from the change of the score of the row.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR

from .....core.components.interactions import SamplingInteraction
from .....core.components.laws import ArrayUpdateLaw, BaseLaw
from .....core.errors import ValidationError
from .....core.specs import BaseSpec
from .....core.types import Array


@dataclass(frozen=True)
class RowScoreSpec(BaseSpec):
    """Spec of an objective that scores a whole row of a lattice state.

    The row is the values along the last axis.

    Args:
        function:
            The function `function(row, **parameters)` of the values of a row and the parameters of the system that it
            names, which returns one number.

        proposals:
            The number of moves proposed per row and unit of time.

    Raises:
        ValidationError:
            The function is not callable, or the number of proposals is not positive.

    Examples:
        >>> from skcomplex.systems.biology.common.sequences import RowScoreSpec
        >>> RowScoreSpec(sum, proposals=2).proposals
        2
    """

    function: Callable[..., float]
    proposals: int = 1

    def _check(self) -> None:
        """Raise an error for a bad function or number of proposals.

        The function must be callable and the number of proposals positive.

        Raises:
            ValidationError:
                The function is not callable, or the number of proposals is not positive.
        """
        if not callable(self.function):
            error_msg = f'The function of a row score must be callable, but it is {self.function!r}.'
            raise ValidationError(error_msg)
        if self.proposals < 1:
            error_msg = f'A row score proposes at least one move per row, but it has proposals={self.proposals}.'
            raise ValidationError(error_msg)


class SequenceSamplingInteraction(SamplingInteraction):
    """Monte Carlo sampling of whole rows of a state on a `LatticeSpace`, scored row by row.

    The objective is a `RowScoreSpec` that scores whole rows along the last axis. Every unit of time proposes its
    `proposals` moves per row, each accepted from the change of the score of the row, and a run records the score of
    every row as `<label>.score`. A named move changes one random site of the row, a function move receives the row,
    the heat-bath rule draws the value of one random site among `values` by the weights `exp(-intensity * score)`, and
    the schedule and the neighbourhood do not apply. An infinite intensity accepts a move that keeps the objective with
    probability 1 under the Metropolis rule and 1/2 under the Glauber rule, and it never accepts a move that raises
    the objective. The arguments are those of `SamplingInteraction`.

    Examples:
        >>> from typing import Annotated
        >>> import numpy as np
        >>> from skcomplex.core.components.elements import Element, GroupElement
        >>> from skcomplex.core.components.spaces import DiscreteTime, LatticeSpace
        >>> from skcomplex.core.components.systems import CompositeSystem
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.core.specs import StateSpec
        >>> from skcomplex.systems.biology.common.sequences import RowScoreSpec, SequenceSamplingInteraction
        >>> class SpinElement(Element):
        ...     def __init__(self, label, s: Annotated[float, StateSpec()] = 1.0):
        ...         super().__init__(label)
        ...         self.s = s
        >>> def flip_one(row, random):
        ...     row[random.integers(row.size)] *= -1
        ...     return row
        >>> walk = SequenceSamplingInteraction(
        ...     'walk', target='spins', state='s', objective=RowScoreSpec(np.sum), move=flip_one, intensity=float('inf')
        ... )
        >>> spins = GroupElement('spins', element=SpinElement, space=LatticeSpace((2, 8)), on='sites')
        >>> system = CompositeSystem(elements=[spins], interactions=[walk], time=DiscreteTime())
        >>> simulate(system, until=3, random_state=0).values('walk.score').shape
        (4, 2)
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (ArrayUpdateLaw,)

    def _check_objective(self, objective: object) -> None:
        """Raise an error for an objective that is not a row score, or a row score the sampling cannot run."""
        if not isinstance(objective, RowScoreSpec):
            error_msg = f'The objective of sequence sampling {self.label!r} must be a RowScoreSpec.'
            raise ValidationError(error_msg)
        self._check_row_score(objective)

    def _check_row_score(self, score: RowScoreSpec) -> None:
        """Raise an error for a bad row score, a cluster move, or a heat-bath rule.

        It raises it for a row score that cannot run, or a heat-bath rule without values.
        """
        score._check()
        if self.move == 'cluster':
            error_msg = f'The row score of sampling {self.label!r} accepts single moves of a row, not cluster moves.'
            raise ValidationError(error_msg)
        if self.rule == 'heat-bath' and (self.move != 'random-value' or not len(self.values)):
            error_msg = (
                f'The row score of sampling {self.label!r} draws by heat bath among values. Give the move '
                "'random-value' and its `values`."
            )
            raise ValidationError(error_msg)

    def _get_row_keywords(self, score: RowScoreSpec, index: int, parameters: Mapping[str, object]) -> dict[str, object]:
        """Return the parameters that a row score names.

        They are those of one value per site read at the row.
        """
        try:
            names = inspect.signature(score.function).parameters
        except (TypeError, ValueError):
            return {}
        return {
            name: value[index] if isinstance(value, np.ndarray) and value.ndim > 1 else value
            for name, value in parameters.items()
            if name in names
        }

    def _score_rows(self, values: Array, **parameters: object) -> npt.NDArray[np.float64]:
        """Return the score of every row of a state."""
        score = getattr(self, self._get_parameter('objective'))
        rows = values.reshape(-1, values.shape[-1])
        return np.array(
            [
                float(score.function(row, **self._get_row_keywords(score, index, parameters)))
                for index, row in enumerate(rows)
            ],
        )

    def _draw_row_value(
        self,
        row: Array,
        score: RowScoreSpec,
        keywords: Mapping[str, object],
        intensity: float,
        rng: np.random.Generator,
    ) -> Array:
        """Return a row with the value of one random site drawn among all values.

        The value is drawn by the weights of their scores.
        """
        candidates = np.asarray(self.values, dtype=float)
        proposal = row.copy()
        site = rng.integers(row.size)
        scores = np.empty(candidates.size)
        for position, value in enumerate(candidates):
            proposal[site] = value
            scores[position] = float(score.function(proposal, **keywords))
        excess = scores - scores.min()
        weights = np.exp(-np.where(excess == 0.0, 0.0, excess * np.where(excess == 0.0, 1.0, intensity)))
        proposal[site] = rng.choice(candidates, p=weights / weights.sum())
        return proposal

    def _propose_row(self, row: Array, rng: np.random.Generator) -> Array:
        """Return a row after one move.

        The move is a function of the row, or a named move at one random site.
        """
        if callable(self.move):
            return np.asarray(self.move(row.copy(), rng), dtype=float)
        proposal = row.copy()
        site = rng.integers(row.size)
        if self.move == 'flip':
            proposal[site] = -row[site]
        elif self.move == 'random-value':
            proposal[site] = rng.choice(np.asarray(self.values, dtype=float))
        elif self.move == 'gaussian':
            proposal[site] = row[site] + rng.normal(0.0, self.width)
        else:
            proposal[site] = row[site] + rng.uniform(-self.width, self.width)
        return proposal

    def _accept_change(self, change: float, intensity: float, rng: np.random.Generator) -> bool:
        """Return whether a change of the score is accepted by the rule at the intensity."""
        exponent = 0.0 if change == 0.0 else change * intensity
        if self.rule == 'metropolis':
            return exponent <= 0.0 or bool(rng.random() < np.exp(-exponent))
        with np.errstate(over='ignore'):
            return bool(rng.random() < 1.0 / (1.0 + np.exp(exponent)))

    def _accept_rows(self, values: Array, *, random: np.random.Generator, time: float, **parameters: object) -> Array:
        """Return a state after the proposed moves of every row.

        Each move is accepted from the change of the row's score.
        """
        score = getattr(self, self._get_parameter('objective'))
        intensity = float(SR(self._get_intensity(SimpleNamespace(**parameters))))
        if self.cooling is not None:
            intensity /= float(SR(self.cooling(time)))
        rows = values.reshape(-1, values.shape[-1]).copy()
        for index in range(len(rows)):
            keywords = self._get_row_keywords(score, index, parameters)
            if self.rule == 'heat-bath':
                for _ in range(score.proposals):
                    rows[index] = self._draw_row_value(rows[index], score, keywords, intensity, random)
                continue
            current = float(score.function(rows[index], **keywords))
            for _ in range(score.proposals):
                proposal = self._propose_row(rows[index], random)
                proposed = float(score.function(proposal, **keywords))
                if self._accept_change(proposed - current, intensity, random):
                    rows[index], current = proposal, proposed
        return rows.reshape(values.shape)

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the law that samples the state by the score.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with `namespace.target`.

        Returns:
            laws:
                The law of the sampling.
        """
        target = getattr(self._get_member(namespace, 'target'), self.state)
        return [ArrayUpdateLaw(target, self._accept_rows, label=self.label, score=self._score_rows)]
