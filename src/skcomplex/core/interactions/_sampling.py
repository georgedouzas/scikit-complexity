"""Define Monte Carlo sampling of a state, by moves accepted from the change of a local objective."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import inspect
from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import RR, SR, ConditionSet, Expression

from ...errors import ValidationError
from ...specs import MoveSpec, ObjectiveSpec, ParameterSpec, RoleSpec, RowScoreSpec, SamplerSpec
from ...types import Array
from ..laws import BaseLaw, DriveLaw, SampleLaw
from ..spaces import BaseNeighborhood, VonNeumannNeighborhood
from ._interaction import Interaction

MOVES = ('flip', 'random-value', 'rotate', 'gaussian', 'cluster')


def _is_positive(value: float) -> bool:
    """Return whether a value is above zero."""
    return bool(value > 0)


POSITIVE_VALUES = ConditionSet(RR, _is_positive)


class SamplingInteraction(Interaction):
    """Monte Carlo sweeps of a state of the members of a lattice or network group, one sweep per unit of time.

    Every sweep proposes a move of the state of every member and accepts it from the change of the local objective of
    the member, the terms of the objective that involve its value, at the intensity. The local objective is a function
    of the namespace, which holds `namespace.target`, the states and parameters of the member, and
    `namespace.target.other`, a neighbour, whose values `neighbors(...)` sums over `neighborhood`. A move is `'flip'`,
    the negative value, `'random-value'`, a value drawn from `values`, `'rotate'`, the value plus a uniform number in
    `[-width, width]`, `'gaussian'`, the value plus a normal number of standard deviation `width`, or a function
    `move(values, random)` of the array of the current values of some members and the random generator that returns
    their proposed values. The move `'cluster'` is the Wolff algorithm instead: a sweep flips clusters until the flipped
    members number as many as the members, each cluster grown from a random member through the bonds to neighbours
    with probability `1 - exp(min(0, -intensity * (f(-s, n) - f(s, n))))`, where `f(s, n)` is the term of the local
    objective of a member of value `s` with a neighbour of value `n`. It takes an objective of such pair terms that the
    flip of both values keeps, and ignores the rule and the schedule. A group on the edges of a cell complex takes a
    local objective that sums over `CofaceNeighborhood`, inside which `exterior_derivative` of the state is the oriented
    sum around the one face, and its `'colored'` schedule updates together only links that share no face.

    An objective that is a `RowScoreSpec` scores whole rows of a state on a `LatticeSpace` instead, along the last axis.
    Every unit of time proposes its `proposals` moves per row, each accepted from the change of the score of the row,
    and a run records the score of every row as `<label>.score`. A named move changes one random site of the row, a
    function move receives the row, the heat-bath rule draws the value of one random site among `values` by the weights
    `exp(-intensity * score)`, and the schedule and the neighbourhood do not apply. An infinite intensity accepts
    a move that keeps the objective with probability 1 under the Metropolis rule and 1/2 under the Glauber rule, and it
    never accepts a move that raises the objective.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group on the sites of a lattice, the nodes of a network, or the edges of a cell complex.

        state:
            The name of the state the moves change.

        objective:
            The function `objective(namespace)` of the local objective of a member, or a `RowScoreSpec`.

        move:
            `'flip'`, `'random-value'`, `'rotate'`, `'gaussian'`, `'cluster'`, or a function that proposes values.

        intensity:
            The factor of the change of the objective in the acceptance `exp(-intensity * change)`, positive and
            possibly infinite.

        rule:
            The acceptance, `'metropolis'`, `'glauber'`, or `'heat-bath'`, which draws among all values of a discrete
            move by the weights `exp(-intensity * objective)`.

        schedule:
            `'checkerboard'`, `'colored'`, or `'random-sequential'`.

        neighborhood:
            The neighbourhood the objective reduces over, or `None` for the von Neumann neighbourhood of a square
            lattice. A network takes `AdjacencyNeighborhood()`.

        values:
            The values a random-value move draws from.

        width:
            The largest change of a rotation, or the standard deviation of a Gaussian move.

        cooling:
            The function of the time symbol that multiplies one over the intensity, or `None` for a constant intensity.

    Raises:
        ValidationError:
            The move, the rule, or the schedule is unknown, the objective is not callable, or a row score takes a
            cluster move, or the heat-bath rule without a random-value move and its values.

    Examples:
        >>> from skcomplex.core.elements import Element, LatticeElement
        >>> from skcomplex.core.interactions import SamplingInteraction
        >>> from skcomplex.core.laws.utils import neighbors
        >>> from skcomplex.core.spaces import DiscreteTime, LatticeSpace
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.specs import StateSpec
        >>> from typing import Annotated
        >>> class SpinElement(Element):
        ...     def __init__(self, label, s: Annotated[float, StateSpec()] = 1.0):
        ...         super().__init__(label)
        ...         self.s = s
        >>> objective = lambda n: -n.target.s * neighbors(n.target.other.s)
        >>> ising = SamplingInteraction('ising', target='spins', state='s', objective=objective, intensity=1.0)
        >>> spins = LatticeElement('spins', element=SpinElement, space=LatticeSpace((8, 8)))
        >>> system = System(elements=[spins], interactions=[ising], time=DiscreteTime())
        >>> states = Simulation(system, random_state=0).check().prepare().run(until=20).trajectory_.states
        >>> bool(abs(states[-1].mean()) > 0.9)
        True
        >>> from sage.all__sagemath_symbolics import cos
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.core.laws.utils import exterior_derivative
        >>> from skcomplex.core.spaces import CellComplexSpace, CofaceNeighborhood
        >>> class LinkElement(Element):
        ...     def __init__(self, label, theta: Annotated[float, StateSpec()] = 0.0):
        ...         super().__init__(label)
        ...         self.theta = theta
        >>> wilson = lambda n: -neighbors(cos(exterior_derivative(n.target.theta)), over=CofaceNeighborhood())
        >>> gauge = SamplingInteraction(
        ...     'wilson', target='links', state='theta', objective=wilson, move='rotate', schedule='colored'
        ... )
        >>> links = GroupElement('links', element=LinkElement, space=CellComplexSpace(LatticeSpace((4, 4))), on='edges')
        >>> system = System(elements=[links], interactions=[gauge], time=DiscreteTime())
        >>> Simulation(system, random_state=0).check().prepare().run(until=2).trajectory_.states.shape
        (3, 32)
        >>> import numpy as np
        >>> from skcomplex.specs import RowScoreSpec
        >>> def flip_one(row, random):
        ...     row[random.integers(row.size)] *= -1
        ...     return row
        >>> walk = SamplingInteraction(
        ...     'walk', target='spins', state='s', objective=RowScoreSpec(np.sum), move=flip_one, intensity=float('inf')
        ... )
        >>> spins = LatticeElement('spins', element=SpinElement, space=LatticeSpace((2, 8)))
        >>> system = System(elements=[spins], interactions=[walk], time=DiscreteTime())
        >>> Simulation(system, random_state=0).check().prepare().run(until=3).trajectory_.values('walk.score').shape
        (4, 2)
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        state: str,
        objective: Callable[[SimpleNamespace], object] | RowScoreSpec,
        move: str | Callable[..., object] = 'flip',
        intensity: Annotated[float, ParameterSpec(domain=POSITIVE_VALUES)] = 1.0,
        rule: str = 'metropolis',
        schedule: str = 'checkerboard',
        neighborhood: BaseNeighborhood | None = None,
        values: Sequence[float] = (),
        width: float = 1.0,
        cooling: Callable[[Expression], object] | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.state = state
        self.objective = objective
        self.move = move
        self.intensity = intensity
        self.rule = rule
        self.schedule = schedule
        self.neighborhood = neighborhood
        self.values = values
        self.width = width
        self.cooling = cooling

    def _get_move(self, target: object) -> MoveSpec:
        """Return the spec of the move of a state."""
        if callable(self.move):
            return MoveSpec(target, 'function', function=self.move)
        return MoveSpec(target, self.move, tuple(self.values), self.width)

    def _get_intensity(self, namespace: SimpleNamespace) -> object:
        """Return the intensity of the acceptance before cooling."""
        return self._get_parameter_symbol('intensity', namespace)

    def _check_row_score(self, score: RowScoreSpec) -> None:
        """Raise an error for a row score that cannot run, a cluster move, or a heat-bath rule without values."""
        score.check()
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
        """Return the parameters that a row score names, those of one value per site read at the row."""
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
        random: np.random.Generator,
    ) -> Array:
        """Return a row with the value of one random site drawn among all values by the weights of their scores."""
        candidates = np.asarray(self.values, dtype=float)
        proposal = row.copy()
        site = random.integers(row.size)
        scores = np.empty(candidates.size)
        for position, value in enumerate(candidates):
            proposal[site] = value
            scores[position] = float(score.function(proposal, **keywords))
        excess = scores - scores.min()
        weights = np.exp(-np.where(excess == 0.0, 0.0, excess * np.where(excess == 0.0, 1.0, intensity)))
        proposal[site] = random.choice(candidates, p=weights / weights.sum())
        return proposal

    def _propose_row(self, row: Array, random: np.random.Generator) -> Array:
        """Return a row after one move: a function of the row, or a named move at one random site."""
        if callable(self.move):
            return np.asarray(self.move(row.copy(), random), dtype=float)
        proposal = row.copy()
        site = random.integers(row.size)
        if self.move == 'flip':
            proposal[site] = -row[site]
        elif self.move == 'random-value':
            proposal[site] = random.choice(np.asarray(self.values, dtype=float))
        elif self.move == 'gaussian':
            proposal[site] = row[site] + random.normal(0.0, self.width)
        else:
            proposal[site] = row[site] + random.uniform(-self.width, self.width)
        return proposal

    def _accept_change(self, change: float, intensity: float, random: np.random.Generator) -> bool:
        """Return whether a change of the score is accepted by the rule at the intensity."""
        exponent = 0.0 if change == 0.0 else change * intensity
        if self.rule == 'metropolis':
            return exponent <= 0.0 or bool(random.random() < np.exp(-exponent))
        with np.errstate(over='ignore'):
            return bool(random.random() < 1.0 / (1.0 + np.exp(exponent)))

    def _accept_rows(self, values: Array, *, random: np.random.Generator, time: float, **parameters: object) -> Array:
        """Return a state after the proposed moves of every row, each accepted from the change of the row's score."""
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

    def _check(self) -> None:
        """Raise an error for an unknown move, rule, or schedule, or an objective that is not callable.

        Raises:
            ValidationError:
                The move, the rule, or the schedule is unknown, or the objective or the cooling is not callable.
        """
        super()._check()
        if self.cooling is not None and not callable(self.cooling):
            error_msg = f'The cooling of sampling {self.label!r} must be a function of the time symbol.'
            raise ValidationError(error_msg)
        self._check_variant('move', self.move, MOVES, functions=True)
        objective = getattr(self, self._get_parameter('objective'))
        if isinstance(objective, RowScoreSpec):
            self._check_row_score(objective)
        elif not callable(objective):
            error_msg = f'The objective of sampling {self.label!r} must be a function of the namespace.'
            raise ValidationError(error_msg)
        SamplerSpec(self.rule, schedule=self.schedule).check()

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the sample law of the state.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with `namespace.target` and `namespace.intensity`.

        Returns:
            laws:
                The sample law.
        """
        target = getattr(self._get_member(namespace, 'target'), self.state)
        declared = getattr(self, self._get_parameter('objective'))
        if isinstance(declared, RowScoreSpec):
            return [DriveLaw(target, self._accept_rows, label=self.label, score=self._score_rows)]
        over = VonNeumannNeighborhood() if self.neighborhood is None else self.neighborhood
        objective = ObjectiveSpec(declared(namespace), over)
        intensity = self._get_intensity(namespace)
        if self.cooling is not None:
            intensity = SR(intensity) / SR(self.cooling(namespace.t))
        sampler = SamplerSpec(self.rule, intensity, self.schedule)
        return [SampleLaw(objective, self._get_move(target), sampler, label=self.label)]
