"""Define the interaction that samples a state by Monte Carlo moves."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import inspect
from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar

from sage.all__sagemath_symbolics import RR, SR, ConditionSet, Expression

from ...errors import ValidationError
from ...specs import (
    BaseSpec,
    MoveSpec,
    ObjectiveSpec,
    ParameterSpec,
    RoleSpec,
    SamplerSpec,
    Tags,
)
from ...types import ElementKind
from ..laws import ArrayUpdateLaw, BaseLaw, SampleLaw
from ..spaces import (
    BallNeighborhood,
    BaseNeighborhood,
    BaseSpace,
)
from ._interaction import Interaction

MOVES = ('flip', 'random-value', 'rotate', 'gaussian', 'cluster')


def _is_positive(value: float) -> bool:
    """Return whether a value is above zero."""
    return bool(value > 0)


POSITIVE_VALUES = ConditionSet(RR, _is_positive)


class SamplingInteraction(Interaction):
    """An interaction that samples a state of the members by Monte Carlo moves.

    In each unit of time the interaction makes one sweep. A sweep proposes a new value of the state of every member and
    accepts it by the change of the local objective of the member, the terms of the objective that involve its value,
    scaled by `intensity`. The objective is a function of the namespace, which holds `namespace.target`, the member, and
    `namespace.target.other`, a neighbour, whose values `neighbors(...)` sums over `neighborhood`. The argument `move`
    proposes the new values:

    - `'flip'`: the negative of the value, as for a spin of plus or minus one.
    - `'random-value'`: a value drawn from `values`.
    - `'rotate'`: the value plus a uniform number in `[-width, width]`.
    - `'gaussian'`: the value plus a normal number of standard deviation `width`.
    - `'cluster'`: the Wolff algorithm, which flips whole clusters of aligned neighbours at once.
    - A function `move(values, random)` that returns the proposed values.

    A group on the edges of a cell complex takes an objective that sums over `CofaceNeighborhood`, and its `'colored'`
    schedule updates together only links that share no face.

    Unlike `MapInteraction`, which sets the next value by a rule, it proposes values and accepts them by a probability,
    so the states follow a stationary distribution. Choose it when the model is a Boltzmann distribution of an energy,
    as in a Markov chain Monte Carlo method.

    It appears in many fields. Spins of a magnet in the Ising model, cells of a tissue in the cellular Potts model, and
    residents who move in a model of segregation are sampled this way.

    Args:
        label:
            The name of the interaction, unique among the parts of its system, such as `'spins'`.

        target:
            The path of the group on the sites of a lattice, the nodes of a network, or the edges of a cell complex.

        state:
            The name of the state the moves change.

        objective:
            The function `objective(namespace)` of the local objective of a member, or the score spec of a subclass
            that samples by its own law.

        move:
            How a new value is proposed, one of the moves above. The move `'cluster'` grows each cluster from a
            random member through the bonds to its neighbours with the probability
            `1 - exp(min(0, -intensity * (f(-s, n) - f(s, n))))`, where `f(s, n)` is the term of the objective for a
            member of value `s` and a neighbour of value `n`. A sweep flips clusters until as many members have
            flipped as there are members. It needs an objective of such pair terms that flipping both values keeps,
            and it ignores `rule` and `schedule`.

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
            The move, the rule, or the schedule is unknown, or the objective is not callable, raised by `prepare`.

    Examples:
        >>> from skcomplex.core.components.elements import Element, GroupElement
        >>> from skcomplex.core.components.interactions import SamplingInteraction
        >>> from skcomplex.core.components.laws.utils import neighbors
        >>> from skcomplex.core.components.spaces import DiscreteTime, LatticeSpace
        >>> from skcomplex.core.components.systems import CompositeSystem
        >>> from skcomplex.core.specs import StateSpec
        >>> from typing import Annotated
        >>> from skcomplex.core.engine.runtime import simulate
        >>> class SpinElement(Element):
        ...     def __init__(self, label, s: Annotated[float, StateSpec()] = 1.0):
        ...         super().__init__(label)
        ...         self.s = s
        >>> objective = lambda n: -n.target.s * neighbors(n.target.other.s)
        >>> ising = SamplingInteraction('ising', target='spins', state='s', objective=objective, intensity=1.0)
        >>> spins = GroupElement('spins', element=SpinElement, space=LatticeSpace((8, 8)), on='sites')
        >>> system = CompositeSystem(elements=[spins], interactions=[ising], time=DiscreteTime())
        >>> states = simulate(system, until=20, random_state=0).states
        >>> bool(abs(states[-1].mean()) > 0.9)
        True
        >>> from sage.all__sagemath_symbolics import cos
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.core.components.laws.utils import exterior_derivative
        >>> from skcomplex.core.components.spaces import CellComplexSpace, CofaceNeighborhood
        >>> from skcomplex.core.engine.runtime import simulate
        >>> class LinkElement(Element):
        ...     def __init__(self, label, theta: Annotated[float, StateSpec()] = 0.0):
        ...         super().__init__(label)
        ...         self.theta = theta
        >>> wilson = lambda n: -neighbors(cos(exterior_derivative(n.target.theta)), over=CofaceNeighborhood())
        >>> gauge = SamplingInteraction(
        ...     'wilson', target='links', state='theta', objective=wilson, move='rotate', schedule='colored'
        ... )
        >>> links = GroupElement('links', element=LinkElement, space=CellComplexSpace(LatticeSpace((4, 4))), on='edges')
        >>> system = CompositeSystem(elements=[links], interactions=[gauge], time=DiscreteTime())
        >>> simulate(system, until=2, random_state=0).states.shape
        (3, 32)
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (SampleLaw,)

    def _get_tags(self) -> Tags:
        """Return the tags of a Monte Carlo sweep of a function, or of score drives.

        The drives are of a declared score, which draw.
        """
        sweep = callable(self._get_argument('objective'))
        return self._get_variant_tags((SampleLaw,) if sweep else (ArrayUpdateLaw,), stochastic=True)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec(elements=(ElementKind.ANY,), states=('state',))],
        state: str,
        objective: Callable[[SimpleNamespace], object] | BaseSpec,
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

    def _get_keywords(self, function: Callable[..., object], parameters: Mapping[str, object]) -> dict[str, object]:
        """Return the parameters that a function names."""
        try:
            names = inspect.signature(function).parameters
        except (TypeError, ValueError):
            return {}
        return {name: value for name, value in parameters.items() if name in names}

    def _check_move(self) -> None:
        """Raise an error for an unknown move."""
        self._check_variant('move', self.move, MOVES, functions=True)

    def _check_objective(self, objective: object) -> None:
        """Raise an error for an objective that is not a function of the namespace.

        A subclass that samples by its own law checks its own score spec.
        """
        if not callable(objective):
            error_msg = f'The objective of sampling {self.label!r} must be a function of the namespace.'
            raise ValidationError(error_msg)

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Raise an error for an unknown move, rule, or schedule, or a bad objective.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                The move, the rule, or the schedule is unknown, or the objective or the cooling is not callable.
        """
        super()._prepare(context)
        if self.cooling is not None and not callable(self.cooling):
            error_msg = f'The cooling of sampling {self.label!r} must be a function of the time symbol.'
            raise ValidationError(error_msg)
        self._check_move()
        self._check_objective(getattr(self, self._get_parameter('objective')))
        SamplerSpec(self.rule, schedule=self.schedule)._check()

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
        over = BallNeighborhood(metric='manhattan') if self.neighborhood is None else self.neighborhood
        objective = ObjectiveSpec(declared(namespace), over)
        intensity = self._get_intensity(namespace)
        if self.cooling is not None:
            intensity = SR(intensity) / SR(self.cooling(namespace.t))
        sampler = SamplerSpec(self.rule, intensity, self.schedule)
        return [SampleLaw(objective, self._get_move(target), sampler, label=self.label)]
