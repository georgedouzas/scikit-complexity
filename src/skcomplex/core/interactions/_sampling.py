"""Define Monte Carlo sampling of a state, by moves accepted from the change of a local objective."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import SR, Expression, oo
from sage.sets.real_set import RealSet

from ...errors import ValidationError
from ...specs import MoveSpec, ObjectiveSpec, ParameterSpec, RoleSpec, SamplerSpec
from ..laws import BaseLaw, SampleLaw
from ..spaces import BaseNeighborhood, VonNeumannNeighborhood
from ._interaction import Interaction

MOVES = ('flip', 'random-value', 'rotate', 'gaussian', 'cluster')


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
    flip of both values keeps, and ignores the rule and the schedule.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group on the sites of a lattice or the nodes of a network.

        state:
            The name of the state the moves change.

        objective:
            The function `objective(namespace)` of the local objective of a member.

        move:
            `'flip'`, `'random-value'`, `'rotate'`, `'gaussian'`, `'cluster'`, or a function that proposes values.

        intensity:
            The factor of the change of the objective in the acceptance `exp(-intensity * change)`.

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
            The move, the rule, or the schedule is unknown, or the objective is not callable.

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
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        state: str,
        objective: Callable[[SimpleNamespace], object],
        move: str | Callable[..., object] = 'flip',
        intensity: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
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
        if not callable(getattr(self, self._get_parameter('objective'))):
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
        over = VonNeumannNeighborhood() if self.neighborhood is None else self.neighborhood
        target = getattr(self._get_member(namespace, 'target'), self.state)
        objective = ObjectiveSpec(getattr(self, self._get_parameter('objective'))(namespace), over)
        intensity = self._get_intensity(namespace)
        if self.cooling is not None:
            intensity = SR(intensity) / SR(self.cooling(namespace.t))
        sampler = SamplerSpec(self.rule, intensity, self.schedule)
        return [SampleLaw(objective, self._get_move(target), sampler, label=self.label)]
