"""Define Monte Carlo sampling of a state, by moves accepted from the change of a local energy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ...errors import ValidationError
from ...specs import EnergySpec, MoveSpec, ParameterSpec, RoleSpec, SamplerSpec
from ..laws import BaseLaw, SampleLaw
from ..neighborhoods import BaseNeighborhood, VonNeumannNeighborhood
from ._interaction import Interaction

MOVES = ('flip', 'random-value', 'rotate', 'gaussian', 'cluster')


class SamplingInteraction(Interaction):
    """Monte Carlo sweeps of a state of the members of a lattice or network group, one sweep per unit of time.

    Every sweep proposes a move of the state of every member and accepts it from the change of the local energy of the
    member, the terms of the energy that involve its value, at the temperature. The local energy is a function of the
    namespace, which holds `namespace.target`, the states and parameters of the member, and `namespace.target.other`, a
    neighbour, whose values `neighbors(...)` sums over `neighborhood`. A move is `'flip'`, the negative value,
    `'random-value'`, a value drawn from `values`, `'rotate'`, the value plus a uniform number in `[-width, width]`,
    `'gaussian'`, the value plus a normal number of standard deviation `width`, or a function
    `move(values, random)` of the array of the current values of some members and the random generator that returns
    their proposed values. The move `'cluster'` is the Wolff algorithm instead: a sweep flips clusters until the flipped
    members number as many as the members, each cluster grown from a random member through the bonds to neighbours
    with probability `1 - exp(min(0, -(f(-s, n) - f(s, n)) / T))`, where `f(s, n)` is the term of the local energy of
    a member of value `s` with a neighbour of value `n`. It takes an energy of such pair terms that the flip of both
    values keeps, and ignores the rule and the schedule.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group on the sites of a lattice or the nodes of a network.

        state:
            The name of the state the moves change.

        energy:
            The function `energy(namespace)` of the local energy of a member.

        move:
            `'flip'`, `'random-value'`, `'rotate'`, `'gaussian'`, `'cluster'`, or a function that proposes values.

        temperature:
            The temperature.

        rule:
            The acceptance, `'metropolis'`, `'glauber'`, or `'heat-bath'`, which draws among all values of a discrete
            move by their Boltzmann weights.

        schedule:
            `'checkerboard'`, `'colored'`, or `'random-sequential'`.

        neighborhood:
            The neighbourhood the energy reduces over, or `None` for the von Neumann neighbourhood of a square lattice.
            A network takes `AdjacencyNeighborhood()`.

        values:
            The values a random-value move draws from.

        width:
            The largest change of a rotation, or the standard deviation of a Gaussian move.

    Raises:
        ValidationError:
            The move, the rule, or the schedule is unknown, or the energy is not callable.

    Examples:
        >>> from skcomplex.core.elements import Element, LatticeElement
        >>> from skcomplex.core.interactions import SamplingInteraction
        >>> from skcomplex.core.laws.utils import neighbors
        >>> from skcomplex.core.spaces import LatticeSpace
        >>> from skcomplex.core.times import DiscreteTime
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.specs import StateSpec
        >>> from typing import Annotated
        >>> class SpinElement(Element):
        ...     def __init__(self, label, s: Annotated[float, StateSpec()] = 1.0):
        ...         super().__init__(label)
        ...         self.s = s
        >>> energy = lambda n: -n.target.s * neighbors(n.target.other.s)
        >>> ising = SamplingInteraction('ising', target='spins', state='s', energy=energy, temperature=1.0)
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
        energy: Callable[[SimpleNamespace], object],
        move: str | Callable[..., object] = 'flip',
        temperature: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        rule: str = 'metropolis',
        schedule: str = 'checkerboard',
        neighborhood: BaseNeighborhood | None = None,
        values: Sequence[float] = (),
        width: float = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.state = state
        self.energy = energy
        self.move = move
        self.temperature = temperature
        self.rule = rule
        self.schedule = schedule
        self.neighborhood = neighborhood
        self.values = values
        self.width = width

    def _get_move(self, target: object) -> MoveSpec:
        """Return the spec of the move of a state."""
        if callable(self.move):
            return MoveSpec(target, 'function', function=self.move)
        return MoveSpec(target, self.move, tuple(self.values), self.width)

    def _check(self) -> None:
        """Raise an error for an unknown move, rule, or schedule, or an energy that is not callable.

        Raises:
            ValidationError:
                The move, the rule, or the schedule is unknown, or the energy is not callable.
        """
        super()._check()
        if not callable(self.move) and self.move not in MOVES:
            error_msg = (
                f'The move of sampling {self.label!r} is {self.move!r}. Use one of {", ".join(MOVES)}, or a function.'
            )
            raise ValidationError(error_msg)
        if not callable(self.energy):
            error_msg = f'The energy of sampling {self.label!r} must be a function of the namespace.'
            raise ValidationError(error_msg)
        SamplerSpec(self.rule, self.temperature, self.schedule).check()

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the sample law of the state.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with `namespace.target` and `namespace.temperature`.

        Returns:
            laws:
                The sample law.
        """
        over = VonNeumannNeighborhood() if self.neighborhood is None else self.neighborhood
        target = getattr(namespace.target, self.state)
        energy = EnergySpec(self.energy(namespace), over)
        sampler = SamplerSpec(self.rule, namespace.temperature, self.schedule)
        return [SampleLaw(energy, self._get_move(target), sampler, label=self.label)]
