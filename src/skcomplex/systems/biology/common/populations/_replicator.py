"""Define replicator dynamics.

Every share of a group grows by its fitness above the mean fitness.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression

from .....core.components.interactions import RateInteraction
from .....core.components.laws.utils import total
from .....core.errors import ValidationError
from .....core.specs import RoleSpec


class ReplicatorInteraction(RateInteraction):
    """Replicator equation of the shares of a group, with a fitness `f`.

    The shares follow `dx_i/dt = x_i (f_i - sum_j x_j f_j)`.

    Every member of the group holds the share of a type, such as a strategy or a molecule. Its share grows with its
    fitness above the mean fitness of the group, so the total of the shares stays constant.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of types.

        fitness:
            The function of the namespace that returns the fitness of a member, such as
            `lambda s: total(s.target.x, weights=payoff)` for the payoff of a strategy against the others.

        state:
            The name of the state of the members that holds the share.

    Raises:
        ValidationError:
            The fitness is not callable, raised by `prepare`.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.core.components.laws.utils import total
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.systems.biology.evolution import ReplicatorInteraction, StrategyElement
        >>> from skcomplex.core.engine.runtime import simulate
        >>> population = GroupElement('population', element=StrategyElement, size=2, values={'x': [0.5, 0.5]})
        >>> payoff = ((1.0, 1.0), (0.0, 0.0))
        >>> replication = ReplicatorInteraction(
        ...     'replication', target='population', fitness=lambda s: total(s.target.x, weights=payoff)
        ... )
        >>> system = System(elements=[population], interactions=[replication])
        >>> trajectory = simulate(system, until=20.0, dt=0.1)
        >>> [round(float(share), 3) for share in trajectory.values('population.x')[-1]]
        [1.0, 0.0]
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        fitness: Callable[[SimpleNamespace], object] | None = None,
        state: str = 'x',
    ) -> None:
        super().__init__(label, target=target, state=state)
        self.fitness = fitness

    def _check_fitness(self) -> None:
        """Raise an error for a fitness that is not callable.

        It is the hook a subclass that derives its fitness overrides.
        """
        if not callable(self.fitness):
            error_msg = f'The fitness of {self.label!r} is {self.fitness!r}. Give a function of the namespace.'
            raise ValidationError(error_msg)

    def _check(self) -> None:
        """Raise an error for a state, functions, or a fitness the interaction cannot take.

        Raises:
            ValidationError:
                The state is not a name, or the fitness is not callable.
        """
        super()._check()
        self._check_fitness()

    def _get_fitness(self, namespace: SimpleNamespace) -> Expression:
        """Return the fitness of a member.

        It is the hook a subclass overrides to derive it from its parameters.
        """
        fitness = self.fitness
        if fitness is None:
            error_msg = f'The fitness of {self.label!r} is missing. Give a function of the namespace.'
            raise ValidationError(error_msg)
        value: Expression = fitness(namespace)
        return value

    def _get_rate(self, namespace: SimpleNamespace) -> Expression:
        """Return the replicator rate `x_i (f_i - sum_j x_j f_j)`.

        Args:
            namespace:
                The namespace of the symbols the rate can use, with the namespace of the target.

        Returns:
            rate:
                The rate of the share.
        """
        share = getattr(namespace.target, self.state)
        fitness = self._get_fitness(namespace)
        rate: Expression = share * (fitness - total(share * fitness))
        return rate
