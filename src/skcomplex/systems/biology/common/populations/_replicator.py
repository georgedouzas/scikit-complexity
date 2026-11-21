"""Define replicator dynamics.

Every share of a group grows by its fitness above the mean fitness.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression

from .....core.components.interactions import Interaction
from .....core.components.laws import BaseLaw, RateLaw
from .....core.components.laws.utils import total
from .....core.errors import ValidationError
from .....core.types import Role


class ReplicatorInteraction(Interaction):
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
        >>> from skcomplex.core.components.laws.utils import total
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.biology.evolution import ReplicatorInteraction, StrategyElement
        >>> population = StrategyElement('population', x=[0.5, 0.5], size=2)
        >>> payoff = ((1.0, 1.0), (0.0, 0.0))
        >>> replication = ReplicatorInteraction(
        ...     'replication', target='population', fitness=lambda s: total(s.target.x, weights=payoff)
        ... )
        >>> system = System(elements=[population], interactions=[replication])
        >>> trajectory = system.prepare().simulate(until=20.0, dt=0.1)
        >>> [round(float(share), 3) for share in trajectory.get_values('population.x')[-1]]
        [1.0, 0.0]
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        fitness: Callable[[SimpleNamespace], object] | None = None,
        state: str = 'x',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.state = state
        self.fitness = fitness

    def _check_fitness(self) -> None:
        """Raise an error for a fitness that is not callable.

        It is the hook a subclass that derives its fitness overrides.
        """
        if not callable(self.fitness):
            error_msg = f'The fitness of {self.label!r} is {self.fitness!r}. Give a function of the namespace.'
            raise ValidationError(error_msg)

    def _check_arguments(self) -> None:
        """Raise an error for a state, functions, or a fitness the interaction cannot take.

        Raises:
            ValidationError:
                The state is not a name, or the fitness is not callable.
        """
        super()._check_arguments()
        self._check_fitness()

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the laws of a binding.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        self.laws_: list[BaseLaw] = [RateLaw(getattr(context.target, self.state), self._get_rate(context))]

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
