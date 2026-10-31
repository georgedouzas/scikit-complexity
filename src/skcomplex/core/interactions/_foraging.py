"""Define foraging, which feeds the members of a group from a resource of the cells they stand on."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ...errors import ValidationError
from ...specs import ParameterSpec, RoleSpec
from ..laws import BaseLaw, RuleLaw
from ..types import AgentViewProtocol, ModelViewProtocol
from ._interaction import Interaction


class ForagingInteraction(Interaction):
    """Foraging of the members of a group, which take a resource from their cell into a reserve that they spend.

    Every step, the foragers act once each in random order. A forager takes from the member of the resource group in
    its cell the least of the `uptake` and the resource there, and adds `gain` times the amount it took to its reserve.
    It then pays the `metabolism` from its reserve, and it dies when its reserve falls below zero. A surviving forager
    whose reserve reaches the `threshold` splits with the `probability`: it halves its reserve and creates a member of
    its own group in its cell that holds the other half. The foraging is a rule, so the system runs in discrete time,
    and the foragers and resources live on one lattice or network, one member of the resource group per cell.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        foragers:
            The path of the group whose members forage, a variable group when they die or split.

        resources:
            The path of the group that holds the resource of every cell.

        resource:
            The name of the state of the resource group that holds the resource of a cell.

        reserve:
            The name of the state of the foragers that holds what they gathered.

        uptake:
            The most resource a forager takes in one step.

        gain:
            The amount the reserve gains per unit of resource taken.

        metabolism:
            The amount a forager pays from its reserve every step.

        threshold:
            The least reserve with which a forager splits.

        probability:
            The probability that a forager whose reserve reaches the threshold splits in one step.

        phase:
            The phase of the step the foraging runs in, or `None` for the default phase.

    Examples:
        >>> from typing import Annotated
        >>> from skcomplex.core.elements import Element, GroupElement
        >>> from skcomplex.core.interactions import ForagingInteraction
        >>> from skcomplex.core.spaces import LatticeSpace
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.specs import StateSpec
        >>> class StockElement(Element):
        ...     def __init__(self, label, visual=None, *, level: Annotated[float, StateSpec()] = 0.0):
        ...         super().__init__(label, visual=visual)
        ...         self.level = level
        >>> cells = GroupElement('cells', element=StockElement, size=1, values={'level': 3.0}, location=[(0, 0)])
        >>> foragers = GroupElement('foragers', element=StockElement, size=1, location=[(0, 0)], variable=True)
        >>> foraging = ForagingInteraction(
        ...     'foraging', foragers='foragers', resources='cells', resource='level', reserve='level', gain=2.0
        ... )
        >>> system = System(elements=[cells, foragers], interactions=[foraging], space=LatticeSpace((1, 1)))
        >>> trajectory = Simulation(system, random_state=0).check().prepare().run(until=1).trajectory_
        >>> [trajectory.agents(group, 1).values['level'].tolist() for group in ('cells', 'foragers')]
        [[2.0], [2.0]]
    """

    def __init__(
        self,
        label: str,
        *,
        foragers: Annotated[str, RoleSpec()],
        resources: Annotated[str, RoleSpec()],
        resource: str = 'resource',
        reserve: str = 'reserve',
        uptake: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        gain: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        metabolism: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        threshold: Annotated[float, ParameterSpec()] = 0.0,
        probability: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.0,
        phase: str | None = None,
    ) -> None:
        super().__init__(label)
        self.foragers = foragers
        self.resources = resources
        self.resource = resource
        self.reserve = reserve
        self.uptake = uptake
        self.gain = gain
        self.metabolism = metabolism
        self.threshold = threshold
        self.probability = probability
        self.phase = phase

    def _check(self) -> None:
        """Raise an error for a resource or a reserve that is not the name of a state.

        Raises:
            ValidationError:
                The resource or the reserve is not a string.
        """
        super()._check()
        for name in ('resource', 'reserve'):
            value = getattr(self, name)
            if not isinstance(value, str):
                error_msg = f'The {name} of {self.label!r} is {value!r}. Give the name of a state.'
                raise ValidationError(error_msg)

    def _get_key(self, location: object) -> object:
        """Return a location as a key of the cells, a tuple of integers on a lattice or the node on a network."""
        if isinstance(location, tuple | list | np.ndarray):
            return tuple(int(item) for item in location)
        return location

    def _feed(self, forager: AgentViewProtocol, cell: AgentViewProtocol | None, model: ModelViewProtocol) -> None:
        """Move the resource a forager takes from its cell into its reserve."""
        if cell is None:
            return
        taken = min(float(getattr(cell, self.resource)), model.parameter('uptake'))
        setattr(cell, self.resource, getattr(cell, self.resource) - taken)
        setattr(forager, self.reserve, getattr(forager, self.reserve) + model.parameter('gain') * taken)

    def _live(self, forager: AgentViewProtocol, model: ModelViewProtocol) -> None:
        """Pay the metabolism of a forager, then remove it below zero reserve, or split it with its probability."""
        level = getattr(forager, self.reserve) - model.parameter('metabolism')
        setattr(forager, self.reserve, level)
        if level < 0:
            forager.remove()
            return
        probability = model.parameter('probability')
        if probability == 0 or level < model.parameter('threshold') or model.rng.random() >= probability:
            return
        setattr(forager, self.reserve, level / 2)
        forager.create(**{self.reserve: level / 2})

    def _forage(self, model: ModelViewProtocol) -> None:
        """Feed every forager from its cell and let it live, in random order."""
        cells = {self._get_key(cell.location): cell for cell in model.agents('resources')}
        foragers = model.agents('foragers')
        for index in model.rng.permutation(len(foragers)):
            forager = foragers[int(index)]
            if forager.alive:
                self._feed(forager, cells.get(self._get_key(forager.location)), model)
                self._live(forager, model)

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule of the foraging.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rule that feeds the foragers every step.
        """
        return [RuleLaw(self._forage, phase=self.phase)]
