"""Calm rats that turn nervous on their own or by meeting nervous rats, at the alarm propensity."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.interactions import ReactionInteraction
from ....specs import ParameterSpec, RoleSpec
from ....types import Stoichiometry
from ._trail_choice import MODES


class AlarmInteraction(ReactionInteraction):
    """Calm rats that turn nervous on their own or by meeting nervous rats, at the alarm propensity.

    A calm rat turns nervous at rate `spontaneous + contagion * nervous / N`, where `N` is the total number of rats, so
    the propensity of the reaction calm to nervous is that rate times the calm rats. It is a `ReactionInteraction` that
    moves one rat from the calm to the nervous rats, a rate equation, or in stochastic mode a random event per rat.

    Args:
        label:
            The label of the interaction.

        calm:
            The path of the calm rats.

        nervous:
            The path of the nervous rats.

        spontaneous:
            The rate at which a calm rat turns nervous on its own.

        contagion:
            The rate at which a calm rat turns nervous when all others are nervous.

        stochastic:
            Whether changes are random events.

    Examples:
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.models.biology.collective_behaviour import AlarmInteraction, RatsElement
        >>> alarm = AlarmInteraction('alarm', calm='calm', nervous='nervous', stochastic=False)
        >>> rats = [RatsElement('calm', n=90.0), RatsElement('nervous', n=10.0)]
        >>> system = System(elements=rats, interactions=[alarm])
        >>> trajectory = Simulation(system, dt=0.1).prepare().run(until=5.0).trajectory_
        >>> round(float(trajectory.values('calm.n')[-1] + trajectory.values('nervous.n')[-1]), 6)
        100.0
    """

    def __init__(
        self,
        label: str,
        *,
        calm: Annotated[str, RoleSpec()],
        nervous: Annotated[str, RoleSpec()],
        spontaneous: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        contagion: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        stochastic: bool = True,
    ) -> None:
        super().__init__(label)
        self.calm = calm
        self.nervous = nervous
        self.spontaneous = spontaneous
        self.contagion = contagion
        self.stochastic = stochastic

    def _get_reactants(self) -> Stoichiometry:
        """Return the calm rats `n`, which every alarm takes one from."""
        return {(self.calm, 'n'): 1}

    def _get_products(self) -> Stoichiometry:
        """Return the nervous rats `n`, which every alarm adds one to."""
        return {(self.nervous, 'n'): 1}

    def _get_mode(self) -> str:
        """Return the mode of the reaction, random events when stochastic and a rate equation otherwise."""
        return MODES[self.stochastic]

    def _get_rate(self, namespace: SimpleNamespace) -> Expression:
        """Return the alarm propensity, the rate at which a calm rat turns nervous times the calm rats.

        Args:
            namespace:
                The namespace of the symbols the rate can use, with the calm and the nervous rats.

        Returns:
            rate:
                The propensity of the alarm.
        """
        s = namespace
        calm, nervous = s.calm.n, s.nervous.n
        propensity: Expression = (s.spontaneous + s.contagion * nervous / (calm + nervous)) * calm
        return propensity
