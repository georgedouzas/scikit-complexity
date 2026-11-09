"""Define cubic autocatalysis, the reaction `A + 2B -> 3B`.

It runs between two states of one reactor or of every site.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from .....core.components.interactions import ReactionInteraction
from .....core.specs import ParameterSpec, RoleSpec
from .....core.types import Stoichiometry


class CubicAutocatalysisInteraction(ReactionInteraction):
    """Cubic autocatalysis `A + 2B -> 3B` at the mass-action rate `k a b^2`.

    It runs between two states of the target.

    The autocatalyst `B` converts the substrate `A` into itself. The reaction is the autocatalytic step of the
    Gray-Scott system, with the substrate `u` and the autocatalyst `v`, and of the Brusselator and the Schnakenberg
    system, `2U + V -> 3U`, with the substrate `v` and the autocatalyst `u`. A target that is a field reacts at every
    site.

    Args:
        label:
            The label of the reaction.

        target:
            The path of the reactor, or of the field whose every site reacts.

        substrate:
            The name of the state of the substrate.

        autocatalyst:
            The name of the state of the autocatalyst.

        k:
            The rate constant.

        mode:
            The form of the laws, `'rate'`, `'jump'`, or `'langevin'`.

    Raises:
        ValidationError:
            The mode is unknown, or a state name is not an identifier.

    Examples:
        >>> from skcomplex.core.components.systems import CompositeSystem
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.systems.chemistry.common.kinetics import CubicAutocatalysisInteraction
        >>> from skcomplex.systems.chemistry.kinetics import CubicAutocatalysisElement
        >>> reaction = CubicAutocatalysisInteraction('autocatalysis', target='reactor', substrate='a', autocatalyst='b')
        >>> reactor = CubicAutocatalysisElement('reactor', a=1.0, b=0.5)
        >>> states = simulate(CompositeSystem(elements=[reactor], interactions=[reaction]), until=10, dt=1.0).states
        >>> round(float(states[-1].sum()), 6)
        1.5
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        substrate: str = 'u',
        autocatalyst: str = 'v',
        k: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        mode: str = 'rate',
    ) -> None:
        super().__init__(label, target=target, k=k, mode=mode)
        self.substrate = substrate
        self.autocatalyst = autocatalyst

    def _get_reactants(self) -> Stoichiometry:
        """Return one substrate and two autocatalysts, the reactants of `A + 2B -> 3B`."""
        return {self.substrate: 1, self.autocatalyst: 2}

    def _get_products(self) -> Stoichiometry:
        """Return three autocatalysts, the products of `A + 2B -> 3B`."""
        return {self.autocatalyst: 3}
