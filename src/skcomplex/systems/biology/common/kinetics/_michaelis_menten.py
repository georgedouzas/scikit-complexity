"""Define Michaelis-Menten reactions, reactions whose propensity saturates in the amount of their substrate."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import SR, Expression

from .....core.components.interactions import ReactionInteraction
from .....core.errors import ValidationError
from .....core.specs import RoleSpec
from .....core.types import Stoichiometry


class MichaelisMentenInteraction(ReactionInteraction):
    """Reaction at the Michaelis-Menten propensity `vmax S / (km + S)`, times every modifier to its coefficient.

    The substrate `S` is a state of the target, or a function of the namespace that returns an amount that is no state
    of its own, such as the phosphorylated form `A_T - A` of a protein whose total is conserved. The modifiers are the
    enzymes that catalyse the reaction, which scale the propensity but do not change, so `vmax` is a catalytic rate when
    the reaction has modifiers and a maximal rate when it has none. The maximal rate and the Michaelis constant are
    functions of the namespace, which reads the parameters of the target, or numbers.

    Args:
        label:
            The label of the reaction, unique among the parts of its system.

        target:
            The path of the element, group, or field whose states the species named by a state alone are.

        reactants:
            The coefficients of the reactants, or `None` for a source.

        products:
            The coefficients of the products, or `None` for a sink.

        substrate:
            The name of the state of the target that saturates the propensity, or a function of the namespace that
            returns the amount that does.

        vmax:
            A function of the namespace that returns the maximal or catalytic rate, or a number.

        km:
            A function of the namespace that returns the Michaelis constant, or a number.

        modifiers:
            The coefficients of the modifiers, the enzymes that scale the propensity, or `None`.

        mode:
            The form of the laws, `'rate'`, `'jump'`, or `'langevin'`.

    Raises:
        ValidationError:
            The substrate is neither a name nor a function, a rate is neither a function nor a number, or the reaction
            cannot take its species or mode.

    Examples:
        >>> from skcomplex.core.components.elements import CompartmentElement
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.core.engine.runtime import Simulation
        >>> from skcomplex.systems.biology.common.kinetics import MichaelisMentenInteraction
        >>> uptake = MichaelisMentenInteraction(
        ...     'uptake', target='cell', reactants={'n': 1}, substrate='n', vmax=1.0, km=100.0
        ... )
        >>> system = System(elements=[CompartmentElement('cell', n=500.0)], interactions=[uptake])
        >>> trajectory = Simulation(system, dt=1.0).prepare().run(until=10.0).trajectory_
        >>> round(float(trajectory.values('cell.n')[-1]), 3)
        491.679
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str | None, RoleSpec()] = None,
        reactants: Annotated[Stoichiometry | None, RoleSpec(variadic=True)] = None,
        products: Annotated[Stoichiometry | None, RoleSpec(variadic=True)] = None,
        substrate: str | Callable[[SimpleNamespace], object] = 'n',
        vmax: Callable[[SimpleNamespace], object] | float = 1.0,
        km: Callable[[SimpleNamespace], object] | float = 1.0,
        modifiers: Annotated[Stoichiometry | None, RoleSpec(variadic=True)] = None,
        mode: str = 'rate',
    ) -> None:
        super().__init__(label, target=target, reactants=reactants, products=products, modifiers=modifiers, mode=mode)
        self.substrate = substrate
        self.vmax = vmax
        self.km = km

    def _check(self) -> None:
        """Raise an error for a substrate or rates the reaction cannot take.

        Raises:
            ValidationError:
                The substrate is neither a name nor a function, a rate is neither a function nor a number, or the
                reaction cannot take its species or mode.
        """
        super()._check()
        if not (callable(self.substrate) or (isinstance(self.substrate, str) and self.substrate.isidentifier())):
            error_msg = (
                f'The substrate of {self.label!r} is {self.substrate!r}. Give the name of a state of the target or a '
                'function of the namespace.'
            )
            raise ValidationError(error_msg)
        for name in ('vmax', 'km'):
            value = getattr(self, name)
            if not (callable(value) or isinstance(value, int | float)):
                error_msg = f'The {name} of {self.label!r} is {value!r}. Give a function of the namespace or a number.'
                raise ValidationError(error_msg)

    def _evaluate(self, value: Callable[[SimpleNamespace], object] | float, namespace: SimpleNamespace) -> object:
        """Return the value of a function of the namespace, or a number as it is."""
        return value(namespace) if callable(value) else value

    def _get_rate(self, namespace: SimpleNamespace) -> Expression:
        """Return the saturating propensity of the substrate, times every modifier to its coefficient.

        Args:
            namespace:
                The namespace of the reaction, with the namespace of the target.

        Returns:
            propensity:
                The propensity of the reaction.
        """
        if callable(self.substrate):
            substrate = SR(self.substrate(namespace))
        else:
            substrate = SR(getattr(namespace.target, self.substrate))
        vmax, km = SR(self._evaluate(self.vmax, namespace)), SR(self._evaluate(self.km, namespace))
        propensity = vmax * substrate / (km + substrate)
        for state, coefficient in self._get_terms(self._get_modifiers(), namespace, 'modifiers'):
            propensity *= state**coefficient
        return propensity
