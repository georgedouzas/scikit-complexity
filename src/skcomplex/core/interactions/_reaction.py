"""Define reactions, interactions that move units between states by mass action."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import SR, Expression, oo, sqrt
from sage.sets.real_set import RealSet

from ...errors import ValidationError
from ...specs import ParameterSpec, RoleSpec
from ..base import BasePart
from ..laws import BaseLaw, JumpLaw, NoiseLaw, RateLaw
from ..laws.utils import where
from ._interaction import Interaction

MODES = ('rate', 'jump', 'langevin')


class ReactionInteraction(Interaction):
    """Reaction that moves units from its reactants to its products at a mass-action propensity.

    The reactants and the products are the paths of the elements whose state `state` holds the units, and a path that
    appears twice counts twice, so `S + I -> 2 I` has the reactants `['S', 'I']` and the products `['I', 'I']`. The
    propensity is `k` times the product of the reactant states, or the expression the `propensity` function returns. The
    reaction changes every state by the number of times it appears among the products minus among the reactants, so
    it conserves the total of the states whose changes sum to zero.

    In the mode `'rate'` the reaction is a rate equation. In the mode `'jump'` it is a random event of the propensity
    that moves single units. In the mode `'langevin'` it is the chemical Langevin equation, the rate equation with one
    Wiener process of amplitude the square root of the propensity driving every state the reaction changes.

    Args:
        label:
            The label of the reaction, unique among the parts of its system.

        reactants:
            The paths of the reactants, one entry per unit the reaction consumes, empty for a source.

        products:
            The paths of the products, one entry per unit the reaction produces, empty for a sink.

        k:
            The rate constant.

        mode:
            The form of the laws, `'rate'`, `'jump'`, or `'langevin'`.

        state:
            The name of the state of every reactant and product that holds its units.

        propensity:
            A function of the namespace of the reaction that returns the propensity, or `None` for mass action. The
            namespace holds the tuples `reactants` and `products` of the namespaces of the bound elements and the
            symbol `k`.

    Raises:
        ValidationError:
            The mode is unknown, the reaction has neither reactants nor products, or the state is not a name.

    Examples:
        >>> from skcomplex.core.elements import CompartmentElement
        >>> from skcomplex.core.interactions import ReactionInteraction
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> system = System(
        ...     elements=[CompartmentElement('S', n=990), CompartmentElement('I', n=10), CompartmentElement('R')],
        ...     interactions=[
        ...         ReactionInteraction('infection', reactants=['S', 'I'], products=['I', 'I'], k=0.0005),
        ...         ReactionInteraction('recovery', reactants=['I'], products=['R'], k=0.1),
        ...     ],
        ... )
        >>> states = Simulation(system, dt=1.0).check().prepare().run(until=50).trajectory_.states
        >>> round(float(states[-1].sum()), 6)
        1000.0
    """

    def __init__(
        self,
        label: str,
        *,
        reactants: Annotated[Sequence[str], RoleSpec(variadic=True)] = (),
        products: Annotated[Sequence[str], RoleSpec(variadic=True)] = (),
        k: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        mode: str = 'rate',
        state: str = 'n',
        propensity: Callable[[SimpleNamespace], Expression] | None = None,
    ) -> None:
        super().__init__(label)
        self.reactants = reactants
        self.products = products
        self.k = k
        self.mode = mode
        self.state = state
        self.propensity = propensity

    def _get_changes(self, namespace: SimpleNamespace) -> dict[Expression, int]:
        """Return the net change of every state the reaction changes, products minus reactants."""
        changes: dict[Expression, int] = {}
        for sign, entries in ((-1, namespace.reactants), (1, namespace.products)):
            for entry in entries:
                target = getattr(entry, self.state)
                changes[target] = changes.get(target, 0) + sign
        return {target: change for target, change in changes.items() if change}

    def _get_propensity(self, namespace: SimpleNamespace) -> Expression:
        """Return the propensity of the reaction, by the user's function or by mass action."""
        if self.propensity is not None:
            return SR(self.propensity(namespace))
        propensity = SR(namespace.k)
        for entry in namespace.reactants:
            propensity *= getattr(entry, self.state)
        return propensity

    def _get_langevin_laws(self, propensity: Expression, changes: dict[Expression, int]) -> list[BaseLaw]:
        """Return the drift of the reaction and the noise of its one Wiener process on every state it changes."""
        amplitude = sqrt(where(propensity > 0, propensity, 0))
        rates: list[BaseLaw] = [RateLaw(target, change * propensity) for target, change in changes.items()]
        noises: list[BaseLaw] = [
            NoiseLaw(target, change * amplitude, process=self.label) for target, change in changes.items()
        ]
        return rates + noises

    def _check_mode(self) -> None:
        """Raise an error for an unknown mode."""
        if self.mode not in MODES:
            error_msg = f'Reaction {self.label!r} has the mode {self.mode!r}. Use one of {", ".join(MODES)}.'
            raise ValidationError(error_msg)

    def _check_species(self) -> None:
        """Raise an error for a reaction without reactants and products, or a state that is not a name."""
        if not self.reactants and not self.products:
            error_msg = f'Reaction {self.label!r} has neither reactants nor products. Give at least one of them.'
            raise ValidationError(error_msg)
        if not isinstance(self.state, str) or not self.state.isidentifier():
            error_msg = f'Reaction {self.label!r} reads the state {self.state!r}, which is not a name of a state.'
            raise ValidationError(error_msg)

    def _check(self) -> None:
        """Raise an error for an unknown mode, a reaction without reactants and products, or a state that is no name.

        Raises:
            ValidationError:
                The type declares states, the mode is unknown, the reaction has neither reactants nor products, or the
                state is not a name.
        """
        super()._check()
        self._check_mode()
        self._check_species()

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BasePart]:
        """Return the laws of the reaction in its mode.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with the namespaces of the reactants and the products.

        Returns:
            laws:
                The rate laws, the jump law, or the rate and noise laws of the reaction.
        """
        propensity = self._get_propensity(namespace)
        changes = self._get_changes(namespace)
        if self.mode == 'jump':
            return [JumpLaw(propensity, changes)]
        if self.mode == 'langevin':
            return self._get_langevin_laws(propensity, changes)
        return [RateLaw(target, change * propensity) for target, change in changes.items()]
