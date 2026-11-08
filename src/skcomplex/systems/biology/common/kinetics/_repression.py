"""Define repression of the synthesis of a species.

Repression is a reaction that synthesizes a species at a rate a Hill function of a repressor lowers.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from .....core.components.interactions import ReactionInteraction
from .....core.components.laws.utils import hill
from .....core.specs import ParameterSpec, RoleSpec
from .....core.types import Stoichiometry


class RepressionInteraction(ReactionInteraction):
    """Synthesis of a product at the rate `alpha0 + alpha (1 - hill(y, K, n))`.

    The synthesis is repressed by a modifier `y`.

    The product, such as the mRNA of a gene, gains units at the leaky rate `alpha0` plus the rate `alpha` lowered by the
    decreasing Hill function `1 / (1 + (y / K)^n)` of the one modifier, the repressor. The repressor enters the rate
    and does not change. In the mode `'jump'` every synthesis is a random event that adds one unit.

    Args:
        label:
            The label of the repression.

        reactants:
            The coefficients of the reactants, or `None` for a synthesis from a source.

        products:
            The coefficient of the synthesized state, such as `{('gene', 'm'): 1}`.

        modifiers:
            The coefficient of the repressor state, such as `{('repressor', 'p'): 1}`.

        alpha:
            The synthesis rate without repressor.

        alpha0:
            The leaky synthesis rate at saturating repressor.

        K:
            The repressor amount at half repression.

        n:
            The Hill coefficient of the repression.

        mode:
            The form of the laws, `'rate'`, `'jump'`, or `'langevin'`.

    Raises:
        ValidationError:
            The mode is unknown, the reaction has neither reactants nor products, or an entry is not a pair of a path
            and a state name with a positive coefficient.

    Examples:
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.systems.biology.systems_biology import RepressionInteraction, SpeciesElement
        >>> repression = RepressionInteraction(
        ...     'repression', products={('mRNA', 'x'): 1}, modifiers={('repressor', 'x'): 1}, alpha=2.0, K=1.0
        ... )
        >>> species = [SpeciesElement('mRNA'), SpeciesElement('repressor', x=1.0)]
        >>> trajectory = simulate(System(elements=species, interactions=[repression]), until=1.0, dt=0.1)
        >>> round(float(trajectory.values('mRNA.x')[-1]), 3)
        1.0
    """

    def __init__(
        self,
        label: str,
        *,
        reactants: Annotated[Stoichiometry | None, RoleSpec(variadic=True)] = None,
        products: Annotated[Stoichiometry | None, RoleSpec(variadic=True)] = None,
        modifiers: Annotated[Stoichiometry | None, RoleSpec(variadic=True)] = None,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        alpha0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        K: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        n: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        mode: str = 'rate',
    ) -> None:
        super().__init__(label, reactants=reactants, products=products, modifiers=modifiers, mode=mode)
        self.alpha = alpha
        self.alpha0 = alpha0
        self.K = K
        self.n = n

    def _get_rate(self, namespace: SimpleNamespace) -> Expression:
        """Return the repressed synthesis rate of the repressor `y`.

        The rate is `alpha0 + alpha (1 - hill(y, K, n))`.

        Args:
            namespace:
                The namespace of the symbols the rate can use, with the namespaces of the modifiers.

        Returns:
            rate:
                The propensity of the synthesis.
        """
        s = namespace
        ((repressor, _),) = self._get_terms(self._get_modifiers(), s, 'modifiers')
        return s.alpha0 + s.alpha * (1 - hill(repressor, s.K, s.n))
