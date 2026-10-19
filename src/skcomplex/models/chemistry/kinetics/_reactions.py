"""Elementary reactions as rate equations, jump processes, and chemical Langevin equations."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping

from sage.all__sagemath_symbolics import sqrt

from ....core.base import Domain, Expression, parameter, role, state
from ....core.model import Element, Interaction, Jump, Law, Noise, Rate, Symbols, System, where
from ....core.runtime import register_model
from ....core.solvers import EulerMaruyama


def _reaction_laws(propensity: Expression, changes: Mapping[Expression, int], *, stochastic: bool) -> list[Law]:
    """Return the laws of one reaction channel, as rates of mass action or as a random event."""
    if stochastic:
        return [Jump(propensity, dict(changes))]
    return [Rate(target, change * propensity) for target, change in changes.items()]


def _schlogl_laws(s: Symbols) -> list[Law]:
    """Return the drift and the independent noise of the four channels of the Schlögl reaction."""
    x = s.x
    increments = [
        s.c1 / 2 * s.A * x * (x - 1),
        s.c3 * s.B,
    ]
    decrements = [
        s.c2 / 6 * x * (x - 1) * (x - 2),
        s.c4 * x,
    ]
    drift = sum(increments) - sum(decrements)
    noises = [Noise(x, sqrt(where(propensity > 0, propensity, 0))) for propensity in [*increments, *decrements]]
    return [Rate(x, drift), *noises]


class Species(Element):
    """Amount of one chemical species, a concentration or a number of molecules.

    Args:
        label:
            The label of the species.

        n:
            The initial amount.

    Examples:
        >>> from skcomplex.models.chemistry.kinetics import Species
        >>> Species('substrate', n=300.0).n
        300.0
    """

    n: float = state(default=0.0, domain=Domain.NONNEGATIVE)


class Binding(Interaction):
    """Second-order reaction `first + second -> product` with the mass-action rate `k * first * second`.

    In stochastic mode every reaction is a random event that changes the amounts by one molecule.

    Args:
        label:
            The label of the reaction.

        first:
            The path of the first reactant.

        second:
            The path of the second reactant.

        product:
            The path of the product.

        k:
            The rate constant.

        stochastic:
            Whether reactions are random events.

    Examples:
        >>> from skcomplex.models.chemistry.kinetics import Binding
        >>> Binding('binding', first='E', second='S', product='C', k=0.01).k
        0.01
    """

    first: str = role()
    second: str = role()
    product: str = role()
    k: float = parameter(default=0.01, domain=Domain.NONNEGATIVE)
    stochastic: bool = False

    def laws(self, s: Symbols) -> list[Law]:
        """Return the reaction at the rate `k * first * second`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        changes = {s.first.n: -1, s.second.n: -1, s.product.n: 1}
        return _reaction_laws(s.k * s.first.n * s.second.n, changes, stochastic=self.stochastic)


class Dissociation(Interaction):
    """First-order reaction `complex -> first + second` with the mass-action rate `k * complex`.

    In stochastic mode every reaction is a random event that changes the amounts by one molecule.

    Args:
        label:
            The label of the reaction.

        complex:
            The path of the dissociating complex.

        first:
            The path of the first product.

        second:
            The path of the second product.

        k:
            The rate constant.

        stochastic:
            Whether reactions are random events.

    Examples:
        >>> from skcomplex.models.chemistry.kinetics import Dissociation
        >>> Dissociation('catalysis', complex='C', first='E', second='P', k=0.1).second
        'P'
    """

    complex: str = role()
    first: str = role()
    second: str = role()
    k: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)
    stochastic: bool = False

    def laws(self, s: Symbols) -> list[Law]:
        """Return the reaction at the rate `k * complex`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        changes = {s.complex.n: -1, s.first.n: 1, s.second.n: 1}
        return _reaction_laws(s.k * s.complex.n, changes, stochastic=self.stochastic)


Schlogl = Element.define(
    'Schlogl',
    states={'x': state(default=250.0, domain=Domain.REAL)},
    parameters={'c1': 3e-7, 'c2': 1e-4, 'c3': 1e-3, 'c4': 3.5, 'A': 1e5, 'B': 2e5},
    laws=_schlogl_laws,
)


@register_model(
    field='chemistry/kinetics',
    tags=('enzyme kinetics', 'ode', 'jump process', 'gillespie'),
    references=('Michaelis and Menten 1913', 'Gillespie 1977', 'Trzaska 2023'),
    defaults={'until': 50.0, 'dt': 0.5},
)
def michaelis_menten(
    enzyme: float = 100.0,
    substrate: float = 300.0,
    *,
    k_binding: float = 0.01,
    k_unbinding: float = 0.1,
    k_catalysis: float = 0.1,
    stochastic: bool = False,
) -> System:
    """Return the enzyme reaction `E + S <-> C -> E + P` of Michaelis and Menten, as rate equations or as jumps.

    The total enzyme `E + C` and the total substrate `S + C + P` are conserved. In stochastic mode the amounts are
    numbers of molecules and the reactions are the random events of Gillespie's algorithm.

    Args:
        enzyme:
            The initial amount of free enzyme.

        substrate:
            The initial amount of substrate.

        k_binding:
            The rate constant of the binding of enzyme and substrate.

        k_unbinding:
            The rate constant of the unbinding of the complex.

        k_catalysis:
            The rate constant of the conversion of the complex into enzyme and product.

        stochastic:
            Whether the reactions are random events.

    Returns:
        system:
            The model, with the amounts `E.n`, `S.n`, `C.n`, and `P.n`.

    Examples:
        >>> from skcomplex.models.chemistry.kinetics import michaelis_menten
        >>> [element.label for element in michaelis_menten().elements]
        ['E', 'S', 'C', 'P']
    """
    return System(
        elements=[Species('E', n=enzyme), Species('S', n=substrate), Species('C'), Species('P')],
        interactions=[
            Binding('binding', first='E', second='S', product='C', k=k_binding, stochastic=stochastic),
            Dissociation('unbinding', complex='C', first='E', second='S', k=k_unbinding, stochastic=stochastic),
            Dissociation('catalysis', complex='C', first='E', second='P', k=k_catalysis, stochastic=stochastic),
        ],
    )


@register_model(
    field='chemistry/kinetics',
    tags=('sde', 'chemical Langevin equation', 'bistability', 'noise'),
    references=('Schlögl 1972', 'Gillespie 2000', 'Trzaska 2023'),
    defaults={'until': 5.0, 'dt': 0.05, 'solver': EulerMaruyama(dt=1e-3)},
)
def chemical_langevin_equation(
    x: float = 250.0,
    *,
    c1: float = 3e-7,
    c2: float = 1e-4,
    c3: float = 1e-3,
    c4: float = 3.5,
    A: float = 1e5,
    B: float = 2e5,
) -> System:
    """Return the chemical Langevin equation of Schlögl's bistable reaction, whose noise switches between two states.

    The channels `A + 2X -> 3X`, `3X -> A + 2X`, `B -> X`, and `X -> B` have the propensities `c1 A x (x - 1) / 2`,
    `c2 x (x - 1) (x - 2) / 6`, `c3 B`, and `c4 x`. Every channel adds its propensity times its change of `x` to the
    drift and drives `x` with its own Wiener process of amplitude the square root of its propensity.

    Args:
        x:
            The initial number of molecules of `X`.

        c1:
            The rate constant of the autocatalytic production.

        c2:
            The rate constant of the reverse of the autocatalytic production.

        c3:
            The rate constant of the production from `B`.

        c4:
            The rate constant of the conversion into `B`.

        A:
            The constant number of molecules of `A`.

        B:
            The constant number of molecules of `B`.

    Returns:
        system:
            The model, with the number of molecules `reactor.x`.

    Examples:
        >>> from skcomplex.models.chemistry.kinetics import chemical_langevin_equation
        >>> chemical_langevin_equation(c4=3.0).elements[0].c4
        3.0
    """
    return System(elements=[Schlogl('reactor', x=x, c1=c1, c2=c2, c3=c3, c4=c4, A=A, B=B)])
