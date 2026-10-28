"""Elementary reactions as rate equations, jump processes, and chemical Langevin equations."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo, sqrt
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.errors import ValidationError
from ....core.interactions import Interaction
from ....core.laws import BaseLaw, JumpLaw, NoiseLaw, RateLaw
from ....core.laws.utils import where
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ....core.system import System
from ....engine.solvers import EulerMaruyamaSolver
from ..._catalog import register_model


def _langevin_laws(propensity: Expression, changes: Mapping[Expression, int]) -> list[BaseLaw]:
    """Return the drift and the noise of one reaction channel, one Wiener process driving every species it changes."""
    amplitude = sqrt(where(propensity > 0, propensity, 0))
    rates: list[BaseLaw] = [RateLaw(target, change * propensity) for target, change in changes.items()]
    return rates + [NoiseLaw(target, change * amplitude, process='channel') for target, change in changes.items()]


def _reaction_laws(
    label: str,
    propensity: Expression,
    changes: Mapping[Expression, int],
    *,
    stochastic: bool,
    langevin: bool,
) -> list[BaseLaw]:
    """Return the laws of one reaction channel, as rates of mass action, a random event, or a Langevin equation."""
    if stochastic and langevin:
        error_msg = (
            f'Reaction {label} is both stochastic and in Langevin mode. Set at most one of `stochastic` and '
            '`langevin`.'
        )
        raise ValidationError(error_msg)
    if stochastic:
        return [JumpLaw(propensity, dict(changes))]
    if langevin:
        return _langevin_laws(propensity, changes)
    return [RateLaw(target, change * propensity) for target, change in changes.items()]


def _schlogl_laws(s: SimpleNamespace) -> list[BaseLaw]:
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
    noises = [NoiseLaw(x, sqrt(where(propensity > 0, propensity, 0))) for propensity in [*increments, *decrements]]
    return [RateLaw(x, drift), *noises]


class SpeciesElement(Element):
    """Amount of one chemical species, a concentration or a number of molecules.

    Args:
        label:
            The label of the species.

        n:
            The initial amount.

    Examples:
        >>> from skcomplex.models.chemistry.kinetics import SpeciesElement
        >>> SpeciesElement('substrate', n=300.0).n
        300.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.n = n


class BindingInteraction(Interaction):
    """Second-order reaction `first + second -> product` with the mass-action rate `k * first * second`.

    In stochastic mode every reaction is a random event that changes the amounts by one molecule. In Langevin mode
    the reaction is a chemical Langevin equation: one Wiener process with the amplitude the square root of the rate
    drives every species the reaction changes.

    Args:
        label:
            The label of the reaction.

        stochastic:
            Whether reactions are random events.

        langevin:
            Whether the reaction is a chemical Langevin equation.

        **values:
            The values of the declared variables, by name:

            first:
                The path of the first reactant.

            second:
                The path of the second reactant.

            product:
                The path of the product.

            k:
                The rate constant.

    Examples:
        >>> from skcomplex.models.chemistry.kinetics import BindingInteraction
        >>> BindingInteraction('binding', first='E', second='S', product='C', k=0.01).k
        0.01
    """

    def __init__(
        self,
        label: str,
        *,
        first: Annotated[str, RoleSpec()],
        second: Annotated[str, RoleSpec()],
        product: Annotated[str, RoleSpec()],
        k: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        stochastic: bool = False,
        langevin: bool = False,
    ) -> None:
        super().__init__(label)
        self.first = first
        self.second = second
        self.product = product
        self.k = k
        self.stochastic = stochastic
        self.langevin = langevin

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the reaction at the rate `k * first * second`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.

        Raises:
            ValidationError:
                The reaction is both stochastic and in Langevin mode.
        """
        changes = {s.first.n: -1, s.second.n: -1, s.product.n: 1}
        propensity = s.k * s.first.n * s.second.n
        return _reaction_laws(self.label, propensity, changes, stochastic=self.stochastic, langevin=self.langevin)


class DissociationInteraction(Interaction):
    """First-order reaction `complex -> first + second` with the mass-action rate `k * complex`.

    In stochastic mode every reaction is a random event that changes the amounts by one molecule. In Langevin mode
    the reaction is a chemical Langevin equation: one Wiener process with the amplitude the square root of the rate
    drives every species the reaction changes.

    Args:
        label:
            The label of the reaction.

        stochastic:
            Whether reactions are random events.

        langevin:
            Whether the reaction is a chemical Langevin equation.

        **values:
            The values of the declared variables, by name:

            complex:
                The path of the dissociating complex.

            first:
                The path of the first product.

            second:
                The path of the second product.

            k:
                The rate constant.

    Examples:
        >>> from skcomplex.models.chemistry.kinetics import DissociationInteraction
        >>> DissociationInteraction('catalysis', complex='C', first='E', second='P', k=0.1).second
        'P'
    """

    def __init__(
        self,
        label: str,
        *,
        complex: Annotated[str, RoleSpec()],  # noqa: A002
        first: Annotated[str, RoleSpec()],
        second: Annotated[str, RoleSpec()],
        k: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        stochastic: bool = False,
        langevin: bool = False,
    ) -> None:
        super().__init__(label)
        self.complex = complex
        self.first = first
        self.second = second
        self.k = k
        self.stochastic = stochastic
        self.langevin = langevin

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the reaction at the rate `k * complex`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.

        Raises:
            ValidationError:
                The reaction is both stochastic and in Langevin mode.
        """
        changes = {s.complex.n: -1, s.first.n: 1, s.second.n: 1}
        propensity = s.k * s.complex.n
        return _reaction_laws(self.label, propensity, changes, stochastic=self.stochastic, langevin=self.langevin)


class SchloglElement(Element):
    """Schlogl element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 50.0,
        c1: Annotated[object, ParameterSpec()] = 1.5e-06,
        c2: Annotated[object, ParameterSpec()] = 0.0025,
        c3: Annotated[object, ParameterSpec()] = 0.0002,
        c4: Annotated[object, ParameterSpec()] = 3.5,
        A: Annotated[object, ParameterSpec()] = 100000.0,
        B: Annotated[object, ParameterSpec()] = 200000.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.A = A
        self.B = B

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the schlogl element."""
        return _schlogl_laws(s)


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
        elements=[
            SpeciesElement('E', n=enzyme),
            SpeciesElement('S', n=substrate),
            SpeciesElement('C'),
            SpeciesElement('P'),
        ],
        interactions=[
            BindingInteraction('binding', first='E', second='S', product='C', k=k_binding, stochastic=stochastic),
            DissociationInteraction(
                'unbinding',
                complex='C',
                first='E',
                second='S',
                k=k_unbinding,
                stochastic=stochastic,
            ),
            DissociationInteraction(
                'catalysis',
                complex='C',
                first='E',
                second='P',
                k=k_catalysis,
                stochastic=stochastic,
            ),
        ],
    )


@register_model(
    field='chemistry/kinetics',
    tags=('sde', 'chemical Langevin equation', 'bistability', 'noise'),
    references=('Schlögl 1972', 'Gillespie 2000', 'Trzaska 2023'),
    defaults={'until': 200.0, 'dt': 0.05, 'solver': EulerMaruyamaSolver(dt=1e-3)},
)
def chemical_langevin_equation(
    x: float = 50.0,
    *,
    c1: float = 1.5e-6,
    c2: float = 2.5e-3,
    c3: float = 2e-4,
    c4: float = 3.5,
    A: float = 1e5,
    B: float = 2e5,
) -> System:
    """Return the chemical Langevin equation of Schlögl's bistable reaction, whose noise switches between two states.

    The channels `A + 2X -> 3X`, `3X -> A + 2X`, `B -> X`, and `X -> B` have the propensities `c1 A x (x - 1) / 2`,
    `c2 x (x - 1) (x - 2) / 6`, `c3 B`, and `c4 x`. Every channel adds its propensity times its change of `x` to the
    drift and drives `x` with its own Wiener process of amplitude the square root of its propensity. The default rate
    constants are those of Gillespie 2000 scaled to a volume five times smaller, so the stable states lie near 17 and
    113 molecules and the noise switches between them within 200 time units.

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
    return System(elements=[SchloglElement('reactor', x=x, c1=c1, c2=c2, c3=c3, c4=c4, A=A, B=B)])


@register_model(
    field='chemistry/kinetics',
    tags=('enzyme kinetics', 'sde', 'chemical Langevin equation', 'noise'),
    references=('Michaelis and Menten 1913', 'Gillespie 2000'),
    defaults={'until': 50.0, 'dt': 0.5, 'solver': EulerMaruyamaSolver(dt=1e-3)},
)
def chemical_langevin_network(
    enzyme: float = 100.0,
    substrate: float = 300.0,
    *,
    k_binding: float = 0.01,
    k_unbinding: float = 0.1,
    k_catalysis: float = 0.1,
) -> System:
    """Return the chemical Langevin equation of the enzyme reaction `E + S <-> C -> E + P`.

    Every reaction adds its rate times its change of every species to the drift, and one Wiener process with the
    amplitude the square root of its rate drives every species it changes. Since one increment changes all of them, the
    total enzyme `E + C` and the total substrate `S + C + P` stay constant under the noise, as in the jump process.

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

    Returns:
        system:
            The model, with the amounts `E.n`, `S.n`, `C.n`, and `P.n`.

    Examples:
        >>> from skcomplex.models.chemistry.kinetics import chemical_langevin_network
        >>> [reaction.langevin for reaction in chemical_langevin_network().interactions]
        [True, True, True]
    """
    return System(
        elements=[
            SpeciesElement('E', n=enzyme),
            SpeciesElement('S', n=substrate),
            SpeciesElement('C'),
            SpeciesElement('P'),
        ],
        interactions=[
            BindingInteraction('binding', first='E', second='S', product='C', k=k_binding, langevin=True),
            DissociationInteraction('unbinding', complex='C', first='E', second='S', k=k_unbinding, langevin=True),
            DissociationInteraction('catalysis', complex='C', first='E', second='P', k=k_catalysis, langevin=True),
        ],
    )
