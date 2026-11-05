"""Elementary reactions as rate equations, jump processes, and chemical Langevin equations."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import CompartmentElement, Element
from ....core.components.interactions import ReactionInteraction
from ....core.components.system import System
from ....core.engine.solvers import EulerMaruyamaSolver
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_system

X = ('reactor', 'x')


def _get_enzyme_reactions(
    k_binding: float,
    k_unbinding: float,
    k_catalysis: float,
    mode: str,
) -> list[ReactionInteraction]:
    """Return the binding, unbinding, and catalysis of the enzyme reaction `E + S <-> C -> E + P` in a mode."""
    enzyme, substrate, complex_, product = ('E', 'n'), ('S', 'n'), ('C', 'n'), ('P', 'n')
    return [
        ReactionInteraction(
            'binding',
            reactants={enzyme: 1, substrate: 1},
            products={complex_: 1},
            k=k_binding,
            mode=mode,
        ),
        ReactionInteraction(
            'unbinding',
            reactants={complex_: 1},
            products={enzyme: 1, substrate: 1},
            k=k_unbinding,
            mode=mode,
        ),
        ReactionInteraction(
            'catalysis',
            reactants={complex_: 1},
            products={enzyme: 1, product: 1},
            k=k_catalysis,
            mode=mode,
        ),
    ]


def _get_autocatalysis(s: SimpleNamespace) -> Expression:
    """Return the propensity `c1 A x (x - 1) / 2` of `A + 2X -> 3X`."""
    reactor = s.reactants[0]
    return reactor.c1 / 2 * reactor.A * reactor.x * (reactor.x - 1)


def _get_reversal(s: SimpleNamespace) -> Expression:
    """Return the propensity `c2 x (x - 1) (x - 2) / 6` of `3X -> A + 2X`."""
    reactor = s.reactants[0]
    return reactor.c2 / 6 * reactor.x * (reactor.x - 1) * (reactor.x - 2)


def _get_production(s: SimpleNamespace) -> Expression:
    """Return the propensity `c3 B` of `B -> X`."""
    reactor = s.products[0]
    return reactor.c3 * reactor.B


def _get_conversion(s: SimpleNamespace) -> Expression:
    """Return the propensity `c4 x` of `X -> B`."""
    reactor = s.reactants[0]
    return reactor.c4 * reactor.x


class SchloglElement(Element):
    """Reactor of Schlögl's reaction, the number of molecules of `X` with the buffered species `A` and `B`.

    The reactions of `chemical_langevin_equation` change `x` and read the rate constants and the constant numbers of
    molecules of `A` and `B` from the reactor.

    Args:
        label:
            The label of the reactor.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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

    Examples:
        >>> from skcomplex.systems.chemistry.kinetics import SchloglElement
        >>> SchloglElement('reactor').x
        50.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 50.0,
        c1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.5e-06,
        c2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0025,
        c3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0002,
        c4: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3.5,
        A: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 100000.0,
        B: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 200000.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.A = A
        self.B = B


@register_system(
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
            The system, with the amounts `E.n`, `S.n`, `C.n`, and `P.n`.

    Examples:
        >>> from skcomplex.systems.chemistry.kinetics import michaelis_menten
        >>> [element.label for element in michaelis_menten().elements]
        ['E', 'S', 'C', 'P']
    """
    return System(
        elements=[
            CompartmentElement('E', n=enzyme),
            CompartmentElement('S', n=substrate),
            CompartmentElement('C'),
            CompartmentElement('P'),
        ],
        interactions=_get_enzyme_reactions(k_binding, k_unbinding, k_catalysis, 'jump' if stochastic else 'rate'),
    )


@register_system(
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
            The system, with the number of molecules `reactor.x`.

    Examples:
        >>> from skcomplex.systems.chemistry.kinetics import chemical_langevin_equation
        >>> chemical_langevin_equation(c4=3.0).elements[0].c4
        3.0
    """
    return System(
        elements=[SchloglElement('reactor', x=x, c1=c1, c2=c2, c3=c3, c4=c4, A=A, B=B)],
        interactions=[
            ReactionInteraction(
                'autocatalysis',
                reactants={X: 2},
                products={X: 3},
                propensity=_get_autocatalysis,
                mode='langevin',
            ),
            ReactionInteraction(
                'reversal',
                reactants={X: 3},
                products={X: 2},
                propensity=_get_reversal,
                mode='langevin',
            ),
            ReactionInteraction('production', products={X: 1}, propensity=_get_production, mode='langevin'),
            ReactionInteraction('conversion', reactants={X: 1}, propensity=_get_conversion, mode='langevin'),
        ],
    )


@register_system(
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
            The system, with the amounts `E.n`, `S.n`, `C.n`, and `P.n`.

    Examples:
        >>> from skcomplex.systems.chemistry.kinetics import chemical_langevin_network
        >>> [reaction.mode for reaction in chemical_langevin_network().interactions]
        ['langevin', 'langevin', 'langevin']
    """
    return System(
        elements=[
            CompartmentElement('E', n=enzyme),
            CompartmentElement('S', n=substrate),
            CompartmentElement('C'),
            CompartmentElement('P'),
        ],
        interactions=_get_enzyme_reactions(k_binding, k_unbinding, k_catalysis, 'langevin'),
    )
