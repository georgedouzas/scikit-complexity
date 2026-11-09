"""Elementary reactions as rate equations, jumps, and Langevin equations.

The jumps are jump processes and the Langevin equations are chemical Langevin equations.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.solvers import EulerMaruyamaSolver
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ...general.reactions import ReactionInteraction, ReactionNetworkSystem
from ...general.reactions.types import ReactionTable

X = ('reactor', 'x')


def _get_enzyme_reactions(k_binding: float, k_unbinding: float, k_catalysis: float) -> ReactionTable:
    """Return the binding, unbinding, and catalysis of the enzyme reaction.

    The reaction is `E + S <-> C -> E + P`.
    """
    return {
        'binding': ({'E': 1, 'S': 1}, {'C': 1}, k_binding),
        'unbinding': ({'C': 1}, {'E': 1, 'S': 1}, k_unbinding),
        'catalysis': ({'C': 1}, {'E': 1, 'P': 1}, k_catalysis),
    }


class SchloglElement(Element):
    """Reactor of Schlögl's reaction, the number of molecules of `X`.

    The buffered species are `A` and `B`.

    The reactions of `ChemicalLangevinEquationSystem` change `x` and read the rate constants and the constant numbers of
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


class MichaelisMentenSystem(ReactionNetworkSystem):
    """The enzyme reaction `E + S <-> C -> E + P` of Michaelis and Menten.

    It runs as rate equations or as jumps.

    The total enzyme `E + C` and the total substrate `S + C + P` are conserved. In stochastic mode the amounts are
    numbers of molecules and the reactions are the random events of Gillespie's algorithm. The amounts are `E.x`,
    `S.x`, `C.x`, and `P.x`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.chemistry.kinetics import MichaelisMentenSystem
        >>> [element.label for element in MichaelisMentenSystem().prepare().system_.elements]
        ['E', 'S', 'C', 'P']
        >>> sorted(MichaelisMentenSystem(stochastic=True).get_tags().laws)
        ['jump']
    """

    field = 'chemistry/kinetics'
    topics = ('enzyme kinetics', 'ode', 'jump process', 'gillespie')
    references = ('Michaelis and Menten 1913', 'Gillespie 1977', 'Trzaska 2023')
    until = 50.0
    dt = 0.5

    def __init__(
        self,
        enzyme: float = 100.0,
        substrate: float = 300.0,
        *,
        k_binding: float = 0.01,
        k_unbinding: float = 0.1,
        k_catalysis: float = 0.1,
        stochastic: bool = False,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.enzyme = enzyme
        self.substrate = substrate
        self.k_binding = k_binding
        self.k_unbinding = k_unbinding
        self.k_catalysis = k_catalysis
        self.stochastic = stochastic

    def _get_species(self) -> dict[str, float]:
        """Return the initial amounts of the enzyme, substrate, complex, and product."""
        return {'E': self.enzyme, 'S': self.substrate, 'C': 0.0, 'P': 0.0}

    def _get_reactions(self) -> ReactionTable:
        """Return the binding, unbinding, and catalysis of the enzyme reaction."""
        return _get_enzyme_reactions(self.k_binding, self.k_unbinding, self.k_catalysis)

    def _get_mode(self) -> str:
        """Return the mode of the reactions.

        It is random events when stochastic and rate equations otherwise.
        """
        return 'jump' if self.stochastic else 'rate'


class ChemicalLangevinEquationSystem(BaseSystem):
    """The chemical Langevin equation of Schlögl's bistable reaction.

    Its noise switches between two states.

    The channels `A + 2X -> 3X`, `3X -> A + 2X`, `B -> X`, and `X -> B` have the propensities `c1 A x (x - 1) / 2`,
    `c2 x (x - 1) (x - 2) / 6`, `c3 B`, and `c4 x`. Every channel adds its propensity times its change of `x` to the
    drift and drives `x` with its own Wiener process of amplitude the square root of its propensity. The default rate
    constants are those of Gillespie 2000 scaled to a volume five times smaller, so the stable states lie near 17 and
    113 molecules and the noise switches between them within 200 time units. The number of molecules is `reactor.x`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.chemistry.kinetics import ChemicalLangevinEquationSystem
        >>> ChemicalLangevinEquationSystem(c4=3.0).prepare().system_.elements[0].c4
        3.0
    """

    field = 'chemistry/kinetics'
    topics = ('sde', 'chemical Langevin equation', 'bistability', 'noise')
    references = ('Schlögl 1972', 'Gillespie 2000', 'Trzaska 2023')
    until = 200.0
    dt = 0.05
    solver = EulerMaruyamaSolver(dt=1e-3)

    def __init__(
        self,
        x: float = 50.0,
        *,
        c1: float = 1.5e-6,
        c2: float = 2.5e-3,
        c3: float = 2e-4,
        c4: float = 3.5,
        A: float = 1e5,
        B: float = 2e5,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.x = x
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.A = A
        self.B = B

    @staticmethod
    def _get_autocatalysis(s: SimpleNamespace) -> Expression:
        """Return the propensity `c1 A x (x - 1) / 2` of `A + 2X -> 3X`."""
        reactor = s.reactants[0]
        return reactor.c1 / 2 * reactor.A * reactor.x * (reactor.x - 1)

    @staticmethod
    def _get_reversal(s: SimpleNamespace) -> Expression:
        """Return the propensity `c2 x (x - 1) (x - 2) / 6` of `3X -> A + 2X`."""
        reactor = s.reactants[0]
        return reactor.c2 / 6 * reactor.x * (reactor.x - 1) * (reactor.x - 2)

    @staticmethod
    def _get_production(s: SimpleNamespace) -> Expression:
        """Return the propensity `c3 B` of `B -> X`."""
        reactor = s.products[0]
        return reactor.c3 * reactor.B

    @staticmethod
    def _get_conversion(s: SimpleNamespace) -> Expression:
        """Return the propensity `c4 x` of `X -> B`."""
        reactor = s.reactants[0]
        return reactor.c4 * reactor.x

    def _build_system(self) -> CompositeSystem:
        """Return the reactor and the four channels of Schlögl's reaction.

        The channels are chemical Langevin equations.
        """
        reactor = SchloglElement(
            'reactor',
            x=self.x,
            c1=self.c1,
            c2=self.c2,
            c3=self.c3,
            c4=self.c4,
            A=self.A,
            B=self.B,
        )
        return CompositeSystem(
            self.label,
            elements=[reactor],
            interactions=[
                ReactionInteraction(
                    'autocatalysis',
                    reactants={X: 2},
                    products={X: 3},
                    propensity=self._get_autocatalysis,
                    mode='langevin',
                ),
                ReactionInteraction(
                    'reversal',
                    reactants={X: 3},
                    products={X: 2},
                    propensity=self._get_reversal,
                    mode='langevin',
                ),
                ReactionInteraction('production', products={X: 1}, propensity=self._get_production, mode='langevin'),
                ReactionInteraction('conversion', reactants={X: 1}, propensity=self._get_conversion, mode='langevin'),
            ],
        )


class ChemicalLangevinNetworkSystem(ReactionNetworkSystem):
    """The chemical Langevin equation of the enzyme reaction `E + S <-> C -> E + P`.

    Every reaction adds its rate times its change of every species to the drift, and one Wiener process with the
    amplitude the square root of its rate drives every species it changes. Since one increment changes all of them, the
    total enzyme `E + C` and the total substrate `S + C + P` stay constant under the noise, as in the jump process. The
    amounts are `E.x`, `S.x`, `C.x`, and `P.x`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.chemistry.kinetics import ChemicalLangevinNetworkSystem
        >>> [reaction.mode for reaction in ChemicalLangevinNetworkSystem().prepare().system_.interactions]
        ['langevin', 'langevin', 'langevin']
    """

    field = 'chemistry/kinetics'
    topics = ('enzyme kinetics', 'sde', 'chemical Langevin equation', 'noise')
    references = ('Michaelis and Menten 1913', 'Gillespie 2000')
    until = 50.0
    dt = 0.5
    solver = EulerMaruyamaSolver(dt=1e-3)

    def __init__(
        self,
        enzyme: float = 100.0,
        substrate: float = 300.0,
        *,
        k_binding: float = 0.01,
        k_unbinding: float = 0.1,
        k_catalysis: float = 0.1,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.enzyme = enzyme
        self.substrate = substrate
        self.k_binding = k_binding
        self.k_unbinding = k_unbinding
        self.k_catalysis = k_catalysis

    def _get_species(self) -> dict[str, float]:
        """Return the initial amounts of the enzyme, substrate, complex, and product."""
        return {'E': self.enzyme, 'S': self.substrate, 'C': 0.0, 'P': 0.0}

    def _get_reactions(self) -> ReactionTable:
        """Return the binding, unbinding, and catalysis of the enzyme reaction."""
        return _get_enzyme_reactions(self.k_binding, self.k_unbinding, self.k_catalysis)

    def _get_mode(self) -> str:
        """Return the mode of the reactions, the chemical Langevin equation."""
        return 'langevin'
