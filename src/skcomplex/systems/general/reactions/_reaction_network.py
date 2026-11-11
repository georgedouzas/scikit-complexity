"""Define the system of species and the reactions between them."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence

from ....core.components.systems import BaseSystem, CompositeSystem
from ..elements import NonnegativeElement
from ._reaction import ReactionInteraction
from .types import ReactionTable


class ReactionNetworkSystem(BaseSystem[CompositeSystem]):
    """A system of amounts and the reactions between them, written as a table.

    Every species is a `NonnegativeElement` named after it, whose state `x` holds its amount. Every reaction is a
    `ReactionInteraction` with a coefficient for each reactant and product and a rate constant `k`. Its propensity is
    `k` times the product of the reactant amounts, each raised to its coefficient. So `S + I -> 2 I` at the rate
    constant `beta` has the reactants `{'S': 1, 'I': 1}`, the products `{'I': 2}`, and the propensity `beta S I`. A
    species that is both a reactant and a product with one coefficient, such as a catalyst, enters the propensity and
    does not change.

    In the mode `'rate'` the reactions are rate equations, in the mode `'jump'` the random events of a Markov chain in
    continuous time, run by the algorithm of Gillespie, and in the mode `'langevin'` the chemical Langevin equations. A
    subclass takes the parameters of its model under its own names and gives its species, reactions, and mode from the
    hooks `_get_species`, `_get_reactions`, and `_get_mode`, or its reactions as interactions from `_get_interactions`.

    Unlike `CompositeSystem`, which takes any components, it builds `NonnegativeElement` and `ReactionInteraction`
    components from a table of reactions. Choose it when a model is a list of reactions with rate constants.

    It appears in many fields. Epidemics of susceptible, infected, and recovered people, enzyme kinetics of Michaelis
    and Menten, and predator and prey cycles of Lotka and Volterra are reaction networks.

    Attributes:
        system_:
            The composite of the species and the reactions built from the parameters at the last `prepare`.

        laws_:
            The laws the system contributes for the namespace of its last run.

    Examples:
        The reversible isomerization `A <-> B`, whose amounts relax to `k_back / (k_forth + k_back)` of the total:

        >>> from skcomplex.systems.general.reactions import ReactionNetworkSystem
        >>> from skcomplex.core.engine.runtime import simulate
        >>> isomerization = ReactionNetworkSystem(
        ...     species={'A': 10.0, 'B': 0.0},
        ...     reactions={'forth': ({'A': 1}, {'B': 1}, 0.3), 'back': ({'B': 1}, {'A': 1}, 0.1)},
        ... )
        >>> [element.label for element in isomerization.prepare().system_.elements]
        ['A', 'B']
        >>> trajectory = simulate(isomerization, until=50.0, dt=1.0)
        >>> round(float(trajectory.get_values('A.x')[-1]), 3)
        2.5

        The same reactions as jumps, whose tags are those of a stochastic system:

        >>> tags = isomerization.set_params(mode='jump').get_tags()
        >>> sorted(tags.laws), tags.randomness
        (['jump'], 'stochastic')
    """

    def __init__(
        self,
        species: Mapping[str, float] | None = None,
        reactions: ReactionTable | None = None,
        *,
        mode: str = 'rate',
        label: str = 'system',
    ) -> None:
        """Store the species, the reactions, the mode, and the label of the system.

        Args:
            species:
                The initial amount of every species, by name, or `None` for no species.

            reactions:
                The reactants, the products, and the rate constant of every reaction, by label, the reactants and the
                products as coefficients by species name, empty for a source or a sink, or `None` for no reactions.

            mode:
                The form of the laws of the reactions, `'rate'`, `'jump'`, or `'langevin'`.

            label:
                The label of the system, which names it when it is nested in another one.
        """
        super().__init__(label)
        self.species = species
        self.reactions = reactions
        self.mode = mode

    def _get_species(self) -> Mapping[str, float]:
        """Return the initial amount of every species by name.

        A subclass with parameters of its own returns it.
        """
        return self.species or {}

    def _get_reactions(self) -> ReactionTable:
        """Return the reactants, the products, and the rate constant of every reaction.

        A subclass returns them.
        """
        return self.reactions or {}

    def _get_mode(self) -> str:
        """Return the form of the laws of the reactions.

        A subclass that sets it by another parameter returns it.
        """
        return self.mode

    def _get_interactions(self) -> Sequence[ReactionInteraction]:
        """Return a reaction for every row of the reactions.

        The reactions act between the states `x` of the species they name. A subclass whose reactions are
        subclasses of `ReactionInteraction` with roles of their own names returns them.

        Returns:
            interactions:
                The reactions, in the order of their rows.
        """
        mode = self._get_mode()
        return [
            ReactionInteraction(
                name,
                reactants={(species, 'x'): coefficient for species, coefficient in reactants.items()} or None,
                products={(species, 'x'): coefficient for species, coefficient in products.items()} or None,
                k=k,
                mode=mode,
            )
            for name, (reactants, products, k) in self._get_reactions().items()
        ]

    def _build_system(self) -> CompositeSystem:
        """Return an element for every species, and the reactions between them.

        Returns:
            system:
                The composite of the species and the reactions.
        """
        return CompositeSystem(
            self.label,
            elements=[NonnegativeElement(name, x=amount) for name, amount in self._get_species().items()],
            interactions=list(self._get_interactions()),
        )
