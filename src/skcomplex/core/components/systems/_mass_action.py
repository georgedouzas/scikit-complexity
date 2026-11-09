"""Define mass-action systems.

A mass-action system holds species in compartments and the reactions between them at mass-action rates.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence

from ...types import ReactionTable
from ..elements import NonnegativeElement
from ..interactions import ReactionInteraction
from ._base import BaseSystem
from ._system import System


class MassActionSystem(BaseSystem[System]):
    """Species in compartments and the reactions between them at mass-action rates.

    Every species is a `NonnegativeElement` named after it, whose state `x` holds its amount. Every reaction is a
    `ReactionInteraction` with a stoichiometric coefficient for each of its reactants and products and a rate constant
    `k`. Its propensity is `k` times the amount of every reactant to the power of its coefficient, and it changes the
    amount of every species by its coefficient among the products minus its coefficient among the reactants. So
    `S + I -> 2 I` at the rate constant `beta` has the reactants `{'S': 1, 'I': 1}`, the products `{'I': 2}`, and the
    propensity `beta S I`. A species among both the reactants and the products with one coefficient, such as a
    catalyst, enters the propensity and does not change.

    In the mode `'rate'` the reactions are the rate equations of the amounts, in the mode `'jump'` they are the random
    events of a continuous-time Markov chain, run by Gillespie's algorithm, and in the mode `'langevin'` they are the
    chemical Langevin equations.

    Compartmental systems of every field are mass-action systems, such as epidemics of susceptible, infected, and
    recovered individuals or enzyme kinetics. A subclass takes the parameters of its model under its own names, and
    gives its species, its reactions, and its mode from the hooks `_get_species`, `_get_reactions`, and `_get_mode`, or
    its reactions as interactions from `_get_interactions`, when they are subclasses of `ReactionInteraction` with
    roles of their own names.

    Attributes:
        system_:
            The composite of the compartments and the reactions built from the parameters at the last `prepare`.

        laws_:
            The laws the system contributes for the namespace of its last run.

    Examples:
        The reversible isomerization `A <-> B`, whose amounts relax to `k_back / (k_forth + k_back)` of the total:

        >>> from skcomplex.core.components.systems import MassActionSystem
        >>> from skcomplex.core.engine.runtime import simulate
        >>> isomerization = MassActionSystem(
        ...     species={'A': 10.0, 'B': 0.0},
        ...     reactions={'forth': ({'A': 1}, {'B': 1}, 0.3), 'back': ({'B': 1}, {'A': 1}, 0.1)},
        ... )
        >>> [element.label for element in isomerization.prepare().system_.elements]
        ['A', 'B']
        >>> trajectory = simulate(isomerization, until=50.0, dt=1.0)
        >>> round(float(trajectory.values('A.x')[-1]), 3)
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

    def _build_system(self) -> System:
        """Return a compartment for every species, and the reactions between them.

        Returns:
            system:
                The composite of the compartments and the reactions.
        """
        return System(
            self.label,
            elements=[NonnegativeElement(name, x=amount) for name, amount in self._get_species().items()],
            interactions=list(self._get_interactions()),
        )
