"""Species of an NKCS ecosystem, a genome whose fitness depends on its own genes and on genes of other species."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec
from ._coevolving_species_view import CoevolvingSpeciesView


class CoevolvingSpeciesElement(Element):
    """Species of an NKCS ecosystem, a genome whose fitness depends on its own genes and on genes of other species.

    A `CoevolutionaryClimbingInteraction` mutates the genome and sets the fitness and the number of accepted moves of
    the last step.

    Args:
        label:
            The label of the species or of the group of species.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        genome:
            The bits of the genome, as an integer.

        fitness:
            The fitness of the genome against the current genomes of the other species.

        moves:
            Whether the species accepted a mutation in the last step.

    Examples:
        >>> from skcomplex.systems.biology.evolution import CoevolvingSpeciesElement
        >>> CoevolvingSpeciesElement('species', genome=5.0).genome
        5.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        genome: Annotated[float, StateSpec(domain=NN)] = 0.0,
        fitness: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        moves: Annotated[float, StateSpec(domain=NN)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.genome = genome
        self.fitness = fitness
        self.moves = moves

    @classmethod
    def _get_view(cls) -> type[CoevolvingSpeciesView]:
        """Return the view of a species, `CoevolvingSpeciesView`."""
        return CoevolvingSpeciesView
