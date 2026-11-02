"""Share of the population with the fittest genome, in a sharp-peak landscape where every other genome is less fit."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....specs import ParameterSpec, VisualSpec
from ._genotype_class import GenotypeClassElement


class MasterSequenceElement(GenotypeClassElement):
    """Share of the population with the fittest genome, in a sharp-peak landscape where every other genome is less fit.

    The master sequence has fitness 1 and every other genome `1 - sigma`. A genome mutates at one locus with
    probability `u` per generation. The master sequence holds the parameters of the landscape that the maps of every
    class read.

    Args:
        label:
            The label of the class.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial share.

        u:
            The mutation rate per genome and generation.

        sigma:
            The selective disadvantage of the other genomes.

    Examples:
        >>> from skcomplex.models.biology.evolution import MasterSequenceElement
        >>> MasterSequenceElement('master', u=0.2).u
        0.2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: float = 0.0,
        u: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        sigma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual, x=x)
        self.u = u
        self.sigma = sigma
