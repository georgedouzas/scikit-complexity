"""Knock-outs of the E. coli core system, whose analyses COBRApy and StrainDesign run."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

from ....core.components.spaces import StaticTime
from ....core.components.system import BaseSystem, System
from ..common.metabolism import load_network


class MetabolicPerturbationsSystem(BaseSystem):
    """The E. coli core system with some genes knocked out, which needs the extra `metabolism`.

    A gene that is off closes the reactions whose rules need it (Raman 2021, Section 9.1). `cobra_measures` scans the
    deletions and adjusts the fluxes by MoMA and ROOM, `strain_design` finds OptKnock designs, and
    `production_scan` scans fluxes with an enforced product.

    Args:
        genes:
            The names of the genes to knock out.

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        MissingExtraError:
            The extra is not installed.

    Examples:
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.systems.biology.systems_biology import MetabolicPerturbationsSystem
        >>> trajectory = simulate(MetabolicPerturbationsSystem(genes=('b4025',)), until=0.0)
        >>> round(float(trajectory.values('cell.objective_0')[0]), 3)
        0.863
    """

    field = 'biology/systems_biology'
    topics = ('flux balance', 'gene knock-out', 'linear program', 'metabolism', 'static')
    references = ('Raman 2021', 'Segre et al. 2002', 'Shlomi et al. 2005', 'Burgard et al. 2003')
    until = 0.0

    def __init__(
        self,
        genes: Sequence[str] = (),
        *,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.genes = genes

    def _build_system(self) -> System:
        """Return the parts of the E. coli core system with some genes knocked out."""
        element = load_network('textbook', 'cell', needed_by='metabolic_perturbations')
        element.set_params(**{element.gene_name(gene): 0.0 for gene in self.genes})
        return System(self.label, elements=[element], time=StaticTime())
