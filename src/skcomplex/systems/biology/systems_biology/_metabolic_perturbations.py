"""Knock-outs of the E. coli core system, whose analyses COBRApy and StrainDesign run."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

from ....core.components.spaces import StaticTime
from ....core.components.system import System
from ..._catalog import register_system
from ..common.metabolism import load_network


@register_system(
    field='biology/systems_biology',
    tags=('flux balance', 'gene knock-out', 'linear program', 'metabolism', 'static'),
    references=('Raman 2021', 'Segre et al. 2002', 'Shlomi et al. 2005', 'Burgard et al. 2003'),
    defaults={'until': 0.0},
)
def metabolic_perturbations(genes: Sequence[str] = ()) -> System:
    """Return the E. coli core system with some genes knocked out, which needs the extra `metabolism`.

    A gene that is off closes the reactions whose rules need it (Raman 2021, Section 9.1). `cobra_measures` scans the
    deletions and adjusts the fluxes by MoMA and ROOM, `strain_design` finds OptKnock designs, and
    `production_scan` scans fluxes with an enforced product.

    Args:
        genes:
            The names of the genes to knock out.

    Returns:
        system:
            The system.

    Raises:
        MissingExtraError:
            The extra is not installed.

    Examples:
        >>> from skcomplex.core.engine.runtime import Simulation
        >>> from skcomplex.systems.biology.systems_biology import metabolic_perturbations
        >>> system = metabolic_perturbations(genes=('b4025',))
        >>> trajectory = Simulation(system).prepare().run(0.0).trajectory_
        >>> round(float(trajectory.values('cell.objective_0')[0]), 3)
        0.863
    """
    element = load_network('textbook', 'cell', needed_by='metabolic_perturbations')
    element.set_params(**{element.gene_name(gene): 0.0 for gene in genes})
    return System(elements=[element], time=StaticTime())
