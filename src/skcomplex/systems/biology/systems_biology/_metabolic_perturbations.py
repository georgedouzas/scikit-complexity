"""Knock-outs of the E. coli core system.

COBRApy and StrainDesign run their analyses.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

from ....core.components import Observable
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ..common.metabolism import load_network


class MetabolicPerturbationsSystem(BaseSystem):
    """The E. coli core system with some genes knocked out.

    It needs the extra `metabolism`.

    A gene that is off closes the reactions whose rules need it (Raman 2021, Section 9.1). `cobra_measures` scans the
    deletions and adjusts the fluxes by MoMA and ROOM, `strain_design` finds OptKnock designs, and
    `production_scan` scans fluxes with an enforced product.

    Args:
        genes:
            The names of the genes to knock out.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        MissingExtraError:
            The extra is not installed.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import MetabolicPerturbationsSystem
        >>> trajectory = MetabolicPerturbationsSystem(genes=('b4025',)).prepare().simulate(until=0.0)
        >>> round(float(trajectory.get_values('cell.objective_0')[0]), 3)
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.genes = genes

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the E. coli core system with some genes knocked out.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        element = load_network('textbook', 'cell', needed_by='metabolic_perturbations')
        element.set_params(**{element.get_gene_name(gene): 0.0 for gene in self.genes})
        self.system_ = System(self.label, elements=[element])
