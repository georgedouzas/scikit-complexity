"""Elementary flux modes and extreme pathways of a metabolic network.

The network is Raman's toy network by default.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

from ....core.components import Observable
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ..common.metabolism import ToyNetworkElement


class ElementaryFluxModesSystem(BaseSystem):
    """The toy network of Raman whose steady-state cone `elementary_modes` enumerates.

    Every steady-state flux of the network is a non-negative combination of the elementary modes of its cone (Raman
    2021, Section 9.8). The toy network has two modes, through `v1` and through `v2` and `v3`, both with growth yield
    `1/12` on `A`, so the bound 1000 on `v6` limits the growth to 50 whichever route carries the flux.

    Args:
        knockouts:
            The reactions to fix at zero.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.core.analysis import elementary_modes
        >>> from skcomplex.systems.biology.systems_biology import ElementaryFluxModesSystem
        >>> analysis = elementary_modes(ElementaryFluxModesSystem().prepare().system_, normalize='cell.v_b1')
        >>> analysis.modes.shape[0]
        2
    """

    field = 'biology/systems_biology'
    topics = ('elementary flux modes', 'extreme pathways', 'linear program', 'metabolism', 'static')
    references = ('Raman 2021', 'Schuster and Hilgetag 1994', 'Schilling et al. 2000', 'Gagneur and Klamt 2004')
    until = 0.0

    def __init__(
        self,
        knockouts: Sequence[str] = (),
        *,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.knockouts = knockouts

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the toy network of Raman.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        closed = {f'{bound}_{name}': 0.0 for name in self.knockouts for bound in ('lower', 'upper')}
        self.system_ = System(self.label, elements=[ToyNetworkElement('cell').set_params(**closed)])
