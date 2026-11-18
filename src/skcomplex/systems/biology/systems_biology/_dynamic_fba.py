"""Dynamic flux balance analysis of a batch culture of the E. coli core system."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from collections.abc import Sequence

from ....core.components import Observable
from ....core.components.spaces import DiscreteTime
from ....core.components.systems import CompositeSystem
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, CoSimulationSolver, RK4Solver
from ....core.errors import ValidationError
from ....core.systems import BaseSystem, System
from ..common.metabolism import BiomassElement, ExchangeInteraction, GrowthInteraction, SubstrateElement, load_network

APPROACHES = ('static', 'direct')
GROWTH = 'Biomass_Ecoli_core'
GLUCOSE_EXCHANGE = 'EX_glc__D_e'
SYNCHRONISATION = 0.05
ODE_STEP = 0.005


class DynamicFBASystem(BaseSystem):
    """A batch culture of the E. coli core system.

    It needs the extra `metabolism`.

    The program solves for the fluxes with Michaelis-Menten glucose uptake and then the smallest total flux. By the
    static optimisation approach it solves in a discrete partition at every synchronisation point, and the biomass and
    the glucose integrate with the fluxes held (Raman 2021, Eqs. 9.8 and 9.9). By the direct approach it solves at every
    evaluation of the drift. Without glucose the program is infeasible, and the policy `'zero'` stops growth, so biomass
    grows in one phase and stays.

    Args:
        approach:
            `'static'` for the static optimisation approach or `'direct'` for the direct approach.

        biomass:
            The initial biomass in gDW per L.

        glucose:
            The initial glucose in mM.

        vmax:
            The largest glucose uptake in mmol per gDW per h.

        km:
            The half-saturation concentration of glucose in mM.

        interval:
            The synchronisation interval of the static optimisation approach in h.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ValidationError:
            The approach is unknown.

        MissingExtraError:
            The extra is not installed.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import DynamicFBASystem
        >>> [system.label for system in DynamicFBASystem().prepare().system_.systems]
        ['fba', 'culture']
    """

    field = 'biology/systems_biology'
    topics = ('dynamic flux balance', 'linear program', 'metabolism', 'batch culture', 'co-simulation')
    references = ('Raman 2021', 'Mahadevan et al. 2002', 'Varma and Palsson 1994')
    until = 10.0
    dt = SYNCHRONISATION

    def __init__(
        self,
        approach: str = 'static',
        *,
        biomass: float = 0.01,
        glucose: float = 10.0,
        vmax: float = 10.0,
        km: float = 0.015,
        interval: float = SYNCHRONISATION,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.approach = approach
        self.biomass = biomass
        self.glucose = glucose
        self.vmax = vmax
        self.km = km
        self.interval = interval

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a batch culture of the E. coli core system.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        if self.approach not in APPROACHES:
            error_msg = f'Unknown approach {self.approach!r}. Use one of {", ".join(APPROACHES)}.'
            raise ValidationError(error_msg)
        cell = load_network('textbook', 'cell', needed_by='dynamic_fba')
        cell.parsimonious, cell.infeasible = True, 'zero'
        amounts = [BiomassElement('biomass', x=self.biomass), SubstrateElement('glucose', c=self.glucose)]
        static = self.approach == 'static'
        cell_path = 'fba/cell' if static else 'cell'
        biomass_path = 'culture/biomass' if static else 'biomass'
        interactions = [
            ExchangeInteraction(
                'uptake',
                cell=cell_path,
                biomass=biomass_path,
                substrate='culture/glucose' if static else 'glucose',
                reaction=GLUCOSE_EXCHANGE,
                vmax=self.vmax,
                km=self.km,
                interval=self.interval if static else None,
            ),
            GrowthInteraction('growth', cell=cell_path, biomass=biomass_path, reaction=GROWTH),
        ]
        if static:
            self.system_ = System(
                self.label,
                systems=[
                    CompositeSystem('fba', elements=[cell], time=DiscreteTime()),
                    CompositeSystem('culture', elements=amounts),
                ],
                interactions=interactions,
                solver=CoSimulationSolver(sync_dt=self.interval, solvers={'culture': RK4Solver(dt=ODE_STEP)}),
            )
            return
        self.system_ = System(
            self.label,
            elements=[cell, *amounts],
            interactions=interactions,
            solver=RK4Solver(dt=ODE_STEP),
        )
