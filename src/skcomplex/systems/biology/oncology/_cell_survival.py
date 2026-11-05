"""Static systems of the survival of irradiated cells."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.components.spaces import StaticTime
from ....core.components.system import System
from ..._catalog import register_system
from ._linear_quadratic_survival import LinearQuadraticSurvivalElement


@register_system(
    field='biology/oncology',
    tags=('radiotherapy', 'static', 'dose response'),
    references=('Padmanabhan, Meskin, and Al Moustafa 2021', 'Fowler 1989'),
)
def lq_cell_survival(
    dose: float = 2.0,
    *,
    fractions: float = 1.0,
    alpha: float = 0.3,
    beta: float = 0.03,
    repair_rate: float = 0.46,
    delivery_time: float = 1e-3,
) -> System:
    """Return the linear-quadratic survival of cells after a total dose, with its biologically effective dose.

    The survival is `exp(-alpha D - beta G D^2)` (Padmanabhan et al. 2021 Eqs. 6.6-6.8), and the Lea-Catcheside factor
    `G` holds the repair between and during fractions. A sweep of the dose through `Experiment` gives the survival
    curve, and 30 fractions of 2 Gy at `alpha / beta = 10` give the biologically effective dose of 72 Gy.

    Args:
        dose:
            The total dose, in Gy.

        fractions:
            The number of fractions.

        alpha:
            The linear coefficient, per Gy.

        beta:
            The quadratic coefficient, per Gy squared.

        repair_rate:
            The rate of repair of sublethal damage, per hour.

        delivery_time:
            The time to deliver one fraction, in hours.

    Returns:
        system:
            The system, with the states `cells.survival`, `cells.protraction`, and `cells.effective_dose`.

    Examples:
        >>> from skcomplex.core.engine.runtime import Simulation
        >>> from skcomplex.systems.biology.oncology import lq_cell_survival
        >>> system = lq_cell_survival(60.0, fractions=30.0)
        >>> trajectory = Simulation(system).prepare().run(1.0).trajectory_
        >>> round(float(trajectory.values('cells.effective_dose')[0]), 1)
        72.0
    """
    cells = LinearQuadraticSurvivalElement(
        'cells',
        alpha=alpha,
        beta=beta,
        dose=dose,
        fractions=fractions,
        repair_rate=repair_rate,
        delivery_time=delivery_time,
    )
    return System(elements=[cells], time=StaticTime())
