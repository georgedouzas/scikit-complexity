"""Thermohaline circulation between ocean boxes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from sage.all__sagemath_symbolics import abs_symbolic

from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations


class StommelSystem(DynamicalSystem):
    """Stommel's two-box thermohaline circulation.

    The circulation is bistable between a thermal and a haline mode.

    The dimensionless differences of temperature `T` and salinity `S` between an equatorial and a polar box relax to
    their forcing and are mixed by a flow proportional to the density difference `|T - S|`: `dT/dt = eta1 - T (1 + |T -
    S|)` and `dS/dt = eta2 - S (eta3 + |T - S|)`. For the default parameters a strong circulation driven by temperature
    and a weak one driven by salinity are both stable. The system has the differences `ocean.T` and `ocean.S`.

    Args:
        T:
            The initial temperature difference.

        S:
            The initial salinity difference.

        eta1:
            The thermal forcing.

        eta2:
            The freshwater forcing.

        eta3:
            The ratio of the relaxation times of temperature and salinity.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.oceanography import StommelSystem
        >>> StommelSystem(eta2=1.0).prepare().system_.elements[0].get_value('eta2')
        1.0
    """

    field = 'physics/oceanography'
    topics = ('ode', 'bistability', 'thermohaline circulation', 'tipping')
    references = ('Stommel 1961', 'Cessi 1994', 'Ghil and Lucarini 2020')
    until = 50.0
    dt = 0.1

    def __init__(
        self,
        T: float = 3.0,
        S: float = 0.0,
        *,
        eta1: float = 3.0,
        eta2: float = 1.2,
        eta3: float = 0.3,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.T = T
        self.S = S
        self.eta1 = eta1
        self.eta2 = eta2
        self.eta3 = eta3

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'ocean'

    def _get_states(self) -> dict[str, object]:
        """Return the initial value of every state."""
        return {
            'T': self.T,
            'S': self.S,
        }

    def _get_parameters(self) -> dict[str, object]:
        """Return the parameters of the equations."""
        return {
            'eta1': self.eta1,
            'eta2': self.eta2,
            'eta3': self.eta3,
        }

    def _get_equations(self) -> Equations:
        """Return the relaxation of both differences to their forcing and their mixing."""
        return {
            'T': lambda s: s.target.eta1 - s.target.T * (1 + abs_symbolic(s.target.T - s.target.S)),
            'S': lambda s: s.target.eta2 - s.target.S * (s.target.eta3 + abs_symbolic(s.target.T - s.target.S)),
        }
