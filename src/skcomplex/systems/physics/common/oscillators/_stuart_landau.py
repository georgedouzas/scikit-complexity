"""Define the Stuart-Landau flow.

It is the normal form of a supercritical Hopf bifurcation of a complex amplitude.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from .....core.components.interactions import Interaction
from .....core.components.laws import RateLaw
from .....core.components.spaces import BaseSpace
from .....core.specs import Role


class StuartLandauInteraction(Interaction):
    """Stuart-Landau flow of `z = x + i y`, `z' = tau ((mu + i omega) z - kappa (1 + i beta) |z|^2 z)`.

    The flow is the normal form of a supercritical Hopf bifurcation. For `mu > 0` the oscillator settles on the limit
    cycle of radius `sqrt(mu / kappa)`, which it turns at the frequency `tau (omega - beta mu)`, and for `mu < 0` it
    decays to the fixed point 0. The target holds the values `x` and `y` of `z` and the parameters `mu`, `omega`,
    `beta`, `kappa`, and `tau`, one of each per member of a group, as for a laser near threshold, a chemical oscillator,
    or a neuron that starts to fire.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the oscillator or the group of oscillators.

    Examples:
        >>> import math
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.physics.common.oscillators import StuartLandauInteraction
        >>> values = {'x': 0.1, 'y': 0.0, 'mu': 4.0, 'omega': 1.0, 'beta': 0.0, 'kappa': 1.0, 'tau': 1.0}
        >>> flow = StuartLandauInteraction('flow', target='oscillator')
        >>> system = System(elements=[Element('oscillator', values=values)], interactions=[flow])
        >>> trajectory = system.prepare().simulate(until=10.0, dt=0.5)
        >>> round(math.hypot(trajectory.get_values('oscillator.x')[-1], trajectory.get_values('oscillator.y')[-1]), 4)
        2.0
    """

    law_classes = (RateLaw,)

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rates of the real and imaginary parts of `z`.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s, t = context.target, context.t
            saturation = s.kappa * (s.x**2 + s.y**2)
            self.laws_ = [
                RateLaw(s.x.diff(t), s.tau * (s.mu * s.x - s.omega * s.y - saturation * (s.x - s.beta * s.y))),
                RateLaw(s.y.diff(t), s.tau * (s.mu * s.y + s.omega * s.x - saturation * (s.y + s.beta * s.x))),
            ]
