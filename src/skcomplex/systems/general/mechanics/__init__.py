"""Define the Lagrangian mechanics that the systems of several fields share.

The equations of motion follow from a Lagrangian, the kinetic minus the potential energy, as for a pendulum, a planet,
or the rotor of a motor.
"""

from ._euler_lagrange import EulerLagrangeInteraction

__all__: list[str] = [
    'EulerLagrangeInteraction',
]
