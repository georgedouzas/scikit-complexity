"""Mechanics: pendulums, oscillators, projectiles, collisions, rotating bodies, and potential wells."""

from ._ballistic_pendulum import BallisticPendulumElement
from ._bouncing_ball import BouncingBallElement
from ._cart_pendulum import CartPendulumElement
from ._cart_pendulum_interaction import CartPendulumInteraction
from ._clock_pendulum import ClockPendulumElement
from ._clock_pendulum_interaction import ClockPendulumInteraction
from ._disc_billiard import DiscBilliardElement
from ._double_pendulum import DoublePendulumElement
from ._double_pendulum_interaction import DoublePendulumInteraction
from ._elastic_collision import ElasticCollisionInteraction
from ._gyroscope import GyroscopeElement
from ._gyroscope_interaction import GyroscopeInteraction
from ._hanging_ball import HangingBallElement
from ._magnet import MagnetInteraction
from ._mechanics import (
    bouncing_ball,
    circular_billiard,
    driven_damped_oscillator,
    harmonic_oscillator,
    newtons_cradle,
    pendulum,
    projectile,
)
from ._parametric_pendulum import ParametricPendulumElement
from ._parametric_pendulum_interaction import ParametricPendulumInteraction
from ._pendulum import PendulumElement
from ._pendulum_interaction import PendulumInteraction
from ._pendulum_spring import PendulumSpringInteraction
from ._pendulums import (
    ballistic_pendulum,
    cart_pendulum,
    coupled_pendulums,
    double_pendulum,
    huygens_clocks,
    magnetic_pendulum,
    parametric_pendulum,
)
from ._rotating_ball import RotatingBallElement
from ._rotating_swing import RotatingSwingElement
from ._rotating_swing_interaction import RotatingSwingInteraction
from ._rotations import gimballed_gyroscope, rotating_balls_on_rods, rotating_swing
from ._springs import HarmonicForceInteraction, two_mass_spring_damper
from ._support_spring import SupportSpringInteraction
from ._wells import ExponentialWellInteraction, particle_in_a_well

__all__: list[str] = [
    'BallisticPendulumElement',
    'BouncingBallElement',
    'CartPendulumElement',
    'CartPendulumInteraction',
    'ClockPendulumElement',
    'ClockPendulumInteraction',
    'DiscBilliardElement',
    'DoublePendulumElement',
    'DoublePendulumInteraction',
    'ElasticCollisionInteraction',
    'ExponentialWellInteraction',
    'GyroscopeElement',
    'GyroscopeInteraction',
    'HangingBallElement',
    'HarmonicForceInteraction',
    'MagnetInteraction',
    'ParametricPendulumElement',
    'ParametricPendulumInteraction',
    'PendulumElement',
    'PendulumInteraction',
    'PendulumSpringInteraction',
    'RotatingBallElement',
    'RotatingSwingElement',
    'RotatingSwingInteraction',
    'SupportSpringInteraction',
    'ballistic_pendulum',
    'bouncing_ball',
    'cart_pendulum',
    'circular_billiard',
    'coupled_pendulums',
    'double_pendulum',
    'driven_damped_oscillator',
    'gimballed_gyroscope',
    'harmonic_oscillator',
    'huygens_clocks',
    'magnetic_pendulum',
    'newtons_cradle',
    'parametric_pendulum',
    'particle_in_a_well',
    'pendulum',
    'projectile',
    'rotating_balls_on_rods',
    'rotating_swing',
    'two_mass_spring_damper',
]
