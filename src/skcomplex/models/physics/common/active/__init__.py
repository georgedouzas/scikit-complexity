"""Active realizations shared by the physics packages: self-propulsion in continuous and in discrete time."""

from ._propulsion_step import PropulsionStepInteraction
from ._self_propulsion import SelfPropulsionInteraction

__all__: list[str] = ['PropulsionStepInteraction', 'SelfPropulsionInteraction']
