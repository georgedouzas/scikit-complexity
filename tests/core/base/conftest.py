"""Share the helpers and fixtures of the tests below this directory."""

from skcomplex.core.base import NumericDomain, parameter, state
from skcomplex.core.components import ElementComponent
from skcomplex.core.interactions import BaseInteraction
from skcomplex.core.laws import UpdateLaw

HEALTH = NumericDomain.categorical('susceptible', 'infected', 'recovered')


PersonComponent = ElementComponent.define('PersonComponent', states={'health': state('susceptible', domain=HEALTH)})


ProgressionInteraction = BaseInteraction.define(
    'ProgressionInteraction',
    roles=['target'],
    parameters={'step': parameter(1.0)},
    laws=lambda s: [UpdateLaw(s.target.health, s.target.health + s.step)],
)
