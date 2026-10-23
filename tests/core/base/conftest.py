"""Share the helpers and fixtures of the tests below this directory."""

from skcomplex.core.base import Domain, parameter, state
from skcomplex.core.components import Element
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import Update

HEALTH = Domain.categorical('susceptible', 'infected', 'recovered')


Person = Element.define('Person', states={'health': state('susceptible', domain=HEALTH)})


Progression = Interaction.define(
    'Progression',
    roles=['target'],
    parameters={'step': parameter(1.0)},
    laws=lambda s: [Update(s.target.health, s.target.health + s.step)],
)
