"""Share the helpers and fixtures of the tests below this directory."""

from skcomplex.core.components import Element
from skcomplex.core.declarations import role

from skcomplex.core.base import BaseInteraction, NumericDomain, parameter, state
from skcomplex.core.laws import UpdateLaw

HEALTH = NumericDomain.categorical('susceptible', 'infected', 'recovered')


class PersonElement(Element):
    """Person element."""

    health = state('susceptible', domain=HEALTH)


class ProgressionInteraction(BaseInteraction):
    """Progression interaction."""

    target = role()
    step = parameter(1.0)

    def laws(self, s):  # noqa: ANN001, ANN201
        """Return the laws of the progression interaction."""
        return [UpdateLaw(s.target.health, s.target.health + s.step)]
