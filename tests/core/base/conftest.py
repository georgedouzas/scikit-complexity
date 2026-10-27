"""Share the helpers and fixtures of the tests below this directory."""

from typing import Annotated

from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from skcomplex.core.elements import Element
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import UpdateLaw
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec

HEALTH = FiniteEnumeratedSet(['susceptible', 'infected', 'recovered'])


class PersonElement(Element):
    """Person element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        health: Annotated[object, StateSpec(domain=HEALTH)] = 'susceptible',
    ) -> None:
        super().__init__(label, visual=visual)
        self.health = health


class ProgressionInteraction(Interaction):
    """Progression interaction."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[object, RoleSpec()],
        step: Annotated[object, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.step = step

    def _get_laws(self, s):  # noqa: ANN001, ANN201
        """Return the laws of the progression interaction."""
        return [UpdateLaw(s.target.health, s.target.health + s.step)]
