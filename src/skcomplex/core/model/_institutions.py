"""Define institutions, single actors whose rules read observables and set the parameters of other components."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from typing import Any

from ._components import Interaction
from ._laws import Law, Rule
from ._symbols import Symbols


class Institution(Interaction):
    """Single actor, such as a central bank, a regulator, or a quota board, that sets parameters from observables.

    Its rule runs once per round of its phase, after the agents of the phase. It reads observables with
    `model.observable(name)`, reporters with `model.history(name)`, and agents with `model.agents(name)`, and sets
    parameters of any group or interaction with `model.set_parameter(name, value)`. A change is visible to every rule
    at once and recorded in the trajectory like any parameter. Subclasses declare their own parameters, such as a
    policy rate, and override `act`; otherwise `rule(model)` acts.

    Args:
        label:
            The label of the institution.

        rule:
            The function of the model view that acts, used when a subclass does not override `act`.

        phase:
            The phase the institution acts in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.core.model import Institution
        >>> Institution('regulator', rule=lambda model: None).phase is None
        True
    """

    rule: Callable[[Any], None] | None = None
    phase: str | None = None

    def laws(self, s: Symbols) -> list[Law]:
        """Return the rule of the institution in its phase.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.act, phase=self.phase)]

    def act(self, model: Any) -> None:  # noqa: ANN401
        """Read the model and set parameters.

        Args:
            model:
                The model view scoped to the institution.

        Raises:
            NotImplementedError:
                The institution has neither a rule nor its own `act`.
        """
        if self.rule is None:
            error_msg = f'Institution {self.label!r} has no rule. Pass `rule` or override `act`.'
            raise NotImplementedError(error_msg)
        self.rule(model)
