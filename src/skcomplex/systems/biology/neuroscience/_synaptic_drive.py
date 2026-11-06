"""Synaptic drive of a population by the rate or the synaptic field of another, `dv/dt += J x`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.errors import ValidationError
from ....core.specs import ParameterSpec, RoleSpec

FIELDS = ('r', 's')


class SynapticDriveInteraction(Interaction):
    """Synaptic drive of a population by the rate or the synaptic field of another, `dv/dt += J x`.

    Instantaneous synapses pass the firing rate `r` of the source, and exponential synapses its synaptic field `s`. A
    negative strength `J` is inhibition.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the driven population.

        source:
            The path of the driving population, which may be the target.

        J:
            The strength of the synapses, negative for inhibition.

        field:
            `'r'` for instantaneous synapses or `'s'` for exponential synapses.

    Raises:
        ValidationError:
            The field is unknown.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import SynapticDriveInteraction
        >>> SynapticDriveInteraction('ie', target='e', source='i', J=-10.0).J
        -10.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        source: Annotated[str, RoleSpec()],
        J: Annotated[float, ParameterSpec()] = 1.0,
        field: str = 'r',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.source = source
        self.J = J
        self.field = field

    def _check(self) -> None:
        """Check the field.

        Raises:
            ValidationError:
                The field is unknown.
        """
        super()._check()
        if self.field not in FIELDS:
            error_msg = f'The field must be one of {", ".join(FIELDS)}, not {self.field!r}.'
            raise ValidationError(error_msg)

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the drive of the potential of the target.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RateLaw(s.target.v, s.J * getattr(s.source, self.field))]
