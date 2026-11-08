"""Define the connector spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ..errors import ValidationError
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class ConnectorSpec(BaseSpec):
    """Spec of the variables a port of an element carries.

    They are a potential that connected ports share and a flow that sums.

    Args:
        name:
            The name of the connector, which connected ports have in common.

        potential:
            The name of the potential variable, the suffix of its state `<port>_<potential>`.

        flow:
            The name of the flow variable, the suffix of its state `<port>_<flow>`, positive into the element.

    Examples:
        >>> from skcomplex.core.specs import ConnectorSpec
        >>> ConnectorSpec('chemical', potential='mu', flow='n').flow
        'n'
    """

    name: str
    potential: str
    flow: str

    def _check(self) -> None:
        """Raise an error for variables that are not distinct names.

        Raises:
            ValidationError:
                The potential or the flow is not a name, or they are the same name.
        """
        names = (self.potential, self.flow)
        if self.potential == self.flow or not all(isinstance(name, str) and name.isidentifier() for name in names):
            error_msg = f'The connector {self.name!r} has the variables {names!r}. Give two different names.'
            raise ValidationError(error_msg)
