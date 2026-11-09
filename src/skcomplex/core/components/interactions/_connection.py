"""Define connection interactions.

A connection is the node where the potentials of ports agree and their flows sum to zero.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar

from sage.all__sagemath_symbolics import SR

from ...errors import ValidationError
from ...specs import RoleSpec
from ...types import ElementKind
from ..base import BasePart
from ..laws import BaseLaw, EquationLaw
from ..spaces import BaseSpace
from ._interaction import Interaction


class ConnectionInteraction(Interaction):
    """Node of connected ports, whose potentials are equal and whose flows sum to zero.

    The interaction binds the elements of one node through a variadic role and names the port of each. A node with many
    ports is one connection, and a port belongs to one connection. A port that no connection names has a zero flow,
    which the compiler adds.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        members:
            The paths of the elements of the node.

        ports:
            The name of the port of every element, in the same order.

    Raises:
        ValidationError:
            The members and the ports are empty or differ in length.

    Examples:
        >>> from typing import Annotated, ClassVar
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.components.interactions import ConnectionInteraction
        >>> from skcomplex.core.components.laws import EquationLaw
        >>> from skcomplex.core.components.systems import System
        >>> from skcomplex.core.specs import ConnectorSpec, ParameterSpec, StateSpec
        >>> from skcomplex.core.engine.runtime import simulate
        >>> PIN = ConnectorSpec('pin', potential='v', flow='i')
        >>> class ConductanceElement(Element):
        ...     ports: ClassVar = {'a': PIN}
        ...
        ...     def __init__(
        ...         self,
        ...         label,
        ...         a_v: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        ...         a_i: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        ...         g: Annotated[float, ParameterSpec()] = 1.0,
        ...         e: Annotated[float, ParameterSpec()] = 0.0,
        ...     ):
        ...         super().__init__(label)
        ...         self.a_v = a_v
        ...         self.a_i = a_i
        ...         self.g = g
        ...         self.e = e
        ...
        ...     def _get_laws(self, s):
        ...         return [EquationLaw(s.a_i, s.g * (s.a_v - s.e))]
        >>> system = System(
        ...     elements=[ConductanceElement('first', e=1.0), ConductanceElement('second', e=3.0)],
        ...     interactions=[ConnectionInteraction('node', members=['first', 'second'], ports=['a', 'a'])],
        ... )
        >>> trajectory = simulate(system, until=1.0, dt=1.0)
        >>> float(trajectory.values('first.a_v')[-1])
        2.0
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (EquationLaw,)

    def __init__(
        self,
        label: str,
        *,
        members: Annotated[Sequence[str], RoleSpec(variadic=True, elements=(ElementKind.ANY,))],
        ports: Sequence[str],
    ) -> None:
        super().__init__(label)
        self.members = members
        self.ports = ports

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Raise an error for members and ports that are empty or differ in length.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                The type declares states, or the members and the ports are empty or differ in length.
        """
        super()._prepare(context)
        if not self.members or len(self.members) != len(self.ports):
            error_msg = (
                f'{self.label!r} connects {len(self.members)} members through {len(self.ports)} ports. Name one port '
                'per member.'
            )
            raise ValidationError(error_msg)

    def _get_laws(self, namespace: SimpleNamespace) -> list[BasePart]:
        """Return the equality of the potentials of the ports and the balance of flows.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with the namespaces of the members.

        Returns:
            laws:
                The equation laws of the node.
        """
        potentials, flows = [], []
        for member, port in zip(namespace.members, self.ports, strict=True):
            connector = member.ports[port]
            potentials.append(getattr(member, f'{port}_{connector.potential}'))
            flows.append(getattr(member, f'{port}_{connector.flow}'))
        equalities: list[BasePart] = [EquationLaw(potentials[0], other) for other in potentials[1:]]
        return [*equalities, EquationLaw(sum(flows, SR(0)), 0)]
