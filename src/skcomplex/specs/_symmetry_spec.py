"""Define the symmetry spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import SimpleNamespace

from ..errors import ValidationError
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class SymmetrySpec(BaseSpec):
    """Spec of a continuous symmetry of a system, the infinitesimal variation of the coordinates it moves.

    The generator is a function of the namespace of the system, where `s.<element>.<state>` is the symbol of a state,
    that maps the path of every coordinate it moves, such as `'body.q'`, to its infinitesimal variation `dq`, an
    expression for a scalar state or a vector for a vector state.

    Args:
        label:
            The label of the symmetry.

        generator:
            A function of the namespace of the system that returns the variation of every coordinate it moves.

    Raises:
        ValidationError:
            The label is empty, or the generator is not callable.

    Examples:
        >>> from sage.all__sagemath_symbolics import vector
        >>> from skcomplex.specs import SymmetrySpec
        >>> rotation = SymmetrySpec('rotation', lambda s: {'body.q': vector([-s.body.q[1], s.body.q[0]])})
        >>> rotation.label
        'rotation'
    """

    label: str
    generator: Callable[[SimpleNamespace], Mapping[str, object]]

    def _check(self) -> None:
        """Raise an error for an empty label or a generator that is not callable.

        Raises:
            ValidationError:
                The label is empty, or the generator is not callable.
        """
        if not self.label:
            error_msg = 'A symmetry has an empty label. Give it a name, such as `rotation`.'
            raise ValidationError(error_msg)
        if not callable(self.generator):
            error_msg = (
                f'The generator of the symmetry {self.label!r} is {self.generator!r}. Give a function of the namespace '
                'that maps the path of every coordinate it moves to its variation.'
            )
            raise ValidationError(error_msg)
