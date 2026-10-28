"""Infer the form of a problem from the classes of the laws of its components."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Iterable
from typing import cast

from ....core.errors import IncompatibleLawsError
from ...problems import Form
from .._compilation import Compilation
from .._forms import _FORMS
from .._owners import _Owners
from .._pipeline import BasePass


class FormPass(BasePass):
    """Pass that infers the form of the problem from the law classes of its owners."""

    reads = ('kinds',)
    writes = ('form',)

    def _run(self, compilation: Compilation) -> None:
        """Infer the form of the problem.

        Args:
            compilation:
                The compilation state.
        """
        compilation.form = self._infer_form(cast('dict[str, set[type]]', compilation.kinds).values())

    @staticmethod
    def _infer_form(owners: Iterable[set[type]]) -> Form:
        """Return the form that the law classes of the components of a system determine.

        A combined class, such as a renewal, drops out of the classes when the rest are its companions and no
        component has laws of that class alone.

        Args:
            owners:
                The law classes of every component.

        Returns:
            form:
                The form.

        Raises:
            IncompatibleLawsError:
                No form takes the law classes.
        """
        kinds = _Owners._form_kinds(owners)
        if kinds not in _FORMS:
            found = ', '.join(sorted(kind.__name__ for kind in kinds))
            supported = '; '.join(
                ', '.join(sorted(kind.__name__ for kind in combination)) for combination in _FORMS if combination
            )
            error_msg = f'The system mixes the law kinds {found}. Supported combinations are: {supported}.'
            raise IncompatibleLawsError(error_msg)
        return _FORMS[kinds]
