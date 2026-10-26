"""Define the base of the parts of a system: elements, interactions, laws, and systems."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from abc import ABC, abstractmethod
from typing import Any, Self

from sklearn.base import BaseEstimator


class BasePart(BaseEstimator, ABC):
    """Base of every part of a system, which has a `label` and speaks the verbs `check`, `prepare`, and `run`.

    The parts of a system are its elements, its interactions, the laws they return, and the system itself. A part
    implements three hooks, and the verbs call them, so every verb is written once here:

    - `_check()` returns the part with its arguments checked;
    - `_prepare(*context)` returns the part prepared against its context, with its prepared attributes set;
    - `_run(*inputs, **settings)` returns what the part contributes.

    An element, an interaction, or a system changes and returns itself, and runs into the laws it contributes. A law
    is a value, so it returns checked and prepared copies, and runs into its contribution at given values.

    Examples:
        >>> from skcomplex.core.base import BaseElement, BaseInteraction, BaseLaw, BasePart
        >>> from skcomplex.core.system import System
        >>> all(issubclass(base, BasePart) for base in (BaseElement, BaseInteraction, BaseLaw, System))
        True
    """

    @abstractmethod
    def _check(self) -> Self:
        """Return the part with its arguments checked, raising a named error for one it cannot take."""

    @abstractmethod
    def _prepare(self, *context: Any) -> Self:  # noqa: ANN401
        """Return the part prepared against its context, with every attribute that ends in `_` set."""

    @abstractmethod
    def _run(self, *inputs: Any, **settings: Any) -> Any:  # noqa: ANN401
        """Return what the part contributes: laws for an element, an interaction, or a system, values for a law."""

    def check(self) -> Self:
        """Return the part with its arguments checked.

        Returns:
            part:
                The checked part, the part itself or, for a law, a checked copy.

        Raises:
            ValidationError:
                An argument is invalid, named in the message.
        """
        return self._check()

    def prepare(self, *context: Any, **settings: Any) -> Self:  # noqa: ANN401
        """Return the part checked and prepared against its context.

        Args:
            *context:
                What the part is prepared against, such as the namespace of its owner.

            **settings:
                Settings of the preparation, which parts do not read.

        Returns:
            part:
                The prepared part, the part itself or, for a law, a prepared copy.
        """
        return self.check()._prepare(*context)

    def run(self, *inputs: Any, **settings: Any) -> Any:  # noqa: ANN401
        """Return what the part contributes.

        Args:
            *inputs:
                The inputs of the part, such as the namespace of symbols an element writes its laws in.

            **settings:
                The values a law is evaluated at, by name.

        Returns:
            contribution:
                The laws of an element, an interaction, or a system, or the value of a law.
        """
        return self._run(*inputs, **settings)
