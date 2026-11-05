"""Define the root of every configurable class of the library."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from abc import ABC, abstractmethod
from typing import Self

from sklearn.base import BaseEstimator

from .errors import NotPreparedError


class BaseConfig[ContextT, InputT](BaseEstimator, ABC):
    """Root of every configurable class of the library: it stores its settings and speaks two verbs.

    The constructor of a child stores every setting unmodified under its own name, as in scikit-learn, so
    `get_params`, `set_params`, and `clone` work on every class. The verbs call the abstract hooks `_check`,
    `_prepare`, and `_run`, which every child implements, itself or through the base of its family, and every verb
    returns the object, as in scikit-learn:

    - `prepare(context)` validates the settings through `_check()`, prepares the object against the one thing it serves
      through `_prepare(context)`, and sets the attributes that end in `_`.
    - `run(inputs, final=False)` performs the one action of the prepared object on its one input through
      `_run(inputs)`, and keeps what the action produces in attributes that end in `_`; a last call with `final=True`
      also releases what preparing acquired through `_release()`, an optional hook that releases nothing by default.

    Examples:
        >>> from skcomplex.base import BaseConfig
        >>> class DoublingConfig(BaseConfig):
        ...     def __init__(self, factor=2):
        ...         self.factor = factor
        ...
        ...     def _check(self):
        ...         pass
        ...
        ...     def _prepare(self, context):
        ...         pass
        ...
        ...     def _run(self, inputs):
        ...         self.doubled_ = self.factor * inputs
        >>> DoublingConfig().prepare().run(3).doubled_
        6
    """

    @abstractmethod
    def _check(self) -> None:
        """Validate the settings, raising a named error for one the object cannot take.

        Raises:
            ValidationError:
                A setting is invalid, named in the message.
        """

    @abstractmethod
    def _prepare(self, context: ContextT | None) -> None:
        """Prepare the object against what it serves and set its attributes that end in `_`.

        Args:
            context:
                What the object serves, such as the system of a compiler or the problem of a solver.
        """

    @abstractmethod
    def _run(self, inputs: InputT | None) -> None:
        """Perform the action of the object on its input and set what it produces as attributes that end in `_`.

        Args:
            inputs:
                The input of the action, such as the state a solver advances or the values a law reads.
        """

    def _release(self) -> None:
        """Release what preparing acquired, such as a viewer or a process; release nothing here."""

    def _is_prepared(self) -> None:
        """Raise an error for an object that was not prepared.

        Raises:
            NotPreparedError:
                The object was not prepared with `prepare`.
        """
        if not getattr(self, '_prepared', False):
            error_msg = f'{type(self).__name__} must be prepared before it runs. Call `prepare(...)` first.'
            raise NotPreparedError(error_msg)

    def _validated(self) -> Self:
        """Validate the settings of the object.

        Returns:
            self:
                The object.

        Raises:
            ValidationError:
                A setting is invalid, named in the message.
        """
        self._check()
        return self

    def prepare(self, context: ContextT | None = None) -> Self:
        """Validate the object and prepare it against the one thing it serves, setting its attributes that end in `_`.

        Args:
            context:
                What the object serves, such as the system of a compiler, the problem of a solver, or the source of an
                analysis.

        Returns:
            self:
                The prepared object.

        Raises:
            ValidationError:
                A setting is invalid, named in the message.
        """
        self._validated()
        self._prepare(context)
        self._prepared = True
        return self

    def run(self, inputs: InputT | None = None, final: bool = False) -> Self:
        """Perform the one action of the prepared object on its one input, keeping its results in attributes.

        Args:
            inputs:
                The input of the action, such as the state a solver advances or the values a law reads.

            final:
                Whether this is the last call, after which the object releases what preparing acquired.

        Returns:
            self:
                The object, with the results of the action in its attributes that end in `_`.

        Raises:
            NotPreparedError:
                The object was not prepared with `prepare`.
        """
        self._is_prepared()
        self._run(inputs)
        if final:
            self._release()
        return self
