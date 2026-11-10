"""Define the abstract base of every configurable class of the library."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from abc import ABC, abstractmethod
from typing import Self

from sklearn.base import BaseEstimator

from .errors import NotPreparedError


class BaseConfig[ContextT, InputT](BaseEstimator, ABC):
    """An abstract configurable class, the root of every class of the library.

    The constructor stores every setting unchanged under its own name, as a scikit-learn estimator does, so
    `get_params`, `set_params`, and `clone` work on every class. The main public methods call the abstract hooks
    `_prepare` and `_run`, which every child implements, itself or through the base of its family, and each returns the
    object:

    - `prepare(context)` prepares the object against the one thing it serves, through `_prepare(context)`. The hook
      first checks the settings, raising a named error for one the object cannot take, and then sets the attributes that
      end in `_`.
    - `run(inputs, final=False)` performs the one action of the prepared object on its input, through `_run(inputs)`,
      and keeps what it produces in attributes that end in `_`. A last call with `final=True` also releases what
      preparing acquired, through the optional hook `_release`.

    Unlike `BaseEstimator` of scikit-learn, which fits data, it prepares against a context and runs on an input. Choose
    it when a new family of classes needs settings that `clone` and `set_params` read, such as a new solver or backend.

    It appears in every part of the library, such as the elements and interactions of a system, its solvers, its
    backends, and the passes of its compiler.

    Examples:
        >>> from skcomplex.core.base import BaseConfig
        >>> class DoublingConfig(BaseConfig):
        ...     def __init__(self, factor=2):
        ...         self.factor = factor
        ...
        ...     def _prepare(self, context):
        ...         if not isinstance(self.factor, int):
        ...             raise TypeError('The factor must be an integer.')
        ...
        ...     def _run(self, inputs):
        ...         self.doubled_ = self.factor * inputs
        >>> DoublingConfig().prepare().run(3).doubled_
        6
    """

    @abstractmethod
    def _prepare(self, context: ContextT | None) -> None:
        """Check the settings of the object and prepare it against what it serves.

        It checks every setting before it sets its attributes that end in `_`, raising a named error for one the
        object cannot take. A child chains with `super()._prepare(context)` when its base prepares something.

        Args:
            context:
                What the object serves, such as the system of a compiler or the problem of a solver.
        """

    @abstractmethod
    def _run(self, inputs: InputT | None) -> None:
        """Perform the action of the object on its input.

        It sets what it produces as attributes that end in `_`.

        Args:
            inputs:
                The input of the action, such as the state a solver advances or the values a law reads.
        """

    def _release(self) -> None:
        """Release what preparing acquired.

        Examples are a viewer or a process. By default, it releases nothing.
        """

    def _is_prepared(self) -> None:
        """Raise an error for an object that was not prepared.

        Raises:
            NotPreparedError:
                The object was not prepared with `prepare`.
        """
        if not getattr(self, '_prepared', False):
            error_msg = f'{type(self).__name__} must be prepared before it runs. Call `prepare(...)` first.'
            raise NotPreparedError(error_msg)

    def prepare(self, context: ContextT | None = None) -> Self:
        """Check the object and prepare it against the one thing it serves.

        It checks its settings and sets its attributes that end in `_`.

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
        self._prepare(context)
        self._prepared = True
        return self

    def run(self, inputs: InputT | None = None, final: bool = False) -> Self:
        """Perform the one action of the prepared object on its one input.

        It keeps its results in attributes.

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
