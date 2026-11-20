"""Define the abstract bases of every engine class and of every spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, is_dataclass
from typing import Self

from sage.all__sagemath_symbolics import Expression
from sage.modules.free_module_element import FreeModuleElement
from sage.structure.element import Matrix
from sage.structure.parent import Parent
from sage.symbolic.function import Function
from sklearn.base import BaseEstimator, clone

from .errors import NotPreparedError


class _BaseEstimator[ContextT](BaseEstimator, ABC):
    """An abstract estimator, the root of the engine classes and of the components.

    The constructor stores every setting unchanged under its own name, as a scikit-learn estimator does, so
    `get_params`, `set_params`, and `clone` work on every class, and a clone shares the symbolic values of Sage. The
    main public method `prepare(context)` calls the abstract hook `_prepare(context)`, which first checks the settings
    and then sets the attributes that end in `_`.
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

    def __sklearn_clone__(self) -> Self:
        """Return a new object of the same class with copies of the settings.

        It is what `sklearn.base.clone` returns. Symbolic values of Sage, such as the expression of a law, a symbolic
        function, or a set of allowed values, are immutable, so the copy shares them. Copying them deeply would rebuild
        them through Sage's pickling, which changes the numbers in conditions and makes new symbolic functions.

        Returns:
            clone:
                The new object, not prepared.
        """
        params = {name: self._get_cloned(value) for name, value in self.get_params(deep=False).items()}
        return type(self)(**params)

    @staticmethod
    def _get_cloned(value: object) -> object:
        """Return a copy of a setting that shares the values that never change.

        They are the symbolic values of Sage and frozen dataclasses, such as specs.
        """
        if isinstance(value, Expression | Function | Parent):
            return value
        params = getattr(type(value), '__dataclass_params__', None)
        if is_dataclass(value) and params is not None and params.frozen:
            return value
        if isinstance(value, FreeModuleElement | Matrix):
            return value if value.is_immutable() else copy.copy(value)
        if type(value) is dict:
            return {key: _BaseEstimator._get_cloned(item) for key, item in value.items()}
        if isinstance(value, list | tuple | set | frozenset) and type(value) in {list, tuple, set, frozenset}:
            return type(value)([_BaseEstimator._get_cloned(item) for item in value])
        return clone(value, safe=False)

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


class BaseConfig[ContextT, InputT](_BaseEstimator[ContextT]):
    """An abstract engine class, such as a solver, a backend, or an observer.

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

    Unlike `BaseEstimator` of scikit-learn, which fits data, it prepares against a context and runs on an input. Unlike
    `BaseComponent`, the base of the components of a system, it runs. Choose it when a new family of engine classes
    needs settings that `clone` and `set_params` read, such as a new solver or backend.

    It appears in the engine of every system, such as its solvers, its backends, its observers, and the passes of its
    compiler.

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


@dataclass(frozen=True)
class BaseSpec:
    """An abstract spec, a frozen value that a class reads, records, or builds.

    A spec is a frozen dataclass whose fields are what it is. A spec declares what a class reads, records what a run
    produced, or holds what a compiler built. It never changes and holds only its fields, the hook `_check`, and readers
    of its own fields. A spec is a value, not an estimator: it has neither `prepare` nor `run`, and the class that owns
    it calls `_check()` when it reads it. A child that validates its fields overrides `_check()`, and a spec is changed
    with `dataclasses.replace`.

    Examples:
        >>> from dataclasses import dataclass
        >>> from skcomplex.core.base import BaseSpec
        >>> @dataclass(frozen=True)
        ... class SpanSpec(BaseSpec):
        ...     start: float
        ...     end: float
        >>> SpanSpec(0.0, 1.0).end
        1.0
    """

    def _check(self) -> None:
        """Validate the fields of the spec.

        It raises a named error for an invalid field. By default, it accepts every field.
        """
