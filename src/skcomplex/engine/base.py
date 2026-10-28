"""Define the root of every configurable engine class."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Self

from sklearn.base import BaseEstimator as SklearnEstimator


class BaseEstimator(SklearnEstimator):
    """Base of every configurable object: it stores its arguments and speaks three verbs.

    The constructor of a child stores every argument unmodified under its own name. `check` validates the arguments,
    `prepare` prepares the object against what it works on and sets every attribute that ends in `_`, and `run` performs
    the object's one action. Each verb calls a private hook that a child overrides, and the hooks of this base do
    nothing, so every object answers every verb. A last call `run(..., final=True)` also releases what `prepare`
    acquired.

    Examples:
        >>> from skcomplex.engine.base import BaseEstimator
        >>> class DoublingEstimator(BaseEstimator):
        ...     def __init__(self, factor=2):
        ...         self.factor = factor
        ...
        ...     def _run(self, value):
        ...         return self.factor * value
        >>> DoublingEstimator().prepare().run(3)
        6
    """

    def _check(self) -> None:
        """Raise an error for an argument the object cannot take; accept every argument by default."""

    def _prepare(self, *context: object, **settings: object) -> None:
        """Prepare the object against its context and set its prepared attributes; derive nothing by default."""

    def _run(self, *inputs: object, **settings: object) -> object:
        """Perform the action of the object; do nothing by default."""
        return None

    def _release(self) -> None:
        """Release what preparing acquired, such as a viewer or a process; release nothing by default."""

    def check(self) -> Self:
        """Validate the arguments of the object.

        Returns:
            self:
                The object.

        Raises:
            ValidationError:
                An argument is invalid, named in the message.
        """
        self._check()
        return self

    def prepare(self, *context: object, **settings: object) -> Self:
        """Check the arguments and prepare the object against its context, setting every attribute that ends in `_`.

        Args:
            *context:
                What the object works on, such as the problem a solver advances.

            **settings:
                Settings of the preparation, such as the end time of a run.

        Returns:
            self:
                The prepared object.
        """
        self.check()
        self._prepare(*context, **settings)
        return self

    def run(self, *inputs: object, final: bool = False, **settings: object) -> object:
        """Perform the one action of the object on its inputs.

        Args:
            *inputs:
                The inputs of the action, such as the state a solver advances.

            final:
                Whether this is the last call, after which the object releases what preparing acquired.

            **settings:
                Settings of the action.

        Returns:
            result:
                The result of the action.
        """
        result = self._run(*inputs, **settings)
        if final:
            self._release()
        return result
