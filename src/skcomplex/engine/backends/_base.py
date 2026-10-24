"""Define the contract of backends that hand a whole problem to a third-party solver library."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from abc import abstractmethod
from typing import ClassVar

import numpy as np

from ..base import BackendError, Form
from ..compile import Problem
from ..solvers import Stepper


class ExternalBackend(Stepper):
    """Base class of solvers that translate a compiled problem into a third-party library.

    Subclasses set the class variables `name` and `forms`, and implement `check`, `build`, and `advance`. Before the
    first frame, `initialize` checks every element, interaction, and law, and raises one error that lists everything the
    library cannot run.
    """

    name: ClassVar[str]
    forms: ClassVar[frozenset[Form]]

    @abstractmethod
    def check(self, problem: Problem) -> list[str]:
        """Return a description of every element, interaction, or law of a problem the library cannot run.

        Args:
            problem:
                The compiled problem.

        Returns:
            unsupported:
                One description per unsupported item, empty when the problem is supported.

        Raises:
            NotImplementedError:
                The subclass does not implement the method.
        """
        raise NotImplementedError

    @abstractmethod
    def build(self, problem: Problem, rng: np.random.Generator) -> None:
        """Build the library model of a supported problem.

        Args:
            problem:
                The compiled problem.

            rng:
                The random number generator that seeds the library.

        Raises:
            NotImplementedError:
                The subclass does not implement the method.
        """
        raise NotImplementedError

    def initialize(self, problem: Problem, rng: np.random.Generator) -> None:
        """Check that the library supports a problem and build its model.

        Args:
            problem:
                The compiled problem.

            rng:
                The random number generator that seeds the library.

        Raises:
            BackendError:
                The problem has a form, a complex state, an element, an interaction, or a law the library cannot run.
        """
        unsupported = [] if problem.form in self.forms else [f'the {problem.form.value} form']
        unsupported += [f'the complex state {entry.name}' for entry in problem.states if entry.is_complex]
        unsupported += self.check(problem)
        if unsupported:
            items = '; '.join(unsupported)
            error_msg = f'The {self.name} backend cannot run this system. Unsupported: {items}.'
            raise BackendError(error_msg)
        self.problem_ = problem
        self.rng_ = rng
        self.build(problem, rng)
