"""Define the learning problem spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass, field

from ...components.laws import GradientStepLaw
from ..types import TracedFunction
from ._base_problem_spec import BaseProblemSpec
from ._problem_spec import ProblemSpec


@dataclass(frozen=True)
class LearningProblemSpec(BaseProblemSpec):
    """Problem of a system that trains the adaptive parameters of a nested system by gradient steps, once per epoch.

    Its states are the trained parameters, named by their paths from the root, and its parameters are the
    hyperparameters of the optimizer. The nested system is compiled to its own problem, a discrete map or ordinary
    differential equations, which the solver unrolls over the sequences of the dataset.

    Args:
        inner:
            The problem of the nested system, compiled by the JAX backend.

        trained_system:
            The path of the nested system that trains.

        training:
            The path of the interaction that returns the law.

        law:
            The gradient-step law.

        trained:
            The names of the trained parameters in the states of the inner problem, in the order of the states.

        hyperparameters:
            A mapping from the name of every hyperparameter of the optimizer to its name in the parameters.

        step:
            The pure JAX function `f(t, x, p)` of the inner problem, its update or its drift.

        measures:
            A mapping from `'loss'` and the label of every metric to the pure JAX function `o(t, x, p)` of its value.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.types import FormKind
        >>> from skcomplex.core.engine._utils import symbol
        >>> from skcomplex.core.specs import LayoutSpec
        >>> from skcomplex.core.engine.problems._learning_problem_spec import LearningProblemSpec
        >>> layout = LayoutSpec(())
        >>> problem = LearningProblemSpec(FormKind.LEARNING, layout, layout, np.zeros(0), np.zeros(0), symbol('t'))
        >>> problem.form.value, problem.trained
        ('learning by gradient steps', ())
    """

    inner: ProblemSpec | None = field(default=None, repr=False)
    trained_system: str = ''
    training: str = ''
    law: GradientStepLaw | None = field(default=None, repr=False)
    trained: tuple[str, ...] = ()
    hyperparameters: Mapping[str, str] = field(default_factory=dict)
    step: TracedFunction | None = field(default=None, repr=False)
    measures: Mapping[str, TracedFunction] = field(default_factory=dict, repr=False)
