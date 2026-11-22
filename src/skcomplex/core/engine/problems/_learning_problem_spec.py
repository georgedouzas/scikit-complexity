"""Define the learning problem spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass, field

from ...components.interactions import TrainingInteraction
from ..types import TracedFunction
from ._base_problem_spec import BaseProblemSpec
from ._problem_spec import ProblemSpec


@dataclass(frozen=True)
class LearningProblemSpec(BaseProblemSpec):
    """A spec of a problem that trains the parameters of a nested system on data.

    Its states are the trained parameters, named by their paths, and its parameters are the settings of the optimizer,
    such as the learning rate. The nested system compiles to its own problem, which the solver runs over the sequences
    of a dataset and differentiates, and one step of the problem is one epoch of gradient steps. It trains a recurrent
    network on a time series, the readout of a reservoir, and the policy of a learning agent.

    Read more in the user guide, under Extend the library.

    Args:
        inner:
            The problem of the nested system, compiled by the JAX backend.

        trained_system:
            The path of the nested system that trains.

        training:
            The path of the training interaction.

        interaction:
            The training interaction, whose optimizer, dataset, batches, truncation, and metrics the solver reads.

        trained:
            The names of the trained parameters in the states of the inner problem, in the order of the states.

        hyperparameters:
            A mapping from the name of every hyperparameter of the optimizer to its name in the parameters.

        step:
            The pure JAX function `f(t, x, p)` of the inner problem, its update or its drift.

        measures:
            A mapping from `'loss'` and the label of every metric to the pure JAX function `o(t, x, p)` of its value.

    See Also:
        TrainingInteraction: The interaction the problem comes from.
        GradientDescentSolver: Runs the epochs.

    Examples:
        >>> from skcomplex.systems.computer_science.neural_networks import ElmanNetworkSystem
        >>> problem = ElmanNetworkSystem().prepare().simulate(until=0).problem
        >>> problem.form.value
        'learning by gradient steps'
    """

    inner: ProblemSpec | None = field(default=None, repr=False)
    trained_system: str = ''
    training: str = ''
    interaction: TrainingInteraction | None = field(default=None, repr=False)
    trained: tuple[str, ...] = ()
    hyperparameters: Mapping[str, str] = field(default_factory=dict)
    step: TracedFunction | None = field(default=None, repr=False)
    measures: Mapping[str, TracedFunction] = field(default_factory=dict, repr=False)
