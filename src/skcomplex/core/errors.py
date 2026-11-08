"""Define the exceptions the library raises."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


class SkcomplexError(Exception):
    """Base exception of the library."""


class NotPreparedError(SkcomplexError):
    """A configurable object is run before it is prepared."""


class ValidationError(SkcomplexError):
    """A system, a value, or a system with its solver and backend is not valid."""


class AlgebraicStateError(ValidationError):
    """A law integrates a state that the equations of its system determine."""


class ConservationError(ValidationError):
    """A conserved quantity or an invariant of a run changes beyond what it may.

    A conserved quantity may change only by its flows, and an invariant only within its tolerance.
    """


class TensorShapeError(ValidationError):
    """A tensor operation is given an expression of the wrong shape.

    An example is the tensor divergence of a scalar.
    """


class TensorSymmetryError(ValidationError):
    """A rate law on a tensor state breaks the declared symmetry of the state."""


class RewritingError(ValidationError):
    """A rewriting system is malformed, or a generation would exceed its maximum size."""


class ParameterError(ValidationError):
    """A parameter name or value is not valid.

    The name is not part of the problem, or the value is not valid for its system or matrix.
    """


class LabelError(ValidationError):
    """A label is missing, duplicated, or produces a name collision."""


class ArgumentError(SkcomplexError, TypeError):
    """A component is given an undeclared argument, or misses a required one.

    An undeclared argument is one its type does not declare.
    """


class RoleError(ValidationError):
    """A role of an interaction refers to an element that does not exist."""


class RoleElementError(RoleError):
    """A role of an interaction is bound to an element it does not accept.

    The element is of a kind the role does not accept, or lacks a state the role reads.
    """


class TagError(ValidationError):
    """The tags of a system disagree or are not supported.

    The parts of a system disagree on a capability tag, or a solver or a backend does not support the tags.
    """


class SpaceError(ValidationError):
    """A part, a vector, or a location does not fit the space of a system.

    A part needs a kind of space the system lacks, a vector has no space, or a location does not fit its space.
    """


class SpaceTooLargeError(SpaceError):
    """A space would hold too many sites, or its elements would overflow.

    It would hold more sites than its limit, or its elements would overflow their integer type.
    """


class PhaseSpaceTooLargeError(SpaceTooLargeError):
    """A phase space would hold more sites than its limit."""


class UnregisteredLawError(ValidationError):
    """A system holds a law of a class that no one registered.

    Neither the shared core nor a formalism registered its class.
    """


class IncompatibleLawsError(ValidationError):
    """A system mixes law kinds that no single form supports."""


class BackendError(SkcomplexError):
    """A backend is not registered or cannot compile a problem."""


class PipelineError(SkcomplexError):
    """The passes of a compilation cannot be ordered, or read a field none writes.

    A pass reads a field that no pass writes.
    """


class SolverError(SkcomplexError):
    """A solver does not fit the problem or fails to advance it."""


class InfeasibleError(SolverError):
    """A linear program has no solution, and its law asks to raise an error."""


class SimulationError(SkcomplexError):
    """A simulation is used in a state that does not allow the operation."""


class CapacityError(SimulationError):
    """A topology request needs more members than a variable group holds rows for.

    The group is a variable group on a network.
    """


class ExperimentError(SkcomplexError):
    """An experiment cannot run its tasks, such as when its worker processes die."""


class AnalysisError(SkcomplexError):
    """An analysis cannot be applied to the system or the data."""


class EnumerationLimitError(AnalysisError):
    """An exact enumeration does not finish within its time limit."""


class MissingExtraError(SkcomplexError):
    """A class needs an optional extra that is not installed."""


class RegistryError(SkcomplexError):
    """A name is registered twice, or a registration is not valid.

    A name is registered twice with different meanings, or a registration is not valid or misses what it needs.
    """


class BackendWarning(UserWarning):
    """A part of a problem runs on another backend, or a compiled kernel in Python.

    The other backend is not the one requested.
    """


class UnobservedStatesWarning(UserWarning):
    """An analysis drops states that have no observed transitions."""


class ValidationWarning(UserWarning):
    """Warning of a problem that does not stop compilation."""
