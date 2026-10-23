"""Define the exceptions the library raises."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


class SkcomplexError(Exception):
    """Base exception of the library."""


class ValidationError(SkcomplexError):
    """A system, a value, or a combination of a system with its solver and backend is not valid."""


class ConservationError(ValidationError):
    """A conserved quantity changes outside its flows, or a balance-sheet identity does not hold."""


class LabelError(ValidationError):
    """A label is missing, duplicated, or produces a name collision."""


class RoleError(ValidationError):
    """A role of an interaction refers to an element that does not exist."""


class SpaceError(ValidationError):
    """A vector variable has no space to take its dimension from."""


class IncompatibleLawsError(ValidationError):
    """A system mixes law kinds that no single form supports."""


class BackendError(SkcomplexError):
    """A backend is not registered or cannot compile a problem."""


class SolverError(SkcomplexError):
    """A solver does not fit the problem or fails to advance it."""


class HorizonError(SolverError):
    """A stepper with a fixed horizon is asked to advance beyond it."""


class ParameterError(SkcomplexError):
    """A parameter name is not part of the problem."""


class SimulationError(SkcomplexError):
    """A simulation is used in a state that does not allow the operation."""


class CapacityError(SimulationError):
    """A topology request needs more members than a variable group on a network holds rows for."""


class ExperimentError(SkcomplexError):
    """An experiment cannot run its tasks, such as when its worker processes die."""


class AnalysisError(SkcomplexError):
    """An analysis cannot be applied to the system or the data."""


class RewritingError(SkcomplexError):
    """A rewriting system is malformed, or a generation would exceed its maximum size."""


class HilbertSpaceError(SkcomplexError):
    """A quantum system has a Hilbert space larger than its declared limit."""


class UnobservedStatesWarning(UserWarning):
    """An analysis drops states that have no observed transitions."""
