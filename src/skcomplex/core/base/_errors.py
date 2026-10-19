"""Define the exceptions the library raises."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


class SkcomplexError(Exception):
    """Base exception of the library."""


class ValidationError(SkcomplexError):
    """A value does not belong to the domain of its variable."""


class LabelError(SkcomplexError):
    """A label is missing, duplicated, or produces a name collision."""


class RoleError(SkcomplexError):
    """A role of an interaction refers to an element that does not exist."""


class SpaceError(SkcomplexError):
    """A vector variable has no space to take its dimension from."""


class IncompatibleLawsError(SkcomplexError):
    """A system mixes law kinds that no single form supports."""


class BackendError(SkcomplexError):
    """A backend is not registered or cannot compile a problem."""


class SolverError(SkcomplexError):
    """A solver does not fit the problem or fails to advance it."""


class ParameterError(SkcomplexError):
    """A parameter name is not part of the problem."""


class SimulationError(SkcomplexError):
    """A simulation is used in a state that does not allow the operation."""


class AnalysisError(SkcomplexError):
    """A symbolic analysis cannot be applied to the system."""
