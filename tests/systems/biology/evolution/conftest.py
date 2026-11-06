"""Share the solver of the evolution system tests."""

from skcomplex.core.engine.solvers import ScipyODESolver

PRECISE = ScipyODESolver(method='DOP853', rtol=1e-12, atol=1e-14)
