"""Share the solver of the evolution model tests."""

from skcomplex.engine.solvers import ScipyODESolver

PRECISE = ScipyODESolver(method='DOP853', rtol=1e-12, atol=1e-14)
