"""Share the solver of the evolution model tests."""

from skcomplex.engine.solvers import ScipyODE

PRECISE = ScipyODE(method='DOP853', rtol=1e-12, atol=1e-14)
