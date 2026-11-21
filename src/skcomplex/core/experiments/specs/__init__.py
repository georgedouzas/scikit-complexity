"""Define the specs that experiments send to the processes that run them.

A spec holds one run of an experiment as a value, such as a run of a sweep with its parameters and seed, so a worker
process receives it as it is.
"""
