# python-support Specification

## Purpose

Define the Python versions the package supports, what must hold before a version is declared, and how every declared
version is verified, so an install on a declared version gets every feature.

## Requirements

### Requirement: The package declares its supported Python versions

The package SHALL declare Python 3.12, 3.13, and 3.14 as supported on Linux and on macOS 13 or later with Apple silicon.
The version range, the classifiers, and the documented versions SHALL agree, and an installer SHALL refuse any version
outside the range.

#### Scenario: Install on Python 3.14

- **WHEN** a user installs the package from the index with Python 3.14 on a supported platform
- **THEN** the install succeeds and the package imports

#### Scenario: Install on a newer Python

- **WHEN** a user installs the package with Python 3.15
- **THEN** the installer reports that no version of the package supports that Python

### Requirement: A Python version is declared only when every dependency installs on it

A Python version SHALL be declared only when every runtime dependency installs on it from a release published on the
package index, on every supported platform. The published metadata SHALL NOT name a dependency by a direct URL.

#### Scenario: A dependency has no release for the version

- **WHEN** a runtime dependency has no published release that installs on a Python version
- **THEN** the version range of the package excludes that Python version

#### Scenario: Published metadata

- **WHEN** a contributor reads the metadata of a built package
- **THEN** every dependency is named by its project and a version range, with no direct URL

### Requirement: Every declared version has the same features

Every model and analysis SHALL be available on every declared Python version, and no feature SHALL be disabled by the
version of Python.

#### Scenario: Markov analyses on Python 3.14

- **WHEN** a user runs the Markov, passage, or emergence analysis on a trajectory with Python 3.14
- **THEN** it returns the same results, within numerical tolerance, as on Python 3.13

### Requirement: Markov results survive a change of library

Replacing the library behind the Markov and passage analyses SHALL keep the names and meaning of their public results,
and their values within numerical tolerance of reference values recorded from the previous library: the transition
matrix, the stationary distribution, the implied timescales, the metastable sets, the committors, and the mean
first-passage times, for the reversible and the non-reversible fit.

#### Scenario: Reversible fit of a known path

- **WHEN** a user estimates a reversible state model from the reference path
- **THEN** its transition matrix, stationary distribution, and implied timescales agree with the recorded reference
  values within 0.001 relative tolerance

#### Scenario: Metastable sets of a double well

- **WHEN** a user asks for two metastable sets of the reference double-well chain
- **THEN** the sets are the same partition of the states as the recorded reference

### Requirement: Every declared version is verified

Continuous integration SHALL run the fast tests on every declared Python version, and on both platforms for the newest
declared version. The full test suite SHALL pass on the newest declared version on Linux before that version is first
declared.

#### Scenario: A test fails only on the newest version

- **WHEN** a change breaks a fast test only on Python 3.14
- **THEN** the continuous integration run fails

### Requirement: A fresh install works

The declared dependency ranges SHALL exclude every version known to break the package, so an install from the index,
which takes the newest versions the ranges allow rather than the lock file, passes the fast tests on the newest declared
Python version.

#### Scenario: Hypergraph model after a fresh install

- **WHEN** a user installs the package from the index into a new environment and simulates `hypergraph_kuramoto`
- **THEN** the simulation runs
