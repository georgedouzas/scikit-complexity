# test-suite Specification

## Purpose
Keeps the default test run fast, while the gate still runs every test with branch coverage.

## Requirements

### Requirement: Slow tests run on request
A test that takes more than 10 seconds on an idle machine SHALL carry the `slow` marker. A plain `pytest` run SHALL
skip slow tests, and `pytest --slow` SHALL run them. The `tests` session of nox SHALL run them.

#### Scenario: Default run
- **WHEN** a contributor runs `pytest` with no options
- **THEN** every test marked `slow` is reported as skipped with the reason that names `--slow`

#### Scenario: Gate run
- **WHEN** the `tests` session of nox runs
- **THEN** the slow tests run

### Requirement: Coverage only in the gate
A plain `pytest` run SHALL not measure coverage, and the `tests` session of nox SHALL measure branch coverage of `src`
and `tests`.

#### Scenario: Plain run
- **WHEN** a contributor runs `pytest tests/core/components/laws`
- **THEN** the run reports no coverage and writes no coverage file

#### Scenario: Gate coverage
- **WHEN** the `tests` session of nox runs
- **THEN** it reports branch coverage and writes the combined coverage report

### Requirement: No test runs for minutes
No single test SHALL take more than 60 seconds on an idle machine.

#### Scenario: Slowest test
- **WHEN** the full suite runs with `--slow --durations=5` on an idle machine
- **THEN** every reported duration is below 60 seconds
