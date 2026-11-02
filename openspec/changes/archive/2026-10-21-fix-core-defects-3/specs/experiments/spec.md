# Spec Delta

## ADDED Requirements

### Requirement: Dead worker processes raise a named error
An experiment run on processes SHALL raise a named error when its worker processes die, and the message SHALL name the
main guard that a script needs and the executors that avoid processes.

#### Scenario: Script without a main guard
- **WHEN** a script without `if __name__ == '__main__':` builds an orbit diagram with `executor='process'`
- **THEN** it raises `ExperimentError`, whose message names the main guard and `executor='thread'`
