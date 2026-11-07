# Spec Delta

## MODIFIED Requirements

### Requirement: Agent traits without a space are drawn
The default visual of an agent group without a space SHALL log the histogram of a numeric trait and the mean and
standard deviation of every numeric trait. The histogram of every trait SHALL bin by the range of its own values,
widened to every value logged so far, and the color range of the visual SHALL only start the range of its colored
trait.

#### Scenario: Moran process
- **WHEN** the default blueprint of `moran_process` is built and its visual logs 10 frames
- **THEN** the first view of the agents is a bar chart of a trait, and the logs hold the mean of every numeric trait

#### Scenario: Integer trait outside the color range
- **WHEN** a histogram visual colored by a trait in `(0, 1)` logs an integer trait with negative values
- **THEN** the histogram of that trait counts every agent and its range covers the negative values
