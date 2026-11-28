## ADDED Requirements

### Requirement: Network models match the graphs they construct
The growing networks and the random graph ensembles SHALL reproduce the degree statistics of their reference
constructions.
A Holme-Kim node SHALL hold all its edges in the frame after the step it joins. The Watts-Strogatz
construction SHALL rewire each lattice edge at most once and keep its lattice end. The Erdős-Rényi construction SHALL
reach the requested mean degree or raise a `ValidationError` naming the smallest number of steps that reaches it.

#### Scenario: Holme-Kim nodes join with all their edges
- **WHEN** Holme-Kim growth with 3 edges runs for any number of steps
- **THEN** every node that joined during the run, the newest included, has degree at least 3

#### Scenario: Holme-Kim transitivity follows the triad probability
- **WHEN** Holme-Kim growth with 3 edges reaches 1000 nodes with `triad` 0 and with `triad` 1
- **THEN** at `triad` 0 its transitivity is within 0.02 of that of `nx.powerlaw_cluster_graph(1000, 3, 0)`, and at
  `triad` 1 within 0.02 of the triad formation of Holme and Kim 2002, where triad edges join neighbours of the first
  preferential contact

#### Scenario: Watts-Strogatz degrees stay near the lattice degree
- **WHEN** the Watts-Strogatz construction on 2000 nodes of mean degree 10 rewires all edges over 1 or 10 steps
- **THEN** the minimum degree is at least 5 and the degree variance is within 20 percent of that of
  `nx.watts_strogatz_graph(2000, 10, 1.0)`

#### Scenario: Erdős-Rényi reaches its mean degree or refuses
- **WHEN** the Erdős-Rényi construction on 1000 nodes asks for mean degree 10 in 100 steps, and in 1 step
- **THEN** the first run ends with a mean degree within 5 percent of 10, and the second raises a `ValidationError`
  that names 5 as the smallest number of steps

#### Scenario: Network measures read a growing network
- **WHEN** `network_measures` is given the trajectory of a Barabási-Albert growth
- **THEN** it returns the measures of the last frame
