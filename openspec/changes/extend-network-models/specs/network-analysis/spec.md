## ADDED Requirements

### Requirement: Summary components are arrays
`network_measures(kind='summary')` SHALL return its `components` as one integer array per frame, the sizes of the
connected components in decreasing order, like the `degrees` of the kind `'degrees'`.

#### Scenario: Two components
- **WHEN** the summary of the disjoint union of a cycle of 5 nodes and a path of 3 nodes is measured
- **THEN** `components` holds one integer NumPy array equal to `[5, 3]`
