# Spec Delta

## ADDED Requirements

### Requirement: New nodes attach by weights that read them
In the action `'add-node'`, the attachment weight, the condition, and the probability of `TopologyInteraction` SHALL
read the drawn values of the node being added as `namespace.source`, beside the symbols of the candidate node.

#### Scenario: Attachment by distance
- **WHEN** a network grows by nodes with a drawn angle that attach with weight 1 to candidates within an angular
  distance of 0.1 and with weight 0 to all others
- **THEN** every edge of a new node joins nodes whose angles differ by at most 0.1
