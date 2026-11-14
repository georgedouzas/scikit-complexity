## MODIFIED Requirements

### Requirement: Connections balance flows
A connected network SHALL be written as node elements holding potentials and component interactions whose `Constraint`
laws add their flows into the balance of each node they join, so the flows into every node sum to zero. No connection
class and no port SHALL be needed.

#### Scenario: Kirchhoff's current law
- **WHEN** three resistors meet at one node
- **THEN** the currents into the node sum to zero at every frame

## REMOVED Requirements

### Requirement: Ports carry potential and flow variables
**Reason**: A component binds the nodes it joins through roles, and its constraints add into the balances of those
nodes, so ports, connectors, and `<port>_<variable>` states are not needed.
**Migration**: Replace an element with ports by an interaction with one role per terminal, whose constraints read the
potentials of the bound nodes and add the flow into their balances.
