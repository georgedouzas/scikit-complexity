## MODIFIED Requirements

### Requirement: Groupings take a symbolic form
`GroupingInteraction(form='symbolic')` SHALL compile a function of one group as an update of a state of its members,
reduced over the groups that hold a member, with pairs and partitions drawn every step from the random stream of the
simulation. The agent form SHALL stay the default.

#### Scenario: Majority in partitions
- **WHEN** a symbolic grouping of partitions of size 3 sets every member to the majority opinion of its group
- **THEN** after one step every group of the drawn partition holds one opinion, and two runs with one seed draw the
  same partitions

#### Scenario: Agent setting in the symbolic form
- **WHEN** a symbolic grouping is given `act`
- **THEN** `prepare` raises `ValidationError` that names `act` as a setting of the agent form
