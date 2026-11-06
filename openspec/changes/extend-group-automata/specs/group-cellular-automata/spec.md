## ADDED Requirements

### Requirement: Local rules read at state-dependent group elements
A local rule on a group space SHALL read, at every site `g`, the cells at the products `g x(g) s` for the words `s` of
a memory set, where `x(g)` is the group element that the first read state holds as the number of its site. On a finite
group every product SHALL be a site. On a ball a product outside it, or reached through a site outside it, SHALL read
the fill value of the space, as SHALL a value that names no site. A rule over such a memory SHALL run on the
synchronous and random schedules, and SHALL be rejected with a named error on the checkerboard and colored schedules
and when it asks for degrees. A space that is no group space SHALL reject such a memory with a named error.

#### Scenario: The self-indexed shift on a finite group
- **WHEN** the rule `τ(x)(g) = x(g · x(g))` runs one step on the symmetric group of degree 3
- **THEN** every cell takes the value of the cell at the product of its site and its value in the group

#### Scenario: Products leave a ball
- **WHEN** the same rule runs on a ball of the free group
- **THEN** a cell whose product lies outside the ball takes the fill value

#### Scenario: Equivariance on a finite group
- **WHEN** a configuration is translated by a group element before one step on a finite group
- **THEN** the image is the translate of the image of the configuration

### Requirement: States take values in truncated power series rings
A state SHALL take values in the ring `K[[t]]` of formal power series over the rationals, the reals, or a prime field,
truncated at a declared order, holding every series by its coefficients as components. The ring SHALL multiply series
by the Cauchy product truncated at the order, reduce coefficients modulo the characteristic, and convert coefficients to
and from Sage's power series ring of the same precision. An order that is not a positive integer or an unknown field
SHALL raise a named error.

#### Scenario: Truncated product
- **WHEN** `1 + t` and `1 - t` are multiplied in the ring truncated at order 4
- **THEN** the coefficients of the product are `1, 0, -1, 0`

#### Scenario: Coefficients round-trip through Sage
- **WHEN** coefficients are converted to a Sage power series and back
- **THEN** they are unchanged

### Requirement: Local rules update states of several components
A function local rule SHALL read and set states of several components per cell, receiving one row of components per
cell for every neighbour, and returning one row per cell. Table and totalistic rules SHALL reject such states with a
named error.

#### Scenario: Power series automaton
- **WHEN** the automaton `τ(x)(n) = x(n) - t x(n + 1)` over the rationals runs three steps from a single 1
- **THEN** the cells hold the coefficients of `(1 - t σ)^3` applied to it, the binomial coefficients with alternating
  signs at the matching powers of `t`

### Requirement: The CC and EX automata are catalog systems
The catalog SHALL include, in `computer_science/automata`, the models of Ceccherini-Silberstein and Coornaert (2023)
and of its exercises book that the reference audit lists as missing: the self-indexed shift (Example 1.8.2), the power
series automaton (Exercise 1.37), the subtraction automaton over the naturals (Example 3.3.3), Muller's pre-injective
but not surjective automaton on F2 (Sections 5.11 and 8.10), the surjective but not pre-injective linear automaton on F2
(Section 8.11), Life in a tree on F4 (Exercise 5.14), the nilpotent automaton of class n (Exercise 1.15), the
involutive automaton (Exercise 1.36), the group-valued difference automaton (Exercise 3.24), the real quadratic
automaton (Exercise 5.64), the even and S-gap subshifts (Exercises 1.85 and 1.92), the Morse subshift (Exercises 3.51
and 3.52), the context-free subshift (Exercise 6.80), and the adding machine (Exercise 6.59). Each SHALL have tests, an
inventory entry citing its section, and a gallery example.

#### Scenario: Muller's automaton is not surjective
- **WHEN** Muller's automaton runs one step
- **THEN** the second coordinate of every cell is 0, and a nonzero configuration of finite support inside the ball has
  a nonzero image

#### Scenario: The surjective linear automaton is not pre-injective
- **WHEN** the surjective linear automaton runs one step from `(0, 1)` at the identity and 0 elsewhere
- **THEN** every cell is 0

#### Scenario: Nilpotency and involution
- **WHEN** the nilpotent automaton of class n runs n steps from any configuration of a ring, and the involutive
  automaton runs two steps
- **THEN** the first ends at the zero configuration, which n - 1 steps do not always reach, and the second returns to
  its start

#### Scenario: Subshift samplers stay admissible
- **WHEN** the S-gap and the context-free subshift samplers run
- **THEN** every configuration they visit satisfies the constraint of its subshift

#### Scenario: The adding machine counts
- **WHEN** the adding machine on `L` digits in base `q` runs `q^L` steps
- **THEN** its digits read every residue modulo `q^L` once, in order, and return to the start
