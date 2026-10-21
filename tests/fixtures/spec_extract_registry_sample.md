<!-- spec-extract widgets/tree@67c292f93010; cited lines are at commit 67c292f93010 -->

## ADDED Requirements

### Requirement: Adds two numbers
add SHALL return the sum of its arguments.

#### Scenario: adding
- **WHEN** add(1, 2) is called
- **THEN** it returns 3
- **CITES** `src/calc.py:1-2`
- **CITES** `tests/test_calc.py:5-5`
- **EVIDENCE** pinned by test

### Requirement: Subtracts two numbers
sub SHALL return a minus b.

#### Scenario: subtracting
- **WHEN** sub(3, 1) is called
- **THEN** it returns 2
- **CITES** `src/calc.py:5-6`
- **EVIDENCE** read from code only
