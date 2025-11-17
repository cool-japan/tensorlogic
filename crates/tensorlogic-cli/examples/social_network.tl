# Social Network Reasoning Rules
# This file demonstrates logic rules for social network analysis

# Transitive friendship: if X knows Y and Y knows Z, then X might know Z
FORALL x IN Person. FORALL y IN Person. FORALL z IN Person.
  (knows(x, y) AND knows(y, z)) -> might_know(x, z)

# Symmetric friendship: if X is friends with Y, then Y is friends with X
FORALL x IN Person. FORALL y IN Person.
  friend(x, y) -> friend(y, x)

# Mutual friends imply potential friendship
FORALL x IN Person. FORALL y IN Person.
  (EXISTS z IN Person. friend(x, z) AND friend(y, z)) -> potential_friend(x, y)

# Compile with:
# tensorlogic social_network.tl --domains Person:100 --validate --analyze
