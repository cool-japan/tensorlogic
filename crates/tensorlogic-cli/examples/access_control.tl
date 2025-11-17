# Access Control Policy Rules
# Demonstrates role-based access control logic

# Admins can access all resources
FORALL u IN User. FORALL r IN Resource.
  has_role(u, admin) -> can_access(u, r)

# Owners can access their own resources if not locked
FORALL u IN User. FORALL r IN Resource.
  (owns(u, r) AND NOT locked(r)) -> can_access(u, r)

# Shared resources are accessible to group members
FORALL u IN User. FORALL r IN Resource. FORALL g IN Group.
  (member(u, g) AND shared_with(r, g)) -> can_access(u, r)

# Read permission implies can_view
FORALL u IN User. FORALL r IN Resource.
  (can_access(u, r) AND has_permission(u, r, read)) -> can_view(u, r)

# Write permission requires both access and write rights
FORALL u IN User. FORALL r IN Resource.
  (can_access(u, r) AND has_permission(u, r, write) AND NOT readonly(r)) -> can_modify(u, r)

# Compile with:
# tensorlogic access_control.tl \
#   --domains User:1000 --domains Resource:5000 --domains Group:100 \
#   --strategy hard_boolean \
#   --validate
