# Data Validation Rules
# Quality checks and constraints for data records

# Age must be within valid range
FORALL r IN Record.
  (age(r) >= 0 AND age(r) <= 150) -> valid_age(r)

# Email must be present for active users
FORALL r IN Record.
  (status(r, active) -> EXISTS e IN Email. has_email(r, e))

# Name must have minimum length
FORALL r IN Record.
  (length(name(r)) > 0 AND length(name(r)) < 200) -> valid_name(r)

# Phone number format validation
FORALL r IN Record.
  (has_phone(r) -> (length(phone(r)) >= 10 AND length(phone(r)) <= 15))

# Complete record validation
FORALL r IN Record.
  (valid_age(r) AND valid_name(r) AND has_email(r, e)) -> valid_record(r)

# Compile with:
# tensorlogic data_validation.tl \
#   --domains Record:1000000 --domains Email:1000000 \
#   --strategy hard_boolean \
#   --validate
