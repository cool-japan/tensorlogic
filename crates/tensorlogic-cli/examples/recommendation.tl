# Recommendation System Rules
# Content-based and collaborative filtering logic

# If user likes an item and items are similar, recommend the similar item
FORALL u IN User. FORALL i IN Item. FORALL j IN Item.
  (liked(u, i) AND similar(i, j) AND NOT liked(u, j)) -> recommend(u, j)

# Collaborative filtering: if similar users like an item, recommend it
FORALL u IN User. FORALL v IN User. FORALL i IN Item.
  (similar_users(u, v) AND liked(v, i) AND NOT liked(u, i)) -> recommend(u, i)

# Popular items in user's category
FORALL u IN User. FORALL i IN Item.
  (interested_in_category(u, c) AND in_category(i, c) AND popular(i)) -> recommend(u, i)

# Trending items
FORALL u IN User. FORALL i IN Item.
  (active(u) AND trending(i) AND NOT viewed(u, i)) -> recommend(u, i)

# Compile with:
# tensorlogic recommendation.tl \
#   --domains User:10000 --domains Item:50000 \
#   --strategy probabilistic \
#   --output-format stats \
#   --analyze
