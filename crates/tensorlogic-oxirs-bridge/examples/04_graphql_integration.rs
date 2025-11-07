//! GraphQL schema to TensorLogic integration example
//!
//! This example demonstrates how to:
//! 1. Define a GraphQL schema
//! 2. Parse and extract types and fields
//! 3. Convert to TensorLogic SymbolTable
//! 4. Map GraphQL types to logical domains and predicates
//!
//! Run with: cargo run --example 04_graphql_integration -p tensorlogic-oxirs-bridge

use anyhow::Result;
use tensorlogic_oxirs_bridge::GraphQLConverter;

fn main() -> Result<()> {
    println!("=== GraphQL Schema to TensorLogic Integration ===\n");

    // Define a GraphQL schema for a social network application
    let graphql_schema = r#"
        type User {
            id: ID!
            username: String!
            email: String!
            age: Int
            bio: String
            friends: [User!]
            posts: [Post!]
            profile: Profile
        }

        type Profile {
            avatarUrl: String
            location: String
            website: String
            verified: Boolean!
        }

        type Post {
            id: ID!
            title: String!
            content: String!
            author: User!
            tags: [String!]
            likes: Int!
            createdAt: String!
            comments: [Comment!]
        }

        type Comment {
            id: ID!
            text: String!
            author: User!
            post: Post!
            createdAt: String!
        }

        type Group {
            id: ID!
            name: String!
            description: String
            members: [User!]
            admins: [User!]
            posts: [Post!]
        }

        type Query {
            user(id: ID!): User
            post(id: ID!): Post
            group(id: ID!): Group
            searchUsers(query: String!): [User!]
        }

        type Mutation {
            createUser(username: String!, email: String!): User
            createPost(title: String!, content: String!): Post
            addFriend(userId: ID!, friendId: ID!): Boolean
        }
    "#;

    // Step 1: Parse GraphQL schema
    println!("Step 1: Parsing GraphQL schema...");
    let mut converter = GraphQLConverter::new();
    let symbol_table = converter.parse_schema(graphql_schema)?;
    println!("✓ Schema parsed successfully\n");

    // Step 2: Display extracted GraphQL types
    println!("Step 2: Extracted GraphQL Types");
    println!("{}", "=".repeat(70));
    for (type_name, type_def) in converter.types() {
        // Skip special types for display
        if matches!(type_name.as_str(), "Query" | "Mutation" | "Subscription") {
            continue;
        }

        println!("\nType: {}", type_name);
        println!("  Fields ({})", type_def.fields.len());
        for field in &type_def.fields {
            let required = if field.is_required { "!" } else { "" };
            let list_marker = if field.is_list {
                format!("[{}{}]", field.field_type, required)
            } else {
                format!("{}{}", field.field_type, required)
            };

            print!("    - {}: {}", field.name, list_marker);

            if !field.arguments.is_empty() {
                print!(" (");
                for (i, arg) in field.arguments.iter().enumerate() {
                    if i > 0 {
                        print!(", ");
                    }
                    let arg_required = if arg.is_required { "!" } else { "" };
                    print!("{}: {}{}", arg.name, arg.arg_type, arg_required);
                }
                print!(")");
            }
            println!();
        }
    }

    // Step 3: Display TensorLogic SymbolTable mapping
    println!("\n\nStep 3: TensorLogic SymbolTable Mapping");
    println!("{}", "=".repeat(70));

    println!("\nDomains (Types → Domains):");
    println!("  Scalar types:");
    for (name, domain) in &symbol_table.domains {
        if matches!(
            name.as_str(),
            "String" | "Int" | "Float" | "Boolean" | "ID" | "Value"
        ) {
            println!("    - {} (cardinality: {})", name, domain.cardinality);
        }
    }

    println!("\n  Object types:");
    for (name, domain) in &symbol_table.domains {
        if !matches!(
            name.as_str(),
            "String" | "Int" | "Float" | "Boolean" | "ID" | "Value"
        ) {
            println!("    - {} (cardinality: {})", name, domain.cardinality);
        }
    }

    println!("\nPredicates (Fields → Predicates):");
    for (name, predicate) in &symbol_table.predicates {
        println!("  - {}", name);
        println!(
            "    Signature: ({}) → {}",
            predicate
                .arg_domains
                .iter()
                .take(predicate.arg_domains.len().saturating_sub(1))
                .cloned()
                .collect::<Vec<_>>()
                .join(", "),
            predicate
                .arg_domains
                .last()
                .unwrap_or(&"Unknown".to_string())
        );
    }

    // Step 4: Example logical rules that could be defined
    println!("\n\nStep 4: Example Logical Rules (Potential)");
    println!("{}", "=".repeat(70));

    println!("\nWith this SymbolTable, you can define rules like:");
    println!();
    println!("1. Friendship Symmetry:");
    println!("   User_friends(u1, u2) → User_friends(u2, u1)");
    println!();
    println!("2. Post Author Consistency:");
    println!("   Post_author(p, u) ∧ User_posts(u, p2) → (p = p2)");
    println!();
    println!("3. Comment on Post:");
    println!("   Comment_post(c, p) → Post_comments(p, c)");
    println!();
    println!("4. Group Membership:");
    println!("   Group_members(g, u) → ∃p. User_posts(u, p) ∧ Group_posts(g, p)");
    println!();
    println!("5. Verified Users:");
    println!("   User_profile(u, prof) ∧ Profile_verified(prof, true) → verifiedUser(u)");

    // Step 5: Statistics
    println!("\n\nStep 5: Conversion Statistics");
    println!("{}", "=".repeat(70));
    println!("GraphQL types parsed: {}", converter.types().len());
    println!(
        "TensorLogic domains created: {}",
        symbol_table.domains.len()
    );
    println!(
        "TensorLogic predicates created: {}",
        symbol_table.predicates.len()
    );

    // Calculate average arity
    let total_arity: usize = symbol_table
        .predicates
        .values()
        .map(|p| p.arg_domains.len())
        .sum();
    let avg_arity = if symbol_table.predicates.is_empty() {
        0.0
    } else {
        total_arity as f64 / symbol_table.predicates.len() as f64
    };
    println!("Average predicate arity: {:.2}", avg_arity);

    // Count predicates by arity
    let mut arity_counts = std::collections::HashMap::new();
    for predicate in symbol_table.predicates.values() {
        *arity_counts.entry(predicate.arg_domains.len()).or_insert(0) += 1;
    }
    println!("\nPredicates by arity:");
    for (arity, count) in arity_counts.iter() {
        println!("  Arity {}: {} predicates", arity, count);
    }

    println!("\n=== Example Complete ===");
    println!("\nNext Steps:");
    println!("  1. Define TLExpr rules using the generated predicates");
    println!("  2. Add GraphQL directives as SHACL-like constraints");
    println!("  3. Compile and execute with TensorLogic backend");
    println!("  4. Use for knowledge graph reasoning over GraphQL data");

    Ok(())
}
