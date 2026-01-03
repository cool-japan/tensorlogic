//! Example 18: Schema Embeddings and Similarity Search
//!
//! This example demonstrates the ML-based schema embedding system,
//! showing how to generate vector embeddings for domains and predicates
//! and use them for similarity search.

use tensorlogic_adapters::{
    DomainInfo, PredicateInfo, SchemaEmbedder, SimilaritySearch, SymbolTable,
};

fn main() {
    println!("=== Schema Embeddings and Similarity Search ===\n");

    // Create a schema with various domains
    let mut table = SymbolTable::new();

    // People-related domains
    table
        .add_domain(DomainInfo::new("Person", 1000).with_description("A human person"))
        .unwrap();
    table
        .add_domain(DomainInfo::new("Student", 500).with_description("A student"))
        .unwrap();
    table
        .add_domain(DomainInfo::new("Teacher", 100).with_description("A teacher"))
        .unwrap();

    // Other domains
    table
        .add_domain(DomainInfo::new("Course", 50).with_description("An academic course"))
        .unwrap();
    table
        .add_domain(DomainInfo::new("Book", 5000).with_description("A book"))
        .unwrap();
    table
        .add_domain(DomainInfo::new("Building", 200).with_description("A physical building"))
        .unwrap();

    // Add predicates
    table
        .add_predicate(PredicateInfo::new(
            "knows",
            vec!["Person".to_string(), "Person".to_string()],
        ))
        .unwrap();
    table
        .add_predicate(PredicateInfo::new(
            "friends_with",
            vec!["Person".to_string(), "Person".to_string()],
        ))
        .unwrap();
    table
        .add_predicate(PredicateInfo::new(
            "teaches",
            vec!["Teacher".to_string(), "Course".to_string()],
        ))
        .unwrap();
    table
        .add_predicate(PredicateInfo::new(
            "enrolled",
            vec!["Student".to_string(), "Course".to_string()],
        ))
        .unwrap();

    println!(
        "Schema indexed with {} domains and {} predicates\n",
        table.domains.len(),
        table.predicates.len()
    );

    // 1. Generate embeddings for individual domains
    println!("=== 1. Domain Embeddings ===\n");
    let embedder = SchemaEmbedder::new();

    let person_domain = table.domains.get("Person").unwrap();
    let person_emb = embedder.embed_domain(person_domain);
    println!(
        "Person domain embedding (first 10 dims): {:?}",
        &person_emb[..10]
    );

    let course_domain = table.domains.get("Course").unwrap();
    let course_emb = embedder.embed_domain(course_domain);
    println!(
        "Course domain embedding (first 10 dims): {:?}\n",
        &course_emb[..10]
    );

    // 2. Compute similarities between domains
    println!("=== 2. Domain Similarity Analysis ===\n");

    let student_domain = table.domains.get("Student").unwrap();
    let teacher_domain = table.domains.get("Teacher").unwrap();
    let building_domain = table.domains.get("Building").unwrap();

    let student_emb = embedder.embed_domain(student_domain);
    let teacher_emb = embedder.embed_domain(teacher_domain);
    let building_emb = embedder.embed_domain(building_domain);

    let sim_person_student = SchemaEmbedder::cosine_similarity(&person_emb, &student_emb);
    let sim_person_teacher = SchemaEmbedder::cosine_similarity(&person_emb, &teacher_emb);
    let sim_person_building = SchemaEmbedder::cosine_similarity(&person_emb, &building_emb);
    let sim_student_teacher = SchemaEmbedder::cosine_similarity(&student_emb, &teacher_emb);

    println!("Similarity scores:");
    println!("  Person <-> Student:  {:.4}", sim_person_student);
    println!("  Person <-> Teacher:  {:.4}", sim_person_teacher);
    println!("  Person <-> Building: {:.4}", sim_person_building);
    println!("  Student <-> Teacher: {:.4}\n", sim_student_teacher);

    println!("Analysis:");
    println!("  - Person/Student and Person/Teacher have high similarity (both people)");
    println!("  - Person/Building has low similarity (different concepts)");
    println!("  - Student/Teacher have moderate similarity (both in education domain)\n");

    // 3. Similarity search
    println!("=== 3. Similarity Search ===\n");

    let mut search = SimilaritySearch::new();
    search.index_table(&table);

    println!("Search statistics:");
    let stats = search.stats();
    println!("  - Indexed domains: {}", stats.num_domains);
    println!("  - Indexed predicates: {}", stats.num_predicates);
    println!("  - Embedding dimension: {}\n", stats.embedding_dim);

    // Find similar domains
    println!("Finding domains similar to 'Person':");
    let similar_to_person = search.find_similar_domains_by_name("Person", 3);
    for (i, (name, similarity)) in similar_to_person.iter().enumerate() {
        println!("  {}. {} (similarity: {:.4})", i + 1, name, similarity);
    }
    println!();

    // Find similar predicates
    println!("Finding predicates similar to 'knows':");
    let similar_to_knows = search.find_similar_predicates_by_name("knows", 3);
    for (i, (name, similarity)) in similar_to_knows.iter().enumerate() {
        println!("  {}. {} (similarity: {:.4})", i + 1, name, similarity);
    }
    println!();

    // 4. Predicate embeddings
    println!("=== 4. Predicate Embeddings ===\n");

    let knows_pred = table.predicates.get("knows").unwrap();
    let teaches_pred = table.predicates.get("teaches").unwrap();

    let knows_emb = embedder.embed_predicate(knows_pred);
    let teaches_emb = embedder.embed_predicate(teaches_pred);

    let sim_knows_teaches = SchemaEmbedder::cosine_similarity(&knows_emb, &teaches_emb);
    println!(
        "Similarity between 'knows' and 'teaches': {:.4}",
        sim_knows_teaches
    );
    println!("Both are binary predicates, hence moderate similarity\n");

    // 5. Schema-level embedding
    println!("=== 5. Schema-Level Embedding ===\n");

    let schema_emb = embedder.embed_schema(&table);
    println!(
        "Complete schema embedding generated (dimension: {})",
        schema_emb.len()
    );
    println!(
        "Schema embedding (first 10 dims): {:?}\n",
        &schema_emb[..10]
    );

    // 6. Distance metrics
    println!("=== 6. Distance Metrics ===\n");

    let dist_person_student = SchemaEmbedder::euclidean_distance(&person_emb, &student_emb);
    let dist_person_building = SchemaEmbedder::euclidean_distance(&person_emb, &building_emb);

    println!("Euclidean distances:");
    println!("  Person <-> Student:  {:.4}", dist_person_student);
    println!("  Person <-> Building: {:.4}", dist_person_building);
    println!("\nSmaller distance = more similar\n");

    // Summary
    println!("=== Summary ===");
    println!("✓ Generated vector embeddings for all schema elements");
    println!("✓ Computed similarity scores using cosine similarity");
    println!("✓ Performed similarity search to find related elements");
    println!("✓ Demonstrated both similarity and distance metrics");
    println!("\nEmbeddings enable ML-based schema analysis, recommendation,");
    println!("and intelligent search across large schema collections!");
}
