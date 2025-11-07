//! Arithmetic and Comparison Operations
//!
//! This example demonstrates how to use arithmetic and comparison operations
//! in TensorLogic IR for mixed logical-numeric reasoning.

use tensorlogic_ir::{IrError, TLExpr, Term};

fn main() -> Result<(), IrError> {
    println!("=== TensorLogic IR: Arithmetic & Comparison ===\n");

    // 1. Arithmetic Operations
    println!("1. Arithmetic Operations:");

    // Add: score(x) + bonus
    let score = TLExpr::pred("score", vec![Term::var("x")]);
    let bonus = TLExpr::constant(10.0);
    let _total_score = TLExpr::add(score.clone(), bonus);
    println!("   Add: score(x) + 10");

    // Subtract: temperature - baseline
    let temp = TLExpr::pred("temperature", vec![Term::var("t")]);
    let baseline = TLExpr::constant(20.0);
    let _temp_diff = TLExpr::sub(temp.clone(), baseline);
    println!("   Subtract: temperature(t) - 20");

    // Multiply: price * quantity
    let price = TLExpr::pred("price", vec![Term::var("item")]);
    let quantity = TLExpr::constant(5.0);
    let _total_cost = TLExpr::mul(price, quantity);
    println!("   Multiply: price(item) * 5");

    // Divide: total / count
    let total = TLExpr::pred("total", vec![Term::var("x")]);
    let count = TLExpr::constant(10.0);
    let _average = TLExpr::div(total, count);
    println!("   Divide: total(x) / 10");

    // 2. Numeric Constants
    println!("\n2. Numeric Constants:");

    let _pi = TLExpr::constant(std::f64::consts::PI);
    let _zero = TLExpr::constant(0.0);
    let _negative = TLExpr::constant(-5.5);

    println!("   π = {:?}", _pi);
    println!("   Zero = {:?}", _zero);
    println!("   Negative = {:?}", _negative);

    // 3. Comparison Operations
    println!("\n3. Comparison Operations:");

    // Equal: age(x) == 18
    let age = TLExpr::pred("age", vec![Term::var("x")]);
    let adult_age = TLExpr::constant(18.0);
    let _is_adult = TLExpr::eq(age.clone(), adult_age);
    println!("   Equal: age(x) == 18");

    // Less than: temperature < 0
    let temp = TLExpr::pred("temperature", vec![Term::var("t")]);
    let freezing = TLExpr::constant(0.0);
    let _is_freezing = TLExpr::lt(temp.clone(), freezing);
    println!("   Less than: temperature(t) < 0");

    // Greater than: score > 100
    let score = TLExpr::pred("score", vec![Term::var("x")]);
    let threshold = TLExpr::constant(100.0);
    let _high_score = TLExpr::gt(score.clone(), threshold);
    println!("   Greater than: score(x) > 100");

    // Less than or equal: age <= 65
    let age = TLExpr::pred("age", vec![Term::var("x")]);
    let retirement = TLExpr::constant(65.0);
    let _working_age = TLExpr::lte(age.clone(), retirement);
    println!("   Less than or equal: age(x) <= 65");

    // Greater than or equal: height >= 180
    let height = TLExpr::pred("height", vec![Term::var("x")]);
    let tall_threshold = TLExpr::constant(180.0);
    let _is_tall = TLExpr::gte(height, tall_threshold);
    println!("   Greater than or equal: height(x) >= 180");

    // 4. Combining Arithmetic and Logic
    println!("\n4. Combining Arithmetic and Logic:");

    // (score(x) + bonus) > threshold ∧ active(x)
    let score = TLExpr::pred("score", vec![Term::var("x")]);
    let bonus = TLExpr::constant(20.0);
    let threshold = TLExpr::constant(100.0);
    let total_score = TLExpr::add(score, bonus);
    let high_enough = TLExpr::gt(total_score, threshold);
    let active = TLExpr::pred("active", vec![Term::var("x")]);
    let _qualified = TLExpr::and(high_enough, active);
    println!("   (score(x) + 20) > 100 ∧ active(x)");

    // 5. Complex Arithmetic Expressions
    println!("\n5. Complex Arithmetic Expressions:");

    // (price * quantity) + shipping
    let price = TLExpr::pred("price", vec![Term::var("item")]);
    let quantity = TLExpr::constant(3.0);
    let shipping = TLExpr::constant(10.0);
    let subtotal = TLExpr::mul(price, quantity);
    let _total = TLExpr::add(subtotal, shipping);
    println!("   (price(item) * 3) + 10");

    // (a + b) * c
    let a = TLExpr::pred("a", vec![Term::var("x")]);
    let b = TLExpr::pred("b", vec![Term::var("x")]);
    let c = TLExpr::constant(2.0);
    let sum = TLExpr::add(a, b);
    let _product = TLExpr::mul(sum, c);
    println!("   (a(x) + b(x)) * 2");

    // ((x + 1) * 2) - 3
    let x = TLExpr::pred("x", vec![Term::var("i")]);
    let plus_one = TLExpr::add(x, TLExpr::constant(1.0));
    let times_two = TLExpr::mul(plus_one, TLExpr::constant(2.0));
    let _final_expr = TLExpr::sub(times_two, TLExpr::constant(3.0));
    println!("   ((x(i) + 1) * 2) - 3");

    // 6. Conditional Expressions (If-Then-Else)
    println!("\n6. Conditional Expressions:");

    // if score > 90 then "A" else "B"
    let score = TLExpr::pred("score", vec![Term::var("x")]);
    let ninety = TLExpr::constant(90.0);
    let condition = TLExpr::gt(score, ninety);
    let grade_a = TLExpr::constant(4.0); // GPA for A
    let grade_b = TLExpr::constant(3.0); // GPA for B
    let _grade = TLExpr::if_then_else(condition, grade_a, grade_b);
    println!("   if score(x) > 90 then 4.0 else 3.0");

    // Nested conditional: if x > 10 then (if x > 20 then 3 else 2) else 1
    let x = TLExpr::pred("x", vec![Term::var("i")]);
    let ten = TLExpr::constant(10.0);
    let twenty = TLExpr::constant(20.0);
    let cond_outer = TLExpr::gt(x.clone(), ten);
    let cond_inner = TLExpr::gt(x, twenty);
    let inner_if = TLExpr::if_then_else(cond_inner, TLExpr::constant(3.0), TLExpr::constant(2.0));
    let _nested_if = TLExpr::if_then_else(cond_outer, inner_if, TLExpr::constant(1.0));
    println!("   Nested: if x > 10 then (if x > 20 then 3 else 2) else 1");

    // 7. Range Checking
    println!("\n7. Range Checking:");

    // Check if value is in range [min, max]: value >= min ∧ value <= max
    let value = TLExpr::pred("value", vec![Term::var("x")]);
    let min = TLExpr::constant(0.0);
    let max = TLExpr::constant(100.0);
    let gte_min = TLExpr::gte(value.clone(), min);
    let lte_max = TLExpr::lte(value, max);
    let _in_range = TLExpr::and(gte_min, lte_max);
    println!("   Range [0, 100]: value(x) >= 0 ∧ value(x) <= 100");

    // 8. Threshold-Based Classification
    println!("\n8. Threshold-Based Classification:");

    // Classify temperature: hot if > 30, cold if < 10, otherwise comfortable
    let temp = TLExpr::pred("temperature", vec![Term::var("t")]);
    let hot_threshold = TLExpr::constant(30.0);
    let cold_threshold = TLExpr::constant(10.0);

    // is_hot = temp > 30
    let is_hot = TLExpr::gt(temp.clone(), hot_threshold);
    println!("   Hot: temperature(t) > 30");

    // is_cold = temp < 10
    let is_cold = TLExpr::lt(temp.clone(), cold_threshold);
    println!("   Cold: temperature(t) < 10");

    // is_comfortable = ¬is_hot ∧ ¬is_cold
    let not_hot = TLExpr::negate(is_hot);
    let not_cold = TLExpr::negate(is_cold);
    let _is_comfortable = TLExpr::and(not_hot, not_cold);
    println!("   Comfortable: ¬(temp > 30) ∧ ¬(temp < 10)");

    // 9. Practical Example: Discount Calculation
    println!("\n9. Practical Example: Discount Calculation");

    // discount = if quantity > 10 then price * 0.9 else price
    let quantity = TLExpr::pred("quantity", vec![Term::var("item")]);
    let price = TLExpr::pred("price", vec![Term::var("item")]);
    let bulk_threshold = TLExpr::constant(10.0);
    let discount_rate = TLExpr::constant(0.9);

    let is_bulk = TLExpr::gt(quantity, bulk_threshold);
    let discounted_price = TLExpr::mul(price.clone(), discount_rate);
    let _final_price = TLExpr::if_then_else(is_bulk, discounted_price, price);

    println!("   if quantity(item) > 10 then price(item) * 0.9 else price(item)");
    println!("   'Apply 10% discount for bulk orders'");

    // 10. Statistical Operations
    println!("\n10. Statistical Operations:");

    // normalized_score = (score - mean) / std_dev
    let score = TLExpr::pred("score", vec![Term::var("x")]);
    let mean = TLExpr::constant(75.0);
    let std_dev = TLExpr::constant(15.0);
    let centered = TLExpr::sub(score, mean);
    let _normalized = TLExpr::div(centered, std_dev);
    println!("   Z-score: (score(x) - 75) / 15");

    // weighted_average = (score1 * 0.6) + (score2 * 0.4)
    let score1 = TLExpr::pred("score1", vec![Term::var("x")]);
    let score2 = TLExpr::pred("score2", vec![Term::var("x")]);
    let weight1 = TLExpr::constant(0.6);
    let weight2 = TLExpr::constant(0.4);
    let weighted1 = TLExpr::mul(score1, weight1);
    let weighted2 = TLExpr::mul(score2, weight2);
    let _weighted_avg = TLExpr::add(weighted1, weighted2);
    println!("   Weighted avg: (score1(x) * 0.6) + (score2(x) * 0.4)");

    println!("\n=== Example Complete ===");

    Ok(())
}
