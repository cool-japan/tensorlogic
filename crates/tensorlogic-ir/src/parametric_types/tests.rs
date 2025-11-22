use super::*;

#[test]
fn test_kind_creation() {
    let star = Kind::Star;
    assert_eq!(star.arity(), 0);
    assert!(star.is_star());

    let arrow1 = Kind::constructor(1);
    assert_eq!(arrow1.arity(), 1);

    let arrow2 = Kind::constructor(2);
    assert_eq!(arrow2.arity(), 2);
}

#[test]
fn test_type_constructor_kind() {
    assert_eq!(TypeConstructor::List.kind().arity(), 1);
    assert_eq!(TypeConstructor::Option.kind().arity(), 1);
    assert_eq!(TypeConstructor::Tuple.kind().arity(), 2);
    assert_eq!(TypeConstructor::Function.kind().arity(), 2);
}

#[test]
fn test_concrete_type() {
    let int_type = ParametricType::concrete("Int");
    assert!(int_type.is_concrete());
    assert!(!int_type.is_variable());
    assert!(int_type.free_variables().is_empty());
    assert!(int_type.is_well_kinded());
    assert_eq!(int_type.to_string(), "Int");
}

#[test]
fn test_type_variable() {
    let t_var = ParametricType::variable("T");
    assert!(t_var.is_variable());
    assert!(!t_var.is_concrete());
    assert_eq!(t_var.free_variables(), vec!["T"]);
    assert!(t_var.is_well_kinded());
    assert_eq!(t_var.to_string(), "T");
}

#[test]
fn test_list_type() {
    let int_type = ParametricType::concrete("Int");
    let list_int = ParametricType::list(int_type.clone());

    assert!(!list_int.is_variable());
    assert!(!list_int.is_concrete());
    assert!(list_int.free_variables().is_empty());
    assert!(list_int.is_well_kinded());
    assert_eq!(list_int.to_string(), "List<Int>");
}

#[test]
fn test_option_type() {
    let string_type = ParametricType::concrete("String");
    let option_string = ParametricType::option(string_type);

    assert!(option_string.is_well_kinded());
    assert_eq!(option_string.to_string(), "Option<String>");
}

#[test]
fn test_tuple_type() {
    let int_type = ParametricType::concrete("Int");
    let string_type = ParametricType::concrete("String");
    let pair = ParametricType::tuple(vec![int_type, string_type]);

    assert!(pair.is_well_kinded());
    assert_eq!(pair.to_string(), "Tuple<Int, String>");
}

#[test]
fn test_nested_parametric_types() {
    let int_type = ParametricType::concrete("Int");
    let list_int = ParametricType::list(int_type.clone());
    let list_list_int = ParametricType::list(list_int);

    assert!(list_list_int.is_well_kinded());
    assert_eq!(list_list_int.to_string(), "List<List<Int>>");
}

#[test]
fn test_free_variables() {
    let t = ParametricType::variable("T");
    let u = ParametricType::variable("U");
    let int_type = ParametricType::concrete("Int");

    // List<T>
    let list_t = ParametricType::list(t.clone());
    assert_eq!(list_t.free_variables(), vec!["T"]);

    // Tuple<T, U>
    let tuple_tu = ParametricType::tuple(vec![t.clone(), u.clone()]);
    let mut free_vars = tuple_tu.free_variables();
    free_vars.sort();
    assert_eq!(free_vars, vec!["T", "U"]);

    // Tuple<T, Int>
    let tuple_t_int = ParametricType::tuple(vec![t.clone(), int_type]);
    assert_eq!(tuple_t_int.free_variables(), vec!["T"]);
}

#[test]
fn test_substitution() {
    let t = ParametricType::variable("T");
    let int_type = ParametricType::concrete("Int");
    let list_t = ParametricType::list(t.clone());

    let mut subst = HashMap::new();
    subst.insert("T".to_string(), int_type.clone());

    let result = list_t.substitute(&subst);
    assert_eq!(result, ParametricType::list(int_type));
}

#[test]
fn test_unify_concrete_types() {
    let int1 = ParametricType::concrete("Int");
    let int2 = ParametricType::concrete("Int");
    let string = ParametricType::concrete("String");

    // Int = Int
    let subst = unify(&int1, &int2).unwrap();
    assert!(subst.is_empty());

    // Int ≠ String
    assert!(unify(&int1, &string).is_err());
}

#[test]
fn test_unify_variable_with_concrete() {
    let t = ParametricType::variable("T");
    let int_type = ParametricType::concrete("Int");

    let subst = unify(&t, &int_type).unwrap();
    assert_eq!(subst.get("T").unwrap(), &int_type);
}

#[test]
fn test_unify_parametric_types() {
    let t = ParametricType::variable("T");
    let int_type = ParametricType::concrete("Int");
    let list_t = ParametricType::list(t.clone());
    let list_int = ParametricType::list(int_type.clone());

    let subst = unify(&list_t, &list_int).unwrap();
    assert_eq!(subst.get("T").unwrap(), &int_type);
}

#[test]
fn test_unify_multiple_variables() {
    let t = ParametricType::variable("T");
    let u = ParametricType::variable("U");
    let int_type = ParametricType::concrete("Int");

    // Tuple<T, U> with Tuple<Int, Int>
    let tuple_tu = ParametricType::tuple(vec![t.clone(), u.clone()]);
    let tuple_int_int = ParametricType::tuple(vec![int_type.clone(), int_type.clone()]);

    let subst = unify(&tuple_tu, &tuple_int_int).unwrap();
    assert_eq!(subst.get("T").unwrap(), &int_type);
    assert_eq!(subst.get("U").unwrap(), &int_type);
}

#[test]
fn test_occurs_check() {
    let t = ParametricType::variable("T");
    let list_t = ParametricType::list(t.clone());

    // T = List<T> should fail (occurs check)
    assert!(unify(&t, &list_t).is_err());
}

#[test]
fn test_unify_constructor_mismatch() {
    let int_type = ParametricType::concrete("Int");
    let list_int = ParametricType::list(int_type.clone());
    let option_int = ParametricType::option(int_type);

    // List<Int> ≠ Option<Int>
    assert!(unify(&list_int, &option_int).is_err());
}

#[test]
fn test_compose_substitutions() {
    let u = ParametricType::variable("U");
    let int_type = ParametricType::concrete("Int");

    let mut subst1 = HashMap::new();
    subst1.insert("T".to_string(), u.clone());

    let mut subst2 = HashMap::new();
    subst2.insert("U".to_string(), int_type.clone());

    let composed = compose_substitutions(&subst1, &subst2);
    assert_eq!(composed.get("T").unwrap(), &int_type);
    assert_eq!(composed.get("U").unwrap(), &int_type);
}

#[test]
fn test_generalize() {
    let t = ParametricType::variable("T");
    let u = ParametricType::variable("U");
    let tuple_tu = ParametricType::tuple(vec![t.clone(), u.clone()]);

    // Generalize with empty environment
    let gen = generalize(&tuple_tu, &[]);
    let free_vars = gen.free_variables();
    assert_eq!(free_vars.len(), 2);
    assert!(free_vars.iter().all(|v| v.starts_with('α')));
}

#[test]
fn test_instantiate() {
    let t = ParametricType::variable("T");
    let list_t = ParametricType::list(t);

    let inst = instantiate(&list_t);
    let free_vars = inst.free_variables();
    assert_eq!(free_vars.len(), 1);
    assert!(free_vars[0].starts_with("'t"));
}

#[test]
fn test_function_type() {
    let int_type = ParametricType::concrete("Int");
    let string_type = ParametricType::concrete("String");
    let func = ParametricType::function(int_type, string_type);

    assert!(func.is_well_kinded());
    assert_eq!(func.to_string(), "-><Int, String>");
}

#[test]
fn test_map_type() {
    let string_type = ParametricType::concrete("String");
    let int_type = ParametricType::concrete("Int");
    let map = ParametricType::map(string_type, int_type);

    assert!(map.is_well_kinded());
    assert_eq!(map.to_string(), "Map<String, Int>");
}

#[test]
fn test_set_type() {
    let int_type = ParametricType::concrete("Int");
    let set = ParametricType::set(int_type);

    assert!(set.is_well_kinded());
    assert_eq!(set.to_string(), "Set<Int>");
}

#[test]
fn test_array_type() {
    let float_type = ParametricType::concrete("Float");
    let array2d = ParametricType::array(float_type, 2);

    assert!(array2d.is_well_kinded());
    assert_eq!(array2d.to_string(), "Array2<Float>");
}

#[test]
fn test_custom_type_constructor() {
    let int_type = ParametricType::concrete("Int");
    let custom = TypeConstructor::custom("MyType", 1);
    let my_int = ParametricType::apply(custom, vec![int_type]);

    assert!(my_int.is_well_kinded());
    assert_eq!(my_int.to_string(), "MyType<Int>");
}

#[test]
fn test_ill_kinded_type() {
    let int_type = ParametricType::concrete("Int");
    // List expects 1 argument, giving it 2
    let ill_kinded =
        ParametricType::apply(TypeConstructor::List, vec![int_type.clone(), int_type]);

    assert!(!ill_kinded.is_well_kinded());
}

#[test]
fn test_unit_type() {
    let unit = ParametricType::unit();
    assert!(unit.is_well_kinded());
    // Empty tuple has no arguments so no angle brackets
    assert_eq!(unit.to_string(), "Tuple0");
}

#[test]
fn test_triple_type() {
    let int_type = ParametricType::concrete("Int");
    let string_type = ParametricType::concrete("String");
    let bool_type = ParametricType::concrete("Bool");
    let triple = ParametricType::triple(int_type, string_type, bool_type);

    assert!(triple.is_well_kinded());
    assert_eq!(triple.to_string(), "Tuple3<Int, String, Bool>");
}

#[test]
fn test_n_ary_tuples() {
    let int_type = ParametricType::concrete("Int");

    // 4-tuple
    let tuple4 = ParametricType::tuple(vec![
        int_type.clone(),
        int_type.clone(),
        int_type.clone(),
        int_type.clone(),
    ]);
    assert!(tuple4.is_well_kinded());
    assert_eq!(tuple4.to_string(), "Tuple4<Int, Int, Int, Int>");

    // 5-tuple
    let tuple5 = ParametricType::tuple(vec![
        int_type.clone(),
        int_type.clone(),
        int_type.clone(),
        int_type.clone(),
        int_type.clone(),
    ]);
    assert!(tuple5.is_well_kinded());
    assert_eq!(tuple5.to_string(), "Tuple5<Int, Int, Int, Int, Int>");

    // 1-tuple (singleton)
    let tuple1 = ParametricType::tuple(vec![int_type.clone()]);
    assert!(tuple1.is_well_kinded());
    assert_eq!(tuple1.to_string(), "Tuple1<Int>");
}

#[test]
fn test_result_type() {
    let int_type = ParametricType::concrete("Int");
    let string_type = ParametricType::concrete("String");
    let result = ParametricType::result(int_type, string_type);

    assert!(result.is_well_kinded());
    assert_eq!(result.to_string(), "Result<Int, String>");
}

#[test]
fn test_either_type() {
    let int_type = ParametricType::concrete("Int");
    let string_type = ParametricType::concrete("String");
    let either = ParametricType::either(int_type, string_type);

    assert!(either.is_well_kinded());
    assert_eq!(either.to_string(), "Either<Int, String>");
}

#[test]
fn test_unify_n_ary_tuples() {
    let t = ParametricType::variable("T");
    let u = ParametricType::variable("U");
    let v = ParametricType::variable("V");
    let int_type = ParametricType::concrete("Int");
    let string_type = ParametricType::concrete("String");
    let bool_type = ParametricType::concrete("Bool");

    // Tuple3<T, U, V> with Tuple3<Int, String, Bool>
    let tuple_tuv = ParametricType::triple(t.clone(), u.clone(), v.clone());
    let tuple_isb =
        ParametricType::triple(int_type.clone(), string_type.clone(), bool_type.clone());

    let subst = unify(&tuple_tuv, &tuple_isb).unwrap();
    assert_eq!(subst.get("T").unwrap(), &int_type);
    assert_eq!(subst.get("U").unwrap(), &string_type);
    assert_eq!(subst.get("V").unwrap(), &bool_type);
}

#[test]
fn test_unify_result_types() {
    let t = ParametricType::variable("T");
    let e = ParametricType::variable("E");
    let int_type = ParametricType::concrete("Int");
    let string_type = ParametricType::concrete("String");

    let result_te = ParametricType::result(t.clone(), e.clone());
    let result_is = ParametricType::result(int_type.clone(), string_type.clone());

    let subst = unify(&result_te, &result_is).unwrap();
    assert_eq!(subst.get("T").unwrap(), &int_type);
    assert_eq!(subst.get("E").unwrap(), &string_type);
}

#[test]
fn test_nested_n_ary_tuples() {
    let int_type = ParametricType::concrete("Int");
    let string_type = ParametricType::concrete("String");

    // Tuple3<Int, Tuple<String, Int>, Int>
    let inner_tuple = ParametricType::tuple(vec![string_type.clone(), int_type.clone()]);
    let outer_tuple = ParametricType::triple(int_type.clone(), inner_tuple, int_type.clone());

    assert!(outer_tuple.is_well_kinded());
    assert_eq!(
        outer_tuple.to_string(),
        "Tuple3<Int, Tuple<String, Int>, Int>"
    );
}

#[test]
fn test_tuple_n_kind() {
    assert_eq!(TypeConstructor::TupleN { arity: 0 }.kind().arity(), 0);
    assert_eq!(TypeConstructor::TupleN { arity: 1 }.kind().arity(), 1);
    assert_eq!(TypeConstructor::TupleN { arity: 3 }.kind().arity(), 3);
    assert_eq!(TypeConstructor::TupleN { arity: 10 }.kind().arity(), 10);
}

#[test]
fn test_result_either_kind() {
    assert_eq!(TypeConstructor::Result.kind().arity(), 2);
    assert_eq!(TypeConstructor::Either.kind().arity(), 2);
}

#[test]
fn test_free_variables_in_n_ary_tuples() {
    let t = ParametricType::variable("T");
    let u = ParametricType::variable("U");
    let v = ParametricType::variable("V");
    let w = ParametricType::variable("W");
    let int_type = ParametricType::concrete("Int");

    let tuple4 = ParametricType::tuple(vec![t.clone(), u.clone(), int_type.clone(), v.clone()]);
    let mut free_vars = tuple4.free_variables();
    free_vars.sort();
    assert_eq!(free_vars, vec!["T", "U", "V"]);

    // With Result type
    let result = ParametricType::result(t.clone(), w.clone());
    let mut result_vars = result.free_variables();
    result_vars.sort();
    assert_eq!(result_vars, vec!["T", "W"]);
}

// =========================================================================
// Type Constraint Tests
// =========================================================================

#[test]
fn test_type_constraint_creation() {
    let c1 = TypeConstraint::numeric("T");
    assert_eq!(c1.type_var(), Some("T"));

    let c2 = TypeConstraint::implements("T", "Clone");
    assert_eq!(c2.to_string(), "T: Clone");

    let c3 = TypeConstraint::ord("X");
    assert_eq!(c3.to_string(), "X: Ord");
}

#[test]
fn test_type_constraint_substitution() {
    let c = TypeConstraint::numeric("T");
    let mut subst = HashMap::new();
    subst.insert("T".to_string(), ParametricType::variable("U"));

    let c2 = c.substitute(&subst);
    assert_eq!(c2.type_var(), Some("U"));
}

#[test]
fn test_constrained_type_creation() {
    let _int_type = ParametricType::concrete("Int");
    let t = ParametricType::variable("T");
    let list_t = ParametricType::list(t.clone());

    // forall T. T: Ord => List<T>
    let ct = ConstrainedType::new(
        vec!["T".to_string()],
        vec![TypeConstraint::ord("T")],
        list_t,
    );

    assert!(ct.has_constraints());
    assert_eq!(ct.type_vars.len(), 1);
    assert_eq!(ct.to_string(), "∀T. (T: Ord) => List<T>");
}

#[test]
fn test_constrained_type_instantiate() {
    let t = ParametricType::variable("T");
    let list_t = ParametricType::list(t.clone());

    let ct = ConstrainedType::new(
        vec!["T".to_string()],
        vec![TypeConstraint::numeric("T")],
        list_t,
    );

    let mut counter = 0;
    let (new_body, new_constraints) = ct.instantiate_fresh(&mut counter);

    // Should have fresh variable
    let free_vars = new_body.free_variables();
    assert_eq!(free_vars.len(), 1);
    assert!(free_vars[0].starts_with("'t"));

    // Constraints should also be updated
    assert_eq!(new_constraints.len(), 1);
}

#[test]
fn test_simple_constrained_type() {
    let int_type = ParametricType::concrete("Int");
    let ct = ConstrainedType::simple(int_type.clone());

    assert!(!ct.has_constraints());
    assert!(ct.type_vars.is_empty());
    assert_eq!(ct.to_string(), "Int");
}

// =========================================================================
// Recursive Type Tests
// =========================================================================

#[test]
fn test_recursive_type_creation() {
    // Natural numbers: μN. 1 + N (simplified as Option<N>)
    let n = ParametricType::variable("N");
    let option_n = ParametricType::option(n.clone());
    let nat = ParametricType::recursive("N", option_n);

    assert!(nat.is_recursive());
    assert_eq!(nat.to_string(), "μN. Option<N>");
}

#[test]
fn test_recursive_type_unfold() {
    let n = ParametricType::variable("N");
    let option_n = ParametricType::option(n.clone());
    let nat = ParametricType::recursive("N", option_n);

    let unfolded = nat.unfold().unwrap();
    // Unfolding substitutes N with the whole recursive type
    assert_eq!(unfolded.to_string(), "Option<μN. Option<N>>");
}

#[test]
fn test_recursive_type_well_formed() {
    // Well-formed: variable occurs in body
    let n = ParametricType::variable("N");
    let option_n = ParametricType::option(n.clone());
    let nat = ParametricType::recursive("N", option_n);
    assert!(nat.is_well_kinded());

    // Not well-formed: variable doesn't occur in body
    let int_type = ParametricType::concrete("Int");
    let bad = ParametricType::recursive("X", int_type);
    assert!(!bad.is_well_kinded());
}

#[test]
fn test_recursive_type_free_variables() {
    let t = ParametricType::variable("T");
    let n = ParametricType::variable("N");

    // μN. Tuple<T, N> - T is free, N is bound
    let tuple = ParametricType::tuple(vec![t.clone(), n.clone()]);
    let rec = ParametricType::recursive("N", tuple);

    let free_vars = rec.free_variables();
    assert_eq!(free_vars, vec!["T"]);
}

#[test]
fn test_recursive_type_substitute() {
    let t = ParametricType::variable("T");
    let n = ParametricType::variable("N");
    let int_type = ParametricType::concrete("Int");

    // μN. Tuple<T, N>
    let tuple = ParametricType::tuple(vec![t.clone(), n.clone()]);
    let rec = ParametricType::recursive("N", tuple);

    // Substitute T -> Int (N should remain)
    let mut subst = HashMap::new();
    subst.insert("T".to_string(), int_type.clone());

    let result = rec.substitute(&subst);
    assert_eq!(result.to_string(), "μN. Tuple<Int, N>");
}

// =========================================================================
// Higher-Kinded Type Tests
// =========================================================================

#[test]
fn test_kinded_var_creation() {
    let type_var = KindedVar::type_var("T");
    assert!(type_var.kind.is_star());

    let constructor_var = KindedVar::constructor_var("F");
    assert_eq!(constructor_var.kind.arity(), 1);

    let higher = KindedVar::higher_kinded("M", 2);
    assert_eq!(higher.kind.arity(), 2);
}

#[test]
fn test_kinded_var_display() {
    let v = KindedVar::type_var("T");
    assert_eq!(v.to_string(), "(T :: *)");

    let c = KindedVar::constructor_var("F");
    assert_eq!(c.to_string(), "(F :: * -> *)");
}

#[test]
fn test_higher_kinded_type_creation() {
    let f_var = KindedVar::constructor_var("F");
    let a_var = KindedVar::type_var("A");
    let b_var = KindedVar::type_var("B");

    // forall F: * -> *, A, B. F A -> F B
    let hkt = HigherKindedType::unconstrained(
        vec![f_var, a_var, b_var],
        ParametricType::function(
            ParametricType::variable("FA"),
            ParametricType::variable("FB"),
        ),
    );

    assert!(hkt.has_higher_kinded_vars());
    assert_eq!(hkt.constructor_vars(), vec!["F"]);
    assert_eq!(hkt.type_vars(), vec!["A", "B"]);
}

#[test]
fn test_higher_kinded_type_display() {
    let f_var = KindedVar::constructor_var("F");
    let t_var = KindedVar::type_var("T");

    let hkt = HigherKindedType::new(
        vec![f_var, t_var],
        vec![TypeConstraint::implements("F", "Functor")],
        ParametricType::variable("Result"),
    );

    let s = hkt.to_string();
    assert!(s.contains("∀"));
    assert!(s.contains("Functor"));
}

// =========================================================================
// Row Polymorphism Tests
// =========================================================================

#[test]
fn test_row_type_closed() {
    let int_type = ParametricType::concrete("Int");
    let string_type = ParametricType::concrete("String");

    let row = RowType::closed(vec![
        ("name".to_string(), string_type),
        ("age".to_string(), int_type),
    ]);

    assert!(row.is_closed());
    assert!(row.has_field("name"));
    assert!(row.has_field("age"));
    assert!(!row.has_field("email"));
    assert_eq!(row.to_string(), "{name: String, age: Int}");
}

#[test]
fn test_row_type_open() {
    let int_type = ParametricType::concrete("Int");

    let row = RowType::open(vec![("x".to_string(), int_type.clone())], "r");

    assert!(!row.is_closed());
    assert!(row.has_field("x"));
    assert_eq!(row.to_string(), "{x: Int | r}");
}

#[test]
fn test_row_type_extend() {
    let int_type = ParametricType::concrete("Int");
    let string_type = ParametricType::concrete("String");

    let row = RowType::closed(vec![("x".to_string(), int_type)]);
    let extended = row.extend(vec![("y".to_string(), string_type)]);

    assert!(extended.has_field("x"));
    assert!(extended.has_field("y"));
    assert_eq!(extended.fields.len(), 2);
}

#[test]
fn test_row_type_free_variables() {
    let t = ParametricType::variable("T");
    let u = ParametricType::variable("U");

    let row = RowType::open(vec![("a".to_string(), t), ("b".to_string(), u)], "r");

    let mut free_vars = row.free_variables();
    free_vars.sort();
    assert_eq!(free_vars, vec!["T", "U", "r"]);
}

#[test]
fn test_record_type_creation() {
    let int_type = ParametricType::concrete("Int");
    let string_type = ParametricType::concrete("String");

    let record = ParametricType::record(vec![
        ("name".to_string(), string_type),
        ("age".to_string(), int_type),
    ]);

    assert!(record.is_record());
    assert!(record.is_well_kinded());

    let row = record.as_row().unwrap();
    assert!(row.is_closed());
    assert_eq!(row.fields.len(), 2);
}

#[test]
fn test_extensible_record_type() {
    let int_type = ParametricType::concrete("Int");

    let record = ParametricType::extensible_record(vec![("x".to_string(), int_type)], "rest");

    let row = record.as_row().unwrap();
    assert!(!row.is_closed());
    assert_eq!(row.rest, Some("rest".to_string()));
}

#[test]
fn test_record_type_substitute() {
    let t = ParametricType::variable("T");
    let int_type = ParametricType::concrete("Int");

    let record = ParametricType::record(vec![("value".to_string(), t.clone())]);

    let mut subst = HashMap::new();
    subst.insert("T".to_string(), int_type.clone());

    let result = record.substitute(&subst);
    let row = result.as_row().unwrap();
    assert_eq!(row.get_field("value").unwrap(), &int_type);
}

#[test]
fn test_row_type_get_field() {
    let int_type = ParametricType::concrete("Int");
    let string_type = ParametricType::concrete("String");

    let row = RowType::closed(vec![
        ("name".to_string(), string_type.clone()),
        ("age".to_string(), int_type.clone()),
    ]);

    assert_eq!(row.get_field("name"), Some(&string_type));
    assert_eq!(row.get_field("age"), Some(&int_type));
    assert_eq!(row.get_field("unknown"), None);
}

#[test]
fn test_empty_row() {
    let row = RowType::empty();
    assert!(row.is_closed());
    assert!(row.fields.is_empty());
    assert_eq!(row.to_string(), "{}");
}

#[test]
fn test_row_field_names() {
    let int_type = ParametricType::concrete("Int");

    let row = RowType::closed(vec![
        ("a".to_string(), int_type.clone()),
        ("b".to_string(), int_type.clone()),
        ("c".to_string(), int_type),
    ]);

    let names = row.field_names();
    assert_eq!(names, vec!["a", "b", "c"]);
}
