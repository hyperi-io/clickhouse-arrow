#![allow(unused_crate_dependencies)]

pub mod common;
pub mod tests;

const TRACING_DIRECTIVES: &[(&str, &str)] = &[("testcontainers", "debug"), ("clickhouse_arrow", "debug")];

// BFloat16 tests
e2e_test!(e2e_bfloat16_basic, tests::new_types::test_bfloat16_basic, TRACING_DIRECTIVES, None);

// Variant tests
e2e_test!(e2e_variant_basic, tests::new_types::test_variant_basic, TRACING_DIRECTIVES, None);
e2e_test!(e2e_variant_complex, tests::new_types::test_variant_complex, TRACING_DIRECTIVES, None);

// Dynamic tests
e2e_test!(e2e_dynamic_basic, tests::new_types::test_dynamic_basic, TRACING_DIRECTIVES, None);
e2e_test!(e2e_dynamic_max_types, tests::new_types::test_dynamic_max_types, TRACING_DIRECTIVES, None);

// Nested tests
e2e_test!(e2e_nested_basic, tests::new_types::test_nested_basic, TRACING_DIRECTIVES, None);
e2e_test!(e2e_nested_flatten, tests::new_types::test_nested_flatten, TRACING_DIRECTIVES, None);

// Time types tests
e2e_test!(e2e_time_simulation, tests::new_types::test_time_simulation, TRACING_DIRECTIVES, None);

// Combined tests
e2e_test!(e2e_new_types_combined, tests::new_types::test_new_types_combined, TRACING_DIRECTIVES, None);
