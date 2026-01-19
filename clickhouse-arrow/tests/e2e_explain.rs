#![allow(unused_crate_dependencies)]

pub mod common;
pub mod tests;

const TRACING_DIRECTIVES: &[(&str, &str)] =
    &[("testcontainers", "debug"), ("clickhouse_arrow", "debug")];

// Test EXPLAIN AST with parallel query execution
#[cfg(feature = "test-utils")]
e2e_test!(
    e2e_explain_ast_parallel,
    tests::explain::test_explain_ast_parallel,
    TRACING_DIRECTIVES,
    None
);

// Test EXPLAIN SYNTAX
#[cfg(feature = "test-utils")]
e2e_test!(e2e_explain_syntax, tests::explain::test_explain_syntax, TRACING_DIRECTIVES, None);

// Test EXPLAIN PLAN
#[cfg(feature = "test-utils")]
e2e_test!(e2e_explain_plan, tests::explain::test_explain_plan, TRACING_DIRECTIVES, None);

// Test EXPLAIN PIPELINE
#[cfg(feature = "test-utils")]
e2e_test!(e2e_explain_pipeline, tests::explain::test_explain_pipeline, TRACING_DIRECTIVES, None);

// Test EXPLAIN ESTIMATE
#[cfg(feature = "test-utils")]
e2e_test!(e2e_explain_estimate, tests::explain::test_explain_estimate, TRACING_DIRECTIVES, None);

// Test query_with_options without explain
#[cfg(feature = "test-utils")]
e2e_test!(
    e2e_query_options_no_explain,
    tests::explain::test_query_options_no_explain,
    TRACING_DIRECTIVES,
    None
);
