//! Micro-benchmarks for array/list type serialization and deserialization.
//!
//! Arrays (nested types) are one of the most complex and potentially expensive
//! operations in the serialization path. These benchmarks isolate the performance
//! characteristics of nested data structures.
#![expect(unused_crate_dependencies)]
// Benchmark code: casts are safe for test data sizes
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]
#![allow(unused_results)]

mod common;

use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::*;
use arrow::datatypes::*;
use arrow::record_batch::RecordBatch;
use clickhouse_arrow::CompressionMethod;
use clickhouse_arrow::prelude::*;
use clickhouse_arrow::test_utils::{arrow_tests, get_or_create_container};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use futures_util::StreamExt;
use tokio::runtime::Runtime;

use self::common::{DISABLE_CLEANUP_ENV, TEST_DB_NAME, init, print_msg};

/// Row counts for array benchmarks
const ROW_COUNTS: &[usize] = &[10_000, 50_000, 100_000];

/// Create a batch with Array<Int64> columns
fn create_int64_array_batch(
    rows: usize,
    elements_per_array: usize,
    num_cols: usize,
) -> RecordBatch {
    let inner_field = Field::new_list_field(DataType::Int64, false);
    let fields: Vec<Field> = (0..num_cols)
        .map(|i| {
            Field::new(format!("arr_{i}"), DataType::List(Arc::new(inner_field.clone())), false)
        })
        .collect();
    let schema = Arc::new(Schema::new(fields));

    let columns: Vec<ArrayRef> = (0..num_cols)
        .map(|col| {
            let mut builder = ListBuilder::new(Int64Builder::new());
            for row in 0..rows {
                for elem in 0..elements_per_array {
                    builder.values().append_value((row * elements_per_array + elem + col) as i64);
                }
                builder.append(true);
            }
            Arc::new(builder.finish()) as ArrayRef
        })
        .collect();

    RecordBatch::try_new(schema, columns).unwrap()
}

/// Create a batch with Array<String> columns
fn create_string_array_batch(
    rows: usize,
    elements_per_array: usize,
    string_len: usize,
    num_cols: usize,
) -> RecordBatch {
    let inner_field = Field::new_list_field(DataType::Utf8, false);
    let fields: Vec<Field> = (0..num_cols)
        .map(|i| {
            Field::new(format!("arr_{i}"), DataType::List(Arc::new(inner_field.clone())), false)
        })
        .collect();
    let schema = Arc::new(Schema::new(fields));

    let pattern: String = "abcdefghijklmnopqrstuvwxyz".chars().cycle().take(string_len).collect();

    let columns: Vec<ArrayRef> = (0..num_cols)
        .map(|_col| {
            let mut builder = ListBuilder::new(StringBuilder::new());
            for _row in 0..rows {
                for _elem in 0..elements_per_array {
                    builder.values().append_value(&pattern);
                }
                builder.append(true);
            }
            Arc::new(builder.finish()) as ArrayRef
        })
        .collect();

    RecordBatch::try_new(schema, columns).unwrap()
}

/// Create a batch with nested Array<Array<Int64>> columns (2 levels deep)
fn create_nested_array_batch(
    rows: usize,
    outer_elements: usize,
    inner_elements: usize,
    num_cols: usize,
) -> RecordBatch {
    let inner_field = Field::new_list_field(DataType::Int64, false);
    let outer_field = Field::new_list_field(DataType::List(Arc::new(inner_field)), false);
    let fields: Vec<Field> = (0..num_cols)
        .map(|i| {
            Field::new(format!("nested_{i}"), DataType::List(Arc::new(outer_field.clone())), false)
        })
        .collect();
    let schema = Arc::new(Schema::new(fields));

    let columns: Vec<ArrayRef> = (0..num_cols)
        .map(|col| {
            let inner_builder = ListBuilder::new(Int64Builder::new());
            let mut outer_builder = ListBuilder::new(inner_builder);

            for row in 0..rows {
                for outer_elem in 0..outer_elements {
                    for inner_elem in 0..inner_elements {
                        let value = (row * outer_elements * inner_elements
                            + outer_elem * inner_elements
                            + inner_elem
                            + col) as i64;
                        outer_builder.values().values().append_value(value);
                    }
                    outer_builder.values().append(true);
                }
                outer_builder.append(true);
            }
            Arc::new(outer_builder.finish()) as ArrayRef
        })
        .collect();

    RecordBatch::try_new(schema, columns).unwrap()
}

/// Create a batch with variable-length arrays
fn create_variable_array_batch(
    rows: usize,
    min_elements: usize,
    max_elements: usize,
    num_cols: usize,
) -> RecordBatch {
    let inner_field = Field::new_list_field(DataType::Int64, false);
    let fields: Vec<Field> = (0..num_cols)
        .map(|i| {
            Field::new(format!("arr_{i}"), DataType::List(Arc::new(inner_field.clone())), false)
        })
        .collect();
    let schema = Arc::new(Schema::new(fields));

    let columns: Vec<ArrayRef> = (0..num_cols)
        .map(|col| {
            let mut builder = ListBuilder::new(Int64Builder::new());
            for row in 0..rows {
                // Deterministic variable length
                let num_elements =
                    min_elements + ((row * 31 + col * 17) % (max_elements - min_elements + 1));
                for elem in 0..num_elements {
                    builder.values().append_value((row * 100 + elem + col) as i64);
                }
                builder.append(true);
            }
            Arc::new(builder.finish()) as ArrayRef
        })
        .collect();

    RecordBatch::try_new(schema, columns).unwrap()
}

/// Create a batch with nullable arrays (some arrays are null)
#[allow(dead_code)] // Reserved for future nullable array benchmarks
fn create_nullable_array_batch(
    rows: usize,
    elements_per_array: usize,
    null_ratio: f64,
    num_cols: usize,
) -> RecordBatch {
    let inner_field = Field::new_list_field(DataType::Int64, false);
    let fields: Vec<Field> = (0..num_cols)
        .map(|i| {
            Field::new(format!("arr_{i}"), DataType::List(Arc::new(inner_field.clone())), true)
        })
        .collect();
    let schema = Arc::new(Schema::new(fields));

    let null_threshold = (null_ratio * 100.0) as usize;

    let columns: Vec<ArrayRef> = (0..num_cols)
        .map(|col| {
            let mut builder = ListBuilder::new(Int64Builder::new());
            for row in 0..rows {
                if (row % 100) < null_threshold {
                    builder.append(false); // null array
                } else {
                    for elem in 0..elements_per_array {
                        builder
                            .values()
                            .append_value((row * elements_per_array + elem + col) as i64);
                    }
                    builder.append(true);
                }
            }
            Arc::new(builder.finish()) as ArrayRef
        })
        .collect();

    RecordBatch::try_new(schema, columns).unwrap()
}

/// Benchmark array inserts with different array sizes
fn bench_array_insert(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    init();

    let ch = rt.block_on(get_or_create_container(None));
    print_msg("Container ready for array benchmarks");

    let client = rt
        .block_on(
            arrow_tests::setup_test_arrow_client(ch.get_native_url(), &ch.user, &ch.password)
                .with_ipv4_only(true)
                .with_compression(CompressionMethod::None)
                .build::<ArrowFormat>(),
        )
        .expect("client setup");

    rt.block_on(arrow_tests::setup_database(TEST_DB_NAME, &client)).expect("database setup");

    let mut group = c.benchmark_group("ArrayInsert");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(15));

    // Test different array sizes
    let array_sizes = [4, 16, 64, 256];

    for &elements in &array_sizes {
        for &rows in ROW_COUNTS {
            let num_cols = 2;
            let batch = create_int64_array_batch(rows, elements, num_cols);
            let total_elements = rows * elements * num_cols;
            let total_bytes = total_elements * 8; // Int64 = 8 bytes

            let table = rt
                .block_on(arrow_tests::setup_table(&client, TEST_DB_NAME, &batch.schema()))
                .expect("table setup");
            let query = format!("INSERT INTO {table} FORMAT NATIVE");

            group.throughput(Throughput::Bytes(total_bytes as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("int64_x{elements}"), rows),
                &(&query, &client, &batch),
                |b, (query, client, batch)| {
                    b.to_async(&rt).iter_batched(
                        || (*batch).clone(),
                        |batch| async {
                            let stream = client.insert(*query, batch, None).await.unwrap();
                            drop(black_box(stream));
                        },
                        criterion::BatchSize::SmallInput,
                    );
                },
            );
        }
    }

    group.finish();

    if std::env::var(DISABLE_CLEANUP_ENV).is_ok_and(|e| e.eq_ignore_ascii_case("true")) {
        return;
    }
    rt.block_on(ch.shutdown()).expect("shutdown");
}

/// Benchmark string array inserts
fn bench_string_array_insert(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    init();

    let ch = rt.block_on(get_or_create_container(None));

    let client = rt
        .block_on(
            arrow_tests::setup_test_arrow_client(ch.get_native_url(), &ch.user, &ch.password)
                .with_ipv4_only(true)
                .with_compression(CompressionMethod::None)
                .build::<ArrowFormat>(),
        )
        .expect("client setup");

    rt.block_on(arrow_tests::setup_database(TEST_DB_NAME, &client)).expect("database setup");

    let mut group = c.benchmark_group("StringArrayInsert");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(15));

    let string_len = 32;
    let array_sizes = [4, 16, 64];

    for &elements in &array_sizes {
        for &rows in &[10_000, 50_000] {
            let num_cols = 2;
            let batch = create_string_array_batch(rows, elements, string_len, num_cols);
            let total_bytes = rows * elements * string_len * num_cols;

            let table = rt
                .block_on(arrow_tests::setup_table(&client, TEST_DB_NAME, &batch.schema()))
                .expect("table setup");
            let query = format!("INSERT INTO {table} FORMAT NATIVE");

            group.throughput(Throughput::Bytes(total_bytes as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("str32_x{elements}"), rows),
                &(&query, &client, &batch),
                |b, (query, client, batch)| {
                    b.to_async(&rt).iter_batched(
                        || (*batch).clone(),
                        |batch| async {
                            let stream = client.insert(*query, batch, None).await.unwrap();
                            drop(black_box(stream));
                        },
                        criterion::BatchSize::SmallInput,
                    );
                },
            );
        }
    }

    group.finish();

    if std::env::var(DISABLE_CLEANUP_ENV).is_ok_and(|e| e.eq_ignore_ascii_case("true")) {
        return;
    }
    rt.block_on(ch.shutdown()).expect("shutdown");
}

/// Benchmark nested array inserts (Array<Array<Int64>>)
fn bench_nested_array_insert(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    init();

    let ch = rt.block_on(get_or_create_container(None));

    let client = rt
        .block_on(
            arrow_tests::setup_test_arrow_client(ch.get_native_url(), &ch.user, &ch.password)
                .with_ipv4_only(true)
                .with_compression(CompressionMethod::None)
                .build::<ArrowFormat>(),
        )
        .expect("client setup");

    rt.block_on(arrow_tests::setup_database(TEST_DB_NAME, &client)).expect("database setup");

    let mut group = c.benchmark_group("NestedArrayInsert");
    group.sample_size(15);
    group.measurement_time(Duration::from_secs(15));

    // Test different nesting configurations
    let configs = [
        (4, 4),  // 4 outer x 4 inner = 16 elements
        (4, 16), // 4 outer x 16 inner = 64 elements
        (8, 8),  // 8 outer x 8 inner = 64 elements
        (4, 64), // 4 outer x 64 inner = 256 elements
    ];

    for &(outer, inner) in &configs {
        for &rows in &[5_000, 20_000] {
            let num_cols = 1;
            let batch = create_nested_array_batch(rows, outer, inner, num_cols);
            let total_elements = rows * outer * inner * num_cols;
            let total_bytes = total_elements * 8;

            let table = rt
                .block_on(arrow_tests::setup_table(&client, TEST_DB_NAME, &batch.schema()))
                .expect("table setup");
            let query = format!("INSERT INTO {table} FORMAT NATIVE");

            group.throughput(Throughput::Bytes(total_bytes as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("nested_{outer}x{inner}"), rows),
                &(&query, &client, &batch),
                |b, (query, client, batch)| {
                    b.to_async(&rt).iter_batched(
                        || (*batch).clone(),
                        |batch| async {
                            let stream = client.insert(*query, batch, None).await.unwrap();
                            drop(black_box(stream));
                        },
                        criterion::BatchSize::SmallInput,
                    );
                },
            );
        }
    }

    group.finish();

    if std::env::var(DISABLE_CLEANUP_ENV).is_ok_and(|e| e.eq_ignore_ascii_case("true")) {
        return;
    }
    rt.block_on(ch.shutdown()).expect("shutdown");
}

/// Benchmark variable-length array inserts
fn bench_variable_array_insert(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    init();

    let ch = rt.block_on(get_or_create_container(None));

    let client = rt
        .block_on(
            arrow_tests::setup_test_arrow_client(ch.get_native_url(), &ch.user, &ch.password)
                .with_ipv4_only(true)
                .with_compression(CompressionMethod::None)
                .build::<ArrowFormat>(),
        )
        .expect("client setup");

    rt.block_on(arrow_tests::setup_database(TEST_DB_NAME, &client)).expect("database setup");

    let mut group = c.benchmark_group("VariableArrayInsert");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(15));

    let ranges = [(1, 16), (4, 64), (8, 128)];

    for &(min, max) in &ranges {
        for &rows in &[20_000, 100_000] {
            let num_cols = 2;
            let batch = create_variable_array_batch(rows, min, max, num_cols);

            // Estimate average bytes
            let avg_elements = usize::midpoint(min, max);
            let total_bytes = rows * avg_elements * 8 * num_cols;

            let table = rt
                .block_on(arrow_tests::setup_table(&client, TEST_DB_NAME, &batch.schema()))
                .expect("table setup");
            let query = format!("INSERT INTO {table} FORMAT NATIVE");

            group.throughput(Throughput::Bytes(total_bytes as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("range_{min}_{max}"), rows),
                &(&query, &client, &batch),
                |b, (query, client, batch)| {
                    b.to_async(&rt).iter_batched(
                        || (*batch).clone(),
                        |batch| async {
                            let stream = client.insert(*query, batch, None).await.unwrap();
                            drop(black_box(stream));
                        },
                        criterion::BatchSize::SmallInput,
                    );
                },
            );
        }
    }

    group.finish();

    if std::env::var(DISABLE_CLEANUP_ENV).is_ok_and(|e| e.eq_ignore_ascii_case("true")) {
        return;
    }
    rt.block_on(ch.shutdown()).expect("shutdown");
}

/// Benchmark array queries
fn bench_array_query(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    init();

    let ch = rt.block_on(get_or_create_container(None));

    let client = rt
        .block_on(
            arrow_tests::setup_test_arrow_client(ch.get_native_url(), &ch.user, &ch.password)
                .with_ipv4_only(true)
                .with_compression(CompressionMethod::None)
                .build::<ArrowFormat>(),
        )
        .expect("client setup");

    rt.block_on(arrow_tests::setup_database(TEST_DB_NAME, &client)).expect("database setup");

    let mut group = c.benchmark_group("ArrayQuery");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(15));

    let array_sizes = [16, 64];

    for &elements in &array_sizes {
        for &rows in &[20_000, 100_000] {
            let num_cols = 2;
            let batch = create_int64_array_batch(rows, elements, num_cols);
            let total_bytes = rows * elements * 8 * num_cols;

            let table = rt
                .block_on(arrow_tests::setup_table(&client, TEST_DB_NAME, &batch.schema()))
                .expect("table setup");

            // Insert data first
            rt.block_on(async {
                let insert_query = format!("INSERT INTO {table} FORMAT NATIVE");
                let mut stream = client.insert(&insert_query, batch, None).await.unwrap();
                while let Some(r) = stream.next().await {
                    r.unwrap();
                }
            });

            let query = format!("SELECT * FROM {table}");

            group.throughput(Throughput::Bytes(total_bytes as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("int64_x{elements}"), rows),
                &(&query, &client),
                |b, (query, client)| {
                    b.to_async(&rt).iter(|| async {
                        let mut stream = client.query(*query, None).await.unwrap();
                        while let Some(result) = stream.next().await {
                            black_box(result.unwrap());
                        }
                    });
                },
            );
        }
    }

    group.finish();

    if std::env::var(DISABLE_CLEANUP_ENV).is_ok_and(|e| e.eq_ignore_ascii_case("true")) {
        return;
    }
    rt.block_on(ch.shutdown()).expect("shutdown");
}

criterion_group!(
    benches,
    bench_array_insert,
    bench_string_array_insert,
    bench_nested_array_insert,
    bench_variable_array_insert,
    bench_array_query,
);
criterion_main!(benches);
