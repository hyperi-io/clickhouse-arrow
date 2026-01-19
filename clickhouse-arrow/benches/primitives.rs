//! Micro-benchmarks for primitive type serialization and deserialization.
//!
//! These benchmarks isolate the critical path performance of individual type
//! operations to enable targeted optimization.
#![expect(unused_crate_dependencies)]
// Benchmark code: casts are safe for test data sizes
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]
#![allow(unused_results)]
#![allow(clippy::explicit_iter_loop)]
// Benchmark functions are necessarily long to set up test infrastructure
#![allow(clippy::too_many_lines)]
// Helper functions reserved for future benchmark variants
#![allow(dead_code)]

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

/// Row counts for benchmarking - focus on larger sizes for throughput measurement
const ROW_COUNTS: &[usize] = &[100_000, 500_000, 1_000_000];

/// Create a batch with only Int8 columns for isolated measurement
fn create_int8_batch(rows: usize, num_cols: usize) -> RecordBatch {
    let fields: Vec<Field> =
        (0..num_cols).map(|i| Field::new(format!("col_{i}"), DataType::Int8, false)).collect();
    let schema = Arc::new(Schema::new(fields));

    let columns: Vec<ArrayRef> = (0..num_cols)
        .map(|col| {
            let array: Int8Array = (0..rows).map(|i| ((i + col) % 128) as i8).collect();
            Arc::new(array) as ArrayRef
        })
        .collect();

    RecordBatch::try_new(schema, columns).unwrap()
}

/// Create a batch with only Int32 columns
fn create_int32_batch(rows: usize, num_cols: usize) -> RecordBatch {
    let fields: Vec<Field> =
        (0..num_cols).map(|i| Field::new(format!("col_{i}"), DataType::Int32, false)).collect();
    let schema = Arc::new(Schema::new(fields));

    let columns: Vec<ArrayRef> = (0..num_cols)
        .map(|col| {
            let array: Int32Array = (0..rows).map(|i| (i + col) as i32).collect();
            Arc::new(array) as ArrayRef
        })
        .collect();

    RecordBatch::try_new(schema, columns).unwrap()
}

/// Create a batch with only Int64 columns
fn create_int64_batch(rows: usize, num_cols: usize) -> RecordBatch {
    let fields: Vec<Field> =
        (0..num_cols).map(|i| Field::new(format!("col_{i}"), DataType::Int64, false)).collect();
    let schema = Arc::new(Schema::new(fields));

    let columns: Vec<ArrayRef> = (0..num_cols)
        .map(|col| {
            let array: Int64Array = (0..rows).map(|i| (i + col) as i64).collect();
            Arc::new(array) as ArrayRef
        })
        .collect();

    RecordBatch::try_new(schema, columns).unwrap()
}

/// Create a batch with only Float32 columns
fn create_float32_batch(rows: usize, num_cols: usize) -> RecordBatch {
    let fields: Vec<Field> =
        (0..num_cols).map(|i| Field::new(format!("col_{i}"), DataType::Float32, false)).collect();
    let schema = Arc::new(Schema::new(fields));

    let columns: Vec<ArrayRef> = (0..num_cols)
        .map(|col| {
            let array: Float32Array = (0..rows).map(|i| ((i + col) as f32) * 1.5).collect();
            Arc::new(array) as ArrayRef
        })
        .collect();

    RecordBatch::try_new(schema, columns).unwrap()
}

/// Create a batch with only Float64 columns
fn create_float64_batch(rows: usize, num_cols: usize) -> RecordBatch {
    let fields: Vec<Field> =
        (0..num_cols).map(|i| Field::new(format!("col_{i}"), DataType::Float64, false)).collect();
    let schema = Arc::new(Schema::new(fields));

    let columns: Vec<ArrayRef> = (0..num_cols)
        .map(|col| {
            let array: Float64Array = (0..rows).map(|i| ((i + col) as f64) * 1.5).collect();
            Arc::new(array) as ArrayRef
        })
        .collect();

    RecordBatch::try_new(schema, columns).unwrap()
}

/// Create a batch with nullable Int64 columns (tests null bitmap handling)
fn create_nullable_int64_batch(rows: usize, num_cols: usize, null_ratio: f64) -> RecordBatch {
    let fields: Vec<Field> =
        (0..num_cols).map(|i| Field::new(format!("col_{i}"), DataType::Int64, true)).collect();
    let schema = Arc::new(Schema::new(fields));

    let null_threshold = (null_ratio * 100.0) as usize;
    let columns: Vec<ArrayRef> = (0..num_cols)
        .map(|col| {
            let array: Int64Array = (0..rows)
                .map(|i| if (i % 100) < null_threshold { None } else { Some((i + col) as i64) })
                .collect();
            Arc::new(array) as ArrayRef
        })
        .collect();

    RecordBatch::try_new(schema, columns).unwrap()
}

/// Create a batch with timestamp columns
fn create_timestamp_batch(rows: usize, num_cols: usize) -> RecordBatch {
    let fields: Vec<Field> = (0..num_cols)
        .map(|i| {
            Field::new(
                format!("ts_{i}"),
                DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())),
                false,
            )
        })
        .collect();
    let schema = Arc::new(Schema::new(fields));

    let columns: Vec<ArrayRef> = (0..num_cols)
        .map(|col| {
            let array = TimestampMillisecondArray::from(
                (0..rows).map(|i| ((i + col) as i64) * 1000).collect::<Vec<_>>(),
            )
            .with_timezone(Arc::from("UTC"));
            Arc::new(array) as ArrayRef
        })
        .collect();

    RecordBatch::try_new(schema, columns).unwrap()
}

/// Benchmark primitive type inserts
fn bench_primitive_insert(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    init();

    let ch = rt.block_on(get_or_create_container(None));
    print_msg("Container ready for primitive benchmarks");

    let client = rt
        .block_on(
            arrow_tests::setup_test_arrow_client(ch.get_native_url(), &ch.user, &ch.password)
                .with_ipv4_only(true)
                .with_compression(CompressionMethod::None)
                .build::<ArrowFormat>(),
        )
        .expect("client setup");

    rt.block_on(arrow_tests::setup_database(TEST_DB_NAME, &client)).expect("database setup");

    let mut group = c.benchmark_group("PrimitiveInsert");
    group.sample_size(30);
    group.measurement_time(Duration::from_secs(20));

    // Benchmark each primitive type
    for &rows in ROW_COUNTS {
        let bytes_per_row = 4; // Int32 = 4 bytes per column
        group.throughput(Throughput::Bytes((rows * bytes_per_row * 4) as u64));

        // Int32 benchmark (4 columns)
        let batch = create_int32_batch(rows, 4);
        let table = rt
            .block_on(arrow_tests::setup_table(&client, TEST_DB_NAME, &batch.schema()))
            .expect("table setup");
        let query = format!("INSERT INTO {table} FORMAT NATIVE");

        group.bench_with_input(
            BenchmarkId::new("int32_4col", rows),
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

        // Int64 benchmark (4 columns)
        let batch = create_int64_batch(rows, 4);
        let table = rt
            .block_on(arrow_tests::setup_table(&client, TEST_DB_NAME, &batch.schema()))
            .expect("table setup");
        let query = format!("INSERT INTO {table} FORMAT NATIVE");

        group.throughput(Throughput::Bytes((rows * 8 * 4) as u64));
        group.bench_with_input(
            BenchmarkId::new("int64_4col", rows),
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

        // Float64 benchmark (4 columns)
        let batch = create_float64_batch(rows, 4);
        let table = rt
            .block_on(arrow_tests::setup_table(&client, TEST_DB_NAME, &batch.schema()))
            .expect("table setup");
        let query = format!("INSERT INTO {table} FORMAT NATIVE");

        group.bench_with_input(
            BenchmarkId::new("float64_4col", rows),
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

        // Nullable Int64 benchmark (10% nulls)
        let batch = create_nullable_int64_batch(rows, 4, 0.1);
        let table = rt
            .block_on(arrow_tests::setup_table(&client, TEST_DB_NAME, &batch.schema()))
            .expect("table setup");
        let query = format!("INSERT INTO {table} FORMAT NATIVE");

        group.bench_with_input(
            BenchmarkId::new("nullable_int64_10pct_4col", rows),
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

        // Timestamp benchmark (4 columns)
        let batch = create_timestamp_batch(rows, 4);
        let table = rt
            .block_on(arrow_tests::setup_table(&client, TEST_DB_NAME, &batch.schema()))
            .expect("table setup");
        let query = format!("INSERT INTO {table} FORMAT NATIVE");

        group.bench_with_input(
            BenchmarkId::new("timestamp_4col", rows),
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

    group.finish();

    if std::env::var(DISABLE_CLEANUP_ENV).is_ok_and(|e| e.eq_ignore_ascii_case("true")) {
        return;
    }
    rt.block_on(ch.shutdown()).expect("shutdown");
}

/// Benchmark primitive type queries
fn bench_primitive_query(c: &mut Criterion) {
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

    let mut group = c.benchmark_group("PrimitiveQuery");
    group.sample_size(30);
    group.measurement_time(Duration::from_secs(20));

    // Pre-populate tables for query benchmarks
    for &rows in ROW_COUNTS {
        // Int64 query benchmark
        let batch = create_int64_batch(rows, 4);
        let table = rt
            .block_on(arrow_tests::setup_table(&client, TEST_DB_NAME, &batch.schema()))
            .expect("table setup");

        // Insert data
        rt.block_on(async {
            let query = format!("INSERT INTO {table} FORMAT NATIVE");
            let mut stream = client.insert(&query, batch.clone(), None).await.unwrap();
            while let Some(r) = stream.next().await {
                r.unwrap();
            }
        });

        group.throughput(Throughput::Bytes((rows * 8 * 4) as u64));
        let query = format!("SELECT * FROM {table}");

        group.bench_with_input(
            BenchmarkId::new("int64_4col", rows),
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

        // Float64 query benchmark
        let batch = create_float64_batch(rows, 4);
        let table = rt
            .block_on(arrow_tests::setup_table(&client, TEST_DB_NAME, &batch.schema()))
            .expect("table setup");

        rt.block_on(async {
            let query = format!("INSERT INTO {table} FORMAT NATIVE");
            let mut stream = client.insert(&query, batch.clone(), None).await.unwrap();
            while let Some(r) = stream.next().await {
                r.unwrap();
            }
        });

        let query = format!("SELECT * FROM {table}");

        group.bench_with_input(
            BenchmarkId::new("float64_4col", rows),
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

    group.finish();

    if std::env::var(DISABLE_CLEANUP_ENV).is_ok_and(|e| e.eq_ignore_ascii_case("true")) {
        return;
    }
    rt.block_on(ch.shutdown()).expect("shutdown");
}

/// Benchmark column count scaling (same total data, different column arrangements)
fn bench_column_scaling(c: &mut Criterion) {
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

    let mut group = c.benchmark_group("ColumnScaling");
    group.sample_size(30);
    group.measurement_time(Duration::from_secs(15));

    // Test with fixed total bytes but varying column counts
    let total_int64s = 4_000_000; // 32 MB total
    let column_counts = [1, 4, 16, 64, 256];

    for &num_cols in &column_counts {
        let rows = total_int64s / num_cols;
        let batch = create_int64_batch(rows, num_cols);
        let table = rt
            .block_on(arrow_tests::setup_table(&client, TEST_DB_NAME, &batch.schema()))
            .expect("table setup");
        let query = format!("INSERT INTO {table} FORMAT NATIVE");

        group.throughput(Throughput::Bytes((rows * 8 * num_cols) as u64));
        group.bench_with_input(
            BenchmarkId::new("int64", format!("{num_cols}col_x_{rows}rows")),
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

    group.finish();

    if std::env::var(DISABLE_CLEANUP_ENV).is_ok_and(|e| e.eq_ignore_ascii_case("true")) {
        return;
    }
    rt.block_on(ch.shutdown()).expect("shutdown");
}

criterion_group!(benches, bench_primitive_insert, bench_primitive_query, bench_column_scaling,);
criterion_main!(benches);
