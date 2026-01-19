//! Micro-benchmarks for string and binary type serialization/deserialization.
//!
//! Tests performance across different string length distributions which is
//! critical for real-world workloads.
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

/// Row counts for string benchmarks
const ROW_COUNTS: &[usize] = &[50_000, 200_000, 500_000];

/// String length profiles to test
const STRING_LENGTHS: &[usize] = &[8, 32, 128, 512, 2048];

/// Create a batch with fixed-length strings
fn create_string_batch(rows: usize, string_len: usize, num_cols: usize) -> RecordBatch {
    let fields: Vec<Field> =
        (0..num_cols).map(|i| Field::new(format!("str_{i}"), DataType::Utf8, false)).collect();
    let schema = Arc::new(Schema::new(fields));

    // Pre-generate a pattern string to repeat
    let pattern: String =
        "abcdefghijklmnopqrstuvwxyz0123456789".chars().cycle().take(string_len).collect();

    let columns: Vec<ArrayRef> = (0..num_cols)
        .map(|_col| {
            let values: Vec<String> = (0..rows).map(|_| pattern.clone()).collect();
            Arc::new(StringArray::from(values)) as ArrayRef
        })
        .collect();

    RecordBatch::try_new(schema, columns).unwrap()
}

/// Create a batch with variable-length strings (realistic distribution)
fn create_variable_string_batch(
    rows: usize,
    min_len: usize,
    max_len: usize,
    num_cols: usize,
) -> RecordBatch {
    let fields: Vec<Field> =
        (0..num_cols).map(|i| Field::new(format!("str_{i}"), DataType::Utf8, false)).collect();
    let schema = Arc::new(Schema::new(fields));

    let columns: Vec<ArrayRef> = (0..num_cols)
        .map(|col| {
            let values: Vec<String> = (0..rows)
                .map(|i| {
                    // Deterministic pseudo-random length
                    let len = min_len + ((i * 31 + col * 17) % (max_len - min_len + 1));
                    "x".repeat(len)
                })
                .collect();
            Arc::new(StringArray::from(values)) as ArrayRef
        })
        .collect();

    RecordBatch::try_new(schema, columns).unwrap()
}

/// Create a batch with binary data
fn create_binary_batch(rows: usize, binary_len: usize, num_cols: usize) -> RecordBatch {
    let fields: Vec<Field> =
        (0..num_cols).map(|i| Field::new(format!("bin_{i}"), DataType::Binary, false)).collect();
    let schema = Arc::new(Schema::new(fields));

    let columns: Vec<ArrayRef> = (0..num_cols)
        .map(|col| {
            let values: Vec<Vec<u8>> = (0..rows)
                .map(|i| {
                    // Generate deterministic binary data
                    let seed = (i + col) as u64;
                    seed.to_le_bytes().iter().cycle().take(binary_len).copied().collect()
                })
                .collect();
            Arc::new(BinaryArray::from(values.iter().map(Vec::as_slice).collect::<Vec<_>>()))
                as ArrayRef
        })
        .collect();

    RecordBatch::try_new(schema, columns).unwrap()
}

/// Create a batch with `LargeUtf8` (for strings > 2GB total)
#[allow(dead_code)] // Reserved for future large string benchmarks
fn create_large_string_batch(rows: usize, string_len: usize, num_cols: usize) -> RecordBatch {
    let fields: Vec<Field> = (0..num_cols)
        .map(|i| Field::new(format!("lstr_{i}"), DataType::LargeUtf8, false))
        .collect();
    let schema = Arc::new(Schema::new(fields));

    let pattern: String =
        "abcdefghijklmnopqrstuvwxyz0123456789".chars().cycle().take(string_len).collect();

    let columns: Vec<ArrayRef> = (0..num_cols)
        .map(|_col| {
            let values: Vec<String> = (0..rows).map(|_| pattern.clone()).collect();
            Arc::new(LargeStringArray::from(values)) as ArrayRef
        })
        .collect();

    RecordBatch::try_new(schema, columns).unwrap()
}

/// Benchmark string inserts with different string lengths
fn bench_string_insert(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    init();

    let ch = rt.block_on(get_or_create_container(None));
    print_msg("Container ready for string benchmarks");

    let client = rt
        .block_on(
            arrow_tests::setup_test_arrow_client(ch.get_native_url(), &ch.user, &ch.password)
                .with_ipv4_only(true)
                .with_compression(CompressionMethod::None)
                .build::<ArrowFormat>(),
        )
        .expect("client setup");

    rt.block_on(arrow_tests::setup_database(TEST_DB_NAME, &client)).expect("database setup");

    let mut group = c.benchmark_group("StringInsert");
    group.sample_size(25);
    group.measurement_time(Duration::from_secs(15));

    // Test different string lengths
    for &string_len in STRING_LENGTHS {
        for &rows in ROW_COUNTS {
            let num_cols = 2;
            let batch = create_string_batch(rows, string_len, num_cols);
            let total_bytes = rows * string_len * num_cols;

            let table = rt
                .block_on(arrow_tests::setup_table(&client, TEST_DB_NAME, &batch.schema()))
                .expect("table setup");
            let query = format!("INSERT INTO {table} FORMAT NATIVE");

            group.throughput(Throughput::Bytes(total_bytes as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("len_{string_len}"), rows),
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

/// Benchmark variable-length string inserts
fn bench_variable_string_insert(c: &mut Criterion) {
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

    let mut group = c.benchmark_group("VariableStringInsert");
    group.sample_size(25);
    group.measurement_time(Duration::from_secs(15));

    // Test different variability ranges
    let variability_ranges = [(8, 32), (16, 256), (32, 1024)];

    for &(min_len, max_len) in &variability_ranges {
        for &rows in &[100_000, 500_000] {
            let num_cols = 2;
            let batch = create_variable_string_batch(rows, min_len, max_len, num_cols);

            // Estimate average bytes
            let avg_len = usize::midpoint(min_len, max_len);
            let total_bytes = rows * avg_len * num_cols;

            let table = rt
                .block_on(arrow_tests::setup_table(&client, TEST_DB_NAME, &batch.schema()))
                .expect("table setup");
            let query = format!("INSERT INTO {table} FORMAT NATIVE");

            group.throughput(Throughput::Bytes(total_bytes as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("range_{min_len}_{max_len}"), rows),
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

/// Benchmark binary data inserts
fn bench_binary_insert(c: &mut Criterion) {
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

    let mut group = c.benchmark_group("BinaryInsert");
    group.sample_size(25);
    group.measurement_time(Duration::from_secs(15));

    let binary_lengths = [16, 64, 256, 1024];

    for &binary_len in &binary_lengths {
        for &rows in &[100_000, 500_000] {
            let num_cols = 2;
            let batch = create_binary_batch(rows, binary_len, num_cols);
            let total_bytes = rows * binary_len * num_cols;

            let table = rt
                .block_on(arrow_tests::setup_table(&client, TEST_DB_NAME, &batch.schema()))
                .expect("table setup");
            let query = format!("INSERT INTO {table} FORMAT NATIVE");

            group.throughput(Throughput::Bytes(total_bytes as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("len_{binary_len}"), rows),
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

/// Benchmark string queries with different lengths
fn bench_string_query(c: &mut Criterion) {
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

    let mut group = c.benchmark_group("StringQuery");
    group.sample_size(25);
    group.measurement_time(Duration::from_secs(15));

    for &string_len in &[32, 128, 512] {
        for &rows in &[100_000, 500_000] {
            let num_cols = 2;
            let batch = create_string_batch(rows, string_len, num_cols);
            let total_bytes = rows * string_len * num_cols;

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
                BenchmarkId::new(format!("len_{string_len}"), rows),
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

/// Compare Utf8 vs Binary performance (strings stored as binary)
fn bench_utf8_vs_binary(c: &mut Criterion) {
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

    let mut group = c.benchmark_group("Utf8VsBinary");
    group.sample_size(25);
    group.measurement_time(Duration::from_secs(15));

    let rows = 500_000;
    let len = 64;
    let num_cols = 2;
    let total_bytes = rows * len * num_cols;

    // Utf8 insert
    let utf8_batch = create_string_batch(rows, len, num_cols);
    let table = rt
        .block_on(arrow_tests::setup_table(&client, TEST_DB_NAME, &utf8_batch.schema()))
        .expect("table setup");
    let query = format!("INSERT INTO {table} FORMAT NATIVE");

    group.throughput(Throughput::Bytes(total_bytes as u64));
    group.bench_with_input(
        BenchmarkId::new("utf8", rows),
        &(&query, &client, &utf8_batch),
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

    // Binary insert
    let binary_batch = create_binary_batch(rows, len, num_cols);
    let table = rt
        .block_on(arrow_tests::setup_table(&client, TEST_DB_NAME, &binary_batch.schema()))
        .expect("table setup");
    let query = format!("INSERT INTO {table} FORMAT NATIVE");

    group.bench_with_input(
        BenchmarkId::new("binary", rows),
        &(&query, &client, &binary_batch),
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

    group.finish();

    if std::env::var(DISABLE_CLEANUP_ENV).is_ok_and(|e| e.eq_ignore_ascii_case("true")) {
        return;
    }
    rt.block_on(ch.shutdown()).expect("shutdown");
}

criterion_group!(
    benches,
    bench_string_insert,
    bench_variable_string_insert,
    bench_binary_insert,
    bench_string_query,
    bench_utf8_vs_binary,
);
criterion_main!(benches);
