//! Batch size sensitivity benchmarks.
//!
//! Tests how different batch sizes affect throughput, latency, and memory usage.
//! Critical for tuning the optimal batch size for different workloads.
#![expect(unused_crate_dependencies)]

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

/// Batch sizes to test - spanning 3 orders of magnitude
const BATCH_SIZES: &[usize] =
    &[1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000];

/// Total rows to insert for measuring multi-batch performance
const TOTAL_ROWS_MULTI: usize = 1_000_000;

/// Create a fixed schema batch with typical column types
fn create_typical_batch(rows: usize) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("ts", DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())), false),
        Field::new("category", DataType::Int32, false),
    ]));

    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from((0..rows).map(|i| i as i64).collect::<Vec<_>>())) as ArrayRef,
        Arc::new(Float64Array::from((0..rows).map(|i| (i as f64) * 1.5).collect::<Vec<_>>()))
            as ArrayRef,
        Arc::new(
            TimestampMillisecondArray::from(
                (0..rows).map(|i| (i as i64) * 1000).collect::<Vec<_>>(),
            )
            .with_timezone(Arc::from("UTC")),
        ) as ArrayRef,
        Arc::new(Int32Array::from((0..rows).map(|i| (i % 100) as i32).collect::<Vec<_>>()))
            as ArrayRef,
    ];

    RecordBatch::try_new(schema, columns).unwrap()
}

/// Create a wide batch with many columns
fn create_wide_batch(rows: usize, num_cols: usize) -> RecordBatch {
    let fields: Vec<Field> =
        (0..num_cols).map(|i| Field::new(format!("col_{i}"), DataType::Int64, false)).collect();
    let schema = Arc::new(Schema::new(fields));

    let columns: Vec<ArrayRef> = (0..num_cols)
        .map(|col| {
            Arc::new(Int64Array::from((0..rows).map(|i| (i + col) as i64).collect::<Vec<_>>()))
                as ArrayRef
        })
        .collect();

    RecordBatch::try_new(schema, columns).unwrap()
}

/// Create a mixed-type batch (more realistic workload)
fn create_mixed_batch(rows: usize) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("small_int", DataType::Int8, false),
        Field::new("medium_int", DataType::Int32, false),
        Field::new("large_int", DataType::Int64, false),
        Field::new("float_val", DataType::Float64, false),
        Field::new("ts", DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())), false),
        Field::new("name", DataType::Binary, false),
    ]));

    let name_pattern: Vec<u8> = b"user_name_12345678".to_vec();

    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from((0..rows).map(|i| i as i64).collect::<Vec<_>>())) as ArrayRef,
        Arc::new(Int8Array::from((0..rows).map(|i| (i % 128) as i8).collect::<Vec<_>>()))
            as ArrayRef,
        Arc::new(Int32Array::from((0..rows).map(|i| i as i32).collect::<Vec<_>>())) as ArrayRef,
        Arc::new(Int64Array::from((0..rows).map(|i| i as i64).collect::<Vec<_>>())) as ArrayRef,
        Arc::new(Float64Array::from((0..rows).map(|i| (i as f64) * 1.5).collect::<Vec<_>>()))
            as ArrayRef,
        Arc::new(
            TimestampMillisecondArray::from(
                (0..rows).map(|i| (i as i64) * 1000).collect::<Vec<_>>(),
            )
            .with_timezone(Arc::from("UTC")),
        ) as ArrayRef,
        Arc::new(BinaryArray::from_iter_values((0..rows).map(|_| name_pattern.as_slice())))
            as ArrayRef,
    ];

    RecordBatch::try_new(schema, columns).unwrap()
}

/// Benchmark single-batch inserts at different sizes
fn bench_batch_size_insert(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    init();

    let ch = rt.block_on(get_or_create_container(None));
    print_msg("Container ready for batch size benchmarks");

    let client = rt
        .block_on(
            arrow_tests::setup_test_arrow_client(ch.get_native_url(), &ch.user, &ch.password)
                .with_ipv4_only(true)
                .with_compression(CompressionMethod::None)
                .build::<ArrowFormat>(),
        )
        .expect("client setup");

    rt.block_on(arrow_tests::setup_database(TEST_DB_NAME, &client)).expect("database setup");

    let mut group = c.benchmark_group("BatchSizeInsert");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(15));

    for &batch_size in BATCH_SIZES {
        let batch = create_typical_batch(batch_size);
        // 4 columns: Int64(8) + Float64(8) + Timestamp(8) + Int32(4) = 28 bytes/row
        let bytes_per_row = 28;
        let total_bytes = batch_size * bytes_per_row;

        let table = rt
            .block_on(arrow_tests::setup_table(&client, TEST_DB_NAME, &batch.schema()))
            .expect("table setup");
        let query = format!("INSERT INTO {table} FORMAT NATIVE");

        group.throughput(Throughput::Bytes(total_bytes as u64));
        group.bench_with_input(
            BenchmarkId::new("typical_4col", batch_size),
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

/// Benchmark multi-batch inserts (same total data, different batch sizes)
fn bench_multi_batch_insert(c: &mut Criterion) {
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

    let mut group = c.benchmark_group("MultiBatchInsert");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    // Test inserting TOTAL_ROWS_MULTI rows with different batch sizes
    let batch_sizes = [10_000, 50_000, 100_000, 500_000];
    let bytes_per_row = 28;

    for &batch_size in &batch_sizes {
        let num_batches = TOTAL_ROWS_MULTI / batch_size;
        let batches: Vec<RecordBatch> =
            (0..num_batches).map(|_| create_typical_batch(batch_size)).collect();

        let table = rt
            .block_on(arrow_tests::setup_table(&client, TEST_DB_NAME, &batches[0].schema()))
            .expect("table setup");
        let query = format!("INSERT INTO {table} FORMAT NATIVE");

        let total_bytes = TOTAL_ROWS_MULTI * bytes_per_row;
        group.throughput(Throughput::Bytes(total_bytes as u64));

        group.bench_with_input(
            BenchmarkId::new("1M_rows", format!("{batch_size}_x_{num_batches}")),
            &(&query, &client, batches),
            |b, (query, client, batches)| {
                b.to_async(&rt).iter_batched(
                    || batches.clone(),
                    |batches| async {
                        for batch in batches {
                            let stream = client.insert(*query, batch, None).await.unwrap();
                            drop(black_box(stream));
                        }
                    },
                    criterion::BatchSize::LargeInput,
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

/// Benchmark wide tables (many columns) at different batch sizes
fn bench_wide_batch_insert(c: &mut Criterion) {
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

    let mut group = c.benchmark_group("WideBatchInsert");
    group.sample_size(15);
    group.measurement_time(Duration::from_secs(15));

    // Test with different column counts
    let column_counts = [10, 50, 100, 200];
    let batch_sizes = [10_000, 50_000, 100_000];

    for &num_cols in &column_counts {
        for &batch_size in &batch_sizes {
            let batch = create_wide_batch(batch_size, num_cols);
            let total_bytes = batch_size * num_cols * 8; // Int64 = 8 bytes

            let table = rt
                .block_on(arrow_tests::setup_table(&client, TEST_DB_NAME, &batch.schema()))
                .expect("table setup");
            let query = format!("INSERT INTO {table} FORMAT NATIVE");

            group.throughput(Throughput::Bytes(total_bytes as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("{num_cols}cols"), batch_size),
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

/// Benchmark batch size impact on queries
fn bench_batch_size_query(c: &mut Criterion) {
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

    let mut group = c.benchmark_group("BatchSizeQuery");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(15));

    let row_counts = [100_000, 500_000, 1_000_000];
    let bytes_per_row = 28;

    for &rows in &row_counts {
        let batch = create_typical_batch(rows);
        let table = rt
            .block_on(arrow_tests::setup_table(&client, TEST_DB_NAME, &batch.schema()))
            .expect("table setup");

        // Insert data
        rt.block_on(async {
            let insert_query = format!("INSERT INTO {table} FORMAT NATIVE");
            let mut stream = client.insert(&insert_query, batch, None).await.unwrap();
            while let Some(r) = stream.next().await {
                r.unwrap();
            }
        });

        let total_bytes = rows * bytes_per_row;
        group.throughput(Throughput::Bytes(total_bytes as u64));

        let query = format!("SELECT * FROM {table}");
        group.bench_with_input(
            BenchmarkId::new("typical_4col", rows),
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

/// Benchmark mixed-type batch performance
fn bench_mixed_type_batch(c: &mut Criterion) {
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

    let mut group = c.benchmark_group("MixedTypeBatch");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(15));

    let batch_sizes = [10_000, 50_000, 100_000, 500_000];
    // Int64(8) + Int8(1) + Int32(4) + Int64(8) + Float64(8) + Timestamp(8) + Binary(~18) = ~55
    // bytes/row
    let bytes_per_row = 55;

    for &batch_size in &batch_sizes {
        let batch = create_mixed_batch(batch_size);
        let total_bytes = batch_size * bytes_per_row;

        let table = rt
            .block_on(arrow_tests::setup_table(&client, TEST_DB_NAME, &batch.schema()))
            .expect("table setup");
        let query = format!("INSERT INTO {table} FORMAT NATIVE");

        group.throughput(Throughput::Bytes(total_bytes as u64));
        group.bench_with_input(
            BenchmarkId::new("mixed_7col", batch_size),
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

criterion_group!(
    benches,
    bench_batch_size_insert,
    bench_multi_batch_insert,
    bench_wide_batch_insert,
    bench_batch_size_query,
    bench_mixed_type_batch,
);
criterion_main!(benches);
