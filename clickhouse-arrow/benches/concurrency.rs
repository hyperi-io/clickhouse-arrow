//! Concurrency benchmarks for parallel insert/query operations.
//!
//! Tests throughput under concurrent load, connection pool effectiveness,
//! and scaling characteristics.
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

/// Concurrency levels to test
const CONCURRENCY_LEVELS: &[usize] = &[1, 2, 4, 8, 16];

/// Batch size for concurrency tests
const BATCH_SIZE: usize = 50_000;

/// Create a test batch
fn create_test_batch(rows: usize, offset: usize) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("ts", DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())), false),
    ]));

    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from((0..rows).map(|i| (offset + i) as i64).collect::<Vec<_>>()))
            as ArrayRef,
        Arc::new(Float64Array::from(
            (0..rows).map(|i| ((offset + i) as f64) * 1.5).collect::<Vec<_>>(),
        )) as ArrayRef,
        Arc::new(
            TimestampMillisecondArray::from(
                (0..rows).map(|i| ((offset + i) as i64) * 1000).collect::<Vec<_>>(),
            )
            .with_timezone(Arc::from("UTC")),
        ) as ArrayRef,
    ];

    RecordBatch::try_new(schema, columns).unwrap()
}

/// Benchmark concurrent inserts with shared client
fn bench_concurrent_insert_shared_client(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    init();

    let ch = rt.block_on(get_or_create_container(None));
    print_msg("Container ready for concurrency benchmarks");

    let client = Arc::new(
        rt.block_on(
            arrow_tests::setup_test_arrow_client(ch.get_native_url(), &ch.user, &ch.password)
                .with_ipv4_only(true)
                .with_compression(CompressionMethod::None)
                .build::<ArrowFormat>(),
        )
        .expect("client setup"),
    );

    rt.block_on(arrow_tests::setup_database(TEST_DB_NAME, &client)).expect("database setup");

    let mut group = c.benchmark_group("ConcurrentInsert");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    let bytes_per_row = 24;

    for &concurrency in CONCURRENCY_LEVELS {
        // Pre-create batches and tables for this concurrency level
        let batches: Arc<Vec<RecordBatch>> = Arc::new(
            (0..concurrency).map(|i| create_test_batch(BATCH_SIZE, i * BATCH_SIZE)).collect(),
        );

        let tables: Arc<Vec<String>> = Arc::new(
            (0..concurrency)
                .map(|_| {
                    rt.block_on(arrow_tests::setup_table(
                        &client,
                        TEST_DB_NAME,
                        &batches[0].schema(),
                    ))
                    .expect("table setup")
                })
                .collect(),
        );

        let total_bytes = BATCH_SIZE * bytes_per_row * concurrency;
        group.throughput(Throughput::Bytes(total_bytes as u64));

        group.bench_function(BenchmarkId::new("shared_client", concurrency), |b| {
            b.to_async(&rt).iter(|| {
                let client = Arc::clone(&client);
                let tables = Arc::clone(&tables);
                let batches = Arc::clone(&batches);
                async move {
                    let handles: Vec<_> = (0..concurrency)
                        .map(|i| {
                            let client = Arc::clone(&client);
                            let table = tables[i].clone();
                            let batch = batches[i].clone();
                            tokio::spawn(async move {
                                let query = format!("INSERT INTO {table} FORMAT NATIVE");
                                let stream = client.insert(&query, batch, None).await.unwrap();
                                drop(black_box(stream));
                            })
                        })
                        .collect();

                    for handle in handles {
                        handle.await.unwrap();
                    }
                }
            });
        });
    }

    group.finish();

    if std::env::var(DISABLE_CLEANUP_ENV).is_ok_and(|e| e.eq_ignore_ascii_case("true")) {
        return;
    }
    rt.block_on(ch.shutdown()).expect("shutdown");
}

/// Benchmark concurrent queries
fn bench_concurrent_query(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    init();

    let ch = rt.block_on(get_or_create_container(None));

    let client = Arc::new(
        rt.block_on(
            arrow_tests::setup_test_arrow_client(ch.get_native_url(), &ch.user, &ch.password)
                .with_ipv4_only(true)
                .with_compression(CompressionMethod::None)
                .build::<ArrowFormat>(),
        )
        .expect("client setup"),
    );

    rt.block_on(arrow_tests::setup_database(TEST_DB_NAME, &client)).expect("database setup");

    let mut group = c.benchmark_group("ConcurrentQuery");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    let bytes_per_row = 24;
    let rows = 200_000;

    // Create and populate a single table for query benchmarks
    let batch = create_test_batch(rows, 0);
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

    let query = Arc::new(format!("SELECT * FROM {table}"));

    for &concurrency in CONCURRENCY_LEVELS {
        let total_bytes = rows * bytes_per_row * concurrency;
        group.throughput(Throughput::Bytes(total_bytes as u64));

        group.bench_function(BenchmarkId::new("shared_client", concurrency), |b| {
            b.to_async(&rt).iter(|| {
                let client = Arc::clone(&client);
                let query = Arc::clone(&query);
                async move {
                    let handles: Vec<_> = (0..concurrency)
                        .map(|_| {
                            let client = Arc::clone(&client);
                            let query = Arc::clone(&query);
                            tokio::spawn(async move {
                                let mut stream = client.query(&*query, None).await.unwrap();
                                while let Some(result) = stream.next().await {
                                    black_box(result.unwrap());
                                }
                            })
                        })
                        .collect();

                    for handle in handles {
                        handle.await.unwrap();
                    }
                }
            });
        });
    }

    group.finish();

    if std::env::var(DISABLE_CLEANUP_ENV).is_ok_and(|e| e.eq_ignore_ascii_case("true")) {
        return;
    }
    rt.block_on(ch.shutdown()).expect("shutdown");
}

/// Benchmark mixed read/write workload
fn bench_mixed_workload(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    init();

    let ch = rt.block_on(get_or_create_container(None));

    let client = Arc::new(
        rt.block_on(
            arrow_tests::setup_test_arrow_client(ch.get_native_url(), &ch.user, &ch.password)
                .with_ipv4_only(true)
                .with_compression(CompressionMethod::None)
                .build::<ArrowFormat>(),
        )
        .expect("client setup"),
    );

    rt.block_on(arrow_tests::setup_database(TEST_DB_NAME, &client)).expect("database setup");

    let mut group = c.benchmark_group("MixedWorkload");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    let insert_batch_size = 25_000;
    let query_rows = 100_000;
    let bytes_per_row = 24;

    // Create and populate query table
    let query_batch = create_test_batch(query_rows, 0);
    let query_table = rt
        .block_on(arrow_tests::setup_table(&client, TEST_DB_NAME, &query_batch.schema()))
        .expect("query table setup");

    rt.block_on(async {
        let insert_query = format!("INSERT INTO {query_table} FORMAT NATIVE");
        let mut stream = client.insert(&insert_query, query_batch.clone(), None).await.unwrap();
        while let Some(r) = stream.next().await {
            r.unwrap();
        }
    });

    let select_query = Arc::new(format!("SELECT * FROM {query_table}"));

    // 50% reads, 50% writes ratio
    let read_write_ratios = [(1, 1), (2, 1), (1, 2), (4, 1)];

    for (reads, writes) in read_write_ratios {
        let total_ops = reads + writes;
        let total_bytes = (reads * query_rows + writes * insert_batch_size) * bytes_per_row;

        // Create insert tables
        let insert_tables: Arc<Vec<String>> = Arc::new(
            (0..writes)
                .map(|_| {
                    let batch = create_test_batch(1, 0);
                    rt.block_on(arrow_tests::setup_table(&client, TEST_DB_NAME, &batch.schema()))
                        .expect("insert table setup")
                })
                .collect(),
        );

        group.throughput(Throughput::Bytes(total_bytes as u64));

        group.bench_function(BenchmarkId::new(format!("r{reads}_w{writes}"), total_ops), |b| {
            b.to_async(&rt).iter(|| {
                let client = Arc::clone(&client);
                let select_query = Arc::clone(&select_query);
                let insert_tables = Arc::clone(&insert_tables);
                async move {
                    let mut handles = Vec::new();

                    // Spawn read tasks
                    for _ in 0..reads {
                        let client = Arc::clone(&client);
                        let query = Arc::clone(&select_query);
                        handles.push(tokio::spawn(async move {
                            let mut stream = client.query(&*query, None).await.unwrap();
                            while let Some(result) = stream.next().await {
                                black_box(result.unwrap());
                            }
                        }));
                    }

                    // Spawn write tasks
                    for (idx, table) in insert_tables.iter().enumerate() {
                        let client = Arc::clone(&client);
                        let table = table.clone();
                        let batch = create_test_batch(insert_batch_size, idx * insert_batch_size);
                        handles.push(tokio::spawn(async move {
                            let query = format!("INSERT INTO {table} FORMAT NATIVE");
                            let stream = client.insert(&query, batch, None).await.unwrap();
                            drop(black_box(stream));
                        }));
                    }

                    for handle in handles {
                        handle.await.unwrap();
                    }
                }
            });
        });
    }

    group.finish();

    if std::env::var(DISABLE_CLEANUP_ENV).is_ok_and(|e| e.eq_ignore_ascii_case("true")) {
        return;
    }
    rt.block_on(ch.shutdown()).expect("shutdown");
}

/// Benchmark throughput scaling with number of concurrent operations
fn bench_throughput_scaling(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    init();

    let ch = rt.block_on(get_or_create_container(None));

    let client = Arc::new(
        rt.block_on(
            arrow_tests::setup_test_arrow_client(ch.get_native_url(), &ch.user, &ch.password)
                .with_ipv4_only(true)
                .with_compression(CompressionMethod::None)
                .build::<ArrowFormat>(),
        )
        .expect("client setup"),
    );

    rt.block_on(arrow_tests::setup_database(TEST_DB_NAME, &client)).expect("database setup");

    let mut group = c.benchmark_group("ThroughputScaling");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    let bytes_per_row = 24;
    // Keep total work constant, vary parallelism
    let total_rows = 500_000;

    for &concurrency in &[1, 2, 4, 8] {
        let rows_per_task = total_rows / concurrency;

        let batches: Arc<Vec<RecordBatch>> = Arc::new(
            (0..concurrency).map(|i| create_test_batch(rows_per_task, i * rows_per_task)).collect(),
        );

        let tables: Arc<Vec<String>> = Arc::new(
            (0..concurrency)
                .map(|_| {
                    rt.block_on(arrow_tests::setup_table(
                        &client,
                        TEST_DB_NAME,
                        &batches[0].schema(),
                    ))
                    .expect("table setup")
                })
                .collect(),
        );

        let total_bytes = total_rows * bytes_per_row;
        group.throughput(Throughput::Bytes(total_bytes as u64));

        group.bench_function(
            BenchmarkId::new("constant_work", format!("{concurrency}x{rows_per_task}")),
            |b| {
                b.to_async(&rt).iter(|| {
                    let client = Arc::clone(&client);
                    let tables = Arc::clone(&tables);
                    let batches = Arc::clone(&batches);
                    async move {
                        let handles: Vec<_> = (0..concurrency)
                            .map(|i| {
                                let client = Arc::clone(&client);
                                let table = tables[i].clone();
                                let batch = batches[i].clone();
                                tokio::spawn(async move {
                                    let query = format!("INSERT INTO {table} FORMAT NATIVE");
                                    let stream = client.insert(&query, batch, None).await.unwrap();
                                    drop(black_box(stream));
                                })
                            })
                            .collect();

                        for handle in handles {
                            handle.await.unwrap();
                        }
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

criterion_group!(
    benches,
    bench_concurrent_insert_shared_client,
    bench_concurrent_query,
    bench_mixed_workload,
    bench_throughput_scaling,
);
criterion_main!(benches);
