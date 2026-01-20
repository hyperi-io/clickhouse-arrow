#![expect(unused_crate_dependencies)]
mod common;

use std::time::Duration;

use arrow::record_batch::RecordBatch;
use clickhouse_arrow::CompressionMethod;
use clickhouse_arrow::prelude::*;
use clickhouse_arrow::test_utils::{arrow_tests, get_or_create_container};
use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main};
use tokio::runtime::Runtime;

use self::common::{init, print_msg};

fn insert_arrow(
    compression: &str,
    table: &str,
    rows: usize,
    client: &ArrowClient,
    batch: &RecordBatch,
    group: &mut BenchmarkGroup<'_, WallTime>,
    rt: &Runtime,
) {
    // Benchmark native arrow insert
    let query = format!("INSERT INTO {table} FORMAT NATIVE");
    let _ = group.sample_size(50).measurement_time(Duration::from_secs(30)).bench_with_input(
        BenchmarkId::new(format!("clickhouse_arrow_{compression}"), rows),
        &(&query, client),
        |b, (query, client)| {
            b.to_async(rt).iter_batched(
                || batch.clone(),
                |batch| async move {
                    let stream = client
                        .insert(query.as_str(), batch, None)
                        .await
                        .inspect_err(|e| print_msg(format!("Insert error\n{e:?}")))
                        .unwrap();
                    drop(stream);
                },
                criterion::BatchSize::SmallInput,
            );
        },
    );
}

#[allow(clippy::too_many_lines)]
fn criterion_benchmark(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    // Init tracing
    init();

    // Setup container once
    let ch = rt.block_on(get_or_create_container(None));
    print_msg("Created container");

    let mut insert_group = c.benchmark_group("Insert");

    // Create a test batch to pull out schema
    let schema = arrow_tests::create_test_batch(1, false).schema();

    // Setup arrow clients
    let client_builder =
        arrow_tests::setup_test_arrow_client(ch.get_native_url(), &ch.user, &ch.password)
            .with_ipv4_only(true);
    let arrow_client = rt
        .block_on(
            client_builder.clone().with_compression(CompressionMethod::None).build::<ArrowFormat>(),
        )
        .expect("clickhouse native arrow setup");
    let arrow_client_lz4 = rt
        .block_on(
            client_builder.clone().with_compression(CompressionMethod::LZ4).build::<ArrowFormat>(),
        )
        .expect("clickhouse native arrow setup");

    // Setup database
    rt.block_on(arrow_tests::setup_database(common::TEST_DB_NAME, &arrow_client))
        .expect("setup database");

    // Setup table
    let arrow_table_ref = rt
        .block_on(arrow_tests::setup_table(&arrow_client, common::TEST_DB_NAME, &schema))
        .expect("arrow table");

    // Test with different row counts
    let row_counts = vec![10_000, 100_000, 200_000, 300_000, 400_000];

    for rows in row_counts {
        print_msg(format!("Insert test for {rows} rows"));

        // Pre-create the batch to avoid including this in benchmark time
        let batch = arrow_tests::create_test_batch(rows, false);

        // Benchmark arrow insert no compression
        insert_arrow("none", &arrow_table_ref, rows, &arrow_client, &batch, &mut insert_group, &rt);

        // Benchmark arrow insert lz4 compression
        insert_arrow(
            "lz4",
            &arrow_table_ref,
            rows,
            &arrow_client_lz4,
            &batch,
            &mut insert_group,
            &rt,
        );
    }

    insert_group.finish();

    if std::env::var(common::DISABLE_CLEANUP_ENV).is_ok_and(|e| e.eq_ignore_ascii_case("true")) {
        return;
    }

    // Shutdown container
    rt.block_on(ch.shutdown()).expect("Shutting down container");
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
