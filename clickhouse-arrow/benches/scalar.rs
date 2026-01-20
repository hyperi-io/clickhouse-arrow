#![expect(unused_crate_dependencies)]
mod common;

use std::sync::Arc;
use std::time::Duration;

use clickhouse_arrow::CompressionMethod;
use clickhouse_arrow::prelude::*;
use clickhouse_arrow::test_utils::{arrow_tests, get_or_create_container};
use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main};
use futures_util::StreamExt;
use tokio::runtime::Runtime;

use self::common::{init, print_msg};

fn query_arrow_native(
    query: &str,
    rows: usize,
    client: &Arc<ArrowClient>,
    group: &mut BenchmarkGroup<'_, WallTime>,
    rt: &Runtime,
) {
    let _ = group.sample_size(10).measurement_time(Duration::from_secs(60)).bench_with_input(
        BenchmarkId::new("clickhouse_arrow", rows),
        &(query, client),
        |b, (query, client)| {
            b.to_async(rt).iter(|| async move {
                let mut stream = client
                    .query(*query, None)
                    .await
                    .inspect_err(|e| print_msg(format!("Query error: {e:?}")))
                    .unwrap();
                while let Some(result) = stream.next().await {
                    drop(result.unwrap());
                }
            });
        },
    );
}

fn criterion_benchmark(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    // Init tracing
    init();

    // Setup container once
    let ch = rt.block_on(get_or_create_container(None));
    print_msg("Created container");

    let mut query_group = c.benchmark_group("Scalar");

    // Test with different row counts
    let rows = 500_000_000;
    let query = format!("SELECT number FROM system.numbers_mt LIMIT {rows}");

    print_msg(format!("Scalar query - default compression - test for {rows} rows"));

    // Setup client
    let arrow_client_builder =
        arrow_tests::setup_test_arrow_client(ch.get_native_url(), &ch.user, &ch.password)
            .with_ipv4_only(true)
            .with_compression(CompressionMethod::LZ4);

    let arrow_client = rt
        .block_on(arrow_client_builder.build::<ArrowFormat>())
        .expect("clickhouse native arrow setup");

    // Wrap client in Arc for sharing across iterations
    let arrow_client = Arc::new(arrow_client);

    // Benchmark native arrow query
    query_arrow_native(&query, rows, &arrow_client, &mut query_group, &rt);

    query_group.finish();

    if std::env::var(common::DISABLE_CLEANUP_ENV).is_ok_and(|e| e.eq_ignore_ascii_case("true")) {
        return;
    }

    rt.block_on(ch.shutdown()).expect("Shutting down container");
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
