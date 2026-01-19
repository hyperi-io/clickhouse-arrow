//! Benchmarks for LowCardinality (Dictionary) type operations.
//!
//! Tests performance of dictionary encoding/decoding with varying:
//! - Dictionary sizes (cardinality)
//! - Key types (UInt8, UInt16, UInt32)
//! - Value types (String, Binary)
//! - Row counts
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

/// Dictionary sizes (cardinalities) to test
/// These map to different key widths in ClickHouse:
/// - 10, 100: UInt8 (0-255)
/// - 1000: UInt16 (256-65535)
/// - 100_000: UInt32 (65536+)
const CARDINALITIES: &[usize] = &[10, 100, 1000, 100_000];

/// String lengths for dictionary values
const VALUE_LENGTHS: &[usize] = &[8, 32, 128];

/// Create a dictionary batch with string values
fn create_dictionary_string_batch(
    rows: usize,
    cardinality: usize,
    value_len: usize,
    num_cols: usize,
) -> RecordBatch {
    // Generate dictionary values
    let dict_values: Vec<String> = (0..cardinality)
        .map(|i| {
            let base = format!("val_{i:08}");
            base.chars().cycle().take(value_len).collect()
        })
        .collect();

    let fields: Vec<Field> = (0..num_cols)
        .map(|i| {
            Field::new(
                format!("dict_{i}"),
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                false,
            )
        })
        .collect();
    let schema = Arc::new(Schema::new(fields));

    let columns: Vec<ArrayRef> = (0..num_cols)
        .map(|col| {
            // Create keys that cycle through the dictionary
            let keys: Vec<i32> = (0..rows).map(|i| ((i + col * 7) % cardinality) as i32).collect();
            let keys_array = Int32Array::from(keys);

            // Create the dictionary values array
            let values_array = StringArray::from(dict_values.clone());

            Arc::new(
                DictionaryArray::<Int32Type>::try_new(keys_array, Arc::new(values_array)).unwrap(),
            ) as ArrayRef
        })
        .collect();

    RecordBatch::try_new(schema, columns).unwrap()
}

/// Create a dictionary batch with binary values
fn create_dictionary_binary_batch(
    rows: usize,
    cardinality: usize,
    value_len: usize,
    num_cols: usize,
) -> RecordBatch {
    // Generate dictionary values
    let dict_values: Vec<Vec<u8>> = (0..cardinality)
        .map(|i| {
            let seed = i as u64;
            seed.to_le_bytes().iter().cycle().take(value_len).copied().collect()
        })
        .collect();

    let fields: Vec<Field> = (0..num_cols)
        .map(|i| {
            Field::new(
                format!("dict_{i}"),
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Binary)),
                false,
            )
        })
        .collect();
    let schema = Arc::new(Schema::new(fields));

    let columns: Vec<ArrayRef> = (0..num_cols)
        .map(|col| {
            // Create keys that cycle through the dictionary
            let keys: Vec<i32> = (0..rows).map(|i| ((i + col * 7) % cardinality) as i32).collect();
            let keys_array = Int32Array::from(keys);

            // Create the dictionary values array
            let values_array =
                BinaryArray::from(dict_values.iter().map(Vec::as_slice).collect::<Vec<_>>());

            Arc::new(
                DictionaryArray::<Int32Type>::try_new(keys_array, Arc::new(values_array)).unwrap(),
            ) as ArrayRef
        })
        .collect();

    RecordBatch::try_new(schema, columns).unwrap()
}

/// Create a non-dictionary string batch for comparison (same data, no dictionary encoding)
fn create_plain_string_batch(
    rows: usize,
    cardinality: usize,
    value_len: usize,
    num_cols: usize,
) -> RecordBatch {
    // Generate dictionary values
    let dict_values: Vec<String> = (0..cardinality)
        .map(|i| {
            let base = format!("val_{i:08}");
            base.chars().cycle().take(value_len).collect()
        })
        .collect();

    let fields: Vec<Field> =
        (0..num_cols).map(|i| Field::new(format!("str_{i}"), DataType::Utf8, false)).collect();
    let schema = Arc::new(Schema::new(fields));

    let columns: Vec<ArrayRef> = (0..num_cols)
        .map(|col| {
            // Create values that cycle through (same as dictionary would)
            let values: Vec<String> =
                (0..rows).map(|i| dict_values[(i + col * 7) % cardinality].clone()).collect();
            Arc::new(StringArray::from(values)) as ArrayRef
        })
        .collect();

    RecordBatch::try_new(schema, columns).unwrap()
}

/// Benchmark LowCardinality insert with varying cardinality
fn bench_low_cardinality_insert_cardinality(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    init();

    let ch = rt.block_on(get_or_create_container(None));
    print_msg("Container ready for LowCardinality cardinality benchmarks");

    let client = rt
        .block_on(
            arrow_tests::setup_test_arrow_client(ch.get_native_url(), &ch.user, &ch.password)
                .with_ipv4_only(true)
                .with_compression(CompressionMethod::None)
                .build::<ArrowFormat>(),
        )
        .expect("client setup");

    rt.block_on(arrow_tests::setup_database(TEST_DB_NAME, &client)).expect("database setup");

    let mut group = c.benchmark_group("LowCardinalityInsert_Cardinality");
    let _ = group.sample_size(10);
    let _ = group.measurement_time(Duration::from_secs(20));

    let rows = 500_000;
    let value_len = 32;
    let num_cols = 1;

    for &cardinality in CARDINALITIES {
        let batch = create_dictionary_string_batch(rows, cardinality, value_len, num_cols);
        let table = rt
            .block_on(arrow_tests::setup_table(&client, TEST_DB_NAME, &batch.schema()))
            .expect("table setup");

        // Estimate bytes: rows * (key_size + avg_dict_overhead)
        // For low cardinality, the actual wire format is more efficient
        let key_size = if cardinality <= 256 {
            1
        } else if cardinality <= 65536 {
            2
        } else {
            4
        };
        let total_bytes = rows * key_size + cardinality * value_len;
        let _ = group.throughput(Throughput::Bytes(total_bytes as u64));

        let _ = group.bench_with_input(
            BenchmarkId::new("dictionary", format!("card_{cardinality}")),
            &(&table, &client, &batch),
            |b, (table, client, batch)| {
                b.to_async(&rt).iter(|| async {
                    let query = format!("INSERT INTO {table} FORMAT NATIVE");
                    let stream = client.insert(&query, (*batch).clone(), None).await.unwrap();
                    drop(black_box(stream));
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

/// Benchmark LowCardinality insert with varying value lengths
fn bench_low_cardinality_insert_value_length(c: &mut Criterion) {
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

    let mut group = c.benchmark_group("LowCardinalityInsert_ValueLength");
    let _ = group.sample_size(10);
    let _ = group.measurement_time(Duration::from_secs(20));

    let rows = 500_000;
    let cardinality = 100; // UInt8 keys
    let num_cols = 1;

    for &value_len in VALUE_LENGTHS {
        let batch = create_dictionary_string_batch(rows, cardinality, value_len, num_cols);
        let table = rt
            .block_on(arrow_tests::setup_table(&client, TEST_DB_NAME, &batch.schema()))
            .expect("table setup");

        let total_bytes = rows * 1 + cardinality * value_len; // UInt8 keys + dict values
        let _ = group.throughput(Throughput::Bytes(total_bytes as u64));

        let _ = group.bench_with_input(
            BenchmarkId::new("dictionary", format!("len_{value_len}")),
            &(&table, &client, &batch),
            |b, (table, client, batch)| {
                b.to_async(&rt).iter(|| async {
                    let query = format!("INSERT INTO {table} FORMAT NATIVE");
                    let stream = client.insert(&query, (*batch).clone(), None).await.unwrap();
                    drop(black_box(stream));
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

/// Benchmark LowCardinality vs plain String comparison
fn bench_low_cardinality_vs_plain(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    init();

    let ch = rt.block_on(get_or_create_container(None));
    print_msg("Container ready for LowCardinality vs plain comparison");

    let client = rt
        .block_on(
            arrow_tests::setup_test_arrow_client(ch.get_native_url(), &ch.user, &ch.password)
                .with_ipv4_only(true)
                .with_compression(CompressionMethod::None)
                .build::<ArrowFormat>(),
        )
        .expect("client setup");

    rt.block_on(arrow_tests::setup_database(TEST_DB_NAME, &client)).expect("database setup");

    let mut group = c.benchmark_group("LowCardinality_vs_Plain");
    let _ = group.sample_size(10);
    let _ = group.measurement_time(Duration::from_secs(20));

    let rows = 500_000;
    let value_len = 32;
    let num_cols = 1;

    // Test with different cardinalities to show when dictionary encoding helps
    for &cardinality in &[10, 100, 1000] {
        // Dictionary encoded batch
        let dict_batch = create_dictionary_string_batch(rows, cardinality, value_len, num_cols);
        let dict_table = rt
            .block_on(arrow_tests::setup_table(&client, TEST_DB_NAME, &dict_batch.schema()))
            .expect("table setup");

        // Plain string batch (same data, no dictionary)
        let plain_batch = create_plain_string_batch(rows, cardinality, value_len, num_cols);
        let plain_table = rt
            .block_on(arrow_tests::setup_table(&client, TEST_DB_NAME, &plain_batch.schema()))
            .expect("table setup");

        // Throughput based on raw string data
        let total_bytes = rows * value_len;
        let _ = group.throughput(Throughput::Bytes(total_bytes as u64));

        // Benchmark dictionary insert
        let _ = group.bench_with_input(
            BenchmarkId::new("dictionary", format!("card_{cardinality}")),
            &(&dict_table, &client, &dict_batch),
            |b, (table, client, batch)| {
                b.to_async(&rt).iter(|| async {
                    let query = format!("INSERT INTO {table} FORMAT NATIVE");
                    let stream = client.insert(&query, (*batch).clone(), None).await.unwrap();
                    drop(black_box(stream));
                });
            },
        );

        // Benchmark plain string insert
        let _ = group.bench_with_input(
            BenchmarkId::new("plain_string", format!("card_{cardinality}")),
            &(&plain_table, &client, &plain_batch),
            |b, (table, client, batch)| {
                b.to_async(&rt).iter(|| async {
                    let query = format!("INSERT INTO {table} FORMAT NATIVE");
                    let stream = client.insert(&query, (*batch).clone(), None).await.unwrap();
                    drop(black_box(stream));
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

/// Benchmark LowCardinality query performance
fn bench_low_cardinality_query(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    init();

    let ch = rt.block_on(get_or_create_container(None));
    print_msg("Container ready for LowCardinality query benchmarks");

    let client = rt
        .block_on(
            arrow_tests::setup_test_arrow_client(ch.get_native_url(), &ch.user, &ch.password)
                .with_ipv4_only(true)
                .with_compression(CompressionMethod::None)
                .build::<ArrowFormat>(),
        )
        .expect("client setup");

    rt.block_on(arrow_tests::setup_database(TEST_DB_NAME, &client)).expect("database setup");

    let mut group = c.benchmark_group("LowCardinalityQuery");
    let _ = group.sample_size(10);
    let _ = group.measurement_time(Duration::from_secs(20));

    let value_len = 32;
    let num_cols = 1;

    for &rows in &[100_000, 500_000] {
        for &cardinality in &[10, 100, 1000] {
            // Create and insert test data
            let batch = create_dictionary_string_batch(rows, cardinality, value_len, num_cols);
            let table = rt
                .block_on(arrow_tests::setup_table(&client, TEST_DB_NAME, &batch.schema()))
                .expect("table setup");

            // Insert the data
            rt.block_on(async {
                let insert_query = format!("INSERT INTO {table} FORMAT NATIVE");
                let mut stream = client.insert(&insert_query, batch, None).await.unwrap();
                while let Some(r) = stream.next().await {
                    r.unwrap();
                }
            });

            let key_size = if cardinality <= 256 { 1 } else { 2 };
            let total_bytes = rows * key_size + cardinality * value_len;
            let _ = group.throughput(Throughput::Bytes(total_bytes as u64));

            let query = format!("SELECT * FROM {table}");

            let _ = group.bench_with_input(
                BenchmarkId::new(format!("rows_{rows}"), format!("card_{cardinality}")),
                &(&query, &client),
                |b, (query, client)| {
                    b.to_async(&rt).iter(|| async {
                        let mut stream = client.query(*query, None).await.unwrap();
                        while let Some(result) = stream.next().await {
                            drop(black_box(result.unwrap()));
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

/// Benchmark LowCardinality with binary values
fn bench_low_cardinality_binary(c: &mut Criterion) {
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

    let mut group = c.benchmark_group("LowCardinalityBinary");
    let _ = group.sample_size(10);
    let _ = group.measurement_time(Duration::from_secs(20));

    let rows = 500_000;
    let num_cols = 1;

    for &cardinality in &[10, 100, 1000] {
        for &value_len in &[16, 64, 256] {
            let batch = create_dictionary_binary_batch(rows, cardinality, value_len, num_cols);
            let table = rt
                .block_on(arrow_tests::setup_table(&client, TEST_DB_NAME, &batch.schema()))
                .expect("table setup");

            let key_size = if cardinality <= 256 { 1 } else { 2 };
            let total_bytes = rows * key_size + cardinality * value_len;
            let _ = group.throughput(Throughput::Bytes(total_bytes as u64));

            let _ = group.bench_with_input(
                BenchmarkId::new(format!("card_{cardinality}"), format!("len_{value_len}")),
                &(&table, &client, &batch),
                |b, (table, client, batch)| {
                    b.to_async(&rt).iter(|| async {
                        let query = format!("INSERT INTO {table} FORMAT NATIVE");
                        let stream = client.insert(&query, (*batch).clone(), None).await.unwrap();
                        drop(black_box(stream));
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

/// Benchmark multi-column LowCardinality
fn bench_low_cardinality_multi_column(c: &mut Criterion) {
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

    let mut group = c.benchmark_group("LowCardinalityMultiColumn");
    let _ = group.sample_size(10);
    let _ = group.measurement_time(Duration::from_secs(20));

    let rows = 500_000;
    let cardinality = 100;
    let value_len = 32;

    for num_cols in [1, 2, 4, 8] {
        let batch = create_dictionary_string_batch(rows, cardinality, value_len, num_cols);
        let table = rt
            .block_on(arrow_tests::setup_table(&client, TEST_DB_NAME, &batch.schema()))
            .expect("table setup");

        // Each column has its own dictionary
        let total_bytes = num_cols * (rows * 1 + cardinality * value_len);
        let _ = group.throughput(Throughput::Bytes(total_bytes as u64));

        let _ = group.bench_with_input(
            BenchmarkId::new("insert", format!("{num_cols}_cols")),
            &(&table, &client, &batch),
            |b, (table, client, batch)| {
                b.to_async(&rt).iter(|| async {
                    let query = format!("INSERT INTO {table} FORMAT NATIVE");
                    let stream = client.insert(&query, (*batch).clone(), None).await.unwrap();
                    drop(black_box(stream));
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
    bench_low_cardinality_insert_cardinality,
    bench_low_cardinality_insert_value_length,
    bench_low_cardinality_vs_plain,
    bench_low_cardinality_query,
    bench_low_cardinality_binary,
    bench_low_cardinality_multi_column,
);
criterion_main!(benches);
