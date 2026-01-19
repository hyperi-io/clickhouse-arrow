//! Version comparison benchmark.
//!
//! This benchmark compares the performance of the SIMD-optimized functions
//! against baseline scalar implementations to measure real-world improvements.
//!
//! Run with: cargo bench --bench version_compare

#![expect(unused_crate_dependencies)]

use std::hint::black_box;

use clickhouse_arrow::simd::{
    BUFFER_POOL, PooledBuffer, encode_varint, expand_null_bitmap, uuid_to_clickhouse,
};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

// ============================================================================
// BASELINE IMPLEMENTATIONS (pre-optimization)
// ============================================================================

/// Original naive null bitmap expansion - per-element loop
fn expand_null_bitmap_naive(bitmap: &[u8], output: &mut [u8], len: usize) {
    for i in 0..len {
        let byte_idx = i / 8;
        let bit_idx = i % 8;
        let is_valid = (bitmap[byte_idx] >> bit_idx) & 1;
        output[i] = (is_valid == 0) as u8;
    }
}

/// Original varint encoding
fn encode_varint_baseline(mut value: u64, buf: &mut [u8]) -> usize {
    let mut pos = 0;
    loop {
        if value < 0x80 {
            buf[pos] = value as u8;
            return pos + 1;
        }
        buf[pos] = (value as u8) | 0x80;
        value >>= 7;
        pos += 1;
    }
}

/// Original UUID conversion (two u64 operations)
fn uuid_to_clickhouse_baseline(uuid: &[u8; 16]) -> [u8; 16] {
    let low = u64::from_le_bytes(uuid[..8].try_into().unwrap());
    let high = u64::from_le_bytes(uuid[8..].try_into().unwrap());

    let mut result = [0u8; 16];
    result[..8].copy_from_slice(&high.to_le_bytes());
    result[8..].copy_from_slice(&low.to_le_bytes());
    result
}

// ============================================================================
// COMPREHENSIVE BENCHMARKS
// ============================================================================

fn bench_null_bitmap_realistic(c: &mut Criterion) {
    let mut group = c.benchmark_group("null_bitmap_realistic");

    // Realistic batch sizes in ClickHouse operations
    for rows in [1000, 10000, 65536, 100000] {
        let bitmap_len = (rows + 7) / 8;
        // Simulate ~25% null rate (every 4th bit is 0)
        let bitmap: Vec<u8> = (0..bitmap_len).map(|_| 0b11101110u8).collect();
        let mut output_baseline = vec![0u8; rows];
        let mut output_optimized = vec![0u8; rows];

        group.throughput(Throughput::Elements(rows as u64));

        group.bench_with_input(BenchmarkId::new("baseline", rows), &rows, |b, _| {
            b.iter(|| {
                expand_null_bitmap_naive(
                    black_box(&bitmap),
                    black_box(&mut output_baseline),
                    black_box(rows),
                );
            });
        });

        group.bench_with_input(BenchmarkId::new("optimized", rows), &rows, |b, _| {
            b.iter(|| {
                expand_null_bitmap(
                    black_box(&bitmap),
                    black_box(&mut output_optimized),
                    black_box(rows),
                );
            });
        });
    }

    group.finish();
}

fn bench_varint_realistic(c: &mut Criterion) {
    let mut group = c.benchmark_group("varint_realistic");

    // String lengths are the most common varint use case
    // Most strings are <128 bytes (1-byte varint)
    let small_lengths: Vec<u64> = (0..10000).map(|i| (i % 100) as u64).collect();
    // Some strings are 128-16383 bytes (2-byte varint)
    let medium_lengths: Vec<u64> = (0..10000).map(|i| 128 + (i % 1000) as u64).collect();
    // Mixed realistic distribution (80% small, 15% medium, 5% large)
    let mixed_lengths: Vec<u64> = (0..10000)
        .map(|i| {
            let r = i % 100;
            if r < 80 {
                (i % 100) as u64
            } else if r < 95 {
                128 + (i % 1000) as u64
            } else {
                16384 + (i % 10000) as u64
            }
        })
        .collect();

    for (name, values) in [
        ("small_strings", &small_lengths),
        ("medium_strings", &medium_lengths),
        ("mixed_realistic", &mixed_lengths),
    ] {
        group.throughput(Throughput::Elements(values.len() as u64));

        group.bench_with_input(BenchmarkId::new("baseline", name), values, |b, vals| {
            let mut buf = [0u8; 10];
            b.iter(|| {
                let mut total = 0usize;
                for &v in vals {
                    total += encode_varint_baseline(black_box(v), &mut buf);
                }
                black_box(total)
            });
        });

        group.bench_with_input(BenchmarkId::new("optimized", name), values, |b, vals| {
            let mut buf = [0u8; 10];
            b.iter(|| {
                let mut total = 0usize;
                for &v in vals {
                    total += encode_varint(black_box(v), &mut buf);
                }
                black_box(total)
            });
        });
    }

    group.finish();
}

fn bench_uuid_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("uuid_batch");

    for batch_size in [100, 1000, 10000] {
        let uuids: Vec<[u8; 16]> = (0..batch_size)
            .map(|i| {
                let mut uuid = [0u8; 16];
                uuid[0] = (i & 0xFF) as u8;
                uuid[1] = ((i >> 8) & 0xFF) as u8;
                uuid[8] = (i as u8).wrapping_mul(7);
                uuid
            })
            .collect();

        group.throughput(Throughput::Elements(batch_size as u64));

        group.bench_with_input(BenchmarkId::new("baseline", batch_size), &batch_size, |b, _| {
            b.iter(|| {
                for uuid in &uuids {
                    black_box(uuid_to_clickhouse_baseline(uuid));
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("optimized", batch_size), &batch_size, |b, _| {
            b.iter(|| {
                for uuid in &uuids {
                    black_box(uuid_to_clickhouse(uuid));
                }
            });
        });
    }

    group.finish();
}

fn bench_buffer_pool_realistic(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_pool_realistic");

    // Simulate a typical serialization workload:
    // Get buffer, use it, return it, repeat
    for size in [4096, 65536] {
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("vec_alloc", size), &size, |b, &sz| {
            b.iter(|| {
                // Simulate 10 serialization operations
                for _ in 0..10 {
                    let mut v = Vec::with_capacity(sz);
                    v.resize(sz, 0u8);
                    // Simulate some work
                    v[0] = 1;
                    v[sz - 1] = 2;
                    black_box(&v);
                    drop(v);
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("pooled", size), &size, |b, &sz| {
            b.iter(|| {
                // Simulate 10 serialization operations with pooling
                for _ in 0..10 {
                    let mut buf = PooledBuffer::with_capacity(sz);
                    buf.resize(sz, 0u8);
                    // Simulate some work
                    buf[0] = 1;
                    buf[sz - 1] = 2;
                    black_box(&*buf);
                    drop(buf);
                }
            });
        });

        // Also test the raw pool API for comparison
        group.bench_with_input(BenchmarkId::new("pool_raw", size), &size, |b, &sz| {
            b.iter(|| {
                for _ in 0..10 {
                    let mut buf = BUFFER_POOL.get(sz);
                    buf.resize(sz, 0u8);
                    buf[0] = 1;
                    buf[sz - 1] = 2;
                    black_box(&buf);
                    BUFFER_POOL.put(buf);
                }
            });
        });
    }

    group.finish();
}

/// Combined benchmark simulating a real serialization workload
fn bench_combined_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("combined_workload");

    // Simulate serializing a batch with nullable columns
    let rows = 10000;
    let bitmap_len = (rows + 7) / 8;
    let bitmap: Vec<u8> = (0..bitmap_len).map(|i| (i * 37) as u8).collect();
    let string_lengths: Vec<u64> = (0..rows).map(|i| 5 + (i % 50) as u64).collect();

    group.throughput(Throughput::Elements(rows as u64));

    group.bench_function("baseline_workload", |b| {
        let mut null_output = vec![0u8; rows];
        let mut varint_buf = [0u8; 10];

        b.iter(|| {
            // Expand null bitmap
            expand_null_bitmap_naive(&bitmap, &mut null_output, rows);

            // Encode string lengths
            let mut total_len = 0usize;
            for &len in &string_lengths {
                total_len += encode_varint_baseline(len, &mut varint_buf);
            }

            black_box(total_len)
        });
    });

    group.bench_function("optimized_workload", |b| {
        let mut null_output = PooledBuffer::with_capacity(rows);
        null_output.resize(rows, 0u8);
        let mut varint_buf = [0u8; 10];

        b.iter(|| {
            // Expand null bitmap with SIMD
            expand_null_bitmap(&bitmap, &mut null_output, rows);

            // Encode string lengths
            let mut total_len = 0usize;
            for &len in &string_lengths {
                total_len += encode_varint(len, &mut varint_buf);
            }

            black_box(total_len)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_null_bitmap_realistic,
    bench_varint_realistic,
    bench_uuid_batch,
    bench_buffer_pool_realistic,
    bench_combined_workload,
);

criterion_main!(benches);
