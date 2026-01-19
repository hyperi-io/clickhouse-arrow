//! A/B Performance benchmark for SIMD optimizations.
//!
//! This benchmark compares the optimized SIMD implementations against
//! the baseline scalar implementations to measure actual performance gains.

use std::hint::black_box as bb;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};

// ============================================================================
// NULL BITMAP EXPANSION BENCHMARKS
// ============================================================================

/// Baseline scalar implementation (from original code)
fn expand_null_bitmap_baseline(bitmap: &[u8], output: &mut [u8], len: usize) {
    let full_bytes = len / 8;
    let remainder = len % 8;

    for (byte_idx, &byte) in bitmap.iter().take(full_bytes).enumerate() {
        let base = byte_idx * 8;
        output[base] = ((byte & 0x01) == 0) as u8;
        output[base + 1] = ((byte & 0x02) == 0) as u8;
        output[base + 2] = ((byte & 0x04) == 0) as u8;
        output[base + 3] = ((byte & 0x08) == 0) as u8;
        output[base + 4] = ((byte & 0x10) == 0) as u8;
        output[base + 5] = ((byte & 0x20) == 0) as u8;
        output[base + 6] = ((byte & 0x40) == 0) as u8;
        output[base + 7] = ((byte & 0x80) == 0) as u8;
    }

    if remainder > 0 {
        let byte = bitmap[full_bytes];
        let base = full_bytes * 8;
        for bit in 0..remainder {
            output[base + bit] = ((byte & (1 << bit)) == 0) as u8;
        }
    }
}

/// Original naive implementation (per-element loop)
fn expand_null_bitmap_naive(bitmap: &[u8], output: &mut [u8], len: usize) {
    for i in 0..len {
        let byte_idx = i / 8;
        let bit_idx = i % 8;
        let is_valid = (bitmap[byte_idx] >> bit_idx) & 1;
        output[i] = (is_valid == 0) as u8;
    }
}

fn bench_null_bitmap_expansion(c: &mut Criterion) {
    let mut group = c.benchmark_group("null_bitmap_expansion");

    for size in [64, 256, 1024, 4096, 16384, 65536] {
        let bitmap_len = (size + 7) / 8;
        let bitmap: Vec<u8> = (0..bitmap_len).map(|i| (i * 37) as u8).collect();
        let mut output_naive = vec![0u8; size];
        let mut output_baseline = vec![0u8; size];
        let mut output_simd = vec![0u8; size];

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("naive", size), &size, |b, _| {
            b.iter(|| {
                expand_null_bitmap_naive(
                    black_box(&bitmap),
                    black_box(&mut output_naive),
                    black_box(size),
                );
            });
        });

        group.bench_with_input(BenchmarkId::new("baseline_unrolled", size), &size, |b, _| {
            b.iter(|| {
                expand_null_bitmap_baseline(
                    black_box(&bitmap),
                    black_box(&mut output_baseline),
                    black_box(size),
                );
            });
        });

        group.bench_with_input(BenchmarkId::new("simd_optimized", size), &size, |b, _| {
            b.iter(|| {
                clickhouse_arrow::simd::expand_null_bitmap(
                    black_box(&bitmap),
                    black_box(&mut output_simd),
                    black_box(size),
                );
            });
        });
    }

    group.finish();
}

// ============================================================================
// UUID CONVERSION BENCHMARKS
// ============================================================================

/// Baseline UUID conversion (two separate u64 operations)
fn uuid_to_clickhouse_baseline(uuid: &[u8]) -> [u8; 16] {
    let bytes: [u8; 16] = uuid.try_into().unwrap();
    let low = u64::from_le_bytes(bytes[..8].try_into().unwrap());
    let high = u64::from_le_bytes(bytes[8..].try_into().unwrap());

    let mut result = [0u8; 16];
    result[..8].copy_from_slice(&high.to_le_bytes());
    result[8..].copy_from_slice(&low.to_le_bytes());
    result
}

fn bench_uuid_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("uuid_conversion");

    // Test with various batch sizes
    for batch_size in [1, 10, 100, 1000, 10000] {
        let uuids: Vec<[u8; 16]> = (0..batch_size)
            .map(|i| {
                let mut uuid = [0u8; 16];
                uuid[0] = (i & 0xFF) as u8;
                uuid[8] = ((i >> 8) & 0xFF) as u8;
                uuid
            })
            .collect();

        group.throughput(Throughput::Elements(batch_size as u64));

        group.bench_with_input(BenchmarkId::new("baseline", batch_size), &batch_size, |b, _| {
            b.iter(|| {
                for uuid in &uuids {
                    bb(uuid_to_clickhouse_baseline(uuid));
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("optimized", batch_size), &batch_size, |b, _| {
            b.iter(|| {
                for uuid in &uuids {
                    bb(clickhouse_arrow::simd::uuid_to_clickhouse(uuid));
                }
            });
        });

        group.bench_with_input(
            BenchmarkId::new("optimized_slice", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    for uuid in &uuids {
                        bb(clickhouse_arrow::simd::uuid_slice_to_clickhouse(uuid.as_slice()));
                    }
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// VARINT ENCODING BENCHMARKS
// ============================================================================

/// Baseline varint encoding
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

fn bench_varint_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("varint_encoding");

    // Test small values (common case: string lengths)
    let small_values: Vec<u64> = (0..1000).map(|i| i % 128).collect();
    // Test medium values
    let medium_values: Vec<u64> = (0..1000).map(|i| 128 + (i * 100)).collect();
    // Test large values
    let large_values: Vec<u64> = (0..1000).map(|i| u32::MAX as u64 + i).collect();

    for (name, values) in [
        ("small_0-127", &small_values),
        ("medium_128-100k", &medium_values),
        ("large_u32+", &large_values),
    ] {
        group.throughput(Throughput::Elements(values.len() as u64));

        group.bench_with_input(BenchmarkId::new("baseline", name), values, |b, vals| {
            let mut buf = [0u8; 10];
            b.iter(|| {
                for &v in vals {
                    bb(encode_varint_baseline(v, &mut buf));
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("optimized", name), values, |b, vals| {
            let mut buf = [0u8; 10];
            b.iter(|| {
                for &v in vals {
                    bb(clickhouse_arrow::simd::encode_varint(v, &mut buf));
                }
            });
        });
    }

    group.finish();
}

// ============================================================================
// BUFFER POOL BENCHMARKS
// ============================================================================

fn bench_buffer_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_allocation");

    for size in [1024, 4096, 65536, 1_048_576] {
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("vec_new", size), &size, |b, &sz| {
            b.iter(|| {
                let mut v = Vec::with_capacity(sz);
                v.resize(sz, 0u8);
                bb(&v);
                drop(v);
            });
        });

        group.bench_with_input(BenchmarkId::new("pooled_buffer", size), &size, |b, &sz| {
            b.iter(|| {
                let mut buf = clickhouse_arrow::simd::PooledBuffer::with_capacity(sz);
                buf.resize(sz, 0u8);
                bb(&*buf);
                drop(buf);
            });
        });
    }

    group.finish();
}

// ============================================================================
// BYTE SWAP BENCHMARKS
// ============================================================================

fn bench_byte_swap(c: &mut Criterion) {
    let mut group = c.benchmark_group("byte_swap");

    for size in [64, 256, 1024, 4096] {
        // u32 swapping
        let mut data_u32: Vec<u32> = (0..size).map(|i| i as u32 * 0x01020304).collect();
        let mut data_u32_baseline = data_u32.clone();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("u32_baseline", size), &size, |b, _| {
            b.iter(|| {
                for v in data_u32_baseline.iter_mut() {
                    *v = v.swap_bytes();
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("u32_simd", size), &size, |b, _| {
            b.iter(|| {
                clickhouse_arrow::simd::swap_bytes_u32_slice(&mut data_u32);
            });
        });

        // u64 swapping
        let mut data_u64: Vec<u64> = (0..size).map(|i| i as u64 * 0x0102030405060708).collect();
        let mut data_u64_baseline = data_u64.clone();

        group.bench_with_input(BenchmarkId::new("u64_baseline", size), &size, |b, _| {
            b.iter(|| {
                for v in data_u64_baseline.iter_mut() {
                    *v = v.swap_bytes();
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("u64_simd", size), &size, |b, _| {
            b.iter(|| {
                clickhouse_arrow::simd::swap_bytes_u64_slice(&mut data_u64);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_null_bitmap_expansion,
    bench_uuid_conversion,
    bench_varint_encoding,
    bench_buffer_allocation,
    bench_byte_swap,
);

criterion_main!(benches);
