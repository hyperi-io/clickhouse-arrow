# ClickHouse Native Protocol Rust Client w/ Arrow Compatibility

Native TCP ClickHouse client with Apache Arrow integration. v0.4.2 "Life the Universe and Everything".

Supports protocol revision `54479` (ClickHouse 25.x).

[![Crates.io](https://img.shields.io/crates/v/clickhouse-arrow.svg)](https://crates.io/crates/clickhouse-arrow)
[![Documentation](https://docs.rs/clickhouse-arrow/badge.svg)](https://docs.rs/clickhouse-arrow)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Build Status](https://img.shields.io/github/actions/workflow/status/GeorgeLeePatterson/clickhouse-arrow/ci.yml?branch=main)](https://github.com/GeorgeLeePatterson/clickhouse-arrow/actions)
[![Coverage](https://codecov.io/gh/GeorgeLeePatterson/clickhouse-arrow/branch/main/graph/badge.svg)](https://codecov.io/gh/GeorgeLeePatterson/clickhouse-arrow)

A high-performance, async Rust client for `ClickHouse` with native Arrow integration. Designed to be faster and more memory-efficient than existing alternatives.

## Why clickhouse-arrow?

- **Performance** – Zero-copy where possible, SIMD-accelerated serialisation
- **Arrow Native** – First-class RecordBatch support for data interchange
- **Async/Await** – Modern async API on Tokio
- **Compression** – LZ4 (default) and ZSTD support
- **Cloud Ready** – ClickHouse Cloud compatible
- **Type Safe** – Compile-time checking with `#[derive(Row)]` macro

## Performance

### Benchmarks

**System:** AMD Ryzen 9 5900X (20 vCPUs), 32GB RAM, ClickHouse 25.12.3, Rust 1.89.0 with LTO.

#### Insert Performance

| Rows | No Compression | LZ4     |
|------|----------------|---------|
| 10k  | 5.6ms          | 6.1ms   |
| 100k | 39.5ms         | 47.7ms  |
| 200k | 87.8ms         | 106.8ms |
| 300k | 140.6ms        | 168.0ms |
| 400k | 200.5ms        | 209.0ms |

#### Query Performance

| Rows | Time    |
|------|---------|
| 10k  | 5.7ms   |
| 100k | 24.5ms  |
| 200k | 36.7ms  |
| 300k | 46.5ms  |
| 400k | 60.4ms  |

#### Internal SIMD Optimisations

| Operation             | Speedup   | Description                                          |
|-----------------------|-----------|------------------------------------------------------|
| Null bitmap expansion | **~2.3x** | AVX2 SIMD for Arrow→ClickHouse null conversion       |
| Buffer allocation     | **~21%**  | Size-tiered buffer pool reduces allocator pressure   |
| Combined workload     | **~1.5x** | Overall serialisation (11.0µs → 7.1µs for 10k rows)  |

**Key insights:**

- **LZ4 compression** adds ~10-20% overhead for CPU-bound inserts (beneficial over network)
- **Zero-copy** Arrow integration minimises memory allocations
- **Linear scaling** with dataset size
- **Connection pooling** (`pool` feature) recommended for sustained throughput

### Running Benchmarks

```bash
# Run all benchmarks with LTO optimizations
just bench-lto

# Run specific benchmark
just bench-one insert

# View detailed results
open target/criterion/report/index.html
```

*Benchmarks use realistic workloads with mixed data types (integers, strings, timestamps, arrays) representative of typical `ClickHouse` usage patterns. To benchmark with scalar data only, similar to the benchmarks in `ch-go`, use the `scalar` bench*

## Details

The crate supports two "modes" of operation:

### `ArrowFormat`

Support allowing interoperability with [arrow](https://docs.rs/arrow/latest/arrow/).

### `NativeFormat`

Uses internal types and custom traits if a dependency on arrow is not required.

### `CreateOptions`, `SchemaConversions`, and Schemas

#### Creating Tables from Arrow Schemas

`clickhouse-arrow` provides powerful DDL capabilities through `CreateOptions`, allowing you to create `ClickHouse` tables directly from Arrow schemas:

```rust,ignore
use clickhouse_arrow::{Client, ArrowFormat, CreateOptions};
use arrow::datatypes::{Schema, Field, DataType};

// Define your Arrow schema
let schema = Schema::new(vec![
    Field::new("id", DataType::UInt64, false),
    Field::new("name", DataType::Utf8, false),
    Field::new("status", DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)), false),
]);

// Configure table creation
let options = CreateOptions::new("MergeTree")
    .with_order_by(&["id".to_string()])
    .with_partition_by("toYYYYMM(created_at)")
    .with_setting("index_granularity", 8192);

// Create the table
client.create_table(None, "my_table", &schema, &options, None).await?;
```

#### Schema Conversions for Type Control

`SchemaConversions` (type alias for `HashMap<String, Type>`) provides fine-grained control over Arrow-to-ClickHouse type mappings. This is especially important for:

1. **Converting Dictionary → Enum**: By default, Arrow Dictionary types map to `LowCardinality(String)`. Use `SchemaConversions` to map them to `Enum8` or `Enum16` instead:

```rust,ignore
use clickhouse_arrow::{Type, CreateOptions};
use std::collections::HashMap;

let schema_conversions = HashMap::from([
    // Convert status column from Dictionary to Enum8
    ("status".to_string(), Type::Enum8(vec![
        ("active".to_string(), 0),
        ("inactive".to_string(), 1),
        ("pending".to_string(), 2),
    ])),
    // Convert category to Enum16 for larger enums
    ("category".to_string(), Type::Enum16(vec![
        ("electronics".to_string(), 0),
        ("clothing".to_string(), 1),
        // ... up to 65k values
    ])),
]);

let options = CreateOptions::new("MergeTree")
    .with_order_by(&["id".to_string()])
    .with_schema_conversions(schema_conversions);
```

2. **Geo Types**: Preserve geographic types during conversion
3. **Date Types**: Choose between `Date` and `Date32`
4. **Custom Type Mappings**: Override any default type conversion

#### Field Naming Constants

When working with complex Arrow types, use these constants to ensure compatibility:

```rust,ignore
use clickhouse_arrow::arrow::types::*;

// For List types - inner field is named "item"
let list_field = Field::new("data", DataType::List(
    Arc::new(Field::new(LIST_ITEM_FIELD_NAME, DataType::Int32, true))
), true);

// For Struct/Tuple types - fields are named "field_0", "field_1", etc.
let tuple_fields = vec![
    Field::new(format!("{}{}", TUPLE_FIELD_NAME_PREFIX, 0), DataType::Int32, false),
    Field::new(format!("{}{}", TUPLE_FIELD_NAME_PREFIX, 1), DataType::Utf8, false),
];

// For Map types - uses specific field names
let map_type = DataType::Map(
    Arc::new(Field::new(MAP_FIELD_NAME, DataType::Struct(
        vec![
            Field::new(STRUCT_KEY_FIELD_NAME, DataType::Utf8, false),
            Field::new(STRUCT_VALUE_FIELD_NAME, DataType::Int32, true),
        ].into()
    ), false)),
    false
);
```

These constants ensure your Arrow schemas align with `ClickHouse`'s expectations and maintain compatibility with arrow-rs conventions.

## Queries

### Query Settings

The `clickhouse_arrow::Settings` type allows configuring `ClickHouse` query settings. You can import it directly:

```rust
use clickhouse_arrow::Settings;
// or via prelude
use clickhouse_arrow::prelude::*;
```

Refer to the settings module documentation for details and examples.

## Arrow Round-Trip

There are cases where a round trip may deserialize a different type by schema or array than the schema and array you used to create the table.

 will try to maintain an accurate and updated list as they occur. In addition, when possible, I will provide options or other functionality to alter this behavior.

#### `(String|Binary)View`/`Large(List|String|Binary)` variations are normalized.
- **Behavior**: `ClickHouse` does not make the same distinction between `Utf8`, `Utf8View`, or `LargeUtf8`. All of these are mapped to either `Type::Binary` (the default, see above) or `Type::String`
- **Option**: None
- **Default**: Unsupported
- **Impact**: When deserializing from `ClickHouse`, manual modification will be necessary to use these data types.

#### `Utf8` -> `Binary`
- **Behavior**: By default, `Type::String`/`DataType::Utf8` will be represented as Binary.
- **Option**: `strings_as_strings` (default: `false`).
- **Default**: Disabled (`false`).
- **Impact**: Set to `true` to strip map `Type::String` -> `DataType::Utf8`. Binary tends to be more efficient to work with in high throughput scenarios

#### Nullable `Array`s
- **Behavior**: `ClickHouse` does not allow `Nullable(Array(...))`, but insertion with non-null data is allowed by default. To modify this behavior, set `array_nullable_error` to `true`.
- **Option**: `array_nullable_error` (default: `false`).
- **Default**: Disabled (`false`).
- **Impact**: Enables flexible insertion but may cause schema mismatches if nulls are present.

#### `LowCardinality(Nullable(...))` vs `Nullable(LowCardinality(...))`
- **Behavior**: Like arrays mentioned above, `ClickHouse` does not allow nullable low cardinality. The default behavior is to push down the nullability.
- **Option**: `low_cardinality_nullable_error` (default: `false`).
- **Default**: Disabled (`false`).
- **Impact**: Enables flexible insertion but may cause schema mismatches if nulls are present.

#### `Enum8`/`Enum16` vs. `LowCardinality`
- **Behavior**: Arrow `Dictionary` types map to `LowCardinality`, but `ClickHouse` `Enum` types may also map to `Dictionary`, altering the type on round-trip.
- **Option**: No options available rather provide hash maps for either `enum_i8` and/or `enum_i16` for `CreateOptions` during schema creation.
- **Impact**: The default behavior will ignore enums when starting from arrow.

> [!NOTE]
> For examples of these cases, refer to the tests in the module [arrow::types](src/arrow/types.rs)

> [!NOTE]
> The configuration for the options above can be found in [options](src/client/options.rs)

> [!NOTE]
> For a builder of create options use during schema creation (eg `Engine`, `Order By`, `Enum8` and `Enum16` lookups), refer to [CreateOptions](src/schema.rs)
