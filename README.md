# zarrow-compute

`zarrow-compute` is a Zig downstream library for Apache Arrow Compute.  
It reuses the compute framework layer (`zarrow-core`) from [`tylitianrui/zarrow`](https://github.com/tylitianrui/zarrow) and implements concrete compute kernels in this repository.

## Implemented Kernels

- `add_i64` (vector)
- `filter` (vector, supports `null/bool/fixed-width/string/binary` value types + `bool` predicate, `Options.filter`)
- `drop_null` (vector, supports the same value type subset as `filter`, `Options.none`)
- `is_null` (vector, null bitmap to bool mask, `Options.none`)
- `is_valid` (vector, inverse null bitmap to bool mask, `Options.none`)
- `true_unless_null` (vector, true for non-null values and false for null values, `Options.none`)
- `if_else` (vector, supports `null/bool/fixed-width/string/binary/struct` subset, `Options.none`)
- `coalesce` (vector, variadic, select first non-null value, `Options.none`)
- `choose` (vector, variadic, select value by 0-based index, `Options.none`)
- `case_when` (vector, Arrow-native `struct<bool...> + *cases` with optional else, `Options.none`)
- `subtract_i64` (vector)
- `divide_i64` (vector)
- `multiply_i64` (vector)
- `cast_i64_to_i32` (vector)
- `count_rows` (aggregate with stateful lifecycle)

## Quick Start

```bash
zig build test
zig build examples
```

To sync `zarrow` to latest `master` before local builds:

```bash
./tools/update_zarrow_master_hash.sh
```

Example output (`zig build examples`):

```text
add_i64 => [6, null, 8]
subtract_i64 => [-4, null, -2]
multiply_i64 => [5, null, 15]
filter => [1, null, null]
drop_null => [1, 3]
is_null => [false, true, false]
is_valid => [true, false, true]
true_unless_null => [true, false, true]
if_else => [1, null, 10]
coalesce => [1, 7, 3]
choose => [1, null, null]
case_when => [10, null, null]
count_rows => 3
```

## Documentation

- Chinese README: [docs/README.zh.md](docs/README.zh.md)
- Apache Compute reference (ZH): [docs/apache-compute-dev-reference.zh.md](docs/apache-compute-dev-reference.zh.md)
- Upstream Compute API doc (no local download required):  
  <https://github.com/tylitianrui/zarrow/blob/master/docs/compute-api-zh.md>
