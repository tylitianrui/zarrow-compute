# zarrow-compute

`zarrow-compute` is a Zig downstream library for Apache Arrow Compute.  
It reuses the compute framework layer (`zarrow-core`) from [`tylitianrui/zarrow`](https://github.com/tylitianrui/zarrow) and implements concrete compute kernels in this repository.

## Implemented Kernels

- `add_i64` (vector)
- `filter` (vector, supports `null/bool/fixed-width/string/binary` value types + `bool` predicate, `Options.filter`)
- `drop_null` (vector, supports the same value type subset as `filter`, `Options.none`)
- `take` / `array_take` (vector, nullable integer indices, `Options.none`)
- `indices_nonzero` (vector, supports `bool/int32/int64`, `Options.none`)
- `is_null` (vector, null bitmap to bool mask, `Options.none`)
- `is_valid` (vector, inverse null bitmap to bool mask, `Options.none`)
- `true_unless_null` (vector, true for non-null values and false for null values, `Options.none`)
- `if_else` (vector, supports `null/bool/fixed-width/string/binary/list/large_list/struct` subset, `Options.none`)
- `coalesce` (vector, variadic, select first non-null value, supports `null/bool/fixed-width/string/binary/list/large_list/struct` subset, `Options.none`)
- `choose` (vector, variadic, select value by 0-based index, supports `null/bool/fixed-width/string/binary/list/large_list/struct` subset, `Options.none`)
- `case_when` (vector, Arrow-native `struct<bool...> + *cases` with optional else, supports `null/bool/fixed-width/string/binary/list/large_list/struct` subset, `Options.none`)
- `fill_null` / `fill_null_forward` / `fill_null_backward` (vector, `Options.none`)
- `equal` / `not_equal` / `less` / `less_equal` / `greater` / `greater_equal` (vector, current `int64` subset, `Options.none`)
- `invert` / `and_` / `or_` / `and_kleene` / `or_kleene` (vector, `bool`, `Options.none`)
- `subtract_i64` (vector)
- `divide_i64` (vector)
- `multiply_i64` (vector)
- `cast_i64_to_i32` (vector)
- `cast` (vector, current numeric/bool subset, `Options.cast`)
- `count_rows` (aggregate with stateful lifecycle)
- `count` / `sum` / `min` / `max` / `mean` (aggregate, current `int64` subset)

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
filter_fixed_size_list => [[1, 2], null, [5, 6]]
drop_null_fixed_size_list => [[11, 12], [15, 16], [17, 18]]
if_else_fixed_size_list => [[1, 2], null, null, [7, 8]]
coalesce_fixed_size_list => [[11, 12], [23, 24], [15, 16], [17, 18]]
choose_fixed_size_list => [[1, 2], null, null, [27, 28]]
case_when_fixed_size_list => [[1, 2], null, null, [17, 18]]
count_rows => 3
```

## Documentation

- Chinese README: [docs/README.zh.md](docs/README.zh.md)
- Apache Compute reference (ZH): [docs/apache-compute-dev-reference.zh.md](docs/apache-compute-dev-reference.zh.md)
- Upstream Compute API doc (no local download required):  
  <https://github.com/tylitianrui/zarrow/blob/master/docs/compute-api-zh.md>
