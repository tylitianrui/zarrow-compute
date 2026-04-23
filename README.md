# zarrow-compute

`zarrow-compute` is a Zig downstream library for Apache Arrow Compute.  
It reuses the compute framework layer (`zarrow-core`) from [`tylitianrui/zarrow`](https://github.com/tylitianrui/zarrow) and implements concrete compute kernels in this repository.

## Implemented Kernels

- `add_i64` (vector)
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

Example output (`zig build examples`):

```text
add_i64 => [6, null, 8]
subtract_i64 => [-4, null, -2]
multiply_i64 => [5, null, 15]
count_rows => 3
```

## Documentation

- Chinese README: [docs/README.zh.md](docs/README.zh.md)
- Apache Compute reference (ZH): [docs/apache-compute-dev-reference.zh.md](docs/apache-compute-dev-reference.zh.md)
- Upstream Compute API doc (no local download required):  
  <https://github.com/tylitianrui/zarrow/blob/master/docs/compute-api-zh.md>
