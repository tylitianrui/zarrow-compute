# zarrow-compute

`zarrow-compute` 是一个面向 Apache Arrow Compute 的 Zig 下游库。  
它引用并复用 [`tylitianrui/zarrow`](https://github.com/tylitianrui/zarrow) 提供的 compute 框架层（`zarrow-core`），在本仓库实现具体 kernels。

## 当前已实现

- `add_i64`（vector）
- `filter`（vector，支持 `null/bool/定长类型/string/binary` values + `bool` predicate，使用 `Options.filter`）
- `drop_null`（vector，支持与 `filter` 相同的值类型子集，使用 `Options.none`）
- `is_null`（vector，将 null 位图映射为 `bool` 掩码，使用 `Options.none`）
- `is_valid`（vector，`is_null` 的反向掩码，使用 `Options.none`）
- `true_unless_null`（vector，非 null 输出 `true`，null 输出 `false`，使用 `Options.none`）
- `if_else`（vector，第一版支持 `fixed-width + string/binary` 子集，使用 `Options.none`）
- `coalesce`（vector，可变参数，按行选择第一个非 null，使用 `Options.none`）
- `choose`（vector，可变参数，按 0-based 索引选择值，使用 `Options.none`）
- `case_when`（vector，主入口为 Arrow 原生 `struct<bool...> + *cases` 且支持可选 else，同时兼容 `cond,value` 可变参数形态，使用 `Options.none`）
- `subtract_i64`（vector）
- `divide_i64`（vector）
- `multiply_i64`（vector）
- `cast_i64_to_i32`（vector）
- `count_rows`（aggregate，含 stateful lifecycle）

## 快速开始

```bash
zig build test
zig build examples
```

示例输出（`zig build examples`）：

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

## 文档说明

- 上游 Compute API 文档（无需下载到本地）：  
  <https://github.com/tylitianrui/zarrow/blob/master/docs/compute-api-zh.md>
- 本仓库 Compute 说明：  
  [compute-api-zh.md](./compute-api-zh.md)
- Apache Arrow Compute 规范与实现梳理（开发参考）：  
  [apache-compute-dev-reference.zh.md](./apache-compute-dev-reference.zh.md)
