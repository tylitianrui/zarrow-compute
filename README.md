# zarrow-compute

`zarrow-compute` 是一个面向 Apache Arrow Compute 的 Zig 下游库。  
它引用并复用 [`tylitianrui/zarrow`](https://github.com/tylitianrui/zarrow) 提供的 compute 框架层（`zarrow-core`），在本仓库实现具体 kernels。

## 当前已实现

- `add_i64`（vector）
- `divide_i64`（vector）
- `cast_i64_to_i32`（vector）
- `count_rows`（aggregate，含 stateful lifecycle）

## 快速开始

```bash
zig build test
zig build example-basic
```

示例输出（`zig build example-basic`）：

```text
add_i64 => [6, null, 8]
count_rows => 3
```
