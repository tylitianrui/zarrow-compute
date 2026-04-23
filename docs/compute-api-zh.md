# Compute API（zarrow-compute）说明

本文档基于上游 `zarrow` 的 Compute API 文档整理，面向当前仓库的下游 kernel 开发与使用。

- 上游参考：<https://github.com/tylitianrui/zarrow/blob/master/docs/compute-api-zh.md>
- 当前仓库核心实现：`src/kernels.zig`

## 1. 目标与边界

- `zarrow-core` 提供 compute 框架层：注册、调度、类型检查、结果类型推导、执行上下文、聚合生命周期。
- `zarrow-compute` 负责实现具体计算 kernel（例如 `add/div/cast/count_rows`）。

## 2. 当前已实现的 kernels

- `add_i64`（vector）
- `filter`（vector，`int64` values + `bool` predicate）
- `subtract_i64`（vector）
- `divide_i64`（vector）
- `multiply_i64`（vector）
- `cast_i64_to_i32`（vector）
- `count_rows`（aggregate，支持 lifecycle）

注册入口：

- `registerBaseKernels(registry: *compute.FunctionRegistry)`
- `registerCompatKernels(registry: *compute.FunctionRegistry)`（当前等价于 `registerBaseKernels`）

## 3. 核心类型（来自 zarrow-core）

- `FunctionRegistry`：管理函数和 kernel
- `KernelSignature`：arity/type/options 检查 + 结果类型推导
- `Kernel`：执行函数 + 可选的聚合生命周期
- `ExecContext`：执行上下文（allocator/config/registry）
- `Datum`：统一输入输出容器（`array` / `chunked` / `scalar`）

## 4. 类型安全 Options

可用 options 类型：

- `Options.none`
- `Options.cast`
- `Options.arithmetic`
- `Options.filter`
- `Options.custom`

本仓库中：

- `add_i64`、`subtract_i64`、`divide_i64` 仅接受 `Options.arithmetic`
- `multiply_i64` 仅接受 `Options.arithmetic`
- `filter` 仅接受 `Options.filter`（`drop_nulls=true` 时丢弃 predicate null；`drop_nulls=false` 时输出 null）
- `cast_i64_to_i32` 仅接受 `Options.cast`
- `count_rows` 仅接受 `Options.none`

## 5. 错误语义

本仓库复用框架层错误模型：

- `InvalidArity`
- `InvalidInput`
- `InvalidOptions`
- `NoMatchingKernel`
- `Overflow`
- `DivideByZero`
- `InvalidCast`

并复用框架 helper：

- `compute.arithmeticDivI64(...)`（除零/溢出语义统一）
- `compute.intCastOrInvalidCast(...)`（cast 失败统一为 `InvalidCast`）

## 6. 数组执行 helper 的使用方式

为支持 null 传播、scalar broadcast、chunked 对齐，vector kernel 建议统一使用：

- `compute.inferBinaryExecLen(lhs, rhs)`
- `compute.BinaryExecChunkIterator`
- `compute.UnaryExecChunkIterator`
- `chunk.binaryNullAt(i)` / `chunk.unaryNullAt(i)`

这也是当前 `add_i64`、`filter`、`subtract_i64`、`divide_i64`、`multiply_i64`、`cast_i64_to_i32` 的实现方式。

## 7. 聚合生命周期（count_rows）

`count_rows` 同时支持：

- 直接调用：`ctx.invokeAggregate("count_rows", ...)`
- 生命周期调用：`ctx.beginAggregate(...) -> update/merge/finalize/deinit`

生命周期函数包括：

- `init`
- `update`
- `merge`
- `finalize`
- `deinit`

## 8. 最小示例

请运行：

```bash
zig build examples
```

示例会展示：

- `add_i64` / `subtract_i64` / `multiply_i64` 的 null 传播行为
- `count_rows` 的聚合结果

## 9. 测试

测试位于 `src/kernels.zig` 中，运行：

```bash
zig build test
```
