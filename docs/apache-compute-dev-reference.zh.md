# Apache Arrow Compute 规范与实现梳理（开发参考）

本文用于 `zarrow-compute` 开发时的对齐参考，聚焦两件事：

- Apache Arrow Compute 在官方文档中的“行为规范”
- Apache Arrow Compute 在 C++ 侧的“实现结构”

基线说明（访问时间）：2026-04-21。  
主要参考官方文档与源码入口（见文末链接）。

## 1. 先厘清“规范”的边界

在 Arrow 体系里，**跨语言强规范**主要是内存格式 / IPC / C Data Interface。  
Compute 更多是“实现层 API 规范 + 函数语义约定”，核心载体是：

- C++ User Guide（Compute Functions）
- C++ API Reference（compute 模块类型与调用约定）
- Python API（函数目录、options、行为说明）

因此，开发 `zarrow-compute` 时，应把“规范对齐”理解为：

1. 输入输出形状与广播规则一致  
2. null / overflow / cast / options 等语义一致  
3. 错误分类可映射到 Arrow 语义

## 2. Apache Compute 关键规范点

## 2.1 函数、kernel、registry

官方定义：Function 由一个或多个 kernel 组成，按输入类型做 dispatch；函数存在全局 `FunctionRegistry` 中并按名称查找。

对开发的直接含义：

- 不要把一个函数写死为单类型实现
- 统一走“签名匹配 + 派发”模型，便于后续补齐更多类型

## 2.2 初始化与可用函数集

C++ User Guide 明确要求 `arrow::compute::Initialize()` 以注册完整函数集；否则只有核心子集可用。

对开发的直接含义：

- 不应假设“所有函数默认已注册”
- 在下游库中应提供明确的 `register*` 入口，避免隐式行为

## 2.3 输入形状（Datum）与调用方式

官方以 `Datum` 表示 compute 输入输出；可承载 `Scalar / Array / ChunkedArray / RecordBatch / Table`。  
调用可走：

- `CallFunction(name, args, options, ctx)`（通用）
- 具体函数入口（如 `Add()`）

同时文档注明：

- `CallFunction` 会处理 kernel dispatch、参数检查、chunked 迭代与输出封装
- Grouped Aggregations 不能通过通用 `CallFunction` 直接调用

## 2.4 隐式类型转换与 common numeric type

官方定义了 numeric common type 规则（如 `uint32 + int32 -> int64`）。  
这影响比较、算术等 kernel 的自动提升行为。

对开发的直接含义：

- 对 binary numeric kernel，要么实现“显式同型检查”，要么实现“统一提升后执行”
- 需要把“提升失败”映射为可诊断错误

## 2.5 标量函数广播规则

官方对 element-wise（scalar）函数给出了广播语义：

- unary: scalar->scalar, array->array
- binary:
  - `(scalar, scalar) -> scalar`
  - `(array, array) -> array`（长度必须一致）
  - `(scalar, array)` / `(array, scalar)` -> array（scalar 按 N 广播）

## 2.6 null 语义与 Kleene 逻辑

官方在逻辑函数中明确：

- 默认逻辑函数：任一输入为 null 时通常输出 null
- `_kleene` 变体使用 Kleene 三值逻辑（与 SQL 类似）

这点对于 `and/or/and_not` 等逻辑函数非常关键。

## 2.7 overflow 与 checked 变体

官方在算术与累计类函数中均强调：

- 默认版本通常不检查溢出（可能 wrap）
- `_checked` 变体在溢出时报 `Invalid` / 异常

对开发的直接含义：

- 需要支持“策略切换”：wrapping vs checked
- 命名和行为建议与 Arrow 一致（`foo` / `foo_checked`）

## 2.8 options 是语义的一部分

官方 API 把 options 单独建模（`FunctionOptions` 及大量具体 options 类型）。  
很多函数需要 options 才能确定完整行为（如 cast/count/round 等）。

对开发的直接含义：

- options 不是附属参数，而是 kernel 语义的一部分
- 需要做 options 类型检查与默认值策略

## 3. Apache Compute 的实现结构（C++视角）

## 3.1 Function 分型

`Function::Kind`（源码/文档）包含：

- `SCALAR`
- `VECTOR`
- `SCALAR_AGGREGATE`
- `HASH_AGGREGATE`
- `META`

这和我们在下游做分类注册时是一致的结构基础。

## 3.2 Kernel dispatch + one-shot 调用

`CallFunction(...)` 是 one-shot 入口，负责：

- 选择匹配 kernel
- 参数检查
- chunked 迭代
- 结果打包

下游实现一般沿用这一模型：  
“registry 负责查找，context 负责执行配置，kernel 负责纯计算”。

## 3.3 与 Acero 的关系

官方文档说明：对于复杂计算，连续直接调用 compute 函数在内存/计算上可能不可行；  
Arrow C++ 提供了 Acero（流式查询执行引擎）来组织大规模执行计划。

可把它理解为：

- Compute 函数：算子语义单元
- Acero：把算子组合成 streaming plan 的执行层

## 4. 对 `zarrow-compute` 的落地映射

结合当前仓库实现（`src/kernels.zig`）：

- 已对齐：`FunctionRegistry + Datum + Options + kernel dispatch` 框架思路
- 已对齐：binary 广播、chunked 对齐迭代、null 传播
- 已对齐：overflow/cast 错误语义（`Overflow` / `DivideByZero` / `InvalidCast`）

当前已实现函数：

- `add_i64`
- `divide_i64`
- `cast_i64_to_i32`
- `count_rows`（含 aggregate lifecycle）

## 5. 新增 kernel 的建议清单（实操）

开发一个新 kernel 时建议按这个顺序：

1. 定义函数类别：`vector` / `aggregate`  
2. 定义签名：arity + type_check + options_check + result_type_fn  
3. 明确 null 语义：默认传播或 Kleene/skip_nulls  
4. 明确 overflow/cast 策略：默认/checked 行为  
5. 明确输入形状：array/chunked/scalar 是否支持广播  
6. 编写单测：正常路径 + 边界 + 错误路径 + chunked 路径  
7. 和上游函数语义做逐项对照（命名、options、错误）

## 6. 参考链接（官方）

- C++ User Guide / Compute Functions  
  <https://arrow.apache.org/docs/cpp/compute.html>
- C++ API / Compute  
  <https://arrow.apache.org/docs/cpp/api/compute.html>
- Python API / Compute  
  <https://arrow.apache.org/docs/python/api/compute.html>
- C++ Acero（流式执行引擎）  
  <https://arrow.apache.org/docs/cpp/acero.html>
- Arrow 源码（compute function 定义）  
  <https://github.com/apache/arrow/blob/main/cpp/src/arrow/compute/function.h>

