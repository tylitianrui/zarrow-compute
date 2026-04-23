# Apache Arrow Compute 全量对齐 TODO（zarrow-compute）

基线日期：2026-04-23  
对齐基线：Apache Arrow `pyarrow.compute` API v24.0.0（函数目录）  
范围说明：本清单覆盖 Arrow Compute 的函数能力；`zarrow-compute` 当前是下游 kernel 仓库，按“先核心类型、后全类型”推进。

## 0. 当前实现状态（本仓库）

- 已实现（但为类型子集实现，不代表该函数全类型完成）：
  - `add`（`add_i64`）
  - `subtract`（`subtract_i64`）
  - `multiply`（`multiply_i64`）
  - `divide`（`divide_i64`）
  - `cast`（当前仅 `int64 -> int32`）
  - `filter`（当前已覆盖 `null/bool/定长类型/string/binary` 值类型；嵌套/字典等复杂类型待补齐）
  - `drop_null`（当前已覆盖 `null/bool/定长类型/string/binary` 值类型；复杂类型待补齐）
  - `is_null`（当前支持 array/chunked 输入，输出 `bool` 掩码）
  - `is_valid`（当前支持 array/chunked 输入，输出 `bool` 掩码）
- 非 Arrow 标准函数（仓库自定义）：
  - `count_rows`

## 1. Aggregations

- [ ] `all`
- [ ] `any`
- [ ] `approximate_median`
- [ ] `count`
- [ ] `count_distinct`
- [ ] `first`
- [ ] `first_last`
- [ ] `index`
- [ ] `kurtosis`
- [ ] `last`
- [ ] `max`
- [ ] `mean`
- [ ] `min`
- [ ] `min_max`
- [ ] `mode`
- [ ] `pivot_wider`
- [ ] `product`
- [ ] `quantile`
- [ ] `skew`
- [ ] `stddev`
- [ ] `sum`
- [ ] `tdigest`
- [ ] `variance`

## 2. Cumulative Functions

- [ ] `cumulative_sum`
- [ ] `cumulative_sum_checked`
- [ ] `cumulative_prod`
- [ ] `cumulative_prod_checked`
- [ ] `cumulative_max`
- [ ] `cumulative_mean`
- [ ] `cumulative_min`

## 3. Arithmetic Functions

- [ ] `abs`
- [ ] `abs_checked`
- [ ] `add`（当前仅 `int64` 子集）
- [ ] `add_checked`
- [ ] `divide`（当前仅 `int64` 子集）
- [ ] `divide_checked`
- [ ] `exp`
- [ ] `expm1`
- [ ] `multiply`（当前仅 `int64` 子集）
- [ ] `multiply_checked`
- [ ] `negate`
- [ ] `negate_checked`
- [ ] `power`
- [ ] `power_checked`
- [ ] `sign`
- [ ] `sqrt`
- [ ] `sqrt_checked`
- [ ] `subtract`（当前仅 `int64` 子集）
- [ ] `subtract_checked`

## 4. Bit-wise Functions

- [ ] `bit_wise_and`
- [ ] `bit_wise_not`
- [ ] `bit_wise_or`
- [ ] `bit_wise_xor`
- [ ] `shift_left`
- [ ] `shift_left_checked`
- [ ] `shift_right`
- [ ] `shift_right_checked`

## 5. Rounding Functions

- [ ] `ceil`
- [ ] `floor`
- [ ] `round`
- [ ] `round_binary`
- [ ] `round_to_multiple`
- [ ] `trunc`

## 6. Logarithmic Functions

- [ ] `ln`
- [ ] `ln_checked`
- [ ] `log10`
- [ ] `log10_checked`
- [ ] `log1p`
- [ ] `log1p_checked`
- [ ] `log2`
- [ ] `log2_checked`
- [ ] `logb`
- [ ] `logb_checked`

## 7. Trigonometric Functions

- [ ] `acos`
- [ ] `acos_checked`
- [ ] `asin`
- [ ] `asin_checked`
- [ ] `atan`
- [ ] `atan2`
- [ ] `cos`
- [ ] `cos_checked`
- [ ] `sin`
- [ ] `sin_checked`
- [ ] `tan`
- [ ] `tan_checked`

## 8. Hyperbolic Trigonometric Functions

- [ ] `acosh`
- [ ] `acosh_checked`
- [ ] `asinh`
- [ ] `atanh`
- [ ] `atanh_checked`
- [ ] `cosh`
- [ ] `sinh`
- [ ] `tanh`

## 9. Comparison Functions

- [ ] `equal`
- [ ] `greater`
- [ ] `greater_equal`
- [ ] `less`
- [ ] `less_equal`
- [ ] `not_equal`
- [ ] `max_element_wise`
- [ ] `min_element_wise`

## 10. Logical Functions

- [ ] `and_`
- [ ] `and_kleene`
- [ ] `and_not`
- [ ] `and_not_kleene`
- [ ] `invert`
- [ ] `or_`
- [ ] `or_kleene`
- [ ] `xor`

## 11. String Predicates

- [ ] `ascii_is_alnum`
- [ ] `ascii_is_alpha`
- [ ] `ascii_is_decimal`
- [ ] `ascii_is_lower`
- [ ] `ascii_is_printable`
- [ ] `ascii_is_space`
- [ ] `ascii_is_upper`
- [ ] `utf8_is_alnum`
- [ ] `utf8_is_alpha`
- [ ] `utf8_is_decimal`
- [ ] `utf8_is_digit`
- [ ] `utf8_is_lower`
- [ ] `utf8_is_numeric`
- [ ] `utf8_is_printable`
- [ ] `utf8_is_space`
- [ ] `utf8_is_upper`
- [ ] `ascii_is_title`
- [ ] `utf8_is_title`
- [ ] `string_is_ascii`

## 12. String Transforms

- [ ] `ascii_capitalize`
- [ ] `ascii_lower`
- [ ] `ascii_reverse`
- [ ] `ascii_swapcase`
- [ ] `ascii_title`
- [ ] `ascii_upper`
- [ ] `binary_length`
- [ ] `binary_repeat`
- [ ] `binary_replace_slice`
- [ ] `binary_reverse`
- [ ] `replace_substring`
- [ ] `replace_substring_regex`
- [ ] `utf8_capitalize`
- [ ] `utf8_length`
- [ ] `utf8_lower`
- [ ] `utf8_normalize`
- [ ] `utf8_replace_slice`
- [ ] `utf8_reverse`
- [ ] `utf8_swapcase`
- [ ] `utf8_title`
- [ ] `utf8_upper`

## 13. String Padding

- [ ] `ascii_center`
- [ ] `ascii_lpad`
- [ ] `ascii_rpad`
- [ ] `utf8_center`
- [ ] `utf8_lpad`
- [ ] `utf8_rpad`
- [ ] `utf8_zero_fill`

## 14. String Trimming

- [ ] `ascii_ltrim`
- [ ] `ascii_ltrim_whitespace`
- [ ] `ascii_rtrim`
- [ ] `ascii_rtrim_whitespace`
- [ ] `ascii_trim`
- [ ] `ascii_trim_whitespace`
- [ ] `utf8_ltrim`
- [ ] `utf8_ltrim_whitespace`
- [ ] `utf8_rtrim`
- [ ] `utf8_rtrim_whitespace`
- [ ] `utf8_trim`
- [ ] `utf8_trim_whitespace`

## 15. String Splitting / Regex Extraction

- [ ] `ascii_split_whitespace`
- [ ] `split_pattern`
- [ ] `split_pattern_regex`
- [ ] `utf8_split_whitespace`
- [ ] `extract_regex`
- [ ] `extract_regex_span`

## 16. String Component / Search / Join

- [ ] `binary_join`
- [ ] `binary_join_element_wise`
- [ ] `binary_slice`
- [ ] `utf8_slice_codeunits`
- [ ] `count_substring`
- [ ] `count_substring_regex`
- [ ] `ends_with`
- [ ] `find_substring`
- [ ] `find_substring_regex`
- [ ] `index_in`
- [ ] `is_in`
- [ ] `match_like`
- [ ] `match_substring`
- [ ] `match_substring_regex`
- [ ] `starts_with`

## 17. Null / Conditional / Selection Core

- [ ] `indices_nonzero`
- [ ] `is_finite`
- [ ] `is_inf`
- [ ] `is_nan`
- [ ] `is_null`（当前已实现 array/chunked 输入基础版）
- [ ] `is_valid`（当前已实现 array/chunked 输入基础版）
- [ ] `true_unless_null`（当前已实现 array/chunked 输入基础版）
- [ ] `case_when`
- [ ] `choose`
- [ ] `coalesce`
- [ ] `if_else`（当前已实现第一版：`fixed-width + string/binary` 子集，`Options.none`，condition null -> output null）
- [ ] `cast`（当前仅 `int64 -> int32` 子集）

## 18. Temporal Rounding / Encoding / Format

- [ ] `ceil_temporal`
- [ ] `floor_temporal`
- [ ] `round_temporal`
- [ ] `run_end_decode`
- [ ] `run_end_encode`
- [ ] `strftime`
- [ ] `strptime`

## 19. Temporal Component Extraction

- [ ] `day`
- [ ] `day_of_week`
- [ ] `day_of_year`
- [ ] `hour`
- [ ] `is_dst`
- [ ] `is_leap_year`
- [ ] `iso_week`
- [ ] `iso_year`
- [ ] `iso_calendar`
- [ ] `microsecond`
- [ ] `millisecond`
- [ ] `minute`
- [ ] `month`
- [ ] `nanosecond`
- [ ] `quarter`
- [ ] `second`
- [ ] `subsecond`
- [ ] `us_week`
- [ ] `us_year`
- [ ] `week`
- [ ] `year`
- [ ] `year_month_day`

## 20. Temporal Difference / Timezone

- [ ] `day_time_interval_between`
- [ ] `days_between`
- [ ] `hours_between`
- [ ] `microseconds_between`
- [ ] `milliseconds_between`
- [ ] `minutes_between`
- [ ] `month_day_nano_interval_between`
- [ ] `month_interval_between`
- [ ] `nanoseconds_between`
- [ ] `quarters_between`
- [ ] `seconds_between`
- [ ] `weeks_between`
- [ ] `years_between`
- [ ] `assume_timezone`
- [ ] `local_timestamp`

## 21. Random / Dictionary / Distinct

- [ ] `random`
- [ ] `dictionary_decode`
- [ ] `dictionary_encode`
- [ ] `unique`
- [ ] `value_counts`

## 22. Filter / Take / Permutation / Scatter

- [ ] `array_filter`
- [ ] `array_take`
- [ ] `drop_null`（当前已覆盖 `null/bool/定长类型/string/binary` 子集，需补齐复杂类型与更高阶输入语义）
- [ ] `filter`（当前已覆盖 `null/bool/定长类型/string/binary` 子集，需补齐复杂类型与更高阶输入语义）
- [ ] `inverse_permutation`
- [ ] `scatter`
- [ ] `take`

## 23. Sorting / Ranking / TopK

- [ ] `array_sort_indices`
- [ ] `bottom_k_unstable`
- [ ] `partition_nth_indices`
- [ ] `rank`
- [ ] `rank_normal`
- [ ] `rank_quantile`
- [ ] `select_k_unstable`
- [ ] `sort_indices`
- [ ] `top_k_unstable`
- [ ] `winsorize`

## 24. Null Filling

- [ ] `fill_null`
- [ ] `fill_null_backward`
- [ ] `fill_null_forward`

## 25. Nested Types / Struct / Mask

- [ ] `list_element`
- [ ] `list_flatten`
- [ ] `list_parent_indices`
- [ ] `list_slice`
- [ ] `list_value_length`
- [ ] `make_struct`
- [ ] `map_lookup`
- [ ] `replace_with_mask`
- [ ] `struct_field`

## 26. Pairwise

- [ ] `pairwise_diff`
- [ ] `pairwise_diff_checked`

## 27. Compute API / UDF / Registry 能力

- [ ] `call_function`
- [ ] `call_tabular_function`
- [ ] `get_function`
- [ ] `list_functions`
- [ ] `register_aggregate_function`
- [ ] `register_scalar_function`
- [ ] `register_tabular_function`
- [ ] `register_vector_function`
- [ ] `UdfContext`
- [ ] `field`
- [ ] `scalar`

## 28. 验收规则（每个函数都要满足）

- [ ] 行为对齐：与 `pyarrow.compute.<fn>` 在同输入下结果一致（或明确列出兼容差异）。
- [ ] 形态对齐：覆盖 `scalar/array/chunked` 至少两种组合。
- [ ] null 语义：默认、Kleene、`skip_nulls`、`drop_nulls` 等行为可测试。
- [ ] options 对齐：Arrow 对应 options 字段均有校验与默认值。
- [ ] 错误语义：`InvalidInput/InvalidOptions/Overflow/InvalidCast` 等映射稳定。
- [ ] CI 回归：Zig `0.15.1` 与 `0.15.2` 双版本全绿。

## 29. 参考

- Arrow Python Compute API（全量目录）：<https://arrow.apache.org/docs/python/api/compute.html>
- Arrow C++ Compute：<https://arrow.apache.org/docs/cpp/compute.html>
- 本仓库开发参考：[`apache-compute-dev-reference.zh.md`](./apache-compute-dev-reference.zh.md)
