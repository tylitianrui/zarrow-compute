const support = @import("test_support.zig");

const std = support.std;
const zcore = support.zcore;
const compute = support.compute;
const registerBaseKernels = support.registerBaseKernels;
const registerCompatKernels = support.registerCompatKernels;

const makeInt64Array = support.makeInt64Array;
const makeInt32Array = support.makeInt32Array;
const makeFloat64Array = support.makeFloat64Array;
const makeBoolArray = support.makeBoolArray;
const makeNullArray = support.makeNullArray;
const makeStructI64BoolArray = support.makeStructI64BoolArray;
const makeStructListI32Array = support.makeStructListI32Array;
const makeStructBool2Array = support.makeStructBool2Array;
const makeStringArray = support.makeStringArray;
const makeLargeStringArray = support.makeLargeStringArray;
const makeBinaryArray = support.makeBinaryArray;
const makeStringViewArray = support.makeStringViewArray;
const makeBinaryViewArray = support.makeBinaryViewArray;
const makeFixedSizeBinaryArray = support.makeFixedSizeBinaryArray;
const makeListInt32Array = support.makeListInt32Array;
const makeListInt32ArrayWithLens = support.makeListInt32ArrayWithLens;
const makeLargeListInt32ArrayWithLens = support.makeLargeListInt32ArrayWithLens;
const makeFixedSizeListInt32Array = support.makeFixedSizeListInt32Array;
const makeNestedScalarDatum = support.makeNestedScalarDatum;

const expectInt64ArrayValues = support.expectInt64ArrayValues;
const expectInt32ArrayValues = support.expectInt32ArrayValues;
const expectFloat64ArrayValues = support.expectFloat64ArrayValues;
const expectBoolArrayValues = support.expectBoolArrayValues;
const expectFixedSizeListAllNullAligned = support.expectFixedSizeListAllNullAligned;

test "subtract_i64 supports null propagation and overflow behavior" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var lhs = try makeInt64Array(allocator, &[_]?i64{ 9, null, -3 });
    defer lhs.release();
    const args = [_]compute.Datum{
        compute.Datum.fromArray(lhs.retain()),
        compute.Datum.fromScalar(.{
            .data_type = .{ .int64 = {} },
            .value = .{ .i64 = 4 },
        }),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    var out = try ctx.invokeVector("subtract_i64", args[0..], .{ .arithmetic = .{} });
    defer out.release();
    try std.testing.expect(out.isArray());
    const view = zcore.Int64Array{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), view.len());
    try std.testing.expectEqual(@as(i64, 5), view.value(0));
    try std.testing.expect(view.isNull(1));
    try std.testing.expectEqual(@as(i64, -7), view.value(2));

    const overflow_args = [_]compute.Datum{
        compute.Datum.fromScalar(.{
            .data_type = .{ .int64 = {} },
            .value = .{ .i64 = std.math.minInt(i64) },
        }),
        compute.Datum.fromScalar(.{
            .data_type = .{ .int64 = {} },
            .value = .{ .i64 = 1 },
        }),
    };
    try std.testing.expectError(
        error.Overflow,
        ctx.invokeVector("subtract_i64", overflow_args[0..], .{
            .arithmetic = .{ .check_overflow = true },
        }),
    );

    var wrapped = try ctx.invokeVector("subtract_i64", overflow_args[0..], .{
        .arithmetic = .{ .check_overflow = false },
    });
    defer wrapped.release();
    try std.testing.expect(wrapped.isArray());
    const wrapped_view = zcore.Int64Array{ .data = wrapped.array.data() };
    try std.testing.expectEqual(@as(usize, 1), wrapped_view.len());
    try std.testing.expectEqual(std.math.minInt(i64) -% @as(i64, 1), wrapped_view.value(0));
}

test "divide_i64 maps divide-by-zero behavior to options" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    const args = [_]compute.Datum{
        compute.Datum.fromScalar(.{
            .data_type = .{ .int64 = {} },
            .value = .{ .i64 = 42 },
        }),
        compute.Datum.fromScalar(.{
            .data_type = .{ .int64 = {} },
            .value = .{ .i64 = 0 },
        }),
    };

    try std.testing.expectError(
        error.DivideByZero,
        ctx.invokeVector("divide_i64", args[0..], .{ .arithmetic = .{} }),
    );

    var out = try ctx.invokeVector(
        "divide_i64",
        args[0..],
        .{ .arithmetic = .{ .check_overflow = true, .divide_by_zero_is_error = false } },
    );
    defer out.release();
    try std.testing.expect(out.isArray());

    const view = zcore.Int64Array{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 1), view.len());
    try std.testing.expectEqual(@as(i64, 0), view.value(0));
}

test "multiply_i64 supports null propagation and overflow behavior" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var lhs = try makeInt64Array(allocator, &[_]?i64{ 2, null, -3 });
    defer lhs.release();
    const args = [_]compute.Datum{
        compute.Datum.fromArray(lhs.retain()),
        compute.Datum.fromScalar(.{
            .data_type = .{ .int64 = {} },
            .value = .{ .i64 = 4 },
        }),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    var out = try ctx.invokeVector("multiply_i64", args[0..], .{ .arithmetic = .{} });
    defer out.release();
    try std.testing.expect(out.isArray());
    const view = zcore.Int64Array{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), view.len());
    try std.testing.expectEqual(@as(i64, 8), view.value(0));
    try std.testing.expect(view.isNull(1));
    try std.testing.expectEqual(@as(i64, -12), view.value(2));

    const overflow_args = [_]compute.Datum{
        compute.Datum.fromScalar(.{
            .data_type = .{ .int64 = {} },
            .value = .{ .i64 = std.math.maxInt(i64) },
        }),
        compute.Datum.fromScalar(.{
            .data_type = .{ .int64 = {} },
            .value = .{ .i64 = 2 },
        }),
    };
    try std.testing.expectError(
        error.Overflow,
        ctx.invokeVector("multiply_i64", overflow_args[0..], .{
            .arithmetic = .{ .check_overflow = true },
        }),
    );

    var wrapped = try ctx.invokeVector("multiply_i64", overflow_args[0..], .{
        .arithmetic = .{ .check_overflow = false },
    });
    defer wrapped.release();
    try std.testing.expect(wrapped.isArray());
    const wrapped_view = zcore.Int64Array{ .data = wrapped.array.data() };
    try std.testing.expectEqual(@as(usize, 1), wrapped_view.len());
    try std.testing.expectEqual(std.math.maxInt(i64) *% @as(i64, 2), wrapped_view.value(0));
}

test "cast_i64_to_i32 enforces safe cast mode" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    const max_i64 = compute.Datum.fromScalar(.{
        .data_type = .{ .int64 = {} },
        .value = .{ .i64 = std.math.maxInt(i64) },
    });
    const args = [_]compute.Datum{max_i64};

    try std.testing.expectError(
        error.InvalidCast,
        ctx.invokeVector("cast_i64_to_i32", args[0..], .{
            .cast = .{
                .safe = true,
                .to_type = .{ .int32 = {} },
            },
        }),
    );

    var out = try ctx.invokeVector("cast_i64_to_i32", args[0..], .{
        .cast = .{
            .safe = false,
            .to_type = .{ .int32 = {} },
        },
    });
    defer out.release();
    try std.testing.expect(out.isArray());

    const view = zcore.Int32Array{ .data = out.array.data() };
    const expected: i32 = @as(i32, @truncate(std.math.maxInt(i64)));
    try std.testing.expectEqual(@as(usize, 1), view.len());
    try std.testing.expectEqual(expected, view.value(0));
}

test "count_rows supports aggregate lifecycle merge/finalize" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var a1 = try makeInt64Array(allocator, &[_]?i64{ 1, 2, 3 });
    defer a1.release();
    var a2 = try makeInt64Array(allocator, &[_]?i64{ 10, 20 });
    defer a2.release();

    const args1 = [_]compute.Datum{compute.Datum.fromArray(a1.retain())};
    defer {
        var d = args1[0];
        d.release();
    }
    const args2 = [_]compute.Datum{compute.Datum.fromArray(a2.retain())};
    defer {
        var d = args2[0];
        d.release();
    }

    var direct = try ctx.invokeAggregate("count_rows", args1[0..], compute.Options.noneValue());
    defer direct.release();
    try std.testing.expect(direct.isScalar());
    try std.testing.expectEqual(@as(i64, 3), direct.scalar.value.i64);

    var s1 = try ctx.beginAggregate("count_rows", args1[0..], compute.Options.noneValue());
    defer s1.deinit();
    var s2 = try ctx.beginAggregate("count_rows", args2[0..], compute.Options.noneValue());
    defer s2.deinit();

    try s1.update(args1[0..]);
    try s2.update(args2[0..]);
    try s1.merge(&s2);

    var out = try s1.finalize();
    defer out.release();
    try std.testing.expect(out.isScalar());
    try std.testing.expectEqual(@as(i64, 5), out.scalar.value.i64);
}

test "take and array_take support nullable indices" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeInt64Array(allocator, &[_]?i64{ 10, null, 30 });
    defer values.release();
    var indices = try makeInt32Array(allocator, &[_]?i32{ 2, 1, null, 0 });
    defer indices.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(values.retain()),
        compute.Datum.fromArray(indices.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    var out_take = try ctx.invokeVector("take", args[0..], compute.Options.noneValue());
    defer out_take.release();
    try expectInt64ArrayValues(out_take, &[_]?i64{ 30, null, null, 10 });

    var out_array_take = try ctx.invokeVector("array_take", args[0..], compute.Options.noneValue());
    defer out_array_take.release();
    try expectInt64ArrayValues(out_array_take, &[_]?i64{ 30, null, null, 10 });
}

test "sort_indices and array_sort_indices support null ordering and chunked input" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeInt32Array(allocator, &[_]?i32{ 3, null, 1, 3, 2 });
    defer values.release();
    const args = [_]compute.Datum{compute.Datum.fromArray(values.retain())};
    defer {
        var d = args[0];
        d.release();
    }

    var out_sort = try ctx.invokeVector("sort_indices", args[0..], .{ .sort = .{ .stable = true } });
    defer out_sort.release();
    try expectInt64ArrayValues(out_sort, &[_]?i64{ 2, 4, 0, 3, 1 });

    var out_array_sort = try ctx.invokeVector("array_sort_indices", args[0..], .{ .sort = .{ .stable = true } });
    defer out_array_sort.release();
    try expectInt64ArrayValues(out_array_sort, &[_]?i64{ 2, 4, 0, 3, 1 });

    var c0 = try makeInt32Array(allocator, &[_]?i32{ 3, null });
    defer c0.release();
    var c1 = try makeInt32Array(allocator, &[_]?i32{ 1, 3, 2 });
    defer c1.release();
    var chunked = try compute.ChunkedArray.init(allocator, .{ .int32 = {} }, &[_]zcore.ArrayRef{ c0, c1 });
    defer chunked.release();

    const chunked_args = [_]compute.Datum{compute.Datum.fromChunked(chunked.retain())};
    defer {
        var d = chunked_args[0];
        d.release();
    }
    var out_chunked = try ctx.invokeVector("sort_indices", chunked_args[0..], .{ .sort = .{ .stable = true } });
    defer out_chunked.release();
    try expectInt64ArrayValues(out_chunked, &[_]?i64{ 2, 4, 0, 3, 1 });
}

test "sort_indices supports sort options and validates type/options via dispatch" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var numeric_values = try makeInt32Array(allocator, &[_]?i32{ 3, null, 1, 3, 2 });
    defer numeric_values.release();
    const numeric_args = [_]compute.Datum{compute.Datum.fromArray(numeric_values.retain())};
    defer {
        var d = numeric_args[0];
        d.release();
    }
    var desc_nulls_start = try ctx.invokeVector("sort_indices", numeric_args[0..], .{
        .sort = .{
            .order = .descending,
            .null_placement = .at_start,
            .stable = true,
        },
    });
    defer desc_nulls_start.release();
    try expectInt64ArrayValues(desc_nulls_start, &[_]?i64{ 1, 0, 3, 4, 2 });

    var floats = try makeFloat64Array(allocator, &[_]?f64{ 2.0, std.math.nan(f64), 1.0 });
    defer floats.release();
    const float_args = [_]compute.Datum{compute.Datum.fromArray(floats.retain())};
    defer {
        var d = float_args[0];
        d.release();
    }
    var nan_start = try ctx.invokeVector("sort_indices", float_args[0..], .{
        .sort = .{
            .order = .ascending,
            .nan_placement = .at_start,
            .stable = true,
        },
    });
    defer nan_start.release();
    try expectInt64ArrayValues(nan_start, &[_]?i64{ 1, 2, 0 });
    try std.testing.expectError(
        error.NoMatchingKernel,
        ctx.invokeVector("sort_indices", numeric_args[0..], compute.Options.noneValue()),
    );

    var bool_values = try makeBoolArray(allocator, &[_]?bool{ true, false });
    defer bool_values.release();
    const bool_args = [_]compute.Datum{compute.Datum.fromArray(bool_values.retain())};
    defer {
        var d = bool_args[0];
        d.release();
    }
    try std.testing.expectError(
        error.NoMatchingKernel,
        ctx.invokeVector("sort_indices", bool_args[0..], .{ .sort = .{} }),
    );

    var int_values = try makeInt32Array(allocator, &[_]?i32{ 1, 0 });
    defer int_values.release();
    const int_args = [_]compute.Datum{compute.Datum.fromArray(int_values.retain())};
    defer {
        var d = int_args[0];
        d.release();
    }
    try std.testing.expectError(
        error.NoMatchingKernel,
        ctx.invokeVector("sort_indices", int_args[0..], .{ .filter = .{} }),
    );
}

test "fill_null family supports directional fill semantics" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeInt64Array(allocator, &[_]?i64{ null, 2, null, 4, null });
    defer values.release();
    const fill_args = [_]compute.Datum{
        compute.Datum.fromArray(values.retain()),
        compute.Datum.fromScalar(.{
            .data_type = .{ .int64 = {} },
            .value = .{ .i64 = 9 },
        }),
    };
    defer {
        var d = fill_args[0];
        d.release();
    }
    defer {
        var d = fill_args[1];
        d.release();
    }

    var filled = try ctx.invokeVector("fill_null", fill_args[0..], compute.Options.noneValue());
    defer filled.release();
    try expectInt64ArrayValues(filled, &[_]?i64{ 9, 2, 9, 4, 9 });

    const unary_args = [_]compute.Datum{compute.Datum.fromArray(values.retain())};
    defer {
        var d = unary_args[0];
        d.release();
    }

    var forward = try ctx.invokeVector("fill_null_forward", unary_args[0..], compute.Options.noneValue());
    defer forward.release();
    try expectInt64ArrayValues(forward, &[_]?i64{ null, 2, 2, 4, 4 });

    var backward = try ctx.invokeVector("fill_null_backward", unary_args[0..], compute.Options.noneValue());
    defer backward.release();
    try expectInt64ArrayValues(backward, &[_]?i64{ 2, 2, 4, 4, null });
}

test "comparison and logical kernels support base semantics" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var lhs = try makeInt64Array(allocator, &[_]?i64{ 1, null, 3 });
    defer lhs.release();
    var rhs = try makeInt64Array(allocator, &[_]?i64{ 1, 2, 4 });
    defer rhs.release();

    const cmp_args = [_]compute.Datum{
        compute.Datum.fromArray(lhs.retain()),
        compute.Datum.fromArray(rhs.retain()),
    };
    defer {
        var d = cmp_args[0];
        d.release();
    }
    defer {
        var d = cmp_args[1];
        d.release();
    }

    var equal_out = try ctx.invokeVector("equal", cmp_args[0..], compute.Options.noneValue());
    defer equal_out.release();
    try expectBoolArrayValues(equal_out, &[_]?bool{ true, null, false });

    var i32_lhs = try makeInt32Array(allocator, &[_]?i32{ 1, null, 3 });
    defer i32_lhs.release();
    var i32_rhs = try makeInt32Array(allocator, &[_]?i32{ 2, 2, 3 });
    defer i32_rhs.release();
    const i32_cmp_args = [_]compute.Datum{
        compute.Datum.fromArray(i32_lhs.retain()),
        compute.Datum.fromArray(i32_rhs.retain()),
    };
    defer {
        var d = i32_cmp_args[0];
        d.release();
    }
    defer {
        var d = i32_cmp_args[1];
        d.release();
    }
    var less_i32_out = try ctx.invokeVector("less", i32_cmp_args[0..], compute.Options.noneValue());
    defer less_i32_out.release();
    try expectBoolArrayValues(less_i32_out, &[_]?bool{ true, null, false });

    var f64_lhs = try makeFloat64Array(allocator, &[_]?f64{ 1.5, null, 2.0 });
    defer f64_lhs.release();
    var f64_rhs = try makeFloat64Array(allocator, &[_]?f64{ 1.5, 3.0, 1.0 });
    defer f64_rhs.release();
    const f64_cmp_args = [_]compute.Datum{
        compute.Datum.fromArray(f64_lhs.retain()),
        compute.Datum.fromArray(f64_rhs.retain()),
    };
    defer {
        var d = f64_cmp_args[0];
        d.release();
    }
    defer {
        var d = f64_cmp_args[1];
        d.release();
    }
    var greater_equal_f64_out = try ctx.invokeVector("greater_equal", f64_cmp_args[0..], compute.Options.noneValue());
    defer greater_equal_f64_out.release();
    try expectBoolArrayValues(greater_equal_f64_out, &[_]?bool{ true, null, true });

    var bl = try makeBoolArray(allocator, &[_]?bool{ false, null, true });
    defer bl.release();
    var br = try makeBoolArray(allocator, &[_]?bool{ null, true, null });
    defer br.release();
    const logical_args = [_]compute.Datum{
        compute.Datum.fromArray(bl.retain()),
        compute.Datum.fromArray(br.retain()),
    };
    defer {
        var d = logical_args[0];
        d.release();
    }
    defer {
        var d = logical_args[1];
        d.release();
    }

    var and_kleene_out = try ctx.invokeVector("and_kleene", logical_args[0..], compute.Options.noneValue());
    defer and_kleene_out.release();
    try expectBoolArrayValues(and_kleene_out, &[_]?bool{ false, null, null });

    var xor_out = try ctx.invokeVector("xor", logical_args[0..], compute.Options.noneValue());
    defer xor_out.release();
    try expectBoolArrayValues(xor_out, &[_]?bool{ null, null, null });

    var and_not_out = try ctx.invokeVector("and_not", logical_args[0..], compute.Options.noneValue());
    defer and_not_out.release();
    try expectBoolArrayValues(and_not_out, &[_]?bool{ null, null, null });

    var and_not_kleene_out = try ctx.invokeVector("and_not_kleene", logical_args[0..], compute.Options.noneValue());
    defer and_not_kleene_out.release();
    try expectBoolArrayValues(and_not_kleene_out, &[_]?bool{ false, false, null });

    var or_kleene_out = try ctx.invokeVector("or_kleene", logical_args[0..], compute.Options.noneValue());
    defer or_kleene_out.release();
    try expectBoolArrayValues(or_kleene_out, &[_]?bool{ null, true, true });
}

test "cast supports numeric and bool subset" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var i32_values = try makeInt32Array(allocator, &[_]?i32{ 1, 0, 2 });
    defer i32_values.release();
    const args = [_]compute.Datum{compute.Datum.fromArray(i32_values.retain())};
    defer {
        var d = args[0];
        d.release();
    }

    var to_i64 = try ctx.invokeVector("cast", args[0..], .{
        .cast = .{
            .safe = true,
            .to_type = .{ .int64 = {} },
        },
    });
    defer to_i64.release();
    try expectInt64ArrayValues(to_i64, &[_]?i64{ 1, 0, 2 });

    try std.testing.expectError(
        error.InvalidCast,
        ctx.invokeVector("cast", args[0..], .{
            .cast = .{
                .safe = true,
                .to_type = .{ .bool = {} },
            },
        }),
    );

    var to_bool = try ctx.invokeVector("cast", args[0..], .{
        .cast = .{
            .safe = false,
            .to_type = .{ .bool = {} },
        },
    });
    defer to_bool.release();
    try expectBoolArrayValues(to_bool, &[_]?bool{ true, false, true });
}

test "aggregate count/sum/min/max/mean support int64" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeInt64Array(allocator, &[_]?i64{ 1, null, 3 });
    defer values.release();
    const args = [_]compute.Datum{compute.Datum.fromArray(values.retain())};
    defer {
        var d = args[0];
        d.release();
    }

    var count_out = try ctx.invokeAggregate("count", args[0..], compute.Options.noneValue());
    defer count_out.release();
    try std.testing.expect(count_out.isScalar());
    try std.testing.expectEqual(@as(i64, 2), count_out.scalar.value.i64);

    var sum_out = try ctx.invokeAggregate("sum", args[0..], compute.Options.noneValue());
    defer sum_out.release();
    try std.testing.expect(sum_out.isScalar());
    try std.testing.expectEqual(@as(i64, 4), sum_out.scalar.value.i64);

    var min_out = try ctx.invokeAggregate("min", args[0..], compute.Options.noneValue());
    defer min_out.release();
    try std.testing.expect(min_out.isScalar());
    try std.testing.expectEqual(@as(i64, 1), min_out.scalar.value.i64);

    var max_out = try ctx.invokeAggregate("max", args[0..], compute.Options.noneValue());
    defer max_out.release();
    try std.testing.expect(max_out.isScalar());
    try std.testing.expectEqual(@as(i64, 3), max_out.scalar.value.i64);

    var mean_out = try ctx.invokeAggregate("mean", args[0..], compute.Options.noneValue());
    defer mean_out.release();
    try std.testing.expect(mean_out.isScalar());
    try std.testing.expectEqual(@as(f64, 2.0), mean_out.scalar.value.f64);
}

test "aggregate all/any support bool with null-aware semantics" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeBoolArray(allocator, &[_]?bool{ true, null, true, false });
    defer values.release();
    const args = [_]compute.Datum{compute.Datum.fromArray(values.retain())};
    defer {
        var d = args[0];
        d.release();
    }

    var all_out = try ctx.invokeAggregate("all", args[0..], compute.Options.noneValue());
    defer all_out.release();
    try std.testing.expect(all_out.isScalar());
    try std.testing.expectEqual(false, all_out.scalar.value.bool);

    var any_out = try ctx.invokeAggregate("any", args[0..], compute.Options.noneValue());
    defer any_out.release();
    try std.testing.expect(any_out.isScalar());
    try std.testing.expectEqual(true, any_out.scalar.value.bool);
}

test "aggregate all/any return null when all inputs are null or empty" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var all_null = try makeBoolArray(allocator, &[_]?bool{ null, null });
    defer all_null.release();
    const all_null_args = [_]compute.Datum{compute.Datum.fromArray(all_null.retain())};
    defer {
        var d = all_null_args[0];
        d.release();
    }

    var all_out = try ctx.invokeAggregate("all", all_null_args[0..], compute.Options.noneValue());
    defer all_out.release();
    try std.testing.expect(all_out.isScalar());
    try std.testing.expect(all_out.scalar.value == .null);

    var any_out = try ctx.invokeAggregate("any", all_null_args[0..], compute.Options.noneValue());
    defer any_out.release();
    try std.testing.expect(any_out.isScalar());
    try std.testing.expect(any_out.scalar.value == .null);

    var empty = try makeBoolArray(allocator, &[_]?bool{});
    defer empty.release();
    const empty_args = [_]compute.Datum{compute.Datum.fromArray(empty.retain())};
    defer {
        var d = empty_args[0];
        d.release();
    }

    var empty_all_out = try ctx.invokeAggregate("all", empty_args[0..], compute.Options.noneValue());
    defer empty_all_out.release();
    try std.testing.expect(empty_all_out.isScalar());
    try std.testing.expect(empty_all_out.scalar.value == .null);

    var empty_any_out = try ctx.invokeAggregate("any", empty_args[0..], compute.Options.noneValue());
    defer empty_any_out.release();
    try std.testing.expect(empty_any_out.isScalar());
    try std.testing.expect(empty_any_out.scalar.value == .null);
}

test "aggregate all/any support chunked bool input and validate signature" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var c0 = try makeBoolArray(allocator, &[_]?bool{ true, null });
    defer c0.release();
    var c1 = try makeBoolArray(allocator, &[_]?bool{true});
    defer c1.release();
    var chunked_all = try compute.ChunkedArray.init(allocator, .{ .bool = {} }, &[_]zcore.ArrayRef{ c0, c1 });
    defer chunked_all.release();
    const all_args = [_]compute.Datum{compute.Datum.fromChunked(chunked_all.retain())};
    defer {
        var d = all_args[0];
        d.release();
    }

    var chunked_all_out = try ctx.invokeAggregate("all", all_args[0..], compute.Options.noneValue());
    defer chunked_all_out.release();
    try std.testing.expect(chunked_all_out.isScalar());
    try std.testing.expectEqual(true, chunked_all_out.scalar.value.bool);

    var c2 = try makeBoolArray(allocator, &[_]?bool{ false, null });
    defer c2.release();
    var c3 = try makeBoolArray(allocator, &[_]?bool{false});
    defer c3.release();
    var chunked_any = try compute.ChunkedArray.init(allocator, .{ .bool = {} }, &[_]zcore.ArrayRef{ c2, c3 });
    defer chunked_any.release();
    const any_args = [_]compute.Datum{compute.Datum.fromChunked(chunked_any.retain())};
    defer {
        var d = any_args[0];
        d.release();
    }

    var chunked_any_out = try ctx.invokeAggregate("any", any_args[0..], compute.Options.noneValue());
    defer chunked_any_out.release();
    try std.testing.expect(chunked_any_out.isScalar());
    try std.testing.expectEqual(false, chunked_any_out.scalar.value.bool);

    var i64_values = try makeInt64Array(allocator, &[_]?i64{ 1, 2 });
    defer i64_values.release();
    const bad_args = [_]compute.Datum{compute.Datum.fromArray(i64_values.retain())};
    defer {
        var d = bad_args[0];
        d.release();
    }
    try std.testing.expectError(
        error.NoMatchingKernel,
        ctx.invokeAggregate("all", bad_args[0..], compute.Options.noneValue()),
    );
    try std.testing.expectError(
        error.NoMatchingKernel,
        ctx.invokeAggregate("any", all_args[0..], .{ .filter = .{} }),
    );
}

test "indices_nonzero supports bool and int64 inputs" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var bool_values = try makeBoolArray(allocator, &[_]?bool{ false, null, true, true });
    defer bool_values.release();
    const bool_args = [_]compute.Datum{compute.Datum.fromArray(bool_values.retain())};
    defer {
        var d = bool_args[0];
        d.release();
    }

    var bool_out = try ctx.invokeVector("indices_nonzero", bool_args[0..], compute.Options.noneValue());
    defer bool_out.release();
    try expectInt64ArrayValues(bool_out, &[_]?i64{ 2, 3 });

    var int_values = try makeInt64Array(allocator, &[_]?i64{ 0, null, -1, 0, 5 });
    defer int_values.release();
    const int_args = [_]compute.Datum{compute.Datum.fromArray(int_values.retain())};
    defer {
        var d = int_args[0];
        d.release();
    }

    var int_out = try ctx.invokeVector("indices_nonzero", int_args[0..], compute.Options.noneValue());
    defer int_out.release();
    try expectInt64ArrayValues(int_out, &[_]?i64{ 2, 4 });
}

test "take supports fixed_size_list values" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeFixedSizeListInt32Array(
        allocator,
        2,
        &[_]bool{ true, false, true },
        &[_]i32{ 1, 2, 3, 4, 5, 6 },
    );
    defer values.release();
    var indices = try makeInt32Array(allocator, &[_]?i32{ 2, 1, null, 0 });
    defer indices.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(values.retain()),
        compute.Datum.fromArray(indices.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    var out = try ctx.invokeVector("take", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType() == .fixed_size_list);

    const out_list = zcore.FixedSizeListArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 4), out_list.len());
    try std.testing.expect(!out_list.isNull(0));
    try std.testing.expect(out_list.isNull(1));
    try std.testing.expect(out_list.isNull(2));
    try std.testing.expect(!out_list.isNull(3));

    var row0 = try out_list.value(0);
    defer row0.release();
    const row0_i32 = zcore.Int32Array{ .data = row0.data() };
    try std.testing.expectEqual(@as(i32, 5), row0_i32.value(0));
    try std.testing.expectEqual(@as(i32, 6), row0_i32.value(1));

    var row3 = try out_list.value(3);
    defer row3.release();
    const row3_i32 = zcore.Int32Array{ .data = row3.data() };
    try std.testing.expectEqual(@as(i32, 1), row3_i32.value(0));
    try std.testing.expectEqual(@as(i32, 2), row3_i32.value(1));
}

test "fill_null supports fixed_size_list values" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var lhs = try makeFixedSizeListInt32Array(
        allocator,
        2,
        &[_]bool{ true, false, true, false },
        &[_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 },
    );
    defer lhs.release();
    var rhs = try makeFixedSizeListInt32Array(
        allocator,
        2,
        &[_]bool{ true, true, true, true },
        &[_]i32{ 11, 12, 13, 14, 15, 16, 17, 18 },
    );
    defer rhs.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(lhs.retain()),
        compute.Datum.fromArray(rhs.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    var out = try ctx.invokeVector("fill_null", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType() == .fixed_size_list);

    const out_list = zcore.FixedSizeListArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 4), out_list.len());
    try std.testing.expect(!out_list.isNull(0));
    try std.testing.expect(!out_list.isNull(1));
    try std.testing.expect(!out_list.isNull(2));
    try std.testing.expect(!out_list.isNull(3));

    var row1 = try out_list.value(1);
    defer row1.release();
    const row1_i32 = zcore.Int32Array{ .data = row1.data() };
    try std.testing.expectEqual(@as(i32, 13), row1_i32.value(0));
    try std.testing.expectEqual(@as(i32, 14), row1_i32.value(1));

    var row3 = try out_list.value(3);
    defer row3.release();
    const row3_i32 = zcore.Int32Array{ .data = row3.data() };
    try std.testing.expectEqual(@as(i32, 17), row3_i32.value(0));
    try std.testing.expectEqual(@as(i32, 18), row3_i32.value(1));
}

test "aggregate sum/min/max/mean return null for all-null int64 input" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeInt64Array(allocator, &[_]?i64{ null, null, null });
    defer values.release();
    const args = [_]compute.Datum{compute.Datum.fromArray(values.retain())};
    defer {
        var d = args[0];
        d.release();
    }

    var sum_out = try ctx.invokeAggregate("sum", args[0..], compute.Options.noneValue());
    defer sum_out.release();
    try std.testing.expect(sum_out.isScalar());
    try std.testing.expect(sum_out.scalar.value == .null);

    var min_out = try ctx.invokeAggregate("min", args[0..], compute.Options.noneValue());
    defer min_out.release();
    try std.testing.expect(min_out.isScalar());
    try std.testing.expect(min_out.scalar.value == .null);

    var max_out = try ctx.invokeAggregate("max", args[0..], compute.Options.noneValue());
    defer max_out.release();
    try std.testing.expect(max_out.isScalar());
    try std.testing.expect(max_out.scalar.value == .null);

    var mean_out = try ctx.invokeAggregate("mean", args[0..], compute.Options.noneValue());
    defer mean_out.release();
    try std.testing.expect(mean_out.isScalar());
    try std.testing.expect(mean_out.scalar.value == .null);
}
