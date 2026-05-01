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

test "register base kernels exposes expected registry surface and resolvable signatures" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);

    try std.testing.expectEqual(@as(usize, 50), registry.functionCount());

    const vector_names = [_][]const u8{
        "add_i64",
        "filter",
        "array_filter",
        "filter_i64",
        "drop_null",
        "take",
        "array_take",
        "sort_indices",
        "array_sort_indices",
        "indices_nonzero",
        "is_null",
        "is_valid",
        "is_finite",
        "is_inf",
        "is_nan",
        "true_unless_null",
        "if_else",
        "coalesce",
        "choose",
        "case_when",
        "fill_null",
        "fill_null_forward",
        "fill_null_backward",
        "equal",
        "not_equal",
        "less",
        "less_equal",
        "greater",
        "greater_equal",
        "invert",
        "and_",
        "or_",
        "xor",
        "and_not",
        "and_kleene",
        "and_not_kleene",
        "or_kleene",
        "subtract_i64",
        "divide_i64",
        "multiply_i64",
        "cast_i64_to_i32",
        "cast",
    };
    for (vector_names) |name| {
        try std.testing.expect(registry.containsFunction(name, .vector));
        try std.testing.expectEqual(@as(usize, 1), registry.kernelCount(name, .vector));
        try std.testing.expect(!registry.containsFunction(name, .aggregate));
    }

    try std.testing.expect(registry.containsFunction("count_rows", .aggregate));
    try std.testing.expectEqual(@as(usize, 1), registry.kernelCount("count_rows", .aggregate));
    try std.testing.expect(!registry.containsFunction("count_rows", .vector));
    try std.testing.expect(registry.containsFunction("all", .aggregate));
    try std.testing.expect(registry.containsFunction("any", .aggregate));
    try std.testing.expect(registry.containsFunction("count", .aggregate));
    try std.testing.expect(registry.containsFunction("sum", .aggregate));
    try std.testing.expect(registry.containsFunction("min", .aggregate));
    try std.testing.expect(registry.containsFunction("max", .aggregate));
    try std.testing.expect(registry.containsFunction("mean", .aggregate));
    try std.testing.expectEqual(@as(usize, 0), registry.kernelCount("not_exist", .vector));

    var add_lhs = try makeInt64Array(allocator, &[_]?i64{1});
    defer add_lhs.release();
    const add_args = [_]compute.Datum{
        compute.Datum.fromArray(add_lhs.retain()),
        compute.Datum.fromScalar(.{
            .data_type = .{ .int64 = {} },
            .value = .{ .i64 = 2 },
        }),
    };
    defer {
        var d = add_args[0];
        d.release();
    }
    defer {
        var d = add_args[1];
        d.release();
    }
    const add_ty = try registry.resolveResultType("add_i64", .vector, add_args[0..], .{
        .arithmetic = .{},
    });
    try std.testing.expect(add_ty.eql(.{ .int64 = {} }));
    const add_i32_args = [_]compute.Datum{
        compute.Datum.fromScalar(.{
            .data_type = .{ .int32 = {} },
            .value = .{ .i32 = 1 },
        }),
        compute.Datum.fromScalar(.{
            .data_type = .{ .int32 = {} },
            .value = .{ .i32 = 2 },
        }),
    };
    defer {
        var d = add_i32_args[0];
        d.release();
    }
    defer {
        var d = add_i32_args[1];
        d.release();
    }
    const add_i32_ty = try registry.resolveResultType("add_i64", .vector, add_i32_args[0..], .{
        .arithmetic = .{},
    });
    try std.testing.expect(add_i32_ty.eql(.{ .int32 = {} }));
    const add_f64_args = [_]compute.Datum{
        compute.Datum.fromScalar(.{
            .data_type = .{ .double = {} },
            .value = .{ .f64 = 1.0 },
        }),
        compute.Datum.fromScalar(.{
            .data_type = .{ .double = {} },
            .value = .{ .f64 = 2.0 },
        }),
    };
    defer {
        var d = add_f64_args[0];
        d.release();
    }
    defer {
        var d = add_f64_args[1];
        d.release();
    }
    const add_f64_ty = try registry.resolveResultType("add_i64", .vector, add_f64_args[0..], .{
        .arithmetic = .{},
    });
    try std.testing.expect(add_f64_ty.eql(.{ .double = {} }));

    var filter_values = try makeInt32Array(allocator, &[_]?i32{ 1, 2 });
    defer filter_values.release();
    var filter_pred = try makeBoolArray(allocator, &[_]?bool{ true, false });
    defer filter_pred.release();
    const filter_args = [_]compute.Datum{
        compute.Datum.fromArray(filter_values.retain()),
        compute.Datum.fromArray(filter_pred.retain()),
    };
    defer {
        var d = filter_args[0];
        d.release();
    }
    defer {
        var d = filter_args[1];
        d.release();
    }
    const filter_ty = try registry.resolveResultType("filter", .vector, filter_args[0..], .{
        .filter = .{},
    });
    try std.testing.expect(filter_ty.eql(.{ .int32 = {} }));
    const array_filter_ty = try registry.resolveResultType("array_filter", .vector, filter_args[0..], .{
        .filter = .{},
    });
    try std.testing.expect(array_filter_ty.eql(.{ .int32 = {} }));

    const drop_null_args = [_]compute.Datum{
        compute.Datum.fromArray(filter_values.retain()),
    };
    defer {
        var d = drop_null_args[0];
        d.release();
    }
    const drop_null_ty = try registry.resolveResultType(
        "drop_null",
        .vector,
        drop_null_args[0..],
        compute.Options.noneValue(),
    );
    try std.testing.expect(drop_null_ty.eql(.{ .int32 = {} }));
    const sort_indices_ty = try registry.resolveResultType(
        "sort_indices",
        .vector,
        drop_null_args[0..],
        compute.Options.noneValue(),
    );
    try std.testing.expect(sort_indices_ty.eql(.{ .int64 = {} }));
    const array_sort_indices_ty = try registry.resolveResultType(
        "array_sort_indices",
        .vector,
        drop_null_args[0..],
        compute.Options.noneValue(),
    );
    try std.testing.expect(array_sort_indices_ty.eql(.{ .int64 = {} }));

    const is_null_ty = try registry.resolveResultType(
        "is_null",
        .vector,
        drop_null_args[0..],
        compute.Options.noneValue(),
    );
    try std.testing.expect(is_null_ty.eql(.{ .bool = {} }));
    const is_valid_ty = try registry.resolveResultType(
        "is_valid",
        .vector,
        drop_null_args[0..],
        compute.Options.noneValue(),
    );
    try std.testing.expect(is_valid_ty.eql(.{ .bool = {} }));
    var finite_values = try makeFloat64Array(allocator, &[_]?f64{ 1.0, null });
    defer finite_values.release();
    const finite_args = [_]compute.Datum{
        compute.Datum.fromArray(finite_values.retain()),
    };
    defer {
        var d = finite_args[0];
        d.release();
    }
    const is_finite_ty = try registry.resolveResultType(
        "is_finite",
        .vector,
        finite_args[0..],
        compute.Options.noneValue(),
    );
    try std.testing.expect(is_finite_ty.eql(.{ .bool = {} }));
    const true_unless_null_ty = try registry.resolveResultType(
        "true_unless_null",
        .vector,
        drop_null_args[0..],
        compute.Options.noneValue(),
    );
    try std.testing.expect(true_unless_null_ty.eql(.{ .bool = {} }));

    const if_else_args = [_]compute.Datum{
        compute.Datum.fromArray(filter_pred.retain()),
        compute.Datum.fromArray(filter_values.retain()),
        compute.Datum.fromScalar(.{
            .data_type = .{ .int32 = {} },
            .value = .{ .i32 = 7 },
        }),
    };
    defer {
        var d = if_else_args[0];
        d.release();
    }
    defer {
        var d = if_else_args[1];
        d.release();
    }
    defer {
        var d = if_else_args[2];
        d.release();
    }
    const if_else_ty = try registry.resolveResultType(
        "if_else",
        .vector,
        if_else_args[0..],
        compute.Options.noneValue(),
    );
    try std.testing.expect(if_else_ty.eql(.{ .int32 = {} }));

    const coalesce_args = [_]compute.Datum{
        compute.Datum.fromArray(filter_values.retain()),
        compute.Datum.fromScalar(.{
            .data_type = .{ .int32 = {} },
            .value = .{ .i32 = 99 },
        }),
    };
    defer {
        var d = coalesce_args[0];
        d.release();
    }
    defer {
        var d = coalesce_args[1];
        d.release();
    }
    const coalesce_ty = try registry.resolveResultType(
        "coalesce",
        .vector,
        coalesce_args[0..],
        compute.Options.noneValue(),
    );
    try std.testing.expect(coalesce_ty.eql(.{ .int32 = {} }));

    const choose_args = [_]compute.Datum{
        compute.Datum.fromArray(filter_values.retain()),
        compute.Datum.fromArray(filter_values.retain()),
        compute.Datum.fromScalar(.{
            .data_type = .{ .int32 = {} },
            .value = .{ .i32 = 42 },
        }),
    };
    defer {
        var d = choose_args[0];
        d.release();
    }
    defer {
        var d = choose_args[1];
        d.release();
    }
    defer {
        var d = choose_args[2];
        d.release();
    }
    const choose_ty = try registry.resolveResultType(
        "choose",
        .vector,
        choose_args[0..],
        compute.Options.noneValue(),
    );
    try std.testing.expect(choose_ty.eql(.{ .int32 = {} }));

    var case_when_cond1 = try makeBoolArray(allocator, &[_]?bool{ false, true });
    defer case_when_cond1.release();
    var case_when_conds = try makeStructBool2Array(allocator, filter_pred, case_when_cond1);
    defer case_when_conds.release();
    const case_when_args = [_]compute.Datum{
        compute.Datum.fromArray(case_when_conds.retain()),
        compute.Datum.fromArray(filter_values.retain()),
        compute.Datum.fromArray(filter_values.retain()),
    };
    defer {
        var d = case_when_args[0];
        d.release();
    }
    defer {
        var d = case_when_args[1];
        d.release();
    }
    defer {
        var d = case_when_args[2];
        d.release();
    }
    const case_when_ty = try registry.resolveResultType(
        "case_when",
        .vector,
        case_when_args[0..],
        compute.Options.noneValue(),
    );
    try std.testing.expect(case_when_ty.eql(.{ .int32 = {} }));

    const cast_args = [_]compute.Datum{
        compute.Datum.fromScalar(.{
            .data_type = .{ .int64 = {} },
            .value = .{ .i64 = 7 },
        }),
    };
    const cast_ty = try registry.resolveResultType("cast_i64_to_i32", .vector, cast_args[0..], .{
        .cast = .{
            .safe = true,
            .to_type = .{ .int32 = {} },
        },
    });
    try std.testing.expect(cast_ty.eql(.{ .int32 = {} }));

    var count_input = try makeInt64Array(allocator, &[_]?i64{ 9, 10, 11 });
    defer count_input.release();
    const count_args = [_]compute.Datum{
        compute.Datum.fromArray(count_input.retain()),
    };
    defer {
        var d = count_args[0];
        d.release();
    }
    const count_ty = try registry.resolveResultType(
        "count_rows",
        .aggregate,
        count_args[0..],
        compute.Options.noneValue(),
    );
    try std.testing.expect(count_ty.eql(.{ .int64 = {} }));
    const all_args = [_]compute.Datum{
        compute.Datum.fromArray(filter_pred.retain()),
    };
    defer {
        var d = all_args[0];
        d.release();
    }
    const all_ty = try registry.resolveResultType(
        "all",
        .aggregate,
        all_args[0..],
        compute.Options.noneValue(),
    );
    try std.testing.expect(all_ty.eql(.{ .bool = {} }));
}

test "register compat kernels matches base registry surface" {
    const allocator = std.testing.allocator;
    var base_registry = compute.FunctionRegistry.init(allocator);
    defer base_registry.deinit();
    try registerBaseKernels(&base_registry);

    var compat_registry = compute.FunctionRegistry.init(allocator);
    defer compat_registry.deinit();
    try registerCompatKernels(&compat_registry);

    try std.testing.expectEqual(base_registry.functionCount(), compat_registry.functionCount());

    const vector_names = [_][]const u8{
        "add_i64",
        "filter",
        "array_filter",
        "filter_i64",
        "drop_null",
        "is_null",
        "is_valid",
        "is_finite",
        "is_inf",
        "is_nan",
        "true_unless_null",
        "if_else",
        "coalesce",
        "choose",
        "case_when",
        "sort_indices",
        "array_sort_indices",
        "subtract_i64",
        "divide_i64",
        "multiply_i64",
        "cast_i64_to_i32",
    };
    for (vector_names) |name| {
        try std.testing.expectEqual(
            base_registry.containsFunction(name, .vector),
            compat_registry.containsFunction(name, .vector),
        );
        try std.testing.expectEqual(
            base_registry.kernelCount(name, .vector),
            compat_registry.kernelCount(name, .vector),
        );
    }

    try std.testing.expectEqual(
        base_registry.containsFunction("count_rows", .aggregate),
        compat_registry.containsFunction("count_rows", .aggregate),
    );
    try std.testing.expectEqual(
        base_registry.kernelCount("count_rows", .aggregate),
        compat_registry.kernelCount("count_rows", .aggregate),
    );
    try std.testing.expectEqual(
        base_registry.containsFunction("all", .aggregate),
        compat_registry.containsFunction("all", .aggregate),
    );
    try std.testing.expectEqual(
        base_registry.containsFunction("any", .aggregate),
        compat_registry.containsFunction("any", .aggregate),
    );
}
