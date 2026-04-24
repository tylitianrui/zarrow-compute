const std = @import("std");
const zcore = @import("zarrow-core");
const impl = @import("kernels/impl.zig");

pub const compute = impl.compute;
pub const registerBaseKernels = impl.registerBaseKernels;
pub const registerCompatKernels = impl.registerCompatKernels;

const DT_BOOL = zcore.DataType{ .bool = {} };
const DT_INT32 = zcore.DataType{ .int32 = {} };
const DT_INT64 = zcore.DataType{ .int64 = {} };
const FIELD_LIST_ITEM_I32 = zcore.Field{
    .name = "item",
    .data_type = &DT_INT32,
    .nullable = true,
};
const DT_LIST_I32 = zcore.DataType{ .list = .{ .value_field = FIELD_LIST_ITEM_I32 } };
const STRUCT_FIELDS_I64_BOOL = [_]zcore.Field{
    .{ .name = "i64", .data_type = &DT_INT64, .nullable = true },
    .{ .name = "b", .data_type = &DT_BOOL, .nullable = true },
};
const STRUCT_FIELDS_BOOL2 = [_]zcore.Field{
    .{ .name = "c0", .data_type = &DT_BOOL, .nullable = true },
    .{ .name = "c1", .data_type = &DT_BOOL, .nullable = true },
};
const STRUCT_FIELDS_LIST_I32 = [_]zcore.Field{
    .{ .name = "items", .data_type = &DT_LIST_I32, .nullable = true },
};

fn makeInt64Array(allocator: std.mem.Allocator, values: []const ?i64) !zcore.ArrayRef {
    var builder = try zcore.Int64Builder.init(allocator, values.len);
    defer builder.deinit();
    for (values) |v| {
        if (v) |x| {
            try builder.append(x);
        } else {
            try builder.appendNull();
        }
    }
    return builder.finish();
}

fn makeInt32Array(allocator: std.mem.Allocator, values: []const ?i32) !zcore.ArrayRef {
    var builder = try zcore.Int32Builder.init(allocator, values.len);
    defer builder.deinit();
    for (values) |v| {
        if (v) |x| {
            try builder.append(x);
        } else {
            try builder.appendNull();
        }
    }
    return builder.finish();
}

fn makeBoolArray(allocator: std.mem.Allocator, values: []const ?bool) !zcore.ArrayRef {
    var builder = try zcore.BooleanBuilder.init(allocator, values.len);
    defer builder.deinit();
    for (values) |v| {
        if (v) |x| {
            try builder.append(x);
        } else {
            try builder.appendNull();
        }
    }
    return builder.finish();
}

fn makeNullArray(allocator: std.mem.Allocator, len: usize) !zcore.ArrayRef {
    var builder = try zcore.NullBuilder.init(allocator, len);
    defer builder.deinit();
    try builder.appendNulls(len);
    return builder.finish();
}

fn makeStructI64BoolArray(
    allocator: std.mem.Allocator,
    present: []const bool,
    ints: []const ?i64,
    bools: []const ?bool,
) !zcore.ArrayRef {
    if (present.len != ints.len or present.len != bools.len) return error.InvalidInput;
    var int_child = try makeInt64Array(allocator, ints);
    defer int_child.release();
    var bool_child = try makeBoolArray(allocator, bools);
    defer bool_child.release();

    var builder = zcore.StructBuilder.init(allocator, STRUCT_FIELDS_I64_BOOL[0..]);
    defer builder.deinit();
    for (present) |is_present| {
        if (is_present) {
            try builder.appendValid();
        } else {
            try builder.appendNull();
        }
    }
    return builder.finish(&[_]zcore.ArrayRef{ int_child, bool_child });
}

fn makeStructListI32Array(
    allocator: std.mem.Allocator,
    field_child: zcore.ArrayRef,
    present: []const bool,
) !zcore.ArrayRef {
    if (present.len != field_child.data().length) return error.InvalidInput;
    var builder = zcore.StructBuilder.init(allocator, STRUCT_FIELDS_LIST_I32[0..]);
    defer builder.deinit();
    for (present) |is_present| {
        if (is_present) {
            try builder.appendValid();
        } else {
            try builder.appendNull();
        }
    }
    return builder.finish(&[_]zcore.ArrayRef{field_child});
}

fn makeStructBool2Array(
    allocator: std.mem.Allocator,
    cond0: zcore.ArrayRef,
    cond1: zcore.ArrayRef,
) !zcore.ArrayRef {
    if (!cond0.data().data_type.eql(.{ .bool = {} }) or !cond1.data().data_type.eql(.{ .bool = {} })) {
        return error.InvalidInput;
    }
    if (cond0.data().length != cond1.data().length) return error.InvalidInput;

    var builder = zcore.StructBuilder.init(allocator, STRUCT_FIELDS_BOOL2[0..]);
    defer builder.deinit();
    var row: usize = 0;
    while (row < cond0.data().length) : (row += 1) {
        try builder.appendValid();
    }

    return builder.finish(&[_]zcore.ArrayRef{ cond0, cond1 });
}

fn makeStringArray(allocator: std.mem.Allocator, values: []const ?[]const u8) !zcore.ArrayRef {
    var data_capacity: usize = 0;
    for (values) |v| {
        if (v) |x| data_capacity += x.len;
    }
    var builder = try zcore.StringBuilder.init(allocator, values.len, data_capacity);
    defer builder.deinit();
    for (values) |v| {
        if (v) |x| {
            try builder.append(x);
        } else {
            try builder.appendNull();
        }
    }
    return builder.finish();
}

fn makeLargeStringArray(allocator: std.mem.Allocator, values: []const ?[]const u8) !zcore.ArrayRef {
    var data_capacity: usize = 0;
    for (values) |v| {
        if (v) |x| data_capacity += x.len;
    }
    var builder = try zcore.LargeStringBuilder.init(allocator, values.len, data_capacity);
    defer builder.deinit();
    for (values) |v| {
        if (v) |x| {
            try builder.append(x);
        } else {
            try builder.appendNull();
        }
    }
    return builder.finish();
}

fn makeBinaryArray(allocator: std.mem.Allocator, values: []const ?[]const u8) !zcore.ArrayRef {
    var data_capacity: usize = 0;
    for (values) |v| {
        if (v) |x| data_capacity += x.len;
    }
    var builder = try zcore.BinaryBuilder.init(allocator, values.len, data_capacity);
    defer builder.deinit();
    for (values) |v| {
        if (v) |x| {
            try builder.append(x);
        } else {
            try builder.appendNull();
        }
    }
    return builder.finish();
}

fn makeStringViewArray(allocator: std.mem.Allocator, values: []const ?[]const u8) !zcore.ArrayRef {
    var data_capacity: usize = 0;
    for (values) |v| {
        if (v) |x| data_capacity += x.len;
    }
    var builder = try zcore.StringViewBuilder.init(allocator, values.len, data_capacity);
    defer builder.deinit();
    for (values) |v| {
        if (v) |x| {
            try builder.append(x);
        } else {
            try builder.appendNull();
        }
    }
    return builder.finish();
}

fn makeBinaryViewArray(allocator: std.mem.Allocator, values: []const ?[]const u8) !zcore.ArrayRef {
    var data_capacity: usize = 0;
    for (values) |v| {
        if (v) |x| data_capacity += x.len;
    }
    var builder = try zcore.BinaryViewBuilder.init(allocator, values.len, data_capacity);
    defer builder.deinit();
    for (values) |v| {
        if (v) |x| {
            try builder.append(x);
        } else {
            try builder.appendNull();
        }
    }
    return builder.finish();
}

fn makeFixedSizeBinaryArray(allocator: std.mem.Allocator, byte_width: usize, values: []const ?[]const u8) !zcore.ArrayRef {
    var builder = try zcore.FixedSizeBinaryBuilder.init(allocator, byte_width, values.len);
    defer builder.deinit();
    for (values) |v| {
        if (v) |x| {
            try builder.append(x);
        } else {
            try builder.appendNull();
        }
    }
    return builder.finish();
}

fn makeListInt32Array(allocator: std.mem.Allocator) !zcore.ArrayRef {
    var values_builder = try zcore.Int32Builder.init(allocator, 3);
    defer values_builder.deinit();
    try values_builder.append(1);
    try values_builder.append(2);
    try values_builder.append(3);
    var values = try values_builder.finish();
    defer values.release();

    var list_builder = try zcore.ListBuilder.init(allocator, 2, FIELD_LIST_ITEM_I32);
    defer list_builder.deinit();
    try list_builder.appendLen(2);
    try list_builder.appendLen(1);
    return list_builder.finish(values);
}

test "register base kernels exposes expected registry surface and resolvable signatures" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);

    try std.testing.expectEqual(@as(usize, 16), registry.functionCount());

    const vector_names = [_][]const u8{
        "add_i64",
        "filter",
        "filter_i64",
        "drop_null",
        "is_null",
        "is_valid",
        "true_unless_null",
        "if_else",
        "coalesce",
        "choose",
        "case_when",
        "subtract_i64",
        "divide_i64",
        "multiply_i64",
        "cast_i64_to_i32",
    };
    for (vector_names) |name| {
        try std.testing.expect(registry.containsFunction(name, .vector));
        try std.testing.expectEqual(@as(usize, 1), registry.kernelCount(name, .vector));
        try std.testing.expect(!registry.containsFunction(name, .aggregate));
    }

    try std.testing.expect(registry.containsFunction("count_rows", .aggregate));
    try std.testing.expectEqual(@as(usize, 1), registry.kernelCount("count_rows", .aggregate));
    try std.testing.expect(!registry.containsFunction("count_rows", .vector));
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
        "filter_i64",
        "drop_null",
        "is_null",
        "is_valid",
        "true_unless_null",
        "if_else",
        "coalesce",
        "choose",
        "case_when",
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
}

test "add_i64 supports scalar broadcast and null propagation" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var lhs = try makeInt64Array(allocator, &[_]?i64{ 1, null, 3 });
    defer lhs.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(lhs.retain()),
        compute.Datum.fromScalar(.{
            .data_type = .{ .int64 = {} },
            .value = .{ .i64 = 10 },
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

    var out = try ctx.invokeVector("add_i64", args[0..], .{ .arithmetic = .{} });
    defer out.release();
    try std.testing.expect(out.isArray());

    const view = zcore.Int64Array{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), view.len());
    try std.testing.expectEqual(@as(i64, 11), view.value(0));
    try std.testing.expect(view.isNull(1));
    try std.testing.expectEqual(@as(i64, 13), view.value(2));
}

test "filter keeps selected values, propagates value nulls, and drops null predicates by default" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeInt64Array(allocator, &[_]?i64{ 1, null, 3, 4 });
    defer values.release();
    var predicate = try makeBoolArray(allocator, &[_]?bool{ true, true, null, false });
    defer predicate.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(values.retain()),
        compute.Datum.fromArray(predicate.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    var out = try ctx.invokeVector("filter", args[0..], .{ .filter = .{} });
    defer out.release();
    try std.testing.expect(out.isArray());

    const view = zcore.Int64Array{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 2), view.len());
    try std.testing.expectEqual(@as(i64, 1), view.value(0));
    try std.testing.expect(view.isNull(1));
}

test "filter emits null for null predicate when drop_nulls is false" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeInt64Array(allocator, &[_]?i64{ 7, 8, 9 });
    defer values.release();
    var predicate = try makeBoolArray(allocator, &[_]?bool{ true, null, true });
    defer predicate.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(values.retain()),
        compute.Datum.fromArray(predicate.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    var out = try ctx.invokeVector("filter_i64", args[0..], .{
        .filter = .{ .drop_nulls = false },
    });
    defer out.release();
    try std.testing.expect(out.isArray());

    const view = zcore.Int64Array{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), view.len());
    try std.testing.expectEqual(@as(i64, 7), view.value(0));
    try std.testing.expect(view.isNull(1));
    try std.testing.expectEqual(@as(i64, 9), view.value(2));
}

test "filter supports int32 value arrays with bool predicate" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeInt32Array(allocator, &[_]?i32{ 10, null, 20, 30 });
    defer values.release();
    var predicate = try makeBoolArray(allocator, &[_]?bool{ true, true, false, true });
    defer predicate.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(values.retain()),
        compute.Datum.fromArray(predicate.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    var out = try ctx.invokeVector("filter", args[0..], .{ .filter = .{} });
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .int32 = {} }));

    const view = zcore.Int32Array{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), view.len());
    try std.testing.expectEqual(@as(i32, 10), try view.value(0));
    try std.testing.expect(view.isNull(1));
    try std.testing.expectEqual(@as(i32, 30), try view.value(2));
}

test "filter supports bool value arrays" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeBoolArray(allocator, &[_]?bool{ true, null, false });
    defer values.release();
    var predicate = try makeBoolArray(allocator, &[_]?bool{ true, true, true });
    defer predicate.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(values.retain()),
        compute.Datum.fromArray(predicate.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    var out = try ctx.invokeVector("filter", args[0..], .{ .filter = .{} });
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .bool = {} }));

    const view = zcore.BooleanArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), view.len());
    try std.testing.expect(view.value(0));
    try std.testing.expect(view.isNull(1));
    try std.testing.expect(!view.value(2));
}

test "filter supports string scalar broadcast and predicate null emission" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var predicate = try makeBoolArray(allocator, &[_]?bool{ true, false, null, true });
    defer predicate.release();

    const args = [_]compute.Datum{
        compute.Datum.fromScalar(.{
            .data_type = .{ .string = {} },
            .value = .{ .string = "x" },
        }),
        compute.Datum.fromArray(predicate.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    var out = try ctx.invokeVector("filter", args[0..], .{
        .filter = .{ .drop_nulls = false },
    });
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .string = {} }));

    const view = zcore.StringArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), view.len());
    try std.testing.expect(std.mem.eql(u8, view.value(0), "x"));
    try std.testing.expect(view.isNull(1));
    try std.testing.expect(std.mem.eql(u8, view.value(2), "x"));
}

test "filter supports binary arrays and predicate null emission" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeBinaryArray(allocator, &[_]?[]const u8{ "aa", null, "bb", "cc" });
    defer values.release();
    var predicate = try makeBoolArray(allocator, &[_]?bool{ true, true, null, false });
    defer predicate.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(values.retain()),
        compute.Datum.fromArray(predicate.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    var out = try ctx.invokeVector("filter", args[0..], .{
        .filter = .{ .drop_nulls = false },
    });
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .binary = {} }));

    const view = zcore.BinaryArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), view.len());
    try std.testing.expect(std.mem.eql(u8, view.value(0), "aa"));
    try std.testing.expect(view.isNull(1));
    try std.testing.expect(view.isNull(2));
}

test "filter supports large_string value arrays" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeLargeStringArray(allocator, &[_]?[]const u8{ "left", null, "right" });
    defer values.release();
    var predicate = try makeBoolArray(allocator, &[_]?bool{ false, true, true });
    defer predicate.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(values.retain()),
        compute.Datum.fromArray(predicate.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    var out = try ctx.invokeVector("filter", args[0..], .{ .filter = .{} });
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .large_string = {} }));

    const view = zcore.LargeStringArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 2), view.len());
    try std.testing.expect(view.isNull(0));
    try std.testing.expect(std.mem.eql(u8, view.value(1), "right"));
}

test "filter supports fixed_size_binary value arrays" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeFixedSizeBinaryArray(allocator, 2, &[_]?[]const u8{ "ab", null, "cd", "ef" });
    defer values.release();
    var predicate = try makeBoolArray(allocator, &[_]?bool{ true, true, false, true });
    defer predicate.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(values.retain()),
        compute.Datum.fromArray(predicate.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    var out = try ctx.invokeVector("filter", args[0..], .{ .filter = .{} });
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{
        .fixed_size_binary = .{ .byte_width = 2 },
    }));

    const view = zcore.FixedSizeBinaryArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), view.len());
    try std.testing.expect(std.mem.eql(u8, view.value(0), "ab"));
    try std.testing.expect(view.isNull(1));
    try std.testing.expect(std.mem.eql(u8, view.value(2), "ef"));
}

test "filter supports chunked values and chunked predicates with misaligned chunk boundaries" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values_chunk0 = try makeInt64Array(allocator, &[_]?i64{ 1, null });
    defer values_chunk0.release();
    var values_chunk1 = try makeInt64Array(allocator, &[_]?i64{ 3, 4 });
    defer values_chunk1.release();
    var values_chunked = try compute.ChunkedArray.init(
        allocator,
        .{ .int64 = {} },
        &[_]zcore.ArrayRef{ values_chunk0, values_chunk1 },
    );
    defer values_chunked.release();

    var pred_chunk0 = try makeBoolArray(allocator, &[_]?bool{true});
    defer pred_chunk0.release();
    var pred_chunk1 = try makeBoolArray(allocator, &[_]?bool{ true, false, true });
    defer pred_chunk1.release();
    var pred_chunked = try compute.ChunkedArray.init(
        allocator,
        .{ .bool = {} },
        &[_]zcore.ArrayRef{ pred_chunk0, pred_chunk1 },
    );
    defer pred_chunked.release();

    const args = [_]compute.Datum{
        compute.Datum.fromChunked(values_chunked.retain()),
        compute.Datum.fromChunked(pred_chunked.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    var out = try ctx.invokeVector("filter", args[0..], .{ .filter = .{} });
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .int64 = {} }));

    const view = zcore.Int64Array{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), view.len());
    try std.testing.expectEqual(@as(i64, 1), view.value(0));
    try std.testing.expect(view.isNull(1));
    try std.testing.expectEqual(@as(i64, 4), view.value(2));
}

test "filter supports string_view value arrays" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeStringViewArray(allocator, &[_]?[]const u8{ "one", null, "two", "three" });
    defer values.release();
    var predicate = try makeBoolArray(allocator, &[_]?bool{ true, true, false, true });
    defer predicate.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(values.retain()),
        compute.Datum.fromArray(predicate.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    var out = try ctx.invokeVector("filter", args[0..], .{ .filter = .{} });
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .string_view = {} }));

    const view = zcore.StringViewArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), view.len());
    try std.testing.expect(std.mem.eql(u8, view.value(0), "one"));
    try std.testing.expect(view.isNull(1));
    try std.testing.expect(std.mem.eql(u8, view.value(2), "three"));
}

test "filter supports binary_view value arrays with predicate null emission" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeBinaryViewArray(allocator, &[_]?[]const u8{ "aa", "bb", null, "cc" });
    defer values.release();
    var predicate = try makeBoolArray(allocator, &[_]?bool{ true, null, true, false });
    defer predicate.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(values.retain()),
        compute.Datum.fromArray(predicate.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    var out = try ctx.invokeVector("filter", args[0..], .{
        .filter = .{ .drop_nulls = false },
    });
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .binary_view = {} }));

    const view = zcore.BinaryViewArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), view.len());
    try std.testing.expect(std.mem.eql(u8, view.value(0), "aa"));
    try std.testing.expect(view.isNull(1));
    try std.testing.expect(view.isNull(2));
}

test "filter rejects unsupported nested value type at dispatch" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeListInt32Array(allocator);
    defer values.release();
    var predicate = try makeBoolArray(allocator, &[_]?bool{ true, true });
    defer predicate.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(values.retain()),
        compute.Datum.fromArray(predicate.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    try std.testing.expectError(
        error.NoMatchingKernel,
        ctx.invokeVector("filter", args[0..], .{ .filter = .{} }),
    );
}

test "drop_null removes nulls from int64 arrays and keeps type" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeInt64Array(allocator, &[_]?i64{ 10, null, 20, null, 30 });
    defer values.release();
    const args = [_]compute.Datum{
        compute.Datum.fromArray(values.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }

    var out = try ctx.invokeVector("drop_null", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .int64 = {} }));

    const view = zcore.Int64Array{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), view.len());
    try std.testing.expectEqual(@as(i64, 10), view.value(0));
    try std.testing.expectEqual(@as(i64, 20), view.value(1));
    try std.testing.expectEqual(@as(i64, 30), view.value(2));
}

test "drop_null supports chunked input and preserves order" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var c0 = try makeInt64Array(allocator, &[_]?i64{ 1, null });
    defer c0.release();
    var c1 = try makeInt64Array(allocator, &[_]?i64{ null, 2, 3 });
    defer c1.release();
    var chunked = try compute.ChunkedArray.init(allocator, .{ .int64 = {} }, &[_]zcore.ArrayRef{ c0, c1 });
    defer chunked.release();

    const args = [_]compute.Datum{
        compute.Datum.fromChunked(chunked.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }

    var out = try ctx.invokeVector("drop_null", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .int64 = {} }));

    const view = zcore.Int64Array{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), view.len());
    try std.testing.expectEqual(@as(i64, 1), view.value(0));
    try std.testing.expectEqual(@as(i64, 2), view.value(1));
    try std.testing.expectEqual(@as(i64, 3), view.value(2));
}

test "drop_null rejects unsupported nested value type at dispatch" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeListInt32Array(allocator);
    defer values.release();
    const args = [_]compute.Datum{
        compute.Datum.fromArray(values.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }

    try std.testing.expectError(
        error.NoMatchingKernel,
        ctx.invokeVector("drop_null", args[0..], compute.Options.noneValue()),
    );
}

test "is_null marks null positions for chunked input" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var c0 = try makeInt64Array(allocator, &[_]?i64{ 1, null });
    defer c0.release();
    var c1 = try makeInt64Array(allocator, &[_]?i64{ null, 2, 3 });
    defer c1.release();
    var chunked = try compute.ChunkedArray.init(allocator, .{ .int64 = {} }, &[_]zcore.ArrayRef{ c0, c1 });
    defer chunked.release();

    const args = [_]compute.Datum{
        compute.Datum.fromChunked(chunked.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }

    var out = try ctx.invokeVector("is_null", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .bool = {} }));

    const view = zcore.BooleanArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 5), view.len());
    try std.testing.expect(!view.value(0));
    try std.testing.expect(view.value(1));
    try std.testing.expect(view.value(2));
    try std.testing.expect(!view.value(3));
    try std.testing.expect(!view.value(4));
}

test "is_valid is inverse of null positions" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeStringArray(allocator, &[_]?[]const u8{ "a", null, "b", null });
    defer values.release();
    const args = [_]compute.Datum{
        compute.Datum.fromArray(values.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }

    var out = try ctx.invokeVector("is_valid", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .bool = {} }));

    const view = zcore.BooleanArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 4), view.len());
    try std.testing.expect(view.value(0));
    try std.testing.expect(!view.value(1));
    try std.testing.expect(view.value(2));
    try std.testing.expect(!view.value(3));
}

test "is_null rejects non-none options" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeInt64Array(allocator, &[_]?i64{ 1, null });
    defer values.release();
    const args = [_]compute.Datum{
        compute.Datum.fromArray(values.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }

    try std.testing.expectError(
        error.InvalidOptions,
        ctx.invokeVector("is_null", args[0..], .{ .filter = .{} }),
    );
}

test "true_unless_null returns true for non-null and false for null" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeBoolArray(allocator, &[_]?bool{ true, false, null, true });
    defer values.release();
    const args = [_]compute.Datum{
        compute.Datum.fromArray(values.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }

    var out = try ctx.invokeVector("true_unless_null", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .bool = {} }));

    const view = zcore.BooleanArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 4), view.len());
    try std.testing.expect(view.value(0));
    try std.testing.expect(view.value(1));
    try std.testing.expect(!view.value(2));
    try std.testing.expect(view.value(3));
}

test "if_else supports fixed-width with scalar broadcast and condition null semantics" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var cond = try makeBoolArray(allocator, &[_]?bool{ true, false, null, true, false });
    defer cond.release();
    var lhs = try makeInt64Array(allocator, &[_]?i64{ 1, null, 3, 4, 5 });
    defer lhs.release();
    const args = [_]compute.Datum{
        compute.Datum.fromArray(cond.retain()),
        compute.Datum.fromArray(lhs.retain()),
        compute.Datum.fromScalar(.{
            .data_type = .{ .int64 = {} },
            .value = .{ .i64 = 9 },
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
    defer {
        var d = args[2];
        d.release();
    }

    var out = try ctx.invokeVector("if_else", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .int64 = {} }));

    const view = zcore.Int64Array{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 5), view.len());
    try std.testing.expectEqual(@as(i64, 1), view.value(0));
    try std.testing.expectEqual(@as(i64, 9), view.value(1));
    try std.testing.expect(view.isNull(2));
    try std.testing.expectEqual(@as(i64, 4), view.value(3));
    try std.testing.expectEqual(@as(i64, 9), view.value(4));
}

test "if_else supports string values and null propagation from selected branch" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var cond = try makeBoolArray(allocator, &[_]?bool{ false, true, null });
    defer cond.release();
    var lhs = try makeStringArray(allocator, &[_]?[]const u8{ "L0", null, "L2" });
    defer lhs.release();
    const args = [_]compute.Datum{
        compute.Datum.fromArray(cond.retain()),
        compute.Datum.fromArray(lhs.retain()),
        compute.Datum.fromScalar(.{
            .data_type = .{ .string = {} },
            .value = .{ .string = "R" },
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
    defer {
        var d = args[2];
        d.release();
    }

    var out = try ctx.invokeVector("if_else", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .string = {} }));

    const view = zcore.StringArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), view.len());
    try std.testing.expect(std.mem.eql(u8, view.value(0), "R"));
    try std.testing.expect(view.isNull(1));
    try std.testing.expect(view.isNull(2));
}

test "if_else supports bool values with branch and condition null propagation" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var cond = try makeBoolArray(allocator, &[_]?bool{ true, false, true, false, null });
    defer cond.release();
    var lhs = try makeBoolArray(allocator, &[_]?bool{ true, null, false, true, true });
    defer lhs.release();
    var rhs = try makeBoolArray(allocator, &[_]?bool{ false, true, null, null, false });
    defer rhs.release();
    const args = [_]compute.Datum{
        compute.Datum.fromArray(cond.retain()),
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
    defer {
        var d = args[2];
        d.release();
    }

    var out = try ctx.invokeVector("if_else", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .bool = {} }));

    const view = zcore.BooleanArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 5), view.len());
    try std.testing.expectEqual(true, view.value(0));
    try std.testing.expectEqual(true, view.value(1));
    try std.testing.expectEqual(false, view.value(2));
    try std.testing.expect(view.isNull(3));
    try std.testing.expect(view.isNull(4));
}

test "if_else supports null type" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var cond = try makeBoolArray(allocator, &[_]?bool{ true, false, null, true });
    defer cond.release();
    var lhs = try makeNullArray(allocator, 4);
    defer lhs.release();
    const args = [_]compute.Datum{
        compute.Datum.fromArray(cond.retain()),
        compute.Datum.fromArray(lhs.retain()),
        compute.Datum.fromScalar(.{
            .data_type = .{ .null = {} },
            .value = .null,
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
    defer {
        var d = args[2];
        d.release();
    }

    var out = try ctx.invokeVector("if_else", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .null = {} }));

    const view = zcore.NullArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 4), view.len());
    try std.testing.expect(view.isNull(0));
    try std.testing.expect(view.isNull(1));
    try std.testing.expect(view.isNull(2));
    try std.testing.expect(view.isNull(3));
}

test "if_else supports struct values with parent null propagation" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var cond = try makeBoolArray(allocator, &[_]?bool{ true, false, null, true });
    defer cond.release();
    var lhs = try makeStructI64BoolArray(
        allocator,
        &[_]bool{ true, true, true, false },
        &[_]?i64{ 1, 2, 3, 4 },
        &[_]?bool{ true, false, true, false },
    );
    defer lhs.release();
    var rhs = try makeStructI64BoolArray(
        allocator,
        &[_]bool{ true, false, true, true },
        &[_]?i64{ 10, 20, 30, 40 },
        &[_]?bool{ false, true, false, true },
    );
    defer rhs.release();
    const args = [_]compute.Datum{
        compute.Datum.fromArray(cond.retain()),
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
    defer {
        var d = args[2];
        d.release();
    }

    var out = try ctx.invokeVector("if_else", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType() == .struct_);

    const out_struct = zcore.StructArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 4), out_struct.len());
    try std.testing.expect(!out_struct.isNull(0));
    try std.testing.expect(out_struct.isNull(1));
    try std.testing.expect(out_struct.isNull(2));
    try std.testing.expect(out_struct.isNull(3));

    const out_i64 = zcore.Int64Array{ .data = out_struct.fieldRef(0).data() };
    const out_bool = zcore.BooleanArray{ .data = out_struct.fieldRef(1).data() };
    try std.testing.expectEqual(@as(i64, 1), out_i64.value(0));
    try std.testing.expectEqual(true, out_bool.value(0));
}

test "if_else rejects struct with unsupported list child at dispatch" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var cond = try makeBoolArray(allocator, &[_]?bool{ true, false });
    defer cond.release();
    var list_child = try makeListInt32Array(allocator);
    defer list_child.release();
    var lhs = try makeStructListI32Array(allocator, list_child, &[_]bool{ true, true });
    defer lhs.release();
    var rhs = try makeStructListI32Array(allocator, list_child, &[_]bool{ true, true });
    defer rhs.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(cond.retain()),
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
    defer {
        var d = args[2];
        d.release();
    }

    try std.testing.expectError(
        error.NoMatchingKernel,
        ctx.invokeVector("if_else", args[0..], compute.Options.noneValue()),
    );
}

test "if_else supports misaligned chunk boundaries across three inputs" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var cond_c0 = try makeBoolArray(allocator, &[_]?bool{true});
    defer cond_c0.release();
    var cond_c1 = try makeBoolArray(allocator, &[_]?bool{ false, null, true });
    defer cond_c1.release();
    var cond_chunked = try compute.ChunkedArray.init(allocator, .{ .bool = {} }, &[_]zcore.ArrayRef{ cond_c0, cond_c1 });
    defer cond_chunked.release();

    var lhs_c0 = try makeInt32Array(allocator, &[_]?i32{ 10, 11 });
    defer lhs_c0.release();
    var lhs_c1 = try makeInt32Array(allocator, &[_]?i32{ 12, 13 });
    defer lhs_c1.release();
    var lhs_chunked = try compute.ChunkedArray.init(allocator, .{ .int32 = {} }, &[_]zcore.ArrayRef{ lhs_c0, lhs_c1 });
    defer lhs_chunked.release();

    var rhs_c0 = try makeInt32Array(allocator, &[_]?i32{20});
    defer rhs_c0.release();
    var rhs_c1 = try makeInt32Array(allocator, &[_]?i32{ 21, 22, 23 });
    defer rhs_c1.release();
    var rhs_chunked = try compute.ChunkedArray.init(allocator, .{ .int32 = {} }, &[_]zcore.ArrayRef{ rhs_c0, rhs_c1 });
    defer rhs_chunked.release();

    const args = [_]compute.Datum{
        compute.Datum.fromChunked(cond_chunked.retain()),
        compute.Datum.fromChunked(lhs_chunked.retain()),
        compute.Datum.fromChunked(rhs_chunked.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }
    defer {
        var d = args[2];
        d.release();
    }

    var out = try ctx.invokeVector("if_else", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .int32 = {} }));

    const view = zcore.Int32Array{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 4), view.len());
    try std.testing.expectEqual(@as(i32, 10), try view.value(0));
    try std.testing.expectEqual(@as(i32, 21), try view.value(1));
    try std.testing.expect(view.isNull(2));
    try std.testing.expectEqual(@as(i32, 13), try view.value(3));
}

test "if_else rejects unsupported nested value type at dispatch" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var cond = try makeBoolArray(allocator, &[_]?bool{ true, false });
    defer cond.release();
    var values = try makeListInt32Array(allocator);
    defer values.release();
    const args = [_]compute.Datum{
        compute.Datum.fromArray(cond.retain()),
        compute.Datum.fromArray(values.retain()),
        compute.Datum.fromArray(values.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }
    defer {
        var d = args[2];
        d.release();
    }

    try std.testing.expectError(
        error.NoMatchingKernel,
        ctx.invokeVector("if_else", args[0..], compute.Options.noneValue()),
    );
}

test "if_else rejects non-none options" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var cond = try makeBoolArray(allocator, &[_]?bool{ true, false });
    defer cond.release();
    var lhs = try makeInt64Array(allocator, &[_]?i64{ 1, 2 });
    defer lhs.release();
    var rhs = try makeInt64Array(allocator, &[_]?i64{ 3, 4 });
    defer rhs.release();
    const args = [_]compute.Datum{
        compute.Datum.fromArray(cond.retain()),
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
    defer {
        var d = args[2];
        d.release();
    }

    try std.testing.expectError(
        error.InvalidOptions,
        ctx.invokeVector("if_else", args[0..], .{ .filter = .{} }),
    );
}

test "coalesce supports variadic scalar broadcast and first-non-null semantics" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var primary = try makeInt64Array(allocator, &[_]?i64{ null, 2, null, 4 });
    defer primary.release();
    var backup = try makeInt64Array(allocator, &[_]?i64{ 7, null, 8, null });
    defer backup.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(primary.retain()),
        compute.Datum.fromScalar(.{
            .data_type = .{ .int64 = {} },
            .value = .{ .i64 = 9 },
        }),
        compute.Datum.fromArray(backup.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }
    defer {
        var d = args[2];
        d.release();
    }

    var out = try ctx.invokeVector("coalesce", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .int64 = {} }));

    const view = zcore.Int64Array{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 4), view.len());
    try std.testing.expectEqual(@as(i64, 9), view.value(0));
    try std.testing.expectEqual(@as(i64, 2), view.value(1));
    try std.testing.expectEqual(@as(i64, 9), view.value(2));
    try std.testing.expectEqual(@as(i64, 4), view.value(3));
}

test "coalesce outputs null only when all candidates are null" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var lhs = try makeStringArray(allocator, &[_]?[]const u8{ null, "x", null });
    defer lhs.release();
    var rhs = try makeStringArray(allocator, &[_]?[]const u8{ null, null, "y" });
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

    var out = try ctx.invokeVector("coalesce", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .string = {} }));

    const view = zcore.StringArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), view.len());
    try std.testing.expect(view.isNull(0));
    try std.testing.expect(std.mem.eql(u8, view.value(1), "x"));
    try std.testing.expect(std.mem.eql(u8, view.value(2), "y"));
}

test "choose supports variadic value selection with null propagation" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var indices = try makeInt32Array(allocator, &[_]?i32{ 0, 1, null, 1, 2 });
    defer indices.release();
    var v0 = try makeInt64Array(allocator, &[_]?i64{ 10, 11, 12, 13, 14 });
    defer v0.release();
    var v1 = try makeInt64Array(allocator, &[_]?i64{ 20, null, 22, 23, 24 });
    defer v1.release();
    const args = [_]compute.Datum{
        compute.Datum.fromArray(indices.retain()),
        compute.Datum.fromArray(v0.retain()),
        compute.Datum.fromArray(v1.retain()),
        compute.Datum.fromScalar(.{
            .data_type = .{ .int64 = {} },
            .value = .{ .i64 = 99 },
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
    defer {
        var d = args[2];
        d.release();
    }
    defer {
        var d = args[3];
        d.release();
    }

    var out = try ctx.invokeVector("choose", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .int64 = {} }));

    const view = zcore.Int64Array{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 5), view.len());
    try std.testing.expectEqual(@as(i64, 10), view.value(0));
    try std.testing.expect(view.isNull(1));
    try std.testing.expect(view.isNull(2));
    try std.testing.expectEqual(@as(i64, 23), view.value(3));
    try std.testing.expectEqual(@as(i64, 99), view.value(4));
}

test "choose rejects out-of-bounds index" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var indices = try makeInt32Array(allocator, &[_]?i32{ 0, 3 });
    defer indices.release();
    var v0 = try makeInt64Array(allocator, &[_]?i64{ 1, 2 });
    defer v0.release();
    var v1 = try makeInt64Array(allocator, &[_]?i64{ 4, 5 });
    defer v1.release();
    const args = [_]compute.Datum{
        compute.Datum.fromArray(indices.retain()),
        compute.Datum.fromArray(v0.retain()),
        compute.Datum.fromArray(v1.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }
    defer {
        var d = args[2];
        d.release();
    }

    try std.testing.expectError(
        error.InvalidInput,
        ctx.invokeVector("choose", args[0..], compute.Options.noneValue()),
    );
}

test "case_when supports Arrow struct<bool...> conditions with optional else" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var cond0 = try makeBoolArray(allocator, &[_]?bool{ false, true, null, false, true });
    defer cond0.release();
    var cond1 = try makeBoolArray(allocator, &[_]?bool{ true, false, true, null, true });
    defer cond1.release();
    var conds = try makeStructBool2Array(allocator, cond0, cond1);
    defer conds.release();
    var v0 = try makeInt64Array(allocator, &[_]?i64{ 1, 1, 1, 1, null });
    defer v0.release();
    var v1 = try makeInt64Array(allocator, &[_]?i64{ 2, 2, null, 2, 2 });
    defer v1.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(conds.retain()),
        compute.Datum.fromArray(v0.retain()),
        compute.Datum.fromArray(v1.retain()),
        compute.Datum.fromScalar(.{
            .data_type = .{ .int64 = {} },
            .value = .{ .i64 = 9 },
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
    defer {
        var d = args[2];
        d.release();
    }
    defer {
        var d = args[3];
        d.release();
    }

    var out = try ctx.invokeVector("case_when", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .int64 = {} }));

    const view = zcore.Int64Array{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 5), view.len());
    try std.testing.expectEqual(@as(i64, 2), view.value(0));
    try std.testing.expectEqual(@as(i64, 1), view.value(1));
    try std.testing.expect(view.isNull(2));
    try std.testing.expectEqual(@as(i64, 9), view.value(3));
    try std.testing.expect(view.isNull(4));
}

test "case_when struct<bool...> without else falls back to null" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var cond0 = try makeBoolArray(allocator, &[_]?bool{ false, null, true });
    defer cond0.release();
    var cond1 = try makeBoolArray(allocator, &[_]?bool{ false, true, false });
    defer cond1.release();
    var conds = try makeStructBool2Array(allocator, cond0, cond1);
    defer conds.release();
    var v0 = try makeStringArray(allocator, &[_]?[]const u8{ "A", null, "C" });
    defer v0.release();
    var v1 = try makeStringArray(allocator, &[_]?[]const u8{ "B", "B", null });
    defer v1.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(conds.retain()),
        compute.Datum.fromArray(v0.retain()),
        compute.Datum.fromArray(v1.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }
    defer {
        var d = args[2];
        d.release();
    }

    var out = try ctx.invokeVector("case_when", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .string = {} }));

    const view = zcore.StringArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), view.len());
    try std.testing.expect(view.isNull(0));
    try std.testing.expect(std.mem.eql(u8, view.value(1), "B"));
    try std.testing.expect(std.mem.eql(u8, view.value(2), "C"));
}

test "case_when struct<bool...> rejects mismatched cases arity" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var cond0 = try makeBoolArray(allocator, &[_]?bool{ true, false });
    defer cond0.release();
    var cond1 = try makeBoolArray(allocator, &[_]?bool{ false, true });
    defer cond1.release();
    var conds = try makeStructBool2Array(allocator, cond0, cond1);
    defer conds.release();
    var v0 = try makeInt64Array(allocator, &[_]?i64{ 1, 2 });
    defer v0.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(conds.retain()),
        compute.Datum.fromArray(v0.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    try std.testing.expectError(
        error.NoMatchingKernel,
        ctx.invokeVector("case_when", args[0..], compute.Options.noneValue()),
    );
}

test "case_when rejects legacy cond-value pair signature" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var cond0 = try makeBoolArray(allocator, &[_]?bool{ false, true, null, false, true });
    defer cond0.release();
    var v0 = try makeInt64Array(allocator, &[_]?i64{ 1, 1, 1, 1, null });
    defer v0.release();
    var cond1 = try makeBoolArray(allocator, &[_]?bool{ true, false, true, null, true });
    defer cond1.release();
    var v1 = try makeInt64Array(allocator, &[_]?i64{ 2, 2, null, 2, 2 });
    defer v1.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(cond0.retain()),
        compute.Datum.fromArray(v0.retain()),
        compute.Datum.fromArray(cond1.retain()),
        compute.Datum.fromArray(v1.retain()),
        compute.Datum.fromScalar(.{
            .data_type = .{ .int64 = {} },
            .value = .{ .i64 = 9 },
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
    defer {
        var d = args[2];
        d.release();
    }
    defer {
        var d = args[3];
        d.release();
    }
    defer {
        var d = args[4];
        d.release();
    }

    try std.testing.expectError(
        error.NoMatchingKernel,
        ctx.invokeVector("case_when", args[0..], compute.Options.noneValue()),
    );
}

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
