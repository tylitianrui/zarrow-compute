const std = @import("std");
const zcore = @import("zarrow-core");
const zcompute = @import("zarrow_compute");

const compute = zcore.compute;
const DT_BOOL = zcore.DataType{ .bool = {} };
const DT_I32 = zcore.DataType{ .int32 = {} };
const STRUCT_FIELDS_BOOL2 = [_]zcore.Field{
    .{ .name = "c0", .data_type = &DT_BOOL, .nullable = true },
    .{ .name = "c1", .data_type = &DT_BOOL, .nullable = true },
};
const FIELD_LIST_ITEM_I32 = zcore.Field{
    .name = "item",
    .data_type = &DT_I32,
    .nullable = true,
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

fn makeFixedSizeListInt32Array(
    allocator: std.mem.Allocator,
    list_size: usize,
    present: []const bool,
    values: []const i32,
) !zcore.ArrayRef {
    const expected_values_len = std.math.mul(usize, present.len, list_size) catch return error.InvalidInput;
    if (expected_values_len != values.len) return error.InvalidInput;

    var values_builder = try zcore.Int32Builder.init(allocator, values.len);
    defer values_builder.deinit();
    for (values) |value| {
        try values_builder.append(value);
    }
    var values_ref = try values_builder.finish();
    defer values_ref.release();

    var builder = try zcore.FixedSizeListBuilder.init(allocator, FIELD_LIST_ITEM_I32, list_size);
    defer builder.deinit();
    for (present) |is_present| {
        if (is_present) {
            try builder.appendValid();
        } else {
            try builder.appendNull();
        }
    }
    return builder.finish(values_ref);
}

fn printInt64DatumLine(label: []const u8, datum: compute.Datum) !void {
    if (!datum.isArray()) return error.InvalidInput;
    const view = zcore.Int64Array{ .data = datum.array.data() };
    std.debug.print("{s} => [", .{label});
    var i: usize = 0;
    while (i < view.len()) : (i += 1) {
        if (i != 0) std.debug.print(", ", .{});
        if (view.isNull(i)) {
            std.debug.print("null", .{});
        } else {
            const value = view.value(i) catch return error.InvalidInput;
            std.debug.print("{d}", .{value});
        }
    }
    std.debug.print("]\n", .{});
}

fn printBoolDatumLine(label: []const u8, datum: compute.Datum) !void {
    if (!datum.isArray()) return error.InvalidInput;
    const view = zcore.BooleanArray{ .data = datum.array.data() };
    std.debug.print("{s} => [", .{label});
    var i: usize = 0;
    while (i < view.len()) : (i += 1) {
        if (i != 0) std.debug.print(", ", .{});
        if (view.isNull(i)) {
            std.debug.print("null", .{});
        } else {
            std.debug.print("{}", .{view.value(i)});
        }
    }
    std.debug.print("]\n", .{});
}

fn printFixedSizeListInt32DatumLine(label: []const u8, datum: compute.Datum) !void {
    if (!datum.isArray()) return error.InvalidInput;
    if (datum.dataType() != .fixed_size_list) return error.InvalidInput;

    const list_view = zcore.FixedSizeListArray{ .data = datum.array.data() };
    std.debug.print("{s} => [", .{label});
    var row: usize = 0;
    while (row < list_view.len()) : (row += 1) {
        if (row != 0) std.debug.print(", ", .{});
        if (list_view.isNull(row)) {
            std.debug.print("null", .{});
            continue;
        }

        var row_ref = try list_view.value(row);
        defer row_ref.release();
        const row_i32 = zcore.Int32Array{ .data = row_ref.data() };
        std.debug.print("[", .{});
        var col: usize = 0;
        while (col < row_i32.len()) : (col += 1) {
            if (col != 0) std.debug.print(", ", .{});
            if (row_i32.isNull(col)) {
                std.debug.print("null", .{});
            } else {
                const value = row_i32.value(col) catch return error.InvalidInput;
                std.debug.print("{d}", .{value});
            }
        }
        std.debug.print("]", .{});
    }
    std.debug.print("]\n", .{});
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try zcompute.registerBaseKernels(&registry);

    var ctx = compute.ExecContext.init(allocator, &registry);
    var left = try makeInt64Array(allocator, &[_]?i64{ 1, null, 3 });
    defer left.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(left.retain()),
        compute.Datum.fromScalar(.{
            .data_type = .{ .int64 = {} },
            .value = .{ .i64 = 5 },
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

    var add_out = try ctx.invokeVector("add_i64", args[0..], .{ .arithmetic = .{} });
    defer add_out.release();
    try printInt64DatumLine("add_i64", add_out);

    var subtract_out = try ctx.invokeVector("subtract_i64", args[0..], .{ .arithmetic = .{} });
    defer subtract_out.release();
    try printInt64DatumLine("subtract_i64", subtract_out);

    var multiply_out = try ctx.invokeVector("multiply_i64", args[0..], .{ .arithmetic = .{} });
    defer multiply_out.release();
    try printInt64DatumLine("multiply_i64", multiply_out);

    var predicate = try makeBoolArray(allocator, &[_]?bool{ true, true, null });
    defer predicate.release();
    const filter_args = [_]compute.Datum{
        compute.Datum.fromArray(left.retain()),
        compute.Datum.fromArray(predicate.retain()),
    };
    defer {
        var d = filter_args[0];
        d.release();
    }
    defer {
        var d = filter_args[1];
        d.release();
    }
    var filter_out = try ctx.invokeVector("filter", filter_args[0..], .{
        .filter = .{ .drop_nulls = false },
    });
    defer filter_out.release();
    try printInt64DatumLine("filter", filter_out);

    const drop_null_args = [_]compute.Datum{
        compute.Datum.fromArray(left.retain()),
    };
    defer {
        var d = drop_null_args[0];
        d.release();
    }
    var drop_null_out = try ctx.invokeVector("drop_null", drop_null_args[0..], compute.Options.noneValue());
    defer drop_null_out.release();
    try printInt64DatumLine("drop_null", drop_null_out);

    var is_null_out = try ctx.invokeVector("is_null", drop_null_args[0..], compute.Options.noneValue());
    defer is_null_out.release();
    try printBoolDatumLine("is_null", is_null_out);

    var is_valid_out = try ctx.invokeVector("is_valid", drop_null_args[0..], compute.Options.noneValue());
    defer is_valid_out.release();
    try printBoolDatumLine("is_valid", is_valid_out);

    var true_unless_null_out = try ctx.invokeVector("true_unless_null", drop_null_args[0..], compute.Options.noneValue());
    defer true_unless_null_out.release();
    try printBoolDatumLine("true_unless_null", true_unless_null_out);

    var cond = try makeBoolArray(allocator, &[_]?bool{ true, null, false });
    defer cond.release();
    const if_else_args = [_]compute.Datum{
        compute.Datum.fromArray(cond.retain()),
        compute.Datum.fromArray(left.retain()),
        compute.Datum.fromScalar(.{
            .data_type = .{ .int64 = {} },
            .value = .{ .i64 = 10 },
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
    var if_else_out = try ctx.invokeVector("if_else", if_else_args[0..], compute.Options.noneValue());
    defer if_else_out.release();
    try printInt64DatumLine("if_else", if_else_out);

    const coalesce_args = [_]compute.Datum{
        compute.Datum.fromArray(left.retain()),
        compute.Datum.fromScalar(.{
            .data_type = .{ .int64 = {} },
            .value = .{ .i64 = 7 },
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
    var coalesce_out = try ctx.invokeVector("coalesce", coalesce_args[0..], compute.Options.noneValue());
    defer coalesce_out.release();
    try printInt64DatumLine("coalesce", coalesce_out);

    var choose_indices = try makeInt32Array(allocator, &[_]?i32{ 0, 1, null });
    defer choose_indices.release();
    var choose_values1 = try makeInt64Array(allocator, &[_]?i64{ 100, null, 300 });
    defer choose_values1.release();
    const choose_args = [_]compute.Datum{
        compute.Datum.fromArray(choose_indices.retain()),
        compute.Datum.fromArray(left.retain()),
        compute.Datum.fromArray(choose_values1.retain()),
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
    var choose_out = try ctx.invokeVector("choose", choose_args[0..], compute.Options.noneValue());
    defer choose_out.release();
    try printInt64DatumLine("choose", choose_out);

    var case_cond0 = try makeBoolArray(allocator, &[_]?bool{ false, true, false });
    defer case_cond0.release();
    var case_cond1 = try makeBoolArray(allocator, &[_]?bool{ true, false, null });
    defer case_cond1.release();
    var case_conds = try makeStructBool2Array(allocator, case_cond0, case_cond1);
    defer case_conds.release();
    const case_when_args = [_]compute.Datum{
        compute.Datum.fromArray(case_conds.retain()),
        compute.Datum.fromArray(left.retain()),
        compute.Datum.fromScalar(.{
            .data_type = .{ .int64 = {} },
            .value = .{ .i64 = 10 },
        }),
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
    var case_when_out = try ctx.invokeVector("case_when", case_when_args[0..], compute.Options.noneValue());
    defer case_when_out.release();
    try printInt64DatumLine("case_when", case_when_out);

    var fs_a = try makeFixedSizeListInt32Array(
        allocator,
        2,
        &[_]bool{ true, true, true, true },
        &[_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 },
    );
    defer fs_a.release();
    var fs_b = try makeFixedSizeListInt32Array(
        allocator,
        2,
        &[_]bool{ true, false, true, true },
        &[_]i32{ 11, 12, 13, 14, 15, 16, 17, 18 },
    );
    defer fs_b.release();
    var fs_else = try makeFixedSizeListInt32Array(
        allocator,
        2,
        &[_]bool{ true, true, false, true },
        &[_]i32{ 21, 22, 23, 24, 25, 26, 27, 28 },
    );
    defer fs_else.release();

    var fs_filter_pred = try makeBoolArray(allocator, &[_]?bool{ true, null, true, false });
    defer fs_filter_pred.release();
    const fs_filter_args = [_]compute.Datum{
        compute.Datum.fromArray(fs_a.retain()),
        compute.Datum.fromArray(fs_filter_pred.retain()),
    };
    defer {
        var d = fs_filter_args[0];
        d.release();
    }
    defer {
        var d = fs_filter_args[1];
        d.release();
    }
    var fs_filter_out = try ctx.invokeVector("filter", fs_filter_args[0..], .{
        .filter = .{ .drop_nulls = false },
    });
    defer fs_filter_out.release();
    try printFixedSizeListInt32DatumLine("filter_fixed_size_list", fs_filter_out);

    const fs_drop_null_args = [_]compute.Datum{
        compute.Datum.fromArray(fs_b.retain()),
    };
    defer {
        var d = fs_drop_null_args[0];
        d.release();
    }
    var fs_drop_null_out = try ctx.invokeVector("drop_null", fs_drop_null_args[0..], compute.Options.noneValue());
    defer fs_drop_null_out.release();
    try printFixedSizeListInt32DatumLine("drop_null_fixed_size_list", fs_drop_null_out);

    var fs_if_else_cond = try makeBoolArray(allocator, &[_]?bool{ true, false, null, true });
    defer fs_if_else_cond.release();
    const fs_if_else_args = [_]compute.Datum{
        compute.Datum.fromArray(fs_if_else_cond.retain()),
        compute.Datum.fromArray(fs_a.retain()),
        compute.Datum.fromArray(fs_b.retain()),
    };
    defer {
        var d = fs_if_else_args[0];
        d.release();
    }
    defer {
        var d = fs_if_else_args[1];
        d.release();
    }
    defer {
        var d = fs_if_else_args[2];
        d.release();
    }
    var fs_if_else_out = try ctx.invokeVector("if_else", fs_if_else_args[0..], compute.Options.noneValue());
    defer fs_if_else_out.release();
    try printFixedSizeListInt32DatumLine("if_else_fixed_size_list", fs_if_else_out);

    const fs_coalesce_args = [_]compute.Datum{
        compute.Datum.fromArray(fs_b.retain()),
        compute.Datum.fromArray(fs_else.retain()),
    };
    defer {
        var d = fs_coalesce_args[0];
        d.release();
    }
    defer {
        var d = fs_coalesce_args[1];
        d.release();
    }
    var fs_coalesce_out = try ctx.invokeVector("coalesce", fs_coalesce_args[0..], compute.Options.noneValue());
    defer fs_coalesce_out.release();
    try printFixedSizeListInt32DatumLine("coalesce_fixed_size_list", fs_coalesce_out);

    var fs_choose_indices = try makeInt32Array(allocator, &[_]?i32{ 0, 1, null, 2 });
    defer fs_choose_indices.release();
    const fs_choose_args = [_]compute.Datum{
        compute.Datum.fromArray(fs_choose_indices.retain()),
        compute.Datum.fromArray(fs_a.retain()),
        compute.Datum.fromArray(fs_b.retain()),
        compute.Datum.fromArray(fs_else.retain()),
    };
    defer {
        var d = fs_choose_args[0];
        d.release();
    }
    defer {
        var d = fs_choose_args[1];
        d.release();
    }
    defer {
        var d = fs_choose_args[2];
        d.release();
    }
    defer {
        var d = fs_choose_args[3];
        d.release();
    }
    var fs_choose_out = try ctx.invokeVector("choose", fs_choose_args[0..], compute.Options.noneValue());
    defer fs_choose_out.release();
    try printFixedSizeListInt32DatumLine("choose_fixed_size_list", fs_choose_out);

    var fs_case_cond0 = try makeBoolArray(allocator, &[_]?bool{ true, false, false, null });
    defer fs_case_cond0.release();
    var fs_case_cond1 = try makeBoolArray(allocator, &[_]?bool{ false, true, false, true });
    defer fs_case_cond1.release();
    var fs_case_conds = try makeStructBool2Array(allocator, fs_case_cond0, fs_case_cond1);
    defer fs_case_conds.release();
    const fs_case_when_args = [_]compute.Datum{
        compute.Datum.fromArray(fs_case_conds.retain()),
        compute.Datum.fromArray(fs_a.retain()),
        compute.Datum.fromArray(fs_b.retain()),
        compute.Datum.fromArray(fs_else.retain()),
    };
    defer {
        var d = fs_case_when_args[0];
        d.release();
    }
    defer {
        var d = fs_case_when_args[1];
        d.release();
    }
    defer {
        var d = fs_case_when_args[2];
        d.release();
    }
    defer {
        var d = fs_case_when_args[3];
        d.release();
    }
    var fs_case_when_out = try ctx.invokeVector("case_when", fs_case_when_args[0..], compute.Options.noneValue());
    defer fs_case_when_out.release();
    try printFixedSizeListInt32DatumLine("case_when_fixed_size_list", fs_case_when_out);

    var count = try ctx.invokeAggregate("count_rows", args[0..1], compute.Options.noneValue());
    defer count.release();
    std.debug.print("count_rows => {d}\n", .{count.scalar.value.i64});
}
