const std = @import("std");
const zcore = @import("zarrow-core");
const zcompute = @import("zarrow_compute");

const compute = zcore.compute;

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

    const bool_type_0 = zcore.DataType{ .bool = {} };
    const bool_type_1 = zcore.DataType{ .bool = {} };
    const fields = &[_]zcore.Field{
        .{ .name = "c0", .data_type = &bool_type_0, .nullable = true },
        .{ .name = "c1", .data_type = &bool_type_1, .nullable = true },
    };

    var builder = zcore.StructBuilder.init(allocator, fields);
    defer builder.deinit();
    var row: usize = 0;
    while (row < cond0.data().length) : (row += 1) {
        try builder.appendValid();
    }

    return builder.finish(&[_]zcore.ArrayRef{ cond0, cond1 });
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

    var count = try ctx.invokeAggregate("count_rows", args[0..1], compute.Options.noneValue());
    defer count.release();
    std.debug.print("count_rows => {d}\n", .{count.scalar.value.i64});
}
