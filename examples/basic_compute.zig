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

    var count = try ctx.invokeAggregate("count_rows", args[0..1], compute.Options.noneValue());
    defer count.release();
    std.debug.print("count_rows => {d}\n", .{count.scalar.value.i64});
}
