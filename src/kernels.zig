const std = @import("std");
const zcore = @import("zarrow-core");
const impl = @import("kernels/impl.zig");

pub const compute = impl.compute;
pub const registerBaseKernels = impl.registerBaseKernels;
pub const registerCompatKernels = impl.registerCompatKernels;

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
    const value_type = zcore.DataType{ .int32 = {} };
    const field = zcore.Field{
        .name = "item",
        .data_type = &value_type,
        .nullable = true,
    };

    var values_builder = try zcore.Int32Builder.init(allocator, 3);
    defer values_builder.deinit();
    try values_builder.append(1);
    try values_builder.append(2);
    try values_builder.append(3);
    var values = try values_builder.finish();
    defer values.release();

    var list_builder = try zcore.ListBuilder.init(allocator, 2, field);
    defer list_builder.deinit();
    try list_builder.appendLen(2);
    try list_builder.appendLen(1);
    return list_builder.finish(values);
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
