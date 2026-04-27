const std = @import("std");
const common = @import("common.zig");

const compute = common.compute;

pub fn countRowsResultType(args: []const compute.Datum, options: compute.Options) compute.KernelError!compute.DataType {
    _ = options;
    if (args.len != 1) return error.InvalidArity;
    return .{ .int64 = {} };
}

fn countRows(datum: compute.Datum) compute.KernelError!usize {
    return switch (datum) {
        .array => |arr| arr.data().length,
        .chunked => |chunks| chunks.len(),
        .scalar => error.InvalidInput,
    };
}

fn countNonNull(datum: compute.Datum) compute.KernelError!usize {
    var iter = compute.UnaryExecChunkIterator.init(datum);
    var count: usize = 0;
    while (try iter.next()) |chunk_value| {
        var chunk = chunk_value;
        defer chunk.deinit();
        var i: usize = 0;
        while (i < chunk.len) : (i += 1) {
            if (!chunk.unaryNullAt(i)) {
                count = std.math.add(usize, count, 1) catch return error.Overflow;
            }
        }
    }
    return count;
}

pub fn countRowsKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    _ = ctx;
    if (!common.onlyNoOptions(options)) return error.InvalidOptions;
    if (!common.unaryArrayLike(args)) return error.InvalidInput;
    const row_count = try countRows(args[0]);
    return compute.Datum.fromScalar(.{
        .data_type = .{ .int64 = {} },
        .value = .{ .i64 = @intCast(row_count) },
    });
}

pub fn countResultType(args: []const compute.Datum, options: compute.Options) compute.KernelError!compute.DataType {
    _ = options;
    if (args.len != 1) return error.InvalidArity;
    return .{ .int64 = {} };
}

pub fn countKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    _ = ctx;
    if (!common.onlyNoOptions(options)) return error.InvalidOptions;
    if (!common.unaryArrayLike(args)) return error.InvalidInput;
    const count = try countNonNull(args[0]);
    return compute.Datum.fromScalar(.{
        .data_type = .{ .int64 = {} },
        .value = .{ .i64 = @intCast(count) },
    });
}

pub fn meanResultType(args: []const compute.Datum, options: compute.Options) compute.KernelError!compute.DataType {
    _ = options;
    if (args.len != 1) return error.InvalidArity;
    return .{ .double = {} };
}

pub fn sumMinMaxMeanKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
    comptime mode: enum { sum, min, max, mean },
) compute.KernelError!compute.Datum {
    _ = ctx;
    if (!common.onlyNoOptions(options)) return error.InvalidOptions;
    if (!common.unaryInt64ArrayLike(args)) return error.InvalidInput;

    var seen: bool = false;
    var sum_i64: i64 = 0;
    var min_i64: i64 = 0;
    var max_i64: i64 = 0;
    var count: usize = 0;

    var iter = compute.UnaryExecChunkIterator.init(args[0]);
    while (try iter.next()) |chunk_value| {
        var chunk = chunk_value;
        defer chunk.deinit();
        var i: usize = 0;
        while (i < chunk.len) : (i += 1) {
            if (chunk.unaryNullAt(i)) continue;
            const value = try common.readI64(chunk.values, i);
            if (!seen) {
                seen = true;
                sum_i64 = value;
                min_i64 = value;
                max_i64 = value;
                count = 1;
                continue;
            }
            sum_i64 = std.math.add(i64, sum_i64, value) catch return error.Overflow;
            if (value < min_i64) min_i64 = value;
            if (value > max_i64) max_i64 = value;
            count = std.math.add(usize, count, 1) catch return error.Overflow;
        }
    }

    return switch (mode) {
        .sum => compute.Datum.fromScalar(.{
            .data_type = .{ .int64 = {} },
            .value = if (seen) .{ .i64 = sum_i64 } else .null,
        }),
        .min => compute.Datum.fromScalar(.{
            .data_type = .{ .int64 = {} },
            .value = if (seen) .{ .i64 = min_i64 } else .null,
        }),
        .max => compute.Datum.fromScalar(.{
            .data_type = .{ .int64 = {} },
            .value = if (seen) .{ .i64 = max_i64 } else .null,
        }),
        .mean => compute.Datum.fromScalar(.{
            .data_type = .{ .double = {} },
            .value = if (seen) .{ .f64 = @as(f64, @floatFromInt(sum_i64)) / @as(f64, @floatFromInt(count)) } else .null,
        }),
    };
}

pub fn sumKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    return sumMinMaxMeanKernel(ctx, args, options, .sum);
}

pub fn minKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    return sumMinMaxMeanKernel(ctx, args, options, .min);
}

pub fn maxKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    return sumMinMaxMeanKernel(ctx, args, options, .max);
}

pub fn meanKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    return sumMinMaxMeanKernel(ctx, args, options, .mean);
}

const CountRowsState = struct {
    count: usize = 0,
};

pub fn countRowsInit(ctx: *compute.ExecContext, options: compute.Options) compute.KernelError!*anyopaque {
    if (!common.onlyNoOptions(options)) return error.InvalidOptions;
    const state = ctx.allocator.create(CountRowsState) catch return error.OutOfMemory;
    state.* = .{};
    return state;
}

pub fn countRowsUpdate(
    ctx: *compute.ExecContext,
    state_ptr: *anyopaque,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!void {
    _ = ctx;
    if (!common.onlyNoOptions(options)) return error.InvalidOptions;
    if (!common.unaryArrayLike(args)) return error.InvalidInput;

    const state: *CountRowsState = @ptrCast(@alignCast(state_ptr));
    const next_rows = try countRows(args[0]);
    state.count = std.math.add(usize, state.count, next_rows) catch return error.Overflow;
}

pub fn countRowsMerge(
    ctx: *compute.ExecContext,
    state_ptr: *anyopaque,
    other_ptr: *anyopaque,
    options: compute.Options,
) compute.KernelError!void {
    _ = ctx;
    if (!common.onlyNoOptions(options)) return error.InvalidOptions;
    const state: *CountRowsState = @ptrCast(@alignCast(state_ptr));
    const other: *CountRowsState = @ptrCast(@alignCast(other_ptr));
    state.count = std.math.add(usize, state.count, other.count) catch return error.Overflow;
}

pub fn countRowsFinalize(
    ctx: *compute.ExecContext,
    state_ptr: *anyopaque,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    _ = ctx;
    if (!common.onlyNoOptions(options)) return error.InvalidOptions;
    const state: *CountRowsState = @ptrCast(@alignCast(state_ptr));
    return compute.Datum.fromScalar(.{
        .data_type = .{ .int64 = {} },
        .value = .{ .i64 = @intCast(state.count) },
    });
}

pub fn countRowsDeinit(ctx: *compute.ExecContext, state_ptr: *anyopaque) void {
    const state: *CountRowsState = @ptrCast(@alignCast(state_ptr));
    ctx.allocator.destroy(state);
}
