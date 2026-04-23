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
