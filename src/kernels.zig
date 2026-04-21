const std = @import("std");
const zcore = @import("zarrow-core");

pub const compute = zcore.compute;

fn onlyNoOptions(options: compute.Options) bool {
    return switch (options) {
        .none => true,
        else => false,
    };
}

fn onlyArithmeticOptions(options: compute.Options) bool {
    return switch (options) {
        .arithmetic => true,
        else => false,
    };
}

fn onlyCastOptions(options: compute.Options) bool {
    return switch (options) {
        .cast => true,
        else => false,
    };
}

fn unaryInt64(args: []const compute.Datum) bool {
    return args.len == 1 and args[0].dataType().eql(.{ .int64 = {} });
}

fn binaryInt64(args: []const compute.Datum) bool {
    return args.len == 2 and
        args[0].dataType().eql(.{ .int64 = {} }) and
        args[1].dataType().eql(.{ .int64 = {} });
}

fn unaryArrayLike(args: []const compute.Datum) bool {
    return args.len == 1 and (args[0].isArray() or args[0].isChunked());
}

fn resultI64(args: []const compute.Datum, options: compute.Options) compute.KernelError!compute.DataType {
    _ = options;
    if (args.len == 0) return error.InvalidArity;
    return .{ .int64 = {} };
}

fn resultI32(args: []const compute.Datum, options: compute.Options) compute.KernelError!compute.DataType {
    if (args.len != 1) return error.InvalidArity;
    return switch (options) {
        .cast => |cast_opts| blk: {
            if (cast_opts.to_type) |to_type| {
                if (!to_type.eql(.{ .int32 = {} })) return error.InvalidCast;
            }
            break :blk .{ .int32 = {} };
        },
        else => error.InvalidOptions,
    };
}

fn kernelAppendError(err: anyerror) compute.KernelError {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidInput,
    };
}

fn readI64(value: compute.ExecChunkValue, logical_index: usize) compute.KernelError!i64 {
    return switch (value) {
        .scalar => |s| switch (s.value) {
            .i64 => |v| v,
            else => error.InvalidInput,
        },
        .array => |arr| blk: {
            const dt = arr.data().data_type;
            if (dt.eql(.{ .int64 = {} })) {
                const view = zcore.Int64Array{ .data = arr.data() };
                break :blk view.value(logical_index);
            }
            if (dt.eql(.{ .int32 = {} })) {
                const view = zcore.Int32Array{ .data = arr.data() };
                break :blk @as(i64, view.value(logical_index));
            }
            break :blk error.UnsupportedType;
        },
    };
}

fn addI64Kernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    if (args.len != 2) return error.InvalidArity;
    const arithmetic_opts = switch (options) {
        .arithmetic => |o| o,
        else => return error.InvalidOptions,
    };

    const out_len = try compute.inferBinaryExecLen(args[0], args[1]);
    var builder = try zcore.Int64Builder.init(ctx.tempAllocator(), out_len);
    defer builder.deinit();

    var iter = try compute.BinaryExecChunkIterator.init(args[0], args[1]);
    while (try iter.next()) |chunk_value| {
        var chunk = chunk_value;
        defer chunk.deinit();

        var i: usize = 0;
        while (i < chunk.len) : (i += 1) {
            if (chunk.binaryNullAt(i)) {
                builder.appendNull() catch |err| return kernelAppendError(err);
                continue;
            }

            const lhs = try readI64(chunk.lhs, i);
            const rhs = try readI64(chunk.rhs, i);
            const sum = if (arithmetic_opts.check_overflow)
                std.math.add(i64, lhs, rhs) catch return error.Overflow
            else
                lhs +% rhs;

            builder.append(sum) catch |err| return kernelAppendError(err);
        }
    }

    const out = builder.finish() catch |err| return kernelAppendError(err);
    return compute.Datum.fromArray(out);
}

fn divideI64Kernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    if (args.len != 2) return error.InvalidArity;
    const arithmetic_opts = switch (options) {
        .arithmetic => |o| o,
        else => return error.InvalidOptions,
    };

    const out_len = try compute.inferBinaryExecLen(args[0], args[1]);
    var builder = try zcore.Int64Builder.init(ctx.tempAllocator(), out_len);
    defer builder.deinit();

    var iter = try compute.BinaryExecChunkIterator.init(args[0], args[1]);
    while (try iter.next()) |chunk_value| {
        var chunk = chunk_value;
        defer chunk.deinit();

        var i: usize = 0;
        while (i < chunk.len) : (i += 1) {
            if (chunk.binaryNullAt(i)) {
                builder.appendNull() catch |err| return kernelAppendError(err);
                continue;
            }

            const lhs = try readI64(chunk.lhs, i);
            const rhs = try readI64(chunk.rhs, i);
            const out = try compute.arithmeticDivI64(lhs, rhs, arithmetic_opts);

            builder.append(out) catch |err| return kernelAppendError(err);
        }
    }

    const out = builder.finish() catch |err| return kernelAppendError(err);
    return compute.Datum.fromArray(out);
}

fn castI64ToI32Kernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    if (args.len != 1) return error.InvalidArity;
    const cast_opts = switch (options) {
        .cast => |o| o,
        else => return error.InvalidOptions,
    };
    if (cast_opts.to_type) |to_type| {
        if (!to_type.eql(.{ .int32 = {} })) return error.InvalidCast;
    }

    const out_len: usize = switch (args[0]) {
        .array => |arr| arr.data().length,
        .chunked => |chunks| chunks.len(),
        .scalar => 1,
    };

    var builder = try zcore.Int32Builder.init(ctx.tempAllocator(), out_len);
    defer builder.deinit();

    var iter = compute.UnaryExecChunkIterator.init(args[0]);
    while (try iter.next()) |chunk_value| {
        var chunk = chunk_value;
        defer chunk.deinit();

        var i: usize = 0;
        while (i < chunk.len) : (i += 1) {
            if (chunk.unaryNullAt(i)) {
                builder.appendNull() catch |err| return kernelAppendError(err);
                continue;
            }

            const value_i64 = try readI64(chunk.values, i);
            const casted: i32 = if (cast_opts.safe)
                try compute.intCastOrInvalidCast(i32, value_i64)
            else
                @as(i32, @truncate(value_i64));

            builder.append(casted) catch |err| return kernelAppendError(err);
        }
    }

    const out = builder.finish() catch |err| return kernelAppendError(err);
    return compute.Datum.fromArray(out);
}

fn countRowsResultType(args: []const compute.Datum, options: compute.Options) compute.KernelError!compute.DataType {
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

fn countRowsKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    _ = ctx;
    if (!onlyNoOptions(options)) return error.InvalidOptions;
    if (!unaryArrayLike(args)) return error.InvalidInput;
    const row_count = try countRows(args[0]);
    return compute.Datum.fromScalar(.{
        .data_type = .{ .int64 = {} },
        .value = .{ .i64 = @intCast(row_count) },
    });
}

const CountRowsState = struct {
    count: usize = 0,
};

fn countRowsInit(ctx: *compute.ExecContext, options: compute.Options) compute.KernelError!*anyopaque {
    if (!onlyNoOptions(options)) return error.InvalidOptions;
    const state = ctx.allocator.create(CountRowsState) catch return error.OutOfMemory;
    state.* = .{};
    return state;
}

fn countRowsUpdate(
    ctx: *compute.ExecContext,
    state_ptr: *anyopaque,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!void {
    _ = ctx;
    if (!onlyNoOptions(options)) return error.InvalidOptions;
    if (!unaryArrayLike(args)) return error.InvalidInput;

    const state: *CountRowsState = @ptrCast(@alignCast(state_ptr));
    const next_rows = try countRows(args[0]);
    state.count = std.math.add(usize, state.count, next_rows) catch return error.Overflow;
}

fn countRowsMerge(
    ctx: *compute.ExecContext,
    state_ptr: *anyopaque,
    other_ptr: *anyopaque,
    options: compute.Options,
) compute.KernelError!void {
    _ = ctx;
    if (!onlyNoOptions(options)) return error.InvalidOptions;
    const state: *CountRowsState = @ptrCast(@alignCast(state_ptr));
    const other: *CountRowsState = @ptrCast(@alignCast(other_ptr));
    state.count = std.math.add(usize, state.count, other.count) catch return error.Overflow;
}

fn countRowsFinalize(
    ctx: *compute.ExecContext,
    state_ptr: *anyopaque,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    _ = ctx;
    if (!onlyNoOptions(options)) return error.InvalidOptions;
    const state: *CountRowsState = @ptrCast(@alignCast(state_ptr));
    return compute.Datum.fromScalar(.{
        .data_type = .{ .int64 = {} },
        .value = .{ .i64 = @intCast(state.count) },
    });
}

fn countRowsDeinit(ctx: *compute.ExecContext, state_ptr: *anyopaque) void {
    const state: *CountRowsState = @ptrCast(@alignCast(state_ptr));
    ctx.allocator.destroy(state);
}

pub fn registerBaseKernels(registry: *compute.FunctionRegistry) compute.KernelError!void {
    try registry.registerVectorKernel("add_i64", .{
        .signature = .{
            .arity = 2,
            .type_check = binaryInt64,
            .options_check = onlyArithmeticOptions,
            .result_type_fn = resultI64,
        },
        .exec = addI64Kernel,
    });

    try registry.registerVectorKernel("divide_i64", .{
        .signature = .{
            .arity = 2,
            .type_check = binaryInt64,
            .options_check = onlyArithmeticOptions,
            .result_type_fn = resultI64,
        },
        .exec = divideI64Kernel,
    });

    try registry.registerVectorKernel("cast_i64_to_i32", .{
        .signature = .{
            .arity = 1,
            .type_check = unaryInt64,
            .options_check = onlyCastOptions,
            .result_type_fn = resultI32,
        },
        .exec = castI64ToI32Kernel,
    });

    try registry.registerAggregateKernel("count_rows", .{
        .signature = .{
            .arity = 1,
            .type_check = unaryArrayLike,
            .options_check = onlyNoOptions,
            .result_type_fn = countRowsResultType,
        },
        .exec = countRowsKernel,
        .aggregate_lifecycle = .{
            .init = countRowsInit,
            .update = countRowsUpdate,
            .merge = countRowsMerge,
            .finalize = countRowsFinalize,
            .deinit = countRowsDeinit,
        },
    });
}

pub fn registerCompatKernels(registry: *compute.FunctionRegistry) compute.KernelError!void {
    return registerBaseKernels(registry);
}

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
