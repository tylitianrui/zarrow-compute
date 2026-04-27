const zcore = @import("zarrow-core");
const common = @import("common.zig");

const compute = common.compute;

pub fn castResultType(
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.DataType {
    if (args.len != 1) return error.InvalidArity;
    return switch (options) {
        .cast => |cast_opts| cast_opts.to_type orelse args[0].dataType(),
        else => error.InvalidOptions,
    };
}

fn castBoolToInt32Kernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
) compute.KernelError!compute.Datum {
    const out_len = switch (args[0]) {
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
                builder.appendNull() catch |err| return common.kernelAppendError(err);
                continue;
            }
            const value = try common.readBool(chunk.values, i);
            builder.append(if (value) 1 else 0) catch |err| return common.kernelAppendError(err);
        }
    }
    return compute.Datum.fromArray(builder.finish() catch |err| return common.kernelAppendError(err));
}

fn castBoolToInt64Kernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
) compute.KernelError!compute.Datum {
    const out_len = switch (args[0]) {
        .array => |arr| arr.data().length,
        .chunked => |chunks| chunks.len(),
        .scalar => 1,
    };
    var builder = try zcore.Int64Builder.init(ctx.tempAllocator(), out_len);
    defer builder.deinit();
    var iter = compute.UnaryExecChunkIterator.init(args[0]);
    while (try iter.next()) |chunk_value| {
        var chunk = chunk_value;
        defer chunk.deinit();
        var i: usize = 0;
        while (i < chunk.len) : (i += 1) {
            if (chunk.unaryNullAt(i)) {
                builder.appendNull() catch |err| return common.kernelAppendError(err);
                continue;
            }
            const value = try common.readBool(chunk.values, i);
            builder.append(if (value) 1 else 0) catch |err| return common.kernelAppendError(err);
        }
    }
    return compute.Datum.fromArray(builder.finish() catch |err| return common.kernelAppendError(err));
}

fn castInt32ToInt64Kernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
) compute.KernelError!compute.Datum {
    const out_len = switch (args[0]) {
        .array => |arr| arr.data().length,
        .chunked => |chunks| chunks.len(),
        .scalar => 1,
    };
    var builder = try zcore.Int64Builder.init(ctx.tempAllocator(), out_len);
    defer builder.deinit();
    var iter = compute.UnaryExecChunkIterator.init(args[0]);
    while (try iter.next()) |chunk_value| {
        var chunk = chunk_value;
        defer chunk.deinit();
        var i: usize = 0;
        while (i < chunk.len) : (i += 1) {
            if (chunk.unaryNullAt(i)) {
                builder.appendNull() catch |err| return common.kernelAppendError(err);
                continue;
            }
            const value = try common.readI64(chunk.values, i);
            builder.append(value) catch |err| return common.kernelAppendError(err);
        }
    }
    return compute.Datum.fromArray(builder.finish() catch |err| return common.kernelAppendError(err));
}

fn castIntToBoolKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    cast_opts: compute.CastOptions,
) compute.KernelError!compute.Datum {
    const out_len = switch (args[0]) {
        .array => |arr| arr.data().length,
        .chunked => |chunks| chunks.len(),
        .scalar => 1,
    };
    var builder = try zcore.BooleanBuilder.init(ctx.tempAllocator(), out_len);
    defer builder.deinit();
    var iter = compute.UnaryExecChunkIterator.init(args[0]);
    while (try iter.next()) |chunk_value| {
        var chunk = chunk_value;
        defer chunk.deinit();
        var i: usize = 0;
        while (i < chunk.len) : (i += 1) {
            if (chunk.unaryNullAt(i)) {
                builder.appendNull() catch |err| return common.kernelAppendError(err);
                continue;
            }
            const value = try common.readI64(chunk.values, i);
            if (cast_opts.safe and value != 0 and value != 1) return error.InvalidCast;
            builder.append(value != 0) catch |err| return common.kernelAppendError(err);
        }
    }
    return compute.Datum.fromArray(builder.finish() catch |err| return common.kernelAppendError(err));
}

pub fn castI64ToI32Kernel(
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
                builder.appendNull() catch |err| return common.kernelAppendError(err);
                continue;
            }

            const value_i64 = try common.readI64(chunk.values, i);
            const casted: i32 = if (cast_opts.safe)
                try compute.intCastOrInvalidCast(i32, value_i64)
            else
                @as(i32, @truncate(value_i64));

            builder.append(casted) catch |err| return common.kernelAppendError(err);
        }
    }

    const out = builder.finish() catch |err| return common.kernelAppendError(err);
    return compute.Datum.fromArray(out);
}

pub fn castKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    if (args.len != 1) return error.InvalidArity;
    const cast_opts = switch (options) {
        .cast => |o| o,
        else => return error.InvalidOptions,
    };

    const from_type = args[0].dataType();
    const to_type = cast_opts.to_type orelse from_type;
    if (from_type.eql(to_type)) return args[0].retain();

    if (from_type.eql(.{ .int64 = {} }) and to_type.eql(.{ .int32 = {} })) {
        return castI64ToI32Kernel(ctx, args, options);
    }
    if (from_type.eql(.{ .int32 = {} }) and to_type.eql(.{ .int64 = {} })) {
        return castInt32ToInt64Kernel(ctx, args);
    }
    if (from_type.eql(.{ .bool = {} }) and to_type.eql(.{ .int32 = {} })) {
        return castBoolToInt32Kernel(ctx, args);
    }
    if (from_type.eql(.{ .bool = {} }) and to_type.eql(.{ .int64 = {} })) {
        return castBoolToInt64Kernel(ctx, args);
    }
    if ((from_type.eql(.{ .int32 = {} }) or from_type.eql(.{ .int64 = {} })) and to_type.eql(.{ .bool = {} })) {
        return castIntToBoolKernel(ctx, args, cast_opts);
    }

    return error.InvalidCast;
}
