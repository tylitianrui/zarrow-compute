const zcore = @import("zarrow-core");
const common = @import("common.zig");

const compute = common.compute;

pub fn invertKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    if (args.len != 1) return error.InvalidArity;
    if (!common.onlyNoOptions(options)) return error.InvalidOptions;
    if (!common.unaryBool(args)) return error.InvalidInput;

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
            const value = try common.readBool(chunk.values, i);
            builder.append(!value) catch |err| return common.kernelAppendError(err);
        }
    }

    const out = builder.finish() catch |err| return common.kernelAppendError(err);
    return compute.Datum.fromArray(out);
}

pub fn andKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    if (args.len != 2) return error.InvalidArity;
    if (!common.onlyNoOptions(options)) return error.InvalidOptions;
    if (!common.binaryBool(args)) return error.InvalidInput;

    const out_len = try compute.inferBinaryExecLen(args[0], args[1]);
    var builder = try zcore.BooleanBuilder.init(ctx.tempAllocator(), out_len);
    defer builder.deinit();

    var iter = try compute.BinaryExecChunkIterator.init(args[0], args[1]);
    while (try iter.next()) |chunk_value| {
        var chunk = chunk_value;
        defer chunk.deinit();

        var i: usize = 0;
        while (i < chunk.len) : (i += 1) {
            if (chunk.binaryNullAt(i)) {
                builder.appendNull() catch |err| return common.kernelAppendError(err);
                continue;
            }
            const lhs = try common.readBool(chunk.lhs, i);
            const rhs = try common.readBool(chunk.rhs, i);
            builder.append(lhs and rhs) catch |err| return common.kernelAppendError(err);
        }
    }

    const out = builder.finish() catch |err| return common.kernelAppendError(err);
    return compute.Datum.fromArray(out);
}

pub fn orKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    if (args.len != 2) return error.InvalidArity;
    if (!common.onlyNoOptions(options)) return error.InvalidOptions;
    if (!common.binaryBool(args)) return error.InvalidInput;

    const out_len = try compute.inferBinaryExecLen(args[0], args[1]);
    var builder = try zcore.BooleanBuilder.init(ctx.tempAllocator(), out_len);
    defer builder.deinit();

    var iter = try compute.BinaryExecChunkIterator.init(args[0], args[1]);
    while (try iter.next()) |chunk_value| {
        var chunk = chunk_value;
        defer chunk.deinit();

        var i: usize = 0;
        while (i < chunk.len) : (i += 1) {
            if (chunk.binaryNullAt(i)) {
                builder.appendNull() catch |err| return common.kernelAppendError(err);
                continue;
            }
            const lhs = try common.readBool(chunk.lhs, i);
            const rhs = try common.readBool(chunk.rhs, i);
            builder.append(lhs or rhs) catch |err| return common.kernelAppendError(err);
        }
    }

    const out = builder.finish() catch |err| return common.kernelAppendError(err);
    return compute.Datum.fromArray(out);
}

pub fn andKleeneKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    if (args.len != 2) return error.InvalidArity;
    if (!common.onlyNoOptions(options)) return error.InvalidOptions;
    if (!common.binaryBool(args)) return error.InvalidInput;

    const out_len = try compute.inferBinaryExecLen(args[0], args[1]);
    var builder = try zcore.BooleanBuilder.init(ctx.tempAllocator(), out_len);
    defer builder.deinit();

    var iter = try compute.BinaryExecChunkIterator.init(args[0], args[1]);
    while (try iter.next()) |chunk_value| {
        var chunk = chunk_value;
        defer chunk.deinit();

        var i: usize = 0;
        while (i < chunk.len) : (i += 1) {
            const lhs_null = chunk.lhs.isNullAt(i);
            const rhs_null = chunk.rhs.isNullAt(i);
            const lhs_false = !lhs_null and !(try common.readBool(chunk.lhs, i));
            const rhs_false = !rhs_null and !(try common.readBool(chunk.rhs, i));
            if (lhs_false or rhs_false) {
                builder.append(false) catch |err| return common.kernelAppendError(err);
                continue;
            }
            if (lhs_null or rhs_null) {
                builder.appendNull() catch |err| return common.kernelAppendError(err);
                continue;
            }
            builder.append(true) catch |err| return common.kernelAppendError(err);
        }
    }

    const out = builder.finish() catch |err| return common.kernelAppendError(err);
    return compute.Datum.fromArray(out);
}

pub fn orKleeneKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    if (args.len != 2) return error.InvalidArity;
    if (!common.onlyNoOptions(options)) return error.InvalidOptions;
    if (!common.binaryBool(args)) return error.InvalidInput;

    const out_len = try compute.inferBinaryExecLen(args[0], args[1]);
    var builder = try zcore.BooleanBuilder.init(ctx.tempAllocator(), out_len);
    defer builder.deinit();

    var iter = try compute.BinaryExecChunkIterator.init(args[0], args[1]);
    while (try iter.next()) |chunk_value| {
        var chunk = chunk_value;
        defer chunk.deinit();

        var i: usize = 0;
        while (i < chunk.len) : (i += 1) {
            const lhs_null = chunk.lhs.isNullAt(i);
            const rhs_null = chunk.rhs.isNullAt(i);
            const lhs_true = !lhs_null and (try common.readBool(chunk.lhs, i));
            const rhs_true = !rhs_null and (try common.readBool(chunk.rhs, i));
            if (lhs_true or rhs_true) {
                builder.append(true) catch |err| return common.kernelAppendError(err);
                continue;
            }
            if (lhs_null or rhs_null) {
                builder.appendNull() catch |err| return common.kernelAppendError(err);
                continue;
            }
            builder.append(false) catch |err| return common.kernelAppendError(err);
        }
    }

    const out = builder.finish() catch |err| return common.kernelAppendError(err);
    return compute.Datum.fromArray(out);
}
