const zcore = @import("zarrow-core");
const common = @import("common.zig");

const compute = common.compute;
const CompareOp = enum { equal, not_equal, less, less_equal, greater, greater_equal };

fn applyCompareI32(lhs: i32, rhs: i32, op: CompareOp) bool {
    return switch (op) {
        .equal => lhs == rhs,
        .not_equal => lhs != rhs,
        .less => lhs < rhs,
        .less_equal => lhs <= rhs,
        .greater => lhs > rhs,
        .greater_equal => lhs >= rhs,
    };
}

fn applyCompareI64(lhs: i64, rhs: i64, op: CompareOp) bool {
    return switch (op) {
        .equal => lhs == rhs,
        .not_equal => lhs != rhs,
        .less => lhs < rhs,
        .less_equal => lhs <= rhs,
        .greater => lhs > rhs,
        .greater_equal => lhs >= rhs,
    };
}

fn applyCompareF64(lhs: f64, rhs: f64, op: CompareOp) bool {
    return switch (op) {
        .equal => lhs == rhs,
        .not_equal => lhs != rhs,
        .less => lhs < rhs,
        .less_equal => lhs <= rhs,
        .greater => lhs > rhs,
        .greater_equal => lhs >= rhs,
    };
}

fn compareKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
    op: CompareOp,
) compute.KernelError!compute.Datum {
    if (args.len != 2) return error.InvalidArity;
    if (!common.onlyNoOptions(options)) return error.InvalidOptions;
    if (!common.binaryArithmeticComparable(args)) return error.InvalidInput;

    const out_len = try compute.inferBinaryExecLen(args[0], args[1]);
    var builder = try zcore.BooleanBuilder.init(ctx.tempAllocator(), out_len);
    defer builder.deinit();
    var iter = try compute.BinaryExecChunkIterator.init(args[0], args[1]);
    switch (args[0].dataType()) {
        .int32 => {
            while (try iter.next()) |chunk_value| {
                var chunk = chunk_value;
                defer chunk.deinit();
                var i: usize = 0;
                while (i < chunk.len) : (i += 1) {
                    if (chunk.binaryNullAt(i)) {
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }
                    const lhs = try common.readI32(chunk.lhs, i);
                    const rhs = try common.readI32(chunk.rhs, i);
                    builder.append(applyCompareI32(lhs, rhs, op)) catch |err| return common.kernelAppendError(err);
                }
            }
        },
        .int64 => {
            while (try iter.next()) |chunk_value| {
                var chunk = chunk_value;
                defer chunk.deinit();
                var i: usize = 0;
                while (i < chunk.len) : (i += 1) {
                    if (chunk.binaryNullAt(i)) {
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }
                    const lhs = try common.readI64(chunk.lhs, i);
                    const rhs = try common.readI64(chunk.rhs, i);
                    builder.append(applyCompareI64(lhs, rhs, op)) catch |err| return common.kernelAppendError(err);
                }
            }
        },
        .double => {
            while (try iter.next()) |chunk_value| {
                var chunk = chunk_value;
                defer chunk.deinit();
                var i: usize = 0;
                while (i < chunk.len) : (i += 1) {
                    if (chunk.binaryNullAt(i)) {
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }
                    const lhs = try common.readF64(chunk.lhs, i);
                    const rhs = try common.readF64(chunk.rhs, i);
                    builder.append(applyCompareF64(lhs, rhs, op)) catch |err| return common.kernelAppendError(err);
                }
            }
        },
        else => return error.InvalidInput,
    }

    const out = builder.finish() catch |err| return common.kernelAppendError(err);
    return compute.Datum.fromArray(out);
}

pub fn equalKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    return compareKernel(ctx, args, options, .equal);
}

pub fn notEqualKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    return compareKernel(ctx, args, options, .not_equal);
}

pub fn lessKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    return compareKernel(ctx, args, options, .less);
}

pub fn lessEqualKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    return compareKernel(ctx, args, options, .less_equal);
}

pub fn greaterKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    return compareKernel(ctx, args, options, .greater);
}

pub fn greaterEqualKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    return compareKernel(ctx, args, options, .greater_equal);
}
