const zcore = @import("zarrow-core");
const common = @import("common.zig");

const compute = common.compute;

fn compareI64Kernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
    comptime op: fn (i64, i64) bool,
) compute.KernelError!compute.Datum {
    if (args.len != 2) return error.InvalidArity;
    if (!common.onlyNoOptions(options)) return error.InvalidOptions;
    if (!common.binaryInt64(args)) return error.InvalidInput;

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
            const lhs = try common.readI64(chunk.lhs, i);
            const rhs = try common.readI64(chunk.rhs, i);
            builder.append(op(lhs, rhs)) catch |err| return common.kernelAppendError(err);
        }
    }

    const out = builder.finish() catch |err| return common.kernelAppendError(err);
    return compute.Datum.fromArray(out);
}

fn opEqual(lhs: i64, rhs: i64) bool {
    return lhs == rhs;
}

fn opNotEqual(lhs: i64, rhs: i64) bool {
    return lhs != rhs;
}

fn opLess(lhs: i64, rhs: i64) bool {
    return lhs < rhs;
}

fn opLessEqual(lhs: i64, rhs: i64) bool {
    return lhs <= rhs;
}

fn opGreater(lhs: i64, rhs: i64) bool {
    return lhs > rhs;
}

fn opGreaterEqual(lhs: i64, rhs: i64) bool {
    return lhs >= rhs;
}

pub fn equalKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    return compareI64Kernel(ctx, args, options, opEqual);
}

pub fn notEqualKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    return compareI64Kernel(ctx, args, options, opNotEqual);
}

pub fn lessKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    return compareI64Kernel(ctx, args, options, opLess);
}

pub fn lessEqualKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    return compareI64Kernel(ctx, args, options, opLessEqual);
}

pub fn greaterKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    return compareI64Kernel(ctx, args, options, opGreater);
}

pub fn greaterEqualKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    return compareI64Kernel(ctx, args, options, opGreaterEqual);
}
