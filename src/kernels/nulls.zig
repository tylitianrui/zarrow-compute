const zcore = @import("zarrow-core");
const std = @import("std");
const common = @import("common.zig");

const compute = common.compute;

pub fn isNullKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    if (args.len != 1) return error.InvalidArity;
    if (!common.onlyNoOptions(options)) return error.InvalidOptions;

    const out_len: usize = switch (args[0]) {
        .array => |arr| arr.data().length,
        .chunked => |chunks| chunks.len(),
        .scalar => return error.InvalidInput,
    };
    var builder = try zcore.BooleanBuilder.init(ctx.tempAllocator(), out_len);
    defer builder.deinit();

    var iter = compute.UnaryExecChunkIterator.init(args[0]);
    while (try iter.next()) |chunk_value| {
        var chunk = chunk_value;
        defer chunk.deinit();

        var i: usize = 0;
        while (i < chunk.len) : (i += 1) {
            const out = chunk.values.isNullAt(i);
            builder.append(out) catch |err| return common.kernelAppendError(err);
        }
    }

    const result = builder.finish() catch |err| return common.kernelAppendError(err);
    return compute.Datum.fromArray(result);
}

pub fn isValidKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    if (args.len != 1) return error.InvalidArity;
    if (!common.onlyNoOptions(options)) return error.InvalidOptions;

    const out_len: usize = switch (args[0]) {
        .array => |arr| arr.data().length,
        .chunked => |chunks| chunks.len(),
        .scalar => return error.InvalidInput,
    };
    var builder = try zcore.BooleanBuilder.init(ctx.tempAllocator(), out_len);
    defer builder.deinit();

    var iter = compute.UnaryExecChunkIterator.init(args[0]);
    while (try iter.next()) |chunk_value| {
        var chunk = chunk_value;
        defer chunk.deinit();

        var i: usize = 0;
        while (i < chunk.len) : (i += 1) {
            const out = !chunk.values.isNullAt(i);
            builder.append(out) catch |err| return common.kernelAppendError(err);
        }
    }

    const result = builder.finish() catch |err| return common.kernelAppendError(err);
    return compute.Datum.fromArray(result);
}

fn unaryFloat64PredicateKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
    comptime pred: fn (f64) bool,
) compute.KernelError!compute.Datum {
    if (args.len != 1) return error.InvalidArity;
    if (!common.onlyNoOptions(options)) return error.InvalidOptions;
    if (!common.unaryFloat64ArrayLike(args)) return error.InvalidInput;

    const out_len: usize = switch (args[0]) {
        .array => |arr| arr.data().length,
        .chunked => |chunks| chunks.len(),
        .scalar => return error.InvalidInput,
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
            const value = try common.readF64(chunk.values, i);
            builder.append(pred(value)) catch |err| return common.kernelAppendError(err);
        }
    }

    const result = builder.finish() catch |err| return common.kernelAppendError(err);
    return compute.Datum.fromArray(result);
}

fn predFinite(value: f64) bool {
    return std.math.isFinite(value);
}

fn predInf(value: f64) bool {
    return std.math.isInf(value);
}

fn predNaN(value: f64) bool {
    return std.math.isNan(value);
}

pub fn isFiniteKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    return unaryFloat64PredicateKernel(ctx, args, options, predFinite);
}

pub fn isInfKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    return unaryFloat64PredicateKernel(ctx, args, options, predInf);
}

pub fn isNaNKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    return unaryFloat64PredicateKernel(ctx, args, options, predNaN);
}
