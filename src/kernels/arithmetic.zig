const std = @import("std");
const zcore = @import("zarrow-core");
const common = @import("common.zig");

const compute = common.compute;

fn divI32(lhs: i32, rhs: i32, options: compute.ArithmeticOptions) compute.KernelError!i32 {
    if (rhs == 0) {
        if (options.divide_by_zero_is_error) return error.DivideByZero;
        return 0;
    }
    if (options.check_overflow and lhs == std.math.minInt(i32) and rhs == -1) {
        return error.Overflow;
    }
    return @divTrunc(lhs, rhs);
}

pub fn addI64Kernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    if (args.len != 2) return error.InvalidArity;
    const arithmetic_opts = switch (options) {
        .arithmetic => |o| o,
        else => return error.InvalidOptions,
    };

    if (args.len != 2 or !args[0].dataType().eql(args[1].dataType())) return error.InvalidInput;
    const out_len = try compute.inferBinaryExecLen(args[0], args[1]);
    switch (args[0].dataType()) {
        .int32 => {
            var builder = try zcore.Int32Builder.init(ctx.tempAllocator(), out_len);
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
                    const lhs = try common.readI32(chunk.lhs, i);
                    const rhs = try common.readI32(chunk.rhs, i);
                    const sum = if (arithmetic_opts.check_overflow)
                        std.math.add(i32, lhs, rhs) catch return error.Overflow
                    else
                        lhs +% rhs;
                    builder.append(sum) catch |err| return common.kernelAppendError(err);
                }
            }
            const out = builder.finish() catch |err| return common.kernelAppendError(err);
            return compute.Datum.fromArray(out);
        },
        .int64 => {
            var builder = try zcore.Int64Builder.init(ctx.tempAllocator(), out_len);
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
                    const sum = if (arithmetic_opts.check_overflow)
                        std.math.add(i64, lhs, rhs) catch return error.Overflow
                    else
                        lhs +% rhs;
                    builder.append(sum) catch |err| return common.kernelAppendError(err);
                }
            }
            const out = builder.finish() catch |err| return common.kernelAppendError(err);
            return compute.Datum.fromArray(out);
        },
        .double => {
            var builder = try zcore.Float64Builder.init(ctx.tempAllocator(), out_len);
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
                    const lhs = try common.readF64(chunk.lhs, i);
                    const rhs = try common.readF64(chunk.rhs, i);
                    builder.append(lhs + rhs) catch |err| return common.kernelAppendError(err);
                }
            }
            const out = builder.finish() catch |err| return common.kernelAppendError(err);
            return compute.Datum.fromArray(out);
        },
        else => return error.InvalidInput,
    }
}

pub fn subtractI64Kernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    if (args.len != 2) return error.InvalidArity;
    const arithmetic_opts = switch (options) {
        .arithmetic => |o| o,
        else => return error.InvalidOptions,
    };

    if (args.len != 2 or !args[0].dataType().eql(args[1].dataType())) return error.InvalidInput;
    const out_len = try compute.inferBinaryExecLen(args[0], args[1]);
    switch (args[0].dataType()) {
        .int32 => {
            var builder = try zcore.Int32Builder.init(ctx.tempAllocator(), out_len);
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
                    const lhs = try common.readI32(chunk.lhs, i);
                    const rhs = try common.readI32(chunk.rhs, i);
                    const diff = if (arithmetic_opts.check_overflow)
                        std.math.sub(i32, lhs, rhs) catch return error.Overflow
                    else
                        lhs -% rhs;
                    builder.append(diff) catch |err| return common.kernelAppendError(err);
                }
            }
            const out = builder.finish() catch |err| return common.kernelAppendError(err);
            return compute.Datum.fromArray(out);
        },
        .int64 => {
            var builder = try zcore.Int64Builder.init(ctx.tempAllocator(), out_len);
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
                    const diff = if (arithmetic_opts.check_overflow)
                        std.math.sub(i64, lhs, rhs) catch return error.Overflow
                    else
                        lhs -% rhs;
                    builder.append(diff) catch |err| return common.kernelAppendError(err);
                }
            }
            const out = builder.finish() catch |err| return common.kernelAppendError(err);
            return compute.Datum.fromArray(out);
        },
        .double => {
            var builder = try zcore.Float64Builder.init(ctx.tempAllocator(), out_len);
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
                    const lhs = try common.readF64(chunk.lhs, i);
                    const rhs = try common.readF64(chunk.rhs, i);
                    builder.append(lhs - rhs) catch |err| return common.kernelAppendError(err);
                }
            }
            const out = builder.finish() catch |err| return common.kernelAppendError(err);
            return compute.Datum.fromArray(out);
        },
        else => return error.InvalidInput,
    }
}

pub fn multiplyI64Kernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    if (args.len != 2) return error.InvalidArity;
    const arithmetic_opts = switch (options) {
        .arithmetic => |o| o,
        else => return error.InvalidOptions,
    };

    if (args.len != 2 or !args[0].dataType().eql(args[1].dataType())) return error.InvalidInput;
    const out_len = try compute.inferBinaryExecLen(args[0], args[1]);
    switch (args[0].dataType()) {
        .int32 => {
            var builder = try zcore.Int32Builder.init(ctx.tempAllocator(), out_len);
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
                    const lhs = try common.readI32(chunk.lhs, i);
                    const rhs = try common.readI32(chunk.rhs, i);
                    const product = if (arithmetic_opts.check_overflow)
                        std.math.mul(i32, lhs, rhs) catch return error.Overflow
                    else
                        lhs *% rhs;
                    builder.append(product) catch |err| return common.kernelAppendError(err);
                }
            }
            const out = builder.finish() catch |err| return common.kernelAppendError(err);
            return compute.Datum.fromArray(out);
        },
        .int64 => {
            var builder = try zcore.Int64Builder.init(ctx.tempAllocator(), out_len);
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
                    const product = if (arithmetic_opts.check_overflow)
                        std.math.mul(i64, lhs, rhs) catch return error.Overflow
                    else
                        lhs *% rhs;
                    builder.append(product) catch |err| return common.kernelAppendError(err);
                }
            }
            const out = builder.finish() catch |err| return common.kernelAppendError(err);
            return compute.Datum.fromArray(out);
        },
        .double => {
            var builder = try zcore.Float64Builder.init(ctx.tempAllocator(), out_len);
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
                    const lhs = try common.readF64(chunk.lhs, i);
                    const rhs = try common.readF64(chunk.rhs, i);
                    builder.append(lhs * rhs) catch |err| return common.kernelAppendError(err);
                }
            }
            const out = builder.finish() catch |err| return common.kernelAppendError(err);
            return compute.Datum.fromArray(out);
        },
        else => return error.InvalidInput,
    }
}

pub fn divideI64Kernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    if (args.len != 2) return error.InvalidArity;
    const arithmetic_opts = switch (options) {
        .arithmetic => |o| o,
        else => return error.InvalidOptions,
    };

    if (args.len != 2 or !args[0].dataType().eql(args[1].dataType())) return error.InvalidInput;
    const out_len = try compute.inferBinaryExecLen(args[0], args[1]);
    switch (args[0].dataType()) {
        .int32 => {
            var builder = try zcore.Int32Builder.init(ctx.tempAllocator(), out_len);
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
                    const lhs = try common.readI32(chunk.lhs, i);
                    const rhs = try common.readI32(chunk.rhs, i);
                    const out = try divI32(lhs, rhs, arithmetic_opts);
                    builder.append(out) catch |err| return common.kernelAppendError(err);
                }
            }
            const out = builder.finish() catch |err| return common.kernelAppendError(err);
            return compute.Datum.fromArray(out);
        },
        .int64 => {
            var builder = try zcore.Int64Builder.init(ctx.tempAllocator(), out_len);
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
                    const out = try compute.arithmeticDivI64(lhs, rhs, arithmetic_opts);
                    builder.append(out) catch |err| return common.kernelAppendError(err);
                }
            }
            const out = builder.finish() catch |err| return common.kernelAppendError(err);
            return compute.Datum.fromArray(out);
        },
        .double => {
            var builder = try zcore.Float64Builder.init(ctx.tempAllocator(), out_len);
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
                    const lhs = try common.readF64(chunk.lhs, i);
                    const rhs = try common.readF64(chunk.rhs, i);
                    if (rhs == 0.0 and arithmetic_opts.divide_by_zero_is_error) return error.DivideByZero;
                    builder.append(lhs / rhs) catch |err| return common.kernelAppendError(err);
                }
            }
            const out = builder.finish() catch |err| return common.kernelAppendError(err);
            return compute.Datum.fromArray(out);
        },
        else => return error.InvalidInput,
    }
}
