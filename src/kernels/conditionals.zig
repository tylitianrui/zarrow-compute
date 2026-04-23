const zcore = @import("zarrow-core");
const common = @import("common.zig");

const compute = common.compute;

fn inferTernaryExecLen(cond: compute.Datum, lhs: compute.Datum, rhs: compute.Datum) compute.KernelError!usize {
    const len_cl = try compute.inferBinaryExecLen(cond, lhs);
    const len_cr = try compute.inferBinaryExecLen(cond, rhs);
    const len_lr = try compute.inferBinaryExecLen(lhs, rhs);
    if (len_cl != len_cr or len_cl != len_lr) return error.InvalidInput;
    return len_cl;
}

fn ensureUnaryChunk(
    iter: *compute.UnaryExecChunkIterator,
    chunk_opt: *?compute.UnaryExecChunk,
    index: *usize,
) compute.KernelError!void {
    while (true) {
        if (chunk_opt.* == null) {
            chunk_opt.* = try iter.next();
            index.* = 0;
            return;
        }
        if (index.* < chunk_opt.*.?.len) return;
        var old = chunk_opt.*.?;
        old.deinit();
        chunk_opt.* = null;
        index.* = 0;
    }
}

fn ensureBinaryChunk(
    iter: *compute.BinaryExecChunkIterator,
    chunk_opt: *?compute.BinaryExecChunk,
    index: *usize,
) compute.KernelError!void {
    while (true) {
        if (chunk_opt.* == null) {
            chunk_opt.* = try iter.next();
            index.* = 0;
            return;
        }
        if (index.* < chunk_opt.*.?.len) return;
        var old = chunk_opt.*.?;
        old.deinit();
        chunk_opt.* = null;
        index.* = 0;
    }
}

pub fn trueUnlessNullKernel(
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

    const out = builder.finish() catch |err| return common.kernelAppendError(err);
    return compute.Datum.fromArray(out);
}

fn ifElseStringKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    kind: common.StringValueKind,
) compute.KernelError!compute.Datum {
    const out_len = try inferTernaryExecLen(args[0], args[1], args[2]);

    var cond_iter = compute.UnaryExecChunkIterator.init(args[0]);
    var values_iter = try compute.BinaryExecChunkIterator.init(args[1], args[2]);
    var cond_chunk_opt: ?compute.UnaryExecChunk = null;
    defer if (cond_chunk_opt) |*c| c.deinit();
    var values_chunk_opt: ?compute.BinaryExecChunk = null;
    defer if (values_chunk_opt) |*c| c.deinit();
    var cond_index: usize = 0;
    var values_index: usize = 0;
    var produced: usize = 0;

    switch (kind) {
        .string => {
            var builder = try zcore.StringBuilder.init(ctx.tempAllocator(), out_len, 0);
            defer builder.deinit();
            while (produced < out_len) {
                try ensureUnaryChunk(&cond_iter, &cond_chunk_opt, &cond_index);
                try ensureBinaryChunk(&values_iter, &values_chunk_opt, &values_index);
                if (cond_chunk_opt == null or values_chunk_opt == null) return error.InvalidInput;

                const cond_chunk = &cond_chunk_opt.?;
                const values_chunk = &values_chunk_opt.?;
                const run_len = @min(cond_chunk.len - cond_index, values_chunk.len - values_index);
                var j: usize = 0;
                while (j < run_len) : (j += 1) {
                    const ci = cond_index + j;
                    const vi = values_index + j;
                    if (cond_chunk.values.isNullAt(ci)) {
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }
                    const take_lhs = try common.readBool(cond_chunk.values, ci);
                    const selected = if (take_lhs) values_chunk.lhs else values_chunk.rhs;
                    if (selected.isNullAt(vi)) {
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }
                    const value = try common.readString(selected, vi, .string);
                    builder.append(value) catch |err| return common.kernelAppendError(err);
                }
                produced += run_len;
                cond_index += run_len;
                values_index += run_len;
            }
            const out = builder.finish() catch |err| return common.kernelAppendError(err);
            return compute.Datum.fromArray(out);
        },
        .large_string => {
            var builder = try zcore.LargeStringBuilder.init(ctx.tempAllocator(), out_len, 0);
            defer builder.deinit();
            while (produced < out_len) {
                try ensureUnaryChunk(&cond_iter, &cond_chunk_opt, &cond_index);
                try ensureBinaryChunk(&values_iter, &values_chunk_opt, &values_index);
                if (cond_chunk_opt == null or values_chunk_opt == null) return error.InvalidInput;

                const cond_chunk = &cond_chunk_opt.?;
                const values_chunk = &values_chunk_opt.?;
                const run_len = @min(cond_chunk.len - cond_index, values_chunk.len - values_index);
                var j: usize = 0;
                while (j < run_len) : (j += 1) {
                    const ci = cond_index + j;
                    const vi = values_index + j;
                    if (cond_chunk.values.isNullAt(ci)) {
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }
                    const take_lhs = try common.readBool(cond_chunk.values, ci);
                    const selected = if (take_lhs) values_chunk.lhs else values_chunk.rhs;
                    if (selected.isNullAt(vi)) {
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }
                    const value = try common.readString(selected, vi, .large_string);
                    builder.append(value) catch |err| return common.kernelAppendError(err);
                }
                produced += run_len;
                cond_index += run_len;
                values_index += run_len;
            }
            const out = builder.finish() catch |err| return common.kernelAppendError(err);
            return compute.Datum.fromArray(out);
        },
        .string_view => {
            var builder = try zcore.StringViewBuilder.init(ctx.tempAllocator(), out_len, 0);
            defer builder.deinit();
            while (produced < out_len) {
                try ensureUnaryChunk(&cond_iter, &cond_chunk_opt, &cond_index);
                try ensureBinaryChunk(&values_iter, &values_chunk_opt, &values_index);
                if (cond_chunk_opt == null or values_chunk_opt == null) return error.InvalidInput;

                const cond_chunk = &cond_chunk_opt.?;
                const values_chunk = &values_chunk_opt.?;
                const run_len = @min(cond_chunk.len - cond_index, values_chunk.len - values_index);
                var j: usize = 0;
                while (j < run_len) : (j += 1) {
                    const ci = cond_index + j;
                    const vi = values_index + j;
                    if (cond_chunk.values.isNullAt(ci)) {
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }
                    const take_lhs = try common.readBool(cond_chunk.values, ci);
                    const selected = if (take_lhs) values_chunk.lhs else values_chunk.rhs;
                    if (selected.isNullAt(vi)) {
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }
                    const value = try common.readString(selected, vi, .string_view);
                    builder.append(value) catch |err| return common.kernelAppendError(err);
                }
                produced += run_len;
                cond_index += run_len;
                values_index += run_len;
            }
            const out = builder.finish() catch |err| return common.kernelAppendError(err);
            return compute.Datum.fromArray(out);
        },
    }
}

fn ifElseBinaryKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    kind: common.BinaryValueKind,
) compute.KernelError!compute.Datum {
    const out_len = try inferTernaryExecLen(args[0], args[1], args[2]);

    var cond_iter = compute.UnaryExecChunkIterator.init(args[0]);
    var values_iter = try compute.BinaryExecChunkIterator.init(args[1], args[2]);
    var cond_chunk_opt: ?compute.UnaryExecChunk = null;
    defer if (cond_chunk_opt) |*c| c.deinit();
    var values_chunk_opt: ?compute.BinaryExecChunk = null;
    defer if (values_chunk_opt) |*c| c.deinit();
    var cond_index: usize = 0;
    var values_index: usize = 0;
    var produced: usize = 0;

    switch (kind) {
        .binary => {
            var builder = try zcore.BinaryBuilder.init(ctx.tempAllocator(), out_len, 0);
            defer builder.deinit();
            while (produced < out_len) {
                try ensureUnaryChunk(&cond_iter, &cond_chunk_opt, &cond_index);
                try ensureBinaryChunk(&values_iter, &values_chunk_opt, &values_index);
                if (cond_chunk_opt == null or values_chunk_opt == null) return error.InvalidInput;

                const cond_chunk = &cond_chunk_opt.?;
                const values_chunk = &values_chunk_opt.?;
                const run_len = @min(cond_chunk.len - cond_index, values_chunk.len - values_index);
                var j: usize = 0;
                while (j < run_len) : (j += 1) {
                    const ci = cond_index + j;
                    const vi = values_index + j;
                    if (cond_chunk.values.isNullAt(ci)) {
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }
                    const take_lhs = try common.readBool(cond_chunk.values, ci);
                    const selected = if (take_lhs) values_chunk.lhs else values_chunk.rhs;
                    if (selected.isNullAt(vi)) {
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }
                    const value = try common.readBinary(selected, vi, .binary);
                    builder.append(value) catch |err| return common.kernelAppendError(err);
                }
                produced += run_len;
                cond_index += run_len;
                values_index += run_len;
            }
            const out = builder.finish() catch |err| return common.kernelAppendError(err);
            return compute.Datum.fromArray(out);
        },
        .large_binary => {
            var builder = try zcore.LargeBinaryBuilder.init(ctx.tempAllocator(), out_len, 0);
            defer builder.deinit();
            while (produced < out_len) {
                try ensureUnaryChunk(&cond_iter, &cond_chunk_opt, &cond_index);
                try ensureBinaryChunk(&values_iter, &values_chunk_opt, &values_index);
                if (cond_chunk_opt == null or values_chunk_opt == null) return error.InvalidInput;

                const cond_chunk = &cond_chunk_opt.?;
                const values_chunk = &values_chunk_opt.?;
                const run_len = @min(cond_chunk.len - cond_index, values_chunk.len - values_index);
                var j: usize = 0;
                while (j < run_len) : (j += 1) {
                    const ci = cond_index + j;
                    const vi = values_index + j;
                    if (cond_chunk.values.isNullAt(ci)) {
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }
                    const take_lhs = try common.readBool(cond_chunk.values, ci);
                    const selected = if (take_lhs) values_chunk.lhs else values_chunk.rhs;
                    if (selected.isNullAt(vi)) {
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }
                    const value = try common.readBinary(selected, vi, .large_binary);
                    builder.append(value) catch |err| return common.kernelAppendError(err);
                }
                produced += run_len;
                cond_index += run_len;
                values_index += run_len;
            }
            const out = builder.finish() catch |err| return common.kernelAppendError(err);
            return compute.Datum.fromArray(out);
        },
        .binary_view => {
            var builder = try zcore.BinaryViewBuilder.init(ctx.tempAllocator(), out_len, 0);
            defer builder.deinit();
            while (produced < out_len) {
                try ensureUnaryChunk(&cond_iter, &cond_chunk_opt, &cond_index);
                try ensureBinaryChunk(&values_iter, &values_chunk_opt, &values_index);
                if (cond_chunk_opt == null or values_chunk_opt == null) return error.InvalidInput;

                const cond_chunk = &cond_chunk_opt.?;
                const values_chunk = &values_chunk_opt.?;
                const run_len = @min(cond_chunk.len - cond_index, values_chunk.len - values_index);
                var j: usize = 0;
                while (j < run_len) : (j += 1) {
                    const ci = cond_index + j;
                    const vi = values_index + j;
                    if (cond_chunk.values.isNullAt(ci)) {
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }
                    const take_lhs = try common.readBool(cond_chunk.values, ci);
                    const selected = if (take_lhs) values_chunk.lhs else values_chunk.rhs;
                    if (selected.isNullAt(vi)) {
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }
                    const value = try common.readBinary(selected, vi, .binary_view);
                    builder.append(value) catch |err| return common.kernelAppendError(err);
                }
                produced += run_len;
                cond_index += run_len;
                values_index += run_len;
            }
            const out = builder.finish() catch |err| return common.kernelAppendError(err);
            return compute.Datum.fromArray(out);
        },
    }
}

fn ifElseFixedWidthKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    data_type: compute.DataType,
) compute.KernelError!compute.Datum {
    if (!common.isFilterFixedWidthType(data_type)) return error.UnsupportedType;
    const bit_width = data_type.bitWidth() orelse return error.UnsupportedType;
    if (bit_width % 8 != 0) return error.UnsupportedType;
    const byte_width = bit_width / 8;
    const out_len = try inferTernaryExecLen(args[0], args[1], args[2]);

    var values = try zcore.OwnedBuffer.init(ctx.tempAllocator(), out_len * byte_width);
    defer values.deinit();
    var validity = try zcore.OwnedBuffer.init(ctx.tempAllocator(), common.bitByteLength(out_len));
    defer validity.deinit();
    var null_count: usize = 0;
    var produced: usize = 0;

    var cond_iter = compute.UnaryExecChunkIterator.init(args[0]);
    var values_iter = try compute.BinaryExecChunkIterator.init(args[1], args[2]);
    var cond_chunk_opt: ?compute.UnaryExecChunk = null;
    defer if (cond_chunk_opt) |*c| c.deinit();
    var values_chunk_opt: ?compute.BinaryExecChunk = null;
    defer if (values_chunk_opt) |*c| c.deinit();
    var cond_index: usize = 0;
    var values_index: usize = 0;

    while (produced < out_len) {
        try ensureUnaryChunk(&cond_iter, &cond_chunk_opt, &cond_index);
        try ensureBinaryChunk(&values_iter, &values_chunk_opt, &values_index);
        if (cond_chunk_opt == null or values_chunk_opt == null) return error.InvalidInput;

        const cond_chunk = &cond_chunk_opt.?;
        const values_chunk = &values_chunk_opt.?;
        const run_len = @min(cond_chunk.len - cond_index, values_chunk.len - values_index);
        var j: usize = 0;
        while (j < run_len) : (j += 1) {
            const ci = cond_index + j;
            const vi = values_index + j;
            const out_i = produced + j;
            if (cond_chunk.values.isNullAt(ci)) {
                null_count += 1;
                continue;
            }
            const take_lhs = try common.readBool(cond_chunk.values, ci);
            const selected = if (take_lhs) values_chunk.lhs else values_chunk.rhs;
            if (selected.isNullAt(vi)) {
                null_count += 1;
                continue;
            }

            var scratch: [32]u8 = undefined;
            const src = try common.readFixedWidthBytes(selected, vi, data_type, byte_width, &scratch);
            const start = out_i * byte_width;
            const end = start + byte_width;
            @memcpy(values.data[start..end], src);
            common.setBit(validity.data[0..], out_i);
        }
        produced += run_len;
        cond_index += run_len;
        values_index += run_len;
    }

    const validity_shared = if (null_count == 0)
        zcore.SharedBuffer.empty
    else
        validity.toShared(common.bitByteLength(out_len)) catch |err| return common.kernelAppendError(err);
    errdefer if (null_count != 0) {
        var owned = validity_shared;
        owned.release();
    };

    const values_shared = values.toShared(out_len * byte_width) catch |err| return common.kernelAppendError(err);
    errdefer {
        var owned = values_shared;
        owned.release();
    }

    const buffers = ctx.tempAllocator().alloc(zcore.SharedBuffer, 2) catch return error.OutOfMemory;
    buffers[0] = validity_shared;
    buffers[1] = values_shared;
    const array_ref = zcore.ArrayRef.fromOwnedUnsafe(ctx.tempAllocator(), .{
        .data_type = data_type,
        .length = out_len,
        .null_count = null_count,
        .buffers = buffers,
    }) catch return error.OutOfMemory;
    return compute.Datum.fromArray(array_ref);
}

pub fn ifElseKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    if (args.len != 3) return error.InvalidArity;
    if (!common.onlyNoOptions(options)) return error.InvalidOptions;
    if (!args[0].dataType().eql(.{ .bool = {} })) return error.InvalidInput;
    const data_type = args[1].dataType();
    if (!data_type.eql(args[2].dataType())) return error.InvalidInput;

    return switch (data_type) {
        .string => ifElseStringKernel(ctx, args, .string),
        .large_string => ifElseStringKernel(ctx, args, .large_string),
        .string_view => ifElseStringKernel(ctx, args, .string_view),
        .binary => ifElseBinaryKernel(ctx, args, .binary),
        .large_binary => ifElseBinaryKernel(ctx, args, .large_binary),
        .binary_view => ifElseBinaryKernel(ctx, args, .binary_view),
        else => blk: {
            if (!common.isFilterFixedWidthType(data_type)) break :blk error.UnsupportedType;
            break :blk ifElseFixedWidthKernel(ctx, args, data_type);
        },
    };
}
