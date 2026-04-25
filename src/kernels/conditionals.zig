const std = @import("std");
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

fn ifElseNullKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
) compute.KernelError!compute.Datum {
    const out_len = try inferTernaryExecLen(args[0], args[1], args[2]);
    var builder = try zcore.NullBuilder.init(ctx.tempAllocator(), out_len);
    defer builder.deinit();

    var i: usize = 0;
    while (i < out_len) : (i += 1) {
        builder.appendNull() catch |err| return common.kernelAppendError(err);
    }

    const out = builder.finish() catch |err| return common.kernelAppendError(err);
    return compute.Datum.fromArray(out);
}

fn ifElseBoolKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
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

    var builder = try zcore.BooleanBuilder.init(ctx.tempAllocator(), out_len);
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
            const value = try common.readBool(selected, vi);
            builder.append(value) catch |err| return common.kernelAppendError(err);
        }
        produced += run_len;
        cond_index += run_len;
        values_index += run_len;
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

fn ifElseStructKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    data_type: compute.DataType,
) compute.KernelError!compute.Datum {
    const struct_type = data_type.struct_;
    const out_len = try inferTernaryExecLen(args[0], args[1], args[2]);

    var validity = try zcore.OwnedBuffer.init(ctx.tempAllocator(), common.bitByteLength(out_len));
    defer validity.deinit();
    var null_count: usize = 0;

    var cond_iter = compute.UnaryExecChunkIterator.init(args[0]);
    var values_iter = try compute.BinaryExecChunkIterator.init(args[1], args[2]);
    var cond_chunk_opt: ?compute.UnaryExecChunk = null;
    defer if (cond_chunk_opt) |*c| c.deinit();
    var values_chunk_opt: ?compute.BinaryExecChunk = null;
    defer if (values_chunk_opt) |*c| c.deinit();
    var cond_index: usize = 0;
    var values_index: usize = 0;
    var produced: usize = 0;

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
            common.setBit(validity.data[0..], out_i);
        }

        produced += run_len;
        cond_index += run_len;
        values_index += run_len;
    }

    const children = ctx.tempAllocator().alloc(zcore.ArrayRef, struct_type.fields.len) catch return error.OutOfMemory;
    var child_count: usize = 0;
    errdefer {
        while (child_count > 0) {
            child_count -= 1;
            var child = children[child_count];
            child.release();
        }
        ctx.tempAllocator().free(children);
    }

    for (struct_type.fields, 0..) |field, field_index| {
        var child_cond = args[0].retain();
        defer child_cond.release();
        var child_lhs = try extractStructFieldDatum(args[1], field_index, field.data_type.*);
        defer child_lhs.release();
        var child_rhs = try extractStructFieldDatum(args[2], field_index, field.data_type.*);
        defer child_rhs.release();

        const child_args = [_]compute.Datum{
            child_cond,
            child_lhs,
            child_rhs,
        };
        var child_out = try ifElseKernel(ctx, child_args[0..], compute.Options.noneValue());
        defer child_out.release();
        if (!child_out.isArray()) return error.InvalidInput;
        children[field_index] = child_out.array.retain();
        child_count += 1;
    }

    const validity_shared = if (null_count == 0)
        zcore.SharedBuffer.empty
    else
        validity.toShared(common.bitByteLength(out_len)) catch |err| return common.kernelAppendError(err);
    errdefer if (null_count != 0) {
        var owned = validity_shared;
        owned.release();
    };

    const buffers = ctx.tempAllocator().alloc(zcore.SharedBuffer, 1) catch return error.OutOfMemory;
    buffers[0] = validity_shared;
    const out = zcore.ArrayRef.fromOwnedUnsafe(ctx.tempAllocator(), .{
        .data_type = data_type,
        .length = out_len,
        .null_count = null_count,
        .buffers = buffers,
        .children = children,
    }) catch return error.OutOfMemory;
    return compute.Datum.fromArray(out);
}

const ListKind = enum {
    list,
    large_list,
    fixed_size_list,
};

fn mapConcatArrayError(err: anyerror) compute.KernelError {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.UnsupportedType => error.UnsupportedType,
        else => error.InvalidInput,
    };
}

fn listValuesRefFromArray(array_ref: zcore.ArrayRef, kind: ListKind) compute.KernelError!zcore.ArrayRef {
    return switch (kind) {
        .list => blk: {
            if (array_ref.data().data_type != .list) return error.InvalidInput;
            const list_array = zcore.ListArray{ .data = array_ref.data() };
            break :blk list_array.valuesRef().retain();
        },
        .large_list => blk: {
            if (array_ref.data().data_type != .large_list) return error.InvalidInput;
            const list_array = zcore.LargeListArray{ .data = array_ref.data() };
            break :blk list_array.valuesRef().retain();
        },
        .fixed_size_list => blk: {
            if (array_ref.data().data_type != .fixed_size_list) return error.InvalidInput;
            const list_array = zcore.FixedSizeListArray{ .data = array_ref.data() };
            break :blk list_array.valuesRef().retain();
        },
    };
}

fn datumListValueSlice(
    datum: compute.Datum,
    logical_index: usize,
    kind: ListKind,
) compute.KernelError!zcore.ArrayRef {
    return switch (kind) {
        .list => compute.datumListValueAt(datum, logical_index),
        .large_list => compute.datumLargeListValueAt(datum, logical_index),
        .fixed_size_list => compute.datumFixedSizeListValueAt(datum, logical_index),
    };
}

fn extractListValueSlice(
    value: compute.ExecChunkValue,
    logical_index: usize,
    kind: ListKind,
) compute.KernelError!zcore.ArrayRef {
    return switch (value) {
        .array => |array_ref| blk: {
            var datum = compute.Datum.fromArray(array_ref.retain());
            defer datum.release();
            break :blk datumListValueSlice(datum, logical_index, kind);
        },
        .scalar => |scalar| blk: {
            var datum = compute.Datum.fromScalar(scalar.retain());
            defer datum.release();
            break :blk datumListValueSlice(datum, logical_index, kind);
        },
    };
}

fn extractListFillerSlice(
    value: compute.ExecChunkValue,
    logical_index: usize,
    kind: ListKind,
) compute.KernelError!zcore.ArrayRef {
    return extractListValueSlice(value, logical_index, kind) catch |err| switch (err) {
        error.InvalidInput => switch (value) {
            .array => err,
            .scalar => |scalar| blk: {
                var payload = scalar.payloadArray() catch return err;
                defer payload.release();
                var payload_datum = compute.Datum.fromArray(payload.retain());
                defer payload_datum.release();
                break :blk datumListValueSlice(payload_datum, 0, kind);
            },
        },
        else => err,
    };
}

fn extractFirstListFillerSlice(
    values: []const compute.ExecChunkValue,
    logical_index: usize,
    kind: ListKind,
) compute.KernelError!zcore.ArrayRef {
    for (values) |value| {
        return extractListFillerSlice(value, logical_index, kind) catch |err| switch (err) {
            error.InvalidInput => continue,
            else => return err,
        };
    }
    return error.InvalidInput;
}

fn extractFirstListFillerSliceFromIndices(
    values: []const compute.ExecChunkValue,
    arg_indices: []const usize,
    logical_index: usize,
    kind: ListKind,
) compute.KernelError!zcore.ArrayRef {
    for (arg_indices) |arg_index| {
        const value = values[arg_index];
        return extractListFillerSlice(value, logical_index, kind) catch |err| switch (err) {
            error.InvalidInput => continue,
            else => return err,
        };
    }
    return error.InvalidInput;
}

fn firstMatchingListValuesRefFromArgs(
    args: []const compute.Datum,
    data_type: compute.DataType,
    kind: ListKind,
) compute.KernelError!zcore.ArrayRef {
    for (args) |arg| {
        if (!arg.dataType().eql(data_type)) continue;
        var source = arg.retain();
        defer source.release();

        var first_value = datumListValueSlice(source, 0, kind) catch |err| switch (err) {
            error.InvalidInput => blk: {
                break :blk switch (source) {
                    .array => |array_ref| blk2: {
                        if (array_ref.data().length != 0) return err;
                        break :blk2 try listValuesRefFromArray(array_ref, kind);
                    },
                    .chunked => |chunks| blk2: {
                        if (chunks.len() != 0 or chunks.numChunks() == 0) return err;
                        const chunk = chunks.chunk(0).*;
                        break :blk2 try listValuesRefFromArray(chunk, kind);
                    },
                    .scalar => |scalar| blk2: {
                        var payload = scalar.payloadArray() catch return err;
                        defer payload.release();
                        break :blk2 try listValuesRefFromArray(payload, kind);
                    },
                };
            },
            else => return err,
        };
        defer first_value.release();
        return first_value.slice(0, 0) catch |err| return mapSelectionInputError(err);
    }
    return error.InvalidInput;
}

fn concatSelectedListValues(
    allocator: std.mem.Allocator,
    value_type: compute.DataType,
    selected_values: []const zcore.ArrayRef,
    args: []const compute.Datum,
    data_type: compute.DataType,
    kind: ListKind,
) compute.KernelError!zcore.ArrayRef {
    if (selected_values.len == 0) {
        return try firstMatchingListValuesRefFromArgs(args, data_type, kind);
    }

    return compute.concatArrayRefs(allocator, value_type, selected_values) catch |err| return mapConcatArrayError(err);
}

fn ifElseListKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    data_type: compute.DataType,
) compute.KernelError!compute.Datum {
    const out_len = try inferTernaryExecLen(args[0], args[1], args[2]);
    const selected_values = ctx.tempAllocator().alloc(zcore.ArrayRef, out_len) catch return error.OutOfMemory;
    var selected_count: usize = 0;
    defer {
        while (selected_count > 0) {
            selected_count -= 1;
            var value = selected_values[selected_count];
            value.release();
        }
        ctx.tempAllocator().free(selected_values);
    }

    var cond_iter = compute.UnaryExecChunkIterator.init(args[0]);
    var values_iter = try compute.BinaryExecChunkIterator.init(args[1], args[2]);
    var cond_chunk_opt: ?compute.UnaryExecChunk = null;
    defer if (cond_chunk_opt) |*c| c.deinit();
    var values_chunk_opt: ?compute.BinaryExecChunk = null;
    defer if (values_chunk_opt) |*c| c.deinit();
    var cond_index: usize = 0;
    var values_index: usize = 0;
    var produced: usize = 0;

    return switch (data_type) {
        .list => |list_type| blk: {
            var builder = try zcore.ListBuilder.init(ctx.tempAllocator(), out_len, list_type.value_field);
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

                    const value = try extractListValueSlice(selected, vi, .list);
                    const value_len = value.data().length;
                    builder.appendLen(value_len) catch |err| return common.kernelAppendError(err);
                    selected_values[selected_count] = value;
                    selected_count += 1;
                }

                produced += run_len;
                cond_index += run_len;
                values_index += run_len;
            }

            var merged_values = try concatSelectedListValues(
                ctx.tempAllocator(),
                list_type.value_field.data_type.*,
                selected_values[0..selected_count],
                args,
                data_type,
                .list,
            );
            defer merged_values.release();

            const out = builder.finish(merged_values) catch |err| return common.kernelAppendError(err);
            break :blk compute.Datum.fromArray(out);
        },
        .large_list => |list_type| blk: {
            var builder = try zcore.LargeListBuilder.init(ctx.tempAllocator(), out_len, list_type.value_field);
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

                    const value = try extractListValueSlice(selected, vi, .large_list);
                    const value_len = value.data().length;
                    builder.appendLen(value_len) catch |err| return common.kernelAppendError(err);
                    selected_values[selected_count] = value;
                    selected_count += 1;
                }

                produced += run_len;
                cond_index += run_len;
                values_index += run_len;
            }

            var merged_values = try concatSelectedListValues(
                ctx.tempAllocator(),
                list_type.value_field.data_type.*,
                selected_values[0..selected_count],
                args,
                data_type,
                .large_list,
            );
            defer merged_values.release();

            const out = builder.finish(merged_values) catch |err| return common.kernelAppendError(err);
            break :blk compute.Datum.fromArray(out);
        },
        .fixed_size_list => |list_type| blk: {
            if (list_type.list_size < 0) break :blk error.InvalidInput;
            const list_size = std.math.cast(usize, list_type.list_size) orelse break :blk error.InvalidInput;
            var builder = zcore.FixedSizeListBuilder.init(ctx.tempAllocator(), list_type.value_field, list_size) catch |err| return common.kernelAppendError(err);
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
                        const fallback_values = [_]compute.ExecChunkValue{
                            values_chunk.lhs,
                            values_chunk.rhs,
                        };
                        const filler = try extractFirstListFillerSlice(
                            fallback_values[0..],
                            vi,
                            .fixed_size_list,
                        );
                        selected_values[selected_count] = filler;
                        selected_count += 1;
                        continue;
                    }

                    const take_lhs = try common.readBool(cond_chunk.values, ci);
                    const selected = if (take_lhs) values_chunk.lhs else values_chunk.rhs;
                    if (selected.isNullAt(vi)) {
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        const fallback_values = [_]compute.ExecChunkValue{
                            values_chunk.lhs,
                            values_chunk.rhs,
                        };
                        const filler = try extractFirstListFillerSlice(
                            fallback_values[0..],
                            vi,
                            .fixed_size_list,
                        );
                        selected_values[selected_count] = filler;
                        selected_count += 1;
                        continue;
                    }

                    const value = try extractListValueSlice(selected, vi, .fixed_size_list);
                    builder.appendValid() catch |err| return common.kernelAppendError(err);
                    selected_values[selected_count] = value;
                    selected_count += 1;
                }

                produced += run_len;
                cond_index += run_len;
                values_index += run_len;
            }

            var merged_values = try concatSelectedListValues(
                ctx.tempAllocator(),
                list_type.value_field.data_type.*,
                selected_values[0..selected_count],
                args,
                data_type,
                .fixed_size_list,
            );
            defer merged_values.release();

            const out = builder.finish(merged_values) catch |err| return common.kernelAppendError(err);
            break :blk compute.Datum.fromArray(out);
        },
        else => error.UnsupportedType,
    };
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
        .null => ifElseNullKernel(ctx, args),
        .bool => ifElseBoolKernel(ctx, args),
        .string => ifElseStringKernel(ctx, args, .string),
        .large_string => ifElseStringKernel(ctx, args, .large_string),
        .string_view => ifElseStringKernel(ctx, args, .string_view),
        .binary => ifElseBinaryKernel(ctx, args, .binary),
        .large_binary => ifElseBinaryKernel(ctx, args, .large_binary),
        .binary_view => ifElseBinaryKernel(ctx, args, .binary_view),
        .list, .large_list, .fixed_size_list => ifElseListKernel(ctx, args, data_type),
        .struct_ => ifElseStructKernel(ctx, args, data_type),
        else => blk: {
            if (!common.isFilterFixedWidthType(data_type)) break :blk error.UnsupportedType;
            break :blk ifElseFixedWidthKernel(ctx, args, data_type);
        },
    };
}

const SelectionKind = enum {
    coalesce,
    choose,
    case_when,
};

const SelectionConfig = struct {
    kind: SelectionKind,
    case_pair_count: usize = 0,
    has_else: bool = false,
};

fn selectionValueArgCount(config: SelectionConfig, args_len: usize) usize {
    return switch (config.kind) {
        .coalesce => args_len,
        .choose => args_len - 1,
        .case_when => config.case_pair_count + (if (config.has_else) @as(usize, 1) else @as(usize, 0)),
    };
}

fn fillSelectionValueArgIndices(
    out: []usize,
    config: SelectionConfig,
    args_len: usize,
) compute.KernelError!void {
    switch (config.kind) {
        .coalesce => {
            if (out.len != args_len) return error.InvalidInput;
            for (out, 0..) |*slot, i| slot.* = i;
        },
        .choose => {
            if (out.len != args_len - 1) return error.InvalidInput;
            for (out, 0..) |*slot, i| slot.* = i + 1;
        },
        .case_when => {
            const expected = config.case_pair_count + (if (config.has_else) @as(usize, 1) else @as(usize, 0));
            if (out.len != expected) return error.InvalidInput;
            for (out, 0..) |*slot, i| {
                if (i < config.case_pair_count) {
                    slot.* = i * 2 + 1;
                } else {
                    if (!config.has_else or args_len == 0) return error.InvalidInput;
                    slot.* = args_len - 1;
                }
            }
        },
    }
}

fn selectionValueSlotForArgIndex(value_arg_indices: []const usize, arg_index: usize) ?usize {
    for (value_arg_indices, 0..) |value_arg_index, slot| {
        if (value_arg_index == arg_index) return slot;
    }
    return null;
}

const CaseWhenStructArgs = struct {
    allocator: std.mem.Allocator,
    args: []compute.Datum,
    config: SelectionConfig,

    fn deinit(self: *CaseWhenStructArgs) void {
        for (self.args) |*arg| {
            arg.release();
        }
        self.allocator.free(self.args);
        self.* = undefined;
    }
};

fn mapSelectionInputError(err: anyerror) compute.KernelError {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidInput,
    };
}

fn parseCaseWhenStructConfig(args: []const compute.Datum) compute.KernelError!SelectionConfig {
    if (args.len < 2) return error.InvalidArity;
    const cond_struct = switch (args[0].dataType()) {
        .struct_ => |value| value,
        else => return error.InvalidInput,
    };
    const case_pair_count = cond_struct.fields.len;
    if (case_pair_count == 0) return error.InvalidArity;

    const value_count = args.len - 1;
    const has_else = value_count == case_pair_count + 1;
    if (!(value_count == case_pair_count or has_else)) return error.InvalidArity;

    return .{
        .kind = .case_when,
        .case_pair_count = case_pair_count,
        .has_else = has_else,
    };
}

fn extractStructFieldDatum(
    struct_conditions: compute.Datum,
    field_index: usize,
    field_type: compute.DataType,
) compute.KernelError!compute.Datum {
    var source = struct_conditions.retain();
    defer source.release();
    var field = try compute.datumStructField(source, field_index);
    if (!field.dataType().eql(field_type)) {
        field.release();
        return error.InvalidInput;
    }
    return field;
}

fn buildCaseWhenStructArgs(
    allocator: std.mem.Allocator,
    args: []const compute.Datum,
) compute.KernelError!CaseWhenStructArgs {
    const config = try parseCaseWhenStructConfig(args);
    const cond_struct = args[0].dataType().struct_;

    const pair_args_len = config.case_pair_count * 2 + (if (config.has_else) @as(usize, 1) else @as(usize, 0));
    const pair_args = allocator.alloc(compute.Datum, pair_args_len) catch return error.OutOfMemory;
    var initialized: usize = 0;
    errdefer {
        while (initialized > 0) {
            initialized -= 1;
            var datum = pair_args[initialized];
            datum.release();
        }
        allocator.free(pair_args);
    }

    var pair_index: usize = 0;
    while (pair_index < config.case_pair_count) : (pair_index += 1) {
        const cond_arg_index = pair_index * 2;
        pair_args[cond_arg_index] = try extractStructFieldDatum(
            args[0],
            pair_index,
            cond_struct.fields[pair_index].data_type.*,
        );
        initialized += 1;

        pair_args[cond_arg_index + 1] = args[pair_index + 1].retain();
        initialized += 1;
    }

    if (config.has_else) {
        pair_args[pair_args.len - 1] = args[args.len - 1].retain();
        initialized += 1;
    }

    return .{
        .allocator = allocator,
        .args = pair_args,
        .config = config,
    };
}

fn resolveSelectedArgIndex(
    values: []const compute.ExecChunkValue,
    row: usize,
    config: SelectionConfig,
) compute.KernelError!?usize {
    return switch (config.kind) {
        .coalesce => blk: {
            for (values, 0..) |value, arg_index| {
                if (!value.isNullAt(row)) break :blk arg_index;
            }
            break :blk null;
        },
        .choose => blk: {
            if (values.len < 2) return error.InvalidArity;
            if (values[0].isNullAt(row)) break :blk null;
            const index = try common.readChooseIndex(values[0], row);
            const value_count = values.len - 1;
            if (index >= value_count) return error.InvalidInput;
            break :blk index + 1;
        },
        .case_when => blk: {
            var pair_index: usize = 0;
            while (pair_index < config.case_pair_count) : (pair_index += 1) {
                const cond_index = pair_index * 2;
                const value_index = cond_index + 1;
                if (values[cond_index].isNullAt(row)) continue;
                const cond = try common.readBool(values[cond_index], row);
                if (cond) break :blk value_index;
            }
            if (config.has_else) break :blk values.len - 1;
            break :blk null;
        },
    };
}

fn selectNullKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    config: SelectionConfig,
) compute.KernelError!compute.Datum {
    const out_len = try compute.inferNaryExecLen(args);
    var builder = try zcore.NullBuilder.init(ctx.tempAllocator(), out_len);
    defer builder.deinit();

    var iter = try compute.NaryExecChunkIterator.init(ctx.tempAllocator(), args);
    defer iter.deinit();
    while (try iter.next()) |chunk_value| {
        var chunk = chunk_value;
        defer chunk.deinit();
        var i: usize = 0;
        while (i < chunk.len) : (i += 1) {
            _ = try resolveSelectedArgIndex(chunk.values, i, config);
            builder.appendNull() catch |err| return common.kernelAppendError(err);
        }
    }

    const out = builder.finish() catch |err| return common.kernelAppendError(err);
    return compute.Datum.fromArray(out);
}

fn selectBoolKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    config: SelectionConfig,
) compute.KernelError!compute.Datum {
    const out_len = try compute.inferNaryExecLen(args);
    var builder = try zcore.BooleanBuilder.init(ctx.tempAllocator(), out_len);
    defer builder.deinit();

    var iter = try compute.NaryExecChunkIterator.init(ctx.tempAllocator(), args);
    defer iter.deinit();
    while (try iter.next()) |chunk_value| {
        var chunk = chunk_value;
        defer chunk.deinit();
        var i: usize = 0;
        while (i < chunk.len) : (i += 1) {
            const selected_index = try resolveSelectedArgIndex(chunk.values, i, config);
            if (selected_index == null) {
                builder.appendNull() catch |err| return common.kernelAppendError(err);
                continue;
            }
            const selected = chunk.values[selected_index.?];
            if (selected.isNullAt(i)) {
                builder.appendNull() catch |err| return common.kernelAppendError(err);
                continue;
            }
            const value = try common.readBool(selected, i);
            builder.append(value) catch |err| return common.kernelAppendError(err);
        }
    }

    const out = builder.finish() catch |err| return common.kernelAppendError(err);
    return compute.Datum.fromArray(out);
}

fn selectStringKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    config: SelectionConfig,
    kind: common.StringValueKind,
) compute.KernelError!compute.Datum {
    const out_len = try compute.inferNaryExecLen(args);
    switch (kind) {
        .string => {
            var builder = try zcore.StringBuilder.init(ctx.tempAllocator(), out_len, 0);
            defer builder.deinit();

            var iter = try compute.NaryExecChunkIterator.init(ctx.tempAllocator(), args);
            defer iter.deinit();
            while (try iter.next()) |chunk_value| {
                var chunk = chunk_value;
                defer chunk.deinit();
                var i: usize = 0;
                while (i < chunk.len) : (i += 1) {
                    const selected_index = try resolveSelectedArgIndex(chunk.values, i, config);
                    if (selected_index == null) {
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }
                    const selected = chunk.values[selected_index.?];
                    if (selected.isNullAt(i)) {
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }
                    const value = try common.readString(selected, i, .string);
                    builder.append(value) catch |err| return common.kernelAppendError(err);
                }
            }

            const out = builder.finish() catch |err| return common.kernelAppendError(err);
            return compute.Datum.fromArray(out);
        },
        .large_string => {
            var builder = try zcore.LargeStringBuilder.init(ctx.tempAllocator(), out_len, 0);
            defer builder.deinit();

            var iter = try compute.NaryExecChunkIterator.init(ctx.tempAllocator(), args);
            defer iter.deinit();
            while (try iter.next()) |chunk_value| {
                var chunk = chunk_value;
                defer chunk.deinit();
                var i: usize = 0;
                while (i < chunk.len) : (i += 1) {
                    const selected_index = try resolveSelectedArgIndex(chunk.values, i, config);
                    if (selected_index == null) {
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }
                    const selected = chunk.values[selected_index.?];
                    if (selected.isNullAt(i)) {
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }
                    const value = try common.readString(selected, i, .large_string);
                    builder.append(value) catch |err| return common.kernelAppendError(err);
                }
            }

            const out = builder.finish() catch |err| return common.kernelAppendError(err);
            return compute.Datum.fromArray(out);
        },
        .string_view => {
            var builder = try zcore.StringViewBuilder.init(ctx.tempAllocator(), out_len, 0);
            defer builder.deinit();

            var iter = try compute.NaryExecChunkIterator.init(ctx.tempAllocator(), args);
            defer iter.deinit();
            while (try iter.next()) |chunk_value| {
                var chunk = chunk_value;
                defer chunk.deinit();
                var i: usize = 0;
                while (i < chunk.len) : (i += 1) {
                    const selected_index = try resolveSelectedArgIndex(chunk.values, i, config);
                    if (selected_index == null) {
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }
                    const selected = chunk.values[selected_index.?];
                    if (selected.isNullAt(i)) {
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }
                    const value = try common.readString(selected, i, .string_view);
                    builder.append(value) catch |err| return common.kernelAppendError(err);
                }
            }

            const out = builder.finish() catch |err| return common.kernelAppendError(err);
            return compute.Datum.fromArray(out);
        },
    }
}

fn selectBinaryKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    config: SelectionConfig,
    kind: common.BinaryValueKind,
) compute.KernelError!compute.Datum {
    const out_len = try compute.inferNaryExecLen(args);
    switch (kind) {
        .binary => {
            var builder = try zcore.BinaryBuilder.init(ctx.tempAllocator(), out_len, 0);
            defer builder.deinit();

            var iter = try compute.NaryExecChunkIterator.init(ctx.tempAllocator(), args);
            defer iter.deinit();
            while (try iter.next()) |chunk_value| {
                var chunk = chunk_value;
                defer chunk.deinit();
                var i: usize = 0;
                while (i < chunk.len) : (i += 1) {
                    const selected_index = try resolveSelectedArgIndex(chunk.values, i, config);
                    if (selected_index == null) {
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }
                    const selected = chunk.values[selected_index.?];
                    if (selected.isNullAt(i)) {
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }
                    const value = try common.readBinary(selected, i, .binary);
                    builder.append(value) catch |err| return common.kernelAppendError(err);
                }
            }

            const out = builder.finish() catch |err| return common.kernelAppendError(err);
            return compute.Datum.fromArray(out);
        },
        .large_binary => {
            var builder = try zcore.LargeBinaryBuilder.init(ctx.tempAllocator(), out_len, 0);
            defer builder.deinit();

            var iter = try compute.NaryExecChunkIterator.init(ctx.tempAllocator(), args);
            defer iter.deinit();
            while (try iter.next()) |chunk_value| {
                var chunk = chunk_value;
                defer chunk.deinit();
                var i: usize = 0;
                while (i < chunk.len) : (i += 1) {
                    const selected_index = try resolveSelectedArgIndex(chunk.values, i, config);
                    if (selected_index == null) {
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }
                    const selected = chunk.values[selected_index.?];
                    if (selected.isNullAt(i)) {
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }
                    const value = try common.readBinary(selected, i, .large_binary);
                    builder.append(value) catch |err| return common.kernelAppendError(err);
                }
            }

            const out = builder.finish() catch |err| return common.kernelAppendError(err);
            return compute.Datum.fromArray(out);
        },
        .binary_view => {
            var builder = try zcore.BinaryViewBuilder.init(ctx.tempAllocator(), out_len, 0);
            defer builder.deinit();

            var iter = try compute.NaryExecChunkIterator.init(ctx.tempAllocator(), args);
            defer iter.deinit();
            while (try iter.next()) |chunk_value| {
                var chunk = chunk_value;
                defer chunk.deinit();
                var i: usize = 0;
                while (i < chunk.len) : (i += 1) {
                    const selected_index = try resolveSelectedArgIndex(chunk.values, i, config);
                    if (selected_index == null) {
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }
                    const selected = chunk.values[selected_index.?];
                    if (selected.isNullAt(i)) {
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }
                    const value = try common.readBinary(selected, i, .binary_view);
                    builder.append(value) catch |err| return common.kernelAppendError(err);
                }
            }

            const out = builder.finish() catch |err| return common.kernelAppendError(err);
            return compute.Datum.fromArray(out);
        },
    }
}

fn selectListKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    config: SelectionConfig,
    data_type: compute.DataType,
) compute.KernelError!compute.Datum {
    const out_len = try compute.inferNaryExecLen(args);
    const value_arg_count = selectionValueArgCount(config, args.len);
    const value_arg_indices = ctx.tempAllocator().alloc(usize, value_arg_count) catch return error.OutOfMemory;
    defer ctx.tempAllocator().free(value_arg_indices);
    try fillSelectionValueArgIndices(value_arg_indices, config, args.len);

    const selected_values = ctx.tempAllocator().alloc(zcore.ArrayRef, out_len) catch return error.OutOfMemory;
    var selected_count: usize = 0;
    defer {
        while (selected_count > 0) {
            selected_count -= 1;
            var value = selected_values[selected_count];
            value.release();
        }
        ctx.tempAllocator().free(selected_values);
    }

    return switch (data_type) {
        .list => |list_type| blk: {
            var builder = try zcore.ListBuilder.init(ctx.tempAllocator(), out_len, list_type.value_field);
            defer builder.deinit();

            var iter = try compute.NaryExecChunkIterator.init(ctx.tempAllocator(), args);
            defer iter.deinit();
            while (try iter.next()) |chunk_value| {
                var chunk = chunk_value;
                defer chunk.deinit();
                var i: usize = 0;
                while (i < chunk.len) : (i += 1) {
                    const selected_index = try resolveSelectedArgIndex(chunk.values, i, config);
                    if (selected_index == null) {
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }

                    const selected = chunk.values[selected_index.?];
                    if (selected.isNullAt(i)) {
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }

                    const value = try extractListValueSlice(selected, i, .list);
                    const value_len = value.data().length;
                    builder.appendLen(value_len) catch |err| return common.kernelAppendError(err);
                    selected_values[selected_count] = value;
                    selected_count += 1;
                }
            }

            var merged_values = try concatSelectedListValues(
                ctx.tempAllocator(),
                list_type.value_field.data_type.*,
                selected_values[0..selected_count],
                args,
                data_type,
                .list,
            );
            defer merged_values.release();

            const out = builder.finish(merged_values) catch |err| return common.kernelAppendError(err);
            break :blk compute.Datum.fromArray(out);
        },
        .large_list => |list_type| blk: {
            var builder = try zcore.LargeListBuilder.init(ctx.tempAllocator(), out_len, list_type.value_field);
            defer builder.deinit();

            var iter = try compute.NaryExecChunkIterator.init(ctx.tempAllocator(), args);
            defer iter.deinit();
            while (try iter.next()) |chunk_value| {
                var chunk = chunk_value;
                defer chunk.deinit();
                var i: usize = 0;
                while (i < chunk.len) : (i += 1) {
                    const selected_index = try resolveSelectedArgIndex(chunk.values, i, config);
                    if (selected_index == null) {
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }

                    const selected = chunk.values[selected_index.?];
                    if (selected.isNullAt(i)) {
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }

                    const value = try extractListValueSlice(selected, i, .large_list);
                    const value_len = value.data().length;
                    builder.appendLen(value_len) catch |err| return common.kernelAppendError(err);
                    selected_values[selected_count] = value;
                    selected_count += 1;
                }
            }

            var merged_values = try concatSelectedListValues(
                ctx.tempAllocator(),
                list_type.value_field.data_type.*,
                selected_values[0..selected_count],
                args,
                data_type,
                .large_list,
            );
            defer merged_values.release();

            const out = builder.finish(merged_values) catch |err| return common.kernelAppendError(err);
            break :blk compute.Datum.fromArray(out);
        },
        .fixed_size_list => |list_type| blk: {
            if (list_type.list_size < 0) break :blk error.InvalidInput;
            const list_size = std.math.cast(usize, list_type.list_size) orelse break :blk error.InvalidInput;
            if (value_arg_count == 0) break :blk error.InvalidInput;

            var builder = zcore.FixedSizeListBuilder.init(ctx.tempAllocator(), list_type.value_field, list_size) catch |err| return common.kernelAppendError(err);
            defer builder.deinit();

            var iter = try compute.NaryExecChunkIterator.init(ctx.tempAllocator(), args);
            defer iter.deinit();
            while (try iter.next()) |chunk_value| {
                var chunk = chunk_value;
                defer chunk.deinit();
                var i: usize = 0;
                while (i < chunk.len) : (i += 1) {
                    const selected_index = try resolveSelectedArgIndex(chunk.values, i, config);
                    if (selected_index == null) {
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        const fallback = try extractFirstListFillerSliceFromIndices(
                            chunk.values,
                            value_arg_indices,
                            i,
                            .fixed_size_list,
                        );
                        selected_values[selected_count] = fallback;
                        selected_count += 1;
                        continue;
                    }

                    const selected = chunk.values[selected_index.?];
                    if (selected.isNullAt(i)) {
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        const filler = try extractFirstListFillerSliceFromIndices(
                            chunk.values,
                            value_arg_indices,
                            i,
                            .fixed_size_list,
                        );
                        selected_values[selected_count] = filler;
                        selected_count += 1;
                        continue;
                    }

                    const value = try extractListValueSlice(selected, i, .fixed_size_list);
                    builder.appendValid() catch |err| return common.kernelAppendError(err);
                    selected_values[selected_count] = value;
                    selected_count += 1;
                }
            }

            var merged_values = try concatSelectedListValues(
                ctx.tempAllocator(),
                list_type.value_field.data_type.*,
                selected_values[0..selected_count],
                args,
                data_type,
                .fixed_size_list,
            );
            defer merged_values.release();

            const out = builder.finish(merged_values) catch |err| return common.kernelAppendError(err);
            break :blk compute.Datum.fromArray(out);
        },
        else => error.UnsupportedType,
    };
}

fn selectStructKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    config: SelectionConfig,
    data_type: compute.DataType,
) compute.KernelError!compute.Datum {
    const struct_type = data_type.struct_;
    const out_len = try compute.inferNaryExecLen(args);
    const value_arg_count = selectionValueArgCount(config, args.len);
    const value_arg_indices = ctx.tempAllocator().alloc(usize, value_arg_count) catch return error.OutOfMemory;
    defer ctx.tempAllocator().free(value_arg_indices);
    try fillSelectionValueArgIndices(value_arg_indices, config, args.len);

    var validity = try zcore.OwnedBuffer.init(ctx.tempAllocator(), common.bitByteLength(out_len));
    defer validity.deinit();
    var null_count: usize = 0;
    var produced: usize = 0;
    var selection_indices_builder = try zcore.Int32Builder.init(ctx.tempAllocator(), out_len);
    defer selection_indices_builder.deinit();

    var iter = try compute.NaryExecChunkIterator.init(ctx.tempAllocator(), args);
    defer iter.deinit();
    while (try iter.next()) |chunk_value| {
        var chunk = chunk_value;
        defer chunk.deinit();
        var i: usize = 0;
        while (i < chunk.len) : (i += 1) {
            const out_index = produced + i;
            const selected_index = try resolveSelectedArgIndex(chunk.values, i, config);
            if (selected_index == null) {
                null_count += 1;
                selection_indices_builder.appendNull() catch |err| return common.kernelAppendError(err);
                continue;
            }
            const selected_arg_index = selected_index.?;
            const selected = chunk.values[selected_arg_index];
            if (selected.isNullAt(i)) {
                null_count += 1;
                selection_indices_builder.appendNull() catch |err| return common.kernelAppendError(err);
                continue;
            }

            const slot = selectionValueSlotForArgIndex(value_arg_indices, selected_arg_index) orelse return error.InvalidInput;
            const slot_i32 = std.math.cast(i32, slot) orelse return error.InvalidInput;
            selection_indices_builder.append(slot_i32) catch |err| return common.kernelAppendError(err);
            common.setBit(validity.data[0..], out_index);
        }
        produced += chunk.len;
    }
    var selection_indices = selection_indices_builder.finish() catch |err| return common.kernelAppendError(err);
    defer selection_indices.release();

    const children = ctx.tempAllocator().alloc(zcore.ArrayRef, struct_type.fields.len) catch return error.OutOfMemory;
    var child_count: usize = 0;
    errdefer {
        while (child_count > 0) {
            child_count -= 1;
            var child = children[child_count];
            child.release();
        }
        ctx.tempAllocator().free(children);
    }

    for (struct_type.fields, 0..) |field, field_index| {
        const child_args = ctx.tempAllocator().alloc(compute.Datum, value_arg_count + 1) catch return error.OutOfMemory;
        var child_arg_count: usize = 0;
        defer ctx.tempAllocator().free(child_args);
        defer {
            while (child_arg_count > 0) {
                child_arg_count -= 1;
                child_args[child_arg_count].release();
            }
        }

        child_args[0] = compute.Datum.fromArray(selection_indices.retain());
        child_arg_count += 1;
        for (value_arg_indices, 0..) |value_arg_index, slot| {
            child_args[slot + 1] = try extractStructFieldDatum(
                args[value_arg_index],
                field_index,
                field.data_type.*,
            );
            child_arg_count += 1;
        }

        var child_out = try chooseKernel(ctx, child_args, compute.Options.noneValue());
        defer child_out.release();
        if (!child_out.isArray()) return error.InvalidInput;
        children[field_index] = child_out.array.retain();
        child_count += 1;
    }

    const validity_shared = if (null_count == 0)
        zcore.SharedBuffer.empty
    else
        validity.toShared(common.bitByteLength(out_len)) catch |err| return common.kernelAppendError(err);
    errdefer if (null_count != 0) {
        var owned = validity_shared;
        owned.release();
    };

    const buffers = ctx.tempAllocator().alloc(zcore.SharedBuffer, 1) catch return error.OutOfMemory;
    buffers[0] = validity_shared;
    const out = zcore.ArrayRef.fromOwnedUnsafe(ctx.tempAllocator(), .{
        .data_type = data_type,
        .length = out_len,
        .null_count = null_count,
        .buffers = buffers,
        .children = children,
    }) catch return error.OutOfMemory;
    return compute.Datum.fromArray(out);
}

fn selectFixedWidthKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    config: SelectionConfig,
    data_type: compute.DataType,
) compute.KernelError!compute.Datum {
    if (!common.isFilterFixedWidthType(data_type)) return error.UnsupportedType;
    const bit_width = data_type.bitWidth() orelse return error.UnsupportedType;
    if (bit_width % 8 != 0) return error.UnsupportedType;
    const byte_width = bit_width / 8;
    const out_len = try compute.inferNaryExecLen(args);

    var values = try zcore.OwnedBuffer.init(ctx.tempAllocator(), out_len * byte_width);
    defer values.deinit();
    var validity = try zcore.OwnedBuffer.init(ctx.tempAllocator(), common.bitByteLength(out_len));
    defer validity.deinit();
    var null_count: usize = 0;
    var produced: usize = 0;

    var iter = try compute.NaryExecChunkIterator.init(ctx.tempAllocator(), args);
    defer iter.deinit();
    while (try iter.next()) |chunk_value| {
        var chunk = chunk_value;
        defer chunk.deinit();
        var i: usize = 0;
        while (i < chunk.len) : (i += 1) {
            const out_index = produced + i;
            const selected_index = try resolveSelectedArgIndex(chunk.values, i, config);
            if (selected_index == null) {
                null_count += 1;
                continue;
            }
            const selected = chunk.values[selected_index.?];
            if (selected.isNullAt(i)) {
                null_count += 1;
                continue;
            }

            var scratch: [32]u8 = undefined;
            const src = try common.readFixedWidthBytes(selected, i, data_type, byte_width, &scratch);
            const start = out_index * byte_width;
            const end = start + byte_width;
            @memcpy(values.data[start..end], src);
            common.setBit(validity.data[0..], out_index);
        }
        produced += chunk.len;
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
    const out = zcore.ArrayRef.fromOwnedUnsafe(ctx.tempAllocator(), .{
        .data_type = data_type,
        .length = out_len,
        .null_count = null_count,
        .buffers = buffers,
    }) catch return error.OutOfMemory;
    return compute.Datum.fromArray(out);
}

fn runSelectionKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    config: SelectionConfig,
    data_type: compute.DataType,
) compute.KernelError!compute.Datum {
    return switch (data_type) {
        .null => selectNullKernel(ctx, args, config),
        .bool => selectBoolKernel(ctx, args, config),
        .string => selectStringKernel(ctx, args, config, .string),
        .large_string => selectStringKernel(ctx, args, config, .large_string),
        .string_view => selectStringKernel(ctx, args, config, .string_view),
        .binary => selectBinaryKernel(ctx, args, config, .binary),
        .large_binary => selectBinaryKernel(ctx, args, config, .large_binary),
        .binary_view => selectBinaryKernel(ctx, args, config, .binary_view),
        .list, .large_list, .fixed_size_list => selectListKernel(ctx, args, config, data_type),
        .struct_ => selectStructKernel(ctx, args, config, data_type),
        else => blk: {
            if (!common.isFilterFixedWidthType(data_type)) break :blk error.UnsupportedType;
            break :blk selectFixedWidthKernel(ctx, args, config, data_type);
        },
    };
}

pub fn coalesceKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    if (args.len < 1) return error.InvalidArity;
    if (!common.onlyNoOptions(options)) return error.InvalidOptions;
    if (!common.variadicCoalesceSupported(args)) return error.InvalidInput;
    return runSelectionKernel(ctx, args, .{ .kind = .coalesce }, args[0].dataType());
}

pub fn chooseKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    if (args.len < 2) return error.InvalidArity;
    if (!common.onlyNoOptions(options)) return error.InvalidOptions;
    if (!common.variadicChooseSupported(args)) return error.InvalidInput;
    return runSelectionKernel(ctx, args, .{ .kind = .choose }, args[1].dataType());
}

pub fn caseWhenKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    if (args.len < 2) return error.InvalidArity;
    if (!common.onlyNoOptions(options)) return error.InvalidOptions;
    if (!common.variadicCaseWhenSupported(args)) return error.InvalidInput;
    var struct_args = try buildCaseWhenStructArgs(ctx.tempAllocator(), args);
    defer struct_args.deinit();
    return runSelectionKernel(ctx, struct_args.args, struct_args.config, args[1].dataType());
}
