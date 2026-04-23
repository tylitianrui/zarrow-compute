const zcore = @import("zarrow-core");
const common = @import("common.zig");

const compute = common.compute;

fn filterBoolKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    filter_opts: compute.FilterOptions,
) compute.KernelError!compute.Datum {
    const input_len = try compute.inferBinaryExecLen(args[0], args[1]);
    var builder = try zcore.BooleanBuilder.init(ctx.tempAllocator(), input_len);
    defer builder.deinit();

    var iter = try compute.BinaryExecChunkIterator.init(args[0], args[1]);
    while (try iter.next()) |chunk_value| {
        var chunk = chunk_value;
        defer chunk.deinit();

        var i: usize = 0;
        while (i < chunk.len) : (i += 1) {
            if (chunk.rhs.isNullAt(i)) {
                if (filter_opts.drop_nulls) continue;
                builder.appendNull() catch |err| return common.kernelAppendError(err);
                continue;
            }

            const keep = try common.readBool(chunk.rhs, i);
            if (!keep) continue;

            if (chunk.lhs.isNullAt(i)) {
                builder.appendNull() catch |err| return common.kernelAppendError(err);
                continue;
            }

            const value = try common.readBool(chunk.lhs, i);
            builder.append(value) catch |err| return common.kernelAppendError(err);
        }
    }

    const out = builder.finish() catch |err| return common.kernelAppendError(err);
    return compute.Datum.fromArray(out);
}

fn filterStringKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    filter_opts: compute.FilterOptions,
    kind: common.StringValueKind,
) compute.KernelError!compute.Datum {
    const input_len = try compute.inferBinaryExecLen(args[0], args[1]);
    switch (kind) {
        .string => {
            var builder = try zcore.StringBuilder.init(ctx.tempAllocator(), input_len, 0);
            defer builder.deinit();
            var iter = try compute.BinaryExecChunkIterator.init(args[0], args[1]);
            while (try iter.next()) |chunk_value| {
                var chunk = chunk_value;
                defer chunk.deinit();
                var i: usize = 0;
                while (i < chunk.len) : (i += 1) {
                    if (chunk.rhs.isNullAt(i)) {
                        if (filter_opts.drop_nulls) continue;
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }
                    if (!(try common.readBool(chunk.rhs, i))) continue;
                    if (chunk.lhs.isNullAt(i)) {
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }
                    const value = try common.readString(chunk.lhs, i, .string);
                    builder.append(value) catch |err| return common.kernelAppendError(err);
                }
            }
            const out = builder.finish() catch |err| return common.kernelAppendError(err);
            return compute.Datum.fromArray(out);
        },
        .large_string => {
            var builder = try zcore.LargeStringBuilder.init(ctx.tempAllocator(), input_len, 0);
            defer builder.deinit();
            var iter = try compute.BinaryExecChunkIterator.init(args[0], args[1]);
            while (try iter.next()) |chunk_value| {
                var chunk = chunk_value;
                defer chunk.deinit();
                var i: usize = 0;
                while (i < chunk.len) : (i += 1) {
                    if (chunk.rhs.isNullAt(i)) {
                        if (filter_opts.drop_nulls) continue;
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }
                    if (!(try common.readBool(chunk.rhs, i))) continue;
                    if (chunk.lhs.isNullAt(i)) {
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }
                    const value = try common.readString(chunk.lhs, i, .large_string);
                    builder.append(value) catch |err| return common.kernelAppendError(err);
                }
            }
            const out = builder.finish() catch |err| return common.kernelAppendError(err);
            return compute.Datum.fromArray(out);
        },
        .string_view => {
            var builder = try zcore.StringViewBuilder.init(ctx.tempAllocator(), input_len, 0);
            defer builder.deinit();
            var iter = try compute.BinaryExecChunkIterator.init(args[0], args[1]);
            while (try iter.next()) |chunk_value| {
                var chunk = chunk_value;
                defer chunk.deinit();
                var i: usize = 0;
                while (i < chunk.len) : (i += 1) {
                    if (chunk.rhs.isNullAt(i)) {
                        if (filter_opts.drop_nulls) continue;
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }
                    if (!(try common.readBool(chunk.rhs, i))) continue;
                    if (chunk.lhs.isNullAt(i)) {
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }
                    const value = try common.readString(chunk.lhs, i, .string_view);
                    builder.append(value) catch |err| return common.kernelAppendError(err);
                }
            }
            const out = builder.finish() catch |err| return common.kernelAppendError(err);
            return compute.Datum.fromArray(out);
        },
    }
}

fn filterBinaryKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    filter_opts: compute.FilterOptions,
    kind: common.BinaryValueKind,
) compute.KernelError!compute.Datum {
    const input_len = try compute.inferBinaryExecLen(args[0], args[1]);
    switch (kind) {
        .binary => {
            var builder = try zcore.BinaryBuilder.init(ctx.tempAllocator(), input_len, 0);
            defer builder.deinit();
            var iter = try compute.BinaryExecChunkIterator.init(args[0], args[1]);
            while (try iter.next()) |chunk_value| {
                var chunk = chunk_value;
                defer chunk.deinit();
                var i: usize = 0;
                while (i < chunk.len) : (i += 1) {
                    if (chunk.rhs.isNullAt(i)) {
                        if (filter_opts.drop_nulls) continue;
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }
                    if (!(try common.readBool(chunk.rhs, i))) continue;
                    if (chunk.lhs.isNullAt(i)) {
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }
                    const value = try common.readBinary(chunk.lhs, i, .binary);
                    builder.append(value) catch |err| return common.kernelAppendError(err);
                }
            }
            const out = builder.finish() catch |err| return common.kernelAppendError(err);
            return compute.Datum.fromArray(out);
        },
        .large_binary => {
            var builder = try zcore.LargeBinaryBuilder.init(ctx.tempAllocator(), input_len, 0);
            defer builder.deinit();
            var iter = try compute.BinaryExecChunkIterator.init(args[0], args[1]);
            while (try iter.next()) |chunk_value| {
                var chunk = chunk_value;
                defer chunk.deinit();
                var i: usize = 0;
                while (i < chunk.len) : (i += 1) {
                    if (chunk.rhs.isNullAt(i)) {
                        if (filter_opts.drop_nulls) continue;
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }
                    if (!(try common.readBool(chunk.rhs, i))) continue;
                    if (chunk.lhs.isNullAt(i)) {
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }
                    const value = try common.readBinary(chunk.lhs, i, .large_binary);
                    builder.append(value) catch |err| return common.kernelAppendError(err);
                }
            }
            const out = builder.finish() catch |err| return common.kernelAppendError(err);
            return compute.Datum.fromArray(out);
        },
        .binary_view => {
            var builder = try zcore.BinaryViewBuilder.init(ctx.tempAllocator(), input_len, 0);
            defer builder.deinit();
            var iter = try compute.BinaryExecChunkIterator.init(args[0], args[1]);
            while (try iter.next()) |chunk_value| {
                var chunk = chunk_value;
                defer chunk.deinit();
                var i: usize = 0;
                while (i < chunk.len) : (i += 1) {
                    if (chunk.rhs.isNullAt(i)) {
                        if (filter_opts.drop_nulls) continue;
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }
                    if (!(try common.readBool(chunk.rhs, i))) continue;
                    if (chunk.lhs.isNullAt(i)) {
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }
                    const value = try common.readBinary(chunk.lhs, i, .binary_view);
                    builder.append(value) catch |err| return common.kernelAppendError(err);
                }
            }
            const out = builder.finish() catch |err| return common.kernelAppendError(err);
            return compute.Datum.fromArray(out);
        },
    }
}

fn filterFixedWidthKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    filter_opts: compute.FilterOptions,
    data_type: compute.DataType,
) compute.KernelError!compute.Datum {
    if (!common.isFilterFixedWidthType(data_type)) return error.UnsupportedType;
    const bit_width = data_type.bitWidth() orelse return error.UnsupportedType;
    if (bit_width % 8 != 0) return error.UnsupportedType;
    const byte_width = bit_width / 8;

    const input_len = try compute.inferBinaryExecLen(args[0], args[1]);
    var values = try zcore.OwnedBuffer.init(ctx.tempAllocator(), input_len * byte_width);
    defer values.deinit();
    var validity = try zcore.OwnedBuffer.init(ctx.tempAllocator(), common.bitByteLength(input_len));
    defer validity.deinit();

    var out_len: usize = 0;
    var null_count: usize = 0;
    var iter = try compute.BinaryExecChunkIterator.init(args[0], args[1]);
    while (try iter.next()) |chunk_value| {
        var chunk = chunk_value;
        defer chunk.deinit();

        var i: usize = 0;
        while (i < chunk.len) : (i += 1) {
            if (chunk.rhs.isNullAt(i)) {
                if (filter_opts.drop_nulls) continue;
                null_count += 1;
                out_len += 1;
                continue;
            }
            if (!(try common.readBool(chunk.rhs, i))) continue;

            if (chunk.lhs.isNullAt(i)) {
                null_count += 1;
                out_len += 1;
                continue;
            }

            var scratch: [32]u8 = undefined;
            const src = try common.readFixedWidthBytes(chunk.lhs, i, data_type, byte_width, &scratch);
            const start = out_len * byte_width;
            const end = start + byte_width;
            @memcpy(values.data[start..end], src);
            common.setBit(validity.data[0..], out_len);
            out_len += 1;
        }
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

pub fn filterKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    if (args.len != 2) return error.InvalidArity;
    const filter_opts = switch (options) {
        .filter => |o| o,
        else => return error.InvalidOptions,
    };
    const data_type = args[0].dataType();
    return switch (data_type) {
        .null => blk: {
            const input_len = try compute.inferBinaryExecLen(args[0], args[1]);
            var builder = try zcore.NullBuilder.init(ctx.tempAllocator(), input_len);
            defer builder.deinit();
            var iter = try compute.BinaryExecChunkIterator.init(args[0], args[1]);
            while (try iter.next()) |chunk_value| {
                var chunk = chunk_value;
                defer chunk.deinit();
                var i: usize = 0;
                while (i < chunk.len) : (i += 1) {
                    if (chunk.rhs.isNullAt(i)) {
                        if (filter_opts.drop_nulls) continue;
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        continue;
                    }
                    if (!(try common.readBool(chunk.rhs, i))) continue;
                    builder.appendNull() catch |err| return common.kernelAppendError(err);
                }
            }
            const out = builder.finish() catch |err| return common.kernelAppendError(err);
            break :blk compute.Datum.fromArray(out);
        },
        .bool => filterBoolKernel(ctx, args, filter_opts),
        .string => filterStringKernel(ctx, args, filter_opts, .string),
        .large_string => filterStringKernel(ctx, args, filter_opts, .large_string),
        .string_view => filterStringKernel(ctx, args, filter_opts, .string_view),
        .binary => filterBinaryKernel(ctx, args, filter_opts, .binary),
        .large_binary => filterBinaryKernel(ctx, args, filter_opts, .large_binary),
        .binary_view => filterBinaryKernel(ctx, args, filter_opts, .binary_view),
        else => blk: {
            if (!common.isFilterFixedWidthType(data_type)) break :blk error.UnsupportedType;
            break :blk filterFixedWidthKernel(ctx, args, filter_opts, data_type);
        },
    };
}
