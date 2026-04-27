const std = @import("std");
const zcore = @import("zarrow-core");
const common = @import("common.zig");

const compute = common.compute;

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

fn mapListInputError(err: anyerror) compute.KernelError {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
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
        return first_value.slice(0, 0) catch |slice_err| return mapListInputError(slice_err);
    }
    return error.InvalidInput;
}

fn concatFilteredListValues(
    allocator: std.mem.Allocator,
    value_type: compute.DataType,
    selected_values: []const zcore.ArrayRef,
    args: []const compute.Datum,
    data_type: compute.DataType,
    kind: ListKind,
) compute.KernelError!zcore.ArrayRef {
    if (selected_values.len == 0) {
        _ = args;
        _ = data_type;
        _ = kind;
        var empty = compute.datumBuildEmptyLikeWithAllocator(allocator, value_type) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            error.UnsupportedType => error.UnsupportedType,
            else => error.InvalidInput,
        };
        errdefer empty.release();
        if (!empty.isArray()) return error.InvalidInput;
        return empty.array.retain();
    }
    return compute.concatArrayRefs(allocator, value_type, selected_values) catch |err| return mapConcatArrayError(err);
}

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

fn filterListKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    filter_opts: compute.FilterOptions,
    data_type: compute.DataType,
) compute.KernelError!compute.Datum {
    const input_len = try compute.inferBinaryExecLen(args[0], args[1]);
    const selected_values = ctx.tempAllocator().alloc(zcore.ArrayRef, input_len) catch return error.OutOfMemory;
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
            var builder = try zcore.ListBuilder.init(ctx.tempAllocator(), input_len, list_type.value_field);
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

                    var value = try extractListValueSlice(chunk.lhs, i, .list);
                    errdefer value.release();
                    builder.appendLen(value.data().length) catch |err| return common.kernelAppendError(err);
                    selected_values[selected_count] = value;
                    selected_count += 1;
                }
            }

            var merged_values = try concatFilteredListValues(
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
            var builder = try zcore.LargeListBuilder.init(ctx.tempAllocator(), input_len, list_type.value_field);
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

                    var value = try extractListValueSlice(chunk.lhs, i, .large_list);
                    errdefer value.release();
                    builder.appendLen(value.data().length) catch |err| return common.kernelAppendError(err);
                    selected_values[selected_count] = value;
                    selected_count += 1;
                }
            }

            var merged_values = try concatFilteredListValues(
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
            const list_size = std.math.cast(usize, list_type.list_size) orelse return error.InvalidInput;
            var builder = zcore.FixedSizeListBuilder.init(ctx.tempAllocator(), list_type.value_field, list_size) catch |err| return common.kernelAppendError(err);
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
                        var filler = try extractListFillerSlice(chunk.lhs, i, .fixed_size_list);
                        errdefer filler.release();
                        selected_values[selected_count] = filler;
                        selected_count += 1;
                        continue;
                    }
                    if (!(try common.readBool(chunk.rhs, i))) continue;
                    if (chunk.lhs.isNullAt(i)) {
                        builder.appendNull() catch |err| return common.kernelAppendError(err);
                        var filler = try extractListFillerSlice(chunk.lhs, i, .fixed_size_list);
                        errdefer filler.release();
                        selected_values[selected_count] = filler;
                        selected_count += 1;
                        continue;
                    }

                    var value = try extractListValueSlice(chunk.lhs, i, .fixed_size_list);
                    errdefer value.release();
                    builder.appendValid() catch |err| return common.kernelAppendError(err);
                    selected_values[selected_count] = value;
                    selected_count += 1;
                }
            }

            var merged_values = try concatFilteredListValues(
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

fn extractStructFieldDatum(
    struct_datum: compute.Datum,
    field_index: usize,
    field_type: compute.DataType,
) compute.KernelError!compute.Datum {
    var source = struct_datum.retain();
    defer source.release();
    var field = try compute.datumStructField(source, field_index);
    if (!field.dataType().eql(field_type)) {
        field.release();
        return error.InvalidInput;
    }
    return field;
}

fn filterStructKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    filter_opts: compute.FilterOptions,
    data_type: compute.DataType,
) compute.KernelError!compute.Datum {
    const struct_type = data_type.struct_;
    const input_len = try compute.inferBinaryExecLen(args[0], args[1]);
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
            common.setBit(validity.data[0..], out_len);
            out_len += 1;
        }
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
        var child_value = try extractStructFieldDatum(args[0], field_index, field.data_type.*);
        defer child_value.release();
        const child_args = [_]compute.Datum{
            child_value.retain(),
            args[1].retain(),
        };
        defer {
            var d = child_args[0];
            d.release();
        }
        defer {
            var d = child_args[1];
            d.release();
        }

        var child_out = try filterKernel(ctx, child_args[0..], .{ .filter = filter_opts });
        defer child_out.release();
        if (!child_out.isArray()) return error.InvalidInput;
        if (child_out.array.data().length != out_len) return error.InvalidInput;
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
    if (data_type == .fixed_size_list) {
        return filterListKernel(ctx, args, filter_opts, data_type);
    }
    return compute.datumFilter(args[0], args[1], filter_opts) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.UnsupportedType => error.UnsupportedType,
        error.InvalidArity => error.InvalidArity,
        else => error.InvalidInput,
    };
}

pub fn dropNullKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    if (args.len != 1) return error.InvalidArity;
    if (!common.onlyNoOptions(options)) return error.InvalidOptions;

    const input_len: usize = switch (args[0]) {
        .array => |arr| arr.data().length,
        .chunked => |chunks| chunks.len(),
        .scalar => return error.InvalidInput,
    };

    var predicate_builder = try zcore.BooleanBuilder.init(ctx.tempAllocator(), input_len);
    defer predicate_builder.deinit();

    var iter = compute.UnaryExecChunkIterator.init(args[0]);
    while (try iter.next()) |chunk_value| {
        var chunk = chunk_value;
        defer chunk.deinit();

        var i: usize = 0;
        while (i < chunk.len) : (i += 1) {
            const keep = !chunk.values.isNullAt(i);
            predicate_builder.append(keep) catch |err| return common.kernelAppendError(err);
        }
    }

    var predicate = predicate_builder.finish() catch |err| return common.kernelAppendError(err);
    defer predicate.release();

    const filter_args = [_]compute.Datum{
        args[0].retain(),
        compute.Datum.fromArray(predicate.retain()),
    };
    defer {
        var d = filter_args[0];
        d.release();
    }
    defer {
        var d = filter_args[1];
        d.release();
    }

    return filterKernel(ctx, filter_args[0..], .{ .filter = .{} });
}
