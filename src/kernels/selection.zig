const std = @import("std");
const zcore = @import("zarrow-core");
const common = @import("common.zig");
const conditionals = @import("conditionals.zig");

const compute = common.compute;

fn mapArraySliceError(err: anyerror) compute.KernelError {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidInput,
    };
}

fn mapConcatArrayError(err: anyerror) compute.KernelError {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.UnsupportedType => error.UnsupportedType,
        else => error.InvalidInput,
    };
}

fn datumLen(datum: compute.Datum) usize {
    return switch (datum) {
        .array => |arr| arr.data().length,
        .chunked => |chunks| chunks.len(),
        .scalar => 1,
    };
}

fn normalizeToArray(ctx: *compute.ExecContext, datum: compute.Datum) compute.KernelError!zcore.ArrayRef {
    return switch (datum) {
        .array => |arr| arr.retain(),
        .chunked => |chunks| blk: {
            if (chunks.numChunks() == 0 or chunks.len() == 0) {
                var empty = try compute.datumBuildEmptyLikeWithAllocator(ctx.tempAllocator(), chunks.dataType());
                errdefer empty.release();
                if (!empty.isArray()) return error.InvalidInput;
                break :blk empty.array.retain();
            }
            break :blk compute.concatArrayRefs(ctx.tempAllocator(), chunks.dataType(), chunks.chunks()) catch |err| return mapConcatArrayError(err);
        },
        .scalar => error.InvalidInput,
    };
}

fn collectNullableIndices(
    allocator: std.mem.Allocator,
    datum: compute.Datum,
) compute.KernelError![]?usize {
    const out_len = datumLen(datum);
    const out = allocator.alloc(?usize, out_len) catch return error.OutOfMemory;
    errdefer allocator.free(out);

    var iter = compute.UnaryExecChunkIterator.init(datum);
    var write_idx: usize = 0;
    while (try iter.next()) |chunk_value| {
        var chunk = chunk_value;
        defer chunk.deinit();

        var i: usize = 0;
        while (i < chunk.len) : (i += 1) {
            if (chunk.unaryNullAt(i)) {
                out[write_idx] = null;
            } else {
                out[write_idx] = try common.readChooseIndex(chunk.values, i);
            }
            write_idx += 1;
        }
    }
    if (write_idx != out_len) return error.InvalidInput;
    return out;
}

fn gatherArrayByNullableIndices(
    ctx: *compute.ExecContext,
    values: zcore.ArrayRef,
    indices: []const ?usize,
) compute.KernelError!compute.Datum {
    const out_len = indices.len;
    if (out_len == 0) {
        return compute.datumBuildEmptyLikeWithAllocator(ctx.tempAllocator(), values.data().data_type);
    }

    const pieces = ctx.tempAllocator().alloc(zcore.ArrayRef, out_len) catch return error.OutOfMemory;
    var piece_count: usize = 0;
    defer {
        while (piece_count > 0) {
            piece_count -= 1;
            var piece = pieces[piece_count];
            piece.release();
        }
        ctx.tempAllocator().free(pieces);
    }

    var null_piece_datum = try compute.datumBuildNullLikeWithAllocator(ctx.tempAllocator(), values.data().data_type, 1);
    defer null_piece_datum.release();
    if (!null_piece_datum.isArray()) return error.InvalidInput;
    var null_piece = null_piece_datum.array.retain();
    defer null_piece.release();

    const input_len = values.data().length;
    for (indices) |maybe_index| {
        const piece = if (maybe_index) |index| blk: {
            if (index >= input_len) return error.InvalidInput;
            break :blk values.slice(index, 1) catch |err| return mapArraySliceError(err);
        } else null_piece.retain();
        pieces[piece_count] = piece;
        piece_count += 1;
    }

    const out = compute.concatArrayRefs(
        ctx.tempAllocator(),
        values.data().data_type,
        pieces[0..piece_count],
    ) catch |err| return mapConcatArrayError(err);
    return compute.Datum.fromArray(out);
}

fn gatherTakeIndices(
    allocator: std.mem.Allocator,
    datum: compute.Datum,
) compute.KernelError![]?usize {
    return collectNullableIndices(allocator, datum);
}

fn computeForwardFillIndices(
    allocator: std.mem.Allocator,
    values: zcore.ArrayRef,
) compute.KernelError![]?usize {
    const len = values.data().length;
    const indices = allocator.alloc(?usize, len) catch return error.OutOfMemory;
    errdefer allocator.free(indices);

    var last_seen: ?usize = null;
    var i: usize = 0;
    while (i < len) : (i += 1) {
        if (values.data().isNull(i)) {
            indices[i] = last_seen;
        } else {
            last_seen = i;
            indices[i] = i;
        }
    }
    return indices;
}

fn computeBackwardFillIndices(
    allocator: std.mem.Allocator,
    values: zcore.ArrayRef,
) compute.KernelError![]?usize {
    const len = values.data().length;
    const indices = allocator.alloc(?usize, len) catch return error.OutOfMemory;
    errdefer allocator.free(indices);

    var next_seen: ?usize = null;
    var i = len;
    while (i > 0) {
        i -= 1;
        if (values.data().isNull(i)) {
            indices[i] = next_seen;
        } else {
            next_seen = i;
            indices[i] = i;
        }
    }
    return indices;
}

pub fn takeKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    if (args.len != 2) return error.InvalidArity;
    if (!common.onlyNoOptions(options)) return error.InvalidOptions;
    if (!common.binaryTakeSupported(args)) return error.InvalidInput;

    var values = try normalizeToArray(ctx, args[0]);
    defer values.release();

    const indices = try gatherTakeIndices(ctx.tempAllocator(), args[1]);
    defer ctx.tempAllocator().free(indices);

    return gatherArrayByNullableIndices(ctx, values, indices);
}

pub fn arrayTakeKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    return takeKernel(ctx, args, options);
}

pub fn fillNullKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    if (args.len != 2) return error.InvalidArity;
    if (!common.binaryFillNullSupported(args)) return error.InvalidInput;
    return conditionals.coalesceKernel(ctx, args, options);
}

pub fn fillNullForwardKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    if (args.len != 1) return error.InvalidArity;
    if (!common.onlyNoOptions(options)) return error.InvalidOptions;
    if (!common.unarySupportedFilter(args)) return error.InvalidInput;

    var values = try normalizeToArray(ctx, args[0]);
    defer values.release();

    const indices = try computeForwardFillIndices(ctx.tempAllocator(), values);
    defer ctx.tempAllocator().free(indices);

    return gatherArrayByNullableIndices(ctx, values, indices);
}

pub fn fillNullBackwardKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    if (args.len != 1) return error.InvalidArity;
    if (!common.onlyNoOptions(options)) return error.InvalidOptions;
    if (!common.unarySupportedFilter(args)) return error.InvalidInput;

    var values = try normalizeToArray(ctx, args[0]);
    defer values.release();

    const indices = try computeBackwardFillIndices(ctx.tempAllocator(), values);
    defer ctx.tempAllocator().free(indices);

    return gatherArrayByNullableIndices(ctx, values, indices);
}

pub fn indicesNonZeroKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    if (args.len != 1) return error.InvalidArity;
    if (!common.onlyNoOptions(options)) return error.InvalidOptions;
    if (!common.unaryIndicesNonZeroSupported(args)) return error.InvalidInput;

    const input_len = datumLen(args[0]);
    var builder = try zcore.Int64Builder.init(ctx.tempAllocator(), input_len);
    defer builder.deinit();

    const value_type = args[0].dataType();
    var iter = compute.UnaryExecChunkIterator.init(args[0]);
    var logical_index: usize = 0;
    while (try iter.next()) |chunk_value| {
        var chunk = chunk_value;
        defer chunk.deinit();

        var i: usize = 0;
        while (i < chunk.len) : (i += 1) {
            if (chunk.unaryNullAt(i)) {
                logical_index += 1;
                continue;
            }

            const non_zero = switch (value_type) {
                .bool => try common.readBool(chunk.values, i),
                .int32, .int64 => (try common.readI64(chunk.values, i)) != 0,
                else => return error.UnsupportedType,
            };
            if (non_zero) {
                builder.append(@intCast(logical_index)) catch |err| return common.kernelAppendError(err);
            }
            logical_index += 1;
        }
    }

    const out = builder.finish() catch |err| return common.kernelAppendError(err);
    return compute.Datum.fromArray(out);
}
