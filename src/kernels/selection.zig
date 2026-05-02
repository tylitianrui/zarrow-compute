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

const SortIndicesI32Context = struct {
    values: []const i32,
    is_null: []const bool,
    sort_opts: compute.SortOptions,
};

fn lessThanSortIndicesI32(ctx: SortIndicesI32Context, lhs: usize, rhs: usize) bool {
    const lhs_null = ctx.is_null[lhs];
    const rhs_null = ctx.is_null[rhs];
    if (lhs_null != rhs_null) {
        const nulls_at_start = ctx.sort_opts.null_placement == .at_start;
        return if (nulls_at_start) lhs_null else !lhs_null;
    }
    if (lhs_null) return tieBreakByIndex(ctx.sort_opts, lhs, rhs);

    const lhs_v = ctx.values[lhs];
    const rhs_v = ctx.values[rhs];
    switch (ctx.sort_opts.order) {
        .ascending => {
            if (lhs_v < rhs_v) return true;
            if (lhs_v > rhs_v) return false;
        },
        .descending => {
            if (lhs_v > rhs_v) return true;
            if (lhs_v < rhs_v) return false;
        },
    }
    return tieBreakByIndex(ctx.sort_opts, lhs, rhs);
}

const SortIndicesI64Context = struct {
    values: []const i64,
    is_null: []const bool,
    sort_opts: compute.SortOptions,
};

fn lessThanSortIndicesI64(ctx: SortIndicesI64Context, lhs: usize, rhs: usize) bool {
    const lhs_null = ctx.is_null[lhs];
    const rhs_null = ctx.is_null[rhs];
    if (lhs_null != rhs_null) {
        const nulls_at_start = ctx.sort_opts.null_placement == .at_start;
        return if (nulls_at_start) lhs_null else !lhs_null;
    }
    if (lhs_null) return tieBreakByIndex(ctx.sort_opts, lhs, rhs);

    const lhs_v = ctx.values[lhs];
    const rhs_v = ctx.values[rhs];
    switch (ctx.sort_opts.order) {
        .ascending => {
            if (lhs_v < rhs_v) return true;
            if (lhs_v > rhs_v) return false;
        },
        .descending => {
            if (lhs_v > rhs_v) return true;
            if (lhs_v < rhs_v) return false;
        },
    }
    return tieBreakByIndex(ctx.sort_opts, lhs, rhs);
}

const SortIndicesF64Context = struct {
    values: []const f64,
    is_null: []const bool,
    sort_opts: compute.SortOptions,
};

fn lessThanSortIndicesF64(ctx: SortIndicesF64Context, lhs: usize, rhs: usize) bool {
    const lhs_null = ctx.is_null[lhs];
    const rhs_null = ctx.is_null[rhs];
    if (lhs_null != rhs_null) {
        const nulls_at_start = ctx.sort_opts.null_placement == .at_start;
        return if (nulls_at_start) lhs_null else !lhs_null;
    }
    if (lhs_null) return tieBreakByIndex(ctx.sort_opts, lhs, rhs);

    const lhs_v = ctx.values[lhs];
    const rhs_v = ctx.values[rhs];
    const lhs_nan = std.math.isNan(lhs_v);
    const rhs_nan = std.math.isNan(rhs_v);
    if (lhs_nan != rhs_nan) {
        const nans_at_start = nanAtStart(ctx.sort_opts);
        return if (nans_at_start) lhs_nan else !lhs_nan;
    }
    if (lhs_nan) return tieBreakByIndex(ctx.sort_opts, lhs, rhs);

    switch (ctx.sort_opts.order) {
        .ascending => {
            if (lhs_v < rhs_v) return true;
            if (lhs_v > rhs_v) return false;
        },
        .descending => {
            if (lhs_v > rhs_v) return true;
            if (lhs_v < rhs_v) return false;
        },
    }
    return tieBreakByIndex(ctx.sort_opts, lhs, rhs);
}

fn tieBreakByIndex(sort_opts: compute.SortOptions, lhs: usize, rhs: usize) bool {
    return sort_opts.stable and lhs < rhs;
}

fn nanAtStart(sort_opts: compute.SortOptions) bool {
    if (sort_opts.nan_placement) |placement| {
        return placement == .at_start;
    }
    return sort_opts.order == .descending;
}

fn sortIndicesForArray(
    ctx: *compute.ExecContext,
    values: zcore.ArrayRef,
    sort_opts: compute.SortOptions,
) compute.KernelError!compute.Datum {
    const len = values.data().length;
    const order = ctx.tempAllocator().alloc(usize, len) catch return error.OutOfMemory;
    defer ctx.tempAllocator().free(order);
    const is_null = ctx.tempAllocator().alloc(bool, len) catch return error.OutOfMemory;
    defer ctx.tempAllocator().free(is_null);

    var i: usize = 0;
    while (i < len) : (i += 1) {
        order[i] = i;
        is_null[i] = values.data().isNull(i);
    }

    switch (values.data().data_type) {
        .int32 => {
            const view = zcore.Int32Array{ .data = values.data() };
            const typed_values = view.values() catch return error.InvalidInput;
            std.mem.sort(usize, order, SortIndicesI32Context{
                .values = typed_values,
                .is_null = is_null,
                .sort_opts = sort_opts,
            }, lessThanSortIndicesI32);
        },
        .int64 => {
            const view = zcore.Int64Array{ .data = values.data() };
            const typed_values = view.values() catch return error.InvalidInput;
            std.mem.sort(usize, order, SortIndicesI64Context{
                .values = typed_values,
                .is_null = is_null,
                .sort_opts = sort_opts,
            }, lessThanSortIndicesI64);
        },
        .double => {
            const view = zcore.Float64Array{ .data = values.data() };
            const typed_values = view.values() catch return error.InvalidInput;
            std.mem.sort(usize, order, SortIndicesF64Context{
                .values = typed_values,
                .is_null = is_null,
                .sort_opts = sort_opts,
            }, lessThanSortIndicesF64);
        },
        else => return error.UnsupportedType,
    }

    var builder = try zcore.Int64Builder.init(ctx.tempAllocator(), len);
    defer builder.deinit();
    for (order) |entry| {
        const index_i64 = std.math.cast(i64, entry) orelse return error.InvalidInput;
        builder.append(index_i64) catch |err| return common.kernelAppendError(err);
    }
    const out = builder.finish() catch |err| return common.kernelAppendError(err);
    return compute.Datum.fromArray(out);
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

pub fn sortIndicesKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    if (args.len != 1) return error.InvalidArity;
    const sort_opts = switch (options) {
        .sort => |opts| opts,
        else => return error.InvalidOptions,
    };
    if (!common.unarySortIndicesSupported(args)) return error.InvalidInput;

    var values = try normalizeToArray(ctx, args[0]);
    defer values.release();
    return sortIndicesForArray(ctx, values, sort_opts);
}

pub fn arraySortIndicesKernel(
    ctx: *compute.ExecContext,
    args: []const compute.Datum,
    options: compute.Options,
) compute.KernelError!compute.Datum {
    return sortIndicesKernel(ctx, args, options);
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
