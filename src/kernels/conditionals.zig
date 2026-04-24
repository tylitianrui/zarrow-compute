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

const CaseWhenCompatArgs = struct {
    allocator: std.mem.Allocator,
    args: []compute.Datum,
    config: SelectionConfig,

    fn deinit(self: *CaseWhenCompatArgs) void {
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

fn parseCaseWhenConfig(args: []const compute.Datum) compute.KernelError!SelectionConfig {
    if (args.len < 2) return error.InvalidArity;
    const has_else = (args.len % 2) == 1;
    const case_pair_count = if (has_else) (args.len - 1) / 2 else args.len / 2;
    if (case_pair_count == 0) return error.InvalidArity;
    return .{
        .kind = .case_when,
        .case_pair_count = case_pair_count,
        .has_else = has_else,
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

fn extractCaseWhenStructFieldDatum(
    allocator: std.mem.Allocator,
    struct_conditions: compute.Datum,
    field_index: usize,
    field_type: compute.DataType,
) compute.KernelError!compute.Datum {
    return switch (struct_conditions) {
        .array => |struct_array_ref| blk: {
            const struct_array = zcore.StructArray{ .data = struct_array_ref.data() };
            if (field_index >= struct_array.fieldCount()) return error.InvalidInput;
            const child = struct_array.field(field_index) catch |err| return mapSelectionInputError(err);
            break :blk compute.Datum.fromArray(child);
        },
        .chunked => |struct_chunks| blk: {
            const chunk_count = struct_chunks.numChunks();
            const child_chunks = allocator.alloc(zcore.ArrayRef, chunk_count) catch return error.OutOfMemory;
            var initialized: usize = 0;
            errdefer {
                while (initialized > 0) {
                    initialized -= 1;
                    child_chunks[initialized].release();
                }
                allocator.free(child_chunks);
            }

            var chunk_index: usize = 0;
            while (chunk_index < chunk_count) : (chunk_index += 1) {
                const struct_chunk = struct_chunks.chunk(chunk_index).*;
                const struct_array = zcore.StructArray{ .data = struct_chunk.data() };
                if (field_index >= struct_array.fieldCount()) return error.InvalidInput;
                child_chunks[chunk_index] = struct_array.field(field_index) catch |err| return mapSelectionInputError(err);
                initialized += 1;
            }

            const child_chunked = zcore.ChunkedArray.init(allocator, field_type, child_chunks) catch |err| return mapSelectionInputError(err);
            for (child_chunks) |*child_chunk| {
                child_chunk.release();
            }
            allocator.free(child_chunks);
            break :blk compute.Datum.fromChunked(child_chunked);
        },
        .scalar => error.InvalidInput,
    };
}

fn buildCaseWhenCompatArgsFromStruct(
    allocator: std.mem.Allocator,
    args: []const compute.Datum,
) compute.KernelError!CaseWhenCompatArgs {
    const config = try parseCaseWhenStructConfig(args);
    const cond_struct = args[0].dataType().struct_;

    const compat_len = config.case_pair_count * 2 + (if (config.has_else) @as(usize, 1) else @as(usize, 0));
    const compat_args = allocator.alloc(compute.Datum, compat_len) catch return error.OutOfMemory;
    var initialized: usize = 0;
    errdefer {
        while (initialized > 0) {
            initialized -= 1;
            var datum = compat_args[initialized];
            datum.release();
        }
        allocator.free(compat_args);
    }

    var pair_index: usize = 0;
    while (pair_index < config.case_pair_count) : (pair_index += 1) {
        const cond_arg_index = pair_index * 2;
        compat_args[cond_arg_index] = try extractCaseWhenStructFieldDatum(
            allocator,
            args[0],
            pair_index,
            cond_struct.fields[pair_index].data_type.*,
        );
        initialized += 1;

        compat_args[cond_arg_index + 1] = args[pair_index + 1].retain();
        initialized += 1;
    }

    if (config.has_else) {
        compat_args[compat_args.len - 1] = args[args.len - 1].retain();
        initialized += 1;
    }

    return .{
        .allocator = allocator,
        .args = compat_args,
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
    if (args[0].dataType() == .struct_) {
        var compat = try buildCaseWhenCompatArgsFromStruct(ctx.tempAllocator(), args);
        defer compat.deinit();
        return runSelectionKernel(ctx, compat.args, compat.config, args[1].dataType());
    }
    const config = try parseCaseWhenConfig(args);
    return runSelectionKernel(ctx, args, config, args[1].dataType());
}
