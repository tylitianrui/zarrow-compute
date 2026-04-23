const std = @import("std");
const zcore = @import("zarrow-core");

pub const compute = zcore.compute;

pub fn onlyNoOptions(options: compute.Options) bool {
    return switch (options) {
        .none => true,
        else => false,
    };
}

pub fn onlyArithmeticOptions(options: compute.Options) bool {
    return switch (options) {
        .arithmetic => true,
        else => false,
    };
}

pub fn onlyCastOptions(options: compute.Options) bool {
    return switch (options) {
        .cast => true,
        else => false,
    };
}

pub fn onlyFilterOptions(options: compute.Options) bool {
    return switch (options) {
        .filter => true,
        else => false,
    };
}

pub fn unaryInt64(args: []const compute.Datum) bool {
    return args.len == 1 and args[0].dataType().eql(.{ .int64 = {} });
}

pub fn binaryInt64(args: []const compute.Datum) bool {
    return args.len == 2 and
        args[0].dataType().eql(.{ .int64 = {} }) and
        args[1].dataType().eql(.{ .int64 = {} });
}

pub fn binaryInt64Bool(args: []const compute.Datum) bool {
    return args.len == 2 and
        args[0].dataType().eql(.{ .int64 = {} }) and
        args[1].dataType().eql(.{ .bool = {} });
}

pub fn isFilterFixedWidthType(data_type: compute.DataType) bool {
    return switch (data_type) {
        .uint8,
        .int8,
        .uint16,
        .int16,
        .uint32,
        .int32,
        .uint64,
        .int64,
        .half_float,
        .float,
        .double,
        .date32,
        .date64,
        .timestamp,
        .time32,
        .time64,
        .duration,
        .interval_months,
        .interval_day_time,
        .interval_month_day_nano,
        .decimal32,
        .decimal64,
        .decimal128,
        .decimal256,
        .fixed_size_binary,
        => true,
        else => false,
    };
}

pub fn isFilterSupportedType(data_type: compute.DataType) bool {
    return switch (data_type) {
        .null, .bool, .string, .large_string, .string_view, .binary, .large_binary, .binary_view => true,
        else => isFilterFixedWidthType(data_type),
    };
}

pub fn isIfElseSupportedType(data_type: compute.DataType) bool {
    return switch (data_type) {
        .string, .large_string, .string_view, .binary, .large_binary, .binary_view => true,
        else => isFilterFixedWidthType(data_type),
    };
}

pub fn binarySupportedFilter(args: []const compute.Datum) bool {
    return args.len == 2 and
        isFilterSupportedType(args[0].dataType()) and
        args[1].dataType().eql(.{ .bool = {} });
}

pub fn unaryArrayLike(args: []const compute.Datum) bool {
    return args.len == 1 and (args[0].isArray() or args[0].isChunked());
}

pub fn unarySupportedFilter(args: []const compute.Datum) bool {
    return args.len == 1 and isFilterSupportedType(args[0].dataType()) and (args[0].isArray() or args[0].isChunked());
}

pub fn ternaryBoolIfElseSupported(args: []const compute.Datum) bool {
    return args.len == 3 and
        args[0].dataType().eql(.{ .bool = {} }) and
        args[1].dataType().eql(args[2].dataType()) and
        isIfElseSupportedType(args[1].dataType());
}

pub fn resultI64(args: []const compute.Datum, options: compute.Options) compute.KernelError!compute.DataType {
    _ = options;
    if (args.len == 0) return error.InvalidArity;
    return .{ .int64 = {} };
}

pub fn resultBool(args: []const compute.Datum, options: compute.Options) compute.KernelError!compute.DataType {
    _ = options;
    if (args.len == 0) return error.InvalidArity;
    return .{ .bool = {} };
}

pub fn resultI32(args: []const compute.Datum, options: compute.Options) compute.KernelError!compute.DataType {
    if (args.len != 1) return error.InvalidArity;
    return switch (options) {
        .cast => |cast_opts| blk: {
            if (cast_opts.to_type) |to_type| {
                if (!to_type.eql(.{ .int32 = {} })) return error.InvalidCast;
            }
            break :blk .{ .int32 = {} };
        },
        else => error.InvalidOptions,
    };
}

pub fn resultSameAsFirst(args: []const compute.Datum, options: compute.Options) compute.KernelError!compute.DataType {
    _ = options;
    if (args.len == 0) return error.InvalidArity;
    return args[0].dataType();
}

pub fn resultSameAsSecond(args: []const compute.Datum, options: compute.Options) compute.KernelError!compute.DataType {
    _ = options;
    if (args.len < 2) return error.InvalidArity;
    return args[1].dataType();
}

pub fn kernelAppendError(err: anyerror) compute.KernelError {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidInput,
    };
}

pub fn bitByteLength(len: usize) usize {
    return (len + 7) / 8;
}

pub fn setBit(data: []u8, bit_index: usize) void {
    const byte_index = bit_index / 8;
    const bit_shift: u3 = @intCast(bit_index & 7);
    data[byte_index] |= @as(u8, 1) << bit_shift;
}

pub fn readI64(value: compute.ExecChunkValue, logical_index: usize) compute.KernelError!i64 {
    return switch (value) {
        .scalar => |s| switch (s.value) {
            .i64 => |v| v,
            else => error.InvalidInput,
        },
        .array => |arr| blk: {
            const dt = arr.data().data_type;
            if (dt.eql(.{ .int64 = {} })) {
                const view = zcore.Int64Array{ .data = arr.data() };
                break :blk view.value(logical_index) catch return error.InvalidInput;
            }
            if (dt.eql(.{ .int32 = {} })) {
                const view = zcore.Int32Array{ .data = arr.data() };
                const v = view.value(logical_index) catch return error.InvalidInput;
                break :blk @as(i64, v);
            }
            break :blk error.UnsupportedType;
        },
    };
}

pub fn readBool(value: compute.ExecChunkValue, logical_index: usize) compute.KernelError!bool {
    return switch (value) {
        .scalar => |s| switch (s.value) {
            .bool => |v| v,
            else => error.InvalidInput,
        },
        .array => |arr| blk: {
            const dt = arr.data().data_type;
            if (!dt.eql(.{ .bool = {} })) break :blk error.UnsupportedType;
            const view = zcore.BooleanArray{ .data = arr.data() };
            break :blk view.value(logical_index);
        },
    };
}

pub const StringValueKind = enum {
    string,
    large_string,
    string_view,
};

pub const BinaryValueKind = enum {
    binary,
    large_binary,
    binary_view,
};

pub fn readString(value: compute.ExecChunkValue, logical_index: usize, kind: StringValueKind) compute.KernelError![]const u8 {
    return switch (value) {
        .scalar => |s| switch (s.value) {
            .string => |v| v,
            else => error.InvalidInput,
        },
        .array => |arr| blk: {
            const dt = arr.data().data_type;
            break :blk switch (kind) {
                .string => blk2: {
                    if (!dt.eql(.{ .string = {} })) break :blk2 error.UnsupportedType;
                    const view = zcore.StringArray{ .data = arr.data() };
                    break :blk2 view.value(logical_index);
                },
                .large_string => blk2: {
                    if (!dt.eql(.{ .large_string = {} })) break :blk2 error.UnsupportedType;
                    const view = zcore.LargeStringArray{ .data = arr.data() };
                    break :blk2 view.value(logical_index);
                },
                .string_view => blk2: {
                    if (!dt.eql(.{ .string_view = {} })) break :blk2 error.UnsupportedType;
                    const view = zcore.StringViewArray{ .data = arr.data() };
                    break :blk2 view.value(logical_index);
                },
            };
        },
    };
}

pub fn readBinary(value: compute.ExecChunkValue, logical_index: usize, kind: BinaryValueKind) compute.KernelError![]const u8 {
    return switch (value) {
        .scalar => |s| switch (s.value) {
            .binary => |v| v,
            else => error.InvalidInput,
        },
        .array => |arr| blk: {
            const dt = arr.data().data_type;
            break :blk switch (kind) {
                .binary => blk2: {
                    if (!dt.eql(.{ .binary = {} })) break :blk2 error.UnsupportedType;
                    const view = zcore.BinaryArray{ .data = arr.data() };
                    break :blk2 view.value(logical_index);
                },
                .large_binary => blk2: {
                    if (!dt.eql(.{ .large_binary = {} })) break :blk2 error.UnsupportedType;
                    const view = zcore.LargeBinaryArray{ .data = arr.data() };
                    break :blk2 view.value(logical_index);
                },
                .binary_view => blk2: {
                    if (!dt.eql(.{ .binary_view = {} })) break :blk2 error.UnsupportedType;
                    const view = zcore.BinaryViewArray{ .data = arr.data() };
                    break :blk2 view.value(logical_index);
                },
            };
        },
    };
}

fn copyScalarBytes(comptime T: type, value: T, scratch: *[32]u8) []const u8 {
    var local = value;
    const bytes = std.mem.asBytes(&local);
    @memcpy(scratch[0..bytes.len], bytes);
    return scratch[0..bytes.len];
}

fn scalarFixedWidthBytes(
    scalar: compute.Scalar,
    data_type: compute.DataType,
    scratch: *[32]u8,
) compute.KernelError![]const u8 {
    if (!scalar.data_type.eql(data_type)) return error.InvalidInput;
    return switch (data_type) {
        .uint8 => switch (scalar.value) {
            .u8 => |v| copyScalarBytes(u8, v, scratch),
            else => error.InvalidInput,
        },
        .int8 => switch (scalar.value) {
            .i8 => |v| copyScalarBytes(i8, v, scratch),
            else => error.InvalidInput,
        },
        .uint16 => switch (scalar.value) {
            .u16 => |v| copyScalarBytes(u16, v, scratch),
            else => error.InvalidInput,
        },
        .int16 => switch (scalar.value) {
            .i16 => |v| copyScalarBytes(i16, v, scratch),
            else => error.InvalidInput,
        },
        .uint32 => switch (scalar.value) {
            .u32 => |v| copyScalarBytes(u32, v, scratch),
            else => error.InvalidInput,
        },
        .int32 => switch (scalar.value) {
            .i32 => |v| copyScalarBytes(i32, v, scratch),
            else => error.InvalidInput,
        },
        .uint64 => switch (scalar.value) {
            .u64 => |v| copyScalarBytes(u64, v, scratch),
            else => error.InvalidInput,
        },
        .int64 => switch (scalar.value) {
            .i64 => |v| copyScalarBytes(i64, v, scratch),
            else => error.InvalidInput,
        },
        .half_float => switch (scalar.value) {
            .f16 => |v| copyScalarBytes(f16, v, scratch),
            else => error.InvalidInput,
        },
        .float => switch (scalar.value) {
            .f32 => |v| copyScalarBytes(f32, v, scratch),
            else => error.InvalidInput,
        },
        .double => switch (scalar.value) {
            .f64 => |v| copyScalarBytes(f64, v, scratch),
            else => error.InvalidInput,
        },
        .date32 => switch (scalar.value) {
            .date32 => |v| copyScalarBytes(i32, v, scratch),
            else => error.InvalidInput,
        },
        .date64 => switch (scalar.value) {
            .date64 => |v| copyScalarBytes(i64, v, scratch),
            else => error.InvalidInput,
        },
        .time32 => switch (scalar.value) {
            .time32 => |v| copyScalarBytes(i32, v, scratch),
            else => error.InvalidInput,
        },
        .time64 => switch (scalar.value) {
            .time64 => |v| copyScalarBytes(i64, v, scratch),
            else => error.InvalidInput,
        },
        .timestamp => switch (scalar.value) {
            .timestamp => |v| copyScalarBytes(i64, v, scratch),
            else => error.InvalidInput,
        },
        .duration => switch (scalar.value) {
            .duration => |v| copyScalarBytes(i64, v, scratch),
            else => error.InvalidInput,
        },
        .interval_months => switch (scalar.value) {
            .interval_months => |v| copyScalarBytes(i32, v, scratch),
            else => error.InvalidInput,
        },
        .interval_day_time => switch (scalar.value) {
            .interval_day_time => |v| copyScalarBytes(i64, v, scratch),
            else => error.InvalidInput,
        },
        .interval_month_day_nano => switch (scalar.value) {
            .interval_month_day_nano => |v| copyScalarBytes(i128, v, scratch),
            else => error.InvalidInput,
        },
        .decimal32 => switch (scalar.value) {
            .decimal32 => |v| copyScalarBytes(i32, v, scratch),
            else => error.InvalidInput,
        },
        .decimal64 => switch (scalar.value) {
            .decimal64 => |v| copyScalarBytes(i64, v, scratch),
            else => error.InvalidInput,
        },
        .decimal128 => switch (scalar.value) {
            .decimal128 => |v| copyScalarBytes(i128, v, scratch),
            else => error.InvalidInput,
        },
        .decimal256 => switch (scalar.value) {
            .decimal256 => |v| copyScalarBytes(i256, v, scratch),
            else => error.InvalidInput,
        },
        .fixed_size_binary => |fsb| switch (scalar.value) {
            .binary => |v| blk: {
                if (fsb.byte_width < 0) break :blk error.UnsupportedType;
                const width: usize = @intCast(fsb.byte_width);
                if (v.len != width) break :blk error.InvalidInput;
                break :blk v;
            },
            else => error.InvalidInput,
        },
        else => error.UnsupportedType,
    };
}

pub fn readFixedWidthBytes(
    value: compute.ExecChunkValue,
    logical_index: usize,
    data_type: compute.DataType,
    byte_width: usize,
    scratch: *[32]u8,
) compute.KernelError![]const u8 {
    return switch (value) {
        .array => |arr| blk: {
            const arr_data = arr.data();
            if (!arr_data.data_type.eql(data_type)) break :blk error.UnsupportedType;
            if (arr_data.buffers.len < 2) break :blk error.InvalidInput;
            const offset = std.math.add(usize, arr_data.offset, logical_index) catch return error.InvalidInput;
            const start = std.math.mul(usize, offset, byte_width) catch return error.InvalidInput;
            const end = std.math.add(usize, start, byte_width) catch return error.InvalidInput;
            if (end > arr_data.buffers[1].data.len) break :blk error.InvalidInput;
            break :blk arr_data.buffers[1].data[start..end];
        },
        .scalar => |s| try scalarFixedWidthBytes(s, data_type, scratch),
    };
}
