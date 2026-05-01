pub const std = @import("std");
pub const zcore = @import("zarrow-core");
const impl = @import("impl.zig");

pub const compute = impl.compute;
pub const registerBaseKernels = impl.registerBaseKernels;
pub const registerCompatKernels = impl.registerCompatKernels;

pub const DT_BOOL = zcore.DataType{ .bool = {} };
pub const DT_INT32 = zcore.DataType{ .int32 = {} };
pub const DT_INT64 = zcore.DataType{ .int64 = {} };
pub const FIELD_LIST_ITEM_I32 = zcore.Field{
    .name = "item",
    .data_type = &DT_INT32,
    .nullable = true,
};
pub const DT_LIST_I32 = zcore.DataType{ .list = .{ .value_field = FIELD_LIST_ITEM_I32 } };
pub const STRUCT_FIELDS_I64_BOOL = [_]zcore.Field{
    .{ .name = "i64", .data_type = &DT_INT64, .nullable = true },
    .{ .name = "b", .data_type = &DT_BOOL, .nullable = true },
};
pub const STRUCT_FIELDS_BOOL2 = [_]zcore.Field{
    .{ .name = "c0", .data_type = &DT_BOOL, .nullable = true },
    .{ .name = "c1", .data_type = &DT_BOOL, .nullable = true },
};
pub const STRUCT_FIELDS_LIST_I32 = [_]zcore.Field{
    .{ .name = "items", .data_type = &DT_LIST_I32, .nullable = true },
};

pub fn makeInt64Array(allocator: std.mem.Allocator, values: []const ?i64) !zcore.ArrayRef {
    var builder = try zcore.Int64Builder.init(allocator, values.len);
    defer builder.deinit();
    for (values) |v| {
        if (v) |x| {
            try builder.append(x);
        } else {
            try builder.appendNull();
        }
    }
    return builder.finish();
}

pub fn makeInt32Array(allocator: std.mem.Allocator, values: []const ?i32) !zcore.ArrayRef {
    var builder = try zcore.Int32Builder.init(allocator, values.len);
    defer builder.deinit();
    for (values) |v| {
        if (v) |x| {
            try builder.append(x);
        } else {
            try builder.appendNull();
        }
    }
    return builder.finish();
}

pub fn makeFloat64Array(allocator: std.mem.Allocator, values: []const ?f64) !zcore.ArrayRef {
    var builder = try zcore.Float64Builder.init(allocator, values.len);
    defer builder.deinit();
    for (values) |v| {
        if (v) |x| {
            try builder.append(x);
        } else {
            try builder.appendNull();
        }
    }
    return builder.finish();
}

pub fn makeBoolArray(allocator: std.mem.Allocator, values: []const ?bool) !zcore.ArrayRef {
    var builder = try zcore.BooleanBuilder.init(allocator, values.len);
    defer builder.deinit();
    for (values) |v| {
        if (v) |x| {
            try builder.append(x);
        } else {
            try builder.appendNull();
        }
    }
    return builder.finish();
}

pub fn makeNullArray(allocator: std.mem.Allocator, len: usize) !zcore.ArrayRef {
    var builder = try zcore.NullBuilder.init(allocator, len);
    defer builder.deinit();
    try builder.appendNulls(len);
    return builder.finish();
}

pub fn makeStructI64BoolArray(
    allocator: std.mem.Allocator,
    present: []const bool,
    ints: []const ?i64,
    bools: []const ?bool,
) !zcore.ArrayRef {
    if (present.len != ints.len or present.len != bools.len) return error.InvalidInput;
    var int_child = try makeInt64Array(allocator, ints);
    defer int_child.release();
    var bool_child = try makeBoolArray(allocator, bools);
    defer bool_child.release();

    var builder = zcore.StructBuilder.init(allocator, STRUCT_FIELDS_I64_BOOL[0..]);
    defer builder.deinit();
    for (present) |is_present| {
        if (is_present) {
            try builder.appendValid();
        } else {
            try builder.appendNull();
        }
    }
    return builder.finish(&[_]zcore.ArrayRef{ int_child, bool_child });
}

pub fn makeStructListI32Array(
    allocator: std.mem.Allocator,
    field_child: zcore.ArrayRef,
    present: []const bool,
) !zcore.ArrayRef {
    if (present.len != field_child.data().length) return error.InvalidInput;
    var builder = zcore.StructBuilder.init(allocator, STRUCT_FIELDS_LIST_I32[0..]);
    defer builder.deinit();
    for (present) |is_present| {
        if (is_present) {
            try builder.appendValid();
        } else {
            try builder.appendNull();
        }
    }
    return builder.finish(&[_]zcore.ArrayRef{field_child});
}

pub fn makeStructBool2Array(
    allocator: std.mem.Allocator,
    cond0: zcore.ArrayRef,
    cond1: zcore.ArrayRef,
) !zcore.ArrayRef {
    if (!cond0.data().data_type.eql(.{ .bool = {} }) or !cond1.data().data_type.eql(.{ .bool = {} })) {
        return error.InvalidInput;
    }
    if (cond0.data().length != cond1.data().length) return error.InvalidInput;

    var builder = zcore.StructBuilder.init(allocator, STRUCT_FIELDS_BOOL2[0..]);
    defer builder.deinit();
    var row: usize = 0;
    while (row < cond0.data().length) : (row += 1) {
        try builder.appendValid();
    }

    return builder.finish(&[_]zcore.ArrayRef{ cond0, cond1 });
}

pub fn makeStringArray(allocator: std.mem.Allocator, values: []const ?[]const u8) !zcore.ArrayRef {
    var data_capacity: usize = 0;
    for (values) |v| {
        if (v) |x| data_capacity += x.len;
    }
    var builder = try zcore.StringBuilder.init(allocator, values.len, data_capacity);
    defer builder.deinit();
    for (values) |v| {
        if (v) |x| {
            try builder.append(x);
        } else {
            try builder.appendNull();
        }
    }
    return builder.finish();
}

pub fn makeLargeStringArray(allocator: std.mem.Allocator, values: []const ?[]const u8) !zcore.ArrayRef {
    var data_capacity: usize = 0;
    for (values) |v| {
        if (v) |x| data_capacity += x.len;
    }
    var builder = try zcore.LargeStringBuilder.init(allocator, values.len, data_capacity);
    defer builder.deinit();
    for (values) |v| {
        if (v) |x| {
            try builder.append(x);
        } else {
            try builder.appendNull();
        }
    }
    return builder.finish();
}

pub fn makeBinaryArray(allocator: std.mem.Allocator, values: []const ?[]const u8) !zcore.ArrayRef {
    var data_capacity: usize = 0;
    for (values) |v| {
        if (v) |x| data_capacity += x.len;
    }
    var builder = try zcore.BinaryBuilder.init(allocator, values.len, data_capacity);
    defer builder.deinit();
    for (values) |v| {
        if (v) |x| {
            try builder.append(x);
        } else {
            try builder.appendNull();
        }
    }
    return builder.finish();
}

pub fn makeStringViewArray(allocator: std.mem.Allocator, values: []const ?[]const u8) !zcore.ArrayRef {
    var data_capacity: usize = 0;
    for (values) |v| {
        if (v) |x| data_capacity += x.len;
    }
    var builder = try zcore.StringViewBuilder.init(allocator, values.len, data_capacity);
    defer builder.deinit();
    for (values) |v| {
        if (v) |x| {
            try builder.append(x);
        } else {
            try builder.appendNull();
        }
    }
    return builder.finish();
}

pub fn makeBinaryViewArray(allocator: std.mem.Allocator, values: []const ?[]const u8) !zcore.ArrayRef {
    var data_capacity: usize = 0;
    for (values) |v| {
        if (v) |x| data_capacity += x.len;
    }
    var builder = try zcore.BinaryViewBuilder.init(allocator, values.len, data_capacity);
    defer builder.deinit();
    for (values) |v| {
        if (v) |x| {
            try builder.append(x);
        } else {
            try builder.appendNull();
        }
    }
    return builder.finish();
}

pub fn makeFixedSizeBinaryArray(allocator: std.mem.Allocator, byte_width: usize, values: []const ?[]const u8) !zcore.ArrayRef {
    var builder = try zcore.FixedSizeBinaryBuilder.init(allocator, byte_width, values.len);
    defer builder.deinit();
    for (values) |v| {
        if (v) |x| {
            try builder.append(x);
        } else {
            try builder.appendNull();
        }
    }
    return builder.finish();
}

pub fn makeListInt32Array(allocator: std.mem.Allocator) !zcore.ArrayRef {
    var values_builder = try zcore.Int32Builder.init(allocator, 3);
    defer values_builder.deinit();
    try values_builder.append(1);
    try values_builder.append(2);
    try values_builder.append(3);
    var values = try values_builder.finish();
    defer values.release();

    var list_builder = try zcore.ListBuilder.init(allocator, 2, FIELD_LIST_ITEM_I32);
    defer list_builder.deinit();
    try list_builder.appendLen(2);
    try list_builder.appendLen(1);
    return list_builder.finish(values);
}

pub fn makeListInt32ArrayWithLens(
    allocator: std.mem.Allocator,
    lengths: []const ?usize,
    values: []const i32,
) !zcore.ArrayRef {
    var expected_values_len: usize = 0;
    for (lengths) |maybe_len| {
        if (maybe_len) |len| expected_values_len += len;
    }
    if (expected_values_len != values.len) return error.InvalidInput;

    var values_builder = try zcore.Int32Builder.init(allocator, values.len);
    defer values_builder.deinit();
    for (values) |value| {
        try values_builder.append(value);
    }
    var value_ref = try values_builder.finish();
    defer value_ref.release();

    var list_builder = try zcore.ListBuilder.init(allocator, lengths.len, FIELD_LIST_ITEM_I32);
    defer list_builder.deinit();
    for (lengths) |maybe_len| {
        if (maybe_len) |len| {
            try list_builder.appendLen(len);
        } else {
            try list_builder.appendNull();
        }
    }

    return list_builder.finish(value_ref);
}

pub fn makeLargeListInt32ArrayWithLens(
    allocator: std.mem.Allocator,
    lengths: []const ?usize,
    values: []const i32,
) !zcore.ArrayRef {
    var expected_values_len: usize = 0;
    for (lengths) |maybe_len| {
        if (maybe_len) |len| expected_values_len += len;
    }
    if (expected_values_len != values.len) return error.InvalidInput;

    var values_builder = try zcore.Int32Builder.init(allocator, values.len);
    defer values_builder.deinit();
    for (values) |value| {
        try values_builder.append(value);
    }
    var value_ref = try values_builder.finish();
    defer value_ref.release();

    var list_builder = try zcore.LargeListBuilder.init(allocator, lengths.len, FIELD_LIST_ITEM_I32);
    defer list_builder.deinit();
    for (lengths) |maybe_len| {
        if (maybe_len) |len| {
            try list_builder.appendLen(len);
        } else {
            try list_builder.appendNull();
        }
    }

    return list_builder.finish(value_ref);
}

pub fn makeFixedSizeListInt32Array(
    allocator: std.mem.Allocator,
    list_size: usize,
    present: []const bool,
    values: []const i32,
) !zcore.ArrayRef {
    const expected_values_len = std.math.mul(usize, present.len, list_size) catch return error.InvalidInput;
    if (values.len != expected_values_len) return error.InvalidInput;

    var values_builder = try zcore.Int32Builder.init(allocator, values.len);
    defer values_builder.deinit();
    for (values) |value| {
        try values_builder.append(value);
    }
    var value_ref = try values_builder.finish();
    defer value_ref.release();

    var builder = try zcore.FixedSizeListBuilder.init(allocator, FIELD_LIST_ITEM_I32, list_size);
    defer builder.deinit();
    for (present) |is_present| {
        if (is_present) {
            try builder.appendValid();
        } else {
            try builder.appendNull();
        }
    }

    return builder.finish(value_ref);
}

pub fn makeNestedScalarDatum(payload: zcore.ArrayRef) !compute.Datum {
    return compute.Datum.fromScalar(try compute.Scalar.initNested(payload.data().data_type, payload));
}

pub fn expectInt64ArrayValues(datum: compute.Datum, expected: []const ?i64) !void {
    try std.testing.expect(datum.isArray());
    const view = zcore.Int64Array{ .data = datum.array.data() };
    try std.testing.expectEqual(expected.len, view.len());
    var i: usize = 0;
    while (i < expected.len) : (i += 1) {
        if (expected[i]) |v| {
            try std.testing.expect(!view.isNull(i));
            try std.testing.expectEqual(v, view.value(i));
        } else {
            try std.testing.expect(view.isNull(i));
        }
    }
}

pub fn expectInt32ArrayValues(datum: compute.Datum, expected: []const ?i32) !void {
    try std.testing.expect(datum.isArray());
    const view = zcore.Int32Array{ .data = datum.array.data() };
    try std.testing.expectEqual(expected.len, view.len());
    var i: usize = 0;
    while (i < expected.len) : (i += 1) {
        if (expected[i]) |v| {
            try std.testing.expect(!view.isNull(i));
            try std.testing.expectEqual(v, view.value(i));
        } else {
            try std.testing.expect(view.isNull(i));
        }
    }
}

pub fn expectFloat64ArrayValues(datum: compute.Datum, expected: []const ?f64) !void {
    try std.testing.expect(datum.isArray());
    const view = zcore.Float64Array{ .data = datum.array.data() };
    try std.testing.expectEqual(expected.len, view.len());
    var i: usize = 0;
    while (i < expected.len) : (i += 1) {
        if (expected[i]) |v| {
            try std.testing.expect(!view.isNull(i));
            try std.testing.expectEqual(v, view.value(i));
        } else {
            try std.testing.expect(view.isNull(i));
        }
    }
}

pub fn expectBoolArrayValues(datum: compute.Datum, expected: []const ?bool) !void {
    try std.testing.expect(datum.isArray());
    const view = zcore.BooleanArray{ .data = datum.array.data() };
    try std.testing.expectEqual(expected.len, view.len());
    var i: usize = 0;
    while (i < expected.len) : (i += 1) {
        if (expected[i]) |v| {
            try std.testing.expect(!view.isNull(i));
            try std.testing.expectEqual(v, view.value(i));
        } else {
            try std.testing.expect(view.isNull(i));
        }
    }
}

pub fn expectFixedSizeListAllNullAligned(out: compute.Datum, expected_len: usize, expected_list_size: usize) !void {
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType() == .fixed_size_list);
    try out.array.data().validateLayout();

    const out_list = zcore.FixedSizeListArray{ .data = out.array.data() };
    try std.testing.expectEqual(expected_len, out_list.len());
    try std.testing.expectEqual(expected_list_size, out_list.listSize());
    var row: usize = 0;
    while (row < expected_len) : (row += 1) {
        try std.testing.expect(out_list.isNull(row));
    }
    const expected_values_len = std.math.mul(usize, expected_len, expected_list_size) catch return error.Overflow;
    try std.testing.expectEqual(expected_values_len, out_list.valuesRef().data().length);
}
