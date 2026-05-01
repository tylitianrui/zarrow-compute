const support = @import("test_support.zig");

const std = support.std;
const zcore = support.zcore;
const compute = support.compute;
const registerBaseKernels = support.registerBaseKernels;
const registerCompatKernels = support.registerCompatKernels;

const makeInt64Array = support.makeInt64Array;
const makeInt32Array = support.makeInt32Array;
const makeFloat64Array = support.makeFloat64Array;
const makeBoolArray = support.makeBoolArray;
const makeNullArray = support.makeNullArray;
const makeStructI64BoolArray = support.makeStructI64BoolArray;
const makeStructListI32Array = support.makeStructListI32Array;
const makeStructBool2Array = support.makeStructBool2Array;
const makeStringArray = support.makeStringArray;
const makeLargeStringArray = support.makeLargeStringArray;
const makeBinaryArray = support.makeBinaryArray;
const makeStringViewArray = support.makeStringViewArray;
const makeBinaryViewArray = support.makeBinaryViewArray;
const makeFixedSizeBinaryArray = support.makeFixedSizeBinaryArray;
const makeListInt32Array = support.makeListInt32Array;
const makeListInt32ArrayWithLens = support.makeListInt32ArrayWithLens;
const makeLargeListInt32ArrayWithLens = support.makeLargeListInt32ArrayWithLens;
const makeFixedSizeListInt32Array = support.makeFixedSizeListInt32Array;
const makeNestedScalarDatum = support.makeNestedScalarDatum;

const expectInt64ArrayValues = support.expectInt64ArrayValues;
const expectInt32ArrayValues = support.expectInt32ArrayValues;
const expectFloat64ArrayValues = support.expectFloat64ArrayValues;
const expectBoolArrayValues = support.expectBoolArrayValues;
const expectFixedSizeListAllNullAligned = support.expectFixedSizeListAllNullAligned;

test "add_i64 supports scalar broadcast and null propagation" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var lhs = try makeInt64Array(allocator, &[_]?i64{ 1, null, 3 });
    defer lhs.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(lhs.retain()),
        compute.Datum.fromScalar(.{
            .data_type = .{ .int64 = {} },
            .value = .{ .i64 = 10 },
        }),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    var out = try ctx.invokeVector("add_i64", args[0..], .{ .arithmetic = .{} });
    defer out.release();
    try std.testing.expect(out.isArray());

    const view = zcore.Int64Array{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), view.len());
    try std.testing.expectEqual(@as(i64, 11), view.value(0));
    try std.testing.expect(view.isNull(1));
    try std.testing.expectEqual(@as(i64, 13), view.value(2));
}

test "arithmetic kernels support int32 and float64 subsets" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var i32_lhs = try makeInt32Array(allocator, &[_]?i32{ 7, null, -3 });
    defer i32_lhs.release();
    var i32_rhs = try makeInt32Array(allocator, &[_]?i32{ 2, 5, 3 });
    defer i32_rhs.release();
    const i32_args = [_]compute.Datum{
        compute.Datum.fromArray(i32_lhs.retain()),
        compute.Datum.fromArray(i32_rhs.retain()),
    };
    defer {
        var d = i32_args[0];
        d.release();
    }
    defer {
        var d = i32_args[1];
        d.release();
    }

    var add_i32_out = try ctx.invokeVector("add_i64", i32_args[0..], .{ .arithmetic = .{} });
    defer add_i32_out.release();
    try expectInt32ArrayValues(add_i32_out, &[_]?i32{ 9, null, 0 });

    var subtract_i32_out = try ctx.invokeVector("subtract_i64", i32_args[0..], .{ .arithmetic = .{} });
    defer subtract_i32_out.release();
    try expectInt32ArrayValues(subtract_i32_out, &[_]?i32{ 5, null, -6 });

    var multiply_i32_out = try ctx.invokeVector("multiply_i64", i32_args[0..], .{ .arithmetic = .{} });
    defer multiply_i32_out.release();
    try expectInt32ArrayValues(multiply_i32_out, &[_]?i32{ 14, null, -9 });

    var divide_i32_out = try ctx.invokeVector(
        "divide_i64",
        i32_args[0..],
        .{ .arithmetic = .{ .divide_by_zero_is_error = false } },
    );
    defer divide_i32_out.release();
    try expectInt32ArrayValues(divide_i32_out, &[_]?i32{ 3, null, -1 });

    var f64_lhs = try makeFloat64Array(allocator, &[_]?f64{ 1.5, null, 8.0 });
    defer f64_lhs.release();
    var f64_rhs = try makeFloat64Array(allocator, &[_]?f64{ 0.5, 2.0, 4.0 });
    defer f64_rhs.release();
    const f64_args = [_]compute.Datum{
        compute.Datum.fromArray(f64_lhs.retain()),
        compute.Datum.fromArray(f64_rhs.retain()),
    };
    defer {
        var d = f64_args[0];
        d.release();
    }
    defer {
        var d = f64_args[1];
        d.release();
    }

    var add_f64_out = try ctx.invokeVector("add_i64", f64_args[0..], .{ .arithmetic = .{} });
    defer add_f64_out.release();
    try expectFloat64ArrayValues(add_f64_out, &[_]?f64{ 2.0, null, 12.0 });

    var divide_f64_out = try ctx.invokeVector("divide_i64", f64_args[0..], .{ .arithmetic = .{} });
    defer divide_f64_out.release();
    try expectFloat64ArrayValues(divide_f64_out, &[_]?f64{ 3.0, null, 2.0 });
}

test "filter keeps selected values, propagates value nulls, and drops null predicates by default" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeInt64Array(allocator, &[_]?i64{ 1, null, 3, 4 });
    defer values.release();
    var predicate = try makeBoolArray(allocator, &[_]?bool{ true, true, null, false });
    defer predicate.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(values.retain()),
        compute.Datum.fromArray(predicate.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    var out = try ctx.invokeVector("filter", args[0..], .{ .filter = .{} });
    defer out.release();
    try std.testing.expect(out.isArray());

    const view = zcore.Int64Array{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 2), view.len());
    try std.testing.expectEqual(@as(i64, 1), view.value(0));
    try std.testing.expect(view.isNull(1));

    var array_filter_out = try ctx.invokeVector("array_filter", args[0..], .{ .filter = .{} });
    defer array_filter_out.release();
    try expectInt64ArrayValues(array_filter_out, &[_]?i64{ 1, null });
}

test "filter emits null for null predicate when drop_nulls is false" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeInt64Array(allocator, &[_]?i64{ 7, 8, 9 });
    defer values.release();
    var predicate = try makeBoolArray(allocator, &[_]?bool{ true, null, true });
    defer predicate.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(values.retain()),
        compute.Datum.fromArray(predicate.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    var out = try ctx.invokeVector("filter_i64", args[0..], .{
        .filter = .{ .drop_nulls = false },
    });
    defer out.release();
    try std.testing.expect(out.isArray());

    const view = zcore.Int64Array{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), view.len());
    try std.testing.expectEqual(@as(i64, 7), view.value(0));
    try std.testing.expect(view.isNull(1));
    try std.testing.expectEqual(@as(i64, 9), view.value(2));
}

test "filter supports int32 value arrays with bool predicate" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeInt32Array(allocator, &[_]?i32{ 10, null, 20, 30 });
    defer values.release();
    var predicate = try makeBoolArray(allocator, &[_]?bool{ true, true, false, true });
    defer predicate.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(values.retain()),
        compute.Datum.fromArray(predicate.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    var out = try ctx.invokeVector("filter", args[0..], .{ .filter = .{} });
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .int32 = {} }));

    const view = zcore.Int32Array{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), view.len());
    try std.testing.expectEqual(@as(i32, 10), try view.value(0));
    try std.testing.expect(view.isNull(1));
    try std.testing.expectEqual(@as(i32, 30), try view.value(2));
}

test "filter supports bool value arrays" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeBoolArray(allocator, &[_]?bool{ true, null, false });
    defer values.release();
    var predicate = try makeBoolArray(allocator, &[_]?bool{ true, true, true });
    defer predicate.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(values.retain()),
        compute.Datum.fromArray(predicate.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    var out = try ctx.invokeVector("filter", args[0..], .{ .filter = .{} });
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .bool = {} }));

    const view = zcore.BooleanArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), view.len());
    try std.testing.expect(view.value(0));
    try std.testing.expect(view.isNull(1));
    try std.testing.expect(!view.value(2));
}

test "filter supports string scalar broadcast and predicate null emission" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var predicate = try makeBoolArray(allocator, &[_]?bool{ true, false, null, true });
    defer predicate.release();

    const args = [_]compute.Datum{
        compute.Datum.fromScalar(.{
            .data_type = .{ .string = {} },
            .value = .{ .string = "x" },
        }),
        compute.Datum.fromArray(predicate.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    var out = try ctx.invokeVector("filter", args[0..], .{
        .filter = .{ .drop_nulls = false },
    });
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .string = {} }));

    const view = zcore.StringArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), view.len());
    try std.testing.expect(std.mem.eql(u8, view.value(0), "x"));
    try std.testing.expect(view.isNull(1));
    try std.testing.expect(std.mem.eql(u8, view.value(2), "x"));
}

test "filter supports binary arrays and predicate null emission" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeBinaryArray(allocator, &[_]?[]const u8{ "aa", null, "bb", "cc" });
    defer values.release();
    var predicate = try makeBoolArray(allocator, &[_]?bool{ true, true, null, false });
    defer predicate.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(values.retain()),
        compute.Datum.fromArray(predicate.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    var out = try ctx.invokeVector("filter", args[0..], .{
        .filter = .{ .drop_nulls = false },
    });
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .binary = {} }));

    const view = zcore.BinaryArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), view.len());
    try std.testing.expect(std.mem.eql(u8, view.value(0), "aa"));
    try std.testing.expect(view.isNull(1));
    try std.testing.expect(view.isNull(2));
}

test "filter supports large_string value arrays" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeLargeStringArray(allocator, &[_]?[]const u8{ "left", null, "right" });
    defer values.release();
    var predicate = try makeBoolArray(allocator, &[_]?bool{ false, true, true });
    defer predicate.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(values.retain()),
        compute.Datum.fromArray(predicate.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    var out = try ctx.invokeVector("filter", args[0..], .{ .filter = .{} });
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .large_string = {} }));

    const view = zcore.LargeStringArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 2), view.len());
    try std.testing.expect(view.isNull(0));
    try std.testing.expect(std.mem.eql(u8, view.value(1), "right"));
}

test "filter supports fixed_size_binary value arrays" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeFixedSizeBinaryArray(allocator, 2, &[_]?[]const u8{ "ab", null, "cd", "ef" });
    defer values.release();
    var predicate = try makeBoolArray(allocator, &[_]?bool{ true, true, false, true });
    defer predicate.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(values.retain()),
        compute.Datum.fromArray(predicate.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    var out = try ctx.invokeVector("filter", args[0..], .{ .filter = .{} });
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{
        .fixed_size_binary = .{ .byte_width = 2 },
    }));

    const view = zcore.FixedSizeBinaryArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), view.len());
    try std.testing.expect(std.mem.eql(u8, view.value(0), "ab"));
    try std.testing.expect(view.isNull(1));
    try std.testing.expect(std.mem.eql(u8, view.value(2), "ef"));
}

test "filter supports chunked values and chunked predicates with misaligned chunk boundaries" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values_chunk0 = try makeInt64Array(allocator, &[_]?i64{ 1, null });
    defer values_chunk0.release();
    var values_chunk1 = try makeInt64Array(allocator, &[_]?i64{ 3, 4 });
    defer values_chunk1.release();
    var values_chunked = try compute.ChunkedArray.init(
        allocator,
        .{ .int64 = {} },
        &[_]zcore.ArrayRef{ values_chunk0, values_chunk1 },
    );
    defer values_chunked.release();

    var pred_chunk0 = try makeBoolArray(allocator, &[_]?bool{true});
    defer pred_chunk0.release();
    var pred_chunk1 = try makeBoolArray(allocator, &[_]?bool{ true, false, true });
    defer pred_chunk1.release();
    var pred_chunked = try compute.ChunkedArray.init(
        allocator,
        .{ .bool = {} },
        &[_]zcore.ArrayRef{ pred_chunk0, pred_chunk1 },
    );
    defer pred_chunked.release();

    const args = [_]compute.Datum{
        compute.Datum.fromChunked(values_chunked.retain()),
        compute.Datum.fromChunked(pred_chunked.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    var out = try ctx.invokeVector("filter", args[0..], .{ .filter = .{} });
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .int64 = {} }));

    const view = zcore.Int64Array{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), view.len());
    try std.testing.expectEqual(@as(i64, 1), view.value(0));
    try std.testing.expect(view.isNull(1));
    try std.testing.expectEqual(@as(i64, 4), view.value(2));
}

test "filter supports string_view value arrays" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeStringViewArray(allocator, &[_]?[]const u8{ "one", null, "two", "three" });
    defer values.release();
    var predicate = try makeBoolArray(allocator, &[_]?bool{ true, true, false, true });
    defer predicate.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(values.retain()),
        compute.Datum.fromArray(predicate.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    var out = try ctx.invokeVector("filter", args[0..], .{ .filter = .{} });
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .string_view = {} }));

    const view = zcore.StringViewArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), view.len());
    try std.testing.expect(std.mem.eql(u8, view.value(0), "one"));
    try std.testing.expect(view.isNull(1));
    try std.testing.expect(std.mem.eql(u8, view.value(2), "three"));
}

test "filter supports binary_view value arrays with predicate null emission" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeBinaryViewArray(allocator, &[_]?[]const u8{ "aa", "bb", null, "cc" });
    defer values.release();
    var predicate = try makeBoolArray(allocator, &[_]?bool{ true, null, true, false });
    defer predicate.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(values.retain()),
        compute.Datum.fromArray(predicate.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    var out = try ctx.invokeVector("filter", args[0..], .{
        .filter = .{ .drop_nulls = false },
    });
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .binary_view = {} }));

    const view = zcore.BinaryViewArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), view.len());
    try std.testing.expect(std.mem.eql(u8, view.value(0), "aa"));
    try std.testing.expect(view.isNull(1));
    try std.testing.expect(view.isNull(2));
}

test "filter supports list values with predicate null emission" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeListInt32ArrayWithLens(
        allocator,
        &[_]?usize{ 2, null, 1, 2 },
        &[_]i32{ 1, 2, 3, 4, 5 },
    );
    defer values.release();
    var predicate = try makeBoolArray(allocator, &[_]?bool{ true, null, false, true });
    defer predicate.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(values.retain()),
        compute.Datum.fromArray(predicate.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    var out = try ctx.invokeVector("filter", args[0..], .{
        .filter = .{ .drop_nulls = false },
    });
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType() == .list);

    const out_list = zcore.ListArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), out_list.len());

    var row0 = try out_list.value(0);
    defer row0.release();
    const row0_i32 = zcore.Int32Array{ .data = row0.data() };
    try std.testing.expectEqual(@as(usize, 2), row0_i32.len());
    try std.testing.expectEqual(@as(i32, 1), row0_i32.value(0));
    try std.testing.expectEqual(@as(i32, 2), row0_i32.value(1));

    try std.testing.expect(out_list.isNull(1));

    var row2 = try out_list.value(2);
    defer row2.release();
    const row2_i32 = zcore.Int32Array{ .data = row2.data() };
    try std.testing.expectEqual(@as(usize, 2), row2_i32.len());
    try std.testing.expectEqual(@as(i32, 4), row2_i32.value(0));
    try std.testing.expectEqual(@as(i32, 5), row2_i32.value(1));
}

test "filter supports large_list values with predicate-driven selection" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeLargeListInt32ArrayWithLens(
        allocator,
        &[_]?usize{ 1, 2, null, 1 },
        &[_]i32{ 10, 11, 12, 13 },
    );
    defer values.release();
    var predicate = try makeBoolArray(allocator, &[_]?bool{ false, true, null, true });
    defer predicate.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(values.retain()),
        compute.Datum.fromArray(predicate.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    var out = try ctx.invokeVector("filter", args[0..], .{ .filter = .{} });
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType() == .large_list);

    const out_list = zcore.LargeListArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 2), out_list.len());

    var row0 = try out_list.value(0);
    defer row0.release();
    const row0_i32 = zcore.Int32Array{ .data = row0.data() };
    try std.testing.expectEqual(@as(usize, 2), row0_i32.len());
    try std.testing.expectEqual(@as(i32, 11), row0_i32.value(0));
    try std.testing.expectEqual(@as(i32, 12), row0_i32.value(1));

    var row1 = try out_list.value(1);
    defer row1.release();
    const row1_i32 = zcore.Int32Array{ .data = row1.data() };
    try std.testing.expectEqual(@as(usize, 1), row1_i32.len());
    try std.testing.expectEqual(@as(i32, 13), row1_i32.value(0));
}

test "filter supports fixed_size_list values with predicate null emission" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeFixedSizeListInt32Array(
        allocator,
        2,
        &[_]bool{ true, true, true, true },
        &[_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 },
    );
    defer values.release();
    var predicate = try makeBoolArray(allocator, &[_]?bool{ true, null, true, false });
    defer predicate.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(values.retain()),
        compute.Datum.fromArray(predicate.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    var out = try ctx.invokeVector("filter", args[0..], .{
        .filter = .{ .drop_nulls = false },
    });
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType() == .fixed_size_list);

    const out_list = zcore.FixedSizeListArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), out_list.len());
    try std.testing.expectEqual(@as(usize, 2), out_list.listSize());
    try std.testing.expect(!out_list.isNull(0));
    try std.testing.expect(out_list.isNull(1));
    try std.testing.expect(!out_list.isNull(2));

    var row0 = try out_list.value(0);
    defer row0.release();
    const row0_i32 = zcore.Int32Array{ .data = row0.data() };
    try std.testing.expectEqual(@as(i32, 1), row0_i32.value(0));
    try std.testing.expectEqual(@as(i32, 2), row0_i32.value(1));

    var row2 = try out_list.value(2);
    defer row2.release();
    const row2_i32 = zcore.Int32Array{ .data = row2.data() };
    try std.testing.expectEqual(@as(i32, 5), row2_i32.value(0));
    try std.testing.expectEqual(@as(i32, 6), row2_i32.value(1));
}

test "filter fixed_size_list all-null predicate keeps aligned null rows" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeFixedSizeListInt32Array(
        allocator,
        2,
        &[_]bool{ true, true, true },
        &[_]i32{ 1, 2, 3, 4, 5, 6 },
    );
    defer values.release();
    var predicate = try makeBoolArray(allocator, &[_]?bool{ null, null, null });
    defer predicate.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(values.retain()),
        compute.Datum.fromArray(predicate.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    var out = try ctx.invokeVector("filter", args[0..], .{
        .filter = .{ .drop_nulls = false },
    });
    defer out.release();
    try expectFixedSizeListAllNullAligned(out, 3, 2);
}

test "filter supports struct values with predicate null emission" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeStructI64BoolArray(
        allocator,
        &[_]bool{ true, false, true, true },
        &[_]?i64{ 1, 2, 3, 4 },
        &[_]?bool{ true, false, null, true },
    );
    defer values.release();
    var predicate = try makeBoolArray(allocator, &[_]?bool{ true, null, true, false });
    defer predicate.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(values.retain()),
        compute.Datum.fromArray(predicate.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    var out = try ctx.invokeVector("filter", args[0..], .{
        .filter = .{ .drop_nulls = false },
    });
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType() == .struct_);

    const out_struct = zcore.StructArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), out_struct.len());
    try std.testing.expect(!out_struct.isNull(0));
    try std.testing.expect(out_struct.isNull(1));
    try std.testing.expect(!out_struct.isNull(2));

    const out_i64 = zcore.Int64Array{ .data = out_struct.fieldRef(0).data() };
    try std.testing.expectEqual(@as(i64, 1), out_i64.value(0));
    try std.testing.expectEqual(@as(i64, 3), out_i64.value(2));

    const out_bool = zcore.BooleanArray{ .data = out_struct.fieldRef(1).data() };
    try std.testing.expect(out_bool.value(0));
    try std.testing.expect(out_bool.isNull(2));
}

test "drop_null removes nulls from int64 arrays and keeps type" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeInt64Array(allocator, &[_]?i64{ 10, null, 20, null, 30 });
    defer values.release();
    const args = [_]compute.Datum{
        compute.Datum.fromArray(values.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }

    var out = try ctx.invokeVector("drop_null", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .int64 = {} }));

    const view = zcore.Int64Array{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), view.len());
    try std.testing.expectEqual(@as(i64, 10), view.value(0));
    try std.testing.expectEqual(@as(i64, 20), view.value(1));
    try std.testing.expectEqual(@as(i64, 30), view.value(2));
}

test "drop_null supports chunked input and preserves order" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var c0 = try makeInt64Array(allocator, &[_]?i64{ 1, null });
    defer c0.release();
    var c1 = try makeInt64Array(allocator, &[_]?i64{ null, 2, 3 });
    defer c1.release();
    var chunked = try compute.ChunkedArray.init(allocator, .{ .int64 = {} }, &[_]zcore.ArrayRef{ c0, c1 });
    defer chunked.release();

    const args = [_]compute.Datum{
        compute.Datum.fromChunked(chunked.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }

    var out = try ctx.invokeVector("drop_null", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .int64 = {} }));

    const view = zcore.Int64Array{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), view.len());
    try std.testing.expectEqual(@as(i64, 1), view.value(0));
    try std.testing.expectEqual(@as(i64, 2), view.value(1));
    try std.testing.expectEqual(@as(i64, 3), view.value(2));
}

test "drop_null supports list values" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeListInt32ArrayWithLens(
        allocator,
        &[_]?usize{ 2, null, 1, null, 1 },
        &[_]i32{ 1, 2, 3, 4 },
    );
    defer values.release();
    const args = [_]compute.Datum{
        compute.Datum.fromArray(values.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }

    var out = try ctx.invokeVector("drop_null", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType() == .list);

    const out_list = zcore.ListArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), out_list.len());

    var row0 = try out_list.value(0);
    defer row0.release();
    const row0_i32 = zcore.Int32Array{ .data = row0.data() };
    try std.testing.expectEqual(@as(i32, 1), row0_i32.value(0));
    try std.testing.expectEqual(@as(i32, 2), row0_i32.value(1));

    var row1 = try out_list.value(1);
    defer row1.release();
    const row1_i32 = zcore.Int32Array{ .data = row1.data() };
    try std.testing.expectEqual(@as(i32, 3), row1_i32.value(0));

    var row2 = try out_list.value(2);
    defer row2.release();
    const row2_i32 = zcore.Int32Array{ .data = row2.data() };
    try std.testing.expectEqual(@as(i32, 4), row2_i32.value(0));
}

test "drop_null supports large_list values" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeLargeListInt32ArrayWithLens(
        allocator,
        &[_]?usize{ null, 1, 2, null, 1 },
        &[_]i32{ 7, 8, 9, 10 },
    );
    defer values.release();
    const args = [_]compute.Datum{
        compute.Datum.fromArray(values.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }

    var out = try ctx.invokeVector("drop_null", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType() == .large_list);

    const out_list = zcore.LargeListArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), out_list.len());

    var row0 = try out_list.value(0);
    defer row0.release();
    const row0_i32 = zcore.Int32Array{ .data = row0.data() };
    try std.testing.expectEqual(@as(i32, 7), row0_i32.value(0));

    var row1 = try out_list.value(1);
    defer row1.release();
    const row1_i32 = zcore.Int32Array{ .data = row1.data() };
    try std.testing.expectEqual(@as(i32, 8), row1_i32.value(0));
    try std.testing.expectEqual(@as(i32, 9), row1_i32.value(1));

    var row2 = try out_list.value(2);
    defer row2.release();
    const row2_i32 = zcore.Int32Array{ .data = row2.data() };
    try std.testing.expectEqual(@as(i32, 10), row2_i32.value(0));
}

test "drop_null supports fixed_size_list values" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeFixedSizeListInt32Array(
        allocator,
        2,
        &[_]bool{ true, false, true, false, true },
        &[_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
    );
    defer values.release();
    const args = [_]compute.Datum{
        compute.Datum.fromArray(values.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }

    var out = try ctx.invokeVector("drop_null", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType() == .fixed_size_list);

    const out_list = zcore.FixedSizeListArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), out_list.len());
    try std.testing.expectEqual(@as(usize, 2), out_list.listSize());
    try std.testing.expect(!out_list.isNull(0));
    try std.testing.expect(!out_list.isNull(1));
    try std.testing.expect(!out_list.isNull(2));

    var row0 = try out_list.value(0);
    defer row0.release();
    const row0_i32 = zcore.Int32Array{ .data = row0.data() };
    try std.testing.expectEqual(@as(i32, 1), row0_i32.value(0));
    try std.testing.expectEqual(@as(i32, 2), row0_i32.value(1));

    var row1 = try out_list.value(1);
    defer row1.release();
    const row1_i32 = zcore.Int32Array{ .data = row1.data() };
    try std.testing.expectEqual(@as(i32, 5), row1_i32.value(0));
    try std.testing.expectEqual(@as(i32, 6), row1_i32.value(1));

    var row2 = try out_list.value(2);
    defer row2.release();
    const row2_i32 = zcore.Int32Array{ .data = row2.data() };
    try std.testing.expectEqual(@as(i32, 9), row2_i32.value(0));
    try std.testing.expectEqual(@as(i32, 10), row2_i32.value(1));
}

test "drop_null supports struct values" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeStructI64BoolArray(
        allocator,
        &[_]bool{ true, false, true, false },
        &[_]?i64{ 10, 20, 30, 40 },
        &[_]?bool{ true, false, null, true },
    );
    defer values.release();
    const args = [_]compute.Datum{
        compute.Datum.fromArray(values.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }

    var out = try ctx.invokeVector("drop_null", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType() == .struct_);

    const out_struct = zcore.StructArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 2), out_struct.len());
    try std.testing.expect(!out_struct.isNull(0));
    try std.testing.expect(!out_struct.isNull(1));

    const out_i64 = zcore.Int64Array{ .data = out_struct.fieldRef(0).data() };
    try std.testing.expectEqual(@as(i64, 10), out_i64.value(0));
    try std.testing.expectEqual(@as(i64, 30), out_i64.value(1));

    const out_bool = zcore.BooleanArray{ .data = out_struct.fieldRef(1).data() };
    try std.testing.expect(out_bool.value(0));
    try std.testing.expect(out_bool.isNull(1));
}
