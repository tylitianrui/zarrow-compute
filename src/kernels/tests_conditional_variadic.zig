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

test "coalesce supports variadic scalar broadcast and first-non-null semantics" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var primary = try makeInt64Array(allocator, &[_]?i64{ null, 2, null, 4 });
    defer primary.release();
    var backup = try makeInt64Array(allocator, &[_]?i64{ 7, null, 8, null });
    defer backup.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(primary.retain()),
        compute.Datum.fromScalar(.{
            .data_type = .{ .int64 = {} },
            .value = .{ .i64 = 9 },
        }),
        compute.Datum.fromArray(backup.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }
    defer {
        var d = args[2];
        d.release();
    }

    var out = try ctx.invokeVector("coalesce", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .int64 = {} }));

    const view = zcore.Int64Array{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 4), view.len());
    try std.testing.expectEqual(@as(i64, 9), view.value(0));
    try std.testing.expectEqual(@as(i64, 2), view.value(1));
    try std.testing.expectEqual(@as(i64, 9), view.value(2));
    try std.testing.expectEqual(@as(i64, 4), view.value(3));
}

test "coalesce outputs null only when all candidates are null" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var lhs = try makeStringArray(allocator, &[_]?[]const u8{ null, "x", null });
    defer lhs.release();
    var rhs = try makeStringArray(allocator, &[_]?[]const u8{ null, null, "y" });
    defer rhs.release();
    const args = [_]compute.Datum{
        compute.Datum.fromArray(lhs.retain()),
        compute.Datum.fromArray(rhs.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    var out = try ctx.invokeVector("coalesce", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .string = {} }));

    const view = zcore.StringArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), view.len());
    try std.testing.expect(view.isNull(0));
    try std.testing.expect(std.mem.eql(u8, view.value(1), "x"));
    try std.testing.expect(std.mem.eql(u8, view.value(2), "y"));
}

test "coalesce supports list values with first-non-null semantics" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var lhs = try makeListInt32ArrayWithLens(
        allocator,
        &[_]?usize{ null, 1, null, 2, null },
        &[_]i32{ 1, 2, 3 },
    );
    defer lhs.release();
    var rhs = try makeListInt32ArrayWithLens(
        allocator,
        &[_]?usize{ 1, 2, 1, null, null },
        &[_]i32{ 10, 11, 12, 13 },
    );
    defer rhs.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(lhs.retain()),
        compute.Datum.fromArray(rhs.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    var out = try ctx.invokeVector("coalesce", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType() == .list);

    const out_list = zcore.ListArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 5), out_list.len());

    var row0 = try out_list.value(0);
    defer row0.release();
    const row0_i32 = zcore.Int32Array{ .data = row0.data() };
    try std.testing.expectEqual(@as(usize, 1), row0_i32.len());
    try std.testing.expectEqual(@as(i32, 10), row0_i32.value(0));

    var row1 = try out_list.value(1);
    defer row1.release();
    const row1_i32 = zcore.Int32Array{ .data = row1.data() };
    try std.testing.expectEqual(@as(usize, 1), row1_i32.len());
    try std.testing.expectEqual(@as(i32, 1), row1_i32.value(0));

    var row2 = try out_list.value(2);
    defer row2.release();
    const row2_i32 = zcore.Int32Array{ .data = row2.data() };
    try std.testing.expectEqual(@as(usize, 1), row2_i32.len());
    try std.testing.expectEqual(@as(i32, 13), row2_i32.value(0));

    var row3 = try out_list.value(3);
    defer row3.release();
    const row3_i32 = zcore.Int32Array{ .data = row3.data() };
    try std.testing.expectEqual(@as(usize, 2), row3_i32.len());
    try std.testing.expectEqual(@as(i32, 2), row3_i32.value(0));
    try std.testing.expectEqual(@as(i32, 3), row3_i32.value(1));

    try std.testing.expect(out_list.isNull(4));
}

test "coalesce supports large_list scalar broadcast" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var lhs = try makeLargeListInt32ArrayWithLens(
        allocator,
        &[_]?usize{ null, 1, null },
        &[_]i32{1},
    );
    defer lhs.release();
    var rhs_payload = try makeLargeListInt32ArrayWithLens(
        allocator,
        &[_]?usize{2},
        &[_]i32{ 9, 10 },
    );
    defer rhs_payload.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(lhs.retain()),
        try makeNestedScalarDatum(rhs_payload),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    var out = try ctx.invokeVector("coalesce", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType() == .large_list);

    const out_list = zcore.LargeListArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), out_list.len());

    var row0 = try out_list.value(0);
    defer row0.release();
    const row0_i32 = zcore.Int32Array{ .data = row0.data() };
    try std.testing.expectEqual(@as(i32, 9), row0_i32.value(0));
    try std.testing.expectEqual(@as(i32, 10), row0_i32.value(1));

    var row1 = try out_list.value(1);
    defer row1.release();
    const row1_i32 = zcore.Int32Array{ .data = row1.data() };
    try std.testing.expectEqual(@as(i32, 1), row1_i32.value(0));

    var row2 = try out_list.value(2);
    defer row2.release();
    const row2_i32 = zcore.Int32Array{ .data = row2.data() };
    try std.testing.expectEqual(@as(i32, 9), row2_i32.value(0));
    try std.testing.expectEqual(@as(i32, 10), row2_i32.value(1));
}

test "coalesce supports fixed_size_list values with first-non-null semantics" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var lhs = try makeFixedSizeListInt32Array(
        allocator,
        2,
        &[_]bool{ false, true, false, true, false },
        &[_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
    );
    defer lhs.release();
    var rhs = try makeFixedSizeListInt32Array(
        allocator,
        2,
        &[_]bool{ true, false, true, true, false },
        &[_]i32{ 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 },
    );
    defer rhs.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(lhs.retain()),
        compute.Datum.fromArray(rhs.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    var out = try ctx.invokeVector("coalesce", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType() == .fixed_size_list);

    const out_list = zcore.FixedSizeListArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 5), out_list.len());
    try std.testing.expectEqual(@as(usize, 2), out_list.listSize());
    try std.testing.expect(!out_list.isNull(0));
    try std.testing.expect(!out_list.isNull(1));
    try std.testing.expect(!out_list.isNull(2));
    try std.testing.expect(!out_list.isNull(3));
    try std.testing.expect(out_list.isNull(4));

    var row0 = try out_list.value(0);
    defer row0.release();
    const row0_i32 = zcore.Int32Array{ .data = row0.data() };
    try std.testing.expectEqual(@as(i32, 11), row0_i32.value(0));
    try std.testing.expectEqual(@as(i32, 12), row0_i32.value(1));

    var row1 = try out_list.value(1);
    defer row1.release();
    const row1_i32 = zcore.Int32Array{ .data = row1.data() };
    try std.testing.expectEqual(@as(i32, 3), row1_i32.value(0));
    try std.testing.expectEqual(@as(i32, 4), row1_i32.value(1));

    var row2 = try out_list.value(2);
    defer row2.release();
    const row2_i32 = zcore.Int32Array{ .data = row2.data() };
    try std.testing.expectEqual(@as(i32, 15), row2_i32.value(0));
    try std.testing.expectEqual(@as(i32, 16), row2_i32.value(1));

    var row3 = try out_list.value(3);
    defer row3.release();
    const row3_i32 = zcore.Int32Array{ .data = row3.data() };
    try std.testing.expectEqual(@as(i32, 7), row3_i32.value(0));
    try std.testing.expectEqual(@as(i32, 8), row3_i32.value(1));
}

test "coalesce supports fixed_size_list null scalar broadcast" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var rhs = try makeFixedSizeListInt32Array(
        allocator,
        2,
        &[_]bool{ true, false, true },
        &[_]i32{ 1, 2, 3, 4, 5, 6 },
    );
    defer rhs.release();

    const args = [_]compute.Datum{
        compute.Datum.fromScalar(.{
            .data_type = rhs.data().data_type,
            .value = .null,
        }),
        compute.Datum.fromArray(rhs.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    var out = try ctx.invokeVector("coalesce", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType() == .fixed_size_list);

    const out_list = zcore.FixedSizeListArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), out_list.len());
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

test "coalesce fixed_size_list all-null candidates emit aligned null rows" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var lhs = try makeFixedSizeListInt32Array(
        allocator,
        2,
        &[_]bool{ false, false, false },
        &[_]i32{ 1, 2, 3, 4, 5, 6 },
    );
    defer lhs.release();
    var rhs = try makeFixedSizeListInt32Array(
        allocator,
        2,
        &[_]bool{ false, false, false },
        &[_]i32{ 11, 12, 13, 14, 15, 16 },
    );
    defer rhs.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(lhs.retain()),
        compute.Datum.fromArray(rhs.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    var out = try ctx.invokeVector("coalesce", args[0..], compute.Options.noneValue());
    defer out.release();
    try expectFixedSizeListAllNullAligned(out, 3, 2);
}

test "coalesce supports struct values with first-non-null semantics" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var lhs = try makeStructI64BoolArray(
        allocator,
        &[_]bool{ false, true, false, true },
        &[_]?i64{ 1, 2, 3, 4 },
        &[_]?bool{ true, false, true, false },
    );
    defer lhs.release();
    var rhs = try makeStructI64BoolArray(
        allocator,
        &[_]bool{ true, true, false, false },
        &[_]?i64{ 10, 20, 30, 40 },
        &[_]?bool{ false, true, false, true },
    );
    defer rhs.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(lhs.retain()),
        compute.Datum.fromArray(rhs.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    var out = try ctx.invokeVector("coalesce", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType() == .struct_);

    const out_struct = zcore.StructArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 4), out_struct.len());
    try std.testing.expect(!out_struct.isNull(0));
    try std.testing.expect(!out_struct.isNull(1));
    try std.testing.expect(out_struct.isNull(2));
    try std.testing.expect(!out_struct.isNull(3));

    const out_i64 = zcore.Int64Array{ .data = out_struct.fieldRef(0).data() };
    const out_bool = zcore.BooleanArray{ .data = out_struct.fieldRef(1).data() };
    try std.testing.expectEqual(@as(i64, 10), out_i64.value(0));
    try std.testing.expectEqual(false, out_bool.value(0));
    try std.testing.expectEqual(@as(i64, 2), out_i64.value(1));
    try std.testing.expectEqual(false, out_bool.value(1));
    try std.testing.expectEqual(@as(i64, 4), out_i64.value(3));
    try std.testing.expectEqual(false, out_bool.value(3));
}

test "choose supports variadic value selection with null propagation" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var indices = try makeInt32Array(allocator, &[_]?i32{ 0, 1, null, 1, 2 });
    defer indices.release();
    var v0 = try makeInt64Array(allocator, &[_]?i64{ 10, 11, 12, 13, 14 });
    defer v0.release();
    var v1 = try makeInt64Array(allocator, &[_]?i64{ 20, null, 22, 23, 24 });
    defer v1.release();
    const args = [_]compute.Datum{
        compute.Datum.fromArray(indices.retain()),
        compute.Datum.fromArray(v0.retain()),
        compute.Datum.fromArray(v1.retain()),
        compute.Datum.fromScalar(.{
            .data_type = .{ .int64 = {} },
            .value = .{ .i64 = 99 },
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
    defer {
        var d = args[2];
        d.release();
    }
    defer {
        var d = args[3];
        d.release();
    }

    var out = try ctx.invokeVector("choose", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .int64 = {} }));

    const view = zcore.Int64Array{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 5), view.len());
    try std.testing.expectEqual(@as(i64, 10), view.value(0));
    try std.testing.expect(view.isNull(1));
    try std.testing.expect(view.isNull(2));
    try std.testing.expectEqual(@as(i64, 23), view.value(3));
    try std.testing.expectEqual(@as(i64, 99), view.value(4));
}

test "choose supports large_list values with null propagation" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var indices = try makeInt32Array(allocator, &[_]?i32{ 0, 1, null, 2, 1 });
    defer indices.release();
    var v0 = try makeLargeListInt32ArrayWithLens(
        allocator,
        &[_]?usize{ 1, 1, 1, 1, 1 },
        &[_]i32{ 1, 2, 3, 4, 5 },
    );
    defer v0.release();
    var v1 = try makeLargeListInt32ArrayWithLens(
        allocator,
        &[_]?usize{ 1, null, 1, 1, 2 },
        &[_]i32{ 10, 11, 12, 13, 14 },
    );
    defer v1.release();
    var v2 = try makeLargeListInt32ArrayWithLens(
        allocator,
        &[_]?usize{ 2, 1, 1, 2, 1 },
        &[_]i32{ 20, 21, 22, 23, 24, 25, 26 },
    );
    defer v2.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(indices.retain()),
        compute.Datum.fromArray(v0.retain()),
        compute.Datum.fromArray(v1.retain()),
        compute.Datum.fromArray(v2.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }
    defer {
        var d = args[2];
        d.release();
    }
    defer {
        var d = args[3];
        d.release();
    }

    var out = try ctx.invokeVector("choose", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType() == .large_list);

    const out_list = zcore.LargeListArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 5), out_list.len());

    var row0 = try out_list.value(0);
    defer row0.release();
    const row0_i32 = zcore.Int32Array{ .data = row0.data() };
    try std.testing.expectEqual(@as(usize, 1), row0_i32.len());
    try std.testing.expectEqual(@as(i32, 1), row0_i32.value(0));

    try std.testing.expect(out_list.isNull(1));
    try std.testing.expect(out_list.isNull(2));

    var row3 = try out_list.value(3);
    defer row3.release();
    const row3_i32 = zcore.Int32Array{ .data = row3.data() };
    try std.testing.expectEqual(@as(usize, 2), row3_i32.len());
    try std.testing.expectEqual(@as(i32, 24), row3_i32.value(0));
    try std.testing.expectEqual(@as(i32, 25), row3_i32.value(1));

    var row4 = try out_list.value(4);
    defer row4.release();
    const row4_i32 = zcore.Int32Array{ .data = row4.data() };
    try std.testing.expectEqual(@as(usize, 2), row4_i32.len());
    try std.testing.expectEqual(@as(i32, 13), row4_i32.value(0));
    try std.testing.expectEqual(@as(i32, 14), row4_i32.value(1));
}

test "choose supports fixed_size_list values with null propagation" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var indices = try makeInt32Array(allocator, &[_]?i32{ 0, 1, null, 1, 0 });
    defer indices.release();
    var v0 = try makeFixedSizeListInt32Array(
        allocator,
        2,
        &[_]bool{ true, true, true, true, true },
        &[_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
    );
    defer v0.release();
    var v1 = try makeFixedSizeListInt32Array(
        allocator,
        2,
        &[_]bool{ true, false, true, true, true },
        &[_]i32{ 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 },
    );
    defer v1.release();
    const args = [_]compute.Datum{
        compute.Datum.fromArray(indices.retain()),
        compute.Datum.fromArray(v0.retain()),
        compute.Datum.fromArray(v1.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }
    defer {
        var d = args[2];
        d.release();
    }

    var out = try ctx.invokeVector("choose", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType() == .fixed_size_list);

    const out_list = zcore.FixedSizeListArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 5), out_list.len());
    try std.testing.expectEqual(@as(usize, 2), out_list.listSize());
    try std.testing.expect(!out_list.isNull(0));
    try std.testing.expect(out_list.isNull(1));
    try std.testing.expect(out_list.isNull(2));
    try std.testing.expect(!out_list.isNull(3));
    try std.testing.expect(!out_list.isNull(4));

    var row0 = try out_list.value(0);
    defer row0.release();
    const row0_i32 = zcore.Int32Array{ .data = row0.data() };
    try std.testing.expectEqual(@as(i32, 1), row0_i32.value(0));
    try std.testing.expectEqual(@as(i32, 2), row0_i32.value(1));

    var row3 = try out_list.value(3);
    defer row3.release();
    const row3_i32 = zcore.Int32Array{ .data = row3.data() };
    try std.testing.expectEqual(@as(i32, 17), row3_i32.value(0));
    try std.testing.expectEqual(@as(i32, 18), row3_i32.value(1));

    var row4 = try out_list.value(4);
    defer row4.release();
    const row4_i32 = zcore.Int32Array{ .data = row4.data() };
    try std.testing.expectEqual(@as(i32, 9), row4_i32.value(0));
    try std.testing.expectEqual(@as(i32, 10), row4_i32.value(1));
}

test "choose supports fixed_size_list scalar broadcast" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var indices = try makeInt32Array(allocator, &[_]?i32{ 0, 1, 0 });
    defer indices.release();
    var v0_payload = try makeFixedSizeListInt32Array(
        allocator,
        2,
        &[_]bool{true},
        &[_]i32{ 7, 8 },
    );
    defer v0_payload.release();
    var v1 = try makeFixedSizeListInt32Array(
        allocator,
        2,
        &[_]bool{ true, true, true },
        &[_]i32{ 11, 12, 13, 14, 15, 16 },
    );
    defer v1.release();
    const args = [_]compute.Datum{
        compute.Datum.fromArray(indices.retain()),
        try makeNestedScalarDatum(v0_payload),
        compute.Datum.fromArray(v1.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }
    defer {
        var d = args[2];
        d.release();
    }

    var out = try ctx.invokeVector("choose", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType() == .fixed_size_list);

    const out_list = zcore.FixedSizeListArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), out_list.len());
    try std.testing.expect(!out_list.isNull(0));
    try std.testing.expect(!out_list.isNull(1));
    try std.testing.expect(!out_list.isNull(2));

    var row0 = try out_list.value(0);
    defer row0.release();
    const row0_i32 = zcore.Int32Array{ .data = row0.data() };
    try std.testing.expectEqual(@as(i32, 7), row0_i32.value(0));
    try std.testing.expectEqual(@as(i32, 8), row0_i32.value(1));

    var row1 = try out_list.value(1);
    defer row1.release();
    const row1_i32 = zcore.Int32Array{ .data = row1.data() };
    try std.testing.expectEqual(@as(i32, 13), row1_i32.value(0));
    try std.testing.expectEqual(@as(i32, 14), row1_i32.value(1));

    var row2 = try out_list.value(2);
    defer row2.release();
    const row2_i32 = zcore.Int32Array{ .data = row2.data() };
    try std.testing.expectEqual(@as(i32, 7), row2_i32.value(0));
    try std.testing.expectEqual(@as(i32, 8), row2_i32.value(1));
}

test "choose supports fixed_size_list null scalar broadcast" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var indices = try makeInt32Array(allocator, &[_]?i32{ 0, 1, null, 0 });
    defer indices.release();
    var v1 = try makeFixedSizeListInt32Array(
        allocator,
        2,
        &[_]bool{ true, true, true, true },
        &[_]i32{ 11, 12, 13, 14, 15, 16, 17, 18 },
    );
    defer v1.release();
    const args = [_]compute.Datum{
        compute.Datum.fromArray(indices.retain()),
        compute.Datum.fromScalar(.{
            .data_type = v1.data().data_type,
            .value = .null,
        }),
        compute.Datum.fromArray(v1.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }
    defer {
        var d = args[2];
        d.release();
    }

    var out = try ctx.invokeVector("choose", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType() == .fixed_size_list);

    const out_list = zcore.FixedSizeListArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 4), out_list.len());
    try std.testing.expect(out_list.isNull(0));
    try std.testing.expect(!out_list.isNull(1));
    try std.testing.expect(out_list.isNull(2));
    try std.testing.expect(out_list.isNull(3));

    var row1 = try out_list.value(1);
    defer row1.release();
    const row1_i32 = zcore.Int32Array{ .data = row1.data() };
    try std.testing.expectEqual(@as(i32, 13), row1_i32.value(0));
    try std.testing.expectEqual(@as(i32, 14), row1_i32.value(1));
}

test "choose fixed_size_list all-null indices emit aligned null rows" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var indices = try makeInt32Array(allocator, &[_]?i32{ null, null, null });
    defer indices.release();
    var v0 = try makeFixedSizeListInt32Array(
        allocator,
        2,
        &[_]bool{ true, true, true },
        &[_]i32{ 1, 2, 3, 4, 5, 6 },
    );
    defer v0.release();
    var v1 = try makeFixedSizeListInt32Array(
        allocator,
        2,
        &[_]bool{ true, true, true },
        &[_]i32{ 11, 12, 13, 14, 15, 16 },
    );
    defer v1.release();
    const args = [_]compute.Datum{
        compute.Datum.fromArray(indices.retain()),
        compute.Datum.fromArray(v0.retain()),
        compute.Datum.fromArray(v1.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }
    defer {
        var d = args[2];
        d.release();
    }

    var out = try ctx.invokeVector("choose", args[0..], compute.Options.noneValue());
    defer out.release();
    try expectFixedSizeListAllNullAligned(out, 3, 2);
}

test "choose supports struct values with null propagation" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var indices = try makeInt32Array(allocator, &[_]?i32{ 0, 1, null, 1 });
    defer indices.release();
    var v0 = try makeStructI64BoolArray(
        allocator,
        &[_]bool{ true, true, true, true },
        &[_]?i64{ 1, 2, 3, 4 },
        &[_]?bool{ true, false, true, false },
    );
    defer v0.release();
    var v1 = try makeStructI64BoolArray(
        allocator,
        &[_]bool{ true, false, true, true },
        &[_]?i64{ 10, 20, 30, 40 },
        &[_]?bool{ false, true, false, true },
    );
    defer v1.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(indices.retain()),
        compute.Datum.fromArray(v0.retain()),
        compute.Datum.fromArray(v1.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }
    defer {
        var d = args[2];
        d.release();
    }

    var out = try ctx.invokeVector("choose", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType() == .struct_);

    const out_struct = zcore.StructArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 4), out_struct.len());
    try std.testing.expect(!out_struct.isNull(0));
    try std.testing.expect(out_struct.isNull(1));
    try std.testing.expect(out_struct.isNull(2));
    try std.testing.expect(!out_struct.isNull(3));

    const out_i64 = zcore.Int64Array{ .data = out_struct.fieldRef(0).data() };
    const out_bool = zcore.BooleanArray{ .data = out_struct.fieldRef(1).data() };
    try std.testing.expectEqual(@as(i64, 1), out_i64.value(0));
    try std.testing.expectEqual(true, out_bool.value(0));
    try std.testing.expectEqual(@as(i64, 40), out_i64.value(3));
    try std.testing.expectEqual(true, out_bool.value(3));
}

test "choose rejects out-of-bounds index" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var indices = try makeInt32Array(allocator, &[_]?i32{ 0, 3 });
    defer indices.release();
    var v0 = try makeInt64Array(allocator, &[_]?i64{ 1, 2 });
    defer v0.release();
    var v1 = try makeInt64Array(allocator, &[_]?i64{ 4, 5 });
    defer v1.release();
    const args = [_]compute.Datum{
        compute.Datum.fromArray(indices.retain()),
        compute.Datum.fromArray(v0.retain()),
        compute.Datum.fromArray(v1.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }
    defer {
        var d = args[2];
        d.release();
    }

    try std.testing.expectError(
        error.InvalidInput,
        ctx.invokeVector("choose", args[0..], compute.Options.noneValue()),
    );
}

test "case_when supports Arrow struct<bool...> conditions with optional else" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var cond0 = try makeBoolArray(allocator, &[_]?bool{ false, true, null, false, true });
    defer cond0.release();
    var cond1 = try makeBoolArray(allocator, &[_]?bool{ true, false, true, null, true });
    defer cond1.release();
    var conds = try makeStructBool2Array(allocator, cond0, cond1);
    defer conds.release();
    var v0 = try makeInt64Array(allocator, &[_]?i64{ 1, 1, 1, 1, null });
    defer v0.release();
    var v1 = try makeInt64Array(allocator, &[_]?i64{ 2, 2, null, 2, 2 });
    defer v1.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(conds.retain()),
        compute.Datum.fromArray(v0.retain()),
        compute.Datum.fromArray(v1.retain()),
        compute.Datum.fromScalar(.{
            .data_type = .{ .int64 = {} },
            .value = .{ .i64 = 9 },
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
    defer {
        var d = args[2];
        d.release();
    }
    defer {
        var d = args[3];
        d.release();
    }

    var out = try ctx.invokeVector("case_when", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .int64 = {} }));

    const view = zcore.Int64Array{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 5), view.len());
    try std.testing.expectEqual(@as(i64, 2), view.value(0));
    try std.testing.expectEqual(@as(i64, 1), view.value(1));
    try std.testing.expect(view.isNull(2));
    try std.testing.expectEqual(@as(i64, 9), view.value(3));
    try std.testing.expect(view.isNull(4));
}

test "case_when supports scalar struct<bool...> condition broadcast" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var cond0 = try makeBoolArray(allocator, &[_]?bool{true});
    defer cond0.release();
    var cond1 = try makeBoolArray(allocator, &[_]?bool{false});
    defer cond1.release();
    var conds_payload = try makeStructBool2Array(allocator, cond0, cond1);
    defer conds_payload.release();
    var v0 = try makeInt64Array(allocator, &[_]?i64{ 1, null, 3 });
    defer v0.release();
    var v1 = try makeInt64Array(allocator, &[_]?i64{ 10, 20, 30 });
    defer v1.release();
    const args = [_]compute.Datum{
        try makeNestedScalarDatum(conds_payload),
        compute.Datum.fromArray(v0.retain()),
        compute.Datum.fromArray(v1.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }
    defer {
        var d = args[2];
        d.release();
    }

    var out = try ctx.invokeVector("case_when", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .int64 = {} }));

    const view = zcore.Int64Array{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), view.len());
    try std.testing.expectEqual(@as(i64, 1), view.value(0));
    try std.testing.expect(view.isNull(1));
    try std.testing.expectEqual(@as(i64, 3), view.value(2));
}

test "case_when supports list values with struct<bool...> conditions" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var cond0 = try makeBoolArray(allocator, &[_]?bool{ true, false, null, false, false });
    defer cond0.release();
    var cond1 = try makeBoolArray(allocator, &[_]?bool{ false, true, true, null, false });
    defer cond1.release();
    var conds = try makeStructBool2Array(allocator, cond0, cond1);
    defer conds.release();
    var v0 = try makeListInt32ArrayWithLens(
        allocator,
        &[_]?usize{ 1, 1, 1, 1, 1 },
        &[_]i32{ 1, 2, 3, 4, 5 },
    );
    defer v0.release();
    var v1 = try makeListInt32ArrayWithLens(
        allocator,
        &[_]?usize{ 1, null, 1, 1, 1 },
        &[_]i32{ 10, 11, 12, 13 },
    );
    defer v1.release();
    var v_else = try makeListInt32ArrayWithLens(
        allocator,
        &[_]?usize{ 2, 1, 1, 1, 1 },
        &[_]i32{ 20, 21, 22, 23, 24, 25 },
    );
    defer v_else.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(conds.retain()),
        compute.Datum.fromArray(v0.retain()),
        compute.Datum.fromArray(v1.retain()),
        compute.Datum.fromArray(v_else.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }
    defer {
        var d = args[2];
        d.release();
    }
    defer {
        var d = args[3];
        d.release();
    }

    var out = try ctx.invokeVector("case_when", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType() == .list);

    const out_list = zcore.ListArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 5), out_list.len());

    var row0 = try out_list.value(0);
    defer row0.release();
    const row0_i32 = zcore.Int32Array{ .data = row0.data() };
    try std.testing.expectEqual(@as(usize, 1), row0_i32.len());
    try std.testing.expectEqual(@as(i32, 1), row0_i32.value(0));

    try std.testing.expect(out_list.isNull(1));

    var row2 = try out_list.value(2);
    defer row2.release();
    const row2_i32 = zcore.Int32Array{ .data = row2.data() };
    try std.testing.expectEqual(@as(usize, 1), row2_i32.len());
    try std.testing.expectEqual(@as(i32, 11), row2_i32.value(0));

    var row3 = try out_list.value(3);
    defer row3.release();
    const row3_i32 = zcore.Int32Array{ .data = row3.data() };
    try std.testing.expectEqual(@as(usize, 1), row3_i32.len());
    try std.testing.expectEqual(@as(i32, 24), row3_i32.value(0));

    var row4 = try out_list.value(4);
    defer row4.release();
    const row4_i32 = zcore.Int32Array{ .data = row4.data() };
    try std.testing.expectEqual(@as(usize, 1), row4_i32.len());
    try std.testing.expectEqual(@as(i32, 25), row4_i32.value(0));
}

test "case_when supports fixed_size_list values with struct<bool...> conditions" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var cond0 = try makeBoolArray(allocator, &[_]?bool{ true, false, false, null, false });
    defer cond0.release();
    var cond1 = try makeBoolArray(allocator, &[_]?bool{ false, true, false, true, false });
    defer cond1.release();
    var conds = try makeStructBool2Array(allocator, cond0, cond1);
    defer conds.release();
    var v0 = try makeFixedSizeListInt32Array(
        allocator,
        2,
        &[_]bool{ true, true, true, true, true },
        &[_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
    );
    defer v0.release();
    var v1 = try makeFixedSizeListInt32Array(
        allocator,
        2,
        &[_]bool{ true, false, true, true, true },
        &[_]i32{ 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 },
    );
    defer v1.release();
    var v_else = try makeFixedSizeListInt32Array(
        allocator,
        2,
        &[_]bool{ true, true, true, true, false },
        &[_]i32{ 21, 22, 23, 24, 25, 26, 27, 28, 29, 30 },
    );
    defer v_else.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(conds.retain()),
        compute.Datum.fromArray(v0.retain()),
        compute.Datum.fromArray(v1.retain()),
        compute.Datum.fromArray(v_else.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }
    defer {
        var d = args[2];
        d.release();
    }
    defer {
        var d = args[3];
        d.release();
    }

    var out = try ctx.invokeVector("case_when", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType() == .fixed_size_list);

    const out_list = zcore.FixedSizeListArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 5), out_list.len());
    try std.testing.expectEqual(@as(usize, 2), out_list.listSize());
    try std.testing.expect(!out_list.isNull(0));
    try std.testing.expect(out_list.isNull(1));
    try std.testing.expect(!out_list.isNull(2));
    try std.testing.expect(!out_list.isNull(3));
    try std.testing.expect(out_list.isNull(4));

    var row0 = try out_list.value(0);
    defer row0.release();
    const row0_i32 = zcore.Int32Array{ .data = row0.data() };
    try std.testing.expectEqual(@as(i32, 1), row0_i32.value(0));
    try std.testing.expectEqual(@as(i32, 2), row0_i32.value(1));

    var row2 = try out_list.value(2);
    defer row2.release();
    const row2_i32 = zcore.Int32Array{ .data = row2.data() };
    try std.testing.expectEqual(@as(i32, 25), row2_i32.value(0));
    try std.testing.expectEqual(@as(i32, 26), row2_i32.value(1));

    var row3 = try out_list.value(3);
    defer row3.release();
    const row3_i32 = zcore.Int32Array{ .data = row3.data() };
    try std.testing.expectEqual(@as(i32, 17), row3_i32.value(0));
    try std.testing.expectEqual(@as(i32, 18), row3_i32.value(1));
}

test "case_when supports fixed_size_list null scalar broadcast" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var cond0 = try makeBoolArray(allocator, &[_]?bool{ true, false, false });
    defer cond0.release();
    var cond1 = try makeBoolArray(allocator, &[_]?bool{ false, true, false });
    defer cond1.release();
    var conds = try makeStructBool2Array(allocator, cond0, cond1);
    defer conds.release();
    var v1 = try makeFixedSizeListInt32Array(
        allocator,
        2,
        &[_]bool{ true, true, true },
        &[_]i32{ 21, 22, 23, 24, 25, 26 },
    );
    defer v1.release();
    const args = [_]compute.Datum{
        compute.Datum.fromArray(conds.retain()),
        compute.Datum.fromScalar(.{
            .data_type = v1.data().data_type,
            .value = .null,
        }),
        compute.Datum.fromArray(v1.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }
    defer {
        var d = args[2];
        d.release();
    }

    var out = try ctx.invokeVector("case_when", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType() == .fixed_size_list);

    const out_list = zcore.FixedSizeListArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), out_list.len());
    try std.testing.expect(out_list.isNull(0));
    try std.testing.expect(!out_list.isNull(1));
    try std.testing.expect(out_list.isNull(2));

    var row1 = try out_list.value(1);
    defer row1.release();
    const row1_i32 = zcore.Int32Array{ .data = row1.data() };
    try std.testing.expectEqual(@as(i32, 23), row1_i32.value(0));
    try std.testing.expectEqual(@as(i32, 24), row1_i32.value(1));
}

test "case_when fixed_size_list without else and no matches emits aligned null rows" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var cond0 = try makeBoolArray(allocator, &[_]?bool{ false, false, false });
    defer cond0.release();
    var cond1 = try makeBoolArray(allocator, &[_]?bool{ false, false, false });
    defer cond1.release();
    var conds = try makeStructBool2Array(allocator, cond0, cond1);
    defer conds.release();
    var v0 = try makeFixedSizeListInt32Array(
        allocator,
        2,
        &[_]bool{ true, true, true },
        &[_]i32{ 1, 2, 3, 4, 5, 6 },
    );
    defer v0.release();
    var v1 = try makeFixedSizeListInt32Array(
        allocator,
        2,
        &[_]bool{ true, true, true },
        &[_]i32{ 11, 12, 13, 14, 15, 16 },
    );
    defer v1.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(conds.retain()),
        compute.Datum.fromArray(v0.retain()),
        compute.Datum.fromArray(v1.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }
    defer {
        var d = args[2];
        d.release();
    }

    var out = try ctx.invokeVector("case_when", args[0..], compute.Options.noneValue());
    defer out.release();
    try expectFixedSizeListAllNullAligned(out, 3, 2);
}

test "case_when supports struct values with struct<bool...> conditions" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var cond0 = try makeBoolArray(allocator, &[_]?bool{ true, false, false, false, false });
    defer cond0.release();
    var cond1 = try makeBoolArray(allocator, &[_]?bool{ false, true, true, false, null });
    defer cond1.release();
    var conds = try makeStructBool2Array(allocator, cond0, cond1);
    defer conds.release();
    var v0 = try makeStructI64BoolArray(
        allocator,
        &[_]bool{ true, true, true, true, true },
        &[_]?i64{ 1, 2, 3, 4, 5 },
        &[_]?bool{ true, false, true, false, true },
    );
    defer v0.release();
    var v1 = try makeStructI64BoolArray(
        allocator,
        &[_]bool{ true, false, true, true, true },
        &[_]?i64{ 10, 20, 30, 40, 50 },
        &[_]?bool{ false, true, false, true, false },
    );
    defer v1.release();
    var v_else = try makeStructI64BoolArray(
        allocator,
        &[_]bool{ true, true, true, true, false },
        &[_]?i64{ 100, 200, 300, 400, 500 },
        &[_]?bool{ true, true, true, true, true },
    );
    defer v_else.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(conds.retain()),
        compute.Datum.fromArray(v0.retain()),
        compute.Datum.fromArray(v1.retain()),
        compute.Datum.fromArray(v_else.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }
    defer {
        var d = args[2];
        d.release();
    }
    defer {
        var d = args[3];
        d.release();
    }

    var out = try ctx.invokeVector("case_when", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType() == .struct_);

    const out_struct = zcore.StructArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 5), out_struct.len());
    try std.testing.expect(!out_struct.isNull(0));
    try std.testing.expect(out_struct.isNull(1));
    try std.testing.expect(!out_struct.isNull(2));
    try std.testing.expect(!out_struct.isNull(3));
    try std.testing.expect(out_struct.isNull(4));

    const out_i64 = zcore.Int64Array{ .data = out_struct.fieldRef(0).data() };
    const out_bool = zcore.BooleanArray{ .data = out_struct.fieldRef(1).data() };
    try std.testing.expectEqual(@as(i64, 1), out_i64.value(0));
    try std.testing.expectEqual(true, out_bool.value(0));
    try std.testing.expectEqual(@as(i64, 30), out_i64.value(2));
    try std.testing.expectEqual(false, out_bool.value(2));
    try std.testing.expectEqual(@as(i64, 400), out_i64.value(3));
    try std.testing.expectEqual(true, out_bool.value(3));
}

test "case_when struct<bool...> without else falls back to null" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var cond0 = try makeBoolArray(allocator, &[_]?bool{ false, null, true });
    defer cond0.release();
    var cond1 = try makeBoolArray(allocator, &[_]?bool{ false, true, false });
    defer cond1.release();
    var conds = try makeStructBool2Array(allocator, cond0, cond1);
    defer conds.release();
    var v0 = try makeStringArray(allocator, &[_]?[]const u8{ "A", null, "C" });
    defer v0.release();
    var v1 = try makeStringArray(allocator, &[_]?[]const u8{ "B", "B", null });
    defer v1.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(conds.retain()),
        compute.Datum.fromArray(v0.retain()),
        compute.Datum.fromArray(v1.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }
    defer {
        var d = args[2];
        d.release();
    }

    var out = try ctx.invokeVector("case_when", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .string = {} }));

    const view = zcore.StringArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), view.len());
    try std.testing.expect(view.isNull(0));
    try std.testing.expect(std.mem.eql(u8, view.value(1), "B"));
    try std.testing.expect(std.mem.eql(u8, view.value(2), "C"));
}

test "case_when struct<bool...> rejects mismatched cases arity" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var cond0 = try makeBoolArray(allocator, &[_]?bool{ true, false });
    defer cond0.release();
    var cond1 = try makeBoolArray(allocator, &[_]?bool{ false, true });
    defer cond1.release();
    var conds = try makeStructBool2Array(allocator, cond0, cond1);
    defer conds.release();
    var v0 = try makeInt64Array(allocator, &[_]?i64{ 1, 2 });
    defer v0.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(conds.retain()),
        compute.Datum.fromArray(v0.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }
    defer {
        var d = args[1];
        d.release();
    }

    try std.testing.expectError(
        error.NoMatchingKernel,
        ctx.invokeVector("case_when", args[0..], compute.Options.noneValue()),
    );
}

test "case_when rejects legacy cond-value pair signature" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var cond0 = try makeBoolArray(allocator, &[_]?bool{ false, true, null, false, true });
    defer cond0.release();
    var v0 = try makeInt64Array(allocator, &[_]?i64{ 1, 1, 1, 1, null });
    defer v0.release();
    var cond1 = try makeBoolArray(allocator, &[_]?bool{ true, false, true, null, true });
    defer cond1.release();
    var v1 = try makeInt64Array(allocator, &[_]?i64{ 2, 2, null, 2, 2 });
    defer v1.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(cond0.retain()),
        compute.Datum.fromArray(v0.retain()),
        compute.Datum.fromArray(cond1.retain()),
        compute.Datum.fromArray(v1.retain()),
        compute.Datum.fromScalar(.{
            .data_type = .{ .int64 = {} },
            .value = .{ .i64 = 9 },
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
    defer {
        var d = args[2];
        d.release();
    }
    defer {
        var d = args[3];
        d.release();
    }
    defer {
        var d = args[4];
        d.release();
    }

    try std.testing.expectError(
        error.NoMatchingKernel,
        ctx.invokeVector("case_when", args[0..], compute.Options.noneValue()),
    );
}
