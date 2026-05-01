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

test "is_null marks null positions for chunked input" {
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

    var out = try ctx.invokeVector("is_null", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .bool = {} }));

    const view = zcore.BooleanArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 5), view.len());
    try std.testing.expect(!view.value(0));
    try std.testing.expect(view.value(1));
    try std.testing.expect(view.value(2));
    try std.testing.expect(!view.value(3));
    try std.testing.expect(!view.value(4));
}

test "is_valid is inverse of null positions" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeStringArray(allocator, &[_]?[]const u8{ "a", null, "b", null });
    defer values.release();
    const args = [_]compute.Datum{
        compute.Datum.fromArray(values.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }

    var out = try ctx.invokeVector("is_valid", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .bool = {} }));

    const view = zcore.BooleanArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 4), view.len());
    try std.testing.expect(view.value(0));
    try std.testing.expect(!view.value(1));
    try std.testing.expect(view.value(2));
    try std.testing.expect(!view.value(3));
}

test "is_finite/is_inf/is_nan support float64 with null propagation" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    const pos_inf = std.math.inf(f64);
    const neg_inf = -std.math.inf(f64);
    const nan_value = std.math.nan(f64);
    var values = try makeFloat64Array(allocator, &[_]?f64{ 1.5, pos_inf, neg_inf, nan_value, null });
    defer values.release();
    const args = [_]compute.Datum{
        compute.Datum.fromArray(values.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }

    var finite_out = try ctx.invokeVector("is_finite", args[0..], compute.Options.noneValue());
    defer finite_out.release();
    try expectBoolArrayValues(finite_out, &[_]?bool{ true, false, false, false, null });

    var inf_out = try ctx.invokeVector("is_inf", args[0..], compute.Options.noneValue());
    defer inf_out.release();
    try expectBoolArrayValues(inf_out, &[_]?bool{ false, true, true, false, null });

    var nan_out = try ctx.invokeVector("is_nan", args[0..], compute.Options.noneValue());
    defer nan_out.release();
    try expectBoolArrayValues(nan_out, &[_]?bool{ false, false, false, true, null });
}

test "is_finite rejects non-float input" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeInt64Array(allocator, &[_]?i64{ 1, null, 3 });
    defer values.release();
    const args = [_]compute.Datum{
        compute.Datum.fromArray(values.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }

    try std.testing.expectError(
        error.NoMatchingKernel,
        ctx.invokeVector("is_finite", args[0..], compute.Options.noneValue()),
    );
}

test "is_null rejects non-none options" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeInt64Array(allocator, &[_]?i64{ 1, null });
    defer values.release();
    const args = [_]compute.Datum{
        compute.Datum.fromArray(values.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }

    try std.testing.expectError(
        error.InvalidOptions,
        ctx.invokeVector("is_null", args[0..], .{ .filter = .{} }),
    );
}

test "true_unless_null returns true for non-null and false for null" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var values = try makeBoolArray(allocator, &[_]?bool{ true, false, null, true });
    defer values.release();
    const args = [_]compute.Datum{
        compute.Datum.fromArray(values.retain()),
    };
    defer {
        var d = args[0];
        d.release();
    }

    var out = try ctx.invokeVector("true_unless_null", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .bool = {} }));

    const view = zcore.BooleanArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 4), view.len());
    try std.testing.expect(view.value(0));
    try std.testing.expect(view.value(1));
    try std.testing.expect(!view.value(2));
    try std.testing.expect(view.value(3));
}

test "if_else supports fixed-width with scalar broadcast and condition null semantics" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var cond = try makeBoolArray(allocator, &[_]?bool{ true, false, null, true, false });
    defer cond.release();
    var lhs = try makeInt64Array(allocator, &[_]?i64{ 1, null, 3, 4, 5 });
    defer lhs.release();
    const args = [_]compute.Datum{
        compute.Datum.fromArray(cond.retain()),
        compute.Datum.fromArray(lhs.retain()),
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

    var out = try ctx.invokeVector("if_else", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .int64 = {} }));

    const view = zcore.Int64Array{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 5), view.len());
    try std.testing.expectEqual(@as(i64, 1), view.value(0));
    try std.testing.expectEqual(@as(i64, 9), view.value(1));
    try std.testing.expect(view.isNull(2));
    try std.testing.expectEqual(@as(i64, 4), view.value(3));
    try std.testing.expectEqual(@as(i64, 9), view.value(4));
}

test "if_else supports string values and null propagation from selected branch" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var cond = try makeBoolArray(allocator, &[_]?bool{ false, true, null });
    defer cond.release();
    var lhs = try makeStringArray(allocator, &[_]?[]const u8{ "L0", null, "L2" });
    defer lhs.release();
    const args = [_]compute.Datum{
        compute.Datum.fromArray(cond.retain()),
        compute.Datum.fromArray(lhs.retain()),
        compute.Datum.fromScalar(.{
            .data_type = .{ .string = {} },
            .value = .{ .string = "R" },
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

    var out = try ctx.invokeVector("if_else", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .string = {} }));

    const view = zcore.StringArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), view.len());
    try std.testing.expect(std.mem.eql(u8, view.value(0), "R"));
    try std.testing.expect(view.isNull(1));
    try std.testing.expect(view.isNull(2));
}

test "if_else supports bool values with branch and condition null propagation" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var cond = try makeBoolArray(allocator, &[_]?bool{ true, false, true, false, null });
    defer cond.release();
    var lhs = try makeBoolArray(allocator, &[_]?bool{ true, null, false, true, true });
    defer lhs.release();
    var rhs = try makeBoolArray(allocator, &[_]?bool{ false, true, null, null, false });
    defer rhs.release();
    const args = [_]compute.Datum{
        compute.Datum.fromArray(cond.retain()),
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
    defer {
        var d = args[2];
        d.release();
    }

    var out = try ctx.invokeVector("if_else", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .bool = {} }));

    const view = zcore.BooleanArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 5), view.len());
    try std.testing.expectEqual(true, view.value(0));
    try std.testing.expectEqual(true, view.value(1));
    try std.testing.expectEqual(false, view.value(2));
    try std.testing.expect(view.isNull(3));
    try std.testing.expect(view.isNull(4));
}

test "if_else supports null type" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var cond = try makeBoolArray(allocator, &[_]?bool{ true, false, null, true });
    defer cond.release();
    var lhs = try makeNullArray(allocator, 4);
    defer lhs.release();
    const args = [_]compute.Datum{
        compute.Datum.fromArray(cond.retain()),
        compute.Datum.fromArray(lhs.retain()),
        compute.Datum.fromScalar(.{
            .data_type = .{ .null = {} },
            .value = .null,
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

    var out = try ctx.invokeVector("if_else", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .null = {} }));

    const view = zcore.NullArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 4), view.len());
    try std.testing.expect(view.isNull(0));
    try std.testing.expect(view.isNull(1));
    try std.testing.expect(view.isNull(2));
    try std.testing.expect(view.isNull(3));
}

test "if_else supports struct values with parent null propagation" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var cond = try makeBoolArray(allocator, &[_]?bool{ true, false, null, true });
    defer cond.release();
    var lhs = try makeStructI64BoolArray(
        allocator,
        &[_]bool{ true, true, true, false },
        &[_]?i64{ 1, 2, 3, 4 },
        &[_]?bool{ true, false, true, false },
    );
    defer lhs.release();
    var rhs = try makeStructI64BoolArray(
        allocator,
        &[_]bool{ true, false, true, true },
        &[_]?i64{ 10, 20, 30, 40 },
        &[_]?bool{ false, true, false, true },
    );
    defer rhs.release();
    const args = [_]compute.Datum{
        compute.Datum.fromArray(cond.retain()),
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
    defer {
        var d = args[2];
        d.release();
    }

    var out = try ctx.invokeVector("if_else", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType() == .struct_);

    const out_struct = zcore.StructArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 4), out_struct.len());
    try std.testing.expect(!out_struct.isNull(0));
    try std.testing.expect(out_struct.isNull(1));
    try std.testing.expect(out_struct.isNull(2));
    try std.testing.expect(out_struct.isNull(3));

    const out_i64 = zcore.Int64Array{ .data = out_struct.fieldRef(0).data() };
    const out_bool = zcore.BooleanArray{ .data = out_struct.fieldRef(1).data() };
    try std.testing.expectEqual(@as(i64, 1), out_i64.value(0));
    try std.testing.expectEqual(true, out_bool.value(0));
}

test "if_else supports list values with selected-branch null propagation" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var cond = try makeBoolArray(allocator, &[_]?bool{ true, false, null, true, false, false });
    defer cond.release();
    var lhs = try makeListInt32ArrayWithLens(
        allocator,
        &[_]?usize{ 2, null, 1, 2, 1, 1 },
        &[_]i32{ 1, 2, 3, 4, 5, 6, 7 },
    );
    defer lhs.release();
    var rhs = try makeListInt32ArrayWithLens(
        allocator,
        &[_]?usize{ 1, 2, 1, 1, 2, null },
        &[_]i32{ 10, 11, 12, 13, 14, 15, 16 },
    );
    defer rhs.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(cond.retain()),
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
    defer {
        var d = args[2];
        d.release();
    }

    var out = try ctx.invokeVector("if_else", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType() == .list);

    const out_list = zcore.ListArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 6), out_list.len());

    var row0 = try out_list.value(0);
    defer row0.release();
    const row0_i32 = zcore.Int32Array{ .data = row0.data() };
    try std.testing.expectEqual(@as(usize, 2), row0_i32.len());
    try std.testing.expectEqual(@as(i32, 1), row0_i32.value(0));
    try std.testing.expectEqual(@as(i32, 2), row0_i32.value(1));

    var row1 = try out_list.value(1);
    defer row1.release();
    const row1_i32 = zcore.Int32Array{ .data = row1.data() };
    try std.testing.expectEqual(@as(usize, 2), row1_i32.len());
    try std.testing.expectEqual(@as(i32, 11), row1_i32.value(0));
    try std.testing.expectEqual(@as(i32, 12), row1_i32.value(1));

    try std.testing.expect(out_list.isNull(2));

    var row3 = try out_list.value(3);
    defer row3.release();
    const row3_i32 = zcore.Int32Array{ .data = row3.data() };
    try std.testing.expectEqual(@as(usize, 2), row3_i32.len());
    try std.testing.expectEqual(@as(i32, 4), row3_i32.value(0));
    try std.testing.expectEqual(@as(i32, 5), row3_i32.value(1));

    var row4 = try out_list.value(4);
    defer row4.release();
    const row4_i32 = zcore.Int32Array{ .data = row4.data() };
    try std.testing.expectEqual(@as(usize, 2), row4_i32.len());
    try std.testing.expectEqual(@as(i32, 15), row4_i32.value(0));
    try std.testing.expectEqual(@as(i32, 16), row4_i32.value(1));

    try std.testing.expect(out_list.isNull(5));
}

test "if_else supports list scalar broadcast" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var cond = try makeBoolArray(allocator, &[_]?bool{ true, false, true });
    defer cond.release();
    var lhs_payload = try makeListInt32ArrayWithLens(
        allocator,
        &[_]?usize{2},
        &[_]i32{ 42, 43 },
    );
    defer lhs_payload.release();
    var rhs = try makeListInt32ArrayWithLens(
        allocator,
        &[_]?usize{ 1, 1, 1 },
        &[_]i32{ 1, 2, 3 },
    );
    defer rhs.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(cond.retain()),
        try makeNestedScalarDatum(lhs_payload),
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
    defer {
        var d = args[2];
        d.release();
    }

    var out = try ctx.invokeVector("if_else", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType() == .list);

    const out_list = zcore.ListArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), out_list.len());

    var row0 = try out_list.value(0);
    defer row0.release();
    const row0_i32 = zcore.Int32Array{ .data = row0.data() };
    try std.testing.expectEqual(@as(i32, 42), row0_i32.value(0));
    try std.testing.expectEqual(@as(i32, 43), row0_i32.value(1));

    var row1 = try out_list.value(1);
    defer row1.release();
    const row1_i32 = zcore.Int32Array{ .data = row1.data() };
    try std.testing.expectEqual(@as(i32, 2), row1_i32.value(0));

    var row2 = try out_list.value(2);
    defer row2.release();
    const row2_i32 = zcore.Int32Array{ .data = row2.data() };
    try std.testing.expectEqual(@as(i32, 42), row2_i32.value(0));
    try std.testing.expectEqual(@as(i32, 43), row2_i32.value(1));
}

test "if_else supports struct with list child field" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var cond = try makeBoolArray(allocator, &[_]?bool{ true, false, null });
    defer cond.release();
    var lhs_list = try makeListInt32ArrayWithLens(
        allocator,
        &[_]?usize{ 1, 2, 1 },
        &[_]i32{ 1, 2, 3, 4 },
    );
    defer lhs_list.release();
    var rhs_list = try makeListInt32ArrayWithLens(
        allocator,
        &[_]?usize{ 1, 1, 2 },
        &[_]i32{ 9, 8, 7, 6 },
    );
    defer rhs_list.release();
    var lhs = try makeStructListI32Array(allocator, lhs_list, &[_]bool{ true, true, true });
    defer lhs.release();
    var rhs = try makeStructListI32Array(allocator, rhs_list, &[_]bool{ true, true, true });
    defer rhs.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(cond.retain()),
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
    defer {
        var d = args[2];
        d.release();
    }

    var out = try ctx.invokeVector("if_else", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType() == .struct_);

    const out_struct = zcore.StructArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), out_struct.len());
    try std.testing.expect(!out_struct.isNull(0));
    try std.testing.expect(!out_struct.isNull(1));
    try std.testing.expect(out_struct.isNull(2));

    const out_items = zcore.ListArray{ .data = out_struct.fieldRef(0).data() };
    var row0 = try out_items.value(0);
    defer row0.release();
    const row0_i32 = zcore.Int32Array{ .data = row0.data() };
    try std.testing.expectEqual(@as(usize, 1), row0_i32.len());
    try std.testing.expectEqual(@as(i32, 1), row0_i32.value(0));

    var row1 = try out_items.value(1);
    defer row1.release();
    const row1_i32 = zcore.Int32Array{ .data = row1.data() };
    try std.testing.expectEqual(@as(usize, 1), row1_i32.len());
    try std.testing.expectEqual(@as(i32, 8), row1_i32.value(0));
    try std.testing.expect(out_items.isNull(2));
}

test "if_else supports misaligned chunk boundaries across three inputs" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var cond_c0 = try makeBoolArray(allocator, &[_]?bool{true});
    defer cond_c0.release();
    var cond_c1 = try makeBoolArray(allocator, &[_]?bool{ false, null, true });
    defer cond_c1.release();
    var cond_chunked = try compute.ChunkedArray.init(allocator, .{ .bool = {} }, &[_]zcore.ArrayRef{ cond_c0, cond_c1 });
    defer cond_chunked.release();

    var lhs_c0 = try makeInt32Array(allocator, &[_]?i32{ 10, 11 });
    defer lhs_c0.release();
    var lhs_c1 = try makeInt32Array(allocator, &[_]?i32{ 12, 13 });
    defer lhs_c1.release();
    var lhs_chunked = try compute.ChunkedArray.init(allocator, .{ .int32 = {} }, &[_]zcore.ArrayRef{ lhs_c0, lhs_c1 });
    defer lhs_chunked.release();

    var rhs_c0 = try makeInt32Array(allocator, &[_]?i32{20});
    defer rhs_c0.release();
    var rhs_c1 = try makeInt32Array(allocator, &[_]?i32{ 21, 22, 23 });
    defer rhs_c1.release();
    var rhs_chunked = try compute.ChunkedArray.init(allocator, .{ .int32 = {} }, &[_]zcore.ArrayRef{ rhs_c0, rhs_c1 });
    defer rhs_chunked.release();

    const args = [_]compute.Datum{
        compute.Datum.fromChunked(cond_chunked.retain()),
        compute.Datum.fromChunked(lhs_chunked.retain()),
        compute.Datum.fromChunked(rhs_chunked.retain()),
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

    var out = try ctx.invokeVector("if_else", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType().eql(.{ .int32 = {} }));

    const view = zcore.Int32Array{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 4), view.len());
    try std.testing.expectEqual(@as(i32, 10), try view.value(0));
    try std.testing.expectEqual(@as(i32, 21), try view.value(1));
    try std.testing.expect(view.isNull(2));
    try std.testing.expectEqual(@as(i32, 13), try view.value(3));
}

test "if_else supports fixed_size_list values with null propagation" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var cond = try makeBoolArray(allocator, &[_]?bool{ true, false, null, true, false });
    defer cond.release();
    var lhs = try makeFixedSizeListInt32Array(
        allocator,
        2,
        &[_]bool{ true, true, true, false, true },
        &[_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
    );
    defer lhs.release();
    var rhs = try makeFixedSizeListInt32Array(
        allocator,
        2,
        &[_]bool{ true, false, true, true, true },
        &[_]i32{ 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 },
    );
    defer rhs.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(cond.retain()),
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
    defer {
        var d = args[2];
        d.release();
    }

    var out = try ctx.invokeVector("if_else", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType() == .fixed_size_list);

    const out_list = zcore.FixedSizeListArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 5), out_list.len());
    try std.testing.expectEqual(@as(usize, 2), out_list.listSize());
    try std.testing.expect(!out_list.isNull(0));
    try std.testing.expect(out_list.isNull(1));
    try std.testing.expect(out_list.isNull(2));
    try std.testing.expect(out_list.isNull(3));
    try std.testing.expect(!out_list.isNull(4));

    var row0 = try out_list.value(0);
    defer row0.release();
    const row0_i32 = zcore.Int32Array{ .data = row0.data() };
    try std.testing.expectEqual(@as(i32, 1), row0_i32.value(0));
    try std.testing.expectEqual(@as(i32, 2), row0_i32.value(1));

    var row4 = try out_list.value(4);
    defer row4.release();
    const row4_i32 = zcore.Int32Array{ .data = row4.data() };
    try std.testing.expectEqual(@as(i32, 19), row4_i32.value(0));
    try std.testing.expectEqual(@as(i32, 20), row4_i32.value(1));
}

test "if_else supports fixed_size_list null scalar broadcast" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var cond = try makeBoolArray(allocator, &[_]?bool{ true, false, true, false });
    defer cond.release();
    var rhs = try makeFixedSizeListInt32Array(
        allocator,
        2,
        &[_]bool{ true, true, true, true },
        &[_]i32{ 11, 12, 13, 14, 15, 16, 17, 18 },
    );
    defer rhs.release();
    const args = [_]compute.Datum{
        compute.Datum.fromArray(cond.retain()),
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
    defer {
        var d = args[2];
        d.release();
    }

    var out = try ctx.invokeVector("if_else", args[0..], compute.Options.noneValue());
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expect(out.dataType() == .fixed_size_list);

    const out_list = zcore.FixedSizeListArray{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 4), out_list.len());
    try std.testing.expect(out_list.isNull(0));
    try std.testing.expect(!out_list.isNull(1));
    try std.testing.expect(out_list.isNull(2));
    try std.testing.expect(!out_list.isNull(3));

    var row1 = try out_list.value(1);
    defer row1.release();
    const row1_i32 = zcore.Int32Array{ .data = row1.data() };
    try std.testing.expectEqual(@as(i32, 13), row1_i32.value(0));
    try std.testing.expectEqual(@as(i32, 14), row1_i32.value(1));

    var row3 = try out_list.value(3);
    defer row3.release();
    const row3_i32 = zcore.Int32Array{ .data = row3.data() };
    try std.testing.expectEqual(@as(i32, 17), row3_i32.value(0));
    try std.testing.expectEqual(@as(i32, 18), row3_i32.value(1));
}

test "if_else fixed_size_list all-null condition emits aligned null rows" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var cond = try makeBoolArray(allocator, &[_]?bool{ null, null, null, null });
    defer cond.release();
    var lhs = try makeFixedSizeListInt32Array(
        allocator,
        2,
        &[_]bool{ true, true, true, true },
        &[_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 },
    );
    defer lhs.release();
    var rhs = try makeFixedSizeListInt32Array(
        allocator,
        2,
        &[_]bool{ true, true, true, true },
        &[_]i32{ 11, 12, 13, 14, 15, 16, 17, 18 },
    );
    defer rhs.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(cond.retain()),
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
    defer {
        var d = args[2];
        d.release();
    }

    var out = try ctx.invokeVector("if_else", args[0..], compute.Options.noneValue());
    defer out.release();
    try expectFixedSizeListAllNullAligned(out, 4, 2);
}

test "if_else rejects non-none options" {
    const allocator = std.testing.allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try registerBaseKernels(&registry);
    var ctx = compute.ExecContext.init(allocator, &registry);

    var cond = try makeBoolArray(allocator, &[_]?bool{ true, false });
    defer cond.release();
    var lhs = try makeInt64Array(allocator, &[_]?i64{ 1, 2 });
    defer lhs.release();
    var rhs = try makeInt64Array(allocator, &[_]?i64{ 3, 4 });
    defer rhs.release();
    const args = [_]compute.Datum{
        compute.Datum.fromArray(cond.retain()),
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
    defer {
        var d = args[2];
        d.release();
    }

    try std.testing.expectError(
        error.InvalidOptions,
        ctx.invokeVector("if_else", args[0..], .{ .filter = .{} }),
    );
}
