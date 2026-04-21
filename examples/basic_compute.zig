const std = @import("std");
const zcore = @import("zarrow-core");
const zcompute = @import("zarrow_compute");

const compute = zcore.compute;

fn makeInt64Array(allocator: std.mem.Allocator, values: []const ?i64) !zcore.ArrayRef {
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

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try zcompute.registerBaseKernels(&registry);

    var ctx = compute.ExecContext.init(allocator, &registry);
    var left = try makeInt64Array(allocator, &[_]?i64{ 1, null, 3 });
    defer left.release();

    const args = [_]compute.Datum{
        compute.Datum.fromArray(left.retain()),
        compute.Datum.fromScalar(.{
            .data_type = .{ .int64 = {} },
            .value = .{ .i64 = 5 },
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

    const view = zcore.Int64Array{ .data = out.array.data() };
    std.debug.print("add_i64 => [", .{});
    var i: usize = 0;
    while (i < view.len()) : (i += 1) {
        if (i != 0) std.debug.print(", ", .{});
        if (view.isNull(i)) {
            std.debug.print("null", .{});
        } else {
            std.debug.print("{d}", .{view.value(i)});
        }
    }
    std.debug.print("]\n", .{});

    var count = try ctx.invokeAggregate("count_rows", args[0..1], compute.Options.noneValue());
    defer count.release();
    std.debug.print("count_rows => {d}\n", .{count.scalar.value.i64});
}
