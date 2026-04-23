const common = @import("common.zig");
const arithmetic = @import("arithmetic.zig");
const filter = @import("filter.zig");
const nulls = @import("nulls.zig");
const conditionals = @import("conditionals.zig");
const cast = @import("cast.zig");
const aggregate = @import("aggregate.zig");

pub const compute = common.compute;

pub fn registerBaseKernels(registry: *compute.FunctionRegistry) compute.KernelError!void {
    try registry.registerVectorKernel("add_i64", .{
        .signature = .{
            .arity = 2,
            .type_check = common.binaryInt64,
            .options_check = common.onlyArithmeticOptions,
            .result_type_fn = common.resultI64,
        },
        .exec = arithmetic.addI64Kernel,
    });

    try registry.registerVectorKernel("filter", .{
        .signature = .{
            .arity = 2,
            .type_check = common.binarySupportedFilter,
            .options_check = common.onlyFilterOptions,
            .result_type_fn = common.resultSameAsFirst,
        },
        .exec = filter.filterKernel,
    });

    try registry.registerVectorKernel("filter_i64", .{
        .signature = .{
            .arity = 2,
            .type_check = common.binaryInt64Bool,
            .options_check = common.onlyFilterOptions,
            .result_type_fn = common.resultI64,
        },
        .exec = filter.filterKernel,
    });

    try registry.registerVectorKernel("drop_null", .{
        .signature = .{
            .arity = 1,
            .type_check = common.unarySupportedFilter,
            .options_check = common.onlyNoOptions,
            .result_type_fn = common.resultSameAsFirst,
        },
        .exec = filter.dropNullKernel,
    });

    try registry.registerVectorKernel("is_null", .{
        .signature = .{
            .arity = 1,
            .type_check = common.unaryArrayLike,
            .options_check = common.onlyNoOptions,
            .result_type_fn = common.resultBool,
        },
        .exec = nulls.isNullKernel,
    });

    try registry.registerVectorKernel("is_valid", .{
        .signature = .{
            .arity = 1,
            .type_check = common.unaryArrayLike,
            .options_check = common.onlyNoOptions,
            .result_type_fn = common.resultBool,
        },
        .exec = nulls.isValidKernel,
    });

    try registry.registerVectorKernel("true_unless_null", .{
        .signature = .{
            .arity = 1,
            .type_check = common.unaryArrayLike,
            .options_check = common.onlyNoOptions,
            .result_type_fn = common.resultBool,
        },
        .exec = conditionals.trueUnlessNullKernel,
    });

    try registry.registerVectorKernel("if_else", .{
        .signature = .{
            .arity = 3,
            .type_check = common.ternaryBoolIfElseSupported,
            .options_check = common.onlyNoOptions,
            .result_type_fn = common.resultSameAsSecond,
        },
        .exec = conditionals.ifElseKernel,
    });

    try registry.registerVectorKernel("subtract_i64", .{
        .signature = .{
            .arity = 2,
            .type_check = common.binaryInt64,
            .options_check = common.onlyArithmeticOptions,
            .result_type_fn = common.resultI64,
        },
        .exec = arithmetic.subtractI64Kernel,
    });

    try registry.registerVectorKernel("divide_i64", .{
        .signature = .{
            .arity = 2,
            .type_check = common.binaryInt64,
            .options_check = common.onlyArithmeticOptions,
            .result_type_fn = common.resultI64,
        },
        .exec = arithmetic.divideI64Kernel,
    });

    try registry.registerVectorKernel("multiply_i64", .{
        .signature = .{
            .arity = 2,
            .type_check = common.binaryInt64,
            .options_check = common.onlyArithmeticOptions,
            .result_type_fn = common.resultI64,
        },
        .exec = arithmetic.multiplyI64Kernel,
    });

    try registry.registerVectorKernel("cast_i64_to_i32", .{
        .signature = .{
            .arity = 1,
            .type_check = common.unaryInt64,
            .options_check = common.onlyCastOptions,
            .result_type_fn = common.resultI32,
        },
        .exec = cast.castI64ToI32Kernel,
    });

    try registry.registerAggregateKernel("count_rows", .{
        .signature = .{
            .arity = 1,
            .type_check = common.unaryArrayLike,
            .options_check = common.onlyNoOptions,
            .result_type_fn = aggregate.countRowsResultType,
        },
        .exec = aggregate.countRowsKernel,
        .aggregate_lifecycle = .{
            .init = aggregate.countRowsInit,
            .update = aggregate.countRowsUpdate,
            .merge = aggregate.countRowsMerge,
            .finalize = aggregate.countRowsFinalize,
            .deinit = aggregate.countRowsDeinit,
        },
    });
}

pub fn registerCompatKernels(registry: *compute.FunctionRegistry) compute.KernelError!void {
    return registerBaseKernels(registry);
}
