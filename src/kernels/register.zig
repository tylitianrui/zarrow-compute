const common = @import("common.zig");
const arithmetic = @import("arithmetic.zig");
const filter = @import("filter.zig");
const nulls = @import("nulls.zig");
const conditionals = @import("conditionals.zig");
const selection = @import("selection.zig");
const compare = @import("compare.zig");
const logical = @import("logical.zig");
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

    try registry.registerVectorKernel("take", .{
        .signature = .{
            .arity = 2,
            .type_check = common.binaryTakeSupported,
            .options_check = common.onlyNoOptions,
            .result_type_fn = common.resultSameAsFirst,
        },
        .exec = selection.takeKernel,
    });

    try registry.registerVectorKernel("array_take", .{
        .signature = .{
            .arity = 2,
            .type_check = common.binaryTakeSupported,
            .options_check = common.onlyNoOptions,
            .result_type_fn = common.resultSameAsFirst,
        },
        .exec = selection.arrayTakeKernel,
    });

    try registry.registerVectorKernel("indices_nonzero", .{
        .signature = .{
            .arity = 1,
            .type_check = common.unaryIndicesNonZeroSupported,
            .options_check = common.onlyNoOptions,
            .result_type_fn = common.resultI64,
        },
        .exec = selection.indicesNonZeroKernel,
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

    try registry.registerVectorKernel("coalesce", .{
        .signature = .{
            .arity = 1,
            .variadic = true,
            .type_check = common.variadicCoalesceSupported,
            .options_check = common.onlyNoOptions,
            .result_type_fn = common.resultSameAsFirst,
        },
        .exec = conditionals.coalesceKernel,
    });

    try registry.registerVectorKernel("choose", .{
        .signature = .{
            .arity = 2,
            .variadic = true,
            .type_check = common.variadicChooseSupported,
            .options_check = common.onlyNoOptions,
            .result_type_fn = common.resultSameAsSecond,
        },
        .exec = conditionals.chooseKernel,
    });

    try registry.registerVectorKernel("case_when", .{
        .signature = .{
            .arity = 2,
            .variadic = true,
            .type_check = common.variadicCaseWhenSupported,
            .options_check = common.onlyNoOptions,
            .result_type_fn = common.resultSameAsSecond,
        },
        .exec = conditionals.caseWhenKernel,
    });

    try registry.registerVectorKernel("fill_null", .{
        .signature = .{
            .arity = 2,
            .type_check = common.binaryFillNullSupported,
            .options_check = common.onlyNoOptions,
            .result_type_fn = common.resultSameAsFirst,
        },
        .exec = selection.fillNullKernel,
    });

    try registry.registerVectorKernel("fill_null_forward", .{
        .signature = .{
            .arity = 1,
            .type_check = common.unarySupportedFilter,
            .options_check = common.onlyNoOptions,
            .result_type_fn = common.resultSameAsFirst,
        },
        .exec = selection.fillNullForwardKernel,
    });

    try registry.registerVectorKernel("fill_null_backward", .{
        .signature = .{
            .arity = 1,
            .type_check = common.unarySupportedFilter,
            .options_check = common.onlyNoOptions,
            .result_type_fn = common.resultSameAsFirst,
        },
        .exec = selection.fillNullBackwardKernel,
    });

    try registry.registerVectorKernel("equal", .{
        .signature = .{
            .arity = 2,
            .type_check = common.binaryInt64,
            .options_check = common.onlyNoOptions,
            .result_type_fn = common.resultBool,
        },
        .exec = compare.equalKernel,
    });

    try registry.registerVectorKernel("not_equal", .{
        .signature = .{
            .arity = 2,
            .type_check = common.binaryInt64,
            .options_check = common.onlyNoOptions,
            .result_type_fn = common.resultBool,
        },
        .exec = compare.notEqualKernel,
    });

    try registry.registerVectorKernel("less", .{
        .signature = .{
            .arity = 2,
            .type_check = common.binaryInt64,
            .options_check = common.onlyNoOptions,
            .result_type_fn = common.resultBool,
        },
        .exec = compare.lessKernel,
    });

    try registry.registerVectorKernel("less_equal", .{
        .signature = .{
            .arity = 2,
            .type_check = common.binaryInt64,
            .options_check = common.onlyNoOptions,
            .result_type_fn = common.resultBool,
        },
        .exec = compare.lessEqualKernel,
    });

    try registry.registerVectorKernel("greater", .{
        .signature = .{
            .arity = 2,
            .type_check = common.binaryInt64,
            .options_check = common.onlyNoOptions,
            .result_type_fn = common.resultBool,
        },
        .exec = compare.greaterKernel,
    });

    try registry.registerVectorKernel("greater_equal", .{
        .signature = .{
            .arity = 2,
            .type_check = common.binaryInt64,
            .options_check = common.onlyNoOptions,
            .result_type_fn = common.resultBool,
        },
        .exec = compare.greaterEqualKernel,
    });

    try registry.registerVectorKernel("invert", .{
        .signature = .{
            .arity = 1,
            .type_check = common.unaryBool,
            .options_check = common.onlyNoOptions,
            .result_type_fn = common.resultBool,
        },
        .exec = logical.invertKernel,
    });

    try registry.registerVectorKernel("and_", .{
        .signature = .{
            .arity = 2,
            .type_check = common.binaryBool,
            .options_check = common.onlyNoOptions,
            .result_type_fn = common.resultBool,
        },
        .exec = logical.andKernel,
    });

    try registry.registerVectorKernel("or_", .{
        .signature = .{
            .arity = 2,
            .type_check = common.binaryBool,
            .options_check = common.onlyNoOptions,
            .result_type_fn = common.resultBool,
        },
        .exec = logical.orKernel,
    });

    try registry.registerVectorKernel("and_kleene", .{
        .signature = .{
            .arity = 2,
            .type_check = common.binaryBool,
            .options_check = common.onlyNoOptions,
            .result_type_fn = common.resultBool,
        },
        .exec = logical.andKleeneKernel,
    });

    try registry.registerVectorKernel("or_kleene", .{
        .signature = .{
            .arity = 2,
            .type_check = common.binaryBool,
            .options_check = common.onlyNoOptions,
            .result_type_fn = common.resultBool,
        },
        .exec = logical.orKleeneKernel,
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

    try registry.registerVectorKernel("cast", .{
        .signature = .{
            .arity = 1,
            .type_check = common.unaryArrayLike,
            .options_check = common.onlyCastOptions,
            .result_type_fn = cast.castResultType,
        },
        .exec = cast.castKernel,
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

    try registry.registerAggregateKernel("count", .{
        .signature = .{
            .arity = 1,
            .type_check = common.unaryArrayLike,
            .options_check = common.onlyNoOptions,
            .result_type_fn = aggregate.countResultType,
        },
        .exec = aggregate.countKernel,
    });

    try registry.registerAggregateKernel("sum", .{
        .signature = .{
            .arity = 1,
            .type_check = common.unaryInt64ArrayLike,
            .options_check = common.onlyNoOptions,
            .result_type_fn = common.resultI64,
        },
        .exec = aggregate.sumKernel,
    });

    try registry.registerAggregateKernel("min", .{
        .signature = .{
            .arity = 1,
            .type_check = common.unaryInt64ArrayLike,
            .options_check = common.onlyNoOptions,
            .result_type_fn = common.resultI64,
        },
        .exec = aggregate.minKernel,
    });

    try registry.registerAggregateKernel("max", .{
        .signature = .{
            .arity = 1,
            .type_check = common.unaryInt64ArrayLike,
            .options_check = common.onlyNoOptions,
            .result_type_fn = common.resultI64,
        },
        .exec = aggregate.maxKernel,
    });

    try registry.registerAggregateKernel("mean", .{
        .signature = .{
            .arity = 1,
            .type_check = common.unaryInt64ArrayLike,
            .options_check = common.onlyNoOptions,
            .result_type_fn = aggregate.meanResultType,
        },
        .exec = aggregate.meanKernel,
    });
}

pub fn registerCompatKernels(registry: *compute.FunctionRegistry) compute.KernelError!void {
    return registerBaseKernels(registry);
}
