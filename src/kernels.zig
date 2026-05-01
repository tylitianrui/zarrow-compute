const impl = @import("kernels/impl.zig");

pub const compute = impl.compute;
pub const registerBaseKernels = impl.registerBaseKernels;
pub const registerCompatKernels = impl.registerCompatKernels;

test {
    _ = @import("kernels/tests_registry.zig");
    _ = @import("kernels/tests_filter_dropnull.zig");
    _ = @import("kernels/tests_null_ifelse.zig");
    _ = @import("kernels/tests_conditional_variadic.zig");
    _ = @import("kernels/tests_misc.zig");
}
