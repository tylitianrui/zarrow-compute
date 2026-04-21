const kernels = @import("kernels.zig");

pub const compute = kernels.compute;
pub const registerBaseKernels = kernels.registerBaseKernels;
pub const registerCompatKernels = kernels.registerCompatKernels;

test {
    _ = @import("kernels.zig");
}
