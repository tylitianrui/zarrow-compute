const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const zarrow_dep = b.dependency("zarrow", .{
        .target = target,
        .optimize = optimize,
    });
    const zarrow_core_mod = zarrow_dep.module("zarrow-core");

    const compute_mod = b.addModule("zarrow_compute", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    compute_mod.addImport("zarrow-core", zarrow_core_mod);

    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_mod.addImport("zarrow-core", zarrow_core_mod);
    const tests = b.addTest(.{ .root_module = test_mod });
    const run_tests = b.addRunArtifact(tests);

    const test_step = b.step("test", "Run zarrow-compute tests");
    test_step.dependOn(&run_tests.step);

    const example_mod = b.createModule(.{
        .root_source_file = b.path("examples/basic_compute.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zarrow_compute", .module = compute_mod },
            .{ .name = "zarrow-core", .module = zarrow_core_mod },
        },
    });
    const example = b.addExecutable(.{
        .name = "example-basic-compute",
        .root_module = example_mod,
    });
    const run_example = b.addRunArtifact(example);
    if (b.args) |args| run_example.addArgs(args);

    const example_step = b.step("example-basic", "Run compute example");
    example_step.dependOn(&run_example.step);

    const run_step = b.step("run", "Run compute example");
    run_step.dependOn(&run_example.step);
}
