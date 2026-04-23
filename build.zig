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

    const examples_step = b.step("examples", "Run all compute examples");
    var basic_example_run_step: ?*std.Build.Step = null;

    var examples_dir = std.fs.cwd().openDir("examples", .{ .iterate = true }) catch |err| {
        std.log.err("failed to open examples directory: {s}", .{@errorName(err)});
        return;
    };
    defer examples_dir.close();

    var iter = examples_dir.iterate();
    while (iter.next() catch |err| {
        std.log.err("failed to iterate examples directory: {s}", .{@errorName(err)});
        return;
    }) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".zig")) continue;

        const source_rel = b.fmt("examples/{s}", .{entry.name});
        const source_stem = std.fs.path.stem(entry.name);
        const step_name = b.fmt("example-{s}", .{source_stem});
        const exe_name = b.fmt("example-{s}", .{source_stem});
        const step_desc = b.fmt("Run {s}", .{source_rel});

        const example_mod = b.createModule(.{
            .root_source_file = b.path(source_rel),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zarrow_compute", .module = compute_mod },
                .{ .name = "zarrow-core", .module = zarrow_core_mod },
            },
        });
        const example = b.addExecutable(.{
            .name = exe_name,
            .root_module = example_mod,
        });
        const run_example = b.addRunArtifact(example);
        if (b.args) |args| run_example.addArgs(args);

        const single_example_step = b.step(step_name, step_desc);
        single_example_step.dependOn(&run_example.step);
        examples_step.dependOn(&run_example.step);

        if (std.mem.eql(u8, entry.name, "basic_compute.zig")) {
            basic_example_run_step = &run_example.step;
        }
    }

    if (basic_example_run_step) |step| {
        const basic_step = b.step("example-basic", "Run compute example");
        basic_step.dependOn(step);
    }

    const run_step = b.step("run", "Run all compute examples");
    run_step.dependOn(examples_step);
}
