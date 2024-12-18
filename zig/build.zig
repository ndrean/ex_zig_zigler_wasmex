const std = @import("std");

// pub fn build(b: *std.Build) void {
//     // the `.os_tag` is `.freestanding` when used in the browser
//     const target = b.resolveTargetQuery(.{
//         .cpu_arch = .wasm32,
//         .os_tag = .wasi,
//         .cpu_features_add = std.Target.wasm.featureSet(&.{ .atomics, .bulk_memory }), // if not single_threaded
//     });

//     // const target = b.standardTargetOptions(.{});
//     // const optimize = b.standardOptimizeOption(.{ .preferred_optimize_mode = .ReleaseSafe });

//     const exe = b.addExecutable(.{
//         .name = "ga_thread",
//         .root_source_file = b.path("ga_thread.zig"),
//         .target = target,
//         .optimize = .ReleaseFast,
//         .single_threaded = false, // ?? <------
//     });
//     exe.entry = .disabled;
//     exe.rdynamic = true;
//     exe.import_memory = false;
//     exe.export_memory = true;
//     exe.shared_memory = true; // ?? <-----

//     exe.initial_memory = 512 * std.wasm.page_size; // 512 pages of 64KiB
//     b.installArtifact(exe);
// }

pub fn build(b: *std.Build) void {
    // the `.os_tag` is `.freestanding` when used in the browser
    const target = b.resolveTargetQuery(.{
        .cpu_arch = .wasm32,
        .os_tag = .wasi,
    });

    // const target = b.standardTargetOptions(.{});
    // const optimize = b.standardOptimizeOption(.{ .preferred_optimize_mode = .ReleaseSafe });

    const exe = b.addExecutable(.{
        .name = "ga",
        .root_source_file = b.path("ga_wasm.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    exe.entry = .disabled;
    exe.rdynamic = true;
    // exe.import_memory = true;
    // exe.export_memory = true;

    exe.initial_memory = 512 * std.wasm.page_size; // 512 pages of 64KiB
    b.installArtifact(exe);
}
