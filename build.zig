// build.zig
// zig build run
//
// Expects wgpu-native pre-built in libs/wgpu-native/:
//   include/wgpu.h
//   lib/libwgpu_native.a   (or .so / .dylib / .dll)
//
// Download release: https://github.com/gfx-rs/wgpu-native/releases
// Pick the archive matching your OS/arch, e.g.:
//   wgpu-linux-x86_64-release.zip     → libwgpu_native.a  + wgpu.h
//   wgpu-macos-aarch64-release.zip
//   wgpu-windows-x86_64-msvc-release.zip

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // 1. Define the module so other projects can import it
    _ = b.addModule("zig-wgpu", .{
        .root_source_file = b.path("src/lib.zig"),
    });

    const exe = b.addExecutable(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/bench.zig"),
            .link_libc = true,
            .target = target,
            .optimize = optimize,
        }),
        .name = "bench",
    });

    // wgpu-native headers + pre-built static library
    exe.root_module.addIncludePath(b.path("libs/wgpu-native/include"));
    exe.root_module.addLibraryPath(b.path("libs/wgpu-native/lib"));
    exe.root_module.addObjectFile(b.path("libs/wgpu-native/lib/libwgpu_native.a"));

    // Platform-specific system frameworks needed by wgpu-native
    const t = target.result;
    if (t.os.tag == .macos) {
        exe.root_module.linkFramework("Metal", .{});
        exe.root_module.linkFramework("QuartzCore", .{});
        exe.root_module.linkFramework("Foundation", .{});
        exe.root_module.linkFramework("CoreGraphics", .{});
    } else if (t.os.tag == .windows) {
        exe.root_module.linkSystemLibrary("d3d12", .{});
        exe.root_module.linkSystemLibrary("dxgi", .{});
        exe.root_module.linkSystemLibrary("user32", .{});
    } else {
        exe.root_module.linkSystemLibrary("vulkan", .{});
        exe.root_module.linkSystemLibrary("gcc_s", .{});
    }

    b.installArtifact(exe);

    const run = b.addRunArtifact(exe);
    run.step.dependOn(b.getInstallStep());
    b.step("bench", "Benchmark a simple add vector.").dependOn(&run.step);

    const exe_examp = b.addExecutable(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/example.zig"),
            .link_libc = true,
            .target = target,
            .optimize = optimize,
        }),
        .name = "bench",
    });

    // wgpu-native headers + pre-built static library
    exe_examp.root_module.addIncludePath(b.path("libs/wgpu-native/include"));
    exe_examp.root_module.addLibraryPath(b.path("libs/wgpu-native/lib"));
    exe_examp.root_module.addObjectFile(b.path("libs/wgpu-native/lib/libwgpu_native.a"));

    if (t.os.tag == .macos) {
        exe_examp.root_module.linkFramework("Metal", .{});
        exe_examp.root_module.linkFramework("QuartzCore", .{});
        exe_examp.root_module.linkFramework("Foundation", .{});
        exe_examp.root_module.linkFramework("CoreGraphics", .{});
    } else if (t.os.tag == .windows) {
        exe_examp.root_module.linkSystemLibrary("d3d12", .{});
        exe_examp.root_module.linkSystemLibrary("dxgi", .{});
        exe_examp.root_module.linkSystemLibrary("user32", .{});
    } else {
        exe_examp.root_module.linkSystemLibrary("vulkan", .{});
        exe_examp.root_module.linkSystemLibrary("gcc_s", .{});
    }

    const examp = b.addRunArtifact(exe_examp);
    run.step.dependOn(b.getInstallStep());
    b.step("example", "Run basic example.").dependOn(&examp.step);
}
