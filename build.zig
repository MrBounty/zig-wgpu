const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mod = b.addModule("zig-wgpu", .{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
    });

    mod.addIncludePath(b.path("libs/wgpu-native/include"));
    mod.addLibraryPath(b.path("libs/wgpu-native/lib"));
    mod.addObjectFile(b.path("libs/wgpu-native/lib/libwgpu_native.a"));

    // Platform-specific system frameworks needed by wgpu-native
    const t = target.result;
    if (t.os.tag == .macos) {
        mod.linkFramework("Metal", .{});
        mod.linkFramework("QuartzCore", .{});
        mod.linkFramework("Foundation", .{});
        mod.linkFramework("CoreGraphics", .{});
    } else if (t.os.tag == .windows) {
        mod.linkSystemLibrary("d3d12", .{});
        mod.linkSystemLibrary("dxgi", .{});
        mod.linkSystemLibrary("user32", .{});
    } else {
        mod.linkSystemLibrary("vulkan", .{});
        mod.linkSystemLibrary("gcc_s", .{});
    }

    var threaded: std.Io.Threaded = .init_single_threaded;
    const io = threaded.io();

    var buf: [1024]u8 = undefined;
    const exemples = try std.Io.Dir.cwd().openDir(io, "examples", .{ .access_sub_paths = false, .iterate = true });
    var iter = exemples.iterate();
    while (try iter.next(io)) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.eql(u8, entry.name[entry.name.len - 4 ..], ".zig")) continue;

        const exe = b.addExecutable(.{
            .name = entry.name[0 .. entry.name.len - 4],
            .root_module = b.createModule(.{
                .root_source_file = b.path(try std.fmt.bufPrint(&buf, "examples/{s}", .{entry.name})),
                .target = target,
                .optimize = optimize,
                .imports = &.{},
            }),
        });
        exe.root_module.addImport("gpu", mod);

        b.installArtifact(exe);

        const run_step = b.step(entry.name[0 .. entry.name.len - 4], try std.fmt.bufPrint(&buf, "Run {s} demo", .{entry.name}));
        const run_cmd = b.addRunArtifact(exe);
        run_step.dependOn(&run_cmd.step);
    }
}
