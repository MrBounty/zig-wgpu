const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mod = b.addModule("zig-wgpu", .{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
    });

    const t = target.result;
    const arch_name = @tagName(t.cpu.arch);
    const os_name = @tagName(t.os.tag);

    // Windows uses .lib, Unix-like systems use .a
    const lib_filename = if (t.os.tag == .windows) "wgpu_native.lib" else "libwgpu_native.a";

    // Example: "libs/wgpu-native/x86_64-windows/wgpu_native.lib"
    const wgpu_lib_path = b.fmt("libs/wgpu-native/{s}-{s}/{s}", .{ arch_name, os_name, lib_filename });

    mod.addIncludePath(b.path("libs/wgpu-native/include"));
    mod.addObjectFile(b.path(wgpu_lib_path));

    // Platform-specific system frameworks needed by wgpu-native
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

    if (b.pkg_hash.len == 0) {
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
}
