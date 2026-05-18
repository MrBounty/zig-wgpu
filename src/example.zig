const std = @import("std");
const GpuDevice = @import("GpuDevice.zig");
const GpuAllocator = @import("GpuAllocator.zig");
const GpuArena = @import("GpuArena.zig");
const GpuPipeline = @import("GpuPipeline.zig");
const Vec = @import("Vec.zig");

const c = @import("utils.zig").c;

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    const device = try GpuDevice.init(.{});
    defer device.deinit();

    var grena = GpuArena.init(allocator, device);
    defer grena.deinit();

    const gloc = grena.gpuAllocator();

    const add_pip = try GpuPipeline.init(device, @embedFile("shaders/add.wgsl"));
    defer add_pip.deinit();

    const data_a = try allocator.alloc(f16, 16);
    defer allocator.free(data_a);
    const data_b = try allocator.alloc(f16, 16);
    defer allocator.free(data_b);

    for (0..16) |i| {
        data_a[i] = @floatFromInt(i);
        data_b[i] = @floatFromInt(16 - 1 - i);
    }

    const a = try Vec.initLoad(gloc, data_a);
    defer a.deinit();
    const b = try Vec.initLoad(gloc, data_b);
    defer b.deinit();

    const sum = try a.run(gloc, b, add_pip);
    // Don't need `sum.deinit()` because grena will deallocate everything when deinit

    std.debug.print("Bytes used: {d} (3 * {d})\n", .{ grena.allocated_vram_bytes, a.byteSize() });

    const out = try sum.read(gloc, allocator);
    defer allocator.free(out);

    std.debug.print("{any}\n", .{out});
}
