const std = @import("std");
const GpuDevice = @import("GpuDevice.zig");
const GpuAllocator = @import("GpuAllocator.zig");
const Mat = @import("Mat.zig");

pub fn main(init: std.process.Init) !void {
    const device = try GpuDevice.init();
    defer device.deinit();

    var gloc = try GpuAllocator.init(init.gpa, device);
    defer gloc.deinit();

    // Define the sizes you want to benchmark
    const sizes = [_]usize{
        1,
        1024,
        4096,
        16384,
        65536,
        262144,
        1024 * 1024,
        // 4 * 1024 * 1024,
        // 4 * 4 * 1024 * 1024,
        // 4 * 4 * 4 * 1024 * 1024,
        // 4 * 4 * 4 * 4 * 1024 * 1024,
        // 4 * 4 * 4 * 4 * 2 * 1024 * 1024,
    };

    // Print table header
    std.debug.print("\n| Element Count | Size (MB) | Time (ms) | Time (ns) |\n", .{});
    std.debug.print("|--------------:|----------:|----------:|----------:|\n", .{});

    const allocator = init.gpa;

    for (sizes) |size| {
        // Dynamically allocate buffers for the current size
        var data_a = try allocator.alloc(f32, size);
        defer allocator.free(data_a);
        var data_b = try allocator.alloc(f32, size);
        defer allocator.free(data_b);

        // Populate data
        for (0..size) |i| {
            data_a[i] = @floatFromInt(i);
            data_b[i] = @floatFromInt(size - 1 - i);
        }

        // Start timing the GPU operations
        const start = std.Io.Clock.awake.now(init.io);

        const a = try Mat.load(&gloc, data_a, size, 1);
        defer a.deinit();
        const b = try Mat.load(&gloc, data_b, size, 1);
        defer b.deinit();

        // a + b
        const sum = try a.add(&gloc, b);
        defer sum.deinit();

        const out = try sum.read(&gloc, allocator);
        defer allocator.free(out);

        const duration = start.durationTo(std.Io.Clock.awake.now(init.io));
        const ns = duration.toNanoseconds();
        const ms = duration.toMilliseconds();
        const mb = @as(f64, @floatFromInt(size * @sizeOf(f32))) / (1024.0 * 1024.0);

        // Print table row
        std.debug.print("| {d:12} | {d:8.2} | {d:9.3} | {d:9} |\n", .{ size, mb, ms, ns });
    }
}

fn printMat(data: []const f32, rows: u32, cols: u32) void {
    for (0..rows) |r| {
        for (0..cols) |col|
            std.debug.print("{d:6.0}", .{data[r * cols + col]});
        std.debug.print("\n", .{});
    }
}
