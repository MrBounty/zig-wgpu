const std = @import("std");
const GpuAllocator = @import("GpuAllocator.zig");
const Mat = @import("Mat.zig");

pub fn main(init: std.process.Init) !void {
    var gloc = try GpuAllocator.init(init.gpa);
    defer gloc.deinit();

    // Input data: a[i] = i, b[i] = 15 - i  →  add should give all 15s
    var data_a: [16]f32 = undefined;
    var data_b: [16]f32 = undefined;
    for (0..16) |i| {
        data_a[i] = @floatFromInt(i);
        data_b[i] = @floatFromInt(15 - i);
    }

    const a = try Mat.load(&gloc, &data_a, 4, 4);
    defer a.deinit();
    const b = try Mat.load(&gloc, &data_b, 4, 4);
    defer b.deinit();

    // a + b
    const sum = try a.add(&gloc, b);
    defer sum.deinit();

    // sum * 2
    const scaled = try sum.scale(&gloc, 2.0);
    defer scaled.deinit();

    // Read back
    var out_sum: [16]f32 = undefined;
    var out_scaled: [16]f32 = undefined;
    try sum.read(&gloc, &out_sum);
    try scaled.read(&gloc, &out_scaled);

    // Print
    std.debug.print("\na + b  (expect all 15):\n", .{});
    printMat(&out_sum, 4, 4);

    std.debug.print("\n(a + b) * 2  (expect all 30):\n", .{});
    printMat(&out_scaled, 4, 4);
}

fn printMat(data: []const f32, rows: u32, cols: u32) void {
    for (0..rows) |r| {
        for (0..cols) |col|
            std.debug.print("{d:6.0}", .{data[r * cols + col]});
        std.debug.print("\n", .{});
    }
}
