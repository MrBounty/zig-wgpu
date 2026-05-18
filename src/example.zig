/// This is a fully self contained example.
/// It set a simple f16 Vector and do a add operation on it
const std = @import("std");
const GpuDevice = @import("GpuDevice.zig");
const GpuArena = @import("GpuArena.zig");
const GpuAllocator = @import("GpuAllocator.zig");
const GpuBuffer = @import("GpuBuffer.zig");
const GpuProcess = @import("GpuProcess.zig");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    // Open GPU Device
    const device = try GpuDevice.init(.{});
    defer device.deinit();

    // Create a GPU Arena to hold GPU memory
    var grena = GpuArena.init(allocator, device);
    defer grena.deinit();
    const gloc = grena.gpuAllocator();

    // Create a GPU process that load the pipeline/shader
    const add = try GpuProcess.init(device, @embedFile("shaders/add.wgsl"));
    defer add.deinit();

    // Allocate CPU memory
    const data_a = try allocator.alloc(f16, 16);
    defer allocator.free(data_a);
    const data_b = try allocator.alloc(f16, 16);
    defer allocator.free(data_b);

    for (0..16) |i| {
        data_a[i] = @floatFromInt(i);
        data_b[i] = @floatFromInt(16 - 1 - i);
    }

    // Allocate GPU memory (Vec.deinit isn't necessary because grena will do it when deinit)
    const a = try Vec.initZero(gloc, 16);
    const b = try Vec.initZero(gloc, 16);

    // Load CPU -> GPU
    try a.load(data_a);
    try b.load(data_b);

    // Run GPU Pipeline
    const sum = try a.run(gloc, b, add);

    // Read GPU -> CPU
    const out = try sum.read(allocator);
    defer allocator.free(out);

    std.debug.print("{any}\n", .{out});
}

/// Minimal implementation of a f16 Vector
const Vec = struct {
    buf: GpuBuffer,
    len: usize,

    pub fn initZero(gloc: GpuAllocator, len: usize) !Vec {
        return .{
            .buf = try GpuBuffer.init(
                gloc,
                len * @sizeOf(f16),
                .initMany(&.{ .Storage, .CopyDst, .CopySrc }),
            ),
            .len = len,
        };
    }

    pub fn deinit(self: Vec) void {
        self.buf.deinit();
    }

    pub fn load(self: Vec, data: []const f16) !void {
        try self.buf.load(f16, data);
    }

    pub fn read(self: Vec, alloc: std.mem.Allocator) ![]f16 {
        return self.buf.read(alloc, f16);
    }

    pub fn run(self: Vec, gloc: GpuAllocator, other: Vec, process: GpuProcess) !Vec {
        std.debug.assert(self.len == other.len);

        const result = try Vec.initZero(gloc, self.len);
        errdefer result.deinit();

        try process.run(gloc, f16, self.buf, other.buf, result.buf);
        return result;
    }
};
