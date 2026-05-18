const std = @import("std");
const GpuDevice = @import("GpuDevice.zig");
const GpuArena = @import("GpuArena.zig");
const GpuAllocator = @import("GpuAllocator.zig");
const GpuBuffer = @import("GpuBuffer.zig");
const GpuProcess = @import("GpuProcess.zig");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    const device = try GpuDevice.init(.{});
    defer device.deinit();

    var grena = GpuArena.init(allocator, device);
    defer grena.deinit();

    const gloc = grena.gpuAllocator();

    const add = try GpuProcess.init(device, @embedFile("shaders/add.wgsl"));
    defer add.deinit();

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

    const sum = try a.run(gloc, b, add);
    // Don't need `sum.deinit()` because grena will deallocate everything when deinit

    std.debug.print("Bytes used: {d} (3 * {d})\n", .{ grena.allocated_vram_bytes, a.byteSize() });

    const out = try sum.read(allocator);
    defer allocator.free(out);

    std.debug.print("{any}\n", .{out});
}

/// Minimal implementation of a f16 Vector
const Vec = struct {
    buf: GpuBuffer,
    len: usize,

    // Changed: gloc is passed by value (const)
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

    // Changed: gloc is passed by value
    pub fn initLoad(gloc: GpuAllocator, data: []const f16) !Vec {
        var self = try initZero(gloc, data.len);
        try self.load(data); // Direct access via the interface copy
        return self;
    }

    pub fn deinit(self: Vec) void {
        self.buf.deinit();
    }

    /// CPU to GPU.
    pub fn load(self: Vec, data: []const f16) !void {
        try self.buf.load(f16, data);
    }

    pub fn byteSize(self: Vec) u64 {
        return @as(u64, self.len) * @sizeOf(f16);
    }

    // Changed: gloc is passed by value instead of *GpuAllocator
    pub fn run(self: Vec, gloc: GpuAllocator, other: Vec, process: GpuProcess) !Vec {
        std.debug.assert(self.len == other.len);

        const result = try Vec.initZero(gloc, self.len);
        errdefer result.deinit();

        try process.run(gloc, f16, self.buf, other.buf, result.buf);
        return result;
    }

    // Changed: gloc is passed by value instead of *GpuAllocator
    pub fn read(self: Vec, alloc: std.mem.Allocator) ![]f16 {
        return self.buf.read(alloc, f16);
    }
};
