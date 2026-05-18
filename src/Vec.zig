const std = @import("std");
const GpuAllocator = @import("GpuAllocator.zig");
const GpuBuffer = @import("GpuBuffer.zig");
const GpuDevice = @import("GpuDevice.zig");
const GpuProcess = @import("GpuProcess.zig");

const Vec = @This();

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

    try process.run(gloc, f32, self.buf, other.buf, result.buf);
    return result;
}

// Changed: gloc is passed by value instead of *GpuAllocator
pub fn read(self: Vec, alloc: std.mem.Allocator) ![]f16 {
    return self.buf.read(alloc, f16);
}
