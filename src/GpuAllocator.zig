const std = @import("std");
const GpuDevice = @import("GpuDevice.zig");
const c = @import("c.zig").c;

const GpuAllocator = @This();

device: GpuDevice,
cpu_allocator: std.mem.Allocator,
tracked_buffers: std.AutoHashMap(c.WGPUBuffer, void),

pub fn init(cpu_allocator: std.mem.Allocator, device: GpuDevice) !GpuAllocator {
    return .{
        .device = device,
        .cpu_allocator = cpu_allocator,
        .tracked_buffers = .init(cpu_allocator),
    };
}

pub fn deinit(self: *GpuAllocator) void {
    var it = self.tracked_buffers.keyIterator();
    while (it.next()) |buf_ptr| {
        const buf = buf_ptr.*;
        c.wgpuBufferDestroy(buf);
        c.wgpuBufferRelease(buf);
    }
    self.tracked_buffers.deinit();
}

pub fn registerBuffer(
    self: *GpuAllocator,
    bytes: u64,
    usage: c.WGPUBufferUsage,
) !c.WGPUBuffer {
    const buf = c.wgpuDeviceCreateBuffer(self.device.device, &.{
        .usage = usage,
        .size = bytes,
    }) orelse return error.BufferAlloc;

    try self.tracked_buffers.put(buf, {});
    return buf;
}

pub fn unregisterAndDestroyBuffer(self: *GpuAllocator, buf: c.WGPUBuffer) void {
    if (self.tracked_buffers.remove(buf)) {
        c.wgpuBufferDestroy(buf);
        c.wgpuBufferRelease(buf);
    }
}
