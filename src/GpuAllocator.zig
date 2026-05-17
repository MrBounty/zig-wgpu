const std = @import("std");
const GpuDevice = @import("GpuDevice.zig");
const GpuBuffer = @import("GpuBuffer.zig");
const c = @import("utils.zig").c;

device: GpuDevice,
tracked_buffers: std.AutoHashMap(c.WGPUBuffer, void),
allocated_vram_bytes: u64 = 0,

pub fn init(cpu_allocator: std.mem.Allocator, device: GpuDevice) !@This() {
    return .{
        .device = device,
        .tracked_buffers = .init(cpu_allocator),
    };
}

pub fn deinit(self: *@This()) void {
    var it = self.tracked_buffers.keyIterator();
    while (it.next()) |buf_ptr| {
        const buf = buf_ptr.*;
        c.wgpuBufferDestroy(buf);
        c.wgpuBufferRelease(buf);
    }
    self.tracked_buffers.deinit();
}

pub fn registerBuffer(
    self: *@This(),
    bytes: u64,
    usage: c.WGPUBufferUsage,
) !c.WGPUBuffer {
    if (bytes > self.device.limits.maxBufferSize)
        return error.SingleBufferExceedsLimit;

    if (bytes + self.allocated_vram_bytes > self.device.config.vram_bytes_limit)
        return error.ExceedsVramBudget;

    const buf = c.wgpuDeviceCreateBuffer(self.device.device, &.{
        .usage = usage,
        .size = bytes,
    }) orelse return error.BufferAlloc;
    errdefer {
        c.wgpuBufferDestroy(buf);
        c.wgpuBufferRelease(buf);
    }

    try self.tracked_buffers.put(buf, {});
    self.allocated_vram_bytes += bytes;
    return buf;
}

pub fn unregisterAndDestroyBuffer(self: *@This(), buf: GpuBuffer) void {
    if (self.tracked_buffers.remove(buf.raw)) {
        c.wgpuBufferDestroy(buf.raw);
        c.wgpuBufferRelease(buf.raw);
        self.allocated_vram_bytes -= buf.size;
    }
}
