// GpuArena.zig
const std = @import("std");
const GpuDevice = @import("GpuDevice.zig");
const GpuAllocator = @import("GpuAllocator.zig");
const c = @import("utils.zig").c;

const GpuArena = @This();

device: GpuDevice,
tracked_buffers: std.AutoHashMap(c.WGPUBuffer, void),
allocated_vram_bytes: u64 = 0,

pub fn init(cpu_allocator: std.mem.Allocator, device: GpuDevice) GpuArena {
    return .{
        .device = device,
        .tracked_buffers = .init(cpu_allocator),
    };
}

pub fn deinit(self: *GpuArena) void {
    var it = self.tracked_buffers.keyIterator();
    while (it.next()) |buf_ptr| {
        c.wgpuBufferDestroy(buf_ptr.*);
        c.wgpuBufferRelease(buf_ptr.*);
    }
    self.tracked_buffers.deinit();
}

/// Returns the type-erased immutable interface wrapper
pub fn gpuAllocator(self: *GpuArena) GpuAllocator {
    return .{
        .device = self.device,
        .ptr = self,
        .vtable = &.{
            .alloc = alloc,
            .free = free,
        },
    };
}

fn alloc(ctx: *anyopaque, bytes: u64, usage: c.WGPUBufferUsage) anyerror!c.WGPUBuffer {
    const self: *GpuArena = @ptrCast(@alignCast(ctx));

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

fn free(ctx: *anyopaque, buf_raw: c.WGPUBuffer, size: u64) void {
    const self: *GpuArena = @ptrCast(@alignCast(ctx));

    if (self.tracked_buffers.remove(buf_raw)) {
        c.wgpuBufferDestroy(buf_raw);
        c.wgpuBufferRelease(buf_raw);
        self.allocated_vram_bytes -= size;
    }
}
