// GpuAllocator.zig
const std = @import("std");
const GpuDevice = @import("GpuDevice.zig");
const c = @import("utils.zig").c;

const GpuAllocator = @This();

/// The function definitions our underlying implementations must satisfy
pub const VTable = struct {
    alloc: *const fn (ctx: *anyopaque, bytes: u64, usage: c.WGPUBufferUsage) anyerror!c.WGPUBuffer,
    free: *const fn (ctx: *anyopaque, buf_raw: c.WGPUBuffer, size: u64) void,
};

device: GpuDevice,
ptr: *anyopaque,
vtable: *const VTable,

pub fn allocBuffer(self: GpuAllocator, bytes: u64, usage: c.WGPUBufferUsage) !c.WGPUBuffer {
    return self.vtable.alloc(self.ptr, bytes, usage);
}

pub fn freeBuffer(self: GpuAllocator, buf_raw: c.WGPUBuffer, size: u64) void {
    self.vtable.free(self.ptr, buf_raw, size);
}
