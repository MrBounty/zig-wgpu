const std = @import("std");
const c = @import("c.zig").c;
const GpuAllocator = @import("GpuAllocator.zig");

raw: c.WGPUBuffer,
size: u64,
usage: c.WGPUBufferUsage,
gloc: *GpuAllocator,

/// Allocates the underlying WebGPU handle and registers it to the parent GpuAllocator
pub fn init(gloc: *GpuAllocator, bytes: u64, usage: c.WGPUBufferUsage) !@This() {
    const raw_handle = try gloc.registerBuffer(bytes, usage);
    return .{
        .raw = raw_handle,
        .size = bytes,
        .usage = usage,
        .gloc = gloc,
    };
}

/// Unregisters from the parent GpuAllocator and cleanly destroys GPU resources
pub fn deinit(self: @This()) void {
    self.gloc.unregisterAndDestroyBuffer(self);
}

/// Native mapAsync wrapper
pub fn mapAsync(
    self: @This(),
    mode: c.WGPUMapMode,
    offset: u64,
    size: u64,
    callback_info: c.WGPUBufferMapCallbackInfo,
) void {
    _ = c.wgpuBufferMapAsync(self.raw, mode, offset, size, callback_info);
}

/// Native getConstMappedRange wrapper
pub fn getConstMappedRange(self: @This(), offset: u64, size: u64) ?*const anyopaque {
    return c.wgpuBufferGetConstMappedRange(self.raw, offset, size);
}

/// Native unmap wrapper
pub fn unmap(self: @This()) void {
    c.wgpuBufferUnmap(self.raw);
}
