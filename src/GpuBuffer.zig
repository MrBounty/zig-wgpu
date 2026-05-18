const std = @import("std");
const c = @import("utils.zig").c;
const GpuAllocator = @import("GpuAllocator.zig");

raw: c.WGPUBuffer,
size: u64,
usage: c.WGPUBufferUsage,
gloc: GpuAllocator,

const BufferUsage = enum(u64) {
    None = 0x0000000000000000,
    MapRead = 0x0000000000000001,
    MapWrite = 0x0000000000000002,
    CopySrc = 0x0000000000000004,
    CopyDst = 0x0000000000000008,
    Index = 0x0000000000000010,
    Vertex = 0x0000000000000020,
    Uniform = 0x0000000000000040,
    Storage = 0x0000000000000080,
    Indirect = 0x0000000000000100,
    QueryResolve = 0x0000000000000200,
};

/// Allocates the underlying WebGPU handle and registers it to the parent GpuAllocator
pub fn init(gloc: GpuAllocator, T: type, len: usize, usage: std.EnumSet(BufferUsage)) !@This() {
    switch (@typeInfo(T)) {
        .int, .float => {},
        else => @compileError("GpuBuffer can only use int and float type"),
    }

    var use: u64 = 0;
    var iter = usage.iterator();
    while (iter.next()) |flag| use |= @intFromEnum(flag);

    const bytes = @sizeOf(T) * len;
    const raw_handle = try gloc.allocBuffer(bytes, use);

    return .{
        .raw = raw_handle,
        .size = bytes,
        .usage = use,
        .gloc = gloc,
    };
}

/// Unregisters from the parent GpuAllocator and cleanly destroys GPU resources
pub fn deinit(self: @This()) void {
    self.gloc.freeBuffer(self.raw, self.size);
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
