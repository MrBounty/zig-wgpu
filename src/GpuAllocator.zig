const GpuDevice = @import("GpuDevice.zig");
const c = @import("utils.zig").c;

pub const VTable = struct {
    allocBuffer: *const fn (ctx: *anyopaque, desc: c.WGPUBufferDescriptor) anyerror!c.WGPUBuffer,
    freeBuffer: *const fn (ctx: *anyopaque, buf_raw: c.WGPUBuffer) void,
    allocTexture: *const fn (ctx: *anyopaque, desc: c.WGPUTextureDescriptor) anyerror!c.WGPUTexture,
    freeTexture: *const fn (ctx: *anyopaque, buf_raw: c.WGPUTexture) void,
};

device: GpuDevice,
ptr: *anyopaque,
vtable: *const VTable,

pub fn allocBuffer(self: @This(), desc: c.WGPUBufferDescriptor) !c.WGPUBuffer {
    return self.vtable.allocBuffer(self.ptr, desc);
}

pub fn freeBuffer(self: @This(), buf_raw: c.WGPUBuffer) void {
    self.vtable.freeBuffer(self.ptr, buf_raw);
}

pub fn allocTexture(self: @This(), desc: c.WGPUTextureDescriptor) !c.WGPUTexture {
    return self.vtable.allocTexture(self.ptr, desc);
}

pub fn freeTexture(self: @This(), buf_raw: c.WGPUTexture) void {
    self.vtable.freeTexture(self.ptr, buf_raw);
}
