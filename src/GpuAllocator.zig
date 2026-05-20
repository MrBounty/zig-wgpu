const GpuDevice = @import("GpuDevice.zig");
const c = @import("utils.zig").c;

pub const VTable = struct {
    allocBuffer: *const fn (ctx: *anyopaque, desc: c.WGPUBufferDescriptor) anyerror!c.WGPUBuffer,
    freeBuffer: *const fn (ctx: *anyopaque, buf_raw: c.WGPUBuffer) void,
    allocTexture: *const fn (ctx: *anyopaque, desc: c.WGPUTextureDescriptor) anyerror!c.WGPUTexture,
    freeTexture: *const fn (ctx: *anyopaque, buf_raw: c.WGPUTexture) void,
    allocTextureView: *const fn (ctx: *anyopaque, texture: c.WGPUTexture, desc: c.WGPUTextureViewDescriptor) anyerror!c.WGPUTextureView,
    freeTextureView: *const fn (ctx: *anyopaque, buf_raw: c.WGPUTextureView) void,
    allocRenderPipeline: *const fn (ctx: *anyopaque, desc: c.WGPURenderPipelineDescriptor) anyerror!c.WGPURenderPipeline,
    freeRenderPipeline: *const fn (ctx: *anyopaque, buf_raw: c.WGPURenderPipeline) void,
    allocComputePipeline: *const fn (ctx: *anyopaque, desc: c.WGPUComputePipelineDescriptor) anyerror!c.WGPUComputePipeline,
    freeComputePipeline: *const fn (ctx: *anyopaque, buf_raw: c.WGPUComputePipeline) void,
};

device: GpuDevice,
ptr: *anyopaque,
vtable: *const VTable,

pub fn allocBuffer(self: @This(), desc: c.WGPUBufferDescriptor) !c.WGPUBuffer {
    return self.vtable.allocBuffer(self.ptr, desc);
}

pub fn freeBuffer(self: @This(), raw: c.WGPUBuffer) void {
    self.vtable.freeBuffer(self.ptr, raw);
}

pub fn allocTexture(self: @This(), desc: c.WGPUTextureDescriptor) !c.WGPUTexture {
    return self.vtable.allocTexture(self.ptr, desc);
}

pub fn freeTexture(self: @This(), raw: c.WGPUTexture) void {
    self.vtable.freeTexture(self.ptr, raw);
}

pub fn allocTextureView(self: @This(), texture: c.WGPUTexture, desc: c.WGPUTextureViewDescriptor) !c.WGPUTextureView {
    return self.vtable.allocTextureView(self.ptr, texture, desc);
}

pub fn freeTextureView(self: @This(), raw: c.WGPUTextureView) void {
    self.vtable.freeTextureView(self.ptr, raw);
}

pub fn allocRenderPipeline(self: @This(), desc: c.WGPURenderPipelineDescriptor) !c.WGPURenderPipeline {
    return self.vtable.allocRenderPipeline(self.ptr, desc);
}

pub fn freeRenderPipeline(self: @This(), raw: c.WGPURenderPipeline) void {
    self.vtable.freeRenderPipeline(self.ptr, raw);
}

pub fn allocComputePipeline(self: @This(), desc: c.WGPUComputePipelineDescriptor) !c.WGPUComputePipeline {
    return self.vtable.allocComputePipeline(self.ptr, desc);
}

pub fn freeComputePipeline(self: @This(), raw: c.WGPUComputePipeline) void {
    self.vtable.freeComputePipeline(self.ptr, raw);
}
