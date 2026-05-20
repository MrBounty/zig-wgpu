const std = @import("std");
const GpuDevice = @import("GpuDevice.zig");
const GpuAllocator = @import("GpuAllocator.zig");
const GpuTextureFormat = @import("lib.zig").GpuTextureFormat;
const c = @import("utils.zig").c;

device: GpuDevice,
tracked_buffers: std.AutoHashMap(c.WGPUBuffer, c.WGPUBufferDescriptor),
tracked_textures: std.AutoHashMap(c.WGPUTexture, c.WGPUTextureDescriptor),
tracked_views: std.AutoHashMap(c.WGPUTextureView, c.WGPUTextureViewDescriptor),
tracked_renders: std.AutoHashMap(c.WGPURenderPipeline, c.WGPURenderPipelineDescriptor),
tracked_computes: std.AutoHashMap(c.WGPUComputePipeline, c.WGPUComputePipelineDescriptor),
allocated_vram_bytes: u64 = 0,

pub fn init(cpu_allocator: std.mem.Allocator, device: GpuDevice) @This() {
    return .{
        .device = device,
        .tracked_buffers = .init(cpu_allocator),
        .tracked_textures = .init(cpu_allocator),
        .tracked_views = .init(cpu_allocator),
        .tracked_computes = .init(cpu_allocator),
        .tracked_renders = .init(cpu_allocator),
    };
}

pub fn deinit(self: *@This()) void {
    const gloc = self.gpuAllocator();

    var it_buffer = self.tracked_buffers.keyIterator();
    while (it_buffer.next()) |buf_ptr|
        gloc.freeBuffer(buf_ptr.*);
    self.tracked_buffers.deinit();

    var it_tex = self.tracked_textures.keyIterator();
    while (it_tex.next()) |buf_ptr|
        gloc.freeTexture(buf_ptr.*);
    self.tracked_textures.deinit();

    var it_view = self.tracked_views.keyIterator();
    while (it_view.next()) |buf_ptr|
        gloc.freeTextureView(buf_ptr.*);
    self.tracked_views.deinit();

    var it_render = self.tracked_renders.keyIterator();
    while (it_render.next()) |buf_ptr|
        gloc.freeRenderPipeline(buf_ptr.*);
    self.tracked_renders.deinit();

    var it_compute = self.tracked_computes.keyIterator();
    while (it_compute.next()) |buf_ptr|
        gloc.freeComputePipeline(buf_ptr.*);
    self.tracked_computes.deinit();
}

/// Returns the type-erased immutable interface wrapper
pub fn gpuAllocator(self: *@This()) GpuAllocator {
    return .{
        .device = self.device,
        .ptr = self,
        .vtable = &.{
            .allocBuffer = allocBuffer,
            .freeBuffer = freeBuffer,
            .allocTexture = allocTexture,
            .freeTexture = freeTexture,
            .allocTextureView = allocTextureView,
            .freeTextureView = freeTextureView,
            .allocRenderPipeline = allocRenderPipeline,
            .freeRenderPipeline = freeRenderPipeline,
            .allocComputePipeline = allocComputePipeline,
            .freeComputePipeline = freeComputePipeline,
        },
    };
}

// NOTE: I use ensureTotalCapacity so I know that try self.tracked_x.put will not fail!
// Like that I dont have to use errdefer to release what I just allocated in VRAM

fn allocBuffer(ctx: *anyopaque, desc: c.WGPUBufferDescriptor) anyerror!c.WGPUBuffer {
    const self: *@This() = @ptrCast(@alignCast(ctx));
    try self.tracked_buffers.ensureTotalCapacity(self.tracked_buffers.count() + 1);

    if (desc.size > self.device.limits.maxBufferSize)
        return error.SingleBufferExceedsLimit;

    if (desc.size + self.allocated_vram_bytes > self.device.config.vram_bytes_limit)
        return error.ExceedsVramBudget;

    const raw = c.wgpuDeviceCreateBuffer(self.device.device, &desc) orelse return error.BufferAlloc;

    self.tracked_buffers.putAssumeCapacity(raw, desc);
    self.allocated_vram_bytes += desc.size;
    return raw;
}

fn freeBuffer(ctx: *anyopaque, raw: c.WGPUBuffer) void {
    const self: *@This() = @ptrCast(@alignCast(ctx));

    if (self.tracked_buffers.fetchRemove(raw)) |kv| {
        c.wgpuBufferDestroy(raw);
        c.wgpuBufferRelease(raw);
        self.allocated_vram_bytes -= kv.value.size;
    }
}

fn allocTexture(ctx: *anyopaque, desc: c.WGPUTextureDescriptor) anyerror!c.WGPUTexture {
    const self: *@This() = @ptrCast(@alignCast(ctx));
    try self.tracked_textures.ensureTotalCapacity(self.tracked_textures.count() + 1);

    const format: GpuTextureFormat = @enumFromInt(desc.format);
    const bytes_size = desc.size.width * desc.size.height * format.bytesPerPixel();
    if (bytes_size > self.device.limits.maxBufferSize)
        return error.SingleBufferExceedsLimit;

    if (bytes_size + self.allocated_vram_bytes > self.device.config.vram_bytes_limit)
        return error.ExceedsVramBudget;

    const raw = c.wgpuDeviceCreateTexture(self.device.device, &desc) orelse return error.Texture;

    self.tracked_textures.putAssumeCapacity(raw, desc);
    self.allocated_vram_bytes += bytes_size;
    return raw;
}

fn freeTexture(ctx: *anyopaque, raw: c.WGPUTexture) void {
    const self: *@This() = @ptrCast(@alignCast(ctx));

    if (self.tracked_textures.fetchRemove(raw)) |kv| {
        c.wgpuTextureRelease(raw);

        const desc = kv.value;
        const format: GpuTextureFormat = @enumFromInt(desc.format);
        const bytes_size = desc.size.width * desc.size.height * format.bytesPerPixel();
        self.allocated_vram_bytes -= bytes_size;
    }
}

fn allocTextureView(ctx: *anyopaque, texture: c.WGPUTexture, desc: c.WGPUTextureViewDescriptor) anyerror!c.WGPUTextureView {
    const self: *@This() = @ptrCast(@alignCast(ctx));
    try self.tracked_views.ensureTotalCapacity(self.tracked_views.count() + 1);
    const raw = c.wgpuTextureCreateView(texture, &desc) orelse return error.View;
    self.tracked_views.putAssumeCapacity(raw, desc);
    return raw;
}

fn freeTextureView(ctx: *anyopaque, raw: c.WGPUTextureView) void {
    const self: *@This() = @ptrCast(@alignCast(ctx));
    if (self.tracked_views.remove(raw))
        c.wgpuTextureViewRelease(raw);
}

fn allocRenderPipeline(ctx: *anyopaque, desc: c.WGPURenderPipelineDescriptor) anyerror!c.WGPURenderPipeline {
    const self: *@This() = @ptrCast(@alignCast(ctx));
    try self.tracked_renders.ensureTotalCapacity(self.tracked_renders.count() + 1);
    const raw = c.wgpuDeviceCreateRenderPipeline(self.device.device, &desc) orelse return error.Pipeline;
    self.tracked_renders.putAssumeCapacity(raw, desc);
    return raw;
}

fn freeRenderPipeline(ctx: *anyopaque, raw: c.WGPURenderPipeline) void {
    const self: *@This() = @ptrCast(@alignCast(ctx));
    if (self.tracked_renders.remove(raw))
        c.wgpuRenderPipelineRelease(raw);
}

fn allocComputePipeline(ctx: *anyopaque, desc: c.WGPUComputePipelineDescriptor) anyerror!c.WGPUComputePipeline {
    const self: *@This() = @ptrCast(@alignCast(ctx));
    try self.tracked_computes.ensureTotalCapacity(self.tracked_computes.count() + 1);
    const raw = c.wgpuDeviceCreateComputePipeline(self.device.device, &desc) orelse return error.Pipeline;
    self.tracked_computes.putAssumeCapacity(raw, desc);
    return raw;
}

fn freeComputePipeline(ctx: *anyopaque, raw: c.WGPUComputePipeline) void {
    const self: *@This() = @ptrCast(@alignCast(ctx));
    if (self.tracked_computes.remove(raw))
        c.wgpuComputePipelineRelease(raw);
}
