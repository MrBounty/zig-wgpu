const std = @import("std");
const GpuDevice = @import("GpuDevice.zig");
const GpuAllocator = @import("GpuAllocator.zig");
const GpuTextureFormat = @import("lib.zig").GpuTextureFormat;
const c = @import("utils.zig").c;

device: GpuDevice,
tracked_buffers: std.AutoHashMap(c.WGPUBuffer, c.WGPUBufferDescriptor),
tracked_textures: std.AutoHashMap(c.WGPUTexture, c.WGPUTextureDescriptor),
tracked_views: std.AutoHashMap(c.WGPUTextureView, c.WGPUTextureViewDescriptor),
allocated_vram_bytes: u64 = 0,

pub fn init(cpu_allocator: std.mem.Allocator, device: GpuDevice) @This() {
    return .{
        .device = device,
        .tracked_buffers = .init(cpu_allocator),
        .tracked_textures = .init(cpu_allocator),
        .tracked_views = .init(cpu_allocator),
    };
}

pub fn deinit(self: *@This()) void {
    var it_buffer = self.tracked_buffers.keyIterator();
    while (it_buffer.next()) |buf_ptr| {
        c.wgpuBufferDestroy(buf_ptr.*);
        c.wgpuBufferRelease(buf_ptr.*);
    }
    self.tracked_buffers.deinit();

    var it_texture = self.tracked_textures.keyIterator();
    while (it_texture.next()) |tex_ptr|
        c.wgpuTextureRelease(tex_ptr.*);
    self.tracked_textures.deinit();

    var it_view = self.tracked_views.keyIterator();
    while (it_view.next()) |view_ptr|
        c.wgpuTextureViewRelease(view_ptr.*);
    self.tracked_views.deinit();
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
        },
    };
}

fn allocBuffer(ctx: *anyopaque, desc: c.WGPUBufferDescriptor) anyerror!c.WGPUBuffer {
    const self: *@This() = @ptrCast(@alignCast(ctx));

    if (desc.size > self.device.limits.maxBufferSize)
        return error.SingleBufferExceedsLimit;

    if (desc.size + self.allocated_vram_bytes > self.device.config.vram_bytes_limit)
        return error.ExceedsVramBudget;

    const buf = c.wgpuDeviceCreateBuffer(self.device.device, &desc) orelse return error.BufferAlloc;
    errdefer {
        c.wgpuBufferDestroy(buf);
        c.wgpuBufferRelease(buf);
    }

    try self.tracked_buffers.put(buf, desc);
    self.allocated_vram_bytes += desc.size;
    return buf;
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

    const format: GpuTextureFormat = @enumFromInt(desc.format);
    const bytes_size = desc.size.width * desc.size.height * format.bytesPerPixel();
    if (bytes_size > self.device.limits.maxBufferSize)
        return error.SingleBufferExceedsLimit;

    if (bytes_size + self.allocated_vram_bytes > self.device.config.vram_bytes_limit)
        return error.ExceedsVramBudget;

    const texture = c.wgpuDeviceCreateTexture(self.device.device, &desc) orelse return error.Texture;

    try self.tracked_textures.put(texture, desc);
    self.allocated_vram_bytes += bytes_size;
    return texture;
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
    const view = c.wgpuTextureCreateView(texture, &desc) orelse return error.View;
    try self.tracked_views.put(view, desc);
    return view;
}

fn freeTextureView(ctx: *anyopaque, raw: c.WGPUTextureView) void {
    const self: *@This() = @ptrCast(@alignCast(ctx));
    if (self.tracked_views.remove(raw))
        c.wgpuTextureViewRelease(raw);
}
