const std = @import("std");
const GpuDevice = @import("GpuDevice.zig");
const GpuAllocator = @import("GpuAllocator.zig");
const GpuTextureFormat = @import("lib.zig").GpuTextureFormat;
const c = @import("utils.zig").c;

device: GpuDevice,
tracked_buffers: std.AutoHashMap(c.WGPUBuffer, c.WGPUBufferDescriptor),
tracked_textures: std.AutoHashMap(c.WGPUTexture, c.WGPUTextureDescriptor),
allocated_vram_bytes: u64 = 0,

pub fn init(cpu_allocator: std.mem.Allocator, device: GpuDevice) @This() {
    return .{
        .device = device,
        .tracked_buffers = .init(cpu_allocator),
        .tracked_textures = .init(cpu_allocator),
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

fn freeBuffer(ctx: *anyopaque, buf_raw: c.WGPUBuffer) void {
    const self: *@This() = @ptrCast(@alignCast(ctx));

    if (self.tracked_buffers.fetchRemove(buf_raw)) |kv| {
        c.wgpuBufferDestroy(buf_raw);
        c.wgpuBufferRelease(buf_raw);
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

fn freeTexture(ctx: *anyopaque, texture_raw: c.WGPUTexture) void {
    const self: *@This() = @ptrCast(@alignCast(ctx));

    if (self.tracked_textures.fetchRemove(texture_raw)) |kv| {
        c.wgpuTextureRelease(texture_raw);

        const desc = kv.value;
        const format: GpuTextureFormat = @enumFromInt(desc.format);
        const bytes_size = desc.size.width * desc.size.height * format.bytesPerPixel();
        self.allocated_vram_bytes -= bytes_size;
    }
}
