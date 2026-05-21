const std = @import("std");
const GpuDevice = @import("GpuDevice.zig");
const GpuAllocator = @import("GpuAllocator.zig");
const GpuTextureFormat = @import("lib.zig").GpuTextureFormat;
const c = @import("utils.zig").c;
const viewStr = @import("utils.zig").viewStr;

child_allocator: GpuAllocator, // I use Zig naming child_allocator, but that should be a parent for me. Likely something idk
tracked_buffers: std.AutoHashMap(c.WGPUBuffer, c.WGPUBufferDescriptor),
tracked_textures: std.AutoHashMap(c.WGPUTexture, c.WGPUTextureDescriptor),
tracked_views: std.AutoHashMap(c.WGPUTextureView, c.WGPUTextureViewDescriptor),
tracked_renders: std.AutoHashMap(c.WGPURenderPipeline, c.WGPURenderPipelineDescriptor),
tracked_computes: std.AutoHashMap(c.WGPUComputePipeline, c.WGPUComputePipelineDescriptor),
allocated_vram_bytes: u64 = 0,

pub fn init(cpu_allocator: std.mem.Allocator, child_allocator: GpuAllocator) @This() {
    return .{
        .child_allocator = child_allocator,
        .tracked_buffers = .init(cpu_allocator),
        .tracked_textures = .init(cpu_allocator),
        .tracked_views = .init(cpu_allocator),
        .tracked_computes = .init(cpu_allocator),
        .tracked_renders = .init(cpu_allocator),
    };
}

pub fn deinit(self: *@This()) void {
    std.log.debug("Freeing GpuArenaAllocator (Used VRAM: {}/{} MB)", .{
        self.allocated_vram_bytes / 1024 / 1024,
        self.child_allocator.device.def.vram_bytes_limit / 1024 / 1024,
    });

    var it_buffer = self.tracked_buffers.keyIterator();
    while (it_buffer.next()) |buf_ptr|
        freeBuffer(self, buf_ptr.*);
    self.tracked_buffers.deinit();

    var it_tex = self.tracked_textures.keyIterator();
    while (it_tex.next()) |buf_ptr|
        freeTexture(self, buf_ptr.*);
    self.tracked_textures.deinit();

    var it_view = self.tracked_views.keyIterator();
    while (it_view.next()) |buf_ptr|
        freeTextureView(self, buf_ptr.*);
    self.tracked_views.deinit();

    var it_render = self.tracked_renders.keyIterator();
    while (it_render.next()) |buf_ptr|
        freeRenderPipeline(self, buf_ptr.*);
    self.tracked_renders.deinit();

    var it_compute = self.tracked_computes.keyIterator();
    while (it_compute.next()) |buf_ptr|
        freeComputePipeline(self, buf_ptr.*);
    self.tracked_computes.deinit();

    std.log.debug("Freed GpuArenaAllocator (Used VRAM: {}/{} MB)", .{
        self.allocated_vram_bytes / 1024 / 1024,
        self.child_allocator.device.def.vram_bytes_limit / 1024 / 1024,
    });
}

/// Returns the type-erased immutable interface wrapper
pub fn gpuAllocator(self: *@This()) GpuAllocator {
    return .{
        .device = self.child_allocator.device,
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
    const raw = try self.child_allocator.allocBuffer(desc);
    self.tracked_buffers.putAssumeCapacity(raw, desc);
    self.allocated_vram_bytes += desc.size;

    std.log.debug("Allocated Buffer '{s}': {d} B (Total VRAM: {}/{} MB)", .{
        viewStr(desc.label),
        desc.size,
        self.allocated_vram_bytes / 1024 / 1024,
        self.child_allocator.device.def.vram_bytes_limit / 1024 / 1024,
    });

    return raw;
}

fn freeBuffer(ctx: *anyopaque, raw: c.WGPUBuffer) void {
    const self: *@This() = @ptrCast(@alignCast(ctx));
    if (self.tracked_buffers.fetchRemove(raw)) |kv| {
        self.child_allocator.freeBuffer(raw);
        self.allocated_vram_bytes -= kv.value.size;

        std.log.debug("Freed Buffer '{s}' (Total VRAM: {}/{} MB)", .{
            viewStr(kv.value.label),
            self.allocated_vram_bytes / 1024 / 1024,
            self.child_allocator.device.def.vram_bytes_limit / 1024 / 1024,
        });
    }
}

fn allocTexture(ctx: *anyopaque, desc: c.WGPUTextureDescriptor) anyerror!c.WGPUTexture {
    const self: *@This() = @ptrCast(@alignCast(ctx));
    try self.tracked_textures.ensureTotalCapacity(self.tracked_textures.count() + 1);

    const format: GpuTextureFormat = @enumFromInt(desc.format);
    const bytes_size = desc.size.width * desc.size.height * format.bytesPerPixel();

    if (bytes_size + self.allocated_vram_bytes > self.child_allocator.device.def.vram_bytes_limit)
        return error.ExceedsVramBudget;

    const raw = try self.child_allocator.allocTexture(desc);

    self.tracked_textures.putAssumeCapacity(raw, desc);
    self.allocated_vram_bytes += bytes_size;

    std.log.debug("Allocated Texture '{s}': {d} B (Total VRAM: {}/{} MB)", .{
        viewStr(desc.label),
        bytes_size,
        self.allocated_vram_bytes / 1024 / 1024,
        self.child_allocator.device.def.vram_bytes_limit / 1024 / 1024,
    });

    return raw;
}

fn freeTexture(ctx: *anyopaque, raw: c.WGPUTexture) void {
    const self: *@This() = @ptrCast(@alignCast(ctx));

    if (self.tracked_textures.fetchRemove(raw)) |kv| {
        self.child_allocator.freeTexture(raw);

        const desc = kv.value;
        const format: GpuTextureFormat = @enumFromInt(desc.format);
        const bytes_size = desc.size.width * desc.size.height * format.bytesPerPixel();
        self.allocated_vram_bytes -= bytes_size;

        std.log.debug("Freed Texture '{s}' (Total VRAM: {}/{} MB)", .{
            viewStr(desc.label),
            self.allocated_vram_bytes / 1024 / 1024,
            self.child_allocator.device.def.vram_bytes_limit / 1024 / 1024,
        });
    }
}

fn allocTextureView(ctx: *anyopaque, texture: c.WGPUTexture, desc: c.WGPUTextureViewDescriptor) anyerror!c.WGPUTextureView {
    const self: *@This() = @ptrCast(@alignCast(ctx));
    try self.tracked_views.ensureTotalCapacity(self.tracked_views.count() + 1);
    const raw = try self.child_allocator.allocTextureView(texture, desc);
    self.tracked_views.putAssumeCapacity(raw, desc);
    std.log.debug("Allocated Texture View '{s}'", .{viewStr(desc.label)});
    return raw;
}

fn freeTextureView(ctx: *anyopaque, raw: c.WGPUTextureView) void {
    const self: *@This() = @ptrCast(@alignCast(ctx));
    if (self.tracked_views.fetchRemove(raw)) |kv| {
        self.child_allocator.freeTextureView(raw);
        const desc = kv.value;
        std.log.debug("Freed Texture View '{s}'", .{viewStr(desc.label)});
    }
}

fn allocRenderPipeline(ctx: *anyopaque, desc: c.WGPURenderPipelineDescriptor) anyerror!c.WGPURenderPipeline {
    const self: *@This() = @ptrCast(@alignCast(ctx));
    try self.tracked_renders.ensureTotalCapacity(self.tracked_renders.count() + 1);
    const raw = try self.child_allocator.allocRenderPipeline(desc);
    self.tracked_renders.putAssumeCapacity(raw, desc);
    std.log.debug("Allocated Render Pipeline '{s}'", .{viewStr(desc.label)});
    return raw;
}

fn freeRenderPipeline(ctx: *anyopaque, raw: c.WGPURenderPipeline) void {
    const self: *@This() = @ptrCast(@alignCast(ctx));
    if (self.tracked_renders.fetchRemove(raw)) |kv| {
        self.child_allocator.freeRenderPipeline(raw);
        const desc = kv.value;
        std.log.debug("Freed Render Pipeline '{s}'", .{viewStr(desc.label)});
    }
}

fn allocComputePipeline(ctx: *anyopaque, desc: c.WGPUComputePipelineDescriptor) anyerror!c.WGPUComputePipeline {
    const self: *@This() = @ptrCast(@alignCast(ctx));
    try self.tracked_computes.ensureTotalCapacity(self.tracked_computes.count() + 1);
    const raw = try self.child_allocator.allocComputePipeline(desc);
    self.tracked_computes.putAssumeCapacity(raw, desc);
    std.log.debug("Allocated Compute Pipeline '{s}'", .{viewStr(desc.label)});
    return raw;
}

fn freeComputePipeline(ctx: *anyopaque, raw: c.WGPUComputePipeline) void {
    const self: *@This() = @ptrCast(@alignCast(ctx));
    if (self.tracked_computes.fetchRemove(raw)) |kv| {
        self.child_allocator.freeComputePipeline(raw);
        const desc = kv.value;
        std.log.debug("Freed Compute Pipeline '{s}'", .{viewStr(desc.label)});
    }
}
