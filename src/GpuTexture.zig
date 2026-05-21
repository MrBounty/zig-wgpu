const std = @import("std");
const c = @import("utils.zig").c;
const svOpt = @import("utils.zig").svOpt;
const GpuAllocator = @import("GpuAllocator.zig");
const GpuBuffer = @import("GpuBuffer.zig");
const GpuTextureFormat = @import("lib.zig").GpuTextureFormat;
const GpuTextureUsage = @import("lib.zig").GpuTextureUsage;

pub const GpuTextureDef = struct {
    label: ?[]const u8 = null,
    size: c.WGPUExtent3D,
    usage: std.EnumSet(GpuTextureUsage),
    format: GpuTextureFormat,
};

raw: c.WGPUTexture,
gloc: GpuAllocator,
def: GpuTextureDef,

pub fn init(gloc: GpuAllocator, def: GpuTextureDef) !@This() {
    var use: u64 = 0;
    var iter = def.usage.iterator();
    while (iter.next()) |flag| use |= @intFromEnum(flag);

    const desc = c.WGPUTextureDescriptor{
        .label = svOpt(def.label),
        .usage = use,
        .dimension = c.WGPUTextureDimension_2D,
        .size = def.size,
        .format = @intFromEnum(def.format),
        .mipLevelCount = 1,
        .sampleCount = 1,
    };
    const raw = try gloc.allocTexture(desc);

    return .{ .gloc = gloc, .raw = raw, .def = def };
}

pub fn deinit(self: @This()) void {
    self.gloc.freeTexture(self.raw);
}

pub fn getConstMappedRange(self: @This(), offset: u64, size: u64) ?*const anyopaque {
    return c.wgpuBufferGetConstMappedRange(self.raw, offset, size);
}

pub fn bytesSize(self: @This()) u32 {
    return self.bytesSizeRow() * self.def.size.height;
}

pub fn bytesSizeRow(self: @This()) u32 {
    return self.def.size.width * self.def.format.bytesPerPixel();
}

/// Return a GpuBuffer containing a copy of the texture.
pub fn buffCopy(self: @This(), gloc: GpuAllocator) !GpuBuffer {
    const buf = try GpuBuffer.init(gloc, .{
        .size = self.bytesSize(),
        .usage = .initMany(&.{ .CopyDst, .CopySrc }),
        .label = "texture_copy_buffer",
    });

    const enc = c.wgpuDeviceCreateCommandEncoder(gloc.device.device, null) orelse return error.Encoder;
    defer c.wgpuCommandEncoderRelease(enc);

    const src_copy = c.WGPUTexelCopyTextureInfo{
        .texture = self.raw,
        .mipLevel = 0,
        .origin = .{ .x = 0, .y = 0, .z = 0 },
        .aspect = c.WGPUTextureAspect_All,
    };
    const dst_copy = c.WGPUTexelCopyBufferInfo{
        .buffer = buf.raw,
        .layout = .{
            .offset = 0,
            .bytesPerRow = self.bytesSizeRow(),
            .rowsPerImage = self.def.size.height,
        },
    };

    c.wgpuCommandEncoderCopyTextureToBuffer(enc, &src_copy, &dst_copy, &self.def.size);

    const cmd = c.wgpuCommandEncoderFinish(enc, null);
    defer c.wgpuCommandBufferRelease(cmd);
    c.wgpuQueueSubmit(gloc.device.queue, 1, &cmd);

    return buf;
}

pub fn mapAsync(
    self: @This(),
    mode: c.WGPUMapMode,
    offset: u64,
    size: u64,
    callback_info: c.WGPUBufferMapCallbackInfo,
) void {
    _ = c.wgpuBufferMapAsync(self.raw, mode, offset, size, callback_info);
}

pub fn unmap(self: @This()) void {
    c.wgpuBufferUnmap(self.raw);
}

/// CPU to GPU
pub fn load(
    self: @This(),
    T: type,
    data: []const T,
) !void {
    const bytes = data.len * @sizeOf(T);

    c.wgpuQueueWriteTexture(
        self.gloc.device.queue,
        &.{
            .texture = self.raw,
            .mipLevel = 0,
            .origin = .{ .x = 0, .y = 0, .z = 0 },
            .aspect = c.WGPUTextureAspect_All,
        },
        data.ptr,
        bytes,
        &.{
            .offset = 0,
            .bytesPerRow = self.bytesSizeRow(),
            .rowsPerImage = self.def.size.height,
        },
        &self.def.size,
    );
}

// GPU to CPU
pub fn read(self: @This(), alloc: std.mem.Allocator, T: type) ![]T {
    const out = try alloc.alloc(T, @divExact(self.size, @sizeOf(T)));

    const staging = try init(self.gloc, .{
        .size = self.size,
        .usage = .initMany(&.{ .MapRead, .CopyDst }),
        .label = "texture_read_staging",
    });
    defer staging.deinit();

    const enc = c.wgpuDeviceCreateCommandEncoder(self.gloc.device.device, null) orelse return error.Encoder;
    const src_copy = c.WGPUTexelCopyTextureInfo{
        .texture = self.raw,
        .mipLevel = 0,
        .origin = .{ .x = 0, .y = 0, .z = 0 },
        .aspect = c.WGPUTextureAspect_All,
    };
    const dst_copy = c.WGPUTexelCopyBufferInfo{
        .buffer = staging.raw,
        .layout = .{
            .offset = 0,
            .bytesPerRow = self.bytesSizeRow(),
            .rowsPerImage = self.def.size.height,
        },
    };
    c.wgpuCommandEncoderCopyTextureToBuffer(enc, &src_copy, &dst_copy, &self.def.size);
    const cmd = c.wgpuCommandEncoderFinish(enc, null);
    defer c.wgpuCommandEncoderRelease(enc);
    defer c.wgpuCommandBufferRelease(cmd);
    c.wgpuQueueSubmit(self.gloc.device.queue, 1, &cmd);

    var mapped = false;
    staging.mapAsync(
        c.WGPUMapMode_Read,
        0,
        self.size,
        .{ .callback = onMapped, .userdata1 = &mapped },
    );
    while (!mapped) self.gloc.device.poll();

    const ptr: [*]const T = @ptrCast(@alignCast(
        staging.getConstMappedRange(0, self.size),
    ));
    @memcpy(out[0..out.len], ptr[0..out.len]);
    staging.unmap();

    return out;
}

fn onMapped(
    status: c.WGPUMapAsyncStatus,
    _: c.WGPUStringView,
    userdata1: ?*anyopaque,
    _: ?*anyopaque,
) callconv(.c) void {
    const flag: *bool = @ptrCast(@alignCast(userdata1.?));
    flag.* = (status == c.WGPUMapAsyncStatus_Success);
}
