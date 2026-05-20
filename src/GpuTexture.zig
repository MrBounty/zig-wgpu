const std = @import("std");
const c = @import("utils.zig").c;
const GpuAllocator = @import("GpuAllocator.zig");
const GpuTextureFormat = @import("lib.zig").GpuTextureFormat;

const TextureUsage = enum(u64) {
    None = 0x0000000000000000,
    CopySrc = 0x0000000000000001,
    CopyDst = 0x0000000000000002,
    TextureBinding = 0x0000000000000004,
    StorageBinding = 0x0000000000000008,
    RenderAttachment = 0x0000000000000010,
    TransientAttachment = 0x0000000000000020,
};

raw: c.WGPUTexture,
size: c.WGPUExtent3D,
usage: c.WGPUTextureUsage,
format: GpuTextureFormat,
gloc: GpuAllocator,

/// Allocates the underlying WebGPU handle and registers it to the parent GpuAllocator
pub fn init(gloc: GpuAllocator, format: GpuTextureFormat, size: c.WGPUExtent3D, usage: std.EnumSet(TextureUsage)) !@This() {
    var use: u64 = 0;
    var iter = usage.iterator();
    while (iter.next()) |flag| use |= @intFromEnum(flag);

    const desc = c.WGPUTextureDescriptor{
        .usage = use,
        .dimension = c.WGPUTextureDimension_2D,
        .size = size,
        .format = @intFromEnum(format),
        .mipLevelCount = 1,
        .sampleCount = 1,
    };
    const raw = try gloc.allocTexture(desc);

    return .{ .gloc = gloc, .raw = raw, .size = size, .format = format, .usage = use };
}

/// Unregisters from the parent GpuAllocator and cleanly destroys GPU resources
pub fn deinit(self: @This()) void {
    self.gloc.freeTexture(self.raw);
}

/// Native getConstMappedRange wrapper
pub fn getConstMappedRange(self: @This(), offset: u64, size: u64) ?*const anyopaque {
    return c.wgpuBufferGetConstMappedRange(self.raw, offset, size);
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

/// Native unmap wrapper
pub fn unmap(self: @This()) void {
    c.wgpuBufferUnmap(self.raw);
}

/// CPU to GPU.
pub fn load(
    self: @This(),
    T: type,
    data: []const T,
) !void {
    const bytes = data.len * @sizeOf(T);

    if (bytes == self.size) {
        // Aligned path: direct download
        c.wgpuQueueWriteBuffer(self.gloc.device.queue, self.raw, 0, data.ptr, self.size);
    } else {
        // Unaligned path: Split the write into an aligned chunk and a padded remainder
        // to support arbitrary lengths without any allocations or large stack arrays.
        const aligned_part = (bytes / 4) * 4;
        if (aligned_part > 0) {
            c.wgpuQueueWriteBuffer(self.gloc.device.queue, self.raw, 0, data.ptr, aligned_part);
        }

        var remainder_buf: [4]u8 = .{ 0, 0, 0, 0 };
        const data_bytes = std.mem.sliceAsBytes(data);
        @memcpy(remainder_buf[0 .. bytes - aligned_part], data_bytes[aligned_part..bytes]);

        c.wgpuQueueWriteBuffer(self.gloc.device.queue, self.raw, aligned_part, &remainder_buf, 4);
    }
}

pub fn read(self: @This(), alloc: std.mem.Allocator, T: type) ![]T {
    const out = try alloc.alloc(T, @divExact(self.size, @sizeOf(T)));

    const staging = try init(
        self.gloc,
        self.size,
        .initMany(&.{ .MapRead, .CopyDst }),
    );
    defer staging.deinit();

    const enc = c.wgpuDeviceCreateCommandEncoder(self.gloc.device.device, null) orelse return error.Encoder;
    c.wgpuCommandEncoderCopyBufferToBuffer(enc, self.raw, 0, staging.raw, 0, self.size);
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
