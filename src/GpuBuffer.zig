const std = @import("std");
const c = @import("utils.zig").c;
const GpuAllocator = @import("GpuAllocator.zig");
const svOpt = @import("utils.zig").svOpt;

raw: c.WGPUBuffer,
glloc: GpuAllocator,
def: GpuBufferDef,

pub const GpuBufferUsage = enum(u64) {
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

    fn enumSetToWGPUBufferUsage(set: std.EnumSet(GpuBufferUsage)) c.WGPUBufferUsage {
        var use: u64 = 0;
        var iter = set.iterator();
        while (iter.next()) |flag| use |= @intFromEnum(flag);
        return use;
    }
};

pub const GpuBufferDef = struct {
    label: ?[]const u8 = null,
    size: u64,
    usage: std.EnumSet(GpuBufferUsage),
};

pub fn init(glloc: GpuAllocator, def: GpuBufferDef) !@This() {

    // Automatically align the buffer size forward to a multiple of 4 bytes under the hood
    const aligned_size = std.mem.alignForward(u64, def.size, 4);

    const raw_handle = try glloc.allocBuffer(.{
        .size = aligned_size,
        .usage = GpuBufferUsage.enumSetToWGPUBufferUsage(def.usage),
        .label = svOpt(def.label),
    });
    return .{
        .raw = raw_handle,
        .def = def,
        .glloc = glloc,
    };
}

pub fn deinit(self: @This()) void {
    self.glloc.freeBuffer(self.raw);
}

pub fn getConstMappedRange(self: @This(), offset: u64, size: u64) ?*const anyopaque {
    return c.wgpuBufferGetConstMappedRange(self.raw, offset, size);
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

    if (bytes == self.def.size) {
        // Aligned path: direct download
        c.wgpuQueueWriteBuffer(self.glloc.device.queue, self.raw, 0, data.ptr, self.def.size);
    } else {
        // Unaligned path: Split the write into an aligned chunk and a padded remainder
        // to support arbitrary lengths without any allocations or large stack arrays.
        const aligned_part = (bytes / 4) * 4;
        if (aligned_part > 0) {
            c.wgpuQueueWriteBuffer(self.glloc.device.queue, self.raw, 0, data.ptr, aligned_part);
        }

        var remainder_buf: [4]u8 = .{ 0, 0, 0, 0 };
        const data_bytes = std.mem.sliceAsBytes(data);
        @memcpy(remainder_buf[0 .. bytes - aligned_part], data_bytes[aligned_part..bytes]);

        c.wgpuQueueWriteBuffer(self.glloc.device.queue, self.raw, aligned_part, &remainder_buf, 4);
    }
}

/// GPU to CPU
/// Buffer must have MapRead usage or returns error.BufferNotMappable.
pub fn read(self: @This(), alloc: std.mem.Allocator, T: type) ![]T {
    if (!self.def.usage.contains(.MapRead)) return error.BufferNotMappable;

    const out = try alloc.alloc(T, @divExact(self.def.size, @sizeOf(T)));

    var mapped = false;
    self.mapAsync(
        c.WGPUMapMode_Read,
        0,
        self.def.size,
        .{ .callback = onMapped, .userdata1 = &mapped },
    );
    while (!mapped) self.glloc.device.poll();

    const ptr: [*]const T = @ptrCast(@alignCast(
        self.getConstMappedRange(0, self.def.size),
    ));
    @memcpy(out[0..out.len], ptr[0..out.len]);
    self.unmap();

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

/// GPU to GPU. Both buffers must be same size, src needs CopySrc, dst needs CopyDst.
pub fn copy(src: @This(), dst: @This()) !void {
    if (src.def.size != dst.def.size) return error.SizeMismatch;

    const copy_src: u64 = @intFromEnum(GpuBufferUsage.CopySrc);
    const copy_dst: u64 = @intFromEnum(GpuBufferUsage.CopyDst);
    if (@as(u64, GpuBufferUsage.enumSetToWGPUBufferUsage(src.def.usage)) & copy_src == 0) return error.SrcNotCopyable;
    if (@as(u64, GpuBufferUsage.enumSetToWGPUBufferUsage(dst.def.usage)) & copy_dst == 0) return error.DstNotWritable;

    const enc = c.wgpuDeviceCreateCommandEncoder(src.glloc.device.device, null) orelse return error.Encoder;
    c.wgpuCommandEncoderCopyBufferToBuffer(enc, src.raw, 0, dst.raw, 0, src.def.size);
    const cmd = c.wgpuCommandEncoderFinish(enc, null);
    defer c.wgpuCommandEncoderRelease(enc);
    defer c.wgpuCommandBufferRelease(cmd);
    c.wgpuQueueSubmit(src.glloc.device.queue, 1, &cmd);
}
