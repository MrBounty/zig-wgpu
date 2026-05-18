const std = @import("std");
const c = @import("utils.zig").c;
const GpuAllocator = @import("GpuAllocator.zig");
const GpuBuffer = @import("GpuBuffer.zig");
const GpuDevice = @import("GpuDevice.zig");
const GpuPipeline = @import("GpuPipeline.zig");

const Vec = @This();

buf: GpuBuffer,
len: usize,

// Changed: gloc is passed by value (const)
pub fn initZero(gloc: GpuAllocator, len: usize) !Vec {
    return .{
        .buf = try GpuBuffer.init(
            gloc,
            len * @sizeOf(f16),
            .initMany(&.{ .Storage, .CopyDst, .CopySrc }),
        ),
        .len = len,
    };
}

// Changed: gloc is passed by value
pub fn initLoad(gloc: GpuAllocator, data: []const f16) !Vec {
    var self = try initZero(gloc, data.len);
    try self.load(data); // Direct access via the interface copy
    return self;
}

pub fn deinit(self: Vec) void {
    self.buf.deinit();
}

/// CPU to GPU.
pub fn load(
    self: Vec,
    data: []const f16,
) !void {
    try self.buf.load(data);
}

pub fn byteSize(self: Vec) u64 {
    return @as(u64, self.len) * @sizeOf(f16);
}

// Changed: gloc is passed by value instead of *GpuAllocator
pub fn run(self: Vec, gloc: GpuAllocator, other: Vec, pip: GpuPipeline) !Vec {
    std.debug.assert(self.len == other.len);

    const result = try Vec.initZero(gloc, self.len);
    errdefer result.deinit();

    try dispatch2in1out(gloc, pip.raw, self.buf, other.buf, result.buf, self.byteSize());

    return result;
}

// Changed: gloc is passed by value instead of *GpuAllocator
pub fn read(self: Vec, alloc: std.mem.Allocator) ![]f16 {
    return self.buf.read(alloc, f16);
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

// Changed: gloc is passed by value instead of *GpuAllocator
fn dispatch2in1out(
    gloc: GpuAllocator,
    pipeline: c.WGPUComputePipeline,
    buf_a: GpuBuffer,
    buf_b: GpuBuffer,
    buf_out: GpuBuffer,
    bytes: u64,
) !void {
    const max_chunk_bytes: u64 = 1024 * 1024 * 1024; // 1 GB

    var offset: u64 = 0;
    while (offset < bytes) {
        const current_chunk_bytes = @min(max_chunk_bytes, bytes - offset);
        const current_chunk_elements: u32 = @intCast(current_chunk_bytes / @sizeOf(f16));

        const info_buf = try GpuBuffer.init(
            gloc,
            @sizeOf(u32),
            .initMany(&.{ .Uniform, .CopyDst }),
        );
        defer info_buf.deinit();

        c.wgpuQueueWriteBuffer(gloc.device.queue, info_buf.raw, 0, &current_chunk_elements, @sizeOf(u32));

        const entries = [_]c.WGPUBindGroupEntry{
            .{ .binding = 0, .buffer = buf_a.raw, .offset = offset, .size = current_chunk_bytes },
            .{ .binding = 1, .buffer = buf_b.raw, .offset = offset, .size = current_chunk_bytes },
            .{ .binding = 2, .buffer = buf_out.raw, .offset = offset, .size = current_chunk_bytes },
            .{ .binding = 3, .buffer = info_buf.raw, .offset = 0, .size = @sizeOf(u32) },
        };

        try submitPass(gloc, pipeline, &entries, current_chunk_elements);

        offset += current_chunk_bytes;
    }
}

// Changed: gloc is passed by value instead of *GpuAllocator
fn submitPass(
    gloc: GpuAllocator,
    pipeline: c.WGPUComputePipeline,
    entries: []const c.WGPUBindGroupEntry,
    n: usize,
) !void {
    const bgl = c.wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
    defer c.wgpuBindGroupLayoutRelease(bgl);

    const bg = c.wgpuDeviceCreateBindGroup(gloc.device.device, &.{
        .layout = bgl,
        .entries = entries.ptr,
        .entryCount = entries.len,
    }) orelse return error.BindGroup;
    defer c.wgpuBindGroupRelease(bg);

    const enc = c.wgpuDeviceCreateCommandEncoder(gloc.device.device, null) orelse
        return error.Encoder;
    const pass = c.wgpuCommandEncoderBeginComputePass(enc, null);
    c.wgpuComputePassEncoderSetPipeline(pass, pipeline);
    c.wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, null);

    const WORKGROUP_SIZE = 256;
    const MAX_WORKGROUPS = 65535;

    const desired_workgroups = ceilDiv(n, WORKGROUP_SIZE);
    const dispatch_count = @min(desired_workgroups, MAX_WORKGROUPS);

    c.wgpuComputePassEncoderDispatchWorkgroups(pass, @intCast(dispatch_count), 1, 1);
    c.wgpuComputePassEncoderEnd(pass);
    c.wgpuComputePassEncoderRelease(pass);

    const cmd = c.wgpuCommandEncoderFinish(enc, null);
    defer c.wgpuCommandEncoderRelease(enc);
    defer c.wgpuCommandBufferRelease(cmd);
    c.wgpuQueueSubmit(gloc.device.queue, 1, &cmd);
}

fn ceilDiv(n: usize, d: usize) usize {
    return (n + d - 1) / d;
}
