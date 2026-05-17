const std = @import("std");
const c = @import("c.zig").c;
const GpuAllocator = @import("GpuAllocator.zig");
const GpuBuffer = @import("GpuBuffer.zig");

const Mat = @This();

buf: GpuBuffer,
rows: usize,
cols: usize,

pub fn load(
    gloc: *GpuAllocator,
    data: []const f32,
    rows: usize,
    cols: usize,
) !Mat {
    std.debug.assert(data.len == @as(usize, rows) * cols);
    const bytes = data.len * @sizeOf(f32);

    // Uses structural constructor initialization
    const buf = try GpuBuffer.init(
        gloc,
        bytes,
        c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst | c.WGPUBufferUsage_CopySrc,
    );

    c.wgpuQueueWriteBuffer(gloc.queue, buf.raw, 0, data.ptr, bytes);
    return .{ .buf = buf, .rows = rows, .cols = cols };
}

pub fn zeros(gloc: *GpuAllocator, rows: usize, cols: usize) !Mat {
    const bytes: u64 = @as(u64, rows) * cols * @sizeOf(f32);
    const buf = try GpuBuffer.init(
        gloc,
        bytes,
        c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst | c.WGPUBufferUsage_CopySrc,
    );
    return .{ .buf = buf, .rows = rows, .cols = cols };
}

pub fn deinit(self: Mat) void {
    self.buf.deinit();
}

pub fn len(self: Mat) usize {
    return self.rows * self.cols;
}

pub fn byteSize(self: Mat) u64 {
    return @as(u64, self.len()) * @sizeOf(f32);
}

pub fn add(self: Mat, gloc: *GpuAllocator, other: Mat) !Mat {
    std.debug.assert(self.rows == other.rows and self.cols == other.cols);

    const result = try Mat.zeros(gloc, self.rows, self.cols);
    errdefer result.deinit();

    const pipeline = try gloc.pipAdd();
    try dispatch2in1out(gloc, pipeline, self.buf, other.buf, result.buf, self.byteSize());

    return result;
}

pub fn read(self: Mat, gloc: *GpuAllocator, out: []f32) !void {
    std.debug.assert(out.len >= self.len());
    const bytes = self.byteSize();

    const staging = try GpuBuffer.init(
        gloc,
        bytes,
        c.WGPUBufferUsage_MapRead | c.WGPUBufferUsage_CopyDst,
    );
    defer staging.deinit();

    const enc = c.wgpuDeviceCreateCommandEncoder(gloc.device, null) orelse return error.Encoder;
    c.wgpuCommandEncoderCopyBufferToBuffer(enc, self.buf.raw, 0, staging.raw, 0, bytes);
    const cmd = c.wgpuCommandEncoderFinish(enc, null);
    defer c.wgpuCommandEncoderRelease(enc);
    defer c.wgpuCommandBufferRelease(cmd);
    c.wgpuQueueSubmit(gloc.queue, 1, &cmd);

    var mapped = false;
    staging.mapAsync(
        c.WGPUMapMode_Read,
        0,
        bytes,
        .{ .callback = onMapped, .userdata1 = &mapped },
    );
    while (!mapped) gloc.poll();

    const ptr: [*]const f32 = @ptrCast(@alignCast(
        staging.getConstMappedRange(0, bytes),
    ));
    @memcpy(out[0..self.len()], ptr[0..self.len()]);
    staging.unmap();
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

// ── Dispatch helpers ──────────────────────────────────────────────────────────

/// Encode + submit a 2-input, 1-output compute pass (used by add).
fn dispatch2in1out(
    gloc: *GpuAllocator,
    pipeline: c.WGPUComputePipeline,
    buf_a: GpuBuffer,
    buf_b: GpuBuffer,
    buf_out: GpuBuffer,
    bytes: u64,
) !void {
    const max_chunk_bytes: u64 = 1024 * 1024 * 1024; // 1 GB

    var offset: u64 = 0;
    while (offset < bytes) {
        // Calculate bounds for the current chunk
        const current_chunk_bytes = @min(max_chunk_bytes, bytes - offset);
        const current_chunk_elements: u32 = @intCast(current_chunk_bytes / @sizeOf(f32));

        // Create uniform buffer for this specific chunk's size
        const info_buf = try GpuBuffer.init(
            gloc,
            @sizeOf(u32),
            c.WGPUBufferUsage_Uniform | c.WGPUBufferUsage_CopyDst,
        );
        defer info_buf.deinit();

        // Write the number of elements *in this chunk* to the uniform buffer
        c.wgpuQueueWriteBuffer(gloc.queue, info_buf.raw, 0, &current_chunk_elements, @sizeOf(u32));

        // Bind only the sub-slice for this chunk using `.offset` and `.size`
        const entries = [_]c.WGPUBindGroupEntry{
            .{ .binding = 0, .buffer = buf_a.raw, .offset = offset, .size = current_chunk_bytes },
            .{ .binding = 1, .buffer = buf_b.raw, .offset = offset, .size = current_chunk_bytes },
            .{ .binding = 2, .buffer = buf_out.raw, .offset = offset, .size = current_chunk_bytes },
            .{ .binding = 3, .buffer = info_buf.raw, .offset = 0, .size = @sizeOf(u32) },
        };

        // Submit the pass for this specific chunk
        try submitPass(gloc, pipeline, &entries, current_chunk_elements);

        offset += current_chunk_bytes;
    }
}

/// Create bind group, encode pass, submit.
fn submitPass(
    gloc: *GpuAllocator,
    pipeline: c.WGPUComputePipeline,
    entries: []const c.WGPUBindGroupEntry,
    n: usize,
) !void {
    const bgl = c.wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
    defer c.wgpuBindGroupLayoutRelease(bgl);

    const bg = c.wgpuDeviceCreateBindGroup(gloc.device, &.{
        .layout = bgl,
        .entries = entries.ptr,
        .entryCount = entries.len,
    }) orelse return error.BindGroup;
    defer c.wgpuBindGroupRelease(bg);

    const enc = c.wgpuDeviceCreateCommandEncoder(gloc.device, null) orelse
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
    c.wgpuQueueSubmit(gloc.queue, 1, &cmd);
}

fn ceilDiv(n: usize, d: usize) usize {
    return (n + d - 1) / d;
}
