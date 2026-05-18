/// GpuProcess is just a pipeline with 2 inpout and 1 output
/// for now, to see if I make it a bit more generic
///
const std = @import("std");
const c = @import("utils.zig").c;
const sv = @import("utils.zig").sv;
const GpuAllocator = @import("GpuAllocator.zig");
const GpuBuffer = @import("GpuBuffer.zig");
const GpuDevice = @import("GpuDevice.zig");

pip: c.WGPUComputePipeline,

pub fn init(device: GpuDevice, wgsl: []const u8) !@This() {
    var wgsl_src = c.WGPUShaderSourceWGSL{
        .chain = .{ .sType = c.WGPUSType_ShaderSourceWGSL },
        .code = sv(wgsl),
    };
    const shader = c.wgpuDeviceCreateShaderModule(device.device, &.{
        .nextInChain = @ptrCast(&wgsl_src),
    }) orelse return error.Shader;
    defer c.wgpuShaderModuleRelease(shader);

    return .{ .pip = c.wgpuDeviceCreateComputePipeline(device.device, &.{
        .compute = .{ .module = shader, .entryPoint = sv("main") },
    }) orelse return error.Pipeline };
}

pub fn deinit(self: @This()) void {
    c.wgpuComputePipelineRelease(self.pip);
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
pub fn run(
    self: @This(),
    gloc: GpuAllocator,
    T: type,
    buf_a: GpuBuffer,
    buf_b: GpuBuffer,
    buf_out: GpuBuffer,
) !void {
    const max_chunk_bytes: u64 = 1024 * 1024 * 1024; // 1 GB

    const bytes = buf_a.size;
    var offset: u64 = 0;
    while (offset < bytes) {
        const current_chunk_bytes = @min(max_chunk_bytes, bytes - offset);
        const current_chunk_elements: u32 = @intCast(current_chunk_bytes / @sizeOf(T));

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

        try submitPass(gloc, self.pip, &entries, current_chunk_elements);

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
