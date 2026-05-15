// Minimal WebGPU compute in Zig: element-wise matrix addition
// Uses wgpu-native C bindings.
// Build: see ../build.zig
//
// Data flow:
//   CPU (mat_a, mat_b) → GPU storage buffers → compute shader → GPU buf_c
//   → staging buffer (mapped) → CPU read → print

const std = @import("std");
const c = @cImport(@cInclude("wgpu.h"));

// ── Config ────────────────────────────────────────────────────────────────────
const ROWS: u32 = 4;
const COLS: u32 = 4;
const N = ROWS * COLS; // 16 elements
const BUF_BYTES = N * @sizeOf(f32);

// ── WGSL Compute Shader ───────────────────────────────────────────────────────
//   workgroup_size(4,4) matches one full 4×4 matrix → dispatch(1,1,1)
const SHADER =
    \\@group(0) @binding(0) var<storage, read>       mat_a : array<f32>;
    \\@group(0) @binding(1) var<storage, read>       mat_b : array<f32>;
    \\@group(0) @binding(2) var<storage, read_write> mat_c : array<f32>;
    \\
    \\@compute @workgroup_size(4, 4)
    \\fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    \\    let idx = gid.y * 4u + gid.x;
    \\    if (idx < arrayLength(&mat_c)) {
    \\        mat_c[idx] = mat_a[idx] + mat_b[idx];
    \\    }
    \\}
;

// ── Callback state ────────────────────────────────────────────────────────────
const Ctx = struct {
    adapter: c.WGPUAdapter = null,
    device: c.WGPUDevice = null,
};

fn onAdapter(
    status: c.WGPURequestAdapterStatus,
    adapter: c.WGPUAdapter,
    _: c.WGPUStringView,
    userdata1: ?*anyopaque,
    _: ?*anyopaque,
) callconv(.c) void {
    if (status != c.WGPURequestAdapterStatus_Success) {
        std.log.err("Adapter request failed (status={d})", .{status});
        return;
    }
    const ctx: *Ctx = @ptrCast(@alignCast(userdata1.?));
    ctx.adapter = adapter;
}

fn onDevice(
    status: c.WGPURequestDeviceStatus,
    device: c.WGPUDevice,
    _: c.WGPUStringView,
    userdata1: ?*anyopaque,
    _: ?*anyopaque,
) callconv(.c) void {
    if (status != c.WGPURequestDeviceStatus_Success) {
        std.log.err("Device request failed (status={d})", .{status});
        return;
    }
    const ctx: *Ctx = @ptrCast(@alignCast(userdata1.?));
    ctx.device = device;
    std.debug.print("{?}", .{device});
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

fn sv(s: []const u8) c.WGPUStringView {
    return .{ .data = s.ptr, .length = s.len };
}

// ── Main ──────────────────────────────────────────────────────────────────────
pub fn main() !void {

    // 1. Instance ──────────────────────────────────────────────────────────────
    const instance = c.wgpuCreateInstance(&std.mem.zeroes(c.WGPUInstanceDescriptor)) orelse
        return error.NoInstance;
    defer c.wgpuInstanceRelease(instance);

    // 2. Adapter (async → poll) ────────────────────────────────────────────────
    var ctx = Ctx{};
    _ = c.wgpuInstanceRequestAdapter(
        instance,
        &.{ .powerPreference = c.WGPUPowerPreference_HighPerformance },
        .{ .callback = onAdapter, .userdata1 = &ctx },
    );
    c.wgpuInstanceProcessEvents(instance); // drive callbacks
    const adapter = ctx.adapter orelse return error.NoAdapter;
    defer c.wgpuAdapterRelease(adapter);

    // 3. Device ────────────────────────────────────────────────────────────────
    _ = c.wgpuAdapterRequestDevice(adapter, null, .{ .callback = onDevice, .userdata1 = &ctx });
    c.wgpuInstanceProcessEvents(instance);
    const device = ctx.device orelse return error.NoDevice;
    defer c.wgpuDeviceRelease(device);

    const queue = c.wgpuDeviceGetQueue(device);
    defer c.wgpuQueueRelease(queue);

    // 4. Input data ────────────────────────────────────────────────────────────
    //   mat_a[i] = i        (0 … 15)
    //   mat_b[i] = 15 − i   → every element of mat_c should equal 15
    var mat_a: [N]f32 = undefined;
    var mat_b: [N]f32 = undefined;
    for (0..N) |i| {
        mat_a[i] = @floatFromInt(i);
        mat_b[i] = @floatFromInt(N - 1 - i);
    }

    // 5. GPU Buffers ───────────────────────────────────────────────────────────
    const buf_a = c.wgpuDeviceCreateBuffer(device, &.{
        .usage = c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst,
        .size = BUF_BYTES,
    }) orelse return error.BufferA;

    const buf_b = c.wgpuDeviceCreateBuffer(device, &.{
        .usage = c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst,
        .size = BUF_BYTES,
    }) orelse return error.BufferB;

    // buf_c: GPU-only result; staging: CPU-readable copy
    const buf_c = c.wgpuDeviceCreateBuffer(device, &.{
        .usage = c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc,
        .size = BUF_BYTES,
    }) orelse return error.BufferC;

    const buf_staging = c.wgpuDeviceCreateBuffer(device, &.{
        .usage = c.WGPUBufferUsage_MapRead | c.WGPUBufferUsage_CopyDst,
        .size = BUF_BYTES,
    }) orelse return error.BufferStaging;

    defer c.wgpuBufferRelease(buf_a);
    defer c.wgpuBufferRelease(buf_b);
    defer c.wgpuBufferRelease(buf_c);
    defer c.wgpuBufferRelease(buf_staging);

    // Upload inputs
    c.wgpuQueueWriteBuffer(queue, buf_a, 0, &mat_a, BUF_BYTES);
    c.wgpuQueueWriteBuffer(queue, buf_b, 0, &mat_b, BUF_BYTES);

    // 6. Shader module ─────────────────────────────────────────────────────────
    // ✅ New API (0.20+)
    var wgsl_src = c.WGPUShaderSourceWGSL{
        .chain = .{ .sType = c.WGPUSType_ShaderSourceWGSL },
        .code = sv(SHADER),
    };
    const shader = c.wgpuDeviceCreateShaderModule(device, &.{
        .nextInChain = @ptrCast(&wgsl_src),
    }) orelse return error.Shader;

    // 7. Compute pipeline (layout auto-inferred from shader) ───────────────────
    // ✅
    const pipeline = c.wgpuDeviceCreateComputePipeline(device, &.{
        .compute = .{
            .module = shader,
            .entryPoint = sv("main"),
        },
    }) orelse return error.Pipeline;
    defer c.wgpuComputePipelineRelease(pipeline);

    // 8. Bind group ────────────────────────────────────────────────────────────
    const bgl = c.wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
    defer c.wgpuBindGroupLayoutRelease(bgl);

    const entries = [_]c.WGPUBindGroupEntry{
        .{ .binding = 0, .buffer = buf_a, .offset = 0, .size = BUF_BYTES },
        .{ .binding = 1, .buffer = buf_b, .offset = 0, .size = BUF_BYTES },
        .{ .binding = 2, .buffer = buf_c, .offset = 0, .size = BUF_BYTES },
    };
    const bind_group = c.wgpuDeviceCreateBindGroup(device, &.{
        .layout = bgl,
        .entries = &entries,
        .entryCount = entries.len,
    }) orelse return error.BindGroup;
    defer c.wgpuBindGroupRelease(bind_group);

    // 9. Encode compute pass + buffer copy ────────────────────────────────────
    const encoder = c.wgpuDeviceCreateCommandEncoder(device, null) orelse
        return error.Encoder;

    const pass = c.wgpuCommandEncoderBeginComputePass(encoder, null);
    c.wgpuComputePassEncoderSetPipeline(pass, pipeline);
    c.wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, null);
    // dispatch(1,1,1): one workgroup of size (4,4) covers the whole 4×4 matrix
    c.wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
    c.wgpuComputePassEncoderEnd(pass);
    c.wgpuComputePassEncoderRelease(pass);

    // Copy result buffer → CPU-readable staging buffer
    c.wgpuCommandEncoderCopyBufferToBuffer(encoder, buf_c, 0, buf_staging, 0, BUF_BYTES);

    const cmdbuf = c.wgpuCommandEncoderFinish(encoder, null);
    defer c.wgpuCommandEncoderRelease(encoder);
    defer c.wgpuCommandBufferRelease(cmdbuf);

    // 10. Submit ───────────────────────────────────────────────────────────────
    c.wgpuQueueSubmit(queue, 1, &cmdbuf);

    // 11. Map staging buffer back to CPU ──────────────────────────────────────
    var mapped = false;
    _ = c.wgpuBufferMapAsync(
        buf_staging,
        c.WGPUMapMode_Read,
        0,
        BUF_BYTES,
        .{ .callback = onMapped, .userdata1 = &mapped },
    );
    // Poll the device until the async map completes
    while (!mapped) _ = c.wgpuDevicePoll(device, 1, null);

    const ptr: [*]const f32 = @ptrCast(@alignCast(
        c.wgpuBufferGetConstMappedRange(buf_staging, 0, BUF_BYTES),
    ));
    const result = ptr[0..N];

    // 12. Print ────────────────────────────────────────────────────────────────
    std.debug.print("\nmat_a + mat_b  ({d}×{d}):\n", .{ ROWS, COLS });
    for (0..ROWS) |r| {
        for (0..COLS) |col|
            std.debug.print("{d:6.0}", .{result[r * COLS + col]});
        std.debug.print("\n", .{});
    }
    // Expected output: every cell = 15.0

    c.wgpuBufferUnmap(buf_staging);
}
