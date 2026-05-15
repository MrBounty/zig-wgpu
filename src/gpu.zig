/// gpu.zig — WebGPU compute module
///
/// Usage:
///   var gloc = try GpuAllocator.init();
///   defer gloc.deinit();
///
///   const a = try Mat.load(&gloc, &data_a, 4, 4);
///   defer a.deinit();
///   const b = try Mat.load(&gloc, &data_b, 4, 4);
///   defer b.deinit();
///
///   const c = try a.add(&gloc, b);
///   defer c.deinit();
///
///   var out: [16]f32 = undefined;
///   try c.read(&gloc, &out);
const std = @import("std");
const c = @cImport(@cInclude("wgpu.h"));

// ── Shaders ───────────────────────────────────────────────────────────────────

const SHADER_ADD =
    \\@group(0) @binding(0) var<storage, read>       a : array<f32>;
    \\@group(0) @binding(1) var<storage, read>       b : array<f32>;
    \\@group(0) @binding(2) var<storage, read_write> out : array<f32>;
    \\
    \\@compute @workgroup_size(64)
    \\fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    \\    let i = gid.x;
    \\    if (i < arrayLength(&out)) {
    \\        out[i] = a[i] + b[i];
    \\    }
    \\}
;

const SHADER_SCALE =
    \\struct Uniforms { scalar : f32 }
    \\@group(0) @binding(0) var<storage, read>       a   : array<f32>;
    \\@group(0) @binding(1) var<storage, read_write> out : array<f32>;
    \\@group(0) @binding(2) var<uniform>             u   : Uniforms;
    \\
    \\@compute @workgroup_size(64)
    \\fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    \\    let i = gid.x;
    \\    if (i < arrayLength(&out)) {
    \\        out[i] = a[i] * u.scalar;
    \\    }
    \\}
;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn sv(s: []const u8) c.WGPUStringView {
    return .{ .data = s.ptr, .length = s.len };
}

fn ceilDiv(n: u32, d: u32) u32 {
    return (n + d - 1) / d;
}

// ── Callbacks ─────────────────────────────────────────────────────────────────

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

// ── Pipeline factory ──────────────────────────────────────────────────────────

fn buildPipeline(device: c.WGPUDevice, wgsl: []const u8) !c.WGPUComputePipeline {
    var wgsl_src = c.WGPUShaderSourceWGSL{
        .chain = .{ .sType = c.WGPUSType_ShaderSourceWGSL },
        .code = sv(wgsl),
    };
    const shader = c.wgpuDeviceCreateShaderModule(device, &.{
        .nextInChain = @ptrCast(&wgsl_src),
    }) orelse return error.Shader;
    defer c.wgpuShaderModuleRelease(shader);

    return c.wgpuDeviceCreateComputePipeline(device, &.{
        .compute = .{ .module = shader, .entryPoint = sv("main") },
    }) orelse error.Pipeline;
}

// ── GpuAllocator ──────────────────────────────────────────────────────────────

/// Owns the WebGPU context. One per application.
/// All Mat instances must not outlive their GpuAllocator.
pub const GpuAllocator = struct {
    instance: c.WGPUInstance,
    adapter: c.WGPUAdapter,
    device: c.WGPUDevice,
    queue: c.WGPUQueue,

    // Lazily created, cached for lifetime of allocator
    _pip_add: c.WGPUComputePipeline = null,
    _pip_scale: c.WGPUComputePipeline = null,

    pub fn init() !GpuAllocator {
        const instance = c.wgpuCreateInstance(
            &std.mem.zeroes(c.WGPUInstanceDescriptor),
        ) orelse return error.NoInstance;
        errdefer c.wgpuInstanceRelease(instance);

        var ctx = Ctx{};
        _ = c.wgpuInstanceRequestAdapter(
            instance,
            &.{ .powerPreference = c.WGPUPowerPreference_HighPerformance },
            .{ .callback = onAdapter, .userdata1 = &ctx },
        );
        c.wgpuInstanceProcessEvents(instance);
        const adapter = ctx.adapter orelse return error.NoAdapter;
        errdefer c.wgpuAdapterRelease(adapter);

        _ = c.wgpuAdapterRequestDevice(
            adapter,
            null,
            .{ .callback = onDevice, .userdata1 = &ctx },
        );
        c.wgpuInstanceProcessEvents(instance);
        const device = ctx.device orelse return error.NoDevice;

        return .{
            .instance = instance,
            .adapter = adapter,
            .device = device,
            .queue = c.wgpuDeviceGetQueue(device),
        };
    }

    pub fn deinit(self: *GpuAllocator) void {
        if (self._pip_add) |p| c.wgpuComputePipelineRelease(p);
        if (self._pip_scale) |p| c.wgpuComputePipelineRelease(p);
        c.wgpuQueueRelease(self.queue);
        c.wgpuDeviceRelease(self.device);
        c.wgpuAdapterRelease(self.adapter);
        c.wgpuInstanceRelease(self.instance);
    }

    // ── Internal ─────────────────────────────────────────────────────────────

    pub fn makeBuffer(
        self: *GpuAllocator,
        bytes: u64,
        usage: c.WGPUBufferUsage,
    ) !c.WGPUBuffer {
        return c.wgpuDeviceCreateBuffer(self.device, &.{
            .usage = usage,
            .size = bytes,
        }) orelse error.BufferAlloc;
    }

    fn pipAdd(self: *GpuAllocator) !c.WGPUComputePipeline {
        if (self._pip_add == null)
            self._pip_add = try buildPipeline(self.device, SHADER_ADD);
        return self._pip_add.?;
    }

    fn pipScale(self: *GpuAllocator) !c.WGPUComputePipeline {
        if (self._pip_scale == null)
            self._pip_scale = try buildPipeline(self.device, SHADER_SCALE);
        return self._pip_scale.?;
    }

    /// Poll until GPU work completes. Use after submit if you need CPU sync.
    pub fn poll(self: *GpuAllocator) void {
        _ = c.wgpuDevicePoll(self.device, 1, null);
    }
};

/// GPU-resident f32 matrix.
/// Buffer usage: Storage | CopyDst | CopySrc (usable as input and output).
pub const Mat = struct {
    buf: c.WGPUBuffer,
    rows: u32,
    cols: u32,

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    /// Allocate GPU buffer and upload `data`. `data.len` must equal rows*cols.
    pub fn load(
        gloc: *GpuAllocator,
        data: []const f32,
        rows: u32,
        cols: u32,
    ) !Mat {
        std.debug.assert(data.len == @as(usize, rows) * cols);
        const bytes = data.len * @sizeOf(f32);
        const buf = try gloc.makeBuffer(
            bytes,
            c.WGPUBufferUsage_Storage |
                c.WGPUBufferUsage_CopyDst |
                c.WGPUBufferUsage_CopySrc,
        );
        c.wgpuQueueWriteBuffer(gloc.queue, buf, 0, data.ptr, bytes);
        return .{ .buf = buf, .rows = rows, .cols = cols };
    }

    /// Allocate zeroed GPU buffer (no upload).
    pub fn zeros(gloc: *GpuAllocator, rows: u32, cols: u32) !Mat {
        const bytes: u64 = @as(u64, rows) * cols * @sizeOf(f32);
        const buf = try gloc.makeBuffer(
            bytes,
            c.WGPUBufferUsage_Storage |
                c.WGPUBufferUsage_CopyDst |
                c.WGPUBufferUsage_CopySrc,
        );
        return .{ .buf = buf, .rows = rows, .cols = cols };
    }

    pub fn deinit(self: Mat) void {
        c.wgpuBufferRelease(self.buf);
    }

    pub fn len(self: Mat) u32 {
        return self.rows * self.cols;
    }

    pub fn byteSize(self: Mat) u64 {
        return @as(u64, self.len()) * @sizeOf(f32);
    }

    /// Element-wise add. Shapes must match. Returns new Mat (caller owns).
    pub fn add(self: Mat, gloc: *GpuAllocator, other: Mat) !Mat {
        std.debug.assert(self.rows == other.rows and self.cols == other.cols);

        const result = try Mat.zeros(gloc, self.rows, self.cols);
        errdefer result.deinit();

        const pipeline = try gloc.pipAdd();
        try dispatch2in1out(gloc, pipeline, self.buf, other.buf, result.buf, self.byteSize(), self.len());

        return result;
    }

    /// Element-wise multiply by scalar. Returns new Mat (caller owns).
    pub fn scale(self: Mat, gloc: *GpuAllocator, scalar: f32) !Mat {
        const result = try Mat.zeros(gloc, self.rows, self.cols);
        errdefer result.deinit();

        const bytes = self.byteSize();
        const n = self.len();

        // Upload scalar as uniform buffer
        const uni_buf = try gloc.makeBuffer(
            @sizeOf(f32),
            c.WGPUBufferUsage_Uniform | c.WGPUBufferUsage_CopyDst,
        );
        defer c.wgpuBufferRelease(uni_buf);
        c.wgpuQueueWriteBuffer(gloc.queue, uni_buf, 0, &scalar, @sizeOf(f32));

        const pipeline = try gloc.pipScale();
        const bgl = c.wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
        defer c.wgpuBindGroupLayoutRelease(bgl);

        const entries = [_]c.WGPUBindGroupEntry{
            .{ .binding = 0, .buffer = self.buf, .offset = 0, .size = bytes },
            .{ .binding = 1, .buffer = result.buf, .offset = 0, .size = bytes },
            .{ .binding = 2, .buffer = uni_buf, .offset = 0, .size = @sizeOf(f32) },
        };
        try submitPass(gloc, pipeline, &entries, n);

        return result;
    }

    /// Read GPU buffer back to CPU. `out.len` must be >= rows*cols.
    pub fn read(self: Mat, gloc: *GpuAllocator, out: []f32) !void {
        std.debug.assert(out.len >= self.len());
        const bytes = self.byteSize();

        const staging = try gloc.makeBuffer(
            bytes,
            c.WGPUBufferUsage_MapRead | c.WGPUBufferUsage_CopyDst,
        );
        defer c.wgpuBufferRelease(staging);

        // Copy result → staging
        const enc = c.wgpuDeviceCreateCommandEncoder(gloc.device, null) orelse
            return error.Encoder;
        c.wgpuCommandEncoderCopyBufferToBuffer(enc, self.buf, 0, staging, 0, bytes);
        const cmd = c.wgpuCommandEncoderFinish(enc, null);
        defer c.wgpuCommandEncoderRelease(enc);
        defer c.wgpuCommandBufferRelease(cmd);
        c.wgpuQueueSubmit(gloc.queue, 1, &cmd);

        // Map and copy to slice
        var mapped = false;
        _ = c.wgpuBufferMapAsync(
            staging,
            c.WGPUMapMode_Read,
            0,
            bytes,
            .{ .callback = onMapped, .userdata1 = &mapped },
        );
        while (!mapped) gloc.poll();

        const ptr: [*]const f32 = @ptrCast(@alignCast(
            c.wgpuBufferGetConstMappedRange(staging, 0, bytes),
        ));
        @memcpy(out[0..self.len()], ptr[0..self.len()]);
        c.wgpuBufferUnmap(staging);
    }
};

// ── Dispatch helpers ──────────────────────────────────────────────────────────

/// Encode + submit a 2-input, 1-output compute pass (used by add).
fn dispatch2in1out(
    gloc: *GpuAllocator,
    pipeline: c.WGPUComputePipeline,
    buf_a: c.WGPUBuffer,
    buf_b: c.WGPUBuffer,
    buf_out: c.WGPUBuffer,
    bytes: u64,
    n: u32,
) !void {
    const bgl = c.wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
    defer c.wgpuBindGroupLayoutRelease(bgl);

    const entries = [_]c.WGPUBindGroupEntry{
        .{ .binding = 0, .buffer = buf_a, .offset = 0, .size = bytes },
        .{ .binding = 1, .buffer = buf_b, .offset = 0, .size = bytes },
        .{ .binding = 2, .buffer = buf_out, .offset = 0, .size = bytes },
    };
    try submitPass(gloc, pipeline, &entries, n);
}

/// Create bind group, encode pass, submit. workgroup_size=64.
fn submitPass(
    gloc: *GpuAllocator,
    pipeline: c.WGPUComputePipeline,
    entries: []const c.WGPUBindGroupEntry,
    n: u32,
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
    c.wgpuComputePassEncoderDispatchWorkgroups(pass, ceilDiv(n, 64), 1, 1);
    c.wgpuComputePassEncoderEnd(pass);
    c.wgpuComputePassEncoderRelease(pass);

    const cmd = c.wgpuCommandEncoderFinish(enc, null);
    defer c.wgpuCommandEncoderRelease(enc);
    defer c.wgpuCommandBufferRelease(cmd);
    c.wgpuQueueSubmit(gloc.queue, 1, &cmd);
}
