const std = @import("std");
const sh = @import("shaders.zig");
const c = @import("c.zig").c;

pub const GpuConfig = struct {
    /// Absolute max footprint of a single Tensor buffer in bytes.
    max_tensor_buffer_bytes: u64,
    /// Absolute max slice size readable inside a single compute binding in bytes.
    max_tensor_binding_bytes: u64,
};

const GpuAllocator = @This();

cpu_allocator: std.mem.Allocator,
instance: c.WGPUInstance,
adapter: c.WGPUAdapter,
device: c.WGPUDevice,
queue: c.WGPUQueue,
config: GpuConfig,

tracked_buffers: std.AutoHashMap(c.WGPUBuffer, void),

pipelines: struct {
    add: c.WGPUComputePipeline,
},

pub fn init(cpu_allocator: std.mem.Allocator) !GpuAllocator {
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

    // --- QUERY HARDWARE LIMITS ---
    var supported_limits = std.mem.zeroes(c.WGPULimits);
    supported_limits.nextInChain = null;

    // Fetch what your physical graphic card can actually handle
    if (c.wgpuAdapterGetLimits(adapter, &supported_limits) != 1) return error.FailedToGetAdapterLimits;

    const device_descriptor = c.WGPUDeviceDescriptor{
        .nextInChain = null,
        .label = sv("TensorCompilerDevice"),
        .requiredFeatureCount = 0,
        .requiredFeatures = null,
        .requiredLimits = &supported_limits,
    };

    _ = c.wgpuAdapterRequestDevice(
        adapter,
        &device_descriptor,
        .{ .callback = onDevice, .userdata1 = &ctx },
    );
    c.wgpuInstanceProcessEvents(instance);
    const device = ctx.device orelse return error.NoDevice;

    // Package configurations into the struct
    const config = GpuConfig{
        .max_tensor_buffer_bytes = supported_limits.maxBufferSize,
        .max_tensor_binding_bytes = supported_limits.maxStorageBufferBindingSize,
    };

    return .{
        .cpu_allocator = cpu_allocator,
        .instance = instance,
        .adapter = adapter,
        .device = device,
        .queue = c.wgpuDeviceGetQueue(device),
        .config = config,
        .tracked_buffers = .init(cpu_allocator),
        .pipelines = .{
            .add = try buildPipeline(device, sh.SHADER_ADD),
        },
    };
}

pub fn deinit(self: *GpuAllocator) void {
    inline for (@typeInfo(@TypeOf(self.pipelines)).@"struct".fields) |field|
        c.wgpuComputePipelineRelease(@field(self.pipelines, field.name));

    var it = self.tracked_buffers.keyIterator();
    while (it.next()) |buf_ptr| {
        const buf = buf_ptr.*;
        c.wgpuBufferDestroy(buf);
        c.wgpuBufferRelease(buf);
    }
    self.tracked_buffers.deinit();

    c.wgpuQueueRelease(self.queue);
    c.wgpuDeviceRelease(self.device);
    c.wgpuAdapterRelease(self.adapter);
    c.wgpuInstanceRelease(self.instance);
}

pub fn registerBuffer(
    self: *GpuAllocator,
    bytes: u64,
    usage: c.WGPUBufferUsage,
) !c.WGPUBuffer {
    const buf = c.wgpuDeviceCreateBuffer(self.device, &.{
        .usage = usage,
        .size = bytes,
    }) orelse return error.BufferAlloc;

    try self.tracked_buffers.put(buf, {});
    return buf;
}

pub fn unregisterAndDestroyBuffer(self: *GpuAllocator, buf: c.WGPUBuffer) void {
    if (self.tracked_buffers.remove(buf)) {
        c.wgpuBufferDestroy(buf);
        c.wgpuBufferRelease(buf);
    }
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

/// Poll until GPU work completes. Use after submit if you need CPU sync.
pub fn poll(self: *GpuAllocator) void {
    _ = c.wgpuDevicePoll(self.device, 1, null);
}

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
}

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

fn sv(s: []const u8) c.WGPUStringView {
    return .{ .data = s.ptr, .length = s.len };
}
