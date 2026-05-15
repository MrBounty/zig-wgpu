const std = @import("std");
const sh = @import("shaders.zig");
const c = @import("c.zig").c;

const GpuAllocator = @This();

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

pub fn pipAdd(self: *GpuAllocator) !c.WGPUComputePipeline {
    if (self._pip_add == null)
        self._pip_add = try buildPipeline(self.device, sh.SHADER_ADD);
    return self._pip_add.?;
}

pub fn pipScale(self: *GpuAllocator) !c.WGPUComputePipeline {
    if (self._pip_scale == null)
        self._pip_scale = try buildPipeline(self.device, sh.SHADER_SCALE);
    return self._pip_scale.?;
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
    std.debug.print("{?}", .{device});
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
