const std = @import("std");
const sh = @import("shaders.zig");
const c = @import("c.zig").c;

const GpuAllocator = @This();

cpu_allocator: std.mem.Allocator,
instance: c.WGPUInstance,
adapter: c.WGPUAdapter,
device: c.WGPUDevice,
queue: c.WGPUQueue,

tracked_buffers: std.AutoHashMap(c.WGPUBuffer, void),

// Lazily created, cached for lifetime of allocator
_pip_add: c.WGPUComputePipeline = null,
_pip_scale: c.WGPUComputePipeline = null,

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

    _ = c.wgpuAdapterRequestDevice(
        adapter,
        null,
        .{ .callback = onDevice, .userdata1 = &ctx },
    );
    c.wgpuInstanceProcessEvents(instance);
    const device = ctx.device orelse return error.NoDevice;

    return .{
        .cpu_allocator = cpu_allocator,
        .instance = instance,
        .adapter = adapter,
        .device = device,
        .queue = c.wgpuDeviceGetQueue(device),
        .tracked_buffers = .init(cpu_allocator),
    };
}

pub fn deinit(self: *GpuAllocator) void {
    if (self._pip_add) |p| c.wgpuComputePipelineRelease(p);
    if (self._pip_scale) |p| c.wgpuComputePipelineRelease(p);

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
