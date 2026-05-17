const std = @import("std");
const c = @import("c.zig").c;

const Ctx = struct {
    adapter: c.WGPUAdapter = null,
    device: c.WGPUDevice = null,
};

const GpuAllocator = @This();

instance: c.WGPUInstance,
adapter: c.WGPUAdapter,
device: c.WGPUDevice,
queue: c.WGPUQueue,
limits: c.WGPULimits,

config: struct {
    vram_bytes_limit: u64 = 10 * 1024 * 1024 * 1024, // 10 GB
} = .{},

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

    return .{
        .instance = instance,
        .adapter = adapter,
        .device = device,
        .queue = c.wgpuDeviceGetQueue(device),
        .limits = supported_limits,
    };
}

pub fn deinit(self: GpuAllocator) void {
    c.wgpuQueueRelease(self.queue);
    c.wgpuDeviceRelease(self.device);
    c.wgpuAdapterRelease(self.adapter);
    c.wgpuInstanceRelease(self.instance);
}

pub fn poll(self: *GpuAllocator) void {
    _ = c.wgpuDevicePoll(self.device, 1, null);
}

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

fn sv(s: []const u8) c.WGPUStringView {
    return .{ .data = s.ptr, .length = s.len };
}
