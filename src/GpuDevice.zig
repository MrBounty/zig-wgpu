const std = @import("std");
const c = @import("utils.zig").c;
const sv = @import("utils.zig").sv;

const Ctx = struct {
    adapter: c.WGPUAdapter = null,
    device: c.WGPUDevice = null,
};

const GpuDeviceConfig = struct {
    /// VRAM limit. Default 2 GB
    vram_bytes_limit: u64 = 2 * 1024 * 1024 * 1024,
    power_preference: enum(c_uint) {
        Undefined = 0x00000000,
        LowPower = 0x00000001,
        HighPerformance = 0x00000002,
        Force32 = 0x7FFFFFFF,
    } = .HighPerformance,
};

instance: c.WGPUInstance,
adapter: c.WGPUAdapter,
device: c.WGPUDevice,
queue: c.WGPUQueue,
limits: c.WGPULimits,

config: GpuDeviceConfig,

pub fn init(config: GpuDeviceConfig) !@This() {
    const instance = c.wgpuCreateInstance(
        &std.mem.zeroes(c.WGPUInstanceDescriptor),
    ) orelse return error.NoInstance;
    errdefer c.wgpuInstanceRelease(instance);

    var ctx = Ctx{};
    _ = c.wgpuInstanceRequestAdapter(
        instance,
        &.{ .powerPreference = @intFromEnum(config.power_preference) },
        .{ .callback = onAdapter, .userdata1 = &ctx },
    );
    c.wgpuInstanceProcessEvents(instance);
    const adapter = ctx.adapter orelse return error.NoAdapter;
    errdefer c.wgpuAdapterRelease(adapter);

    var supported_features = std.mem.zeroes(c.WGPUSupportedFeatures);
    c.wgpuAdapterGetFeatures(adapter, &supported_features);

    var supported_limits = std.mem.zeroes(c.WGPULimits);
    supported_limits.nextInChain = null;
    if (c.wgpuAdapterGetLimits(adapter, &supported_limits) != 1) return error.FailedToGetAdapterLimits;

    var has_f16 = false;
    for (0..supported_features.featureCount) |i| {
        if (supported_features.features[i] == c.WGPUFeatureName_ShaderF16) {
            has_f16 = true;
            break;
        }
    }

    var feature_buf = [_]c.WGPUFeatureName{c.WGPUFeatureName_ShaderF16};
    const required_features: []const c.WGPUFeatureName =
        if (has_f16) feature_buf[0..1] else &.{};

    const device_descriptor = c.WGPUDeviceDescriptor{
        .nextInChain = null,
        .label = sv("TensorCompilerDevice"),
        .requiredFeatureCount = required_features.len,
        .requiredFeatures = if (required_features.len > 0) required_features.ptr else null,
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
        .config = config,
    };
}

pub fn deinit(self: @This()) void {
    c.wgpuQueueRelease(self.queue);
    c.wgpuDeviceRelease(self.device);
    c.wgpuAdapterRelease(self.adapter);
    c.wgpuInstanceRelease(self.instance);
}

/// Wait for thing to be done
pub fn poll(self: @This()) void {
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
