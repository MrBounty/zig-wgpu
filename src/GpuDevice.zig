const std = @import("std");
const c = @import("utils.zig").c;
const sv = @import("utils.zig").sv;
const GpuAllocator = @import("GpuAllocator.zig");
const GpuTextureFormat = @import("lib.zig").GpuTextureFormat;

// TODO: Make Allocator more zig like
//  - GpuDevice can return a GpuAllocator that just allocate and nothing else
//  - From this GpuAllocator, can create a GpuArena like std.heap.ArenaAllocator.init(allocator)
//  - Rename GpuArenaAllocator too

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

// Allocation stuff

/// Returns the type-erased immutable interface wrapper
pub fn gpuAllocator(self: *const @This()) GpuAllocator {
    return .{
        .device = self.*,
        .ptr = @ptrCast(@constCast(self)),
        .vtable = &.{
            .allocBuffer = allocBuffer,
            .freeBuffer = freeBuffer,
            .allocTexture = allocTexture,
            .freeTexture = freeTexture,
            .allocTextureView = allocTextureView,
            .freeTextureView = freeTextureView,
            .allocRenderPipeline = allocRenderPipeline,
            .freeRenderPipeline = freeRenderPipeline,
            .allocComputePipeline = allocComputePipeline,
            .freeComputePipeline = freeComputePipeline,
        },
    };
}

fn allocBuffer(ctx: *anyopaque, desc: c.WGPUBufferDescriptor) anyerror!c.WGPUBuffer {
    const self: *@This() = @ptrCast(@alignCast(ctx));
    if (desc.size > self.limits.maxBufferSize)
        return error.SingleBufferExceedsLimit;
    return c.wgpuDeviceCreateBuffer(self.device, &desc) orelse return error.BufferAlloc;
}

fn freeBuffer(_: *anyopaque, raw: c.WGPUBuffer) void {
    c.wgpuBufferDestroy(raw);
    c.wgpuBufferRelease(raw);
}

fn allocTexture(ctx: *anyopaque, desc: c.WGPUTextureDescriptor) anyerror!c.WGPUTexture {
    const self: *@This() = @ptrCast(@alignCast(ctx));
    const format: GpuTextureFormat = @enumFromInt(desc.format);
    if (desc.size.width * desc.size.height * format.bytesPerPixel() > self.limits.maxBufferSize)
        return error.SingleBufferExceedsLimit;
    return c.wgpuDeviceCreateTexture(self.device, &desc) orelse return error.Texture;
}

fn freeTexture(_: *anyopaque, raw: c.WGPUTexture) void {
    c.wgpuTextureRelease(raw);
}

fn allocTextureView(_: *anyopaque, texture: c.WGPUTexture, desc: c.WGPUTextureViewDescriptor) anyerror!c.WGPUTextureView {
    return c.wgpuTextureCreateView(texture, &desc) orelse return error.View;
}

fn freeTextureView(_: *anyopaque, raw: c.WGPUTextureView) void {
    c.wgpuTextureViewRelease(raw);
}

fn allocRenderPipeline(ctx: *anyopaque, desc: c.WGPURenderPipelineDescriptor) anyerror!c.WGPURenderPipeline {
    const self: *@This() = @ptrCast(@alignCast(ctx));
    return c.wgpuDeviceCreateRenderPipeline(self.device, &desc) orelse return error.Pipeline;
}

fn freeRenderPipeline(_: *anyopaque, raw: c.WGPURenderPipeline) void {
    c.wgpuRenderPipelineRelease(raw);
}

fn allocComputePipeline(ctx: *anyopaque, desc: c.WGPUComputePipelineDescriptor) anyerror!c.WGPUComputePipeline {
    const self: *@This() = @ptrCast(@alignCast(ctx));
    return c.wgpuDeviceCreateComputePipeline(self.device, &desc) orelse return error.Pipeline;
}

fn freeComputePipeline(_: *anyopaque, raw: c.WGPUComputePipeline) void {
    c.wgpuComputePipelineRelease(raw);
}
