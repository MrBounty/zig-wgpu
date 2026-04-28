const std = @import("std");
const c = @cImport(@cInclude("wgpu.h"));

/// Replace enum_WGPURequestAdapterStatus
pub const RequestAdapterStatus = enum {
    Success,
    CallbackCancelled,
    Unavailable,
    Error,
    Force32,
};

pub const BufferUsage = enum(u64) {
    None = 0,
    MapRead = 1, // CPU can read after GPU finishes
    MapWrite = 2,
    CopySrc = 4, // can copy from GPU to staging.
    CopyDst = 8, // CPU can write to it
    Index = 16,
    Vertex = 32,
    Uniform = 64,
    Storage = 128,
    Indirect = 256,
    QueryResolve = 512,
};

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

const AllocatorGPU = @This();

allocator: std.mem.Allocator,
instance: *c.struct_WGPUInstanceImpl,
adapter: *c.struct_WGPUAdapterImpl,
device: c.struct_WGPUDeviceImpl,
queue: c.struct_WGPUQueueImpl,
ctx: Ctx,

buffers: std.AutoHashMap(*c.struct_WGPUBufferImpl, void),

pub fn init(allocator: std.mem.Allocator) !AllocatorGPU {
    var self: AllocatorGPU = undefined;
    self.allocator = allocator;
    self.ctx = .{};
    self.buffers = try .init(allocator);

    // 1. Instance ──────────────────────────────────────────────────────────────
    self.instance = c.wgpuCreateInstance(&std.mem.zeroes(c.WGPUInstanceDescriptor)) orelse
        return error.NoInstance;

    // 2. Adapter (async → poll) ────────────────────────────────────────────────
    _ = c.wgpuInstanceRequestAdapter(
        self.instance,
        &.{ .powerPreference = c.WGPUPowerPreference_HighPerformance },
        .{ .callback = onAdapter, .userdata1 = &self.ctx },
    );
    c.wgpuInstanceProcessEvents(self.instance); // drive callbacks
    self.adapter = self.ctx.adapter orelse return error.NoAdapter;

    // 3. Device ────────────────────────────────────────────────────────────────
    _ = c.wgpuAdapterRequestDevice(self.adapter, null, .{ .callback = onDevice, .userdata1 = &self.ctx });
    c.wgpuInstanceProcessEvents(self.instance);
    self.device = self.ctx.device orelse return error.NoDevice;

    self.queue = c.wgpuDeviceGetQueue(self.device);

    return self;
}

pub fn deinit(self: AllocatorGPU) void {
    c.wgpuInstanceRelease(self.instance);
    defer c.wgpuAdapterRelease(self.adapter);
    defer c.wgpuDeviceRelease(self.device);
    defer c.wgpuQueueRelease(self.queue);
}

pub fn addBuff(
    self: AllocatorGPU,
    comptime T: type,
    comptime len: comptime_int,
    comptime opt: struct {},
) !void {
    self.buffers.put(
        c.wgpuDeviceCreateBuffer(self.device, &.{
            .usage = c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst,
            .size = len * @bitSizeOf(T),
        }) orelse return error.Buffer,
        {},
    );
}
