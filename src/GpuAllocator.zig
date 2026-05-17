const std = @import("std");
const sh = @import("shaders.zig");
const GpuDevice = @import("GpuDevice.zig");
const c = @import("c.zig").c;

const GpuAllocator = @This();

device: GpuDevice,
cpu_allocator: std.mem.Allocator,
tracked_buffers: std.AutoHashMap(c.WGPUBuffer, void),
pipelines: struct {
    add: c.WGPUComputePipeline,
},

pub fn init(cpu_allocator: std.mem.Allocator, device: GpuDevice) !GpuAllocator {
    return .{
        .device = device,
        .cpu_allocator = cpu_allocator,
        .tracked_buffers = .init(cpu_allocator),
        .pipelines = .{
            .add = try buildPipeline(device.device, sh.SHADER_ADD),
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
}

pub fn registerBuffer(
    self: *GpuAllocator,
    bytes: u64,
    usage: c.WGPUBufferUsage,
) !c.WGPUBuffer {
    const buf = c.wgpuDeviceCreateBuffer(self.device.device, &.{
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
