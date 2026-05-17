const std = @import("std");
const GpuDevice = @import("GpuDevice.zig");
const c = @import("c.zig").c;

raw: c.WGPUComputePipeline,

pub fn init(device: GpuDevice, wgsl: []const u8) !@This() {
    var wgsl_src = c.WGPUShaderSourceWGSL{
        .chain = .{ .sType = c.WGPUSType_ShaderSourceWGSL },
        .code = sv(wgsl),
    };
    const shader = c.wgpuDeviceCreateShaderModule(device.device, &.{
        .nextInChain = @ptrCast(&wgsl_src),
    }) orelse return error.Shader;
    defer c.wgpuShaderModuleRelease(shader);

    return .{ .raw = c.wgpuDeviceCreateComputePipeline(device.device, &.{
        .compute = .{ .module = shader, .entryPoint = sv("main") },
    }) orelse return error.Pipeline };
}

pub fn deinit(self: @This()) void {
    c.wgpuComputePipelineRelease(self.raw);
}

fn sv(s: []const u8) c.WGPUStringView {
    return .{ .data = s.ptr, .length = s.len };
}
