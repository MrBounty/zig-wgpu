const std = @import("std");
const c = @import("utils.zig").c;
const sv = @import("utils.zig").sv;
const GpuAllocator = @import("GpuAllocator.zig");
const GpuBuffer = @import("GpuBuffer.zig");
const GpuDevice = @import("GpuDevice.zig");
const GpuTextureFormat = @import("lib.zig").GpuTextureFormat;

pub const Binding = struct {
    element_size: u32 = 0,
};

pub const GpuRenderDef = struct {
    bindings: []const Binding = &.{},
    /// The surface texture format we are rendering to (e.g., BGRA8Unorm)
    texture_format: GpuTextureFormat,
    /// The names of the entry points inside your WGSL code
    vertex_entry: []const u8 = "vs_main",
    fragment_entry: []const u8 = "fs_main",
    /// Primitive topology, default to triangle list
    topology: GpuPrimitiveTopology = .TriangleList,
};

const GpuPrimitiveTopology = enum(c_uint) {
    Undefined = 0x00000000,
    PointList = 0x00000001,
    LineList = 0x00000002,
    LineStrip = 0x00000003,
    TriangleList = 0x00000004,
    TriangleStrip = 0x00000005,
    Force32 = 0x7FFFFFFF,
};

pip: c.WGPURenderPipeline,
def: GpuRenderDef,

pub fn init(device: GpuDevice, wgsl: []const u8, def: GpuRenderDef) !@This() {
    var wgsl_src = c.WGPUShaderSourceWGSL{
        .chain = .{ .sType = c.WGPUSType_ShaderSourceWGSL },
        .code = sv(wgsl),
    };
    const shader = c.wgpuDeviceCreateShaderModule(device.device, &.{
        .nextInChain = @ptrCast(&wgsl_src),
    }) orelse return error.Shader;
    defer c.wgpuShaderModuleRelease(shader);

    // 1. Setup the Color Target State (where the fragment shader outputs)
    const blend = c.WGPUBlendState{
        .color = .{ .operation = c.WGPUBlendOperation_Add, .srcFactor = c.WGPUBlendFactor_SrcAlpha, .dstFactor = c.WGPUBlendFactor_OneMinusSrcAlpha },
        .alpha = .{ .operation = c.WGPUBlendOperation_Add, .srcFactor = c.WGPUBlendFactor_One, .dstFactor = c.WGPUBlendFactor_Zero },
    };

    const color_target = c.WGPUColorTargetState{
        .format = @intFromEnum(def.texture_format),
        .blend = &blend,
        .writeMask = c.WGPUColorWriteMask_All,
    };

    // 2. Setup the Fragment State
    const fragment_state = c.WGPUFragmentState{
        .module = shader,
        .entryPoint = sv(def.fragment_entry),
        .targetCount = 1,
        .targets = &color_target,
    };

    // 3. Compile the Complete Render Pipeline
    const pip = c.wgpuDeviceCreateRenderPipeline(device.device, &.{
        .vertex = .{
            .module = shader,
            .entryPoint = sv(def.vertex_entry),
        },
        .primitive = .{
            .topology = @intFromEnum(def.topology),
            .stripIndexFormat = c.WGPUIndexFormat_Undefined,
            .frontFace = c.WGPUFrontFace_CCW,
            .cullMode = c.WGPUCullMode_None,
        },
        .multisample = .{
            .count = 1,
            .mask = 0xFFFFFFFF,
            .alphaToCoverageEnabled = 0,
        },
        .fragment = &fragment_state,
    }) orelse return error.Pipeline;

    return .{
        .pip = pip,
        .def = def,
    };
}

pub fn deinit(self: @This()) void {
    c.wgpuRenderPipelineRelease(self.pip);
}

/// Execute the render pass targeting a specific frame texture view.
/// Passes bind groups via a tuple exactly like your original compute setup.
pub fn draw(
    self: @This(),
    gloc: GpuAllocator,
    target_view: c.WGPUTextureView,
    vertex_count: u32,
    args: anytype,
) !void {
    const type_info = @typeInfo(@TypeOf(args));
    if (type_info != .@"struct" or !type_info.@"struct".is_tuple)
        @compileError("Expected a tuple of GpuBuffers for args. E.g. .{ uniform_buf }");

    const fields = type_info.@"struct".fields;
    if (fields.len != self.def.bindings.len)
        return error.InvalidArgumentCount;

    var entries_buf: [32]c.WGPUBindGroupEntry = undefined;

    inline for (fields, 0..) |field, i| {
        const buf = @field(args, field.name);
        if (@TypeOf(buf) != GpuBuffer) {
            @compileError("All arguments in the tuple must be of type GpuBuffer");
        }
        entries_buf[i] = .{
            .binding = @intCast(i),
            .buffer = buf.raw,
            .offset = 0,
            .size = buf.size,
        };
    }

    const entries = entries_buf[0..fields.len];

    // Create Render Bind Group from layout
    const bgl = c.wgpuRenderPipelineGetBindGroupLayout(self.pip, 0);
    defer c.wgpuBindGroupLayoutRelease(bgl);

    const bg = c.wgpuDeviceCreateBindGroup(gloc.device.device, &.{
        .layout = bgl,
        .entries = entries.ptr,
        .entryCount = @intCast(entries.len),
    }) orelse return error.BindGroup;
    defer c.wgpuBindGroupRelease(bg);

    // Encode Render Command
    const enc = c.wgpuDeviceCreateCommandEncoder(gloc.device.device, null) orelse return error.Encoder;
    defer c.wgpuCommandEncoderRelease(enc);

    const color_attachment = c.WGPURenderPassColorAttachment{
        .view = target_view,
        .resolveTarget = null,
        .loadOp = c.WGPULoadOp_Clear,
        .storeOp = c.WGPUStoreOp_Store,
        .clearValue = .{ .r = 0.1, .g = 0.1, .b = 0.1, .a = 1.0 },
        .depthSlice = c.WGPU_DEPTH_SLICE_UNDEFINED,
    };

    const pass_desc = c.WGPURenderPassDescriptor{
        .colorAttachmentCount = 1,
        .colorAttachments = &color_attachment,
        .depthStencilAttachment = null,
    };

    const pass = c.wgpuCommandEncoderBeginRenderPass(enc, &pass_desc);
    c.wgpuRenderPassEncoderSetPipeline(pass, self.pip);

    if (fields.len > 0) {
        c.wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 0, null);
    }

    // Draw! (Instead of Compute Dispatch)
    c.wgpuRenderPassEncoderDraw(pass, vertex_count, 1, 0, 0);

    c.wgpuRenderPassEncoderEnd(pass);
    c.wgpuRenderPassEncoderRelease(pass);

    const cmd = c.wgpuCommandEncoderFinish(enc, null);
    defer c.wgpuCommandBufferRelease(cmd);

    c.wgpuQueueSubmit(gloc.device.queue, 1, &cmd);
}
