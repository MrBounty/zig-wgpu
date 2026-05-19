const std = @import("std");
const c = @import("utils.zig").c;
const sv = @import("utils.zig").sv;
const GpuAllocator = @import("GpuAllocator.zig");
const GpuBuffer = @import("GpuBuffer.zig");
const GpuDevice = @import("GpuDevice.zig");

pub const Binding = struct {
    /// Element size in bytes for this binding. E.g. @sizeOf(f32).
    /// If 0, no element-based size validation is performed for this buffer.
    element_size: u32 = 0,
};

pub const ProcessDef = struct {
    bindings: []const Binding,
    workgroup_size: u32 = 256,
    max_workgroups: u32 = 65535,
    /// If true, automatically adds a Uniform Buffer containing `elements_count` as a `u32`
    /// to the next available binding slot.
    append_info_buffer: bool = true,
};

pip: c.WGPUComputePipeline,
def: ProcessDef,

pub fn init(device: GpuDevice, wgsl: []const u8, def: ProcessDef) !@This() {
    var wgsl_src = c.WGPUShaderSourceWGSL{
        .chain = .{ .sType = c.WGPUSType_ShaderSourceWGSL },
        .code = sv(wgsl),
    };
    const shader = c.wgpuDeviceCreateShaderModule(device.device, &.{
        .nextInChain = @ptrCast(&wgsl_src),
    }) orelse return error.Shader;
    defer c.wgpuShaderModuleRelease(shader);

    const pip = c.wgpuDeviceCreateComputePipeline(device.device, &.{
        .compute = .{ .module = shader, .entryPoint = sv("main") },
    }) orelse return error.Pipeline;

    return .{
        .pip = pip,
        .def = def,
    };
}

pub fn deinit(self: @This()) void {
    c.wgpuComputePipelineRelease(self.pip);
}

/// Execute the compute pass with arbitrary buffer bindings via a tuple.
/// Example: `try proc.run(gloc, .{ buf_a, buf_b, buf_out });`
pub fn run(
    self: @This(),
    gloc: GpuAllocator,
    args: anytype,
) !void {
    const type_info = @typeInfo(@TypeOf(args));
    if (type_info != .@"struct" or !type_info.@"struct".is_tuple)
        @compileError("Expected a tuple of GpuBuffers for args. E.g. .{ buf_a, buf_b }");

    const fields = type_info.@"struct".fields;
    if (fields.len != self.def.bindings.len) {
        std.log.err("Process expected {d} arguments, got {d}", .{ self.def.bindings.len, fields.len });
        return error.InvalidArgumentCount;
    }

    var elements_count: u32 = 0;

    // Infer elements_count from the first arg with a defined element_size
    inline for (fields, 0..) |field, i| {
        if (elements_count == 0) {
            const buf = @field(args, field.name);
            const el_size = self.def.bindings[i].element_size;
            if (el_size > 0) {
                elements_count = @intCast(buf.size / el_size);
            }
        }
    }

    // Validate runtime buffer sizes before dispatching
    inline for (fields, 0..) |field, i| {
        const buf = @field(args, field.name);
        const el_size = self.def.bindings[i].element_size;
        if (el_size > 0) {
            const expected_min_bytes = @as(u64, elements_count) * el_size;
            if (buf.size < expected_min_bytes) {
                std.log.err("Argument {d} size mismatch: expected at least {d} bytes, got {d}", .{ i, expected_min_bytes, buf.size });
                return error.BufferTooSmall;
            }
        }
    }

    var entries_buf: [32]c.WGPUBindGroupEntry = undefined;
    var entry_count: usize = 0;

    // Unpack tuple into WebGPU BindGroupEntries
    inline for (fields, 0..) |field, i| {
        const buf = @field(args, field.name);
        if (@TypeOf(buf) != GpuBuffer) {
            @compileError("All arguments in the tuple must be of type GpuBuffer");
        }
        entries_buf[entry_count] = .{
            .binding = @intCast(i),
            .buffer = buf.raw,
            .offset = 0,
            .size = buf.size, // Size exposes the fully allocated length
        };
        entry_count += 1;
    }

    // Optional uniform dispatch buffer appended at the end
    var info_buf: ?GpuBuffer = null;
    defer if (info_buf) |b| b.deinit();

    if (self.def.append_info_buffer) {
        info_buf = try GpuBuffer.init(
            gloc,
            @sizeOf(u32),
            .initMany(&.{ .Uniform, .CopyDst }),
        );
        c.wgpuQueueWriteBuffer(gloc.device.queue, info_buf.?.raw, 0, &elements_count, @sizeOf(u32));

        entries_buf[entry_count] = .{
            .binding = @intCast(entry_count),
            .buffer = info_buf.?.raw,
            .offset = 0,
            .size = @sizeOf(u32),
        };
        entry_count += 1;
    }

    const entries = entries_buf[0..entry_count];
    try submitPass(gloc, self.pip, entries, elements_count, self.def.workgroup_size, self.def.max_workgroups);
}

fn submitPass(
    gloc: GpuAllocator,
    pipeline: c.WGPUComputePipeline,
    entries: []const c.WGPUBindGroupEntry,
    n: usize,
    workgroup_size: u32,
    max_workgroups: u32,
) !void {
    if (n == 0) return;

    const bgl = c.wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
    defer c.wgpuBindGroupLayoutRelease(bgl);

    const bg = c.wgpuDeviceCreateBindGroup(gloc.device.device, &.{
        .layout = bgl,
        .entries = entries.ptr,
        .entryCount = entries.len,
    }) orelse return error.BindGroup;
    defer c.wgpuBindGroupRelease(bg);

    const enc = c.wgpuDeviceCreateCommandEncoder(gloc.device.device, null) orelse return error.Encoder;
    const pass = c.wgpuCommandEncoderBeginComputePass(enc, null);
    c.wgpuComputePassEncoderSetPipeline(pass, pipeline);
    c.wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, null);

    const desired_workgroups = ceilDiv(n, workgroup_size);
    const dispatch_count = @min(desired_workgroups, max_workgroups);

    c.wgpuComputePassEncoderDispatchWorkgroups(pass, @intCast(dispatch_count), 1, 1);
    c.wgpuComputePassEncoderEnd(pass);
    c.wgpuComputePassEncoderRelease(pass);

    const cmd = c.wgpuCommandEncoderFinish(enc, null);
    defer c.wgpuCommandEncoderRelease(enc);
    defer c.wgpuCommandBufferRelease(cmd);
    c.wgpuQueueSubmit(gloc.device.queue, 1, &cmd);
}

fn ceilDiv(n: usize, d: usize) usize {
    return (n + d - 1) / d;
}
