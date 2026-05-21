const std = @import("std");
const gpu = @import("gpu");
const GpuDevice = gpu.GpuDevice;
const GpuArenaAllocator = gpu.GpuArenaAllocator;
const GpuBuffer = gpu.GpuBuffer;
const GpuCompute = gpu.GpuCompute;

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    // 1. Open GPU Device
    const device = try GpuDevice.init(.{});
    defer device.deinit();

    // 2. Create a GPU Arena to manage VRAM
    var grena = GpuArenaAllocator.init(allocator, device.gpuAllocator());
    defer grena.deinit();
    const gloc = grena.gpuAllocator();

    // 3. Load the WGSL compute pipeline
    const add_cp = try GpuCompute.init(
        gloc,
        @embedFile("shaders/add.wgsl"),
        .{
            .label = "add",
            .bindings = &.{
                .{ .element_size = @sizeOf(f16) },
                .{ .element_size = @sizeOf(f16) },
                .{ .element_size = @sizeOf(f16) },
            },
        },
    );

    // 4. Setup CPU data
    const len: usize = 1024;
    const data_a = try allocator.alloc(f16, len);
    defer allocator.free(data_a);
    const data_b = try allocator.alloc(f16, len);
    defer allocator.free(data_b);

    for (0..len) |i| {
        data_a[i] = @floatFromInt(i);
        data_b[i] = @floatFromInt(len - 1 - i);
    }

    // 5. Initialize raw GPU Buffers
    const byte_size = len * @sizeOf(f16);
    const buf_a = try GpuBuffer.init(gloc, .{ .label = "a", .size = byte_size, .usage = .initMany(&.{ .Storage, .CopyDst, .CopySrc }) });
    const buf_b = try GpuBuffer.init(gloc, .{ .label = "b", .size = byte_size, .usage = .initMany(&.{ .Storage, .CopyDst, .CopySrc }) });
    const buf_out = try GpuBuffer.init(gloc, .{ .label = "out", .size = byte_size, .usage = .initMany(&.{ .Storage, .CopyDst, .CopySrc }) });

    // Note: Buffers are safely tied to the GpuArenaAllocator which will automatically
    // release them at the end. You can also manually call buf_x.deinit() if desired.
    // This will also release pipelines, textures, ect. Everything using a GpuAllocator to init.

    // 6. Transfer data from CPU slices to GPU Buffers
    try buf_a.load(f16, data_a);
    try buf_b.load(f16, data_b);

    // 7. Dispatch the Compute
    try add_cp.run(gloc, .{ buf_a, buf_b, buf_out });

    // 8. Map and copy the resulting buffer back to the CPU
    const staging = try GpuBuffer.init(gloc, .{
        .size = byte_size,
        .usage = .initMany(&.{ .MapRead, .CopyDst }),
    });
    defer staging.deinit();

    try buf_out.copy(staging);
    const out = try staging.read(allocator, f16);
    defer allocator.free(out);

    std.debug.print("Result: {any}\n", .{out[0..@min(6, len)]});
}
