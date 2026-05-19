const std = @import("std");
const gpu = @import("gpu");
const GpuDevice = gpu.GpuDevice;
const GpuArena = gpu.GpuArena;
const GpuBuffer = gpu.GpuBuffer;
const GpuCompute = gpu.GpuCompute;

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    // 1. Open GPU Device
    const device = try GpuDevice.init(.{});
    defer device.deinit();

    // 2. Create a GPU Arena to manage VRAM
    var grena = GpuArena.init(allocator, device);
    defer grena.deinit();
    const gloc = grena.gpuAllocator();

    // 3. Load the WGSL compute pipeline
    const add_cp = try GpuCompute.init(
        device,
        @embedFile("shaders/add.wgsl"),
        .{ .bindings = &.{
            .{ .element_size = @sizeOf(f16) },
            .{ .element_size = @sizeOf(f16) },
            .{ .element_size = @sizeOf(f16) },
        } },
    );
    defer add_cp.deinit();

    // 4. Setup CPU data
    const len: usize = 16;
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
    const buf_a = try GpuBuffer.init(gloc, byte_size, .initMany(&.{ .Storage, .CopyDst, .CopySrc }));
    const buf_b = try GpuBuffer.init(gloc, byte_size, .initMany(&.{ .Storage, .CopyDst, .CopySrc }));
    const buf_out = try GpuBuffer.init(gloc, byte_size, .initMany(&.{ .Storage, .CopyDst, .CopySrc }));

    // Note: The buffers are safely tied to the GpuArena which will automatically
    // release them at the end. You can also manually call buf_x.deinit() if desired.

    // 6. Transfer data from CPU slices to GPU Buffers
    try buf_a.load(f16, data_a);
    try buf_b.load(f16, data_b);

    // 7. Dispatch the Compute Process
    try add_cp.run(gloc, .{ buf_a, buf_b, buf_out });

    // 8. Map and copy the resulting buffer back to the CPU
    const out = try buf_out.read(allocator, f16);
    defer allocator.free(out);

    std.debug.print("Result: {any}\n", .{out});
}
