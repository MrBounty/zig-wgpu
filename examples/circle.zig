const std = @import("std");
const gpu = @import("gpu");
const GpuDevice = gpu.GpuDevice;
const GpuArenaAllocator = gpu.GpuArenaAllocator;
const GpuBuffer = gpu.GpuBuffer;
const GpuRender = gpu.GpuRender;
const GpuTexture = gpu.GpuTexture;
const GpuTextureView = gpu.GpuTextureView;

const width: u32 = 512;
const height: u32 = 512;

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    // 1. Open GPU Device
    const device = try GpuDevice.init(.{});
    defer device.deinit();

    // 2. Init VRAM Arena
    var grena = GpuArenaAllocator.init(allocator, device.gpuAllocator());
    defer grena.deinit();
    const glloc = grena.gpuAllocator();

    // 3. Load Render Pipeline
    const circle_rp = try GpuRender.init(
        glloc,
        @embedFile("shaders/circle.wgsl"),
        .{ .bindings = &.{}, .texture_format = .RGBA8Unorm, .topology = .TriangleStrip },
    );
    defer circle_rp.deinit();

    // 4. Create VRAM texture to render into
    const texture = try GpuTexture.init(glloc, .{
        .format = .RGBA8Unorm,
        .size = .{ .width = width, .height = height, .depthOrArrayLayers = 1 },
        .usage = .initMany(&.{ .RenderAttachment, .CopySrc }),
    });
    defer texture.deinit();

    // 5. Create a view from texture
    const view = try GpuTextureView.init(glloc, texture, .{});
    defer view.deinit();

    // 6. Run the rendering pipeline
    try circle_rp.draw(glloc, view, 4, .{});

    // 7. Load Texture into GpuBuffer
    const cpu_staging_cpu = try texture.buffCopy(glloc);
    defer cpu_staging_cpu.deinit();

    // 8. Read GpuBuffer to CPU
    // This need to be free manually because CPU memory
    const pixels = try cpu_staging_cpu.read(allocator, u8);
    defer allocator.free(pixels);

    // 9. Write a simple ppm image
    try savePpm(init.io, "circle.ppm", width, height, pixels);
}

fn savePpm(io: std.Io, filename: []const u8, w: u32, h: u32, rgba_pixels: []const u8) !void {
    const file = try std.Io.Dir.cwd().createFile(io, filename, .{});
    defer file.close(io);

    var buf: [255]u8 = undefined;
    var writer = file.writer(io, &buf);

    // PPM Header: P6 format means raw RGB bytes
    try writer.interface.print("P6\n{d} {d}\n255\n", .{ w, h });

    // Strip Alpha channel when writing out to standard RGB PPM format
    var i: usize = 0;
    while (i < rgba_pixels.len) : (i += 4) {
        try writer.interface.writeAll(rgba_pixels[i .. i + 3]);
    }
}
