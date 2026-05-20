const std = @import("std");
const gpu = @import("lib.zig");
const c = @import("utils.zig").c;
const sv = @import("utils.zig").sv;
const GpuDevice = gpu.GpuDevice;
const GpuArena = gpu.GpuArena;
const GpuBuffer = gpu.GpuBuffer;
const GpuRender = gpu.GpuRender;
const GpuTexture = gpu.GpuTexture;

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    // 1. Open the raw headless GPU Device you shared
    const device = try GpuDevice.init(.{});
    defer device.deinit();

    var grena = GpuArena.init(allocator, device);
    defer grena.deinit();
    const gloc = grena.gpuAllocator();

    const width: u32 = 512;
    const height: u32 = 512;

    // 2. Load our Render Pipeline (Procedural Triangle Strip)
    const circle_rp = try GpuRender.init(
        device,
        @embedFile("shaders/circle.wgsl"),
        .{
            .bindings = &.{},
            .texture_format = .RGBA8Unorm,
            .topology = c.WGPUPrimitiveTopology_TriangleStrip,
        },
    );
    defer circle_rp.deinit();

    // 3. Create the offscreen VRAM texture to render into
    const texture = try GpuTexture.init(
        gloc,
        .RGBA8Unorm,
        .{ .width = width, .height = height, .depthOrArrayLayers = 1 },
        .initMany(&.{ .RenderAttachment, .CopySrc }),
    );
    defer texture.deinit();

    const target_view = c.wgpuTextureCreateView(texture.raw, null) orelse return error.View;
    defer c.wgpuTextureViewRelease(target_view);

    // 4. Create a staging buffer to pull pixels from VRAM to CPU
    // 4 bytes per pixel (RGBA8)
    const row_bytes = width * 4;
    const buffer_bytes = row_bytes * height;

    // Create a regular GpuBuffer set up to receive texture copy transfers
    const cpu_staging_buf = try GpuBuffer.init(gloc, buffer_bytes, .initMany(&.{ .CopyDst, .CopySrc }));

    // 5. Draw the Circle Frame into the texture view!
    try circle_rp.draw(gloc, target_view, 4, .{});

    // 6. Copy the texture data into our CPU staging buffer
    const enc = c.wgpuDeviceCreateCommandEncoder(device.device, null) orelse return error.Encoder;
    defer c.wgpuCommandEncoderRelease(enc);

    const src_copy = c.WGPUTexelCopyTextureInfo{
        .texture = texture.raw,
        .mipLevel = 0,
        .origin = .{ .x = 0, .y = 0, .z = 0 },
        .aspect = c.WGPUTextureAspect_All,
    };
    const dst_copy = c.WGPUTexelCopyBufferInfo{
        .buffer = cpu_staging_buf.raw,
        .layout = .{
            .offset = 0,
            .bytesPerRow = row_bytes,
            .rowsPerImage = height,
        },
    };
    const copy_size = c.WGPUExtent3D{ .width = width, .height = height, .depthOrArrayLayers = 1 };

    c.wgpuCommandEncoderCopyTextureToBuffer(enc, &src_copy, &dst_copy, &copy_size);

    const cmd = c.wgpuCommandEncoderFinish(enc, null);
    defer c.wgpuCommandBufferRelease(cmd);
    c.wgpuQueueSubmit(device.queue, 1, &cmd);

    // 7. Map and read the raw image bytes back to CPU
    // (This uses whatever slice-reading helpers your `GpuBuffer` wrapper provides)
    const pixels = try cpu_staging_buf.read(allocator, u8);
    defer allocator.free(pixels);

    // Now you have the raw binary image data! Let's output a simple Netpbm PPM image file
    // so you can actually open and look at your rendered circle.
    try savePpm(init.io, "circle.ppm", width, height, pixels);
    std.debug.print("Successfully rendered circle to circle.ppm!\n", .{});
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
