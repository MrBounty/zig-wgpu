# Minimal Zig WebGPU Compute & Render Library

This is a minimal, self-contained Zig library designed to simplify running compute shaders and rendering pipelines using WebGPU. It abstracts away much of the boilerplate required for GPU device initialization, memory management, bind groups, and pipeline execution.

## Core Modules

The library exports the following primary components:

* **`GpuDevice`**: Initializes the WebGPU instance, adapter, device, and queue. It is configured to prioritize high performance and automatically requests the `ShaderF16` feature if the adapter supports it. It provides the base `GpuAllocator` for raw VRAM allocations.
* **`GpuArenaAllocator`**: A memory management layer that wraps a base allocator to track and automatically destroy all allocated WebGPU buffers, textures, views, and pipelines when deinitialized.
* **`GpuBuffer`**: Wraps native WebGPU buffers. It provides a `.load()` method for CPU-to-GPU data transfers and a `.read()` method that utilizes a staging buffer to map GPU data back to the CPU.
* **`GpuCompute`**: Compiles WGSL source code into a compute pipeline and dispatches compute workgroups.
* **`GpuRender` / `GpuTexture` / `GpuTextureView`**: Components used to initialize render pipelines, set up render attachments (textures), and bind render targets for offscreen drawing.

---

## Example 1: Compute Pipeline

Below is a complete example demonstrating how to initialize the GPU via the device allocator, manage VRAM using a GPU Arena, run a compute shader, and read the results back to the CPU:

```zig
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
    const out = try buf_out.read(allocator, f16);
    defer allocator.free(out);

    std.debug.print("Result: {any}\n", .{out});
}
```

---

## Example 2: Rendering Pipeline (Offscreen to PPM Image)

This example demonstrates how to initialize a rendering pipeline, allocate an output texture target, draw primitives via WebGPU,
and pull the frame pixels back to the CPU to write a standard image file:

```zig
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
    const gloc = grena.gpuAllocator();

    // 3. Load Render Pipeline
    const circle_rp = try GpuRender.init(
        gloc,
        @embedFile("shaders/circle.wgsl"),
        .{ .bindings = &.{}, .texture_format = .RGBA8Unorm, .topology = .TriangleStrip },
    );
    defer circle_rp.deinit();

    // 4. Create VRAM texture to render into
    const texture = try GpuTexture.init(gloc, .{
        .format = .RGBA8Unorm,
        .size = .{ .width = width, .height = height, .depthOrArrayLayers = 1 },
        .usage = .initMany(&.{ .RenderAttachment, .CopySrc }),
    });
    defer texture.deinit();

    // 5. Create a view from texture
    const view = try GpuTextureView.init(gloc, texture, .{});
    defer view.deinit();

    // 6. Run the rendering pipeline
    try circle_rp.draw(gloc, view, 4, .{});

    // 7. Load Texture into GpuBuffer
    const cpu_staging_cpu = try texture.buffCopy(gloc);
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
```

---

## Running Examples Locally

If you have cloned the repository, you can run the included examples directly using the Zig build system:

```bash
# Run the rendering example (generates circle.ppm)
zig build circle

# Run the compute example
zig build compute

# Run the compute benchmark
zig build bench_cp
```

---

## Dependencies

* **`wgpu.h`**: The library relies on WebGPU C API headers to bind to the native system graphics.

## System Requirements

Because this library binds to native system graphics APIs via `wgpu-native`, you must ensure the appropriate development headers and libraries are available on your system before compiling.

### Linux (Vulkan)

* **Ubuntu / Debian:** `sudo apt update && sudo apt install libvulkan-dev mesa-vulkan-drivers`
* **Fedora / RHEL:** `sudo dnf install vulkan-devel mesa-vulkan-drivers`
* **Arch Linux:** `sudo pacman -S vulkan-headers vulkan-icd-loader`

### macOS (Metal)

No extra installation required. Automatically links against `Metal`, `QuartzCore`, `Foundation`, and `CoreGraphics`.

### Windows (DirectX 12)

No extra installation required. Automatically links against `d3d12`, `dxgi`, and `user32`. Ensure you have MSVC build tools installed.

---

## Adding to your project

Add it to your `build.zig.zon`:

```bash
zig fetch --save git+[https://git.bouvais.lu/adrien/zig-wgpu#ref=0.2.0](https://git.bouvais.lu/adrien/zig-wgpu)
```

Then, expose it in your `build.zig`:

```zig
const zig_wgpu = b.dependency("zig-wgpu", .{
    .target = target,
    .optimize = optimize,
});

exe.root_module.addImport("gpu", zig_wgpu.module("zig-wgpu"));
```
