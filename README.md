# Minimal Zig WebGPU Compute Library

This is a minimal, self-contained Zig library designed to simplify running compute shaders using WebGPU.
It abstracts away much of the boilerplate required for GPU device initialization, memory management, and pipeline execution.

## Core Modules

The library exports five primary components:

* **`GpuDevice`**: Initializes the WebGPU instance, adapter, device, and queue. It is configured to prioritize high performance and automatically requests the `ShaderF16` feature if the adapter supports it. By default, it enforces a 2 GB VRAM limit.
* **`GpuArena` / `GpuAllocator`**: A memory management layer that tracks allocated VRAM bytes to prevent exceeding the device budget. The arena automatically destroys and releases all tracked WebGPU buffers when deinitialized.
* **`GpuBuffer`**: Wraps native WebGPU buffers. It automatically aligns buffer sizes forward to a multiple of 4 bytes. It provides a `.load()` method for CPU-to-GPU data transfers (handling both aligned and unaligned lengths smoothly) and a `.read()` method that utilizes a staging buffer to map GPU data back to the CPU.
* **`GpuProcess`**: Compiles WGSL source code into a compute pipeline. When running a process, it automatically splits the work into manageable chunks (up to 1 GB at a time) and dispatches workgroups of size 256.

## Quick Start Example

Below is a complete, self-contained example demonstrating how to initialize the GPU, load data, run a compute shader, and read the results back to the CPU:

```zig
const std = @import("std");
const gpu = @import("gpu");
const GpuDevice = gpu.GpuDevice;
const GpuArena = gpu.GpuArena;
const GpuBuffer = gpu.GpuBuffer;
const GpuProcess = gpu.GpuProcess;

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
    const add_process = try GpuProcess.init(
        device,
        @embedFile("shaders/add.wgsl"),
        .{ .bindings = &.{
            .{ .element_size = @sizeOf(f16) },
            .{ .element_size = @sizeOf(f16) },
            .{ .element_size = @sizeOf(f16) },
        } },
    );
    defer add_process.deinit();

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
    try add_process.run(gloc, .{ buf_a, buf_b, buf_out });

    // 8. Map and copy the resulting buffer back to the CPU
    const out = try buf_out.read(allocator, f16);
    defer allocator.free(out);

    std.debug.print("Result: {any}\n", .{out});
}

```

## Dependencies

* **`wgpu.h`**: The library relies on the WebGPU C API headers to bind to the native system graphics.

## System Requirements

Because this library binds to native system graphics APIs via `wgpu-native`,
you must ensure the appropriate development headers and libraries are available on your system before compiling.

### Linux (Vulkan)
You need the Vulkan development headers to compile the project, and a Vulkan-compatible driver to run it. 

Depending on your distribution, install the following packages:
* **Ubuntu / Debian:**
  ```bash
  sudo apt update
  sudo apt install libvulkan-dev mesa-vulkan-drivers
  ```
* **Fedora / RHEL:**
  ```bash
  sudo dnf install vulkan-devel mesa-vulkan-drivers
  ```
* **Arch Linux:**
  ```bash
  sudo pacman -S vulkan-headers vulkan-icd-loader
  ```

### macOS (Metal)
No extra installation is required.
The build script automatically links against the standard Apple frameworks provided by the Xcode Command Line Tools
(`Metal`, `QuartzCore`, `Foundation`, `CoreGraphics`).

### Windows (DirectX 12)
No extra installation is required.
The build script automatically links against the standard Windows SDK libraries (`d3d12`, `dxgi`, `user32`).
Ensure you have the MSVC build tools installed.

---

## Adding to your project

To use this module in your own Zig project, add it to your `build.zig.zon`:
```bash
zig fetch --save git+https://git.bouvais.lu/adrien/zig-wgpu
```

Then, in your `build.zig`, import and add the module to your executable:
```zig
const zig_wgpu = b.dependency("zig-wgpu", .{
    .target = target,
    .optimize = optimize,
});

exe.root_module.addImport("gpu", zig_wgpu.module("zig-wgpu"));
```
