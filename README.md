# Minimal Zig WebGPU Compute Library

This is a minimal, self-contained Zig library designed to simplify running compute shaders using WebGPU. It abstracts away much of the boilerplate required for GPU device initialization, memory management, and pipeline execution.

## Core Modules

The library exports five primary components:

* 
**`GpuDevice`**: Initializes the WebGPU instance, adapter, device, and queue. It is configured to prioritize high performance and automatically requests the `ShaderF16` feature if the adapter supports it. By default, it enforces a 2 GB VRAM limit.


* 
**`GpuArena` / `GpuAllocator**`: A memory management layer that tracks allocated VRAM bytes to prevent exceeding the device budget. The arena automatically destroys and releases all tracked WebGPU buffers when deinitialized.


* **`GpuBuffer`**: Wraps native WebGPU buffers. It automatically aligns buffer sizes forward to a multiple of 4 bytes. It provides a `.load()` method for CPU-to-GPU data transfers (handling both aligned and unaligned lengths smoothly) and a `.read()` method that utilizes a staging buffer to map GPU data back to the CPU.


* 
**`GpuProcess`**: Compiles WGSL source code into a compute pipeline. When running a process, it automatically splits the work into manageable chunks (up to 1 GB at a time) and dispatches workgroups of size 256.



## Quick Start Example

Below is a complete, self-contained example demonstrating how to initialize the GPU, load data, run a compute shader, and read the results back to the CPU:

```zig
const std = @import("std");
const GpuDevice = @import("GpuDevice.zig");
const GpuArena = @import("GpuArena.zig");
const GpuProcess = @import("GpuProcess.zig");
// Note: Assuming Vec is implemented via GpuBuffer as shown in example.zig

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    [cite_start]// 1. Open GPU Device [cite: 46]
    const device = try GpuDevice.init(.{});
    defer device.deinit();

    [cite_start]// 2. Create a GPU Arena to hold GPU memory [cite: 47]
    var grena = GpuArena.init(allocator, device);
    defer grena.deinit();
    [cite_start]const gloc = grena.gpuAllocator(); [cite: 48]

    [cite_start]// 3. Create a GPU process that loads the WGSL pipeline/shader [cite: 48]
    const add = try GpuProcess.init(device, @embedFile("shaders/add.wgsl"));
    [cite_start]defer add.deinit(); [cite: 49]

    [cite_start]// 4. Allocate and populate CPU memory [cite: 49, 50, 51]
    const data_a = try allocator.alloc(f16, 16);
    defer allocator.free(data_a);
    const data_b = try allocator.alloc(f16, 16);
    defer allocator.free(data_b);

    for (0..16) |i| {
        data_a[i] = @floatFromInt(i);
        data_b[i] = @floatFromInt(16 - 1 - i);
    }

    [cite_start]// 5. Allocate GPU memory (deinit handled automatically by grena) [cite: 52]
    const a = try Vec.initZero(gloc, 16);
    [cite_start]const b = try Vec.initZero(gloc, 16); [cite: 53]

    [cite_start]// 6. Load CPU -> GPU [cite: 53]
    try a.load(data_a);
    try b.load(data_b);

    [cite_start]// 7. Run GPU Pipeline [cite: 54]
    const sum = try a.run(gloc, b, add);

    [cite_start]// 8. Read GPU -> CPU [cite: 55]
    const out = try sum.read(allocator);
    defer allocator.free(out);

    [cite_start]std.debug.print("{any}\n", .{out}); [cite: 55]
}

```

## Dependencies

* 
**`wgpu.h`**: The library relies on the WebGPU C API headers to bind to the native system graphics.
