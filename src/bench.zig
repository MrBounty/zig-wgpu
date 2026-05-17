const std = @import("std");
const GpuDevice = @import("GpuDevice.zig");
const GpuAllocator = @import("GpuAllocator.zig");
const GpuPipeline = @import("GpuPipeline.zig");
const Vec = @import("Vec.zig");

const c = @import("utils.zig").c;

pub fn main(init: std.process.Init) !void {
    const device = try GpuDevice.init(.{ .vram_bytes_limit = 4 * 1024 * 1024 * 1024 });
    defer device.deinit();

    var gloc = try GpuAllocator.init(init.gpa, device);
    defer gloc.deinit();

    const add_pip = try GpuPipeline.init(device, @embedFile("shaders/add.wgsl"));
    defer add_pip.deinit();

    const allocator = init.gpa;

    // --- WARM-UP PHASE ---
    {
        var warmup_a = [_]f32{1.0};
        var warmup_b = [_]f32{1.0};
        const wa = try Vec.initLoad(&gloc, &warmup_a);
        defer wa.deinit();
        const wb = try Vec.initLoad(&gloc, &warmup_b);
        defer wb.deinit();
        const wsum = try wa.run(&gloc, wb, add_pip);
        defer wsum.deinit();
        const wout = try wsum.read(&gloc, allocator);
        defer allocator.free(wout);
    }

    const sizes = [_]usize{
        1,
        256,
        1024,
        4 * 1024,
        4 * 4 * 1024,
        4 * 4 * 4 * 1024,
        4 * 4 * 4 * 4 * 1024,
        1024 * 1024,
        // 4 * 1024 * 1024,
        // 4 * 4 * 1024 * 1024,
        // 4 * 4 * 4 * 1024 * 1024,
        // 4 * 4 * 4 * 4 * 1024 * 1024,
        // 4 * 4 * 4 * 4 * 4 * 1024 * 1024,
    };

    const iterations = 10;

    // Updated headers to include VRAM footprint info
    std.debug.print("\n| Size (MB) | Phase             | Time (ms)  |   GB/s   | VRAM Peak |\n", .{});
    std.debug.print("|----------:|:------------------|-----------:|---------:|----------:|\n", .{});

    for (sizes) |size| {
        // --- Phase 1: Host Init/Alloc (Outside the iteration loop for pure host prep) ---
        const data_a = try allocator.alloc(f32, size);
        defer allocator.free(data_a);
        const data_b = try allocator.alloc(f32, size);
        defer allocator.free(data_b);

        for (0..size) |i| {
            data_a[i] = @floatFromInt(i);
            data_b[i] = @floatFromInt(size - 1 - i);
        }

        // Track best times across iterations
        var min_alloc_ns: u64 = std.math.maxInt(u64);
        var min_transfer_ns: u64 = std.math.maxInt(u64);
        var min_compute_ns: u64 = std.math.maxInt(u64);

        // Track peak VRAM usage observed during the iterations
        var peak_vram_bytes: u64 = 0;

        for (0..iterations) |_| {
            // --- 1. GPU ALLOCATION PHASE ---
            const alloc_start = std.Io.Clock.awake.now(init.io);

            const a = try Vec.initLoad(&gloc, data_a);
            defer a.deinit();
            const b = try Vec.initLoad(&gloc, data_b);
            defer b.deinit();

            const alloc_duration = alloc_start.durationTo(std.Io.Clock.awake.now(init.io));
            const alloc_ns = @as(u64, @intCast(alloc_duration.toNanoseconds()));
            if (alloc_ns < min_alloc_ns) min_alloc_ns = alloc_ns;

            // --- 2. COMPUTE PHASE ---
            const compute_start = std.Io.Clock.awake.now(init.io);

            const sum = try a.run(&gloc, b, add_pip);
            defer sum.deinit();

            // All 3 buffers (a, b, sum) are currently resident in VRAM here.
            // Querying now catches the true peak allocation step.
            if (gloc.allocated_vram_bytes > peak_vram_bytes)
                peak_vram_bytes = gloc.allocated_vram_bytes;

            _ = c.wgpuDevicePoll(device.device, 1, null);

            const compute_duration = compute_start.durationTo(std.Io.Clock.awake.now(init.io));
            const compute_ns = @as(u64, @intCast(compute_duration.toNanoseconds()));
            if (compute_ns < min_compute_ns) min_compute_ns = compute_ns;

            // --- 3. TRANSFER PHASE (Device -> Host) ---
            const transfer_start = std.Io.Clock.awake.now(init.io);

            const out = try sum.read(&gloc, allocator);
            defer allocator.free(out);

            const transfer_duration = transfer_start.durationTo(std.Io.Clock.awake.now(init.io));
            const transfer_ns = @as(u64, @intCast(transfer_duration.toNanoseconds()));
            if (transfer_ns < min_transfer_ns) min_transfer_ns = transfer_ns;
        }

        // --- Metrics Calculations ---
        const f_size = @as(f64, @floatFromInt(size));
        const element_bytes = f_size * @as(f64, @floatFromInt(@sizeOf(f32)));
        const mb = element_bytes / (1024.0 * 1024.0);

        // Individual Phase Timings (ms)
        const alloc_ms = @as(f64, @floatFromInt(min_alloc_ns)) / 1_000_000.0;
        const compute_ms = @as(f64, @floatFromInt(min_compute_ns)) / 1_000_000.0;
        const transfer_ms = @as(f64, @floatFromInt(min_transfer_ns)) / 1_000_000.0;

        // Bandwidth Calculations
        const alloc_gb_s = (element_bytes * 2.0 / 1_000_000_000.0) / (@as(f64, @floatFromInt(min_alloc_ns)) / 1_000_000_000.0);
        const compute_gb_s = (element_bytes * 3.0 / 1_000_000_000.0) / (@as(f64, @floatFromInt(min_compute_ns)) / 1_000_000_000.0);
        const transfer_gb_s = (element_bytes * 1.0 / 1_000_000_000.0) / (@as(f64, @floatFromInt(min_transfer_ns)) / 1_000_000_000.0);

        // Convert Peak VRAM bytes to Megabytes for clean display
        const peak_vram_mb = @as(f64, @floatFromInt(peak_vram_bytes)) / (1024.0 * 1024.0);

        // Print Results per Size Block with VRAM column aligned
        std.debug.print("| {d:9.2} | 1. GPU Alloc/Load | {d:10.3} | {d:8.2} |           |\n", .{ mb, alloc_ms, alloc_gb_s });
        std.debug.print("|           | 2. Compute        | {d:10.3} | {d:8.2} | {d:7.2} MB|\n", .{ compute_ms, compute_gb_s, peak_vram_mb });
        std.debug.print("|           | 3. Transfer (D->H)| {d:10.3} | {d:8.2} |           |\n", .{ transfer_ms, transfer_gb_s });
        std.debug.print("|-----------|-------------------|------------|---------:|----------:|\n", .{});
    }
}
