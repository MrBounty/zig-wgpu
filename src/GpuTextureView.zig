const std = @import("std");
const c = @import("utils.zig").c;
const GpuAllocator = @import("GpuAllocator.zig");
const GpuTexture = @import("lib.zig").GpuTexture;
const GpuTextureFormat = @import("lib.zig").GpuTextureFormat;
const GpuTextureUsage = @import("lib.zig").GpuTextureUsage;

pub const GpuViewDef = struct {
    usage: std.EnumSet(GpuTextureUsage) = .empty,
    format: GpuTextureFormat = .Undefined,
};

raw: c.WGPUTextureView,
gloc: GpuAllocator,

pub fn init(gloc: GpuAllocator, texture: GpuTexture, def: GpuViewDef) !@This() {
    var use: u64 = 0;
    var iter = def.usage.iterator();
    while (iter.next()) |flag| use |= @intFromEnum(flag);

    const raw = try gloc.allocTextureView(texture.raw, .{
        .format = @intFromEnum(def.format),
        .usage = use,
        .mipLevelCount = 1,
        .arrayLayerCount = 1,
    });
    return .{ .gloc = gloc, .raw = raw };
}

pub fn deinit(self: @This()) void {
    self.gloc.freeTextureView(self.raw);
}
