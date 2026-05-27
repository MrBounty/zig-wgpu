const std = @import("std");
const c = @import("utils.zig").c;
const svOpt = @import("utils.zig").svOpt;
const GpuAllocator = @import("GpuAllocator.zig");
const GpuTexture = @import("lib.zig").GpuTexture;
const GpuTextureFormat = @import("lib.zig").GpuTextureFormat;
const GpuTextureUsage = @import("lib.zig").GpuTextureUsage;

pub const GpuViewDef = struct {
    label: ?[]const u8 = null,
    usage: std.EnumSet(GpuTextureUsage) = .empty,
    format: GpuTextureFormat = .Undefined,
};

raw: c.WGPUTextureView,
glloc: GpuAllocator,

pub fn init(glloc: GpuAllocator, texture: GpuTexture, def: GpuViewDef) !@This() {
    var use: u64 = 0;
    var iter = def.usage.iterator();
    while (iter.next()) |flag| use |= @intFromEnum(flag);

    const raw = try glloc.allocTextureView(texture.raw, .{
        .label = svOpt(def.label),
        .format = @intFromEnum(def.format),
        .usage = use,
        .mipLevelCount = 1,
        .arrayLayerCount = 1,
    });
    return .{ .glloc = glloc, .raw = raw };
}

pub fn deinit(self: @This()) void {
    self.glloc.freeTextureView(self.raw);
}
