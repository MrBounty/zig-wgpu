pub const c = @cImport(@cInclude("wgpu.h"));

pub fn sv(s: []const u8) c.WGPUStringView {
    return .{ .data = s.ptr, .length = s.len };
}
