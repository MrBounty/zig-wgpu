pub const c = @cImport(@cInclude("wgpu.h"));

pub fn sv(s: []const u8) c.WGPUStringView {
    return .{ .data = s.ptr, .length = s.len };
}

/// Allows safely passing an optional Zig string to a WebGPU string view.
pub fn svOpt(s: ?[]const u8) c.WGPUStringView {
    if (s) |str| return sv(str);
    return .{ .data = null, .length = 0 };
}

/// Helper to print a WGPUStringView in your logs.
pub fn viewStr(view: c.WGPUStringView) []const u8 {
    if (view.data != null and view.length > 0) {
        return view.data[0..view.length];
    }
    return "unnamed";
}
