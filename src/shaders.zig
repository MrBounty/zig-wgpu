pub const SHADER_ADD =
    \\@group(0) @binding(0) var<storage, read>       a : array<f32>;
    \\@group(0) @binding(1) var<storage, read>       b : array<f32>;
    \\@group(0) @binding(2) var<storage, read_write> out : array<f32>;
    \\
    \\@compute @workgroup_size(64)
    \\fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    \\    let i = gid.x;
    \\    if (i < arrayLength(&out)) {
    \\        out[i] = a[i] + b[i];
    \\    }
    \\}
;

pub const SHADER_SCALE =
    \\struct Uniforms { scalar : f32 }
    \\@group(0) @binding(0) var<storage, read>       a   : array<f32>;
    \\@group(0) @binding(1) var<storage, read_write> out : array<f32>;
    \\@group(0) @binding(2) var<uniform>             u   : Uniforms;
    \\
    \\@compute @workgroup_size(64)
    \\fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    \\    let i = gid.x;
    \\    if (i < arrayLength(&out)) {
    \\        out[i] = a[i] * u.scalar;
    \\    }
    \\}
;
