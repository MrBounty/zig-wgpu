struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var output: VertexOutput;
    // Hardcoded fullscreen quad layout using 4 vertices (Triangle Strip)
    // Indexes: 0: Top-Left, 1: Bottom-Left, 2: Top-Right, 3: Bottom-Right
    var pos = array<vec2f, 4>(
        vec2f(-1.0,  1.0),
        vec2f(-1.0, -1.0),
        vec2f( 1.0,  1.0),
        vec2f( 1.0, -1.0)
    );
    
    output.position = vec4f(pos[vertex_index], 0.0, 1.0);
    output.uv = pos[vertex_index]; // Ranges cleanly from -1.0 to 1.0
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    // Distance from the center (0,0)
    let distance = length(input.uv);
    let radius = 0.5;
    
    // Smooth out pixel edges (anti-aliasing)
    let edge_softness = 0.005;
    let alpha = 1.0 - smoothstep(radius - edge_softness, radius + edge_softness, distance);
    
    if (alpha <= 0.0) {
        discard; 
    }
    
    // Draw a sharp/smooth red circle
    return vec4f(1.0, 0.3, 0.3, alpha);
}
