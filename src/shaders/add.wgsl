enable f16;

@group(0) @binding(0) var<storage, read> A: array<f16>;
@group(0) @binding(1) var<storage, read> B: array<f16>;
@group(0) @binding(2) var<storage, read_write> C: array<f16>;

struct TensorInfo {
    size: u32,
};
@group(0) @binding(3) var<uniform> info: TensorInfo; 

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id : vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    // 1. Calculate the total number of threads across the entire grid
    let total_threads = num_workgroups.x * 256u; 
    
    // 2. Start at this thread's unique global ID
    var index = global_id.x;
    
    // 3. Stride through the tensor elements
    while (index < info.size) {
        C[index] = A[index] + B[index];
        index += total_threads; // Jump forward by the total thread count
    }
}
