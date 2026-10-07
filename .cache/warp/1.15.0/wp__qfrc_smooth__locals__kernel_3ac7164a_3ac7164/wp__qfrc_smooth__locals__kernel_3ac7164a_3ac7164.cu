#define WP_NO_BFLOAT16

#define WP_TILE_BLOCK_DIM 256
#define WP_NO_CRT
#include "builtin.h"
#include "deterministic.h"

// Map wp.breakpoint() to a device brkpt at the call site so cuda-gdb attributes the stop to the generated .cu line
#if defined(__CUDACC__) && !defined(_MSC_VER)
#define __debugbreak() __brkpt()
#endif

// avoid namespacing of float type for casting to float type, this is to avoid wp::float(x), which is not valid in C++
#define float(x) cast_float(x)
#define adj_float(x, adj_x, adj_ret) adj_cast_float(x, adj_x, adj_ret)

#define int(x) cast_int(x)
#define adj_int(x, adj_x, adj_ret) adj_cast_int(x, adj_x, adj_ret)

#define builtin_tid1d() wp::tid(_idx, dim)
#define builtin_tid2d(x, y) wp::tid(x, y, _idx, dim)
#define builtin_tid3d(x, y, z) wp::tid(x, y, z, _idx, dim)
#define builtin_tid4d(x, y, z, w) wp::tid(x, y, z, w, _idx, dim)

#define builtin_block_dim() wp::block_dim()

// CUDA Thread Block Cluster shape declaration. Expands to __cluster_dims__
// only on devices that support clusters (compute capability 9.0+); otherwise
// expands to nothing so the same source compiles cleanly for any target arch.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define WP_CLUSTER_DIMS(x, y, z) __cluster_dims__(x, y, z)
#else
#define WP_CLUSTER_DIMS(x, y, z)
#endif



extern "C" __global__ void _qfrc_smooth__locals__kernel_01eec87a_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_treeid,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::float32> var_qfrc_applied_in,
    wp::array_t<wp::int32> var_tree_awake_in,
    wp::array_t<wp::float32> var_qfrc_bias_in,
    wp::array_t<wp::float32> var_qfrc_passive_in,
    wp::array_t<wp::float32> var_qfrc_actuator_in,
    wp::array_t<wp::float32> var_qfrc_smooth_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        const bool var_2 = false;
        wp::float32* var_3;
        wp::float32* var_4;
        wp::float32 var_5;
        wp::float32 var_6;
        wp::float32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        //---------
        // forward
        // def kernel(                                                                            <L 1258>
        // worldid, dofid = wp.tid()                                                              <L 1271>
        builtin_tid2d(var_0, var_1);
        // if wp.static(enable_sleep):                                                            <L 1273>
        // qfrc_smooth_out[worldid, dofid] = (                                                    <L 1280>
        // qfrc_passive_in[worldid, dofid]                                                        <L 1281>
        var_3 = wp::address(var_qfrc_passive_in, var_0, var_1);
        // - qfrc_bias_in[worldid, dofid]                                                         <L 1282>
        var_4 = wp::address(var_qfrc_bias_in, var_0, var_1);
        var_6 = wp::load(var_3);
        var_7 = wp::load(var_4);
        var_5 = wp::sub(var_6, var_7);
        // + qfrc_actuator_in[worldid, dofid]                                                     <L 1283>
        var_8 = wp::address(var_qfrc_actuator_in, var_0, var_1);
        var_10 = wp::load(var_8);
        var_9 = wp::add(var_5, var_10);
        // + qfrc_applied_in[worldid, dofid]                                                      <L 1284>
        var_11 = wp::address(var_qfrc_applied_in, var_0, var_1);
        var_13 = wp::load(var_11);
        var_12 = wp::add(var_9, var_13);
        // qfrc_smooth_out[worldid, dofid] = (                                                    <L 1280>
        wp::array_store(var_qfrc_smooth_out, var_0, var_1, var_12);
    }
}

