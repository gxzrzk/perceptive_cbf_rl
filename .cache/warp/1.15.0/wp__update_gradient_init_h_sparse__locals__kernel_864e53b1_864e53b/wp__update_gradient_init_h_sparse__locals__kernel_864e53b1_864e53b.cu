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



extern "C" __global__ void _update_gradient_init_h_sparse__locals__kernel_0631694a_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::int32 var_nv,
    wp::array_t<wp::int32> var_M_elemid,
    wp::array_t<wp::float32> var_M_in,
    wp::array_t<wp::int32> var_cdof_dof_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_ctx_h_out)
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
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        bool var_5;
        bool var_6;
        const bool var_7 = false;
        wp::int32 var_8;
        wp::int32 var_9;
        bool var_10;
        bool var_11;
        bool var_12;
        const wp::float32 var_13 = 0.0;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        const wp::int32 var_19 = 0;
        bool var_20;
        wp::float32* var_21;
        wp::float32 var_22;
        const wp::float32 var_23 = 0.0;
        //---------
        // forward
        // def kernel(                                                                            <L 2193>
        // worldid, i, j = wp.tid()                                                               <L 2205>
        builtin_tid3d(var_0, var_1, var_2);
        // if ctx_done_in[worldid]:                                                               <L 2207>
        var_3 = wp::address(var_ctx_done_in, var_0);
        var_4 = wp::load(var_3);
        if (var_4) {
            // return                                                                             <L 2208>
            continue;
        }
        var_5 = wp::load(var_3);
        // if j < i:                                                                              <L 2211>
        var_6 = (var_2 < var_1);
        if (var_6) {
            // return                                                                             <L 2212>
            continue;
        }
        // if wp.static(COMPACT):                                                                 <L 2214>
        // dof_i = i                                                                              <L 2222>
        var_8 = wp::copy(var_1);
        // dof_j = j                                                                              <L 2223>
        var_9 = wp::copy(var_2);
        // if i >= nv or j >= nv:                                                                 <L 2224>
        var_11 = (var_1 >= var_nv);
        var_10 = var_11;
        if (!var_10) {
            var_12 = (var_2 >= var_nv);
            var_10 = var_10 || var_12;
        }
        if (var_10) {
            // ctx_h_out[worldid, i, j] = 0.0                                                     <L 2225>
            wp::array_store(var_ctx_h_out, var_0, var_1, var_2, var_13);
            // return                                                                             <L 2226>
            continue;
        }
        // elemid = M_elemid[wp.max(dof_i, dof_j), wp.min(dof_i, dof_j)]                          <L 2229>
        var_14 = wp::max(var_8, var_9);
        var_15 = wp::min(var_8, var_9);
        var_16 = wp::address(var_M_elemid, var_14, var_15);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // if elemid >= 0:                                                                        <L 2230>
        var_20 = (var_17 >= var_19);
        if (var_20) {
            // ctx_h_out[worldid, i, j] = M_in[worldid, elemid]                                   <L 2231>
            var_21 = wp::address(var_M_in, var_0, var_17);
            var_22 = wp::load(var_21);
            wp::array_store(var_ctx_h_out, var_0, var_1, var_2, var_22);
        }
        if (!var_20) {
            // ctx_h_out[worldid, i, j] = 0.0                                                     <L 2233>
            wp::array_store(var_ctx_h_out, var_0, var_1, var_2, var_23);
        }
    }
}

