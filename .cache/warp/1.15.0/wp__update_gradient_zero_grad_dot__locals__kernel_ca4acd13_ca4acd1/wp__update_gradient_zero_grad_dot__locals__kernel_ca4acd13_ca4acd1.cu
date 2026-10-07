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



extern "C" __global__ void _update_gradient_zero_grad_dot__locals__kernel_0798e2bb_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::int32> var_state_changed_count_in,
    wp::array_t<wp::float32> var_ctx_alpha_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_ctx_grad_dot_out,
    wp::array_t<wp::float32> var_ctx_newton_decrement_out,
    wp::array_t<wp::float32> var_ctx_grad_scale_out,
    wp::array_t<bool> var_ctx_search_unchanged_out)
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
        const bool var_1 = false;
        const bool var_2 = false;
        bool* var_3;
        bool var_4;
        bool var_5;
        const bool var_6 = false;
        const wp::float32 var_7 = 0.0;
        const wp::float32 var_8 = 0.0;
        const wp::float32 var_9 = 1.0;
        //---------
        // forward
        // def kernel(                                                                            <L 2076>
        // worldid = wp.tid()                                                                     <L 2087>
        var_0 = builtin_tid1d();
        // if wp.static(STABLE_FAST):                                                             <L 2089>
        // ctx_search_unchanged_out[worldid] = False                                              <L 2092>
        wp::array_store(var_ctx_search_unchanged_out, var_0, var_2);
        // if ctx_done_in[worldid]:                                                               <L 2094>
        var_3 = wp::address(var_ctx_done_in, var_0);
        var_4 = wp::load(var_3);
        if (var_4) {
            // return                                                                             <L 2095>
            continue;
        }
        var_5 = wp::load(var_3);
        // if wp.static(STABLE_FAST):                                                             <L 2100>
        // ctx_grad_dot_out[worldid] = 0.0                                                        <L 2113>
        wp::array_store(var_ctx_grad_dot_out, var_0, var_7);
        // ctx_newton_decrement_out[worldid] = 0.0                                                <L 2114>
        wp::array_store(var_ctx_newton_decrement_out, var_0, var_8);
        // ctx_grad_scale_out[worldid] = 1.0                                                      <L 2115>
        wp::array_store(var_ctx_grad_scale_out, var_0, var_9);
    }
}

