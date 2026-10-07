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



extern "C" __global__ void _update_gradient_zero_grad_dot__locals__kernel_999bba89_cuda_kernel_forward(
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
        const bool var_1 = true;
        bool var_2;
        bool* var_3;
        bool var_4;
        wp::int32* var_5;
        const wp::int32 var_6 = 0;
        bool var_7;
        wp::int32 var_8;
        bool* var_9;
        bool var_10;
        bool var_11;
        const bool var_12 = true;
        wp::int32* var_13;
        const wp::int32 var_14 = 0;
        bool var_15;
        wp::int32 var_16;
        wp::float32* var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::float32* var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        const wp::float32 var_23 = 0.0;
        wp::float32 var_24;
        const wp::float32 var_25 = 0.0;
        bool var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        wp::float32 var_29;
        wp::float32* var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        wp::float32* var_33;
        wp::float32 var_34;
        wp::float32 var_35;
        const wp::float32 var_36 = 0.0;
        const wp::float32 var_37 = 0.0;
        const wp::float32 var_38 = 1.0;
        //---------
        // forward
        // def kernel(                                                                            <L 2076>
        // worldid = wp.tid()                                                                     <L 2087>
        var_0 = builtin_tid1d();
        // if wp.static(STABLE_FAST):                                                             <L 2089>
        // ctx_search_unchanged_out[worldid] = ctx_done_in[worldid] or state_changed_count_in[worldid] == 0       <L 2090>
        var_3 = wp::address(var_ctx_done_in, var_0);
        var_4 = wp::load(var_3);
        var_2 = var_4;
        if (!var_2) {
            var_5 = wp::address(var_state_changed_count_in, var_0);
            var_8 = wp::load(var_5);
            var_7 = (var_8 == var_6);
            var_2 = var_2 || var_7;
        }
        wp::array_store(var_ctx_search_unchanged_out, var_0, var_2);
        // if ctx_done_in[worldid]:                                                               <L 2094>
        var_9 = wp::address(var_ctx_done_in, var_0);
        var_10 = wp::load(var_9);
        if (var_10) {
            // return                                                                             <L 2095>
            continue;
        }
        var_11 = wp::load(var_9);
        // if wp.static(STABLE_FAST):                                                             <L 2100>
        // if state_changed_count_in[worldid] == 0:                                               <L 2101>
        var_13 = wp::address(var_state_changed_count_in, var_0);
        var_16 = wp::load(var_13);
        var_15 = (var_16 == var_14);
        if (var_15) {
            // sigma = ctx_grad_scale_out[worldid]                                                <L 2102>
            var_17 = wp::address(var_ctx_grad_scale_out, var_0);
            var_19 = wp::load(var_17);
            var_18 = wp::copy(var_19);
            // new_sigma = sigma - ctx_alpha_in[worldid]                                          <L 2103>
            var_20 = wp::address(var_ctx_alpha_in, var_0);
            var_22 = wp::load(var_20);
            var_21 = wp::sub(var_18, var_22);
            // ratio = float(0.0)                                                                 <L 2104>
            var_24 = wp::float(var_23);
            // if sigma != 0.0:                                                                   <L 2105>
            var_26 = (var_18 != var_25);
            if (var_26) {
                // ratio = new_sigma / sigma                                                      <L 2106>
                var_27 = wp::div(var_21, var_18);
            }
            var_28 = wp::where(var_26, var_27, var_24);
            // ratio_sq = ratio * ratio                                                           <L 2107>
            var_29 = wp::mul(var_28, var_28);
            // ctx_grad_dot_out[worldid] *= ratio_sq                                              <L 2108>
            var_30 = wp::address(var_ctx_grad_dot_out, var_0);
            var_32 = wp::load(var_30);
            var_31 = wp::mul(var_32, var_29);
            wp::array_store(var_ctx_grad_dot_out, var_0, var_31);
            // ctx_newton_decrement_out[worldid] *= ratio_sq                                      <L 2109>
            var_33 = wp::address(var_ctx_newton_decrement_out, var_0);
            var_35 = wp::load(var_33);
            var_34 = wp::mul(var_35, var_29);
            wp::array_store(var_ctx_newton_decrement_out, var_0, var_34);
            // ctx_grad_scale_out[worldid] = new_sigma                                            <L 2110>
            wp::array_store(var_ctx_grad_scale_out, var_0, var_21);
            // return                                                                             <L 2111>
            continue;
        }
        // ctx_grad_dot_out[worldid] = 0.0                                                        <L 2113>
        wp::array_store(var_ctx_grad_dot_out, var_0, var_36);
        // ctx_newton_decrement_out[worldid] = 0.0                                                <L 2114>
        wp::array_store(var_ctx_newton_decrement_out, var_0, var_37);
        // ctx_grad_scale_out[worldid] = 1.0                                                      <L 2115>
        wp::array_store(var_ctx_grad_scale_out, var_0, var_38);
    }
}

