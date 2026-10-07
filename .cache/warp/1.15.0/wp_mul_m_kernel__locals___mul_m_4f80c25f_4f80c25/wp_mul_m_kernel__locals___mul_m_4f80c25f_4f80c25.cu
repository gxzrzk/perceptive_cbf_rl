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



extern "C" __global__ void mul_m_kernel__locals___mul_m_ff3bc617_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_M_mulm_rowadr,
    wp::array_t<wp::int32> var_M_mulm_col,
    wp::array_t<wp::int32> var_M_mulm_madr,
    wp::array_t<wp::float32> var_M_in,
    wp::array_t<wp::float32> var_vec,
    wp::array_t<bool> var_skip,
    wp::array_t<wp::float32> var_res)
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
        const bool var_2 = true;
        bool* var_3;
        bool var_4;
        bool var_5;
        const wp::float32 var_6 = 0.0;
        wp::float32 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        const wp::int32 var_11 = 1;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::range_t var_16;
        wp::int32 var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::int32* var_21;
        wp::int32 var_22;
        wp::int32 var_23;
        wp::float32* var_24;
        wp::float32* var_25;
        wp::float32 var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        wp::float32 var_29;
        //---------
        // forward
        // def _mul_m(                                                                            <L 156>
        // worldid, dofid = wp.tid()                                                              <L 170>
        builtin_tid2d(var_0, var_1);
        // if wp.static(check_skip):                                                              <L 172>
        // if skip[worldid]:                                                                      <L 173>
        var_3 = wp::address(var_skip, var_0);
        var_4 = wp::load(var_3);
        if (var_4) {
            // return                                                                             <L 174>
            continue;
        }
        var_5 = wp::load(var_3);
        // acc = float(0.0)                                                                       <L 177>
        var_7 = wp::float(var_6);
        // start = M_mulm_rowadr[dofid]                                                           <L 178>
        var_8 = wp::address(var_M_mulm_rowadr, var_1);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // end = M_mulm_rowadr[dofid + 1]                                                         <L 179>
        var_12 = wp::add(var_1, var_11);
        var_13 = wp::address(var_M_mulm_rowadr, var_12);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // for k in range(start, end):                                                            <L 180>
        var_16 = wp::range(var_9, var_14);
        start_for_1:;
            if (iter_cmp(var_16) == 0) goto end_for_1;
            var_17 = wp::iter_next(var_16);
            // col = M_mulm_col[k]                                                                <L 181>
            var_18 = wp::address(var_M_mulm_col, var_17);
            var_20 = wp::load(var_18);
            var_19 = wp::copy(var_20);
            // madr = M_mulm_madr[k]                                                              <L 182>
            var_21 = wp::address(var_M_mulm_madr, var_17);
            var_23 = wp::load(var_21);
            var_22 = wp::copy(var_23);
            // acc += M_in[worldid, madr] * vec[worldid, col]                                     <L 183>
            var_24 = wp::address(var_M_in, var_0, var_22);
            var_25 = wp::address(var_vec, var_0, var_19);
            var_27 = wp::load(var_24);
            var_28 = wp::load(var_25);
            var_26 = wp::mul(var_27, var_28);
            var_29 = wp::add(var_7, var_26);
            wp::assign(var_7, var_29);
            goto start_for_1;
        end_for_1:;
        // res[worldid, dofid] = acc                                                              <L 185>
        wp::array_store(var_res, var_0, var_1, var_7);
    }
}

