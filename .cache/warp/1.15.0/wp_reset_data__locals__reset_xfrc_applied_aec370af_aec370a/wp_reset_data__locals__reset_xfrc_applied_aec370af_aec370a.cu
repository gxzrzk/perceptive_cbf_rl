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



extern "C" __global__ void reset_data__locals__reset_xfrc_applied_87b343f1_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<bool> var_reset_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_xfrc_applied_out)
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
        const bool var_3 = true;
        bool* var_4;
        bool var_5;
        bool var_6;
        const wp::float32 var_7 = 0.0;
        //---------
        // forward
        // def reset_xfrc_applied(reset_in: wp.array[bool], xfrc_applied_out: wp.array2d[wp.spatial_vector]):       <L 2384>
        // worldid, bodyid, elemid = wp.tid()                                                     <L 2385>
        builtin_tid3d(var_0, var_1, var_2);
        // if wp.static(reset is not None):                                                       <L 2387>
        // if not reset_in[worldid]:                                                              <L 2388>
        var_4 = wp::address(var_reset_in, var_0);
        var_6 = wp::load(var_4);
        var_5 = wp::unot(var_6);
        if (var_5) {
            // return                                                                             <L 2389>
            continue;
        }
        // xfrc_applied_out[worldid, bodyid][elemid] = 0.0                                        <L 2391>
        wp::index(var_xfrc_applied_out, var_0, var_1)[var_2] = var_7;
    }
}

