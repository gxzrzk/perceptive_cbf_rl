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



extern "C" __global__ void _zero_constraint_counts_82d999b9_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::int32> var_ne_out,
    wp::array_t<wp::int32> var_nf_out,
    wp::array_t<wp::int32> var_nl_out,
    wp::array_t<wp::int32> var_nefc_out,
    wp::array_t<wp::int32> var_efc_jtdaj_nblock_out,
    wp::array_t<wp::int32> var_efc_nnz_out)
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
        const wp::int32 var_1 = 0;
        const wp::int32 var_2 = 0;
        const wp::int32 var_3 = 0;
        const wp::int32 var_4 = 0;
        const wp::int32 var_5 = 0;
        const wp::int32 var_6 = 0;
        //---------
        // forward
        // def _zero_constraint_counts(                                                           <L 61>
        // worldid = wp.tid()                                                                     <L 71>
        var_0 = builtin_tid1d();
        // ne_out[worldid] = 0                                                                    <L 74>
        wp::array_store(var_ne_out, var_0, var_1);
        // nf_out[worldid] = 0                                                                    <L 75>
        wp::array_store(var_nf_out, var_0, var_2);
        // nl_out[worldid] = 0                                                                    <L 76>
        wp::array_store(var_nl_out, var_0, var_3);
        // nefc_out[worldid] = 0                                                                  <L 77>
        wp::array_store(var_nefc_out, var_0, var_4);
        // efc_jtdaj_nblock_out[worldid] = 0                                                      <L 78>
        wp::array_store(var_efc_jtdaj_nblock_out, var_0, var_5);
        // efc_nnz_out[worldid] = 0                                                               <L 79>
        wp::array_store(var_efc_nnz_out, var_0, var_6);
    }
}

