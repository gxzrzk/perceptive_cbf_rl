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



extern "C" __global__ void _solve_init_jaref_kernel__locals__kernel_abfbd8b7_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::int32> var_nefc_in,
    wp::array_t<wp::float32> var_qacc_in,
    wp::array_t<wp::int32> var_efc_J_rownnz_in,
    wp::array_t<wp::int32> var_efc_J_rowadr_in,
    wp::array_t<wp::int32> var_efc_J_colind_in,
    wp::array_t<wp::float32> var_efc_J_in,
    wp::array_t<wp::float32> var_efc_aref_in,
    wp::array_t<wp::int32> var_dof_cdof_in,
    wp::array_t<wp::float32> var_ctx_Jaref_out)
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
        wp::int32* var_3;
        bool var_4;
        wp::int32 var_5;
        const wp::float32 var_6 = 0.0;
        wp::float32 var_7;
        const bool var_8 = true;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::range_t var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        const wp::int32 var_18 = 0;
        wp::int32* var_19;
        wp::int32 var_20;
        wp::int32 var_21;
        const bool var_22 = false;
        const wp::int32 var_23 = 0;
        wp::float32* var_24;
        wp::float32* var_25;
        wp::float32 var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        wp::float32 var_29;
        wp::float32* var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        //---------
        // forward
        // def kernel(                                                                            <L 1565>
        // worldid, efcid, dofstart = wp.tid()                                                    <L 1578>
        builtin_tid3d(var_0, var_1, var_2);
        // if efcid >= nefc_in[worldid]:                                                          <L 1580>
        var_3 = wp::address(var_nefc_in, var_0);
        var_5 = wp::load(var_3);
        var_4 = (var_1 >= var_5);
        if (var_4) {
            // return                                                                             <L 1581>
            continue;
        }
        // jaref = float(0.0)                                                                     <L 1583>
        var_7 = wp::float(var_6);
        // if wp.static(is_sparse):                                                               <L 1584>
        // rownnz = efc_J_rownnz_in[worldid, efcid]                                               <L 1585>
        var_9 = wp::address(var_efc_J_rownnz_in, var_0, var_1);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // rowadr = efc_J_rowadr_in[worldid, efcid]                                               <L 1586>
        var_12 = wp::address(var_efc_J_rowadr_in, var_0, var_1);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // for i in range(rownnz):                                                                <L 1587>
        var_15 = wp::range(var_10);
        start_for_1:;
            if (iter_cmp(var_15) == 0) goto end_for_1;
            var_16 = wp::iter_next(var_15);
            // sparseid = rowadr + i                                                              <L 1588>
            var_17 = wp::add(var_13, var_16);
            // colind = efc_J_colind_in[worldid, 0, sparseid]                                     <L 1589>
            var_19 = wp::address(var_efc_J_colind_in, var_0, var_18, var_17);
            var_21 = wp::load(var_19);
            var_20 = wp::copy(var_21);
            // if wp.static(COMPACT):                                                             <L 1590>
            // jaref += efc_J_in[worldid, 0, sparseid] * qacc_in[worldid, colind]                 <L 1594>
            var_24 = wp::address(var_efc_J_in, var_0, var_23, var_17);
            var_25 = wp::address(var_qacc_in, var_0, var_20);
            var_27 = wp::load(var_24);
            var_28 = wp::load(var_25);
            var_26 = wp::mul(var_27, var_28);
            var_29 = wp::add(var_7, var_26);
            wp::assign(var_7, var_29);
            goto start_for_1;
        end_for_1:;
        // ctx_Jaref_out[worldid, efcid] = jaref - efc_aref_in[worldid, efcid]                    <L 1595>
        var_30 = wp::address(var_efc_aref_in, var_0, var_1);
        var_32 = wp::load(var_30);
        var_31 = wp::sub(var_7, var_32);
        wp::array_store(var_ctx_Jaref_out, var_0, var_1, var_31);
    }
}

