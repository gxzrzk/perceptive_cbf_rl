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



extern "C" __global__ void _update_constraint_init_qfrc_constraint_sparse__locals__kernel_47cbed64_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_nefc_in,
    wp::array_t<wp::int32> var_efc_J_rownnz_in,
    wp::array_t<wp::int32> var_efc_J_rowadr_in,
    wp::array_t<wp::int32> var_efc_J_colind_in,
    wp::array_t<wp::float32> var_efc_J_in,
    wp::array_t<wp::float32> var_efc_force_in,
    wp::array_t<wp::int32> var_dof_cdof_in,
    wp::array_t<wp::int32> var_state_changed_count_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_qfrc_constraint_out)
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
        bool* var_2;
        bool var_3;
        bool var_4;
        wp::int32* var_5;
        const wp::int32 var_6 = 0;
        bool var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        bool var_10;
        wp::int32 var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        const wp::float32 var_15 = 0.0;
        bool var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::int32* var_20;
        wp::int32 var_21;
        wp::int32 var_22;
        wp::range_t var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        const wp::int32 var_26 = 0;
        wp::int32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        const bool var_30 = false;
        const wp::int32 var_31 = 0;
        wp::float32* var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        wp::slice_t var_35;
        const wp::int32 var_36 = 0;
        wp::array_t<wp::float32> var_37;
        wp::float32 var_38;
        wp::float32 var_39;
        //---------
        // forward
        // def kernel(                                                                            <L 1802>
        // worldid, efcid = wp.tid()                                                              <L 1817>
        builtin_tid2d(var_0, var_1);
        // if ctx_done_in[worldid]:                                                               <L 1819>
        var_2 = wp::address(var_ctx_done_in, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // return                                                                             <L 1820>
            continue;
        }
        var_4 = wp::load(var_2);
        // if state_changed_count_in[worldid] == 0:                                               <L 1822>
        var_5 = wp::address(var_state_changed_count_in, var_0);
        var_8 = wp::load(var_5);
        var_7 = (var_8 == var_6);
        if (var_7) {
            // return                                                                             <L 1823>
            continue;
        }
        // if efcid >= nefc_in[worldid]:                                                          <L 1825>
        var_9 = wp::address(var_nefc_in, var_0);
        var_11 = wp::load(var_9);
        var_10 = (var_1 >= var_11);
        if (var_10) {
            // return                                                                             <L 1826>
            continue;
        }
        // force = efc_force_in[worldid, efcid]                                                   <L 1828>
        var_12 = wp::address(var_efc_force_in, var_0, var_1);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // if force == 0.0:                                                                       <L 1829>
        var_16 = (var_13 == var_15);
        if (var_16) {
            // return                                                                             <L 1830>
            continue;
        }
        // rownnz = efc_J_rownnz_in[worldid, efcid]                                               <L 1832>
        var_17 = wp::address(var_efc_J_rownnz_in, var_0, var_1);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // rowadr = efc_J_rowadr_in[worldid, efcid]                                               <L 1833>
        var_20 = wp::address(var_efc_J_rowadr_in, var_0, var_1);
        var_22 = wp::load(var_20);
        var_21 = wp::copy(var_22);
        // for i in range(rownnz):                                                                <L 1834>
        var_23 = wp::range(var_18);
        start_for_4:;
            if (iter_cmp(var_23) == 0) goto end_for_4;
            var_24 = wp::iter_next(var_23);
            // sparseid = rowadr + i                                                              <L 1835>
            var_25 = wp::add(var_21, var_24);
            // colind = efc_J_colind_in[worldid, 0, sparseid]                                     <L 1836>
            var_27 = wp::address(var_efc_J_colind_in, var_0, var_26, var_25);
            var_29 = wp::load(var_27);
            var_28 = wp::copy(var_29);
            // if wp.static(COMPACT):                                                             <L 1837>
            // efc_J = efc_J_in[worldid, 0, sparseid]                                             <L 1841>
            var_32 = wp::address(var_efc_J_in, var_0, var_31, var_25);
            var_34 = wp::load(var_32);
            var_33 = wp::copy(var_34);
            // wp.atomic_add(qfrc_constraint_out[worldid], colind, efc_J * force)                 <L 1842>
            var_35 = wp::slice_t(var_0, var_0, var_36);
            var_37 = wp::view(var_qfrc_constraint_out, var_35);
            var_38 = wp::mul(var_33, var_13);
            var_39 = wp::atomic_add(var_37, var_28, var_38);
            goto start_for_4;
        end_for_4:;
    }
}

