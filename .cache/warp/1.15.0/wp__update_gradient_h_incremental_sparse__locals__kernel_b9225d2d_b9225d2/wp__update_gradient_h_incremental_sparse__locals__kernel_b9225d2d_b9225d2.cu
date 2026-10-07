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



extern "C" __global__ void _update_gradient_h_incremental_sparse__locals__kernel_0425b575_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::int32> var_efc_J_rownnz_in,
    wp::array_t<wp::int32> var_efc_J_rowadr_in,
    wp::array_t<wp::int32> var_efc_J_colind_in,
    wp::array_t<wp::float32> var_efc_J_in,
    wp::array_t<wp::float32> var_efc_D_in,
    wp::array_t<wp::int32> var_efc_state_in,
    wp::array_t<wp::int32> var_dof_cdof_in,
    wp::array_t<wp::int32> var_quad_changed_ids_in,
    wp::array_t<wp::int32> var_quad_changed_count_in,
    wp::int32 var_slots_per_world,
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
        wp::int32* var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::range_t var_6;
        wp::int32 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        const wp::float32 var_14 = 0.0;
        wp::float32 var_15;
        wp::int32* var_16;
        const wp::int32 var_17 = 1;
        bool var_18;
        wp::int32 var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::int32* var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        wp::int32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        const wp::int32 var_30 = 1;
        wp::int32 var_31;
        wp::int32 var_32;
        const wp::int32 var_33 = 2;
        wp::int32 var_34;
        const wp::int32 var_35 = 32;
        wp::range_t var_36;
        wp::int32 var_37;
        const wp::int32 var_38 = 8;
        wp::int32 var_39;
        const wp::int32 var_40 = 1;
        wp::int32 var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        const wp::float32 var_44 = 1.0;
        wp::float32 var_45;
        const wp::float32 var_46 = 0.5;
        wp::float32 var_47;
        wp::int32 var_48;
        const wp::int32 var_49 = 1;
        wp::int32 var_50;
        wp::int32 var_51;
        const wp::int32 var_52 = 2;
        wp::int32 var_53;
        wp::int32 var_54;
        const wp::int32 var_55 = 0;
        wp::int32 var_56;
        wp::float32* var_57;
        wp::float32 var_58;
        wp::float32 var_59;
        const wp::int32 var_60 = 0;
        wp::int32 var_61;
        wp::float32* var_62;
        wp::float32 var_63;
        wp::float32 var_64;
        wp::float32 var_65;
        wp::float32 var_66;
        const wp::float32 var_67 = 0.0;
        bool var_68;
        const wp::int32 var_69 = 0;
        wp::int32 var_70;
        wp::int32* var_71;
        wp::int32 var_72;
        wp::int32 var_73;
        const wp::int32 var_74 = 0;
        wp::int32 var_75;
        wp::int32* var_76;
        wp::int32 var_77;
        wp::int32 var_78;
        const bool var_79 = false;
        wp::int32 var_80;
        wp::slice_t var_81;
        const wp::int32 var_82 = 0;
        wp::slice_t var_83;
        const wp::int32 var_84 = 0;
        wp::array_t<wp::float32> var_85;
        wp::int32 var_86;
        wp::float32 var_87;
        //---------
        // forward
        // def kernel(                                                                            <L 1954>
        // worldid, slot, lane = wp.tid()                                                         <L 1976>
        builtin_tid3d(var_0, var_1, var_2);
        // n_changes = quad_changed_count_in[worldid]                                             <L 1978>
        var_3 = wp::address(var_quad_changed_count_in, var_0);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // for change_idx in range(slot, n_changes, slots_per_world):                             <L 1979>
        var_6 = wp::range(var_1, var_4, var_slots_per_world);
        start_for_0:;
            if (iter_cmp(var_6) == 0) goto end_for_0;
            var_7 = wp::iter_next(var_6);
            // efcid = quad_changed_ids_in[worldid, change_idx]                                   <L 1980>
            var_8 = wp::address(var_quad_changed_ids_in, var_0, var_7);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
            // D = efc_D_in[worldid, efcid]                                                       <L 1981>
            var_11 = wp::address(var_efc_D_in, var_0, var_9);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
            // sign = float(0.0)                                                                  <L 1982>
            var_15 = wp::float(var_14);
            // if efc_state_in[worldid, efcid] == types.ConstraintState.QUADRATIC.value:          <L 1983>
            var_16 = wp::address(var_efc_state_in, var_0, var_9);
            var_19 = wp::load(var_16);
            var_18 = (var_19 == var_17);
            if (var_18) {
                // sign = D                                                                       <L 1984>
                var_20 = wp::copy(var_12);
            }
            var_21 = wp::where(var_18, var_20, var_15);
            if (!var_18) {
                // sign = -D                                                                      <L 1986>
                var_22 = wp::neg(var_12);
            }
            var_23 = wp::where(var_18, var_21, var_22);
            // rownnz = efc_J_rownnz_in[worldid, efcid]                                           <L 1988>
            var_24 = wp::address(var_efc_J_rownnz_in, var_0, var_9);
            var_26 = wp::load(var_24);
            var_25 = wp::copy(var_26);
            // rowadr = efc_J_rowadr_in[worldid, efcid]                                           <L 1989>
            var_27 = wp::address(var_efc_J_rowadr_in, var_0, var_9);
            var_29 = wp::load(var_27);
            var_28 = wp::copy(var_29);
            // n_entries = rownnz * (rownnz + 1) // 2                                             <L 1990>
            var_31 = wp::add(var_25, var_30);
            var_32 = wp::mul(var_25, var_31);
            var_34 = wp::floordiv(var_32, var_33);
            // for entry in range(lane, n_entries, wp.static(_JTDAJ_THREADS_PER_GROUP)):          <L 1992>
            var_36 = wp::range(var_2, var_34, var_35);
            start_for_2:;
                if (iter_cmp(var_36) == 0) goto end_for_2;
                var_37 = wp::iter_next(var_36);
                // ii = int((wp.sqrt(float(8 * entry + 1)) - 1.0) * 0.5)                          <L 1993>
                var_39 = wp::mul(var_38, var_37);
                var_41 = wp::add(var_39, var_40);
                var_42 = wp::float(var_41);
                var_43 = wp::sqrt(var_42);
                var_45 = wp::sub(var_43, var_44);
                var_47 = wp::mul(var_45, var_46);
                var_48 = wp::int(var_47);
                // jj = entry - ii * (ii + 1) // 2                                                <L 1994>
                var_50 = wp::add(var_48, var_49);
                var_51 = wp::mul(var_48, var_50);
                var_53 = wp::floordiv(var_51, var_52);
                var_54 = wp::sub(var_37, var_53);
                // Ji = efc_J_in[worldid, 0, rowadr + ii]                                         <L 1995>
                var_56 = wp::add(var_28, var_48);
                var_57 = wp::address(var_efc_J_in, var_0, var_55, var_56);
                var_59 = wp::load(var_57);
                var_58 = wp::copy(var_59);
                // Jj = efc_J_in[worldid, 0, rowadr + jj]                                         <L 1996>
                var_61 = wp::add(var_28, var_54);
                var_62 = wp::address(var_efc_J_in, var_0, var_60, var_61);
                var_64 = wp::load(var_62);
                var_63 = wp::copy(var_64);
                // h = sign * Ji * Jj                                                             <L 1997>
                var_65 = wp::mul(var_23, var_58);
                var_66 = wp::mul(var_65, var_63);
                // if h != 0.0:                                                                   <L 1998>
                var_68 = (var_66 != var_67);
                if (var_68) {
                    // colindi = efc_J_colind_in[worldid, 0, rowadr + ii]                         <L 1999>
                    var_70 = wp::add(var_28, var_48);
                    var_71 = wp::address(var_efc_J_colind_in, var_0, var_69, var_70);
                    var_73 = wp::load(var_71);
                    var_72 = wp::copy(var_73);
                    // colindj = efc_J_colind_in[worldid, 0, rowadr + jj]                         <L 2000>
                    var_75 = wp::add(var_28, var_54);
                    var_76 = wp::address(var_efc_J_colind_in, var_0, var_74, var_75);
                    var_78 = wp::load(var_76);
                    var_77 = wp::copy(var_78);
                    // if wp.static(COMPACT):                                                     <L 2001>
                    // wp.atomic_add(ctx_h_out[worldid, wp.min(colindi, colindj)], wp.max(colindi, colindj), h)       <L 2006>
                    var_80 = wp::min(var_72, var_77);
                    var_81 = wp::slice_t(var_0, var_0, var_82);
                    var_83 = wp::slice_t(var_80, var_80, var_84);
                    var_85 = wp::view(var_ctx_h_out, var_81, var_83);
                    var_86 = wp::max(var_72, var_77);
                    var_87 = wp::atomic_add(var_85, var_86, var_66);
                }
                goto start_for_2;
            end_for_2:;
            goto start_for_0;
        end_for_0:;
    }
}

