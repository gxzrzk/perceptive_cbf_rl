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



extern "C" __global__ void _JTDAJ_sparse__locals__kernel_5e7a16ac_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::int32> var_efc_jtdaj_adr_in,
    wp::array_t<wp::int32> var_efc_jtdaj_nrow_in,
    wp::array_t<wp::int32> var_efc_jtdaj_nblock_in,
    wp::array_t<wp::int32> var_efc_J_rownnz_in,
    wp::array_t<wp::int32> var_efc_J_rowadr_in,
    wp::array_t<wp::int32> var_efc_J_colind_in,
    wp::array_t<wp::float32> var_efc_J_in,
    wp::array_t<wp::float32> var_efc_D_in,
    wp::array_t<wp::int32> var_efc_state_in,
    wp::array_t<wp::int32> var_dof_cdof_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::int32 var_groups_per_world,
    wp::array_t<wp::float32> var_h_out)
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
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::range_t var_9;
        wp::int32 var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::int32* var_20;
        wp::int32 var_21;
        wp::int32 var_22;
        const wp::int32 var_23 = 1;
        wp::int32 var_24;
        wp::int32 var_25;
        const wp::int32 var_26 = 2;
        wp::int32 var_27;
        const wp::int32 var_28 = 32;
        wp::range_t var_29;
        wp::int32 var_30;
        const wp::int32 var_31 = 8;
        wp::int32 var_32;
        const wp::int32 var_33 = 1;
        wp::int32 var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        const wp::float32 var_37 = 1.0;
        wp::float32 var_38;
        const wp::float32 var_39 = 0.5;
        wp::float32 var_40;
        wp::int32 var_41;
        const wp::int32 var_42 = 1;
        wp::int32 var_43;
        wp::int32 var_44;
        const wp::int32 var_45 = 2;
        wp::int32 var_46;
        wp::int32 var_47;
        const wp::int32 var_48 = 0;
        wp::int32 var_49;
        wp::int32* var_50;
        wp::int32 var_51;
        wp::int32 var_52;
        const wp::int32 var_53 = 0;
        wp::int32 var_54;
        wp::int32* var_55;
        wp::int32 var_56;
        wp::int32 var_57;
        const wp::float32 var_58 = 0.0;
        wp::float32 var_59;
        wp::range_t var_60;
        wp::int32 var_61;
        wp::int32 var_62;
        wp::int32* var_63;
        const wp::int32 var_64 = 1;
        bool var_65;
        wp::int32 var_66;
        wp::int32* var_67;
        wp::int32 var_68;
        wp::int32 var_69;
        const wp::int32 var_70 = 0;
        wp::int32 var_71;
        wp::float32* var_72;
        wp::float32 var_73;
        wp::float32 var_74;
        const wp::int32 var_75 = 0;
        wp::int32 var_76;
        wp::float32* var_77;
        wp::float32 var_78;
        wp::float32 var_79;
        wp::float32* var_80;
        wp::float32 var_81;
        wp::float32 var_82;
        wp::float32 var_83;
        wp::float32 var_84;
        wp::float32 var_85;
        const wp::float32 var_86 = 0.0;
        bool var_87;
        const bool var_88 = false;
        wp::int32 var_89;
        wp::slice_t var_90;
        const wp::int32 var_91 = 0;
        wp::slice_t var_92;
        const wp::int32 var_93 = 0;
        wp::array_t<wp::float32> var_94;
        wp::int32 var_95;
        wp::float32 var_96;
        //---------
        // forward
        // def kernel(                                                                            <L 3107>
        // worldid, slot, lane = wp.tid()                                                         <L 3125>
        builtin_tid3d(var_0, var_1, var_2);
        // if ctx_done_in[worldid]:                                                               <L 3126>
        var_3 = wp::address(var_ctx_done_in, var_0);
        var_4 = wp::load(var_3);
        if (var_4) {
            // return                                                                             <L 3127>
            continue;
        }
        var_5 = wp::load(var_3);
        // count = efc_jtdaj_nblock_in[worldid]                                                   <L 3128>
        var_6 = wp::address(var_efc_jtdaj_nblock_in, var_0);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // for groupid in range(slot, count, groups_per_world):  # grid-stride this world's group list       <L 3129>
        var_9 = wp::range(var_1, var_7, var_groups_per_world);
        start_for_1:;
            if (iter_cmp(var_9) == 0) goto end_for_1;
            var_10 = wp::iter_next(var_9);
            // head_row = efc_jtdaj_adr_in[worldid, groupid]                                      <L 3130>
            var_11 = wp::address(var_efc_jtdaj_adr_in, var_0, var_10);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
            // block_rows = efc_jtdaj_nrow_in[worldid, groupid]                                   <L 3131>
            var_14 = wp::address(var_efc_jtdaj_nrow_in, var_0, var_10);
            var_16 = wp::load(var_14);
            var_15 = wp::copy(var_16);
            // head_adr = efc_J_rowadr_in[worldid, head_row]                                      <L 3132>
            var_17 = wp::address(var_efc_J_rowadr_in, var_0, var_12);
            var_19 = wp::load(var_17);
            var_18 = wp::copy(var_19);
            // support = efc_J_rownnz_in[worldid, head_row]  # dofs the constraint touches = block dimension       <L 3133>
            var_20 = wp::address(var_efc_J_rownnz_in, var_0, var_12);
            var_22 = wp::load(var_20);
            var_21 = wp::copy(var_22);
            // n_entries = support * (support + 1) // 2  # upper-triangular entries of the |S|x|S| block       <L 3134>
            var_24 = wp::add(var_21, var_23);
            var_25 = wp::mul(var_21, var_24);
            var_27 = wp::floordiv(var_25, var_26);
            // for entry in range(lane, n_entries, wp.static(_JTDAJ_THREADS_PER_GROUP)):          <L 3135>
            var_29 = wp::range(var_2, var_27, var_28);
            start_for_3:;
                if (iter_cmp(var_29) == 0) goto end_for_3;
                var_30 = wp::iter_next(var_29);
                // block_col = int((wp.sqrt(float(8 * entry + 1)) - 1.0) * 0.5)                   <L 3136>
                var_32 = wp::mul(var_31, var_30);
                var_34 = wp::add(var_32, var_33);
                var_35 = wp::float(var_34);
                var_36 = wp::sqrt(var_35);
                var_38 = wp::sub(var_36, var_37);
                var_40 = wp::mul(var_38, var_39);
                var_41 = wp::int(var_40);
                // block_row = entry - block_col * (block_col + 1) // 2                           <L 3137>
                var_43 = wp::add(var_41, var_42);
                var_44 = wp::mul(var_41, var_43);
                var_46 = wp::floordiv(var_44, var_45);
                var_47 = wp::sub(var_30, var_46);
                // dof_row = efc_J_colind_in[worldid, 0, head_adr + block_row]                    <L 3138>
                var_49 = wp::add(var_18, var_47);
                var_50 = wp::address(var_efc_J_colind_in, var_0, var_48, var_49);
                var_52 = wp::load(var_50);
                var_51 = wp::copy(var_52);
                // dof_col = efc_J_colind_in[worldid, 0, head_adr + block_col]                    <L 3139>
                var_54 = wp::add(var_18, var_41);
                var_55 = wp::address(var_efc_J_colind_in, var_0, var_53, var_54);
                var_57 = wp::load(var_55);
                var_56 = wp::copy(var_57);
                // hval = float(0.0)                                                              <L 3140>
                var_59 = wp::float(var_58);
                // for member in range(block_rows):                                               <L 3141>
                var_60 = wp::range(var_15);
                start_for_5:;
                    if (iter_cmp(var_60) == 0) goto end_for_5;
                    var_61 = wp::iter_next(var_60);
                    // member_row = head_row + member                                             <L 3142>
                    var_62 = wp::add(var_12, var_61);
                    // if efc_state_in[worldid, member_row] == types.ConstraintState.QUADRATIC.value:       <L 3143>
                    var_63 = wp::address(var_efc_state_in, var_0, var_62);
                    var_66 = wp::load(var_63);
                    var_65 = (var_66 == var_64);
                    if (var_65) {
                        // member_adr = efc_J_rowadr_in[worldid, member_row]                      <L 3144>
                        var_67 = wp::address(var_efc_J_rowadr_in, var_0, var_62);
                        var_69 = wp::load(var_67);
                        var_68 = wp::copy(var_69);
                        // j_row = efc_J_in[worldid, 0, member_adr + block_row]                   <L 3145>
                        var_71 = wp::add(var_68, var_47);
                        var_72 = wp::address(var_efc_J_in, var_0, var_70, var_71);
                        var_74 = wp::load(var_72);
                        var_73 = wp::copy(var_74);
                        // j_col = efc_J_in[worldid, 0, member_adr + block_col]                   <L 3146>
                        var_76 = wp::add(var_68, var_41);
                        var_77 = wp::address(var_efc_J_in, var_0, var_75, var_76);
                        var_79 = wp::load(var_77);
                        var_78 = wp::copy(var_79);
                        // hval += j_row * efc_D_in[worldid, member_row] * j_col                  <L 3147>
                        var_80 = wp::address(var_efc_D_in, var_0, var_62);
                        var_82 = wp::load(var_80);
                        var_81 = wp::mul(var_73, var_82);
                        var_83 = wp::mul(var_81, var_78);
                        var_84 = wp::add(var_59, var_83);
                    }
                    var_85 = wp::where(var_65, var_84, var_59);
                    wp::assign(var_59, var_85);
                    goto start_for_5;
                end_for_5:;
                // if hval != 0.0:  # skip the atomic when no member row is active                <L 3148>
                var_87 = (var_59 != var_86);
                if (var_87) {
                    // if wp.static(COMPACT):                                                     <L 3149>
                    // wp.atomic_add(h_out[worldid, wp.min(dof_row, dof_col)], wp.max(dof_row, dof_col), hval)       <L 3154>
                    var_89 = wp::min(var_51, var_56);
                    var_90 = wp::slice_t(var_0, var_0, var_91);
                    var_92 = wp::slice_t(var_89, var_89, var_93);
                    var_94 = wp::view(var_h_out, var_90, var_92);
                    var_95 = wp::max(var_51, var_56);
                    var_96 = wp::atomic_add(var_94, var_95, var_59);
                }
                goto start_for_3;
            end_for_3:;
            goto start_for_1;
        end_for_1:;
    }
}

