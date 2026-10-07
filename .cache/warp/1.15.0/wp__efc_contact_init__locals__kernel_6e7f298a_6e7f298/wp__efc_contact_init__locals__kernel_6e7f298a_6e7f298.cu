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



extern "C" __global__ void _efc_contact_init__locals__kernel_80d75754_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::int32> var_body_weldid,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_dof_parentid,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::int32 var_njmax_in,
    wp::int32 var_njmax_nnz_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::array_t<wp::float32> var_dist_in,
    wp::array_t<wp::int32> var_condim_in,
    wp::array_t<wp::float32> var_includemargin_in,
    wp::array_t<wp::int32> var_worldid_in,
    wp::array_t<wp::vec_t<2, wp::int32>> var_geom_in,
    wp::array_t<wp::int32> var_type_in,
    wp::array_t<wp::int32> var_nefc_out,
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_efc_id_out,
    wp::array_t<wp::int32> var_efc_jtdaj_adr_out,
    wp::array_t<wp::int32> var_efc_jtdaj_nrow_out,
    wp::array_t<wp::int32> var_efc_jtdaj_nblock_out,
    wp::array_t<wp::int32> var_efc_J_rownnz_out,
    wp::array_t<wp::int32> var_efc_J_rowadr_out,
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
        wp::int32* var_2;
        bool var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        const wp::int32 var_6 = 1;
        wp::int32 var_7;
        wp::int32 var_8;
        bool var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::float32* var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        wp::float32* var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        const wp::int32 var_19 = 0;
        bool var_20;
        bool var_21;
        const bool var_22 = false;
        const wp::int32 var_23 = 1;
        bool var_24;
        const wp::int32 var_25 = 1;
        const wp::int32 var_26 = 2;
        const wp::int32 var_27 = 1;
        wp::int32 var_28;
        wp::int32 var_29;
        wp::int32 var_30;
        wp::int32* var_31;
        wp::int32 var_32;
        wp::int32 var_33;
        wp::int32 var_34;
        wp::range_t var_35;
        wp::int32 var_36;
        wp::int32 var_37;
        bool var_38;
        const wp::int32 var_39 = -1;
        const bool var_40 = true;
        bool var_41;
        const wp::int32 var_42 = 1;
        wp::int32 var_43;
        wp::int32 var_44;
        wp::int32 var_45;
        const bool var_46 = true;
        wp::vec_t<2, wp::int32>* var_47;
        wp::vec_t<2, wp::int32> var_48;
        wp::vec_t<2, wp::int32> var_49;
        const wp::int32 var_50 = 0;
        wp::int32 var_51;
        wp::int32* var_52;
        wp::int32* var_53;
        wp::int32 var_54;
        wp::int32 var_55;
        wp::int32 var_56;
        const wp::int32 var_57 = 1;
        wp::int32 var_58;
        wp::int32* var_59;
        wp::int32* var_60;
        wp::int32 var_61;
        wp::int32 var_62;
        wp::int32 var_63;
        wp::int32* var_64;
        wp::int32* var_65;
        wp::int32 var_66;
        wp::int32 var_67;
        wp::int32 var_68;
        const wp::int32 var_69 = 1;
        wp::int32 var_70;
        wp::int32 var_71;
        wp::int32* var_72;
        wp::int32* var_73;
        wp::int32 var_74;
        wp::int32 var_75;
        wp::int32 var_76;
        const wp::int32 var_77 = 1;
        wp::int32 var_78;
        wp::int32 var_79;
        const wp::int32 var_80 = 0;
        wp::int32 var_81;
        bool var_82;
        const wp::int32 var_83 = 0;
        bool var_84;
        const wp::int32 var_85 = 0;
        bool var_86;
        wp::int32 var_87;
        bool var_88;
        bool var_89;
        bool var_90;
        bool var_91;
        wp::int32* var_92;
        wp::int32 var_93;
        wp::int32 var_94;
        wp::int32 var_95;
        bool var_96;
        wp::int32* var_97;
        wp::int32 var_98;
        wp::int32 var_99;
        wp::int32 var_100;
        const wp::int32 var_101 = 1;
        wp::int32 var_102;
        wp::int32 var_103;
        wp::int32 var_104;
        wp::int32 var_105;
        wp::int32 var_106;
        bool var_107;
        wp::range_t var_108;
        wp::int32 var_109;
        wp::int32 var_110;
        bool var_111;
        wp::int32 var_112;
        wp::int32 var_113;
        //---------
        // forward
        // def kernel(                                                                            <L 2646>
        // conid = wp.tid()                                                                       <L 2676>
        var_0 = builtin_tid1d();
        // if conid >= nacon_in[0]:                                                               <L 2678>
        var_2 = wp::address(var_nacon_in, var_1);
        var_4 = wp::load(var_2);
        var_3 = (var_0 >= var_4);
        if (var_3) {
            // return                                                                             <L 2679>
            continue;
        }
        // if not type_in[conid] & ContactType.CONSTRAINT:                                        <L 2681>
        var_5 = wp::address(var_type_in, var_0);
        var_8 = wp::load(var_5);
        var_7 = wp::bit_and(var_8, var_6);
        var_9 = wp::unot(var_7);
        if (var_9) {
            // return                                                                             <L 2682>
            continue;
        }
        // condim = condim_in[conid]                                                              <L 2684>
        var_10 = wp::address(var_condim_in, var_0);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // includemargin = includemargin_in[conid]                                                <L 2686>
        var_13 = wp::address(var_includemargin_in, var_0);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // pos = dist_in[conid] - includemargin                                                   <L 2687>
        var_16 = wp::address(var_dist_in, var_0);
        var_18 = wp::load(var_16);
        var_17 = wp::sub(var_18, var_14);
        // active = pos < 0                                                                       <L 2688>
        var_20 = (var_17 < var_19);
        // if not active:                                                                         <L 2690>
        var_21 = wp::unot(var_20);
        if (var_21) {
            // return                                                                             <L 2691>
            continue;
        }
        // if wp.static(IS_ELLIPTIC):                                                             <L 2693>
        // if condim == 1:                                                                        <L 2696>
        var_24 = (var_11 == var_23);
        if (var_24) {
            // ndim = 1                                                                           <L 2697>
        }
        if (!var_24) {
            // ndim = 2 * (condim - 1)                                                            <L 2699>
            var_28 = wp::sub(var_11, var_27);
            var_29 = wp::mul(var_26, var_28);
        }
        var_30 = wp::where(var_24, var_25, var_29);
        // worldid = worldid_in[conid]                                                            <L 2701>
        var_31 = wp::address(var_worldid_in, var_0);
        var_33 = wp::load(var_31);
        var_32 = wp::copy(var_33);
        // base_efcid = wp.atomic_add(nefc_out, worldid, ndim)                                    <L 2704>
        var_34 = wp::atomic_add(var_nefc_out, var_32, var_30);
        // for dim in range(ndim):                                                                <L 2705>
        var_35 = wp::range(var_30);
        start_for_3:;
            if (iter_cmp(var_35) == 0) goto end_for_3;
            var_36 = wp::iter_next(var_35);
            // efcid = base_efcid + dim                                                           <L 2706>
            var_37 = wp::add(var_34, var_36);
            // if efcid >= njmax_in:                                                              <L 2707>
            var_38 = (var_37 >= var_njmax_in);
            if (var_38) {
                // contact_efc_address_out[conid, dim] = -1                                       <L 2708>
                wp::array_store(var_contact_efc_address_out, var_0, var_36, var_39);
            }
            if (!var_38) {
                // contact_efc_address_out[conid, dim] = efcid                                    <L 2710>
                wp::array_store(var_contact_efc_address_out, var_0, var_36, var_37);
                // efc_id_out[worldid, efcid] = conid                                             <L 2712>
                wp::array_store(var_efc_id_out, var_32, var_37, var_0);
            }
            goto start_for_3;
        end_for_3:;
        // if wp.static(is_sparse and newton):                                                    <L 2714>
        // if base_efcid < njmax_in:                                                              <L 2715>
        var_41 = (var_34 < var_njmax_in);
        if (var_41) {
            // jgid = wp.atomic_add(efc_jtdaj_nblock_out, worldid, 1)                             <L 2716>
            var_43 = wp::atomic_add(var_efc_jtdaj_nblock_out, var_32, var_42);
            // efc_jtdaj_adr_out[worldid, jgid] = base_efcid                                      <L 2717>
            wp::array_store(var_efc_jtdaj_adr_out, var_32, var_43, var_34);
            // efc_jtdaj_nrow_out[worldid, jgid] = wp.min(ndim, njmax_in - base_efcid)            <L 2718>
            var_44 = wp::sub(var_njmax_in, var_34);
            var_45 = wp::min(var_30, var_44);
            wp::array_store(var_efc_jtdaj_nrow_out, var_32, var_43, var_45);
        }
        // if wp.static(IS_SPARSE):                                                               <L 2720>
        // geom = geom_in[conid]                                                                  <L 2721>
        var_47 = wp::address(var_geom_in, var_0);
        var_49 = wp::load(var_47);
        var_48 = wp::copy(var_49);
        // body1 = body_weldid[geom_bodyid[geom[0]]]                                              <L 2722>
        var_51 = wp::extract(var_48, var_50);
        var_52 = wp::address(var_geom_bodyid, var_51);
        var_54 = wp::load(var_52);
        var_53 = wp::address(var_body_weldid, var_54);
        var_56 = wp::load(var_53);
        var_55 = wp::copy(var_56);
        // body2 = body_weldid[geom_bodyid[geom[1]]]                                              <L 2723>
        var_58 = wp::extract(var_48, var_57);
        var_59 = wp::address(var_geom_bodyid, var_58);
        var_61 = wp::load(var_59);
        var_60 = wp::address(var_body_weldid, var_61);
        var_63 = wp::load(var_60);
        var_62 = wp::copy(var_63);
        // da1 = int(body_dofadr[body1] + body_dofnum[body1] - 1)                                 <L 2725>
        var_64 = wp::address(var_body_dofadr, var_55);
        var_65 = wp::address(var_body_dofnum, var_55);
        var_67 = wp::load(var_64);
        var_68 = wp::load(var_65);
        var_66 = wp::add(var_67, var_68);
        var_70 = wp::sub(var_66, var_69);
        var_71 = wp::int(var_70);
        // da2 = int(body_dofadr[body2] + body_dofnum[body2] - 1)                                 <L 2726>
        var_72 = wp::address(var_body_dofadr, var_62);
        var_73 = wp::address(var_body_dofnum, var_62);
        var_75 = wp::load(var_72);
        var_76 = wp::load(var_73);
        var_74 = wp::add(var_75, var_76);
        var_78 = wp::sub(var_74, var_77);
        var_79 = wp::int(var_78);
        // rownnz = int(0)                                                                        <L 2729>
        var_81 = wp::int(var_80);
        // while da1 >= 0 or da2 >= 0:                                                            <L 2730>
        start_while_5:;
        var_84 = (var_71 >= var_83);
        var_82 = var_84;
        if (!var_82) {
            var_86 = (var_79 >= var_85);
            var_82 = var_82 || var_86;
        }
        if ((var_82) == false) goto end_while_5;
            // da = wp.max(da1, da2)                                                              <L 2731>
            var_87 = wp::max(var_71, var_79);
            // if da1 == da and da2 == da:                                                        <L 2733>
            var_89 = (var_71 == var_87);
            var_88 = var_89;
            if (var_88) {
                var_90 = (var_79 == var_87);
                var_88 = var_88 && var_90;
            }
            if (var_88) {
                // break                                                                          <L 2734>
                goto end_while_5;
            }
            // if da1 == da:                                                                      <L 2735>
            var_91 = (var_71 == var_87);
            if (var_91) {
                // da1 = dof_parentid[da1]                                                        <L 2736>
                var_92 = wp::address(var_dof_parentid, var_71);
                var_94 = wp::load(var_92);
                var_93 = wp::copy(var_94);
            }
            var_95 = wp::where(var_91, var_93, var_71);
            // if da2 == da:                                                                      <L 2737>
            var_96 = (var_79 == var_87);
            if (var_96) {
                // da2 = dof_parentid[da2]                                                        <L 2738>
                var_97 = wp::address(var_dof_parentid, var_79);
                var_99 = wp::load(var_97);
                var_98 = wp::copy(var_99);
            }
            var_100 = wp::where(var_96, var_98, var_79);
            // rownnz += 1                                                                        <L 2739>
            var_102 = wp::add(var_81, var_101);
            wp::assign(var_71, var_95);
            wp::assign(var_79, var_100);
            wp::assign(var_81, var_102);
        goto start_while_5;
        end_while_5:;
        // rowadr = wp.atomic_add(efc_nnz_out, worldid, rownnz * ndim)                            <L 2741>
        var_103 = wp::mul(var_81, var_30);
        var_104 = wp::atomic_add(var_efc_nnz_out, var_32, var_103);
        // if rowadr + rownnz * ndim > njmax_nnz_in:                                              <L 2742>
        var_105 = wp::mul(var_81, var_30);
        var_106 = wp::add(var_104, var_105);
        var_107 = (var_106 > var_njmax_nnz_in);
        if (var_107) {
            // return                                                                             <L 2743>
            continue;
        }
        // for dim in range(ndim):                                                                <L 2744>
        var_108 = wp::range(var_30);
        start_for_8:;
            if (iter_cmp(var_108) == 0) goto end_for_8;
            var_109 = wp::iter_next(var_108);
            // efcid = base_efcid + dim                                                           <L 2745>
            var_110 = wp::add(var_34, var_109);
            // if efcid < njmax_in:                                                               <L 2746>
            var_111 = (var_110 < var_njmax_in);
            if (var_111) {
                // efc_J_rowadr_out[worldid, efcid] = rowadr + dim * rownnz                       <L 2747>
                var_112 = wp::mul(var_109, var_81);
                var_113 = wp::add(var_104, var_112);
                wp::array_store(var_efc_J_rowadr_out, var_32, var_110, var_113);
                // efc_J_rownnz_out[worldid, efcid] = rownnz                                      <L 2748>
                wp::array_store(var_efc_J_rownnz_out, var_32, var_110, var_81);
            }
            wp::assign(var_37, var_110);
            goto start_for_8;
        end_for_8:;
    }
}

