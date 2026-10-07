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


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/constraint.py:82
static CUDA_CALLABLE void _efc_row_0(
    wp::int32 var_opt_disableflags,
    wp::int32 var_worldid,
    wp::float32 var_timestep,
    wp::int32 var_efcid,
    wp::float32 var_pos_aref,
    wp::float32 var_pos_imp,
    wp::float32 var_invweight,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::float32 var_margin,
    wp::float32 var_vel,
    wp::float32 var_frictionloss,
    wp::int32 var_type,
    wp::int32 var_id,
    wp::array_t<wp::int32> var_type_out,
    wp::array_t<wp::int32> var_id_out,
    wp::array_t<wp::float32> var_pos_out,
    wp::array_t<wp::float32> var_margin_out,
    wp::array_t<wp::float32> var_D_out,
    wp::array_t<wp::float32> var_vel_out,
    wp::array_t<wp::float32> var_aref_out,
    wp::array_t<wp::float32> var_frictionloss_out)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    const wp::int32 var_4 = 0;
    wp::float32 var_5;
    const wp::int32 var_6 = 1;
    wp::float32 var_7;
    const wp::int32 var_8 = 2;
    wp::float32 var_9;
    const wp::int32 var_10 = 3;
    wp::float32 var_11;
    const wp::int32 var_12 = 4;
    wp::float32 var_13;
    const wp::int32 var_14 = 4096;
    wp::int32 var_15;
    bool var_16;
    const wp::float32 var_17 = 2.0;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    const wp::float32 var_21 = 0.0001;
    const wp::float32 var_22 = 0.0001;
    const wp::float32 var_23 = 0.9999;
    const wp::float32 var_24 = 0.9999;
    wp::float32 var_25;
    const wp::float32 var_26 = 0.0001;
    const wp::float32 var_27 = 0.0001;
    const wp::float32 var_28 = 0.9999;
    const wp::float32 var_29 = 0.9999;
    wp::float32 var_30;
    const wp::float32 var_31 = 1e-15;
    const wp::float32 var_32 = 1e-15;
    wp::float32 var_33;
    const wp::float32 var_34 = 0.0001;
    const wp::float32 var_35 = 0.0001;
    const wp::float32 var_36 = 0.9999;
    const wp::float32 var_37 = 0.9999;
    wp::float32 var_38;
    const wp::float32 var_39 = 1.0;
    wp::float32 var_40;
    wp::float32 var_41;
    const wp::float32 var_42 = 1.0;
    wp::float32 var_43;
    wp::float32 var_44;
    wp::float32 var_45;
    wp::float32 var_46;
    wp::float32 var_47;
    const wp::float32 var_48 = 2.0;
    wp::float32 var_49;
    wp::float32 var_50;
    const wp::int32 var_51 = 0;
    wp::float32 var_52;
    const wp::int32 var_53 = 0;
    bool var_54;
    const wp::int32 var_55 = 0;
    wp::float32 var_56;
    wp::float32 var_57;
    wp::float32 var_58;
    wp::float32 var_59;
    const wp::int32 var_60 = 1;
    wp::float32 var_61;
    const wp::int32 var_62 = 0;
    bool var_63;
    const wp::int32 var_64 = 1;
    wp::float32 var_65;
    wp::float32 var_66;
    wp::float32 var_67;
    wp::float32 var_68;
    wp::float32 var_69;
    wp::float32 var_70;
    const wp::float32 var_71 = 1.0;
    const wp::float32 var_72 = 1.0;
    wp::float32 var_73;
    wp::float32 var_74;
    wp::float32 var_75;
    wp::float32 var_76;
    wp::float32 var_77;
    const wp::float32 var_78 = 1.0;
    const wp::float32 var_79 = 1.0;
    const wp::float32 var_80 = 1.0;
    wp::float32 var_81;
    const wp::float32 var_82 = 1.0;
    wp::float32 var_83;
    wp::float32 var_84;
    wp::float32 var_85;
    const wp::float32 var_86 = 1.0;
    wp::float32 var_87;
    wp::float32 var_88;
    wp::float32 var_89;
    wp::float32 var_90;
    bool var_91;
    wp::float32 var_92;
    wp::float32 var_93;
    wp::float32 var_94;
    wp::float32 var_95;
    wp::float32 var_96;
    const wp::float32 var_97 = 1.0;
    bool var_98;
    wp::float32 var_99;
    const wp::float32 var_100 = 1.0;
    const wp::float32 var_101 = 1.0;
    wp::float32 var_102;
    wp::float32 var_103;
    wp::float32 var_104;
    const wp::float32 var_105 = 1e-15;
    const wp::float32 var_106 = 1e-15;
    wp::float32 var_107;
    wp::float32 var_108;
    wp::float32 var_109;
    wp::float32 var_110;
    wp::float32 var_111;
    wp::float32 var_112;
    wp::float32 var_113;
    wp::float32 var_114;
    //---------
    // forward
    // def _efc_row(                                                                          <L 83>
    // timeconst = solref[0]                                                                  <L 111>
    var_1 = wp::extract(var_solref, var_0);
    // dampratio = solref[1]                                                                  <L 112>
    var_3 = wp::extract(var_solref, var_2);
    // dmin = solimp[0]                                                                       <L 113>
    var_5 = wp::extract(var_solimp, var_4);
    // dmax = solimp[1]                                                                       <L 114>
    var_7 = wp::extract(var_solimp, var_6);
    // width = solimp[2]                                                                      <L 115>
    var_9 = wp::extract(var_solimp, var_8);
    // mid = solimp[3]                                                                        <L 116>
    var_11 = wp::extract(var_solimp, var_10);
    // power = solimp[4]                                                                      <L 117>
    var_13 = wp::extract(var_solimp, var_12);
    // if not (opt_disableflags & DisableBit.REFSAFE):                                        <L 119>
    var_15 = wp::bit_and(var_opt_disableflags, var_14);
    var_16 = wp::unot(var_15);
    if (var_16) {
        // timeconst = wp.max(timeconst, 2.0 * timestep)                                      <L 120>
        var_18 = wp::mul(var_17, var_timestep);
        var_19 = wp::max(var_1, var_18);
    }
    var_20 = wp::where(var_16, var_19, var_1);
    // dmin = wp.clamp(dmin, types.MJ_MINIMP, types.MJ_MAXIMP)                                <L 122>
    var_25 = wp::clamp(var_5, var_22, var_24);
    // dmax = wp.clamp(dmax, types.MJ_MINIMP, types.MJ_MAXIMP)                                <L 123>
    var_30 = wp::clamp(var_7, var_27, var_29);
    // width = wp.max(types.MJ_MINVAL, width)                                                 <L 124>
    var_33 = wp::max(var_32, var_9);
    // mid = wp.clamp(mid, types.MJ_MINIMP, types.MJ_MAXIMP)                                  <L 125>
    var_38 = wp::clamp(var_11, var_35, var_37);
    // power = wp.max(1.0, power)                                                             <L 126>
    var_40 = wp::max(var_39, var_13);
    // dmax_sq = dmax * dmax                                                                  <L 129>
    var_41 = wp::mul(var_30, var_30);
    // k = 1.0 / (dmax_sq * timeconst * timeconst * dampratio * dampratio)                    <L 130>
    var_43 = wp::mul(var_41, var_20);
    var_44 = wp::mul(var_43, var_20);
    var_45 = wp::mul(var_44, var_3);
    var_46 = wp::mul(var_45, var_3);
    var_47 = wp::div(var_42, var_46);
    // b = 2.0 / (dmax * timeconst)                                                           <L 131>
    var_49 = wp::mul(var_30, var_20);
    var_50 = wp::div(var_48, var_49);
    // k = wp.where(solref[0] <= 0, -solref[0] / dmax_sq, k)                                  <L 132>
    var_52 = wp::extract(var_solref, var_51);
    var_54 = (var_52 <= var_53);
    var_56 = wp::extract(var_solref, var_55);
    var_57 = wp::neg(var_56);
    var_58 = wp::div(var_57, var_41);
    var_59 = wp::where(var_54, var_58, var_47);
    // b = wp.where(solref[1] <= 0, -solref[1] / dmax, b)                                     <L 133>
    var_61 = wp::extract(var_solref, var_60);
    var_63 = (var_61 <= var_62);
    var_65 = wp::extract(var_solref, var_64);
    var_66 = wp::neg(var_65);
    var_67 = wp::div(var_66, var_30);
    var_68 = wp::where(var_63, var_67, var_50);
    // imp_x = wp.abs(pos_imp) / width                                                        <L 135>
    var_69 = wp::abs(var_pos_imp);
    var_70 = wp::div(var_69, var_33);
    // imp_a = (1.0 / wp.pow(mid, power - 1.0)) * wp.pow(imp_x, power)                        <L 136>
    var_73 = wp::sub(var_40, var_72);
    var_74 = wp::pow(var_38, var_73);
    var_75 = wp::div(var_71, var_74);
    var_76 = wp::pow(var_70, var_40);
    var_77 = wp::mul(var_75, var_76);
    // imp_b = 1.0 - (1.0 / wp.pow(1.0 - mid, power - 1.0)) * wp.pow(1.0 - imp_x, power)       <L 137>
    var_81 = wp::sub(var_80, var_38);
    var_83 = wp::sub(var_40, var_82);
    var_84 = wp::pow(var_81, var_83);
    var_85 = wp::div(var_79, var_84);
    var_87 = wp::sub(var_86, var_70);
    var_88 = wp::pow(var_87, var_40);
    var_89 = wp::mul(var_85, var_88);
    var_90 = wp::sub(var_78, var_89);
    // imp_y = wp.where(imp_x < mid, imp_a, imp_b)                                            <L 138>
    var_91 = (var_70 < var_38);
    var_92 = wp::where(var_91, var_77, var_90);
    // imp = dmin + imp_y * (dmax - dmin)                                                     <L 139>
    var_93 = wp::sub(var_30, var_25);
    var_94 = wp::mul(var_92, var_93);
    var_95 = wp::add(var_25, var_94);
    // imp = wp.clamp(imp, dmin, dmax)                                                        <L 140>
    var_96 = wp::clamp(var_95, var_25, var_30);
    // imp = wp.where(imp_x > 1.0, dmax, imp)                                                 <L 141>
    var_98 = (var_70 > var_97);
    var_99 = wp::where(var_98, var_30, var_96);
    // D_out[worldid, efcid] = 1.0 / wp.max(invweight * (1.0 - imp) / imp, types.MJ_MINVAL)       <L 144>
    var_102 = wp::sub(var_101, var_99);
    var_103 = wp::mul(var_invweight, var_102);
    var_104 = wp::div(var_103, var_99);
    var_107 = wp::max(var_104, var_106);
    var_108 = wp::div(var_100, var_107);
    wp::array_store(var_D_out, var_worldid, var_efcid, var_108);
    // vel_out[worldid, efcid] = vel                                                          <L 145>
    wp::array_store(var_vel_out, var_worldid, var_efcid, var_vel);
    // aref_out[worldid, efcid] = -k * imp * pos_aref - b * vel                               <L 146>
    var_109 = wp::neg(var_59);
    var_110 = wp::mul(var_109, var_99);
    var_111 = wp::mul(var_110, var_pos_aref);
    var_112 = wp::mul(var_68, var_vel);
    var_113 = wp::sub(var_111, var_112);
    wp::array_store(var_aref_out, var_worldid, var_efcid, var_113);
    // pos_out[worldid, efcid] = pos_aref + margin                                            <L 147>
    var_114 = wp::add(var_pos_aref, var_margin);
    wp::array_store(var_pos_out, var_worldid, var_efcid, var_114);
    // margin_out[worldid, efcid] = margin                                                    <L 148>
    wp::array_store(var_margin_out, var_worldid, var_efcid, var_margin);
    // frictionloss_out[worldid, efcid] = frictionloss                                        <L 149>
    wp::array_store(var_frictionloss_out, var_worldid, var_efcid, var_frictionloss);
    // type_out[worldid, efcid] = type                                                        <L 150>
    wp::array_store(var_type_out, var_worldid, var_efcid, var_type);
    // id_out[worldid, efcid] = id                                                            <L 151>
    wp::array_store(var_id_out, var_worldid, var_efcid, var_id);
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/constraint.py:82
static CUDA_CALLABLE void adj__efc_row_0(
    wp::int32 var_opt_disableflags,
    wp::int32 var_worldid,
    wp::float32 var_timestep,
    wp::int32 var_efcid,
    wp::float32 var_pos_aref,
    wp::float32 var_pos_imp,
    wp::float32 var_invweight,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::float32 var_margin,
    wp::float32 var_vel,
    wp::float32 var_frictionloss,
    wp::int32 var_type,
    wp::int32 var_id,
    wp::array_t<wp::int32> var_type_out,
    wp::array_t<wp::int32> var_id_out,
    wp::array_t<wp::float32> var_pos_out,
    wp::array_t<wp::float32> var_margin_out,
    wp::array_t<wp::float32> var_D_out,
    wp::array_t<wp::float32> var_vel_out,
    wp::array_t<wp::float32> var_aref_out,
    wp::array_t<wp::float32> var_frictionloss_out,
    wp::int32 & adj_opt_disableflags,
    wp::int32 & adj_worldid,
    wp::float32 & adj_timestep,
    wp::int32 & adj_efcid,
    wp::float32 & adj_pos_aref,
    wp::float32 & adj_pos_imp,
    wp::float32 & adj_invweight,
    wp::vec_t<2, wp::float32> & adj_solref,
    wp::vec_t<5, wp::float32> & adj_solimp,
    wp::float32 & adj_margin,
    wp::float32 & adj_vel,
    wp::float32 & adj_frictionloss,
    wp::int32 & adj_type,
    wp::int32 & adj_id,
    wp::array_t<wp::int32> & adj_type_out,
    wp::array_t<wp::int32> & adj_id_out,
    wp::array_t<wp::float32> & adj_pos_out,
    wp::array_t<wp::float32> & adj_margin_out,
    wp::array_t<wp::float32> & adj_D_out,
    wp::array_t<wp::float32> & adj_vel_out,
    wp::array_t<wp::float32> & adj_aref_out,
    wp::array_t<wp::float32> & adj_frictionloss_out)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void _limit_slide_hinge__locals__kernel_d3bc0891_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_nv,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::int32 var_opt_disableflags,
    wp::array_t<wp::int32> var_jnt_qposadr,
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::vec_t<2, wp::float32>> var_jnt_solref,
    wp::array_t<wp::vec_t<5, wp::float32>> var_jnt_solimp,
    wp::array_t<wp::vec_t<2, wp::float32>> var_jnt_range,
    wp::array_t<wp::float32> var_jnt_margin,
    wp::array_t<wp::float32> var_dof_invweight0,
    wp::array_t<wp::int32> var_jnt_limited_slide_hinge_adr,
    wp::array_t<wp::float32> var_qpos_in,
    wp::array_t<wp::float32> var_qvel_in,
    wp::int32 var_njmax_in,
    wp::int32 var_njmax_nnz_in,
    wp::array_t<wp::int32> var_nl_out,
    wp::array_t<wp::int32> var_nefc_out,
    wp::array_t<wp::int32> var_efc_type_out,
    wp::array_t<wp::int32> var_efc_id_out,
    wp::array_t<wp::int32> var_efc_jtdaj_adr_out,
    wp::array_t<wp::int32> var_efc_jtdaj_nrow_out,
    wp::array_t<wp::int32> var_efc_jtdaj_nblock_out,
    wp::array_t<wp::int32> var_efc_J_rownnz_out,
    wp::array_t<wp::int32> var_efc_J_rowadr_out,
    wp::array_t<wp::int32> var_efc_J_colind_out,
    wp::array_t<wp::float32> var_efc_J_out,
    wp::array_t<wp::float32> var_efc_pos_out,
    wp::array_t<wp::float32> var_efc_margin_out,
    wp::array_t<wp::float32> var_efc_D_out,
    wp::array_t<wp::float32> var_efc_vel_out,
    wp::array_t<wp::float32> var_efc_aref_out,
    wp::array_t<wp::float32> var_efc_frictionloss_out,
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
        wp::int32 var_1;
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::shape_t* var_5;
        const wp::int32 var_6 = 0;
        wp::int32 var_7;
        wp::shape_t var_8;
        wp::int32 var_9;
        wp::vec_t<2, wp::float32>* var_10;
        wp::vec_t<2, wp::float32> var_11;
        wp::vec_t<2, wp::float32> var_12;
        wp::int32* var_13;
        wp::float32* var_14;
        wp::int32 var_15;
        wp::float32 var_16;
        wp::float32 var_17;
        wp::shape_t* var_18;
        const wp::int32 var_19 = 0;
        wp::int32 var_20;
        wp::shape_t var_21;
        wp::int32 var_22;
        wp::float32* var_23;
        wp::float32 var_24;
        wp::float32 var_25;
        const wp::int32 var_26 = 0;
        wp::float32 var_27;
        wp::float32 var_28;
        const wp::int32 var_29 = 1;
        wp::float32 var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        const wp::int32 var_34 = 0;
        bool var_35;
        const wp::int32 var_36 = 1;
        wp::int32 var_37;
        const wp::int32 var_38 = 1;
        wp::int32 var_39;
        bool var_40;
        const bool var_41 = true;
        const wp::int32 var_42 = 1;
        wp::int32 var_43;
        const wp::int32 var_44 = 1;
        wp::int32* var_45;
        wp::int32 var_46;
        wp::int32 var_47;
        bool var_48;
        wp::float32 var_49;
        const wp::float32 var_50 = 2.0;
        wp::float32 var_51;
        const wp::float32 var_52 = 1.0;
        wp::float32 var_53;
        const bool var_54 = true;
        const wp::int32 var_55 = 1;
        const wp::int32 var_56 = 1;
        wp::int32 var_57;
        const wp::int32 var_58 = 1;
        wp::int32 var_59;
        bool var_60;
        const wp::int32 var_61 = 0;
        const wp::int32 var_62 = 0;
        wp::float32* var_63;
        wp::float32 var_64;
        wp::float32 var_65;
        wp::shape_t* var_66;
        const wp::int32 var_67 = 0;
        wp::int32 var_68;
        wp::shape_t var_69;
        wp::int32 var_70;
        wp::shape_t* var_71;
        const wp::int32 var_72 = 0;
        wp::int32 var_73;
        wp::shape_t var_74;
        wp::int32 var_75;
        wp::shape_t* var_76;
        const wp::int32 var_77 = 0;
        wp::int32 var_78;
        wp::shape_t var_79;
        wp::int32 var_80;
        wp::shape_t* var_81;
        const wp::int32 var_82 = 0;
        wp::int32 var_83;
        wp::shape_t var_84;
        wp::int32 var_85;
        wp::float32* var_86;
        wp::float32* var_87;
        wp::vec_t<2, wp::float32>* var_88;
        wp::vec_t<5, wp::float32>* var_89;
        const wp::float32 var_90 = 0.0;
        const wp::int32 var_91 = 3;
        const wp::int32 var_92 = 3;
        wp::float32 var_93;
        wp::float32 var_94;
        wp::vec_t<2, wp::float32> var_95;
        wp::vec_t<5, wp::float32> var_96;
        //---------
        // forward
        // def kernel(                                                                            <L 1992>
        // worldid, jntlimitedid = wp.tid()                                                       <L 2031>
        builtin_tid2d(var_0, var_1);
        // jntid = jnt_limited_slide_hinge_adr[jntlimitedid]                                      <L 2032>
        var_2 = wp::address(var_jnt_limited_slide_hinge_adr, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // jnt_range_id = worldid % jnt_range.shape[0]                                            <L 2033>
        var_5 = &(var_jnt_range.shape);
        var_8 = wp::load(var_5);
        var_7 = wp::extract(var_8, var_6);
        var_9 = wp::mod(var_0, var_7);
        // jntrange = jnt_range[jnt_range_id, jntid]                                              <L 2034>
        var_10 = wp::address(var_jnt_range, var_9, var_3);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // qpos = qpos_in[worldid, jnt_qposadr[jntid]]                                            <L 2036>
        var_13 = wp::address(var_jnt_qposadr, var_3);
        var_15 = wp::load(var_13);
        var_14 = wp::address(var_qpos_in, var_0, var_15);
        var_17 = wp::load(var_14);
        var_16 = wp::copy(var_17);
        // jnt_margin_id = worldid % jnt_margin.shape[0]                                          <L 2037>
        var_18 = &(var_jnt_margin.shape);
        var_21 = wp::load(var_18);
        var_20 = wp::extract(var_21, var_19);
        var_22 = wp::mod(var_0, var_20);
        // jntmargin = jnt_margin[jnt_margin_id, jntid]                                           <L 2038>
        var_23 = wp::address(var_jnt_margin, var_22, var_3);
        var_25 = wp::load(var_23);
        var_24 = wp::copy(var_25);
        // dist_min, dist_max = qpos - jntrange[0], jntrange[1] - qpos                            <L 2039>
        var_27 = wp::extract(var_11, var_26);
        var_28 = wp::sub(var_16, var_27);
        var_30 = wp::extract(var_11, var_29);
        var_31 = wp::sub(var_30, var_16);
        // pos = wp.min(dist_min, dist_max) - jntmargin                                           <L 2040>
        var_32 = wp::min(var_28, var_31);
        var_33 = wp::sub(var_32, var_24);
        // active = pos < 0                                                                       <L 2041>
        var_35 = (var_33 < var_34);
        // if active:                                                                             <L 2043>
        if (var_35) {
            // wp.atomic_add(nl_out, worldid, 1)                                                  <L 2044>
            var_37 = wp::atomic_add(var_nl_out, var_0, var_36);
            // efcid = wp.atomic_add(nefc_out, worldid, 1)                                        <L 2045>
            var_39 = wp::atomic_add(var_nefc_out, var_0, var_38);
            // if efcid >= njmax_in:                                                              <L 2047>
            var_40 = (var_39 >= var_njmax_in);
            if (var_40) {
                // return                                                                         <L 2048>
                continue;
            }
            // if wp.static(is_sparse and newton):                                                <L 2050>
            // jgid = wp.atomic_add(efc_jtdaj_nblock_out, worldid, 1)                             <L 2051>
            var_43 = wp::atomic_add(var_efc_jtdaj_nblock_out, var_0, var_42);
            // efc_jtdaj_adr_out[worldid, jgid] = efcid                                           <L 2052>
            wp::array_store(var_efc_jtdaj_adr_out, var_0, var_43, var_39);
            // efc_jtdaj_nrow_out[worldid, jgid] = 1                                              <L 2053>
            wp::array_store(var_efc_jtdaj_nrow_out, var_0, var_43, var_44);
            // dofadr = jnt_dofadr[jntid]                                                         <L 2055>
            var_45 = wp::address(var_jnt_dofadr, var_3);
            var_47 = wp::load(var_45);
            var_46 = wp::copy(var_47);
            // J = float(dist_min < dist_max) * 2.0 - 1.0                                         <L 2057>
            var_48 = (var_28 < var_31);
            var_49 = wp::float(var_48);
            var_51 = wp::mul(var_49, var_50);
            var_53 = wp::sub(var_51, var_52);
            // if wp.static(is_sparse):                                                           <L 2059>
            // efc_J_rownnz_out[worldid, efcid] = 1                                               <L 2060>
            wp::array_store(var_efc_J_rownnz_out, var_0, var_39, var_55);
            // rowadr = wp.atomic_add(efc_nnz_out, worldid, 1)                                    <L 2061>
            var_57 = wp::atomic_add(var_efc_nnz_out, var_0, var_56);
            // if rowadr + 1 > njmax_nnz_in:                                                      <L 2062>
            var_59 = wp::add(var_57, var_58);
            var_60 = (var_59 > var_njmax_nnz_in);
            if (var_60) {
                // return                                                                         <L 2063>
                continue;
            }
            // efc_J_rowadr_out[worldid, efcid] = rowadr                                          <L 2064>
            wp::array_store(var_efc_J_rowadr_out, var_0, var_39, var_57);
            // efc_J_colind_out[worldid, 0, rowadr] = dofadr                                      <L 2065>
            wp::array_store(var_efc_J_colind_out, var_0, var_61, var_57, var_46);
            // efc_J_out[worldid, 0, rowadr] = J                                                  <L 2066>
            wp::array_store(var_efc_J_out, var_0, var_62, var_57, var_53);
            // Jqvel = J * qvel_in[worldid, dofadr]                                               <L 2072>
            var_63 = wp::address(var_qvel_in, var_0, var_46);
            var_65 = wp::load(var_63);
            var_64 = wp::mul(var_53, var_65);
            // dof_invweight0_id = worldid % dof_invweight0.shape[0]                              <L 2074>
            var_66 = &(var_dof_invweight0.shape);
            var_69 = wp::load(var_66);
            var_68 = wp::extract(var_69, var_67);
            var_70 = wp::mod(var_0, var_68);
            // jnt_solref_id = worldid % jnt_solref.shape[0]                                      <L 2075>
            var_71 = &(var_jnt_solref.shape);
            var_74 = wp::load(var_71);
            var_73 = wp::extract(var_74, var_72);
            var_75 = wp::mod(var_0, var_73);
            // jnt_solimp_id = worldid % jnt_solimp.shape[0]                                      <L 2076>
            var_76 = &(var_jnt_solimp.shape);
            var_79 = wp::load(var_76);
            var_78 = wp::extract(var_79, var_77);
            var_80 = wp::mod(var_0, var_78);
            // _efc_row(                                                                          <L 2077>
            // opt_disableflags,                                                                  <L 2078>
            // worldid,                                                                           <L 2079>
            // opt_timestep[worldid % opt_timestep.shape[0]],                                     <L 2080>
            var_81 = &(var_opt_timestep.shape);
            var_84 = wp::load(var_81);
            var_83 = wp::extract(var_84, var_82);
            var_85 = wp::mod(var_0, var_83);
            var_86 = wp::address(var_opt_timestep, var_85);
            // efcid,                                                                             <L 2081>
            // pos,                                                                               <L 2082>
            // pos,                                                                               <L 2083>
            // dof_invweight0[dof_invweight0_id, dofadr],                                         <L 2084>
            var_87 = wp::address(var_dof_invweight0, var_70, var_46);
            // jnt_solref[jnt_solref_id, jntid],                                                  <L 2085>
            var_88 = wp::address(var_jnt_solref, var_75, var_3);
            // jnt_solimp[jnt_solimp_id, jntid],                                                  <L 2086>
            var_89 = wp::address(var_jnt_solimp, var_80, var_3);
            // jntmargin,                                                                         <L 2087>
            // Jqvel,                                                                             <L 2088>
            // 0.0,                                                                               <L 2089>
            // ConstraintType.LIMIT_JOINT,                                                        <L 2090>
            // jntid,                                                                             <L 2091>
            // efc_type_out,                                                                      <L 2092>
            // efc_id_out,                                                                        <L 2093>
            // efc_pos_out,                                                                       <L 2094>
            // efc_margin_out,                                                                    <L 2095>
            // efc_D_out,                                                                         <L 2096>
            // efc_vel_out,                                                                       <L 2097>
            // efc_aref_out,                                                                      <L 2098>
            // efc_frictionloss_out,                                                              <L 2099>
            var_93 = wp::load(var_86);
            var_94 = wp::load(var_87);
            var_95 = wp::load(var_88);
            var_96 = wp::load(var_89);
            _efc_row_0(var_opt_disableflags, var_0, var_93, var_39, var_33, var_33, var_94, var_95, var_96, var_24, var_64, var_90, var_92, var_3, var_efc_type_out, var_efc_id_out, var_efc_pos_out, var_efc_margin_out, var_efc_D_out, var_efc_vel_out, var_efc_aref_out, var_efc_frictionloss_out);
        }
    }
}

