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


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/math.py:0
static CUDA_CALLABLE wp::float32 safe_div_0(
    wp::float32 var_x,
    wp::float32 var_y)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    bool var_1;
    const wp::float32 var_2 = 1e-15;
    const wp::float32 var_3 = 1e-15;
    wp::float32 var_4;
    wp::float32 var_5;
    //---------
    // forward
    // def safe_div(x: Any, y: Any) -> Any:                                                   <L 1>
    // return x / wp.where(y != 0.0, y, types.MJ_MINVAL)                                      <L 2>
    var_1 = (var_y != var_0);
    var_4 = wp::where(var_1, var_y, var_3);
    var_5 = wp::div(var_x, var_4);
    return var_5;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/solver.py:2740
static CUDA_CALLABLE wp::float32 _elliptic_hessian_entry_from_projections_0(
    wp::float32 var_dm,
    wp::float32 var_mu_over_t,
    wp::float32 var_mu_n_over_ttt,
    wp::float32 var_tangent_diag,
    wp::float32 var_z01,
    wp::float32 var_z02,
    wp::float32 var_projection1,
    wp::float32 var_projection2,
    wp::float32 var_tangent_dot)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    //---------
    // forward
    // def _elliptic_hessian_entry_from_projections(                                          <L 2741>
    // return dm * (                                                                          <L 2754>
    // z01 * z02                                                                              <L 2755>
    var_0 = wp::mul(var_z01, var_z02);
    // - mu_over_t * (z01 * projection2 + z02 * projection1)                                  <L 2756>
    var_1 = wp::mul(var_z01, var_projection2);
    var_2 = wp::mul(var_z02, var_projection1);
    var_3 = wp::add(var_1, var_2);
    var_4 = wp::mul(var_mu_over_t, var_3);
    var_5 = wp::sub(var_0, var_4);
    // + mu_n_over_ttt * projection1 * projection2                                            <L 2757>
    var_6 = wp::mul(var_mu_n_over_ttt, var_projection1);
    var_7 = wp::mul(var_6, var_projection2);
    var_8 = wp::add(var_5, var_7);
    // + tangent_diag * tangent_dot                                                           <L 2758>
    var_9 = wp::mul(var_tangent_diag, var_tangent_dot);
    var_10 = wp::add(var_8, var_9);
    var_11 = wp::mul(var_dm, var_10);
    return var_11;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/solver.py:112
static CUDA_CALLABLE wp::float32 _rescale_0(
    wp::int32 var_nv,
    wp::float32 var_meaninertia,
    wp::float32 var_value)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    //---------
    // forward
    // def _rescale(nv: int, meaninertia: float, value: float) -> float:                      <L 113>
    // return value / (meaninertia * float(nv))                                               <L 114>
    var_0 = wp::float(var_nv);
    var_1 = wp::mul(var_meaninertia, var_0);
    var_2 = wp::div(var_value, var_1);
    return var_2;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/math.py:0
static CUDA_CALLABLE void adj_safe_div_0(
    wp::float32 var_x,
    wp::float32 var_y,
    wp::float32 & adj_x,
    wp::float32 & adj_y,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/solver.py:2740
static CUDA_CALLABLE void adj__elliptic_hessian_entry_from_projections_0(
    wp::float32 var_dm,
    wp::float32 var_mu_over_t,
    wp::float32 var_mu_n_over_ttt,
    wp::float32 var_tangent_diag,
    wp::float32 var_z01,
    wp::float32 var_z02,
    wp::float32 var_projection1,
    wp::float32 var_projection2,
    wp::float32 var_tangent_dot,
    wp::float32 & adj_dm,
    wp::float32 & adj_mu_over_t,
    wp::float32 & adj_mu_n_over_ttt,
    wp::float32 & adj_tangent_diag,
    wp::float32 & adj_z01,
    wp::float32 & adj_z02,
    wp::float32 & adj_projection1,
    wp::float32 & adj_projection2,
    wp::float32 & adj_tangent_dot,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/solver.py:112
static CUDA_CALLABLE void adj__rescale_0(
    wp::int32 var_nv,
    wp::float32 var_meaninertia,
    wp::float32 var_value,
    wp::int32 & adj_nv,
    wp::float32 & adj_meaninertia,
    wp::float32 & adj_value,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void _update_gradient_JTCJ_compact_2f0889de_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_opt_impratio_invsqrt,
    wp::array_t<wp::float32> var_contact_dist_in,
    wp::array_t<wp::float32> var_contact_includemargin_in,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_in,
    wp::array_t<wp::int32> var_contact_dim_in,
    wp::array_t<wp::int32> var_contact_efc_address_in,
    wp::array_t<wp::int32> var_contact_worldid_in,
    wp::array_t<wp::int32> var_efc_J_rownnz_in,
    wp::array_t<wp::int32> var_efc_J_rowadr_in,
    wp::array_t<wp::int32> var_efc_J_colind_in,
    wp::array_t<wp::float32> var_efc_J_in,
    wp::array_t<wp::float32> var_efc_D_in,
    wp::array_t<wp::int32> var_efc_state_in,
    wp::array_t<wp::int32> var_dof_cdof_in,
    wp::int32 var_naconmax_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::array_t<wp::float32> var_ctx_Jaref_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::int32 var_nblocks_perblock,
    wp::int32 var_dim_block,
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
        wp::range_t var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        const wp::int32 var_6 = 0;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        bool var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        bool* var_14;
        bool var_15;
        bool var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        const wp::int32 var_20 = 1;
        bool var_21;
        wp::float32* var_22;
        wp::float32* var_23;
        wp::float32 var_24;
        wp::float32 var_25;
        wp::float32 var_26;
        const wp::float32 var_27 = 0.0;
        bool var_28;
        const wp::int32 var_29 = 0;
        wp::int32* var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        const wp::int32 var_33 = 0;
        bool var_34;
        wp::int32* var_35;
        const wp::int32 var_36 = 4;
        bool var_37;
        wp::int32 var_38;
        wp::int32* var_39;
        wp::int32 var_40;
        wp::int32 var_41;
        const wp::int32 var_42 = 1;
        wp::int32 var_43;
        wp::int32 var_44;
        const wp::int32 var_45 = 2;
        wp::int32 var_46;
        bool var_47;
        wp::int32* var_48;
        wp::int32 var_49;
        wp::int32 var_50;
        const wp::int32 var_51 = 0;
        wp::int32 var_52;
        wp::int32 var_53;
        wp::int32 var_54;
        bool var_55;
        wp::int32 var_56;
        wp::int32 var_57;
        const wp::int32 var_58 = 1;
        wp::int32 var_59;
        wp::int32 var_60;
        const wp::int32 var_61 = 0;
        wp::int32 var_62;
        wp::int32* var_63;
        wp::int32 var_64;
        wp::int32 var_65;
        const wp::int32 var_66 = 0;
        wp::int32 var_67;
        wp::int32* var_68;
        wp::int32 var_69;
        wp::int32 var_70;
        wp::int32* var_71;
        wp::int32 var_72;
        wp::int32 var_73;
        wp::int32* var_74;
        wp::int32 var_75;
        wp::int32 var_76;
        bool var_77;
        const wp::int32 var_78 = 0;
        bool var_79;
        const wp::int32 var_80 = 0;
        bool var_81;
        wp::int32 var_82;
        wp::int32 var_83;
        wp::vec_t<5, wp::float32>* var_84;
        wp::vec_t<5, wp::float32> var_85;
        wp::vec_t<5, wp::float32> var_86;
        const wp::int32 var_87 = 0;
        wp::float32 var_88;
        wp::shape_t* var_89;
        const wp::int32 var_90 = 0;
        wp::int32 var_91;
        wp::shape_t var_92;
        wp::int32 var_93;
        wp::float32* var_94;
        wp::float32 var_95;
        wp::float32 var_96;
        wp::float32 var_97;
        wp::float32* var_98;
        const wp::float32 var_99 = 1.0;
        wp::float32 var_100;
        wp::float32 var_101;
        wp::float32 var_102;
        wp::float32 var_103;
        const wp::float32 var_104 = 0.0;
        bool var_105;
        wp::float32* var_106;
        wp::float32 var_107;
        wp::float32 var_108;
        const wp::float32 var_109 = 0.0;
        const wp::float32 var_110 = 0.0;
        const wp::float32 var_111 = 0.0;
        const wp::float32 var_112 = 0.0;
        const wp::float32 var_113 = 0.0;
        wp::vec_t<6, wp::float32> var_114;
        const wp::float32 var_115 = 0.0;
        wp::float32 var_116;
        const wp::int32 var_117 = 1;
        wp::range_t var_118;
        wp::int32 var_119;
        wp::int32* var_120;
        wp::int32 var_121;
        wp::int32 var_122;
        const wp::int32 var_123 = 0;
        bool var_124;
        wp::float32* var_125;
        const wp::int32 var_126 = 1;
        wp::int32 var_127;
        wp::float32 var_128;
        wp::float32 var_129;
        wp::float32 var_130;
        const wp::float32 var_131 = 0.0;
        wp::float32 var_132;
        wp::float32 var_133;
        wp::float32 var_134;
        const wp::float32 var_135 = 0.0;
        bool var_136;
        const wp::float32 var_137 = 0.0;
        wp::float32 var_138;
        wp::float32 var_139;
        const wp::float32 var_140 = 1e-15;
        const wp::float32 var_141 = 1e-15;
        wp::float32 var_142;
        wp::float32 var_143;
        wp::float32 var_144;
        const wp::float32 var_145 = 1e-15;
        const wp::float32 var_146 = 1e-15;
        wp::float32 var_147;
        wp::float32 var_148;
        wp::float32 var_149;
        wp::float32 var_150;
        wp::float32 var_151;
        wp::float32 var_152;
        wp::float32 var_153;
        const wp::float32 var_154 = 0.0;
        wp::float32 var_155;
        wp::range_t var_156;
        wp::int32 var_157;
        const wp::int32 var_158 = 0;
        bool var_159;
        wp::int32 var_160;
        wp::float32 var_161;
        wp::int32* var_162;
        wp::int32 var_163;
        wp::int32 var_164;
        const wp::int32 var_165 = 0;
        bool var_166;
        const wp::int32 var_167 = 1;
        wp::int32 var_168;
        wp::float32 var_169;
        wp::float32 var_170;
        wp::int32 var_171;
        wp::float32 var_172;
        wp::float32* var_173;
        wp::float32 var_174;
        wp::float32 var_175;
        wp::float32* var_176;
        wp::float32 var_177;
        wp::float32 var_178;
        wp::float32 var_179;
        const wp::int32 var_180 = 1;
        wp::int32 var_181;
        const wp::int32 var_182 = 0;
        wp::range_t var_183;
        wp::int32 var_184;
        const wp::int32 var_185 = 0;
        bool var_186;
        wp::int32 var_187;
        wp::float32 var_188;
        wp::int32* var_189;
        wp::int32 var_190;
        wp::int32 var_191;
        const wp::int32 var_192 = 0;
        bool var_193;
        const wp::int32 var_194 = 1;
        wp::int32 var_195;
        wp::float32 var_196;
        wp::float32 var_197;
        wp::int32 var_198;
        wp::float32 var_199;
        wp::float32* var_200;
        wp::float32 var_201;
        wp::float32 var_202;
        wp::float32* var_203;
        wp::float32 var_204;
        wp::float32 var_205;
        wp::float32 var_206;
        bool var_207;
        const wp::int32 var_208 = 0;
        bool var_209;
        const wp::int32 var_210 = 0;
        bool var_211;
        const wp::float32 var_212 = 1.0;
        const wp::int32 var_213 = 0;
        bool var_214;
        wp::float32 var_215;
        wp::float32 var_216;
        wp::float32 var_217;
        const wp::int32 var_218 = 0;
        bool var_219;
        wp::float32 var_220;
        wp::float32 var_221;
        wp::float32 var_222;
        wp::float32 var_223;
        wp::float32 var_224;
        bool var_225;
        wp::float32 var_226;
        wp::float32 var_227;
        wp::float32 var_228;
        wp::float32 var_229;
        wp::float32 var_230;
        wp::float32 var_231;
        const wp::float32 var_232 = 0.0;
        bool var_233;
        wp::float32 var_234;
        wp::float32 var_235;
        wp::float32 var_236;
        bool var_237;
        wp::float32 var_238;
        wp::float32 var_239;
        wp::float32 var_240;
        wp::float32 var_241;
        wp::float32 var_242;
        wp::slice_t var_243;
        const wp::int32 var_244 = 0;
        wp::slice_t var_245;
        const wp::int32 var_246 = 0;
        wp::array_t<wp::float32> var_247;
        wp::float32 var_248;
        //---------
        // forward
        // def _update_gradient_JTCJ_compact(                                                     <L 2566>
        // conid_start, pairid = wp.tid()                                                         <L 2593>
        builtin_tid2d(var_0, var_1);
        // for i in range(nblocks_perblock):                                                      <L 2595>
        var_2 = wp::range(var_nblocks_perblock);
        start_for_0:;
            if (iter_cmp(var_2) == 0) goto end_for_0;
            var_3 = wp::iter_next(var_2);
            // conid = conid_start + i * dim_block                                                <L 2596>
            var_4 = wp::mul(var_3, var_dim_block);
            var_5 = wp::add(var_0, var_4);
            // if conid >= min(nacon_in[0], naconmax_in):                                         <L 2598>
            var_7 = wp::address(var_nacon_in, var_6);
            var_9 = wp::load(var_7);
            var_8 = wp::min(var_9, var_naconmax_in);
            var_10 = (var_5 >= var_8);
            if (var_10) {
                // return                                                                         <L 2599>
                continue;
            }
            // worldid = contact_worldid_in[conid]                                                <L 2601>
            var_11 = wp::address(var_contact_worldid_in, var_5);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
            // if ctx_done_in[worldid]:                                                           <L 2602>
            var_14 = wp::address(var_ctx_done_in, var_12);
            var_15 = wp::load(var_14);
            if (var_15) {
                // continue                                                                       <L 2603>
                goto start_for_0;
            }
            var_16 = wp::load(var_14);
            // condim = contact_dim_in[conid]                                                     <L 2605>
            var_17 = wp::address(var_contact_dim_in, var_5);
            var_19 = wp::load(var_17);
            var_18 = wp::copy(var_19);
            // if condim == 1:                                                                    <L 2607>
            var_21 = (var_18 == var_20);
            if (var_21) {
                // continue                                                                       <L 2608>
                goto start_for_0;
            }
            // if contact_dist_in[conid] - contact_includemargin_in[conid] >= 0.0:                <L 2611>
            var_22 = wp::address(var_contact_dist_in, var_5);
            var_23 = wp::address(var_contact_includemargin_in, var_5);
            var_25 = wp::load(var_22);
            var_26 = wp::load(var_23);
            var_24 = wp::sub(var_25, var_26);
            var_28 = (var_24 >= var_27);
            if (var_28) {
                // continue                                                                       <L 2612>
                goto start_for_0;
            }
            // efcid0 = contact_efc_address_in[conid, 0]                                          <L 2614>
            var_30 = wp::address(var_contact_efc_address_in, var_5, var_29);
            var_32 = wp::load(var_30);
            var_31 = wp::copy(var_32);
            // if efcid0 < 0:                                                                     <L 2615>
            var_34 = (var_31 < var_33);
            if (var_34) {
                // continue                                                                       <L 2616>
                goto start_for_0;
            }
            // if efc_state_in[worldid, efcid0] != types.ConstraintState.CONE:                    <L 2617>
            var_35 = wp::address(var_efc_state_in, var_12, var_31);
            var_38 = wp::load(var_35);
            var_37 = (var_38 != var_36);
            if (var_37) {
                // continue                                                                       <L 2618>
                goto start_for_0;
            }
            // rownnz = efc_J_rownnz_in[worldid, efcid0]                                          <L 2620>
            var_39 = wp::address(var_efc_J_rownnz_in, var_12, var_31);
            var_41 = wp::load(var_39);
            var_40 = wp::copy(var_41);
            // npairs = rownnz * (rownnz + 1) // 2                                                <L 2621>
            var_43 = wp::add(var_40, var_42);
            var_44 = wp::mul(var_40, var_43);
            var_46 = wp::floordiv(var_44, var_45);
            // if pairid >= npairs:                                                               <L 2622>
            var_47 = (var_1 >= var_46);
            if (var_47) {
                // continue                                                                       <L 2623>
                goto start_for_0;
            }
            // rowadr0 = efc_J_rowadr_in[worldid, efcid0]                                         <L 2625>
            var_48 = wp::address(var_efc_J_rowadr_in, var_12, var_31);
            var_50 = wp::load(var_48);
            var_49 = wp::copy(var_50);
            // pos1 = int(0)                                                                      <L 2626>
            var_52 = wp::int(var_51);
            // rem = pairid                                                                       <L 2627>
            var_53 = wp::copy(var_1);
            // while rem >= rownnz - pos1:                                                        <L 2628>
        start_while_3:;
            var_54 = wp::sub(var_40, var_52);
            var_55 = (var_53 >= var_54);
        if ((var_55) == false) goto end_while_3;
                // rem -= rownnz - pos1                                                           <L 2629>
                var_56 = wp::sub(var_40, var_52);
                var_57 = wp::sub(var_53, var_56);
                // pos1 += 1                                                                      <L 2630>
                var_59 = wp::add(var_52, var_58);
                wp::assign(var_52, var_59);
                wp::assign(var_53, var_57);
        goto start_while_3;
        end_while_3:;
            // pos2 = pos1 + rem                                                                  <L 2631>
            var_60 = wp::add(var_52, var_53);
            // dofa = efc_J_colind_in[worldid, 0, rowadr0 + pos1]                                 <L 2633>
            var_62 = wp::add(var_49, var_52);
            var_63 = wp::address(var_efc_J_colind_in, var_12, var_61, var_62);
            var_65 = wp::load(var_63);
            var_64 = wp::copy(var_65);
            // dofb = efc_J_colind_in[worldid, 0, rowadr0 + pos2]                                 <L 2634>
            var_67 = wp::add(var_49, var_60);
            var_68 = wp::address(var_efc_J_colind_in, var_12, var_66, var_67);
            var_70 = wp::load(var_68);
            var_69 = wp::copy(var_70);
            // dof1id = dof_cdof_in[worldid, dofa]                                                <L 2637>
            var_71 = wp::address(var_dof_cdof_in, var_12, var_64);
            var_73 = wp::load(var_71);
            var_72 = wp::copy(var_73);
            // dof2id = dof_cdof_in[worldid, dofb]                                                <L 2638>
            var_74 = wp::address(var_dof_cdof_in, var_12, var_69);
            var_76 = wp::load(var_74);
            var_75 = wp::copy(var_76);
            // if dof1id < 0 or dof2id < 0:                                                       <L 2640>
            var_79 = (var_72 < var_78);
            var_77 = var_79;
            if (!var_77) {
                var_81 = (var_75 < var_80);
                var_77 = var_77 || var_81;
            }
            if (var_77) {
                // continue                                                                       <L 2641>
                goto start_for_0;
            }
            // c_dof1 = wp.min(dof1id, dof2id)                                                    <L 2643>
            var_82 = wp::min(var_72, var_75);
            // c_dof2 = wp.max(dof1id, dof2id)                                                    <L 2644>
            var_83 = wp::max(var_72, var_75);
            // fri = contact_friction_in[conid]                                                   <L 2646>
            var_84 = wp::address(var_contact_friction_in, var_5);
            var_86 = wp::load(var_84);
            var_85 = wp::copy(var_86);
            // mu = fri[0] * opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]        <L 2647>
            var_88 = wp::extract(var_85, var_87);
            var_89 = &(var_opt_impratio_invsqrt.shape);
            var_92 = wp::load(var_89);
            var_91 = wp::extract(var_92, var_90);
            var_93 = wp::mod(var_12, var_91);
            var_94 = wp::address(var_opt_impratio_invsqrt, var_93);
            var_96 = wp::load(var_94);
            var_95 = wp::mul(var_88, var_96);
            // mu2 = mu * mu                                                                      <L 2649>
            var_97 = wp::mul(var_95, var_95);
            // dm = math.safe_div(efc_D_in[worldid, efcid0], mu2 * (1.0 + mu2))                   <L 2650>
            var_98 = wp::address(var_efc_D_in, var_12, var_31);
            var_100 = wp::add(var_99, var_97);
            var_101 = wp::mul(var_97, var_100);
            var_103 = wp::load(var_98);
            var_102 = safe_div_0(var_103, var_101);
            // if dm == 0.0:                                                                      <L 2652>
            var_105 = (var_102 == var_104);
            if (var_105) {
                // continue                                                                       <L 2653>
                goto start_for_0;
            }
            // n = ctx_Jaref_in[worldid, efcid0] * mu                                             <L 2655>
            var_106 = wp::address(var_ctx_Jaref_in, var_12, var_31);
            var_108 = wp::load(var_106);
            var_107 = wp::mul(var_108, var_95);
            // u = types.vec6(n, 0.0, 0.0, 0.0, 0.0, 0.0)                                         <L 2656>
            var_114 = wp::vec_t<6, wp::float32>({var_107, var_109, var_110, var_111, var_112, var_113});
            // tt = float(0.0)                                                                    <L 2658>
            var_116 = wp::float(var_115);
            // for j in range(1, condim):                                                         <L 2659>
            var_118 = wp::range(var_117, var_18);
            start_for_5:;
                if (iter_cmp(var_118) == 0) goto end_for_5;
                var_119 = wp::iter_next(var_118);
                // efcidj = contact_efc_address_in[conid, j]                                      <L 2660>
                var_120 = wp::address(var_contact_efc_address_in, var_5, var_119);
                var_122 = wp::load(var_120);
                var_121 = wp::copy(var_122);
                // if efcidj >= 0:                                                                <L 2661>
                var_124 = (var_121 >= var_123);
                if (var_124) {
                    // uj = ctx_Jaref_in[worldid, efcidj] * fri[j - 1]                            <L 2662>
                    var_125 = wp::address(var_ctx_Jaref_in, var_12, var_121);
                    var_127 = wp::sub(var_119, var_126);
                    var_128 = wp::extract(var_85, var_127);
                    var_130 = wp::load(var_125);
                    var_129 = wp::mul(var_130, var_128);
                }
                if (!var_124) {
                    // uj = 0.0                                                                   <L 2664>
                }
                var_132 = wp::where(var_124, var_129, var_131);
                // tt += uj * uj                                                                  <L 2665>
                var_133 = wp::mul(var_132, var_132);
                var_134 = wp::add(var_116, var_133);
                // u[j] = uj                                                                      <L 2666>
                wp::assign_inplace(var_114, var_119, var_132);
                wp::assign(var_116, var_134);
                goto start_for_5;
            end_for_5:;
            // if tt <= 0.0:                                                                      <L 2668>
            var_136 = (var_116 <= var_135);
            if (var_136) {
                // t = 0.0                                                                        <L 2669>
            }
            if (!var_136) {
                // t = wp.sqrt(tt)                                                                <L 2671>
                var_138 = wp::sqrt(var_116);
            }
            var_139 = wp::where(var_136, var_137, var_138);
            // t = wp.max(t, types.MJ_MINVAL)                                                     <L 2672>
            var_142 = wp::max(var_139, var_141);
            // ttt = wp.max(t * t * t, types.MJ_MINVAL)                                           <L 2673>
            var_143 = wp::mul(var_142, var_142);
            var_144 = wp::mul(var_143, var_142);
            var_147 = wp::max(var_144, var_146);
            // mu_over_t = math.safe_div(mu, t)                                                   <L 2676>
            var_148 = safe_div_0(var_95, var_142);
            // mu_n_over_ttt = mu * math.safe_div(n, ttt)                                         <L 2677>
            var_149 = safe_div_0(var_107, var_147);
            var_150 = wp::mul(var_95, var_149);
            // mu2_minus_mu_n_over_t = mu2 - mu * math.safe_div(n, t)                             <L 2678>
            var_151 = safe_div_0(var_107, var_142);
            var_152 = wp::mul(var_95, var_151);
            var_153 = wp::sub(var_97, var_152);
            // h = float(0.0)                                                                     <L 2680>
            var_155 = wp::float(var_154);
            // for dim1id in range(condim):                                                       <L 2682>
            var_156 = wp::range(var_18);
            start_for_7:;
                if (iter_cmp(var_156) == 0) goto end_for_7;
                var_157 = wp::iter_next(var_156);
                // if dim1id == 0:                                                                <L 2683>
                var_159 = (var_157 == var_158);
                if (var_159) {
                    // efcid1 = efcid0                                                            <L 2684>
                    var_160 = wp::copy(var_31);
                    // dm_fri1 = dm * mu                                                          <L 2685>
                    var_161 = wp::mul(var_102, var_95);
                }
                if (!var_159) {
                    // efcid1 = contact_efc_address_in[conid, dim1id]                             <L 2687>
                    var_162 = wp::address(var_contact_efc_address_in, var_5, var_157);
                    var_164 = wp::load(var_162);
                    var_163 = wp::copy(var_164);
                    // if efcid1 < 0:                                                             <L 2688>
                    var_166 = (var_163 < var_165);
                    if (var_166) {
                        // continue                                                               <L 2689>
                        goto start_for_7;
                    }
                    // dm_fri1 = dm * fri[dim1id - 1]                                             <L 2690>
                    var_168 = wp::sub(var_157, var_167);
                    var_169 = wp::extract(var_85, var_168);
                    var_170 = wp::mul(var_102, var_169);
                }
                var_171 = wp::where(var_159, var_160, var_163);
                var_172 = wp::where(var_159, var_161, var_170);
                // efc_J11 = efc_J_in[worldid, efcid1, c_dof1]                                    <L 2693>
                var_173 = wp::address(var_efc_J_in, var_12, var_171, var_82);
                var_175 = wp::load(var_173);
                var_174 = wp::copy(var_175);
                // efc_J12 = efc_J_in[worldid, efcid1, c_dof2]                                    <L 2694>
                var_176 = wp::address(var_efc_J_in, var_12, var_171, var_83);
                var_178 = wp::load(var_176);
                var_177 = wp::copy(var_178);
                // ui = u[dim1id]                                                                 <L 2696>
                var_179 = wp::extract(var_114, var_157);
                // for dim2id in range(0, dim1id + 1):                                            <L 2698>
                var_181 = wp::add(var_157, var_180);
                var_183 = wp::range(var_182, var_181);
                start_for_9:;
                    if (iter_cmp(var_183) == 0) goto end_for_9;
                    var_184 = wp::iter_next(var_183);
                    // if dim2id == 0:                                                            <L 2699>
                    var_186 = (var_184 == var_185);
                    if (var_186) {
                        // efcid2 = efcid0                                                        <L 2700>
                        var_187 = wp::copy(var_31);
                        // dm_fri12 = dm_fri1 * mu                                                <L 2701>
                        var_188 = wp::mul(var_172, var_95);
                    }
                    if (!var_186) {
                        // efcid2 = contact_efc_address_in[conid, dim2id]                         <L 2703>
                        var_189 = wp::address(var_contact_efc_address_in, var_5, var_184);
                        var_191 = wp::load(var_189);
                        var_190 = wp::copy(var_191);
                        // if efcid2 < 0:                                                         <L 2704>
                        var_193 = (var_190 < var_192);
                        if (var_193) {
                            // continue                                                           <L 2705>
                            goto start_for_9;
                        }
                        // dm_fri12 = dm_fri1 * fri[dim2id - 1]                                   <L 2706>
                        var_195 = wp::sub(var_184, var_194);
                        var_196 = wp::extract(var_85, var_195);
                        var_197 = wp::mul(var_172, var_196);
                    }
                    var_198 = wp::where(var_186, var_187, var_190);
                    var_199 = wp::where(var_186, var_188, var_197);
                    // efc_J21 = efc_J_in[worldid, efcid2, c_dof1]                                <L 2709>
                    var_200 = wp::address(var_efc_J_in, var_12, var_198, var_82);
                    var_202 = wp::load(var_200);
                    var_201 = wp::copy(var_202);
                    // efc_J22 = efc_J_in[worldid, efcid2, c_dof2]                                <L 2710>
                    var_203 = wp::address(var_efc_J_in, var_12, var_198, var_83);
                    var_205 = wp::load(var_203);
                    var_204 = wp::copy(var_205);
                    // uj = u[dim2id]                                                             <L 2712>
                    var_206 = wp::extract(var_114, var_184);
                    // if dim1id == 0 and dim2id == 0:                                            <L 2715>
                    var_209 = (var_157 == var_208);
                    var_207 = var_209;
                    if (var_207) {
                        var_211 = (var_184 == var_210);
                        var_207 = var_207 && var_211;
                    }
                    if (var_207) {
                        // hcone = 1.0                                                            <L 2716>
                    }
                    if (!var_207) {
                        // elif dim1id == 0:                                                      <L 2717>
                        var_214 = (var_157 == var_213);
                        if (var_214) {
                            // hcone = -mu_over_t * uj                                            <L 2718>
                            var_215 = wp::neg(var_148);
                            var_216 = wp::mul(var_215, var_206);
                        }
                        var_217 = wp::where(var_214, var_216, var_212);
                        if (!var_214) {
                            // elif dim2id == 0:                                                  <L 2719>
                            var_219 = (var_184 == var_218);
                            if (var_219) {
                                // hcone = -mu_over_t * ui                                        <L 2720>
                                var_220 = wp::neg(var_148);
                                var_221 = wp::mul(var_220, var_179);
                            }
                            var_222 = wp::where(var_219, var_221, var_217);
                            if (!var_219) {
                                // hcone = mu_n_over_ttt * ui * uj                                <L 2722>
                                var_223 = wp::mul(var_150, var_179);
                                var_224 = wp::mul(var_223, var_206);
                                // if dim1id == dim2id:                                           <L 2725>
                                var_225 = (var_157 == var_184);
                                if (var_225) {
                                    // hcone += mu2_minus_mu_n_over_t                             <L 2726>
                                    var_226 = wp::add(var_224, var_153);
                                }
                                var_227 = wp::where(var_225, var_226, var_224);
                            }
                            var_228 = wp::where(var_219, var_222, var_227);
                        }
                        var_229 = wp::where(var_214, var_217, var_228);
                    }
                    var_230 = wp::where(var_207, var_212, var_229);
                    // hcone *= dm_fri12                                                          <L 2728>
                    var_231 = wp::mul(var_230, var_199);
                    // if hcone != 0.0:                                                           <L 2730>
                    var_233 = (var_231 != var_232);
                    if (var_233) {
                        // h += hcone * efc_J11 * efc_J22                                         <L 2731>
                        var_234 = wp::mul(var_231, var_174);
                        var_235 = wp::mul(var_234, var_204);
                        var_236 = wp::add(var_155, var_235);
                        // if dim1id != dim2id:                                                   <L 2733>
                        var_237 = (var_157 != var_184);
                        if (var_237) {
                            // h += hcone * efc_J12 * efc_J21                                     <L 2734>
                            var_238 = wp::mul(var_231, var_177);
                            var_239 = wp::mul(var_238, var_201);
                            var_240 = wp::add(var_236, var_239);
                        }
                        var_241 = wp::where(var_237, var_240, var_236);
                    }
                    var_242 = wp::where(var_233, var_241, var_155);
                    wp::assign(var_132, var_206);
                    wp::assign(var_155, var_242);
                    goto start_for_9;
                end_for_9:;
                goto start_for_7;
            end_for_7:;
            // wp.atomic_add(ctx_h_out[worldid, c_dof1], c_dof2, h)                               <L 2737>
            var_243 = wp::slice_t(var_12, var_12, var_244);
            var_245 = wp::slice_t(var_82, var_82, var_246);
            var_247 = wp::view(var_ctx_h_out, var_243, var_245);
            var_248 = wp::atomic_add(var_247, var_83, var_155);
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void _update_gradient_JTCJ_dense_f52c2ac2_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_opt_impratio_invsqrt,
    wp::array_t<wp::int32> var_dof_tri_row,
    wp::array_t<wp::int32> var_dof_tri_col,
    wp::array_t<wp::float32> var_contact_dist_in,
    wp::array_t<wp::float32> var_contact_includemargin_in,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_in,
    wp::array_t<wp::int32> var_contact_dim_in,
    wp::array_t<wp::int32> var_contact_efc_address_in,
    wp::array_t<wp::int32> var_contact_worldid_in,
    wp::array_t<wp::float32> var_efc_J_in,
    wp::array_t<wp::float32> var_efc_D_in,
    wp::array_t<wp::int32> var_efc_state_in,
    wp::int32 var_naconmax_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::array_t<wp::float32> var_ctx_Jaref_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::int32 var_nblocks_perblock,
    wp::int32 var_dim_block,
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
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::range_t var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        const wp::int32 var_12 = 0;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        bool var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        bool* var_20;
        bool var_21;
        bool var_22;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        const wp::int32 var_26 = 1;
        bool var_27;
        wp::float32* var_28;
        wp::float32* var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        const wp::float32 var_33 = 0.0;
        bool var_34;
        const wp::int32 var_35 = 0;
        wp::int32* var_36;
        wp::int32 var_37;
        wp::int32 var_38;
        const wp::int32 var_39 = 0;
        bool var_40;
        wp::int32* var_41;
        const wp::int32 var_42 = 4;
        bool var_43;
        wp::int32 var_44;
        wp::vec_t<5, wp::float32>* var_45;
        wp::vec_t<5, wp::float32> var_46;
        wp::vec_t<5, wp::float32> var_47;
        const wp::int32 var_48 = 0;
        wp::float32 var_49;
        wp::shape_t* var_50;
        const wp::int32 var_51 = 0;
        wp::int32 var_52;
        wp::shape_t var_53;
        wp::int32 var_54;
        wp::float32* var_55;
        wp::float32 var_56;
        wp::float32 var_57;
        wp::float32 var_58;
        wp::float32* var_59;
        const wp::float32 var_60 = 1.0;
        wp::float32 var_61;
        wp::float32 var_62;
        wp::float32 var_63;
        wp::float32 var_64;
        const wp::float32 var_65 = 0.0;
        bool var_66;
        wp::float32* var_67;
        wp::float32 var_68;
        wp::float32 var_69;
        wp::float32* var_70;
        wp::float32 var_71;
        wp::float32 var_72;
        wp::float32* var_73;
        wp::float32 var_74;
        wp::float32 var_75;
        const wp::float32 var_76 = 0.0;
        wp::float32 var_77;
        const wp::float32 var_78 = 0.0;
        wp::float32 var_79;
        const wp::float32 var_80 = 0.0;
        wp::float32 var_81;
        const wp::float32 var_82 = 0.0;
        wp::float32 var_83;
        const wp::int32 var_84 = 1;
        wp::range_t var_85;
        wp::int32 var_86;
        wp::int32* var_87;
        wp::int32 var_88;
        wp::int32 var_89;
        const wp::int32 var_90 = 0;
        bool var_91;
        const wp::int32 var_92 = 1;
        wp::int32 var_93;
        wp::float32 var_94;
        wp::float32* var_95;
        wp::float32 var_96;
        wp::float32 var_97;
        wp::float32* var_98;
        wp::float32 var_99;
        wp::float32 var_100;
        wp::float32* var_101;
        wp::float32 var_102;
        wp::float32 var_103;
        wp::float32 var_104;
        wp::float32 var_105;
        wp::float32 var_106;
        wp::float32 var_107;
        wp::float32 var_108;
        wp::float32 var_109;
        wp::float32 var_110;
        wp::float32 var_111;
        wp::float32 var_112;
        wp::float32 var_113;
        wp::float32 var_114;
        wp::float32 var_115;
        wp::float32 var_116;
        const wp::float32 var_117 = 1e-15;
        const wp::float32 var_118 = 1e-15;
        wp::float32 var_119;
        wp::float32 var_120;
        wp::float32 var_121;
        const wp::float32 var_122 = 1e-15;
        const wp::float32 var_123 = 1e-15;
        wp::float32 var_124;
        wp::float32 var_125;
        wp::float32 var_126;
        wp::float32 var_127;
        wp::float32 var_128;
        wp::float32 var_129;
        wp::float32 var_130;
        wp::float32 var_131;
        //---------
        // forward
        // def _update_gradient_JTCJ_dense(                                                       <L 2763>
        // conid_start, elementid = wp.tid()                                                      <L 2788>
        builtin_tid2d(var_0, var_1);
        // dof1id = dof_tri_row[elementid]                                                        <L 2790>
        var_2 = wp::address(var_dof_tri_row, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // dof2id = dof_tri_col[elementid]                                                        <L 2791>
        var_5 = wp::address(var_dof_tri_col, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // for i in range(nblocks_perblock):                                                      <L 2793>
        var_8 = wp::range(var_nblocks_perblock);
        start_for_0:;
            if (iter_cmp(var_8) == 0) goto end_for_0;
            var_9 = wp::iter_next(var_8);
            // conid = conid_start + i * dim_block                                                <L 2794>
            var_10 = wp::mul(var_9, var_dim_block);
            var_11 = wp::add(var_0, var_10);
            // if conid >= min(nacon_in[0], naconmax_in):                                         <L 2796>
            var_13 = wp::address(var_nacon_in, var_12);
            var_15 = wp::load(var_13);
            var_14 = wp::min(var_15, var_naconmax_in);
            var_16 = (var_11 >= var_14);
            if (var_16) {
                // return                                                                         <L 2797>
                continue;
            }
            // worldid = contact_worldid_in[conid]                                                <L 2799>
            var_17 = wp::address(var_contact_worldid_in, var_11);
            var_19 = wp::load(var_17);
            var_18 = wp::copy(var_19);
            // if ctx_done_in[worldid]:                                                           <L 2800>
            var_20 = wp::address(var_ctx_done_in, var_18);
            var_21 = wp::load(var_20);
            if (var_21) {
                // continue                                                                       <L 2801>
                goto start_for_0;
            }
            var_22 = wp::load(var_20);
            // condim = contact_dim_in[conid]                                                     <L 2803>
            var_23 = wp::address(var_contact_dim_in, var_11);
            var_25 = wp::load(var_23);
            var_24 = wp::copy(var_25);
            // if condim == 1:                                                                    <L 2805>
            var_27 = (var_24 == var_26);
            if (var_27) {
                // continue                                                                       <L 2806>
                goto start_for_0;
            }
            // if contact_dist_in[conid] - contact_includemargin_in[conid] >= 0.0:                <L 2809>
            var_28 = wp::address(var_contact_dist_in, var_11);
            var_29 = wp::address(var_contact_includemargin_in, var_11);
            var_31 = wp::load(var_28);
            var_32 = wp::load(var_29);
            var_30 = wp::sub(var_31, var_32);
            var_34 = (var_30 >= var_33);
            if (var_34) {
                // continue                                                                       <L 2810>
                goto start_for_0;
            }
            // efcid0 = contact_efc_address_in[conid, 0]                                          <L 2812>
            var_36 = wp::address(var_contact_efc_address_in, var_11, var_35);
            var_38 = wp::load(var_36);
            var_37 = wp::copy(var_38);
            // if efcid0 < 0:                                                                     <L 2813>
            var_40 = (var_37 < var_39);
            if (var_40) {
                // continue                                                                       <L 2814>
                goto start_for_0;
            }
            // if efc_state_in[worldid, efcid0] != types.ConstraintState.CONE:                    <L 2815>
            var_41 = wp::address(var_efc_state_in, var_18, var_37);
            var_44 = wp::load(var_41);
            var_43 = (var_44 != var_42);
            if (var_43) {
                // continue                                                                       <L 2816>
                goto start_for_0;
            }
            // fri = contact_friction_in[conid]                                                   <L 2818>
            var_45 = wp::address(var_contact_friction_in, var_11);
            var_47 = wp::load(var_45);
            var_46 = wp::copy(var_47);
            // mu = fri[0] * opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]        <L 2819>
            var_49 = wp::extract(var_46, var_48);
            var_50 = &(var_opt_impratio_invsqrt.shape);
            var_53 = wp::load(var_50);
            var_52 = wp::extract(var_53, var_51);
            var_54 = wp::mod(var_18, var_52);
            var_55 = wp::address(var_opt_impratio_invsqrt, var_54);
            var_57 = wp::load(var_55);
            var_56 = wp::mul(var_49, var_57);
            // mu2 = mu * mu                                                                      <L 2821>
            var_58 = wp::mul(var_56, var_56);
            // dm = math.safe_div(efc_D_in[worldid, efcid0], mu2 * (1.0 + mu2))                   <L 2822>
            var_59 = wp::address(var_efc_D_in, var_18, var_37);
            var_61 = wp::add(var_60, var_58);
            var_62 = wp::mul(var_58, var_61);
            var_64 = wp::load(var_59);
            var_63 = safe_div_0(var_64, var_62);
            // if dm == 0.0:                                                                      <L 2824>
            var_66 = (var_63 == var_65);
            if (var_66) {
                // continue                                                                       <L 2825>
                goto start_for_0;
            }
            // n = ctx_Jaref_in[worldid, efcid0] * mu                                             <L 2827>
            var_67 = wp::address(var_ctx_Jaref_in, var_18, var_37);
            var_69 = wp::load(var_67);
            var_68 = wp::mul(var_69, var_56);
            // z01 = mu * efc_J_in[worldid, efcid0, dof1id]                                       <L 2828>
            var_70 = wp::address(var_efc_J_in, var_18, var_37, var_3);
            var_72 = wp::load(var_70);
            var_71 = wp::mul(var_56, var_72);
            // z02 = mu * efc_J_in[worldid, efcid0, dof2id]                                       <L 2829>
            var_73 = wp::address(var_efc_J_in, var_18, var_37, var_6);
            var_75 = wp::load(var_73);
            var_74 = wp::mul(var_56, var_75);
            // tt = float(0.0)                                                                    <L 2830>
            var_77 = wp::float(var_76);
            // projection1 = float(0.0)                                                           <L 2831>
            var_79 = wp::float(var_78);
            // projection2 = float(0.0)                                                           <L 2832>
            var_81 = wp::float(var_80);
            // tangent_dot = float(0.0)                                                           <L 2833>
            var_83 = wp::float(var_82);
            // for dim in range(1, condim):                                                       <L 2834>
            var_85 = wp::range(var_84, var_24);
            start_for_3:;
                if (iter_cmp(var_85) == 0) goto end_for_3;
                var_86 = wp::iter_next(var_85);
                // efcid = contact_efc_address_in[conid, dim]                                     <L 2835>
                var_87 = wp::address(var_contact_efc_address_in, var_11, var_86);
                var_89 = wp::load(var_87);
                var_88 = wp::copy(var_89);
                // if efcid >= 0:                                                                 <L 2836>
                var_91 = (var_88 >= var_90);
                if (var_91) {
                    // scale = fri[dim - 1]                                                       <L 2837>
                    var_93 = wp::sub(var_86, var_92);
                    var_94 = wp::extract(var_46, var_93);
                    // u = ctx_Jaref_in[worldid, efcid] * scale                                   <L 2838>
                    var_95 = wp::address(var_ctx_Jaref_in, var_18, var_88);
                    var_97 = wp::load(var_95);
                    var_96 = wp::mul(var_97, var_94);
                    // z1 = scale * efc_J_in[worldid, efcid, dof1id]                              <L 2839>
                    var_98 = wp::address(var_efc_J_in, var_18, var_88, var_3);
                    var_100 = wp::load(var_98);
                    var_99 = wp::mul(var_94, var_100);
                    // z2 = scale * efc_J_in[worldid, efcid, dof2id]                              <L 2840>
                    var_101 = wp::address(var_efc_J_in, var_18, var_88, var_6);
                    var_103 = wp::load(var_101);
                    var_102 = wp::mul(var_94, var_103);
                    // tt += u * u                                                                <L 2841>
                    var_104 = wp::mul(var_96, var_96);
                    var_105 = wp::add(var_77, var_104);
                    // projection1 += u * z1                                                      <L 2842>
                    var_106 = wp::mul(var_96, var_99);
                    var_107 = wp::add(var_79, var_106);
                    // projection2 += u * z2                                                      <L 2843>
                    var_108 = wp::mul(var_96, var_102);
                    var_109 = wp::add(var_81, var_108);
                    // tangent_dot += z1 * z2                                                     <L 2844>
                    var_110 = wp::mul(var_99, var_102);
                    var_111 = wp::add(var_83, var_110);
                }
                var_112 = wp::where(var_91, var_105, var_77);
                var_113 = wp::where(var_91, var_107, var_79);
                var_114 = wp::where(var_91, var_109, var_81);
                var_115 = wp::where(var_91, var_111, var_83);
                wp::assign(var_77, var_112);
                wp::assign(var_79, var_113);
                wp::assign(var_81, var_114);
                wp::assign(var_83, var_115);
                goto start_for_3;
            end_for_3:;
            // t = wp.max(wp.sqrt(tt), types.MJ_MINVAL)                                           <L 2846>
            var_116 = wp::sqrt(var_77);
            var_119 = wp::max(var_116, var_118);
            // ttt = wp.max(t * t * t, types.MJ_MINVAL)                                           <L 2847>
            var_120 = wp::mul(var_119, var_119);
            var_121 = wp::mul(var_120, var_119);
            var_124 = wp::max(var_121, var_123);
            // mu_tinv = math.safe_div(mu, t)                                                     <L 2848>
            var_125 = safe_div_0(var_56, var_119);
            // h = _elliptic_hessian_entry_from_projections(                                      <L 2849>
            // dm,                                                                                <L 2850>
            // mu_tinv,                                                                           <L 2851>
            // mu * math.safe_div(n, ttt),                                                        <L 2852>
            var_126 = safe_div_0(var_68, var_124);
            var_127 = wp::mul(var_56, var_126);
            // mu2 - n * mu_tinv,                                                                 <L 2853>
            var_128 = wp::mul(var_68, var_125);
            var_129 = wp::sub(var_58, var_128);
            // z01,                                                                               <L 2854>
            // z02,                                                                               <L 2855>
            // projection1,                                                                       <L 2856>
            // projection2,                                                                       <L 2857>
            // tangent_dot,                                                                       <L 2858>
            var_130 = _elliptic_hessian_entry_from_projections_0(var_63, var_125, var_127, var_129, var_71, var_74, var_79, var_81, var_83);
            // ctx_h_out[worldid, dof1id, dof2id] += h                                            <L 2861>
            var_131 = wp::atomic_add(var_ctx_h_out, var_18, var_3, var_6, var_130);
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void _solve_cg_finalize_97c56132_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_nv,
    wp::array_t<wp::float32> var_opt_tolerance,
    wp::int32 var_opt_iterations,
    wp::array_t<wp::float32> var_stat_meaninertia,
    wp::array_t<wp::float32> var_ctx_beta_num_in,
    wp::array_t<wp::float32> var_ctx_beta_den_in,
    wp::array_t<wp::float32> var_ctx_improvement_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_ctx_grad_dot_in,
    wp::array_t<wp::int32> var_solver_niter_out,
    wp::array_t<wp::float32> var_ctx_beta_out,
    wp::array_t<wp::int32> var_nsolving_out,
    wp::array_t<bool> var_ctx_done_out)
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
        bool* var_1;
        bool var_2;
        bool var_3;
        const wp::float32 var_4 = 0.0;
        wp::float32* var_5;
        const wp::float32 var_6 = 1e-15;
        const wp::float32 var_7 = 1e-15;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32 var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        const wp::int32 var_14 = 1;
        wp::int32 var_15;
        wp::shape_t* var_16;
        const wp::int32 var_17 = 0;
        wp::int32 var_18;
        wp::shape_t var_19;
        wp::int32 var_20;
        wp::float32* var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::shape_t* var_24;
        const wp::int32 var_25 = 0;
        wp::int32 var_26;
        wp::shape_t var_27;
        wp::int32 var_28;
        wp::float32* var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        wp::float32* var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        wp::float32* var_35;
        wp::float32 var_36;
        wp::float32 var_37;
        wp::float32 var_38;
        wp::float32 var_39;
        bool var_40;
        bool var_41;
        bool var_42;
        bool var_43;
        wp::int32* var_44;
        bool var_45;
        wp::int32 var_46;
        const bool var_47 = true;
        const wp::int32 var_48 = 0;
        const wp::int32 var_49 = -1;
        wp::int32 var_50;
        //---------
        // forward
        // def _solve_cg_finalize(                                                                <L 3660>
        // worldid = wp.tid()                                                                     <L 3679>
        var_0 = builtin_tid1d();
        // if ctx_done_in[worldid]:                                                               <L 3681>
        var_1 = wp::address(var_ctx_done_in, var_0);
        var_2 = wp::load(var_1);
        if (var_2) {
            // return                                                                             <L 3682>
            continue;
        }
        var_3 = wp::load(var_1);
        // ctx_beta_out[worldid] = wp.max(0.0, ctx_beta_num_in[worldid] / wp.max(types.MJ_MINVAL, ctx_beta_den_in[worldid]))       <L 3685>
        var_5 = wp::address(var_ctx_beta_num_in, var_0);
        var_8 = wp::address(var_ctx_beta_den_in, var_0);
        var_10 = wp::load(var_8);
        var_9 = wp::max(var_7, var_10);
        var_12 = wp::load(var_5);
        var_11 = wp::div(var_12, var_9);
        var_13 = wp::max(var_4, var_11);
        wp::array_store(var_ctx_beta_out, var_0, var_13);
        // solver_niter_out[worldid] += 1                                                         <L 3688>
        var_15 = wp::atomic_add(var_solver_niter_out, var_0, var_14);
        // tolerance = opt_tolerance[worldid % opt_tolerance.shape[0]]                            <L 3689>
        var_16 = &(var_opt_tolerance.shape);
        var_19 = wp::load(var_16);
        var_18 = wp::extract(var_19, var_17);
        var_20 = wp::mod(var_0, var_18);
        var_21 = wp::address(var_opt_tolerance, var_20);
        var_23 = wp::load(var_21);
        var_22 = wp::copy(var_23);
        // meaninertia = stat_meaninertia[worldid % stat_meaninertia.shape[0]]                    <L 3690>
        var_24 = &(var_stat_meaninertia.shape);
        var_27 = wp::load(var_24);
        var_26 = wp::extract(var_27, var_25);
        var_28 = wp::mod(var_0, var_26);
        var_29 = wp::address(var_stat_meaninertia, var_28);
        var_31 = wp::load(var_29);
        var_30 = wp::copy(var_31);
        // grad_dot = ctx_grad_dot_in[worldid]                                                    <L 3692>
        var_32 = wp::address(var_ctx_grad_dot_in, var_0);
        var_34 = wp::load(var_32);
        var_33 = wp::copy(var_34);
        // improvement = _rescale(nv, meaninertia, ctx_improvement_in[worldid])                   <L 3694>
        var_35 = wp::address(var_ctx_improvement_in, var_0);
        var_37 = wp::load(var_35);
        var_36 = _rescale_0(var_nv, var_30, var_37);
        // gradient = _rescale(nv, meaninertia, wp.sqrt(grad_dot))                                <L 3695>
        var_38 = wp::sqrt(var_33);
        var_39 = _rescale_0(var_nv, var_30, var_38);
        // done = (improvement < tolerance) or (gradient < tolerance)                             <L 3696>
        var_41 = (var_36 < var_22);
        var_40 = var_41;
        if (!var_40) {
            var_42 = (var_39 < var_22);
            var_40 = var_40 || var_42;
        }
        // if done or solver_niter_out[worldid] == opt_iterations:                                <L 3697>
        var_43 = var_40;
        if (!var_43) {
            var_44 = wp::address(var_solver_niter_out, var_0);
            var_46 = wp::load(var_44);
            var_45 = (var_46 == var_opt_iterations);
            var_43 = var_43 || var_45;
        }
        if (var_43) {
            // ctx_done_out[worldid] = True                                                       <L 3698>
            wp::array_store(var_ctx_done_out, var_0, var_47);
            // wp.atomic_add(nsolving_out, 0, -1)                                                 <L 3699>
            var_50 = wp::atomic_add(var_nsolving_out, var_48, var_49);
        }
    }
}



extern "C" __global__ void _diag_precond_add_JTDJ_a3224b51_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_simple,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::int32> var_nefc_in,
    wp::array_t<wp::int32> var_efc_J_rownnz_in,
    wp::array_t<wp::int32> var_efc_J_rowadr_in,
    wp::array_t<wp::int32> var_efc_J_colind_in,
    wp::array_t<wp::float32> var_efc_J_in,
    wp::array_t<wp::float32> var_efc_D_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_diag_out)
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
        bool var_6;
        wp::int32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        const wp::float32 var_11 = 0.0;
        bool var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        wp::range_t var_19;
        wp::int32 var_20;
        const wp::int32 var_21 = 0;
        wp::int32 var_22;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        wp::int32* var_26;
        wp::int32* var_27;
        wp::int32 var_28;
        const wp::int32 var_29 = 2;
        bool var_30;
        wp::int32 var_31;
        const wp::int32 var_32 = 0;
        wp::int32 var_33;
        wp::float32* var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        const wp::float32 var_37 = 0.0;
        bool var_38;
        wp::float32 var_39;
        wp::float32 var_40;
        wp::float32 var_41;
        //---------
        // forward
        // def _diag_precond_add_JTDJ(                                                            <L 3196>
        // worldid, efcid = wp.tid()                                                              <L 3213>
        builtin_tid2d(var_0, var_1);
        // if ctx_done_in[worldid]:                                                               <L 3214>
        var_2 = wp::address(var_ctx_done_in, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // return                                                                             <L 3215>
            continue;
        }
        var_4 = wp::load(var_2);
        // if efcid >= nefc_in[worldid]:                                                          <L 3216>
        var_5 = wp::address(var_nefc_in, var_0);
        var_7 = wp::load(var_5);
        var_6 = (var_1 >= var_7);
        if (var_6) {
            // return                                                                             <L 3217>
            continue;
        }
        // D = efc_D_in[worldid, efcid]                                                           <L 3218>
        var_8 = wp::address(var_efc_D_in, var_0, var_1);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // if D == 0.0:                                                                           <L 3219>
        var_12 = (var_9 == var_11);
        if (var_12) {
            // return                                                                             <L 3220>
            continue;
        }
        // rownnz = efc_J_rownnz_in[worldid, efcid]                                               <L 3221>
        var_13 = wp::address(var_efc_J_rownnz_in, var_0, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // rowadr = efc_J_rowadr_in[worldid, efcid]                                               <L 3222>
        var_16 = wp::address(var_efc_J_rowadr_in, var_0, var_1);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // for i in range(rownnz):                                                                <L 3223>
        var_19 = wp::range(var_14);
        start_for_3:;
            if (iter_cmp(var_19) == 0) goto end_for_3;
            var_20 = wp::iter_next(var_19);
            // col = efc_J_colind_in[worldid, 0, rowadr + i]                                      <L 3224>
            var_22 = wp::add(var_17, var_20);
            var_23 = wp::address(var_efc_J_colind_in, var_0, var_21, var_22);
            var_25 = wp::load(var_23);
            var_24 = wp::copy(var_25);
            // if body_simple[dof_bodyid[col]] != 2:                                              <L 3225>
            var_26 = wp::address(var_dof_bodyid, var_24);
            var_28 = wp::load(var_26);
            var_27 = wp::address(var_body_simple, var_28);
            var_31 = wp::load(var_27);
            var_30 = (var_31 != var_29);
            if (var_30) {
                // continue                                                                       <L 3226>
                goto start_for_3;
            }
            // Jval = efc_J_in[worldid, 0, rowadr + i]                                            <L 3227>
            var_33 = wp::add(var_17, var_20);
            var_34 = wp::address(var_efc_J_in, var_0, var_32, var_33);
            var_36 = wp::load(var_34);
            var_35 = wp::copy(var_36);
            // if Jval != 0.0:                                                                    <L 3228>
            var_38 = (var_35 != var_37);
            if (var_38) {
                // wp.atomic_add(diag_out, worldid, col, D * Jval * Jval)                         <L 3229>
                var_39 = wp::mul(var_9, var_35);
                var_40 = wp::mul(var_39, var_35);
                var_41 = wp::atomic_add(var_diag_out, var_0, var_24, var_40);
            }
            goto start_for_3;
        end_for_3:;
    }
}



extern "C" __global__ void _gather_J_sparse_0dd96708_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_nefc_in,
    wp::array_t<wp::int32> var_dof_cdof_in,
    wp::array_t<wp::int32> var_J_rownnz_in,
    wp::array_t<wp::int32> var_J_rowadr_in,
    wp::array_t<wp::int32> var_J_colind_in,
    wp::array_t<wp::float32> var_J_in,
    wp::array_t<wp::float32> var_J_c_out)
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
        bool var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32* var_8;
        wp::range_t var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        const wp::int32 var_13 = 0;
        wp::int32* var_14;
        wp::int32* var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        const wp::int32 var_19 = 0;
        bool var_20;
        const wp::int32 var_21 = 0;
        wp::float32* var_22;
        wp::float32 var_23;
        //---------
        // forward
        // def _gather_J_sparse(                                                                  <L 4244>
        // worldid, efcid = wp.tid()                                                              <L 4256>
        builtin_tid2d(var_0, var_1);
        // if efcid >= nefc_in[worldid]:                                                          <L 4257>
        var_2 = wp::address(var_nefc_in, var_0);
        var_4 = wp::load(var_2);
        var_3 = (var_1 >= var_4);
        if (var_3) {
            // return                                                                             <L 4258>
            continue;
        }
        // rowadr = J_rowadr_in[worldid, efcid]                                                   <L 4259>
        var_5 = wp::address(var_J_rowadr_in, var_0, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // for k in range(J_rownnz_in[worldid, efcid]):                                           <L 4260>
        var_8 = wp::address(var_J_rownnz_in, var_0, var_1);
        var_10 = wp::load(var_8);
        var_9 = wp::range(var_10);
        start_for_1:;
            if (iter_cmp(var_9) == 0) goto end_for_1;
            var_11 = wp::iter_next(var_9);
            // adr = rowadr + k                                                                   <L 4261>
            var_12 = wp::add(var_6, var_11);
            // cj = dof_cdof_in[worldid, J_colind_in[worldid, 0, adr]]                            <L 4262>
            var_14 = wp::address(var_J_colind_in, var_0, var_13, var_12);
            var_16 = wp::load(var_14);
            var_15 = wp::address(var_dof_cdof_in, var_0, var_16);
            var_18 = wp::load(var_15);
            var_17 = wp::copy(var_18);
            // if cj >= 0:                                                                        <L 4263>
            var_20 = (var_17 >= var_19);
            if (var_20) {
                // J_c_out[worldid, efcid, cj] = J_in[worldid, 0, adr]                            <L 4264>
                var_22 = wp::address(var_J_in, var_0, var_21, var_12);
                var_23 = wp::load(var_22);
                wp::array_store(var_J_c_out, var_0, var_1, var_17, var_23);
            }
            goto start_for_1;
        end_for_1:;
    }
}



extern "C" __global__ void _diag_precond_apply_d65d9ee0_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_simple,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::float32> var_qLDiagInv_in,
    wp::array_t<wp::float32> var_diag_in,
    wp::array_t<wp::float32> var_grad_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_Mgrad_out)
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
        wp::int32* var_6;
        wp::int32 var_7;
        const wp::int32 var_8 = 2;
        bool var_9;
        wp::int32 var_10;
        wp::float32* var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        wp::float32* var_16;
        wp::float32* var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        //---------
        // forward
        // def _diag_precond_apply(                                                               <L 3233>
        // worldid, dofid = wp.tid()                                                              <L 3247>
        builtin_tid2d(var_0, var_1);
        // if ctx_done_in[worldid]:                                                               <L 3248>
        var_2 = wp::address(var_ctx_done_in, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // return                                                                             <L 3249>
            continue;
        }
        var_4 = wp::load(var_2);
        // if body_simple[dof_bodyid[dofid]] == 2:                                                <L 3250>
        var_5 = wp::address(var_dof_bodyid, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::address(var_body_simple, var_7);
        var_10 = wp::load(var_6);
        var_9 = (var_10 == var_8);
        if (var_9) {
            // Mgrad_out[worldid, dofid] = grad_in[worldid, dofid] / diag_in[worldid, dofid]       <L 3251>
            var_11 = wp::address(var_grad_in, var_0, var_1);
            var_12 = wp::address(var_diag_in, var_0, var_1);
            var_14 = wp::load(var_11);
            var_15 = wp::load(var_12);
            var_13 = wp::div(var_14, var_15);
            wp::array_store(var_Mgrad_out, var_0, var_1, var_13);
        }
        if (!var_9) {
            // Mgrad_out[worldid, dofid] = qLDiagInv_in[worldid, dofid] * grad_in[worldid, dofid]       <L 3253>
            var_16 = wp::address(var_qLDiagInv_in, var_0, var_1);
            var_17 = wp::address(var_grad_in, var_0, var_1);
            var_19 = wp::load(var_16);
            var_20 = wp::load(var_17);
            var_18 = wp::mul(var_19, var_20);
            wp::array_store(var_Mgrad_out, var_0, var_1, var_18);
        }
    }
}



extern "C" __global__ void _solve_beta_accumulate_95bec0bb_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_ctx_grad_in,
    wp::array_t<wp::float32> var_ctx_Mgrad_in,
    wp::array_t<wp::float32> var_ctx_prev_grad_in,
    wp::array_t<wp::float32> var_ctx_prev_Mgrad_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_ctx_beta_num_out,
    wp::array_t<wp::float32> var_ctx_beta_den_out)
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
        wp::float32* var_5;
        wp::float32 var_6;
        wp::float32 var_7;
        wp::float32* var_8;
        wp::float32* var_9;
        wp::float32 var_10;
        wp::float32 var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        //---------
        // forward
        // def _solve_beta_accumulate(                                                            <L 3594>
        // worldid, dofid = wp.tid()                                                              <L 3605>
        builtin_tid2d(var_0, var_1);
        // if ctx_done_in[worldid]:                                                               <L 3607>
        var_2 = wp::address(var_ctx_done_in, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // return                                                                             <L 3608>
            continue;
        }
        var_4 = wp::load(var_2);
        // prev_Mgrad = ctx_prev_Mgrad_in[worldid, dofid]                                         <L 3610>
        var_5 = wp::address(var_ctx_prev_Mgrad_in, var_0, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // num = ctx_grad_in[worldid, dofid] * (ctx_Mgrad_in[worldid, dofid] - prev_Mgrad)        <L 3611>
        var_8 = wp::address(var_ctx_grad_in, var_0, var_1);
        var_9 = wp::address(var_ctx_Mgrad_in, var_0, var_1);
        var_11 = wp::load(var_9);
        var_10 = wp::sub(var_11, var_6);
        var_13 = wp::load(var_8);
        var_12 = wp::mul(var_13, var_10);
        // den = ctx_prev_grad_in[worldid, dofid] * prev_Mgrad                                    <L 3612>
        var_14 = wp::address(var_ctx_prev_grad_in, var_0, var_1);
        var_16 = wp::load(var_14);
        var_15 = wp::mul(var_16, var_6);
        // wp.atomic_add(ctx_beta_num_out, worldid, num)                                          <L 3613>
        var_17 = wp::atomic_add(var_ctx_beta_num_out, var_0, var_12);
        // wp.atomic_add(ctx_beta_den_out, worldid, den)                                          <L 3614>
        var_18 = wp::atomic_add(var_ctx_beta_den_out, var_0, var_15);
    }
}



extern "C" __global__ void _scatter_dof_vecs_54d8b54e_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_dof_cdof_in,
    wp::array_t<wp::float32> var_qacc_c_in,
    wp::array_t<wp::float32> var_qfrc_constraint_c_in,
    wp::array_t<wp::float32> var_qacc_out,
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
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 0;
        bool var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        wp::float32* var_9;
        wp::float32 var_10;
        const wp::float32 var_11 = 0.0;
        const wp::float32 var_12 = 0.0;
        //---------
        // forward
        // def _scatter_dof_vecs(                                                                 <L 4223>
        // worldid, i = wp.tid()                                                                  <L 4233>
        builtin_tid2d(var_0, var_1);
        // ci = dof_cdof_in[worldid, i]                                                           <L 4234>
        var_2 = wp::address(var_dof_cdof_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if ci >= 0:                                                                            <L 4235>
        var_6 = (var_3 >= var_5);
        if (var_6) {
            // qacc_out[worldid, i] = qacc_c_in[worldid, ci]                                      <L 4236>
            var_7 = wp::address(var_qacc_c_in, var_0, var_3);
            var_8 = wp::load(var_7);
            wp::array_store(var_qacc_out, var_0, var_1, var_8);
            // qfrc_constraint_out[worldid, i] = qfrc_constraint_c_in[worldid, ci]                <L 4237>
            var_9 = wp::address(var_qfrc_constraint_c_in, var_0, var_3);
            var_10 = wp::load(var_9);
            wp::array_store(var_qfrc_constraint_out, var_0, var_1, var_10);
        }
        if (!var_6) {
            // qacc_out[worldid, i] = 0.0                                                         <L 4239>
            wp::array_store(var_qacc_out, var_0, var_1, var_11);
            // qfrc_constraint_out[worldid, i] = 0.0                                              <L 4240>
            wp::array_store(var_qfrc_constraint_out, var_0, var_1, var_12);
        }
    }
}



extern "C" __global__ void _solve_init_search_cg_tiled_5992b048_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_nv,
    wp::array_t<wp::float32> var_ctx_grad_in,
    wp::array_t<wp::float32> var_ctx_Mgrad_in,
    wp::array_t<wp::float32> var_ctx_search_out,
    wp::array_t<wp::float32> var_ctx_search_dot_out,
    wp::array_t<wp::float32> var_ctx_prev_grad_out,
    wp::array_t<wp::float32> var_ctx_prev_Mgrad_out)
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
        const wp::float32 var_2 = 0.0;
        wp::float32 var_3;
        wp::int32 var_4;
        wp::range_t var_5;
        wp::int32 var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        const wp::float32 var_10 = -1.0;
        wp::float32 var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32* var_14;
        wp::float32 var_15;
        const bool var_16 = true;
        wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>> var_17 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>>{};
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_18 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
        const wp::int32 var_19 = 0;
        bool var_20;
        const wp::int32 var_21 = 0;
        wp::float32 var_22;
        //---------
        // forward
        // def _solve_init_search_cg_tiled(                                                       <L 1617>
        // worldid, tid = wp.tid()                                                                <L 1629>
        builtin_tid2d(var_0, var_1);
        // local_search_dot = float(0.0)                                                          <L 1631>
        var_3 = wp::float(var_2);
        // BLOCK_DIM = wp.block_dim()                                                             <L 1632>
        var_4 = builtin_block_dim();
        // for dofid in range(tid, nv, BLOCK_DIM):                                                <L 1634>
        var_5 = wp::range(var_1, var_nv, var_4);
        start_for_0:;
            if (iter_cmp(var_5) == 0) goto end_for_0;
            var_6 = wp::iter_next(var_5);
            // mgrad = ctx_Mgrad_in[worldid, dofid]                                               <L 1635>
            var_7 = wp::address(var_ctx_Mgrad_in, var_0, var_6);
            var_9 = wp::load(var_7);
            var_8 = wp::copy(var_9);
            // search = -1.0 * mgrad                                                              <L 1636>
            var_11 = wp::mul(var_10, var_8);
            // ctx_search_out[worldid, dofid] = search                                            <L 1637>
            wp::array_store(var_ctx_search_out, var_0, var_6, var_11);
            // local_search_dot += search * search                                                <L 1638>
            var_12 = wp::mul(var_11, var_11);
            var_13 = wp::add(var_3, var_12);
            // ctx_prev_grad_out[worldid, dofid] = ctx_grad_in[worldid, dofid]                    <L 1640>
            var_14 = wp::address(var_ctx_grad_in, var_0, var_6);
            var_15 = wp::load(var_14);
            wp::array_store(var_ctx_prev_grad_out, var_0, var_6, var_15);
            // ctx_prev_Mgrad_out[worldid, dofid] = mgrad                                         <L 1641>
            wp::array_store(var_ctx_prev_Mgrad_out, var_0, var_6, var_8);
            wp::assign(var_3, var_13);
            goto start_for_0;
        end_for_0:;
        // search_dot_tile = wp.tile(local_search_dot, preserve_type=True)                        <L 1643>
        var_17 = wp::tile<wp::float32>(var_3);
        // search_dot_sum = wp.tile_reduce(wp.add, search_dot_tile)                               <L 1644>
        var_18 = wp::tile_reduce(wp::add, var_17);
        // if tid == 0:                                                                           <L 1646>
        var_20 = (var_1 == var_19);
        if (var_20) {
            // ctx_search_dot_out[worldid] = search_dot_sum[0]                                    <L 1647>
            var_22 = wp::tile_extract(var_18, var_21);
            wp::array_store(var_ctx_search_dot_out, var_0, var_22);
        }
    }
}



extern "C" __global__ void _zero_qfrc_constraint_sparse_6b3cbe94_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
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
        const wp::float32 var_9 = 0.0;
        //---------
        // forward
        // def _zero_qfrc_constraint_sparse(                                                      <L 1778>
        // worldid, dofid = wp.tid()                                                              <L 1786>
        builtin_tid2d(var_0, var_1);
        // if ctx_done_in[worldid]:                                                               <L 1788>
        var_2 = wp::address(var_ctx_done_in, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // return                                                                             <L 1789>
            continue;
        }
        var_4 = wp::load(var_2);
        // if state_changed_count_in[worldid] == 0:                                               <L 1791>
        var_5 = wp::address(var_state_changed_count_in, var_0);
        var_8 = wp::load(var_5);
        var_7 = (var_8 == var_6);
        if (var_7) {
            // return                                                                             <L 1792>
            continue;
        }
        // qfrc_constraint_out[worldid, dofid] = 0.0                                              <L 1794>
        wp::array_store(var_qfrc_constraint_out, var_0, var_1, var_9);
    }
}



extern "C" __global__ void _scatter_solution_3c25285e_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_dof_cdof_in,
    wp::array_t<wp::float32> var_x_in,
    wp::array_t<wp::float32> var_vec_out)
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
        const wp::int32 var_5 = 0;
        bool var_6;
        const wp::int32 var_7 = 0;
        wp::float32* var_8;
        wp::float32 var_9;
        const wp::float32 var_10 = 0.0;
        //---------
        // forward
        // def _scatter_solution(                                                                 <L 4148>
        // worldid, i = wp.tid()                                                                  <L 4156>
        builtin_tid2d(var_0, var_1);
        // ci = dof_cdof_in[worldid, i]                                                           <L 4157>
        var_2 = wp::address(var_dof_cdof_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if ci >= 0:                                                                            <L 4158>
        var_6 = (var_3 >= var_5);
        if (var_6) {
            // vec_out[worldid, i] = x_in[worldid, ci, 0]                                         <L 4159>
            var_8 = wp::address(var_x_in, var_0, var_3, var_7);
            var_9 = wp::load(var_8);
            wp::array_store(var_vec_out, var_0, var_1, var_9);
        }
        if (!var_6) {
            // vec_out[worldid, i] = 0.0  # frozen inactive DOF                                   <L 4161>
            wp::array_store(var_vec_out, var_0, var_1, var_10);
        }
    }
}



extern "C" __global__ void _solve_init_efc_a1db7ec5_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::int32> var_solver_niter_out,
    wp::array_t<wp::float32> var_ctx_search_dot_out,
    wp::array_t<bool> var_ctx_done_out)
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
        const bool var_2 = false;
        const wp::float32 var_3 = 0.0;
        //---------
        // forward
        // def _solve_init_efc(                                                                   <L 1547>
        // worldid = wp.tid()                                                                     <L 1554>
        var_0 = builtin_tid1d();
        // solver_niter_out[worldid] = 0                                                          <L 1555>
        wp::array_store(var_solver_niter_out, var_0, var_1);
        // ctx_done_out[worldid] = False                                                          <L 1556>
        wp::array_store(var_ctx_done_out, var_0, var_2);
        // ctx_search_dot_out[worldid] = 0.0                                                      <L 1557>
        wp::array_store(var_ctx_search_dot_out, var_0, var_3);
    }
}



extern "C" __global__ void _solve_beta_zero_8deac91b_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::float32> var_ctx_beta_num_out,
    wp::array_t<wp::float32> var_ctx_beta_den_out)
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
        const wp::float32 var_1 = 0.0;
        const wp::float32 var_2 = 0.0;
        //---------
        // forward
        // def _solve_beta_zero(                                                                  <L 3542>
        // worldid = wp.tid()                                                                     <L 3547>
        var_0 = builtin_tid1d();
        // ctx_beta_num_out[worldid] = 0.0                                                        <L 3548>
        wp::array_store(var_ctx_beta_num_out, var_0, var_1);
        // ctx_beta_den_out[worldid] = 0.0                                                        <L 3549>
        wp::array_store(var_ctx_beta_den_out, var_0, var_2);
    }
}



extern "C" __global__ void _mul_m_sparse_compact_fbf15af2_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_M_mulm_rowadr,
    wp::array_t<wp::int32> var_M_mulm_col,
    wp::array_t<wp::int32> var_M_mulm_madr,
    wp::array_t<wp::float32> var_M_in,
    wp::array_t<wp::int32> var_dof_cdof_in,
    wp::array_t<wp::int32> var_cdof_dof_in,
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
        bool* var_2;
        bool var_3;
        bool var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        const wp::int32 var_8 = 0;
        bool var_9;
        const wp::float32 var_10 = 0.0;
        const wp::float32 var_11 = 0.0;
        wp::float32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        const wp::int32 var_16 = 1;
        wp::int32 var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::range_t var_21;
        wp::int32 var_22;
        wp::int32* var_23;
        wp::int32* var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        wp::int32 var_27;
        const wp::int32 var_28 = 0;
        bool var_29;
        wp::int32* var_30;
        wp::float32* var_31;
        wp::int32 var_32;
        wp::float32* var_33;
        wp::float32 var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        wp::float32 var_37;
        wp::float32 var_38;
        //---------
        // forward
        // def _mul_m_sparse_compact(                                                             <L 4037>
        // worldid, ci = wp.tid()                                                                 <L 4053>
        builtin_tid2d(var_0, var_1);
        // if skip[worldid]:                                                                      <L 4055>
        var_2 = wp::address(var_skip, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // return                                                                             <L 4056>
            continue;
        }
        var_4 = wp::load(var_2);
        // dof = cdof_dof_in[worldid, ci]                                                         <L 4058>
        var_5 = wp::address(var_cdof_dof_in, var_0, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // if dof < 0:                                                                            <L 4059>
        var_9 = (var_6 < var_8);
        if (var_9) {
            // res[worldid, ci] = 0.0                                                             <L 4060>
            wp::array_store(var_res, var_0, var_1, var_10);
            // return                                                                             <L 4061>
            continue;
        }
        // acc = float(0.0)                                                                       <L 4063>
        var_12 = wp::float(var_11);
        // start = M_mulm_rowadr[dof]                                                             <L 4064>
        var_13 = wp::address(var_M_mulm_rowadr, var_6);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // end = M_mulm_rowadr[dof + 1]                                                           <L 4065>
        var_17 = wp::add(var_6, var_16);
        var_18 = wp::address(var_M_mulm_rowadr, var_17);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // for k in range(start, end):                                                            <L 4066>
        var_21 = wp::range(var_14, var_19);
        start_for_2:;
            if (iter_cmp(var_21) == 0) goto end_for_2;
            var_22 = wp::iter_next(var_21);
            // cj = dof_cdof_in[worldid, M_mulm_col[k]]                                           <L 4069>
            var_23 = wp::address(var_M_mulm_col, var_22);
            var_25 = wp::load(var_23);
            var_24 = wp::address(var_dof_cdof_in, var_0, var_25);
            var_27 = wp::load(var_24);
            var_26 = wp::copy(var_27);
            // if cj >= 0:                                                                        <L 4070>
            var_29 = (var_26 >= var_28);
            if (var_29) {
                // acc += M_in[worldid, M_mulm_madr[k]] * vec[worldid, cj]                        <L 4071>
                var_30 = wp::address(var_M_mulm_madr, var_22);
                var_32 = wp::load(var_30);
                var_31 = wp::address(var_M_in, var_0, var_32);
                var_33 = wp::address(var_vec, var_0, var_26);
                var_35 = wp::load(var_31);
                var_36 = wp::load(var_33);
                var_34 = wp::mul(var_35, var_36);
                var_37 = wp::add(var_12, var_34);
            }
            var_38 = wp::where(var_29, var_37, var_12);
            wp::assign(var_12, var_38);
            goto start_for_2;
        end_for_2:;
        // res[worldid, ci] = acc                                                                 <L 4072>
        wp::array_store(var_res, var_0, var_1, var_12);
    }
}



extern "C" __global__ void _qfrc_constraint_from_grad_4e7e88b1_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_qfrc_smooth_in,
    wp::array_t<wp::float32> var_efc_Ma_in,
    wp::array_t<wp::float32> var_ctx_grad_in,
    wp::array_t<wp::float32> var_ctx_grad_scale_in,
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
        wp::float32* var_2;
        wp::float32* var_3;
        wp::float32 var_4;
        wp::float32 var_5;
        wp::float32 var_6;
        wp::float32* var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32 var_11;
        wp::float32 var_12;
        //---------
        // forward
        // def _qfrc_constraint_from_grad(                                                        <L 1848>
        // worldid, dofid = wp.tid()                                                              <L 1858>
        builtin_tid2d(var_0, var_1);
        // grad = ctx_grad_scale_in[worldid] * ctx_grad_in[worldid, dofid]                        <L 1860>
        var_2 = wp::address(var_ctx_grad_scale_in, var_0);
        var_3 = wp::address(var_ctx_grad_in, var_0, var_1);
        var_5 = wp::load(var_2);
        var_6 = wp::load(var_3);
        var_4 = wp::mul(var_5, var_6);
        // qfrc_constraint_out[worldid, dofid] = efc_Ma_in[worldid, dofid] - qfrc_smooth_in[worldid, dofid] - grad       <L 1861>
        var_7 = wp::address(var_efc_Ma_in, var_0, var_1);
        var_8 = wp::address(var_qfrc_smooth_in, var_0, var_1);
        var_10 = wp::load(var_7);
        var_11 = wp::load(var_8);
        var_9 = wp::sub(var_10, var_11);
        var_12 = wp::sub(var_9, var_4);
        wp::array_store(var_qfrc_constraint_out, var_0, var_1, var_12);
    }
}



extern "C" __global__ void _diag_precond_build_8760076b_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_simple,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::int32> var_M_rownnz,
    wp::array_t<wp::int32> var_M_rowadr,
    wp::array_t<wp::float32> var_M_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_diag_out)
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
        wp::int32* var_6;
        wp::int32 var_7;
        const wp::int32 var_8 = 2;
        bool var_9;
        wp::int32 var_10;
        wp::int32* var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        const wp::int32 var_16 = 1;
        wp::int32 var_17;
        wp::float32* var_18;
        const wp::float32 var_19 = 1e-12;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        //---------
        // forward
        // def _diag_precond_build(                                                               <L 3172>
        // worldid, dofid = wp.tid()                                                              <L 3186>
        builtin_tid2d(var_0, var_1);
        // if ctx_done_in[worldid]:                                                               <L 3187>
        var_2 = wp::address(var_ctx_done_in, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // return                                                                             <L 3188>
            continue;
        }
        var_4 = wp::load(var_2);
        // if body_simple[dof_bodyid[dofid]] != 2:                                                <L 3189>
        var_5 = wp::address(var_dof_bodyid, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::address(var_body_simple, var_7);
        var_10 = wp::load(var_6);
        var_9 = (var_10 != var_8);
        if (var_9) {
            // return                                                                             <L 3190>
            continue;
        }
        // madr_ii = M_rowadr[dofid] + M_rownnz[dofid] - 1                                        <L 3191>
        var_11 = wp::address(var_M_rowadr, var_1);
        var_12 = wp::address(var_M_rownnz, var_1);
        var_14 = wp::load(var_11);
        var_15 = wp::load(var_12);
        var_13 = wp::add(var_14, var_15);
        var_17 = wp::sub(var_13, var_16);
        // diag_out[worldid, dofid] = M_in[worldid, madr_ii] + float(1.0e-12)                     <L 3192>
        var_18 = wp::address(var_M_in, var_0, var_17);
        var_20 = wp::float(var_19);
        var_22 = wp::load(var_18);
        var_21 = wp::add(var_22, var_20);
        wp::array_store(var_diag_out, var_0, var_1, var_21);
    }
}



extern "C" __global__ void _update_gradient_h_incremental_9c0c45bc_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_efc_J_in,
    wp::array_t<wp::float32> var_efc_D_in,
    wp::array_t<wp::int32> var_efc_state_in,
    wp::array_t<wp::int32> var_quad_changed_ids_in,
    wp::array_t<wp::int32> var_quad_changed_count_in,
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
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 0;
        bool var_6;
        const wp::int32 var_7 = 1;
        const wp::int32 var_8 = 8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::float32 var_11;
        wp::float32 var_12;
        wp::int32 var_13;
        const wp::int32 var_14 = 1;
        wp::int32 var_15;
        const wp::int32 var_16 = 2;
        wp::int32 var_17;
        const wp::int32 var_18 = 1;
        wp::int32 var_19;
        wp::int32 var_20;
        const wp::int32 var_21 = 2;
        wp::int32 var_22;
        wp::int32 var_23;
        const wp::float32 var_24 = 0.0;
        wp::float32 var_25;
        wp::range_t var_26;
        wp::int32 var_27;
        wp::int32* var_28;
        wp::int32 var_29;
        wp::int32 var_30;
        wp::float32* var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        const wp::float32 var_34 = 0.0;
        bool var_35;
        wp::float32* var_36;
        wp::float32 var_37;
        wp::float32 var_38;
        const wp::float32 var_39 = 0.0;
        bool var_40;
        wp::float32* var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::int32* var_44;
        const wp::int32 var_45 = 1;
        bool var_46;
        wp::int32 var_47;
        wp::float32 var_48;
        wp::float32 var_49;
        wp::float32 var_50;
        wp::float32 var_51;
        wp::float32 var_52;
        wp::float32 var_53;
        wp::float32 var_54;
        wp::float32 var_55;
        const wp::float32 var_56 = 0.0;
        bool var_57;
        wp::float32 var_58;
        //---------
        // forward
        // def _update_gradient_h_incremental(                                                    <L 1903>
        // worldid, elementid = wp.tid()                                                          <L 1919>
        builtin_tid2d(var_0, var_1);
        // n_changes = quad_changed_count_in[worldid]                                             <L 1921>
        var_2 = wp::address(var_quad_changed_count_in, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if n_changes == 0:                                                                     <L 1922>
        var_6 = (var_3 == var_5);
        if (var_6) {
            // return                                                                             <L 1923>
            continue;
        }
        // col = (int(wp.sqrt(float(1 + 8 * elementid))) - 1) // 2                                <L 1926>
        var_9 = wp::mul(var_8, var_1);
        var_10 = wp::add(var_7, var_9);
        var_11 = wp::float(var_10);
        var_12 = wp::sqrt(var_11);
        var_13 = wp::int(var_12);
        var_15 = wp::sub(var_13, var_14);
        var_17 = wp::floordiv(var_15, var_16);
        // row = elementid - (col * (col + 1)) // 2                                               <L 1927>
        var_19 = wp::add(var_17, var_18);
        var_20 = wp::mul(var_17, var_19);
        var_22 = wp::floordiv(var_20, var_21);
        var_23 = wp::sub(var_1, var_22);
        // delta = float(0.0)                                                                     <L 1929>
        var_25 = wp::float(var_24);
        // for change_idx in range(n_changes):                                                    <L 1930>
        var_26 = wp::range(var_3);
        start_for_1:;
            if (iter_cmp(var_26) == 0) goto end_for_1;
            var_27 = wp::iter_next(var_26);
            // efcid = quad_changed_ids_in[worldid, change_idx]                                   <L 1931>
            var_28 = wp::address(var_quad_changed_ids_in, var_0, var_27);
            var_30 = wp::load(var_28);
            var_29 = wp::copy(var_30);
            // Jrow = efc_J_in[worldid, efcid, row]                                               <L 1932>
            var_31 = wp::address(var_efc_J_in, var_0, var_29, var_23);
            var_33 = wp::load(var_31);
            var_32 = wp::copy(var_33);
            // if Jrow == 0.0:                                                                    <L 1933>
            var_35 = (var_32 == var_34);
            if (var_35) {
                // continue                                                                       <L 1934>
                goto start_for_1;
            }
            // Jcol = efc_J_in[worldid, efcid, col]                                               <L 1935>
            var_36 = wp::address(var_efc_J_in, var_0, var_29, var_17);
            var_38 = wp::load(var_36);
            var_37 = wp::copy(var_38);
            // if Jcol == 0.0:                                                                    <L 1936>
            var_40 = (var_37 == var_39);
            if (var_40) {
                // continue                                                                       <L 1937>
                goto start_for_1;
            }
            // D = efc_D_in[worldid, efcid]                                                       <L 1939>
            var_41 = wp::address(var_efc_D_in, var_0, var_29);
            var_43 = wp::load(var_41);
            var_42 = wp::copy(var_43);
            // if efc_state_in[worldid, efcid] == types.ConstraintState.QUADRATIC.value:          <L 1940>
            var_44 = wp::address(var_efc_state_in, var_0, var_29);
            var_47 = wp::load(var_44);
            var_46 = (var_47 == var_45);
            if (var_46) {
                // delta += D * Jrow * Jcol                                                       <L 1941>
                var_48 = wp::mul(var_42, var_32);
                var_49 = wp::mul(var_48, var_37);
                var_50 = wp::add(var_25, var_49);
            }
            var_51 = wp::where(var_46, var_50, var_25);
            if (!var_46) {
                // delta -= D * Jrow * Jcol                                                       <L 1943>
                var_52 = wp::mul(var_42, var_32);
                var_53 = wp::mul(var_52, var_37);
                var_54 = wp::sub(var_51, var_53);
            }
            var_55 = wp::where(var_46, var_51, var_54);
            wp::assign(var_25, var_55);
            goto start_for_1;
        end_for_1:;
        // if delta != 0.0:                                                                       <L 1945>
        var_57 = (var_25 != var_56);
        if (var_57) {
            // ctx_h_out[worldid, row, col] += delta                                              <L 1946>
            var_58 = wp::atomic_add(var_ctx_h_out, var_0, var_23, var_17, var_25);
        }
    }
}



extern "C" __global__ void _update_gradient_grad_tiled_c0425fb5_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_nv,
    wp::array_t<wp::float32> var_qfrc_smooth_in,
    wp::array_t<wp::float32> var_qfrc_constraint_in,
    wp::array_t<wp::float32> var_efc_Ma_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_ctx_grad_out,
    wp::array_t<wp::float32> var_ctx_grad_dot_out)
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
        const wp::float32 var_5 = 0.0;
        wp::float32 var_6;
        wp::int32 var_7;
        wp::range_t var_8;
        wp::int32 var_9;
        wp::float32* var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        wp::float32* var_15;
        wp::float32 var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        const bool var_20 = true;
        wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>> var_21 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>>{};
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_22 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
        const wp::int32 var_23 = 0;
        bool var_24;
        const wp::int32 var_25 = 0;
        wp::float32 var_26;
        //---------
        // forward
        // def _update_gradient_grad_tiled(                                                       <L 2155>
        // worldid, tid = wp.tid()                                                                <L 2168>
        builtin_tid2d(var_0, var_1);
        // if ctx_done_in[worldid]:                                                               <L 2170>
        var_2 = wp::address(var_ctx_done_in, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // return                                                                             <L 2171>
            continue;
        }
        var_4 = wp::load(var_2);
        // local_grad_dot = float(0.0)                                                            <L 2173>
        var_6 = wp::float(var_5);
        // BLOCK_DIM = wp.block_dim()                                                             <L 2174>
        var_7 = builtin_block_dim();
        // for dofid in range(tid, nv, BLOCK_DIM):                                                <L 2176>
        var_8 = wp::range(var_1, var_nv, var_7);
        start_for_1:;
            if (iter_cmp(var_8) == 0) goto end_for_1;
            var_9 = wp::iter_next(var_8);
            // grad = efc_Ma_in[worldid, dofid] - qfrc_smooth_in[worldid, dofid] - qfrc_constraint_in[worldid, dofid]       <L 2177>
            var_10 = wp::address(var_efc_Ma_in, var_0, var_9);
            var_11 = wp::address(var_qfrc_smooth_in, var_0, var_9);
            var_13 = wp::load(var_10);
            var_14 = wp::load(var_11);
            var_12 = wp::sub(var_13, var_14);
            var_15 = wp::address(var_qfrc_constraint_in, var_0, var_9);
            var_17 = wp::load(var_15);
            var_16 = wp::sub(var_12, var_17);
            // ctx_grad_out[worldid, dofid] = grad                                                <L 2178>
            wp::array_store(var_ctx_grad_out, var_0, var_9, var_16);
            // local_grad_dot += grad * grad                                                      <L 2179>
            var_18 = wp::mul(var_16, var_16);
            var_19 = wp::add(var_6, var_18);
            wp::assign(var_6, var_19);
            goto start_for_1;
        end_for_1:;
        // grad_dot_tile = wp.tile(local_grad_dot, preserve_type=True)                            <L 2181>
        var_21 = wp::tile<wp::float32>(var_6);
        // grad_dot_sum = wp.tile_reduce(wp.add, grad_dot_tile)                                   <L 2182>
        var_22 = wp::tile_reduce(wp::add, var_21);
        // if tid == 0:                                                                           <L 2184>
        var_24 = (var_1 == var_23);
        if (var_24) {
            // ctx_grad_dot_out[worldid] = grad_dot_sum[0]                                        <L 2185>
            var_26 = wp::tile_extract(var_22, var_25);
            wp::array_store(var_ctx_grad_dot_out, var_0, var_26);
        }
    }
}



extern "C" __global__ void _solve_search_update_cg_tiled_9a891d26_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_nv,
    wp::array_t<wp::float32> var_ctx_grad_in,
    wp::array_t<wp::float32> var_ctx_Mgrad_in,
    wp::array_t<wp::float32> var_ctx_search_in,
    wp::array_t<wp::float32> var_ctx_beta_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_ctx_search_out,
    wp::array_t<wp::float32> var_ctx_search_dot_out,
    wp::array_t<wp::float32> var_ctx_prev_grad_out,
    wp::array_t<wp::float32> var_ctx_prev_Mgrad_out)
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
        const wp::float32 var_5 = 0.0;
        wp::float32 var_6;
        wp::int32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::range_t var_11;
        wp::int32 var_12;
        wp::float32* var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        const wp::float32 var_16 = -1.0;
        wp::float32 var_17;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::float32* var_24;
        wp::float32 var_25;
        const bool var_26 = true;
        wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>> var_27 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>>{};
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_28 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
        const wp::int32 var_29 = 0;
        bool var_30;
        const wp::int32 var_31 = 0;
        wp::float32 var_32;
        //---------
        // forward
        // def _solve_search_update_cg_tiled(                                                     <L 3618>
        // worldid, tid = wp.tid()                                                                <L 3633>
        builtin_tid2d(var_0, var_1);
        // if ctx_done_in[worldid]:                                                               <L 3635>
        var_2 = wp::address(var_ctx_done_in, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // return                                                                             <L 3636>
            continue;
        }
        var_4 = wp::load(var_2);
        // local_search_dot = float(0.0)                                                          <L 3638>
        var_6 = wp::float(var_5);
        // BLOCK_DIM = wp.block_dim()                                                             <L 3639>
        var_7 = builtin_block_dim();
        // beta = ctx_beta_in[worldid]                                                            <L 3640>
        var_8 = wp::address(var_ctx_beta_in, var_0);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // for dofid in range(tid, nv, BLOCK_DIM):                                                <L 3642>
        var_11 = wp::range(var_1, var_nv, var_7);
        start_for_1:;
            if (iter_cmp(var_11) == 0) goto end_for_1;
            var_12 = wp::iter_next(var_11);
            // mgrad = ctx_Mgrad_in[worldid, dofid]                                               <L 3643>
            var_13 = wp::address(var_ctx_Mgrad_in, var_0, var_12);
            var_15 = wp::load(var_13);
            var_14 = wp::copy(var_15);
            // search = -1.0 * mgrad + beta * ctx_search_in[worldid, dofid]                       <L 3644>
            var_17 = wp::mul(var_16, var_14);
            var_18 = wp::address(var_ctx_search_in, var_0, var_12);
            var_20 = wp::load(var_18);
            var_19 = wp::mul(var_9, var_20);
            var_21 = wp::add(var_17, var_19);
            // ctx_search_out[worldid, dofid] = search                                            <L 3646>
            wp::array_store(var_ctx_search_out, var_0, var_12, var_21);
            // local_search_dot += search * search                                                <L 3647>
            var_22 = wp::mul(var_21, var_21);
            var_23 = wp::add(var_6, var_22);
            // ctx_prev_grad_out[worldid, dofid] = ctx_grad_in[worldid, dofid]                    <L 3649>
            var_24 = wp::address(var_ctx_grad_in, var_0, var_12);
            var_25 = wp::load(var_24);
            wp::array_store(var_ctx_prev_grad_out, var_0, var_12, var_25);
            // ctx_prev_Mgrad_out[worldid, dofid] = mgrad                                         <L 3650>
            wp::array_store(var_ctx_prev_Mgrad_out, var_0, var_12, var_14);
            wp::assign(var_6, var_23);
            goto start_for_1;
        end_for_1:;
        // search_dot_tile = wp.tile(local_search_dot, preserve_type=True)                        <L 3652>
        var_27 = wp::tile<wp::float32>(var_6);
        // search_dot_sum = wp.tile_reduce(wp.add, search_dot_tile)                               <L 3653>
        var_28 = wp::tile_reduce(wp::add, var_27);
        // if tid == 0:                                                                           <L 3655>
        var_30 = (var_1 == var_29);
        if (var_30) {
            // ctx_search_dot_out[worldid] = search_dot_sum[0]                                    <L 3656>
            var_32 = wp::tile_extract(var_28, var_31);
            wp::array_store(var_ctx_search_dot_out, var_0, var_32);
        }
    }
}



extern "C" __global__ void _gather_rhs_compact_48b89a4c_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_cdof_dof_in,
    wp::array_t<wp::float32> var_vec_in,
    wp::array_t<wp::float32> var_rhs_out)
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
        const wp::int32 var_5 = 0;
        bool var_6;
        wp::float32* var_7;
        const wp::int32 var_8 = 0;
        wp::float32 var_9;
        const wp::float32 var_10 = 0.0;
        const wp::int32 var_11 = 0;
        //---------
        // forward
        // def _gather_rhs_compact(                                                               <L 4131>
        // worldid, ci = wp.tid()                                                                 <L 4139>
        builtin_tid2d(var_0, var_1);
        // dof = cdof_dof_in[worldid, ci]                                                         <L 4140>
        var_2 = wp::address(var_cdof_dof_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if dof >= 0:                                                                           <L 4141>
        var_6 = (var_3 >= var_5);
        if (var_6) {
            // rhs_out[worldid, ci, 0] = vec_in[worldid, dof]                                     <L 4142>
            var_7 = wp::address(var_vec_in, var_0, var_3);
            var_9 = wp::load(var_7);
            wp::array_store(var_rhs_out, var_0, var_1, var_8, var_9);
        }
        if (!var_6) {
            // rhs_out[worldid, ci, 0] = 0.0                                                      <L 4144>
            wp::array_store(var_rhs_out, var_0, var_1, var_11, var_10);
        }
    }
}



extern "C" __global__ void _solve_done_c8485047_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_nv,
    wp::array_t<wp::float32> var_opt_tolerance,
    wp::int32 var_opt_iterations,
    wp::array_t<wp::float32> var_stat_meaninertia,
    wp::array_t<wp::float32> var_ctx_grad_dot_in,
    wp::array_t<wp::float32> var_ctx_newton_decrement_in,
    wp::array_t<wp::float32> var_ctx_improvement_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::int32> var_solver_niter_out,
    wp::array_t<wp::int32> var_nsolving_out,
    wp::array_t<bool> var_ctx_done_out)
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
        bool* var_1;
        bool var_2;
        bool var_3;
        const wp::int32 var_4 = 1;
        wp::int32 var_5;
        wp::shape_t* var_6;
        const wp::int32 var_7 = 0;
        wp::int32 var_8;
        wp::shape_t var_9;
        wp::int32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::shape_t* var_14;
        const wp::int32 var_15 = 0;
        wp::int32 var_16;
        wp::shape_t var_17;
        wp::int32 var_18;
        wp::float32* var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::float32* var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        wp::float32* var_25;
        wp::float32 var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        const wp::float32 var_29 = 0.5;
        wp::float32* var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        bool var_34;
        bool var_35;
        bool var_36;
        bool var_37;
        bool var_38;
        wp::int32* var_39;
        bool var_40;
        wp::int32 var_41;
        const bool var_42 = true;
        const wp::int32 var_43 = 0;
        const wp::int32 var_44 = -1;
        wp::int32 var_45;
        //---------
        // forward
        // def _solve_done(                                                                       <L 3703>
        // worldid = wp.tid()                                                                     <L 3720>
        var_0 = builtin_tid1d();
        // if ctx_done_in[worldid]:                                                               <L 3722>
        var_1 = wp::address(var_ctx_done_in, var_0);
        var_2 = wp::load(var_1);
        if (var_2) {
            // return                                                                             <L 3723>
            continue;
        }
        var_3 = wp::load(var_1);
        // solver_niter_out[worldid] += 1                                                         <L 3725>
        var_5 = wp::atomic_add(var_solver_niter_out, var_0, var_4);
        // tolerance = opt_tolerance[worldid % opt_tolerance.shape[0]]                            <L 3726>
        var_6 = &(var_opt_tolerance.shape);
        var_9 = wp::load(var_6);
        var_8 = wp::extract(var_9, var_7);
        var_10 = wp::mod(var_0, var_8);
        var_11 = wp::address(var_opt_tolerance, var_10);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // meaninertia = stat_meaninertia[worldid % stat_meaninertia.shape[0]]                    <L 3727>
        var_14 = &(var_stat_meaninertia.shape);
        var_17 = wp::load(var_14);
        var_16 = wp::extract(var_17, var_15);
        var_18 = wp::mod(var_0, var_16);
        var_19 = wp::address(var_stat_meaninertia, var_18);
        var_21 = wp::load(var_19);
        var_20 = wp::copy(var_21);
        // improvement = _rescale(nv, meaninertia, ctx_improvement_in[worldid])                   <L 3729>
        var_22 = wp::address(var_ctx_improvement_in, var_0);
        var_24 = wp::load(var_22);
        var_23 = _rescale_0(var_nv, var_20, var_24);
        // gradient = _rescale(nv, meaninertia, wp.sqrt(ctx_grad_dot_in[worldid]))                <L 3730>
        var_25 = wp::address(var_ctx_grad_dot_in, var_0);
        var_27 = wp::load(var_25);
        var_26 = wp::sqrt(var_27);
        var_28 = _rescale_0(var_nv, var_20, var_26);
        // model_improvement = _rescale(nv, meaninertia, 0.5 * ctx_newton_decrement_in[worldid])       <L 3731>
        var_30 = wp::address(var_ctx_newton_decrement_in, var_0);
        var_32 = wp::load(var_30);
        var_31 = wp::mul(var_29, var_32);
        var_33 = _rescale_0(var_nv, var_20, var_31);
        // done = (improvement < tolerance) or (gradient < tolerance) or (model_improvement < tolerance)       <L 3732>
        var_35 = (var_23 < var_12);
        var_34 = var_35;
        if (!var_34) {
            var_36 = (var_28 < var_12);
            var_34 = var_34 || var_36;
        }
        if (!var_34) {
            var_37 = (var_33 < var_12);
            var_34 = var_34 || var_37;
        }
        // if done or solver_niter_out[worldid] == opt_iterations:                                <L 3733>
        var_38 = var_34;
        if (!var_38) {
            var_39 = wp::address(var_solver_niter_out, var_0);
            var_41 = wp::load(var_39);
            var_40 = (var_41 == var_opt_iterations);
            var_38 = var_38 || var_40;
        }
        if (var_38) {
            // ctx_done_out[worldid] = True                                                       <L 3736>
            wp::array_store(var_ctx_done_out, var_0, var_42);
            // wp.atomic_add(nsolving_out, 0, -1)                                                 <L 3737>
            var_45 = wp::atomic_add(var_nsolving_out, var_43, var_44);
        }
    }
}



extern "C" __global__ void _update_gradient_JTCJ_sparse_05d3c7f1_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_opt_impratio_invsqrt,
    wp::array_t<wp::float32> var_contact_dist_in,
    wp::array_t<wp::float32> var_contact_includemargin_in,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_in,
    wp::array_t<wp::int32> var_contact_dim_in,
    wp::array_t<wp::int32> var_contact_efc_address_in,
    wp::array_t<wp::int32> var_contact_worldid_in,
    wp::array_t<wp::int32> var_efc_J_rownnz_in,
    wp::array_t<wp::int32> var_efc_J_rowadr_in,
    wp::array_t<wp::int32> var_efc_J_colind_in,
    wp::array_t<wp::float32> var_efc_J_in,
    wp::array_t<wp::float32> var_efc_D_in,
    wp::array_t<wp::int32> var_efc_state_in,
    wp::int32 var_naconmax_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::array_t<wp::float32> var_ctx_Jaref_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::int32 var_nblocks_perblock,
    wp::int32 var_dim_block,
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
        wp::range_t var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        const wp::int32 var_6 = 0;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        bool var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        bool* var_14;
        bool var_15;
        bool var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        const wp::int32 var_20 = 1;
        bool var_21;
        wp::float32* var_22;
        wp::float32* var_23;
        wp::float32 var_24;
        wp::float32 var_25;
        wp::float32 var_26;
        const wp::float32 var_27 = 0.0;
        bool var_28;
        const wp::int32 var_29 = 0;
        wp::int32* var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        const wp::int32 var_33 = 0;
        bool var_34;
        wp::int32* var_35;
        const wp::int32 var_36 = 4;
        bool var_37;
        wp::int32 var_38;
        wp::int32* var_39;
        wp::int32 var_40;
        wp::int32 var_41;
        const wp::int32 var_42 = 1;
        wp::int32 var_43;
        wp::int32 var_44;
        const wp::int32 var_45 = 2;
        wp::int32 var_46;
        bool var_47;
        wp::int32* var_48;
        wp::int32 var_49;
        wp::int32 var_50;
        const wp::int32 var_51 = 0;
        wp::int32 var_52;
        wp::int32 var_53;
        wp::int32 var_54;
        bool var_55;
        wp::int32 var_56;
        wp::int32 var_57;
        const wp::int32 var_58 = 1;
        wp::int32 var_59;
        wp::int32 var_60;
        const wp::int32 var_61 = 0;
        wp::int32 var_62;
        wp::int32* var_63;
        wp::int32 var_64;
        wp::int32 var_65;
        const wp::int32 var_66 = 0;
        wp::int32 var_67;
        wp::int32* var_68;
        wp::int32 var_69;
        wp::int32 var_70;
        wp::int32 var_71;
        wp::int32 var_72;
        wp::vec_t<5, wp::float32>* var_73;
        wp::vec_t<5, wp::float32> var_74;
        wp::vec_t<5, wp::float32> var_75;
        const wp::int32 var_76 = 0;
        wp::float32 var_77;
        wp::shape_t* var_78;
        const wp::int32 var_79 = 0;
        wp::int32 var_80;
        wp::shape_t var_81;
        wp::int32 var_82;
        wp::float32* var_83;
        wp::float32 var_84;
        wp::float32 var_85;
        wp::float32 var_86;
        wp::float32* var_87;
        const wp::float32 var_88 = 1.0;
        wp::float32 var_89;
        wp::float32 var_90;
        wp::float32 var_91;
        wp::float32 var_92;
        const wp::float32 var_93 = 0.0;
        bool var_94;
        wp::float32* var_95;
        wp::float32 var_96;
        wp::float32 var_97;
        const wp::float32 var_98 = 0.0;
        const wp::float32 var_99 = 0.0;
        const wp::float32 var_100 = 0.0;
        const wp::float32 var_101 = 0.0;
        const wp::float32 var_102 = 0.0;
        wp::vec_t<6, wp::float32> var_103;
        const wp::float32 var_104 = 0.0;
        wp::float32 var_105;
        const wp::int32 var_106 = 1;
        wp::range_t var_107;
        wp::int32 var_108;
        wp::int32* var_109;
        wp::int32 var_110;
        wp::int32 var_111;
        const wp::int32 var_112 = 0;
        bool var_113;
        wp::float32* var_114;
        const wp::int32 var_115 = 1;
        wp::int32 var_116;
        wp::float32 var_117;
        wp::float32 var_118;
        wp::float32 var_119;
        const wp::float32 var_120 = 0.0;
        wp::float32 var_121;
        wp::float32 var_122;
        wp::float32 var_123;
        const wp::float32 var_124 = 0.0;
        bool var_125;
        const wp::float32 var_126 = 0.0;
        wp::float32 var_127;
        wp::float32 var_128;
        const wp::float32 var_129 = 1e-15;
        const wp::float32 var_130 = 1e-15;
        wp::float32 var_131;
        wp::float32 var_132;
        wp::float32 var_133;
        const wp::float32 var_134 = 1e-15;
        const wp::float32 var_135 = 1e-15;
        wp::float32 var_136;
        wp::float32 var_137;
        wp::float32 var_138;
        wp::float32 var_139;
        wp::float32 var_140;
        wp::float32 var_141;
        wp::float32 var_142;
        const wp::float32 var_143 = 0.0;
        wp::float32 var_144;
        wp::range_t var_145;
        wp::int32 var_146;
        const wp::int32 var_147 = 0;
        bool var_148;
        wp::int32 var_149;
        wp::float32 var_150;
        wp::int32* var_151;
        wp::int32 var_152;
        wp::int32 var_153;
        const wp::int32 var_154 = 0;
        bool var_155;
        wp::int32* var_156;
        wp::int32 var_157;
        wp::int32 var_158;
        const wp::int32 var_159 = 1;
        wp::int32 var_160;
        wp::float32 var_161;
        wp::float32 var_162;
        wp::int32 var_163;
        wp::float32 var_164;
        const wp::int32 var_165 = 0;
        wp::int32 var_166;
        wp::float32* var_167;
        wp::float32 var_168;
        wp::float32 var_169;
        const wp::int32 var_170 = 0;
        wp::int32 var_171;
        wp::float32* var_172;
        wp::float32 var_173;
        wp::float32 var_174;
        wp::float32 var_175;
        const wp::int32 var_176 = 1;
        wp::int32 var_177;
        const wp::int32 var_178 = 0;
        wp::range_t var_179;
        wp::int32 var_180;
        const wp::int32 var_181 = 0;
        bool var_182;
        wp::int32 var_183;
        wp::float32 var_184;
        wp::int32* var_185;
        wp::int32 var_186;
        wp::int32 var_187;
        const wp::int32 var_188 = 0;
        bool var_189;
        wp::int32* var_190;
        wp::int32 var_191;
        wp::int32 var_192;
        const wp::int32 var_193 = 1;
        wp::int32 var_194;
        wp::float32 var_195;
        wp::float32 var_196;
        wp::int32 var_197;
        wp::float32 var_198;
        const wp::int32 var_199 = 0;
        wp::int32 var_200;
        wp::float32* var_201;
        wp::float32 var_202;
        wp::float32 var_203;
        const wp::int32 var_204 = 0;
        wp::int32 var_205;
        wp::float32* var_206;
        wp::float32 var_207;
        wp::float32 var_208;
        wp::float32 var_209;
        bool var_210;
        const wp::int32 var_211 = 0;
        bool var_212;
        const wp::int32 var_213 = 0;
        bool var_214;
        const wp::float32 var_215 = 1.0;
        const wp::int32 var_216 = 0;
        bool var_217;
        wp::float32 var_218;
        wp::float32 var_219;
        wp::float32 var_220;
        const wp::int32 var_221 = 0;
        bool var_222;
        wp::float32 var_223;
        wp::float32 var_224;
        wp::float32 var_225;
        wp::float32 var_226;
        wp::float32 var_227;
        bool var_228;
        wp::float32 var_229;
        wp::float32 var_230;
        wp::float32 var_231;
        wp::float32 var_232;
        wp::float32 var_233;
        wp::float32 var_234;
        const wp::float32 var_235 = 0.0;
        bool var_236;
        wp::float32 var_237;
        wp::float32 var_238;
        wp::float32 var_239;
        bool var_240;
        wp::float32 var_241;
        wp::float32 var_242;
        wp::float32 var_243;
        wp::float32 var_244;
        wp::float32 var_245;
        wp::slice_t var_246;
        const wp::int32 var_247 = 0;
        wp::slice_t var_248;
        const wp::int32 var_249 = 0;
        wp::array_t<wp::float32> var_250;
        wp::float32 var_251;
        //---------
        // forward
        // def _update_gradient_JTCJ_sparse(                                                      <L 2397>
        // conid_start, pairid = wp.tid()                                                         <L 2423>
        builtin_tid2d(var_0, var_1);
        // for i in range(nblocks_perblock):                                                      <L 2425>
        var_2 = wp::range(var_nblocks_perblock);
        start_for_0:;
            if (iter_cmp(var_2) == 0) goto end_for_0;
            var_3 = wp::iter_next(var_2);
            // conid = conid_start + i * dim_block                                                <L 2426>
            var_4 = wp::mul(var_3, var_dim_block);
            var_5 = wp::add(var_0, var_4);
            // if conid >= min(nacon_in[0], naconmax_in):                                         <L 2428>
            var_7 = wp::address(var_nacon_in, var_6);
            var_9 = wp::load(var_7);
            var_8 = wp::min(var_9, var_naconmax_in);
            var_10 = (var_5 >= var_8);
            if (var_10) {
                // return                                                                         <L 2429>
                continue;
            }
            // worldid = contact_worldid_in[conid]                                                <L 2431>
            var_11 = wp::address(var_contact_worldid_in, var_5);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
            // if ctx_done_in[worldid]:                                                           <L 2432>
            var_14 = wp::address(var_ctx_done_in, var_12);
            var_15 = wp::load(var_14);
            if (var_15) {
                // continue                                                                       <L 2433>
                goto start_for_0;
            }
            var_16 = wp::load(var_14);
            // condim = contact_dim_in[conid]                                                     <L 2435>
            var_17 = wp::address(var_contact_dim_in, var_5);
            var_19 = wp::load(var_17);
            var_18 = wp::copy(var_19);
            // if condim == 1:                                                                    <L 2437>
            var_21 = (var_18 == var_20);
            if (var_21) {
                // continue                                                                       <L 2438>
                goto start_for_0;
            }
            // if contact_dist_in[conid] - contact_includemargin_in[conid] >= 0.0:                <L 2441>
            var_22 = wp::address(var_contact_dist_in, var_5);
            var_23 = wp::address(var_contact_includemargin_in, var_5);
            var_25 = wp::load(var_22);
            var_26 = wp::load(var_23);
            var_24 = wp::sub(var_25, var_26);
            var_28 = (var_24 >= var_27);
            if (var_28) {
                // continue                                                                       <L 2442>
                goto start_for_0;
            }
            // efcid0 = contact_efc_address_in[conid, 0]                                          <L 2444>
            var_30 = wp::address(var_contact_efc_address_in, var_5, var_29);
            var_32 = wp::load(var_30);
            var_31 = wp::copy(var_32);
            // if efcid0 < 0:                                                                     <L 2445>
            var_34 = (var_31 < var_33);
            if (var_34) {
                // continue                                                                       <L 2446>
                goto start_for_0;
            }
            // if efc_state_in[worldid, efcid0] != types.ConstraintState.CONE:                    <L 2447>
            var_35 = wp::address(var_efc_state_in, var_12, var_31);
            var_38 = wp::load(var_35);
            var_37 = (var_38 != var_36);
            if (var_37) {
                // continue                                                                       <L 2448>
                goto start_for_0;
            }
            // rownnz = efc_J_rownnz_in[worldid, efcid0]                                          <L 2453>
            var_39 = wp::address(var_efc_J_rownnz_in, var_12, var_31);
            var_41 = wp::load(var_39);
            var_40 = wp::copy(var_41);
            // npairs = rownnz * (rownnz + 1) // 2                                                <L 2454>
            var_43 = wp::add(var_40, var_42);
            var_44 = wp::mul(var_40, var_43);
            var_46 = wp::floordiv(var_44, var_45);
            // if pairid >= npairs:                                                               <L 2455>
            var_47 = (var_1 >= var_46);
            if (var_47) {
                // continue                                                                       <L 2456>
                goto start_for_0;
            }
            // rowadr0 = efc_J_rowadr_in[worldid, efcid0]                                         <L 2457>
            var_48 = wp::address(var_efc_J_rowadr_in, var_12, var_31);
            var_50 = wp::load(var_48);
            var_49 = wp::copy(var_50);
            // pos1 = int(0)                                                                      <L 2458>
            var_52 = wp::int(var_51);
            // rem = pairid                                                                       <L 2459>
            var_53 = wp::copy(var_1);
            // while rem >= rownnz - pos1:                                                        <L 2460>
        start_while_3:;
            var_54 = wp::sub(var_40, var_52);
            var_55 = (var_53 >= var_54);
        if ((var_55) == false) goto end_while_3;
                // rem -= rownnz - pos1                                                           <L 2461>
                var_56 = wp::sub(var_40, var_52);
                var_57 = wp::sub(var_53, var_56);
                // pos1 += 1                                                                      <L 2462>
                var_59 = wp::add(var_52, var_58);
                wp::assign(var_52, var_59);
                wp::assign(var_53, var_57);
        goto start_while_3;
        end_while_3:;
            // pos2 = pos1 + rem                                                                  <L 2463>
            var_60 = wp::add(var_52, var_53);
            // dofa = efc_J_colind_in[worldid, 0, rowadr0 + pos1]                                 <L 2464>
            var_62 = wp::add(var_49, var_52);
            var_63 = wp::address(var_efc_J_colind_in, var_12, var_61, var_62);
            var_65 = wp::load(var_63);
            var_64 = wp::copy(var_65);
            // dofb = efc_J_colind_in[worldid, 0, rowadr0 + pos2]                                 <L 2465>
            var_67 = wp::add(var_49, var_60);
            var_68 = wp::address(var_efc_J_colind_in, var_12, var_66, var_67);
            var_70 = wp::load(var_68);
            var_69 = wp::copy(var_70);
            // dof1id = wp.min(dofa, dofb)                                                        <L 2466>
            var_71 = wp::min(var_64, var_69);
            // dof2id = wp.max(dofa, dofb)                                                        <L 2467>
            var_72 = wp::max(var_64, var_69);
            // fri = contact_friction_in[conid]                                                   <L 2469>
            var_73 = wp::address(var_contact_friction_in, var_5);
            var_75 = wp::load(var_73);
            var_74 = wp::copy(var_75);
            // mu = fri[0] * opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]        <L 2470>
            var_77 = wp::extract(var_74, var_76);
            var_78 = &(var_opt_impratio_invsqrt.shape);
            var_81 = wp::load(var_78);
            var_80 = wp::extract(var_81, var_79);
            var_82 = wp::mod(var_12, var_80);
            var_83 = wp::address(var_opt_impratio_invsqrt, var_82);
            var_85 = wp::load(var_83);
            var_84 = wp::mul(var_77, var_85);
            // mu2 = mu * mu                                                                      <L 2472>
            var_86 = wp::mul(var_84, var_84);
            // dm = math.safe_div(efc_D_in[worldid, efcid0], mu2 * (1.0 + mu2))                   <L 2473>
            var_87 = wp::address(var_efc_D_in, var_12, var_31);
            var_89 = wp::add(var_88, var_86);
            var_90 = wp::mul(var_86, var_89);
            var_92 = wp::load(var_87);
            var_91 = safe_div_0(var_92, var_90);
            // if dm == 0.0:                                                                      <L 2475>
            var_94 = (var_91 == var_93);
            if (var_94) {
                // continue                                                                       <L 2476>
                goto start_for_0;
            }
            // n = ctx_Jaref_in[worldid, efcid0] * mu                                             <L 2478>
            var_95 = wp::address(var_ctx_Jaref_in, var_12, var_31);
            var_97 = wp::load(var_95);
            var_96 = wp::mul(var_97, var_84);
            // u = types.vec6(n, 0.0, 0.0, 0.0, 0.0, 0.0)                                         <L 2479>
            var_103 = wp::vec_t<6, wp::float32>({var_96, var_98, var_99, var_100, var_101, var_102});
            // tt = float(0.0)                                                                    <L 2481>
            var_105 = wp::float(var_104);
            // for j in range(1, condim):                                                         <L 2482>
            var_107 = wp::range(var_106, var_18);
            start_for_5:;
                if (iter_cmp(var_107) == 0) goto end_for_5;
                var_108 = wp::iter_next(var_107);
                // efcidj = contact_efc_address_in[conid, j]                                      <L 2483>
                var_109 = wp::address(var_contact_efc_address_in, var_5, var_108);
                var_111 = wp::load(var_109);
                var_110 = wp::copy(var_111);
                // if efcidj >= 0:                                                                <L 2484>
                var_113 = (var_110 >= var_112);
                if (var_113) {
                    // uj = ctx_Jaref_in[worldid, efcidj] * fri[j - 1]                            <L 2485>
                    var_114 = wp::address(var_ctx_Jaref_in, var_12, var_110);
                    var_116 = wp::sub(var_108, var_115);
                    var_117 = wp::extract(var_74, var_116);
                    var_119 = wp::load(var_114);
                    var_118 = wp::mul(var_119, var_117);
                }
                if (!var_113) {
                    // uj = 0.0                                                                   <L 2487>
                }
                var_121 = wp::where(var_113, var_118, var_120);
                // tt += uj * uj                                                                  <L 2488>
                var_122 = wp::mul(var_121, var_121);
                var_123 = wp::add(var_105, var_122);
                // u[j] = uj                                                                      <L 2489>
                wp::assign_inplace(var_103, var_108, var_121);
                wp::assign(var_105, var_123);
                goto start_for_5;
            end_for_5:;
            // if tt <= 0.0:                                                                      <L 2491>
            var_125 = (var_105 <= var_124);
            if (var_125) {
                // t = 0.0                                                                        <L 2492>
            }
            if (!var_125) {
                // t = wp.sqrt(tt)                                                                <L 2494>
                var_127 = wp::sqrt(var_105);
            }
            var_128 = wp::where(var_125, var_126, var_127);
            // t = wp.max(t, types.MJ_MINVAL)                                                     <L 2495>
            var_131 = wp::max(var_128, var_130);
            // ttt = wp.max(t * t * t, types.MJ_MINVAL)                                           <L 2496>
            var_132 = wp::mul(var_131, var_131);
            var_133 = wp::mul(var_132, var_131);
            var_136 = wp::max(var_133, var_135);
            // mu_over_t = math.safe_div(mu, t)                                                   <L 2499>
            var_137 = safe_div_0(var_84, var_131);
            // mu_n_over_ttt = mu * math.safe_div(n, ttt)                                         <L 2500>
            var_138 = safe_div_0(var_96, var_136);
            var_139 = wp::mul(var_84, var_138);
            // mu2_minus_mu_n_over_t = mu2 - mu * math.safe_div(n, t)                             <L 2501>
            var_140 = safe_div_0(var_96, var_131);
            var_141 = wp::mul(var_84, var_140);
            var_142 = wp::sub(var_86, var_141);
            // h = float(0.0)                                                                     <L 2503>
            var_144 = wp::float(var_143);
            // for dim1id in range(condim):                                                       <L 2505>
            var_145 = wp::range(var_18);
            start_for_7:;
                if (iter_cmp(var_145) == 0) goto end_for_7;
                var_146 = wp::iter_next(var_145);
                // if dim1id == 0:                                                                <L 2506>
                var_148 = (var_146 == var_147);
                if (var_148) {
                    // rowadr1 = rowadr0                                                          <L 2507>
                    var_149 = wp::copy(var_49);
                    // dm_fri1 = dm * mu                                                          <L 2508>
                    var_150 = wp::mul(var_91, var_84);
                }
                if (!var_148) {
                    // efcid1 = contact_efc_address_in[conid, dim1id]                             <L 2510>
                    var_151 = wp::address(var_contact_efc_address_in, var_5, var_146);
                    var_153 = wp::load(var_151);
                    var_152 = wp::copy(var_153);
                    // if efcid1 < 0:                                                             <L 2511>
                    var_155 = (var_152 < var_154);
                    if (var_155) {
                        // continue                                                               <L 2512>
                        goto start_for_7;
                    }
                    // rowadr1 = efc_J_rowadr_in[worldid, efcid1]                                 <L 2513>
                    var_156 = wp::address(var_efc_J_rowadr_in, var_12, var_152);
                    var_158 = wp::load(var_156);
                    var_157 = wp::copy(var_158);
                    // dm_fri1 = dm * fri[dim1id - 1]                                             <L 2514>
                    var_160 = wp::sub(var_146, var_159);
                    var_161 = wp::extract(var_74, var_160);
                    var_162 = wp::mul(var_91, var_161);
                }
                var_163 = wp::where(var_148, var_149, var_157);
                var_164 = wp::where(var_148, var_150, var_162);
                // efc_J11 = efc_J_in[worldid, 0, rowadr1 + pos1]                                 <L 2517>
                var_166 = wp::add(var_163, var_52);
                var_167 = wp::address(var_efc_J_in, var_12, var_165, var_166);
                var_169 = wp::load(var_167);
                var_168 = wp::copy(var_169);
                // efc_J12 = efc_J_in[worldid, 0, rowadr1 + pos2]                                 <L 2518>
                var_171 = wp::add(var_163, var_60);
                var_172 = wp::address(var_efc_J_in, var_12, var_170, var_171);
                var_174 = wp::load(var_172);
                var_173 = wp::copy(var_174);
                // ui = u[dim1id]                                                                 <L 2520>
                var_175 = wp::extract(var_103, var_146);
                // for dim2id in range(0, dim1id + 1):                                            <L 2522>
                var_177 = wp::add(var_146, var_176);
                var_179 = wp::range(var_178, var_177);
                start_for_9:;
                    if (iter_cmp(var_179) == 0) goto end_for_9;
                    var_180 = wp::iter_next(var_179);
                    // if dim2id == 0:                                                            <L 2523>
                    var_182 = (var_180 == var_181);
                    if (var_182) {
                        // rowadr2 = rowadr0                                                      <L 2524>
                        var_183 = wp::copy(var_49);
                        // dm_fri12 = dm_fri1 * mu                                                <L 2525>
                        var_184 = wp::mul(var_164, var_84);
                    }
                    if (!var_182) {
                        // efcid2 = contact_efc_address_in[conid, dim2id]                         <L 2527>
                        var_185 = wp::address(var_contact_efc_address_in, var_5, var_180);
                        var_187 = wp::load(var_185);
                        var_186 = wp::copy(var_187);
                        // if efcid2 < 0:                                                         <L 2528>
                        var_189 = (var_186 < var_188);
                        if (var_189) {
                            // continue                                                           <L 2529>
                            goto start_for_9;
                        }
                        // rowadr2 = efc_J_rowadr_in[worldid, efcid2]                             <L 2530>
                        var_190 = wp::address(var_efc_J_rowadr_in, var_12, var_186);
                        var_192 = wp::load(var_190);
                        var_191 = wp::copy(var_192);
                        // dm_fri12 = dm_fri1 * fri[dim2id - 1]                                   <L 2531>
                        var_194 = wp::sub(var_180, var_193);
                        var_195 = wp::extract(var_74, var_194);
                        var_196 = wp::mul(var_164, var_195);
                    }
                    var_197 = wp::where(var_182, var_183, var_191);
                    var_198 = wp::where(var_182, var_184, var_196);
                    // efc_J21 = efc_J_in[worldid, 0, rowadr2 + pos1]                             <L 2534>
                    var_200 = wp::add(var_197, var_52);
                    var_201 = wp::address(var_efc_J_in, var_12, var_199, var_200);
                    var_203 = wp::load(var_201);
                    var_202 = wp::copy(var_203);
                    // efc_J22 = efc_J_in[worldid, 0, rowadr2 + pos2]                             <L 2535>
                    var_205 = wp::add(var_197, var_60);
                    var_206 = wp::address(var_efc_J_in, var_12, var_204, var_205);
                    var_208 = wp::load(var_206);
                    var_207 = wp::copy(var_208);
                    // uj = u[dim2id]                                                             <L 2537>
                    var_209 = wp::extract(var_103, var_180);
                    // if dim1id == 0 and dim2id == 0:                                            <L 2540>
                    var_212 = (var_146 == var_211);
                    var_210 = var_212;
                    if (var_210) {
                        var_214 = (var_180 == var_213);
                        var_210 = var_210 && var_214;
                    }
                    if (var_210) {
                        // hcone = 1.0                                                            <L 2541>
                    }
                    if (!var_210) {
                        // elif dim1id == 0:                                                      <L 2542>
                        var_217 = (var_146 == var_216);
                        if (var_217) {
                            // hcone = -mu_over_t * uj                                            <L 2543>
                            var_218 = wp::neg(var_137);
                            var_219 = wp::mul(var_218, var_209);
                        }
                        var_220 = wp::where(var_217, var_219, var_215);
                        if (!var_217) {
                            // elif dim2id == 0:                                                  <L 2544>
                            var_222 = (var_180 == var_221);
                            if (var_222) {
                                // hcone = -mu_over_t * ui                                        <L 2545>
                                var_223 = wp::neg(var_137);
                                var_224 = wp::mul(var_223, var_175);
                            }
                            var_225 = wp::where(var_222, var_224, var_220);
                            if (!var_222) {
                                // hcone = mu_n_over_ttt * ui * uj                                <L 2547>
                                var_226 = wp::mul(var_139, var_175);
                                var_227 = wp::mul(var_226, var_209);
                                // if dim1id == dim2id:                                           <L 2550>
                                var_228 = (var_146 == var_180);
                                if (var_228) {
                                    // hcone += mu2_minus_mu_n_over_t                             <L 2551>
                                    var_229 = wp::add(var_227, var_142);
                                }
                                var_230 = wp::where(var_228, var_229, var_227);
                            }
                            var_231 = wp::where(var_222, var_225, var_230);
                        }
                        var_232 = wp::where(var_217, var_220, var_231);
                    }
                    var_233 = wp::where(var_210, var_215, var_232);
                    // hcone *= dm_fri12                                                          <L 2553>
                    var_234 = wp::mul(var_233, var_198);
                    // if hcone != 0.0:                                                           <L 2555>
                    var_236 = (var_234 != var_235);
                    if (var_236) {
                        // h += hcone * efc_J11 * efc_J22                                         <L 2556>
                        var_237 = wp::mul(var_234, var_168);
                        var_238 = wp::mul(var_237, var_207);
                        var_239 = wp::add(var_144, var_238);
                        // if dim1id != dim2id:                                                   <L 2558>
                        var_240 = (var_146 != var_180);
                        if (var_240) {
                            // h += hcone * efc_J12 * efc_J21                                     <L 2559>
                            var_241 = wp::mul(var_234, var_173);
                            var_242 = wp::mul(var_241, var_202);
                            var_243 = wp::add(var_239, var_242);
                        }
                        var_244 = wp::where(var_240, var_243, var_239);
                    }
                    var_245 = wp::where(var_236, var_244, var_144);
                    wp::assign(var_121, var_209);
                    wp::assign(var_144, var_245);
                    goto start_for_9;
                end_for_9:;
                goto start_for_7;
            end_for_7:;
            // wp.atomic_add(ctx_h_out[worldid, dof1id], dof2id, h)                               <L 2562>
            var_246 = wp::slice_t(var_12, var_12, var_247);
            var_248 = wp::slice_t(var_71, var_71, var_249);
            var_250 = wp::view(var_ctx_h_out, var_246, var_248);
            var_251 = wp::atomic_add(var_250, var_72, var_144);
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void _zero_change_counters_939878a2_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::int32> var_quad_changed_count_out,
    wp::array_t<wp::int32> var_state_changed_count_out)
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
        //---------
        // forward
        // def _zero_change_counters(                                                             <L 3766>
        // worldid = wp.tid()                                                                     <L 3771>
        var_0 = builtin_tid1d();
        // quad_changed_count_out[worldid] = 0                                                    <L 3772>
        wp::array_store(var_quad_changed_count_out, var_0, var_1);
        // state_changed_count_out[worldid] = 0                                                   <L 3773>
        wp::array_store(var_state_changed_count_out, var_0, var_2);
    }
}



extern "C" __global__ void _linesearch_zero_jv_ef61d94f_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_nefc_in,
    wp::array_t<bool> var_skip_in,
    wp::array_t<wp::float32> var_ctx_jv_out)
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
        bool var_3;
        wp::int32 var_4;
        bool* var_5;
        bool var_6;
        bool var_7;
        const wp::float32 var_8 = 0.0;
        //---------
        // forward
        // def _linesearch_zero_jv(                                                               <L 1369>
        // worldid, efcid = wp.tid()                                                              <L 1377>
        builtin_tid2d(var_0, var_1);
        // if efcid >= nefc_in[worldid]:                                                          <L 1379>
        var_2 = wp::address(var_nefc_in, var_0);
        var_4 = wp::load(var_2);
        var_3 = (var_1 >= var_4);
        if (var_3) {
            // return                                                                             <L 1380>
            continue;
        }
        // if skip_in[worldid]:                                                                   <L 1382>
        var_5 = wp::address(var_skip_in, var_0);
        var_6 = wp::load(var_5);
        if (var_6) {
            // return                                                                             <L 1383>
            continue;
        }
        var_7 = wp::load(var_5);
        // ctx_jv_out[worldid, efcid] = 0.0                                                       <L 1385>
        wp::array_store(var_ctx_jv_out, var_0, var_1, var_8);
    }
}



extern "C" __global__ void _gather_M_sparse_8250c474_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_M_rownnz,
    wp::array_t<wp::int32> var_M_rowadr,
    wp::array_t<wp::int32> var_M_colind,
    wp::array_t<wp::float32> var_M_in,
    wp::array_t<wp::int32> var_dof_cdof_in,
    wp::array_t<wp::float32> var_M_c_out)
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
        const wp::int32 var_5 = 0;
        bool var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::range_t var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::int32* var_15;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        const wp::int32 var_20 = 0;
        bool var_21;
        wp::float32* var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        //---------
        // forward
        // def _gather_M_sparse(                                                                  <L 4105>
        // worldid, i = wp.tid()                                                                  <L 4116>
        builtin_tid2d(var_0, var_1);
        // ci = dof_cdof_in[worldid, i]                                                           <L 4117>
        var_2 = wp::address(var_dof_cdof_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if ci < 0:                                                                             <L 4118>
        var_6 = (var_3 < var_5);
        if (var_6) {
            // return                                                                             <L 4119>
            continue;
        }
        // rowadr = M_rowadr[i]                                                                   <L 4120>
        var_7 = wp::address(var_M_rowadr, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // for k in range(M_rownnz[i]):                                                           <L 4121>
        var_10 = wp::address(var_M_rownnz, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::range(var_12);
        start_for_1:;
            if (iter_cmp(var_11) == 0) goto end_for_1;
            var_13 = wp::iter_next(var_11);
            // adr = rowadr + k                                                                   <L 4122>
            var_14 = wp::add(var_8, var_13);
            // cj = dof_cdof_in[worldid, M_colind[adr]]                                           <L 4123>
            var_15 = wp::address(var_M_colind, var_14);
            var_17 = wp::load(var_15);
            var_16 = wp::address(var_dof_cdof_in, var_0, var_17);
            var_19 = wp::load(var_16);
            var_18 = wp::copy(var_19);
            // if cj >= 0:                                                                        <L 4124>
            var_21 = (var_18 >= var_20);
            if (var_21) {
                // val = M_in[worldid, adr]                                                       <L 4125>
                var_22 = wp::address(var_M_in, var_0, var_14);
                var_24 = wp::load(var_22);
                var_23 = wp::copy(var_24);
                // M_c_out[worldid, ci, cj] = val                                                 <L 4126>
                wp::array_store(var_M_c_out, var_0, var_3, var_18, var_23);
                // M_c_out[worldid, cj, ci] = val                                                 <L 4127>
                wp::array_store(var_M_c_out, var_0, var_18, var_3, var_23);
            }
            goto start_for_1;
        end_for_1:;
    }
}



extern "C" __global__ void _solve_beta_accumulate_tiled_2c9d0ce3_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_nv,
    wp::array_t<wp::float32> var_ctx_grad_in,
    wp::array_t<wp::float32> var_ctx_Mgrad_in,
    wp::array_t<wp::float32> var_ctx_prev_grad_in,
    wp::array_t<wp::float32> var_ctx_prev_Mgrad_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_ctx_beta_num_out,
    wp::array_t<wp::float32> var_ctx_beta_den_out)
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
        const wp::float32 var_5 = 0.0;
        wp::float32 var_6;
        const wp::float32 var_7 = 0.0;
        wp::float32 var_8;
        wp::int32 var_9;
        wp::range_t var_10;
        wp::int32 var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        wp::float32* var_15;
        wp::float32* var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        wp::float32* var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        wp::float32 var_25;
        const bool var_26 = true;
        wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>> var_27 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>>{};
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_28 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
        const bool var_29 = true;
        wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>> var_30 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<256>>>{};
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_31 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
        const wp::int32 var_32 = 0;
        bool var_33;
        const wp::int32 var_34 = 0;
        wp::float32 var_35;
        const wp::int32 var_36 = 0;
        wp::float32 var_37;
        //---------
        // forward
        // def _solve_beta_accumulate_tiled(                                                      <L 3553>
        // worldid, tid = wp.tid()                                                                <L 3566>
        builtin_tid2d(var_0, var_1);
        // if ctx_done_in[worldid]:                                                               <L 3568>
        var_2 = wp::address(var_ctx_done_in, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // return                                                                             <L 3569>
            continue;
        }
        var_4 = wp::load(var_2);
        // local_num = float(0.0)                                                                 <L 3571>
        var_6 = wp::float(var_5);
        // local_den = float(0.0)                                                                 <L 3572>
        var_8 = wp::float(var_7);
        // BLOCK_DIM = wp.block_dim()                                                             <L 3573>
        var_9 = builtin_block_dim();
        // for dofid in range(tid, nv, BLOCK_DIM):                                                <L 3575>
        var_10 = wp::range(var_1, var_nv, var_9);
        start_for_1:;
            if (iter_cmp(var_10) == 0) goto end_for_1;
            var_11 = wp::iter_next(var_10);
            // prev_Mgrad = ctx_prev_Mgrad_in[worldid, dofid]                                     <L 3576>
            var_12 = wp::address(var_ctx_prev_Mgrad_in, var_0, var_11);
            var_14 = wp::load(var_12);
            var_13 = wp::copy(var_14);
            // num = ctx_grad_in[worldid, dofid] * (ctx_Mgrad_in[worldid, dofid] - prev_Mgrad)       <L 3577>
            var_15 = wp::address(var_ctx_grad_in, var_0, var_11);
            var_16 = wp::address(var_ctx_Mgrad_in, var_0, var_11);
            var_18 = wp::load(var_16);
            var_17 = wp::sub(var_18, var_13);
            var_20 = wp::load(var_15);
            var_19 = wp::mul(var_20, var_17);
            // den = ctx_prev_grad_in[worldid, dofid] * prev_Mgrad                                <L 3578>
            var_21 = wp::address(var_ctx_prev_grad_in, var_0, var_11);
            var_23 = wp::load(var_21);
            var_22 = wp::mul(var_23, var_13);
            // local_num += num                                                                   <L 3579>
            var_24 = wp::add(var_6, var_19);
            // local_den += den                                                                   <L 3580>
            var_25 = wp::add(var_8, var_22);
            wp::assign(var_6, var_24);
            wp::assign(var_8, var_25);
            goto start_for_1;
        end_for_1:;
        // num_tile = wp.tile(local_num, preserve_type=True)                                      <L 3582>
        var_27 = wp::tile<wp::float32>(var_6);
        // num_sum = wp.tile_reduce(wp.add, num_tile)                                             <L 3583>
        var_28 = wp::tile_reduce(wp::add, var_27);
        // den_tile = wp.tile(local_den, preserve_type=True)                                      <L 3585>
        var_30 = wp::tile<wp::float32>(var_8);
        // den_sum = wp.tile_reduce(wp.add, den_tile)                                             <L 3586>
        var_31 = wp::tile_reduce(wp::add, var_30);
        // if tid == 0:                                                                           <L 3588>
        var_33 = (var_1 == var_32);
        if (var_33) {
            // ctx_beta_num_out[worldid] = num_sum[0]                                             <L 3589>
            var_35 = wp::tile_extract(var_28, var_34);
            wp::array_store(var_ctx_beta_num_out, var_0, var_35);
            // ctx_beta_den_out[worldid] = den_sum[0]                                             <L 3590>
            var_37 = wp::tile_extract(var_31, var_36);
            wp::array_store(var_ctx_beta_den_out, var_0, var_37);
        }
    }
}



extern "C" __global__ void _padding_h_cc932c7a_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_nv,
    wp::array_t<bool> var_ctx_done_in,
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
        bool* var_2;
        bool var_3;
        bool var_4;
        wp::int32 var_5;
        const wp::float32 var_6 = 1.0;
        //---------
        // forward
        // def _padding_h(nv: int, ctx_done_in: wp.array[bool], ctx_h_out: wp.array3d[float]):       <L 3022>
        // worldid, elementid = wp.tid()                                                          <L 3023>
        builtin_tid2d(var_0, var_1);
        // if ctx_done_in[worldid]:                                                               <L 3025>
        var_2 = wp::address(var_ctx_done_in, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // return                                                                             <L 3026>
            continue;
        }
        var_4 = wp::load(var_2);
        // dofid = nv + elementid                                                                 <L 3028>
        var_5 = wp::add(var_nv, var_1);
        // ctx_h_out[worldid, dofid, dofid] = 1.0                                                 <L 3029>
        wp::array_store(var_ctx_h_out, var_0, var_5, var_5, var_6);
    }
}



extern "C" __global__ void _init_compact_inertia_48546221_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::int32> var_ncdof_in,
    wp::array_t<wp::float32> var_M_c_out)
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
        const wp::float32 var_3 = 0.0;
        bool var_4;
        bool var_5;
        wp::int32* var_6;
        bool var_7;
        wp::int32 var_8;
        const wp::float32 var_9 = 1.0;
        wp::float32 var_10;
        //---------
        // forward
        // def _init_compact_inertia(                                                             <L 4023>
        // worldid, i, j = wp.tid()                                                               <L 4029>
        builtin_tid3d(var_0, var_1, var_2);
        // val = 0.0                                                                              <L 4030>
        // if i == j and i >= ncdof_in[worldid]:                                                  <L 4031>
        var_5 = (var_1 == var_2);
        var_4 = var_5;
        if (var_4) {
            var_6 = wp::address(var_ncdof_in, var_0);
            var_8 = wp::load(var_6);
            var_7 = (var_1 >= var_8);
            var_4 = var_4 && var_7;
        }
        if (var_4) {
            // val = 1.0                                                                          <L 4032>
        }
        var_10 = wp::where(var_4, var_9, var_3);
        // M_c_out[worldid, i, j] = val                                                           <L 4033>
        wp::array_store(var_M_c_out, var_0, var_1, var_2, var_10);
    }
}



extern "C" __global__ void _gather_dof_vecs_compact_a4bcec76_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_qacc_warmstart_in,
    wp::array_t<wp::float32> var_qfrc_smooth_in,
    wp::array_t<wp::float32> var_qacc_smooth_in,
    wp::array_t<wp::int32> var_cdof_dof_in,
    wp::array_t<wp::float32> var_qfrc_smooth_c_out,
    wp::array_t<wp::float32> var_qacc_smooth_c_out,
    wp::array_t<wp::float32> var_qacc_warmstart_c_out)
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
        const wp::int32 var_5 = 0;
        bool var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        wp::float32* var_9;
        wp::float32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        const wp::float32 var_13 = 0.0;
        const wp::float32 var_14 = 0.0;
        const wp::float32 var_15 = 0.0;
        //---------
        // forward
        // def _gather_dof_vecs_compact(                                                          <L 4199>
        // worldid, ci = wp.tid()                                                                 <L 4210>
        builtin_tid2d(var_0, var_1);
        // dof = cdof_dof_in[worldid, ci]                                                         <L 4211>
        var_2 = wp::address(var_cdof_dof_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if dof >= 0:                                                                           <L 4212>
        var_6 = (var_3 >= var_5);
        if (var_6) {
            // qfrc_smooth_c_out[worldid, ci] = qfrc_smooth_in[worldid, dof]                      <L 4213>
            var_7 = wp::address(var_qfrc_smooth_in, var_0, var_3);
            var_8 = wp::load(var_7);
            wp::array_store(var_qfrc_smooth_c_out, var_0, var_1, var_8);
            // qacc_smooth_c_out[worldid, ci] = qacc_smooth_in[worldid, dof]                      <L 4214>
            var_9 = wp::address(var_qacc_smooth_in, var_0, var_3);
            var_10 = wp::load(var_9);
            wp::array_store(var_qacc_smooth_c_out, var_0, var_1, var_10);
            // qacc_warmstart_c_out[worldid, ci] = qacc_warmstart_in[worldid, dof]                <L 4215>
            var_11 = wp::address(var_qacc_warmstart_in, var_0, var_3);
            var_12 = wp::load(var_11);
            wp::array_store(var_qacc_warmstart_c_out, var_0, var_1, var_12);
        }
        if (!var_6) {
            // qfrc_smooth_c_out[worldid, ci] = 0.0                                               <L 4217>
            wp::array_store(var_qfrc_smooth_c_out, var_0, var_1, var_13);
            // qacc_smooth_c_out[worldid, ci] = 0.0                                               <L 4218>
            wp::array_store(var_qacc_smooth_c_out, var_0, var_1, var_14);
            // qacc_warmstart_c_out[worldid, ci] = 0.0                                            <L 4219>
            wp::array_store(var_qacc_warmstart_c_out, var_0, var_1, var_15);
        }
    }
}



extern "C" __global__ void _gather_J_dense_7f22c6f3_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_nefc_in,
    wp::array_t<wp::int32> var_dof_cdof_in,
    wp::array_t<wp::float32> var_J_in,
    wp::array_t<wp::float32> var_J_c_out)
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
        bool var_3;
        wp::int32 var_4;
        wp::shape_t* var_5;
        const wp::int32 var_6 = 1;
        wp::int32 var_7;
        wp::shape_t var_8;
        wp::range_t var_9;
        wp::int32 var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        const wp::int32 var_14 = 0;
        bool var_15;
        wp::float32* var_16;
        wp::float32 var_17;
        //---------
        // forward
        // def _gather_J_dense(                                                                   <L 4268>
        // worldid, efcid = wp.tid()                                                              <L 4277>
        builtin_tid2d(var_0, var_1);
        // if efcid >= nefc_in[worldid]:                                                          <L 4278>
        var_2 = wp::address(var_nefc_in, var_0);
        var_4 = wp::load(var_2);
        var_3 = (var_1 >= var_4);
        if (var_3) {
            // return                                                                             <L 4279>
            continue;
        }
        // nv = dof_cdof_in.shape[1]                                                              <L 4280>
        var_5 = &(var_dof_cdof_in.shape);
        var_8 = wp::load(var_5);
        var_7 = wp::extract(var_8, var_6);
        // for j in range(nv):                                                                    <L 4281>
        var_9 = wp::range(var_7);
        start_for_1:;
            if (iter_cmp(var_9) == 0) goto end_for_1;
            var_10 = wp::iter_next(var_9);
            // cj = dof_cdof_in[worldid, j]                                                       <L 4282>
            var_11 = wp::address(var_dof_cdof_in, var_0, var_10);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
            // if cj >= 0:                                                                        <L 4283>
            var_15 = (var_12 >= var_14);
            if (var_15) {
                // J_c_out[worldid, efcid, cj] = J_in[worldid, efcid, j]                          <L 4284>
                var_16 = wp::address(var_J_in, var_0, var_1, var_10);
                var_17 = wp::load(var_16);
                wp::array_store(var_J_c_out, var_0, var_1, var_12, var_17);
            }
            goto start_for_1;
        end_for_1:;
    }
}

