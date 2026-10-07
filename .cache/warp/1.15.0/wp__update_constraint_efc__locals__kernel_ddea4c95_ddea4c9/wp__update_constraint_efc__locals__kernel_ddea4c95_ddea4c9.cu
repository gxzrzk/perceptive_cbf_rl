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


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/solver.py:402
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _eval_constraint_0(
    bool var_is_equality,
    bool var_is_friction,
    bool var_is_elliptic,
    wp::float32 var_jaref,
    wp::float32 var_D,
    wp::float32 var_frictionloss,
    wp::int32 var_efcid,
    wp::int32 var_efcid0,
    wp::float32 var_jaref0,
    wp::float32 var_D0,
    wp::float32 var_mu,
    wp::float32 var_ufrictionj,
    wp::float32 var_TT)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::float32 var_1;
    const wp::float32 var_2 = 0.5;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    const wp::int32 var_6 = 1;
    const wp::int32 var_7 = 1;
    wp::float32 var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    bool var_12;
    const wp::int32 var_13 = 2;
    const wp::int32 var_14 = 2;
    wp::float32 var_15;
    wp::float32 var_16;
    const wp::float32 var_17 = 0.5;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    wp::vec_t<3, wp::float32> var_21;
    bool var_22;
    wp::float32 var_23;
    const wp::int32 var_24 = 3;
    const wp::int32 var_25 = 3;
    wp::float32 var_26;
    wp::float32 var_27;
    const wp::float32 var_28 = 0.5;
    wp::float32 var_29;
    wp::float32 var_30;
    wp::float32 var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    const wp::int32 var_35 = 1;
    const wp::int32 var_36 = 1;
    wp::float32 var_37;
    const wp::float32 var_38 = 0.5;
    wp::float32 var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::vec_t<3, wp::float32> var_42;
    wp::float32 var_43;
    const wp::float32 var_44 = 0.0;
    bool var_45;
    const wp::float32 var_46 = 0.0;
    wp::float32 var_47;
    wp::float32 var_48;
    bool var_49;
    wp::float32 var_50;
    bool var_51;
    bool var_52;
    const wp::float32 var_53 = 0.0;
    bool var_54;
    const wp::float32 var_55 = 0.0;
    bool var_56;
    const wp::float32 var_57 = 0.0;
    const wp::int32 var_58 = 0;
    const wp::int32 var_59 = 0;
    wp::float32 var_60;
    const wp::float32 var_61 = 0.0;
    wp::vec_t<3, wp::float32> var_62;
    bool var_63;
    wp::float32 var_64;
    wp::float32 var_65;
    const wp::float32 var_66 = 0.0;
    bool var_67;
    bool var_68;
    const wp::float32 var_69 = 0.0;
    bool var_70;
    const wp::float32 var_71 = 0.0;
    bool var_72;
    wp::float32 var_73;
    wp::float32 var_74;
    const wp::int32 var_75 = 1;
    const wp::int32 var_76 = 1;
    wp::float32 var_77;
    const wp::float32 var_78 = 0.5;
    wp::float32 var_79;
    wp::float32 var_80;
    wp::float32 var_81;
    wp::vec_t<3, wp::float32> var_82;
    wp::float32 var_83;
    const wp::float32 var_84 = 1.0;
    wp::float32 var_85;
    wp::float32 var_86;
    wp::float32 var_87;
    wp::float32 var_88;
    wp::float32 var_89;
    wp::float32 var_90;
    wp::float32 var_91;
    wp::float32 var_92;
    wp::float32 var_93;
    bool var_94;
    const wp::int32 var_95 = 4;
    const wp::int32 var_96 = 4;
    wp::float32 var_97;
    const wp::float32 var_98 = 0.5;
    wp::float32 var_99;
    wp::float32 var_100;
    wp::float32 var_101;
    wp::vec_t<3, wp::float32> var_102;
    wp::float32 var_103;
    wp::float32 var_104;
    wp::float32 var_105;
    const wp::int32 var_106 = 4;
    const wp::int32 var_107 = 4;
    wp::float32 var_108;
    const wp::float32 var_109 = 0.0;
    wp::vec_t<3, wp::float32> var_110;
    const wp::float32 var_111 = 0.0;
    bool var_112;
    const wp::float32 var_113 = 0.0;
    const wp::int32 var_114 = 0;
    const wp::int32 var_115 = 0;
    wp::float32 var_116;
    const wp::float32 var_117 = 0.0;
    wp::vec_t<3, wp::float32> var_118;
    wp::float32 var_119;
    wp::float32 var_120;
    const wp::int32 var_121 = 1;
    const wp::int32 var_122 = 1;
    wp::float32 var_123;
    const wp::float32 var_124 = 0.5;
    wp::float32 var_125;
    wp::float32 var_126;
    wp::float32 var_127;
    wp::vec_t<3, wp::float32> var_128;
    //---------
    // forward
    // def _eval_constraint(                                                                  <L 403>
    // if is_equality:                                                                        <L 419>
    if (var_is_equality) {
        // force = -D * jaref                                                                 <L 420>
        var_0 = wp::neg(var_D);
        var_1 = wp::mul(var_0, var_jaref);
        // cost = 0.5 * D * jaref * jaref                                                     <L 421>
        var_3 = wp::mul(var_2, var_D);
        var_4 = wp::mul(var_3, var_jaref);
        var_5 = wp::mul(var_4, var_jaref);
        // return wp.vec3(force, float(types.ConstraintState.QUADRATIC.value), cost)          <L 422>
        var_8 = wp::float(var_7);
        var_9 = wp::vec_t<3, wp::float32>(var_1, var_8, var_5);
        return var_9;
    }
    // if is_friction:                                                                        <L 424>
    if (var_is_friction) {
        // rf = math.safe_div(frictionloss, D)                                                <L 425>
        var_10 = safe_div_0(var_frictionloss, var_D);
        // if jaref <= -rf:                                                                   <L 426>
        var_11 = wp::neg(var_10);
        var_12 = (var_jaref <= var_11);
        if (var_12) {
            // return wp.vec3(frictionloss, float(types.ConstraintState.LINEARNEG.value), -frictionloss * (0.5 * rf + jaref))       <L 427>
            var_15 = wp::float(var_14);
            var_16 = wp::neg(var_frictionloss);
            var_18 = wp::mul(var_17, var_10);
            var_19 = wp::add(var_18, var_jaref);
            var_20 = wp::mul(var_16, var_19);
            var_21 = wp::vec_t<3, wp::float32>(var_frictionloss, var_15, var_20);
            return var_21;
        }
        if (!var_12) {
            // elif jaref >= rf:                                                              <L 428>
            var_22 = (var_jaref >= var_10);
            if (var_22) {
                // return wp.vec3(-frictionloss, float(types.ConstraintState.LINEARPOS.value), -frictionloss * (0.5 * rf - jaref))       <L 429>
                var_23 = wp::neg(var_frictionloss);
                var_26 = wp::float(var_25);
                var_27 = wp::neg(var_frictionloss);
                var_29 = wp::mul(var_28, var_10);
                var_30 = wp::sub(var_29, var_jaref);
                var_31 = wp::mul(var_27, var_30);
                var_32 = wp::vec_t<3, wp::float32>(var_23, var_26, var_31);
                return var_32;
            }
            if (!var_22) {
                // return wp.vec3(-D * jaref, float(types.ConstraintState.QUADRATIC.value), 0.5 * D * jaref * jaref)       <L 431>
                var_33 = wp::neg(var_D);
                var_34 = wp::mul(var_33, var_jaref);
                var_37 = wp::float(var_36);
                var_39 = wp::mul(var_38, var_D);
                var_40 = wp::mul(var_39, var_jaref);
                var_41 = wp::mul(var_40, var_jaref);
                var_42 = wp::vec_t<3, wp::float32>(var_34, var_37, var_41);
                return var_42;
            }
        }
    }
    // if is_elliptic:                                                                        <L 433>
    if (var_is_elliptic) {
        // N = jaref0 * mu                                                                    <L 434>
        var_43 = wp::mul(var_jaref0, var_mu);
        // if TT <= 0.0:                                                                      <L 435>
        var_45 = (var_TT <= var_44);
        if (var_45) {
            // T = 0.0                                                                        <L 436>
        }
        if (!var_45) {
            // T = wp.sqrt(TT)                                                                <L 438>
            var_47 = wp::sqrt(var_TT);
        }
        var_48 = wp::where(var_45, var_46, var_47);
        // if (N >= mu * T) or ((T <= 0.0) and (N >= 0.0)):                                   <L 441>
        var_50 = wp::mul(var_mu, var_48);
        var_51 = (var_43 >= var_50);
        var_49 = var_51;
        if (!var_49) {
            var_54 = (var_48 <= var_53);
            var_52 = var_54;
            if (var_52) {
                var_56 = (var_43 >= var_55);
                var_52 = var_52 && var_56;
            }
            var_49 = var_49 || var_52;
        }
        if (var_49) {
            // return wp.vec3(0.0, float(types.ConstraintState.SATISFIED.value), 0.0)         <L 442>
            var_60 = wp::float(var_59);
            var_62 = wp::vec_t<3, wp::float32>(var_57, var_60, var_61);
            return var_62;
        }
        if (!var_49) {
            // elif (mu * N + T <= 0.0) or ((T <= 0.0) and (N < 0.0)):                        <L 444>
            var_64 = wp::mul(var_mu, var_43);
            var_65 = wp::add(var_64, var_48);
            var_67 = (var_65 <= var_66);
            var_63 = var_67;
            if (!var_63) {
                var_70 = (var_48 <= var_69);
                var_68 = var_70;
                if (var_68) {
                    var_72 = (var_43 < var_71);
                    var_68 = var_68 && var_72;
                }
                var_63 = var_63 || var_68;
            }
            if (var_63) {
                // return wp.vec3(-D * jaref, float(types.ConstraintState.QUADRATIC.value), 0.5 * D * jaref * jaref)       <L 445>
                var_73 = wp::neg(var_D);
                var_74 = wp::mul(var_73, var_jaref);
                var_77 = wp::float(var_76);
                var_79 = wp::mul(var_78, var_D);
                var_80 = wp::mul(var_79, var_jaref);
                var_81 = wp::mul(var_80, var_jaref);
                var_82 = wp::vec_t<3, wp::float32>(var_74, var_77, var_81);
                return var_82;
            }
            if (!var_63) {
                // dm = math.safe_div(D0, mu * mu * (1.0 + mu * mu))                          <L 448>
                var_83 = wp::mul(var_mu, var_mu);
                var_85 = wp::mul(var_mu, var_mu);
                var_86 = wp::add(var_84, var_85);
                var_87 = wp::mul(var_83, var_86);
                var_88 = safe_div_0(var_D0, var_87);
                // nmt = N - mu * T                                                           <L 449>
                var_89 = wp::mul(var_mu, var_48);
                var_90 = wp::sub(var_43, var_89);
                // force_normal = -dm * nmt * mu                                              <L 450>
                var_91 = wp::neg(var_88);
                var_92 = wp::mul(var_91, var_90);
                var_93 = wp::mul(var_92, var_mu);
                // if efcid == efcid0:                                                        <L 452>
                var_94 = (var_efcid == var_efcid0);
                if (var_94) {
                    // return wp.vec3(force_normal, float(types.ConstraintState.CONE.value), 0.5 * dm * nmt * nmt)       <L 453>
                    var_97 = wp::float(var_96);
                    var_99 = wp::mul(var_98, var_88);
                    var_100 = wp::mul(var_99, var_90);
                    var_101 = wp::mul(var_100, var_90);
                    var_102 = wp::vec_t<3, wp::float32>(var_93, var_97, var_101);
                    return var_102;
                }
                if (!var_94) {
                    // force_tangent = -math.safe_div(force_normal, T) * ufrictionj           <L 455>
                    var_103 = safe_div_0(var_93, var_48);
                    var_104 = wp::neg(var_103);
                    var_105 = wp::mul(var_104, var_ufrictionj);
                }
                // return wp.vec3(force_tangent, float(types.ConstraintState.CONE.value), 0.0)       <L 457>
                var_108 = wp::float(var_107);
                var_110 = wp::vec_t<3, wp::float32>(var_105, var_108, var_109);
                return var_110;
            }
        }
    }
    // if jaref >= 0.0:                                                                       <L 459>
    var_112 = (var_jaref >= var_111);
    if (var_112) {
        // return wp.vec3(0.0, float(types.ConstraintState.SATISFIED.value), 0.0)             <L 460>
        var_116 = wp::float(var_115);
        var_118 = wp::vec_t<3, wp::float32>(var_113, var_116, var_117);
        return var_118;
    }
    if (!var_112) {
        // return wp.vec3(-D * jaref, float(types.ConstraintState.QUADRATIC.value), 0.5 * D * jaref * jaref)       <L 462>
        var_119 = wp::neg(var_D);
        var_120 = wp::mul(var_119, var_jaref);
        var_123 = wp::float(var_122);
        var_125 = wp::mul(var_124, var_D);
        var_126 = wp::mul(var_125, var_jaref);
        var_127 = wp::mul(var_126, var_jaref);
        var_128 = wp::vec_t<3, wp::float32>(var_120, var_123, var_127);
        return var_128;
    }
    return {};
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


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/solver.py:402
static CUDA_CALLABLE void adj__eval_constraint_0(
    bool var_is_equality,
    bool var_is_friction,
    bool var_is_elliptic,
    wp::float32 var_jaref,
    wp::float32 var_D,
    wp::float32 var_frictionloss,
    wp::int32 var_efcid,
    wp::int32 var_efcid0,
    wp::float32 var_jaref0,
    wp::float32 var_D0,
    wp::float32 var_mu,
    wp::float32 var_ufrictionj,
    wp::float32 var_TT,
    bool & adj_is_equality,
    bool & adj_is_friction,
    bool & adj_is_elliptic,
    wp::float32 & adj_jaref,
    wp::float32 & adj_D,
    wp::float32 & adj_frictionloss,
    wp::int32 & adj_efcid,
    wp::int32 & adj_efcid0,
    wp::float32 & adj_jaref0,
    wp::float32 & adj_D0,
    wp::float32 & adj_mu,
    wp::float32 & adj_ufrictionj,
    wp::float32 & adj_TT,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void _update_constraint_efc__locals__kernel_2c99286b_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_opt_impratio_invsqrt,
    wp::array_t<wp::int32> var_ne_in,
    wp::array_t<wp::int32> var_nf_in,
    wp::array_t<wp::int32> var_nefc_in,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_in,
    wp::array_t<wp::int32> var_contact_dim_in,
    wp::array_t<wp::int32> var_contact_efc_address_in,
    wp::array_t<wp::int32> var_efc_type_in,
    wp::array_t<wp::int32> var_efc_id_in,
    wp::array_t<wp::float32> var_efc_D_in,
    wp::array_t<wp::float32> var_efc_frictionloss_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::array_t<wp::float32> var_ctx_Jaref_in,
    wp::array_t<bool> var_ctx_ls_exhausted_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_efc_force_out,
    wp::array_t<wp::int32> var_efc_state_out,
    wp::array_t<wp::int32> var_quad_changed_ids_out,
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
        wp::int32 var_1;
        bool* var_2;
        bool var_3;
        bool var_4;
        const bool var_5 = true;
        bool var_6;
        const wp::int32 var_7 = 0;
        bool var_8;
        bool* var_9;
        bool var_10;
        const wp::int32 var_11 = 1;
        wp::int32 var_12;
        wp::int32* var_13;
        bool var_14;
        wp::int32 var_15;
        const bool var_16 = true;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::float32* var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::float32* var_23;
        wp::float32 var_24;
        wp::float32 var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::int32* var_29;
        wp::int32 var_30;
        wp::int32 var_31;
        bool var_32;
        bool var_33;
        bool var_34;
        wp::int32 var_35;
        bool var_36;
        wp::int32* var_37;
        const wp::int32 var_38 = 7;
        bool var_39;
        wp::int32 var_40;
        wp::float32* var_41;
        wp::float32 var_42;
        const wp::float32 var_43 = 0.0;
        wp::float32 var_44;
        const wp::int32 var_45 = -1;
        const wp::float32 var_46 = 0.0;
        wp::float32 var_47;
        const wp::float32 var_48 = 0.0;
        wp::float32 var_49;
        const wp::float32 var_50 = 0.0;
        wp::float32 var_51;
        const wp::float32 var_52 = 0.0;
        wp::float32 var_53;
        const wp::float32 var_54 = 0.0;
        wp::float32 var_55;
        wp::int32* var_56;
        wp::int32 var_57;
        wp::int32 var_58;
        const wp::int32 var_59 = 0;
        wp::int32* var_60;
        bool var_61;
        wp::int32 var_62;
        const wp::int32 var_63 = 0;
        wp::int32* var_64;
        wp::int32 var_65;
        wp::int32 var_66;
        const wp::int32 var_67 = 0;
        bool var_68;
        wp::int32* var_69;
        wp::int32 var_70;
        wp::int32 var_71;
        wp::vec_t<5, wp::float32>* var_72;
        wp::vec_t<5, wp::float32> var_73;
        wp::vec_t<5, wp::float32> var_74;
        const wp::int32 var_75 = 0;
        wp::float32 var_76;
        wp::shape_t* var_77;
        const wp::int32 var_78 = 0;
        wp::int32 var_79;
        wp::shape_t var_80;
        wp::int32 var_81;
        wp::float32* var_82;
        wp::float32 var_83;
        wp::float32 var_84;
        wp::float32* var_85;
        wp::float32 var_86;
        wp::float32 var_87;
        wp::float32* var_88;
        wp::float32 var_89;
        wp::float32 var_90;
        const wp::int32 var_91 = 1;
        wp::range_t var_92;
        wp::int32 var_93;
        wp::int32* var_94;
        wp::int32 var_95;
        wp::int32 var_96;
        const wp::int32 var_97 = 0;
        bool var_98;
        const wp::int32 var_99 = 1;
        wp::int32 var_100;
        wp::float32 var_101;
        wp::float32* var_102;
        wp::float32 var_103;
        wp::float32 var_104;
        wp::float32 var_105;
        wp::float32 var_106;
        bool var_107;
        wp::float32 var_108;
        wp::float32 var_109;
        wp::int32 var_110;
        wp::float32 var_111;
        wp::float32 var_112;
        wp::float32 var_113;
        wp::vec_t<3, wp::float32> var_114;
        const wp::int32 var_115 = 1;
        wp::float32 var_116;
        wp::int32 var_117;
        const wp::int32 var_118 = 0;
        wp::float32 var_119;
        const bool var_120 = true;
        const wp::int32 var_121 = 1;
        bool var_122;
        const wp::int32 var_123 = 1;
        bool var_124;
        bool var_125;
        const wp::int32 var_126 = 1;
        wp::int32 var_127;
        bool var_128;
        const wp::int32 var_129 = 1;
        wp::int32 var_130;
        //---------
        // forward
        // def kernel(                                                                            <L 1655>
        // worldid, efcid = wp.tid()                                                              <L 1682>
        builtin_tid2d(var_0, var_1);
        // if ctx_done_in[worldid]:                                                               <L 1684>
        var_2 = wp::address(var_ctx_done_in, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // return                                                                             <L 1685>
            continue;
        }
        var_4 = wp::load(var_2);
        // if wp.static(TRACK_CHANGES):                                                           <L 1689>
        // if efcid == 0 and ctx_ls_exhausted_in[worldid]:                                        <L 1690>
        var_8 = (var_1 == var_7);
        var_6 = var_8;
        if (var_6) {
            var_9 = wp::address(var_ctx_ls_exhausted_in, var_0);
            var_10 = wp::load(var_9);
            var_6 = var_6 && var_10;
        }
        if (var_6) {
            // wp.atomic_add(state_changed_count_out, worldid, 1)                                 <L 1691>
            var_12 = wp::atomic_add(var_state_changed_count_out, var_0, var_11);
        }
        // if efcid >= nefc_in[worldid]:                                                          <L 1693>
        var_13 = wp::address(var_nefc_in, var_0);
        var_15 = wp::load(var_13);
        var_14 = (var_1 >= var_15);
        if (var_14) {
            // return                                                                             <L 1694>
            continue;
        }
        // if wp.static(TRACK_CHANGES):                                                           <L 1697>
        // old_state = efc_state_out[worldid, efcid]                                              <L 1698>
        var_17 = wp::address(var_efc_state_out, var_0, var_1);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // efc_D = efc_D_in[worldid, efcid]                                                       <L 1700>
        var_20 = wp::address(var_efc_D_in, var_0, var_1);
        var_22 = wp::load(var_20);
        var_21 = wp::copy(var_22);
        // Jaref = ctx_Jaref_in[worldid, efcid]                                                   <L 1701>
        var_23 = wp::address(var_ctx_Jaref_in, var_0, var_1);
        var_25 = wp::load(var_23);
        var_24 = wp::copy(var_25);
        // ne = ne_in[worldid]                                                                    <L 1703>
        var_26 = wp::address(var_ne_in, var_0);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // nf = nf_in[worldid]                                                                    <L 1704>
        var_29 = wp::address(var_nf_in, var_0);
        var_31 = wp::load(var_29);
        var_30 = wp::copy(var_31);
        // is_equality = efcid < ne                                                               <L 1706>
        var_32 = (var_1 < var_27);
        // is_friction = (not is_equality) and (efcid < ne + nf)                                  <L 1707>
        var_34 = wp::unot(var_32);
        var_33 = var_34;
        if (var_33) {
            var_35 = wp::add(var_27, var_30);
            var_36 = (var_1 < var_35);
            var_33 = var_33 && var_36;
        }
        // is_elliptic = efc_type_in[worldid, efcid] == types.ConstraintType.CONTACT_ELLIPTIC       <L 1708>
        var_37 = wp::address(var_efc_type_in, var_0, var_1);
        var_40 = wp::load(var_37);
        var_39 = (var_40 == var_38);
        // frictionloss = efc_frictionloss_in[worldid, efcid] if is_friction else 0.0             <L 1710>
        if (var_33) {
            var_41 = wp::address(var_efc_frictionloss_in, var_0, var_1);
            var_42 = wp::load(var_41);
        }
        if (!var_33) {
        }
        var_44 = wp::where(var_33, var_42, var_43);
        // efcid0 = -1                                                                            <L 1712>
        // jaref0 = float(0.0)                                                                    <L 1713>
        var_47 = wp::float(var_46);
        // D0 = float(0.0)                                                                        <L 1714>
        var_49 = wp::float(var_48);
        // mu = float(0.0)                                                                        <L 1715>
        var_51 = wp::float(var_50);
        // ufrictionj = float(0.0)                                                                <L 1716>
        var_53 = wp::float(var_52);
        // TT = float(0.0)                                                                        <L 1717>
        var_55 = wp::float(var_54);
        // if is_elliptic:                                                                        <L 1719>
        if (var_39) {
            // conid = efc_id_in[worldid, efcid]                                                  <L 1720>
            var_56 = wp::address(var_efc_id_in, var_0, var_1);
            var_58 = wp::load(var_56);
            var_57 = wp::copy(var_58);
            // if conid >= nacon_in[0]:                                                           <L 1721>
            var_60 = wp::address(var_nacon_in, var_59);
            var_62 = wp::load(var_60);
            var_61 = (var_57 >= var_62);
            if (var_61) {
                // return                                                                         <L 1722>
                continue;
            }
            // efcid0 = contact_efc_address_in[conid, 0]                                          <L 1723>
            var_64 = wp::address(var_contact_efc_address_in, var_57, var_63);
            var_66 = wp::load(var_64);
            var_65 = wp::copy(var_66);
            // if efcid0 < 0:                                                                     <L 1724>
            var_68 = (var_65 < var_67);
            if (var_68) {
                // return                                                                         <L 1725>
                continue;
            }
            // dim = contact_dim_in[conid]                                                        <L 1727>
            var_69 = wp::address(var_contact_dim_in, var_57);
            var_71 = wp::load(var_69);
            var_70 = wp::copy(var_71);
            // friction = contact_friction_in[conid]                                              <L 1728>
            var_72 = wp::address(var_contact_friction_in, var_57);
            var_74 = wp::load(var_72);
            var_73 = wp::copy(var_74);
            // mu = friction[0] * opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]       <L 1729>
            var_76 = wp::extract(var_73, var_75);
            var_77 = &(var_opt_impratio_invsqrt.shape);
            var_80 = wp::load(var_77);
            var_79 = wp::extract(var_80, var_78);
            var_81 = wp::mod(var_0, var_79);
            var_82 = wp::address(var_opt_impratio_invsqrt, var_81);
            var_84 = wp::load(var_82);
            var_83 = wp::mul(var_76, var_84);
            // jaref0 = ctx_Jaref_in[worldid, efcid0]                                             <L 1730>
            var_85 = wp::address(var_ctx_Jaref_in, var_0, var_65);
            var_87 = wp::load(var_85);
            var_86 = wp::copy(var_87);
            // D0 = efc_D_in[worldid, efcid0]                                                     <L 1731>
            var_88 = wp::address(var_efc_D_in, var_0, var_65);
            var_90 = wp::load(var_88);
            var_89 = wp::copy(var_90);
            // for j in range(1, dim):                                                            <L 1733>
            var_92 = wp::range(var_91, var_70);
            start_for_4:;
                if (iter_cmp(var_92) == 0) goto end_for_4;
                var_93 = wp::iter_next(var_92);
                // efcidj = contact_efc_address_in[conid, j]                                      <L 1734>
                var_94 = wp::address(var_contact_efc_address_in, var_57, var_93);
                var_96 = wp::load(var_94);
                var_95 = wp::copy(var_96);
                // if efcidj < 0:                                                                 <L 1735>
                var_98 = (var_95 < var_97);
                if (var_98) {
                    // return                                                                     <L 1736>
                    continue;
                }
                // frictionj = friction[j - 1]                                                    <L 1737>
                var_100 = wp::sub(var_93, var_99);
                var_101 = wp::extract(var_73, var_100);
                // uj = ctx_Jaref_in[worldid, efcidj] * frictionj                                 <L 1738>
                var_102 = wp::address(var_ctx_Jaref_in, var_0, var_95);
                var_104 = wp::load(var_102);
                var_103 = wp::mul(var_104, var_101);
                // TT += uj * uj                                                                  <L 1739>
                var_105 = wp::mul(var_103, var_103);
                var_106 = wp::add(var_55, var_105);
                // if efcid == efcidj:                                                            <L 1740>
                var_107 = (var_1 == var_95);
                if (var_107) {
                    // ufrictionj = uj * frictionj                                                <L 1741>
                    var_108 = wp::mul(var_103, var_101);
                }
                var_109 = wp::where(var_107, var_108, var_53);
                wp::assign(var_53, var_109);
                wp::assign(var_55, var_106);
                goto start_for_4;
            end_for_4:;
        }
        var_110 = wp::where(var_39, var_65, var_45);
        var_111 = wp::where(var_39, var_86, var_47);
        var_112 = wp::where(var_39, var_89, var_49);
        var_113 = wp::where(var_39, var_83, var_51);
        // res = _eval_constraint(                                                                <L 1743>
        // is_equality,                                                                           <L 1744>
        // is_friction,                                                                           <L 1745>
        // is_elliptic,                                                                           <L 1746>
        // Jaref,                                                                                 <L 1747>
        // efc_D,                                                                                 <L 1748>
        // frictionloss,                                                                          <L 1749>
        // efcid,                                                                                 <L 1750>
        // efcid0,                                                                                <L 1751>
        // jaref0,                                                                                <L 1752>
        // D0,                                                                                    <L 1753>
        // mu,                                                                                    <L 1754>
        // ufrictionj,                                                                            <L 1755>
        // TT,                                                                                    <L 1756>
        var_114 = _eval_constraint_0(var_32, var_33, var_39, var_24, var_21, var_44, var_1, var_110, var_111, var_112, var_113, var_53, var_55);
        // new_state = int(res[1])                                                                <L 1759>
        var_116 = wp::extract(var_114, var_115);
        var_117 = wp::int(var_116);
        // efc_force_out[worldid, efcid] = res[0]                                                 <L 1760>
        var_119 = wp::extract(var_114, var_118);
        wp::array_store(var_efc_force_out, var_0, var_1, var_119);
        // efc_state_out[worldid, efcid] = new_state                                              <L 1761>
        wp::array_store(var_efc_state_out, var_0, var_1, var_117);
        // if wp.static(TRACK_CHANGES):                                                           <L 1763>
        // old_quad = old_state == types.ConstraintState.QUADRATIC.value                          <L 1764>
        var_122 = (var_18 == var_121);
        // new_quad = new_state == types.ConstraintState.QUADRATIC.value                          <L 1765>
        var_124 = (var_117 == var_123);
        // if old_quad != new_quad:                                                               <L 1766>
        var_125 = (var_122 != var_124);
        if (var_125) {
            // idx = wp.atomic_add(quad_changed_count_out, worldid, 1)                            <L 1767>
            var_127 = wp::atomic_add(var_quad_changed_count_out, var_0, var_126);
            // quad_changed_ids_out[worldid, idx] = efcid                                         <L 1768>
            wp::array_store(var_quad_changed_ids_out, var_0, var_127, var_1);
        }
        // if old_state != new_state:                                                             <L 1771>
        var_128 = (var_18 != var_117);
        if (var_128) {
            // wp.atomic_add(state_changed_count_out, worldid, 1)                                 <L 1772>
            var_130 = wp::atomic_add(var_state_changed_count_out, var_0, var_129);
        }
    }
}

