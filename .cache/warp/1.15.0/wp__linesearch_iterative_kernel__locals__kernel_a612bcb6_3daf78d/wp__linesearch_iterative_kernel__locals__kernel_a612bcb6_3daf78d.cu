#define WP_NO_BFLOAT16

#define WP_TILE_BLOCK_DIM 32
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


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/solver.py:122
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _eval_pt_direct_alpha_zero_0(
    wp::float32 var_jaref,
    wp::float32 var_jv,
    wp::float32 var_d)
{
    //---------
    // primal vars
    wp::float32 var_0;
    const wp::float32 var_1 = 0.5;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::vec_t<3, wp::float32> var_7;
    //---------
    // forward
    // def _eval_pt_direct_alpha_zero(jaref: float, jv: float, d: float) -> wp.vec3:          <L 123>
    // jvD = jv * d                                                                           <L 125>
    var_0 = wp::mul(var_jv, var_d);
    // return wp.vec3(0.5 * d * jaref * jaref, jvD * jaref, jv * jvD)                         <L 126>
    var_2 = wp::mul(var_1, var_d);
    var_3 = wp::mul(var_2, var_jaref);
    var_4 = wp::mul(var_3, var_jaref);
    var_5 = wp::mul(var_0, var_jaref);
    var_6 = wp::mul(var_jv, var_0);
    var_7 = wp::vec_t<3, wp::float32>(var_4, var_5, var_6);
    return var_7;
}


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


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/solver.py:223
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _eval_frictionloss_pt_0(
    wp::float32 var_x,
    wp::float32 var_f,
    wp::float32 var_rf,
    wp::float32 var_jv,
    wp::float32 var_d)
{
    //---------
    // primal vars
    bool var_0;
    wp::float32 var_1;
    bool var_2;
    bool var_3;
    wp::float32 var_4;
    const wp::float32 var_5 = 0.5;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::float32 var_12;
    bool var_13;
    const wp::float32 var_14 = -0.5;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    const wp::float32 var_20 = 0.0;
    wp::vec_t<3, wp::float32> var_21;
    const wp::float32 var_22 = -0.5;
    wp::float32 var_23;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::float32 var_26;
    const wp::float32 var_27 = 0.0;
    wp::vec_t<3, wp::float32> var_28;
    //---------
    // forward
    // def _eval_frictionloss_pt(x: float, f: float, rf: float, jv: float, d: float) -> wp.vec3:       <L 224>
    // if (-rf < x) and (x < rf):                                                             <L 226>
    var_1 = wp::neg(var_rf);
    var_2 = (var_1 < var_x);
    var_0 = var_2;
    if (var_0) {
        var_3 = (var_x < var_rf);
        var_0 = var_0 && var_3;
    }
    if (var_0) {
        // jvD = jv * d                                                                       <L 227>
        var_4 = wp::mul(var_jv, var_d);
        // return wp.vec3(0.5 * d * x * x, jvD * x, jv * jvD)                                 <L 228>
        var_6 = wp::mul(var_5, var_d);
        var_7 = wp::mul(var_6, var_x);
        var_8 = wp::mul(var_7, var_x);
        var_9 = wp::mul(var_4, var_x);
        var_10 = wp::mul(var_jv, var_4);
        var_11 = wp::vec_t<3, wp::float32>(var_8, var_9, var_10);
        return var_11;
    }
    if (!var_0) {
        // elif x <= -rf:                                                                     <L 229>
        var_12 = wp::neg(var_rf);
        var_13 = (var_x <= var_12);
        if (var_13) {
            // return wp.vec3(f * (-0.5 * rf - x), -f * jv, 0.0)                              <L 230>
            var_15 = wp::mul(var_14, var_rf);
            var_16 = wp::sub(var_15, var_x);
            var_17 = wp::mul(var_f, var_16);
            var_18 = wp::neg(var_f);
            var_19 = wp::mul(var_18, var_jv);
            var_21 = wp::vec_t<3, wp::float32>(var_17, var_19, var_20);
            return var_21;
        }
        if (!var_13) {
            // return wp.vec3(f * (-0.5 * rf + x), f * jv, 0.0)                               <L 232>
            var_23 = wp::mul(var_22, var_rf);
            var_24 = wp::add(var_23, var_x);
            var_25 = wp::mul(var_f, var_24);
            var_26 = wp::mul(var_f, var_jv);
            var_28 = wp::vec_t<3, wp::float32>(var_25, var_26, var_27);
            return var_28;
        }
    }
    return {};
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/solver.py:562
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _compute_efc_eval_pt_alpha_zero_pyramidal_0(
    wp::int32 var_efcid,
    wp::int32 var_ne,
    wp::int32 var_nf,
    wp::float32 var_efc_D,
    wp::array_t<wp::float32> var_efc_frictionloss,
    wp::float32 var_ctx_Jaref,
    wp::float32 var_ctx_jv)
{
    //---------
    // primal vars
    wp::int32 var_0;
    bool var_1;
    const wp::float32 var_2 = 0.0;
    bool var_3;
    wp::vec_t<3, wp::float32> var_4;
    const wp::float32 var_5 = 0.0;
    wp::vec_t<3, wp::float32> var_6;
    bool var_7;
    wp::float32* var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::vec_t<3, wp::float32> var_13;
    //---------
    // forward
    // def _compute_efc_eval_pt_alpha_zero_pyramidal(                                         <L 563>
    // if efcid >= ne + nf:                                                                   <L 575>
    var_0 = wp::add(var_ne, var_nf);
    var_1 = (var_efcid >= var_0);
    if (var_1) {
        // if ctx_Jaref < 0.0:                                                                <L 576>
        var_3 = (var_ctx_Jaref < var_2);
        if (var_3) {
            // return _eval_pt_direct_alpha_zero(ctx_Jaref, ctx_jv, efc_D)                    <L 577>
            var_4 = _eval_pt_direct_alpha_zero_0(var_ctx_Jaref, var_ctx_jv, var_efc_D);
            return var_4;
        }
        // return wp.vec3(0.0)                                                                <L 578>
        var_6 = wp::vec_t<3, wp::float32>(var_5);
        return var_6;
    }
    // if efcid >= ne:                                                                        <L 581>
    var_7 = (var_efcid >= var_ne);
    if (var_7) {
        // f = efc_frictionloss[efcid]                                                        <L 582>
        var_8 = wp::address(var_efc_frictionloss, var_efcid);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // rf = math.safe_div(f, efc_D)                                                       <L 583>
        var_11 = safe_div_0(var_9, var_efc_D);
        // return _eval_frictionloss_pt(ctx_Jaref, f, rf, ctx_jv, efc_D)                      <L 584>
        var_12 = _eval_frictionloss_pt_0(var_ctx_Jaref, var_9, var_11, var_ctx_jv, var_efc_D);
        return var_12;
    }
    // return _eval_pt_direct_alpha_zero(ctx_Jaref, ctx_jv, efc_D)                            <L 587>
    var_13 = _eval_pt_direct_alpha_zero_0(var_ctx_Jaref, var_ctx_jv, var_efc_D);
    return var_13;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/solver.py:137
static CUDA_CALLABLE wp::float32 _eval_pt_direct_cost_alpha_zero_0(
    wp::float32 var_jaref,
    wp::float32 var_d)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.5;
    wp::float32 var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    //---------
    // forward
    // def _eval_pt_direct_cost_alpha_zero(jaref: float, d: float) -> float:                  <L 138>
    // return 0.5 * d * jaref * jaref                                                         <L 139>
    var_1 = wp::mul(var_0, var_d);
    var_2 = wp::mul(var_1, var_jaref);
    var_3 = wp::mul(var_2, var_jaref);
    return var_3;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/solver.py:142
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _eval_pt_direct_shifted_0(
    wp::float32 var_jaref,
    wp::float32 var_jv,
    wp::float32 var_d,
    wp::float32 var_alpha,
    wp::float32 var_offset)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    const wp::float32 var_4 = 0.5;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::vec_t<3, wp::float32> var_11;
    //---------
    // forward
    // def _eval_pt_direct_shifted(jaref: float, jv: float, d: float, alpha: float, offset: float) -> wp.vec3:       <L 143>
    // jvD = jv * d                                                                           <L 145>
    var_0 = wp::mul(var_jv, var_d);
    // hessian = jv * jvD                                                                     <L 146>
    var_1 = wp::mul(var_jv, var_0);
    // alpha_h = alpha * hessian                                                              <L 147>
    var_2 = wp::mul(var_alpha, var_1);
    // return wp.vec3(alpha * (jvD * jaref + 0.5 * alpha_h) + offset, jvD * jaref + alpha_h, hessian)       <L 148>
    var_3 = wp::mul(var_0, var_jaref);
    var_5 = wp::mul(var_4, var_2);
    var_6 = wp::add(var_3, var_5);
    var_7 = wp::mul(var_alpha, var_6);
    var_8 = wp::add(var_7, var_offset);
    var_9 = wp::mul(var_0, var_jaref);
    var_10 = wp::add(var_9, var_2);
    var_11 = wp::vec_t<3, wp::float32>(var_8, var_10, var_1);
    return var_11;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/solver.py:235
static CUDA_CALLABLE wp::float32 _eval_frictionloss_cost_0(
    wp::float32 var_x,
    wp::float32 var_f,
    wp::float32 var_rf,
    wp::float32 var_d)
{
    //---------
    // primal vars
    bool var_0;
    wp::float32 var_1;
    bool var_2;
    bool var_3;
    const wp::float32 var_4 = 0.5;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    bool var_9;
    const wp::float32 var_10 = -0.5;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    const wp::float32 var_14 = -0.5;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    //---------
    // forward
    // def _eval_frictionloss_cost(x: float, f: float, rf: float, d: float) -> float:         <L 236>
    // if (-rf < x) and (x < rf):                                                             <L 237>
    var_1 = wp::neg(var_rf);
    var_2 = (var_1 < var_x);
    var_0 = var_2;
    if (var_0) {
        var_3 = (var_x < var_rf);
        var_0 = var_0 && var_3;
    }
    if (var_0) {
        // return 0.5 * d * x * x                                                             <L 238>
        var_5 = wp::mul(var_4, var_d);
        var_6 = wp::mul(var_5, var_x);
        var_7 = wp::mul(var_6, var_x);
        return var_7;
    }
    if (!var_0) {
        // elif x <= -rf:                                                                     <L 239>
        var_8 = wp::neg(var_rf);
        var_9 = (var_x <= var_8);
        if (var_9) {
            // return f * (-0.5 * rf - x)                                                     <L 240>
            var_11 = wp::mul(var_10, var_rf);
            var_12 = wp::sub(var_11, var_x);
            var_13 = wp::mul(var_f, var_12);
            return var_13;
        }
    }
    // return f * (-0.5 * rf + x)                                                             <L 241>
    var_15 = wp::mul(var_14, var_rf);
    var_16 = wp::add(var_15, var_x);
    var_17 = wp::mul(var_f, var_16);
    return var_17;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/solver.py:218
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _shift_cost_0(
    wp::vec_t<3, wp::float32> var_pt,
    wp::float32 var_cost0)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    wp::float32 var_2;
    const wp::int32 var_3 = 1;
    wp::float32 var_4;
    const wp::int32 var_5 = 2;
    wp::float32 var_6;
    wp::vec_t<3, wp::float32> var_7;
    //---------
    // forward
    // def _shift_cost(pt: wp.vec3, cost0: float) -> wp.vec3:                                 <L 219>
    // return wp.vec3(pt[0] - cost0, pt[1], pt[2])                                            <L 220>
    var_1 = wp::extract(var_pt, var_0);
    var_2 = wp::sub(var_1, var_cost0);
    var_4 = wp::extract(var_pt, var_3);
    var_6 = wp::extract(var_pt, var_5);
    var_7 = wp::vec_t<3, wp::float32>(var_2, var_4, var_6);
    return var_7;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/solver.py:466
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _compute_efc_eval_pt_pyramidal_0(
    wp::int32 var_efcid,
    wp::float32 var_alpha,
    wp::int32 var_ne,
    wp::int32 var_nf,
    wp::float32 var_efc_D,
    wp::array_t<wp::float32> var_efc_frictionloss,
    wp::float32 var_ctx_Jaref,
    wp::float32 var_ctx_jv)
{
    //---------
    // primal vars
    wp::int32 var_0;
    bool var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::float32 var_5 = 0.0;
    bool var_6;
    const wp::float32 var_7 = 0.0;
    wp::float32 var_8;
    wp::float32 var_9;
    const wp::float32 var_10 = 0.0;
    bool var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::float32 var_13;
    const wp::float32 var_14 = 0.0;
    const wp::float32 var_15 = 0.0;
    wp::vec_t<3, wp::float32> var_16;
    bool var_17;
    wp::float32* var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::float32 var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::float32 var_27;
    const wp::float32 var_28 = 0.0;
    wp::vec_t<3, wp::float32> var_29;
    //---------
    // forward
    // def _compute_efc_eval_pt_pyramidal(                                                    <L 467>
    // if efcid >= ne + nf:                                                                   <L 483>
    var_0 = wp::add(var_ne, var_nf);
    var_1 = (var_efcid >= var_0);
    if (var_1) {
        // x = ctx_Jaref + alpha * ctx_jv                                                     <L 484>
        var_2 = wp::mul(var_alpha, var_ctx_jv);
        var_3 = wp::add(var_ctx_Jaref, var_2);
        // quad0 = _eval_pt_direct_cost_alpha_zero(ctx_Jaref, efc_D)                          <L 485>
        var_4 = _eval_pt_direct_cost_alpha_zero_0(var_ctx_Jaref, var_efc_D);
        // cost0 = wp.where(ctx_Jaref < 0.0, quad0, 0.0)                                      <L 486>
        var_6 = (var_ctx_Jaref < var_5);
        var_8 = wp::where(var_6, var_4, var_7);
        // offset = quad0 - cost0                                                             <L 489>
        var_9 = wp::sub(var_4, var_8);
        // if x < 0.0:                                                                        <L 490>
        var_11 = (var_3 < var_10);
        if (var_11) {
            // return _eval_pt_direct_shifted(ctx_Jaref, ctx_jv, efc_D, alpha, offset)        <L 491>
            var_12 = _eval_pt_direct_shifted_0(var_ctx_Jaref, var_ctx_jv, var_efc_D, var_alpha, var_9);
            return var_12;
        }
        // return wp.vec3(-cost0, 0.0, 0.0)                                                   <L 492>
        var_13 = wp::neg(var_8);
        var_16 = wp::vec_t<3, wp::float32>(var_13, var_14, var_15);
        return var_16;
    }
    // if efcid >= ne:                                                                        <L 495>
    var_17 = (var_efcid >= var_ne);
    if (var_17) {
        // f = efc_frictionloss[efcid]                                                        <L 496>
        var_18 = wp::address(var_efc_frictionloss, var_efcid);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // x = ctx_Jaref + alpha * ctx_jv                                                     <L 497>
        var_21 = wp::mul(var_alpha, var_ctx_jv);
        var_22 = wp::add(var_ctx_Jaref, var_21);
        // rf = math.safe_div(f, efc_D)                                                       <L 498>
        var_23 = safe_div_0(var_19, var_efc_D);
        // return _shift_cost(_eval_frictionloss_pt(x, f, rf, ctx_jv, efc_D), _eval_frictionloss_cost(ctx_Jaref, f, rf, efc_D))       <L 499>
        var_24 = _eval_frictionloss_pt_0(var_22, var_19, var_23, var_ctx_jv, var_efc_D);
        var_25 = _eval_frictionloss_cost_0(var_ctx_Jaref, var_19, var_23, var_efc_D);
        var_26 = _shift_cost_0(var_24, var_25);
        return var_26;
    }
    var_27 = wp::where(var_17, var_22, var_3);
    // return _eval_pt_direct_shifted(ctx_Jaref, ctx_jv, efc_D, alpha, 0.0)                   <L 502>
    var_29 = _eval_pt_direct_shifted_0(var_ctx_Jaref, var_ctx_jv, var_efc_D, var_alpha, var_28);
    return var_29;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/solver.py:192
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _eval_pt_0(
    wp::vec_t<3, wp::float32> var_quad,
    wp::float32 var_alpha)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 2;
    wp::float32 var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    const wp::int32 var_4 = 1;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    const wp::int32 var_8 = 0;
    wp::float32 var_9;
    wp::float32 var_10;
    const wp::float32 var_11 = 2.0;
    wp::float32 var_12;
    const wp::int32 var_13 = 1;
    wp::float32 var_14;
    wp::float32 var_15;
    const wp::float32 var_16 = 2.0;
    const wp::int32 var_17 = 2;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::vec_t<3, wp::float32> var_20;
    //---------
    // forward
    // def _eval_pt(quad: wp.vec3, alpha: float) -> wp.vec3:                                  <L 193>
    // aq2 = alpha * quad[2]                                                                  <L 195>
    var_1 = wp::extract(var_quad, var_0);
    var_2 = wp::mul(var_alpha, var_1);
    // return wp.vec3(                                                                        <L 196>
    // alpha * aq2 + alpha * quad[1] + quad[0],                                               <L 197>
    var_3 = wp::mul(var_alpha, var_2);
    var_5 = wp::extract(var_quad, var_4);
    var_6 = wp::mul(var_alpha, var_5);
    var_7 = wp::add(var_3, var_6);
    var_9 = wp::extract(var_quad, var_8);
    var_10 = wp::add(var_7, var_9);
    // 2.0 * aq2 + quad[1],                                                                   <L 198>
    var_12 = wp::mul(var_11, var_2);
    var_14 = wp::extract(var_quad, var_13);
    var_15 = wp::add(var_12, var_14);
    // 2.0 * quad[2],                                                                         <L 199>
    var_18 = wp::extract(var_quad, var_17);
    var_19 = wp::mul(var_16, var_18);
    var_20 = wp::vec_t<3, wp::float32>(var_10, var_15, var_19);
    return var_20;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/solver.py:169
static CUDA_CALLABLE void _eval_pt_direct_shifted_3alphas_0(
    wp::float32 var_jaref,
    wp::float32 var_jv,
    wp::float32 var_d,
    wp::float32 var_lo_alpha,
    wp::float32 var_hi_alpha,
    wp::float32 var_mid_alpha,
    wp::float32 var_offset,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    const wp::float32 var_6 = 0.5;
    wp::float32 var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::vec_t<3, wp::float32> var_12;
    const wp::float32 var_13 = 0.5;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::vec_t<3, wp::float32> var_19;
    const wp::float32 var_20 = 0.5;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::vec_t<3, wp::float32> var_26;
    //---------
    // forward
    // def _eval_pt_direct_shifted_3alphas(                                                   <L 170>
    // jvD = jv * d                                                                           <L 174>
    var_0 = wp::mul(var_jv, var_d);
    // grad0 = jvD * jaref                                                                    <L 175>
    var_1 = wp::mul(var_0, var_jaref);
    // hessian = jv * jvD                                                                     <L 176>
    var_2 = wp::mul(var_jv, var_0);
    // lo_ah = lo_alpha * hessian                                                             <L 177>
    var_3 = wp::mul(var_lo_alpha, var_2);
    // hi_ah = hi_alpha * hessian                                                             <L 178>
    var_4 = wp::mul(var_hi_alpha, var_2);
    // mid_ah = mid_alpha * hessian                                                           <L 179>
    var_5 = wp::mul(var_mid_alpha, var_2);
    // return (                                                                               <L 180>
    // wp.vec3(lo_alpha * (grad0 + 0.5 * lo_ah) + offset, grad0 + lo_ah, hessian),            <L 181>
    var_7 = wp::mul(var_6, var_3);
    var_8 = wp::add(var_1, var_7);
    var_9 = wp::mul(var_lo_alpha, var_8);
    var_10 = wp::add(var_9, var_offset);
    var_11 = wp::add(var_1, var_3);
    var_12 = wp::vec_t<3, wp::float32>(var_10, var_11, var_2);
    // wp.vec3(hi_alpha * (grad0 + 0.5 * hi_ah) + offset, grad0 + hi_ah, hessian),            <L 182>
    var_14 = wp::mul(var_13, var_4);
    var_15 = wp::add(var_1, var_14);
    var_16 = wp::mul(var_hi_alpha, var_15);
    var_17 = wp::add(var_16, var_offset);
    var_18 = wp::add(var_1, var_4);
    var_19 = wp::vec_t<3, wp::float32>(var_17, var_18, var_2);
    // wp.vec3(mid_alpha * (grad0 + 0.5 * mid_ah) + offset, grad0 + mid_ah, hessian),         <L 183>
    var_21 = wp::mul(var_20, var_5);
    var_22 = wp::add(var_1, var_21);
    var_23 = wp::mul(var_mid_alpha, var_22);
    var_24 = wp::add(var_23, var_offset);
    var_25 = wp::add(var_1, var_5);
    var_26 = wp::vec_t<3, wp::float32>(var_24, var_25, var_2);
    ret_0 = var_12;
    ret_1 = var_19;
    ret_2 = var_26;
    return;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/solver.py:244
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _eval_frictionloss_pt_one_0(
    wp::float32 var_x,
    wp::float32 var_f,
    wp::float32 var_rf,
    wp::float32 var_half_d,
    wp::float32 var_jvD,
    wp::float32 var_hessian,
    wp::float32 var_f_jv)
{
    //---------
    // primal vars
    bool var_0;
    wp::float32 var_1;
    bool var_2;
    bool var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::float32 var_8;
    bool var_9;
    const wp::float32 var_10 = -0.5;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    const wp::float32 var_15 = 0.0;
    wp::vec_t<3, wp::float32> var_16;
    const wp::float32 var_17 = -0.5;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    const wp::float32 var_21 = 0.0;
    wp::vec_t<3, wp::float32> var_22;
    //---------
    // forward
    // def _eval_frictionloss_pt_one(x: float, f: float, rf: float, half_d: float, jvD: float, hessian: float, f_jv: float) -> wp.vec3:       <L 245>
    // if (-rf < x) and (x < rf):                                                             <L 247>
    var_1 = wp::neg(var_rf);
    var_2 = (var_1 < var_x);
    var_0 = var_2;
    if (var_0) {
        var_3 = (var_x < var_rf);
        var_0 = var_0 && var_3;
    }
    if (var_0) {
        // return wp.vec3(half_d * x * x, jvD * x, hessian)                                   <L 248>
        var_4 = wp::mul(var_half_d, var_x);
        var_5 = wp::mul(var_4, var_x);
        var_6 = wp::mul(var_jvD, var_x);
        var_7 = wp::vec_t<3, wp::float32>(var_5, var_6, var_hessian);
        return var_7;
    }
    if (!var_0) {
        // elif x <= -rf:                                                                     <L 249>
        var_8 = wp::neg(var_rf);
        var_9 = (var_x <= var_8);
        if (var_9) {
            // return wp.vec3(f * (-0.5 * rf - x), -f_jv, 0.0)                                <L 250>
            var_11 = wp::mul(var_10, var_rf);
            var_12 = wp::sub(var_11, var_x);
            var_13 = wp::mul(var_f, var_12);
            var_14 = wp::neg(var_f_jv);
            var_16 = wp::vec_t<3, wp::float32>(var_13, var_14, var_15);
            return var_16;
        }
        if (!var_9) {
            // return wp.vec3(f * (-0.5 * rf + x), f_jv, 0.0)                                 <L 252>
            var_18 = wp::mul(var_17, var_rf);
            var_19 = wp::add(var_18, var_x);
            var_20 = wp::mul(var_f, var_19);
            var_22 = wp::vec_t<3, wp::float32>(var_20, var_f_jv, var_21);
            return var_22;
        }
    }
    return {};
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/solver.py:255
static CUDA_CALLABLE void _eval_frictionloss_pt_3alphas_0(
    wp::float32 var_x_lo,
    wp::float32 var_x_hi,
    wp::float32 var_x_mid,
    wp::float32 var_f,
    wp::float32 var_rf,
    wp::float32 var_jv,
    wp::float32 var_d,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::float32 var_0;
    const wp::float32 var_1 = 0.5;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    //---------
    // forward
    // def _eval_frictionloss_pt_3alphas(                                                     <L 256>
    // jvD = jv * d                                                                           <L 260>
    var_0 = wp::mul(var_jv, var_d);
    // half_d = 0.5 * d                                                                       <L 261>
    var_2 = wp::mul(var_1, var_d);
    // hessian = jv * jvD                                                                     <L 262>
    var_3 = wp::mul(var_jv, var_0);
    // f_jv = f * jv                                                                          <L 263>
    var_4 = wp::mul(var_f, var_jv);
    // return (                                                                               <L 264>
    // _eval_frictionloss_pt_one(x_lo, f, rf, half_d, jvD, hessian, f_jv),                    <L 265>
    var_5 = _eval_frictionloss_pt_one_0(var_x_lo, var_f, var_rf, var_2, var_0, var_3, var_4);
    // _eval_frictionloss_pt_one(x_hi, f, rf, half_d, jvD, hessian, f_jv),                    <L 266>
    var_6 = _eval_frictionloss_pt_one_0(var_x_hi, var_f, var_rf, var_2, var_0, var_3, var_4);
    // _eval_frictionloss_pt_one(x_mid, f, rf, half_d, jvD, hessian, f_jv),                   <L 267>
    var_7 = _eval_frictionloss_pt_one_0(var_x_mid, var_f, var_rf, var_2, var_0, var_3, var_4);
    ret_0 = var_5;
    ret_1 = var_6;
    ret_2 = var_7;
    return;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/solver.py:634
static CUDA_CALLABLE void _compute_efc_eval_pt_3alphas_pyramidal_0(
    wp::int32 var_efcid,
    wp::float32 var_lo_alpha,
    wp::float32 var_hi_alpha,
    wp::float32 var_mid_alpha,
    wp::int32 var_ne,
    wp::int32 var_nf,
    wp::float32 var_efc_D,
    wp::array_t<wp::float32> var_efc_frictionloss,
    wp::float32 var_ctx_Jaref,
    wp::float32 var_ctx_jv,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::int32 var_0;
    bool var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    const wp::float32 var_9 = 0.0;
    bool var_10;
    const wp::float32 var_11 = 0.0;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::vec_t<3, wp::float32> var_16;
    wp::float32 var_17;
    const wp::float32 var_18 = 0.0;
    const wp::float32 var_19 = 0.0;
    wp::vec_t<3, wp::float32> var_20;
    const wp::float32 var_21 = 0.0;
    bool var_22;
    wp::vec_t<3, wp::float32> var_23;
    const wp::float32 var_24 = 0.0;
    bool var_25;
    wp::vec_t<3, wp::float32> var_26;
    const wp::float32 var_27 = 0.0;
    bool var_28;
    wp::vec_t<3, wp::float32> var_29;
    bool var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    wp::float32 var_35;
    wp::float32 var_36;
    wp::float32* var_37;
    wp::float32 var_38;
    wp::float32 var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::vec_t<3, wp::float32> var_42;
    wp::vec_t<3, wp::float32> var_43;
    wp::vec_t<3, wp::float32> var_44;
    wp::vec_t<3, wp::float32> var_45;
    wp::vec_t<3, wp::float32> var_46;
    wp::vec_t<3, wp::float32> var_47;
    wp::float32 var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::float32 var_51;
    const wp::float32 var_52 = 0.0;
    wp::vec_t<3, wp::float32> var_53;
    wp::vec_t<3, wp::float32> var_54;
    wp::vec_t<3, wp::float32> var_55;
    //---------
    // forward
    // def _compute_efc_eval_pt_3alphas_pyramidal(                                            <L 635>
    // if efcid >= ne + nf:                                                                   <L 654>
    var_0 = wp::add(var_ne, var_nf);
    var_1 = (var_efcid >= var_0);
    if (var_1) {
        // x_lo = ctx_Jaref + lo_alpha * ctx_jv                                               <L 655>
        var_2 = wp::mul(var_lo_alpha, var_ctx_jv);
        var_3 = wp::add(var_ctx_Jaref, var_2);
        // x_hi = ctx_Jaref + hi_alpha * ctx_jv                                               <L 656>
        var_4 = wp::mul(var_hi_alpha, var_ctx_jv);
        var_5 = wp::add(var_ctx_Jaref, var_4);
        // x_mid = ctx_Jaref + mid_alpha * ctx_jv                                             <L 657>
        var_6 = wp::mul(var_mid_alpha, var_ctx_jv);
        var_7 = wp::add(var_ctx_Jaref, var_6);
        // quad0 = _eval_pt_direct_cost_alpha_zero(ctx_Jaref, efc_D)                          <L 658>
        var_8 = _eval_pt_direct_cost_alpha_zero_0(var_ctx_Jaref, var_efc_D);
        // cost0 = wp.where(ctx_Jaref < 0.0, quad0, 0.0)                                      <L 659>
        var_10 = (var_ctx_Jaref < var_9);
        var_12 = wp::where(var_10, var_8, var_11);
        // offset = quad0 - cost0                                                             <L 661>
        var_13 = wp::sub(var_8, var_12);
        // pt_lo, pt_hi, pt_mid = _eval_pt_direct_shifted_3alphas(ctx_Jaref, ctx_jv, efc_D, lo_alpha, hi_alpha, mid_alpha, offset)       <L 662>
        _eval_pt_direct_shifted_3alphas_0(var_ctx_Jaref, var_ctx_jv, var_efc_D, var_lo_alpha, var_hi_alpha, var_mid_alpha, var_13, var_14, var_15, var_16);
        // inactive = wp.vec3(-cost0, 0.0, 0.0)                                               <L 663>
        var_17 = wp::neg(var_12);
        var_20 = wp::vec_t<3, wp::float32>(var_17, var_18, var_19);
        // return (                                                                           <L 664>
        // wp.where(x_lo < 0.0, pt_lo, inactive),                                             <L 665>
        var_22 = (var_3 < var_21);
        var_23 = wp::where(var_22, var_14, var_20);
        // wp.where(x_hi < 0.0, pt_hi, inactive),                                             <L 666>
        var_25 = (var_5 < var_24);
        var_26 = wp::where(var_25, var_15, var_20);
        // wp.where(x_mid < 0.0, pt_mid, inactive),                                           <L 667>
        var_28 = (var_7 < var_27);
        var_29 = wp::where(var_28, var_16, var_20);
        ret_0 = var_23;
        ret_1 = var_26;
        ret_2 = var_29;
        return;
    }
    // if efcid >= ne:                                                                        <L 671>
    var_30 = (var_efcid >= var_ne);
    if (var_30) {
        // x_lo = ctx_Jaref + lo_alpha * ctx_jv                                               <L 672>
        var_31 = wp::mul(var_lo_alpha, var_ctx_jv);
        var_32 = wp::add(var_ctx_Jaref, var_31);
        // x_hi = ctx_Jaref + hi_alpha * ctx_jv                                               <L 673>
        var_33 = wp::mul(var_hi_alpha, var_ctx_jv);
        var_34 = wp::add(var_ctx_Jaref, var_33);
        // x_mid = ctx_Jaref + mid_alpha * ctx_jv                                             <L 674>
        var_35 = wp::mul(var_mid_alpha, var_ctx_jv);
        var_36 = wp::add(var_ctx_Jaref, var_35);
        // f = efc_frictionloss[efcid]                                                        <L 675>
        var_37 = wp::address(var_efc_frictionloss, var_efcid);
        var_39 = wp::load(var_37);
        var_38 = wp::copy(var_39);
        // rf = math.safe_div(f, efc_D)                                                       <L 676>
        var_40 = safe_div_0(var_38, var_efc_D);
        // cost0 = _eval_frictionloss_cost(ctx_Jaref, f, rf, efc_D)                           <L 677>
        var_41 = _eval_frictionloss_cost_0(var_ctx_Jaref, var_38, var_40, var_efc_D);
        // lo, hi, mid = _eval_frictionloss_pt_3alphas(x_lo, x_hi, x_mid, f, rf, ctx_jv, efc_D)       <L 678>
        _eval_frictionloss_pt_3alphas_0(var_32, var_34, var_36, var_38, var_40, var_ctx_jv, var_efc_D, var_42, var_43, var_44);
        // return (_shift_cost(lo, cost0), _shift_cost(hi, cost0), _shift_cost(mid, cost0))       <L 679>
        var_45 = _shift_cost_0(var_42, var_41);
        var_46 = _shift_cost_0(var_43, var_41);
        var_47 = _shift_cost_0(var_44, var_41);
        ret_0 = var_45;
        ret_1 = var_46;
        ret_2 = var_47;
        return;
    }
    var_48 = wp::where(var_30, var_32, var_3);
    var_49 = wp::where(var_30, var_34, var_5);
    var_50 = wp::where(var_30, var_36, var_7);
    var_51 = wp::where(var_30, var_41, var_12);
    // return _eval_pt_direct_shifted_3alphas(ctx_Jaref, ctx_jv, efc_D, lo_alpha, hi_alpha, mid_alpha, 0.0)       <L 682>
    _eval_pt_direct_shifted_3alphas_0(var_ctx_Jaref, var_ctx_jv, var_efc_D, var_lo_alpha, var_hi_alpha, var_mid_alpha, var_52, var_53, var_54, var_55);
    ret_0 = var_53;
    ret_1 = var_54;
    ret_2 = var_55;
    return;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/solver.py:203
static CUDA_CALLABLE void _eval_pt_3alphas_0(
    wp::vec_t<3, wp::float32> var_quad,
    wp::float32 var_lo_alpha,
    wp::float32 var_hi_alpha,
    wp::float32 var_mid_alpha,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    const wp::int32 var_4 = 2;
    wp::float32 var_5;
    const wp::float32 var_6 = 2.0;
    wp::float32 var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    const wp::float32 var_15 = 2.0;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::vec_t<3, wp::float32> var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    const wp::float32 var_23 = 2.0;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::float32 var_30;
    const wp::float32 var_31 = 2.0;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::vec_t<3, wp::float32> var_34;
    //---------
    // forward
    // def _eval_pt_3alphas(quad: wp.vec3, lo_alpha: float, hi_alpha: float, mid_alpha: float) -> tuple[wp.vec3, wp.vec3, wp.vec3]:       <L 204>
    // q0, q1, q2 = quad[0], quad[1], quad[2]                                                 <L 206>
    var_1 = wp::extract(var_quad, var_0);
    var_3 = wp::extract(var_quad, var_2);
    var_5 = wp::extract(var_quad, var_4);
    // hessian = 2.0 * q2                                                                     <L 207>
    var_7 = wp::mul(var_6, var_5);
    // lo_aq2 = lo_alpha * q2                                                                 <L 208>
    var_8 = wp::mul(var_lo_alpha, var_5);
    // hi_aq2 = hi_alpha * q2                                                                 <L 209>
    var_9 = wp::mul(var_hi_alpha, var_5);
    // mid_aq2 = mid_alpha * q2                                                               <L 210>
    var_10 = wp::mul(var_mid_alpha, var_5);
    // return (                                                                               <L 211>
    // wp.vec3(lo_alpha * lo_aq2 + lo_alpha * q1 + q0, 2.0 * lo_aq2 + q1, hessian),           <L 212>
    var_11 = wp::mul(var_lo_alpha, var_8);
    var_12 = wp::mul(var_lo_alpha, var_3);
    var_13 = wp::add(var_11, var_12);
    var_14 = wp::add(var_13, var_1);
    var_16 = wp::mul(var_15, var_8);
    var_17 = wp::add(var_16, var_3);
    var_18 = wp::vec_t<3, wp::float32>(var_14, var_17, var_7);
    // wp.vec3(hi_alpha * hi_aq2 + hi_alpha * q1 + q0, 2.0 * hi_aq2 + q1, hessian),           <L 213>
    var_19 = wp::mul(var_hi_alpha, var_9);
    var_20 = wp::mul(var_hi_alpha, var_3);
    var_21 = wp::add(var_19, var_20);
    var_22 = wp::add(var_21, var_1);
    var_24 = wp::mul(var_23, var_9);
    var_25 = wp::add(var_24, var_3);
    var_26 = wp::vec_t<3, wp::float32>(var_22, var_25, var_7);
    // wp.vec3(mid_alpha * mid_aq2 + mid_alpha * q1 + q0, 2.0 * mid_aq2 + q1, hessian),       <L 214>
    var_27 = wp::mul(var_mid_alpha, var_10);
    var_28 = wp::mul(var_mid_alpha, var_3);
    var_29 = wp::add(var_27, var_28);
    var_30 = wp::add(var_29, var_1);
    var_32 = wp::mul(var_31, var_10);
    var_33 = wp::add(var_32, var_3);
    var_34 = wp::vec_t<3, wp::float32>(var_30, var_33, var_7);
    ret_0 = var_18;
    ret_1 = var_26;
    ret_2 = var_34;
    return;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/solver.py:117
static CUDA_CALLABLE bool _in_bracket_0(
    wp::vec_t<3, wp::float32> var_x,
    wp::vec_t<3, wp::float32> var_y)
{
    //---------
    // primal vars
    bool var_0;
    bool var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    const wp::int32 var_4 = 1;
    wp::float32 var_5;
    bool var_6;
    const wp::int32 var_7 = 1;
    wp::float32 var_8;
    const wp::float32 var_9 = 0.0;
    bool var_10;
    bool var_11;
    const wp::int32 var_12 = 1;
    wp::float32 var_13;
    const wp::int32 var_14 = 1;
    wp::float32 var_15;
    bool var_16;
    const wp::int32 var_17 = 1;
    wp::float32 var_18;
    const wp::float32 var_19 = 0.0;
    bool var_20;
    //---------
    // forward
    // def _in_bracket(x: wp.vec3, y: wp.vec3) -> bool:                                       <L 118>
    // return (x[1] < y[1] and y[1] < 0.0) or (x[1] > y[1] and y[1] > 0.0)                    <L 119>
    var_3 = wp::extract(var_x, var_2);
    var_5 = wp::extract(var_y, var_4);
    var_6 = (var_3 < var_5);
    var_1 = var_6;
    if (var_1) {
        var_8 = wp::extract(var_y, var_7);
        var_10 = (var_8 < var_9);
        var_1 = var_1 && var_10;
    }
    var_0 = var_1;
    if (!var_0) {
        var_13 = wp::extract(var_x, var_12);
        var_15 = wp::extract(var_y, var_14);
        var_16 = (var_13 > var_15);
        var_11 = var_16;
        if (var_11) {
            var_18 = wp::extract(var_y, var_17);
            var_20 = (var_18 > var_19);
            var_11 = var_11 && var_20;
        }
        var_0 = var_0 || var_11;
    }
    return var_0;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/solver.py:122
static CUDA_CALLABLE void adj__eval_pt_direct_alpha_zero_0(
    wp::float32 var_jaref,
    wp::float32 var_jv,
    wp::float32 var_d,
    wp::float32 & adj_jaref,
    wp::float32 & adj_jv,
    wp::float32 & adj_d,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
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


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/solver.py:223
static CUDA_CALLABLE void adj__eval_frictionloss_pt_0(
    wp::float32 var_x,
    wp::float32 var_f,
    wp::float32 var_rf,
    wp::float32 var_jv,
    wp::float32 var_d,
    wp::float32 & adj_x,
    wp::float32 & adj_f,
    wp::float32 & adj_rf,
    wp::float32 & adj_jv,
    wp::float32 & adj_d,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/solver.py:562
static CUDA_CALLABLE void adj__compute_efc_eval_pt_alpha_zero_pyramidal_0(
    wp::int32 var_efcid,
    wp::int32 var_ne,
    wp::int32 var_nf,
    wp::float32 var_efc_D,
    wp::array_t<wp::float32> var_efc_frictionloss,
    wp::float32 var_ctx_Jaref,
    wp::float32 var_ctx_jv,
    wp::int32 & adj_efcid,
    wp::int32 & adj_ne,
    wp::int32 & adj_nf,
    wp::float32 & adj_efc_D,
    wp::array_t<wp::float32> & adj_efc_frictionloss,
    wp::float32 & adj_ctx_Jaref,
    wp::float32 & adj_ctx_jv,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/solver.py:137
static CUDA_CALLABLE void adj__eval_pt_direct_cost_alpha_zero_0(
    wp::float32 var_jaref,
    wp::float32 var_d,
    wp::float32 & adj_jaref,
    wp::float32 & adj_d,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/solver.py:142
static CUDA_CALLABLE void adj__eval_pt_direct_shifted_0(
    wp::float32 var_jaref,
    wp::float32 var_jv,
    wp::float32 var_d,
    wp::float32 var_alpha,
    wp::float32 var_offset,
    wp::float32 & adj_jaref,
    wp::float32 & adj_jv,
    wp::float32 & adj_d,
    wp::float32 & adj_alpha,
    wp::float32 & adj_offset,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/solver.py:235
static CUDA_CALLABLE void adj__eval_frictionloss_cost_0(
    wp::float32 var_x,
    wp::float32 var_f,
    wp::float32 var_rf,
    wp::float32 var_d,
    wp::float32 & adj_x,
    wp::float32 & adj_f,
    wp::float32 & adj_rf,
    wp::float32 & adj_d,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/solver.py:218
static CUDA_CALLABLE void adj__shift_cost_0(
    wp::vec_t<3, wp::float32> var_pt,
    wp::float32 var_cost0,
    wp::vec_t<3, wp::float32> & adj_pt,
    wp::float32 & adj_cost0,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/solver.py:466
static CUDA_CALLABLE void adj__compute_efc_eval_pt_pyramidal_0(
    wp::int32 var_efcid,
    wp::float32 var_alpha,
    wp::int32 var_ne,
    wp::int32 var_nf,
    wp::float32 var_efc_D,
    wp::array_t<wp::float32> var_efc_frictionloss,
    wp::float32 var_ctx_Jaref,
    wp::float32 var_ctx_jv,
    wp::int32 & adj_efcid,
    wp::float32 & adj_alpha,
    wp::int32 & adj_ne,
    wp::int32 & adj_nf,
    wp::float32 & adj_efc_D,
    wp::array_t<wp::float32> & adj_efc_frictionloss,
    wp::float32 & adj_ctx_Jaref,
    wp::float32 & adj_ctx_jv,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/solver.py:192
static CUDA_CALLABLE void adj__eval_pt_0(
    wp::vec_t<3, wp::float32> var_quad,
    wp::float32 var_alpha,
    wp::vec_t<3, wp::float32> & adj_quad,
    wp::float32 & adj_alpha,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/solver.py:169
static CUDA_CALLABLE void adj__eval_pt_direct_shifted_3alphas_0(
    wp::float32 var_jaref,
    wp::float32 var_jv,
    wp::float32 var_d,
    wp::float32 var_lo_alpha,
    wp::float32 var_hi_alpha,
    wp::float32 var_mid_alpha,
    wp::float32 var_offset,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::float32 & adj_jaref,
    wp::float32 & adj_jv,
    wp::float32 & adj_d,
    wp::float32 & adj_lo_alpha,
    wp::float32 & adj_hi_alpha,
    wp::float32 & adj_mid_alpha,
    wp::float32 & adj_offset,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/solver.py:244
static CUDA_CALLABLE void adj__eval_frictionloss_pt_one_0(
    wp::float32 var_x,
    wp::float32 var_f,
    wp::float32 var_rf,
    wp::float32 var_half_d,
    wp::float32 var_jvD,
    wp::float32 var_hessian,
    wp::float32 var_f_jv,
    wp::float32 & adj_x,
    wp::float32 & adj_f,
    wp::float32 & adj_rf,
    wp::float32 & adj_half_d,
    wp::float32 & adj_jvD,
    wp::float32 & adj_hessian,
    wp::float32 & adj_f_jv,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/solver.py:255
static CUDA_CALLABLE void adj__eval_frictionloss_pt_3alphas_0(
    wp::float32 var_x_lo,
    wp::float32 var_x_hi,
    wp::float32 var_x_mid,
    wp::float32 var_f,
    wp::float32 var_rf,
    wp::float32 var_jv,
    wp::float32 var_d,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::float32 & adj_x_lo,
    wp::float32 & adj_x_hi,
    wp::float32 & adj_x_mid,
    wp::float32 & adj_f,
    wp::float32 & adj_rf,
    wp::float32 & adj_jv,
    wp::float32 & adj_d,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/solver.py:634
static CUDA_CALLABLE void adj__compute_efc_eval_pt_3alphas_pyramidal_0(
    wp::int32 var_efcid,
    wp::float32 var_lo_alpha,
    wp::float32 var_hi_alpha,
    wp::float32 var_mid_alpha,
    wp::int32 var_ne,
    wp::int32 var_nf,
    wp::float32 var_efc_D,
    wp::array_t<wp::float32> var_efc_frictionloss,
    wp::float32 var_ctx_Jaref,
    wp::float32 var_ctx_jv,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::int32 & adj_efcid,
    wp::float32 & adj_lo_alpha,
    wp::float32 & adj_hi_alpha,
    wp::float32 & adj_mid_alpha,
    wp::int32 & adj_ne,
    wp::int32 & adj_nf,
    wp::float32 & adj_efc_D,
    wp::array_t<wp::float32> & adj_efc_frictionloss,
    wp::float32 & adj_ctx_Jaref,
    wp::float32 & adj_ctx_jv,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/solver.py:203
static CUDA_CALLABLE void adj__eval_pt_3alphas_0(
    wp::vec_t<3, wp::float32> var_quad,
    wp::float32 var_lo_alpha,
    wp::float32 var_hi_alpha,
    wp::float32 var_mid_alpha,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & adj_quad,
    wp::float32 & adj_lo_alpha,
    wp::float32 & adj_hi_alpha,
    wp::float32 & adj_mid_alpha,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/solver.py:117
static CUDA_CALLABLE void adj__in_bracket_0(
    wp::vec_t<3, wp::float32> var_x,
    wp::vec_t<3, wp::float32> var_y,
    wp::vec_t<3, wp::float32> & adj_x,
    wp::vec_t<3, wp::float32> & adj_y,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void _linesearch_iterative_kernel__locals__kernel_28c5ca0b_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_nv,
    wp::array_t<wp::float32> var_opt_tolerance,
    wp::array_t<wp::float32> var_opt_ls_tolerance,
    wp::array_t<wp::float32> var_opt_impratio_invsqrt,
    wp::array_t<wp::float32> var_stat_meaninertia,
    wp::array_t<wp::int32> var_ne_in,
    wp::array_t<wp::int32> var_nf_in,
    wp::array_t<wp::int32> var_nefc_in,
    wp::array_t<wp::float32> var_qfrc_smooth_in,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_in,
    wp::array_t<wp::int32> var_contact_dim_in,
    wp::array_t<wp::int32> var_contact_efc_address_in,
    wp::array_t<wp::int32> var_efc_type_in,
    wp::array_t<wp::int32> var_efc_id_in,
    wp::array_t<wp::int32> var_efc_J_rownnz_in,
    wp::array_t<wp::int32> var_efc_J_rowadr_in,
    wp::array_t<wp::int32> var_efc_J_colind_in,
    wp::array_t<wp::float32> var_efc_J_in,
    wp::array_t<wp::float32> var_efc_D_in,
    wp::array_t<wp::float32> var_efc_frictionloss_in,
    wp::int32 var_njmax_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::array_t<bool> var_ctx_search_unchanged_in,
    wp::array_t<wp::float32> var_ctx_Jaref_in,
    wp::array_t<wp::float32> var_ctx_search_in,
    wp::array_t<wp::float32> var_ctx_search_dot_in,
    wp::array_t<wp::float32> var_ctx_mv_in,
    wp::array_t<wp::float32> var_ctx_jv_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ctx_quad_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_qacc_out,
    wp::array_t<wp::float32> var_efc_Ma_out,
    wp::array_t<wp::float32> var_ctx_Jaref_out,
    wp::array_t<wp::float32> var_ctx_jv_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ctx_quad_out,
    wp::array_t<wp::float32> var_ctx_improvement_out,
    wp::array_t<wp::float32> var_ctx_alpha_out,
    wp::array_t<bool> var_ctx_ls_exhausted_out)
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
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        const bool var_14 = false;
        const bool var_15 = false;
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
        wp::shape_t* var_35;
        const wp::int32 var_36 = 0;
        wp::int32 var_37;
        wp::shape_t var_38;
        wp::int32 var_39;
        wp::float32* var_40;
        wp::float32 var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::float32 var_44;
        wp::float32 var_45;
        wp::float32 var_46;
        wp::float32 var_47;
        const wp::float32 var_48 = 1e-06;
        wp::float32 var_49;
        const wp::float32 var_50 = 0.0;
        wp::vec_t<3, wp::float32> var_51;
        wp::int32 var_52;
        wp::range_t var_53;
        wp::int32 var_54;
        const bool var_55 = false;
        wp::float32* var_56;
        wp::slice_t var_57;
        const wp::int32 var_58 = 0;
        wp::array_t<wp::float32> var_59;
        wp::float32* var_60;
        wp::float32* var_61;
        wp::vec_t<3, wp::float32> var_62;
        wp::float32 var_63;
        wp::float32 var_64;
        wp::float32 var_65;
        wp::vec_t<3, wp::float32> var_66;
        const bool var_67 = true;
        wp::tile_register_t<wp::vec_t<3, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_68 = wp::tile_register_t<wp::vec_t<3, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        wp::tile_shared_t<wp::vec_t<3, wp::float32>,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_69 = wp::tile_alloc_empty<wp::vec_t<3, wp::float32>,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
        const wp::float32 var_70 = 0.0;
        wp::vec_t<2, wp::float32> var_71;
        wp::int32 var_72;
        wp::range_t var_73;
        wp::int32 var_74;
        wp::float32* var_75;
        wp::float32 var_76;
        wp::float32 var_77;
        wp::float32* var_78;
        wp::float32* var_79;
        wp::float32 var_80;
        wp::float32 var_81;
        wp::float32 var_82;
        wp::float32 var_83;
        const wp::float32 var_84 = 0.5;
        wp::float32 var_85;
        wp::float32* var_86;
        wp::float32 var_87;
        wp::float32 var_88;
        wp::vec_t<2, wp::float32> var_89;
        wp::vec_t<2, wp::float32> var_90;
        const bool var_91 = true;
        wp::tile_register_t<wp::vec_t<2, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_92 = wp::tile_register_t<wp::vec_t<2, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        wp::tile_shared_t<wp::vec_t<2, wp::float32>,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_93 = wp::tile_alloc_empty<wp::vec_t<2, wp::float32>,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
        const wp::int32 var_94 = 0;
        wp::vec_t<2, wp::float32> var_95;
        const wp::float32 var_96 = 0.0;
        const wp::int32 var_97 = 0;
        wp::float32 var_98;
        const wp::int32 var_99 = 1;
        wp::float32 var_100;
        wp::vec_t<3, wp::float32> var_101;
        const wp::int32 var_102 = 0;
        wp::float32 var_103;
        const wp::int32 var_104 = 1;
        wp::float32 var_105;
        const wp::float32 var_106 = 2.0;
        const wp::int32 var_107 = 2;
        wp::float32 var_108;
        wp::float32 var_109;
        wp::vec_t<3, wp::float32> var_110;
        const wp::int32 var_111 = 0;
        wp::vec_t<3, wp::float32> var_112;
        wp::vec_t<3, wp::float32> var_113;
        const wp::float32 var_114 = 0.0;
        const wp::int32 var_115 = 1;
        wp::float32 var_116;
        const wp::int32 var_117 = 2;
        wp::float32 var_118;
        wp::vec_t<3, wp::float32> var_119;
        const wp::float32 var_120 = 0.0;
        wp::float32 var_121;
        const bool var_122 = true;
        const wp::int32 var_123 = 0;
        wp::vec_t<3, wp::float32> var_124;
        const wp::float32 var_125 = 2.0;
        const wp::int32 var_126 = 0;
        wp::float32 var_127;
        const wp::float32 var_128 = 0.0;
        wp::float32 var_129;
        wp::float32 var_130;
        const wp::int32 var_131 = 2;
        wp::float32 var_132;
        const wp::float32 var_133 = 0.0;
        wp::float32 var_134;
        wp::float32 var_135;
        wp::float32 var_136;
        const wp::int32 var_137 = 1;
        wp::float32 var_138;
        wp::float32 var_139;
        wp::float32 var_140;
        const wp::float32 var_141 = 9.5367432e-07;
        const wp::float32 var_142 = 1.0;
        const wp::int32 var_143 = 2;
        wp::float32 var_144;
        wp::float32 var_145;
        wp::float32 var_146;
        wp::float32 var_147;
        const wp::int32 var_148 = 1;
        wp::float32 var_149;
        const wp::int32 var_150 = 2;
        wp::float32 var_151;
        wp::float32 var_152;
        wp::float32 var_153;
        const wp::float32 var_154 = 0.0;
        wp::vec_t<3, wp::float32> var_155;
        wp::int32 var_156;
        wp::range_t var_157;
        wp::int32 var_158;
        const bool var_159 = false;
        wp::float32* var_160;
        wp::slice_t var_161;
        const wp::int32 var_162 = 0;
        wp::array_t<wp::float32> var_163;
        wp::float32* var_164;
        wp::float32* var_165;
        wp::vec_t<3, wp::float32> var_166;
        wp::float32 var_167;
        wp::float32 var_168;
        wp::float32 var_169;
        wp::vec_t<3, wp::float32> var_170;
        const bool var_171 = true;
        wp::tile_register_t<wp::vec_t<3, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_172 = wp::tile_register_t<wp::vec_t<3, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        wp::tile_shared_t<wp::vec_t<3, wp::float32>,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_173 = wp::tile_alloc_empty<wp::vec_t<3, wp::float32>,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
        wp::vec_t<3, wp::float32> var_174;
        const wp::int32 var_175 = 0;
        wp::vec_t<3, wp::float32> var_176;
        wp::vec_t<3, wp::float32> var_177;
        bool var_178;
        const wp::int32 var_179 = 1;
        wp::float32 var_180;
        wp::float32 var_181;
        bool var_182;
        const wp::int32 var_183 = 0;
        wp::float32 var_184;
        const wp::float32 var_185 = 0.0;
        bool var_186;
        bool var_187;
        const wp::float32 var_188 = 0.0;
        wp::float32 var_189;
        const wp::float32 var_190 = 0.0;
        wp::float32 var_191;
        const wp::int32 var_192 = 1;
        wp::float32 var_193;
        const wp::int32 var_194 = 1;
        wp::float32 var_195;
        bool var_196;
        wp::vec_t<3, wp::float32> var_197;
        const wp::float32 var_198 = 0.0;
        wp::float32 var_199;
        wp::vec_t<3, wp::float32> var_200;
        const wp::float32 var_201 = 0.0;
        wp::float32 var_202;
        const wp::int32 var_203 = 20;
        wp::range_t var_204;
        wp::int32 var_205;
        const wp::int32 var_206 = 1;
        wp::float32 var_207;
        const wp::int32 var_208 = 2;
        wp::float32 var_209;
        wp::float32 var_210;
        wp::float32 var_211;
        const wp::int32 var_212 = 1;
        wp::float32 var_213;
        const wp::int32 var_214 = 2;
        wp::float32 var_215;
        wp::float32 var_216;
        wp::float32 var_217;
        const wp::float32 var_218 = 0.5;
        wp::float32 var_219;
        wp::float32 var_220;
        const wp::float32 var_221 = 0.0;
        wp::vec_t<3, wp::float32> var_222;
        const wp::float32 var_223 = 0.0;
        wp::vec_t<3, wp::float32> var_224;
        const wp::float32 var_225 = 0.0;
        wp::vec_t<3, wp::float32> var_226;
        wp::int32 var_227;
        wp::range_t var_228;
        wp::int32 var_229;
        const bool var_230 = false;
        wp::float32* var_231;
        wp::slice_t var_232;
        const wp::int32 var_233 = 0;
        wp::array_t<wp::float32> var_234;
        wp::float32* var_235;
        wp::float32* var_236;
        wp::vec_t<3, wp::float32> var_237;
        wp::vec_t<3, wp::float32> var_238;
        wp::vec_t<3, wp::float32> var_239;
        wp::float32 var_240;
        wp::float32 var_241;
        wp::float32 var_242;
        wp::vec_t<3, wp::float32> var_243;
        wp::vec_t<3, wp::float32> var_244;
        wp::vec_t<3, wp::float32> var_245;
        const wp::int32 var_246 = 0;
        wp::float32 var_247;
        const wp::int32 var_248 = 0;
        wp::float32 var_249;
        const wp::int32 var_250 = 0;
        wp::float32 var_251;
        const wp::int32 var_252 = 1;
        wp::float32 var_253;
        const wp::int32 var_254 = 1;
        wp::float32 var_255;
        const wp::int32 var_256 = 1;
        wp::float32 var_257;
        const wp::int32 var_258 = 2;
        wp::float32 var_259;
        const wp::int32 var_260 = 2;
        wp::float32 var_261;
        const wp::int32 var_262 = 2;
        wp::float32 var_263;
        wp::mat_t<3, 3, wp::float32> var_264;
        const bool var_265 = true;
        wp::tile_register_t<wp::mat_t<3, 3, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_266 = wp::tile_register_t<wp::mat_t<3, 3, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        wp::tile_shared_t<wp::mat_t<3, 3, wp::float32>,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_267 = wp::tile_alloc_empty<wp::mat_t<3, 3, wp::float32>,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
        const wp::int32 var_268 = 0;
        wp::mat_t<3, 3, wp::float32> var_269;
        wp::vec_t<3, wp::float32> var_270;
        wp::vec_t<3, wp::float32> var_271;
        wp::vec_t<3, wp::float32> var_272;
        const wp::int32 var_273 = 0;
        const wp::int32 var_274 = 0;
        wp::float32 var_275;
        const wp::int32 var_276 = 1;
        const wp::int32 var_277 = 0;
        wp::float32 var_278;
        const wp::int32 var_279 = 2;
        const wp::int32 var_280 = 0;
        wp::float32 var_281;
        wp::vec_t<3, wp::float32> var_282;
        wp::vec_t<3, wp::float32> var_283;
        const wp::int32 var_284 = 0;
        const wp::int32 var_285 = 1;
        wp::float32 var_286;
        const wp::int32 var_287 = 1;
        const wp::int32 var_288 = 1;
        wp::float32 var_289;
        const wp::int32 var_290 = 2;
        const wp::int32 var_291 = 1;
        wp::float32 var_292;
        wp::vec_t<3, wp::float32> var_293;
        wp::vec_t<3, wp::float32> var_294;
        const wp::int32 var_295 = 0;
        const wp::int32 var_296 = 2;
        wp::float32 var_297;
        const wp::int32 var_298 = 1;
        const wp::int32 var_299 = 2;
        wp::float32 var_300;
        const wp::int32 var_301 = 2;
        const wp::int32 var_302 = 2;
        wp::float32 var_303;
        wp::vec_t<3, wp::float32> var_304;
        wp::vec_t<3, wp::float32> var_305;
        bool var_306;
        wp::vec_t<3, wp::float32> var_307;
        wp::float32 var_308;
        bool var_309;
        wp::vec_t<3, wp::float32> var_310;
        wp::float32 var_311;
        bool var_312;
        wp::vec_t<3, wp::float32> var_313;
        wp::float32 var_314;
        bool var_315;
        bool var_316;
        wp::vec_t<3, wp::float32> var_317;
        wp::float32 var_318;
        bool var_319;
        wp::vec_t<3, wp::float32> var_320;
        wp::float32 var_321;
        bool var_322;
        wp::vec_t<3, wp::float32> var_323;
        wp::float32 var_324;
        bool var_325;
        bool var_326;
        bool var_327;
        bool var_328;
        bool var_329;
        bool var_330;
        const wp::int32 var_331 = 0;
        wp::float32 var_332;
        const wp::float32 var_333 = 0.0;
        bool var_334;
        const wp::int32 var_335 = 1;
        wp::float32 var_336;
        const wp::float32 var_337 = 0.0;
        bool var_338;
        const wp::int32 var_339 = 1;
        wp::float32 var_340;
        wp::float32 var_341;
        bool var_342;
        bool var_343;
        const wp::int32 var_344 = 0;
        wp::float32 var_345;
        const wp::float32 var_346 = 0.0;
        bool var_347;
        const wp::int32 var_348 = 1;
        wp::float32 var_349;
        const wp::float32 var_350 = 0.0;
        bool var_351;
        const wp::int32 var_352 = 1;
        wp::float32 var_353;
        bool var_354;
        bool var_355;
        const wp::int32 var_356 = 0;
        wp::float32 var_357;
        const wp::float32 var_358 = 0.0;
        bool var_359;
        const wp::int32 var_360 = 0;
        wp::float32 var_361;
        const wp::float32 var_362 = 0.0;
        bool var_363;
        const wp::int32 var_364 = 0;
        wp::float32 var_365;
        const wp::int32 var_366 = 0;
        wp::float32 var_367;
        bool var_368;
        wp::float32 var_369;
        const wp::int32 var_370 = 0;
        wp::float32 var_371;
        const wp::int32 var_372 = 0;
        wp::float32 var_373;
        wp::float32 var_374;
        wp::float32 var_375;
        wp::float32 var_376;
        wp::float32 var_377;
        wp::int32 var_378;
        wp::float32 var_379;
        wp::float32 var_380;
        wp::vec_t<3, wp::float32> var_381;
        wp::float32 var_382;
        wp::vec_t<3, wp::float32> var_383;
        wp::float32 var_384;
        wp::float32 var_385;
        const wp::int32 var_386 = 0;
        wp::float32 var_387;
        wp::float32 var_388;
        wp::float32 var_389;
        wp::float32 var_390;
        wp::int32 var_391;
        wp::range_t var_392;
        wp::int32 var_393;
        wp::float32* var_394;
        wp::float32 var_395;
        wp::float32 var_396;
        wp::float32 var_397;
        wp::float32* var_398;
        wp::float32 var_399;
        wp::float32 var_400;
        wp::float32 var_401;
        wp::int32 var_402;
        wp::range_t var_403;
        wp::int32 var_404;
        wp::float32* var_405;
        wp::float32 var_406;
        wp::float32 var_407;
        wp::float32 var_408;
        const wp::int32 var_409 = 0;
        bool var_410;
        const bool var_411 = true;
        wp::float32 var_412;
        bool var_413;
        //---------
        // forward
        // def kernel(                                                                            <L 855>
        // worldid, tid = wp.tid()                                                                <L 900>
        builtin_tid2d(var_0, var_1);
        // if ctx_done_in[worldid]:                                                               <L 902>
        var_2 = wp::address(var_ctx_done_in, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // return                                                                             <L 903>
            continue;
        }
        var_4 = wp::load(var_2);
        // ne = ne_in[worldid]                                                                    <L 905>
        var_5 = wp::address(var_ne_in, var_0);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // nf = nf_in[worldid]                                                                    <L 906>
        var_8 = wp::address(var_nf_in, var_0);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // nefc = wp.min(njmax_in, nefc_in[worldid])                                              <L 907>
        var_11 = wp::address(var_nefc_in, var_0);
        var_13 = wp::load(var_11);
        var_12 = wp::min(var_njmax_in, var_13);
        // if wp.static(FUSE_JV):                                                                 <L 911>
        // if wp.static(IS_ELLIPTIC):                                                             <L 935>
        // tolerance = opt_tolerance[worldid % opt_tolerance.shape[0]]                            <L 996>
        var_16 = &(var_opt_tolerance.shape);
        var_19 = wp::load(var_16);
        var_18 = wp::extract(var_19, var_17);
        var_20 = wp::mod(var_0, var_18);
        var_21 = wp::address(var_opt_tolerance, var_20);
        var_23 = wp::load(var_21);
        var_22 = wp::copy(var_23);
        // ls_tolerance = opt_ls_tolerance[worldid % opt_ls_tolerance.shape[0]]                   <L 997>
        var_24 = &(var_opt_ls_tolerance.shape);
        var_27 = wp::load(var_24);
        var_26 = wp::extract(var_27, var_25);
        var_28 = wp::mod(var_0, var_26);
        var_29 = wp::address(var_opt_ls_tolerance, var_28);
        var_31 = wp::load(var_29);
        var_30 = wp::copy(var_31);
        // snorm = wp.sqrt(ctx_search_dot_in[worldid])                                            <L 998>
        var_32 = wp::address(var_ctx_search_dot_in, var_0);
        var_34 = wp::load(var_32);
        var_33 = wp::sqrt(var_34);
        // meaninertia = stat_meaninertia[worldid % stat_meaninertia.shape[0]]                    <L 999>
        var_35 = &(var_stat_meaninertia.shape);
        var_38 = wp::load(var_35);
        var_37 = wp::extract(var_38, var_36);
        var_39 = wp::mod(var_0, var_37);
        var_40 = wp::address(var_stat_meaninertia, var_39);
        var_42 = wp::load(var_40);
        var_41 = wp::copy(var_42);
        // scale = meaninertia * wp.float(nv)                                                     <L 1000>
        var_43 = wp::float(var_nv);
        var_44 = wp::mul(var_41, var_43);
        // gtol = wp.max(tolerance * ls_tolerance * snorm * scale, 1e-6)                          <L 1001>
        var_45 = wp::mul(var_22, var_30);
        var_46 = wp::mul(var_45, var_33);
        var_47 = wp::mul(var_46, var_44);
        var_49 = wp::max(var_47, var_48);
        // local_p0 = wp.vec3(0.0)                                                                <L 1004>
        var_51 = wp::vec_t<3, wp::float32>(var_50);
        // for efcid in range(tid, nefc, wp.block_dim()):                                         <L 1005>
        var_52 = builtin_block_dim();
        var_53 = wp::range(var_1, var_12, var_52);
        start_for_1:;
            if (iter_cmp(var_53) == 0) goto end_for_1;
            var_54 = wp::iter_next(var_53);
            // if wp.static(IS_ELLIPTIC):                                                         <L 1006>
            // local_p0 += _compute_efc_eval_pt_alpha_zero(                                       <L 1043>
            // efcid,                                                                             <L 1044>
            // ne,                                                                                <L 1045>
            // nf,                                                                                <L 1046>
            // efc_D_in[worldid, efcid],                                                          <L 1047>
            var_56 = wp::address(var_efc_D_in, var_0, var_54);
            // efc_frictionloss_in[worldid],                                                      <L 1048>
            var_57 = wp::slice_t(var_0, var_0, var_58);
            var_59 = wp::view(var_efc_frictionloss_in, var_57);
            // ctx_Jaref_in[worldid, efcid],                                                      <L 1049>
            var_60 = wp::address(var_ctx_Jaref_in, var_0, var_54);
            // ctx_jv_in[worldid, efcid],                                                         <L 1050>
            var_61 = wp::address(var_ctx_jv_in, var_0, var_54);
            var_63 = wp::load(var_56);
            var_64 = wp::load(var_60);
            var_65 = wp::load(var_61);
            var_62 = _compute_efc_eval_pt_alpha_zero_pyramidal_0(var_54, var_6, var_9, var_63, var_59, var_64, var_65);
            // local_p0 += _compute_efc_eval_pt_alpha_zero(                                       <L 1043>
            var_66 = wp::add(var_51, var_62);
            wp::assign(var_51, var_66);
            goto start_for_1;
        end_for_1:;
        // p0_tile = wp.tile(local_p0, preserve_type=True)                                        <L 1056>
        var_68 = wp::tile<wp::vec_t<3, wp::float32>>(var_51);
        // p0_sum = wp.tile_reduce(wp.add, p0_tile)                                               <L 1057>
        var_69 = wp::tile_reduce(wp::add, var_68);
        // local_gauss = wp.vec2(0.0)                                                             <L 1060>
        var_71 = wp::vec_t<2, wp::float32>(var_70);
        // for dofid in range(tid, nv, wp.block_dim()):                                           <L 1061>
        var_72 = builtin_block_dim();
        var_73 = wp::range(var_1, var_nv, var_72);
        start_for_3:;
            if (iter_cmp(var_73) == 0) goto end_for_3;
            var_74 = wp::iter_next(var_73);
            // search = ctx_search_in[worldid, dofid]                                             <L 1062>
            var_75 = wp::address(var_ctx_search_in, var_0, var_74);
            var_77 = wp::load(var_75);
            var_76 = wp::copy(var_77);
            // local_gauss += wp.vec2(                                                            <L 1063>
            // search * (efc_Ma_out[worldid, dofid] - qfrc_smooth_in[worldid, dofid]),            <L 1064>
            var_78 = wp::address(var_efc_Ma_out, var_0, var_74);
            var_79 = wp::address(var_qfrc_smooth_in, var_0, var_74);
            var_81 = wp::load(var_78);
            var_82 = wp::load(var_79);
            var_80 = wp::sub(var_81, var_82);
            var_83 = wp::mul(var_76, var_80);
            // 0.5 * search * ctx_mv_in[worldid, dofid],                                          <L 1065>
            var_85 = wp::mul(var_84, var_76);
            var_86 = wp::address(var_ctx_mv_in, var_0, var_74);
            var_88 = wp::load(var_86);
            var_87 = wp::mul(var_85, var_88);
            var_89 = wp::vec_t<2, wp::float32>(var_83, var_87);
            // local_gauss += wp.vec2(                                                            <L 1063>
            var_90 = wp::add(var_71, var_89);
            wp::assign(var_71, var_90);
            goto start_for_3;
        end_for_3:;
        // gauss_tile = wp.tile(local_gauss, preserve_type=True)                                  <L 1068>
        var_92 = wp::tile<wp::vec_t<2, wp::float32>>(var_71);
        // gauss_sum = wp.tile_reduce(wp.add, gauss_tile)                                         <L 1069>
        var_93 = wp::tile_reduce(wp::add, var_92);
        // gauss_reduced = gauss_sum[0]                                                           <L 1070>
        var_95 = wp::tile_extract(var_93, var_94);
        // ctx_quad_gauss = wp.vec3(0.0, gauss_reduced[0], gauss_reduced[1])                      <L 1071>
        var_98 = wp::extract(var_95, var_97);
        var_100 = wp::extract(var_95, var_99);
        var_101 = wp::vec_t<3, wp::float32>(var_96, var_98, var_100);
        // p0 = wp.vec3(ctx_quad_gauss[0], ctx_quad_gauss[1], 2.0 * ctx_quad_gauss[2]) + p0_sum[0]       <L 1074>
        var_103 = wp::extract(var_101, var_102);
        var_105 = wp::extract(var_101, var_104);
        var_108 = wp::extract(var_101, var_107);
        var_109 = wp::mul(var_106, var_108);
        var_110 = wp::vec_t<3, wp::float32>(var_103, var_105, var_109);
        var_112 = wp::tile_extract(var_69, var_111);
        var_113 = wp::add(var_110, var_112);
        // p0_delta = wp.vec3(0.0, p0[1], p0[2])                                                  <L 1075>
        var_116 = wp::extract(var_113, var_115);
        var_118 = wp::extract(var_113, var_117);
        var_119 = wp::vec_t<3, wp::float32>(var_114, var_116, var_118);
        // noise_floor = float(0.0)                                                               <L 1085>
        var_121 = wp::float(var_120);
        // if wp.static(INCREMENTAL):                                                             <L 1086>
        // rows = p0_sum[0]                                                                       <L 1087>
        var_124 = wp::tile_extract(var_69, var_123);
        // q1_abs = wp.sqrt(2.0 * wp.max(rows[0], 0.0) * wp.max(rows[2], 0.0)) + wp.abs(ctx_quad_gauss[1])       <L 1088>
        var_127 = wp::extract(var_124, var_126);
        var_129 = wp::max(var_127, var_128);
        var_130 = wp::mul(var_125, var_129);
        var_132 = wp::extract(var_124, var_131);
        var_134 = wp::max(var_132, var_133);
        var_135 = wp::mul(var_130, var_134);
        var_136 = wp::sqrt(var_135);
        var_138 = wp::extract(var_101, var_137);
        var_139 = wp::abs(var_138);
        var_140 = wp::add(var_136, var_139);
        // noise_floor = _ALPHA_NOISE_EPS * wp.max(1.0, math.safe_div(q1_abs, p0[2]))             <L 1089>
        var_144 = wp::extract(var_113, var_143);
        var_145 = safe_div_0(var_140, var_144);
        var_146 = wp::max(var_142, var_145);
        var_147 = wp::mul(var_141, var_146);
        // lo_alpha_in = -math.safe_div(p0[1], p0[2])                                             <L 1092>
        var_149 = wp::extract(var_113, var_148);
        var_151 = wp::extract(var_113, var_150);
        var_152 = safe_div_0(var_149, var_151);
        var_153 = wp::neg(var_152);
        // local_lo_in = wp.vec3(0.0)                                                             <L 1094>
        var_155 = wp::vec_t<3, wp::float32>(var_154);
        // for efcid in range(tid, nefc, wp.block_dim()):                                         <L 1095>
        var_156 = builtin_block_dim();
        var_157 = wp::range(var_1, var_12, var_156);
        start_for_5:;
            if (iter_cmp(var_157) == 0) goto end_for_5;
            var_158 = wp::iter_next(var_157);
            // if wp.static(IS_ELLIPTIC):                                                         <L 1096>
            // local_lo_in += _compute_efc_eval_pt(                                               <L 1134>
            // efcid,                                                                             <L 1135>
            // lo_alpha_in,                                                                       <L 1136>
            // ne,                                                                                <L 1137>
            // nf,                                                                                <L 1138>
            // efc_D_in[worldid, efcid],                                                          <L 1139>
            var_160 = wp::address(var_efc_D_in, var_0, var_158);
            // efc_frictionloss_in[worldid],                                                      <L 1140>
            var_161 = wp::slice_t(var_0, var_0, var_162);
            var_163 = wp::view(var_efc_frictionloss_in, var_161);
            // ctx_Jaref_in[worldid, efcid],                                                      <L 1141>
            var_164 = wp::address(var_ctx_Jaref_in, var_0, var_158);
            // ctx_jv_in[worldid, efcid],                                                         <L 1142>
            var_165 = wp::address(var_ctx_jv_in, var_0, var_158);
            var_167 = wp::load(var_160);
            var_168 = wp::load(var_164);
            var_169 = wp::load(var_165);
            var_166 = _compute_efc_eval_pt_pyramidal_0(var_158, var_153, var_6, var_9, var_167, var_163, var_168, var_169);
            // local_lo_in += _compute_efc_eval_pt(                                               <L 1134>
            var_170 = wp::add(var_155, var_166);
            wp::assign(var_155, var_170);
            goto start_for_5;
        end_for_5:;
        // lo_in_tile = wp.tile(local_lo_in, preserve_type=True)                                  <L 1145>
        var_172 = wp::tile<wp::vec_t<3, wp::float32>>(var_155);
        // lo_in_sum = wp.tile_reduce(wp.add, lo_in_tile)                                         <L 1146>
        var_173 = wp::tile_reduce(wp::add, var_172);
        // lo_in = _eval_pt(ctx_quad_gauss, lo_alpha_in) + lo_in_sum[0]                           <L 1147>
        var_174 = _eval_pt_0(var_101, var_153);
        var_176 = wp::tile_extract(var_173, var_175);
        var_177 = wp::add(var_174, var_176);
        // initial_converged = wp.abs(lo_in[1]) < gtol and lo_in[0] < 0.0                         <L 1150>
        var_180 = wp::extract(var_177, var_179);
        var_181 = wp::abs(var_180);
        var_182 = (var_181 < var_49);
        var_178 = var_182;
        if (var_178) {
            var_184 = wp::extract(var_177, var_183);
            var_186 = (var_184 < var_185);
            var_178 = var_178 && var_186;
        }
        // if not initial_converged:                                                              <L 1153>
        var_187 = wp::unot(var_178);
        if (var_187) {
            // alpha = float(0.0)                                                                 <L 1154>
            var_189 = wp::float(var_188);
            // improvement = float(0.0)                                                           <L 1155>
            var_191 = wp::float(var_190);
            // lo_less = lo_in[1] < p0[1]                                                         <L 1158>
            var_193 = wp::extract(var_177, var_192);
            var_195 = wp::extract(var_113, var_194);
            var_196 = (var_193 < var_195);
            // lo = wp.where(lo_less, lo_in, p0_delta)                                            <L 1159>
            var_197 = wp::where(var_196, var_177, var_119);
            // lo_alpha = wp.where(lo_less, lo_alpha_in, 0.0)                                     <L 1160>
            var_199 = wp::where(var_196, var_153, var_198);
            // hi = wp.where(lo_less, p0_delta, lo_in)                                            <L 1161>
            var_200 = wp::where(var_196, var_119, var_177);
            // hi_alpha = wp.where(lo_less, 0.0, lo_alpha_in)                                     <L 1162>
            var_202 = wp::where(var_196, var_201, var_153);
            // for _ in range(LS_ITERATIONS):                                                     <L 1164>
            var_204 = wp::range(var_203);
            start_for_7:;
                if (iter_cmp(var_204) == 0) goto end_for_7;
                var_205 = wp::iter_next(var_204);
                // lo_next_alpha = lo_alpha - math.safe_div(lo[1], lo[2])                         <L 1165>
                var_207 = wp::extract(var_197, var_206);
                var_209 = wp::extract(var_197, var_208);
                var_210 = safe_div_0(var_207, var_209);
                var_211 = wp::sub(var_199, var_210);
                // hi_next_alpha = hi_alpha - math.safe_div(hi[1], hi[2])                         <L 1166>
                var_213 = wp::extract(var_200, var_212);
                var_215 = wp::extract(var_200, var_214);
                var_216 = safe_div_0(var_213, var_215);
                var_217 = wp::sub(var_202, var_216);
                // mid_alpha = 0.5 * (lo_alpha + hi_alpha)                                        <L 1167>
                var_219 = wp::add(var_199, var_202);
                var_220 = wp::mul(var_218, var_219);
                // local_lo = wp.vec3(0.0)                                                        <L 1169>
                var_222 = wp::vec_t<3, wp::float32>(var_221);
                // local_hi = wp.vec3(0.0)                                                        <L 1170>
                var_224 = wp::vec_t<3, wp::float32>(var_223);
                // local_mid = wp.vec3(0.0)                                                       <L 1171>
                var_226 = wp::vec_t<3, wp::float32>(var_225);
                // for efcid in range(tid, nefc, wp.block_dim()):                                 <L 1173>
                var_227 = builtin_block_dim();
                var_228 = wp::range(var_1, var_12, var_227);
                start_for_9:;
                    if (iter_cmp(var_228) == 0) goto end_for_9;
                    var_229 = wp::iter_next(var_228);
                    // if wp.static(IS_ELLIPTIC):                                                 <L 1174>
                    // r_lo, r_hi, r_mid = _compute_efc_eval_pt_3alphas(                          <L 1214>
                    // efcid,                                                                     <L 1215>
                    // lo_next_alpha,                                                             <L 1216>
                    // hi_next_alpha,                                                             <L 1217>
                    // mid_alpha,                                                                 <L 1218>
                    // ne,                                                                        <L 1219>
                    // nf,                                                                        <L 1220>
                    // efc_D_in[worldid, efcid],                                                  <L 1221>
                    var_231 = wp::address(var_efc_D_in, var_0, var_229);
                    // efc_frictionloss_in[worldid],                                              <L 1222>
                    var_232 = wp::slice_t(var_0, var_0, var_233);
                    var_234 = wp::view(var_efc_frictionloss_in, var_232);
                    // ctx_Jaref_in[worldid, efcid],                                              <L 1223>
                    var_235 = wp::address(var_ctx_Jaref_in, var_0, var_229);
                    // ctx_jv_in[worldid, efcid],                                                 <L 1224>
                    var_236 = wp::address(var_ctx_jv_in, var_0, var_229);
                    var_240 = wp::load(var_231);
                    var_241 = wp::load(var_235);
                    var_242 = wp::load(var_236);
                    _compute_efc_eval_pt_3alphas_pyramidal_0(var_229, var_211, var_217, var_220, var_6, var_9, var_240, var_234, var_241, var_242, var_237, var_238, var_239);
                    // local_lo += r_lo                                                           <L 1226>
                    var_243 = wp::add(var_222, var_237);
                    // local_hi += r_hi                                                           <L 1227>
                    var_244 = wp::add(var_224, var_238);
                    // local_mid += r_mid                                                         <L 1228>
                    var_245 = wp::add(var_226, var_239);
                    wp::assign(var_222, var_243);
                    wp::assign(var_224, var_244);
                    wp::assign(var_226, var_245);
                    goto start_for_9;
                end_for_9:;
                // local_combined = wp.mat33(                                                     <L 1231>
                // local_lo[0],                                                                   <L 1232>
                var_247 = wp::extract(var_222, var_246);
                // local_hi[0],                                                                   <L 1233>
                var_249 = wp::extract(var_224, var_248);
                // local_mid[0],                                                                  <L 1234>
                var_251 = wp::extract(var_226, var_250);
                // local_lo[1],                                                                   <L 1235>
                var_253 = wp::extract(var_222, var_252);
                // local_hi[1],                                                                   <L 1236>
                var_255 = wp::extract(var_224, var_254);
                // local_mid[1],                                                                  <L 1237>
                var_257 = wp::extract(var_226, var_256);
                // local_lo[2],                                                                   <L 1238>
                var_259 = wp::extract(var_222, var_258);
                // local_hi[2],                                                                   <L 1239>
                var_261 = wp::extract(var_224, var_260);
                // local_mid[2],                                                                  <L 1240>
                var_263 = wp::extract(var_226, var_262);
                var_264 = wp::mat_t<3, 3, wp::float32>(var_247, var_249, var_251, var_253, var_255, var_257, var_259, var_261, var_263);
                // combined_tile = wp.tile(local_combined, preserve_type=True)                    <L 1245>
                var_266 = wp::tile<wp::mat_t<3, 3, wp::float32>>(var_264);
                // combined_sum = wp.tile_reduce(wp.add, combined_tile)                           <L 1246>
                var_267 = wp::tile_reduce(wp::add, var_266);
                // result = combined_sum[0]                                                       <L 1247>
                var_269 = wp::tile_extract(var_267, var_268);
                // gauss_lo, gauss_hi, gauss_mid = _eval_pt_3alphas(ctx_quad_gauss, lo_next_alpha, hi_next_alpha, mid_alpha)       <L 1250>
                _eval_pt_3alphas_0(var_101, var_211, var_217, var_220, var_270, var_271, var_272);
                // lo_next = gauss_lo + wp.vec3(result[0, 0], result[1, 0], result[2, 0])         <L 1251>
                var_275 = wp::extract(var_269, var_273, var_274);
                var_278 = wp::extract(var_269, var_276, var_277);
                var_281 = wp::extract(var_269, var_279, var_280);
                var_282 = wp::vec_t<3, wp::float32>(var_275, var_278, var_281);
                var_283 = wp::add(var_270, var_282);
                // hi_next = gauss_hi + wp.vec3(result[0, 1], result[1, 1], result[2, 1])         <L 1252>
                var_286 = wp::extract(var_269, var_284, var_285);
                var_289 = wp::extract(var_269, var_287, var_288);
                var_292 = wp::extract(var_269, var_290, var_291);
                var_293 = wp::vec_t<3, wp::float32>(var_286, var_289, var_292);
                var_294 = wp::add(var_271, var_293);
                // mid = gauss_mid + wp.vec3(result[0, 2], result[1, 2], result[2, 2])            <L 1253>
                var_297 = wp::extract(var_269, var_295, var_296);
                var_300 = wp::extract(var_269, var_298, var_299);
                var_303 = wp::extract(var_269, var_301, var_302);
                var_304 = wp::vec_t<3, wp::float32>(var_297, var_300, var_303);
                var_305 = wp::add(var_272, var_304);
                // swap_lo_lo_next = _in_bracket(lo, lo_next)                                     <L 1257>
                var_306 = _in_bracket_0(var_197, var_283);
                // lo = wp.where(swap_lo_lo_next, lo_next, lo)                                    <L 1258>
                var_307 = wp::where(var_306, var_283, var_197);
                // lo_alpha = wp.where(swap_lo_lo_next, lo_next_alpha, lo_alpha)                  <L 1259>
                var_308 = wp::where(var_306, var_211, var_199);
                // swap_lo_mid = _in_bracket(lo, mid)                                             <L 1260>
                var_309 = _in_bracket_0(var_307, var_305);
                // lo = wp.where(swap_lo_mid, mid, lo)                                            <L 1261>
                var_310 = wp::where(var_309, var_305, var_307);
                // lo_alpha = wp.where(swap_lo_mid, mid_alpha, lo_alpha)                          <L 1262>
                var_311 = wp::where(var_309, var_220, var_308);
                // swap_lo_hi_next = _in_bracket(lo, hi_next)                                     <L 1263>
                var_312 = _in_bracket_0(var_310, var_294);
                // lo = wp.where(swap_lo_hi_next, hi_next, lo)                                    <L 1264>
                var_313 = wp::where(var_312, var_294, var_310);
                // lo_alpha = wp.where(swap_lo_hi_next, hi_next_alpha, lo_alpha)                  <L 1265>
                var_314 = wp::where(var_312, var_217, var_311);
                // swap_lo = swap_lo_lo_next or swap_lo_mid or swap_lo_hi_next                    <L 1266>
                var_315 = var_306;
                if (!var_315) {
                    var_315 = var_315 || var_309;
                }
                if (!var_315) {
                    var_315 = var_315 || var_312;
                }
                // swap_hi_hi_next = _in_bracket(hi, hi_next)                                     <L 1269>
                var_316 = _in_bracket_0(var_200, var_294);
                // hi = wp.where(swap_hi_hi_next, hi_next, hi)                                    <L 1270>
                var_317 = wp::where(var_316, var_294, var_200);
                // hi_alpha = wp.where(swap_hi_hi_next, hi_next_alpha, hi_alpha)                  <L 1271>
                var_318 = wp::where(var_316, var_217, var_202);
                // swap_hi_mid = _in_bracket(hi, mid)                                             <L 1272>
                var_319 = _in_bracket_0(var_317, var_305);
                // hi = wp.where(swap_hi_mid, mid, hi)                                            <L 1273>
                var_320 = wp::where(var_319, var_305, var_317);
                // hi_alpha = wp.where(swap_hi_mid, mid_alpha, hi_alpha)                          <L 1274>
                var_321 = wp::where(var_319, var_220, var_318);
                // swap_hi_lo_next = _in_bracket(hi, lo_next)                                     <L 1275>
                var_322 = _in_bracket_0(var_320, var_283);
                // hi = wp.where(swap_hi_lo_next, lo_next, hi)                                    <L 1276>
                var_323 = wp::where(var_322, var_283, var_320);
                // hi_alpha = wp.where(swap_hi_lo_next, lo_next_alpha, hi_alpha)                  <L 1277>
                var_324 = wp::where(var_322, var_211, var_321);
                // swap_hi = swap_hi_hi_next or swap_hi_mid or swap_hi_lo_next                    <L 1278>
                var_325 = var_316;
                if (!var_325) {
                    var_325 = var_325 || var_319;
                }
                if (!var_325) {
                    var_325 = var_325 || var_322;
                }
                // ls_done = (                                                                    <L 1281>
                // (not swap_lo and not swap_hi)                                                  <L 1282>
                var_328 = wp::unot(var_315);
                var_327 = var_328;
                if (var_327) {
                    var_329 = wp::unot(var_325);
                    var_327 = var_327 && var_329;
                }
                var_326 = var_327;
                if (!var_326) {
                    // or (lo[0] < 0.0 and lo[1] < 0.0 and lo[1] > -gtol)                         <L 1283>
                    var_332 = wp::extract(var_313, var_331);
                    var_334 = (var_332 < var_333);
                    var_330 = var_334;
                    if (var_330) {
                        var_336 = wp::extract(var_313, var_335);
                        var_338 = (var_336 < var_337);
                        var_330 = var_330 && var_338;
                    }
                    if (var_330) {
                        var_340 = wp::extract(var_313, var_339);
                        var_341 = wp::neg(var_49);
                        var_342 = (var_340 > var_341);
                        var_330 = var_330 && var_342;
                    }
                    var_326 = var_326 || var_330;
                }
                if (!var_326) {
                    // or (hi[0] < 0.0 and hi[1] > 0.0 and hi[1] < gtol)                          <L 1284>
                    var_345 = wp::extract(var_323, var_344);
                    var_347 = (var_345 < var_346);
                    var_343 = var_347;
                    if (var_343) {
                        var_349 = wp::extract(var_323, var_348);
                        var_351 = (var_349 > var_350);
                        var_343 = var_343 && var_351;
                    }
                    if (var_343) {
                        var_353 = wp::extract(var_323, var_352);
                        var_354 = (var_353 < var_49);
                        var_343 = var_343 && var_354;
                    }
                    var_326 = var_326 || var_343;
                }
                // improved = lo[0] < 0.0 or hi[0] < 0.0                                          <L 1288>
                var_357 = wp::extract(var_313, var_356);
                var_359 = (var_357 < var_358);
                var_355 = var_359;
                if (!var_355) {
                    var_361 = wp::extract(var_323, var_360);
                    var_363 = (var_361 < var_362);
                    var_355 = var_355 || var_363;
                }
                // lo_better = lo[0] < hi[0]                                                      <L 1289>
                var_365 = wp::extract(var_313, var_364);
                var_367 = wp::extract(var_323, var_366);
                var_368 = (var_365 < var_367);
                // best_alpha = wp.where(lo_better, lo_alpha, hi_alpha)                           <L 1290>
                var_369 = wp::where(var_368, var_314, var_324);
                // best_delta = wp.where(lo_better, lo[0], hi[0])                                 <L 1291>
                var_371 = wp::extract(var_313, var_370);
                var_373 = wp::extract(var_323, var_372);
                var_374 = wp::where(var_368, var_371, var_373);
                // alpha = wp.where(improved, best_alpha, alpha)                                  <L 1292>
                var_375 = wp::where(var_355, var_369, var_189);
                // improvement = wp.where(improved, -best_delta, improvement)                     <L 1293>
                var_376 = wp::neg(var_374);
                var_377 = wp::where(var_355, var_376, var_191);
                // if ls_done:                                                                    <L 1295>
                if (var_326) {
                    // break                                                                      <L 1296>
                    wp::assign(var_158, var_229);
                    wp::assign(var_189, var_375);
                    wp::assign(var_191, var_377);
                    wp::assign(var_197, var_313);
                    wp::assign(var_199, var_314);
                    wp::assign(var_200, var_323);
                    wp::assign(var_202, var_324);
                    goto end_for_7;
                }
                var_378 = wp::where(var_326, var_158, var_229);
                var_379 = wp::where(var_326, var_189, var_375);
                var_380 = wp::where(var_326, var_191, var_377);
                var_381 = wp::where(var_326, var_197, var_313);
                var_382 = wp::where(var_326, var_199, var_314);
                var_383 = wp::where(var_326, var_200, var_323);
                var_384 = wp::where(var_326, var_202, var_324);
                wp::assign(var_158, var_378);
                wp::assign(var_189, var_379);
                wp::assign(var_191, var_380);
                wp::assign(var_197, var_381);
                wp::assign(var_199, var_382);
                wp::assign(var_200, var_383);
                wp::assign(var_202, var_384);
                goto start_for_7;
            end_for_7:;
        }
        if (!var_187) {
            // alpha = lo_alpha_in                                                                <L 1298>
            var_385 = wp::copy(var_153);
            // improvement = -lo_in[0]                                                            <L 1299>
            var_387 = wp::extract(var_177, var_386);
            var_388 = wp::neg(var_387);
        }
        var_389 = wp::where(var_187, var_189, var_385);
        var_390 = wp::where(var_187, var_191, var_388);
        // for dofid in range(tid, nv, wp.block_dim()):                                           <L 1302>
        var_391 = builtin_block_dim();
        var_392 = wp::range(var_1, var_nv, var_391);
        start_for_11:;
            if (iter_cmp(var_392) == 0) goto end_for_11;
            var_393 = wp::iter_next(var_392);
            // qacc_out[worldid, dofid] += alpha * ctx_search_in[worldid, dofid]                  <L 1303>
            var_394 = wp::address(var_ctx_search_in, var_0, var_393);
            var_396 = wp::load(var_394);
            var_395 = wp::mul(var_389, var_396);
            var_397 = wp::atomic_add(var_qacc_out, var_0, var_393, var_395);
            // efc_Ma_out[worldid, dofid] += alpha * ctx_mv_in[worldid, dofid]                    <L 1304>
            var_398 = wp::address(var_ctx_mv_in, var_0, var_393);
            var_400 = wp::load(var_398);
            var_399 = wp::mul(var_389, var_400);
            var_401 = wp::atomic_add(var_efc_Ma_out, var_0, var_393, var_399);
            goto start_for_11;
        end_for_11:;
        // for efcid in range(tid, nefc, wp.block_dim()):                                         <L 1307>
        var_402 = builtin_block_dim();
        var_403 = wp::range(var_1, var_12, var_402);
        start_for_13:;
            if (iter_cmp(var_403) == 0) goto end_for_13;
            var_404 = wp::iter_next(var_403);
            // ctx_Jaref_out[worldid, efcid] += alpha * ctx_jv_in[worldid, efcid]                 <L 1308>
            var_405 = wp::address(var_ctx_jv_in, var_0, var_404);
            var_407 = wp::load(var_405);
            var_406 = wp::mul(var_389, var_407);
            var_408 = wp::atomic_add(var_ctx_Jaref_out, var_0, var_404, var_406);
            goto start_for_13;
        end_for_13:;
        // if tid == 0:                                                                           <L 1310>
        var_410 = (var_1 == var_409);
        if (var_410) {
            // ctx_improvement_out[worldid] = improvement                                         <L 1311>
            wp::array_store(var_ctx_improvement_out, var_0, var_390);
            // ctx_alpha_out[worldid] = alpha                                                     <L 1312>
            wp::array_store(var_ctx_alpha_out, var_0, var_389);
            // if wp.static(INCREMENTAL):                                                         <L 1313>
            // ctx_ls_exhausted_out[worldid] = wp.abs(alpha) < noise_floor                        <L 1314>
            var_412 = wp::abs(var_389);
            var_413 = (var_412 < var_147);
            wp::array_store(var_ctx_ls_exhausted_out, var_0, var_413);
        }
    }
}

