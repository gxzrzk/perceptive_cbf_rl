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


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/util_misc.py:603
static CUDA_CALLABLE wp::vec_t<6, wp::int32> dcmotor_slots_0(
    wp::vec_t<10, wp::float32> var_dynprm,
    wp::vec_t<10, wp::float32> var_gainprm)
{
    //---------
    // primal vars
    const wp::int32 var_0 = -1;
    const wp::int32 var_1 = -1;
    const wp::int32 var_2 = -1;
    const wp::int32 var_3 = -1;
    const wp::int32 var_4 = -1;
    const wp::int32 var_5 = 0;
    wp::vec_t<6, wp::int32> var_6;
    const wp::int32 var_7 = 0;
    const wp::int32 var_8 = 7;
    wp::float32 var_9;
    const wp::float32 var_10 = 0.0;
    bool var_11;
    const wp::int32 var_12 = 0;
    const wp::int32 var_13 = 1;
    wp::int32 var_14;
    wp::int32 var_15;
    const wp::int32 var_16 = 5;
    wp::float32 var_17;
    const wp::float32 var_18 = 0.0;
    bool var_19;
    const wp::int32 var_20 = 1;
    const wp::int32 var_21 = 1;
    wp::int32 var_22;
    wp::int32 var_23;
    const wp::int32 var_24 = 2;
    wp::float32 var_25;
    const wp::float32 var_26 = 0.0;
    bool var_27;
    const wp::int32 var_28 = 2;
    const wp::int32 var_29 = 1;
    wp::int32 var_30;
    wp::int32 var_31;
    const wp::int32 var_32 = 5;
    wp::float32 var_33;
    const wp::float32 var_34 = 0.0;
    bool var_35;
    const wp::int32 var_36 = 3;
    const wp::int32 var_37 = 1;
    wp::int32 var_38;
    wp::int32 var_39;
    const wp::int32 var_40 = 0;
    wp::float32 var_41;
    const wp::float32 var_42 = 0.0;
    bool var_43;
    const wp::int32 var_44 = 4;
    const wp::int32 var_45 = 1;
    wp::int32 var_46;
    wp::int32 var_47;
    const wp::int32 var_48 = 5;
    //---------
    // forward
    // def dcmotor_slots(dynprm: types.vec10, gainprm: types.vec10) -> types.vec6i:           <L 604>
    // s = types.vec6i(-1, -1, -1, -1, -1, 0)                                                 <L 622>
    var_6 = wp::vec_t<6, wp::int32>({var_0, var_1, var_2, var_3, var_4, var_5});
    // num_slots = 0                                                                          <L 623>
    // if dynprm[7] > 0.0:                                                                    <L 624>
    var_9 = wp::extract(var_dynprm, var_8);
    var_11 = (var_9 > var_10);
    if (var_11) {
        // s[0] = num_slots                                                                   <L 625>
        wp::assign_inplace(var_6, var_12, var_7);
        // num_slots += 1                                                                     <L 626>
        var_14 = wp::add(var_7, var_13);
    }
    var_15 = wp::where(var_11, var_14, var_7);
    // if gainprm[5] > 0.0:                                                                   <L 627>
    var_17 = wp::extract(var_gainprm, var_16);
    var_19 = (var_17 > var_18);
    if (var_19) {
        // s[1] = num_slots                                                                   <L 628>
        wp::assign_inplace(var_6, var_20, var_15);
        // num_slots += 1                                                                     <L 629>
        var_22 = wp::add(var_15, var_21);
    }
    var_23 = wp::where(var_19, var_22, var_15);
    // if dynprm[2] > 0.0:                                                                    <L 630>
    var_25 = wp::extract(var_dynprm, var_24);
    var_27 = (var_25 > var_26);
    if (var_27) {
        // s[2] = num_slots                                                                   <L 631>
        wp::assign_inplace(var_6, var_28, var_23);
        // num_slots += 1                                                                     <L 632>
        var_30 = wp::add(var_23, var_29);
    }
    var_31 = wp::where(var_27, var_30, var_23);
    // if dynprm[5] > 0.0:                                                                    <L 633>
    var_33 = wp::extract(var_dynprm, var_32);
    var_35 = (var_33 > var_34);
    if (var_35) {
        // s[3] = num_slots                                                                   <L 634>
        wp::assign_inplace(var_6, var_36, var_31);
        // num_slots += 1                                                                     <L 635>
        var_38 = wp::add(var_31, var_37);
    }
    var_39 = wp::where(var_35, var_38, var_31);
    // if dynprm[0] > 0.0:                                                                    <L 636>
    var_41 = wp::extract(var_dynprm, var_40);
    var_43 = (var_41 > var_42);
    if (var_43) {
        // s[4] = num_slots                                                                   <L 637>
        wp::assign_inplace(var_6, var_44, var_39);
        // num_slots += 1                                                                     <L 638>
        var_46 = wp::add(var_39, var_45);
    }
    var_47 = wp::where(var_43, var_46, var_39);
    // s[5] = num_slots                                                                       <L 639>
    wp::assign_inplace(var_6, var_48, var_47);
    // return s                                                                               <L 640>
    return var_6;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/util_misc.py:643
static CUDA_CALLABLE wp::float32 lugre_stribeck_0(
    wp::float32 var_velocity,
    wp::float32 var_F_C,
    wp::float32 var_F_S,
    wp::float32 var_v_S)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 1e-15;
    wp::float32 var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    //---------
    // forward
    // def lugre_stribeck(velocity: float, F_C: float, F_S: float, v_S: float) -> float:       <L 644>
    // ratio = velocity / wp.max(MJ_MINVAL, v_S)                                              <L 645>
    var_1 = wp::max(var_0, var_v_S);
    var_2 = wp::div(var_velocity, var_1);
    // return F_C + (F_S - F_C) * wp.exp(-ratio * ratio)                                      <L 646>
    var_3 = wp::sub(var_F_S, var_F_C);
    var_4 = wp::neg(var_2);
    var_5 = wp::mul(var_4, var_2);
    var_6 = wp::exp(var_5);
    var_7 = wp::mul(var_3, var_6);
    var_8 = wp::add(var_F_C, var_7);
    return var_8;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/support.py:37
static CUDA_CALLABLE wp::float32 next_act_0(
    wp::float32 var_opt_timestep,
    wp::int32 var_actuator_dyntype,
    wp::vec_t<10, wp::float32> var_actuator_dynprm,
    wp::vec_t<2, wp::float32> var_actuator_actrange,
    wp::float32 var_act_in,
    wp::float32 var_act_dot_in,
    wp::float32 var_act_dot_scale,
    bool var_clamp)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 3;
    bool var_1;
    const wp::float32 var_2 = 1e-15;
    const wp::int32 var_3 = 0;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    const wp::float32 var_8 = 1.0;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    const wp::int32 var_15 = 6;
    bool var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    wp::float32 var_21;
    const wp::int32 var_22 = 0;
    wp::float32 var_23;
    const wp::int32 var_24 = 1;
    wp::float32 var_25;
    wp::float32 var_26;
    wp::float32 var_27;
    //---------
    // forward
    // def next_act(                                                                          <L 38>
    // if actuator_dyntype == DynType.FILTEREXACT:                                            <L 52>
    var_1 = (var_actuator_dyntype == var_0);
    if (var_1) {
        // tau = wp.max(MJ_MINVAL, actuator_dynprm[0])                                        <L 53>
        var_4 = wp::extract(var_actuator_dynprm, var_3);
        var_5 = wp::max(var_2, var_4);
        // act = act_in + act_dot_scale * act_dot_in * tau * (1.0 - wp.exp(-opt_timestep / tau))       <L 54>
        var_6 = wp::mul(var_act_dot_scale, var_act_dot_in);
        var_7 = wp::mul(var_6, var_5);
        var_9 = wp::neg(var_opt_timestep);
        var_10 = wp::div(var_9, var_5);
        var_11 = wp::exp(var_10);
        var_12 = wp::sub(var_8, var_11);
        var_13 = wp::mul(var_7, var_12);
        var_14 = wp::add(var_act_in, var_13);
    }
    if (!var_1) {
        // elif actuator_dyntype == DynType.USER:                                             <L 55>
        var_16 = (var_actuator_dyntype == var_15);
        if (var_16) {
            // return act_in                                                                  <L 56>
            return var_act_in;
        }
        if (!var_16) {
            // act = act_in + act_dot_scale * act_dot_in * opt_timestep                       <L 58>
            var_17 = wp::mul(var_act_dot_scale, var_act_dot_in);
            var_18 = wp::mul(var_17, var_opt_timestep);
            var_19 = wp::add(var_act_in, var_18);
        }
        var_20 = wp::where(var_16, var_14, var_19);
    }
    var_21 = wp::where(var_1, var_14, var_20);
    // if clamp:                                                                              <L 61>
    if (var_clamp) {
        // act = wp.clamp(act, actuator_actrange[0], actuator_actrange[1])                    <L 62>
        var_23 = wp::extract(var_actuator_actrange, var_22);
        var_25 = wp::extract(var_actuator_actrange, var_24);
        var_26 = wp::clamp(var_21, var_23, var_25);
    }
    var_27 = wp::where(var_clamp, var_26, var_21);
    // return act                                                                             <L 64>
    return var_27;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/util_misc.py:554
static CUDA_CALLABLE wp::float32 _sigmoid_0(
    wp::float32 var_x)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    bool var_1;
    const wp::float32 var_2 = 0.0;
    const wp::float32 var_3 = 1.0;
    bool var_4;
    const wp::float32 var_5 = 1.0;
    wp::float32 var_6;
    wp::float32 var_7;
    const wp::float32 var_8 = 3.0;
    wp::float32 var_9;
    const wp::float32 var_10 = 2.0;
    wp::float32 var_11;
    const wp::float32 var_12 = 5.0;
    wp::float32 var_13;
    wp::float32 var_14;
    const wp::float32 var_15 = 10.0;
    wp::float32 var_16;
    wp::float32 var_17;
    //---------
    // forward
    // def _sigmoid(x: float) -> float:                                                       <L 555>
    // if x <= 0.0:                                                                           <L 557>
    var_1 = (var_x <= var_0);
    if (var_1) {
        // return 0.0                                                                         <L 558>
        return var_2;
    }
    // if x >= 1.0:                                                                           <L 560>
    var_4 = (var_x >= var_3);
    if (var_4) {
        // return 1.0                                                                         <L 561>
        return var_5;
    }
    // return x * x * x * (3.0 * x * (2.0 * x - 5.0) + 10.0)                                  <L 565>
    var_6 = wp::mul(var_x, var_x);
    var_7 = wp::mul(var_6, var_x);
    var_9 = wp::mul(var_8, var_x);
    var_11 = wp::mul(var_10, var_x);
    var_13 = wp::sub(var_11, var_12);
    var_14 = wp::mul(var_9, var_13);
    var_16 = wp::add(var_14, var_15);
    var_17 = wp::mul(var_7, var_16);
    return var_17;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/util_misc.py:568
static CUDA_CALLABLE wp::float32 muscle_dynamics_timescale_0(
    wp::float32 var_dctrl,
    wp::float32 var_tau_act,
    wp::float32 var_tau_deact,
    wp::float32 var_smooth_width)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 1e-15;
    bool var_1;
    const wp::float32 var_2 = 0.0;
    bool var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    const wp::float32 var_6 = 0.5;
    wp::float32 var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    //---------
    // forward
    // def muscle_dynamics_timescale(dctrl: float, tau_act: float, tau_deact: float, smooth_width: float) -> float:       <L 569>
    // if smooth_width < MJ_MINVAL:                                                           <L 572>
    var_1 = (var_smooth_width < var_0);
    if (var_1) {
        // if dctrl > 0.0:                                                                    <L 573>
        var_3 = (var_dctrl > var_2);
        if (var_3) {
            // return tau_act                                                                 <L 574>
            return var_tau_act;
        }
        if (!var_3) {
            // return tau_deact                                                               <L 576>
            return var_tau_deact;
        }
    }
    if (!var_1) {
        // return tau_deact + (tau_act - tau_deact) * _sigmoid(dctrl / smooth_width + 0.5)       <L 579>
        var_4 = wp::sub(var_tau_act, var_tau_deact);
        var_5 = wp::div(var_dctrl, var_smooth_width);
        var_7 = wp::add(var_5, var_6);
        var_8 = _sigmoid_0(var_7);
        var_9 = wp::mul(var_4, var_8);
        var_10 = wp::add(var_tau_deact, var_9);
        return var_10;
    }
    return {};
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/util_misc.py:582
static CUDA_CALLABLE wp::float32 muscle_dynamics_0(
    wp::float32 var_control,
    wp::float32 var_activation,
    wp::vec_t<10, wp::float32> var_prm)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    const wp::float32 var_1 = 1.0;
    wp::float32 var_2;
    const wp::float32 var_3 = 0.0;
    const wp::float32 var_4 = 1.0;
    wp::float32 var_5;
    const wp::int32 var_6 = 0;
    wp::float32 var_7;
    const wp::float32 var_8 = 0.5;
    const wp::float32 var_9 = 1.5;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::float32 var_12;
    const wp::int32 var_13 = 1;
    wp::float32 var_14;
    const wp::float32 var_15 = 0.5;
    const wp::float32 var_16 = 1.5;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    const wp::int32 var_20 = 2;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    const wp::float32 var_24 = 1e-15;
    wp::float32 var_25;
    wp::float32 var_26;
    //---------
    // forward
    // def muscle_dynamics(control: float, activation: float, prm: vec10) -> float:           <L 583>
    // ctrlclamp = wp.clamp(control, 0.0, 1.0)                                                <L 586>
    var_2 = wp::clamp(var_control, var_0, var_1);
    // actclamp = wp.clamp(activation, 0.0, 1.0)                                              <L 589>
    var_5 = wp::clamp(var_activation, var_3, var_4);
    // tau_act = prm[0] * (0.5 + 1.5 * actclamp)  # activation timescale                      <L 592>
    var_7 = wp::extract(var_prm, var_6);
    var_10 = wp::mul(var_9, var_5);
    var_11 = wp::add(var_8, var_10);
    var_12 = wp::mul(var_7, var_11);
    // tau_deact = prm[1] / (0.5 + 1.5 * actclamp)  # deactivation timescale                  <L 593>
    var_14 = wp::extract(var_prm, var_13);
    var_17 = wp::mul(var_16, var_5);
    var_18 = wp::add(var_15, var_17);
    var_19 = wp::div(var_14, var_18);
    // smooth_width = prm[2]  # width of smoothing sigmoid                                    <L 594>
    var_21 = wp::extract(var_prm, var_20);
    // dctrl = ctrlclamp - activation  # excess excitation                                    <L 595>
    var_22 = wp::sub(var_2, var_activation);
    // tau = muscle_dynamics_timescale(dctrl, tau_act, tau_deact, smooth_width)               <L 597>
    var_23 = muscle_dynamics_timescale_0(var_22, var_12, var_19, var_21);
    // return dctrl / wp.max(MJ_MINVAL, tau)                                                  <L 600>
    var_25 = wp::max(var_24, var_23);
    var_26 = wp::div(var_22, var_25);
    return var_26;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/util_misc.py:649
static CUDA_CALLABLE wp::float32 dcmotor_voltage_0(
    wp::float32 var_u,
    wp::float32 var_length,
    wp::float32 var_velocity,
    wp::float32 var_x_I,
    wp::vec_t<10, wp::float32> var_gainprm)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 8;
    wp::float32 var_1;
    wp::int32 var_2;
    const wp::int32 var_3 = 7;
    wp::float32 var_4;
    const wp::float32 var_5 = 0.0;
    const wp::int32 var_6 = 0;
    bool var_7;
    const wp::int32 var_8 = 4;
    wp::float32 var_9;
    const wp::int32 var_10 = 5;
    wp::float32 var_11;
    const wp::int32 var_12 = 6;
    wp::float32 var_13;
    const wp::int32 var_14 = 1;
    bool var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::float32 var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::float32 var_30;
    wp::float32 var_31;
    const wp::float32 var_32 = 0.0;
    bool var_33;
    wp::float32 var_34;
    wp::float32 var_35;
    wp::float32 var_36;
    //---------
    // forward
    // def dcmotor_voltage(u: float, length: float, velocity: float, x_I: float, gainprm: types.vec10) -> float:       <L 650>
    // input_mode = int(gainprm[8])                                                           <L 651>
    var_1 = wp::extract(var_gainprm, var_0);
    var_2 = wp::int(var_1);
    // Vmax = gainprm[7]                                                                      <L 652>
    var_4 = wp::extract(var_gainprm, var_3);
    // voltage = 0.0                                                                          <L 653>
    // if input_mode > 0:                                                                     <L 655>
    var_7 = (var_2 > var_6);
    if (var_7) {
        // kp = gainprm[4]                                                                    <L 656>
        var_9 = wp::extract(var_gainprm, var_8);
        // ki = gainprm[5]                                                                    <L 657>
        var_11 = wp::extract(var_gainprm, var_10);
        // kd = gainprm[6]                                                                    <L 658>
        var_13 = wp::extract(var_gainprm, var_12);
        // if input_mode == 1:                                                                <L 660>
        var_15 = (var_2 == var_14);
        if (var_15) {
            // voltage = kp * (u - length) + ki * x_I - kd * velocity                         <L 662>
            var_16 = wp::sub(var_u, var_length);
            var_17 = wp::mul(var_9, var_16);
            var_18 = wp::mul(var_11, var_x_I);
            var_19 = wp::add(var_17, var_18);
            var_20 = wp::mul(var_13, var_velocity);
            var_21 = wp::sub(var_19, var_20);
        }
        var_22 = wp::where(var_15, var_21, var_5);
        if (!var_15) {
            // voltage = kp * (u - velocity) + ki * (x_I - length)                            <L 665>
            var_23 = wp::sub(var_u, var_velocity);
            var_24 = wp::mul(var_9, var_23);
            var_25 = wp::sub(var_x_I, var_length);
            var_26 = wp::mul(var_11, var_25);
            var_27 = wp::add(var_24, var_26);
        }
        var_28 = wp::where(var_15, var_22, var_27);
    }
    var_29 = wp::where(var_7, var_28, var_5);
    if (!var_7) {
        // voltage = u                                                                        <L 667>
        var_30 = wp::copy(var_u);
    }
    var_31 = wp::where(var_7, var_29, var_30);
    // if Vmax > 0.0:                                                                         <L 669>
    var_33 = (var_4 > var_32);
    if (var_33) {
        // voltage = wp.clamp(voltage, -Vmax, Vmax)                                           <L 670>
        var_34 = wp::neg(var_4);
        var_35 = wp::clamp(var_31, var_34, var_4);
    }
    var_36 = wp::where(var_33, var_35, var_31);
    // return voltage                                                                         <L 672>
    return var_36;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/util_misc.py:454
static CUDA_CALLABLE wp::float32 muscle_gain_length_0(
    wp::float32 var_length,
    wp::float32 var_lmin,
    wp::float32 var_lmax)
{
    //---------
    // primal vars
    bool var_0;
    bool var_1;
    bool var_2;
    const wp::float32 var_3 = 0.0;
    const wp::float32 var_4 = 0.5;
    const wp::float32 var_5 = 1.0;
    wp::float32 var_6;
    wp::float32 var_7;
    const wp::float32 var_8 = 0.5;
    const wp::float32 var_9 = 1.0;
    wp::float32 var_10;
    wp::float32 var_11;
    bool var_12;
    wp::float32 var_13;
    const wp::float32 var_14 = 1e-15;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    const wp::float32 var_18 = 0.5;
    wp::float32 var_19;
    wp::float32 var_20;
    const wp::float32 var_21 = 1.0;
    bool var_22;
    const wp::float32 var_23 = 1.0;
    wp::float32 var_24;
    const wp::float32 var_25 = 1.0;
    wp::float32 var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    const wp::float32 var_29 = 1.0;
    const wp::float32 var_30 = 0.5;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    bool var_35;
    const wp::float32 var_36 = 1.0;
    wp::float32 var_37;
    const wp::float32 var_38 = 1.0;
    wp::float32 var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    const wp::float32 var_42 = 1.0;
    const wp::float32 var_43 = 0.5;
    wp::float32 var_44;
    wp::float32 var_45;
    wp::float32 var_46;
    wp::float32 var_47;
    wp::float32 var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::float32 var_51;
    const wp::float32 var_52 = 0.5;
    wp::float32 var_53;
    wp::float32 var_54;
    wp::float32 var_55;
    wp::float32 var_56;
    wp::float32 var_57;
    //---------
    // forward
    // def muscle_gain_length(length: float, lmin: float, lmax: float) -> float:              <L 455>
    // if (lmin > length) or (length > lmax):                                                 <L 457>
    var_1 = (var_lmin > var_length);
    var_0 = var_1;
    if (!var_0) {
        var_2 = (var_length > var_lmax);
        var_0 = var_0 || var_2;
    }
    if (var_0) {
        // return 0.0                                                                         <L 458>
        return var_3;
    }
    // a = 0.5 * (lmin + 1.0)                                                                 <L 461>
    var_6 = wp::add(var_lmin, var_5);
    var_7 = wp::mul(var_4, var_6);
    // b = 0.5 * (1.0 + lmax)                                                                 <L 462>
    var_10 = wp::add(var_9, var_lmax);
    var_11 = wp::mul(var_8, var_10);
    // if length <= a:                                                                        <L 464>
    var_12 = (var_length <= var_7);
    if (var_12) {
        // x = (length - lmin) / wp.max(MJ_MINVAL, a - lmin)                                  <L 465>
        var_13 = wp::sub(var_length, var_lmin);
        var_15 = wp::sub(var_7, var_lmin);
        var_16 = wp::max(var_14, var_15);
        var_17 = wp::div(var_13, var_16);
        // return 0.5 * x * x                                                                 <L 466>
        var_19 = wp::mul(var_18, var_17);
        var_20 = wp::mul(var_19, var_17);
        return var_20;
    }
    if (!var_12) {
        // elif length <= 1.0:                                                                <L 467>
        var_22 = (var_length <= var_21);
        if (var_22) {
            // x = (1.0 - length) / wp.max(MJ_MINVAL, 1.0 - a)                                <L 468>
            var_24 = wp::sub(var_23, var_length);
            var_26 = wp::sub(var_25, var_7);
            var_27 = wp::max(var_14, var_26);
            var_28 = wp::div(var_24, var_27);
            // return 1.0 - 0.5 * x * x                                                       <L 469>
            var_31 = wp::mul(var_30, var_28);
            var_32 = wp::mul(var_31, var_28);
            var_33 = wp::sub(var_29, var_32);
            return var_33;
        }
        var_34 = wp::where(var_22, var_28, var_17);
        if (!var_22) {
            // elif length <= b:                                                              <L 470>
            var_35 = (var_length <= var_11);
            if (var_35) {
                // x = (length - 1.0) / wp.max(MJ_MINVAL, b - 1.0)                            <L 471>
                var_37 = wp::sub(var_length, var_36);
                var_39 = wp::sub(var_11, var_38);
                var_40 = wp::max(var_14, var_39);
                var_41 = wp::div(var_37, var_40);
                // return 1.0 - 0.5 * x * x                                                   <L 472>
                var_44 = wp::mul(var_43, var_41);
                var_45 = wp::mul(var_44, var_41);
                var_46 = wp::sub(var_42, var_45);
                return var_46;
            }
            var_47 = wp::where(var_35, var_41, var_34);
            if (!var_35) {
                // x = (lmax - length) / wp.max(MJ_MINVAL, lmax - b)                          <L 474>
                var_48 = wp::sub(var_lmax, var_length);
                var_49 = wp::sub(var_lmax, var_11);
                var_50 = wp::max(var_14, var_49);
                var_51 = wp::div(var_48, var_50);
                // return 0.5 * x * x                                                         <L 475>
                var_53 = wp::mul(var_52, var_51);
                var_54 = wp::mul(var_53, var_51);
                return var_54;
            }
            var_55 = wp::where(var_35, var_47, var_51);
        }
        var_56 = wp::where(var_22, var_34, var_55);
    }
    var_57 = wp::where(var_12, var_17, var_56);
    return {};
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/util_misc.py:478
static CUDA_CALLABLE wp::float32 muscle_gain_0(
    wp::float32 var_len,
    wp::float32 var_vel,
    wp::vec_t<2, wp::float32> var_lengthrange,
    wp::float32 var_acc0,
    wp::vec_t<10, wp::float32> var_prm)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    wp::vec_t<2, wp::float32> var_4;
    const wp::int32 var_5 = 2;
    wp::float32 var_6;
    const wp::int32 var_7 = 3;
    wp::float32 var_8;
    const wp::int32 var_9 = 4;
    wp::float32 var_10;
    const wp::int32 var_11 = 5;
    wp::float32 var_12;
    const wp::int32 var_13 = 6;
    wp::float32 var_14;
    const wp::int32 var_15 = 8;
    wp::float32 var_16;
    const wp::float32 var_17 = 0.0;
    bool var_18;
    const wp::float32 var_19 = 1e-15;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    const wp::int32 var_23 = 1;
    wp::float32 var_24;
    const wp::int32 var_25 = 0;
    wp::float32 var_26;
    wp::float32 var_27;
    const wp::int32 var_28 = 1;
    wp::float32 var_29;
    const wp::int32 var_30 = 0;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    const wp::int32 var_35 = 0;
    wp::float32 var_36;
    const wp::int32 var_37 = 0;
    wp::float32 var_38;
    wp::float32 var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::float32 var_42;
    wp::float32 var_43;
    wp::float32 var_44;
    wp::float32 var_45;
    wp::float32 var_46;
    const wp::float32 var_47 = 1.0;
    wp::float32 var_48;
    const wp::float32 var_49 = -1.0;
    bool var_50;
    const wp::float32 var_51 = 0.0;
    const wp::float32 var_52 = 0.0;
    bool var_53;
    const wp::float32 var_54 = 1.0;
    wp::float32 var_55;
    const wp::float32 var_56 = 1.0;
    wp::float32 var_57;
    wp::float32 var_58;
    wp::float32 var_59;
    bool var_60;
    wp::float32 var_61;
    wp::float32 var_62;
    wp::float32 var_63;
    wp::float32 var_64;
    wp::float32 var_65;
    wp::float32 var_66;
    wp::float32 var_67;
    wp::float32 var_68;
    wp::float32 var_69;
    wp::float32 var_70;
    wp::float32 var_71;
    wp::float32 var_72;
    wp::float32 var_73;
    wp::float32 var_74;
    //---------
    // forward
    // def muscle_gain(len: float, vel: float, lengthrange: wp.vec2, acc0: float, prm: vec10) -> float:       <L 479>
    // range_ = wp.vec2(prm[0], prm[1])                                                       <L 482>
    var_1 = wp::extract(var_prm, var_0);
    var_3 = wp::extract(var_prm, var_2);
    var_4 = wp::vec_t<2, wp::float32>(var_1, var_3);
    // force = prm[2]                                                                         <L 483>
    var_6 = wp::extract(var_prm, var_5);
    // scale = prm[3]                                                                         <L 484>
    var_8 = wp::extract(var_prm, var_7);
    // lmin = prm[4]                                                                          <L 485>
    var_10 = wp::extract(var_prm, var_9);
    // lmax = prm[5]                                                                          <L 486>
    var_12 = wp::extract(var_prm, var_11);
    // vmax = prm[6]                                                                          <L 487>
    var_14 = wp::extract(var_prm, var_13);
    // fvmax = prm[8]                                                                         <L 488>
    var_16 = wp::extract(var_prm, var_15);
    // if force < 0.0:                                                                        <L 491>
    var_18 = (var_6 < var_17);
    if (var_18) {
        // force = scale / wp.max(MJ_MINVAL, acc0)                                            <L 492>
        var_20 = wp::max(var_19, var_acc0);
        var_21 = wp::div(var_8, var_20);
    }
    var_22 = wp::where(var_18, var_21, var_6);
    // L0 = (lengthrange[1] - lengthrange[0]) / wp.max(MJ_MINVAL, range_[1] - range_[0])       <L 495>
    var_24 = wp::extract(var_lengthrange, var_23);
    var_26 = wp::extract(var_lengthrange, var_25);
    var_27 = wp::sub(var_24, var_26);
    var_29 = wp::extract(var_4, var_28);
    var_31 = wp::extract(var_4, var_30);
    var_32 = wp::sub(var_29, var_31);
    var_33 = wp::max(var_19, var_32);
    var_34 = wp::div(var_27, var_33);
    // L = range_[0] + (len - lengthrange[0]) / wp.max(MJ_MINVAL, L0)                         <L 498>
    var_36 = wp::extract(var_4, var_35);
    var_38 = wp::extract(var_lengthrange, var_37);
    var_39 = wp::sub(var_len, var_38);
    var_40 = wp::max(var_19, var_34);
    var_41 = wp::div(var_39, var_40);
    var_42 = wp::add(var_36, var_41);
    // V = vel / wp.max(MJ_MINVAL, L0 * vmax)                                                 <L 499>
    var_43 = wp::mul(var_34, var_14);
    var_44 = wp::max(var_19, var_43);
    var_45 = wp::div(var_vel, var_44);
    // FL = muscle_gain_length(L, lmin, lmax)                                                 <L 502>
    var_46 = muscle_gain_length_0(var_42, var_10, var_12);
    // y = fvmax - 1.0                                                                        <L 505>
    var_48 = wp::sub(var_16, var_47);
    // if V <= -1.0:                                                                          <L 506>
    var_50 = (var_45 <= var_49);
    if (var_50) {
        // FV = 0.0                                                                           <L 507>
    }
    if (!var_50) {
        // elif V <= 0.0:                                                                     <L 508>
        var_53 = (var_45 <= var_52);
        if (var_53) {
            // FV = (V + 1.0) * (V + 1.0)                                                     <L 509>
            var_55 = wp::add(var_45, var_54);
            var_57 = wp::add(var_45, var_56);
            var_58 = wp::mul(var_55, var_57);
        }
        var_59 = wp::where(var_53, var_58, var_51);
        if (!var_53) {
            // elif V <= y:                                                                   <L 510>
            var_60 = (var_45 <= var_48);
            if (var_60) {
                // FV = fvmax - (y - V) * (y - V) / wp.max(MJ_MINVAL, y)                      <L 511>
                var_61 = wp::sub(var_48, var_45);
                var_62 = wp::sub(var_48, var_45);
                var_63 = wp::mul(var_61, var_62);
                var_64 = wp::max(var_19, var_48);
                var_65 = wp::div(var_63, var_64);
                var_66 = wp::sub(var_16, var_65);
            }
            var_67 = wp::where(var_60, var_66, var_59);
            if (!var_60) {
                // FV = fvmax                                                                 <L 513>
                var_68 = wp::copy(var_16);
            }
            var_69 = wp::where(var_60, var_67, var_68);
        }
        var_70 = wp::where(var_53, var_59, var_69);
    }
    var_71 = wp::where(var_50, var_51, var_70);
    // return -force * FL * FV                                                                <L 516>
    var_72 = wp::neg(var_22);
    var_73 = wp::mul(var_72, var_46);
    var_74 = wp::mul(var_73, var_71);
    return var_74;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/util_misc.py:519
static CUDA_CALLABLE wp::float32 muscle_bias_0(
    wp::float32 var_len,
    wp::vec_t<2, wp::float32> var_lengthrange,
    wp::float32 var_acc0,
    wp::vec_t<10, wp::float32> var_prm)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    wp::vec_t<2, wp::float32> var_4;
    const wp::int32 var_5 = 2;
    wp::float32 var_6;
    const wp::int32 var_7 = 3;
    wp::float32 var_8;
    const wp::int32 var_9 = 5;
    wp::float32 var_10;
    const wp::int32 var_11 = 7;
    wp::float32 var_12;
    const wp::float32 var_13 = 0.0;
    bool var_14;
    const wp::float32 var_15 = 1e-15;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    const wp::int32 var_19 = 1;
    wp::float32 var_20;
    const wp::int32 var_21 = 0;
    wp::float32 var_22;
    wp::float32 var_23;
    const wp::int32 var_24 = 1;
    wp::float32 var_25;
    const wp::int32 var_26 = 0;
    wp::float32 var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::float32 var_30;
    const wp::int32 var_31 = 0;
    wp::float32 var_32;
    const wp::int32 var_33 = 0;
    wp::float32 var_34;
    wp::float32 var_35;
    wp::float32 var_36;
    wp::float32 var_37;
    wp::float32 var_38;
    const wp::float32 var_39 = 0.5;
    const wp::float32 var_40 = 1.0;
    wp::float32 var_41;
    wp::float32 var_42;
    const wp::float32 var_43 = 1.0;
    bool var_44;
    const wp::float32 var_45 = 0.0;
    bool var_46;
    const wp::float32 var_47 = 1.0;
    wp::float32 var_48;
    const wp::float32 var_49 = 1.0;
    wp::float32 var_50;
    wp::float32 var_51;
    wp::float32 var_52;
    wp::float32 var_53;
    wp::float32 var_54;
    const wp::float32 var_55 = 0.5;
    wp::float32 var_56;
    wp::float32 var_57;
    wp::float32 var_58;
    wp::float32 var_59;
    const wp::float32 var_60 = 1.0;
    wp::float32 var_61;
    wp::float32 var_62;
    wp::float32 var_63;
    wp::float32 var_64;
    wp::float32 var_65;
    const wp::float32 var_66 = 0.5;
    wp::float32 var_67;
    wp::float32 var_68;
    wp::float32 var_69;
    //---------
    // forward
    // def muscle_bias(len: float, lengthrange: wp.vec2, acc0: float, prm: vec10) -> float:       <L 520>
    // range_ = wp.vec2(prm[0], prm[1])                                                       <L 526>
    var_1 = wp::extract(var_prm, var_0);
    var_3 = wp::extract(var_prm, var_2);
    var_4 = wp::vec_t<2, wp::float32>(var_1, var_3);
    // force = prm[2]                                                                         <L 527>
    var_6 = wp::extract(var_prm, var_5);
    // scale = prm[3]                                                                         <L 528>
    var_8 = wp::extract(var_prm, var_7);
    // lmax = prm[5]                                                                          <L 529>
    var_10 = wp::extract(var_prm, var_9);
    // fpmax = prm[7]                                                                         <L 530>
    var_12 = wp::extract(var_prm, var_11);
    // if force < 0.0:                                                                        <L 533>
    var_14 = (var_6 < var_13);
    if (var_14) {
        // force = scale / wp.max(MJ_MINVAL, acc0)                                            <L 534>
        var_16 = wp::max(var_15, var_acc0);
        var_17 = wp::div(var_8, var_16);
    }
    var_18 = wp::where(var_14, var_17, var_6);
    // L0 = (lengthrange[1] - lengthrange[0]) / wp.max(MJ_MINVAL, range_[1] - range_[0])       <L 537>
    var_20 = wp::extract(var_lengthrange, var_19);
    var_22 = wp::extract(var_lengthrange, var_21);
    var_23 = wp::sub(var_20, var_22);
    var_25 = wp::extract(var_4, var_24);
    var_27 = wp::extract(var_4, var_26);
    var_28 = wp::sub(var_25, var_27);
    var_29 = wp::max(var_15, var_28);
    var_30 = wp::div(var_23, var_29);
    // L = range_[0] + (len - lengthrange[0]) / wp.max(MJ_MINVAL, L0)                         <L 540>
    var_32 = wp::extract(var_4, var_31);
    var_34 = wp::extract(var_lengthrange, var_33);
    var_35 = wp::sub(var_len, var_34);
    var_36 = wp::max(var_15, var_30);
    var_37 = wp::div(var_35, var_36);
    var_38 = wp::add(var_32, var_37);
    // b = 0.5 * (1.0 + lmax)                                                                 <L 543>
    var_41 = wp::add(var_40, var_10);
    var_42 = wp::mul(var_39, var_41);
    // if L <= 1.0:                                                                           <L 544>
    var_44 = (var_38 <= var_43);
    if (var_44) {
        // return 0.0                                                                         <L 545>
        return var_45;
    }
    if (!var_44) {
        // elif L <= b:                                                                       <L 546>
        var_46 = (var_38 <= var_42);
        if (var_46) {
            // x = (L - 1.0) / wp.max(MJ_MINVAL, b - 1.0)                                     <L 547>
            var_48 = wp::sub(var_38, var_47);
            var_50 = wp::sub(var_42, var_49);
            var_51 = wp::max(var_15, var_50);
            var_52 = wp::div(var_48, var_51);
            // return -force * fpmax * 0.5 * x * x                                            <L 548>
            var_53 = wp::neg(var_18);
            var_54 = wp::mul(var_53, var_12);
            var_56 = wp::mul(var_54, var_55);
            var_57 = wp::mul(var_56, var_52);
            var_58 = wp::mul(var_57, var_52);
            return var_58;
        }
        if (!var_46) {
            // x = (L - b) / wp.max(MJ_MINVAL, b - 1.0)                                       <L 550>
            var_59 = wp::sub(var_38, var_42);
            var_61 = wp::sub(var_42, var_60);
            var_62 = wp::max(var_15, var_61);
            var_63 = wp::div(var_59, var_62);
            // return -force * fpmax * (0.5 + x)                                              <L 551>
            var_64 = wp::neg(var_18);
            var_65 = wp::mul(var_64, var_12);
            var_67 = wp::add(var_66, var_63);
            var_68 = wp::mul(var_65, var_67);
            return var_68;
        }
        var_69 = wp::where(var_46, var_52, var_63);
    }
    return {};
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/util_misc.py:717
static CUDA_CALLABLE wp::float32 _poly_force_deriv_0(
    wp::float32 var_linear,
    wp::vec_t<2, wp::float32> var_poly,
    wp::float32 var_x,
    wp::int32 var_flg_odd)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    bool var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::float32 var_5 = 2.0;
    const wp::int32 var_6 = 0;
    wp::float32 var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    const wp::float32 var_11 = 3.0;
    const wp::int32 var_12 = 1;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    //---------
    // forward
    // def _poly_force_deriv(linear: float, poly: wp.vec2, x: float, flg_odd: int) -> float:       <L 718>
    // x_val = wp.where(flg_odd == 1, wp.abs(x), x)                                           <L 719>
    var_1 = (var_flg_odd == var_0);
    var_2 = wp::abs(var_x);
    var_3 = wp::where(var_1, var_2, var_x);
    // res = linear                                                                           <L 720>
    var_4 = wp::copy(var_linear);
    // res += 2.0 * poly[0] * x_val                                                           <L 721>
    var_7 = wp::extract(var_poly, var_6);
    var_8 = wp::mul(var_5, var_7);
    var_9 = wp::mul(var_8, var_3);
    var_10 = wp::add(var_4, var_9);
    // res += 3.0 * poly[1] * x_val * x_val                                                   <L 722>
    var_13 = wp::extract(var_poly, var_12);
    var_14 = wp::mul(var_11, var_13);
    var_15 = wp::mul(var_14, var_3);
    var_16 = wp::mul(var_15, var_3);
    var_17 = wp::add(var_10, var_16);
    // return res                                                                             <L 723>
    return var_17;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/math.py:52
static CUDA_CALLABLE wp::quat_t<wp::float32> axis_angle_to_quat_0(
    wp::vec_t<3, wp::float32> var_axis,
    wp::float32 var_angle)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.5;
    wp::float32 var_1;
    wp::float32 var_2;
    const wp::float32 var_3 = 0.5;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::vec_t<3, wp::float32> var_6;
    const wp::int32 var_7 = 0;
    wp::float32 var_8;
    const wp::int32 var_9 = 1;
    wp::float32 var_10;
    const wp::int32 var_11 = 2;
    wp::float32 var_12;
    wp::quat_t<wp::float32> var_13;
    //---------
    // forward
    // def axis_angle_to_quat(axis: wp.vec3, angle: float) -> wp.quat:                        <L 53>
    // s, c = wp.sin(angle * 0.5), wp.cos(angle * 0.5)                                        <L 54>
    var_1 = wp::mul(var_angle, var_0);
    var_2 = wp::sin(var_1);
    var_4 = wp::mul(var_angle, var_3);
    var_5 = wp::cos(var_4);
    // axis = axis * s                                                                        <L 55>
    var_6 = wp::mul(var_axis, var_2);
    // return wp.quat(c, axis[0], axis[1], axis[2])                                           <L 56>
    var_8 = wp::extract(var_6, var_7);
    var_10 = wp::extract(var_6, var_9);
    var_12 = wp::extract(var_6, var_11);
    var_13 = wp::quat_t<wp::float32>(var_5, var_8, var_10, var_12);
    return var_13;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/math.py:23
static CUDA_CALLABLE wp::quat_t<wp::float32> mul_quat_0(
    wp::quat_t<wp::float32> var_u,
    wp::quat_t<wp::float32> var_v)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 0;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::int32 var_5 = 1;
    wp::float32 var_6;
    const wp::int32 var_7 = 1;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    const wp::int32 var_11 = 2;
    wp::float32 var_12;
    const wp::int32 var_13 = 2;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    const wp::int32 var_17 = 3;
    wp::float32 var_18;
    const wp::int32 var_19 = 3;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    const wp::int32 var_23 = 0;
    wp::float32 var_24;
    const wp::int32 var_25 = 1;
    wp::float32 var_26;
    wp::float32 var_27;
    const wp::int32 var_28 = 1;
    wp::float32 var_29;
    const wp::int32 var_30 = 0;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    const wp::int32 var_34 = 2;
    wp::float32 var_35;
    const wp::int32 var_36 = 3;
    wp::float32 var_37;
    wp::float32 var_38;
    wp::float32 var_39;
    const wp::int32 var_40 = 3;
    wp::float32 var_41;
    const wp::int32 var_42 = 2;
    wp::float32 var_43;
    wp::float32 var_44;
    wp::float32 var_45;
    const wp::int32 var_46 = 0;
    wp::float32 var_47;
    const wp::int32 var_48 = 2;
    wp::float32 var_49;
    wp::float32 var_50;
    const wp::int32 var_51 = 1;
    wp::float32 var_52;
    const wp::int32 var_53 = 3;
    wp::float32 var_54;
    wp::float32 var_55;
    wp::float32 var_56;
    const wp::int32 var_57 = 2;
    wp::float32 var_58;
    const wp::int32 var_59 = 0;
    wp::float32 var_60;
    wp::float32 var_61;
    wp::float32 var_62;
    const wp::int32 var_63 = 3;
    wp::float32 var_64;
    const wp::int32 var_65 = 1;
    wp::float32 var_66;
    wp::float32 var_67;
    wp::float32 var_68;
    const wp::int32 var_69 = 0;
    wp::float32 var_70;
    const wp::int32 var_71 = 3;
    wp::float32 var_72;
    wp::float32 var_73;
    const wp::int32 var_74 = 1;
    wp::float32 var_75;
    const wp::int32 var_76 = 2;
    wp::float32 var_77;
    wp::float32 var_78;
    wp::float32 var_79;
    const wp::int32 var_80 = 2;
    wp::float32 var_81;
    const wp::int32 var_82 = 1;
    wp::float32 var_83;
    wp::float32 var_84;
    wp::float32 var_85;
    const wp::int32 var_86 = 3;
    wp::float32 var_87;
    const wp::int32 var_88 = 0;
    wp::float32 var_89;
    wp::float32 var_90;
    wp::float32 var_91;
    wp::quat_t<wp::float32> var_92;
    //---------
    // forward
    // def mul_quat(u: wp.quat, v: wp.quat) -> wp.quat:                                       <L 24>
    // return wp.quat(                                                                        <L 25>
    // u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],                                 <L 26>
    var_1 = wp::extract(var_u, var_0);
    var_3 = wp::extract(var_v, var_2);
    var_4 = wp::mul(var_1, var_3);
    var_6 = wp::extract(var_u, var_5);
    var_8 = wp::extract(var_v, var_7);
    var_9 = wp::mul(var_6, var_8);
    var_10 = wp::sub(var_4, var_9);
    var_12 = wp::extract(var_u, var_11);
    var_14 = wp::extract(var_v, var_13);
    var_15 = wp::mul(var_12, var_14);
    var_16 = wp::sub(var_10, var_15);
    var_18 = wp::extract(var_u, var_17);
    var_20 = wp::extract(var_v, var_19);
    var_21 = wp::mul(var_18, var_20);
    var_22 = wp::sub(var_16, var_21);
    // u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],                                 <L 27>
    var_24 = wp::extract(var_u, var_23);
    var_26 = wp::extract(var_v, var_25);
    var_27 = wp::mul(var_24, var_26);
    var_29 = wp::extract(var_u, var_28);
    var_31 = wp::extract(var_v, var_30);
    var_32 = wp::mul(var_29, var_31);
    var_33 = wp::add(var_27, var_32);
    var_35 = wp::extract(var_u, var_34);
    var_37 = wp::extract(var_v, var_36);
    var_38 = wp::mul(var_35, var_37);
    var_39 = wp::add(var_33, var_38);
    var_41 = wp::extract(var_u, var_40);
    var_43 = wp::extract(var_v, var_42);
    var_44 = wp::mul(var_41, var_43);
    var_45 = wp::sub(var_39, var_44);
    // u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],                                 <L 28>
    var_47 = wp::extract(var_u, var_46);
    var_49 = wp::extract(var_v, var_48);
    var_50 = wp::mul(var_47, var_49);
    var_52 = wp::extract(var_u, var_51);
    var_54 = wp::extract(var_v, var_53);
    var_55 = wp::mul(var_52, var_54);
    var_56 = wp::sub(var_50, var_55);
    var_58 = wp::extract(var_u, var_57);
    var_60 = wp::extract(var_v, var_59);
    var_61 = wp::mul(var_58, var_60);
    var_62 = wp::add(var_56, var_61);
    var_64 = wp::extract(var_u, var_63);
    var_66 = wp::extract(var_v, var_65);
    var_67 = wp::mul(var_64, var_66);
    var_68 = wp::add(var_62, var_67);
    // u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],                                 <L 29>
    var_70 = wp::extract(var_u, var_69);
    var_72 = wp::extract(var_v, var_71);
    var_73 = wp::mul(var_70, var_72);
    var_75 = wp::extract(var_u, var_74);
    var_77 = wp::extract(var_v, var_76);
    var_78 = wp::mul(var_75, var_77);
    var_79 = wp::add(var_73, var_78);
    var_81 = wp::extract(var_u, var_80);
    var_83 = wp::extract(var_v, var_82);
    var_84 = wp::mul(var_81, var_83);
    var_85 = wp::sub(var_79, var_84);
    var_87 = wp::extract(var_u, var_86);
    var_89 = wp::extract(var_v, var_88);
    var_90 = wp::mul(var_87, var_89);
    var_91 = wp::add(var_85, var_90);
    var_92 = wp::quat_t<wp::float32>(var_22, var_45, var_68, var_91);
    return var_92;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/math.py:188
static CUDA_CALLABLE wp::quat_t<wp::float32> quat_integrate_0(
    wp::quat_t<wp::float32> var_q,
    wp::vec_t<3, wp::float32> var_v,
    wp::float32 var_dt)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::float32 var_2;
    wp::quat_t<wp::float32> var_3;
    wp::quat_t<wp::float32> var_4;
    wp::quat_t<wp::float32> var_5;
    wp::quat_t<wp::float32> var_6;
    //---------
    // forward
    // def quat_integrate(q: wp.quat, v: wp.vec3, dt: float) -> wp.quat:                      <L 189>
    // norm_ = wp.length(v)                                                                   <L 191>
    var_0 = wp::length(var_v);
    // v = wp.normalize(v)  # does that need proper zero gradient handling?                   <L 192>
    var_1 = wp::normalize(var_v);
    // angle = dt * norm_                                                                     <L 193>
    var_2 = wp::mul(var_dt, var_0);
    // q_res = axis_angle_to_quat(v, angle)                                                   <L 195>
    var_3 = axis_angle_to_quat_0(var_1, var_2);
    // q = wp.normalize(q)                                                                    <L 196>
    var_4 = wp::normalize(var_q);
    // q_res = mul_quat(q, q_res)                                                             <L 197>
    var_5 = mul_quat_0(var_4, var_3);
    // return wp.normalize(q_res)                                                             <L 199>
    var_6 = wp::normalize(var_5);
    return var_6;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/util_misc.py:603
static CUDA_CALLABLE void adj_dcmotor_slots_0(
    wp::vec_t<10, wp::float32> var_dynprm,
    wp::vec_t<10, wp::float32> var_gainprm,
    wp::vec_t<10, wp::float32> & adj_dynprm,
    wp::vec_t<10, wp::float32> & adj_gainprm,
    wp::vec_t<6, wp::int32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/util_misc.py:643
static CUDA_CALLABLE void adj_lugre_stribeck_0(
    wp::float32 var_velocity,
    wp::float32 var_F_C,
    wp::float32 var_F_S,
    wp::float32 var_v_S,
    wp::float32 & adj_velocity,
    wp::float32 & adj_F_C,
    wp::float32 & adj_F_S,
    wp::float32 & adj_v_S,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/support.py:37
static CUDA_CALLABLE void adj_next_act_0(
    wp::float32 var_opt_timestep,
    wp::int32 var_actuator_dyntype,
    wp::vec_t<10, wp::float32> var_actuator_dynprm,
    wp::vec_t<2, wp::float32> var_actuator_actrange,
    wp::float32 var_act_in,
    wp::float32 var_act_dot_in,
    wp::float32 var_act_dot_scale,
    bool var_clamp,
    wp::float32 & adj_opt_timestep,
    wp::int32 & adj_actuator_dyntype,
    wp::vec_t<10, wp::float32> & adj_actuator_dynprm,
    wp::vec_t<2, wp::float32> & adj_actuator_actrange,
    wp::float32 & adj_act_in,
    wp::float32 & adj_act_dot_in,
    wp::float32 & adj_act_dot_scale,
    bool & adj_clamp,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/util_misc.py:554
static CUDA_CALLABLE void adj__sigmoid_0(
    wp::float32 var_x,
    wp::float32 & adj_x,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/util_misc.py:568
static CUDA_CALLABLE void adj_muscle_dynamics_timescale_0(
    wp::float32 var_dctrl,
    wp::float32 var_tau_act,
    wp::float32 var_tau_deact,
    wp::float32 var_smooth_width,
    wp::float32 & adj_dctrl,
    wp::float32 & adj_tau_act,
    wp::float32 & adj_tau_deact,
    wp::float32 & adj_smooth_width,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/util_misc.py:582
static CUDA_CALLABLE void adj_muscle_dynamics_0(
    wp::float32 var_control,
    wp::float32 var_activation,
    wp::vec_t<10, wp::float32> var_prm,
    wp::float32 & adj_control,
    wp::float32 & adj_activation,
    wp::vec_t<10, wp::float32> & adj_prm,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/util_misc.py:649
static CUDA_CALLABLE void adj_dcmotor_voltage_0(
    wp::float32 var_u,
    wp::float32 var_length,
    wp::float32 var_velocity,
    wp::float32 var_x_I,
    wp::vec_t<10, wp::float32> var_gainprm,
    wp::float32 & adj_u,
    wp::float32 & adj_length,
    wp::float32 & adj_velocity,
    wp::float32 & adj_x_I,
    wp::vec_t<10, wp::float32> & adj_gainprm,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/util_misc.py:454
static CUDA_CALLABLE void adj_muscle_gain_length_0(
    wp::float32 var_length,
    wp::float32 var_lmin,
    wp::float32 var_lmax,
    wp::float32 & adj_length,
    wp::float32 & adj_lmin,
    wp::float32 & adj_lmax,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/util_misc.py:478
static CUDA_CALLABLE void adj_muscle_gain_0(
    wp::float32 var_len,
    wp::float32 var_vel,
    wp::vec_t<2, wp::float32> var_lengthrange,
    wp::float32 var_acc0,
    wp::vec_t<10, wp::float32> var_prm,
    wp::float32 & adj_len,
    wp::float32 & adj_vel,
    wp::vec_t<2, wp::float32> & adj_lengthrange,
    wp::float32 & adj_acc0,
    wp::vec_t<10, wp::float32> & adj_prm,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/util_misc.py:519
static CUDA_CALLABLE void adj_muscle_bias_0(
    wp::float32 var_len,
    wp::vec_t<2, wp::float32> var_lengthrange,
    wp::float32 var_acc0,
    wp::vec_t<10, wp::float32> var_prm,
    wp::float32 & adj_len,
    wp::vec_t<2, wp::float32> & adj_lengthrange,
    wp::float32 & adj_acc0,
    wp::vec_t<10, wp::float32> & adj_prm,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/util_misc.py:717
static CUDA_CALLABLE void adj__poly_force_deriv_0(
    wp::float32 var_linear,
    wp::vec_t<2, wp::float32> var_poly,
    wp::float32 var_x,
    wp::int32 var_flg_odd,
    wp::float32 & adj_linear,
    wp::vec_t<2, wp::float32> & adj_poly,
    wp::float32 & adj_x,
    wp::int32 & adj_flg_odd,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/math.py:52
static CUDA_CALLABLE void adj_axis_angle_to_quat_0(
    wp::vec_t<3, wp::float32> var_axis,
    wp::float32 var_angle,
    wp::vec_t<3, wp::float32> & adj_axis,
    wp::float32 & adj_angle,
    wp::quat_t<wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/math.py:23
static CUDA_CALLABLE void adj_mul_quat_0(
    wp::quat_t<wp::float32> var_u,
    wp::quat_t<wp::float32> var_v,
    wp::quat_t<wp::float32> & adj_u,
    wp::quat_t<wp::float32> & adj_v,
    wp::quat_t<wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/math.py:188
static CUDA_CALLABLE void adj_quat_integrate_0(
    wp::quat_t<wp::float32> var_q,
    wp::vec_t<3, wp::float32> var_v,
    wp::float32 var_dt,
    wp::quat_t<wp::float32> & adj_q,
    wp::vec_t<3, wp::float32> & adj_v,
    wp::float32 & adj_dt,
    wp::quat_t<wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void _next_activation_4a03322b_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::array_t<wp::int32> var_actuator_dyntype,
    wp::array_t<wp::int32> var_actuator_actadr,
    wp::array_t<wp::int32> var_actuator_actnum,
    wp::array_t<bool> var_actuator_actlimited,
    wp::array_t<wp::vec_t<10, wp::float32>> var_actuator_dynprm,
    wp::array_t<wp::vec_t<10, wp::float32>> var_actuator_gainprm,
    wp::array_t<wp::vec_t<10, wp::float32>> var_actuator_biasprm,
    wp::array_t<wp::vec_t<2, wp::float32>> var_actuator_actrange,
    wp::array_t<wp::float32> var_act_in,
    wp::array_t<wp::float32> var_act_dot_in,
    wp::array_t<wp::float32> var_actuator_velocity_in,
    wp::float32 var_act_dot_scale,
    bool var_limit,
    wp::array_t<wp::float32> var_act_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::shape_t* var_7;
        const wp::int32 var_8 = 0;
        wp::int32 var_9;
        wp::shape_t var_10;
        wp::int32 var_11;
        wp::shape_t* var_12;
        const wp::int32 var_13 = 0;
        wp::int32 var_14;
        wp::shape_t var_15;
        wp::int32 var_16;
        wp::shape_t* var_17;
        const wp::int32 var_18 = 0;
        wp::int32 var_19;
        wp::shape_t var_20;
        wp::int32 var_21;
        wp::shape_t* var_22;
        const wp::int32 var_23 = 0;
        wp::int32 var_24;
        wp::shape_t var_25;
        wp::int32 var_26;
        wp::int32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        wp::int32* var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        wp::int32* var_33;
        wp::int32 var_34;
        wp::int32 var_35;
        const wp::int32 var_36 = 5;
        bool var_37;
        wp::vec_t<10, wp::float32>* var_38;
        wp::vec_t<10, wp::float32> var_39;
        wp::vec_t<10, wp::float32> var_40;
        wp::vec_t<10, wp::float32>* var_41;
        wp::vec_t<10, wp::float32> var_42;
        wp::vec_t<10, wp::float32> var_43;
        wp::vec_t<10, wp::float32>* var_44;
        wp::vec_t<10, wp::float32> var_45;
        wp::vec_t<10, wp::float32> var_46;
        wp::vec_t<6, wp::int32> var_47;
        wp::int32 var_48;
        wp::range_t var_49;
        wp::int32 var_50;
        wp::int32 var_51;
        wp::float32* var_52;
        wp::float32 var_53;
        wp::float32 var_54;
        wp::float32* var_55;
        wp::float32 var_56;
        wp::float32 var_57;
        const wp::int32 var_58 = 4;
        wp::int32 var_59;
        bool var_60;
        const wp::int32 var_61 = 0;
        wp::float32 var_62;
        const wp::float32 var_63 = 1e-15;
        const wp::int32 var_64 = 0;
        wp::float32 var_65;
        wp::float32 var_66;
        wp::float32 var_67;
        const wp::float32 var_68 = 1.0;
        wp::float32* var_69;
        wp::float32 var_70;
        wp::float32 var_71;
        wp::float32 var_72;
        wp::float32 var_73;
        wp::float32 var_74;
        wp::float32 var_75;
        wp::float32 var_76;
        wp::float32 var_77;
        const wp::int32 var_78 = 3;
        wp::int32 var_79;
        bool var_80;
        const wp::int32 var_81 = 3;
        wp::float32 var_82;
        const wp::int32 var_83 = 4;
        wp::float32 var_84;
        const wp::int32 var_85 = 5;
        wp::float32 var_86;
        const wp::int32 var_87 = 5;
        wp::float32 var_88;
        wp::float32* var_89;
        wp::float32 var_90;
        wp::float32 var_91;
        wp::float32 var_92;
        wp::float32 var_93;
        wp::float32 var_94;
        wp::float32 var_95;
        wp::float32 var_96;
        wp::float32 var_97;
        wp::float32* var_98;
        wp::float32 var_99;
        wp::float32 var_100;
        wp::float32 var_101;
        wp::float32 var_102;
        wp::float32 var_103;
        wp::float32 var_104;
        bool var_105;
        const wp::float32 var_106 = 1.0;
        wp::float32 var_107;
        wp::float32 var_108;
        wp::float32 var_109;
        wp::float32 var_110;
        wp::float32 var_111;
        wp::float32 var_112;
        wp::float32 var_113;
        const wp::int32 var_114 = 1;
        wp::int32 var_115;
        bool var_116;
        wp::float32* var_117;
        wp::float32 var_118;
        wp::float32 var_119;
        wp::float32 var_120;
        const wp::int32 var_121 = 8;
        wp::float32 var_122;
        const wp::float32 var_123 = 0.0;
        bool var_124;
        wp::float32 var_125;
        wp::float32 var_126;
        wp::float32 var_127;
        wp::float32 var_128;
        wp::float32* var_129;
        wp::float32 var_130;
        wp::float32 var_131;
        wp::float32 var_132;
        wp::float32 var_133;
        wp::float32 var_134;
        wp::float32 var_135;
        wp::int32 var_136;
        wp::range_t var_137;
        wp::int32 var_138;
        wp::float32* var_139;
        wp::vec_t<10, wp::float32>* var_140;
        wp::vec_t<2, wp::float32>* var_141;
        wp::float32* var_142;
        wp::float32* var_143;
        bool var_144;
        bool* var_145;
        bool var_146;
        wp::float32 var_147;
        wp::float32 var_148;
        wp::vec_t<10, wp::float32> var_149;
        wp::vec_t<2, wp::float32> var_150;
        wp::float32 var_151;
        wp::float32 var_152;
        wp::int32 var_153;
        //---------
        // forward
        // def _next_activation(                                                                  <L 135>
        // worldid, uid = wp.tid()                                                                <L 156>
        builtin_tid2d(var_0, var_1);
        // opt_timestep_id = worldid % opt_timestep.shape[0]                                      <L 157>
        var_2 = &(var_opt_timestep.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // actuator_dynprm_id = worldid % actuator_dynprm.shape[0]                                <L 158>
        var_7 = &(var_actuator_dynprm.shape);
        var_10 = wp::load(var_7);
        var_9 = wp::extract(var_10, var_8);
        var_11 = wp::mod(var_0, var_9);
        // actuator_actrange_id = worldid % actuator_actrange.shape[0]                            <L 159>
        var_12 = &(var_actuator_actrange.shape);
        var_15 = wp::load(var_12);
        var_14 = wp::extract(var_15, var_13);
        var_16 = wp::mod(var_0, var_14);
        // actuator_gainprm_id = worldid % actuator_gainprm.shape[0]                              <L 160>
        var_17 = &(var_actuator_gainprm.shape);
        var_20 = wp::load(var_17);
        var_19 = wp::extract(var_20, var_18);
        var_21 = wp::mod(var_0, var_19);
        // actuator_biasprm_id = worldid % actuator_biasprm.shape[0]                              <L 161>
        var_22 = &(var_actuator_biasprm.shape);
        var_25 = wp::load(var_22);
        var_24 = wp::extract(var_25, var_23);
        var_26 = wp::mod(var_0, var_24);
        // actadr = actuator_actadr[uid]                                                          <L 163>
        var_27 = wp::address(var_actuator_actadr, var_1);
        var_29 = wp::load(var_27);
        var_28 = wp::copy(var_29);
        // actnum = actuator_actnum[uid]                                                          <L 164>
        var_30 = wp::address(var_actuator_actnum, var_1);
        var_32 = wp::load(var_30);
        var_31 = wp::copy(var_32);
        // dyntype = actuator_dyntype[uid]                                                        <L 165>
        var_33 = wp::address(var_actuator_dyntype, var_1);
        var_35 = wp::load(var_33);
        var_34 = wp::copy(var_35);
        // if dyntype == DynType.DCMOTOR:                                                         <L 167>
        var_37 = (var_34 == var_36);
        if (var_37) {
            // dynprm = actuator_dynprm[actuator_dynprm_id, uid]                                  <L 168>
            var_38 = wp::address(var_actuator_dynprm, var_11, var_1);
            var_40 = wp::load(var_38);
            var_39 = wp::copy(var_40);
            // gainprm = actuator_gainprm[actuator_gainprm_id, uid]                               <L 169>
            var_41 = wp::address(var_actuator_gainprm, var_21, var_1);
            var_43 = wp::load(var_41);
            var_42 = wp::copy(var_43);
            // biasprm = actuator_biasprm[actuator_biasprm_id, uid]                               <L 170>
            var_44 = wp::address(var_actuator_biasprm, var_26, var_1);
            var_46 = wp::load(var_44);
            var_45 = wp::copy(var_46);
            // slots = util_misc.dcmotor_slots(dynprm, gainprm)                                   <L 171>
            var_47 = dcmotor_slots_0(var_39, var_42);
            // for j in range(actadr, actadr + actnum):                                           <L 173>
            var_48 = wp::add(var_28, var_31);
            var_49 = wp::range(var_28, var_48);
            start_for_0:;
                if (iter_cmp(var_49) == 0) goto end_for_0;
                var_50 = wp::iter_next(var_49);
                // offset = j - actadr                                                            <L 174>
                var_51 = wp::sub(var_50, var_28);
                // act = act_in[worldid, j]                                                       <L 175>
                var_52 = wp::address(var_act_in, var_0, var_50);
                var_54 = wp::load(var_52);
                var_53 = wp::copy(var_54);
                // act_dot = act_dot_in[worldid, j]                                               <L 176>
                var_55 = wp::address(var_act_dot_in, var_0, var_50);
                var_57 = wp::load(var_55);
                var_56 = wp::copy(var_57);
                // if offset == slots[4]:  # current                                              <L 178>
                var_59 = wp::extract(var_47, var_58);
                var_60 = (var_51 == var_59);
                if (var_60) {
                    // R = gainprm[0]                                                             <L 179>
                    var_62 = wp::extract(var_42, var_61);
                    // te = wp.max(MJ_MINVAL, dynprm[0])                                          <L 180>
                    var_65 = wp::extract(var_39, var_64);
                    var_66 = wp::max(var_63, var_65);
                    // act = act + act_dot * te * (1.0 - wp.exp(-opt_timestep[opt_timestep_id] / te))       <L 181>
                    var_67 = wp::mul(var_56, var_66);
                    var_69 = wp::address(var_opt_timestep, var_6);
                    var_71 = wp::load(var_69);
                    var_70 = wp::neg(var_71);
                    var_72 = wp::div(var_70, var_66);
                    var_73 = wp::exp(var_72);
                    var_74 = wp::sub(var_68, var_73);
                    var_75 = wp::mul(var_67, var_74);
                    var_76 = wp::add(var_53, var_75);
                }
                var_77 = wp::where(var_60, var_76, var_53);
                if (!var_60) {
                    // elif offset == slots[3]:  # bristle                                        <L 182>
                    var_79 = wp::extract(var_47, var_78);
                    var_80 = (var_51 == var_79);
                    if (var_80) {
                        // F_C = biasprm[3]                                                       <L 183>
                        var_82 = wp::extract(var_45, var_81);
                        // F_S = biasprm[4]                                                       <L 184>
                        var_84 = wp::extract(var_45, var_83);
                        // v_S = biasprm[5]                                                       <L 185>
                        var_86 = wp::extract(var_45, var_85);
                        // sigma0 = dynprm[5]                                                     <L 186>
                        var_88 = wp::extract(var_39, var_87);
                        // velocity = actuator_velocity_in[worldid, uid]                          <L 187>
                        var_89 = wp::address(var_actuator_velocity_in, var_0, var_1);
                        var_91 = wp::load(var_89);
                        var_90 = wp::copy(var_91);
                        // g = util_misc.lugre_stribeck(velocity, F_C, F_S, v_S)                  <L 188>
                        var_92 = lugre_stribeck_0(var_90, var_82, var_84, var_86);
                        // a = -sigma0 * wp.abs(velocity) / wp.max(MJ_MINVAL, g)                  <L 190>
                        var_93 = wp::neg(var_88);
                        var_94 = wp::abs(var_90);
                        var_95 = wp::mul(var_93, var_94);
                        var_96 = wp::max(var_63, var_92);
                        var_97 = wp::div(var_95, var_96);
                        // h = opt_timestep[opt_timestep_id]                                      <L 191>
                        var_98 = wp::address(var_opt_timestep, var_6);
                        var_100 = wp::load(var_98);
                        var_99 = wp::copy(var_100);
                        // exp_ah = wp.exp(a * h)                                                 <L 192>
                        var_101 = wp::mul(var_97, var_99);
                        var_102 = wp::exp(var_101);
                        // int_h = h                                                              <L 193>
                        var_103 = wp::copy(var_99);
                        // if wp.abs(a) > MJ_MINVAL:                                              <L 194>
                        var_104 = wp::abs(var_97);
                        var_105 = (var_104 > var_63);
                        if (var_105) {
                            // int_h = (exp_ah - 1.0) / a                                         <L 195>
                            var_107 = wp::sub(var_102, var_106);
                            var_108 = wp::div(var_107, var_97);
                        }
                        var_109 = wp::where(var_105, var_108, var_103);
                        // act = exp_ah * act + int_h * velocity                                  <L 196>
                        var_110 = wp::mul(var_102, var_77);
                        var_111 = wp::mul(var_109, var_90);
                        var_112 = wp::add(var_110, var_111);
                    }
                    var_113 = wp::where(var_80, var_112, var_77);
                    if (!var_80) {
                        // elif offset == slots[1]:  # integral                                   <L 197>
                        var_115 = wp::extract(var_47, var_114);
                        var_116 = (var_51 == var_115);
                        if (var_116) {
                            // act = act + act_dot * opt_timestep[opt_timestep_id]                <L 198>
                            var_117 = wp::address(var_opt_timestep, var_6);
                            var_119 = wp::load(var_117);
                            var_118 = wp::mul(var_56, var_119);
                            var_120 = wp::add(var_113, var_118);
                            // Imax = dynprm[8]                                                   <L 199>
                            var_122 = wp::extract(var_39, var_121);
                            // if Imax > 0.0:                                                     <L 200>
                            var_124 = (var_122 > var_123);
                            if (var_124) {
                                // act = wp.clamp(act, -Imax, Imax)                               <L 201>
                                var_125 = wp::neg(var_122);
                                var_126 = wp::clamp(var_120, var_125, var_122);
                            }
                            var_127 = wp::where(var_124, var_126, var_120);
                        }
                        var_128 = wp::where(var_116, var_127, var_113);
                        if (!var_116) {
                            // act = act + act_dot * opt_timestep[opt_timestep_id]                <L 203>
                            var_129 = wp::address(var_opt_timestep, var_6);
                            var_131 = wp::load(var_129);
                            var_130 = wp::mul(var_56, var_131);
                            var_132 = wp::add(var_128, var_130);
                        }
                        var_133 = wp::where(var_116, var_128, var_132);
                    }
                    var_134 = wp::where(var_80, var_113, var_133);
                }
                var_135 = wp::where(var_60, var_77, var_134);
                // act_out[worldid, j] = act                                                      <L 205>
                wp::array_store(var_act_out, var_0, var_50, var_135);
                goto start_for_0;
            end_for_0:;
        }
        if (!var_37) {
            // for j in range(actadr, actadr + actnum):                                           <L 207>
            var_136 = wp::add(var_28, var_31);
            var_137 = wp::range(var_28, var_136);
            start_for_2:;
                if (iter_cmp(var_137) == 0) goto end_for_2;
                var_138 = wp::iter_next(var_137);
                // act = next_act(                                                                <L 208>
                // opt_timestep[opt_timestep_id],                                                 <L 209>
                var_139 = wp::address(var_opt_timestep, var_6);
                // dyntype,                                                                       <L 210>
                // actuator_dynprm[actuator_dynprm_id, uid],                                      <L 211>
                var_140 = wp::address(var_actuator_dynprm, var_11, var_1);
                // actuator_actrange[actuator_actrange_id, uid],                                  <L 212>
                var_141 = wp::address(var_actuator_actrange, var_16, var_1);
                // act_in[worldid, j],                                                            <L 213>
                var_142 = wp::address(var_act_in, var_0, var_138);
                // act_dot_in[worldid, j],                                                        <L 214>
                var_143 = wp::address(var_act_dot_in, var_0, var_138);
                // act_dot_scale,                                                                 <L 215>
                // limit and actuator_actlimited[uid],                                            <L 216>
                var_144 = var_limit;
                if (var_144) {
                    var_145 = wp::address(var_actuator_actlimited, var_1);
                    var_146 = wp::load(var_145);
                    var_144 = var_144 && var_146;
                }
                var_148 = wp::load(var_139);
                var_149 = wp::load(var_140);
                var_150 = wp::load(var_141);
                var_151 = wp::load(var_142);
                var_152 = wp::load(var_143);
                var_147 = next_act_0(var_148, var_34, var_149, var_150, var_151, var_152, var_act_dot_scale, var_144);
                // act_out[worldid, j] = act                                                      <L 218>
                wp::array_store(var_act_out, var_0, var_138, var_147);
                wp::assign(var_135, var_147);
                goto start_for_2;
            end_for_2:;
        }
        var_153 = wp::where(var_37, var_50, var_138);
    }
}



extern "C" __global__ void _tendon_actuator_force_2da6552b_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_actuator_trntype,
    wp::array_t<wp::vec_t<2, wp::int32>> var_actuator_trnid,
    wp::array_t<wp::float32> var_actuator_force_in,
    wp::array_t<wp::float32> var_ten_actfrc_out)
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
        const wp::int32 var_3 = 3;
        bool var_4;
        wp::int32 var_5;
        wp::vec_t<2, wp::int32>* var_6;
        const wp::int32 var_7 = 0;
        wp::int32 var_8;
        wp::vec_t<2, wp::int32> var_9;
        wp::slice_t var_10;
        const wp::int32 var_11 = 0;
        wp::array_t<wp::float32> var_12;
        wp::float32* var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        //---------
        // forward
        // def _tendon_actuator_force(                                                            <L 1054>
        // worldid, actid = wp.tid()                                                              <L 1063>
        builtin_tid2d(var_0, var_1);
        // if actuator_trntype[actid] == TrnType.TENDON:                                          <L 1065>
        var_2 = wp::address(var_actuator_trntype, var_1);
        var_5 = wp::load(var_2);
        var_4 = (var_5 == var_3);
        if (var_4) {
            // tenid = actuator_trnid[actid][0]                                                   <L 1066>
            var_6 = wp::address(var_actuator_trnid, var_1);
            var_9 = wp::load(var_6);
            var_8 = wp::extract(var_9, var_7);
            // wp.atomic_add(ten_actfrc_out[worldid], tenid, actuator_force_in[worldid, actid])       <L 1068>
            var_10 = wp::slice_t(var_0, var_0, var_11);
            var_12 = wp::view(var_ten_actfrc_out, var_10);
            var_13 = wp::address(var_actuator_force_in, var_0, var_1);
            var_15 = wp::load(var_13);
            var_14 = wp::atomic_add(var_12, var_8, var_15);
        }
    }
}



extern "C" __global__ void _tendon_velocity_ca816a63_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::float32> var_ten_J_in,
    wp::array_t<wp::float32> var_ten_velocity_out)
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
        wp::int32* var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::range_t var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::float32* var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        const wp::float32 var_16 = 0.0;
        bool var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::float32* var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        wp::float32 var_25;
        //---------
        // forward
        // def _tendon_velocity(                                                                  <L 706>
        // worldid, tenid = wp.tid()                                                              <L 717>
        builtin_tid2d(var_0, var_1);
        // velocity = float(0.0)                                                                  <L 719>
        var_3 = wp::float(var_2);
        // rownnz = ten_J_rownnz[tenid]                                                           <L 720>
        var_4 = wp::address(var_ten_J_rownnz, var_1);
        var_6 = wp::load(var_4);
        var_5 = wp::copy(var_6);
        // rowadr = ten_J_rowadr[tenid]                                                           <L 721>
        var_7 = wp::address(var_ten_J_rowadr, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // for i in range(rownnz):                                                                <L 722>
        var_10 = wp::range(var_5);
        start_for_0:;
            if (iter_cmp(var_10) == 0) goto end_for_0;
            var_11 = wp::iter_next(var_10);
            // sparseid = rowadr + i                                                              <L 723>
            var_12 = wp::add(var_8, var_11);
            // J = ten_J_in[worldid, sparseid]                                                    <L 724>
            var_13 = wp::address(var_ten_J_in, var_0, var_12);
            var_15 = wp::load(var_13);
            var_14 = wp::copy(var_15);
            // if J != 0.0:                                                                       <L 725>
            var_17 = (var_14 != var_16);
            if (var_17) {
                // colind = ten_J_colind[sparseid]                                                <L 726>
                var_18 = wp::address(var_ten_J_colind, var_12);
                var_20 = wp::load(var_18);
                var_19 = wp::copy(var_20);
                // velocity += J * qvel_in[worldid, colind]                                       <L 727>
                var_21 = wp::address(var_qvel_in, var_0, var_19);
                var_23 = wp::load(var_21);
                var_22 = wp::mul(var_14, var_23);
                var_24 = wp::add(var_3, var_22);
            }
            var_25 = wp::where(var_17, var_24, var_3);
            wp::assign(var_3, var_25);
            goto start_for_0;
        end_for_0:;
        // ten_velocity_out[worldid, tenid] = velocity                                            <L 729>
        wp::array_store(var_ten_velocity_out, var_0, var_1, var_3);
    }
}



extern "C" __global__ void _tendon_actuator_force_clamp_fd85b13d_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<bool> var_tendon_actfrclimited,
    wp::array_t<wp::vec_t<2, wp::float32>> var_tendon_actfrcrange,
    wp::array_t<wp::int32> var_actuator_trntype,
    wp::array_t<wp::vec_t<2, wp::int32>> var_actuator_trnid,
    wp::array_t<wp::float32> var_ten_actfrc_in,
    wp::array_t<wp::float32> var_actuator_force_out)
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
        const wp::int32 var_3 = 3;
        bool var_4;
        wp::int32 var_5;
        wp::vec_t<2, wp::int32>* var_6;
        const wp::int32 var_7 = 0;
        wp::int32 var_8;
        wp::vec_t<2, wp::int32> var_9;
        bool* var_10;
        bool var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        wp::shape_t* var_15;
        const wp::int32 var_16 = 0;
        wp::int32 var_17;
        wp::shape_t var_18;
        wp::int32 var_19;
        wp::vec_t<2, wp::float32>* var_20;
        wp::vec_t<2, wp::float32> var_21;
        wp::vec_t<2, wp::float32> var_22;
        const wp::int32 var_23 = 0;
        wp::float32 var_24;
        bool var_25;
        const wp::int32 var_26 = 0;
        wp::float32 var_27;
        wp::float32 var_28;
        wp::float32* var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        const wp::int32 var_32 = 1;
        wp::float32 var_33;
        bool var_34;
        const wp::int32 var_35 = 1;
        wp::float32 var_36;
        wp::float32 var_37;
        wp::float32* var_38;
        wp::float32 var_39;
        wp::float32 var_40;
        bool var_41;
        //---------
        // forward
        // def _tendon_actuator_force_clamp(                                                      <L 1072>
        // worldid, actid = wp.tid()                                                              <L 1083>
        builtin_tid2d(var_0, var_1);
        // if actuator_trntype[actid] == TrnType.TENDON:                                          <L 1085>
        var_2 = wp::address(var_actuator_trntype, var_1);
        var_5 = wp::load(var_2);
        var_4 = (var_5 == var_3);
        if (var_4) {
            // tenid = actuator_trnid[actid][0]                                                   <L 1086>
            var_6 = wp::address(var_actuator_trnid, var_1);
            var_9 = wp::load(var_6);
            var_8 = wp::extract(var_9, var_7);
            // if tendon_actfrclimited[tenid]:                                                    <L 1087>
            var_10 = wp::address(var_tendon_actfrclimited, var_8);
            var_11 = wp::load(var_10);
            if (var_11) {
                // ten_actfrc = ten_actfrc_in[worldid, tenid]                                     <L 1088>
                var_12 = wp::address(var_ten_actfrc_in, var_0, var_8);
                var_14 = wp::load(var_12);
                var_13 = wp::copy(var_14);
                // actfrcrange = tendon_actfrcrange[worldid % tendon_actfrcrange.shape[0], tenid]       <L 1089>
                var_15 = &(var_tendon_actfrcrange.shape);
                var_18 = wp::load(var_15);
                var_17 = wp::extract(var_18, var_16);
                var_19 = wp::mod(var_0, var_17);
                var_20 = wp::address(var_tendon_actfrcrange, var_19, var_8);
                var_22 = wp::load(var_20);
                var_21 = wp::copy(var_22);
                // if ten_actfrc < actfrcrange[0]:                                                <L 1091>
                var_24 = wp::extract(var_21, var_23);
                var_25 = (var_13 < var_24);
                if (var_25) {
                    // actuator_force_out[worldid, actid] *= actfrcrange[0] / ten_actfrc          <L 1092>
                    var_27 = wp::extract(var_21, var_26);
                    var_28 = wp::div(var_27, var_13);
                    var_29 = wp::address(var_actuator_force_out, var_0, var_1);
                    var_31 = wp::load(var_29);
                    var_30 = wp::mul(var_31, var_28);
                    wp::array_store(var_actuator_force_out, var_0, var_1, var_30);
                }
                if (!var_25) {
                    // elif ten_actfrc > actfrcrange[1]:                                          <L 1093>
                    var_33 = wp::extract(var_21, var_32);
                    var_34 = (var_13 > var_33);
                    if (var_34) {
                        // actuator_force_out[worldid, actid] *= actfrcrange[1] / ten_actfrc       <L 1094>
                        var_36 = wp::extract(var_21, var_35);
                        var_37 = wp::div(var_36, var_13);
                        var_38 = wp::address(var_actuator_force_out, var_0, var_1);
                        var_40 = wp::load(var_38);
                        var_39 = wp::mul(var_40, var_37);
                        wp::array_store(var_actuator_force_out, var_0, var_1, var_39);
                    }
                }
            }
            var_41 = wp::load(var_10);
        }
    }
}



extern "C" __global__ void _qfrc_actuator_gravcomp_limits_a91eb6a0_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<bool> var_jnt_actfrclimited,
    wp::array_t<wp::int32> var_jnt_actgravcomp,
    wp::array_t<wp::vec_t<2, wp::float32>> var_jnt_actfrcrange,
    wp::array_t<wp::int32> var_dof_jntid,
    wp::array_t<wp::float32> var_qfrc_gravcomp_in,
    wp::array_t<wp::float32> var_qfrc_actuator_in,
    bool var_gravity_enabled,
    wp::array_t<wp::float32> var_qfrc_actuator_out)
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
        wp::float32* var_5;
        wp::float32 var_6;
        wp::float32 var_7;
        bool var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        bool* var_15;
        bool var_16;
        wp::shape_t* var_17;
        const wp::int32 var_18 = 0;
        wp::int32 var_19;
        wp::shape_t var_20;
        wp::int32 var_21;
        wp::vec_t<2, wp::float32>* var_22;
        wp::vec_t<2, wp::float32> var_23;
        wp::vec_t<2, wp::float32> var_24;
        const wp::int32 var_25 = 0;
        wp::float32 var_26;
        const wp::int32 var_27 = 1;
        wp::float32 var_28;
        wp::float32 var_29;
        bool var_30;
        wp::float32 var_31;
        bool var_32;
        //---------
        // forward
        // def _qfrc_actuator_gravcomp_limits(                                                    <L 1121>
        // worldid, dofid = wp.tid()                                                              <L 1135>
        builtin_tid2d(var_0, var_1);
        // jntid = dof_jntid[dofid]                                                               <L 1136>
        var_2 = wp::address(var_dof_jntid, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // qfrc = qfrc_actuator_in[worldid, dofid]                                                <L 1138>
        var_5 = wp::address(var_qfrc_actuator_in, var_0, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // if gravity_enabled and jnt_actgravcomp[jntid]:                                         <L 1141>
        var_8 = var_gravity_enabled;
        if (var_8) {
            var_9 = wp::address(var_jnt_actgravcomp, var_3);
            var_10 = wp::load(var_9);
            var_8 = var_8 && var_10;
        }
        if (var_8) {
            // qfrc += qfrc_gravcomp_in[worldid, dofid]                                           <L 1142>
            var_11 = wp::address(var_qfrc_gravcomp_in, var_0, var_1);
            var_13 = wp::load(var_11);
            var_12 = wp::add(var_6, var_13);
        }
        var_14 = wp::where(var_8, var_12, var_6);
        // if jnt_actfrclimited[jntid]:                                                           <L 1145>
        var_15 = wp::address(var_jnt_actfrclimited, var_3);
        var_16 = wp::load(var_15);
        if (var_16) {
            // frcrange = jnt_actfrcrange[worldid % jnt_actfrcrange.shape[0], jntid]              <L 1146>
            var_17 = &(var_jnt_actfrcrange.shape);
            var_20 = wp::load(var_17);
            var_19 = wp::extract(var_20, var_18);
            var_21 = wp::mod(var_0, var_19);
            var_22 = wp::address(var_jnt_actfrcrange, var_21, var_3);
            var_24 = wp::load(var_22);
            var_23 = wp::copy(var_24);
            // qfrc = wp.clamp(qfrc, frcrange[0], frcrange[1])                                    <L 1147>
            var_26 = wp::extract(var_23, var_25);
            var_28 = wp::extract(var_23, var_27);
            var_29 = wp::clamp(var_14, var_26, var_28);
        }
        var_30 = wp::load(var_15);
        var_32 = wp::load(var_15);
        var_31 = wp::where(var_32, var_29, var_14);
        // qfrc_actuator_out[worldid, dofid] = qfrc                                               <L 1149>
        wp::array_store(var_qfrc_actuator_out, var_0, var_1, var_31);
    }
}



extern "C" __global__ void _qfrc_actuator_6de5c51c_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_moment_rownnz_in,
    wp::array_t<wp::int32> var_moment_rowadr_in,
    wp::array_t<wp::int32> var_moment_colind_in,
    wp::array_t<wp::float32> var_actuator_moment_in,
    wp::array_t<wp::float32> var_actuator_force_in,
    wp::array_t<wp::float32> var_qfrc_actuator_out)
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
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::float32* var_14;
        wp::float32* var_15;
        wp::float32 var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        wp::slice_t var_19;
        const wp::int32 var_20 = 0;
        wp::array_t<wp::float32> var_21;
        wp::float32 var_22;
        //---------
        // forward
        // def _qfrc_actuator(                                                                    <L 1098>
        // worldid, actid = wp.tid()                                                              <L 1108>
        builtin_tid2d(var_0, var_1);
        // rownnz = moment_rownnz_in[worldid, actid]                                              <L 1110>
        var_2 = wp::address(var_moment_rownnz_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // rowadr = moment_rowadr_in[worldid, actid]                                              <L 1111>
        var_5 = wp::address(var_moment_rowadr_in, var_0, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // for i in range(rownnz):                                                                <L 1113>
        var_8 = wp::range(var_3);
        start_for_0:;
            if (iter_cmp(var_8) == 0) goto end_for_0;
            var_9 = wp::iter_next(var_8);
            // sparseid = rowadr + i                                                              <L 1114>
            var_10 = wp::add(var_6, var_9);
            // colind = moment_colind_in[worldid, sparseid]                                       <L 1115>
            var_11 = wp::address(var_moment_colind_in, var_0, var_10);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
            // qfrc = actuator_moment_in[worldid, sparseid] * actuator_force_in[worldid, actid]       <L 1116>
            var_14 = wp::address(var_actuator_moment_in, var_0, var_10);
            var_15 = wp::address(var_actuator_force_in, var_0, var_1);
            var_17 = wp::load(var_14);
            var_18 = wp::load(var_15);
            var_16 = wp::mul(var_17, var_18);
            // wp.atomic_add(qfrc_actuator_out[worldid], colind, qfrc)                            <L 1117>
            var_19 = wp::slice_t(var_0, var_0, var_20);
            var_21 = wp::view(var_qfrc_actuator_out, var_19);
            var_22 = wp::atomic_add(var_21, var_12, var_16);
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void _euler_damp_qfrc_de139d07_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::array_t<wp::int32> var_M_rownnz,
    wp::array_t<wp::int32> var_M_rowadr,
    wp::array_t<wp::float32> var_damp_deriv,
    wp::array_t<wp::float32> var_M_integration_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        wp::int32* var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        const wp::int32 var_15 = 1;
        wp::int32 var_16;
        wp::float32* var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        //---------
        // forward
        // def _euler_damp_qfrc(                                                                  <L 370>
        // worldid, tid = wp.tid()                                                                <L 380>
        builtin_tid2d(var_0, var_1);
        // timestep = opt_timestep[worldid % opt_timestep.shape[0]]                               <L 381>
        var_2 = &(var_opt_timestep.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        var_7 = wp::address(var_opt_timestep, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // adr = M_rowadr[tid] + M_rownnz[tid] - 1                                                <L 383>
        var_10 = wp::address(var_M_rowadr, var_1);
        var_11 = wp::address(var_M_rownnz, var_1);
        var_13 = wp::load(var_10);
        var_14 = wp::load(var_11);
        var_12 = wp::add(var_13, var_14);
        var_16 = wp::sub(var_12, var_15);
        // M_integration_out[worldid, adr] += timestep * damp_deriv[worldid, tid]                 <L 384>
        var_17 = wp::address(var_damp_deriv, var_0, var_1);
        var_19 = wp::load(var_17);
        var_18 = wp::mul(var_8, var_19);
        var_20 = wp::atomic_add(var_M_integration_out, var_0, var_16, var_18);
    }
}



extern "C" __global__ void _next_velocity_d66c2f53_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::float32> var_qacc_in,
    wp::float32 var_qacc_scale_in,
    wp::array_t<wp::float32> var_qvel_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        wp::float32* var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        //---------
        // forward
        // def _next_velocity(                                                                    <L 118>
        // worldid, dofid = wp.tid()                                                              <L 129>
        builtin_tid2d(var_0, var_1);
        // timestep = opt_timestep[worldid % opt_timestep.shape[0]]                               <L 130>
        var_2 = &(var_opt_timestep.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        var_7 = wp::address(var_opt_timestep, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // qvel_out[worldid, dofid] = qvel_in[worldid, dofid] + qacc_scale_in * qacc_in[worldid, dofid] * timestep       <L 131>
        var_10 = wp::address(var_qvel_in, var_0, var_1);
        var_11 = wp::address(var_qacc_in, var_0, var_1);
        var_13 = wp::load(var_11);
        var_12 = wp::mul(var_qacc_scale_in, var_13);
        var_14 = wp::mul(var_12, var_8);
        var_16 = wp::load(var_10);
        var_15 = wp::add(var_16, var_14);
        wp::array_store(var_qvel_out, var_0, var_1, var_15);
    }
}



extern "C" __global__ void _rk_accumulate_velocity_acceleration_829bb7cc_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::float32> var_qacc_in,
    wp::float32 var_scale,
    wp::array_t<wp::float32> var_qvel_out,
    wp::array_t<wp::float32> var_qacc_out)
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
        wp::float32 var_3;
        wp::float32 var_4;
        wp::float32 var_5;
        wp::float32* var_6;
        wp::float32 var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        //---------
        // forward
        // def _rk_accumulate_velocity_acceleration(                                              <L 470>
        // worldid, dofid = wp.tid()                                                              <L 480>
        builtin_tid2d(var_0, var_1);
        // qvel_out[worldid, dofid] += scale * qvel_in[worldid, dofid]                            <L 481>
        var_2 = wp::address(var_qvel_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::mul(var_scale, var_4);
        var_5 = wp::atomic_add(var_qvel_out, var_0, var_1, var_3);
        // qacc_out[worldid, dofid] += scale * qacc_in[worldid, dofid]                            <L 482>
        var_6 = wp::address(var_qacc_in, var_0, var_1);
        var_8 = wp::load(var_6);
        var_7 = wp::mul(var_scale, var_8);
        var_9 = wp::atomic_add(var_qacc_out, var_0, var_1, var_7);
    }
}



extern "C" __global__ void _map_m2d_273b5a0a_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_mapM2D,
    wp::array_t<wp::float32> var_qH_M,
    wp::array_t<wp::float32> var_qLU_out)
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
        const wp::float32 var_9 = 0.0;
        //---------
        // forward
        // def _map_m2d(                                                                          <L 561>
        // worldid, elemid = wp.tid()                                                             <L 570>
        builtin_tid2d(var_0, var_1);
        // m_idx = mapM2D[elemid]                                                                 <L 571>
        var_2 = wp::address(var_mapM2D, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if m_idx >= 0:                                                                         <L 572>
        var_6 = (var_3 >= var_5);
        if (var_6) {
            // qLU_out[worldid, elemid] = qH_M[worldid, m_idx]                                    <L 573>
            var_7 = wp::address(var_qH_M, var_0, var_3);
            var_8 = wp::load(var_7);
            wp::array_store(var_qLU_out, var_0, var_1, var_8);
        }
        if (!var_6) {
            // qLU_out[worldid, elemid] = 0.0                                                     <L 575>
            wp::array_store(var_qLU_out, var_0, var_1, var_9);
        }
    }
}



extern "C" __global__ void _actuator_force_311863ad_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_na,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::array_t<wp::int32> var_actuator_dyntype,
    wp::array_t<wp::int32> var_actuator_gaintype,
    wp::array_t<wp::int32> var_actuator_biastype,
    wp::array_t<wp::int32> var_actuator_actadr,
    wp::array_t<wp::int32> var_actuator_actnum,
    wp::array_t<bool> var_actuator_ctrllimited,
    wp::array_t<bool> var_actuator_forcelimited,
    wp::array_t<bool> var_actuator_actlimited,
    wp::array_t<wp::vec_t<10, wp::float32>> var_actuator_dynprm,
    wp::array_t<wp::vec_t<10, wp::float32>> var_actuator_gainprm,
    wp::array_t<wp::vec_t<10, wp::float32>> var_actuator_biasprm,
    wp::array_t<bool> var_actuator_actearly,
    wp::array_t<wp::vec_t<2, wp::float32>> var_actuator_ctrlrange,
    wp::array_t<wp::vec_t<2, wp::float32>> var_actuator_forcerange,
    wp::array_t<wp::vec_t<2, wp::float32>> var_actuator_actrange,
    wp::array_t<wp::float32> var_actuator_acc0,
    wp::array_t<wp::vec_t<2, wp::float32>> var_actuator_lengthrange,
    wp::array_t<wp::float32> var_act_in,
    wp::array_t<wp::float32> var_ctrl_in,
    wp::array_t<wp::float32> var_actuator_length_in,
    wp::array_t<wp::float32> var_actuator_velocity_in,
    wp::int32 var_dsbl_clampctrl,
    wp::array_t<wp::float32> var_act_dot_out,
    wp::array_t<wp::float32> var_actuator_force_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        bool var_10;
        bool* var_11;
        bool var_12;
        bool var_13;
        wp::vec_t<2, wp::float32>* var_14;
        wp::vec_t<2, wp::float32> var_15;
        wp::vec_t<2, wp::float32> var_16;
        const wp::int32 var_17 = 0;
        wp::float32 var_18;
        const wp::int32 var_19 = 1;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::int32* var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        bool var_27;
        const wp::int32 var_28 = 0;
        bool var_29;
        wp::int32* var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        const wp::int32 var_33 = 1;
        wp::int32 var_34;
        wp::int32* var_35;
        wp::int32 var_36;
        wp::int32 var_37;
        wp::shape_t* var_38;
        const wp::int32 var_39 = 0;
        wp::int32 var_40;
        wp::shape_t var_41;
        wp::int32 var_42;
        wp::vec_t<10, wp::float32>* var_43;
        wp::vec_t<10, wp::float32> var_44;
        wp::vec_t<10, wp::float32> var_45;
        const wp::int32 var_46 = 1;
        bool var_47;
        wp::float32 var_48;
        bool var_49;
        const wp::int32 var_50 = 2;
        bool var_51;
        const wp::int32 var_52 = 3;
        bool var_53;
        wp::float32* var_54;
        wp::float32 var_55;
        wp::float32 var_56;
        wp::float32 var_57;
        const wp::int32 var_58 = 0;
        wp::float32 var_59;
        const wp::float32 var_60 = 1e-15;
        wp::float32 var_61;
        wp::float32 var_62;
        wp::float32 var_63;
        const wp::int32 var_64 = 4;
        bool var_65;
        wp::shape_t* var_66;
        const wp::int32 var_67 = 0;
        wp::int32 var_68;
        wp::shape_t var_69;
        wp::int32 var_70;
        wp::vec_t<10, wp::float32>* var_71;
        wp::vec_t<10, wp::float32> var_72;
        wp::vec_t<10, wp::float32> var_73;
        wp::float32* var_74;
        wp::float32 var_75;
        wp::float32 var_76;
        wp::float32 var_77;
        wp::vec_t<10, wp::float32> var_78;
        wp::float32 var_79;
        wp::float32 var_80;
        const wp::int32 var_81 = 5;
        bool var_82;
        wp::shape_t* var_83;
        const wp::int32 var_84 = 0;
        wp::int32 var_85;
        wp::shape_t var_86;
        wp::int32 var_87;
        wp::vec_t<10, wp::float32>* var_88;
        wp::vec_t<10, wp::float32> var_89;
        wp::vec_t<10, wp::float32> var_90;
        wp::vec_t<6, wp::int32> var_91;
        wp::int32 var_92;
        const wp::float32 var_93 = 0.0;
        const wp::int32 var_94 = 0;
        wp::int32 var_95;
        const wp::int32 var_96 = 0;
        bool var_97;
        wp::float32* var_98;
        wp::float32 var_99;
        wp::float32 var_100;
        const wp::int32 var_101 = 7;
        wp::float32 var_102;
        wp::shape_t* var_103;
        const wp::int32 var_104 = 0;
        wp::int32 var_105;
        wp::shape_t var_106;
        wp::int32 var_107;
        wp::float32* var_108;
        wp::float32 var_109;
        wp::float32 var_110;
        wp::float32 var_111;
        wp::float32 var_112;
        wp::float32 var_113;
        wp::float32 var_114;
        wp::shape_t* var_115;
        const wp::int32 var_116 = 0;
        wp::int32 var_117;
        wp::shape_t var_118;
        wp::int32 var_119;
        wp::float32* var_120;
        wp::float32 var_121;
        wp::float32 var_122;
        wp::float32 var_123;
        const wp::int32 var_124 = 1;
        wp::int32 var_125;
        wp::float32 var_126;
        wp::float32 var_127;
        wp::int32 var_128;
        const wp::float32 var_129 = 0.0;
        const wp::int32 var_130 = 1;
        wp::int32 var_131;
        const wp::int32 var_132 = 0;
        bool var_133;
        wp::float32* var_134;
        wp::float32 var_135;
        wp::float32 var_136;
        const wp::int32 var_137 = 8;
        wp::float32 var_138;
        wp::int32 var_139;
        const wp::int32 var_140 = 8;
        wp::float32 var_141;
        wp::float32 var_142;
        const wp::int32 var_143 = 1;
        bool var_144;
        wp::float32* var_145;
        wp::float32 var_146;
        wp::float32 var_147;
        wp::float32 var_148;
        const wp::float32 var_149 = 0.0;
        bool var_150;
        bool var_151;
        const wp::float32 var_152 = 0.0;
        wp::float32 var_153;
        wp::float32 var_154;
        wp::float32 var_155;
        bool var_156;
        const wp::float32 var_157 = 0.0;
        wp::float32 var_158;
        wp::float32 var_159;
        wp::float32 var_160;
        wp::float32 var_161;
        const wp::int32 var_162 = 1;
        wp::int32 var_163;
        wp::float32 var_164;
        wp::int32 var_165;
        wp::float32 var_166;
        wp::float32* var_167;
        wp::float32* var_168;
        wp::float32 var_169;
        wp::float32 var_170;
        wp::float32 var_171;
        const wp::int32 var_172 = 0;
        wp::float32 var_173;
        const wp::int32 var_174 = 1;
        wp::float32 var_175;
        const wp::int32 var_176 = 0;
        wp::float32 var_177;
        wp::float32 var_178;
        const wp::int32 var_179 = 2;
        wp::int32 var_180;
        const wp::int32 var_181 = 0;
        bool var_182;
        const wp::int32 var_183 = 2;
        wp::float32 var_184;
        const wp::int32 var_185 = 3;
        wp::float32 var_186;
        const wp::int32 var_187 = 4;
        wp::float32 var_188;
        const wp::int32 var_189 = 2;
        wp::float32 var_190;
        const wp::int32 var_191 = 3;
        wp::float32 var_192;
        wp::float32* var_193;
        wp::float32 var_194;
        wp::float32 var_195;
        const wp::float32 var_196 = 1.0;
        wp::float32 var_197;
        wp::float32 var_198;
        wp::float32 var_199;
        wp::float32 var_200;
        wp::float32 var_201;
        wp::float32* var_202;
        wp::float32 var_203;
        wp::float32 var_204;
        wp::float32 var_205;
        wp::float32 var_206;
        const wp::int32 var_207 = 4;
        wp::int32 var_208;
        const wp::int32 var_209 = 0;
        bool var_210;
        wp::float32* var_211;
        wp::float32 var_212;
        wp::float32 var_213;
        wp::float32 var_214;
        wp::float32 var_215;
        wp::float32 var_216;
        wp::float32 var_217;
        wp::float32 var_218;
        wp::float32 var_219;
        const wp::int32 var_220 = 1;
        wp::int32 var_221;
        wp::float32 var_222;
        wp::float32 var_223;
        wp::int32 var_224;
        wp::float32 var_225;
        const wp::int32 var_226 = 3;
        wp::int32 var_227;
        const wp::int32 var_228 = 0;
        bool var_229;
        const wp::int32 var_230 = 5;
        wp::float32 var_231;
        wp::shape_t* var_232;
        const wp::int32 var_233 = 0;
        wp::int32 var_234;
        wp::shape_t var_235;
        wp::int32 var_236;
        wp::vec_t<10, wp::float32>* var_237;
        wp::vec_t<10, wp::float32> var_238;
        wp::vec_t<10, wp::float32> var_239;
        const wp::int32 var_240 = 3;
        wp::float32 var_241;
        const wp::int32 var_242 = 4;
        wp::float32 var_243;
        const wp::int32 var_244 = 5;
        wp::float32 var_245;
        wp::float32* var_246;
        wp::float32 var_247;
        wp::float32 var_248;
        wp::float32* var_249;
        wp::float32 var_250;
        wp::float32 var_251;
        wp::float32 var_252;
        wp::float32* var_253;
        wp::float32 var_254;
        wp::float32 var_255;
        wp::float32 var_256;
        wp::float32 var_257;
        wp::float32 var_258;
        wp::float32 var_259;
        wp::float32* var_260;
        wp::float32 var_261;
        wp::float32 var_262;
        const wp::int32 var_263 = 1;
        wp::int32 var_264;
        wp::float32 var_265;
        wp::int32 var_266;
        const wp::int32 var_267 = 4;
        wp::int32 var_268;
        const wp::int32 var_269 = 0;
        bool var_270;
        const wp::int32 var_271 = 1;
        wp::float32 var_272;
        wp::float32 var_273;
        wp::float32 var_274;
        wp::float32* var_275;
        wp::float32 var_276;
        wp::float32 var_277;
        wp::float32 var_278;
        wp::float32* var_279;
        wp::float32 var_280;
        wp::float32 var_281;
        wp::float32 var_282;
        const wp::float32 var_283 = 0.0;
        bool var_284;
        wp::float32 var_285;
        wp::float32 var_286;
        wp::float32 var_287;
        wp::float32 var_288;
        wp::float32 var_289;
        wp::float32 var_290;
        const wp::int32 var_291 = 6;
        bool var_292;
        const wp::float32 var_293 = 0.0;
        wp::float32 var_294;
        const wp::float32 var_295 = 0.0;
        wp::float32 var_296;
        wp::float32 var_297;
        wp::float32 var_298;
        wp::float32 var_299;
        wp::float32 var_300;
        wp::vec_t<10, wp::float32> var_301;
        wp::float32 var_302;
        wp::float32 var_303;
        wp::float32 var_304;
        wp::vec_t<10, wp::float32> var_305;
        wp::float32 var_306;
        bool* var_307;
        bool var_308;
        bool var_309;
        const wp::int32 var_310 = 1;
        bool var_311;
        const wp::int32 var_312 = 0;
        bool var_313;
        const wp::int32 var_314 = 5;
        bool var_315;
        wp::float32* var_316;
        wp::float32 var_317;
        wp::float32 var_318;
        wp::float32 var_319;
        const wp::int32 var_320 = 5;
        bool var_321;
        wp::shape_t* var_322;
        const wp::int32 var_323 = 0;
        wp::int32 var_324;
        wp::shape_t var_325;
        wp::int32 var_326;
        wp::vec_t<10, wp::float32>* var_327;
        wp::vec_t<10, wp::float32> var_328;
        wp::vec_t<10, wp::float32> var_329;
        wp::vec_t<6, wp::int32> var_330;
        wp::int32* var_331;
        const wp::int32 var_332 = 1;
        wp::int32 var_333;
        wp::int32 var_334;
        const wp::int32 var_335 = 4;
        wp::int32 var_336;
        bool var_337;
        const wp::int32 var_338 = 0;
        wp::float32 var_339;
        wp::float32 var_340;
        wp::float32 var_341;
        const wp::float32 var_342 = 1.0;
        wp::shape_t* var_343;
        const wp::int32 var_344 = 0;
        wp::int32 var_345;
        wp::shape_t var_346;
        wp::int32 var_347;
        wp::float32* var_348;
        wp::float32 var_349;
        wp::float32 var_350;
        wp::float32 var_351;
        wp::float32 var_352;
        wp::float32 var_353;
        wp::float32 var_354;
        wp::float32 var_355;
        wp::float32 var_356;
        wp::float32 var_357;
        const wp::int32 var_358 = 3;
        wp::int32 var_359;
        bool var_360;
        const wp::int32 var_361 = 5;
        wp::float32 var_362;
        wp::shape_t* var_363;
        const wp::int32 var_364 = 0;
        wp::int32 var_365;
        wp::shape_t var_366;
        wp::int32 var_367;
        wp::vec_t<10, wp::float32>* var_368;
        wp::vec_t<10, wp::float32> var_369;
        wp::vec_t<10, wp::float32> var_370;
        const wp::int32 var_371 = 3;
        wp::float32 var_372;
        const wp::int32 var_373 = 4;
        wp::float32 var_374;
        const wp::int32 var_375 = 5;
        wp::float32 var_376;
        wp::float32* var_377;
        wp::float32 var_378;
        wp::float32 var_379;
        wp::float32 var_380;
        wp::float32 var_381;
        wp::float32 var_382;
        wp::float32 var_383;
        wp::float32 var_384;
        wp::float32 var_385;
        wp::shape_t* var_386;
        const wp::int32 var_387 = 0;
        wp::int32 var_388;
        wp::shape_t var_389;
        wp::int32 var_390;
        wp::float32* var_391;
        wp::float32 var_392;
        wp::float32 var_393;
        wp::float32 var_394;
        wp::float32 var_395;
        wp::float32 var_396;
        wp::float32 var_397;
        bool var_398;
        const wp::float32 var_399 = 1.0;
        wp::float32 var_400;
        wp::float32 var_401;
        wp::float32 var_402;
        wp::float32 var_403;
        wp::float32 var_404;
        wp::float32 var_405;
        wp::float32 var_406;
        wp::float32 var_407;
        wp::vec_t<10, wp::float32> var_408;
        wp::float32 var_409;
        wp::float32 var_410;
        wp::float32 var_411;
        wp::float32 var_412;
        wp::float32 var_413;
        const wp::int32 var_414 = 1;
        wp::int32 var_415;
        bool var_416;
        wp::shape_t* var_417;
        const wp::int32 var_418 = 0;
        wp::int32 var_419;
        wp::shape_t var_420;
        wp::int32 var_421;
        wp::float32* var_422;
        wp::float32 var_423;
        wp::float32 var_424;
        wp::float32 var_425;
        const wp::int32 var_426 = 8;
        wp::float32 var_427;
        const wp::float32 var_428 = 0.0;
        bool var_429;
        wp::float32 var_430;
        wp::float32 var_431;
        wp::float32 var_432;
        wp::float32 var_433;
        wp::float32 var_434;
        wp::shape_t* var_435;
        const wp::int32 var_436 = 0;
        wp::int32 var_437;
        wp::shape_t var_438;
        wp::int32 var_439;
        wp::float32* var_440;
        wp::float32 var_441;
        wp::float32 var_442;
        wp::float32 var_443;
        wp::float32 var_444;
        wp::float32 var_445;
        wp::float32 var_446;
        wp::float32 var_447;
        wp::float32 var_448;
        wp::float32 var_449;
        wp::vec_t<10, wp::float32> var_450;
        wp::float32 var_451;
        wp::float32 var_452;
        wp::float32 var_453;
        wp::float32 var_454;
        wp::float32 var_455;
        bool* var_456;
        bool var_457;
        wp::shape_t* var_458;
        const wp::int32 var_459 = 0;
        wp::int32 var_460;
        wp::shape_t var_461;
        wp::int32 var_462;
        wp::vec_t<2, wp::float32>* var_463;
        wp::vec_t<2, wp::float32> var_464;
        wp::vec_t<2, wp::float32> var_465;
        const wp::int32 var_466 = 0;
        wp::float32 var_467;
        const wp::int32 var_468 = 1;
        wp::float32 var_469;
        wp::float32 var_470;
        bool var_471;
        wp::float32 var_472;
        bool var_473;
        wp::float32 var_474;
        wp::vec_t<10, wp::float32> var_475;
        wp::vec_t<6, wp::int32> var_476;
        wp::float32 var_477;
        wp::float32 var_478;
        wp::float32 var_479;
        wp::vec_t<10, wp::float32> var_480;
        wp::float32 var_481;
        wp::float32 var_482;
        wp::float32 var_483;
        wp::float32 var_484;
        wp::float32 var_485;
        wp::shape_t* var_486;
        const wp::int32 var_487 = 0;
        wp::int32 var_488;
        wp::shape_t var_489;
        wp::int32 var_490;
        wp::float32* var_491;
        wp::shape_t* var_492;
        const wp::int32 var_493 = 0;
        wp::int32 var_494;
        wp::shape_t var_495;
        wp::int32 var_496;
        wp::vec_t<2, wp::float32>* var_497;
        const wp::float32 var_498 = 1.0;
        bool* var_499;
        wp::float32 var_500;
        wp::float32 var_501;
        wp::vec_t<2, wp::float32> var_502;
        bool var_503;
        wp::float32 var_504;
        bool var_505;
        wp::float32 var_506;
        bool var_507;
        wp::float32 var_508;
        bool var_509;
        wp::vec_t<10, wp::float32> var_510;
        bool var_511;
        wp::vec_t<6, wp::int32> var_512;
        bool var_513;
        wp::float32 var_514;
        bool var_515;
        wp::float32 var_516;
        bool var_517;
        wp::float32 var_518;
        bool var_519;
        wp::vec_t<10, wp::float32> var_520;
        bool var_521;
        wp::float32 var_522;
        bool var_523;
        wp::float32 var_524;
        bool var_525;
        wp::float32 var_526;
        bool var_527;
        wp::float32 var_528;
        bool var_529;
        wp::float32 var_530;
        bool var_531;
        bool var_532;
        wp::float32* var_533;
        wp::float32 var_534;
        wp::float32 var_535;
        bool var_536;
        wp::float32 var_537;
        bool var_538;
        wp::float32 var_539;
        wp::float32 var_540;
        wp::float32* var_541;
        wp::float32 var_542;
        wp::float32 var_543;
        wp::float32* var_544;
        wp::float32 var_545;
        wp::float32 var_546;
        wp::int32* var_547;
        wp::int32 var_548;
        wp::int32 var_549;
        wp::shape_t* var_550;
        const wp::int32 var_551 = 0;
        wp::int32 var_552;
        wp::shape_t var_553;
        wp::int32 var_554;
        wp::vec_t<10, wp::float32>* var_555;
        wp::vec_t<10, wp::float32> var_556;
        wp::vec_t<10, wp::float32> var_557;
        const wp::float32 var_558 = 0.0;
        const wp::int32 var_559 = 0;
        bool var_560;
        const wp::int32 var_561 = 0;
        wp::float32 var_562;
        wp::float32 var_563;
        const wp::int32 var_564 = 1;
        bool var_565;
        const wp::int32 var_566 = 0;
        wp::float32 var_567;
        const wp::int32 var_568 = 1;
        wp::float32 var_569;
        wp::float32 var_570;
        wp::float32 var_571;
        const wp::int32 var_572 = 2;
        wp::float32 var_573;
        wp::float32 var_574;
        wp::float32 var_575;
        wp::float32 var_576;
        const wp::int32 var_577 = 2;
        bool var_578;
        wp::shape_t* var_579;
        const wp::int32 var_580 = 0;
        wp::int32 var_581;
        wp::shape_t var_582;
        wp::int32 var_583;
        wp::float32* var_584;
        wp::float32 var_585;
        wp::float32 var_586;
        wp::shape_t* var_587;
        const wp::int32 var_588 = 0;
        wp::int32 var_589;
        wp::shape_t var_590;
        wp::int32 var_591;
        wp::vec_t<2, wp::float32>* var_592;
        wp::vec_t<2, wp::float32> var_593;
        wp::vec_t<2, wp::float32> var_594;
        wp::float32 var_595;
        wp::float32 var_596;
        const wp::int32 var_597 = 3;
        bool var_598;
        const wp::int32 var_599 = 0;
        wp::float32 var_600;
        const wp::int32 var_601 = 1;
        wp::float32 var_602;
        const wp::int32 var_603 = 0;
        wp::float32 var_604;
        wp::vec_t<6, wp::int32> var_605;
        wp::int32 var_606;
        const wp::int32 var_607 = 2;
        wp::int32 var_608;
        const wp::int32 var_609 = 0;
        bool var_610;
        const wp::int32 var_611 = 2;
        wp::int32 var_612;
        wp::int32 var_613;
        wp::float32* var_614;
        wp::float32 var_615;
        wp::float32 var_616;
        const wp::int32 var_617 = 2;
        wp::float32 var_618;
        const wp::int32 var_619 = 3;
        wp::float32 var_620;
        const wp::int32 var_621 = 4;
        wp::float32 var_622;
        const wp::float32 var_623 = 1.0;
        wp::float32 var_624;
        wp::float32 var_625;
        wp::float32 var_626;
        wp::float32 var_627;
        wp::float32 var_628;
        wp::float32 var_629;
        wp::float32 var_630;
        wp::float32 var_631;
        wp::float32 var_632;
        wp::float32 var_633;
        const wp::float32 var_634 = 0.0;
        bool var_635;
        wp::float32 var_636;
        wp::float32 var_637;
        wp::float32 var_638;
        const wp::float32 var_639 = 0.0;
        bool var_640;
        const wp::int32 var_641 = 8;
        wp::float32 var_642;
        wp::int32 var_643;
        const wp::int32 var_644 = 0;
        bool var_645;
        const wp::float32 var_646 = 0.0;
        const wp::int32 var_647 = 1;
        wp::int32 var_648;
        const wp::int32 var_649 = 0;
        bool var_650;
        const wp::int32 var_651 = 1;
        wp::int32 var_652;
        wp::int32 var_653;
        wp::float32* var_654;
        wp::float32 var_655;
        wp::float32 var_656;
        wp::float32 var_657;
        wp::float32 var_658;
        wp::float32 var_659;
        wp::float32 var_660;
        wp::float32 var_661;
        wp::float32 var_662;
        wp::float32 var_663;
        wp::float32 var_664;
        wp::int32 var_665;
        wp::float32 var_666;
        wp::vec_t<6, wp::int32> var_667;
        wp::int32 var_668;
        wp::float32 var_669;
        wp::int32 var_670;
        wp::float32 var_671;
        wp::float32 var_672;
        wp::float32 var_673;
        wp::float32 var_674;
        wp::float32 var_675;
        wp::float32 var_676;
        wp::float32 var_677;
        wp::float32 var_678;
        wp::float32 var_679;
        wp::vec_t<6, wp::int32> var_680;
        wp::int32 var_681;
        wp::float32 var_682;
        wp::int32 var_683;
        wp::float32 var_684;
        wp::float32 var_685;
        wp::float32 var_686;
        wp::float32 var_687;
        wp::float32 var_688;
        wp::float32 var_689;
        wp::float32 var_690;
        wp::float32 var_691;
        wp::float32 var_692;
        wp::vec_t<6, wp::int32> var_693;
        wp::int32 var_694;
        wp::float32 var_695;
        wp::int32 var_696;
        wp::float32 var_697;
        wp::float32 var_698;
        wp::float32 var_699;
        wp::float32 var_700;
        wp::float32 var_701;
        wp::float32 var_702;
        wp::float32 var_703;
        wp::float32 var_704;
        wp::float32 var_705;
        wp::vec_t<6, wp::int32> var_706;
        wp::int32 var_707;
        wp::float32 var_708;
        wp::int32 var_709;
        wp::float32 var_710;
        wp::float32 var_711;
        wp::float32 var_712;
        wp::float32 var_713;
        wp::float32 var_714;
        wp::float32 var_715;
        wp::float32 var_716;
        wp::float32 var_717;
        wp::int32* var_718;
        wp::int32 var_719;
        wp::int32 var_720;
        wp::shape_t* var_721;
        const wp::int32 var_722 = 0;
        wp::int32 var_723;
        wp::shape_t var_724;
        wp::int32 var_725;
        wp::vec_t<10, wp::float32>* var_726;
        wp::vec_t<10, wp::float32> var_727;
        wp::vec_t<10, wp::float32> var_728;
        const wp::float32 var_729 = 0.0;
        const wp::int32 var_730 = 1;
        bool var_731;
        const wp::int32 var_732 = 0;
        wp::float32 var_733;
        const wp::int32 var_734 = 1;
        wp::float32 var_735;
        wp::float32 var_736;
        wp::float32 var_737;
        const wp::int32 var_738 = 2;
        wp::float32 var_739;
        wp::float32 var_740;
        wp::float32 var_741;
        wp::float32 var_742;
        const wp::int32 var_743 = 2;
        bool var_744;
        wp::shape_t* var_745;
        const wp::int32 var_746 = 0;
        wp::int32 var_747;
        wp::shape_t var_748;
        wp::int32 var_749;
        wp::float32* var_750;
        wp::float32 var_751;
        wp::float32 var_752;
        wp::shape_t* var_753;
        const wp::int32 var_754 = 0;
        wp::int32 var_755;
        wp::shape_t var_756;
        wp::int32 var_757;
        wp::vec_t<2, wp::float32>* var_758;
        wp::vec_t<2, wp::float32> var_759;
        wp::vec_t<2, wp::float32> var_760;
        wp::float32 var_761;
        wp::float32 var_762;
        wp::vec_t<2, wp::float32> var_763;
        wp::float32 var_764;
        const wp::int32 var_765 = 3;
        bool var_766;
        const wp::int32 var_767 = 0;
        wp::float32 var_768;
        const wp::float32 var_769 = 0.0;
        bool var_770;
        const wp::int32 var_771 = 1;
        wp::float32 var_772;
        wp::float32 var_773;
        wp::float32 var_774;
        wp::float32 var_775;
        wp::float32 var_776;
        wp::float32 var_777;
        wp::float32 var_778;
        wp::float32 var_779;
        wp::float32 var_780;
        wp::float32 var_781;
        wp::float32 var_782;
        wp::float32 var_783;
        wp::vec_t<2, wp::float32> var_784;
        wp::float32 var_785;
        wp::float32 var_786;
        wp::float32 var_787;
        bool* var_788;
        bool var_789;
        wp::shape_t* var_790;
        const wp::int32 var_791 = 0;
        wp::int32 var_792;
        wp::shape_t var_793;
        wp::int32 var_794;
        wp::vec_t<2, wp::float32>* var_795;
        wp::vec_t<2, wp::float32> var_796;
        wp::vec_t<2, wp::float32> var_797;
        const wp::int32 var_798 = 0;
        wp::float32 var_799;
        const wp::int32 var_800 = 1;
        wp::float32 var_801;
        wp::float32 var_802;
        bool var_803;
        wp::float32 var_804;
        bool var_805;
        const wp::int32 var_806 = 3;
        bool var_807;
        const wp::int32 var_808 = 0;
        wp::float32 var_809;
        const wp::float32 var_810 = 0.0;
        bool var_811;
        const wp::int32 var_812 = 1;
        wp::float32 var_813;
        const wp::int32 var_814 = 2;
        wp::float32 var_815;
        wp::float32 var_816;
        wp::float32 var_817;
        wp::float32 var_818;
        wp::float32 var_819;
        wp::float32 var_820;
        wp::float32 var_821;
        const wp::int32 var_822 = 5;
        wp::float32 var_823;
        const wp::float32 var_824 = 0.0;
        bool var_825;
        const wp::int32 var_826 = 6;
        wp::float32 var_827;
        wp::vec_t<6, wp::int32> var_828;
        const wp::int32 var_829 = 3;
        wp::int32 var_830;
        wp::int32 var_831;
        wp::float32* var_832;
        wp::float32 var_833;
        wp::float32 var_834;
        wp::float32* var_835;
        wp::float32 var_836;
        wp::float32 var_837;
        wp::float32 var_838;
        wp::float32 var_839;
        wp::float32 var_840;
        wp::float32 var_841;
        wp::vec_t<6, wp::int32> var_842;
        wp::int32 var_843;
        wp::float32 var_844;
        wp::float32 var_845;
        wp::vec_t<6, wp::int32> var_846;
        wp::int32 var_847;
        wp::float32 var_848;
        wp::float32 var_849;
        wp::float32 var_850;
        //---------
        // forward
        // def _actuator_force(                                                                   <L 757>
        // worldid, uid = wp.tid()                                                                <L 789>
        builtin_tid2d(var_0, var_1);
        // actuator_ctrlrange_id = worldid % actuator_ctrlrange.shape[0]                          <L 791>
        var_2 = &(var_actuator_ctrlrange.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // ctrl = ctrl_in[worldid, uid]                                                           <L 793>
        var_7 = wp::address(var_ctrl_in, var_0, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // if actuator_ctrllimited[uid] and not dsbl_clampctrl:                                   <L 795>
        var_11 = wp::address(var_actuator_ctrllimited, var_1);
        var_12 = wp::load(var_11);
        var_10 = var_12;
        if (var_10) {
            var_13 = wp::unot(var_dsbl_clampctrl);
            var_10 = var_10 && var_13;
        }
        if (var_10) {
            // ctrlrange = actuator_ctrlrange[actuator_ctrlrange_id, uid]                         <L 796>
            var_14 = wp::address(var_actuator_ctrlrange, var_6, var_1);
            var_16 = wp::load(var_14);
            var_15 = wp::copy(var_16);
            // ctrl = wp.clamp(ctrl, ctrlrange[0], ctrlrange[1])                                  <L 797>
            var_18 = wp::extract(var_15, var_17);
            var_20 = wp::extract(var_15, var_19);
            var_21 = wp::clamp(var_8, var_18, var_20);
        }
        var_22 = wp::where(var_10, var_21, var_8);
        // ctrl_act = ctrl                                                                        <L 798>
        var_23 = wp::copy(var_22);
        // act_first = actuator_actadr[uid]                                                       <L 800>
        var_24 = wp::address(var_actuator_actadr, var_1);
        var_26 = wp::load(var_24);
        var_25 = wp::copy(var_26);
        // if na and act_first >= 0:                                                              <L 801>
        var_27 = var_na;
        if (var_27) {
            var_29 = (var_25 >= var_28);
            var_27 = var_27 && var_29;
        }
        if (var_27) {
            // act_last = act_first + actuator_actnum[uid] - 1                                    <L 802>
            var_30 = wp::address(var_actuator_actnum, var_1);
            var_32 = wp::load(var_30);
            var_31 = wp::add(var_25, var_32);
            var_34 = wp::sub(var_31, var_33);
            // dyntype = actuator_dyntype[uid]                                                    <L 803>
            var_35 = wp::address(var_actuator_dyntype, var_1);
            var_37 = wp::load(var_35);
            var_36 = wp::copy(var_37);
            // dynprm = actuator_dynprm[worldid % actuator_dynprm.shape[0], uid]                  <L 804>
            var_38 = &(var_actuator_dynprm.shape);
            var_41 = wp::load(var_38);
            var_40 = wp::extract(var_41, var_39);
            var_42 = wp::mod(var_0, var_40);
            var_43 = wp::address(var_actuator_dynprm, var_42, var_1);
            var_45 = wp::load(var_43);
            var_44 = wp::copy(var_45);
            // if dyntype == DynType.INTEGRATOR:                                                  <L 806>
            var_47 = (var_36 == var_46);
            if (var_47) {
                // act_dot = ctrl                                                                 <L 807>
                var_48 = wp::copy(var_22);
            }
            if (!var_47) {
                // elif dyntype == DynType.FILTER or dyntype == DynType.FILTEREXACT:              <L 808>
                var_51 = (var_36 == var_50);
                var_49 = var_51;
                if (!var_49) {
                    var_53 = (var_36 == var_52);
                    var_49 = var_49 || var_53;
                }
                if (var_49) {
                    // act = act_in[worldid, act_last]                                            <L 809>
                    var_54 = wp::address(var_act_in, var_0, var_34);
                    var_56 = wp::load(var_54);
                    var_55 = wp::copy(var_56);
                    // act_dot = (ctrl - act) / wp.max(dynprm[0], MJ_MINVAL)                      <L 810>
                    var_57 = wp::sub(var_22, var_55);
                    var_59 = wp::extract(var_44, var_58);
                    var_61 = wp::max(var_59, var_60);
                    var_62 = wp::div(var_57, var_61);
                }
                var_63 = wp::where(var_49, var_62, var_48);
                if (!var_49) {
                    // elif dyntype == DynType.MUSCLE:                                            <L 811>
                    var_65 = (var_36 == var_64);
                    if (var_65) {
                        // dynprm = actuator_dynprm[worldid % actuator_dynprm.shape[0], uid]       <L 812>
                        var_66 = &(var_actuator_dynprm.shape);
                        var_69 = wp::load(var_66);
                        var_68 = wp::extract(var_69, var_67);
                        var_70 = wp::mod(var_0, var_68);
                        var_71 = wp::address(var_actuator_dynprm, var_70, var_1);
                        var_73 = wp::load(var_71);
                        var_72 = wp::copy(var_73);
                        // act = act_in[worldid, act_last]                                        <L 813>
                        var_74 = wp::address(var_act_in, var_0, var_34);
                        var_76 = wp::load(var_74);
                        var_75 = wp::copy(var_76);
                        // act_dot = util_misc.muscle_dynamics(ctrl, act, dynprm)                 <L 814>
                        var_77 = muscle_dynamics_0(var_22, var_75, var_72);
                    }
                    var_78 = wp::where(var_65, var_72, var_44);
                    var_79 = wp::where(var_65, var_77, var_63);
                    var_80 = wp::where(var_65, var_75, var_55);
                    if (!var_65) {
                        // elif dyntype == DynType.DCMOTOR:                                       <L 815>
                        var_82 = (var_36 == var_81);
                        if (var_82) {
                            // gainprm = actuator_gainprm[worldid % actuator_gainprm.shape[0], uid]       <L 816>
                            var_83 = &(var_actuator_gainprm.shape);
                            var_86 = wp::load(var_83);
                            var_85 = wp::extract(var_86, var_84);
                            var_87 = wp::mod(var_0, var_85);
                            var_88 = wp::address(var_actuator_gainprm, var_87, var_1);
                            var_90 = wp::load(var_88);
                            var_89 = wp::copy(var_90);
                            // slots = util_misc.dcmotor_slots(dynprm, gainprm)                   <L 817>
                            var_91 = dcmotor_slots_0(var_78, var_89);
                            // adr = act_first                                                    <L 818>
                            var_92 = wp::copy(var_25);
                            // act_dot = 0.0                                                      <L 820>
                            // if slots[0] >= 0:                                                  <L 823>
                            var_95 = wp::extract(var_91, var_94);
                            var_97 = (var_95 >= var_96);
                            if (var_97) {
                                // u_prev = act_in[worldid, adr]                                  <L 824>
                                var_98 = wp::address(var_act_in, var_0, var_92);
                                var_100 = wp::load(var_98);
                                var_99 = wp::copy(var_100);
                                // slew_s = dynprm[7]                                             <L 825>
                                var_102 = wp::extract(var_78, var_101);
                                // slew = slew_s * opt_timestep[worldid % opt_timestep.shape[0]]       <L 826>
                                var_103 = &(var_opt_timestep.shape);
                                var_106 = wp::load(var_103);
                                var_105 = wp::extract(var_106, var_104);
                                var_107 = wp::mod(var_0, var_105);
                                var_108 = wp::address(var_opt_timestep, var_107);
                                var_110 = wp::load(var_108);
                                var_109 = wp::mul(var_102, var_110);
                                // u_eff = wp.clamp(ctrl, u_prev - slew, u_prev + slew)           <L 827>
                                var_111 = wp::sub(var_99, var_109);
                                var_112 = wp::add(var_99, var_109);
                                var_113 = wp::clamp(var_22, var_111, var_112);
                                // act_dot = (u_eff - u_prev) / opt_timestep[worldid % opt_timestep.shape[0]]       <L 828>
                                var_114 = wp::sub(var_113, var_99);
                                var_115 = &(var_opt_timestep.shape);
                                var_118 = wp::load(var_115);
                                var_117 = wp::extract(var_118, var_116);
                                var_119 = wp::mod(var_0, var_117);
                                var_120 = wp::address(var_opt_timestep, var_119);
                                var_122 = wp::load(var_120);
                                var_121 = wp::div(var_114, var_122);
                                // act_dot_out[worldid, adr] = act_dot                            <L 829>
                                wp::array_store(var_act_dot_out, var_0, var_92, var_121);
                                // ctrl = u_eff                                                   <L 830>
                                var_123 = wp::copy(var_113);
                                // adr += 1                                                       <L 831>
                                var_125 = wp::add(var_92, var_124);
                            }
                            var_126 = wp::where(var_97, var_123, var_22);
                            var_127 = wp::where(var_97, var_121, var_93);
                            var_128 = wp::where(var_97, var_125, var_92);
                            // x_I = 0.0                                                          <L 834>
                            // if slots[1] >= 0:                                                  <L 835>
                            var_131 = wp::extract(var_91, var_130);
                            var_133 = (var_131 >= var_132);
                            if (var_133) {
                                // x_I = act_in[worldid, adr]                                     <L 836>
                                var_134 = wp::address(var_act_in, var_0, var_128);
                                var_136 = wp::load(var_134);
                                var_135 = wp::copy(var_136);
                                // input_mode = int(gainprm[8])                                   <L 837>
                                var_138 = wp::extract(var_89, var_137);
                                var_139 = wp::int(var_138);
                                // Imax = dynprm[8]                                               <L 838>
                                var_141 = wp::extract(var_78, var_140);
                                // act_dot = ctrl                                                 <L 839>
                                var_142 = wp::copy(var_126);
                                // if input_mode == 1:                                            <L 840>
                                var_144 = (var_139 == var_143);
                                if (var_144) {
                                    // act_dot = ctrl - actuator_length_in[worldid, uid]          <L 841>
                                    var_145 = wp::address(var_actuator_length_in, var_0, var_1);
                                    var_147 = wp::load(var_145);
                                    var_146 = wp::sub(var_126, var_147);
                                }
                                var_148 = wp::where(var_144, var_146, var_142);
                                // if Imax > 0.0:                                                 <L 843>
                                var_150 = (var_141 > var_149);
                                if (var_150) {
                                    // if x_I >= Imax:                                            <L 844>
                                    var_151 = (var_135 >= var_141);
                                    if (var_151) {
                                        // act_dot = wp.min(act_dot, 0.0)                         <L 845>
                                        var_153 = wp::min(var_148, var_152);
                                    }
                                    var_154 = wp::where(var_151, var_153, var_148);
                                    if (!var_151) {
                                        // elif x_I <= -Imax:                                     <L 846>
                                        var_155 = wp::neg(var_141);
                                        var_156 = (var_135 <= var_155);
                                        if (var_156) {
                                            // act_dot = wp.max(act_dot, 0.0)                     <L 847>
                                            var_158 = wp::max(var_154, var_157);
                                        }
                                        var_159 = wp::where(var_156, var_158, var_154);
                                    }
                                    var_160 = wp::where(var_151, var_154, var_159);
                                }
                                var_161 = wp::where(var_150, var_160, var_148);
                                // act_dot_out[worldid, adr] = act_dot                            <L 849>
                                wp::array_store(var_act_dot_out, var_0, var_128, var_161);
                                // adr += 1                                                       <L 850>
                                var_163 = wp::add(var_128, var_162);
                            }
                            var_164 = wp::where(var_133, var_161, var_127);
                            var_165 = wp::where(var_133, var_163, var_128);
                            var_166 = wp::where(var_133, var_135, var_129);
                            // V = util_misc.dcmotor_voltage(                                     <L 853>
                            // ctrl,                                                              <L 854>
                            // actuator_length_in[worldid, uid],                                  <L 855>
                            var_167 = wp::address(var_actuator_length_in, var_0, var_1);
                            // actuator_velocity_in[worldid, uid],                                <L 856>
                            var_168 = wp::address(var_actuator_velocity_in, var_0, var_1);
                            // x_I,                                                               <L 857>
                            // gainprm,                                                           <L 858>
                            var_170 = wp::load(var_167);
                            var_171 = wp::load(var_168);
                            var_169 = dcmotor_voltage_0(var_126, var_170, var_171, var_166, var_89);
                            // R = gainprm[0]                                                     <L 862>
                            var_173 = wp::extract(var_89, var_172);
                            // K = gainprm[1]                                                     <L 863>
                            var_175 = wp::extract(var_89, var_174);
                            // te = wp.max(MJ_MINVAL, dynprm[0])                                  <L 864>
                            var_177 = wp::extract(var_78, var_176);
                            var_178 = wp::max(var_60, var_177);
                            // if slots[2] >= 0:                                                  <L 866>
                            var_180 = wp::extract(var_91, var_179);
                            var_182 = (var_180 >= var_181);
                            if (var_182) {
                                // RT = dynprm[2]                                                 <L 867>
                                var_184 = wp::extract(var_78, var_183);
                                // C = dynprm[3]                                                  <L 868>
                                var_186 = wp::extract(var_78, var_185);
                                // Ta = dynprm[4]                                                 <L 869>
                                var_188 = wp::extract(var_78, var_187);
                                // alpha = gainprm[2]                                             <L 870>
                                var_190 = wp::extract(var_89, var_189);
                                // T0 = gainprm[3]                                                <L 871>
                                var_192 = wp::extract(var_89, var_191);
                                // T = act_in[worldid, adr]                                       <L 872>
                                var_193 = wp::address(var_act_in, var_0, var_165);
                                var_195 = wp::load(var_193);
                                var_194 = wp::copy(var_195);
                                // R_eff = R * (1.0 + alpha * (T + Ta - T0))                      <L 873>
                                var_197 = wp::add(var_194, var_188);
                                var_198 = wp::sub(var_197, var_192);
                                var_199 = wp::mul(var_190, var_198);
                                var_200 = wp::add(var_196, var_199);
                                var_201 = wp::mul(var_173, var_200);
                                // current = (V - K * actuator_velocity_in[worldid, uid]) / R_eff       <L 875>
                                var_202 = wp::address(var_actuator_velocity_in, var_0, var_1);
                                var_204 = wp::load(var_202);
                                var_203 = wp::mul(var_175, var_204);
                                var_205 = wp::sub(var_169, var_203);
                                var_206 = wp::div(var_205, var_201);
                                // if slots[4] >= 0:                                              <L 876>
                                var_208 = wp::extract(var_91, var_207);
                                var_210 = (var_208 >= var_209);
                                if (var_210) {
                                    // current = act_in[worldid, act_last]                        <L 877>
                                    var_211 = wp::address(var_act_in, var_0, var_34);
                                    var_213 = wp::load(var_211);
                                    var_212 = wp::copy(var_213);
                                }
                                var_214 = wp::where(var_210, var_212, var_206);
                                // act_dot = (R_eff * current * current - T / RT) / C             <L 879>
                                var_215 = wp::mul(var_201, var_214);
                                var_216 = wp::mul(var_215, var_214);
                                var_217 = wp::div(var_194, var_184);
                                var_218 = wp::sub(var_216, var_217);
                                var_219 = wp::div(var_218, var_186);
                                // act_dot_out[worldid, adr] = act_dot                            <L 880>
                                wp::array_store(var_act_dot_out, var_0, var_165, var_219);
                                // adr += 1                                                       <L 881>
                                var_221 = wp::add(var_165, var_220);
                                // R = R_eff                                                      <L 882>
                                var_222 = wp::copy(var_201);
                            }
                            var_223 = wp::where(var_182, var_219, var_164);
                            var_224 = wp::where(var_182, var_221, var_165);
                            var_225 = wp::where(var_182, var_222, var_173);
                            // if slots[3] >= 0:                                                  <L 885>
                            var_227 = wp::extract(var_91, var_226);
                            var_229 = (var_227 >= var_228);
                            if (var_229) {
                                // sigma0 = dynprm[5]                                             <L 886>
                                var_231 = wp::extract(var_78, var_230);
                                // biasprm = actuator_biasprm[worldid % actuator_biasprm.shape[0], uid]       <L 887>
                                var_232 = &(var_actuator_biasprm.shape);
                                var_235 = wp::load(var_232);
                                var_234 = wp::extract(var_235, var_233);
                                var_236 = wp::mod(var_0, var_234);
                                var_237 = wp::address(var_actuator_biasprm, var_236, var_1);
                                var_239 = wp::load(var_237);
                                var_238 = wp::copy(var_239);
                                // F_C = biasprm[3]                                               <L 888>
                                var_241 = wp::extract(var_238, var_240);
                                // F_S = biasprm[4]                                               <L 889>
                                var_243 = wp::extract(var_238, var_242);
                                // v_S = biasprm[5]                                               <L 890>
                                var_245 = wp::extract(var_238, var_244);
                                // z = act_in[worldid, adr]                                       <L 891>
                                var_246 = wp::address(var_act_in, var_0, var_224);
                                var_248 = wp::load(var_246);
                                var_247 = wp::copy(var_248);
                                // g = util_misc.lugre_stribeck(actuator_velocity_in[worldid, uid], F_C, F_S, v_S)       <L 892>
                                var_249 = wp::address(var_actuator_velocity_in, var_0, var_1);
                                var_251 = wp::load(var_249);
                                var_250 = lugre_stribeck_0(var_251, var_241, var_243, var_245);
                                // a = -sigma0 * wp.abs(actuator_velocity_in[worldid, uid]) / wp.max(MJ_MINVAL, g)       <L 893>
                                var_252 = wp::neg(var_231);
                                var_253 = wp::address(var_actuator_velocity_in, var_0, var_1);
                                var_255 = wp::load(var_253);
                                var_254 = wp::abs(var_255);
                                var_256 = wp::mul(var_252, var_254);
                                var_257 = wp::max(var_60, var_250);
                                var_258 = wp::div(var_256, var_257);
                                // act_dot = a * z + actuator_velocity_in[worldid, uid]           <L 894>
                                var_259 = wp::mul(var_258, var_247);
                                var_260 = wp::address(var_actuator_velocity_in, var_0, var_1);
                                var_262 = wp::load(var_260);
                                var_261 = wp::add(var_259, var_262);
                                // act_dot_out[worldid, adr] = act_dot                            <L 895>
                                wp::array_store(var_act_dot_out, var_0, var_224, var_261);
                                // adr += 1                                                       <L 896>
                                var_264 = wp::add(var_224, var_263);
                            }
                            var_265 = wp::where(var_229, var_261, var_223);
                            var_266 = wp::where(var_229, var_264, var_224);
                            // if slots[4] >= 0:                                                  <L 899>
                            var_268 = wp::extract(var_91, var_267);
                            var_270 = (var_268 >= var_269);
                            if (var_270) {
                                // dimax = dynprm[1]                                              <L 900>
                                var_272 = wp::extract(var_78, var_271);
                                // act_dot = (V / R - K / R * actuator_velocity_in[worldid, uid] - act_in[worldid, act_last]) / te       <L 901>
                                var_273 = wp::div(var_169, var_225);
                                var_274 = wp::div(var_175, var_225);
                                var_275 = wp::address(var_actuator_velocity_in, var_0, var_1);
                                var_277 = wp::load(var_275);
                                var_276 = wp::mul(var_274, var_277);
                                var_278 = wp::sub(var_273, var_276);
                                var_279 = wp::address(var_act_in, var_0, var_34);
                                var_281 = wp::load(var_279);
                                var_280 = wp::sub(var_278, var_281);
                                var_282 = wp::div(var_280, var_178);
                                // if dimax > 0.0:                                                <L 902>
                                var_284 = (var_272 > var_283);
                                if (var_284) {
                                    // act_dot = wp.clamp(act_dot, -dimax, dimax)                 <L 903>
                                    var_285 = wp::neg(var_272);
                                    var_286 = wp::clamp(var_282, var_285, var_272);
                                }
                                var_287 = wp::where(var_284, var_286, var_282);
                                // act_dot_out[worldid, act_last] = act_dot                       <L 904>
                                wp::array_store(var_act_dot_out, var_0, var_34, var_287);
                            }
                            var_288 = wp::where(var_270, var_287, var_265);
                        }
                        var_289 = wp::where(var_82, var_126, var_22);
                        var_290 = wp::where(var_82, var_288, var_79);
                        if (!var_82) {
                            // elif dyntype == DynType.USER:                                      <L 906>
                            var_292 = (var_36 == var_291);
                            if (var_292) {
                                // act_dot = 0.0  # set by act_dyn_callback                       <L 907>
                            }
                            var_294 = wp::where(var_292, var_293, var_290);
                            if (!var_292) {
                                // act_dot = 0.0                                                  <L 909>
                            }
                            var_296 = wp::where(var_292, var_294, var_295);
                        }
                        var_297 = wp::where(var_82, var_290, var_296);
                    }
                    var_298 = wp::where(var_65, var_22, var_289);
                    var_299 = wp::where(var_65, var_79, var_297);
                }
                var_300 = wp::where(var_49, var_22, var_298);
                var_301 = wp::where(var_49, var_44, var_78);
                var_302 = wp::where(var_49, var_63, var_299);
                var_303 = wp::where(var_49, var_55, var_80);
            }
            var_304 = wp::where(var_47, var_22, var_300);
            var_305 = wp::where(var_47, var_44, var_301);
            var_306 = wp::where(var_47, var_48, var_302);
            // act_dot_out[worldid, act_last] = act_dot                                           <L 911>
            wp::array_store(var_act_dot_out, var_0, var_34, var_306);
            // if actuator_actearly[uid]:                                                         <L 913>
            var_307 = wp::address(var_actuator_actearly, var_1);
            var_308 = wp::load(var_307);
            if (var_308) {
                // if dyntype == DynType.INTEGRATOR or dyntype == DynType.NONE or dyntype == DynType.DCMOTOR:       <L 914>
                var_311 = (var_36 == var_310);
                var_309 = var_311;
                if (!var_309) {
                    var_313 = (var_36 == var_312);
                    var_309 = var_309 || var_313;
                }
                if (!var_309) {
                    var_315 = (var_36 == var_314);
                    var_309 = var_309 || var_315;
                }
                if (var_309) {
                    // act = act_in[worldid, act_last]                                            <L 915>
                    var_316 = wp::address(var_act_in, var_0, var_34);
                    var_318 = wp::load(var_316);
                    var_317 = wp::copy(var_318);
                }
                var_319 = wp::where(var_309, var_317, var_303);
                // if dyntype == DynType.DCMOTOR:                                                 <L 917>
                var_321 = (var_36 == var_320);
                if (var_321) {
                    // gainprm = actuator_gainprm[worldid % actuator_gainprm.shape[0], uid]       <L 918>
                    var_322 = &(var_actuator_gainprm.shape);
                    var_325 = wp::load(var_322);
                    var_324 = wp::extract(var_325, var_323);
                    var_326 = wp::mod(var_0, var_324);
                    var_327 = wp::address(var_actuator_gainprm, var_326, var_1);
                    var_329 = wp::load(var_327);
                    var_328 = wp::copy(var_329);
                    // slots = util_misc.dcmotor_slots(dynprm, gainprm)                           <L 919>
                    var_330 = dcmotor_slots_0(var_305, var_328);
                    // offset = actuator_actnum[uid] - 1                                          <L 920>
                    var_331 = wp::address(var_actuator_actnum, var_1);
                    var_334 = wp::load(var_331);
                    var_333 = wp::sub(var_334, var_332);
                    // if offset == slots[4]:  # current                                          <L 922>
                    var_336 = wp::extract(var_330, var_335);
                    var_337 = (var_333 == var_336);
                    if (var_337) {
                        // te = wp.max(MJ_MINVAL, dynprm[0])                                      <L 923>
                        var_339 = wp::extract(var_305, var_338);
                        var_340 = wp::max(var_60, var_339);
                        // ctrl_act = act + act_dot * te * (1.0 - wp.exp(-opt_timestep[worldid % opt_timestep.shape[0]] / te))       <L 924>
                        var_341 = wp::mul(var_306, var_340);
                        var_343 = &(var_opt_timestep.shape);
                        var_346 = wp::load(var_343);
                        var_345 = wp::extract(var_346, var_344);
                        var_347 = wp::mod(var_0, var_345);
                        var_348 = wp::address(var_opt_timestep, var_347);
                        var_350 = wp::load(var_348);
                        var_349 = wp::neg(var_350);
                        var_351 = wp::div(var_349, var_340);
                        var_352 = wp::exp(var_351);
                        var_353 = wp::sub(var_342, var_352);
                        var_354 = wp::mul(var_341, var_353);
                        var_355 = wp::add(var_319, var_354);
                    }
                    var_356 = wp::where(var_337, var_355, var_23);
                    var_357 = wp::where(var_337, var_340, var_178);
                    if (!var_337) {
                        // elif offset == slots[3]:  # bristle                                    <L 925>
                        var_359 = wp::extract(var_330, var_358);
                        var_360 = (var_333 == var_359);
                        if (var_360) {
                            // sigma0 = dynprm[5]                                                 <L 926>
                            var_362 = wp::extract(var_305, var_361);
                            // biasprm = actuator_biasprm[worldid % actuator_biasprm.shape[0], uid]       <L 927>
                            var_363 = &(var_actuator_biasprm.shape);
                            var_366 = wp::load(var_363);
                            var_365 = wp::extract(var_366, var_364);
                            var_367 = wp::mod(var_0, var_365);
                            var_368 = wp::address(var_actuator_biasprm, var_367, var_1);
                            var_370 = wp::load(var_368);
                            var_369 = wp::copy(var_370);
                            // F_C = biasprm[3]                                                   <L 928>
                            var_372 = wp::extract(var_369, var_371);
                            // F_S = biasprm[4]                                                   <L 929>
                            var_374 = wp::extract(var_369, var_373);
                            // v_S = biasprm[5]                                                   <L 930>
                            var_376 = wp::extract(var_369, var_375);
                            // velocity = actuator_velocity_in[worldid, uid]                      <L 931>
                            var_377 = wp::address(var_actuator_velocity_in, var_0, var_1);
                            var_379 = wp::load(var_377);
                            var_378 = wp::copy(var_379);
                            // g = util_misc.lugre_stribeck(velocity, F_C, F_S, v_S)              <L 932>
                            var_380 = lugre_stribeck_0(var_378, var_372, var_374, var_376);
                            // a = -sigma0 * wp.abs(velocity) / wp.max(MJ_MINVAL, g)              <L 933>
                            var_381 = wp::neg(var_362);
                            var_382 = wp::abs(var_378);
                            var_383 = wp::mul(var_381, var_382);
                            var_384 = wp::max(var_60, var_380);
                            var_385 = wp::div(var_383, var_384);
                            // h = opt_timestep[worldid % opt_timestep.shape[0]]                  <L 934>
                            var_386 = &(var_opt_timestep.shape);
                            var_389 = wp::load(var_386);
                            var_388 = wp::extract(var_389, var_387);
                            var_390 = wp::mod(var_0, var_388);
                            var_391 = wp::address(var_opt_timestep, var_390);
                            var_393 = wp::load(var_391);
                            var_392 = wp::copy(var_393);
                            // exp_ah = wp.exp(a * h)                                             <L 935>
                            var_394 = wp::mul(var_385, var_392);
                            var_395 = wp::exp(var_394);
                            // int_h = h                                                          <L 936>
                            var_396 = wp::copy(var_392);
                            // if wp.abs(a) > MJ_MINVAL:                                          <L 937>
                            var_397 = wp::abs(var_385);
                            var_398 = (var_397 > var_60);
                            if (var_398) {
                                // int_h = (exp_ah - 1.0) / a                                     <L 938>
                                var_400 = wp::sub(var_395, var_399);
                                var_401 = wp::div(var_400, var_385);
                            }
                            var_402 = wp::where(var_398, var_401, var_396);
                            // ctrl_act = exp_ah * act + int_h * velocity                         <L 939>
                            var_403 = wp::mul(var_395, var_319);
                            var_404 = wp::mul(var_402, var_378);
                            var_405 = wp::add(var_403, var_404);
                        }
                        var_406 = wp::where(var_360, var_405, var_356);
                        var_407 = wp::where(var_360, var_362, var_231);
                        var_408 = wp::where(var_360, var_369, var_238);
                        var_409 = wp::where(var_360, var_372, var_241);
                        var_410 = wp::where(var_360, var_374, var_243);
                        var_411 = wp::where(var_360, var_376, var_245);
                        var_412 = wp::where(var_360, var_380, var_250);
                        var_413 = wp::where(var_360, var_385, var_258);
                        if (!var_360) {
                            // elif offset == slots[1]:  # integral                               <L 940>
                            var_415 = wp::extract(var_330, var_414);
                            var_416 = (var_333 == var_415);
                            if (var_416) {
                                // ctrl_act = act + act_dot * opt_timestep[worldid % opt_timestep.shape[0]]       <L 941>
                                var_417 = &(var_opt_timestep.shape);
                                var_420 = wp::load(var_417);
                                var_419 = wp::extract(var_420, var_418);
                                var_421 = wp::mod(var_0, var_419);
                                var_422 = wp::address(var_opt_timestep, var_421);
                                var_424 = wp::load(var_422);
                                var_423 = wp::mul(var_306, var_424);
                                var_425 = wp::add(var_319, var_423);
                                // Imax = dynprm[8]                                               <L 942>
                                var_427 = wp::extract(var_305, var_426);
                                // if Imax > 0.0:                                                 <L 943>
                                var_429 = (var_427 > var_428);
                                if (var_429) {
                                    // ctrl_act = wp.clamp(ctrl_act, -Imax, Imax)                 <L 944>
                                    var_430 = wp::neg(var_427);
                                    var_431 = wp::clamp(var_425, var_430, var_427);
                                }
                                var_432 = wp::where(var_429, var_431, var_425);
                            }
                            var_433 = wp::where(var_416, var_432, var_406);
                            var_434 = wp::where(var_416, var_427, var_141);
                            if (!var_416) {
                                // ctrl_act = act + act_dot * opt_timestep[worldid % opt_timestep.shape[0]]       <L 946>
                                var_435 = &(var_opt_timestep.shape);
                                var_438 = wp::load(var_435);
                                var_437 = wp::extract(var_438, var_436);
                                var_439 = wp::mod(var_0, var_437);
                                var_440 = wp::address(var_opt_timestep, var_439);
                                var_442 = wp::load(var_440);
                                var_441 = wp::mul(var_306, var_442);
                                var_443 = wp::add(var_319, var_441);
                            }
                            var_444 = wp::where(var_416, var_433, var_443);
                        }
                        var_445 = wp::where(var_360, var_406, var_444);
                        var_446 = wp::where(var_360, var_141, var_434);
                    }
                    var_447 = wp::where(var_337, var_356, var_445);
                    var_448 = wp::where(var_337, var_141, var_446);
                    var_449 = wp::where(var_337, var_231, var_407);
                    var_450 = wp::where(var_337, var_238, var_408);
                    var_451 = wp::where(var_337, var_241, var_409);
                    var_452 = wp::where(var_337, var_243, var_410);
                    var_453 = wp::where(var_337, var_245, var_411);
                    var_454 = wp::where(var_337, var_250, var_412);
                    var_455 = wp::where(var_337, var_258, var_413);
                    // if actuator_actlimited[uid]:                                               <L 948>
                    var_456 = wp::address(var_actuator_actlimited, var_1);
                    var_457 = wp::load(var_456);
                    if (var_457) {
                        // actrange = actuator_actrange[worldid % actuator_actrange.shape[0], uid]       <L 949>
                        var_458 = &(var_actuator_actrange.shape);
                        var_461 = wp::load(var_458);
                        var_460 = wp::extract(var_461, var_459);
                        var_462 = wp::mod(var_0, var_460);
                        var_463 = wp::address(var_actuator_actrange, var_462, var_1);
                        var_465 = wp::load(var_463);
                        var_464 = wp::copy(var_465);
                        // ctrl_act = wp.clamp(ctrl_act, actrange[0], actrange[1])                <L 950>
                        var_467 = wp::extract(var_464, var_466);
                        var_469 = wp::extract(var_464, var_468);
                        var_470 = wp::clamp(var_447, var_467, var_469);
                    }
                    var_471 = wp::load(var_456);
                    var_473 = wp::load(var_456);
                    var_472 = wp::where(var_473, var_470, var_447);
                }
                var_474 = wp::where(var_321, var_472, var_23);
                var_475 = wp::where(var_321, var_328, var_89);
                var_476 = wp::where(var_321, var_330, var_91);
                var_477 = wp::where(var_321, var_448, var_141);
                var_478 = wp::where(var_321, var_357, var_178);
                var_479 = wp::where(var_321, var_449, var_231);
                var_480 = wp::where(var_321, var_450, var_238);
                var_481 = wp::where(var_321, var_451, var_241);
                var_482 = wp::where(var_321, var_452, var_243);
                var_483 = wp::where(var_321, var_453, var_245);
                var_484 = wp::where(var_321, var_454, var_250);
                var_485 = wp::where(var_321, var_455, var_258);
                if (!var_321) {
                    // ctrl_act = next_act(                                                       <L 952>
                    // opt_timestep[worldid % opt_timestep.shape[0]],                             <L 953>
                    var_486 = &(var_opt_timestep.shape);
                    var_489 = wp::load(var_486);
                    var_488 = wp::extract(var_489, var_487);
                    var_490 = wp::mod(var_0, var_488);
                    var_491 = wp::address(var_opt_timestep, var_490);
                    // dyntype,                                                                   <L 954>
                    // dynprm,                                                                    <L 955>
                    // actuator_actrange[worldid % actuator_actrange.shape[0], uid],              <L 956>
                    var_492 = &(var_actuator_actrange.shape);
                    var_495 = wp::load(var_492);
                    var_494 = wp::extract(var_495, var_493);
                    var_496 = wp::mod(var_0, var_494);
                    var_497 = wp::address(var_actuator_actrange, var_496, var_1);
                    // act,                                                                       <L 957>
                    // act_dot,                                                                   <L 958>
                    // 1.0,                                                                       <L 959>
                    // actuator_actlimited[uid],                                                  <L 960>
                    var_499 = wp::address(var_actuator_actlimited, var_1);
                    var_501 = wp::load(var_491);
                    var_502 = wp::load(var_497);
                    var_503 = wp::load(var_499);
                    var_500 = next_act_0(var_501, var_36, var_305, var_502, var_319, var_306, var_498, var_503);
                }
                var_504 = wp::where(var_321, var_474, var_500);
            }
            var_505 = wp::load(var_307);
            var_507 = wp::load(var_307);
            var_506 = wp::where(var_507, var_504, var_23);
            var_509 = wp::load(var_307);
            var_508 = wp::where(var_509, var_319, var_303);
            var_511 = wp::load(var_307);
            var_510 = wp::where(var_511, var_475, var_89);
            var_513 = wp::load(var_307);
            var_512 = wp::where(var_513, var_476, var_91);
            var_515 = wp::load(var_307);
            var_514 = wp::where(var_515, var_477, var_141);
            var_517 = wp::load(var_307);
            var_516 = wp::where(var_517, var_478, var_178);
            var_519 = wp::load(var_307);
            var_518 = wp::where(var_519, var_479, var_231);
            var_521 = wp::load(var_307);
            var_520 = wp::where(var_521, var_480, var_238);
            var_523 = wp::load(var_307);
            var_522 = wp::where(var_523, var_481, var_241);
            var_525 = wp::load(var_307);
            var_524 = wp::where(var_525, var_482, var_243);
            var_527 = wp::load(var_307);
            var_526 = wp::where(var_527, var_483, var_245);
            var_529 = wp::load(var_307);
            var_528 = wp::where(var_529, var_484, var_250);
            var_531 = wp::load(var_307);
            var_530 = wp::where(var_531, var_485, var_258);
            var_532 = wp::load(var_307);
            if (!var_532) {
                // ctrl_act = act_in[worldid, act_last]                                           <L 963>
                var_533 = wp::address(var_act_in, var_0, var_34);
                var_535 = wp::load(var_533);
                var_534 = wp::copy(var_535);
            }
            var_536 = wp::load(var_307);
            var_538 = wp::load(var_307);
            var_537 = wp::where(var_538, var_506, var_534);
        }
        var_539 = wp::where(var_27, var_304, var_22);
        var_540 = wp::where(var_27, var_537, var_23);
        // length = actuator_length_in[worldid, uid]                                              <L 965>
        var_541 = wp::address(var_actuator_length_in, var_0, var_1);
        var_543 = wp::load(var_541);
        var_542 = wp::copy(var_543);
        // velocity = actuator_velocity_in[worldid, uid]                                          <L 966>
        var_544 = wp::address(var_actuator_velocity_in, var_0, var_1);
        var_546 = wp::load(var_544);
        var_545 = wp::copy(var_546);
        // gaintype = actuator_gaintype[uid]                                                      <L 969>
        var_547 = wp::address(var_actuator_gaintype, var_1);
        var_549 = wp::load(var_547);
        var_548 = wp::copy(var_549);
        // gainprm = actuator_gainprm[worldid % actuator_gainprm.shape[0], uid]                   <L 970>
        var_550 = &(var_actuator_gainprm.shape);
        var_553 = wp::load(var_550);
        var_552 = wp::extract(var_553, var_551);
        var_554 = wp::mod(var_0, var_552);
        var_555 = wp::address(var_actuator_gainprm, var_554, var_1);
        var_557 = wp::load(var_555);
        var_556 = wp::copy(var_557);
        // gain = 0.0                                                                             <L 972>
        // if gaintype == GainType.FIXED:                                                         <L 973>
        var_560 = (var_548 == var_559);
        if (var_560) {
            // gain = gainprm[0]                                                                  <L 974>
            var_562 = wp::extract(var_556, var_561);
        }
        var_563 = wp::where(var_560, var_562, var_558);
        if (!var_560) {
            // elif gaintype == GainType.AFFINE:                                                  <L 975>
            var_565 = (var_548 == var_564);
            if (var_565) {
                // gain = gainprm[0] + gainprm[1] * length + gainprm[2] * velocity                <L 976>
                var_567 = wp::extract(var_556, var_566);
                var_569 = wp::extract(var_556, var_568);
                var_570 = wp::mul(var_569, var_542);
                var_571 = wp::add(var_567, var_570);
                var_573 = wp::extract(var_556, var_572);
                var_574 = wp::mul(var_573, var_545);
                var_575 = wp::add(var_571, var_574);
            }
            var_576 = wp::where(var_565, var_575, var_563);
            if (!var_565) {
                // elif gaintype == GainType.MUSCLE:                                              <L 977>
                var_578 = (var_548 == var_577);
                if (var_578) {
                    // acc0 = actuator_acc0[worldid % actuator_acc0.shape[0], uid]                <L 978>
                    var_579 = &(var_actuator_acc0.shape);
                    var_582 = wp::load(var_579);
                    var_581 = wp::extract(var_582, var_580);
                    var_583 = wp::mod(var_0, var_581);
                    var_584 = wp::address(var_actuator_acc0, var_583, var_1);
                    var_586 = wp::load(var_584);
                    var_585 = wp::copy(var_586);
                    // lengthrange = actuator_lengthrange[worldid % actuator_lengthrange.shape[0], uid]       <L 979>
                    var_587 = &(var_actuator_lengthrange.shape);
                    var_590 = wp::load(var_587);
                    var_589 = wp::extract(var_590, var_588);
                    var_591 = wp::mod(var_0, var_589);
                    var_592 = wp::address(var_actuator_lengthrange, var_591, var_1);
                    var_594 = wp::load(var_592);
                    var_593 = wp::copy(var_594);
                    // gain = util_misc.muscle_gain(length, velocity, lengthrange, acc0, gainprm)       <L 980>
                    var_595 = muscle_gain_0(var_542, var_545, var_593, var_585, var_556);
                }
                var_596 = wp::where(var_578, var_595, var_576);
                if (!var_578) {
                    // elif gaintype == GainType.DCMOTOR:                                         <L 981>
                    var_598 = (var_548 == var_597);
                    if (var_598) {
                        // R = gainprm[0]                                                         <L 982>
                        var_600 = wp::extract(var_556, var_599);
                        // K = gainprm[1]                                                         <L 983>
                        var_602 = wp::extract(var_556, var_601);
                        // te = dynprm[0]                                                         <L 984>
                        var_604 = wp::extract(var_305, var_603);
                        // slots = util_misc.dcmotor_slots(dynprm, gainprm)                       <L 986>
                        var_605 = dcmotor_slots_0(var_305, var_556);
                        // adr = act_first                                                        <L 987>
                        var_606 = wp::copy(var_25);
                        // if slots[2] >= 0:                                                      <L 989>
                        var_608 = wp::extract(var_605, var_607);
                        var_610 = (var_608 >= var_609);
                        if (var_610) {
                            // T = act_in[worldid, adr + slots[2]]                                <L 990>
                            var_612 = wp::extract(var_605, var_611);
                            var_613 = wp::add(var_606, var_612);
                            var_614 = wp::address(var_act_in, var_0, var_613);
                            var_616 = wp::load(var_614);
                            var_615 = wp::copy(var_616);
                            // alpha = gainprm[2]                                                 <L 991>
                            var_618 = wp::extract(var_556, var_617);
                            // T0 = gainprm[3]                                                    <L 992>
                            var_620 = wp::extract(var_556, var_619);
                            // Ta = dynprm[4]                                                     <L 993>
                            var_622 = wp::extract(var_305, var_621);
                            // R *= 1.0 + alpha * (T + Ta - T0)                                   <L 994>
                            var_624 = wp::add(var_615, var_622);
                            var_625 = wp::sub(var_624, var_620);
                            var_626 = wp::mul(var_618, var_625);
                            var_627 = wp::add(var_623, var_626);
                            var_628 = wp::mul(var_600, var_627);
                        }
                        var_629 = wp::where(var_610, var_628, var_600);
                        var_630 = wp::where(var_610, var_622, var_188);
                        var_631 = wp::where(var_610, var_618, var_190);
                        var_632 = wp::where(var_610, var_620, var_192);
                        var_633 = wp::where(var_610, var_615, var_194);
                        // gain = K if te > 0.0 else K / wp.max(MJ_MINVAL, R)                     <L 996>
                        var_635 = (var_604 > var_634);
                        if (var_635) {
                        }
                        if (!var_635) {
                            var_636 = wp::max(var_60, var_629);
                            var_637 = wp::div(var_602, var_636);
                        }
                        var_638 = wp::where(var_635, var_602, var_637);
                        // if te <= 0.0:                                                          <L 998>
                        var_640 = (var_604 <= var_639);
                        if (var_640) {
                            // input_mode = int(gainprm[8])                                       <L 999>
                            var_642 = wp::extract(var_556, var_641);
                            var_643 = wp::int(var_642);
                            // if input_mode > 0:                                                 <L 1000>
                            var_645 = (var_643 > var_644);
                            if (var_645) {
                                // x_I = 0.0                                                      <L 1001>
                                // if slots[1] >= 0:                                              <L 1002>
                                var_648 = wp::extract(var_605, var_647);
                                var_650 = (var_648 >= var_649);
                                if (var_650) {
                                    // x_I = act_in[worldid, adr + slots[1]]                      <L 1003>
                                    var_652 = wp::extract(var_605, var_651);
                                    var_653 = wp::add(var_606, var_652);
                                    var_654 = wp::address(var_act_in, var_0, var_653);
                                    var_656 = wp::load(var_654);
                                    var_655 = wp::copy(var_656);
                                }
                                var_657 = wp::where(var_650, var_655, var_646);
                                // ctrl_act = util_misc.dcmotor_voltage(ctrl, length, velocity, x_I, gainprm)       <L 1004>
                                var_658 = dcmotor_voltage_0(var_539, var_542, var_545, var_657, var_556);
                            }
                            var_659 = wp::where(var_645, var_658, var_540);
                            var_660 = wp::where(var_645, var_657, var_166);
                            if (!var_645) {
                                // ctrl_act = ctrl                                                <L 1006>
                                var_661 = wp::copy(var_539);
                            }
                            var_662 = wp::where(var_645, var_659, var_661);
                        }
                        var_663 = wp::where(var_640, var_662, var_540);
                        var_664 = wp::where(var_640, var_660, var_166);
                        var_665 = wp::where(var_640, var_643, var_139);
                    }
                    var_666 = wp::where(var_598, var_663, var_540);
                    var_667 = wp::where(var_598, var_605, var_512);
                    var_668 = wp::where(var_598, var_606, var_266);
                    var_669 = wp::where(var_598, var_664, var_166);
                    var_670 = wp::where(var_598, var_665, var_139);
                    var_671 = wp::where(var_598, var_629, var_225);
                    var_672 = wp::where(var_598, var_602, var_175);
                    var_673 = wp::where(var_598, var_604, var_516);
                    var_674 = wp::where(var_598, var_630, var_188);
                    var_675 = wp::where(var_598, var_631, var_190);
                    var_676 = wp::where(var_598, var_632, var_192);
                    var_677 = wp::where(var_598, var_633, var_194);
                    var_678 = wp::where(var_598, var_638, var_596);
                }
                var_679 = wp::where(var_578, var_540, var_666);
                var_680 = wp::where(var_578, var_512, var_667);
                var_681 = wp::where(var_578, var_266, var_668);
                var_682 = wp::where(var_578, var_166, var_669);
                var_683 = wp::where(var_578, var_139, var_670);
                var_684 = wp::where(var_578, var_225, var_671);
                var_685 = wp::where(var_578, var_175, var_672);
                var_686 = wp::where(var_578, var_516, var_673);
                var_687 = wp::where(var_578, var_188, var_674);
                var_688 = wp::where(var_578, var_190, var_675);
                var_689 = wp::where(var_578, var_192, var_676);
                var_690 = wp::where(var_578, var_194, var_677);
                var_691 = wp::where(var_578, var_596, var_678);
            }
            var_692 = wp::where(var_565, var_540, var_679);
            var_693 = wp::where(var_565, var_512, var_680);
            var_694 = wp::where(var_565, var_266, var_681);
            var_695 = wp::where(var_565, var_166, var_682);
            var_696 = wp::where(var_565, var_139, var_683);
            var_697 = wp::where(var_565, var_225, var_684);
            var_698 = wp::where(var_565, var_175, var_685);
            var_699 = wp::where(var_565, var_516, var_686);
            var_700 = wp::where(var_565, var_188, var_687);
            var_701 = wp::where(var_565, var_190, var_688);
            var_702 = wp::where(var_565, var_192, var_689);
            var_703 = wp::where(var_565, var_194, var_690);
            var_704 = wp::where(var_565, var_576, var_691);
        }
        var_705 = wp::where(var_560, var_540, var_692);
        var_706 = wp::where(var_560, var_512, var_693);
        var_707 = wp::where(var_560, var_266, var_694);
        var_708 = wp::where(var_560, var_166, var_695);
        var_709 = wp::where(var_560, var_139, var_696);
        var_710 = wp::where(var_560, var_225, var_697);
        var_711 = wp::where(var_560, var_175, var_698);
        var_712 = wp::where(var_560, var_516, var_699);
        var_713 = wp::where(var_560, var_188, var_700);
        var_714 = wp::where(var_560, var_190, var_701);
        var_715 = wp::where(var_560, var_192, var_702);
        var_716 = wp::where(var_560, var_194, var_703);
        var_717 = wp::where(var_560, var_563, var_704);
        // biastype = actuator_biastype[uid]                                                      <L 1010>
        var_718 = wp::address(var_actuator_biastype, var_1);
        var_720 = wp::load(var_718);
        var_719 = wp::copy(var_720);
        // biasprm = actuator_biasprm[worldid % actuator_biasprm.shape[0], uid]                   <L 1011>
        var_721 = &(var_actuator_biasprm.shape);
        var_724 = wp::load(var_721);
        var_723 = wp::extract(var_724, var_722);
        var_725 = wp::mod(var_0, var_723);
        var_726 = wp::address(var_actuator_biasprm, var_725, var_1);
        var_728 = wp::load(var_726);
        var_727 = wp::copy(var_728);
        // bias = 0.0  # BiasType.NONE or BiasType.USER (modified by act_bias_callback)           <L 1013>
        // if biastype == BiasType.AFFINE:                                                        <L 1014>
        var_731 = (var_719 == var_730);
        if (var_731) {
            // bias = biasprm[0] + biasprm[1] * length + biasprm[2] * velocity                    <L 1015>
            var_733 = wp::extract(var_727, var_732);
            var_735 = wp::extract(var_727, var_734);
            var_736 = wp::mul(var_735, var_542);
            var_737 = wp::add(var_733, var_736);
            var_739 = wp::extract(var_727, var_738);
            var_740 = wp::mul(var_739, var_545);
            var_741 = wp::add(var_737, var_740);
        }
        var_742 = wp::where(var_731, var_741, var_729);
        if (!var_731) {
            // elif biastype == BiasType.MUSCLE:                                                  <L 1016>
            var_744 = (var_719 == var_743);
            if (var_744) {
                // acc0 = actuator_acc0[worldid % actuator_acc0.shape[0], uid]                    <L 1017>
                var_745 = &(var_actuator_acc0.shape);
                var_748 = wp::load(var_745);
                var_747 = wp::extract(var_748, var_746);
                var_749 = wp::mod(var_0, var_747);
                var_750 = wp::address(var_actuator_acc0, var_749, var_1);
                var_752 = wp::load(var_750);
                var_751 = wp::copy(var_752);
                // lengthrange = actuator_lengthrange[worldid % actuator_lengthrange.shape[0], uid]       <L 1018>
                var_753 = &(var_actuator_lengthrange.shape);
                var_756 = wp::load(var_753);
                var_755 = wp::extract(var_756, var_754);
                var_757 = wp::mod(var_0, var_755);
                var_758 = wp::address(var_actuator_lengthrange, var_757, var_1);
                var_760 = wp::load(var_758);
                var_759 = wp::copy(var_760);
                // bias = util_misc.muscle_bias(length, lengthrange, acc0, biasprm)               <L 1019>
                var_761 = muscle_bias_0(var_542, var_759, var_751, var_727);
            }
            var_762 = wp::where(var_744, var_751, var_585);
            var_763 = wp::where(var_744, var_759, var_593);
            var_764 = wp::where(var_744, var_761, var_742);
            if (!var_744) {
                // elif biastype == BiasType.DCMOTOR:                                             <L 1020>
                var_766 = (var_719 == var_765);
                if (var_766) {
                    // if dynprm[0] <= 0.0:                                                       <L 1021>
                    var_768 = wp::extract(var_305, var_767);
                    var_770 = (var_768 <= var_769);
                    if (var_770) {
                        // K = gainprm[1]                                                         <L 1022>
                        var_772 = wp::extract(var_556, var_771);
                        // bias -= gain * K * velocity                                            <L 1023>
                        var_773 = wp::mul(var_717, var_772);
                        var_774 = wp::mul(var_773, var_545);
                        var_775 = wp::sub(var_764, var_774);
                    }
                    var_776 = wp::where(var_770, var_772, var_711);
                    var_777 = wp::where(var_770, var_775, var_764);
                }
                var_778 = wp::where(var_766, var_776, var_711);
                var_779 = wp::where(var_766, var_777, var_764);
            }
            var_780 = wp::where(var_744, var_711, var_778);
            var_781 = wp::where(var_744, var_764, var_779);
        }
        var_782 = wp::where(var_731, var_711, var_780);
        var_783 = wp::where(var_731, var_585, var_762);
        var_784 = wp::where(var_731, var_593, var_763);
        var_785 = wp::where(var_731, var_742, var_781);
        // force = gain * ctrl_act + bias                                                         <L 1025>
        var_786 = wp::mul(var_717, var_705);
        var_787 = wp::add(var_786, var_785);
        // if actuator_forcelimited[uid]:                                                         <L 1027>
        var_788 = wp::address(var_actuator_forcelimited, var_1);
        var_789 = wp::load(var_788);
        if (var_789) {
            // forcerange = actuator_forcerange[worldid % actuator_forcerange.shape[0], uid]       <L 1028>
            var_790 = &(var_actuator_forcerange.shape);
            var_793 = wp::load(var_790);
            var_792 = wp::extract(var_793, var_791);
            var_794 = wp::mod(var_0, var_792);
            var_795 = wp::address(var_actuator_forcerange, var_794, var_1);
            var_797 = wp::load(var_795);
            var_796 = wp::copy(var_797);
            // force = wp.clamp(force, forcerange[0], forcerange[1])                              <L 1029>
            var_799 = wp::extract(var_796, var_798);
            var_801 = wp::extract(var_796, var_800);
            var_802 = wp::clamp(var_787, var_799, var_801);
        }
        var_803 = wp::load(var_788);
        var_805 = wp::load(var_788);
        var_804 = wp::where(var_805, var_802, var_787);
        // if biastype == BiasType.DCMOTOR:                                                       <L 1032>
        var_807 = (var_719 == var_806);
        if (var_807) {
            // A = biasprm[0]                                                                     <L 1034>
            var_809 = wp::extract(var_727, var_808);
            // if A != 0.0:                                                                       <L 1035>
            var_811 = (var_809 != var_810);
            if (var_811) {
                // Np = biasprm[1]                                                                <L 1036>
                var_813 = wp::extract(var_727, var_812);
                // phi = biasprm[2]                                                               <L 1037>
                var_815 = wp::extract(var_727, var_814);
                // force += A * wp.sin(Np * length + phi)                                         <L 1038>
                var_816 = wp::mul(var_813, var_542);
                var_817 = wp::add(var_816, var_815);
                var_818 = wp::sin(var_817);
                var_819 = wp::mul(var_809, var_818);
                var_820 = wp::add(var_804, var_819);
            }
            var_821 = wp::where(var_811, var_820, var_804);
            // sigma0 = dynprm[5]                                                                 <L 1041>
            var_823 = wp::extract(var_305, var_822);
            // if sigma0 > 0.0:                                                                   <L 1042>
            var_825 = (var_823 > var_824);
            if (var_825) {
                // sigma1 = dynprm[6]                                                             <L 1043>
                var_827 = wp::extract(var_305, var_826);
                // slots = util_misc.dcmotor_slots(dynprm, gainprm)                               <L 1044>
                var_828 = dcmotor_slots_0(var_305, var_556);
                // adr = act_first + slots[3]  # slots[3] is bristle                              <L 1045>
                var_830 = wp::extract(var_828, var_829);
                var_831 = wp::add(var_25, var_830);
                // z = act_in[worldid, adr]                                                       <L 1046>
                var_832 = wp::address(var_act_in, var_0, var_831);
                var_834 = wp::load(var_832);
                var_833 = wp::copy(var_834);
                // z_dot = act_dot_out[worldid, adr]                                              <L 1047>
                var_835 = wp::address(var_act_dot_out, var_0, var_831);
                var_837 = wp::load(var_835);
                var_836 = wp::copy(var_837);
                // force -= sigma0 * z + sigma1 * z_dot                                           <L 1048>
                var_838 = wp::mul(var_823, var_833);
                var_839 = wp::mul(var_827, var_836);
                var_840 = wp::add(var_838, var_839);
                var_841 = wp::sub(var_821, var_840);
            }
            var_842 = wp::where(var_825, var_828, var_706);
            var_843 = wp::where(var_825, var_831, var_707);
            var_844 = wp::where(var_825, var_833, var_247);
            var_845 = wp::where(var_825, var_841, var_821);
        }
        var_846 = wp::where(var_807, var_842, var_706);
        var_847 = wp::where(var_807, var_843, var_707);
        var_848 = wp::where(var_807, var_823, var_518);
        var_849 = wp::where(var_807, var_844, var_247);
        var_850 = wp::where(var_807, var_845, var_804);
        // actuator_force_out[worldid, uid] = force                                               <L 1050>
        wp::array_store(var_actuator_force_out, var_0, var_1, var_850);
    }
}



extern "C" __global__ void _actuator_velocity_00a052f5_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::int32> var_moment_rownnz_in,
    wp::array_t<wp::int32> var_moment_rowadr_in,
    wp::array_t<wp::int32> var_moment_colind_in,
    wp::array_t<wp::float32> var_actuator_moment_in,
    wp::array_t<wp::float32> var_actuator_velocity_out)
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
        const wp::float32 var_8 = 0.0;
        wp::float32 var_9;
        wp::range_t var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::float32* var_16;
        wp::float32* var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        //---------
        // forward
        // def _actuator_velocity(                                                                <L 681>
        // worldid, actid = wp.tid()                                                              <L 691>
        builtin_tid2d(var_0, var_1);
        // rownnz = moment_rownnz_in[worldid, actid]                                              <L 693>
        var_2 = wp::address(var_moment_rownnz_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // rowadr = moment_rowadr_in[worldid, actid]                                              <L 694>
        var_5 = wp::address(var_moment_rowadr_in, var_0, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // vel = float(0.0)                                                                       <L 696>
        var_9 = wp::float(var_8);
        // for i in range(rownnz):                                                                <L 697>
        var_10 = wp::range(var_3);
        start_for_0:;
            if (iter_cmp(var_10) == 0) goto end_for_0;
            var_11 = wp::iter_next(var_10);
            // sparseid = rowadr + i                                                              <L 698>
            var_12 = wp::add(var_6, var_11);
            // colind = moment_colind_in[worldid, sparseid]                                       <L 699>
            var_13 = wp::address(var_moment_colind_in, var_0, var_12);
            var_15 = wp::load(var_13);
            var_14 = wp::copy(var_15);
            // vel += actuator_moment_in[worldid, sparseid] * qvel_in[worldid, colind]            <L 700>
            var_16 = wp::address(var_actuator_moment_in, var_0, var_12);
            var_17 = wp::address(var_qvel_in, var_0, var_14);
            var_19 = wp::load(var_16);
            var_20 = wp::load(var_17);
            var_18 = wp::mul(var_19, var_20);
            var_21 = wp::add(var_9, var_18);
            wp::assign(var_9, var_21);
            goto start_for_0;
        end_for_0:;
        // actuator_velocity_out[worldid, actid] = vel                                            <L 702>
        wp::array_store(var_actuator_velocity_out, var_0, var_1, var_9);
    }
}



extern "C" __global__ void _compute_damping_deriv_cfbf3e5a_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_dof_damping,
    wp::array_t<wp::vec_t<2, wp::float32>> var_dof_dampingpoly,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::float32> var_deriv_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        wp::shape_t* var_10;
        const wp::int32 var_11 = 0;
        wp::int32 var_12;
        wp::shape_t var_13;
        wp::int32 var_14;
        wp::vec_t<2, wp::float32>* var_15;
        wp::vec_t<2, wp::float32> var_16;
        wp::vec_t<2, wp::float32> var_17;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        const wp::int32 var_21 = 1;
        wp::float32 var_22;
        //---------
        // forward
        // def _compute_damping_deriv(                                                            <L 353>
        // worldid, tid = wp.tid()                                                                <L 362>
        builtin_tid2d(var_0, var_1);
        // damping = dof_damping[worldid % dof_damping.shape[0], tid]                             <L 363>
        var_2 = &(var_dof_damping.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        var_7 = wp::address(var_dof_damping, var_6, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // dpoly = dof_dampingpoly[worldid % dof_dampingpoly.shape[0], tid]                       <L 364>
        var_10 = &(var_dof_dampingpoly.shape);
        var_13 = wp::load(var_10);
        var_12 = wp::extract(var_13, var_11);
        var_14 = wp::mod(var_0, var_12);
        var_15 = wp::address(var_dof_dampingpoly, var_14, var_1);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
        // v = qvel_in[worldid, tid]                                                              <L 365>
        var_18 = wp::address(var_qvel_in, var_0, var_1);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // deriv_out[worldid, tid] = util_misc._poly_force_deriv(damping, dpoly, v, 1)            <L 366>
        var_22 = _poly_force_deriv_0(var_8, var_16, var_19, var_21);
        wp::array_store(var_deriv_out, var_0, var_1, var_22);
    }
}



extern "C" __global__ void _next_position_1c101e8d_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::array_t<wp::int32> var_jnt_type,
    wp::array_t<wp::int32> var_jnt_qposadr,
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::float32> var_qpos_in,
    wp::array_t<wp::float32> var_qvel_in,
    wp::float32 var_qvel_scale_in,
    wp::array_t<wp::float32> var_qpos_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        wp::slice_t var_19;
        const wp::int32 var_20 = 0;
        wp::array_t<wp::float32> var_21;
        wp::slice_t var_22;
        const wp::int32 var_23 = 0;
        wp::array_t<wp::float32> var_24;
        wp::slice_t var_25;
        const wp::int32 var_26 = 0;
        wp::array_t<wp::float32> var_27;
        const wp::int32 var_28 = 0;
        bool var_29;
        wp::float32* var_30;
        const wp::int32 var_31 = 1;
        wp::int32 var_32;
        wp::float32* var_33;
        const wp::int32 var_34 = 2;
        wp::int32 var_35;
        wp::float32* var_36;
        wp::vec_t<3, wp::float32> var_37;
        wp::float32 var_38;
        wp::float32 var_39;
        wp::float32 var_40;
        wp::float32* var_41;
        const wp::int32 var_42 = 1;
        wp::int32 var_43;
        wp::float32* var_44;
        const wp::int32 var_45 = 2;
        wp::int32 var_46;
        wp::float32* var_47;
        wp::vec_t<3, wp::float32> var_48;
        wp::float32 var_49;
        wp::float32 var_50;
        wp::float32 var_51;
        wp::vec_t<3, wp::float32> var_52;
        wp::vec_t<3, wp::float32> var_53;
        wp::vec_t<3, wp::float32> var_54;
        const wp::int32 var_55 = 3;
        wp::int32 var_56;
        wp::float32* var_57;
        const wp::int32 var_58 = 4;
        wp::int32 var_59;
        wp::float32* var_60;
        const wp::int32 var_61 = 5;
        wp::int32 var_62;
        wp::float32* var_63;
        const wp::int32 var_64 = 6;
        wp::int32 var_65;
        wp::float32* var_66;
        wp::quat_t<wp::float32> var_67;
        wp::float32 var_68;
        wp::float32 var_69;
        wp::float32 var_70;
        wp::float32 var_71;
        const wp::int32 var_72 = 3;
        wp::int32 var_73;
        wp::float32* var_74;
        const wp::int32 var_75 = 4;
        wp::int32 var_76;
        wp::float32* var_77;
        const wp::int32 var_78 = 5;
        wp::int32 var_79;
        wp::float32* var_80;
        wp::vec_t<3, wp::float32> var_81;
        wp::float32 var_82;
        wp::float32 var_83;
        wp::float32 var_84;
        wp::vec_t<3, wp::float32> var_85;
        wp::quat_t<wp::float32> var_86;
        const wp::int32 var_87 = 0;
        wp::float32 var_88;
        const wp::int32 var_89 = 0;
        wp::int32 var_90;
        const wp::int32 var_91 = 1;
        wp::float32 var_92;
        const wp::int32 var_93 = 1;
        wp::int32 var_94;
        const wp::int32 var_95 = 2;
        wp::float32 var_96;
        const wp::int32 var_97 = 2;
        wp::int32 var_98;
        const wp::int32 var_99 = 0;
        wp::float32 var_100;
        const wp::int32 var_101 = 3;
        wp::int32 var_102;
        const wp::int32 var_103 = 1;
        wp::float32 var_104;
        const wp::int32 var_105 = 4;
        wp::int32 var_106;
        const wp::int32 var_107 = 2;
        wp::float32 var_108;
        const wp::int32 var_109 = 5;
        wp::int32 var_110;
        const wp::int32 var_111 = 3;
        wp::float32 var_112;
        const wp::int32 var_113 = 6;
        wp::int32 var_114;
        const wp::int32 var_115 = 1;
        bool var_116;
        const wp::int32 var_117 = 0;
        wp::int32 var_118;
        wp::float32* var_119;
        const wp::int32 var_120 = 1;
        wp::int32 var_121;
        wp::float32* var_122;
        const wp::int32 var_123 = 2;
        wp::int32 var_124;
        wp::float32* var_125;
        const wp::int32 var_126 = 3;
        wp::int32 var_127;
        wp::float32* var_128;
        wp::quat_t<wp::float32> var_129;
        wp::float32 var_130;
        wp::float32 var_131;
        wp::float32 var_132;
        wp::float32 var_133;
        wp::float32* var_134;
        const wp::int32 var_135 = 1;
        wp::int32 var_136;
        wp::float32* var_137;
        const wp::int32 var_138 = 2;
        wp::int32 var_139;
        wp::float32* var_140;
        wp::vec_t<3, wp::float32> var_141;
        wp::float32 var_142;
        wp::float32 var_143;
        wp::float32 var_144;
        wp::vec_t<3, wp::float32> var_145;
        wp::quat_t<wp::float32> var_146;
        const wp::int32 var_147 = 0;
        wp::float32 var_148;
        const wp::int32 var_149 = 0;
        wp::int32 var_150;
        const wp::int32 var_151 = 1;
        wp::float32 var_152;
        const wp::int32 var_153 = 1;
        wp::int32 var_154;
        const wp::int32 var_155 = 2;
        wp::float32 var_156;
        const wp::int32 var_157 = 2;
        wp::int32 var_158;
        const wp::int32 var_159 = 3;
        wp::float32 var_160;
        const wp::int32 var_161 = 3;
        wp::int32 var_162;
        wp::quat_t<wp::float32> var_163;
        wp::vec_t<3, wp::float32> var_164;
        wp::quat_t<wp::float32> var_165;
        wp::float32* var_166;
        wp::float32* var_167;
        wp::float32 var_168;
        wp::float32 var_169;
        wp::float32 var_170;
        wp::float32 var_171;
        wp::float32 var_172;
        wp::quat_t<wp::float32> var_173;
        wp::vec_t<3, wp::float32> var_174;
        wp::quat_t<wp::float32> var_175;
        //---------
        // forward
        // def _next_position(                                                                    <L 54>
        // worldid, jntid = wp.tid()                                                              <L 68>
        builtin_tid2d(var_0, var_1);
        // timestep = opt_timestep[worldid % opt_timestep.shape[0]]                               <L 69>
        var_2 = &(var_opt_timestep.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        var_7 = wp::address(var_opt_timestep, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // jnttype = jnt_type[jntid]                                                              <L 71>
        var_10 = wp::address(var_jnt_type, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // qpos_adr = jnt_qposadr[jntid]                                                          <L 72>
        var_13 = wp::address(var_jnt_qposadr, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // dof_adr = jnt_dofadr[jntid]                                                            <L 73>
        var_16 = wp::address(var_jnt_dofadr, var_1);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // qpos = qpos_in[worldid]                                                                <L 74>
        var_19 = wp::slice_t(var_0, var_0, var_20);
        var_21 = wp::view(var_qpos_in, var_19);
        // qpos_next = qpos_out[worldid]                                                          <L 75>
        var_22 = wp::slice_t(var_0, var_0, var_23);
        var_24 = wp::view(var_qpos_out, var_22);
        // qvel = qvel_in[worldid]                                                                <L 76>
        var_25 = wp::slice_t(var_0, var_0, var_26);
        var_27 = wp::view(var_qvel_in, var_25);
        // if jnttype == JointType.FREE:                                                          <L 78>
        var_29 = (var_11 == var_28);
        if (var_29) {
            // qpos_pos = wp.vec3(qpos[qpos_adr], qpos[qpos_adr + 1], qpos[qpos_adr + 2])         <L 79>
            var_30 = wp::address(var_21, var_14);
            var_32 = wp::add(var_14, var_31);
            var_33 = wp::address(var_21, var_32);
            var_35 = wp::add(var_14, var_34);
            var_36 = wp::address(var_21, var_35);
            var_38 = wp::load(var_30);
            var_39 = wp::load(var_33);
            var_40 = wp::load(var_36);
            var_37 = wp::vec_t<3, wp::float32>(var_38, var_39, var_40);
            // qvel_lin = wp.vec3(qvel[dof_adr], qvel[dof_adr + 1], qvel[dof_adr + 2]) * qvel_scale_in       <L 80>
            var_41 = wp::address(var_27, var_17);
            var_43 = wp::add(var_17, var_42);
            var_44 = wp::address(var_27, var_43);
            var_46 = wp::add(var_17, var_45);
            var_47 = wp::address(var_27, var_46);
            var_49 = wp::load(var_41);
            var_50 = wp::load(var_44);
            var_51 = wp::load(var_47);
            var_48 = wp::vec_t<3, wp::float32>(var_49, var_50, var_51);
            var_52 = wp::mul(var_48, var_qvel_scale_in);
            // qpos_new = qpos_pos + timestep * qvel_lin                                          <L 82>
            var_53 = wp::mul(var_8, var_52);
            var_54 = wp::add(var_37, var_53);
            // qpos_quat = wp.quat(                                                               <L 84>
            // qpos[qpos_adr + 3],                                                                <L 85>
            var_56 = wp::add(var_14, var_55);
            var_57 = wp::address(var_21, var_56);
            // qpos[qpos_adr + 4],                                                                <L 86>
            var_59 = wp::add(var_14, var_58);
            var_60 = wp::address(var_21, var_59);
            // qpos[qpos_adr + 5],                                                                <L 87>
            var_62 = wp::add(var_14, var_61);
            var_63 = wp::address(var_21, var_62);
            // qpos[qpos_adr + 6],                                                                <L 88>
            var_65 = wp::add(var_14, var_64);
            var_66 = wp::address(var_21, var_65);
            var_68 = wp::load(var_57);
            var_69 = wp::load(var_60);
            var_70 = wp::load(var_63);
            var_71 = wp::load(var_66);
            var_67 = wp::quat_t<wp::float32>(var_68, var_69, var_70, var_71);
            // qvel_ang = wp.vec3(qvel[dof_adr + 3], qvel[dof_adr + 4], qvel[dof_adr + 5]) * qvel_scale_in       <L 90>
            var_73 = wp::add(var_17, var_72);
            var_74 = wp::address(var_27, var_73);
            var_76 = wp::add(var_17, var_75);
            var_77 = wp::address(var_27, var_76);
            var_79 = wp::add(var_17, var_78);
            var_80 = wp::address(var_27, var_79);
            var_82 = wp::load(var_74);
            var_83 = wp::load(var_77);
            var_84 = wp::load(var_80);
            var_81 = wp::vec_t<3, wp::float32>(var_82, var_83, var_84);
            var_85 = wp::mul(var_81, var_qvel_scale_in);
            // qpos_quat_new = math.quat_integrate(qpos_quat, qvel_ang, timestep)                 <L 92>
            var_86 = quat_integrate_0(var_67, var_85, var_8);
            // qpos_next[qpos_adr + 0] = qpos_new[0]                                              <L 94>
            var_88 = wp::extract(var_54, var_87);
            var_90 = wp::add(var_14, var_89);
            wp::array_store(var_24, var_90, var_88);
            // qpos_next[qpos_adr + 1] = qpos_new[1]                                              <L 95>
            var_92 = wp::extract(var_54, var_91);
            var_94 = wp::add(var_14, var_93);
            wp::array_store(var_24, var_94, var_92);
            // qpos_next[qpos_adr + 2] = qpos_new[2]                                              <L 96>
            var_96 = wp::extract(var_54, var_95);
            var_98 = wp::add(var_14, var_97);
            wp::array_store(var_24, var_98, var_96);
            // qpos_next[qpos_adr + 3] = qpos_quat_new[0]                                         <L 97>
            var_100 = wp::extract(var_86, var_99);
            var_102 = wp::add(var_14, var_101);
            wp::array_store(var_24, var_102, var_100);
            // qpos_next[qpos_adr + 4] = qpos_quat_new[1]                                         <L 98>
            var_104 = wp::extract(var_86, var_103);
            var_106 = wp::add(var_14, var_105);
            wp::array_store(var_24, var_106, var_104);
            // qpos_next[qpos_adr + 5] = qpos_quat_new[2]                                         <L 99>
            var_108 = wp::extract(var_86, var_107);
            var_110 = wp::add(var_14, var_109);
            wp::array_store(var_24, var_110, var_108);
            // qpos_next[qpos_adr + 6] = qpos_quat_new[3]                                         <L 100>
            var_112 = wp::extract(var_86, var_111);
            var_114 = wp::add(var_14, var_113);
            wp::array_store(var_24, var_114, var_112);
        }
        if (!var_29) {
            // elif jnttype == JointType.BALL:                                                    <L 102>
            var_116 = (var_11 == var_115);
            if (var_116) {
                // qpos_quat = wp.quat(qpos[qpos_adr + 0], qpos[qpos_adr + 1], qpos[qpos_adr + 2], qpos[qpos_adr + 3])       <L 103>
                var_118 = wp::add(var_14, var_117);
                var_119 = wp::address(var_21, var_118);
                var_121 = wp::add(var_14, var_120);
                var_122 = wp::address(var_21, var_121);
                var_124 = wp::add(var_14, var_123);
                var_125 = wp::address(var_21, var_124);
                var_127 = wp::add(var_14, var_126);
                var_128 = wp::address(var_21, var_127);
                var_130 = wp::load(var_119);
                var_131 = wp::load(var_122);
                var_132 = wp::load(var_125);
                var_133 = wp::load(var_128);
                var_129 = wp::quat_t<wp::float32>(var_130, var_131, var_132, var_133);
                // qvel_ang = wp.vec3(qvel[dof_adr], qvel[dof_adr + 1], qvel[dof_adr + 2]) * qvel_scale_in       <L 104>
                var_134 = wp::address(var_27, var_17);
                var_136 = wp::add(var_17, var_135);
                var_137 = wp::address(var_27, var_136);
                var_139 = wp::add(var_17, var_138);
                var_140 = wp::address(var_27, var_139);
                var_142 = wp::load(var_134);
                var_143 = wp::load(var_137);
                var_144 = wp::load(var_140);
                var_141 = wp::vec_t<3, wp::float32>(var_142, var_143, var_144);
                var_145 = wp::mul(var_141, var_qvel_scale_in);
                // qpos_quat_new = math.quat_integrate(qpos_quat, qvel_ang, timestep)             <L 106>
                var_146 = quat_integrate_0(var_129, var_145, var_8);
                // qpos_next[qpos_adr + 0] = qpos_quat_new[0]                                     <L 108>
                var_148 = wp::extract(var_146, var_147);
                var_150 = wp::add(var_14, var_149);
                wp::array_store(var_24, var_150, var_148);
                // qpos_next[qpos_adr + 1] = qpos_quat_new[1]                                     <L 109>
                var_152 = wp::extract(var_146, var_151);
                var_154 = wp::add(var_14, var_153);
                wp::array_store(var_24, var_154, var_152);
                // qpos_next[qpos_adr + 2] = qpos_quat_new[2]                                     <L 110>
                var_156 = wp::extract(var_146, var_155);
                var_158 = wp::add(var_14, var_157);
                wp::array_store(var_24, var_158, var_156);
                // qpos_next[qpos_adr + 3] = qpos_quat_new[3]                                     <L 111>
                var_160 = wp::extract(var_146, var_159);
                var_162 = wp::add(var_14, var_161);
                wp::array_store(var_24, var_162, var_160);
            }
            var_163 = wp::where(var_116, var_129, var_67);
            var_164 = wp::where(var_116, var_145, var_85);
            var_165 = wp::where(var_116, var_146, var_86);
            if (!var_116) {
                // qpos_next[qpos_adr] = qpos[qpos_adr] + timestep * qvel[dof_adr] * qvel_scale_in       <L 114>
                var_166 = wp::address(var_21, var_14);
                var_167 = wp::address(var_27, var_17);
                var_169 = wp::load(var_167);
                var_168 = wp::mul(var_8, var_169);
                var_170 = wp::mul(var_168, var_qvel_scale_in);
                var_172 = wp::load(var_166);
                var_171 = wp::add(var_172, var_170);
                wp::array_store(var_24, var_14, var_171);
            }
        }
        var_173 = wp::where(var_29, var_67, var_163);
        var_174 = wp::where(var_29, var_85, var_164);
        var_175 = wp::where(var_29, var_86, var_165);
    }
}



extern "C" __global__ void _rk_accumulate_activation_velocity_5935ab3f_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_act_dot_in,
    wp::float32 var_scale,
    wp::array_t<wp::float32> var_act_dot_out)
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
        wp::float32 var_3;
        wp::float32 var_4;
        wp::float32 var_5;
        //---------
        // forward
        // def _rk_accumulate_activation_velocity(                                                <L 486>
        // worldid, actid = wp.tid()                                                              <L 494>
        builtin_tid2d(var_0, var_1);
        // act_dot_out[worldid, actid] += scale * act_dot_in[worldid, actid]                      <L 495>
        var_2 = wp::address(var_act_dot_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::mul(var_scale, var_4);
        var_5 = wp::atomic_add(var_act_dot_out, var_0, var_1, var_3);
    }
}

