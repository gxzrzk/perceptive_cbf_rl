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


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/util_misc.py:708
static CUDA_CALLABLE wp::float32 _poly_force_0(
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
    const wp::int32 var_5 = 0;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    const wp::int32 var_9 = 1;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    //---------
    // forward
    // def _poly_force(linear: float, poly: wp.vec2, x: float, flg_odd: int) -> float:        <L 709>
    // x_val = wp.where(flg_odd == 1, wp.abs(x), x)                                           <L 710>
    var_1 = (var_flg_odd == var_0);
    var_2 = wp::abs(var_x);
    var_3 = wp::where(var_1, var_2, var_x);
    // res = linear                                                                           <L 711>
    var_4 = wp::copy(var_linear);
    // res += poly[0] * x_val                                                                 <L 712>
    var_6 = wp::extract(var_poly, var_5);
    var_7 = wp::mul(var_6, var_3);
    var_8 = wp::add(var_4, var_7);
    // res += poly[1] * x_val * x_val                                                         <L 713>
    var_10 = wp::extract(var_poly, var_9);
    var_11 = wp::mul(var_10, var_3);
    var_12 = wp::mul(var_11, var_3);
    var_13 = wp::add(var_8, var_12);
    // return res                                                                             <L 714>
    return var_13;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/support.py:1270
static CUDA_CALLABLE wp::float32 flex_dphi_0(
    wp::float32 var_s,
    wp::int32 var_i,
    wp::int32 var_order)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    bool var_1;
    const wp::float32 var_2 = -1.0;
    const wp::float32 var_3 = 1.0;
    wp::float32 var_4;
    //---------
    // forward
    // def flex_dphi(s: float, i: int, order: int) -> float:                                  <L 1271>
    // return -1.0 if i == 0 else 1.0                                                         <L 1272>
    var_1 = (var_i == var_0);
    if (var_1) {
    }
    if (!var_1) {
    }
    var_4 = wp::where(var_1, var_2, var_3);
    return var_4;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/support.py:1265
static CUDA_CALLABLE wp::float32 flex_phi_0(
    wp::float32 var_s,
    wp::int32 var_i,
    wp::int32 var_order)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    bool var_1;
    const wp::float32 var_2 = 1.0;
    wp::float32 var_3;
    wp::float32 var_4;
    //---------
    // forward
    // def flex_phi(s: float, i: int, order: int) -> float:                                   <L 1266>
    // return 1.0 - s if i == 0 else s                                                        <L 1267>
    var_1 = (var_i == var_0);
    if (var_1) {
        var_3 = wp::sub(var_2, var_s);
    }
    if (!var_1) {
    }
    var_4 = wp::where(var_1, var_3, var_s);
    return var_4;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/support.py:1275
static CUDA_CALLABLE wp::float32 dphi2D_0(
    wp::float32 var_s0,
    wp::int32 var_l0,
    wp::float32 var_s1,
    wp::int32 var_l1,
    wp::int32 var_order,
    wp::int32 var_direction)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    bool var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    //---------
    // forward
    // def dphi2D(s0: float, l0: int, s1: float, l1: int, order: int, direction: int) -> float:       <L 1276>
    // if direction == 0:                                                                     <L 1277>
    var_1 = (var_direction == var_0);
    if (var_1) {
        // return flex_dphi(s0, l0, order) * flex_phi(s1, l1, order)                          <L 1278>
        var_2 = flex_dphi_0(var_s0, var_l0, var_order);
        var_3 = flex_phi_0(var_s1, var_l1, var_order);
        var_4 = wp::mul(var_2, var_3);
        return var_4;
    }
    if (!var_1) {
        // return flex_phi(s0, l0, order) * flex_dphi(s1, l1, order)                          <L 1280>
        var_5 = flex_phi_0(var_s0, var_l0, var_order);
        var_6 = flex_dphi_0(var_s1, var_l1, var_order);
        var_7 = wp::mul(var_5, var_6);
        return var_7;
    }
    return {};
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/passive.py:980
static CUDA_CALLABLE void _apply_face_forces_0(
    wp::array_t<wp::int32> var_flex_nodebodyid,
    wp::array_t<wp::int32> var_flex_face,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexnode_xpos_in,
    wp::int32 var_face_id,
    wp::vec_t<2, wp::float32> var_local_coords,
    wp::vec_t<3, wp::float32> var_wt1,
    wp::vec_t<3, wp::float32> var_wt2,
    wp::float32 var_stiffness_scale,
    wp::int32 var_order_abs,
    wp::int32 var_worldid,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_force_out)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::int32 var_1;
    const wp::int32 var_2 = 3;
    wp::range_t var_3;
    wp::int32 var_4;
    bool var_5;
    const wp::int32 var_6 = 3;
    wp::range_t var_7;
    wp::int32 var_8;
    bool var_9;
    const wp::int32 var_10 = 0;
    wp::float32 var_11;
    const wp::int32 var_12 = 1;
    wp::float32 var_13;
    const wp::int32 var_14 = 0;
    wp::float32 var_15;
    const wp::int32 var_16 = 0;
    wp::float32 var_17;
    const wp::int32 var_18 = 1;
    wp::float32 var_19;
    const wp::int32 var_20 = 1;
    wp::float32 var_21;
    wp::int32* var_22;
    wp::int32 var_23;
    wp::int32 var_24;
    wp::vec_t<3, wp::float32> var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::vec_t<3, wp::float32> var_28;
    wp::int32* var_29;
    wp::int32 var_30;
    wp::int32 var_31;
    wp::vec_t<3, wp::float32>* var_32;
    wp::vec_t<3, wp::float32> var_33;
    wp::vec_t<3, wp::float32> var_34;
    wp::vec_t<3, wp::float32>* var_35;
    wp::vec_t<3, wp::float32> var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::vec_t<3, wp::float32> var_39;
    wp::vec_t<3, wp::float32> var_40;
    wp::vec_t<6, wp::float32> var_41;
    wp::vec_t<6, wp::float32> var_42;
    const wp::int32 var_43 = 1;
    wp::int32 var_44;
    //---------
    // forward
    // def _apply_face_forces(                                                                <L 981>
    // idx = int(0)                                                                           <L 999>
    var_1 = wp::int(var_0);
    // for l0 in range(3):                                                                    <L 1000>
    var_3 = wp::range(var_2);
    start_for_0:;
        if (iter_cmp(var_3) == 0) goto end_for_0;
        var_4 = wp::iter_next(var_3);
        // if l0 > order_abs:                                                                 <L 1001>
        var_5 = (var_4 > var_order_abs);
        if (var_5) {
            // continue                                                                       <L 1002>
            goto start_for_0;
        }
        // for l1 in range(3):                                                                <L 1003>
        var_7 = wp::range(var_6);
        start_for_2:;
            if (iter_cmp(var_7) == 0) goto end_for_2;
            var_8 = wp::iter_next(var_7);
            // if l1 > order_abs:                                                             <L 1004>
            var_9 = (var_8 > var_order_abs);
            if (var_9) {
                // continue                                                                   <L 1005>
                goto start_for_2;
            }
            // g0 = support.dphi2D(local_coords[0], l0, local_coords[1], l1, order_abs, 0)       <L 1006>
            var_11 = wp::extract(var_local_coords, var_10);
            var_13 = wp::extract(var_local_coords, var_12);
            var_15 = dphi2D_0(var_11, var_4, var_13, var_8, var_order_abs, var_14);
            // g1 = support.dphi2D(local_coords[0], l0, local_coords[1], l1, order_abs, 1)       <L 1007>
            var_17 = wp::extract(var_local_coords, var_16);
            var_19 = wp::extract(var_local_coords, var_18);
            var_21 = dphi2D_0(var_17, var_4, var_19, var_8, var_order_abs, var_20);
            // gidx = flex_face[face_id, idx]                                                 <L 1009>
            var_22 = wp::address(var_flex_face, var_face_id, var_1);
            var_24 = wp::load(var_22);
            var_23 = wp::copy(var_24);
            // frc = (wt2 * g0 - wt1 * g1) * stiffness_scale                                  <L 1011>
            var_25 = wp::mul(var_wt2, var_15);
            var_26 = wp::mul(var_wt1, var_21);
            var_27 = wp::sub(var_25, var_26);
            var_28 = wp::mul(var_27, var_stiffness_scale);
            // bid = flex_nodebodyid[gidx]                                                    <L 1013>
            var_29 = wp::address(var_flex_nodebodyid, var_23);
            var_31 = wp::load(var_29);
            var_30 = wp::copy(var_31);
            // node_pos = flexnode_xpos_in[worldid, gidx]                                     <L 1014>
            var_32 = wp::address(var_flexnode_xpos_in, var_worldid, var_23);
            var_34 = wp::load(var_32);
            var_33 = wp::copy(var_34);
            // body_xipos = xipos_in[worldid, bid]                                            <L 1016>
            var_35 = wp::address(var_xipos_in, var_worldid, var_30);
            var_37 = wp::load(var_35);
            var_36 = wp::copy(var_37);
            // offset = body_xipos - node_pos                                                 <L 1017>
            var_38 = wp::sub(var_36, var_33);
            // spatial_frc = wp.spatial_vector(frc, -wp.cross(offset, frc))                   <L 1018>
            var_39 = wp::cross(var_38, var_28);
            var_40 = wp::neg(var_39);
            var_41 = wp::vec_t<6, wp::float32>(var_28, var_40);
            // wp.atomic_add(body_force_out, worldid, bid, spatial_frc)                       <L 1019>
            var_42 = wp::atomic_add(var_body_force_out, var_worldid, var_30, var_41);
            // idx += 1                                                                       <L 1020>
            var_44 = wp::add(var_1, var_43);
            wp::assign(var_1, var_44);
            goto start_for_2;
        end_for_2:;
        goto start_for_0;
    end_for_0:;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/support.py:497
static CUDA_CALLABLE void jac_dof_0(
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::int32> var_body_isdofancestor,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::vec_t<3, wp::float32> var_point,
    wp::int32 var_bodyid,
    wp::int32 var_dofid,
    wp::int32 var_worldid,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    wp::int32* var_0;
    const wp::int32 var_1 = 0;
    bool var_2;
    wp::int32 var_3;
    const wp::float32 var_4 = 0.0;
    wp::vec_t<3, wp::float32> var_5;
    const wp::float32 var_6 = 0.0;
    wp::vec_t<3, wp::float32> var_7;
    wp::int32* var_8;
    wp::vec_t<3, wp::float32>* var_9;
    wp::int32 var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::vec_t<3, wp::float32> var_13;
    wp::vec_t<6, wp::float32>* var_14;
    wp::vec_t<6, wp::float32> var_15;
    wp::vec_t<6, wp::float32> var_16;
    wp::vec_t<3, wp::float32> var_17;
    wp::vec_t<3, wp::float32> var_18;
    wp::vec_t<3, wp::float32> var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32> var_21;
    //---------
    // forward
    // def jac_dof(                                                                           <L 498>
    // if body_isdofancestor[bodyid, dofid] == 0:                                             <L 513>
    var_0 = wp::address(var_body_isdofancestor, var_bodyid, var_dofid);
    var_3 = wp::load(var_0);
    var_2 = (var_3 == var_1);
    if (var_2) {
        // return wp.vec3(0.0), wp.vec3(0.0)                                                  <L 514>
        var_5 = wp::vec_t<3, wp::float32>(var_4);
        var_7 = wp::vec_t<3, wp::float32>(var_6);
        ret_0 = var_5;
        ret_1 = var_7;
        return;
    }
    // offset = point - wp.vec3(subtree_com_in[worldid, body_rootid[bodyid]])                 <L 516>
    var_8 = wp::address(var_body_rootid, var_bodyid);
    var_10 = wp::load(var_8);
    var_9 = wp::address(var_subtree_com_in, var_worldid, var_10);
    var_12 = wp::load(var_9);
    var_11 = wp::vec_t<3, wp::float32>(var_12);
    var_13 = wp::sub(var_point, var_11);
    // cdof = cdof_in[worldid, dofid]                                                         <L 518>
    var_14 = wp::address(var_cdof_in, var_worldid, var_dofid);
    var_16 = wp::load(var_14);
    var_15 = wp::copy(var_16);
    // cdof_ang = wp.spatial_top(cdof)                                                        <L 519>
    var_17 = wp::spatial_top(var_15);
    // cdof_lin = wp.spatial_bottom(cdof)                                                     <L 520>
    var_18 = wp::spatial_bottom(var_15);
    // jacp = cdof_lin + wp.cross(cdof_ang, offset)                                           <L 522>
    var_19 = wp::cross(var_17, var_13);
    var_20 = wp::add(var_18, var_19);
    // jacr = cdof_ang                                                                        <L 523>
    var_21 = wp::copy(var_17);
    // return jacp, jacr                                                                      <L 525>
    ret_0 = var_20;
    ret_1 = var_21;
    return;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/passive.py:44
static CUDA_CALLABLE wp::vec_t<3, wp::float32> geom_semiaxes_0(
    wp::vec_t<3, wp::float32> var_size,
    wp::int32 var_geom_type)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 2;
    bool var_1;
    const wp::int32 var_2 = 0;
    wp::float32 var_3;
    wp::vec_t<3, wp::float32> var_4;
    const wp::int32 var_5 = 3;
    bool var_6;
    const wp::int32 var_7 = 0;
    wp::float32 var_8;
    const wp::int32 var_9 = 1;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::vec_t<3, wp::float32> var_12;
    const wp::int32 var_13 = 5;
    bool var_14;
    const wp::int32 var_15 = 0;
    wp::float32 var_16;
    const wp::int32 var_17 = 1;
    wp::float32 var_18;
    wp::vec_t<3, wp::float32> var_19;
    wp::float32 var_20;
    wp::float32 var_21;
    //---------
    // forward
    // def geom_semiaxes(size: wp.vec3, geom_type: int) -> wp.vec3:  # kernel_analyzer: ignore       <L 45>
    // if geom_type == GeomType.SPHERE:                                                       <L 46>
    var_1 = (var_geom_type == var_0);
    if (var_1) {
        // r = size[0]                                                                        <L 47>
        var_3 = wp::extract(var_size, var_2);
        // return wp.vec3(r, r, r)                                                            <L 48>
        var_4 = wp::vec_t<3, wp::float32>(var_3, var_3, var_3);
        return var_4;
    }
    // if geom_type == GeomType.CAPSULE:                                                      <L 50>
    var_6 = (var_geom_type == var_5);
    if (var_6) {
        // radius = size[0]                                                                   <L 51>
        var_8 = wp::extract(var_size, var_7);
        // half_length = size[1]                                                              <L 52>
        var_10 = wp::extract(var_size, var_9);
        // return wp.vec3(radius, radius, half_length + radius)                               <L 53>
        var_11 = wp::add(var_10, var_8);
        var_12 = wp::vec_t<3, wp::float32>(var_8, var_8, var_11);
        return var_12;
    }
    // if geom_type == GeomType.CYLINDER:                                                     <L 55>
    var_14 = (var_geom_type == var_13);
    if (var_14) {
        // radius = size[0]                                                                   <L 56>
        var_16 = wp::extract(var_size, var_15);
        // half_length = size[1]                                                              <L 57>
        var_18 = wp::extract(var_size, var_17);
        // return wp.vec3(radius, radius, half_length)                                        <L 58>
        var_19 = wp::vec_t<3, wp::float32>(var_16, var_16, var_18);
        return var_19;
    }
    var_20 = wp::where(var_14, var_16, var_8);
    var_21 = wp::where(var_14, var_18, var_10);
    // return size                                                                            <L 61>
    return var_size;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/passive.py:38
static CUDA_CALLABLE wp::float32 _pow4_0(
    wp::float32 var_val)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::float32 var_1;
    //---------
    // forward
    // def _pow4(val: float) -> float:                                                        <L 39>
    // sq = val * val                                                                         <L 40>
    var_0 = wp::mul(var_val, var_val);
    // return sq * sq                                                                         <L 41>
    var_1 = wp::mul(var_0, var_0);
    return var_1;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/passive.py:33
static CUDA_CALLABLE wp::float32 _pow2_0(
    wp::float32 var_val)
{
    //---------
    // primal vars
    wp::float32 var_0;
    //---------
    // forward
    // def _pow2(val: float) -> float:                                                        <L 34>
    // return val * val                                                                       <L 35>
    var_0 = wp::mul(var_val, var_val);
    return var_0;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/passive.py:64
static CUDA_CALLABLE wp::float32 ellipsoid_max_moment_0(
    wp::vec_t<3, wp::float32> var_size,
    wp::int32 var_dir)
{
    //---------
    // primal vars
    wp::float32 var_0;
    const wp::int32 var_1 = 1;
    wp::int32 var_2;
    const wp::int32 var_3 = 3;
    wp::int32 var_4;
    wp::float32 var_5;
    const wp::int32 var_6 = 2;
    wp::int32 var_7;
    const wp::int32 var_8 = 3;
    wp::int32 var_9;
    wp::float32 var_10;
    const wp::float32 var_11 = 1.6755160819145563;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    //---------
    // forward
    // def ellipsoid_max_moment(size: wp.vec3, dir: int) -> float:                            <L 65>
    // d0 = size[dir]                                                                         <L 66>
    var_0 = wp::extract(var_size, var_dir);
    // d1 = size[(dir + 1) % 3]                                                               <L 67>
    var_2 = wp::add(var_dir, var_1);
    var_4 = wp::mod(var_2, var_3);
    var_5 = wp::extract(var_size, var_4);
    // d2 = size[(dir + 2) % 3]                                                               <L 68>
    var_7 = wp::add(var_dir, var_6);
    var_9 = wp::mod(var_7, var_8);
    var_10 = wp::extract(var_size, var_9);
    // return wp.static(8.0 / 15.0 * wp.pi) * d0 * _pow4(wp.max(d1, d2))                      <L 69>
    var_12 = wp::mul(var_11, var_0);
    var_13 = wp::max(var_5, var_10);
    var_14 = _pow4_0(var_13);
    var_15 = wp::mul(var_12, var_14);
    return var_15;
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


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/warp/_src/math.py:0
static CUDA_CALLABLE wp::float32 norm_l2_0(
    wp::vec_t<3, wp::float32> var_v)
{
    //---------
    // primal vars
    wp::float32 var_0;
    //---------
    // forward
    // def norm_l2(v: Any) -> float:                                                          <L 1>
    // return wp.length(v)                                                                    <L 12>
    var_0 = wp::length(var_v);
    return var_0;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/math.py:161
static CUDA_CALLABLE wp::vec_t<3, wp::float32> quat_to_vel_0(
    wp::quat_t<wp::float32> var_quat)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    wp::float32 var_1;
    const wp::int32 var_2 = 2;
    wp::float32 var_3;
    const wp::int32 var_4 = 3;
    wp::float32 var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::float32 var_7;
    const wp::float32 var_8 = 0.0;
    bool var_9;
    const wp::float32 var_10 = 0.0;
    wp::vec_t<3, wp::float32> var_11;
    const wp::float32 var_12 = 2.0;
    const wp::int32 var_13 = 0;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    const wp::float32 var_17 = 3.141592653589793;
    bool var_18;
    const wp::float32 var_19 = 2.0;
    const wp::float32 var_20 = 3.141592653589793;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::vec_t<3, wp::float32> var_25;
    //---------
    // forward
    // def quat_to_vel(quat: wp.quat) -> wp.vec3:                                             <L 162>
    // axis = wp.vec3(quat[1], quat[2], quat[3])                                              <L 163>
    var_1 = wp::extract(var_quat, var_0);
    var_3 = wp::extract(var_quat, var_2);
    var_5 = wp::extract(var_quat, var_4);
    var_6 = wp::vec_t<3, wp::float32>(var_1, var_3, var_5);
    // sin_a_2 = wp.norm_l2(axis)                                                             <L 164>
    var_7 = norm_l2_0(var_6);
    // if sin_a_2 == 0.0:                                                                     <L 166>
    var_9 = (var_7 == var_8);
    if (var_9) {
        // return wp.vec3(0.0)                                                                <L 167>
        var_11 = wp::vec_t<3, wp::float32>(var_10);
        return var_11;
    }
    // speed = 2.0 * wp.atan2(sin_a_2, quat[0])                                               <L 169>
    var_14 = wp::extract(var_quat, var_13);
    var_15 = wp::atan2(var_7, var_14);
    var_16 = wp::mul(var_12, var_15);
    // if speed > wp.pi:                                                                      <L 171>
    var_18 = (var_16 > var_17);
    if (var_18) {
        // speed -= 2.0 * wp.pi                                                               <L 172>
        var_21 = wp::mul(var_19, var_20);
        var_22 = wp::sub(var_16, var_21);
    }
    var_23 = wp::where(var_18, var_22, var_16);
    // return axis * speed / sin_a_2                                                          <L 174>
    var_24 = wp::mul(var_6, var_23);
    var_25 = wp::div(var_24, var_7);
    return var_25;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/math.py:177
static CUDA_CALLABLE wp::vec_t<3, wp::float32> quat_sub_0(
    wp::quat_t<wp::float32> var_qa,
    wp::quat_t<wp::float32> var_qb)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::int32 var_5 = 2;
    wp::float32 var_6;
    wp::float32 var_7;
    const wp::int32 var_8 = 3;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::quat_t<wp::float32> var_11;
    wp::quat_t<wp::float32> var_12;
    wp::vec_t<3, wp::float32> var_13;
    //---------
    // forward
    // def quat_sub(qa: wp.quat, qb: wp.quat) -> wp.vec3:                                     <L 178>
    // qneg = wp.quat(qb[0], -qb[1], -qb[2], -qb[3])                                          <L 181>
    var_1 = wp::extract(var_qb, var_0);
    var_3 = wp::extract(var_qb, var_2);
    var_4 = wp::neg(var_3);
    var_6 = wp::extract(var_qb, var_5);
    var_7 = wp::neg(var_6);
    var_9 = wp::extract(var_qb, var_8);
    var_10 = wp::neg(var_9);
    var_11 = wp::quat_t<wp::float32>(var_1, var_4, var_7, var_10);
    // qdif = mul_quat(qneg, qa)                                                              <L 182>
    var_12 = mul_quat_0(var_11, var_qa);
    // return quat_to_vel(qdif)                                                               <L 185>
    var_13 = quat_to_vel_0(var_12);
    return var_13;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/support.py:67
static CUDA_CALLABLE wp::quat_t<wp::float32> mat33_to_quat_polar_0(
    wp::mat_t<3, 3, wp::float32> var_F)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    const wp::float32 var_1 = 0.0;
    const wp::float32 var_2 = 0.0;
    const wp::float32 var_3 = 1.0;
    wp::quat_t<wp::float32> var_4;
    const wp::int32 var_5 = 50;
    wp::range_t var_6;
    wp::int32 var_7;
    wp::mat_t<3, 3, wp::float32> var_8;
    wp::mat_t<3, 3, wp::float32> var_9;
    const wp::int32 var_10 = 0;
    wp::vec_t<3, wp::float32> var_11;
    const wp::int32 var_12 = 1;
    wp::vec_t<3, wp::float32> var_13;
    const wp::int32 var_14 = 2;
    wp::vec_t<3, wp::float32> var_15;
    wp::mat_t<3, 3, wp::float32> var_16;
    const wp::int32 var_17 = 0;
    wp::vec_t<3, wp::float32> var_18;
    const wp::int32 var_19 = 1;
    wp::vec_t<3, wp::float32> var_20;
    const wp::int32 var_21 = 2;
    wp::vec_t<3, wp::float32> var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::vec_t<3, wp::float32> var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::float32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    const wp::float32 var_34 = 1e-10;
    wp::float32 var_35;
    wp::vec_t<3, wp::float32> var_36;
    wp::float32 var_37;
    const wp::float32 var_38 = 1e-06;
    bool var_39;
    wp::vec_t<3, wp::float32> var_40;
    const wp::float32 var_41 = 0.5;
    wp::float32 var_42;
    const wp::int32 var_43 = 0;
    wp::float32 var_44;
    wp::float32 var_45;
    wp::float32 var_46;
    const wp::int32 var_47 = 1;
    wp::float32 var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    const wp::int32 var_51 = 2;
    wp::float32 var_52;
    wp::float32 var_53;
    wp::float32 var_54;
    wp::float32 var_55;
    wp::quat_t<wp::float32> var_56;
    wp::quat_t<wp::float32> var_57;
    wp::quat_t<wp::float32> var_58;
    //---------
    // forward
    // def mat33_to_quat_polar(F: wp.mat33) -> wp.quat:                                       <L 68>
    // cell_quat = wp.quat(0.0, 0.0, 0.0, 1.0)                                                <L 69>
    var_4 = wp::quat_t<wp::float32>(var_0, var_1, var_2, var_3);
    // for _iter in range(50):                                                                <L 70>
    var_6 = wp::range(var_5);
    start_for_0:;
        if (iter_cmp(var_6) == 0) goto end_for_0;
        var_7 = wp::iter_next(var_6);
        // rot = wp.quat_to_matrix(cell_quat)                                                 <L 71>
        var_8 = wp::quat_to_matrix(var_4);
        // rot_t = wp.transpose(rot)                                                          <L 72>
        var_9 = wp::transpose(var_8);
        // col1_rot = rot_t[0]                                                                <L 73>
        var_11 = wp::extract(var_9, var_10);
        // col2_rot = rot_t[1]                                                                <L 74>
        var_13 = wp::extract(var_9, var_12);
        // col3_rot = rot_t[2]                                                                <L 75>
        var_15 = wp::extract(var_9, var_14);
        // F_t = wp.transpose(F)                                                              <L 76>
        var_16 = wp::transpose(var_F);
        // col1_mat = F_t[0]                                                                  <L 77>
        var_18 = wp::extract(var_16, var_17);
        // col2_mat = F_t[1]                                                                  <L 78>
        var_20 = wp::extract(var_16, var_19);
        // col3_mat = F_t[2]                                                                  <L 79>
        var_22 = wp::extract(var_16, var_21);
        // omega = wp.cross(col1_rot, col1_mat) + wp.cross(col2_rot, col2_mat) + wp.cross(col3_rot, col3_mat)       <L 81>
        var_23 = wp::cross(var_11, var_18);
        var_24 = wp::cross(var_13, var_20);
        var_25 = wp::add(var_23, var_24);
        var_26 = wp::cross(var_15, var_22);
        var_27 = wp::add(var_25, var_26);
        // denom = wp.abs(wp.dot(col1_rot, col1_mat) + wp.dot(col2_rot, col2_mat) + wp.dot(col3_rot, col3_mat)) + 1.0e-10       <L 82>
        var_28 = wp::dot(var_11, var_18);
        var_29 = wp::dot(var_13, var_20);
        var_30 = wp::add(var_28, var_29);
        var_31 = wp::dot(var_15, var_22);
        var_32 = wp::add(var_30, var_31);
        var_33 = wp::abs(var_32);
        var_35 = wp::add(var_33, var_34);
        // omega = omega / denom                                                              <L 83>
        var_36 = wp::div(var_27, var_35);
        // w = wp.length(omega)                                                               <L 85>
        var_37 = wp::length(var_36);
        // if w < 1.0e-6:                                                                     <L 86>
        var_39 = (var_37 < var_38);
        if (var_39) {
            // break                                                                          <L 87>
            goto end_for_0;
        }
        // axis = omega / w                                                                   <L 89>
        var_40 = wp::div(var_36, var_37);
        // half_w = 0.5 * w                                                                   <L 90>
        var_42 = wp::mul(var_41, var_37);
        // qrot = wp.quat(                                                                    <L 91>
        // axis[0] * wp.sin(half_w),                                                          <L 92>
        var_44 = wp::extract(var_40, var_43);
        var_45 = wp::sin(var_42);
        var_46 = wp::mul(var_44, var_45);
        // axis[1] * wp.sin(half_w),                                                          <L 93>
        var_48 = wp::extract(var_40, var_47);
        var_49 = wp::sin(var_42);
        var_50 = wp::mul(var_48, var_49);
        // axis[2] * wp.sin(half_w),                                                          <L 94>
        var_52 = wp::extract(var_40, var_51);
        var_53 = wp::sin(var_42);
        var_54 = wp::mul(var_52, var_53);
        // wp.cos(half_w),                                                                    <L 95>
        var_55 = wp::cos(var_42);
        var_56 = wp::quat_t<wp::float32>(var_46, var_50, var_54, var_55);
        // cell_quat = wp.normalize(qrot * cell_quat)                                         <L 97>
        var_57 = wp::mul(var_56, var_4);
        var_58 = wp::normalize(var_57);
        wp::assign(var_4, var_58);
        goto start_for_0;
    end_for_0:;
    // return cell_quat                                                                       <L 98>
    return var_4;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/support.py:101
static CUDA_CALLABLE wp::quat_t<wp::float32> compute_interp_cell_quat_0(
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexnode_xpos_in,
    wp::int32 var_order,
    wp::int32 var_ci,
    wp::int32 var_cj,
    wp::int32 var_ck,
    wp::int32 var_cy,
    wp::int32 var_cz,
    wp::int32 var_ny_g,
    wp::int32 var_nz_g,
    wp::int32 var_nstart,
    wp::int32 var_worldid)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    wp::int32 var_1;
    const wp::int32 var_2 = 1;
    wp::int32 var_3;
    wp::int32 var_4;
    const wp::int32 var_5 = 1;
    wp::int32 var_6;
    wp::int32 var_7;
    const wp::float32 var_8 = 0.0;
    wp::mat_t<3, 3, wp::float32> var_9;
    const wp::int32 var_10 = 0;
    wp::int32 var_11;
    const wp::int32 var_12 = 1;
    wp::int32 var_13;
    wp::range_t var_14;
    wp::int32 var_15;
    const wp::int32 var_16 = 1;
    wp::int32 var_17;
    wp::range_t var_18;
    wp::int32 var_19;
    const wp::int32 var_20 = 1;
    wp::int32 var_21;
    wp::range_t var_22;
    wp::int32 var_23;
    bool var_24;
    wp::int32 var_25;
    wp::int32 var_26;
    wp::int32 var_27;
    wp::int32 var_28;
    wp::int32 var_29;
    wp::int32 var_30;
    wp::int32 var_31;
    wp::int32 var_32;
    wp::int32 var_33;
    wp::int32 var_34;
    wp::int32 var_35;
    wp::int32 var_36;
    wp::vec_t<3, wp::float32>* var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::vec_t<3, wp::float32> var_39;
    const wp::int32 var_40 = 0;
    bool var_41;
    const wp::int32 var_42 = -1;
    wp::float32 var_43;
    const wp::int32 var_44 = 1;
    wp::float32 var_45;
    wp::float32 var_46;
    const wp::int32 var_47 = 0;
    bool var_48;
    const wp::int32 var_49 = -1;
    wp::float32 var_50;
    const wp::int32 var_51 = 1;
    wp::float32 var_52;
    wp::float32 var_53;
    const wp::int32 var_54 = 0;
    bool var_55;
    const wp::int32 var_56 = -1;
    wp::float32 var_57;
    const wp::int32 var_58 = 1;
    wp::float32 var_59;
    wp::float32 var_60;
    const wp::float32 var_61 = 0.5;
    wp::float32 var_62;
    const wp::float32 var_63 = 0.5;
    wp::float32 var_64;
    const wp::float32 var_65 = 0.5;
    wp::float32 var_66;
    wp::float32 var_67;
    wp::float32 var_68;
    wp::float32 var_69;
    wp::float32 var_70;
    wp::float32 var_71;
    wp::float32 var_72;
    const wp::int32 var_73 = 0;
    wp::float32 var_74;
    wp::float32 var_75;
    const wp::int32 var_76 = 0;
    wp::float32 var_77;
    wp::float32 var_78;
    const wp::int32 var_79 = 1;
    wp::float32 var_80;
    wp::float32 var_81;
    const wp::int32 var_82 = 2;
    const wp::int32 var_83 = 1;
    wp::float32 var_84;
    wp::float32 var_85;
    const wp::int32 var_86 = 0;
    wp::float32 var_87;
    wp::float32 var_88;
    const wp::int32 var_89 = 1;
    wp::float32 var_90;
    wp::float32 var_91;
    const wp::int32 var_92 = 2;
    const wp::int32 var_93 = 2;
    wp::float32 var_94;
    wp::float32 var_95;
    const wp::int32 var_96 = 0;
    wp::float32 var_97;
    wp::float32 var_98;
    const wp::int32 var_99 = 1;
    wp::float32 var_100;
    wp::float32 var_101;
    const wp::int32 var_102 = 2;
    const wp::int32 var_103 = 1;
    wp::int32 var_104;
    wp::int32 var_105;
    wp::quat_t<wp::float32> var_106;
    //---------
    // forward
    // def compute_interp_cell_quat(                                                          <L 102>
    // npc = (order + 1) * (order + 1) * (order + 1)                                          <L 118>
    var_1 = wp::add(var_order, var_0);
    var_3 = wp::add(var_order, var_2);
    var_4 = wp::mul(var_1, var_3);
    var_6 = wp::add(var_order, var_5);
    var_7 = wp::mul(var_4, var_6);
    // F = wp.mat33(0.0)                                                                      <L 119>
    var_9 = wp::mat_t<3, 3, wp::float32>(var_8);
    // idx = int(0)                                                                           <L 120>
    var_11 = wp::int(var_10);
    // for li in range(order + 1):                                                            <L 121>
    var_13 = wp::add(var_order, var_12);
    var_14 = wp::range(var_13);
    start_for_0:;
        if (iter_cmp(var_14) == 0) goto end_for_0;
        var_15 = wp::iter_next(var_14);
        // for lj in range(order + 1):                                                        <L 122>
        var_17 = wp::add(var_order, var_16);
        var_18 = wp::range(var_17);
        start_for_2:;
            if (iter_cmp(var_18) == 0) goto end_for_2;
            var_19 = wp::iter_next(var_18);
            // for lk in range(order + 1):                                                    <L 123>
            var_21 = wp::add(var_order, var_20);
            var_22 = wp::range(var_21);
            start_for_4:;
                if (iter_cmp(var_22) == 0) goto end_for_4;
                var_23 = wp::iter_next(var_22);
                // if idx < npc:                                                              <L 124>
                var_24 = (var_11 < var_7);
                if (var_24) {
                    // gi = ci * order + li                                                   <L 125>
                    var_25 = wp::mul(var_ci, var_order);
                    var_26 = wp::add(var_25, var_15);
                    // gj = cj * order + lj                                                   <L 126>
                    var_27 = wp::mul(var_cj, var_order);
                    var_28 = wp::add(var_27, var_19);
                    // gk = ck * order + lk                                                   <L 127>
                    var_29 = wp::mul(var_ck, var_order);
                    var_30 = wp::add(var_29, var_23);
                    // gidx = gi * ny_g * nz_g + gj * nz_g + gk                               <L 128>
                    var_31 = wp::mul(var_26, var_ny_g);
                    var_32 = wp::mul(var_31, var_nz_g);
                    var_33 = wp::mul(var_28, var_nz_g);
                    var_34 = wp::add(var_32, var_33);
                    var_35 = wp::add(var_34, var_30);
                    // node_pos = flexnode_xpos_in[worldid, nstart + gidx]                    <L 130>
                    var_36 = wp::add(var_nstart, var_35);
                    var_37 = wp::address(var_flexnode_xpos_in, var_worldid, var_36);
                    var_39 = wp::load(var_37);
                    var_38 = wp::copy(var_39);
                    // dphi_x = float(-1) if li == 0 else float(1)                            <L 132>
                    var_41 = (var_15 == var_40);
                    if (var_41) {
                        var_43 = wp::float(var_42);
                    }
                    if (!var_41) {
                        var_45 = wp::float(var_44);
                    }
                    var_46 = wp::where(var_41, var_43, var_45);
                    // dphi_y = float(-1) if lj == 0 else float(1)                            <L 133>
                    var_48 = (var_19 == var_47);
                    if (var_48) {
                        var_50 = wp::float(var_49);
                    }
                    if (!var_48) {
                        var_52 = wp::float(var_51);
                    }
                    var_53 = wp::where(var_48, var_50, var_52);
                    // dphi_z = float(-1) if lk == 0 else float(1)                            <L 134>
                    var_55 = (var_23 == var_54);
                    if (var_55) {
                        var_57 = wp::float(var_56);
                    }
                    if (!var_55) {
                        var_59 = wp::float(var_58);
                    }
                    var_60 = wp::where(var_55, var_57, var_59);
                    // phi_x = float(0.5)                                                     <L 135>
                    var_62 = wp::float(var_61);
                    // phi_y = float(0.5)                                                     <L 136>
                    var_64 = wp::float(var_63);
                    // phi_z = float(0.5)                                                     <L 137>
                    var_66 = wp::float(var_65);
                    // grad_x = dphi_x * phi_y * phi_z                                        <L 139>
                    var_67 = wp::mul(var_46, var_64);
                    var_68 = wp::mul(var_67, var_66);
                    // grad_y = phi_x * dphi_y * phi_z                                        <L 140>
                    var_69 = wp::mul(var_62, var_53);
                    var_70 = wp::mul(var_69, var_66);
                    // grad_z = phi_x * phi_y * dphi_z                                        <L 141>
                    var_71 = wp::mul(var_62, var_64);
                    var_72 = wp::mul(var_71, var_60);
                    // for r in range(3):                                                     <L 143>
                    // F[r, 0] += node_pos[r] * grad_x                                        <L 144>
                    var_74 = wp::extract(var_38, var_73);
                    var_75 = wp::mul(var_74, var_68);
                    wp::add_inplace(var_9, var_73, var_76, var_75);
                    // F[r, 1] += node_pos[r] * grad_y                                        <L 145>
                    var_77 = wp::extract(var_38, var_73);
                    var_78 = wp::mul(var_77, var_70);
                    wp::add_inplace(var_9, var_73, var_79, var_78);
                    // F[r, 2] += node_pos[r] * grad_z                                        <L 146>
                    var_80 = wp::extract(var_38, var_73);
                    var_81 = wp::mul(var_80, var_72);
                    wp::add_inplace(var_9, var_73, var_82, var_81);
                    // F[r, 0] += node_pos[r] * grad_x                                        <L 144>
                    var_84 = wp::extract(var_38, var_83);
                    var_85 = wp::mul(var_84, var_68);
                    wp::add_inplace(var_9, var_83, var_86, var_85);
                    // F[r, 1] += node_pos[r] * grad_y                                        <L 145>
                    var_87 = wp::extract(var_38, var_83);
                    var_88 = wp::mul(var_87, var_70);
                    wp::add_inplace(var_9, var_83, var_89, var_88);
                    // F[r, 2] += node_pos[r] * grad_z                                        <L 146>
                    var_90 = wp::extract(var_38, var_83);
                    var_91 = wp::mul(var_90, var_72);
                    wp::add_inplace(var_9, var_83, var_92, var_91);
                    // F[r, 0] += node_pos[r] * grad_x                                        <L 144>
                    var_94 = wp::extract(var_38, var_93);
                    var_95 = wp::mul(var_94, var_68);
                    wp::add_inplace(var_9, var_93, var_96, var_95);
                    // F[r, 1] += node_pos[r] * grad_y                                        <L 145>
                    var_97 = wp::extract(var_38, var_93);
                    var_98 = wp::mul(var_97, var_70);
                    wp::add_inplace(var_9, var_93, var_99, var_98);
                    // F[r, 2] += node_pos[r] * grad_z                                        <L 146>
                    var_100 = wp::extract(var_38, var_93);
                    var_101 = wp::mul(var_100, var_72);
                    wp::add_inplace(var_9, var_93, var_102, var_101);
                    // idx += 1                                                               <L 148>
                    var_104 = wp::add(var_11, var_103);
                }
                var_105 = wp::where(var_24, var_104, var_11);
                wp::assign(var_11, var_105);
                goto start_for_4;
            end_for_4:;
            goto start_for_2;
        end_for_2:;
        goto start_for_0;
    end_for_0:;
    // return mat33_to_quat_polar(F)                                                          <L 150>
    var_106 = mat33_to_quat_polar_0(var_9);
    return var_106;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/util_misc.py:708
static CUDA_CALLABLE void adj__poly_force_0(
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


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/support.py:1270
static CUDA_CALLABLE void adj_flex_dphi_0(
    wp::float32 var_s,
    wp::int32 var_i,
    wp::int32 var_order,
    wp::float32 & adj_s,
    wp::int32 & adj_i,
    wp::int32 & adj_order,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/support.py:1265
static CUDA_CALLABLE void adj_flex_phi_0(
    wp::float32 var_s,
    wp::int32 var_i,
    wp::int32 var_order,
    wp::float32 & adj_s,
    wp::int32 & adj_i,
    wp::int32 & adj_order,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/support.py:1275
static CUDA_CALLABLE void adj_dphi2D_0(
    wp::float32 var_s0,
    wp::int32 var_l0,
    wp::float32 var_s1,
    wp::int32 var_l1,
    wp::int32 var_order,
    wp::int32 var_direction,
    wp::float32 & adj_s0,
    wp::int32 & adj_l0,
    wp::float32 & adj_s1,
    wp::int32 & adj_l1,
    wp::int32 & adj_order,
    wp::int32 & adj_direction,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/passive.py:980
static CUDA_CALLABLE void adj__apply_face_forces_0(
    wp::array_t<wp::int32> var_flex_nodebodyid,
    wp::array_t<wp::int32> var_flex_face,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexnode_xpos_in,
    wp::int32 var_face_id,
    wp::vec_t<2, wp::float32> var_local_coords,
    wp::vec_t<3, wp::float32> var_wt1,
    wp::vec_t<3, wp::float32> var_wt2,
    wp::float32 var_stiffness_scale,
    wp::int32 var_order_abs,
    wp::int32 var_worldid,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_force_out,
    wp::array_t<wp::int32> & adj_flex_nodebodyid,
    wp::array_t<wp::int32> & adj_flex_face,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_flexnode_xpos_in,
    wp::int32 & adj_face_id,
    wp::vec_t<2, wp::float32> & adj_local_coords,
    wp::vec_t<3, wp::float32> & adj_wt1,
    wp::vec_t<3, wp::float32> & adj_wt2,
    wp::float32 & adj_stiffness_scale,
    wp::int32 & adj_order_abs,
    wp::int32 & adj_worldid,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_body_force_out)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/support.py:497
static CUDA_CALLABLE void adj_jac_dof_0(
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::int32> var_body_isdofancestor,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::vec_t<3, wp::float32> var_point,
    wp::int32 var_bodyid,
    wp::int32 var_dofid,
    wp::int32 var_worldid,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::array_t<wp::int32> & adj_body_parentid,
    wp::array_t<wp::int32> & adj_body_rootid,
    wp::array_t<wp::int32> & adj_dof_bodyid,
    wp::array_t<wp::int32> & adj_body_isdofancestor,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cdof_in,
    wp::vec_t<3, wp::float32> & adj_point,
    wp::int32 & adj_bodyid,
    wp::int32 & adj_dofid,
    wp::int32 & adj_worldid,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/passive.py:44
static CUDA_CALLABLE void adj_geom_semiaxes_0(
    wp::vec_t<3, wp::float32> var_size,
    wp::int32 var_geom_type,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::int32 & adj_geom_type,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/passive.py:38
static CUDA_CALLABLE void adj__pow4_0(
    wp::float32 var_val,
    wp::float32 & adj_val,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/passive.py:33
static CUDA_CALLABLE void adj__pow2_0(
    wp::float32 var_val,
    wp::float32 & adj_val,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/passive.py:64
static CUDA_CALLABLE void adj_ellipsoid_max_moment_0(
    wp::vec_t<3, wp::float32> var_size,
    wp::int32 var_dir,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::int32 & adj_dir,
    wp::float32 & adj_ret)
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


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/warp/_src/math.py:0
static CUDA_CALLABLE void adj_norm_l2_0(
    wp::vec_t<3, wp::float32> var_v,
    wp::vec_t<3, wp::float32> & adj_v,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/math.py:161
static CUDA_CALLABLE void adj_quat_to_vel_0(
    wp::quat_t<wp::float32> var_quat,
    wp::quat_t<wp::float32> & adj_quat,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/math.py:177
static CUDA_CALLABLE void adj_quat_sub_0(
    wp::quat_t<wp::float32> var_qa,
    wp::quat_t<wp::float32> var_qb,
    wp::quat_t<wp::float32> & adj_qa,
    wp::quat_t<wp::float32> & adj_qb,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/support.py:67
static CUDA_CALLABLE void adj_mat33_to_quat_polar_0(
    wp::mat_t<3, 3, wp::float32> var_F,
    wp::mat_t<3, 3, wp::float32> & adj_F,
    wp::quat_t<wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/support.py:101
static CUDA_CALLABLE void adj_compute_interp_cell_quat_0(
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexnode_xpos_in,
    wp::int32 var_order,
    wp::int32 var_ci,
    wp::int32 var_cj,
    wp::int32 var_ck,
    wp::int32 var_cy,
    wp::int32 var_cz,
    wp::int32 var_ny_g,
    wp::int32 var_nz_g,
    wp::int32 var_nstart,
    wp::int32 var_worldid,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_flexnode_xpos_in,
    wp::int32 & adj_order,
    wp::int32 & adj_ci,
    wp::int32 & adj_cj,
    wp::int32 & adj_ck,
    wp::int32 & adj_cy,
    wp::int32 & adj_cz,
    wp::int32 & adj_ny_g,
    wp::int32 & adj_nz_g,
    wp::int32 & adj_nstart,
    wp::int32 & adj_worldid,
    wp::quat_t<wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void _flex_bending_c3cc3c88_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_nflex,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_flex_dim,
    wp::array_t<wp::int32> var_flex_vertadr,
    wp::array_t<wp::int32> var_flex_edgeadr,
    wp::array_t<wp::int32> var_flex_edgenum,
    wp::array_t<wp::int32> var_flex_bendingadr,
    wp::array_t<wp::int32> var_flex_vertbodyid,
    wp::array_t<wp::vec_t<2, wp::int32>> var_flex_edge,
    wp::array_t<wp::vec_t<2, wp::int32>> var_flex_edgeflap,
    wp::array_t<wp::float32> var_flex_bending,
    wp::array_t<wp::float32> var_flex_damping,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexvert_xpos_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    bool var_dsbl_damper,
    wp::array_t<wp::vec_t<6, wp::float32>> var_flex_spring_body_force_out,
    wp::array_t<wp::vec_t<6, wp::float32>> var_flex_damper_body_force_out)
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
        wp::int32* var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        bool var_7;
        const wp::int32 var_8 = 0;
        bool var_9;
        wp::int32* var_10;
        bool var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        const wp::int32 var_17 = 0;
        bool var_18;
        wp::int32* var_19;
        const wp::int32 var_20 = 2;
        bool var_21;
        wp::int32 var_22;
        wp::vec_t<2, wp::int32>* var_23;
        const wp::int32 var_24 = 1;
        wp::int32 var_25;
        wp::vec_t<2, wp::int32> var_26;
        const wp::int32 var_27 = -1;
        bool var_28;
        wp::int32* var_29;
        wp::vec_t<2, wp::int32>* var_30;
        const wp::int32 var_31 = 0;
        wp::int32 var_32;
        wp::vec_t<2, wp::int32> var_33;
        wp::int32 var_34;
        wp::int32 var_35;
        wp::int32* var_36;
        wp::vec_t<2, wp::int32>* var_37;
        const wp::int32 var_38 = 1;
        wp::int32 var_39;
        wp::vec_t<2, wp::int32> var_40;
        wp::int32 var_41;
        wp::int32 var_42;
        wp::int32* var_43;
        wp::vec_t<2, wp::int32>* var_44;
        const wp::int32 var_45 = 0;
        wp::int32 var_46;
        wp::vec_t<2, wp::int32> var_47;
        wp::int32 var_48;
        wp::int32 var_49;
        wp::int32* var_50;
        wp::vec_t<2, wp::int32>* var_51;
        const wp::int32 var_52 = 1;
        wp::int32 var_53;
        wp::vec_t<2, wp::int32> var_54;
        wp::int32 var_55;
        wp::int32 var_56;
        wp::vec_t<4, wp::int32> var_57;
        wp::mat_t<4, 3, wp::float32> var_58;
        const wp::int32 var_59 = 17;
        wp::int32 var_60;
        wp::int32 var_61;
        const wp::int32 var_62 = 16;
        wp::int32 var_63;
        wp::float32* var_64;
        wp::float32 var_65;
        const wp::int32 var_66 = 0;
        wp::int32 var_67;
        wp::vec_t<3, wp::float32>* var_68;
        wp::vec_t<3, wp::float32> var_69;
        wp::vec_t<3, wp::float32> var_70;
        const wp::int32 var_71 = 1;
        wp::int32 var_72;
        wp::vec_t<3, wp::float32>* var_73;
        wp::vec_t<3, wp::float32> var_74;
        wp::vec_t<3, wp::float32> var_75;
        const wp::int32 var_76 = 2;
        wp::int32 var_77;
        wp::vec_t<3, wp::float32>* var_78;
        wp::vec_t<3, wp::float32> var_79;
        wp::vec_t<3, wp::float32> var_80;
        const wp::int32 var_81 = 3;
        wp::int32 var_82;
        wp::vec_t<3, wp::float32>* var_83;
        wp::vec_t<3, wp::float32> var_84;
        wp::vec_t<3, wp::float32> var_85;
        wp::vec_t<3, wp::float32> var_86;
        wp::vec_t<3, wp::float32> var_87;
        wp::vec_t<3, wp::float32> var_88;
        wp::vec_t<3, wp::float32> var_89;
        const wp::int32 var_90 = 1;
        wp::vec_t<3, wp::float32> var_91;
        const wp::int32 var_92 = 2;
        wp::vec_t<3, wp::float32> var_93;
        const wp::int32 var_94 = 3;
        const wp::int32 var_95 = 1;
        wp::vec_t<3, wp::float32> var_96;
        const wp::int32 var_97 = 2;
        wp::vec_t<3, wp::float32> var_98;
        wp::vec_t<3, wp::float32> var_99;
        const wp::int32 var_100 = 3;
        wp::vec_t<3, wp::float32> var_101;
        wp::vec_t<3, wp::float32> var_102;
        wp::vec_t<3, wp::float32> var_103;
        const wp::int32 var_104 = 0;
        wp::float32 var_105;
        wp::mat_t<4, 3, wp::float32> var_106;
        bool var_107;
        bool var_108;
        wp::float32* var_109;
        const wp::float32 var_110 = 0.0;
        bool var_111;
        wp::float32 var_112;
        const wp::int32 var_113 = 0;
        wp::int32 var_114;
        wp::int32* var_115;
        wp::int32 var_116;
        wp::int32 var_117;
        wp::vec_t<6, wp::float32>* var_118;
        wp::vec_t<6, wp::float32> var_119;
        wp::vec_t<6, wp::float32> var_120;
        wp::vec_t<3, wp::float32> var_121;
        wp::vec_t<3, wp::float32> var_122;
        wp::int32* var_123;
        wp::vec_t<3, wp::float32>* var_124;
        wp::int32 var_125;
        wp::vec_t<3, wp::float32> var_126;
        wp::vec_t<3, wp::float32> var_127;
        wp::int32 var_128;
        wp::vec_t<3, wp::float32>* var_129;
        wp::vec_t<3, wp::float32> var_130;
        wp::vec_t<3, wp::float32> var_131;
        wp::vec_t<3, wp::float32> var_132;
        wp::vec_t<3, wp::float32> var_133;
        const wp::int32 var_134 = 1;
        wp::int32 var_135;
        wp::int32* var_136;
        wp::int32 var_137;
        wp::int32 var_138;
        wp::vec_t<6, wp::float32>* var_139;
        wp::vec_t<6, wp::float32> var_140;
        wp::vec_t<6, wp::float32> var_141;
        wp::vec_t<3, wp::float32> var_142;
        wp::vec_t<3, wp::float32> var_143;
        wp::int32* var_144;
        wp::vec_t<3, wp::float32>* var_145;
        wp::int32 var_146;
        wp::vec_t<3, wp::float32> var_147;
        wp::vec_t<3, wp::float32> var_148;
        wp::int32 var_149;
        wp::vec_t<3, wp::float32>* var_150;
        wp::vec_t<3, wp::float32> var_151;
        wp::vec_t<3, wp::float32> var_152;
        wp::vec_t<3, wp::float32> var_153;
        wp::vec_t<3, wp::float32> var_154;
        const wp::int32 var_155 = 2;
        wp::int32 var_156;
        wp::int32* var_157;
        wp::int32 var_158;
        wp::int32 var_159;
        wp::vec_t<6, wp::float32>* var_160;
        wp::vec_t<6, wp::float32> var_161;
        wp::vec_t<6, wp::float32> var_162;
        wp::vec_t<3, wp::float32> var_163;
        wp::vec_t<3, wp::float32> var_164;
        wp::int32* var_165;
        wp::vec_t<3, wp::float32>* var_166;
        wp::int32 var_167;
        wp::vec_t<3, wp::float32> var_168;
        wp::vec_t<3, wp::float32> var_169;
        wp::int32 var_170;
        wp::vec_t<3, wp::float32>* var_171;
        wp::vec_t<3, wp::float32> var_172;
        wp::vec_t<3, wp::float32> var_173;
        wp::vec_t<3, wp::float32> var_174;
        wp::vec_t<3, wp::float32> var_175;
        const wp::int32 var_176 = 3;
        wp::int32 var_177;
        wp::int32* var_178;
        wp::int32 var_179;
        wp::int32 var_180;
        wp::vec_t<6, wp::float32>* var_181;
        wp::vec_t<6, wp::float32> var_182;
        wp::vec_t<6, wp::float32> var_183;
        wp::vec_t<3, wp::float32> var_184;
        wp::vec_t<3, wp::float32> var_185;
        wp::int32* var_186;
        wp::vec_t<3, wp::float32>* var_187;
        wp::int32 var_188;
        wp::vec_t<3, wp::float32> var_189;
        wp::vec_t<3, wp::float32> var_190;
        wp::int32 var_191;
        wp::vec_t<3, wp::float32>* var_192;
        wp::vec_t<3, wp::float32> var_193;
        wp::vec_t<3, wp::float32> var_194;
        wp::vec_t<3, wp::float32> var_195;
        wp::vec_t<3, wp::float32> var_196;
        wp::mat_t<4, 3, wp::float32> var_197;
        wp::mat_t<4, 3, wp::float32> var_198;
        const wp::int32 var_199 = 0;
        const wp::int32 var_200 = 0;
        const wp::float32 var_201 = 0.0;
        wp::float32 var_202;
        const wp::float32 var_203 = 0.0;
        wp::float32 var_204;
        const wp::int32 var_205 = 0;
        const wp::int32 var_206 = 17;
        wp::int32 var_207;
        wp::int32 var_208;
        const wp::int32 var_209 = 4;
        wp::int32 var_210;
        wp::int32 var_211;
        wp::int32 var_212;
        wp::float32* var_213;
        wp::float32 var_214;
        wp::float32 var_215;
        wp::int32 var_216;
        wp::vec_t<3, wp::float32>* var_217;
        wp::float32 var_218;
        wp::vec_t<3, wp::float32> var_219;
        wp::float32 var_220;
        wp::float32 var_221;
        bool var_222;
        bool var_223;
        wp::float32* var_224;
        const wp::float32 var_225 = 0.0;
        bool var_226;
        wp::float32 var_227;
        wp::float32 var_228;
        wp::float32 var_229;
        wp::float32 var_230;
        wp::float32 var_231;
        const wp::int32 var_232 = 1;
        const wp::int32 var_233 = 17;
        wp::int32 var_234;
        wp::int32 var_235;
        const wp::int32 var_236 = 4;
        wp::int32 var_237;
        wp::int32 var_238;
        wp::int32 var_239;
        wp::float32* var_240;
        wp::float32 var_241;
        wp::float32 var_242;
        wp::int32 var_243;
        wp::vec_t<3, wp::float32>* var_244;
        wp::float32 var_245;
        wp::vec_t<3, wp::float32> var_246;
        wp::float32 var_247;
        wp::float32 var_248;
        bool var_249;
        bool var_250;
        wp::float32* var_251;
        const wp::float32 var_252 = 0.0;
        bool var_253;
        wp::float32 var_254;
        wp::float32 var_255;
        wp::float32 var_256;
        wp::float32 var_257;
        wp::float32 var_258;
        const wp::int32 var_259 = 2;
        const wp::int32 var_260 = 17;
        wp::int32 var_261;
        wp::int32 var_262;
        const wp::int32 var_263 = 4;
        wp::int32 var_264;
        wp::int32 var_265;
        wp::int32 var_266;
        wp::float32* var_267;
        wp::float32 var_268;
        wp::float32 var_269;
        wp::int32 var_270;
        wp::vec_t<3, wp::float32>* var_271;
        wp::float32 var_272;
        wp::vec_t<3, wp::float32> var_273;
        wp::float32 var_274;
        wp::float32 var_275;
        bool var_276;
        bool var_277;
        wp::float32* var_278;
        const wp::float32 var_279 = 0.0;
        bool var_280;
        wp::float32 var_281;
        wp::float32 var_282;
        wp::float32 var_283;
        wp::float32 var_284;
        wp::float32 var_285;
        const wp::int32 var_286 = 3;
        const wp::int32 var_287 = 17;
        wp::int32 var_288;
        wp::int32 var_289;
        const wp::int32 var_290 = 4;
        wp::int32 var_291;
        wp::int32 var_292;
        wp::int32 var_293;
        wp::float32* var_294;
        wp::float32 var_295;
        wp::float32 var_296;
        wp::int32 var_297;
        wp::vec_t<3, wp::float32>* var_298;
        wp::float32 var_299;
        wp::vec_t<3, wp::float32> var_300;
        wp::float32 var_301;
        wp::float32 var_302;
        bool var_303;
        bool var_304;
        wp::float32* var_305;
        const wp::float32 var_306 = 0.0;
        bool var_307;
        wp::float32 var_308;
        wp::float32 var_309;
        wp::float32 var_310;
        wp::float32 var_311;
        wp::float32 var_312;
        const wp::int32 var_313 = 17;
        wp::int32 var_314;
        wp::int32 var_315;
        const wp::int32 var_316 = 16;
        wp::int32 var_317;
        wp::float32* var_318;
        wp::float32 var_319;
        wp::float32 var_320;
        wp::float32 var_321;
        wp::float32 var_322;
        wp::float32 var_323;
        bool var_324;
        bool var_325;
        wp::float32* var_326;
        const wp::float32 var_327 = 0.0;
        bool var_328;
        wp::float32 var_329;
        wp::float32 var_330;
        const wp::int32 var_331 = 1;
        const wp::float32 var_332 = 0.0;
        wp::float32 var_333;
        const wp::float32 var_334 = 0.0;
        wp::float32 var_335;
        const wp::int32 var_336 = 0;
        const wp::int32 var_337 = 17;
        wp::int32 var_338;
        wp::int32 var_339;
        const wp::int32 var_340 = 4;
        wp::int32 var_341;
        wp::int32 var_342;
        wp::int32 var_343;
        wp::float32* var_344;
        wp::float32 var_345;
        wp::float32 var_346;
        wp::int32 var_347;
        wp::vec_t<3, wp::float32>* var_348;
        wp::float32 var_349;
        wp::vec_t<3, wp::float32> var_350;
        wp::float32 var_351;
        wp::float32 var_352;
        bool var_353;
        bool var_354;
        wp::float32* var_355;
        const wp::float32 var_356 = 0.0;
        bool var_357;
        wp::float32 var_358;
        wp::float32 var_359;
        wp::float32 var_360;
        wp::float32 var_361;
        wp::float32 var_362;
        const wp::int32 var_363 = 1;
        const wp::int32 var_364 = 17;
        wp::int32 var_365;
        wp::int32 var_366;
        const wp::int32 var_367 = 4;
        wp::int32 var_368;
        wp::int32 var_369;
        wp::int32 var_370;
        wp::float32* var_371;
        wp::float32 var_372;
        wp::float32 var_373;
        wp::int32 var_374;
        wp::vec_t<3, wp::float32>* var_375;
        wp::float32 var_376;
        wp::vec_t<3, wp::float32> var_377;
        wp::float32 var_378;
        wp::float32 var_379;
        bool var_380;
        bool var_381;
        wp::float32* var_382;
        const wp::float32 var_383 = 0.0;
        bool var_384;
        wp::float32 var_385;
        wp::float32 var_386;
        wp::float32 var_387;
        wp::float32 var_388;
        wp::float32 var_389;
        const wp::int32 var_390 = 2;
        const wp::int32 var_391 = 17;
        wp::int32 var_392;
        wp::int32 var_393;
        const wp::int32 var_394 = 4;
        wp::int32 var_395;
        wp::int32 var_396;
        wp::int32 var_397;
        wp::float32* var_398;
        wp::float32 var_399;
        wp::float32 var_400;
        wp::int32 var_401;
        wp::vec_t<3, wp::float32>* var_402;
        wp::float32 var_403;
        wp::vec_t<3, wp::float32> var_404;
        wp::float32 var_405;
        wp::float32 var_406;
        bool var_407;
        bool var_408;
        wp::float32* var_409;
        const wp::float32 var_410 = 0.0;
        bool var_411;
        wp::float32 var_412;
        wp::float32 var_413;
        wp::float32 var_414;
        wp::float32 var_415;
        wp::float32 var_416;
        const wp::int32 var_417 = 3;
        const wp::int32 var_418 = 17;
        wp::int32 var_419;
        wp::int32 var_420;
        const wp::int32 var_421 = 4;
        wp::int32 var_422;
        wp::int32 var_423;
        wp::int32 var_424;
        wp::float32* var_425;
        wp::float32 var_426;
        wp::float32 var_427;
        wp::int32 var_428;
        wp::vec_t<3, wp::float32>* var_429;
        wp::float32 var_430;
        wp::vec_t<3, wp::float32> var_431;
        wp::float32 var_432;
        wp::float32 var_433;
        bool var_434;
        bool var_435;
        wp::float32* var_436;
        const wp::float32 var_437 = 0.0;
        bool var_438;
        wp::float32 var_439;
        wp::float32 var_440;
        wp::float32 var_441;
        wp::float32 var_442;
        wp::float32 var_443;
        const wp::int32 var_444 = 17;
        wp::int32 var_445;
        wp::int32 var_446;
        const wp::int32 var_447 = 16;
        wp::int32 var_448;
        wp::float32* var_449;
        wp::float32 var_450;
        wp::float32 var_451;
        wp::float32 var_452;
        wp::float32 var_453;
        wp::float32 var_454;
        bool var_455;
        bool var_456;
        wp::float32* var_457;
        const wp::float32 var_458 = 0.0;
        bool var_459;
        wp::float32 var_460;
        wp::float32 var_461;
        const wp::int32 var_462 = 2;
        const wp::float32 var_463 = 0.0;
        wp::float32 var_464;
        const wp::float32 var_465 = 0.0;
        wp::float32 var_466;
        const wp::int32 var_467 = 0;
        const wp::int32 var_468 = 17;
        wp::int32 var_469;
        wp::int32 var_470;
        const wp::int32 var_471 = 4;
        wp::int32 var_472;
        wp::int32 var_473;
        wp::int32 var_474;
        wp::float32* var_475;
        wp::float32 var_476;
        wp::float32 var_477;
        wp::int32 var_478;
        wp::vec_t<3, wp::float32>* var_479;
        wp::float32 var_480;
        wp::vec_t<3, wp::float32> var_481;
        wp::float32 var_482;
        wp::float32 var_483;
        bool var_484;
        bool var_485;
        wp::float32* var_486;
        const wp::float32 var_487 = 0.0;
        bool var_488;
        wp::float32 var_489;
        wp::float32 var_490;
        wp::float32 var_491;
        wp::float32 var_492;
        wp::float32 var_493;
        const wp::int32 var_494 = 1;
        const wp::int32 var_495 = 17;
        wp::int32 var_496;
        wp::int32 var_497;
        const wp::int32 var_498 = 4;
        wp::int32 var_499;
        wp::int32 var_500;
        wp::int32 var_501;
        wp::float32* var_502;
        wp::float32 var_503;
        wp::float32 var_504;
        wp::int32 var_505;
        wp::vec_t<3, wp::float32>* var_506;
        wp::float32 var_507;
        wp::vec_t<3, wp::float32> var_508;
        wp::float32 var_509;
        wp::float32 var_510;
        bool var_511;
        bool var_512;
        wp::float32* var_513;
        const wp::float32 var_514 = 0.0;
        bool var_515;
        wp::float32 var_516;
        wp::float32 var_517;
        wp::float32 var_518;
        wp::float32 var_519;
        wp::float32 var_520;
        const wp::int32 var_521 = 2;
        const wp::int32 var_522 = 17;
        wp::int32 var_523;
        wp::int32 var_524;
        const wp::int32 var_525 = 4;
        wp::int32 var_526;
        wp::int32 var_527;
        wp::int32 var_528;
        wp::float32* var_529;
        wp::float32 var_530;
        wp::float32 var_531;
        wp::int32 var_532;
        wp::vec_t<3, wp::float32>* var_533;
        wp::float32 var_534;
        wp::vec_t<3, wp::float32> var_535;
        wp::float32 var_536;
        wp::float32 var_537;
        bool var_538;
        bool var_539;
        wp::float32* var_540;
        const wp::float32 var_541 = 0.0;
        bool var_542;
        wp::float32 var_543;
        wp::float32 var_544;
        wp::float32 var_545;
        wp::float32 var_546;
        wp::float32 var_547;
        const wp::int32 var_548 = 3;
        const wp::int32 var_549 = 17;
        wp::int32 var_550;
        wp::int32 var_551;
        const wp::int32 var_552 = 4;
        wp::int32 var_553;
        wp::int32 var_554;
        wp::int32 var_555;
        wp::float32* var_556;
        wp::float32 var_557;
        wp::float32 var_558;
        wp::int32 var_559;
        wp::vec_t<3, wp::float32>* var_560;
        wp::float32 var_561;
        wp::vec_t<3, wp::float32> var_562;
        wp::float32 var_563;
        wp::float32 var_564;
        bool var_565;
        bool var_566;
        wp::float32* var_567;
        const wp::float32 var_568 = 0.0;
        bool var_569;
        wp::float32 var_570;
        wp::float32 var_571;
        wp::float32 var_572;
        wp::float32 var_573;
        wp::float32 var_574;
        const wp::int32 var_575 = 17;
        wp::int32 var_576;
        wp::int32 var_577;
        const wp::int32 var_578 = 16;
        wp::int32 var_579;
        wp::float32* var_580;
        wp::float32 var_581;
        wp::float32 var_582;
        wp::float32 var_583;
        wp::float32 var_584;
        wp::float32 var_585;
        bool var_586;
        bool var_587;
        wp::float32* var_588;
        const wp::float32 var_589 = 0.0;
        bool var_590;
        wp::float32 var_591;
        wp::float32 var_592;
        const wp::int32 var_593 = 1;
        const wp::int32 var_594 = 0;
        const wp::float32 var_595 = 0.0;
        wp::float32 var_596;
        const wp::float32 var_597 = 0.0;
        wp::float32 var_598;
        const wp::int32 var_599 = 0;
        const wp::int32 var_600 = 17;
        wp::int32 var_601;
        wp::int32 var_602;
        const wp::int32 var_603 = 4;
        wp::int32 var_604;
        wp::int32 var_605;
        wp::int32 var_606;
        wp::float32* var_607;
        wp::float32 var_608;
        wp::float32 var_609;
        wp::int32 var_610;
        wp::vec_t<3, wp::float32>* var_611;
        wp::float32 var_612;
        wp::vec_t<3, wp::float32> var_613;
        wp::float32 var_614;
        wp::float32 var_615;
        bool var_616;
        bool var_617;
        wp::float32* var_618;
        const wp::float32 var_619 = 0.0;
        bool var_620;
        wp::float32 var_621;
        wp::float32 var_622;
        wp::float32 var_623;
        wp::float32 var_624;
        wp::float32 var_625;
        const wp::int32 var_626 = 1;
        const wp::int32 var_627 = 17;
        wp::int32 var_628;
        wp::int32 var_629;
        const wp::int32 var_630 = 4;
        wp::int32 var_631;
        wp::int32 var_632;
        wp::int32 var_633;
        wp::float32* var_634;
        wp::float32 var_635;
        wp::float32 var_636;
        wp::int32 var_637;
        wp::vec_t<3, wp::float32>* var_638;
        wp::float32 var_639;
        wp::vec_t<3, wp::float32> var_640;
        wp::float32 var_641;
        wp::float32 var_642;
        bool var_643;
        bool var_644;
        wp::float32* var_645;
        const wp::float32 var_646 = 0.0;
        bool var_647;
        wp::float32 var_648;
        wp::float32 var_649;
        wp::float32 var_650;
        wp::float32 var_651;
        wp::float32 var_652;
        const wp::int32 var_653 = 2;
        const wp::int32 var_654 = 17;
        wp::int32 var_655;
        wp::int32 var_656;
        const wp::int32 var_657 = 4;
        wp::int32 var_658;
        wp::int32 var_659;
        wp::int32 var_660;
        wp::float32* var_661;
        wp::float32 var_662;
        wp::float32 var_663;
        wp::int32 var_664;
        wp::vec_t<3, wp::float32>* var_665;
        wp::float32 var_666;
        wp::vec_t<3, wp::float32> var_667;
        wp::float32 var_668;
        wp::float32 var_669;
        bool var_670;
        bool var_671;
        wp::float32* var_672;
        const wp::float32 var_673 = 0.0;
        bool var_674;
        wp::float32 var_675;
        wp::float32 var_676;
        wp::float32 var_677;
        wp::float32 var_678;
        wp::float32 var_679;
        const wp::int32 var_680 = 3;
        const wp::int32 var_681 = 17;
        wp::int32 var_682;
        wp::int32 var_683;
        const wp::int32 var_684 = 4;
        wp::int32 var_685;
        wp::int32 var_686;
        wp::int32 var_687;
        wp::float32* var_688;
        wp::float32 var_689;
        wp::float32 var_690;
        wp::int32 var_691;
        wp::vec_t<3, wp::float32>* var_692;
        wp::float32 var_693;
        wp::vec_t<3, wp::float32> var_694;
        wp::float32 var_695;
        wp::float32 var_696;
        bool var_697;
        bool var_698;
        wp::float32* var_699;
        const wp::float32 var_700 = 0.0;
        bool var_701;
        wp::float32 var_702;
        wp::float32 var_703;
        wp::float32 var_704;
        wp::float32 var_705;
        wp::float32 var_706;
        const wp::int32 var_707 = 17;
        wp::int32 var_708;
        wp::int32 var_709;
        const wp::int32 var_710 = 16;
        wp::int32 var_711;
        wp::float32* var_712;
        wp::float32 var_713;
        wp::float32 var_714;
        wp::float32 var_715;
        wp::float32 var_716;
        wp::float32 var_717;
        bool var_718;
        bool var_719;
        wp::float32* var_720;
        const wp::float32 var_721 = 0.0;
        bool var_722;
        wp::float32 var_723;
        wp::float32 var_724;
        const wp::int32 var_725 = 1;
        const wp::float32 var_726 = 0.0;
        wp::float32 var_727;
        const wp::float32 var_728 = 0.0;
        wp::float32 var_729;
        const wp::int32 var_730 = 0;
        const wp::int32 var_731 = 17;
        wp::int32 var_732;
        wp::int32 var_733;
        const wp::int32 var_734 = 4;
        wp::int32 var_735;
        wp::int32 var_736;
        wp::int32 var_737;
        wp::float32* var_738;
        wp::float32 var_739;
        wp::float32 var_740;
        wp::int32 var_741;
        wp::vec_t<3, wp::float32>* var_742;
        wp::float32 var_743;
        wp::vec_t<3, wp::float32> var_744;
        wp::float32 var_745;
        wp::float32 var_746;
        bool var_747;
        bool var_748;
        wp::float32* var_749;
        const wp::float32 var_750 = 0.0;
        bool var_751;
        wp::float32 var_752;
        wp::float32 var_753;
        wp::float32 var_754;
        wp::float32 var_755;
        wp::float32 var_756;
        const wp::int32 var_757 = 1;
        const wp::int32 var_758 = 17;
        wp::int32 var_759;
        wp::int32 var_760;
        const wp::int32 var_761 = 4;
        wp::int32 var_762;
        wp::int32 var_763;
        wp::int32 var_764;
        wp::float32* var_765;
        wp::float32 var_766;
        wp::float32 var_767;
        wp::int32 var_768;
        wp::vec_t<3, wp::float32>* var_769;
        wp::float32 var_770;
        wp::vec_t<3, wp::float32> var_771;
        wp::float32 var_772;
        wp::float32 var_773;
        bool var_774;
        bool var_775;
        wp::float32* var_776;
        const wp::float32 var_777 = 0.0;
        bool var_778;
        wp::float32 var_779;
        wp::float32 var_780;
        wp::float32 var_781;
        wp::float32 var_782;
        wp::float32 var_783;
        const wp::int32 var_784 = 2;
        const wp::int32 var_785 = 17;
        wp::int32 var_786;
        wp::int32 var_787;
        const wp::int32 var_788 = 4;
        wp::int32 var_789;
        wp::int32 var_790;
        wp::int32 var_791;
        wp::float32* var_792;
        wp::float32 var_793;
        wp::float32 var_794;
        wp::int32 var_795;
        wp::vec_t<3, wp::float32>* var_796;
        wp::float32 var_797;
        wp::vec_t<3, wp::float32> var_798;
        wp::float32 var_799;
        wp::float32 var_800;
        bool var_801;
        bool var_802;
        wp::float32* var_803;
        const wp::float32 var_804 = 0.0;
        bool var_805;
        wp::float32 var_806;
        wp::float32 var_807;
        wp::float32 var_808;
        wp::float32 var_809;
        wp::float32 var_810;
        const wp::int32 var_811 = 3;
        const wp::int32 var_812 = 17;
        wp::int32 var_813;
        wp::int32 var_814;
        const wp::int32 var_815 = 4;
        wp::int32 var_816;
        wp::int32 var_817;
        wp::int32 var_818;
        wp::float32* var_819;
        wp::float32 var_820;
        wp::float32 var_821;
        wp::int32 var_822;
        wp::vec_t<3, wp::float32>* var_823;
        wp::float32 var_824;
        wp::vec_t<3, wp::float32> var_825;
        wp::float32 var_826;
        wp::float32 var_827;
        bool var_828;
        bool var_829;
        wp::float32* var_830;
        const wp::float32 var_831 = 0.0;
        bool var_832;
        wp::float32 var_833;
        wp::float32 var_834;
        wp::float32 var_835;
        wp::float32 var_836;
        wp::float32 var_837;
        const wp::int32 var_838 = 17;
        wp::int32 var_839;
        wp::int32 var_840;
        const wp::int32 var_841 = 16;
        wp::int32 var_842;
        wp::float32* var_843;
        wp::float32 var_844;
        wp::float32 var_845;
        wp::float32 var_846;
        wp::float32 var_847;
        wp::float32 var_848;
        bool var_849;
        bool var_850;
        wp::float32* var_851;
        const wp::float32 var_852 = 0.0;
        bool var_853;
        wp::float32 var_854;
        wp::float32 var_855;
        const wp::int32 var_856 = 2;
        const wp::float32 var_857 = 0.0;
        wp::float32 var_858;
        const wp::float32 var_859 = 0.0;
        wp::float32 var_860;
        const wp::int32 var_861 = 0;
        const wp::int32 var_862 = 17;
        wp::int32 var_863;
        wp::int32 var_864;
        const wp::int32 var_865 = 4;
        wp::int32 var_866;
        wp::int32 var_867;
        wp::int32 var_868;
        wp::float32* var_869;
        wp::float32 var_870;
        wp::float32 var_871;
        wp::int32 var_872;
        wp::vec_t<3, wp::float32>* var_873;
        wp::float32 var_874;
        wp::vec_t<3, wp::float32> var_875;
        wp::float32 var_876;
        wp::float32 var_877;
        bool var_878;
        bool var_879;
        wp::float32* var_880;
        const wp::float32 var_881 = 0.0;
        bool var_882;
        wp::float32 var_883;
        wp::float32 var_884;
        wp::float32 var_885;
        wp::float32 var_886;
        wp::float32 var_887;
        const wp::int32 var_888 = 1;
        const wp::int32 var_889 = 17;
        wp::int32 var_890;
        wp::int32 var_891;
        const wp::int32 var_892 = 4;
        wp::int32 var_893;
        wp::int32 var_894;
        wp::int32 var_895;
        wp::float32* var_896;
        wp::float32 var_897;
        wp::float32 var_898;
        wp::int32 var_899;
        wp::vec_t<3, wp::float32>* var_900;
        wp::float32 var_901;
        wp::vec_t<3, wp::float32> var_902;
        wp::float32 var_903;
        wp::float32 var_904;
        bool var_905;
        bool var_906;
        wp::float32* var_907;
        const wp::float32 var_908 = 0.0;
        bool var_909;
        wp::float32 var_910;
        wp::float32 var_911;
        wp::float32 var_912;
        wp::float32 var_913;
        wp::float32 var_914;
        const wp::int32 var_915 = 2;
        const wp::int32 var_916 = 17;
        wp::int32 var_917;
        wp::int32 var_918;
        const wp::int32 var_919 = 4;
        wp::int32 var_920;
        wp::int32 var_921;
        wp::int32 var_922;
        wp::float32* var_923;
        wp::float32 var_924;
        wp::float32 var_925;
        wp::int32 var_926;
        wp::vec_t<3, wp::float32>* var_927;
        wp::float32 var_928;
        wp::vec_t<3, wp::float32> var_929;
        wp::float32 var_930;
        wp::float32 var_931;
        bool var_932;
        bool var_933;
        wp::float32* var_934;
        const wp::float32 var_935 = 0.0;
        bool var_936;
        wp::float32 var_937;
        wp::float32 var_938;
        wp::float32 var_939;
        wp::float32 var_940;
        wp::float32 var_941;
        const wp::int32 var_942 = 3;
        const wp::int32 var_943 = 17;
        wp::int32 var_944;
        wp::int32 var_945;
        const wp::int32 var_946 = 4;
        wp::int32 var_947;
        wp::int32 var_948;
        wp::int32 var_949;
        wp::float32* var_950;
        wp::float32 var_951;
        wp::float32 var_952;
        wp::int32 var_953;
        wp::vec_t<3, wp::float32>* var_954;
        wp::float32 var_955;
        wp::vec_t<3, wp::float32> var_956;
        wp::float32 var_957;
        wp::float32 var_958;
        bool var_959;
        bool var_960;
        wp::float32* var_961;
        const wp::float32 var_962 = 0.0;
        bool var_963;
        wp::float32 var_964;
        wp::float32 var_965;
        wp::float32 var_966;
        wp::float32 var_967;
        wp::float32 var_968;
        const wp::int32 var_969 = 17;
        wp::int32 var_970;
        wp::int32 var_971;
        const wp::int32 var_972 = 16;
        wp::int32 var_973;
        wp::float32* var_974;
        wp::float32 var_975;
        wp::float32 var_976;
        wp::float32 var_977;
        wp::float32 var_978;
        wp::float32 var_979;
        bool var_980;
        bool var_981;
        wp::float32* var_982;
        const wp::float32 var_983 = 0.0;
        bool var_984;
        wp::float32 var_985;
        wp::float32 var_986;
        const wp::int32 var_987 = 2;
        const wp::int32 var_988 = 0;
        const wp::float32 var_989 = 0.0;
        wp::float32 var_990;
        const wp::float32 var_991 = 0.0;
        wp::float32 var_992;
        const wp::int32 var_993 = 0;
        const wp::int32 var_994 = 17;
        wp::int32 var_995;
        wp::int32 var_996;
        const wp::int32 var_997 = 4;
        wp::int32 var_998;
        wp::int32 var_999;
        wp::int32 var_1000;
        wp::float32* var_1001;
        wp::float32 var_1002;
        wp::float32 var_1003;
        wp::int32 var_1004;
        wp::vec_t<3, wp::float32>* var_1005;
        wp::float32 var_1006;
        wp::vec_t<3, wp::float32> var_1007;
        wp::float32 var_1008;
        wp::float32 var_1009;
        bool var_1010;
        bool var_1011;
        wp::float32* var_1012;
        const wp::float32 var_1013 = 0.0;
        bool var_1014;
        wp::float32 var_1015;
        wp::float32 var_1016;
        wp::float32 var_1017;
        wp::float32 var_1018;
        wp::float32 var_1019;
        const wp::int32 var_1020 = 1;
        const wp::int32 var_1021 = 17;
        wp::int32 var_1022;
        wp::int32 var_1023;
        const wp::int32 var_1024 = 4;
        wp::int32 var_1025;
        wp::int32 var_1026;
        wp::int32 var_1027;
        wp::float32* var_1028;
        wp::float32 var_1029;
        wp::float32 var_1030;
        wp::int32 var_1031;
        wp::vec_t<3, wp::float32>* var_1032;
        wp::float32 var_1033;
        wp::vec_t<3, wp::float32> var_1034;
        wp::float32 var_1035;
        wp::float32 var_1036;
        bool var_1037;
        bool var_1038;
        wp::float32* var_1039;
        const wp::float32 var_1040 = 0.0;
        bool var_1041;
        wp::float32 var_1042;
        wp::float32 var_1043;
        wp::float32 var_1044;
        wp::float32 var_1045;
        wp::float32 var_1046;
        const wp::int32 var_1047 = 2;
        const wp::int32 var_1048 = 17;
        wp::int32 var_1049;
        wp::int32 var_1050;
        const wp::int32 var_1051 = 4;
        wp::int32 var_1052;
        wp::int32 var_1053;
        wp::int32 var_1054;
        wp::float32* var_1055;
        wp::float32 var_1056;
        wp::float32 var_1057;
        wp::int32 var_1058;
        wp::vec_t<3, wp::float32>* var_1059;
        wp::float32 var_1060;
        wp::vec_t<3, wp::float32> var_1061;
        wp::float32 var_1062;
        wp::float32 var_1063;
        bool var_1064;
        bool var_1065;
        wp::float32* var_1066;
        const wp::float32 var_1067 = 0.0;
        bool var_1068;
        wp::float32 var_1069;
        wp::float32 var_1070;
        wp::float32 var_1071;
        wp::float32 var_1072;
        wp::float32 var_1073;
        const wp::int32 var_1074 = 3;
        const wp::int32 var_1075 = 17;
        wp::int32 var_1076;
        wp::int32 var_1077;
        const wp::int32 var_1078 = 4;
        wp::int32 var_1079;
        wp::int32 var_1080;
        wp::int32 var_1081;
        wp::float32* var_1082;
        wp::float32 var_1083;
        wp::float32 var_1084;
        wp::int32 var_1085;
        wp::vec_t<3, wp::float32>* var_1086;
        wp::float32 var_1087;
        wp::vec_t<3, wp::float32> var_1088;
        wp::float32 var_1089;
        wp::float32 var_1090;
        bool var_1091;
        bool var_1092;
        wp::float32* var_1093;
        const wp::float32 var_1094 = 0.0;
        bool var_1095;
        wp::float32 var_1096;
        wp::float32 var_1097;
        wp::float32 var_1098;
        wp::float32 var_1099;
        wp::float32 var_1100;
        const wp::int32 var_1101 = 17;
        wp::int32 var_1102;
        wp::int32 var_1103;
        const wp::int32 var_1104 = 16;
        wp::int32 var_1105;
        wp::float32* var_1106;
        wp::float32 var_1107;
        wp::float32 var_1108;
        wp::float32 var_1109;
        wp::float32 var_1110;
        wp::float32 var_1111;
        bool var_1112;
        bool var_1113;
        wp::float32* var_1114;
        const wp::float32 var_1115 = 0.0;
        bool var_1116;
        wp::float32 var_1117;
        wp::float32 var_1118;
        const wp::int32 var_1119 = 1;
        const wp::float32 var_1120 = 0.0;
        wp::float32 var_1121;
        const wp::float32 var_1122 = 0.0;
        wp::float32 var_1123;
        const wp::int32 var_1124 = 0;
        const wp::int32 var_1125 = 17;
        wp::int32 var_1126;
        wp::int32 var_1127;
        const wp::int32 var_1128 = 4;
        wp::int32 var_1129;
        wp::int32 var_1130;
        wp::int32 var_1131;
        wp::float32* var_1132;
        wp::float32 var_1133;
        wp::float32 var_1134;
        wp::int32 var_1135;
        wp::vec_t<3, wp::float32>* var_1136;
        wp::float32 var_1137;
        wp::vec_t<3, wp::float32> var_1138;
        wp::float32 var_1139;
        wp::float32 var_1140;
        bool var_1141;
        bool var_1142;
        wp::float32* var_1143;
        const wp::float32 var_1144 = 0.0;
        bool var_1145;
        wp::float32 var_1146;
        wp::float32 var_1147;
        wp::float32 var_1148;
        wp::float32 var_1149;
        wp::float32 var_1150;
        const wp::int32 var_1151 = 1;
        const wp::int32 var_1152 = 17;
        wp::int32 var_1153;
        wp::int32 var_1154;
        const wp::int32 var_1155 = 4;
        wp::int32 var_1156;
        wp::int32 var_1157;
        wp::int32 var_1158;
        wp::float32* var_1159;
        wp::float32 var_1160;
        wp::float32 var_1161;
        wp::int32 var_1162;
        wp::vec_t<3, wp::float32>* var_1163;
        wp::float32 var_1164;
        wp::vec_t<3, wp::float32> var_1165;
        wp::float32 var_1166;
        wp::float32 var_1167;
        bool var_1168;
        bool var_1169;
        wp::float32* var_1170;
        const wp::float32 var_1171 = 0.0;
        bool var_1172;
        wp::float32 var_1173;
        wp::float32 var_1174;
        wp::float32 var_1175;
        wp::float32 var_1176;
        wp::float32 var_1177;
        const wp::int32 var_1178 = 2;
        const wp::int32 var_1179 = 17;
        wp::int32 var_1180;
        wp::int32 var_1181;
        const wp::int32 var_1182 = 4;
        wp::int32 var_1183;
        wp::int32 var_1184;
        wp::int32 var_1185;
        wp::float32* var_1186;
        wp::float32 var_1187;
        wp::float32 var_1188;
        wp::int32 var_1189;
        wp::vec_t<3, wp::float32>* var_1190;
        wp::float32 var_1191;
        wp::vec_t<3, wp::float32> var_1192;
        wp::float32 var_1193;
        wp::float32 var_1194;
        bool var_1195;
        bool var_1196;
        wp::float32* var_1197;
        const wp::float32 var_1198 = 0.0;
        bool var_1199;
        wp::float32 var_1200;
        wp::float32 var_1201;
        wp::float32 var_1202;
        wp::float32 var_1203;
        wp::float32 var_1204;
        const wp::int32 var_1205 = 3;
        const wp::int32 var_1206 = 17;
        wp::int32 var_1207;
        wp::int32 var_1208;
        const wp::int32 var_1209 = 4;
        wp::int32 var_1210;
        wp::int32 var_1211;
        wp::int32 var_1212;
        wp::float32* var_1213;
        wp::float32 var_1214;
        wp::float32 var_1215;
        wp::int32 var_1216;
        wp::vec_t<3, wp::float32>* var_1217;
        wp::float32 var_1218;
        wp::vec_t<3, wp::float32> var_1219;
        wp::float32 var_1220;
        wp::float32 var_1221;
        bool var_1222;
        bool var_1223;
        wp::float32* var_1224;
        const wp::float32 var_1225 = 0.0;
        bool var_1226;
        wp::float32 var_1227;
        wp::float32 var_1228;
        wp::float32 var_1229;
        wp::float32 var_1230;
        wp::float32 var_1231;
        const wp::int32 var_1232 = 17;
        wp::int32 var_1233;
        wp::int32 var_1234;
        const wp::int32 var_1235 = 16;
        wp::int32 var_1236;
        wp::float32* var_1237;
        wp::float32 var_1238;
        wp::float32 var_1239;
        wp::float32 var_1240;
        wp::float32 var_1241;
        wp::float32 var_1242;
        bool var_1243;
        bool var_1244;
        wp::float32* var_1245;
        const wp::float32 var_1246 = 0.0;
        bool var_1247;
        wp::float32 var_1248;
        wp::float32 var_1249;
        const wp::int32 var_1250 = 2;
        const wp::float32 var_1251 = 0.0;
        wp::float32 var_1252;
        const wp::float32 var_1253 = 0.0;
        wp::float32 var_1254;
        const wp::int32 var_1255 = 0;
        const wp::int32 var_1256 = 17;
        wp::int32 var_1257;
        wp::int32 var_1258;
        const wp::int32 var_1259 = 4;
        wp::int32 var_1260;
        wp::int32 var_1261;
        wp::int32 var_1262;
        wp::float32* var_1263;
        wp::float32 var_1264;
        wp::float32 var_1265;
        wp::int32 var_1266;
        wp::vec_t<3, wp::float32>* var_1267;
        wp::float32 var_1268;
        wp::vec_t<3, wp::float32> var_1269;
        wp::float32 var_1270;
        wp::float32 var_1271;
        bool var_1272;
        bool var_1273;
        wp::float32* var_1274;
        const wp::float32 var_1275 = 0.0;
        bool var_1276;
        wp::float32 var_1277;
        wp::float32 var_1278;
        wp::float32 var_1279;
        wp::float32 var_1280;
        wp::float32 var_1281;
        const wp::int32 var_1282 = 1;
        const wp::int32 var_1283 = 17;
        wp::int32 var_1284;
        wp::int32 var_1285;
        const wp::int32 var_1286 = 4;
        wp::int32 var_1287;
        wp::int32 var_1288;
        wp::int32 var_1289;
        wp::float32* var_1290;
        wp::float32 var_1291;
        wp::float32 var_1292;
        wp::int32 var_1293;
        wp::vec_t<3, wp::float32>* var_1294;
        wp::float32 var_1295;
        wp::vec_t<3, wp::float32> var_1296;
        wp::float32 var_1297;
        wp::float32 var_1298;
        bool var_1299;
        bool var_1300;
        wp::float32* var_1301;
        const wp::float32 var_1302 = 0.0;
        bool var_1303;
        wp::float32 var_1304;
        wp::float32 var_1305;
        wp::float32 var_1306;
        wp::float32 var_1307;
        wp::float32 var_1308;
        const wp::int32 var_1309 = 2;
        const wp::int32 var_1310 = 17;
        wp::int32 var_1311;
        wp::int32 var_1312;
        const wp::int32 var_1313 = 4;
        wp::int32 var_1314;
        wp::int32 var_1315;
        wp::int32 var_1316;
        wp::float32* var_1317;
        wp::float32 var_1318;
        wp::float32 var_1319;
        wp::int32 var_1320;
        wp::vec_t<3, wp::float32>* var_1321;
        wp::float32 var_1322;
        wp::vec_t<3, wp::float32> var_1323;
        wp::float32 var_1324;
        wp::float32 var_1325;
        bool var_1326;
        bool var_1327;
        wp::float32* var_1328;
        const wp::float32 var_1329 = 0.0;
        bool var_1330;
        wp::float32 var_1331;
        wp::float32 var_1332;
        wp::float32 var_1333;
        wp::float32 var_1334;
        wp::float32 var_1335;
        const wp::int32 var_1336 = 3;
        const wp::int32 var_1337 = 17;
        wp::int32 var_1338;
        wp::int32 var_1339;
        const wp::int32 var_1340 = 4;
        wp::int32 var_1341;
        wp::int32 var_1342;
        wp::int32 var_1343;
        wp::float32* var_1344;
        wp::float32 var_1345;
        wp::float32 var_1346;
        wp::int32 var_1347;
        wp::vec_t<3, wp::float32>* var_1348;
        wp::float32 var_1349;
        wp::vec_t<3, wp::float32> var_1350;
        wp::float32 var_1351;
        wp::float32 var_1352;
        bool var_1353;
        bool var_1354;
        wp::float32* var_1355;
        const wp::float32 var_1356 = 0.0;
        bool var_1357;
        wp::float32 var_1358;
        wp::float32 var_1359;
        wp::float32 var_1360;
        wp::float32 var_1361;
        wp::float32 var_1362;
        const wp::int32 var_1363 = 17;
        wp::int32 var_1364;
        wp::int32 var_1365;
        const wp::int32 var_1366 = 16;
        wp::int32 var_1367;
        wp::float32* var_1368;
        wp::float32 var_1369;
        wp::float32 var_1370;
        wp::float32 var_1371;
        wp::float32 var_1372;
        wp::float32 var_1373;
        bool var_1374;
        bool var_1375;
        wp::float32* var_1376;
        const wp::float32 var_1377 = 0.0;
        bool var_1378;
        wp::float32 var_1379;
        wp::float32 var_1380;
        const wp::int32 var_1381 = 3;
        const wp::int32 var_1382 = 0;
        const wp::float32 var_1383 = 0.0;
        wp::float32 var_1384;
        const wp::float32 var_1385 = 0.0;
        wp::float32 var_1386;
        const wp::int32 var_1387 = 0;
        const wp::int32 var_1388 = 17;
        wp::int32 var_1389;
        wp::int32 var_1390;
        const wp::int32 var_1391 = 4;
        wp::int32 var_1392;
        wp::int32 var_1393;
        wp::int32 var_1394;
        wp::float32* var_1395;
        wp::float32 var_1396;
        wp::float32 var_1397;
        wp::int32 var_1398;
        wp::vec_t<3, wp::float32>* var_1399;
        wp::float32 var_1400;
        wp::vec_t<3, wp::float32> var_1401;
        wp::float32 var_1402;
        wp::float32 var_1403;
        bool var_1404;
        bool var_1405;
        wp::float32* var_1406;
        const wp::float32 var_1407 = 0.0;
        bool var_1408;
        wp::float32 var_1409;
        wp::float32 var_1410;
        wp::float32 var_1411;
        wp::float32 var_1412;
        wp::float32 var_1413;
        const wp::int32 var_1414 = 1;
        const wp::int32 var_1415 = 17;
        wp::int32 var_1416;
        wp::int32 var_1417;
        const wp::int32 var_1418 = 4;
        wp::int32 var_1419;
        wp::int32 var_1420;
        wp::int32 var_1421;
        wp::float32* var_1422;
        wp::float32 var_1423;
        wp::float32 var_1424;
        wp::int32 var_1425;
        wp::vec_t<3, wp::float32>* var_1426;
        wp::float32 var_1427;
        wp::vec_t<3, wp::float32> var_1428;
        wp::float32 var_1429;
        wp::float32 var_1430;
        bool var_1431;
        bool var_1432;
        wp::float32* var_1433;
        const wp::float32 var_1434 = 0.0;
        bool var_1435;
        wp::float32 var_1436;
        wp::float32 var_1437;
        wp::float32 var_1438;
        wp::float32 var_1439;
        wp::float32 var_1440;
        const wp::int32 var_1441 = 2;
        const wp::int32 var_1442 = 17;
        wp::int32 var_1443;
        wp::int32 var_1444;
        const wp::int32 var_1445 = 4;
        wp::int32 var_1446;
        wp::int32 var_1447;
        wp::int32 var_1448;
        wp::float32* var_1449;
        wp::float32 var_1450;
        wp::float32 var_1451;
        wp::int32 var_1452;
        wp::vec_t<3, wp::float32>* var_1453;
        wp::float32 var_1454;
        wp::vec_t<3, wp::float32> var_1455;
        wp::float32 var_1456;
        wp::float32 var_1457;
        bool var_1458;
        bool var_1459;
        wp::float32* var_1460;
        const wp::float32 var_1461 = 0.0;
        bool var_1462;
        wp::float32 var_1463;
        wp::float32 var_1464;
        wp::float32 var_1465;
        wp::float32 var_1466;
        wp::float32 var_1467;
        const wp::int32 var_1468 = 3;
        const wp::int32 var_1469 = 17;
        wp::int32 var_1470;
        wp::int32 var_1471;
        const wp::int32 var_1472 = 4;
        wp::int32 var_1473;
        wp::int32 var_1474;
        wp::int32 var_1475;
        wp::float32* var_1476;
        wp::float32 var_1477;
        wp::float32 var_1478;
        wp::int32 var_1479;
        wp::vec_t<3, wp::float32>* var_1480;
        wp::float32 var_1481;
        wp::vec_t<3, wp::float32> var_1482;
        wp::float32 var_1483;
        wp::float32 var_1484;
        bool var_1485;
        bool var_1486;
        wp::float32* var_1487;
        const wp::float32 var_1488 = 0.0;
        bool var_1489;
        wp::float32 var_1490;
        wp::float32 var_1491;
        wp::float32 var_1492;
        wp::float32 var_1493;
        wp::float32 var_1494;
        const wp::int32 var_1495 = 17;
        wp::int32 var_1496;
        wp::int32 var_1497;
        const wp::int32 var_1498 = 16;
        wp::int32 var_1499;
        wp::float32* var_1500;
        wp::float32 var_1501;
        wp::float32 var_1502;
        wp::float32 var_1503;
        wp::float32 var_1504;
        wp::float32 var_1505;
        bool var_1506;
        bool var_1507;
        wp::float32* var_1508;
        const wp::float32 var_1509 = 0.0;
        bool var_1510;
        wp::float32 var_1511;
        wp::float32 var_1512;
        const wp::int32 var_1513 = 1;
        const wp::float32 var_1514 = 0.0;
        wp::float32 var_1515;
        const wp::float32 var_1516 = 0.0;
        wp::float32 var_1517;
        const wp::int32 var_1518 = 0;
        const wp::int32 var_1519 = 17;
        wp::int32 var_1520;
        wp::int32 var_1521;
        const wp::int32 var_1522 = 4;
        wp::int32 var_1523;
        wp::int32 var_1524;
        wp::int32 var_1525;
        wp::float32* var_1526;
        wp::float32 var_1527;
        wp::float32 var_1528;
        wp::int32 var_1529;
        wp::vec_t<3, wp::float32>* var_1530;
        wp::float32 var_1531;
        wp::vec_t<3, wp::float32> var_1532;
        wp::float32 var_1533;
        wp::float32 var_1534;
        bool var_1535;
        bool var_1536;
        wp::float32* var_1537;
        const wp::float32 var_1538 = 0.0;
        bool var_1539;
        wp::float32 var_1540;
        wp::float32 var_1541;
        wp::float32 var_1542;
        wp::float32 var_1543;
        wp::float32 var_1544;
        const wp::int32 var_1545 = 1;
        const wp::int32 var_1546 = 17;
        wp::int32 var_1547;
        wp::int32 var_1548;
        const wp::int32 var_1549 = 4;
        wp::int32 var_1550;
        wp::int32 var_1551;
        wp::int32 var_1552;
        wp::float32* var_1553;
        wp::float32 var_1554;
        wp::float32 var_1555;
        wp::int32 var_1556;
        wp::vec_t<3, wp::float32>* var_1557;
        wp::float32 var_1558;
        wp::vec_t<3, wp::float32> var_1559;
        wp::float32 var_1560;
        wp::float32 var_1561;
        bool var_1562;
        bool var_1563;
        wp::float32* var_1564;
        const wp::float32 var_1565 = 0.0;
        bool var_1566;
        wp::float32 var_1567;
        wp::float32 var_1568;
        wp::float32 var_1569;
        wp::float32 var_1570;
        wp::float32 var_1571;
        const wp::int32 var_1572 = 2;
        const wp::int32 var_1573 = 17;
        wp::int32 var_1574;
        wp::int32 var_1575;
        const wp::int32 var_1576 = 4;
        wp::int32 var_1577;
        wp::int32 var_1578;
        wp::int32 var_1579;
        wp::float32* var_1580;
        wp::float32 var_1581;
        wp::float32 var_1582;
        wp::int32 var_1583;
        wp::vec_t<3, wp::float32>* var_1584;
        wp::float32 var_1585;
        wp::vec_t<3, wp::float32> var_1586;
        wp::float32 var_1587;
        wp::float32 var_1588;
        bool var_1589;
        bool var_1590;
        wp::float32* var_1591;
        const wp::float32 var_1592 = 0.0;
        bool var_1593;
        wp::float32 var_1594;
        wp::float32 var_1595;
        wp::float32 var_1596;
        wp::float32 var_1597;
        wp::float32 var_1598;
        const wp::int32 var_1599 = 3;
        const wp::int32 var_1600 = 17;
        wp::int32 var_1601;
        wp::int32 var_1602;
        const wp::int32 var_1603 = 4;
        wp::int32 var_1604;
        wp::int32 var_1605;
        wp::int32 var_1606;
        wp::float32* var_1607;
        wp::float32 var_1608;
        wp::float32 var_1609;
        wp::int32 var_1610;
        wp::vec_t<3, wp::float32>* var_1611;
        wp::float32 var_1612;
        wp::vec_t<3, wp::float32> var_1613;
        wp::float32 var_1614;
        wp::float32 var_1615;
        bool var_1616;
        bool var_1617;
        wp::float32* var_1618;
        const wp::float32 var_1619 = 0.0;
        bool var_1620;
        wp::float32 var_1621;
        wp::float32 var_1622;
        wp::float32 var_1623;
        wp::float32 var_1624;
        wp::float32 var_1625;
        const wp::int32 var_1626 = 17;
        wp::int32 var_1627;
        wp::int32 var_1628;
        const wp::int32 var_1629 = 16;
        wp::int32 var_1630;
        wp::float32* var_1631;
        wp::float32 var_1632;
        wp::float32 var_1633;
        wp::float32 var_1634;
        wp::float32 var_1635;
        wp::float32 var_1636;
        bool var_1637;
        bool var_1638;
        wp::float32* var_1639;
        const wp::float32 var_1640 = 0.0;
        bool var_1641;
        wp::float32 var_1642;
        wp::float32 var_1643;
        const wp::int32 var_1644 = 2;
        const wp::float32 var_1645 = 0.0;
        wp::float32 var_1646;
        const wp::float32 var_1647 = 0.0;
        wp::float32 var_1648;
        const wp::int32 var_1649 = 0;
        const wp::int32 var_1650 = 17;
        wp::int32 var_1651;
        wp::int32 var_1652;
        const wp::int32 var_1653 = 4;
        wp::int32 var_1654;
        wp::int32 var_1655;
        wp::int32 var_1656;
        wp::float32* var_1657;
        wp::float32 var_1658;
        wp::float32 var_1659;
        wp::int32 var_1660;
        wp::vec_t<3, wp::float32>* var_1661;
        wp::float32 var_1662;
        wp::vec_t<3, wp::float32> var_1663;
        wp::float32 var_1664;
        wp::float32 var_1665;
        bool var_1666;
        bool var_1667;
        wp::float32* var_1668;
        const wp::float32 var_1669 = 0.0;
        bool var_1670;
        wp::float32 var_1671;
        wp::float32 var_1672;
        wp::float32 var_1673;
        wp::float32 var_1674;
        wp::float32 var_1675;
        const wp::int32 var_1676 = 1;
        const wp::int32 var_1677 = 17;
        wp::int32 var_1678;
        wp::int32 var_1679;
        const wp::int32 var_1680 = 4;
        wp::int32 var_1681;
        wp::int32 var_1682;
        wp::int32 var_1683;
        wp::float32* var_1684;
        wp::float32 var_1685;
        wp::float32 var_1686;
        wp::int32 var_1687;
        wp::vec_t<3, wp::float32>* var_1688;
        wp::float32 var_1689;
        wp::vec_t<3, wp::float32> var_1690;
        wp::float32 var_1691;
        wp::float32 var_1692;
        bool var_1693;
        bool var_1694;
        wp::float32* var_1695;
        const wp::float32 var_1696 = 0.0;
        bool var_1697;
        wp::float32 var_1698;
        wp::float32 var_1699;
        wp::float32 var_1700;
        wp::float32 var_1701;
        wp::float32 var_1702;
        const wp::int32 var_1703 = 2;
        const wp::int32 var_1704 = 17;
        wp::int32 var_1705;
        wp::int32 var_1706;
        const wp::int32 var_1707 = 4;
        wp::int32 var_1708;
        wp::int32 var_1709;
        wp::int32 var_1710;
        wp::float32* var_1711;
        wp::float32 var_1712;
        wp::float32 var_1713;
        wp::int32 var_1714;
        wp::vec_t<3, wp::float32>* var_1715;
        wp::float32 var_1716;
        wp::vec_t<3, wp::float32> var_1717;
        wp::float32 var_1718;
        wp::float32 var_1719;
        bool var_1720;
        bool var_1721;
        wp::float32* var_1722;
        const wp::float32 var_1723 = 0.0;
        bool var_1724;
        wp::float32 var_1725;
        wp::float32 var_1726;
        wp::float32 var_1727;
        wp::float32 var_1728;
        wp::float32 var_1729;
        const wp::int32 var_1730 = 3;
        const wp::int32 var_1731 = 17;
        wp::int32 var_1732;
        wp::int32 var_1733;
        const wp::int32 var_1734 = 4;
        wp::int32 var_1735;
        wp::int32 var_1736;
        wp::int32 var_1737;
        wp::float32* var_1738;
        wp::float32 var_1739;
        wp::float32 var_1740;
        wp::int32 var_1741;
        wp::vec_t<3, wp::float32>* var_1742;
        wp::float32 var_1743;
        wp::vec_t<3, wp::float32> var_1744;
        wp::float32 var_1745;
        wp::float32 var_1746;
        bool var_1747;
        bool var_1748;
        wp::float32* var_1749;
        const wp::float32 var_1750 = 0.0;
        bool var_1751;
        wp::float32 var_1752;
        wp::float32 var_1753;
        wp::float32 var_1754;
        wp::float32 var_1755;
        wp::float32 var_1756;
        const wp::int32 var_1757 = 17;
        wp::int32 var_1758;
        wp::int32 var_1759;
        const wp::int32 var_1760 = 16;
        wp::int32 var_1761;
        wp::float32* var_1762;
        wp::float32 var_1763;
        wp::float32 var_1764;
        wp::float32 var_1765;
        wp::float32 var_1766;
        wp::float32 var_1767;
        bool var_1768;
        bool var_1769;
        wp::float32* var_1770;
        const wp::float32 var_1771 = 0.0;
        bool var_1772;
        wp::float32 var_1773;
        wp::float32 var_1774;
        const wp::int32 var_1775 = 0;
        wp::int32 var_1776;
        wp::int32* var_1777;
        wp::int32 var_1778;
        wp::int32 var_1779;
        wp::vec_t<3, wp::float32> var_1780;
        wp::int32 var_1781;
        wp::vec_t<3, wp::float32>* var_1782;
        wp::vec_t<3, wp::float32> var_1783;
        wp::vec_t<3, wp::float32> var_1784;
        wp::vec_t<3, wp::float32>* var_1785;
        wp::vec_t<3, wp::float32> var_1786;
        wp::vec_t<3, wp::float32> var_1787;
        wp::vec_t<3, wp::float32> var_1788;
        wp::vec_t<3, wp::float32> var_1789;
        wp::vec_t<3, wp::float32> var_1790;
        wp::vec_t<6, wp::float32> var_1791;
        wp::vec_t<6, wp::float32> var_1792;
        bool var_1793;
        bool var_1794;
        wp::float32* var_1795;
        const wp::float32 var_1796 = 0.0;
        bool var_1797;
        wp::float32 var_1798;
        wp::vec_t<3, wp::float32> var_1799;
        wp::float32* var_1800;
        wp::vec_t<3, wp::float32> var_1801;
        wp::float32 var_1802;
        wp::vec_t<3, wp::float32> var_1803;
        wp::vec_t<3, wp::float32> var_1804;
        wp::vec_t<6, wp::float32> var_1805;
        wp::vec_t<6, wp::float32> var_1806;
        const wp::int32 var_1807 = 1;
        wp::int32 var_1808;
        wp::int32* var_1809;
        wp::int32 var_1810;
        wp::int32 var_1811;
        wp::vec_t<3, wp::float32> var_1812;
        wp::int32 var_1813;
        wp::vec_t<3, wp::float32>* var_1814;
        wp::vec_t<3, wp::float32> var_1815;
        wp::vec_t<3, wp::float32> var_1816;
        wp::vec_t<3, wp::float32>* var_1817;
        wp::vec_t<3, wp::float32> var_1818;
        wp::vec_t<3, wp::float32> var_1819;
        wp::vec_t<3, wp::float32> var_1820;
        wp::vec_t<3, wp::float32> var_1821;
        wp::vec_t<3, wp::float32> var_1822;
        wp::vec_t<6, wp::float32> var_1823;
        wp::vec_t<6, wp::float32> var_1824;
        bool var_1825;
        bool var_1826;
        wp::float32* var_1827;
        const wp::float32 var_1828 = 0.0;
        bool var_1829;
        wp::float32 var_1830;
        wp::vec_t<3, wp::float32> var_1831;
        wp::float32* var_1832;
        wp::vec_t<3, wp::float32> var_1833;
        wp::float32 var_1834;
        wp::vec_t<3, wp::float32> var_1835;
        wp::vec_t<3, wp::float32> var_1836;
        wp::vec_t<6, wp::float32> var_1837;
        wp::vec_t<6, wp::float32> var_1838;
        wp::vec_t<3, wp::float32> var_1839;
        wp::vec_t<6, wp::float32> var_1840;
        const wp::int32 var_1841 = 2;
        wp::int32 var_1842;
        wp::int32* var_1843;
        wp::int32 var_1844;
        wp::int32 var_1845;
        wp::vec_t<3, wp::float32> var_1846;
        wp::int32 var_1847;
        wp::vec_t<3, wp::float32>* var_1848;
        wp::vec_t<3, wp::float32> var_1849;
        wp::vec_t<3, wp::float32> var_1850;
        wp::vec_t<3, wp::float32>* var_1851;
        wp::vec_t<3, wp::float32> var_1852;
        wp::vec_t<3, wp::float32> var_1853;
        wp::vec_t<3, wp::float32> var_1854;
        wp::vec_t<3, wp::float32> var_1855;
        wp::vec_t<3, wp::float32> var_1856;
        wp::vec_t<6, wp::float32> var_1857;
        wp::vec_t<6, wp::float32> var_1858;
        bool var_1859;
        bool var_1860;
        wp::float32* var_1861;
        const wp::float32 var_1862 = 0.0;
        bool var_1863;
        wp::float32 var_1864;
        wp::vec_t<3, wp::float32> var_1865;
        wp::float32* var_1866;
        wp::vec_t<3, wp::float32> var_1867;
        wp::float32 var_1868;
        wp::vec_t<3, wp::float32> var_1869;
        wp::vec_t<3, wp::float32> var_1870;
        wp::vec_t<6, wp::float32> var_1871;
        wp::vec_t<6, wp::float32> var_1872;
        wp::vec_t<3, wp::float32> var_1873;
        wp::vec_t<6, wp::float32> var_1874;
        const wp::int32 var_1875 = 3;
        wp::int32 var_1876;
        wp::int32* var_1877;
        wp::int32 var_1878;
        wp::int32 var_1879;
        wp::vec_t<3, wp::float32> var_1880;
        wp::int32 var_1881;
        wp::vec_t<3, wp::float32>* var_1882;
        wp::vec_t<3, wp::float32> var_1883;
        wp::vec_t<3, wp::float32> var_1884;
        wp::vec_t<3, wp::float32>* var_1885;
        wp::vec_t<3, wp::float32> var_1886;
        wp::vec_t<3, wp::float32> var_1887;
        wp::vec_t<3, wp::float32> var_1888;
        wp::vec_t<3, wp::float32> var_1889;
        wp::vec_t<3, wp::float32> var_1890;
        wp::vec_t<6, wp::float32> var_1891;
        wp::vec_t<6, wp::float32> var_1892;
        bool var_1893;
        bool var_1894;
        wp::float32* var_1895;
        const wp::float32 var_1896 = 0.0;
        bool var_1897;
        wp::float32 var_1898;
        wp::vec_t<3, wp::float32> var_1899;
        wp::float32* var_1900;
        wp::vec_t<3, wp::float32> var_1901;
        wp::float32 var_1902;
        wp::vec_t<3, wp::float32> var_1903;
        wp::vec_t<3, wp::float32> var_1904;
        wp::vec_t<6, wp::float32> var_1905;
        wp::vec_t<6, wp::float32> var_1906;
        wp::vec_t<3, wp::float32> var_1907;
        wp::vec_t<6, wp::float32> var_1908;
        //---------
        // forward
        // def _flex_bending(                                                                     <L 711>
        // worldid, edgeid = wp.tid()                                                             <L 736>
        builtin_tid2d(var_0, var_1);
        // for i in range(nflex):                                                                 <L 738>
        var_2 = wp::range(var_nflex);
        start_for_0:;
            if (iter_cmp(var_2) == 0) goto end_for_0;
            var_3 = wp::iter_next(var_2);
            // eid = edgeid - flex_edgeadr[i]                                                     <L 739>
            var_4 = wp::address(var_flex_edgeadr, var_3);
            var_6 = wp::load(var_4);
            var_5 = wp::sub(var_1, var_6);
            // if eid >= 0 and eid < flex_edgenum[i]:                                             <L 740>
            var_9 = (var_5 >= var_8);
            var_7 = var_9;
            if (var_7) {
                var_10 = wp::address(var_flex_edgenum, var_3);
                var_12 = wp::load(var_10);
                var_11 = (var_5 < var_12);
                var_7 = var_7 && var_11;
            }
            if (var_7) {
                // f = i                                                                          <L 741>
                var_13 = wp::copy(var_3);
                // break                                                                          <L 742>
                goto end_for_0;
            }
            goto start_for_0;
        end_for_0:;
        // bendingadr = flex_bendingadr[f]                                                        <L 744>
        var_14 = wp::address(var_flex_bendingadr, var_13);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // if bendingadr < 0:                                                                     <L 745>
        var_18 = (var_15 < var_17);
        if (var_18) {
            // return                                                                             <L 746>
            continue;
        }
        // if flex_dim[f] != 2:                                                                   <L 748>
        var_19 = wp::address(var_flex_dim, var_13);
        var_22 = wp::load(var_19);
        var_21 = (var_22 != var_20);
        if (var_21) {
            // return                                                                             <L 749>
            continue;
        }
        // if flex_edgeflap[edgeid][1] == -1:                                                     <L 751>
        var_23 = wp::address(var_flex_edgeflap, var_1);
        var_26 = wp::load(var_23);
        var_25 = wp::extract(var_26, var_24);
        var_28 = (var_25 == var_27);
        if (var_28) {
            // return                                                                             <L 752>
            continue;
        }
        // v = wp.vec4i(                                                                          <L 754>
        // flex_vertadr[f] + flex_edge[edgeid][0],                                                <L 755>
        var_29 = wp::address(var_flex_vertadr, var_13);
        var_30 = wp::address(var_flex_edge, var_1);
        var_33 = wp::load(var_30);
        var_32 = wp::extract(var_33, var_31);
        var_35 = wp::load(var_29);
        var_34 = wp::add(var_35, var_32);
        // flex_vertadr[f] + flex_edge[edgeid][1],                                                <L 756>
        var_36 = wp::address(var_flex_vertadr, var_13);
        var_37 = wp::address(var_flex_edge, var_1);
        var_40 = wp::load(var_37);
        var_39 = wp::extract(var_40, var_38);
        var_42 = wp::load(var_36);
        var_41 = wp::add(var_42, var_39);
        // flex_vertadr[f] + flex_edgeflap[edgeid][0],                                            <L 757>
        var_43 = wp::address(var_flex_vertadr, var_13);
        var_44 = wp::address(var_flex_edgeflap, var_1);
        var_47 = wp::load(var_44);
        var_46 = wp::extract(var_47, var_45);
        var_49 = wp::load(var_43);
        var_48 = wp::add(var_49, var_46);
        // flex_vertadr[f] + flex_edgeflap[edgeid][1],                                            <L 758>
        var_50 = wp::address(var_flex_vertadr, var_13);
        var_51 = wp::address(var_flex_edgeflap, var_1);
        var_54 = wp::load(var_51);
        var_53 = wp::extract(var_54, var_52);
        var_56 = wp::load(var_50);
        var_55 = wp::add(var_56, var_53);
        var_57 = wp::vec_t<4, wp::int32>(var_34, var_41, var_48, var_55);
        // frc = mat43()                                                                          <L 761>
        var_58 = wp::mat_t<4, 3, wp::float32>();
        // if flex_bending[bendingadr + 17 * eid + 16]:                                           <L 762>
        var_60 = wp::mul(var_59, var_5);
        var_61 = wp::add(var_15, var_60);
        var_63 = wp::add(var_61, var_62);
        var_64 = wp::address(var_flex_bending, var_63);
        var_65 = wp::load(var_64);
        if (var_65) {
            // v0 = flexvert_xpos_in[worldid, v[0]]                                               <L 763>
            var_67 = wp::extract(var_57, var_66);
            var_68 = wp::address(var_flexvert_xpos_in, var_0, var_67);
            var_70 = wp::load(var_68);
            var_69 = wp::copy(var_70);
            // v1 = flexvert_xpos_in[worldid, v[1]]                                               <L 764>
            var_72 = wp::extract(var_57, var_71);
            var_73 = wp::address(var_flexvert_xpos_in, var_0, var_72);
            var_75 = wp::load(var_73);
            var_74 = wp::copy(var_75);
            // v2 = flexvert_xpos_in[worldid, v[2]]                                               <L 765>
            var_77 = wp::extract(var_57, var_76);
            var_78 = wp::address(var_flexvert_xpos_in, var_0, var_77);
            var_80 = wp::load(var_78);
            var_79 = wp::copy(var_80);
            // v3 = flexvert_xpos_in[worldid, v[3]]                                               <L 766>
            var_82 = wp::extract(var_57, var_81);
            var_83 = wp::address(var_flexvert_xpos_in, var_0, var_82);
            var_85 = wp::load(var_83);
            var_84 = wp::copy(var_85);
            // ed0 = v1 - v0                                                                      <L 768>
            var_86 = wp::sub(var_74, var_69);
            // ed1 = v2 - v0                                                                      <L 769>
            var_87 = wp::sub(var_79, var_69);
            // ed2 = v3 - v0                                                                      <L 770>
            var_88 = wp::sub(var_84, var_69);
            // frc[1] = wp.cross(ed1, ed2)                                                        <L 772>
            var_89 = wp::cross(var_87, var_88);
            wp::assign_inplace(var_58, var_90, var_89);
            // frc[2] = wp.cross(ed2, ed0)                                                        <L 773>
            var_91 = wp::cross(var_88, var_86);
            wp::assign_inplace(var_58, var_92, var_91);
            // frc[3] = wp.cross(ed0, ed1)                                                        <L 774>
            var_93 = wp::cross(var_86, var_87);
            wp::assign_inplace(var_58, var_94, var_93);
            // frc[0] = -(frc[1] + frc[2] + frc[3])                                               <L 775>
            var_96 = wp::extract(var_58, var_95);
            var_98 = wp::extract(var_58, var_97);
            var_99 = wp::add(var_96, var_98);
            var_101 = wp::extract(var_58, var_100);
            var_102 = wp::add(var_99, var_101);
            var_103 = wp::neg(var_102);
            wp::assign_inplace(var_58, var_104, var_103);
        }
        var_105 = wp::load(var_64);
        // vel = mat43()                                                                          <L 778>
        var_106 = wp::mat_t<4, 3, wp::float32>();
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 779>
        var_108 = wp::unot(var_dsbl_damper);
        var_107 = var_108;
        if (var_107) {
            var_109 = wp::address(var_flex_damping, var_13);
            var_112 = wp::load(var_109);
            var_111 = (var_112 > var_110);
            var_107 = var_107 && var_111;
        }
        if (var_107) {
            // for j in range(4):                                                                 <L 780>
            // bodyid_j = flex_vertbodyid[v[j]]                                                   <L 781>
            var_114 = wp::extract(var_57, var_113);
            var_115 = wp::address(var_flex_vertbodyid, var_114);
            var_117 = wp::load(var_115);
            var_116 = wp::copy(var_117);
            // cvel_j = cvel_in[worldid, bodyid_j]                                                <L 782>
            var_118 = wp::address(var_cvel_in, var_0, var_116);
            var_120 = wp::load(var_118);
            var_119 = wp::copy(var_120);
            // omega_j = wp.spatial_top(cvel_j)                                                   <L 783>
            var_121 = wp::spatial_top(var_119);
            // vcom_j = wp.spatial_bottom(cvel_j)                                                 <L 784>
            var_122 = wp::spatial_bottom(var_119);
            // com_j = subtree_com_in[worldid, body_rootid[bodyid_j]]                             <L 785>
            var_123 = wp::address(var_body_rootid, var_116);
            var_125 = wp::load(var_123);
            var_124 = wp::address(var_subtree_com_in, var_0, var_125);
            var_127 = wp::load(var_124);
            var_126 = wp::copy(var_127);
            // r_j = flexvert_xpos_in[worldid, v[j]] - com_j                                      <L 786>
            var_128 = wp::extract(var_57, var_113);
            var_129 = wp::address(var_flexvert_xpos_in, var_0, var_128);
            var_131 = wp::load(var_129);
            var_130 = wp::sub(var_131, var_126);
            // vel[j] = vcom_j + wp.cross(omega_j, r_j)                                           <L 787>
            var_132 = wp::cross(var_121, var_130);
            var_133 = wp::add(var_122, var_132);
            wp::assign_inplace(var_106, var_113, var_133);
            // bodyid_j = flex_vertbodyid[v[j]]                                                   <L 781>
            var_135 = wp::extract(var_57, var_134);
            var_136 = wp::address(var_flex_vertbodyid, var_135);
            var_138 = wp::load(var_136);
            var_137 = wp::copy(var_138);
            // cvel_j = cvel_in[worldid, bodyid_j]                                                <L 782>
            var_139 = wp::address(var_cvel_in, var_0, var_137);
            var_141 = wp::load(var_139);
            var_140 = wp::copy(var_141);
            // omega_j = wp.spatial_top(cvel_j)                                                   <L 783>
            var_142 = wp::spatial_top(var_140);
            // vcom_j = wp.spatial_bottom(cvel_j)                                                 <L 784>
            var_143 = wp::spatial_bottom(var_140);
            // com_j = subtree_com_in[worldid, body_rootid[bodyid_j]]                             <L 785>
            var_144 = wp::address(var_body_rootid, var_137);
            var_146 = wp::load(var_144);
            var_145 = wp::address(var_subtree_com_in, var_0, var_146);
            var_148 = wp::load(var_145);
            var_147 = wp::copy(var_148);
            // r_j = flexvert_xpos_in[worldid, v[j]] - com_j                                      <L 786>
            var_149 = wp::extract(var_57, var_134);
            var_150 = wp::address(var_flexvert_xpos_in, var_0, var_149);
            var_152 = wp::load(var_150);
            var_151 = wp::sub(var_152, var_147);
            // vel[j] = vcom_j + wp.cross(omega_j, r_j)                                           <L 787>
            var_153 = wp::cross(var_142, var_151);
            var_154 = wp::add(var_143, var_153);
            wp::assign_inplace(var_106, var_134, var_154);
            // bodyid_j = flex_vertbodyid[v[j]]                                                   <L 781>
            var_156 = wp::extract(var_57, var_155);
            var_157 = wp::address(var_flex_vertbodyid, var_156);
            var_159 = wp::load(var_157);
            var_158 = wp::copy(var_159);
            // cvel_j = cvel_in[worldid, bodyid_j]                                                <L 782>
            var_160 = wp::address(var_cvel_in, var_0, var_158);
            var_162 = wp::load(var_160);
            var_161 = wp::copy(var_162);
            // omega_j = wp.spatial_top(cvel_j)                                                   <L 783>
            var_163 = wp::spatial_top(var_161);
            // vcom_j = wp.spatial_bottom(cvel_j)                                                 <L 784>
            var_164 = wp::spatial_bottom(var_161);
            // com_j = subtree_com_in[worldid, body_rootid[bodyid_j]]                             <L 785>
            var_165 = wp::address(var_body_rootid, var_158);
            var_167 = wp::load(var_165);
            var_166 = wp::address(var_subtree_com_in, var_0, var_167);
            var_169 = wp::load(var_166);
            var_168 = wp::copy(var_169);
            // r_j = flexvert_xpos_in[worldid, v[j]] - com_j                                      <L 786>
            var_170 = wp::extract(var_57, var_155);
            var_171 = wp::address(var_flexvert_xpos_in, var_0, var_170);
            var_173 = wp::load(var_171);
            var_172 = wp::sub(var_173, var_168);
            // vel[j] = vcom_j + wp.cross(omega_j, r_j)                                           <L 787>
            var_174 = wp::cross(var_163, var_172);
            var_175 = wp::add(var_164, var_174);
            wp::assign_inplace(var_106, var_155, var_175);
            // bodyid_j = flex_vertbodyid[v[j]]                                                   <L 781>
            var_177 = wp::extract(var_57, var_176);
            var_178 = wp::address(var_flex_vertbodyid, var_177);
            var_180 = wp::load(var_178);
            var_179 = wp::copy(var_180);
            // cvel_j = cvel_in[worldid, bodyid_j]                                                <L 782>
            var_181 = wp::address(var_cvel_in, var_0, var_179);
            var_183 = wp::load(var_181);
            var_182 = wp::copy(var_183);
            // omega_j = wp.spatial_top(cvel_j)                                                   <L 783>
            var_184 = wp::spatial_top(var_182);
            // vcom_j = wp.spatial_bottom(cvel_j)                                                 <L 784>
            var_185 = wp::spatial_bottom(var_182);
            // com_j = subtree_com_in[worldid, body_rootid[bodyid_j]]                             <L 785>
            var_186 = wp::address(var_body_rootid, var_179);
            var_188 = wp::load(var_186);
            var_187 = wp::address(var_subtree_com_in, var_0, var_188);
            var_190 = wp::load(var_187);
            var_189 = wp::copy(var_190);
            // r_j = flexvert_xpos_in[worldid, v[j]] - com_j                                      <L 786>
            var_191 = wp::extract(var_57, var_176);
            var_192 = wp::address(var_flexvert_xpos_in, var_0, var_191);
            var_194 = wp::load(var_192);
            var_193 = wp::sub(var_194, var_189);
            // vel[j] = vcom_j + wp.cross(omega_j, r_j)                                           <L 787>
            var_195 = wp::cross(var_184, var_193);
            var_196 = wp::add(var_185, var_195);
            wp::assign_inplace(var_106, var_176, var_196);
        }
        // force_spring = mat43()                                                                 <L 789>
        var_197 = wp::mat_t<4, 3, wp::float32>();
        // force_damper = mat43()                                                                 <L 790>
        var_198 = wp::mat_t<4, 3, wp::float32>();
        // for i in range(4):                                                                     <L 791>
        // for x in range(3):                                                                     <L 792>
        // acc_spring = float(0.0)                                                                <L 793>
        var_202 = wp::float(var_201);
        // acc_damper = float(0.0)                                                                <L 794>
        var_204 = wp::float(var_203);
        // for j in range(4):                                                                     <L 795>
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_207 = wp::mul(var_206, var_5);
        var_208 = wp::add(var_15, var_207);
        var_210 = wp::mul(var_209, var_199);
        var_211 = wp::add(var_208, var_210);
        var_212 = wp::add(var_211, var_205);
        var_213 = wp::address(var_flex_bending, var_212);
        var_215 = wp::load(var_213);
        var_214 = wp::copy(var_215);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_216 = wp::extract(var_57, var_205);
        var_217 = wp::address(var_flexvert_xpos_in, var_0, var_216);
        var_219 = wp::load(var_217);
        var_218 = wp::extract(var_219, var_200);
        var_220 = wp::mul(var_214, var_218);
        var_221 = wp::add(var_202, var_220);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_223 = wp::unot(var_dsbl_damper);
        var_222 = var_223;
        if (var_222) {
            var_224 = wp::address(var_flex_damping, var_13);
            var_227 = wp::load(var_224);
            var_226 = (var_227 > var_225);
            var_222 = var_222 && var_226;
        }
        if (var_222) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_228 = wp::extract(var_106, var_205, var_200);
            var_229 = wp::mul(var_214, var_228);
            var_230 = wp::add(var_204, var_229);
        }
        var_231 = wp::where(var_222, var_230, var_204);
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_234 = wp::mul(var_233, var_5);
        var_235 = wp::add(var_15, var_234);
        var_237 = wp::mul(var_236, var_199);
        var_238 = wp::add(var_235, var_237);
        var_239 = wp::add(var_238, var_232);
        var_240 = wp::address(var_flex_bending, var_239);
        var_242 = wp::load(var_240);
        var_241 = wp::copy(var_242);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_243 = wp::extract(var_57, var_232);
        var_244 = wp::address(var_flexvert_xpos_in, var_0, var_243);
        var_246 = wp::load(var_244);
        var_245 = wp::extract(var_246, var_200);
        var_247 = wp::mul(var_241, var_245);
        var_248 = wp::add(var_221, var_247);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_250 = wp::unot(var_dsbl_damper);
        var_249 = var_250;
        if (var_249) {
            var_251 = wp::address(var_flex_damping, var_13);
            var_254 = wp::load(var_251);
            var_253 = (var_254 > var_252);
            var_249 = var_249 && var_253;
        }
        if (var_249) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_255 = wp::extract(var_106, var_232, var_200);
            var_256 = wp::mul(var_241, var_255);
            var_257 = wp::add(var_231, var_256);
        }
        var_258 = wp::where(var_249, var_257, var_231);
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_261 = wp::mul(var_260, var_5);
        var_262 = wp::add(var_15, var_261);
        var_264 = wp::mul(var_263, var_199);
        var_265 = wp::add(var_262, var_264);
        var_266 = wp::add(var_265, var_259);
        var_267 = wp::address(var_flex_bending, var_266);
        var_269 = wp::load(var_267);
        var_268 = wp::copy(var_269);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_270 = wp::extract(var_57, var_259);
        var_271 = wp::address(var_flexvert_xpos_in, var_0, var_270);
        var_273 = wp::load(var_271);
        var_272 = wp::extract(var_273, var_200);
        var_274 = wp::mul(var_268, var_272);
        var_275 = wp::add(var_248, var_274);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_277 = wp::unot(var_dsbl_damper);
        var_276 = var_277;
        if (var_276) {
            var_278 = wp::address(var_flex_damping, var_13);
            var_281 = wp::load(var_278);
            var_280 = (var_281 > var_279);
            var_276 = var_276 && var_280;
        }
        if (var_276) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_282 = wp::extract(var_106, var_259, var_200);
            var_283 = wp::mul(var_268, var_282);
            var_284 = wp::add(var_258, var_283);
        }
        var_285 = wp::where(var_276, var_284, var_258);
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_288 = wp::mul(var_287, var_5);
        var_289 = wp::add(var_15, var_288);
        var_291 = wp::mul(var_290, var_199);
        var_292 = wp::add(var_289, var_291);
        var_293 = wp::add(var_292, var_286);
        var_294 = wp::address(var_flex_bending, var_293);
        var_296 = wp::load(var_294);
        var_295 = wp::copy(var_296);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_297 = wp::extract(var_57, var_286);
        var_298 = wp::address(var_flexvert_xpos_in, var_0, var_297);
        var_300 = wp::load(var_298);
        var_299 = wp::extract(var_300, var_200);
        var_301 = wp::mul(var_295, var_299);
        var_302 = wp::add(var_275, var_301);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_304 = wp::unot(var_dsbl_damper);
        var_303 = var_304;
        if (var_303) {
            var_305 = wp::address(var_flex_damping, var_13);
            var_308 = wp::load(var_305);
            var_307 = (var_308 > var_306);
            var_303 = var_303 && var_307;
        }
        if (var_303) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_309 = wp::extract(var_106, var_286, var_200);
            var_310 = wp::mul(var_295, var_309);
            var_311 = wp::add(var_285, var_310);
        }
        var_312 = wp::where(var_303, var_311, var_285);
        // force_spring[i, x] = -(acc_spring + flex_bending[bendingadr + 17 * eid + 16] * frc[i, x])       <L 801>
        var_314 = wp::mul(var_313, var_5);
        var_315 = wp::add(var_15, var_314);
        var_317 = wp::add(var_315, var_316);
        var_318 = wp::address(var_flex_bending, var_317);
        var_319 = wp::extract(var_58, var_199, var_200);
        var_321 = wp::load(var_318);
        var_320 = wp::mul(var_321, var_319);
        var_322 = wp::add(var_302, var_320);
        var_323 = wp::neg(var_322);
        wp::assign_inplace(var_197, var_199, var_200, var_323);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 802>
        var_325 = wp::unot(var_dsbl_damper);
        var_324 = var_325;
        if (var_324) {
            var_326 = wp::address(var_flex_damping, var_13);
            var_329 = wp::load(var_326);
            var_328 = (var_329 > var_327);
            var_324 = var_324 && var_328;
        }
        if (var_324) {
            // force_damper[i, x] = -acc_damper                                                   <L 803>
            var_330 = wp::neg(var_312);
            wp::assign_inplace(var_198, var_199, var_200, var_330);
        }
        // acc_spring = float(0.0)                                                                <L 793>
        var_333 = wp::float(var_332);
        // acc_damper = float(0.0)                                                                <L 794>
        var_335 = wp::float(var_334);
        // for j in range(4):                                                                     <L 795>
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_338 = wp::mul(var_337, var_5);
        var_339 = wp::add(var_15, var_338);
        var_341 = wp::mul(var_340, var_199);
        var_342 = wp::add(var_339, var_341);
        var_343 = wp::add(var_342, var_336);
        var_344 = wp::address(var_flex_bending, var_343);
        var_346 = wp::load(var_344);
        var_345 = wp::copy(var_346);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_347 = wp::extract(var_57, var_336);
        var_348 = wp::address(var_flexvert_xpos_in, var_0, var_347);
        var_350 = wp::load(var_348);
        var_349 = wp::extract(var_350, var_331);
        var_351 = wp::mul(var_345, var_349);
        var_352 = wp::add(var_333, var_351);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_354 = wp::unot(var_dsbl_damper);
        var_353 = var_354;
        if (var_353) {
            var_355 = wp::address(var_flex_damping, var_13);
            var_358 = wp::load(var_355);
            var_357 = (var_358 > var_356);
            var_353 = var_353 && var_357;
        }
        if (var_353) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_359 = wp::extract(var_106, var_336, var_331);
            var_360 = wp::mul(var_345, var_359);
            var_361 = wp::add(var_335, var_360);
        }
        var_362 = wp::where(var_353, var_361, var_335);
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_365 = wp::mul(var_364, var_5);
        var_366 = wp::add(var_15, var_365);
        var_368 = wp::mul(var_367, var_199);
        var_369 = wp::add(var_366, var_368);
        var_370 = wp::add(var_369, var_363);
        var_371 = wp::address(var_flex_bending, var_370);
        var_373 = wp::load(var_371);
        var_372 = wp::copy(var_373);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_374 = wp::extract(var_57, var_363);
        var_375 = wp::address(var_flexvert_xpos_in, var_0, var_374);
        var_377 = wp::load(var_375);
        var_376 = wp::extract(var_377, var_331);
        var_378 = wp::mul(var_372, var_376);
        var_379 = wp::add(var_352, var_378);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_381 = wp::unot(var_dsbl_damper);
        var_380 = var_381;
        if (var_380) {
            var_382 = wp::address(var_flex_damping, var_13);
            var_385 = wp::load(var_382);
            var_384 = (var_385 > var_383);
            var_380 = var_380 && var_384;
        }
        if (var_380) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_386 = wp::extract(var_106, var_363, var_331);
            var_387 = wp::mul(var_372, var_386);
            var_388 = wp::add(var_362, var_387);
        }
        var_389 = wp::where(var_380, var_388, var_362);
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_392 = wp::mul(var_391, var_5);
        var_393 = wp::add(var_15, var_392);
        var_395 = wp::mul(var_394, var_199);
        var_396 = wp::add(var_393, var_395);
        var_397 = wp::add(var_396, var_390);
        var_398 = wp::address(var_flex_bending, var_397);
        var_400 = wp::load(var_398);
        var_399 = wp::copy(var_400);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_401 = wp::extract(var_57, var_390);
        var_402 = wp::address(var_flexvert_xpos_in, var_0, var_401);
        var_404 = wp::load(var_402);
        var_403 = wp::extract(var_404, var_331);
        var_405 = wp::mul(var_399, var_403);
        var_406 = wp::add(var_379, var_405);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_408 = wp::unot(var_dsbl_damper);
        var_407 = var_408;
        if (var_407) {
            var_409 = wp::address(var_flex_damping, var_13);
            var_412 = wp::load(var_409);
            var_411 = (var_412 > var_410);
            var_407 = var_407 && var_411;
        }
        if (var_407) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_413 = wp::extract(var_106, var_390, var_331);
            var_414 = wp::mul(var_399, var_413);
            var_415 = wp::add(var_389, var_414);
        }
        var_416 = wp::where(var_407, var_415, var_389);
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_419 = wp::mul(var_418, var_5);
        var_420 = wp::add(var_15, var_419);
        var_422 = wp::mul(var_421, var_199);
        var_423 = wp::add(var_420, var_422);
        var_424 = wp::add(var_423, var_417);
        var_425 = wp::address(var_flex_bending, var_424);
        var_427 = wp::load(var_425);
        var_426 = wp::copy(var_427);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_428 = wp::extract(var_57, var_417);
        var_429 = wp::address(var_flexvert_xpos_in, var_0, var_428);
        var_431 = wp::load(var_429);
        var_430 = wp::extract(var_431, var_331);
        var_432 = wp::mul(var_426, var_430);
        var_433 = wp::add(var_406, var_432);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_435 = wp::unot(var_dsbl_damper);
        var_434 = var_435;
        if (var_434) {
            var_436 = wp::address(var_flex_damping, var_13);
            var_439 = wp::load(var_436);
            var_438 = (var_439 > var_437);
            var_434 = var_434 && var_438;
        }
        if (var_434) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_440 = wp::extract(var_106, var_417, var_331);
            var_441 = wp::mul(var_426, var_440);
            var_442 = wp::add(var_416, var_441);
        }
        var_443 = wp::where(var_434, var_442, var_416);
        // force_spring[i, x] = -(acc_spring + flex_bending[bendingadr + 17 * eid + 16] * frc[i, x])       <L 801>
        var_445 = wp::mul(var_444, var_5);
        var_446 = wp::add(var_15, var_445);
        var_448 = wp::add(var_446, var_447);
        var_449 = wp::address(var_flex_bending, var_448);
        var_450 = wp::extract(var_58, var_199, var_331);
        var_452 = wp::load(var_449);
        var_451 = wp::mul(var_452, var_450);
        var_453 = wp::add(var_433, var_451);
        var_454 = wp::neg(var_453);
        wp::assign_inplace(var_197, var_199, var_331, var_454);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 802>
        var_456 = wp::unot(var_dsbl_damper);
        var_455 = var_456;
        if (var_455) {
            var_457 = wp::address(var_flex_damping, var_13);
            var_460 = wp::load(var_457);
            var_459 = (var_460 > var_458);
            var_455 = var_455 && var_459;
        }
        if (var_455) {
            // force_damper[i, x] = -acc_damper                                                   <L 803>
            var_461 = wp::neg(var_443);
            wp::assign_inplace(var_198, var_199, var_331, var_461);
        }
        // acc_spring = float(0.0)                                                                <L 793>
        var_464 = wp::float(var_463);
        // acc_damper = float(0.0)                                                                <L 794>
        var_466 = wp::float(var_465);
        // for j in range(4):                                                                     <L 795>
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_469 = wp::mul(var_468, var_5);
        var_470 = wp::add(var_15, var_469);
        var_472 = wp::mul(var_471, var_199);
        var_473 = wp::add(var_470, var_472);
        var_474 = wp::add(var_473, var_467);
        var_475 = wp::address(var_flex_bending, var_474);
        var_477 = wp::load(var_475);
        var_476 = wp::copy(var_477);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_478 = wp::extract(var_57, var_467);
        var_479 = wp::address(var_flexvert_xpos_in, var_0, var_478);
        var_481 = wp::load(var_479);
        var_480 = wp::extract(var_481, var_462);
        var_482 = wp::mul(var_476, var_480);
        var_483 = wp::add(var_464, var_482);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_485 = wp::unot(var_dsbl_damper);
        var_484 = var_485;
        if (var_484) {
            var_486 = wp::address(var_flex_damping, var_13);
            var_489 = wp::load(var_486);
            var_488 = (var_489 > var_487);
            var_484 = var_484 && var_488;
        }
        if (var_484) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_490 = wp::extract(var_106, var_467, var_462);
            var_491 = wp::mul(var_476, var_490);
            var_492 = wp::add(var_466, var_491);
        }
        var_493 = wp::where(var_484, var_492, var_466);
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_496 = wp::mul(var_495, var_5);
        var_497 = wp::add(var_15, var_496);
        var_499 = wp::mul(var_498, var_199);
        var_500 = wp::add(var_497, var_499);
        var_501 = wp::add(var_500, var_494);
        var_502 = wp::address(var_flex_bending, var_501);
        var_504 = wp::load(var_502);
        var_503 = wp::copy(var_504);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_505 = wp::extract(var_57, var_494);
        var_506 = wp::address(var_flexvert_xpos_in, var_0, var_505);
        var_508 = wp::load(var_506);
        var_507 = wp::extract(var_508, var_462);
        var_509 = wp::mul(var_503, var_507);
        var_510 = wp::add(var_483, var_509);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_512 = wp::unot(var_dsbl_damper);
        var_511 = var_512;
        if (var_511) {
            var_513 = wp::address(var_flex_damping, var_13);
            var_516 = wp::load(var_513);
            var_515 = (var_516 > var_514);
            var_511 = var_511 && var_515;
        }
        if (var_511) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_517 = wp::extract(var_106, var_494, var_462);
            var_518 = wp::mul(var_503, var_517);
            var_519 = wp::add(var_493, var_518);
        }
        var_520 = wp::where(var_511, var_519, var_493);
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_523 = wp::mul(var_522, var_5);
        var_524 = wp::add(var_15, var_523);
        var_526 = wp::mul(var_525, var_199);
        var_527 = wp::add(var_524, var_526);
        var_528 = wp::add(var_527, var_521);
        var_529 = wp::address(var_flex_bending, var_528);
        var_531 = wp::load(var_529);
        var_530 = wp::copy(var_531);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_532 = wp::extract(var_57, var_521);
        var_533 = wp::address(var_flexvert_xpos_in, var_0, var_532);
        var_535 = wp::load(var_533);
        var_534 = wp::extract(var_535, var_462);
        var_536 = wp::mul(var_530, var_534);
        var_537 = wp::add(var_510, var_536);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_539 = wp::unot(var_dsbl_damper);
        var_538 = var_539;
        if (var_538) {
            var_540 = wp::address(var_flex_damping, var_13);
            var_543 = wp::load(var_540);
            var_542 = (var_543 > var_541);
            var_538 = var_538 && var_542;
        }
        if (var_538) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_544 = wp::extract(var_106, var_521, var_462);
            var_545 = wp::mul(var_530, var_544);
            var_546 = wp::add(var_520, var_545);
        }
        var_547 = wp::where(var_538, var_546, var_520);
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_550 = wp::mul(var_549, var_5);
        var_551 = wp::add(var_15, var_550);
        var_553 = wp::mul(var_552, var_199);
        var_554 = wp::add(var_551, var_553);
        var_555 = wp::add(var_554, var_548);
        var_556 = wp::address(var_flex_bending, var_555);
        var_558 = wp::load(var_556);
        var_557 = wp::copy(var_558);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_559 = wp::extract(var_57, var_548);
        var_560 = wp::address(var_flexvert_xpos_in, var_0, var_559);
        var_562 = wp::load(var_560);
        var_561 = wp::extract(var_562, var_462);
        var_563 = wp::mul(var_557, var_561);
        var_564 = wp::add(var_537, var_563);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_566 = wp::unot(var_dsbl_damper);
        var_565 = var_566;
        if (var_565) {
            var_567 = wp::address(var_flex_damping, var_13);
            var_570 = wp::load(var_567);
            var_569 = (var_570 > var_568);
            var_565 = var_565 && var_569;
        }
        if (var_565) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_571 = wp::extract(var_106, var_548, var_462);
            var_572 = wp::mul(var_557, var_571);
            var_573 = wp::add(var_547, var_572);
        }
        var_574 = wp::where(var_565, var_573, var_547);
        // force_spring[i, x] = -(acc_spring + flex_bending[bendingadr + 17 * eid + 16] * frc[i, x])       <L 801>
        var_576 = wp::mul(var_575, var_5);
        var_577 = wp::add(var_15, var_576);
        var_579 = wp::add(var_577, var_578);
        var_580 = wp::address(var_flex_bending, var_579);
        var_581 = wp::extract(var_58, var_199, var_462);
        var_583 = wp::load(var_580);
        var_582 = wp::mul(var_583, var_581);
        var_584 = wp::add(var_564, var_582);
        var_585 = wp::neg(var_584);
        wp::assign_inplace(var_197, var_199, var_462, var_585);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 802>
        var_587 = wp::unot(var_dsbl_damper);
        var_586 = var_587;
        if (var_586) {
            var_588 = wp::address(var_flex_damping, var_13);
            var_591 = wp::load(var_588);
            var_590 = (var_591 > var_589);
            var_586 = var_586 && var_590;
        }
        if (var_586) {
            // force_damper[i, x] = -acc_damper                                                   <L 803>
            var_592 = wp::neg(var_574);
            wp::assign_inplace(var_198, var_199, var_462, var_592);
        }
        // for x in range(3):                                                                     <L 792>
        // acc_spring = float(0.0)                                                                <L 793>
        var_596 = wp::float(var_595);
        // acc_damper = float(0.0)                                                                <L 794>
        var_598 = wp::float(var_597);
        // for j in range(4):                                                                     <L 795>
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_601 = wp::mul(var_600, var_5);
        var_602 = wp::add(var_15, var_601);
        var_604 = wp::mul(var_603, var_593);
        var_605 = wp::add(var_602, var_604);
        var_606 = wp::add(var_605, var_599);
        var_607 = wp::address(var_flex_bending, var_606);
        var_609 = wp::load(var_607);
        var_608 = wp::copy(var_609);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_610 = wp::extract(var_57, var_599);
        var_611 = wp::address(var_flexvert_xpos_in, var_0, var_610);
        var_613 = wp::load(var_611);
        var_612 = wp::extract(var_613, var_594);
        var_614 = wp::mul(var_608, var_612);
        var_615 = wp::add(var_596, var_614);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_617 = wp::unot(var_dsbl_damper);
        var_616 = var_617;
        if (var_616) {
            var_618 = wp::address(var_flex_damping, var_13);
            var_621 = wp::load(var_618);
            var_620 = (var_621 > var_619);
            var_616 = var_616 && var_620;
        }
        if (var_616) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_622 = wp::extract(var_106, var_599, var_594);
            var_623 = wp::mul(var_608, var_622);
            var_624 = wp::add(var_598, var_623);
        }
        var_625 = wp::where(var_616, var_624, var_598);
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_628 = wp::mul(var_627, var_5);
        var_629 = wp::add(var_15, var_628);
        var_631 = wp::mul(var_630, var_593);
        var_632 = wp::add(var_629, var_631);
        var_633 = wp::add(var_632, var_626);
        var_634 = wp::address(var_flex_bending, var_633);
        var_636 = wp::load(var_634);
        var_635 = wp::copy(var_636);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_637 = wp::extract(var_57, var_626);
        var_638 = wp::address(var_flexvert_xpos_in, var_0, var_637);
        var_640 = wp::load(var_638);
        var_639 = wp::extract(var_640, var_594);
        var_641 = wp::mul(var_635, var_639);
        var_642 = wp::add(var_615, var_641);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_644 = wp::unot(var_dsbl_damper);
        var_643 = var_644;
        if (var_643) {
            var_645 = wp::address(var_flex_damping, var_13);
            var_648 = wp::load(var_645);
            var_647 = (var_648 > var_646);
            var_643 = var_643 && var_647;
        }
        if (var_643) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_649 = wp::extract(var_106, var_626, var_594);
            var_650 = wp::mul(var_635, var_649);
            var_651 = wp::add(var_625, var_650);
        }
        var_652 = wp::where(var_643, var_651, var_625);
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_655 = wp::mul(var_654, var_5);
        var_656 = wp::add(var_15, var_655);
        var_658 = wp::mul(var_657, var_593);
        var_659 = wp::add(var_656, var_658);
        var_660 = wp::add(var_659, var_653);
        var_661 = wp::address(var_flex_bending, var_660);
        var_663 = wp::load(var_661);
        var_662 = wp::copy(var_663);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_664 = wp::extract(var_57, var_653);
        var_665 = wp::address(var_flexvert_xpos_in, var_0, var_664);
        var_667 = wp::load(var_665);
        var_666 = wp::extract(var_667, var_594);
        var_668 = wp::mul(var_662, var_666);
        var_669 = wp::add(var_642, var_668);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_671 = wp::unot(var_dsbl_damper);
        var_670 = var_671;
        if (var_670) {
            var_672 = wp::address(var_flex_damping, var_13);
            var_675 = wp::load(var_672);
            var_674 = (var_675 > var_673);
            var_670 = var_670 && var_674;
        }
        if (var_670) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_676 = wp::extract(var_106, var_653, var_594);
            var_677 = wp::mul(var_662, var_676);
            var_678 = wp::add(var_652, var_677);
        }
        var_679 = wp::where(var_670, var_678, var_652);
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_682 = wp::mul(var_681, var_5);
        var_683 = wp::add(var_15, var_682);
        var_685 = wp::mul(var_684, var_593);
        var_686 = wp::add(var_683, var_685);
        var_687 = wp::add(var_686, var_680);
        var_688 = wp::address(var_flex_bending, var_687);
        var_690 = wp::load(var_688);
        var_689 = wp::copy(var_690);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_691 = wp::extract(var_57, var_680);
        var_692 = wp::address(var_flexvert_xpos_in, var_0, var_691);
        var_694 = wp::load(var_692);
        var_693 = wp::extract(var_694, var_594);
        var_695 = wp::mul(var_689, var_693);
        var_696 = wp::add(var_669, var_695);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_698 = wp::unot(var_dsbl_damper);
        var_697 = var_698;
        if (var_697) {
            var_699 = wp::address(var_flex_damping, var_13);
            var_702 = wp::load(var_699);
            var_701 = (var_702 > var_700);
            var_697 = var_697 && var_701;
        }
        if (var_697) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_703 = wp::extract(var_106, var_680, var_594);
            var_704 = wp::mul(var_689, var_703);
            var_705 = wp::add(var_679, var_704);
        }
        var_706 = wp::where(var_697, var_705, var_679);
        // force_spring[i, x] = -(acc_spring + flex_bending[bendingadr + 17 * eid + 16] * frc[i, x])       <L 801>
        var_708 = wp::mul(var_707, var_5);
        var_709 = wp::add(var_15, var_708);
        var_711 = wp::add(var_709, var_710);
        var_712 = wp::address(var_flex_bending, var_711);
        var_713 = wp::extract(var_58, var_593, var_594);
        var_715 = wp::load(var_712);
        var_714 = wp::mul(var_715, var_713);
        var_716 = wp::add(var_696, var_714);
        var_717 = wp::neg(var_716);
        wp::assign_inplace(var_197, var_593, var_594, var_717);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 802>
        var_719 = wp::unot(var_dsbl_damper);
        var_718 = var_719;
        if (var_718) {
            var_720 = wp::address(var_flex_damping, var_13);
            var_723 = wp::load(var_720);
            var_722 = (var_723 > var_721);
            var_718 = var_718 && var_722;
        }
        if (var_718) {
            // force_damper[i, x] = -acc_damper                                                   <L 803>
            var_724 = wp::neg(var_706);
            wp::assign_inplace(var_198, var_593, var_594, var_724);
        }
        // acc_spring = float(0.0)                                                                <L 793>
        var_727 = wp::float(var_726);
        // acc_damper = float(0.0)                                                                <L 794>
        var_729 = wp::float(var_728);
        // for j in range(4):                                                                     <L 795>
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_732 = wp::mul(var_731, var_5);
        var_733 = wp::add(var_15, var_732);
        var_735 = wp::mul(var_734, var_593);
        var_736 = wp::add(var_733, var_735);
        var_737 = wp::add(var_736, var_730);
        var_738 = wp::address(var_flex_bending, var_737);
        var_740 = wp::load(var_738);
        var_739 = wp::copy(var_740);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_741 = wp::extract(var_57, var_730);
        var_742 = wp::address(var_flexvert_xpos_in, var_0, var_741);
        var_744 = wp::load(var_742);
        var_743 = wp::extract(var_744, var_725);
        var_745 = wp::mul(var_739, var_743);
        var_746 = wp::add(var_727, var_745);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_748 = wp::unot(var_dsbl_damper);
        var_747 = var_748;
        if (var_747) {
            var_749 = wp::address(var_flex_damping, var_13);
            var_752 = wp::load(var_749);
            var_751 = (var_752 > var_750);
            var_747 = var_747 && var_751;
        }
        if (var_747) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_753 = wp::extract(var_106, var_730, var_725);
            var_754 = wp::mul(var_739, var_753);
            var_755 = wp::add(var_729, var_754);
        }
        var_756 = wp::where(var_747, var_755, var_729);
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_759 = wp::mul(var_758, var_5);
        var_760 = wp::add(var_15, var_759);
        var_762 = wp::mul(var_761, var_593);
        var_763 = wp::add(var_760, var_762);
        var_764 = wp::add(var_763, var_757);
        var_765 = wp::address(var_flex_bending, var_764);
        var_767 = wp::load(var_765);
        var_766 = wp::copy(var_767);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_768 = wp::extract(var_57, var_757);
        var_769 = wp::address(var_flexvert_xpos_in, var_0, var_768);
        var_771 = wp::load(var_769);
        var_770 = wp::extract(var_771, var_725);
        var_772 = wp::mul(var_766, var_770);
        var_773 = wp::add(var_746, var_772);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_775 = wp::unot(var_dsbl_damper);
        var_774 = var_775;
        if (var_774) {
            var_776 = wp::address(var_flex_damping, var_13);
            var_779 = wp::load(var_776);
            var_778 = (var_779 > var_777);
            var_774 = var_774 && var_778;
        }
        if (var_774) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_780 = wp::extract(var_106, var_757, var_725);
            var_781 = wp::mul(var_766, var_780);
            var_782 = wp::add(var_756, var_781);
        }
        var_783 = wp::where(var_774, var_782, var_756);
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_786 = wp::mul(var_785, var_5);
        var_787 = wp::add(var_15, var_786);
        var_789 = wp::mul(var_788, var_593);
        var_790 = wp::add(var_787, var_789);
        var_791 = wp::add(var_790, var_784);
        var_792 = wp::address(var_flex_bending, var_791);
        var_794 = wp::load(var_792);
        var_793 = wp::copy(var_794);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_795 = wp::extract(var_57, var_784);
        var_796 = wp::address(var_flexvert_xpos_in, var_0, var_795);
        var_798 = wp::load(var_796);
        var_797 = wp::extract(var_798, var_725);
        var_799 = wp::mul(var_793, var_797);
        var_800 = wp::add(var_773, var_799);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_802 = wp::unot(var_dsbl_damper);
        var_801 = var_802;
        if (var_801) {
            var_803 = wp::address(var_flex_damping, var_13);
            var_806 = wp::load(var_803);
            var_805 = (var_806 > var_804);
            var_801 = var_801 && var_805;
        }
        if (var_801) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_807 = wp::extract(var_106, var_784, var_725);
            var_808 = wp::mul(var_793, var_807);
            var_809 = wp::add(var_783, var_808);
        }
        var_810 = wp::where(var_801, var_809, var_783);
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_813 = wp::mul(var_812, var_5);
        var_814 = wp::add(var_15, var_813);
        var_816 = wp::mul(var_815, var_593);
        var_817 = wp::add(var_814, var_816);
        var_818 = wp::add(var_817, var_811);
        var_819 = wp::address(var_flex_bending, var_818);
        var_821 = wp::load(var_819);
        var_820 = wp::copy(var_821);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_822 = wp::extract(var_57, var_811);
        var_823 = wp::address(var_flexvert_xpos_in, var_0, var_822);
        var_825 = wp::load(var_823);
        var_824 = wp::extract(var_825, var_725);
        var_826 = wp::mul(var_820, var_824);
        var_827 = wp::add(var_800, var_826);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_829 = wp::unot(var_dsbl_damper);
        var_828 = var_829;
        if (var_828) {
            var_830 = wp::address(var_flex_damping, var_13);
            var_833 = wp::load(var_830);
            var_832 = (var_833 > var_831);
            var_828 = var_828 && var_832;
        }
        if (var_828) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_834 = wp::extract(var_106, var_811, var_725);
            var_835 = wp::mul(var_820, var_834);
            var_836 = wp::add(var_810, var_835);
        }
        var_837 = wp::where(var_828, var_836, var_810);
        // force_spring[i, x] = -(acc_spring + flex_bending[bendingadr + 17 * eid + 16] * frc[i, x])       <L 801>
        var_839 = wp::mul(var_838, var_5);
        var_840 = wp::add(var_15, var_839);
        var_842 = wp::add(var_840, var_841);
        var_843 = wp::address(var_flex_bending, var_842);
        var_844 = wp::extract(var_58, var_593, var_725);
        var_846 = wp::load(var_843);
        var_845 = wp::mul(var_846, var_844);
        var_847 = wp::add(var_827, var_845);
        var_848 = wp::neg(var_847);
        wp::assign_inplace(var_197, var_593, var_725, var_848);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 802>
        var_850 = wp::unot(var_dsbl_damper);
        var_849 = var_850;
        if (var_849) {
            var_851 = wp::address(var_flex_damping, var_13);
            var_854 = wp::load(var_851);
            var_853 = (var_854 > var_852);
            var_849 = var_849 && var_853;
        }
        if (var_849) {
            // force_damper[i, x] = -acc_damper                                                   <L 803>
            var_855 = wp::neg(var_837);
            wp::assign_inplace(var_198, var_593, var_725, var_855);
        }
        // acc_spring = float(0.0)                                                                <L 793>
        var_858 = wp::float(var_857);
        // acc_damper = float(0.0)                                                                <L 794>
        var_860 = wp::float(var_859);
        // for j in range(4):                                                                     <L 795>
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_863 = wp::mul(var_862, var_5);
        var_864 = wp::add(var_15, var_863);
        var_866 = wp::mul(var_865, var_593);
        var_867 = wp::add(var_864, var_866);
        var_868 = wp::add(var_867, var_861);
        var_869 = wp::address(var_flex_bending, var_868);
        var_871 = wp::load(var_869);
        var_870 = wp::copy(var_871);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_872 = wp::extract(var_57, var_861);
        var_873 = wp::address(var_flexvert_xpos_in, var_0, var_872);
        var_875 = wp::load(var_873);
        var_874 = wp::extract(var_875, var_856);
        var_876 = wp::mul(var_870, var_874);
        var_877 = wp::add(var_858, var_876);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_879 = wp::unot(var_dsbl_damper);
        var_878 = var_879;
        if (var_878) {
            var_880 = wp::address(var_flex_damping, var_13);
            var_883 = wp::load(var_880);
            var_882 = (var_883 > var_881);
            var_878 = var_878 && var_882;
        }
        if (var_878) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_884 = wp::extract(var_106, var_861, var_856);
            var_885 = wp::mul(var_870, var_884);
            var_886 = wp::add(var_860, var_885);
        }
        var_887 = wp::where(var_878, var_886, var_860);
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_890 = wp::mul(var_889, var_5);
        var_891 = wp::add(var_15, var_890);
        var_893 = wp::mul(var_892, var_593);
        var_894 = wp::add(var_891, var_893);
        var_895 = wp::add(var_894, var_888);
        var_896 = wp::address(var_flex_bending, var_895);
        var_898 = wp::load(var_896);
        var_897 = wp::copy(var_898);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_899 = wp::extract(var_57, var_888);
        var_900 = wp::address(var_flexvert_xpos_in, var_0, var_899);
        var_902 = wp::load(var_900);
        var_901 = wp::extract(var_902, var_856);
        var_903 = wp::mul(var_897, var_901);
        var_904 = wp::add(var_877, var_903);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_906 = wp::unot(var_dsbl_damper);
        var_905 = var_906;
        if (var_905) {
            var_907 = wp::address(var_flex_damping, var_13);
            var_910 = wp::load(var_907);
            var_909 = (var_910 > var_908);
            var_905 = var_905 && var_909;
        }
        if (var_905) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_911 = wp::extract(var_106, var_888, var_856);
            var_912 = wp::mul(var_897, var_911);
            var_913 = wp::add(var_887, var_912);
        }
        var_914 = wp::where(var_905, var_913, var_887);
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_917 = wp::mul(var_916, var_5);
        var_918 = wp::add(var_15, var_917);
        var_920 = wp::mul(var_919, var_593);
        var_921 = wp::add(var_918, var_920);
        var_922 = wp::add(var_921, var_915);
        var_923 = wp::address(var_flex_bending, var_922);
        var_925 = wp::load(var_923);
        var_924 = wp::copy(var_925);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_926 = wp::extract(var_57, var_915);
        var_927 = wp::address(var_flexvert_xpos_in, var_0, var_926);
        var_929 = wp::load(var_927);
        var_928 = wp::extract(var_929, var_856);
        var_930 = wp::mul(var_924, var_928);
        var_931 = wp::add(var_904, var_930);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_933 = wp::unot(var_dsbl_damper);
        var_932 = var_933;
        if (var_932) {
            var_934 = wp::address(var_flex_damping, var_13);
            var_937 = wp::load(var_934);
            var_936 = (var_937 > var_935);
            var_932 = var_932 && var_936;
        }
        if (var_932) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_938 = wp::extract(var_106, var_915, var_856);
            var_939 = wp::mul(var_924, var_938);
            var_940 = wp::add(var_914, var_939);
        }
        var_941 = wp::where(var_932, var_940, var_914);
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_944 = wp::mul(var_943, var_5);
        var_945 = wp::add(var_15, var_944);
        var_947 = wp::mul(var_946, var_593);
        var_948 = wp::add(var_945, var_947);
        var_949 = wp::add(var_948, var_942);
        var_950 = wp::address(var_flex_bending, var_949);
        var_952 = wp::load(var_950);
        var_951 = wp::copy(var_952);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_953 = wp::extract(var_57, var_942);
        var_954 = wp::address(var_flexvert_xpos_in, var_0, var_953);
        var_956 = wp::load(var_954);
        var_955 = wp::extract(var_956, var_856);
        var_957 = wp::mul(var_951, var_955);
        var_958 = wp::add(var_931, var_957);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_960 = wp::unot(var_dsbl_damper);
        var_959 = var_960;
        if (var_959) {
            var_961 = wp::address(var_flex_damping, var_13);
            var_964 = wp::load(var_961);
            var_963 = (var_964 > var_962);
            var_959 = var_959 && var_963;
        }
        if (var_959) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_965 = wp::extract(var_106, var_942, var_856);
            var_966 = wp::mul(var_951, var_965);
            var_967 = wp::add(var_941, var_966);
        }
        var_968 = wp::where(var_959, var_967, var_941);
        // force_spring[i, x] = -(acc_spring + flex_bending[bendingadr + 17 * eid + 16] * frc[i, x])       <L 801>
        var_970 = wp::mul(var_969, var_5);
        var_971 = wp::add(var_15, var_970);
        var_973 = wp::add(var_971, var_972);
        var_974 = wp::address(var_flex_bending, var_973);
        var_975 = wp::extract(var_58, var_593, var_856);
        var_977 = wp::load(var_974);
        var_976 = wp::mul(var_977, var_975);
        var_978 = wp::add(var_958, var_976);
        var_979 = wp::neg(var_978);
        wp::assign_inplace(var_197, var_593, var_856, var_979);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 802>
        var_981 = wp::unot(var_dsbl_damper);
        var_980 = var_981;
        if (var_980) {
            var_982 = wp::address(var_flex_damping, var_13);
            var_985 = wp::load(var_982);
            var_984 = (var_985 > var_983);
            var_980 = var_980 && var_984;
        }
        if (var_980) {
            // force_damper[i, x] = -acc_damper                                                   <L 803>
            var_986 = wp::neg(var_968);
            wp::assign_inplace(var_198, var_593, var_856, var_986);
        }
        // for x in range(3):                                                                     <L 792>
        // acc_spring = float(0.0)                                                                <L 793>
        var_990 = wp::float(var_989);
        // acc_damper = float(0.0)                                                                <L 794>
        var_992 = wp::float(var_991);
        // for j in range(4):                                                                     <L 795>
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_995 = wp::mul(var_994, var_5);
        var_996 = wp::add(var_15, var_995);
        var_998 = wp::mul(var_997, var_987);
        var_999 = wp::add(var_996, var_998);
        var_1000 = wp::add(var_999, var_993);
        var_1001 = wp::address(var_flex_bending, var_1000);
        var_1003 = wp::load(var_1001);
        var_1002 = wp::copy(var_1003);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_1004 = wp::extract(var_57, var_993);
        var_1005 = wp::address(var_flexvert_xpos_in, var_0, var_1004);
        var_1007 = wp::load(var_1005);
        var_1006 = wp::extract(var_1007, var_988);
        var_1008 = wp::mul(var_1002, var_1006);
        var_1009 = wp::add(var_990, var_1008);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_1011 = wp::unot(var_dsbl_damper);
        var_1010 = var_1011;
        if (var_1010) {
            var_1012 = wp::address(var_flex_damping, var_13);
            var_1015 = wp::load(var_1012);
            var_1014 = (var_1015 > var_1013);
            var_1010 = var_1010 && var_1014;
        }
        if (var_1010) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_1016 = wp::extract(var_106, var_993, var_988);
            var_1017 = wp::mul(var_1002, var_1016);
            var_1018 = wp::add(var_992, var_1017);
        }
        var_1019 = wp::where(var_1010, var_1018, var_992);
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_1022 = wp::mul(var_1021, var_5);
        var_1023 = wp::add(var_15, var_1022);
        var_1025 = wp::mul(var_1024, var_987);
        var_1026 = wp::add(var_1023, var_1025);
        var_1027 = wp::add(var_1026, var_1020);
        var_1028 = wp::address(var_flex_bending, var_1027);
        var_1030 = wp::load(var_1028);
        var_1029 = wp::copy(var_1030);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_1031 = wp::extract(var_57, var_1020);
        var_1032 = wp::address(var_flexvert_xpos_in, var_0, var_1031);
        var_1034 = wp::load(var_1032);
        var_1033 = wp::extract(var_1034, var_988);
        var_1035 = wp::mul(var_1029, var_1033);
        var_1036 = wp::add(var_1009, var_1035);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_1038 = wp::unot(var_dsbl_damper);
        var_1037 = var_1038;
        if (var_1037) {
            var_1039 = wp::address(var_flex_damping, var_13);
            var_1042 = wp::load(var_1039);
            var_1041 = (var_1042 > var_1040);
            var_1037 = var_1037 && var_1041;
        }
        if (var_1037) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_1043 = wp::extract(var_106, var_1020, var_988);
            var_1044 = wp::mul(var_1029, var_1043);
            var_1045 = wp::add(var_1019, var_1044);
        }
        var_1046 = wp::where(var_1037, var_1045, var_1019);
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_1049 = wp::mul(var_1048, var_5);
        var_1050 = wp::add(var_15, var_1049);
        var_1052 = wp::mul(var_1051, var_987);
        var_1053 = wp::add(var_1050, var_1052);
        var_1054 = wp::add(var_1053, var_1047);
        var_1055 = wp::address(var_flex_bending, var_1054);
        var_1057 = wp::load(var_1055);
        var_1056 = wp::copy(var_1057);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_1058 = wp::extract(var_57, var_1047);
        var_1059 = wp::address(var_flexvert_xpos_in, var_0, var_1058);
        var_1061 = wp::load(var_1059);
        var_1060 = wp::extract(var_1061, var_988);
        var_1062 = wp::mul(var_1056, var_1060);
        var_1063 = wp::add(var_1036, var_1062);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_1065 = wp::unot(var_dsbl_damper);
        var_1064 = var_1065;
        if (var_1064) {
            var_1066 = wp::address(var_flex_damping, var_13);
            var_1069 = wp::load(var_1066);
            var_1068 = (var_1069 > var_1067);
            var_1064 = var_1064 && var_1068;
        }
        if (var_1064) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_1070 = wp::extract(var_106, var_1047, var_988);
            var_1071 = wp::mul(var_1056, var_1070);
            var_1072 = wp::add(var_1046, var_1071);
        }
        var_1073 = wp::where(var_1064, var_1072, var_1046);
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_1076 = wp::mul(var_1075, var_5);
        var_1077 = wp::add(var_15, var_1076);
        var_1079 = wp::mul(var_1078, var_987);
        var_1080 = wp::add(var_1077, var_1079);
        var_1081 = wp::add(var_1080, var_1074);
        var_1082 = wp::address(var_flex_bending, var_1081);
        var_1084 = wp::load(var_1082);
        var_1083 = wp::copy(var_1084);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_1085 = wp::extract(var_57, var_1074);
        var_1086 = wp::address(var_flexvert_xpos_in, var_0, var_1085);
        var_1088 = wp::load(var_1086);
        var_1087 = wp::extract(var_1088, var_988);
        var_1089 = wp::mul(var_1083, var_1087);
        var_1090 = wp::add(var_1063, var_1089);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_1092 = wp::unot(var_dsbl_damper);
        var_1091 = var_1092;
        if (var_1091) {
            var_1093 = wp::address(var_flex_damping, var_13);
            var_1096 = wp::load(var_1093);
            var_1095 = (var_1096 > var_1094);
            var_1091 = var_1091 && var_1095;
        }
        if (var_1091) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_1097 = wp::extract(var_106, var_1074, var_988);
            var_1098 = wp::mul(var_1083, var_1097);
            var_1099 = wp::add(var_1073, var_1098);
        }
        var_1100 = wp::where(var_1091, var_1099, var_1073);
        // force_spring[i, x] = -(acc_spring + flex_bending[bendingadr + 17 * eid + 16] * frc[i, x])       <L 801>
        var_1102 = wp::mul(var_1101, var_5);
        var_1103 = wp::add(var_15, var_1102);
        var_1105 = wp::add(var_1103, var_1104);
        var_1106 = wp::address(var_flex_bending, var_1105);
        var_1107 = wp::extract(var_58, var_987, var_988);
        var_1109 = wp::load(var_1106);
        var_1108 = wp::mul(var_1109, var_1107);
        var_1110 = wp::add(var_1090, var_1108);
        var_1111 = wp::neg(var_1110);
        wp::assign_inplace(var_197, var_987, var_988, var_1111);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 802>
        var_1113 = wp::unot(var_dsbl_damper);
        var_1112 = var_1113;
        if (var_1112) {
            var_1114 = wp::address(var_flex_damping, var_13);
            var_1117 = wp::load(var_1114);
            var_1116 = (var_1117 > var_1115);
            var_1112 = var_1112 && var_1116;
        }
        if (var_1112) {
            // force_damper[i, x] = -acc_damper                                                   <L 803>
            var_1118 = wp::neg(var_1100);
            wp::assign_inplace(var_198, var_987, var_988, var_1118);
        }
        // acc_spring = float(0.0)                                                                <L 793>
        var_1121 = wp::float(var_1120);
        // acc_damper = float(0.0)                                                                <L 794>
        var_1123 = wp::float(var_1122);
        // for j in range(4):                                                                     <L 795>
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_1126 = wp::mul(var_1125, var_5);
        var_1127 = wp::add(var_15, var_1126);
        var_1129 = wp::mul(var_1128, var_987);
        var_1130 = wp::add(var_1127, var_1129);
        var_1131 = wp::add(var_1130, var_1124);
        var_1132 = wp::address(var_flex_bending, var_1131);
        var_1134 = wp::load(var_1132);
        var_1133 = wp::copy(var_1134);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_1135 = wp::extract(var_57, var_1124);
        var_1136 = wp::address(var_flexvert_xpos_in, var_0, var_1135);
        var_1138 = wp::load(var_1136);
        var_1137 = wp::extract(var_1138, var_1119);
        var_1139 = wp::mul(var_1133, var_1137);
        var_1140 = wp::add(var_1121, var_1139);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_1142 = wp::unot(var_dsbl_damper);
        var_1141 = var_1142;
        if (var_1141) {
            var_1143 = wp::address(var_flex_damping, var_13);
            var_1146 = wp::load(var_1143);
            var_1145 = (var_1146 > var_1144);
            var_1141 = var_1141 && var_1145;
        }
        if (var_1141) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_1147 = wp::extract(var_106, var_1124, var_1119);
            var_1148 = wp::mul(var_1133, var_1147);
            var_1149 = wp::add(var_1123, var_1148);
        }
        var_1150 = wp::where(var_1141, var_1149, var_1123);
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_1153 = wp::mul(var_1152, var_5);
        var_1154 = wp::add(var_15, var_1153);
        var_1156 = wp::mul(var_1155, var_987);
        var_1157 = wp::add(var_1154, var_1156);
        var_1158 = wp::add(var_1157, var_1151);
        var_1159 = wp::address(var_flex_bending, var_1158);
        var_1161 = wp::load(var_1159);
        var_1160 = wp::copy(var_1161);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_1162 = wp::extract(var_57, var_1151);
        var_1163 = wp::address(var_flexvert_xpos_in, var_0, var_1162);
        var_1165 = wp::load(var_1163);
        var_1164 = wp::extract(var_1165, var_1119);
        var_1166 = wp::mul(var_1160, var_1164);
        var_1167 = wp::add(var_1140, var_1166);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_1169 = wp::unot(var_dsbl_damper);
        var_1168 = var_1169;
        if (var_1168) {
            var_1170 = wp::address(var_flex_damping, var_13);
            var_1173 = wp::load(var_1170);
            var_1172 = (var_1173 > var_1171);
            var_1168 = var_1168 && var_1172;
        }
        if (var_1168) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_1174 = wp::extract(var_106, var_1151, var_1119);
            var_1175 = wp::mul(var_1160, var_1174);
            var_1176 = wp::add(var_1150, var_1175);
        }
        var_1177 = wp::where(var_1168, var_1176, var_1150);
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_1180 = wp::mul(var_1179, var_5);
        var_1181 = wp::add(var_15, var_1180);
        var_1183 = wp::mul(var_1182, var_987);
        var_1184 = wp::add(var_1181, var_1183);
        var_1185 = wp::add(var_1184, var_1178);
        var_1186 = wp::address(var_flex_bending, var_1185);
        var_1188 = wp::load(var_1186);
        var_1187 = wp::copy(var_1188);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_1189 = wp::extract(var_57, var_1178);
        var_1190 = wp::address(var_flexvert_xpos_in, var_0, var_1189);
        var_1192 = wp::load(var_1190);
        var_1191 = wp::extract(var_1192, var_1119);
        var_1193 = wp::mul(var_1187, var_1191);
        var_1194 = wp::add(var_1167, var_1193);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_1196 = wp::unot(var_dsbl_damper);
        var_1195 = var_1196;
        if (var_1195) {
            var_1197 = wp::address(var_flex_damping, var_13);
            var_1200 = wp::load(var_1197);
            var_1199 = (var_1200 > var_1198);
            var_1195 = var_1195 && var_1199;
        }
        if (var_1195) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_1201 = wp::extract(var_106, var_1178, var_1119);
            var_1202 = wp::mul(var_1187, var_1201);
            var_1203 = wp::add(var_1177, var_1202);
        }
        var_1204 = wp::where(var_1195, var_1203, var_1177);
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_1207 = wp::mul(var_1206, var_5);
        var_1208 = wp::add(var_15, var_1207);
        var_1210 = wp::mul(var_1209, var_987);
        var_1211 = wp::add(var_1208, var_1210);
        var_1212 = wp::add(var_1211, var_1205);
        var_1213 = wp::address(var_flex_bending, var_1212);
        var_1215 = wp::load(var_1213);
        var_1214 = wp::copy(var_1215);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_1216 = wp::extract(var_57, var_1205);
        var_1217 = wp::address(var_flexvert_xpos_in, var_0, var_1216);
        var_1219 = wp::load(var_1217);
        var_1218 = wp::extract(var_1219, var_1119);
        var_1220 = wp::mul(var_1214, var_1218);
        var_1221 = wp::add(var_1194, var_1220);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_1223 = wp::unot(var_dsbl_damper);
        var_1222 = var_1223;
        if (var_1222) {
            var_1224 = wp::address(var_flex_damping, var_13);
            var_1227 = wp::load(var_1224);
            var_1226 = (var_1227 > var_1225);
            var_1222 = var_1222 && var_1226;
        }
        if (var_1222) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_1228 = wp::extract(var_106, var_1205, var_1119);
            var_1229 = wp::mul(var_1214, var_1228);
            var_1230 = wp::add(var_1204, var_1229);
        }
        var_1231 = wp::where(var_1222, var_1230, var_1204);
        // force_spring[i, x] = -(acc_spring + flex_bending[bendingadr + 17 * eid + 16] * frc[i, x])       <L 801>
        var_1233 = wp::mul(var_1232, var_5);
        var_1234 = wp::add(var_15, var_1233);
        var_1236 = wp::add(var_1234, var_1235);
        var_1237 = wp::address(var_flex_bending, var_1236);
        var_1238 = wp::extract(var_58, var_987, var_1119);
        var_1240 = wp::load(var_1237);
        var_1239 = wp::mul(var_1240, var_1238);
        var_1241 = wp::add(var_1221, var_1239);
        var_1242 = wp::neg(var_1241);
        wp::assign_inplace(var_197, var_987, var_1119, var_1242);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 802>
        var_1244 = wp::unot(var_dsbl_damper);
        var_1243 = var_1244;
        if (var_1243) {
            var_1245 = wp::address(var_flex_damping, var_13);
            var_1248 = wp::load(var_1245);
            var_1247 = (var_1248 > var_1246);
            var_1243 = var_1243 && var_1247;
        }
        if (var_1243) {
            // force_damper[i, x] = -acc_damper                                                   <L 803>
            var_1249 = wp::neg(var_1231);
            wp::assign_inplace(var_198, var_987, var_1119, var_1249);
        }
        // acc_spring = float(0.0)                                                                <L 793>
        var_1252 = wp::float(var_1251);
        // acc_damper = float(0.0)                                                                <L 794>
        var_1254 = wp::float(var_1253);
        // for j in range(4):                                                                     <L 795>
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_1257 = wp::mul(var_1256, var_5);
        var_1258 = wp::add(var_15, var_1257);
        var_1260 = wp::mul(var_1259, var_987);
        var_1261 = wp::add(var_1258, var_1260);
        var_1262 = wp::add(var_1261, var_1255);
        var_1263 = wp::address(var_flex_bending, var_1262);
        var_1265 = wp::load(var_1263);
        var_1264 = wp::copy(var_1265);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_1266 = wp::extract(var_57, var_1255);
        var_1267 = wp::address(var_flexvert_xpos_in, var_0, var_1266);
        var_1269 = wp::load(var_1267);
        var_1268 = wp::extract(var_1269, var_1250);
        var_1270 = wp::mul(var_1264, var_1268);
        var_1271 = wp::add(var_1252, var_1270);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_1273 = wp::unot(var_dsbl_damper);
        var_1272 = var_1273;
        if (var_1272) {
            var_1274 = wp::address(var_flex_damping, var_13);
            var_1277 = wp::load(var_1274);
            var_1276 = (var_1277 > var_1275);
            var_1272 = var_1272 && var_1276;
        }
        if (var_1272) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_1278 = wp::extract(var_106, var_1255, var_1250);
            var_1279 = wp::mul(var_1264, var_1278);
            var_1280 = wp::add(var_1254, var_1279);
        }
        var_1281 = wp::where(var_1272, var_1280, var_1254);
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_1284 = wp::mul(var_1283, var_5);
        var_1285 = wp::add(var_15, var_1284);
        var_1287 = wp::mul(var_1286, var_987);
        var_1288 = wp::add(var_1285, var_1287);
        var_1289 = wp::add(var_1288, var_1282);
        var_1290 = wp::address(var_flex_bending, var_1289);
        var_1292 = wp::load(var_1290);
        var_1291 = wp::copy(var_1292);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_1293 = wp::extract(var_57, var_1282);
        var_1294 = wp::address(var_flexvert_xpos_in, var_0, var_1293);
        var_1296 = wp::load(var_1294);
        var_1295 = wp::extract(var_1296, var_1250);
        var_1297 = wp::mul(var_1291, var_1295);
        var_1298 = wp::add(var_1271, var_1297);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_1300 = wp::unot(var_dsbl_damper);
        var_1299 = var_1300;
        if (var_1299) {
            var_1301 = wp::address(var_flex_damping, var_13);
            var_1304 = wp::load(var_1301);
            var_1303 = (var_1304 > var_1302);
            var_1299 = var_1299 && var_1303;
        }
        if (var_1299) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_1305 = wp::extract(var_106, var_1282, var_1250);
            var_1306 = wp::mul(var_1291, var_1305);
            var_1307 = wp::add(var_1281, var_1306);
        }
        var_1308 = wp::where(var_1299, var_1307, var_1281);
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_1311 = wp::mul(var_1310, var_5);
        var_1312 = wp::add(var_15, var_1311);
        var_1314 = wp::mul(var_1313, var_987);
        var_1315 = wp::add(var_1312, var_1314);
        var_1316 = wp::add(var_1315, var_1309);
        var_1317 = wp::address(var_flex_bending, var_1316);
        var_1319 = wp::load(var_1317);
        var_1318 = wp::copy(var_1319);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_1320 = wp::extract(var_57, var_1309);
        var_1321 = wp::address(var_flexvert_xpos_in, var_0, var_1320);
        var_1323 = wp::load(var_1321);
        var_1322 = wp::extract(var_1323, var_1250);
        var_1324 = wp::mul(var_1318, var_1322);
        var_1325 = wp::add(var_1298, var_1324);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_1327 = wp::unot(var_dsbl_damper);
        var_1326 = var_1327;
        if (var_1326) {
            var_1328 = wp::address(var_flex_damping, var_13);
            var_1331 = wp::load(var_1328);
            var_1330 = (var_1331 > var_1329);
            var_1326 = var_1326 && var_1330;
        }
        if (var_1326) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_1332 = wp::extract(var_106, var_1309, var_1250);
            var_1333 = wp::mul(var_1318, var_1332);
            var_1334 = wp::add(var_1308, var_1333);
        }
        var_1335 = wp::where(var_1326, var_1334, var_1308);
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_1338 = wp::mul(var_1337, var_5);
        var_1339 = wp::add(var_15, var_1338);
        var_1341 = wp::mul(var_1340, var_987);
        var_1342 = wp::add(var_1339, var_1341);
        var_1343 = wp::add(var_1342, var_1336);
        var_1344 = wp::address(var_flex_bending, var_1343);
        var_1346 = wp::load(var_1344);
        var_1345 = wp::copy(var_1346);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_1347 = wp::extract(var_57, var_1336);
        var_1348 = wp::address(var_flexvert_xpos_in, var_0, var_1347);
        var_1350 = wp::load(var_1348);
        var_1349 = wp::extract(var_1350, var_1250);
        var_1351 = wp::mul(var_1345, var_1349);
        var_1352 = wp::add(var_1325, var_1351);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_1354 = wp::unot(var_dsbl_damper);
        var_1353 = var_1354;
        if (var_1353) {
            var_1355 = wp::address(var_flex_damping, var_13);
            var_1358 = wp::load(var_1355);
            var_1357 = (var_1358 > var_1356);
            var_1353 = var_1353 && var_1357;
        }
        if (var_1353) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_1359 = wp::extract(var_106, var_1336, var_1250);
            var_1360 = wp::mul(var_1345, var_1359);
            var_1361 = wp::add(var_1335, var_1360);
        }
        var_1362 = wp::where(var_1353, var_1361, var_1335);
        // force_spring[i, x] = -(acc_spring + flex_bending[bendingadr + 17 * eid + 16] * frc[i, x])       <L 801>
        var_1364 = wp::mul(var_1363, var_5);
        var_1365 = wp::add(var_15, var_1364);
        var_1367 = wp::add(var_1365, var_1366);
        var_1368 = wp::address(var_flex_bending, var_1367);
        var_1369 = wp::extract(var_58, var_987, var_1250);
        var_1371 = wp::load(var_1368);
        var_1370 = wp::mul(var_1371, var_1369);
        var_1372 = wp::add(var_1352, var_1370);
        var_1373 = wp::neg(var_1372);
        wp::assign_inplace(var_197, var_987, var_1250, var_1373);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 802>
        var_1375 = wp::unot(var_dsbl_damper);
        var_1374 = var_1375;
        if (var_1374) {
            var_1376 = wp::address(var_flex_damping, var_13);
            var_1379 = wp::load(var_1376);
            var_1378 = (var_1379 > var_1377);
            var_1374 = var_1374 && var_1378;
        }
        if (var_1374) {
            // force_damper[i, x] = -acc_damper                                                   <L 803>
            var_1380 = wp::neg(var_1362);
            wp::assign_inplace(var_198, var_987, var_1250, var_1380);
        }
        // for x in range(3):                                                                     <L 792>
        // acc_spring = float(0.0)                                                                <L 793>
        var_1384 = wp::float(var_1383);
        // acc_damper = float(0.0)                                                                <L 794>
        var_1386 = wp::float(var_1385);
        // for j in range(4):                                                                     <L 795>
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_1389 = wp::mul(var_1388, var_5);
        var_1390 = wp::add(var_15, var_1389);
        var_1392 = wp::mul(var_1391, var_1381);
        var_1393 = wp::add(var_1390, var_1392);
        var_1394 = wp::add(var_1393, var_1387);
        var_1395 = wp::address(var_flex_bending, var_1394);
        var_1397 = wp::load(var_1395);
        var_1396 = wp::copy(var_1397);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_1398 = wp::extract(var_57, var_1387);
        var_1399 = wp::address(var_flexvert_xpos_in, var_0, var_1398);
        var_1401 = wp::load(var_1399);
        var_1400 = wp::extract(var_1401, var_1382);
        var_1402 = wp::mul(var_1396, var_1400);
        var_1403 = wp::add(var_1384, var_1402);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_1405 = wp::unot(var_dsbl_damper);
        var_1404 = var_1405;
        if (var_1404) {
            var_1406 = wp::address(var_flex_damping, var_13);
            var_1409 = wp::load(var_1406);
            var_1408 = (var_1409 > var_1407);
            var_1404 = var_1404 && var_1408;
        }
        if (var_1404) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_1410 = wp::extract(var_106, var_1387, var_1382);
            var_1411 = wp::mul(var_1396, var_1410);
            var_1412 = wp::add(var_1386, var_1411);
        }
        var_1413 = wp::where(var_1404, var_1412, var_1386);
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_1416 = wp::mul(var_1415, var_5);
        var_1417 = wp::add(var_15, var_1416);
        var_1419 = wp::mul(var_1418, var_1381);
        var_1420 = wp::add(var_1417, var_1419);
        var_1421 = wp::add(var_1420, var_1414);
        var_1422 = wp::address(var_flex_bending, var_1421);
        var_1424 = wp::load(var_1422);
        var_1423 = wp::copy(var_1424);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_1425 = wp::extract(var_57, var_1414);
        var_1426 = wp::address(var_flexvert_xpos_in, var_0, var_1425);
        var_1428 = wp::load(var_1426);
        var_1427 = wp::extract(var_1428, var_1382);
        var_1429 = wp::mul(var_1423, var_1427);
        var_1430 = wp::add(var_1403, var_1429);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_1432 = wp::unot(var_dsbl_damper);
        var_1431 = var_1432;
        if (var_1431) {
            var_1433 = wp::address(var_flex_damping, var_13);
            var_1436 = wp::load(var_1433);
            var_1435 = (var_1436 > var_1434);
            var_1431 = var_1431 && var_1435;
        }
        if (var_1431) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_1437 = wp::extract(var_106, var_1414, var_1382);
            var_1438 = wp::mul(var_1423, var_1437);
            var_1439 = wp::add(var_1413, var_1438);
        }
        var_1440 = wp::where(var_1431, var_1439, var_1413);
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_1443 = wp::mul(var_1442, var_5);
        var_1444 = wp::add(var_15, var_1443);
        var_1446 = wp::mul(var_1445, var_1381);
        var_1447 = wp::add(var_1444, var_1446);
        var_1448 = wp::add(var_1447, var_1441);
        var_1449 = wp::address(var_flex_bending, var_1448);
        var_1451 = wp::load(var_1449);
        var_1450 = wp::copy(var_1451);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_1452 = wp::extract(var_57, var_1441);
        var_1453 = wp::address(var_flexvert_xpos_in, var_0, var_1452);
        var_1455 = wp::load(var_1453);
        var_1454 = wp::extract(var_1455, var_1382);
        var_1456 = wp::mul(var_1450, var_1454);
        var_1457 = wp::add(var_1430, var_1456);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_1459 = wp::unot(var_dsbl_damper);
        var_1458 = var_1459;
        if (var_1458) {
            var_1460 = wp::address(var_flex_damping, var_13);
            var_1463 = wp::load(var_1460);
            var_1462 = (var_1463 > var_1461);
            var_1458 = var_1458 && var_1462;
        }
        if (var_1458) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_1464 = wp::extract(var_106, var_1441, var_1382);
            var_1465 = wp::mul(var_1450, var_1464);
            var_1466 = wp::add(var_1440, var_1465);
        }
        var_1467 = wp::where(var_1458, var_1466, var_1440);
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_1470 = wp::mul(var_1469, var_5);
        var_1471 = wp::add(var_15, var_1470);
        var_1473 = wp::mul(var_1472, var_1381);
        var_1474 = wp::add(var_1471, var_1473);
        var_1475 = wp::add(var_1474, var_1468);
        var_1476 = wp::address(var_flex_bending, var_1475);
        var_1478 = wp::load(var_1476);
        var_1477 = wp::copy(var_1478);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_1479 = wp::extract(var_57, var_1468);
        var_1480 = wp::address(var_flexvert_xpos_in, var_0, var_1479);
        var_1482 = wp::load(var_1480);
        var_1481 = wp::extract(var_1482, var_1382);
        var_1483 = wp::mul(var_1477, var_1481);
        var_1484 = wp::add(var_1457, var_1483);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_1486 = wp::unot(var_dsbl_damper);
        var_1485 = var_1486;
        if (var_1485) {
            var_1487 = wp::address(var_flex_damping, var_13);
            var_1490 = wp::load(var_1487);
            var_1489 = (var_1490 > var_1488);
            var_1485 = var_1485 && var_1489;
        }
        if (var_1485) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_1491 = wp::extract(var_106, var_1468, var_1382);
            var_1492 = wp::mul(var_1477, var_1491);
            var_1493 = wp::add(var_1467, var_1492);
        }
        var_1494 = wp::where(var_1485, var_1493, var_1467);
        // force_spring[i, x] = -(acc_spring + flex_bending[bendingadr + 17 * eid + 16] * frc[i, x])       <L 801>
        var_1496 = wp::mul(var_1495, var_5);
        var_1497 = wp::add(var_15, var_1496);
        var_1499 = wp::add(var_1497, var_1498);
        var_1500 = wp::address(var_flex_bending, var_1499);
        var_1501 = wp::extract(var_58, var_1381, var_1382);
        var_1503 = wp::load(var_1500);
        var_1502 = wp::mul(var_1503, var_1501);
        var_1504 = wp::add(var_1484, var_1502);
        var_1505 = wp::neg(var_1504);
        wp::assign_inplace(var_197, var_1381, var_1382, var_1505);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 802>
        var_1507 = wp::unot(var_dsbl_damper);
        var_1506 = var_1507;
        if (var_1506) {
            var_1508 = wp::address(var_flex_damping, var_13);
            var_1511 = wp::load(var_1508);
            var_1510 = (var_1511 > var_1509);
            var_1506 = var_1506 && var_1510;
        }
        if (var_1506) {
            // force_damper[i, x] = -acc_damper                                                   <L 803>
            var_1512 = wp::neg(var_1494);
            wp::assign_inplace(var_198, var_1381, var_1382, var_1512);
        }
        // acc_spring = float(0.0)                                                                <L 793>
        var_1515 = wp::float(var_1514);
        // acc_damper = float(0.0)                                                                <L 794>
        var_1517 = wp::float(var_1516);
        // for j in range(4):                                                                     <L 795>
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_1520 = wp::mul(var_1519, var_5);
        var_1521 = wp::add(var_15, var_1520);
        var_1523 = wp::mul(var_1522, var_1381);
        var_1524 = wp::add(var_1521, var_1523);
        var_1525 = wp::add(var_1524, var_1518);
        var_1526 = wp::address(var_flex_bending, var_1525);
        var_1528 = wp::load(var_1526);
        var_1527 = wp::copy(var_1528);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_1529 = wp::extract(var_57, var_1518);
        var_1530 = wp::address(var_flexvert_xpos_in, var_0, var_1529);
        var_1532 = wp::load(var_1530);
        var_1531 = wp::extract(var_1532, var_1513);
        var_1533 = wp::mul(var_1527, var_1531);
        var_1534 = wp::add(var_1515, var_1533);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_1536 = wp::unot(var_dsbl_damper);
        var_1535 = var_1536;
        if (var_1535) {
            var_1537 = wp::address(var_flex_damping, var_13);
            var_1540 = wp::load(var_1537);
            var_1539 = (var_1540 > var_1538);
            var_1535 = var_1535 && var_1539;
        }
        if (var_1535) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_1541 = wp::extract(var_106, var_1518, var_1513);
            var_1542 = wp::mul(var_1527, var_1541);
            var_1543 = wp::add(var_1517, var_1542);
        }
        var_1544 = wp::where(var_1535, var_1543, var_1517);
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_1547 = wp::mul(var_1546, var_5);
        var_1548 = wp::add(var_15, var_1547);
        var_1550 = wp::mul(var_1549, var_1381);
        var_1551 = wp::add(var_1548, var_1550);
        var_1552 = wp::add(var_1551, var_1545);
        var_1553 = wp::address(var_flex_bending, var_1552);
        var_1555 = wp::load(var_1553);
        var_1554 = wp::copy(var_1555);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_1556 = wp::extract(var_57, var_1545);
        var_1557 = wp::address(var_flexvert_xpos_in, var_0, var_1556);
        var_1559 = wp::load(var_1557);
        var_1558 = wp::extract(var_1559, var_1513);
        var_1560 = wp::mul(var_1554, var_1558);
        var_1561 = wp::add(var_1534, var_1560);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_1563 = wp::unot(var_dsbl_damper);
        var_1562 = var_1563;
        if (var_1562) {
            var_1564 = wp::address(var_flex_damping, var_13);
            var_1567 = wp::load(var_1564);
            var_1566 = (var_1567 > var_1565);
            var_1562 = var_1562 && var_1566;
        }
        if (var_1562) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_1568 = wp::extract(var_106, var_1545, var_1513);
            var_1569 = wp::mul(var_1554, var_1568);
            var_1570 = wp::add(var_1544, var_1569);
        }
        var_1571 = wp::where(var_1562, var_1570, var_1544);
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_1574 = wp::mul(var_1573, var_5);
        var_1575 = wp::add(var_15, var_1574);
        var_1577 = wp::mul(var_1576, var_1381);
        var_1578 = wp::add(var_1575, var_1577);
        var_1579 = wp::add(var_1578, var_1572);
        var_1580 = wp::address(var_flex_bending, var_1579);
        var_1582 = wp::load(var_1580);
        var_1581 = wp::copy(var_1582);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_1583 = wp::extract(var_57, var_1572);
        var_1584 = wp::address(var_flexvert_xpos_in, var_0, var_1583);
        var_1586 = wp::load(var_1584);
        var_1585 = wp::extract(var_1586, var_1513);
        var_1587 = wp::mul(var_1581, var_1585);
        var_1588 = wp::add(var_1561, var_1587);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_1590 = wp::unot(var_dsbl_damper);
        var_1589 = var_1590;
        if (var_1589) {
            var_1591 = wp::address(var_flex_damping, var_13);
            var_1594 = wp::load(var_1591);
            var_1593 = (var_1594 > var_1592);
            var_1589 = var_1589 && var_1593;
        }
        if (var_1589) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_1595 = wp::extract(var_106, var_1572, var_1513);
            var_1596 = wp::mul(var_1581, var_1595);
            var_1597 = wp::add(var_1571, var_1596);
        }
        var_1598 = wp::where(var_1589, var_1597, var_1571);
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_1601 = wp::mul(var_1600, var_5);
        var_1602 = wp::add(var_15, var_1601);
        var_1604 = wp::mul(var_1603, var_1381);
        var_1605 = wp::add(var_1602, var_1604);
        var_1606 = wp::add(var_1605, var_1599);
        var_1607 = wp::address(var_flex_bending, var_1606);
        var_1609 = wp::load(var_1607);
        var_1608 = wp::copy(var_1609);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_1610 = wp::extract(var_57, var_1599);
        var_1611 = wp::address(var_flexvert_xpos_in, var_0, var_1610);
        var_1613 = wp::load(var_1611);
        var_1612 = wp::extract(var_1613, var_1513);
        var_1614 = wp::mul(var_1608, var_1612);
        var_1615 = wp::add(var_1588, var_1614);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_1617 = wp::unot(var_dsbl_damper);
        var_1616 = var_1617;
        if (var_1616) {
            var_1618 = wp::address(var_flex_damping, var_13);
            var_1621 = wp::load(var_1618);
            var_1620 = (var_1621 > var_1619);
            var_1616 = var_1616 && var_1620;
        }
        if (var_1616) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_1622 = wp::extract(var_106, var_1599, var_1513);
            var_1623 = wp::mul(var_1608, var_1622);
            var_1624 = wp::add(var_1598, var_1623);
        }
        var_1625 = wp::where(var_1616, var_1624, var_1598);
        // force_spring[i, x] = -(acc_spring + flex_bending[bendingadr + 17 * eid + 16] * frc[i, x])       <L 801>
        var_1627 = wp::mul(var_1626, var_5);
        var_1628 = wp::add(var_15, var_1627);
        var_1630 = wp::add(var_1628, var_1629);
        var_1631 = wp::address(var_flex_bending, var_1630);
        var_1632 = wp::extract(var_58, var_1381, var_1513);
        var_1634 = wp::load(var_1631);
        var_1633 = wp::mul(var_1634, var_1632);
        var_1635 = wp::add(var_1615, var_1633);
        var_1636 = wp::neg(var_1635);
        wp::assign_inplace(var_197, var_1381, var_1513, var_1636);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 802>
        var_1638 = wp::unot(var_dsbl_damper);
        var_1637 = var_1638;
        if (var_1637) {
            var_1639 = wp::address(var_flex_damping, var_13);
            var_1642 = wp::load(var_1639);
            var_1641 = (var_1642 > var_1640);
            var_1637 = var_1637 && var_1641;
        }
        if (var_1637) {
            // force_damper[i, x] = -acc_damper                                                   <L 803>
            var_1643 = wp::neg(var_1625);
            wp::assign_inplace(var_198, var_1381, var_1513, var_1643);
        }
        // acc_spring = float(0.0)                                                                <L 793>
        var_1646 = wp::float(var_1645);
        // acc_damper = float(0.0)                                                                <L 794>
        var_1648 = wp::float(var_1647);
        // for j in range(4):                                                                     <L 795>
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_1651 = wp::mul(var_1650, var_5);
        var_1652 = wp::add(var_15, var_1651);
        var_1654 = wp::mul(var_1653, var_1381);
        var_1655 = wp::add(var_1652, var_1654);
        var_1656 = wp::add(var_1655, var_1649);
        var_1657 = wp::address(var_flex_bending, var_1656);
        var_1659 = wp::load(var_1657);
        var_1658 = wp::copy(var_1659);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_1660 = wp::extract(var_57, var_1649);
        var_1661 = wp::address(var_flexvert_xpos_in, var_0, var_1660);
        var_1663 = wp::load(var_1661);
        var_1662 = wp::extract(var_1663, var_1644);
        var_1664 = wp::mul(var_1658, var_1662);
        var_1665 = wp::add(var_1646, var_1664);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_1667 = wp::unot(var_dsbl_damper);
        var_1666 = var_1667;
        if (var_1666) {
            var_1668 = wp::address(var_flex_damping, var_13);
            var_1671 = wp::load(var_1668);
            var_1670 = (var_1671 > var_1669);
            var_1666 = var_1666 && var_1670;
        }
        if (var_1666) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_1672 = wp::extract(var_106, var_1649, var_1644);
            var_1673 = wp::mul(var_1658, var_1672);
            var_1674 = wp::add(var_1648, var_1673);
        }
        var_1675 = wp::where(var_1666, var_1674, var_1648);
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_1678 = wp::mul(var_1677, var_5);
        var_1679 = wp::add(var_15, var_1678);
        var_1681 = wp::mul(var_1680, var_1381);
        var_1682 = wp::add(var_1679, var_1681);
        var_1683 = wp::add(var_1682, var_1676);
        var_1684 = wp::address(var_flex_bending, var_1683);
        var_1686 = wp::load(var_1684);
        var_1685 = wp::copy(var_1686);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_1687 = wp::extract(var_57, var_1676);
        var_1688 = wp::address(var_flexvert_xpos_in, var_0, var_1687);
        var_1690 = wp::load(var_1688);
        var_1689 = wp::extract(var_1690, var_1644);
        var_1691 = wp::mul(var_1685, var_1689);
        var_1692 = wp::add(var_1665, var_1691);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_1694 = wp::unot(var_dsbl_damper);
        var_1693 = var_1694;
        if (var_1693) {
            var_1695 = wp::address(var_flex_damping, var_13);
            var_1698 = wp::load(var_1695);
            var_1697 = (var_1698 > var_1696);
            var_1693 = var_1693 && var_1697;
        }
        if (var_1693) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_1699 = wp::extract(var_106, var_1676, var_1644);
            var_1700 = wp::mul(var_1685, var_1699);
            var_1701 = wp::add(var_1675, var_1700);
        }
        var_1702 = wp::where(var_1693, var_1701, var_1675);
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_1705 = wp::mul(var_1704, var_5);
        var_1706 = wp::add(var_15, var_1705);
        var_1708 = wp::mul(var_1707, var_1381);
        var_1709 = wp::add(var_1706, var_1708);
        var_1710 = wp::add(var_1709, var_1703);
        var_1711 = wp::address(var_flex_bending, var_1710);
        var_1713 = wp::load(var_1711);
        var_1712 = wp::copy(var_1713);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_1714 = wp::extract(var_57, var_1703);
        var_1715 = wp::address(var_flexvert_xpos_in, var_0, var_1714);
        var_1717 = wp::load(var_1715);
        var_1716 = wp::extract(var_1717, var_1644);
        var_1718 = wp::mul(var_1712, var_1716);
        var_1719 = wp::add(var_1692, var_1718);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_1721 = wp::unot(var_dsbl_damper);
        var_1720 = var_1721;
        if (var_1720) {
            var_1722 = wp::address(var_flex_damping, var_13);
            var_1725 = wp::load(var_1722);
            var_1724 = (var_1725 > var_1723);
            var_1720 = var_1720 && var_1724;
        }
        if (var_1720) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_1726 = wp::extract(var_106, var_1703, var_1644);
            var_1727 = wp::mul(var_1712, var_1726);
            var_1728 = wp::add(var_1702, var_1727);
        }
        var_1729 = wp::where(var_1720, var_1728, var_1702);
        // coeff = flex_bending[bendingadr + 17 * eid + 4 * i + j]                                <L 796>
        var_1732 = wp::mul(var_1731, var_5);
        var_1733 = wp::add(var_15, var_1732);
        var_1735 = wp::mul(var_1734, var_1381);
        var_1736 = wp::add(var_1733, var_1735);
        var_1737 = wp::add(var_1736, var_1730);
        var_1738 = wp::address(var_flex_bending, var_1737);
        var_1740 = wp::load(var_1738);
        var_1739 = wp::copy(var_1740);
        // acc_spring += coeff * flexvert_xpos_in[worldid, v[j]][x]                               <L 797>
        var_1741 = wp::extract(var_57, var_1730);
        var_1742 = wp::address(var_flexvert_xpos_in, var_0, var_1741);
        var_1744 = wp::load(var_1742);
        var_1743 = wp::extract(var_1744, var_1644);
        var_1745 = wp::mul(var_1739, var_1743);
        var_1746 = wp::add(var_1719, var_1745);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 798>
        var_1748 = wp::unot(var_dsbl_damper);
        var_1747 = var_1748;
        if (var_1747) {
            var_1749 = wp::address(var_flex_damping, var_13);
            var_1752 = wp::load(var_1749);
            var_1751 = (var_1752 > var_1750);
            var_1747 = var_1747 && var_1751;
        }
        if (var_1747) {
            // acc_damper += coeff * vel[j, x]                                                    <L 799>
            var_1753 = wp::extract(var_106, var_1730, var_1644);
            var_1754 = wp::mul(var_1739, var_1753);
            var_1755 = wp::add(var_1729, var_1754);
        }
        var_1756 = wp::where(var_1747, var_1755, var_1729);
        // force_spring[i, x] = -(acc_spring + flex_bending[bendingadr + 17 * eid + 16] * frc[i, x])       <L 801>
        var_1758 = wp::mul(var_1757, var_5);
        var_1759 = wp::add(var_15, var_1758);
        var_1761 = wp::add(var_1759, var_1760);
        var_1762 = wp::address(var_flex_bending, var_1761);
        var_1763 = wp::extract(var_58, var_1381, var_1644);
        var_1765 = wp::load(var_1762);
        var_1764 = wp::mul(var_1765, var_1763);
        var_1766 = wp::add(var_1746, var_1764);
        var_1767 = wp::neg(var_1766);
        wp::assign_inplace(var_197, var_1381, var_1644, var_1767);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 802>
        var_1769 = wp::unot(var_dsbl_damper);
        var_1768 = var_1769;
        if (var_1768) {
            var_1770 = wp::address(var_flex_damping, var_13);
            var_1773 = wp::load(var_1770);
            var_1772 = (var_1773 > var_1771);
            var_1768 = var_1768 && var_1772;
        }
        if (var_1768) {
            // force_damper[i, x] = -acc_damper                                                   <L 803>
            var_1774 = wp::neg(var_1756);
            wp::assign_inplace(var_198, var_1381, var_1644, var_1774);
        }
        // for i in range(4):                                                                     <L 805>
        // bodyid = flex_vertbodyid[v[i]]                                                         <L 806>
        var_1776 = wp::extract(var_57, var_1775);
        var_1777 = wp::address(var_flex_vertbodyid, var_1776);
        var_1779 = wp::load(var_1777);
        var_1778 = wp::copy(var_1779);
        // frc_s = force_spring[i]                                                                <L 807>
        var_1780 = wp::extract(var_197, var_1775);
        // node_pos = flexvert_xpos_in[worldid, v[i]]                                             <L 808>
        var_1781 = wp::extract(var_57, var_1775);
        var_1782 = wp::address(var_flexvert_xpos_in, var_0, var_1781);
        var_1784 = wp::load(var_1782);
        var_1783 = wp::copy(var_1784);
        // body_xipos = xipos_in[worldid, bodyid]                                                 <L 809>
        var_1785 = wp::address(var_xipos_in, var_0, var_1778);
        var_1787 = wp::load(var_1785);
        var_1786 = wp::copy(var_1787);
        // offset = body_xipos - node_pos                                                         <L 810>
        var_1788 = wp::sub(var_1786, var_1783);
        // spatial_frc_s = wp.spatial_vector(frc_s, -wp.cross(offset, frc_s))                     <L 812>
        var_1789 = wp::cross(var_1788, var_1780);
        var_1790 = wp::neg(var_1789);
        var_1791 = wp::vec_t<6, wp::float32>(var_1780, var_1790);
        // wp.atomic_add(flex_spring_body_force_out, worldid, bodyid, spatial_frc_s)              <L 813>
        var_1792 = wp::atomic_add(var_flex_spring_body_force_out, var_0, var_1778, var_1791);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 815>
        var_1794 = wp::unot(var_dsbl_damper);
        var_1793 = var_1794;
        if (var_1793) {
            var_1795 = wp::address(var_flex_damping, var_13);
            var_1798 = wp::load(var_1795);
            var_1797 = (var_1798 > var_1796);
            var_1793 = var_1793 && var_1797;
        }
        if (var_1793) {
            // frc_d = force_damper[i] * flex_damping[f]                                          <L 816>
            var_1799 = wp::extract(var_198, var_1775);
            var_1800 = wp::address(var_flex_damping, var_13);
            var_1802 = wp::load(var_1800);
            var_1801 = wp::mul(var_1799, var_1802);
            // spatial_frc_d = wp.spatial_vector(frc_d, -wp.cross(offset, frc_d))                 <L 817>
            var_1803 = wp::cross(var_1788, var_1801);
            var_1804 = wp::neg(var_1803);
            var_1805 = wp::vec_t<6, wp::float32>(var_1801, var_1804);
            // wp.atomic_add(flex_damper_body_force_out, worldid, bodyid, spatial_frc_d)          <L 818>
            var_1806 = wp::atomic_add(var_flex_damper_body_force_out, var_0, var_1778, var_1805);
        }
        // bodyid = flex_vertbodyid[v[i]]                                                         <L 806>
        var_1808 = wp::extract(var_57, var_1807);
        var_1809 = wp::address(var_flex_vertbodyid, var_1808);
        var_1811 = wp::load(var_1809);
        var_1810 = wp::copy(var_1811);
        // frc_s = force_spring[i]                                                                <L 807>
        var_1812 = wp::extract(var_197, var_1807);
        // node_pos = flexvert_xpos_in[worldid, v[i]]                                             <L 808>
        var_1813 = wp::extract(var_57, var_1807);
        var_1814 = wp::address(var_flexvert_xpos_in, var_0, var_1813);
        var_1816 = wp::load(var_1814);
        var_1815 = wp::copy(var_1816);
        // body_xipos = xipos_in[worldid, bodyid]                                                 <L 809>
        var_1817 = wp::address(var_xipos_in, var_0, var_1810);
        var_1819 = wp::load(var_1817);
        var_1818 = wp::copy(var_1819);
        // offset = body_xipos - node_pos                                                         <L 810>
        var_1820 = wp::sub(var_1818, var_1815);
        // spatial_frc_s = wp.spatial_vector(frc_s, -wp.cross(offset, frc_s))                     <L 812>
        var_1821 = wp::cross(var_1820, var_1812);
        var_1822 = wp::neg(var_1821);
        var_1823 = wp::vec_t<6, wp::float32>(var_1812, var_1822);
        // wp.atomic_add(flex_spring_body_force_out, worldid, bodyid, spatial_frc_s)              <L 813>
        var_1824 = wp::atomic_add(var_flex_spring_body_force_out, var_0, var_1810, var_1823);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 815>
        var_1826 = wp::unot(var_dsbl_damper);
        var_1825 = var_1826;
        if (var_1825) {
            var_1827 = wp::address(var_flex_damping, var_13);
            var_1830 = wp::load(var_1827);
            var_1829 = (var_1830 > var_1828);
            var_1825 = var_1825 && var_1829;
        }
        if (var_1825) {
            // frc_d = force_damper[i] * flex_damping[f]                                          <L 816>
            var_1831 = wp::extract(var_198, var_1807);
            var_1832 = wp::address(var_flex_damping, var_13);
            var_1834 = wp::load(var_1832);
            var_1833 = wp::mul(var_1831, var_1834);
            // spatial_frc_d = wp.spatial_vector(frc_d, -wp.cross(offset, frc_d))                 <L 817>
            var_1835 = wp::cross(var_1820, var_1833);
            var_1836 = wp::neg(var_1835);
            var_1837 = wp::vec_t<6, wp::float32>(var_1833, var_1836);
            // wp.atomic_add(flex_damper_body_force_out, worldid, bodyid, spatial_frc_d)          <L 818>
            var_1838 = wp::atomic_add(var_flex_damper_body_force_out, var_0, var_1810, var_1837);
        }
        var_1839 = wp::where(var_1825, var_1833, var_1801);
        var_1840 = wp::where(var_1825, var_1837, var_1805);
        // bodyid = flex_vertbodyid[v[i]]                                                         <L 806>
        var_1842 = wp::extract(var_57, var_1841);
        var_1843 = wp::address(var_flex_vertbodyid, var_1842);
        var_1845 = wp::load(var_1843);
        var_1844 = wp::copy(var_1845);
        // frc_s = force_spring[i]                                                                <L 807>
        var_1846 = wp::extract(var_197, var_1841);
        // node_pos = flexvert_xpos_in[worldid, v[i]]                                             <L 808>
        var_1847 = wp::extract(var_57, var_1841);
        var_1848 = wp::address(var_flexvert_xpos_in, var_0, var_1847);
        var_1850 = wp::load(var_1848);
        var_1849 = wp::copy(var_1850);
        // body_xipos = xipos_in[worldid, bodyid]                                                 <L 809>
        var_1851 = wp::address(var_xipos_in, var_0, var_1844);
        var_1853 = wp::load(var_1851);
        var_1852 = wp::copy(var_1853);
        // offset = body_xipos - node_pos                                                         <L 810>
        var_1854 = wp::sub(var_1852, var_1849);
        // spatial_frc_s = wp.spatial_vector(frc_s, -wp.cross(offset, frc_s))                     <L 812>
        var_1855 = wp::cross(var_1854, var_1846);
        var_1856 = wp::neg(var_1855);
        var_1857 = wp::vec_t<6, wp::float32>(var_1846, var_1856);
        // wp.atomic_add(flex_spring_body_force_out, worldid, bodyid, spatial_frc_s)              <L 813>
        var_1858 = wp::atomic_add(var_flex_spring_body_force_out, var_0, var_1844, var_1857);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 815>
        var_1860 = wp::unot(var_dsbl_damper);
        var_1859 = var_1860;
        if (var_1859) {
            var_1861 = wp::address(var_flex_damping, var_13);
            var_1864 = wp::load(var_1861);
            var_1863 = (var_1864 > var_1862);
            var_1859 = var_1859 && var_1863;
        }
        if (var_1859) {
            // frc_d = force_damper[i] * flex_damping[f]                                          <L 816>
            var_1865 = wp::extract(var_198, var_1841);
            var_1866 = wp::address(var_flex_damping, var_13);
            var_1868 = wp::load(var_1866);
            var_1867 = wp::mul(var_1865, var_1868);
            // spatial_frc_d = wp.spatial_vector(frc_d, -wp.cross(offset, frc_d))                 <L 817>
            var_1869 = wp::cross(var_1854, var_1867);
            var_1870 = wp::neg(var_1869);
            var_1871 = wp::vec_t<6, wp::float32>(var_1867, var_1870);
            // wp.atomic_add(flex_damper_body_force_out, worldid, bodyid, spatial_frc_d)          <L 818>
            var_1872 = wp::atomic_add(var_flex_damper_body_force_out, var_0, var_1844, var_1871);
        }
        var_1873 = wp::where(var_1859, var_1867, var_1839);
        var_1874 = wp::where(var_1859, var_1871, var_1840);
        // bodyid = flex_vertbodyid[v[i]]                                                         <L 806>
        var_1876 = wp::extract(var_57, var_1875);
        var_1877 = wp::address(var_flex_vertbodyid, var_1876);
        var_1879 = wp::load(var_1877);
        var_1878 = wp::copy(var_1879);
        // frc_s = force_spring[i]                                                                <L 807>
        var_1880 = wp::extract(var_197, var_1875);
        // node_pos = flexvert_xpos_in[worldid, v[i]]                                             <L 808>
        var_1881 = wp::extract(var_57, var_1875);
        var_1882 = wp::address(var_flexvert_xpos_in, var_0, var_1881);
        var_1884 = wp::load(var_1882);
        var_1883 = wp::copy(var_1884);
        // body_xipos = xipos_in[worldid, bodyid]                                                 <L 809>
        var_1885 = wp::address(var_xipos_in, var_0, var_1878);
        var_1887 = wp::load(var_1885);
        var_1886 = wp::copy(var_1887);
        // offset = body_xipos - node_pos                                                         <L 810>
        var_1888 = wp::sub(var_1886, var_1883);
        // spatial_frc_s = wp.spatial_vector(frc_s, -wp.cross(offset, frc_s))                     <L 812>
        var_1889 = wp::cross(var_1888, var_1880);
        var_1890 = wp::neg(var_1889);
        var_1891 = wp::vec_t<6, wp::float32>(var_1880, var_1890);
        // wp.atomic_add(flex_spring_body_force_out, worldid, bodyid, spatial_frc_s)              <L 813>
        var_1892 = wp::atomic_add(var_flex_spring_body_force_out, var_0, var_1878, var_1891);
        // if not dsbl_damper and flex_damping[f] > 0.0:                                          <L 815>
        var_1894 = wp::unot(var_dsbl_damper);
        var_1893 = var_1894;
        if (var_1893) {
            var_1895 = wp::address(var_flex_damping, var_13);
            var_1898 = wp::load(var_1895);
            var_1897 = (var_1898 > var_1896);
            var_1893 = var_1893 && var_1897;
        }
        if (var_1893) {
            // frc_d = force_damper[i] * flex_damping[f]                                          <L 816>
            var_1899 = wp::extract(var_198, var_1875);
            var_1900 = wp::address(var_flex_damping, var_13);
            var_1902 = wp::load(var_1900);
            var_1901 = wp::mul(var_1899, var_1902);
            // spatial_frc_d = wp.spatial_vector(frc_d, -wp.cross(offset, frc_d))                 <L 817>
            var_1903 = wp::cross(var_1888, var_1901);
            var_1904 = wp::neg(var_1903);
            var_1905 = wp::vec_t<6, wp::float32>(var_1901, var_1904);
            // wp.atomic_add(flex_damper_body_force_out, worldid, bodyid, spatial_frc_d)          <L 818>
            var_1906 = wp::atomic_add(var_flex_damper_body_force_out, var_0, var_1878, var_1905);
        }
        var_1907 = wp::where(var_1893, var_1901, var_1873);
        var_1908 = wp::where(var_1893, var_1905, var_1874);
    }
}



extern "C" __global__ void _qfrc_passive_ccf6028a_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_jnt_actgravcomp,
    wp::array_t<wp::int32> var_dof_jntid,
    bool var_has_fluid,
    wp::array_t<wp::float32> var_qfrc_spring_in,
    wp::array_t<wp::float32> var_qfrc_damper_in,
    wp::array_t<wp::float32> var_qfrc_gravcomp_in,
    wp::array_t<wp::float32> var_qfrc_fluid_in,
    bool var_gravity_enabled,
    wp::array_t<wp::float32> var_qfrc_passive_out)
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
        wp::float32* var_5;
        wp::float32 var_6;
        wp::float32 var_7;
        bool var_8;
        wp::int32* var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        bool var_12;
        wp::int32 var_13;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        wp::float32 var_17;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        //---------
        // forward
        // def _qfrc_passive(                                                                     <L 562>
        // worldid, dofid = wp.tid()                                                              <L 577>
        builtin_tid2d(var_0, var_1);
        // qfrc_passive = qfrc_spring_in[worldid, dofid]                                          <L 578>
        var_2 = wp::address(var_qfrc_spring_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // qfrc_passive += qfrc_damper_in[worldid, dofid]                                         <L 579>
        var_5 = wp::address(var_qfrc_damper_in, var_0, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::add(var_3, var_7);
        // if gravity_enabled and not jnt_actgravcomp[dof_jntid[dofid]]:                          <L 582>
        var_8 = var_gravity_enabled;
        if (var_8) {
            var_9 = wp::address(var_dof_jntid, var_1);
            var_11 = wp::load(var_9);
            var_10 = wp::address(var_jnt_actgravcomp, var_11);
            var_13 = wp::load(var_10);
            var_12 = wp::unot(var_13);
            var_8 = var_8 && var_12;
        }
        if (var_8) {
            // qfrc_passive += qfrc_gravcomp_in[worldid, dofid]                                   <L 583>
            var_14 = wp::address(var_qfrc_gravcomp_in, var_0, var_1);
            var_16 = wp::load(var_14);
            var_15 = wp::add(var_6, var_16);
        }
        var_17 = wp::where(var_8, var_15, var_6);
        // if has_fluid:                                                                          <L 586>
        if (var_has_fluid) {
            // qfrc_passive += qfrc_fluid_in[worldid, dofid]                                      <L 587>
            var_18 = wp::address(var_qfrc_fluid_in, var_0, var_1);
            var_20 = wp::load(var_18);
            var_19 = wp::add(var_17, var_20);
        }
        var_21 = wp::where(var_has_fluid, var_19, var_17);
        // qfrc_passive_out[worldid, dofid] = qfrc_passive                                        <L 589>
        wp::array_store(var_qfrc_passive_out, var_0, var_1, var_21);
    }
}



extern "C" __global__ void _spring_damper_tendon_passive_d76f76d8_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::float32> var_tendon_stiffness,
    wp::array_t<wp::vec_t<2, wp::float32>> var_tendon_stiffnesspoly,
    wp::array_t<wp::float32> var_tendon_damping,
    wp::array_t<wp::vec_t<2, wp::float32>> var_tendon_dampingpoly,
    wp::array_t<wp::vec_t<2, wp::float32>> var_tendon_lengthspring,
    wp::array_t<wp::float32> var_ten_J_in,
    wp::array_t<wp::float32> var_ten_length_in,
    wp::array_t<wp::float32> var_ten_velocity_in,
    bool var_dsbl_spring,
    bool var_dsbl_damper,
    wp::array_t<wp::float32> var_qfrc_spring_out,
    wp::array_t<wp::float32> var_qfrc_damper_out)
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
        wp::shape_t* var_3;
        const wp::int32 var_4 = 0;
        wp::int32 var_5;
        wp::shape_t var_6;
        wp::int32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::shape_t* var_11;
        const wp::int32 var_12 = 0;
        wp::int32 var_13;
        wp::shape_t var_14;
        wp::int32 var_15;
        wp::vec_t<2, wp::float32>* var_16;
        wp::vec_t<2, wp::float32> var_17;
        wp::vec_t<2, wp::float32> var_18;
        wp::shape_t* var_19;
        const wp::int32 var_20 = 0;
        wp::int32 var_21;
        wp::shape_t var_22;
        wp::int32 var_23;
        wp::float32* var_24;
        wp::float32 var_25;
        wp::float32 var_26;
        wp::shape_t* var_27;
        const wp::int32 var_28 = 0;
        wp::int32 var_29;
        wp::shape_t var_30;
        wp::int32 var_31;
        wp::vec_t<2, wp::float32>* var_32;
        wp::vec_t<2, wp::float32> var_33;
        wp::vec_t<2, wp::float32> var_34;
        bool var_35;
        bool var_36;
        const wp::float32 var_37 = 0.0;
        bool var_38;
        const wp::int32 var_39 = 0;
        wp::float32 var_40;
        const wp::float32 var_41 = 0.0;
        bool var_42;
        const wp::int32 var_43 = 1;
        wp::float32 var_44;
        const wp::float32 var_45 = 0.0;
        bool var_46;
        bool var_47;
        bool var_48;
        bool var_49;
        const wp::float32 var_50 = 0.0;
        bool var_51;
        const wp::int32 var_52 = 0;
        wp::float32 var_53;
        const wp::float32 var_54 = 0.0;
        bool var_55;
        const wp::int32 var_56 = 1;
        wp::float32 var_57;
        const wp::float32 var_58 = 0.0;
        bool var_59;
        bool var_60;
        bool var_61;
        bool var_62;
        bool var_63;
        wp::int32* var_64;
        wp::int32 var_65;
        wp::int32 var_66;
        bool var_67;
        wp::int32* var_68;
        wp::int32 var_69;
        wp::int32 var_70;
        wp::int32 var_71;
        wp::float32* var_72;
        wp::float32 var_73;
        wp::float32 var_74;
        wp::int32* var_75;
        wp::int32 var_76;
        wp::int32 var_77;
        wp::float32* var_78;
        wp::float32 var_79;
        wp::float32 var_80;
        wp::shape_t* var_81;
        const wp::int32 var_82 = 0;
        wp::int32 var_83;
        wp::shape_t var_84;
        wp::int32 var_85;
        wp::vec_t<2, wp::float32>* var_86;
        wp::vec_t<2, wp::float32> var_87;
        wp::vec_t<2, wp::float32> var_88;
        const wp::int32 var_89 = 0;
        wp::float32 var_90;
        const wp::int32 var_91 = 1;
        wp::float32 var_92;
        bool var_93;
        wp::float32 var_94;
        bool var_95;
        wp::float32 var_96;
        const wp::float32 var_97 = 0.0;
        wp::float32 var_98;
        wp::float32 var_99;
        wp::float32 var_100;
        const wp::int32 var_101 = 0;
        wp::float32 var_102;
        wp::float32 var_103;
        wp::slice_t var_104;
        const wp::int32 var_105 = 0;
        wp::array_t<wp::float32> var_106;
        wp::float32 var_107;
        wp::float32 var_108;
        wp::float32* var_109;
        wp::float32 var_110;
        wp::float32 var_111;
        wp::float32 var_112;
        const wp::int32 var_113 = 1;
        wp::float32 var_114;
        wp::float32 var_115;
        wp::slice_t var_116;
        const wp::int32 var_117 = 0;
        wp::array_t<wp::float32> var_118;
        wp::float32 var_119;
        wp::float32 var_120;
        //---------
        // forward
        // def _spring_damper_tendon_passive(                                                     <L 209>
        // worldid, tenid, dofid_sparse = wp.tid()                                                <L 230>
        builtin_tid3d(var_0, var_1, var_2);
        // stiffness = tendon_stiffness[worldid % tendon_stiffness.shape[0], tenid]               <L 232>
        var_3 = &(var_tendon_stiffness.shape);
        var_6 = wp::load(var_3);
        var_5 = wp::extract(var_6, var_4);
        var_7 = wp::mod(var_0, var_5);
        var_8 = wp::address(var_tendon_stiffness, var_7, var_1);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // spoly = tendon_stiffnesspoly[worldid % tendon_stiffnesspoly.shape[0], tenid]           <L 233>
        var_11 = &(var_tendon_stiffnesspoly.shape);
        var_14 = wp::load(var_11);
        var_13 = wp::extract(var_14, var_12);
        var_15 = wp::mod(var_0, var_13);
        var_16 = wp::address(var_tendon_stiffnesspoly, var_15, var_1);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // damping = tendon_damping[worldid % tendon_damping.shape[0], tenid]                     <L 234>
        var_19 = &(var_tendon_damping.shape);
        var_22 = wp::load(var_19);
        var_21 = wp::extract(var_22, var_20);
        var_23 = wp::mod(var_0, var_21);
        var_24 = wp::address(var_tendon_damping, var_23, var_1);
        var_26 = wp::load(var_24);
        var_25 = wp::copy(var_26);
        // dpoly = tendon_dampingpoly[worldid % tendon_dampingpoly.shape[0], tenid]               <L 235>
        var_27 = &(var_tendon_dampingpoly.shape);
        var_30 = wp::load(var_27);
        var_29 = wp::extract(var_30, var_28);
        var_31 = wp::mod(var_0, var_29);
        var_32 = wp::address(var_tendon_dampingpoly, var_31, var_1);
        var_34 = wp::load(var_32);
        var_33 = wp::copy(var_34);
        // has_stiffness = (stiffness != 0.0 or spoly[0] != 0.0 or spoly[1] != 0.0) and not dsbl_spring       <L 237>
        var_38 = (var_9 != var_37);
        var_36 = var_38;
        if (!var_36) {
            var_40 = wp::extract(var_17, var_39);
            var_42 = (var_40 != var_41);
            var_36 = var_36 || var_42;
        }
        if (!var_36) {
            var_44 = wp::extract(var_17, var_43);
            var_46 = (var_44 != var_45);
            var_36 = var_36 || var_46;
        }
        var_35 = var_36;
        if (var_35) {
            var_47 = wp::unot(var_dsbl_spring);
            var_35 = var_35 && var_47;
        }
        // has_damping = (damping != 0.0 or dpoly[0] != 0.0 or dpoly[1] != 0.0) and not dsbl_damper       <L 238>
        var_51 = (var_25 != var_50);
        var_49 = var_51;
        if (!var_49) {
            var_53 = wp::extract(var_33, var_52);
            var_55 = (var_53 != var_54);
            var_49 = var_49 || var_55;
        }
        if (!var_49) {
            var_57 = wp::extract(var_33, var_56);
            var_59 = (var_57 != var_58);
            var_49 = var_49 || var_59;
        }
        var_48 = var_49;
        if (var_48) {
            var_60 = wp::unot(var_dsbl_damper);
            var_48 = var_48 && var_60;
        }
        // if not has_stiffness and not has_damping:                                              <L 240>
        var_62 = wp::unot(var_35);
        var_61 = var_62;
        if (var_61) {
            var_63 = wp::unot(var_48);
            var_61 = var_61 && var_63;
        }
        if (var_61) {
            // return                                                                             <L 241>
            continue;
        }
        // rownnz = ten_J_rownnz[tenid]                                                           <L 243>
        var_64 = wp::address(var_ten_J_rownnz, var_1);
        var_66 = wp::load(var_64);
        var_65 = wp::copy(var_66);
        // if dofid_sparse >= rownnz:                                                             <L 244>
        var_67 = (var_2 >= var_65);
        if (var_67) {
            // return                                                                             <L 245>
            continue;
        }
        // rowadr = ten_J_rowadr[tenid]                                                           <L 246>
        var_68 = wp::address(var_ten_J_rowadr, var_1);
        var_70 = wp::load(var_68);
        var_69 = wp::copy(var_70);
        // sparseid = rowadr + dofid_sparse                                                       <L 247>
        var_71 = wp::add(var_69, var_2);
        // J = ten_J_in[worldid, sparseid]                                                        <L 248>
        var_72 = wp::address(var_ten_J_in, var_0, var_71);
        var_74 = wp::load(var_72);
        var_73 = wp::copy(var_74);
        // dofid = ten_J_colind[sparseid]                                                         <L 249>
        var_75 = wp::address(var_ten_J_colind, var_71);
        var_77 = wp::load(var_75);
        var_76 = wp::copy(var_77);
        // if has_stiffness:                                                                      <L 251>
        if (var_35) {
            // length = ten_length_in[worldid, tenid]                                             <L 253>
            var_78 = wp::address(var_ten_length_in, var_0, var_1);
            var_80 = wp::load(var_78);
            var_79 = wp::copy(var_80);
            // lengthspring = tendon_lengthspring[worldid % tendon_lengthspring.shape[0], tenid]       <L 254>
            var_81 = &(var_tendon_lengthspring.shape);
            var_84 = wp::load(var_81);
            var_83 = wp::extract(var_84, var_82);
            var_85 = wp::mod(var_0, var_83);
            var_86 = wp::address(var_tendon_lengthspring, var_85, var_1);
            var_88 = wp::load(var_86);
            var_87 = wp::copy(var_88);
            // lower = lengthspring[0]                                                            <L 255>
            var_90 = wp::extract(var_87, var_89);
            // upper = lengthspring[1]                                                            <L 256>
            var_92 = wp::extract(var_87, var_91);
            // x = wp.where(length > upper, length - upper, wp.where(length < lower, length - lower, 0.0))       <L 258>
            var_93 = (var_79 > var_92);
            var_94 = wp::sub(var_79, var_92);
            var_95 = (var_79 < var_90);
            var_96 = wp::sub(var_79, var_90);
            var_98 = wp::where(var_95, var_96, var_97);
            var_99 = wp::where(var_93, var_94, var_98);
            // frc_spring = -x * util_misc._poly_force(stiffness, spoly, x, 0)                    <L 259>
            var_100 = wp::neg(var_99);
            var_102 = _poly_force_0(var_9, var_17, var_99, var_101);
            var_103 = wp::mul(var_100, var_102);
            // wp.atomic_add(qfrc_spring_out[worldid], dofid, J * frc_spring)                     <L 262>
            var_104 = wp::slice_t(var_0, var_0, var_105);
            var_106 = wp::view(var_qfrc_spring_out, var_104);
            var_107 = wp::mul(var_73, var_103);
            var_108 = wp::atomic_add(var_106, var_76, var_107);
        }
        // if has_damping:                                                                        <L 264>
        if (var_48) {
            // v = ten_velocity_in[worldid, tenid]                                                <L 266>
            var_109 = wp::address(var_ten_velocity_in, var_0, var_1);
            var_111 = wp::load(var_109);
            var_110 = wp::copy(var_111);
            // frc_damper = -v * util_misc._poly_force(damping, dpoly, v, 1)                      <L 267>
            var_112 = wp::neg(var_110);
            var_114 = _poly_force_0(var_25, var_33, var_110, var_113);
            var_115 = wp::mul(var_112, var_114);
            // wp.atomic_add(qfrc_damper_out[worldid], dofid, J * frc_damper)                     <L 270>
            var_116 = wp::slice_t(var_0, var_0, var_117);
            var_118 = wp::view(var_qfrc_damper_out, var_116);
            var_119 = wp::mul(var_73, var_115);
            var_120 = wp::atomic_add(var_118, var_76, var_119);
        }
    }
}



extern "C" __global__ void _flex_passive_bend_interp_6fd16e7b_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_nflex,
    wp::array_t<wp::int32> var_flex_interp,
    wp::array_t<wp::vec_t<3, wp::int32>> var_flex_cellnum,
    wp::array_t<wp::int32> var_flex_nodeadr,
    wp::array_t<wp::int32> var_flex_nodenum,
    wp::array_t<wp::int32> var_flex_bendingadr,
    wp::array_t<wp::int32> var_flex_nodebodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flex_node,
    wp::array_t<wp::float32> var_flex_bending,
    wp::array_t<bool> var_flex_centered,
    wp::array_t<wp::int32> var_flex_faceadr,
    wp::array_t<wp::vec_t<2, wp::int32>> var_flex_bend_interp_map,
    wp::array_t<wp::int32> var_flex_face,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexnode_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_face_xpos_in,
    wp::array_t<wp::quat_t<wp::float32>> var_face_quat_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_flex_spring_body_force_out)
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
        wp::vec_t<2, wp::int32>* var_2;
        wp::vec_t<2, wp::int32> var_3;
        wp::vec_t<2, wp::int32> var_4;
        const wp::int32 var_5 = 0;
        wp::int32 var_6;
        const wp::int32 var_7 = 1;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::vec_t<3, wp::int32>* var_16;
        wp::vec_t<3, wp::int32> var_17;
        wp::vec_t<3, wp::int32> var_18;
        const wp::int32 var_19 = 0;
        wp::int32 var_20;
        const wp::int32 var_21 = 1;
        wp::int32 var_22;
        const wp::int32 var_23 = 2;
        wp::int32 var_24;
        wp::int32* var_25;
        wp::int32 var_26;
        wp::int32 var_27;
        const wp::int32 var_28 = 1;
        wp::int32 var_29;
        const wp::int32 var_30 = 10;
        wp::int32 var_31;
        wp::int32 var_32;
        const wp::int32 var_33 = 0;
        wp::int32 var_34;
        wp::float32* var_35;
        wp::int32 var_36;
        wp::float32 var_37;
        const wp::int32 var_38 = 1;
        wp::int32 var_39;
        wp::float32* var_40;
        wp::int32 var_41;
        wp::float32 var_42;
        const wp::int32 var_43 = 2;
        wp::int32 var_44;
        wp::float32* var_45;
        const wp::int32 var_46 = 3;
        wp::int32 var_47;
        wp::float32* var_48;
        wp::vec_t<2, wp::float32> var_49;
        wp::float32 var_50;
        wp::float32 var_51;
        const wp::int32 var_52 = 4;
        wp::int32 var_53;
        wp::float32* var_54;
        const wp::int32 var_55 = 5;
        wp::int32 var_56;
        wp::float32* var_57;
        wp::vec_t<2, wp::float32> var_58;
        wp::float32 var_59;
        wp::float32 var_60;
        const wp::int32 var_61 = 6;
        wp::int32 var_62;
        wp::float32* var_63;
        wp::float32 var_64;
        wp::float32 var_65;
        const wp::int32 var_66 = 7;
        wp::int32 var_67;
        wp::float32* var_68;
        const wp::int32 var_69 = 8;
        wp::int32 var_70;
        wp::float32* var_71;
        const wp::int32 var_72 = 9;
        wp::int32 var_73;
        wp::float32* var_74;
        wp::vec_t<3, wp::float32> var_75;
        wp::float32 var_76;
        wp::float32 var_77;
        wp::float32 var_78;
        const wp::float32 var_79 = 0.0;
        bool var_80;
        wp::int32* var_81;
        wp::int32 var_82;
        wp::int32 var_83;
        wp::int32* var_84;
        wp::int32 var_85;
        wp::int32 var_86;
        wp::quat_t<wp::float32>* var_87;
        wp::quat_t<wp::float32> var_88;
        wp::quat_t<wp::float32> var_89;
        wp::quat_t<wp::float32>* var_90;
        wp::quat_t<wp::float32> var_91;
        wp::quat_t<wp::float32> var_92;
        const wp::float32 var_93 = 0.0;
        wp::vec_t<3, wp::float32> var_94;
        const wp::float32 var_95 = 0.0;
        wp::vec_t<3, wp::float32> var_96;
        const wp::float32 var_97 = 0.0;
        wp::vec_t<3, wp::float32> var_98;
        const wp::float32 var_99 = 0.0;
        wp::vec_t<3, wp::float32> var_100;
        const wp::int32 var_101 = 0;
        wp::int32 var_102;
        const wp::int32 var_103 = 3;
        wp::range_t var_104;
        wp::int32 var_105;
        bool var_106;
        const wp::int32 var_107 = 3;
        wp::range_t var_108;
        wp::int32 var_109;
        bool var_110;
        wp::vec_t<3, wp::float32>* var_111;
        wp::vec_t<3, wp::float32> var_112;
        wp::vec_t<3, wp::float32> var_113;
        wp::vec_t<3, wp::float32>* var_114;
        wp::vec_t<3, wp::float32> var_115;
        wp::vec_t<3, wp::float32> var_116;
        const wp::int32 var_117 = 0;
        wp::float32 var_118;
        wp::float32 var_119;
        const wp::int32 var_120 = 1;
        wp::float32 var_121;
        wp::float32 var_122;
        wp::float32 var_123;
        const wp::int32 var_124 = 0;
        wp::float32 var_125;
        wp::float32 var_126;
        const wp::int32 var_127 = 1;
        wp::float32 var_128;
        wp::float32 var_129;
        wp::float32 var_130;
        const wp::int32 var_131 = 0;
        wp::float32 var_132;
        wp::float32 var_133;
        const wp::int32 var_134 = 1;
        wp::float32 var_135;
        wp::float32 var_136;
        wp::float32 var_137;
        const wp::int32 var_138 = 0;
        wp::float32 var_139;
        wp::float32 var_140;
        const wp::int32 var_141 = 1;
        wp::float32 var_142;
        wp::float32 var_143;
        wp::float32 var_144;
        wp::vec_t<3, wp::float32> var_145;
        wp::vec_t<3, wp::float32> var_146;
        wp::vec_t<3, wp::float32> var_147;
        wp::vec_t<3, wp::float32> var_148;
        wp::vec_t<3, wp::float32> var_149;
        wp::vec_t<3, wp::float32> var_150;
        wp::vec_t<3, wp::float32> var_151;
        wp::vec_t<3, wp::float32> var_152;
        const wp::int32 var_153 = 1;
        wp::int32 var_154;
        wp::vec_t<3, wp::float32> var_155;
        wp::vec_t<3, wp::float32> var_156;
        wp::float32 var_157;
        wp::float32 var_158;
        bool var_159;
        const wp::float32 var_160 = 1e-15;
        bool var_161;
        bool var_162;
        const wp::float32 var_163 = 1.0;
        wp::float32 var_164;
        const wp::float32 var_165 = 1.0;
        wp::float32 var_166;
        wp::vec_t<3, wp::float32> var_167;
        wp::vec_t<3, wp::float32> var_168;
        wp::float32 var_169;
        const wp::float32 var_170 = 0.0;
        bool var_171;
        wp::quat_t<wp::float32> var_172;
        wp::quat_t<wp::float32> var_173;
        wp::quat_t<wp::float32> var_174;
        wp::quat_t<wp::float32> var_175;
        wp::vec_t<3, wp::float32> var_176;
        wp::vec_t<3, wp::float32> var_177;
        wp::vec_t<3, wp::float32> var_178;
        wp::float32 var_179;
        wp::vec_t<3, wp::float32> var_180;
        wp::vec_t<3, wp::float32> var_181;
        wp::vec_t<3, wp::float32> var_182;
        wp::float32 var_183;
        wp::vec_t<3, wp::float32> var_184;
        wp::vec_t<3, wp::float32> var_185;
        wp::vec_t<3, wp::float32> var_186;
        wp::vec_t<3, wp::float32> var_187;
        wp::vec_t<3, wp::float32> var_188;
        wp::vec_t<3, wp::float32> var_189;
        wp::vec_t<3, wp::float32> var_190;
        wp::float32 var_191;
        //---------
        // forward
        // def _flex_passive_bend_interp(                                                         <L 1024>
        // worldid, bend_edge_id = wp.tid()                                                       <L 1047>
        builtin_tid2d(var_0, var_1);
        // mapping = flex_bend_interp_map[bend_edge_id]                                           <L 1049>
        var_2 = wp::address(var_flex_bend_interp_map, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // f = mapping[0]                                                                         <L 1050>
        var_6 = wp::extract(var_3, var_5);
        // e = mapping[1]                                                                         <L 1051>
        var_8 = wp::extract(var_3, var_7);
        // order = flex_interp[f]                                                                 <L 1053>
        var_9 = wp::address(var_flex_interp, var_6);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // order_abs = -order                                                                     <L 1054>
        var_12 = wp::neg(var_10);
        // bendingadr = flex_bendingadr[f]                                                        <L 1055>
        var_13 = wp::address(var_flex_bendingadr, var_6);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // cellnum = flex_cellnum[f]                                                              <L 1057>
        var_16 = wp::address(var_flex_cellnum, var_6);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // cx = cellnum[0]                                                                        <L 1058>
        var_20 = wp::extract(var_17, var_19);
        // cy = cellnum[1]                                                                        <L 1059>
        var_22 = wp::extract(var_17, var_21);
        // cz = cellnum[2]                                                                        <L 1060>
        var_24 = wp::extract(var_17, var_23);
        // nstart = flex_nodeadr[f]                                                               <L 1061>
        var_25 = wp::address(var_flex_nodeadr, var_6);
        var_27 = wp::load(var_25);
        var_26 = wp::copy(var_27);
        // edata_base = bendingadr + 1 + e * 10                                                   <L 1063>
        var_29 = wp::add(var_14, var_28);
        var_31 = wp::mul(var_8, var_30);
        var_32 = wp::add(var_29, var_31);
        // fe_A = int(flex_bending[edata_base + 0])                                               <L 1064>
        var_34 = wp::add(var_32, var_33);
        var_35 = wp::address(var_flex_bending, var_34);
        var_37 = wp::load(var_35);
        var_36 = wp::int(var_37);
        // fe_B = int(flex_bending[edata_base + 1])                                               <L 1065>
        var_39 = wp::add(var_32, var_38);
        var_40 = wp::address(var_flex_bending, var_39);
        var_42 = wp::load(var_40);
        var_41 = wp::int(var_42);
        // local_A = wp.vec2(flex_bending[edata_base + 2], flex_bending[edata_base + 3])          <L 1066>
        var_44 = wp::add(var_32, var_43);
        var_45 = wp::address(var_flex_bending, var_44);
        var_47 = wp::add(var_32, var_46);
        var_48 = wp::address(var_flex_bending, var_47);
        var_50 = wp::load(var_45);
        var_51 = wp::load(var_48);
        var_49 = wp::vec_t<2, wp::float32>(var_50, var_51);
        // local_B = wp.vec2(flex_bending[edata_base + 4], flex_bending[edata_base + 5])          <L 1067>
        var_53 = wp::add(var_32, var_52);
        var_54 = wp::address(var_flex_bending, var_53);
        var_56 = wp::add(var_32, var_55);
        var_57 = wp::address(var_flex_bending, var_56);
        var_59 = wp::load(var_54);
        var_60 = wp::load(var_57);
        var_58 = wp::vec_t<2, wp::float32>(var_59, var_60);
        // stiffness = flex_bending[edata_base + 6]                                               <L 1068>
        var_62 = wp::add(var_32, var_61);
        var_63 = wp::address(var_flex_bending, var_62);
        var_65 = wp::load(var_63);
        var_64 = wp::copy(var_65);
        // dn0 = wp.vec3(flex_bending[edata_base + 7], flex_bending[edata_base + 8], flex_bending[edata_base + 9])       <L 1069>
        var_67 = wp::add(var_32, var_66);
        var_68 = wp::address(var_flex_bending, var_67);
        var_70 = wp::add(var_32, var_69);
        var_71 = wp::address(var_flex_bending, var_70);
        var_73 = wp::add(var_32, var_72);
        var_74 = wp::address(var_flex_bending, var_73);
        var_76 = wp::load(var_68);
        var_77 = wp::load(var_71);
        var_78 = wp::load(var_74);
        var_75 = wp::vec_t<3, wp::float32>(var_76, var_77, var_78);
        // if stiffness <= 0.0:                                                                   <L 1071>
        var_80 = (var_64 <= var_79);
        if (var_80) {
            // return                                                                             <L 1072>
            continue;
        }
        // face_id_A = flex_faceadr[f] + fe_A                                                     <L 1075>
        var_81 = wp::address(var_flex_faceadr, var_6);
        var_83 = wp::load(var_81);
        var_82 = wp::add(var_83, var_36);
        // face_id_B = flex_faceadr[f] + fe_B                                                     <L 1076>
        var_84 = wp::address(var_flex_faceadr, var_6);
        var_86 = wp::load(var_84);
        var_85 = wp::add(var_86, var_41);
        // quat_A = face_quat_in[worldid, face_id_A]                                              <L 1078>
        var_87 = wp::address(var_face_quat_in, var_0, var_82);
        var_89 = wp::load(var_87);
        var_88 = wp::copy(var_89);
        // quat_B = face_quat_in[worldid, face_id_B]                                              <L 1079>
        var_90 = wp::address(var_face_quat_in, var_0, var_85);
        var_92 = wp::load(var_90);
        var_91 = wp::copy(var_92);
        // t1_A = wp.vec3(0.0)                                                                    <L 1082>
        var_94 = wp::vec_t<3, wp::float32>(var_93);
        // t2_A = wp.vec3(0.0)                                                                    <L 1083>
        var_96 = wp::vec_t<3, wp::float32>(var_95);
        // t1_B = wp.vec3(0.0)                                                                    <L 1084>
        var_98 = wp::vec_t<3, wp::float32>(var_97);
        // t2_B = wp.vec3(0.0)                                                                    <L 1085>
        var_100 = wp::vec_t<3, wp::float32>(var_99);
        // idx = int(0)                                                                           <L 1087>
        var_102 = wp::int(var_101);
        // for l0 in range(3):                                                                    <L 1088>
        var_104 = wp::range(var_103);
        start_for_1:;
            if (iter_cmp(var_104) == 0) goto end_for_1;
            var_105 = wp::iter_next(var_104);
            // if l0 > order_abs:                                                                 <L 1089>
            var_106 = (var_105 > var_12);
            if (var_106) {
                // continue                                                                       <L 1090>
                goto start_for_1;
            }
            // for l1 in range(3):                                                                <L 1091>
            var_108 = wp::range(var_107);
            start_for_3:;
                if (iter_cmp(var_108) == 0) goto end_for_3;
                var_109 = wp::iter_next(var_108);
                // if l1 > order_abs:                                                             <L 1092>
                var_110 = (var_109 > var_12);
                if (var_110) {
                    // continue                                                                   <L 1093>
                    goto start_for_3;
                }
                // pos_A = face_xpos_in[worldid, face_id_A, idx]                                  <L 1094>
                var_111 = wp::address(var_face_xpos_in, var_0, var_82, var_102);
                var_113 = wp::load(var_111);
                var_112 = wp::copy(var_113);
                // pos_B = face_xpos_in[worldid, face_id_B, idx]                                  <L 1095>
                var_114 = wp::address(var_face_xpos_in, var_0, var_85, var_102);
                var_116 = wp::load(var_114);
                var_115 = wp::copy(var_116);
                // grad0_A = support.flex_dphi(local_A[0], l0, order_abs) * support.flex_phi(local_A[1], l1, order_abs)       <L 1097>
                var_118 = wp::extract(var_49, var_117);
                var_119 = flex_dphi_0(var_118, var_105, var_12);
                var_121 = wp::extract(var_49, var_120);
                var_122 = flex_phi_0(var_121, var_109, var_12);
                var_123 = wp::mul(var_119, var_122);
                // grad1_A = support.flex_phi(local_A[0], l0, order_abs) * support.flex_dphi(local_A[1], l1, order_abs)       <L 1098>
                var_125 = wp::extract(var_49, var_124);
                var_126 = flex_phi_0(var_125, var_105, var_12);
                var_128 = wp::extract(var_49, var_127);
                var_129 = flex_dphi_0(var_128, var_109, var_12);
                var_130 = wp::mul(var_126, var_129);
                // grad0_B = support.flex_dphi(local_B[0], l0, order_abs) * support.flex_phi(local_B[1], l1, order_abs)       <L 1100>
                var_132 = wp::extract(var_58, var_131);
                var_133 = flex_dphi_0(var_132, var_105, var_12);
                var_135 = wp::extract(var_58, var_134);
                var_136 = flex_phi_0(var_135, var_109, var_12);
                var_137 = wp::mul(var_133, var_136);
                // grad1_B = support.flex_phi(local_B[0], l0, order_abs) * support.flex_dphi(local_B[1], l1, order_abs)       <L 1101>
                var_139 = wp::extract(var_58, var_138);
                var_140 = flex_phi_0(var_139, var_105, var_12);
                var_142 = wp::extract(var_58, var_141);
                var_143 = flex_dphi_0(var_142, var_109, var_12);
                var_144 = wp::mul(var_140, var_143);
                // t1_A += pos_A * grad0_A                                                        <L 1103>
                var_145 = wp::mul(var_112, var_123);
                var_146 = wp::add(var_94, var_145);
                // t2_A += pos_A * grad1_A                                                        <L 1104>
                var_147 = wp::mul(var_112, var_130);
                var_148 = wp::add(var_96, var_147);
                // t1_B += pos_B * grad0_B                                                        <L 1106>
                var_149 = wp::mul(var_115, var_137);
                var_150 = wp::add(var_98, var_149);
                // t2_B += pos_B * grad1_B                                                        <L 1107>
                var_151 = wp::mul(var_115, var_144);
                var_152 = wp::add(var_100, var_151);
                // idx += 1                                                                       <L 1108>
                var_154 = wp::add(var_102, var_153);
                wp::assign(var_94, var_146);
                wp::assign(var_96, var_148);
                wp::assign(var_98, var_150);
                wp::assign(var_100, var_152);
                wp::assign(var_102, var_154);
                goto start_for_3;
            end_for_3:;
            goto start_for_1;
        end_for_1:;
        // n_A = wp.cross(t1_A, t2_A)                                                             <L 1110>
        var_155 = wp::cross(var_94, var_96);
        // n_B = wp.cross(t1_B, t2_B)                                                             <L 1111>
        var_156 = wp::cross(var_98, var_100);
        // len_A = wp.length(n_A)                                                                 <L 1113>
        var_157 = wp::length(var_155);
        // len_B = wp.length(n_B)                                                                 <L 1114>
        var_158 = wp::length(var_156);
        // if len_A < MJ_MINVAL or len_B < MJ_MINVAL:                                             <L 1116>
        var_161 = (var_157 < var_160);
        var_159 = var_161;
        if (!var_159) {
            var_162 = (var_158 < var_160);
            var_159 = var_159 || var_162;
        }
        if (var_159) {
            // return                                                                             <L 1117>
            continue;
        }
        // inv_A = 1.0 / len_A                                                                    <L 1119>
        var_164 = wp::div(var_163, var_157);
        // inv_B = 1.0 / len_B                                                                    <L 1120>
        var_166 = wp::div(var_165, var_158);
        // n_A_norm = n_A * inv_A                                                                 <L 1121>
        var_167 = wp::mul(var_155, var_164);
        // n_B_norm = n_B * inv_B                                                                 <L 1122>
        var_168 = wp::mul(var_156, var_166);
        // if wp.dot(quat_A, quat_B) < 0.0:                                                       <L 1125>
        var_169 = wp::dot(var_88, var_91);
        var_171 = (var_169 < var_170);
        if (var_171) {
            // quat_B = -quat_B                                                                   <L 1126>
            var_172 = wp::neg(var_91);
        }
        var_173 = wp::where(var_171, var_172, var_91);
        // quat_avg = quat_A + quat_B                                                             <L 1127>
        var_174 = wp::add(var_88, var_173);
        // quat_avg = wp.normalize(quat_avg)                                                      <L 1128>
        var_175 = wp::normalize(var_174);
        // dn0_rot = wp.quat_rotate(quat_avg, dn0)                                                <L 1131>
        var_176 = wp::quat_rotate(var_175, var_75);
        // r = n_A_norm - n_B_norm - dn0_rot                                                      <L 1134>
        var_177 = wp::sub(var_167, var_168);
        var_178 = wp::sub(var_177, var_176);
        // dot_A = wp.dot(n_A_norm, r)                                                            <L 1137>
        var_179 = wp::dot(var_167, var_178);
        // w_A = (r - n_A_norm * dot_A) * inv_A                                                   <L 1138>
        var_180 = wp::mul(var_167, var_179);
        var_181 = wp::sub(var_178, var_180);
        var_182 = wp::mul(var_181, var_164);
        // dot_B = wp.dot(n_B_norm, r)                                                            <L 1140>
        var_183 = wp::dot(var_168, var_178);
        // w_B = (r - n_B_norm * dot_B) * inv_B                                                   <L 1141>
        var_184 = wp::mul(var_168, var_183);
        var_185 = wp::sub(var_178, var_184);
        var_186 = wp::mul(var_185, var_166);
        // wAt2 = wp.cross(w_A, t2_A)                                                             <L 1143>
        var_187 = wp::cross(var_182, var_96);
        // wAt1 = wp.cross(w_A, t1_A)                                                             <L 1144>
        var_188 = wp::cross(var_182, var_94);
        // wBt2 = wp.cross(w_B, t2_B)                                                             <L 1145>
        var_189 = wp::cross(var_186, var_100);
        // wBt1 = wp.cross(w_B, t1_B)                                                             <L 1146>
        var_190 = wp::cross(var_186, var_98);
        // _apply_face_forces(                                                                    <L 1149>
        // flex_nodebodyid,                                                                       <L 1150>
        // flex_face,                                                                             <L 1151>
        // xipos_in,                                                                              <L 1152>
        // flexnode_xpos_in,                                                                      <L 1153>
        // face_id_A,                                                                             <L 1154>
        // local_A,                                                                               <L 1155>
        // wAt1,                                                                                  <L 1156>
        // wAt2,                                                                                  <L 1157>
        // stiffness,                                                                             <L 1158>
        // order_abs,                                                                             <L 1159>
        // worldid,                                                                               <L 1160>
        // flex_spring_body_force_out,                                                            <L 1161>
        _apply_face_forces_0(var_flex_nodebodyid, var_flex_face, var_xipos_in, var_flexnode_xpos_in, var_82, var_49, var_188, var_187, var_64, var_12, var_0, var_flex_spring_body_force_out);
        // _apply_face_forces(                                                                    <L 1165>
        // flex_nodebodyid,                                                                       <L 1166>
        // flex_face,                                                                             <L 1167>
        // xipos_in,                                                                              <L 1168>
        // flexnode_xpos_in,                                                                      <L 1169>
        // face_id_B,                                                                             <L 1170>
        // local_B,                                                                               <L 1171>
        // wBt1,                                                                                  <L 1172>
        // wBt2,                                                                                  <L 1173>
        // -stiffness,                                                                            <L 1174>
        var_191 = wp::neg(var_64);
        // order_abs,                                                                             <L 1175>
        // worldid,                                                                               <L 1176>
        // flex_spring_body_force_out,                                                            <L 1177>
        _apply_face_forces_0(var_flex_nodebodyid, var_flex_face, var_xipos_in, var_flexnode_xpos_in, var_85, var_58, var_190, var_189, var_191, var_12, var_0, var_flex_spring_body_force_out);
    }
}



extern "C" __global__ void _gravity_force_7c63ac28_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_opt_gravity,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::float32> var_body_mass,
    wp::array_t<wp::float32> var_body_gravcomp,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::int32> var_body_isdofancestor,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::float32> var_qfrc_gravcomp_out)
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
        const wp::int32 var_3 = 1;
        wp::int32 var_4;
        wp::shape_t* var_5;
        const wp::int32 var_6 = 0;
        wp::int32 var_7;
        wp::shape_t var_8;
        wp::int32 var_9;
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32 var_12;
        wp::shape_t* var_13;
        const wp::int32 var_14 = 0;
        wp::int32 var_15;
        wp::shape_t var_16;
        wp::int32 var_17;
        wp::vec_t<3, wp::float32>* var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32> var_20;
        wp::vec_t<3, wp::float32> var_21;
        wp::shape_t* var_22;
        const wp::int32 var_23 = 0;
        wp::int32 var_24;
        wp::shape_t var_25;
        wp::int32 var_26;
        wp::float32* var_27;
        wp::vec_t<3, wp::float32> var_28;
        wp::float32 var_29;
        wp::vec_t<3, wp::float32> var_30;
        wp::vec_t<3, wp::float32>* var_31;
        wp::vec_t<3, wp::float32> var_32;
        wp::vec_t<3, wp::float32> var_33;
        wp::vec_t<3, wp::float32> var_34;
        wp::vec_t<3, wp::float32> var_35;
        wp::slice_t var_36;
        const wp::int32 var_37 = 0;
        wp::array_t<wp::float32> var_38;
        wp::float32 var_39;
        wp::float32 var_40;
        //---------
        // forward
        // def _gravity_force(                                                                    <L 274>
        // worldid, bodyid, dofid = wp.tid()                                                      <L 290>
        builtin_tid3d(var_0, var_1, var_2);
        // bodyid += 1  # skip world body                                                         <L 291>
        var_4 = wp::add(var_1, var_3);
        // gravcomp = body_gravcomp[worldid % body_gravcomp.shape[0], bodyid]                     <L 292>
        var_5 = &(var_body_gravcomp.shape);
        var_8 = wp::load(var_5);
        var_7 = wp::extract(var_8, var_6);
        var_9 = wp::mod(var_0, var_7);
        var_10 = wp::address(var_body_gravcomp, var_9, var_4);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // gravity = opt_gravity[worldid % opt_gravity.shape[0]]                                  <L 293>
        var_13 = &(var_opt_gravity.shape);
        var_16 = wp::load(var_13);
        var_15 = wp::extract(var_16, var_14);
        var_17 = wp::mod(var_0, var_15);
        var_18 = wp::address(var_opt_gravity, var_17);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // if gravcomp:                                                                           <L 295>
        if (var_11) {
            // force = -gravity * body_mass[worldid % body_mass.shape[0], bodyid] * gravcomp       <L 296>
            var_21 = wp::neg(var_19);
            var_22 = &(var_body_mass.shape);
            var_25 = wp::load(var_22);
            var_24 = wp::extract(var_25, var_23);
            var_26 = wp::mod(var_0, var_24);
            var_27 = wp::address(var_body_mass, var_26, var_4);
            var_29 = wp::load(var_27);
            var_28 = wp::mul(var_21, var_29);
            var_30 = wp::mul(var_28, var_11);
            // pos = xipos_in[worldid, bodyid]                                                    <L 297>
            var_31 = wp::address(var_xipos_in, var_0, var_4);
            var_33 = wp::load(var_31);
            var_32 = wp::copy(var_33);
            // jac, _ = support.jac_dof(                                                          <L 298>
            // body_parentid, body_rootid, dof_bodyid, body_isdofancestor, subtree_com_in, cdof_in, pos, bodyid, dofid, worldid       <L 299>
            jac_dof_0(var_body_parentid, var_body_rootid, var_dof_bodyid, var_body_isdofancestor, var_subtree_com_in, var_cdof_in, var_32, var_4, var_2, var_0, var_34, var_35);
            // wp.atomic_add(qfrc_gravcomp_out[worldid], dofid, wp.dot(jac, force))               <L 302>
            var_36 = wp::slice_t(var_0, var_0, var_37);
            var_38 = wp::view(var_qfrc_gravcomp_out, var_36);
            var_39 = wp::dot(var_34, var_30);
            var_40 = wp::atomic_add(var_38, var_2, var_39);
        }
    }
}



extern "C" __global__ void _fluid_force_fcd94129_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_opt_wind,
    wp::array_t<wp::float32> var_opt_density,
    wp::array_t<wp::float32> var_opt_viscosity,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_body_geomnum,
    wp::array_t<wp::int32> var_body_geomadr,
    wp::array_t<wp::float32> var_body_mass,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_inertia,
    wp::array_t<wp::int32> var_geom_type,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_size,
    wp::array_t<wp::float32> var_geom_fluid,
    wp::array_t<bool> var_body_fluid_ellipsoid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_ximat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_fluid_applied_out)
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
        wp::vec_t<3, wp::float32> var_3;
        const wp::float32 var_4 = 0.0;
        wp::vec_t<3, wp::float32> var_5;
        wp::vec_t<6, wp::float32> var_6;
        const wp::int32 var_7 = 0;
        bool var_8;
        wp::shape_t* var_9;
        const wp::int32 var_10 = 0;
        wp::int32 var_11;
        wp::shape_t var_12;
        wp::int32 var_13;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        const wp::float32 var_17 = 1e-15;
        bool var_18;
        wp::shape_t* var_19;
        const wp::int32 var_20 = 0;
        wp::int32 var_21;
        wp::shape_t var_22;
        wp::int32 var_23;
        wp::vec_t<3, wp::float32>* var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::vec_t<3, wp::float32> var_26;
        wp::shape_t* var_27;
        const wp::int32 var_28 = 0;
        wp::int32 var_29;
        wp::shape_t var_30;
        wp::int32 var_31;
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
        wp::vec_t<3, wp::float32>* var_43;
        wp::vec_t<3, wp::float32> var_44;
        wp::vec_t<3, wp::float32> var_45;
        wp::mat_t<3, 3, wp::float32>* var_46;
        wp::mat_t<3, 3, wp::float32> var_47;
        wp::mat_t<3, 3, wp::float32> var_48;
        wp::mat_t<3, 3, wp::float32> var_49;
        wp::vec_t<6, wp::float32>* var_50;
        wp::vec_t<6, wp::float32> var_51;
        wp::vec_t<6, wp::float32> var_52;
        wp::vec_t<3, wp::float32> var_53;
        wp::vec_t<3, wp::float32> var_54;
        wp::int32* var_55;
        wp::vec_t<3, wp::float32>* var_56;
        wp::int32 var_57;
        wp::vec_t<3, wp::float32> var_58;
        wp::vec_t<3, wp::float32> var_59;
        wp::vec_t<3, wp::float32> var_60;
        wp::vec_t<3, wp::float32> var_61;
        wp::vec_t<3, wp::float32> var_62;
        bool* var_63;
        bool var_64;
        const wp::float32 var_65 = 0.0;
        wp::vec_t<3, wp::float32> var_66;
        const wp::float32 var_67 = 0.0;
        wp::vec_t<3, wp::float32> var_68;
        wp::int32* var_69;
        wp::int32 var_70;
        wp::int32 var_71;
        wp::int32* var_72;
        wp::int32 var_73;
        wp::int32 var_74;
        wp::range_t var_75;
        wp::int32 var_76;
        wp::int32 var_77;
        const wp::int32 var_78 = 0;
        wp::float32* var_79;
        wp::float32 var_80;
        wp::float32 var_81;
        const wp::float32 var_82 = 0.0;
        bool var_83;
        wp::shape_t* var_84;
        const wp::int32 var_85 = 0;
        wp::int32 var_86;
        wp::shape_t var_87;
        wp::int32 var_88;
        wp::vec_t<3, wp::float32>* var_89;
        wp::vec_t<3, wp::float32> var_90;
        wp::vec_t<3, wp::float32> var_91;
        wp::int32* var_92;
        wp::vec_t<3, wp::float32> var_93;
        wp::int32 var_94;
        wp::mat_t<3, 3, wp::float32>* var_95;
        wp::mat_t<3, 3, wp::float32> var_96;
        wp::mat_t<3, 3, wp::float32> var_97;
        wp::mat_t<3, 3, wp::float32> var_98;
        wp::vec_t<3, wp::float32>* var_99;
        wp::vec_t<3, wp::float32> var_100;
        wp::vec_t<3, wp::float32> var_101;
        wp::vec_t<3, wp::float32> var_102;
        wp::vec_t<3, wp::float32> var_103;
        wp::vec_t<3, wp::float32> var_104;
        wp::vec_t<3, wp::float32> var_105;
        wp::vec_t<3, wp::float32> var_106;
        bool var_107;
        const wp::int32 var_108 = 0;
        wp::float32 var_109;
        const wp::int32 var_110 = 1;
        wp::float32 var_111;
        const wp::int32 var_112 = 2;
        wp::float32 var_113;
        wp::vec_t<3, wp::float32> var_114;
        wp::vec_t<3, wp::float32> var_115;
        wp::vec_t<3, wp::float32> var_116;
        const wp::float32 var_117 = 0.0;
        wp::vec_t<3, wp::float32> var_118;
        const wp::float32 var_119 = 0.0;
        wp::vec_t<3, wp::float32> var_120;
        const wp::float32 var_121 = 0.0;
        bool var_122;
        const wp::int32 var_123 = 6;
        wp::float32* var_124;
        const wp::int32 var_125 = 7;
        wp::float32* var_126;
        const wp::int32 var_127 = 8;
        wp::float32* var_128;
        wp::vec_t<3, wp::float32> var_129;
        wp::float32 var_130;
        wp::float32 var_131;
        wp::float32 var_132;
        const wp::int32 var_133 = 9;
        wp::float32* var_134;
        const wp::int32 var_135 = 10;
        wp::float32* var_136;
        const wp::int32 var_137 = 11;
        wp::float32* var_138;
        wp::vec_t<3, wp::float32> var_139;
        wp::float32 var_140;
        wp::float32 var_141;
        wp::float32 var_142;
        const wp::int32 var_143 = 0;
        wp::float32 var_144;
        wp::float32 var_145;
        const wp::int32 var_146 = 0;
        wp::float32 var_147;
        wp::float32 var_148;
        const wp::int32 var_149 = 1;
        wp::float32 var_150;
        wp::float32 var_151;
        const wp::int32 var_152 = 1;
        wp::float32 var_153;
        wp::float32 var_154;
        const wp::int32 var_155 = 2;
        wp::float32 var_156;
        wp::float32 var_157;
        const wp::int32 var_158 = 2;
        wp::float32 var_159;
        wp::float32 var_160;
        wp::vec_t<3, wp::float32> var_161;
        const wp::int32 var_162 = 0;
        wp::float32 var_163;
        wp::float32 var_164;
        const wp::int32 var_165 = 0;
        wp::float32 var_166;
        wp::float32 var_167;
        const wp::int32 var_168 = 1;
        wp::float32 var_169;
        wp::float32 var_170;
        const wp::int32 var_171 = 1;
        wp::float32 var_172;
        wp::float32 var_173;
        const wp::int32 var_174 = 2;
        wp::float32 var_175;
        wp::float32 var_176;
        const wp::int32 var_177 = 2;
        wp::float32 var_178;
        wp::float32 var_179;
        wp::vec_t<3, wp::float32> var_180;
        wp::vec_t<3, wp::float32> var_181;
        wp::vec_t<3, wp::float32> var_182;
        wp::vec_t<3, wp::float32> var_183;
        wp::vec_t<3, wp::float32> var_184;
        wp::vec_t<3, wp::float32> var_185;
        wp::vec_t<3, wp::float32> var_186;
        wp::vec_t<3, wp::float32> var_187;
        wp::vec_t<3, wp::float32> var_188;
        const wp::int32 var_189 = 5;
        wp::float32* var_190;
        wp::float32 var_191;
        wp::float32 var_192;
        const wp::int32 var_193 = 4;
        wp::float32* var_194;
        wp::float32 var_195;
        wp::float32 var_196;
        const wp::int32 var_197 = 1;
        wp::float32* var_198;
        wp::float32 var_199;
        wp::float32 var_200;
        const wp::int32 var_201 = 2;
        wp::float32* var_202;
        wp::float32 var_203;
        wp::float32 var_204;
        const wp::int32 var_205 = 3;
        wp::float32* var_206;
        wp::float32 var_207;
        wp::float32 var_208;
        const wp::float32 var_209 = 4.1887902047863905;
        const wp::int32 var_210 = 0;
        wp::float32 var_211;
        wp::float32 var_212;
        const wp::int32 var_213 = 1;
        wp::float32 var_214;
        wp::float32 var_215;
        const wp::int32 var_216 = 2;
        wp::float32 var_217;
        wp::float32 var_218;
        const wp::int32 var_219 = 0;
        wp::float32 var_220;
        const wp::int32 var_221 = 1;
        wp::float32 var_222;
        wp::float32 var_223;
        const wp::int32 var_224 = 2;
        wp::float32 var_225;
        wp::float32 var_226;
        const wp::int32 var_227 = 0;
        wp::float32 var_228;
        const wp::int32 var_229 = 1;
        wp::float32 var_230;
        wp::float32 var_231;
        const wp::int32 var_232 = 2;
        wp::float32 var_233;
        wp::float32 var_234;
        const wp::int32 var_235 = 0;
        wp::float32 var_236;
        const wp::int32 var_237 = 1;
        wp::float32 var_238;
        wp::float32 var_239;
        const wp::int32 var_240 = 2;
        wp::float32 var_241;
        wp::float32 var_242;
        wp::float32 var_243;
        wp::float32 var_244;
        const wp::float32 var_245 = 3.141592653589793;
        wp::float32 var_246;
        wp::float32 var_247;
        wp::float32 var_248;
        wp::vec_t<3, wp::float32> var_249;
        wp::float32 var_250;
        wp::float32 var_251;
        wp::vec_t<3, wp::float32> var_252;
        const wp::int32 var_253 = 1;
        wp::float32 var_254;
        const wp::int32 var_255 = 2;
        wp::float32 var_256;
        wp::float32 var_257;
        const wp::int32 var_258 = 2;
        wp::float32 var_259;
        const wp::int32 var_260 = 0;
        wp::float32 var_261;
        wp::float32 var_262;
        const wp::int32 var_263 = 0;
        wp::float32 var_264;
        const wp::int32 var_265 = 1;
        wp::float32 var_266;
        wp::float32 var_267;
        wp::float32 var_268;
        const wp::int32 var_269 = 0;
        wp::float32 var_270;
        wp::float32 var_271;
        wp::float32 var_272;
        wp::float32 var_273;
        const wp::int32 var_274 = 1;
        wp::float32 var_275;
        wp::float32 var_276;
        wp::float32 var_277;
        wp::float32 var_278;
        wp::float32 var_279;
        const wp::int32 var_280 = 2;
        wp::float32 var_281;
        wp::float32 var_282;
        wp::float32 var_283;
        wp::float32 var_284;
        const wp::int32 var_285 = 0;
        wp::float32 var_286;
        wp::float32 var_287;
        wp::float32 var_288;
        const wp::int32 var_289 = 1;
        wp::float32 var_290;
        wp::float32 var_291;
        wp::float32 var_292;
        wp::float32 var_293;
        const wp::int32 var_294 = 2;
        wp::float32 var_295;
        wp::float32 var_296;
        wp::float32 var_297;
        wp::float32 var_298;
        const wp::float32 var_299 = 3.141592653589793;
        wp::float32 var_300;
        wp::float32 var_301;
        wp::float32 var_302;
        wp::float32 var_303;
        wp::float32 var_304;
        wp::float32 var_305;
        wp::float32 var_306;
        wp::float32 var_307;
        const wp::int32 var_308 = 0;
        wp::float32 var_309;
        wp::float32 var_310;
        wp::float32 var_311;
        const wp::int32 var_312 = 1;
        wp::float32 var_313;
        wp::float32 var_314;
        wp::float32 var_315;
        const wp::int32 var_316 = 2;
        wp::float32 var_317;
        wp::float32 var_318;
        wp::vec_t<3, wp::float32> var_319;
        const wp::float32 var_320 = 0.0;
        wp::vec_t<3, wp::float32> var_321;
        bool var_322;
        const wp::float32 var_323 = 0.0;
        bool var_324;
        const wp::float32 var_325 = 0.0;
        bool var_326;
        bool var_327;
        wp::vec_t<3, wp::float32> var_328;
        wp::float32 var_329;
        wp::float32 var_330;
        wp::float32 var_331;
        wp::vec_t<3, wp::float32> var_332;
        wp::vec_t<3, wp::float32> var_333;
        wp::vec_t<3, wp::float32> var_334;
        const wp::float32 var_335 = 0.6666666666666666;
        const wp::int32 var_336 = 0;
        wp::float32 var_337;
        const wp::int32 var_338 = 1;
        wp::float32 var_339;
        wp::float32 var_340;
        const wp::int32 var_341 = 2;
        wp::float32 var_342;
        wp::float32 var_343;
        wp::float32 var_344;
        const wp::float32 var_345 = 9.42477796076938;
        wp::float32 var_346;
        const wp::float32 var_347 = 3.141592653589793;
        wp::float32 var_348;
        wp::float32 var_349;
        wp::float32 var_350;
        const wp::float32 var_351 = 1.6755160819145563;
        wp::float32 var_352;
        wp::float32 var_353;
        wp::float32 var_354;
        const wp::int32 var_355 = 0;
        wp::float32 var_356;
        const wp::int32 var_357 = 1;
        wp::float32 var_358;
        const wp::int32 var_359 = 2;
        wp::float32 var_360;
        const wp::int32 var_361 = 0;
        wp::float32 var_362;
        wp::float32 var_363;
        wp::float32 var_364;
        wp::float32 var_365;
        wp::float32 var_366;
        wp::float32 var_367;
        const wp::int32 var_368 = 1;
        wp::float32 var_369;
        wp::float32 var_370;
        wp::float32 var_371;
        wp::float32 var_372;
        wp::float32 var_373;
        wp::float32 var_374;
        const wp::int32 var_375 = 2;
        wp::float32 var_376;
        wp::float32 var_377;
        wp::float32 var_378;
        wp::float32 var_379;
        wp::float32 var_380;
        wp::float32 var_381;
        wp::vec_t<3, wp::float32> var_382;
        wp::float32 var_383;
        wp::float32 var_384;
        wp::float32 var_385;
        wp::float32 var_386;
        wp::float32 var_387;
        wp::float32 var_388;
        wp::float32 var_389;
        wp::float32 var_390;
        wp::float32 var_391;
        wp::float32 var_392;
        wp::float32 var_393;
        wp::float32 var_394;
        wp::vec_t<3, wp::float32> var_395;
        wp::vec_t<3, wp::float32> var_396;
        wp::vec_t<3, wp::float32> var_397;
        wp::vec_t<3, wp::float32> var_398;
        wp::vec_t<3, wp::float32> var_399;
        wp::vec_t<3, wp::float32> var_400;
        wp::vec_t<3, wp::float32> var_401;
        wp::vec_t<3, wp::float32> var_402;
        wp::vec_t<3, wp::float32> var_403;
        wp::vec_t<3, wp::float32> var_404;
        wp::vec_t<3, wp::float32> var_405;
        wp::vec_t<3, wp::float32> var_406;
        wp::vec_t<6, wp::float32> var_407;
        bool var_408;
        wp::vec_t<3, wp::float32> var_409;
        wp::vec_t<3, wp::float32> var_410;
        bool var_411;
        const wp::int32 var_412 = 0;
        wp::float32 var_413;
        const wp::int32 var_414 = 1;
        wp::float32 var_415;
        const wp::int32 var_416 = 2;
        wp::float32 var_417;
        wp::vec_t<3, wp::float32> var_418;
        wp::vec_t<3, wp::float32> var_419;
        wp::vec_t<3, wp::float32> var_420;
        const wp::float32 var_421 = 0.0;
        wp::vec_t<3, wp::float32> var_422;
        const wp::float32 var_423 = 0.0;
        wp::vec_t<3, wp::float32> var_424;
        const wp::float32 var_425 = 0.0;
        bool var_426;
        const wp::float32 var_427 = 0.0;
        bool var_428;
        bool var_429;
        wp::shape_t* var_430;
        const wp::int32 var_431 = 0;
        wp::int32 var_432;
        wp::shape_t var_433;
        wp::int32 var_434;
        wp::vec_t<3, wp::float32>* var_435;
        wp::vec_t<3, wp::float32> var_436;
        wp::vec_t<3, wp::float32> var_437;
        wp::shape_t* var_438;
        const wp::int32 var_439 = 0;
        wp::int32 var_440;
        wp::shape_t var_441;
        wp::int32 var_442;
        wp::float32* var_443;
        wp::float32 var_444;
        wp::float32 var_445;
        const wp::float32 var_446 = 6.0;
        wp::float32 var_447;
        const wp::int32 var_448 = 1;
        wp::float32 var_449;
        const wp::int32 var_450 = 2;
        wp::float32 var_451;
        wp::float32 var_452;
        const wp::int32 var_453 = 0;
        wp::float32 var_454;
        wp::float32 var_455;
        wp::float32 var_456;
        wp::float32 var_457;
        wp::float32 var_458;
        const wp::int32 var_459 = 0;
        wp::float32 var_460;
        const wp::int32 var_461 = 2;
        wp::float32 var_462;
        wp::float32 var_463;
        const wp::int32 var_464 = 1;
        wp::float32 var_465;
        wp::float32 var_466;
        wp::float32 var_467;
        wp::float32 var_468;
        wp::float32 var_469;
        const wp::int32 var_470 = 0;
        wp::float32 var_471;
        const wp::int32 var_472 = 1;
        wp::float32 var_473;
        wp::float32 var_474;
        const wp::int32 var_475 = 2;
        wp::float32 var_476;
        wp::float32 var_477;
        wp::float32 var_478;
        wp::float32 var_479;
        wp::float32 var_480;
        wp::float32 var_481;
        wp::float32 var_482;
        wp::float32 var_483;
        const wp::float32 var_484 = 3.0;
        wp::float32 var_485;
        wp::vec_t<3, wp::float32> var_486;
        const wp::float32 var_487 = 3.0;
        wp::float32 var_488;
        wp::vec_t<3, wp::float32> var_489;
        const wp::float32 var_490 = 3.141592653589793;
        wp::vec_t<3, wp::float32> var_491;
        wp::vec_t<3, wp::float32> var_492;
        const wp::float32 var_493 = -3.0;
        wp::vec_t<3, wp::float32> var_494;
        wp::vec_t<3, wp::float32> var_495;
        const wp::float32 var_496 = 3.141592653589793;
        wp::vec_t<3, wp::float32> var_497;
        wp::vec_t<3, wp::float32> var_498;
        wp::vec_t<3, wp::float32> var_499;
        wp::vec_t<3, wp::float32> var_500;
        const wp::float32 var_501 = 0.5;
        wp::float32 var_502;
        wp::float32 var_503;
        wp::float32 var_504;
        const wp::int32 var_505 = 0;
        wp::float32 var_506;
        wp::float32 var_507;
        wp::float32 var_508;
        const wp::int32 var_509 = 0;
        wp::float32 var_510;
        wp::float32 var_511;
        const wp::float32 var_512 = 0.5;
        wp::float32 var_513;
        wp::float32 var_514;
        wp::float32 var_515;
        const wp::int32 var_516 = 1;
        wp::float32 var_517;
        wp::float32 var_518;
        wp::float32 var_519;
        const wp::int32 var_520 = 1;
        wp::float32 var_521;
        wp::float32 var_522;
        const wp::float32 var_523 = 0.5;
        wp::float32 var_524;
        wp::float32 var_525;
        wp::float32 var_526;
        const wp::int32 var_527 = 2;
        wp::float32 var_528;
        wp::float32 var_529;
        wp::float32 var_530;
        const wp::int32 var_531 = 2;
        wp::float32 var_532;
        wp::float32 var_533;
        wp::vec_t<3, wp::float32> var_534;
        wp::vec_t<3, wp::float32> var_535;
        const wp::float32 var_536 = 64.0;
        wp::float32 var_537;
        const wp::float32 var_538 = 4.0;
        wp::float32 var_539;
        const wp::float32 var_540 = 4.0;
        wp::float32 var_541;
        const wp::float32 var_542 = 4.0;
        wp::float32 var_543;
        wp::float32 var_544;
        wp::float32 var_545;
        const wp::int32 var_546 = 0;
        wp::float32 var_547;
        wp::float32 var_548;
        wp::float32 var_549;
        const wp::int32 var_550 = 0;
        wp::float32 var_551;
        wp::float32 var_552;
        wp::float32 var_553;
        wp::float32 var_554;
        wp::float32 var_555;
        const wp::int32 var_556 = 1;
        wp::float32 var_557;
        wp::float32 var_558;
        wp::float32 var_559;
        const wp::int32 var_560 = 1;
        wp::float32 var_561;
        wp::float32 var_562;
        wp::float32 var_563;
        wp::float32 var_564;
        wp::float32 var_565;
        const wp::int32 var_566 = 2;
        wp::float32 var_567;
        wp::float32 var_568;
        wp::float32 var_569;
        const wp::int32 var_570 = 2;
        wp::float32 var_571;
        wp::float32 var_572;
        wp::float32 var_573;
        wp::vec_t<3, wp::float32> var_574;
        wp::vec_t<3, wp::float32> var_575;
        wp::vec_t<3, wp::float32> var_576;
        wp::vec_t<3, wp::float32> var_577;
        wp::float32 var_578;
        wp::vec_t<3, wp::float32> var_579;
        wp::vec_t<3, wp::float32> var_580;
        wp::vec_t<6, wp::float32> var_581;
        //---------
        // forward
        // def _fluid_force(                                                                      <L 306>
        // worldid, bodyid = wp.tid()                                                             <L 331>
        builtin_tid2d(var_0, var_1);
        // zero_force = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))                             <L 332>
        var_3 = wp::vec_t<3, wp::float32>(var_2);
        var_5 = wp::vec_t<3, wp::float32>(var_4);
        var_6 = wp::vec_t<6, wp::float32>(var_3, var_5);
        // if bodyid == 0:                                                                        <L 334>
        var_8 = (var_1 == var_7);
        if (var_8) {
            // fluid_applied_out[worldid, bodyid] = zero_force                                    <L 335>
            wp::array_store(var_fluid_applied_out, var_0, var_1, var_6);
            // return                                                                             <L 336>
            continue;
        }
        // mass = body_mass[worldid % body_mass.shape[0], bodyid]                                 <L 339>
        var_9 = &(var_body_mass.shape);
        var_12 = wp::load(var_9);
        var_11 = wp::extract(var_12, var_10);
        var_13 = wp::mod(var_0, var_11);
        var_14 = wp::address(var_body_mass, var_13, var_1);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // if mass < MJ_MINVAL:                                                                   <L 340>
        var_18 = (var_15 < var_17);
        if (var_18) {
            // fluid_applied_out[worldid, bodyid] = zero_force                                    <L 341>
            wp::array_store(var_fluid_applied_out, var_0, var_1, var_6);
            // return                                                                             <L 342>
            continue;
        }
        // wind = opt_wind[worldid % opt_wind.shape[0]]                                           <L 344>
        var_19 = &(var_opt_wind.shape);
        var_22 = wp::load(var_19);
        var_21 = wp::extract(var_22, var_20);
        var_23 = wp::mod(var_0, var_21);
        var_24 = wp::address(var_opt_wind, var_23);
        var_26 = wp::load(var_24);
        var_25 = wp::copy(var_26);
        // density = opt_density[worldid % opt_density.shape[0]]                                  <L 345>
        var_27 = &(var_opt_density.shape);
        var_30 = wp::load(var_27);
        var_29 = wp::extract(var_30, var_28);
        var_31 = wp::mod(var_0, var_29);
        var_32 = wp::address(var_opt_density, var_31);
        var_34 = wp::load(var_32);
        var_33 = wp::copy(var_34);
        // viscosity = opt_viscosity[worldid % opt_viscosity.shape[0]]                            <L 346>
        var_35 = &(var_opt_viscosity.shape);
        var_38 = wp::load(var_35);
        var_37 = wp::extract(var_38, var_36);
        var_39 = wp::mod(var_0, var_37);
        var_40 = wp::address(var_opt_viscosity, var_39);
        var_42 = wp::load(var_40);
        var_41 = wp::copy(var_42);
        // xipos = xipos_in[worldid, bodyid]                                                      <L 349>
        var_43 = wp::address(var_xipos_in, var_0, var_1);
        var_45 = wp::load(var_43);
        var_44 = wp::copy(var_45);
        // rot = ximat_in[worldid, bodyid]                                                        <L 350>
        var_46 = wp::address(var_ximat_in, var_0, var_1);
        var_48 = wp::load(var_46);
        var_47 = wp::copy(var_48);
        // rotT = wp.transpose(rot)                                                               <L 351>
        var_49 = wp::transpose(var_47);
        // cvel = cvel_in[worldid, bodyid]                                                        <L 352>
        var_50 = wp::address(var_cvel_in, var_0, var_1);
        var_52 = wp::load(var_50);
        var_51 = wp::copy(var_52);
        // ang_global = wp.spatial_top(cvel)                                                      <L 353>
        var_53 = wp::spatial_top(var_51);
        // lin_global = wp.spatial_bottom(cvel)                                                   <L 354>
        var_54 = wp::spatial_bottom(var_51);
        // subtree_root = subtree_com_in[worldid, body_rootid[bodyid]]                            <L 355>
        var_55 = wp::address(var_body_rootid, var_1);
        var_57 = wp::load(var_55);
        var_56 = wp::address(var_subtree_com_in, var_0, var_57);
        var_59 = wp::load(var_56);
        var_58 = wp::copy(var_59);
        // lin_com = lin_global - wp.cross(xipos - subtree_root, ang_global)                      <L 356>
        var_60 = wp::sub(var_44, var_58);
        var_61 = wp::cross(var_60, var_53);
        var_62 = wp::sub(var_54, var_61);
        // if body_fluid_ellipsoid[bodyid]:                                                       <L 358>
        var_63 = wp::address(var_body_fluid_ellipsoid, var_1);
        var_64 = wp::load(var_63);
        if (var_64) {
            // force_global = wp.vec3(0.0)                                                        <L 359>
            var_66 = wp::vec_t<3, wp::float32>(var_65);
            // torque_global = wp.vec3(0.0)                                                       <L 360>
            var_68 = wp::vec_t<3, wp::float32>(var_67);
            // start = body_geomadr[bodyid]                                                       <L 362>
            var_69 = wp::address(var_body_geomadr, var_1);
            var_71 = wp::load(var_69);
            var_70 = wp::copy(var_71);
            // count = body_geomnum[bodyid]                                                       <L 363>
            var_72 = wp::address(var_body_geomnum, var_1);
            var_74 = wp::load(var_72);
            var_73 = wp::copy(var_74);
            // for i in range(count):                                                             <L 365>
            var_75 = wp::range(var_73);
            start_for_2:;
                if (iter_cmp(var_75) == 0) goto end_for_2;
                var_76 = wp::iter_next(var_75);
                // geomid = start + i                                                             <L 366>
                var_77 = wp::add(var_70, var_76);
                // coef = geom_fluid[geomid, 0]                                                   <L 367>
                var_79 = wp::address(var_geom_fluid, var_77, var_78);
                var_81 = wp::load(var_79);
                var_80 = wp::copy(var_81);
                // if coef <= 0.0:                                                                <L 368>
                var_83 = (var_80 <= var_82);
                if (var_83) {
                    // continue                                                                   <L 369>
                    goto start_for_2;
                }
                // size = geom_size[worldid % geom_size.shape[0], geomid]                         <L 371>
                var_84 = &(var_geom_size.shape);
                var_87 = wp::load(var_84);
                var_86 = wp::extract(var_87, var_85);
                var_88 = wp::mod(var_0, var_86);
                var_89 = wp::address(var_geom_size, var_88, var_77);
                var_91 = wp::load(var_89);
                var_90 = wp::copy(var_91);
                // semiaxes = geom_semiaxes(size, geom_type[geomid])                              <L 372>
                var_92 = wp::address(var_geom_type, var_77);
                var_94 = wp::load(var_92);
                var_93 = geom_semiaxes_0(var_90, var_94);
                // geom_rot = geom_xmat_in[worldid, geomid]                                       <L 373>
                var_95 = wp::address(var_geom_xmat_in, var_0, var_77);
                var_97 = wp::load(var_95);
                var_96 = wp::copy(var_97);
                // geom_rotT = wp.transpose(geom_rot)                                             <L 374>
                var_98 = wp::transpose(var_96);
                // geom_pos = geom_xpos_in[worldid, geomid]                                       <L 375>
                var_99 = wp::address(var_geom_xpos_in, var_0, var_77);
                var_101 = wp::load(var_99);
                var_100 = wp::copy(var_101);
                // lin_point = lin_com + wp.cross(ang_global, geom_pos - xipos)                   <L 377>
                var_102 = wp::sub(var_100, var_44);
                var_103 = wp::cross(var_53, var_102);
                var_104 = wp::add(var_62, var_103);
                // l_ang = geom_rotT @ ang_global                                                 <L 379>
                var_105 = wp::mul(var_98, var_53);
                // l_lin = geom_rotT @ lin_point                                                  <L 380>
                var_106 = wp::mul(var_98, var_104);
                // if wind[0] or wind[1] or wind[2]:                                              <L 382>
                var_109 = wp::extract(var_25, var_108);
                var_107 = var_109;
                if (!var_107) {
                    var_111 = wp::extract(var_25, var_110);
                    var_107 = var_107 || var_111;
                }
                if (!var_107) {
                    var_113 = wp::extract(var_25, var_112);
                    var_107 = var_107 || var_113;
                }
                if (var_107) {
                    // l_lin -= geom_rotT @ wind                                                  <L 383>
                    var_114 = wp::mul(var_98, var_25);
                    var_115 = wp::sub(var_106, var_114);
                }
                var_116 = wp::where(var_107, var_115, var_106);
                // lfrc_torque = wp.vec3(0.0)                                                     <L 385>
                var_118 = wp::vec_t<3, wp::float32>(var_117);
                // lfrc_force = wp.vec3(0.0)                                                      <L 386>
                var_120 = wp::vec_t<3, wp::float32>(var_119);
                // if density > 0.0:                                                              <L 388>
                var_122 = (var_33 > var_121);
                if (var_122) {
                    // virtual_mass = wp.vec3(geom_fluid[geomid, 6], geom_fluid[geomid, 7], geom_fluid[geomid, 8])       <L 390>
                    var_124 = wp::address(var_geom_fluid, var_77, var_123);
                    var_126 = wp::address(var_geom_fluid, var_77, var_125);
                    var_128 = wp::address(var_geom_fluid, var_77, var_127);
                    var_130 = wp::load(var_124);
                    var_131 = wp::load(var_126);
                    var_132 = wp::load(var_128);
                    var_129 = wp::vec_t<3, wp::float32>(var_130, var_131, var_132);
                    // virtual_inertia = wp.vec3(geom_fluid[geomid, 9], geom_fluid[geomid, 10], geom_fluid[geomid, 11])       <L 391>
                    var_134 = wp::address(var_geom_fluid, var_77, var_133);
                    var_136 = wp::address(var_geom_fluid, var_77, var_135);
                    var_138 = wp::address(var_geom_fluid, var_77, var_137);
                    var_140 = wp::load(var_134);
                    var_141 = wp::load(var_136);
                    var_142 = wp::load(var_138);
                    var_139 = wp::vec_t<3, wp::float32>(var_140, var_141, var_142);
                    // virtual_lin_mom = wp.vec3(                                                 <L 393>
                    // density * virtual_mass[0] * l_lin[0],                                      <L 394>
                    var_144 = wp::extract(var_129, var_143);
                    var_145 = wp::mul(var_33, var_144);
                    var_147 = wp::extract(var_116, var_146);
                    var_148 = wp::mul(var_145, var_147);
                    // density * virtual_mass[1] * l_lin[1],                                      <L 395>
                    var_150 = wp::extract(var_129, var_149);
                    var_151 = wp::mul(var_33, var_150);
                    var_153 = wp::extract(var_116, var_152);
                    var_154 = wp::mul(var_151, var_153);
                    // density * virtual_mass[2] * l_lin[2],                                      <L 396>
                    var_156 = wp::extract(var_129, var_155);
                    var_157 = wp::mul(var_33, var_156);
                    var_159 = wp::extract(var_116, var_158);
                    var_160 = wp::mul(var_157, var_159);
                    var_161 = wp::vec_t<3, wp::float32>(var_148, var_154, var_160);
                    // virtual_ang_mom = wp.vec3(                                                 <L 398>
                    // density * virtual_inertia[0] * l_ang[0],                                   <L 399>
                    var_163 = wp::extract(var_139, var_162);
                    var_164 = wp::mul(var_33, var_163);
                    var_166 = wp::extract(var_105, var_165);
                    var_167 = wp::mul(var_164, var_166);
                    // density * virtual_inertia[1] * l_ang[1],                                   <L 400>
                    var_169 = wp::extract(var_139, var_168);
                    var_170 = wp::mul(var_33, var_169);
                    var_172 = wp::extract(var_105, var_171);
                    var_173 = wp::mul(var_170, var_172);
                    // density * virtual_inertia[2] * l_ang[2],                                   <L 401>
                    var_175 = wp::extract(var_139, var_174);
                    var_176 = wp::mul(var_33, var_175);
                    var_178 = wp::extract(var_105, var_177);
                    var_179 = wp::mul(var_176, var_178);
                    var_180 = wp::vec_t<3, wp::float32>(var_167, var_173, var_179);
                    // added_mass_force = wp.cross(virtual_lin_mom, l_ang)                        <L 404>
                    var_181 = wp::cross(var_161, var_105);
                    // added_mass_torque = wp.cross(virtual_lin_mom, l_lin) + wp.cross(virtual_ang_mom, l_ang)       <L 405>
                    var_182 = wp::cross(var_161, var_116);
                    var_183 = wp::cross(var_180, var_105);
                    var_184 = wp::add(var_182, var_183);
                    // lfrc_force += added_mass_force                                             <L 407>
                    var_185 = wp::add(var_120, var_181);
                    // lfrc_torque += added_mass_torque                                           <L 408>
                    var_186 = wp::add(var_118, var_184);
                }
                var_187 = wp::where(var_122, var_186, var_118);
                var_188 = wp::where(var_122, var_185, var_120);
                // magnus_coef = geom_fluid[geomid, 5]                                            <L 411>
                var_190 = wp::address(var_geom_fluid, var_77, var_189);
                var_192 = wp::load(var_190);
                var_191 = wp::copy(var_192);
                // kutta_coef = geom_fluid[geomid, 4]                                             <L 412>
                var_194 = wp::address(var_geom_fluid, var_77, var_193);
                var_196 = wp::load(var_194);
                var_195 = wp::copy(var_196);
                // blunt_drag_coef = geom_fluid[geomid, 1]                                        <L 413>
                var_198 = wp::address(var_geom_fluid, var_77, var_197);
                var_200 = wp::load(var_198);
                var_199 = wp::copy(var_200);
                // slender_drag_coef = geom_fluid[geomid, 2]                                      <L 414>
                var_202 = wp::address(var_geom_fluid, var_77, var_201);
                var_204 = wp::load(var_202);
                var_203 = wp::copy(var_204);
                // ang_drag_coef = geom_fluid[geomid, 3]                                          <L 415>
                var_206 = wp::address(var_geom_fluid, var_77, var_205);
                var_208 = wp::load(var_206);
                var_207 = wp::copy(var_208);
                // volume = wp.static(4.0 / 3.0 * wp.pi) * semiaxes[0] * semiaxes[1] * semiaxes[2]       <L 417>
                var_211 = wp::extract(var_93, var_210);
                var_212 = wp::mul(var_209, var_211);
                var_214 = wp::extract(var_93, var_213);
                var_215 = wp::mul(var_212, var_214);
                var_217 = wp::extract(var_93, var_216);
                var_218 = wp::mul(var_215, var_217);
                // d_max = wp.max(wp.max(semiaxes[0], semiaxes[1]), semiaxes[2])                  <L 418>
                var_220 = wp::extract(var_93, var_219);
                var_222 = wp::extract(var_93, var_221);
                var_223 = wp::max(var_220, var_222);
                var_225 = wp::extract(var_93, var_224);
                var_226 = wp::max(var_223, var_225);
                // d_min = wp.min(wp.min(semiaxes[0], semiaxes[1]), semiaxes[2])                  <L 419>
                var_228 = wp::extract(var_93, var_227);
                var_230 = wp::extract(var_93, var_229);
                var_231 = wp::min(var_228, var_230);
                var_233 = wp::extract(var_93, var_232);
                var_234 = wp::min(var_231, var_233);
                // d_mid = semiaxes[0] + semiaxes[1] + semiaxes[2] - d_max - d_min                <L 420>
                var_236 = wp::extract(var_93, var_235);
                var_238 = wp::extract(var_93, var_237);
                var_239 = wp::add(var_236, var_238);
                var_241 = wp::extract(var_93, var_240);
                var_242 = wp::add(var_239, var_241);
                var_243 = wp::sub(var_242, var_226);
                var_244 = wp::sub(var_243, var_234);
                // A_max = wp.pi * d_max * d_mid                                                  <L 421>
                var_246 = wp::mul(var_245, var_226);
                var_247 = wp::mul(var_246, var_244);
                // lin_speed = wp.length(l_lin)                                                   <L 423>
                var_248 = wp::length(var_116);
                // magnus_force = wp.cross(l_ang, l_lin) * (magnus_coef * density * volume)       <L 425>
                var_249 = wp::cross(var_105, var_116);
                var_250 = wp::mul(var_191, var_33);
                var_251 = wp::mul(var_250, var_218);
                var_252 = wp::mul(var_249, var_251);
                // s12 = semiaxes[1] * semiaxes[2]                                                <L 427>
                var_254 = wp::extract(var_93, var_253);
                var_256 = wp::extract(var_93, var_255);
                var_257 = wp::mul(var_254, var_256);
                // s20 = semiaxes[2] * semiaxes[0]                                                <L 428>
                var_259 = wp::extract(var_93, var_258);
                var_261 = wp::extract(var_93, var_260);
                var_262 = wp::mul(var_259, var_261);
                // s01 = semiaxes[0] * semiaxes[1]                                                <L 429>
                var_264 = wp::extract(var_93, var_263);
                var_266 = wp::extract(var_93, var_265);
                var_267 = wp::mul(var_264, var_266);
                // proj_denom = _pow4(s12) * _pow2(l_lin[0]) + _pow4(s20) * _pow2(l_lin[1]) + _pow4(s01) * _pow2(l_lin[2])       <L 431>
                var_268 = _pow4_0(var_257);
                var_270 = wp::extract(var_116, var_269);
                var_271 = _pow2_0(var_270);
                var_272 = wp::mul(var_268, var_271);
                var_273 = _pow4_0(var_262);
                var_275 = wp::extract(var_116, var_274);
                var_276 = _pow2_0(var_275);
                var_277 = wp::mul(var_273, var_276);
                var_278 = wp::add(var_272, var_277);
                var_279 = _pow4_0(var_267);
                var_281 = wp::extract(var_116, var_280);
                var_282 = _pow2_0(var_281);
                var_283 = wp::mul(var_279, var_282);
                var_284 = wp::add(var_278, var_283);
                // proj_num = _pow2(s12 * l_lin[0]) + _pow2(s20 * l_lin[1]) + _pow2(s01 * l_lin[2])       <L 432>
                var_286 = wp::extract(var_116, var_285);
                var_287 = wp::mul(var_257, var_286);
                var_288 = _pow2_0(var_287);
                var_290 = wp::extract(var_116, var_289);
                var_291 = wp::mul(var_262, var_290);
                var_292 = _pow2_0(var_291);
                var_293 = wp::add(var_288, var_292);
                var_295 = wp::extract(var_116, var_294);
                var_296 = wp::mul(var_267, var_295);
                var_297 = _pow2_0(var_296);
                var_298 = wp::add(var_293, var_297);
                // A_proj = wp.pi * wp.sqrt(proj_denom / wp.max(MJ_MINVAL, proj_num))             <L 434>
                var_300 = wp::max(var_17, var_298);
                var_301 = wp::div(var_284, var_300);
                var_302 = wp::sqrt(var_301);
                var_303 = wp::mul(var_299, var_302);
                // cos_alpha = proj_num / wp.max(MJ_MINVAL, lin_speed * proj_denom)               <L 435>
                var_304 = wp::mul(var_248, var_284);
                var_305 = wp::max(var_17, var_304);
                var_306 = wp::div(var_298, var_305);
                // norm = wp.vec3(                                                                <L 437>
                // _pow2(s12) * l_lin[0],                                                         <L 438>
                var_307 = _pow2_0(var_257);
                var_309 = wp::extract(var_116, var_308);
                var_310 = wp::mul(var_307, var_309);
                // _pow2(s20) * l_lin[1],                                                         <L 439>
                var_311 = _pow2_0(var_262);
                var_313 = wp::extract(var_116, var_312);
                var_314 = wp::mul(var_311, var_313);
                // _pow2(s01) * l_lin[2],                                                         <L 440>
                var_315 = _pow2_0(var_267);
                var_317 = wp::extract(var_116, var_316);
                var_318 = wp::mul(var_315, var_317);
                var_319 = wp::vec_t<3, wp::float32>(var_310, var_314, var_318);
                // kutta_force = wp.vec3(0.0)                                                     <L 443>
                var_321 = wp::vec_t<3, wp::float32>(var_320);
                // if density > 0.0 and kutta_coef != 0.0 and lin_speed > MJ_MINVAL:              <L 444>
                var_324 = (var_33 > var_323);
                var_322 = var_324;
                if (var_322) {
                    var_326 = (var_195 != var_325);
                    var_322 = var_322 && var_326;
                }
                if (var_322) {
                    var_327 = (var_248 > var_17);
                    var_322 = var_322 && var_327;
                }
                if (var_322) {
                    // kutta_circ = wp.cross(norm, l_lin) * (kutta_coef * density * cos_alpha * A_proj)       <L 445>
                    var_328 = wp::cross(var_319, var_116);
                    var_329 = wp::mul(var_195, var_33);
                    var_330 = wp::mul(var_329, var_306);
                    var_331 = wp::mul(var_330, var_303);
                    var_332 = wp::mul(var_328, var_331);
                    // kutta_force = wp.cross(kutta_circ, l_lin)                                  <L 446>
                    var_333 = wp::cross(var_332, var_116);
                }
                var_334 = wp::where(var_322, var_333, var_321);
                // eq_sphere_D = wp.static(2.0 / 3.0) * (semiaxes[0] + semiaxes[1] + semiaxes[2])       <L 448>
                var_337 = wp::extract(var_93, var_336);
                var_339 = wp::extract(var_93, var_338);
                var_340 = wp::add(var_337, var_339);
                var_342 = wp::extract(var_93, var_341);
                var_343 = wp::add(var_340, var_342);
                var_344 = wp::mul(var_335, var_343);
                // lin_visc_force_coef = wp.static(3.0 * wp.pi) * eq_sphere_D                     <L 449>
                var_346 = wp::mul(var_345, var_344);
                // lin_visc_torq_coef = wp.pi * eq_sphere_D * eq_sphere_D * eq_sphere_D           <L 450>
                var_348 = wp::mul(var_347, var_344);
                var_349 = wp::mul(var_348, var_344);
                var_350 = wp::mul(var_349, var_344);
                // I_max = wp.static(8.0 / 15.0 * wp.pi) * d_mid * _pow4(d_max)                   <L 452>
                var_352 = wp::mul(var_351, var_244);
                var_353 = _pow4_0(var_226);
                var_354 = wp::mul(var_352, var_353);
                // II0 = ellipsoid_max_moment(semiaxes, 0)                                        <L 453>
                var_356 = ellipsoid_max_moment_0(var_93, var_355);
                // II1 = ellipsoid_max_moment(semiaxes, 1)                                        <L 454>
                var_358 = ellipsoid_max_moment_0(var_93, var_357);
                // II2 = ellipsoid_max_moment(semiaxes, 2)                                        <L 455>
                var_360 = ellipsoid_max_moment_0(var_93, var_359);
                // mom_visc = wp.vec3(                                                            <L 457>
                // l_ang[0] * (ang_drag_coef * II0 + slender_drag_coef * (I_max - II0)),          <L 458>
                var_362 = wp::extract(var_105, var_361);
                var_363 = wp::mul(var_207, var_356);
                var_364 = wp::sub(var_354, var_356);
                var_365 = wp::mul(var_203, var_364);
                var_366 = wp::add(var_363, var_365);
                var_367 = wp::mul(var_362, var_366);
                // l_ang[1] * (ang_drag_coef * II1 + slender_drag_coef * (I_max - II1)),          <L 459>
                var_369 = wp::extract(var_105, var_368);
                var_370 = wp::mul(var_207, var_358);
                var_371 = wp::sub(var_354, var_358);
                var_372 = wp::mul(var_203, var_371);
                var_373 = wp::add(var_370, var_372);
                var_374 = wp::mul(var_369, var_373);
                // l_ang[2] * (ang_drag_coef * II2 + slender_drag_coef * (I_max - II2)),          <L 460>
                var_376 = wp::extract(var_105, var_375);
                var_377 = wp::mul(var_207, var_360);
                var_378 = wp::sub(var_354, var_360);
                var_379 = wp::mul(var_203, var_378);
                var_380 = wp::add(var_377, var_379);
                var_381 = wp::mul(var_376, var_380);
                var_382 = wp::vec_t<3, wp::float32>(var_367, var_374, var_381);
                // drag_lin_coef = viscosity * lin_visc_force_coef + density * lin_speed * (       <L 463>
                var_383 = wp::mul(var_41, var_346);
                var_384 = wp::mul(var_33, var_248);
                // A_proj * blunt_drag_coef + slender_drag_coef * (A_max - A_proj)                <L 464>
                var_385 = wp::mul(var_303, var_199);
                var_386 = wp::sub(var_247, var_303);
                var_387 = wp::mul(var_203, var_386);
                var_388 = wp::add(var_385, var_387);
                var_389 = wp::mul(var_384, var_388);
                var_390 = wp::add(var_383, var_389);
                // drag_ang_coef = viscosity * lin_visc_torq_coef + density * wp.length(mom_visc)       <L 466>
                var_391 = wp::mul(var_41, var_350);
                var_392 = wp::length(var_382);
                var_393 = wp::mul(var_33, var_392);
                var_394 = wp::add(var_391, var_393);
                // lfrc_torque -= drag_ang_coef * l_ang                                           <L 468>
                var_395 = wp::mul(var_394, var_105);
                var_396 = wp::sub(var_187, var_395);
                // lfrc_force += magnus_force + kutta_force - drag_lin_coef * l_lin               <L 469>
                var_397 = wp::add(var_252, var_334);
                var_398 = wp::mul(var_390, var_116);
                var_399 = wp::sub(var_397, var_398);
                var_400 = wp::add(var_188, var_399);
                // lfrc_torque *= coef                                                            <L 471>
                var_401 = wp::mul(var_396, var_80);
                // lfrc_force *= coef                                                             <L 472>
                var_402 = wp::mul(var_400, var_80);
                // torque_global += geom_rot @ lfrc_torque                                        <L 475>
                var_403 = wp::mul(var_96, var_401);
                var_404 = wp::add(var_68, var_403);
                // force_global += geom_rot @ lfrc_force                                          <L 476>
                var_405 = wp::mul(var_96, var_402);
                var_406 = wp::add(var_66, var_405);
                wp::assign(var_66, var_406);
                wp::assign(var_68, var_404);
                goto start_for_2;
            end_for_2:;
            // fluid_applied_out[worldid, bodyid] = wp.spatial_vector(force_global, torque_global)       <L 478>
            var_407 = wp::vec_t<6, wp::float32>(var_66, var_68);
            wp::array_store(var_fluid_applied_out, var_0, var_1, var_407);
            // return                                                                             <L 479>
            continue;
        }
        var_408 = wp::load(var_63);
        // l_ang = rotT @ ang_global                                                              <L 481>
        var_409 = wp::mul(var_49, var_53);
        // l_lin = rotT @ lin_com                                                                 <L 482>
        var_410 = wp::mul(var_49, var_62);
        // if wind[0] or wind[1] or wind[2]:                                                      <L 484>
        var_413 = wp::extract(var_25, var_412);
        var_411 = var_413;
        if (!var_411) {
            var_415 = wp::extract(var_25, var_414);
            var_411 = var_411 || var_415;
        }
        if (!var_411) {
            var_417 = wp::extract(var_25, var_416);
            var_411 = var_411 || var_417;
        }
        if (var_411) {
            // l_lin -= rotT @ wind                                                               <L 485>
            var_418 = wp::mul(var_49, var_25);
            var_419 = wp::sub(var_410, var_418);
        }
        var_420 = wp::where(var_411, var_419, var_410);
        // lfrc_torque = wp.vec3(0.0)                                                             <L 487>
        var_422 = wp::vec_t<3, wp::float32>(var_421);
        // lfrc_force = wp.vec3(0.0)                                                              <L 488>
        var_424 = wp::vec_t<3, wp::float32>(var_423);
        // has_viscosity = viscosity > 0.0                                                        <L 490>
        var_426 = (var_41 > var_425);
        // has_density = density > 0.0                                                            <L 491>
        var_428 = (var_33 > var_427);
        // if has_viscosity or has_density:                                                       <L 493>
        var_429 = var_426;
        if (!var_429) {
            var_429 = var_429 || var_428;
        }
        if (var_429) {
            // inertia = body_inertia[worldid % body_inertia.shape[0], bodyid]                    <L 494>
            var_430 = &(var_body_inertia.shape);
            var_433 = wp::load(var_430);
            var_432 = wp::extract(var_433, var_431);
            var_434 = wp::mod(var_0, var_432);
            var_435 = wp::address(var_body_inertia, var_434, var_1);
            var_437 = wp::load(var_435);
            var_436 = wp::copy(var_437);
            // mass = body_mass[worldid % body_mass.shape[0], bodyid]                             <L 495>
            var_438 = &(var_body_mass.shape);
            var_441 = wp::load(var_438);
            var_440 = wp::extract(var_441, var_439);
            var_442 = wp::mod(var_0, var_440);
            var_443 = wp::address(var_body_mass, var_442, var_1);
            var_445 = wp::load(var_443);
            var_444 = wp::copy(var_445);
            // scl = 6.0 / mass                                                                   <L 496>
            var_447 = wp::div(var_446, var_444);
            // box0 = wp.sqrt(wp.max(MJ_MINVAL, inertia[1] + inertia[2] - inertia[0]) * scl)       <L 497>
            var_449 = wp::extract(var_436, var_448);
            var_451 = wp::extract(var_436, var_450);
            var_452 = wp::add(var_449, var_451);
            var_454 = wp::extract(var_436, var_453);
            var_455 = wp::sub(var_452, var_454);
            var_456 = wp::max(var_17, var_455);
            var_457 = wp::mul(var_456, var_447);
            var_458 = wp::sqrt(var_457);
            // box1 = wp.sqrt(wp.max(MJ_MINVAL, inertia[0] + inertia[2] - inertia[1]) * scl)       <L 498>
            var_460 = wp::extract(var_436, var_459);
            var_462 = wp::extract(var_436, var_461);
            var_463 = wp::add(var_460, var_462);
            var_465 = wp::extract(var_436, var_464);
            var_466 = wp::sub(var_463, var_465);
            var_467 = wp::max(var_17, var_466);
            var_468 = wp::mul(var_467, var_447);
            var_469 = wp::sqrt(var_468);
            // box2 = wp.sqrt(wp.max(MJ_MINVAL, inertia[0] + inertia[1] - inertia[2]) * scl)       <L 499>
            var_471 = wp::extract(var_436, var_470);
            var_473 = wp::extract(var_436, var_472);
            var_474 = wp::add(var_471, var_473);
            var_476 = wp::extract(var_436, var_475);
            var_477 = wp::sub(var_474, var_476);
            var_478 = wp::max(var_17, var_477);
            var_479 = wp::mul(var_478, var_447);
            var_480 = wp::sqrt(var_479);
        }
        var_481 = wp::where(var_429, var_444, var_15);
        // if has_viscosity:                                                                      <L 501>
        if (var_426) {
            // diam = (box0 + box1 + box2) / 3.0                                                  <L 502>
            var_482 = wp::add(var_458, var_469);
            var_483 = wp::add(var_482, var_480);
            var_485 = wp::div(var_483, var_484);
            // lfrc_torque = -l_ang * wp.pow(diam, 3.0) * wp.pi * viscosity                       <L 503>
            var_486 = wp::neg(var_409);
            var_488 = wp::pow(var_485, var_487);
            var_489 = wp::mul(var_486, var_488);
            var_491 = wp::mul(var_489, var_490);
            var_492 = wp::mul(var_491, var_41);
            // lfrc_force = -3.0 * l_lin * diam * wp.pi * viscosity                               <L 504>
            var_494 = wp::mul(var_493, var_420);
            var_495 = wp::mul(var_494, var_485);
            var_497 = wp::mul(var_495, var_496);
            var_498 = wp::mul(var_497, var_41);
        }
        var_499 = wp::where(var_426, var_492, var_422);
        var_500 = wp::where(var_426, var_498, var_424);
        // if has_density:                                                                        <L 506>
        if (var_428) {
            // lfrc_force -= wp.vec3(                                                             <L 507>
            // 0.5 * density * box1 * box2 * wp.abs(l_lin[0]) * l_lin[0],                         <L 508>
            var_502 = wp::mul(var_501, var_33);
            var_503 = wp::mul(var_502, var_469);
            var_504 = wp::mul(var_503, var_480);
            var_506 = wp::extract(var_420, var_505);
            var_507 = wp::abs(var_506);
            var_508 = wp::mul(var_504, var_507);
            var_510 = wp::extract(var_420, var_509);
            var_511 = wp::mul(var_508, var_510);
            // 0.5 * density * box0 * box2 * wp.abs(l_lin[1]) * l_lin[1],                         <L 509>
            var_513 = wp::mul(var_512, var_33);
            var_514 = wp::mul(var_513, var_458);
            var_515 = wp::mul(var_514, var_480);
            var_517 = wp::extract(var_420, var_516);
            var_518 = wp::abs(var_517);
            var_519 = wp::mul(var_515, var_518);
            var_521 = wp::extract(var_420, var_520);
            var_522 = wp::mul(var_519, var_521);
            // 0.5 * density * box0 * box1 * wp.abs(l_lin[2]) * l_lin[2],                         <L 510>
            var_524 = wp::mul(var_523, var_33);
            var_525 = wp::mul(var_524, var_458);
            var_526 = wp::mul(var_525, var_469);
            var_528 = wp::extract(var_420, var_527);
            var_529 = wp::abs(var_528);
            var_530 = wp::mul(var_526, var_529);
            var_532 = wp::extract(var_420, var_531);
            var_533 = wp::mul(var_530, var_532);
            var_534 = wp::vec_t<3, wp::float32>(var_511, var_522, var_533);
            // lfrc_force -= wp.vec3(                                                             <L 507>
            var_535 = wp::sub(var_500, var_534);
            // scl = density / 64.0                                                               <L 513>
            var_537 = wp::div(var_33, var_536);
            // box0_pow4 = wp.pow(box0, 4.0)                                                      <L 514>
            var_539 = wp::pow(var_458, var_538);
            // box1_pow4 = wp.pow(box1, 4.0)                                                      <L 515>
            var_541 = wp::pow(var_469, var_540);
            // box2_pow4 = wp.pow(box2, 4.0)                                                      <L 516>
            var_543 = wp::pow(var_480, var_542);
            // lfrc_torque -= wp.vec3(                                                            <L 517>
            // box0 * (box1_pow4 + box2_pow4) * wp.abs(l_ang[0]) * l_ang[0] * scl,                <L 518>
            var_544 = wp::add(var_541, var_543);
            var_545 = wp::mul(var_458, var_544);
            var_547 = wp::extract(var_409, var_546);
            var_548 = wp::abs(var_547);
            var_549 = wp::mul(var_545, var_548);
            var_551 = wp::extract(var_409, var_550);
            var_552 = wp::mul(var_549, var_551);
            var_553 = wp::mul(var_552, var_537);
            // box1 * (box0_pow4 + box2_pow4) * wp.abs(l_ang[1]) * l_ang[1] * scl,                <L 519>
            var_554 = wp::add(var_539, var_543);
            var_555 = wp::mul(var_469, var_554);
            var_557 = wp::extract(var_409, var_556);
            var_558 = wp::abs(var_557);
            var_559 = wp::mul(var_555, var_558);
            var_561 = wp::extract(var_409, var_560);
            var_562 = wp::mul(var_559, var_561);
            var_563 = wp::mul(var_562, var_537);
            // box2 * (box0_pow4 + box1_pow4) * wp.abs(l_ang[2]) * l_ang[2] * scl,                <L 520>
            var_564 = wp::add(var_539, var_541);
            var_565 = wp::mul(var_480, var_564);
            var_567 = wp::extract(var_409, var_566);
            var_568 = wp::abs(var_567);
            var_569 = wp::mul(var_565, var_568);
            var_571 = wp::extract(var_409, var_570);
            var_572 = wp::mul(var_569, var_571);
            var_573 = wp::mul(var_572, var_537);
            var_574 = wp::vec_t<3, wp::float32>(var_553, var_563, var_573);
            // lfrc_torque -= wp.vec3(                                                            <L 517>
            var_575 = wp::sub(var_499, var_574);
        }
        var_576 = wp::where(var_428, var_575, var_499);
        var_577 = wp::where(var_428, var_535, var_500);
        var_578 = wp::where(var_428, var_537, var_447);
        // torque_global = rot @ lfrc_torque                                                      <L 523>
        var_579 = wp::mul(var_47, var_576);
        // force_global = rot @ lfrc_force                                                        <L 524>
        var_580 = wp::mul(var_47, var_577);
        // fluid_applied_out[worldid, bodyid] = wp.spatial_vector(force_global, torque_global)       <L 526>
        var_581 = wp::vec_t<6, wp::float32>(var_580, var_579);
        wp::array_store(var_fluid_applied_out, var_0, var_1, var_581);
    }
}



extern "C" __global__ void _spring_damper_dof_passive_7b46d40c_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_opt_disableflags,
    wp::array_t<wp::float32> var_qpos_spring,
    wp::array_t<wp::int32> var_jnt_type,
    wp::array_t<wp::int32> var_jnt_qposadr,
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::float32> var_jnt_stiffness,
    wp::array_t<wp::vec_t<2, wp::float32>> var_jnt_stiffnesspoly,
    wp::array_t<wp::float32> var_dof_damping,
    wp::array_t<wp::vec_t<2, wp::float32>> var_dof_dampingpoly,
    wp::array_t<wp::float32> var_qpos_in,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::float32> var_qfrc_spring_out,
    wp::array_t<wp::float32> var_qfrc_damper_out)
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
        wp::shape_t* var_8;
        const wp::int32 var_9 = 0;
        wp::int32 var_10;
        wp::shape_t var_11;
        wp::int32 var_12;
        wp::float32* var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        wp::shape_t* var_16;
        const wp::int32 var_17 = 0;
        wp::int32 var_18;
        wp::shape_t var_19;
        wp::int32 var_20;
        wp::vec_t<2, wp::float32>* var_21;
        wp::vec_t<2, wp::float32> var_22;
        wp::vec_t<2, wp::float32> var_23;
        wp::shape_t* var_24;
        const wp::int32 var_25 = 0;
        wp::int32 var_26;
        wp::shape_t var_27;
        wp::int32 var_28;
        wp::float32* var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        wp::shape_t* var_32;
        const wp::int32 var_33 = 0;
        wp::int32 var_34;
        wp::shape_t var_35;
        wp::int32 var_36;
        wp::vec_t<2, wp::float32>* var_37;
        wp::vec_t<2, wp::float32> var_38;
        wp::vec_t<2, wp::float32> var_39;
        bool var_40;
        bool var_41;
        const wp::float32 var_42 = 0.0;
        bool var_43;
        const wp::int32 var_44 = 0;
        wp::float32 var_45;
        const wp::float32 var_46 = 0.0;
        bool var_47;
        const wp::int32 var_48 = 1;
        wp::float32 var_49;
        const wp::float32 var_50 = 0.0;
        bool var_51;
        const wp::int32 var_52 = 32;
        wp::int32 var_53;
        bool var_54;
        bool var_55;
        bool var_56;
        const wp::float32 var_57 = 0.0;
        bool var_58;
        const wp::int32 var_59 = 0;
        wp::float32 var_60;
        const wp::float32 var_61 = 0.0;
        bool var_62;
        const wp::int32 var_63 = 1;
        wp::float32 var_64;
        const wp::float32 var_65 = 0.0;
        bool var_66;
        const wp::int32 var_67 = 64;
        wp::int32 var_68;
        bool var_69;
        bool var_70;
        const wp::int32 var_71 = 0;
        bool var_72;
        const wp::int32 var_73 = 0;
        const wp::float32 var_74 = 0.0;
        wp::int32 var_75;
        const wp::int32 var_76 = 1;
        const wp::float32 var_77 = 0.0;
        wp::int32 var_78;
        const wp::int32 var_79 = 2;
        const wp::float32 var_80 = 0.0;
        wp::int32 var_81;
        const wp::int32 var_82 = 3;
        const wp::float32 var_83 = 0.0;
        wp::int32 var_84;
        const wp::int32 var_85 = 4;
        const wp::float32 var_86 = 0.0;
        wp::int32 var_87;
        const wp::int32 var_88 = 5;
        const wp::float32 var_89 = 0.0;
        wp::int32 var_90;
        const wp::int32 var_91 = 1;
        bool var_92;
        const wp::int32 var_93 = 0;
        const wp::float32 var_94 = 0.0;
        wp::int32 var_95;
        const wp::int32 var_96 = 1;
        const wp::float32 var_97 = 0.0;
        wp::int32 var_98;
        const wp::int32 var_99 = 2;
        const wp::float32 var_100 = 0.0;
        wp::int32 var_101;
        wp::int32 var_102;
        const wp::float32 var_103 = 0.0;
        wp::int32 var_104;
        bool var_105;
        const wp::int32 var_106 = 0;
        bool var_107;
        const wp::int32 var_108 = 0;
        const wp::float32 var_109 = 0.0;
        wp::int32 var_110;
        const wp::int32 var_111 = 1;
        const wp::float32 var_112 = 0.0;
        wp::int32 var_113;
        const wp::int32 var_114 = 2;
        const wp::float32 var_115 = 0.0;
        wp::int32 var_116;
        const wp::int32 var_117 = 3;
        const wp::float32 var_118 = 0.0;
        wp::int32 var_119;
        const wp::int32 var_120 = 4;
        const wp::float32 var_121 = 0.0;
        wp::int32 var_122;
        const wp::int32 var_123 = 5;
        const wp::float32 var_124 = 0.0;
        wp::int32 var_125;
        wp::int32 var_126;
        const wp::int32 var_127 = 1;
        bool var_128;
        const wp::int32 var_129 = 0;
        const wp::float32 var_130 = 0.0;
        wp::int32 var_131;
        const wp::int32 var_132 = 1;
        const wp::float32 var_133 = 0.0;
        wp::int32 var_134;
        const wp::int32 var_135 = 2;
        const wp::float32 var_136 = 0.0;
        wp::int32 var_137;
        wp::int32 var_138;
        const wp::float32 var_139 = 0.0;
        wp::int32 var_140;
        wp::int32 var_141;
        bool var_142;
        bool var_143;
        wp::int32* var_144;
        wp::int32 var_145;
        wp::int32 var_146;
        wp::shape_t* var_147;
        const wp::int32 var_148 = 0;
        wp::int32 var_149;
        wp::shape_t var_150;
        wp::int32 var_151;
        const wp::int32 var_152 = 0;
        bool var_153;
        const wp::int32 var_154 = 0;
        wp::int32 var_155;
        wp::float32* var_156;
        const wp::int32 var_157 = 0;
        wp::int32 var_158;
        wp::float32* var_159;
        wp::float32 var_160;
        wp::float32 var_161;
        wp::float32 var_162;
        const wp::int32 var_163 = 1;
        wp::int32 var_164;
        wp::float32* var_165;
        const wp::int32 var_166 = 1;
        wp::int32 var_167;
        wp::float32* var_168;
        wp::float32 var_169;
        wp::float32 var_170;
        wp::float32 var_171;
        const wp::int32 var_172 = 2;
        wp::int32 var_173;
        wp::float32* var_174;
        const wp::int32 var_175 = 2;
        wp::int32 var_176;
        wp::float32* var_177;
        wp::float32 var_178;
        wp::float32 var_179;
        wp::float32 var_180;
        wp::vec_t<3, wp::float32> var_181;
        wp::float32 var_182;
        const wp::int32 var_183 = 0;
        wp::float32 var_184;
        wp::float32 var_185;
        const wp::int32 var_186 = 0;
        wp::float32 var_187;
        wp::float32 var_188;
        const wp::int32 var_189 = 0;
        wp::int32 var_190;
        wp::float32 var_191;
        const wp::int32 var_192 = 1;
        wp::float32 var_193;
        wp::float32 var_194;
        const wp::int32 var_195 = 1;
        wp::int32 var_196;
        wp::float32 var_197;
        const wp::int32 var_198 = 2;
        wp::float32 var_199;
        wp::float32 var_200;
        const wp::int32 var_201 = 2;
        wp::int32 var_202;
        const wp::int32 var_203 = 3;
        wp::int32 var_204;
        wp::float32* var_205;
        const wp::int32 var_206 = 4;
        wp::int32 var_207;
        wp::float32* var_208;
        const wp::int32 var_209 = 5;
        wp::int32 var_210;
        wp::float32* var_211;
        const wp::int32 var_212 = 6;
        wp::int32 var_213;
        wp::float32* var_214;
        wp::quat_t<wp::float32> var_215;
        wp::float32 var_216;
        wp::float32 var_217;
        wp::float32 var_218;
        wp::float32 var_219;
        wp::quat_t<wp::float32> var_220;
        const wp::int32 var_221 = 3;
        wp::int32 var_222;
        wp::float32* var_223;
        const wp::int32 var_224 = 4;
        wp::int32 var_225;
        wp::float32* var_226;
        const wp::int32 var_227 = 5;
        wp::int32 var_228;
        wp::float32* var_229;
        const wp::int32 var_230 = 6;
        wp::int32 var_231;
        wp::float32* var_232;
        wp::quat_t<wp::float32> var_233;
        wp::float32 var_234;
        wp::float32 var_235;
        wp::float32 var_236;
        wp::float32 var_237;
        wp::vec_t<3, wp::float32> var_238;
        wp::float32 var_239;
        const wp::int32 var_240 = 0;
        wp::float32 var_241;
        wp::float32 var_242;
        const wp::int32 var_243 = 0;
        wp::float32 var_244;
        wp::float32 var_245;
        const wp::int32 var_246 = 3;
        wp::int32 var_247;
        wp::float32 var_248;
        const wp::int32 var_249 = 1;
        wp::float32 var_250;
        wp::float32 var_251;
        const wp::int32 var_252 = 4;
        wp::int32 var_253;
        wp::float32 var_254;
        const wp::int32 var_255 = 2;
        wp::float32 var_256;
        wp::float32 var_257;
        const wp::int32 var_258 = 5;
        wp::int32 var_259;
        const wp::int32 var_260 = 0;
        wp::int32 var_261;
        wp::float32* var_262;
        wp::float32 var_263;
        wp::float32 var_264;
        wp::float32 var_265;
        const wp::int32 var_266 = 1;
        wp::float32 var_267;
        wp::float32 var_268;
        wp::int32 var_269;
        const wp::int32 var_270 = 1;
        wp::int32 var_271;
        wp::float32* var_272;
        wp::float32 var_273;
        wp::float32 var_274;
        wp::float32 var_275;
        const wp::int32 var_276 = 1;
        wp::float32 var_277;
        wp::float32 var_278;
        wp::int32 var_279;
        const wp::int32 var_280 = 2;
        wp::int32 var_281;
        wp::float32* var_282;
        wp::float32 var_283;
        wp::float32 var_284;
        wp::float32 var_285;
        const wp::int32 var_286 = 1;
        wp::float32 var_287;
        wp::float32 var_288;
        wp::int32 var_289;
        const wp::int32 var_290 = 3;
        wp::int32 var_291;
        wp::float32* var_292;
        wp::float32 var_293;
        wp::float32 var_294;
        wp::float32 var_295;
        const wp::int32 var_296 = 1;
        wp::float32 var_297;
        wp::float32 var_298;
        wp::int32 var_299;
        const wp::int32 var_300 = 4;
        wp::int32 var_301;
        wp::float32* var_302;
        wp::float32 var_303;
        wp::float32 var_304;
        wp::float32 var_305;
        const wp::int32 var_306 = 1;
        wp::float32 var_307;
        wp::float32 var_308;
        wp::int32 var_309;
        const wp::int32 var_310 = 5;
        wp::int32 var_311;
        wp::float32* var_312;
        wp::float32 var_313;
        wp::float32 var_314;
        wp::float32 var_315;
        const wp::int32 var_316 = 1;
        wp::float32 var_317;
        wp::float32 var_318;
        wp::int32 var_319;
        wp::int32 var_320;
        wp::int32 var_321;
        const wp::int32 var_322 = 1;
        bool var_323;
        const wp::int32 var_324 = 0;
        wp::int32 var_325;
        wp::float32* var_326;
        const wp::int32 var_327 = 1;
        wp::int32 var_328;
        wp::float32* var_329;
        const wp::int32 var_330 = 2;
        wp::int32 var_331;
        wp::float32* var_332;
        const wp::int32 var_333 = 3;
        wp::int32 var_334;
        wp::float32* var_335;
        wp::quat_t<wp::float32> var_336;
        wp::float32 var_337;
        wp::float32 var_338;
        wp::float32 var_339;
        wp::float32 var_340;
        wp::quat_t<wp::float32> var_341;
        const wp::int32 var_342 = 0;
        wp::int32 var_343;
        wp::float32* var_344;
        const wp::int32 var_345 = 1;
        wp::int32 var_346;
        wp::float32* var_347;
        const wp::int32 var_348 = 2;
        wp::int32 var_349;
        wp::float32* var_350;
        const wp::int32 var_351 = 3;
        wp::int32 var_352;
        wp::float32* var_353;
        wp::quat_t<wp::float32> var_354;
        wp::float32 var_355;
        wp::float32 var_356;
        wp::float32 var_357;
        wp::float32 var_358;
        wp::vec_t<3, wp::float32> var_359;
        wp::float32 var_360;
        const wp::int32 var_361 = 0;
        wp::float32 var_362;
        wp::float32 var_363;
        const wp::int32 var_364 = 0;
        wp::float32 var_365;
        wp::float32 var_366;
        const wp::int32 var_367 = 0;
        wp::int32 var_368;
        wp::float32 var_369;
        const wp::int32 var_370 = 1;
        wp::float32 var_371;
        wp::float32 var_372;
        const wp::int32 var_373 = 1;
        wp::int32 var_374;
        wp::float32 var_375;
        const wp::int32 var_376 = 2;
        wp::float32 var_377;
        wp::float32 var_378;
        const wp::int32 var_379 = 2;
        wp::int32 var_380;
        wp::vec_t<3, wp::float32> var_381;
        wp::float32 var_382;
        wp::float32 var_383;
        wp::quat_t<wp::float32> var_384;
        wp::quat_t<wp::float32> var_385;
        const wp::int32 var_386 = 0;
        wp::int32 var_387;
        wp::float32* var_388;
        wp::float32 var_389;
        wp::float32 var_390;
        wp::float32 var_391;
        const wp::int32 var_392 = 1;
        wp::float32 var_393;
        wp::float32 var_394;
        wp::int32 var_395;
        const wp::int32 var_396 = 1;
        wp::int32 var_397;
        wp::float32* var_398;
        wp::float32 var_399;
        wp::float32 var_400;
        wp::float32 var_401;
        const wp::int32 var_402 = 1;
        wp::float32 var_403;
        wp::float32 var_404;
        wp::int32 var_405;
        const wp::int32 var_406 = 2;
        wp::int32 var_407;
        wp::float32* var_408;
        wp::float32 var_409;
        wp::float32 var_410;
        wp::float32 var_411;
        const wp::int32 var_412 = 1;
        wp::float32 var_413;
        wp::float32 var_414;
        wp::int32 var_415;
        wp::int32 var_416;
        wp::float32 var_417;
        wp::int32 var_418;
        wp::vec_t<3, wp::float32> var_419;
        wp::float32 var_420;
        wp::float32 var_421;
        wp::quat_t<wp::float32> var_422;
        wp::quat_t<wp::float32> var_423;
        wp::float32 var_424;
        wp::float32* var_425;
        wp::float32* var_426;
        wp::float32 var_427;
        wp::float32 var_428;
        wp::float32 var_429;
        wp::float32 var_430;
        const wp::int32 var_431 = 0;
        wp::float32 var_432;
        wp::float32 var_433;
        wp::float32* var_434;
        wp::float32 var_435;
        wp::float32 var_436;
        wp::float32 var_437;
        const wp::int32 var_438 = 1;
        wp::float32 var_439;
        wp::float32 var_440;
        wp::float32 var_441;
        wp::float32 var_442;
        wp::int32 var_443;
        wp::vec_t<3, wp::float32> var_444;
        wp::float32 var_445;
        wp::float32 var_446;
        wp::quat_t<wp::float32> var_447;
        wp::quat_t<wp::float32> var_448;
        wp::float32 var_449;
        //---------
        // forward
        // def _spring_damper_dof_passive(                                                        <L 73>
        // worldid, jntid = wp.tid()                                                              <L 91>
        builtin_tid2d(var_0, var_1);
        // dofid = jnt_dofadr[jntid]                                                              <L 92>
        var_2 = wp::address(var_jnt_dofadr, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // jnttype = jnt_type[jntid]                                                              <L 93>
        var_5 = wp::address(var_jnt_type, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // stiffness = jnt_stiffness[worldid % jnt_stiffness.shape[0], jntid]                     <L 94>
        var_8 = &(var_jnt_stiffness.shape);
        var_11 = wp::load(var_8);
        var_10 = wp::extract(var_11, var_9);
        var_12 = wp::mod(var_0, var_10);
        var_13 = wp::address(var_jnt_stiffness, var_12, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // spoly = jnt_stiffnesspoly[worldid % jnt_stiffnesspoly.shape[0], jntid]                 <L 95>
        var_16 = &(var_jnt_stiffnesspoly.shape);
        var_19 = wp::load(var_16);
        var_18 = wp::extract(var_19, var_17);
        var_20 = wp::mod(var_0, var_18);
        var_21 = wp::address(var_jnt_stiffnesspoly, var_20, var_1);
        var_23 = wp::load(var_21);
        var_22 = wp::copy(var_23);
        // damping = dof_damping[worldid % dof_damping.shape[0], dofid]                           <L 96>
        var_24 = &(var_dof_damping.shape);
        var_27 = wp::load(var_24);
        var_26 = wp::extract(var_27, var_25);
        var_28 = wp::mod(var_0, var_26);
        var_29 = wp::address(var_dof_damping, var_28, var_3);
        var_31 = wp::load(var_29);
        var_30 = wp::copy(var_31);
        // dpoly = dof_dampingpoly[worldid % dof_dampingpoly.shape[0], dofid]                     <L 97>
        var_32 = &(var_dof_dampingpoly.shape);
        var_35 = wp::load(var_32);
        var_34 = wp::extract(var_35, var_33);
        var_36 = wp::mod(var_0, var_34);
        var_37 = wp::address(var_dof_dampingpoly, var_36, var_3);
        var_39 = wp::load(var_37);
        var_38 = wp::copy(var_39);
        // has_stiffness = (stiffness != 0.0 or spoly[0] != 0.0 or spoly[1] != 0.0) and not (opt_disableflags & DisableBit.SPRING)       <L 99>
        var_43 = (var_14 != var_42);
        var_41 = var_43;
        if (!var_41) {
            var_45 = wp::extract(var_22, var_44);
            var_47 = (var_45 != var_46);
            var_41 = var_41 || var_47;
        }
        if (!var_41) {
            var_49 = wp::extract(var_22, var_48);
            var_51 = (var_49 != var_50);
            var_41 = var_41 || var_51;
        }
        var_40 = var_41;
        if (var_40) {
            var_53 = wp::bit_and(var_opt_disableflags, var_52);
            var_54 = wp::unot(var_53);
            var_40 = var_40 && var_54;
        }
        // has_damping = (damping != 0.0 or dpoly[0] != 0.0 or dpoly[1] != 0.0) and not (opt_disableflags & DisableBit.DAMPER)       <L 100>
        var_58 = (var_30 != var_57);
        var_56 = var_58;
        if (!var_56) {
            var_60 = wp::extract(var_38, var_59);
            var_62 = (var_60 != var_61);
            var_56 = var_56 || var_62;
        }
        if (!var_56) {
            var_64 = wp::extract(var_38, var_63);
            var_66 = (var_64 != var_65);
            var_56 = var_56 || var_66;
        }
        var_55 = var_56;
        if (var_55) {
            var_68 = wp::bit_and(var_opt_disableflags, var_67);
            var_69 = wp::unot(var_68);
            var_55 = var_55 && var_69;
        }
        // if not has_stiffness:                                                                  <L 102>
        var_70 = wp::unot(var_40);
        if (var_70) {
            // if jnttype == JointType.FREE:                                                      <L 103>
            var_72 = (var_6 == var_71);
            if (var_72) {
                // for i in range(6):                                                             <L 104>
                // qfrc_spring_out[worldid, dofid + i] = 0.0                                      <L 105>
                var_75 = wp::add(var_3, var_73);
                wp::array_store(var_qfrc_spring_out, var_0, var_75, var_74);
                var_78 = wp::add(var_3, var_76);
                wp::array_store(var_qfrc_spring_out, var_0, var_78, var_77);
                var_81 = wp::add(var_3, var_79);
                wp::array_store(var_qfrc_spring_out, var_0, var_81, var_80);
                var_84 = wp::add(var_3, var_82);
                wp::array_store(var_qfrc_spring_out, var_0, var_84, var_83);
                var_87 = wp::add(var_3, var_85);
                wp::array_store(var_qfrc_spring_out, var_0, var_87, var_86);
                var_90 = wp::add(var_3, var_88);
                wp::array_store(var_qfrc_spring_out, var_0, var_90, var_89);
            }
            if (!var_72) {
                // elif jnttype == JointType.BALL:                                                <L 106>
                var_92 = (var_6 == var_91);
                if (var_92) {
                    // for i in range(3):                                                         <L 107>
                    // qfrc_spring_out[worldid, dofid + i] = 0.0                                  <L 108>
                    var_95 = wp::add(var_3, var_93);
                    wp::array_store(var_qfrc_spring_out, var_0, var_95, var_94);
                    var_98 = wp::add(var_3, var_96);
                    wp::array_store(var_qfrc_spring_out, var_0, var_98, var_97);
                    var_101 = wp::add(var_3, var_99);
                    wp::array_store(var_qfrc_spring_out, var_0, var_101, var_100);
                }
                var_102 = wp::where(var_92, var_99, var_88);
                if (!var_92) {
                    // qfrc_spring_out[worldid, dofid] = 0.0                                      <L 110>
                    wp::array_store(var_qfrc_spring_out, var_0, var_3, var_103);
                }
            }
            var_104 = wp::where(var_72, var_88, var_102);
        }
        // if not has_damping:                                                                    <L 112>
        var_105 = wp::unot(var_55);
        if (var_105) {
            // if jnttype == JointType.FREE:                                                      <L 113>
            var_107 = (var_6 == var_106);
            if (var_107) {
                // for i in range(6):                                                             <L 114>
                // qfrc_damper_out[worldid, dofid + i] = 0.0                                      <L 115>
                var_110 = wp::add(var_3, var_108);
                wp::array_store(var_qfrc_damper_out, var_0, var_110, var_109);
                var_113 = wp::add(var_3, var_111);
                wp::array_store(var_qfrc_damper_out, var_0, var_113, var_112);
                var_116 = wp::add(var_3, var_114);
                wp::array_store(var_qfrc_damper_out, var_0, var_116, var_115);
                var_119 = wp::add(var_3, var_117);
                wp::array_store(var_qfrc_damper_out, var_0, var_119, var_118);
                var_122 = wp::add(var_3, var_120);
                wp::array_store(var_qfrc_damper_out, var_0, var_122, var_121);
                var_125 = wp::add(var_3, var_123);
                wp::array_store(var_qfrc_damper_out, var_0, var_125, var_124);
            }
            var_126 = wp::where(var_107, var_123, var_104);
            if (!var_107) {
                // elif jnttype == JointType.BALL:                                                <L 116>
                var_128 = (var_6 == var_127);
                if (var_128) {
                    // for i in range(3):                                                         <L 117>
                    // qfrc_damper_out[worldid, dofid + i] = 0.0                                  <L 118>
                    var_131 = wp::add(var_3, var_129);
                    wp::array_store(var_qfrc_damper_out, var_0, var_131, var_130);
                    var_134 = wp::add(var_3, var_132);
                    wp::array_store(var_qfrc_damper_out, var_0, var_134, var_133);
                    var_137 = wp::add(var_3, var_135);
                    wp::array_store(var_qfrc_damper_out, var_0, var_137, var_136);
                }
                var_138 = wp::where(var_128, var_135, var_126);
                if (!var_128) {
                    // qfrc_damper_out[worldid, dofid] = 0.0                                      <L 120>
                    wp::array_store(var_qfrc_damper_out, var_0, var_3, var_139);
                }
            }
            var_140 = wp::where(var_107, var_126, var_138);
        }
        var_141 = wp::where(var_105, var_140, var_104);
        // if not (has_stiffness or has_damping):                                                 <L 122>
        var_142 = var_40;
        if (!var_142) {
            var_142 = var_142 || var_55;
        }
        var_143 = wp::unot(var_142);
        if (var_143) {
            // return                                                                             <L 123>
            continue;
        }
        // qposid = jnt_qposadr[jntid]                                                            <L 124>
        var_144 = wp::address(var_jnt_qposadr, var_1);
        var_146 = wp::load(var_144);
        var_145 = wp::copy(var_146);
        // qpos_spring_id = worldid % qpos_spring.shape[0]                                        <L 125>
        var_147 = &(var_qpos_spring.shape);
        var_150 = wp::load(var_147);
        var_149 = wp::extract(var_150, var_148);
        var_151 = wp::mod(var_0, var_149);
        // if jnttype == JointType.FREE:                                                          <L 127>
        var_153 = (var_6 == var_152);
        if (var_153) {
            // if has_stiffness:                                                                  <L 129>
            if (var_40) {
                // dif = wp.vec3(                                                                 <L 130>
                // qpos_in[worldid, qposid + 0] - qpos_spring[qpos_spring_id, qposid + 0],        <L 131>
                var_155 = wp::add(var_145, var_154);
                var_156 = wp::address(var_qpos_in, var_0, var_155);
                var_158 = wp::add(var_145, var_157);
                var_159 = wp::address(var_qpos_spring, var_151, var_158);
                var_161 = wp::load(var_156);
                var_162 = wp::load(var_159);
                var_160 = wp::sub(var_161, var_162);
                // qpos_in[worldid, qposid + 1] - qpos_spring[qpos_spring_id, qposid + 1],        <L 132>
                var_164 = wp::add(var_145, var_163);
                var_165 = wp::address(var_qpos_in, var_0, var_164);
                var_167 = wp::add(var_145, var_166);
                var_168 = wp::address(var_qpos_spring, var_151, var_167);
                var_170 = wp::load(var_165);
                var_171 = wp::load(var_168);
                var_169 = wp::sub(var_170, var_171);
                // qpos_in[worldid, qposid + 2] - qpos_spring[qpos_spring_id, qposid + 2],        <L 133>
                var_173 = wp::add(var_145, var_172);
                var_174 = wp::address(var_qpos_in, var_0, var_173);
                var_176 = wp::add(var_145, var_175);
                var_177 = wp::address(var_qpos_spring, var_151, var_176);
                var_179 = wp::load(var_174);
                var_180 = wp::load(var_177);
                var_178 = wp::sub(var_179, var_180);
                var_181 = wp::vec_t<3, wp::float32>(var_160, var_169, var_178);
                // r = wp.length(dif)                                                             <L 135>
                var_182 = wp::length(var_181);
                // k = util_misc._poly_force(stiffness, spoly, r, 0)                              <L 136>
                var_184 = _poly_force_0(var_14, var_22, var_182, var_183);
                // qfrc_spring_out[worldid, dofid + 0] = -k * dif[0]                              <L 137>
                var_185 = wp::neg(var_184);
                var_187 = wp::extract(var_181, var_186);
                var_188 = wp::mul(var_185, var_187);
                var_190 = wp::add(var_3, var_189);
                wp::array_store(var_qfrc_spring_out, var_0, var_190, var_188);
                // qfrc_spring_out[worldid, dofid + 1] = -k * dif[1]                              <L 138>
                var_191 = wp::neg(var_184);
                var_193 = wp::extract(var_181, var_192);
                var_194 = wp::mul(var_191, var_193);
                var_196 = wp::add(var_3, var_195);
                wp::array_store(var_qfrc_spring_out, var_0, var_196, var_194);
                // qfrc_spring_out[worldid, dofid + 2] = -k * dif[2]                              <L 139>
                var_197 = wp::neg(var_184);
                var_199 = wp::extract(var_181, var_198);
                var_200 = wp::mul(var_197, var_199);
                var_202 = wp::add(var_3, var_201);
                wp::array_store(var_qfrc_spring_out, var_0, var_202, var_200);
                // rot = wp.quat(                                                                 <L 141>
                // qpos_in[worldid, qposid + 3],                                                  <L 142>
                var_204 = wp::add(var_145, var_203);
                var_205 = wp::address(var_qpos_in, var_0, var_204);
                // qpos_in[worldid, qposid + 4],                                                  <L 143>
                var_207 = wp::add(var_145, var_206);
                var_208 = wp::address(var_qpos_in, var_0, var_207);
                // qpos_in[worldid, qposid + 5],                                                  <L 144>
                var_210 = wp::add(var_145, var_209);
                var_211 = wp::address(var_qpos_in, var_0, var_210);
                // qpos_in[worldid, qposid + 6],                                                  <L 145>
                var_213 = wp::add(var_145, var_212);
                var_214 = wp::address(var_qpos_in, var_0, var_213);
                var_216 = wp::load(var_205);
                var_217 = wp::load(var_208);
                var_218 = wp::load(var_211);
                var_219 = wp::load(var_214);
                var_215 = wp::quat_t<wp::float32>(var_216, var_217, var_218, var_219);
                // rot = wp.normalize(rot)                                                        <L 147>
                var_220 = wp::normalize(var_215);
                // ref = wp.quat(                                                                 <L 148>
                // qpos_spring[qpos_spring_id, qposid + 3],                                       <L 149>
                var_222 = wp::add(var_145, var_221);
                var_223 = wp::address(var_qpos_spring, var_151, var_222);
                // qpos_spring[qpos_spring_id, qposid + 4],                                       <L 150>
                var_225 = wp::add(var_145, var_224);
                var_226 = wp::address(var_qpos_spring, var_151, var_225);
                // qpos_spring[qpos_spring_id, qposid + 5],                                       <L 151>
                var_228 = wp::add(var_145, var_227);
                var_229 = wp::address(var_qpos_spring, var_151, var_228);
                // qpos_spring[qpos_spring_id, qposid + 6],                                       <L 152>
                var_231 = wp::add(var_145, var_230);
                var_232 = wp::address(var_qpos_spring, var_151, var_231);
                var_234 = wp::load(var_223);
                var_235 = wp::load(var_226);
                var_236 = wp::load(var_229);
                var_237 = wp::load(var_232);
                var_233 = wp::quat_t<wp::float32>(var_234, var_235, var_236, var_237);
                // dif = math.quat_sub(rot, ref)                                                  <L 154>
                var_238 = quat_sub_0(var_220, var_233);
                // r_rot = wp.length(dif)                                                         <L 155>
                var_239 = wp::length(var_238);
                // k_rot = util_misc._poly_force(stiffness, spoly, r_rot, 0)                      <L 156>
                var_241 = _poly_force_0(var_14, var_22, var_239, var_240);
                // qfrc_spring_out[worldid, dofid + 3] = -k_rot * dif[0]                          <L 157>
                var_242 = wp::neg(var_241);
                var_244 = wp::extract(var_238, var_243);
                var_245 = wp::mul(var_242, var_244);
                var_247 = wp::add(var_3, var_246);
                wp::array_store(var_qfrc_spring_out, var_0, var_247, var_245);
                // qfrc_spring_out[worldid, dofid + 4] = -k_rot * dif[1]                          <L 158>
                var_248 = wp::neg(var_241);
                var_250 = wp::extract(var_238, var_249);
                var_251 = wp::mul(var_248, var_250);
                var_253 = wp::add(var_3, var_252);
                wp::array_store(var_qfrc_spring_out, var_0, var_253, var_251);
                // qfrc_spring_out[worldid, dofid + 5] = -k_rot * dif[2]                          <L 159>
                var_254 = wp::neg(var_241);
                var_256 = wp::extract(var_238, var_255);
                var_257 = wp::mul(var_254, var_256);
                var_259 = wp::add(var_3, var_258);
                wp::array_store(var_qfrc_spring_out, var_0, var_259, var_257);
            }
            // if has_damping:                                                                    <L 162>
            if (var_55) {
                // for i in range(6):                                                             <L 163>
                // v = qvel_in[worldid, dofid + i]                                                <L 164>
                var_261 = wp::add(var_3, var_260);
                var_262 = wp::address(var_qvel_in, var_0, var_261);
                var_264 = wp::load(var_262);
                var_263 = wp::copy(var_264);
                // qfrc_damper_out[worldid, dofid + i] = -v * util_misc._poly_force(damping, dpoly, v, 1)       <L 165>
                var_265 = wp::neg(var_263);
                var_267 = _poly_force_0(var_30, var_38, var_263, var_266);
                var_268 = wp::mul(var_265, var_267);
                var_269 = wp::add(var_3, var_260);
                wp::array_store(var_qfrc_damper_out, var_0, var_269, var_268);
                // v = qvel_in[worldid, dofid + i]                                                <L 164>
                var_271 = wp::add(var_3, var_270);
                var_272 = wp::address(var_qvel_in, var_0, var_271);
                var_274 = wp::load(var_272);
                var_273 = wp::copy(var_274);
                // qfrc_damper_out[worldid, dofid + i] = -v * util_misc._poly_force(damping, dpoly, v, 1)       <L 165>
                var_275 = wp::neg(var_273);
                var_277 = _poly_force_0(var_30, var_38, var_273, var_276);
                var_278 = wp::mul(var_275, var_277);
                var_279 = wp::add(var_3, var_270);
                wp::array_store(var_qfrc_damper_out, var_0, var_279, var_278);
                // v = qvel_in[worldid, dofid + i]                                                <L 164>
                var_281 = wp::add(var_3, var_280);
                var_282 = wp::address(var_qvel_in, var_0, var_281);
                var_284 = wp::load(var_282);
                var_283 = wp::copy(var_284);
                // qfrc_damper_out[worldid, dofid + i] = -v * util_misc._poly_force(damping, dpoly, v, 1)       <L 165>
                var_285 = wp::neg(var_283);
                var_287 = _poly_force_0(var_30, var_38, var_283, var_286);
                var_288 = wp::mul(var_285, var_287);
                var_289 = wp::add(var_3, var_280);
                wp::array_store(var_qfrc_damper_out, var_0, var_289, var_288);
                // v = qvel_in[worldid, dofid + i]                                                <L 164>
                var_291 = wp::add(var_3, var_290);
                var_292 = wp::address(var_qvel_in, var_0, var_291);
                var_294 = wp::load(var_292);
                var_293 = wp::copy(var_294);
                // qfrc_damper_out[worldid, dofid + i] = -v * util_misc._poly_force(damping, dpoly, v, 1)       <L 165>
                var_295 = wp::neg(var_293);
                var_297 = _poly_force_0(var_30, var_38, var_293, var_296);
                var_298 = wp::mul(var_295, var_297);
                var_299 = wp::add(var_3, var_290);
                wp::array_store(var_qfrc_damper_out, var_0, var_299, var_298);
                // v = qvel_in[worldid, dofid + i]                                                <L 164>
                var_301 = wp::add(var_3, var_300);
                var_302 = wp::address(var_qvel_in, var_0, var_301);
                var_304 = wp::load(var_302);
                var_303 = wp::copy(var_304);
                // qfrc_damper_out[worldid, dofid + i] = -v * util_misc._poly_force(damping, dpoly, v, 1)       <L 165>
                var_305 = wp::neg(var_303);
                var_307 = _poly_force_0(var_30, var_38, var_303, var_306);
                var_308 = wp::mul(var_305, var_307);
                var_309 = wp::add(var_3, var_300);
                wp::array_store(var_qfrc_damper_out, var_0, var_309, var_308);
                // v = qvel_in[worldid, dofid + i]                                                <L 164>
                var_311 = wp::add(var_3, var_310);
                var_312 = wp::address(var_qvel_in, var_0, var_311);
                var_314 = wp::load(var_312);
                var_313 = wp::copy(var_314);
                // qfrc_damper_out[worldid, dofid + i] = -v * util_misc._poly_force(damping, dpoly, v, 1)       <L 165>
                var_315 = wp::neg(var_313);
                var_317 = _poly_force_0(var_30, var_38, var_313, var_316);
                var_318 = wp::mul(var_315, var_317);
                var_319 = wp::add(var_3, var_310);
                wp::array_store(var_qfrc_damper_out, var_0, var_319, var_318);
            }
            var_320 = wp::where(var_55, var_310, var_141);
        }
        var_321 = wp::where(var_153, var_320, var_141);
        if (!var_153) {
            // elif jnttype == JointType.BALL:                                                    <L 167>
            var_323 = (var_6 == var_322);
            if (var_323) {
                // if has_stiffness:                                                              <L 169>
                if (var_40) {
                    // rot = wp.quat(                                                             <L 170>
                    // qpos_in[worldid, qposid + 0],                                              <L 171>
                    var_325 = wp::add(var_145, var_324);
                    var_326 = wp::address(var_qpos_in, var_0, var_325);
                    // qpos_in[worldid, qposid + 1],                                              <L 172>
                    var_328 = wp::add(var_145, var_327);
                    var_329 = wp::address(var_qpos_in, var_0, var_328);
                    // qpos_in[worldid, qposid + 2],                                              <L 173>
                    var_331 = wp::add(var_145, var_330);
                    var_332 = wp::address(var_qpos_in, var_0, var_331);
                    // qpos_in[worldid, qposid + 3],                                              <L 174>
                    var_334 = wp::add(var_145, var_333);
                    var_335 = wp::address(var_qpos_in, var_0, var_334);
                    var_337 = wp::load(var_326);
                    var_338 = wp::load(var_329);
                    var_339 = wp::load(var_332);
                    var_340 = wp::load(var_335);
                    var_336 = wp::quat_t<wp::float32>(var_337, var_338, var_339, var_340);
                    // rot = wp.normalize(rot)                                                    <L 176>
                    var_341 = wp::normalize(var_336);
                    // ref = wp.quat(                                                             <L 177>
                    // qpos_spring[qpos_spring_id, qposid + 0],                                   <L 178>
                    var_343 = wp::add(var_145, var_342);
                    var_344 = wp::address(var_qpos_spring, var_151, var_343);
                    // qpos_spring[qpos_spring_id, qposid + 1],                                   <L 179>
                    var_346 = wp::add(var_145, var_345);
                    var_347 = wp::address(var_qpos_spring, var_151, var_346);
                    // qpos_spring[qpos_spring_id, qposid + 2],                                   <L 180>
                    var_349 = wp::add(var_145, var_348);
                    var_350 = wp::address(var_qpos_spring, var_151, var_349);
                    // qpos_spring[qpos_spring_id, qposid + 3],                                   <L 181>
                    var_352 = wp::add(var_145, var_351);
                    var_353 = wp::address(var_qpos_spring, var_151, var_352);
                    var_355 = wp::load(var_344);
                    var_356 = wp::load(var_347);
                    var_357 = wp::load(var_350);
                    var_358 = wp::load(var_353);
                    var_354 = wp::quat_t<wp::float32>(var_355, var_356, var_357, var_358);
                    // dif = math.quat_sub(rot, ref)                                              <L 183>
                    var_359 = quat_sub_0(var_341, var_354);
                    // r = wp.length(dif)                                                         <L 184>
                    var_360 = wp::length(var_359);
                    // k = util_misc._poly_force(stiffness, spoly, r, 0)                          <L 185>
                    var_362 = _poly_force_0(var_14, var_22, var_360, var_361);
                    // qfrc_spring_out[worldid, dofid + 0] = -k * dif[0]                          <L 186>
                    var_363 = wp::neg(var_362);
                    var_365 = wp::extract(var_359, var_364);
                    var_366 = wp::mul(var_363, var_365);
                    var_368 = wp::add(var_3, var_367);
                    wp::array_store(var_qfrc_spring_out, var_0, var_368, var_366);
                    // qfrc_spring_out[worldid, dofid + 1] = -k * dif[1]                          <L 187>
                    var_369 = wp::neg(var_362);
                    var_371 = wp::extract(var_359, var_370);
                    var_372 = wp::mul(var_369, var_371);
                    var_374 = wp::add(var_3, var_373);
                    wp::array_store(var_qfrc_spring_out, var_0, var_374, var_372);
                    // qfrc_spring_out[worldid, dofid + 2] = -k * dif[2]                          <L 188>
                    var_375 = wp::neg(var_362);
                    var_377 = wp::extract(var_359, var_376);
                    var_378 = wp::mul(var_375, var_377);
                    var_380 = wp::add(var_3, var_379);
                    wp::array_store(var_qfrc_spring_out, var_0, var_380, var_378);
                }
                var_381 = wp::where(var_40, var_359, var_238);
                var_382 = wp::where(var_40, var_360, var_182);
                var_383 = wp::where(var_40, var_362, var_184);
                var_384 = wp::where(var_40, var_341, var_220);
                var_385 = wp::where(var_40, var_354, var_233);
                // if has_damping:                                                                <L 191>
                if (var_55) {
                    // for i in range(3):                                                         <L 192>
                    // v = qvel_in[worldid, dofid + i]                                            <L 193>
                    var_387 = wp::add(var_3, var_386);
                    var_388 = wp::address(var_qvel_in, var_0, var_387);
                    var_390 = wp::load(var_388);
                    var_389 = wp::copy(var_390);
                    // qfrc_damper_out[worldid, dofid + i] = -v * util_misc._poly_force(damping, dpoly, v, 1)       <L 194>
                    var_391 = wp::neg(var_389);
                    var_393 = _poly_force_0(var_30, var_38, var_389, var_392);
                    var_394 = wp::mul(var_391, var_393);
                    var_395 = wp::add(var_3, var_386);
                    wp::array_store(var_qfrc_damper_out, var_0, var_395, var_394);
                    // v = qvel_in[worldid, dofid + i]                                            <L 193>
                    var_397 = wp::add(var_3, var_396);
                    var_398 = wp::address(var_qvel_in, var_0, var_397);
                    var_400 = wp::load(var_398);
                    var_399 = wp::copy(var_400);
                    // qfrc_damper_out[worldid, dofid + i] = -v * util_misc._poly_force(damping, dpoly, v, 1)       <L 194>
                    var_401 = wp::neg(var_399);
                    var_403 = _poly_force_0(var_30, var_38, var_399, var_402);
                    var_404 = wp::mul(var_401, var_403);
                    var_405 = wp::add(var_3, var_396);
                    wp::array_store(var_qfrc_damper_out, var_0, var_405, var_404);
                    // v = qvel_in[worldid, dofid + i]                                            <L 193>
                    var_407 = wp::add(var_3, var_406);
                    var_408 = wp::address(var_qvel_in, var_0, var_407);
                    var_410 = wp::load(var_408);
                    var_409 = wp::copy(var_410);
                    // qfrc_damper_out[worldid, dofid + i] = -v * util_misc._poly_force(damping, dpoly, v, 1)       <L 194>
                    var_411 = wp::neg(var_409);
                    var_413 = _poly_force_0(var_30, var_38, var_409, var_412);
                    var_414 = wp::mul(var_411, var_413);
                    var_415 = wp::add(var_3, var_406);
                    wp::array_store(var_qfrc_damper_out, var_0, var_415, var_414);
                }
                var_416 = wp::where(var_55, var_406, var_321);
                var_417 = wp::where(var_55, var_409, var_313);
            }
            var_418 = wp::where(var_323, var_416, var_321);
            var_419 = wp::where(var_323, var_381, var_238);
            var_420 = wp::where(var_323, var_382, var_182);
            var_421 = wp::where(var_323, var_383, var_184);
            var_422 = wp::where(var_323, var_384, var_220);
            var_423 = wp::where(var_323, var_385, var_233);
            var_424 = wp::where(var_323, var_417, var_313);
            if (!var_323) {
                // if has_stiffness:                                                              <L 198>
                if (var_40) {
                    // fdif = qpos_in[worldid, qposid] - qpos_spring[qpos_spring_id, qposid]       <L 199>
                    var_425 = wp::address(var_qpos_in, var_0, var_145);
                    var_426 = wp::address(var_qpos_spring, var_151, var_145);
                    var_428 = wp::load(var_425);
                    var_429 = wp::load(var_426);
                    var_427 = wp::sub(var_428, var_429);
                    // qfrc_spring_out[worldid, dofid] = -fdif * util_misc._poly_force(stiffness, spoly, fdif, 0)       <L 200>
                    var_430 = wp::neg(var_427);
                    var_432 = _poly_force_0(var_14, var_22, var_427, var_431);
                    var_433 = wp::mul(var_430, var_432);
                    wp::array_store(var_qfrc_spring_out, var_0, var_3, var_433);
                }
                // if has_damping:                                                                <L 203>
                if (var_55) {
                    // v = qvel_in[worldid, dofid]                                                <L 204>
                    var_434 = wp::address(var_qvel_in, var_0, var_3);
                    var_436 = wp::load(var_434);
                    var_435 = wp::copy(var_436);
                    // qfrc_damper_out[worldid, dofid] = -v * util_misc._poly_force(damping, dpoly, v, 1)       <L 205>
                    var_437 = wp::neg(var_435);
                    var_439 = _poly_force_0(var_30, var_38, var_435, var_438);
                    var_440 = wp::mul(var_437, var_439);
                    wp::array_store(var_qfrc_damper_out, var_0, var_3, var_440);
                }
                var_441 = wp::where(var_55, var_435, var_424);
            }
            var_442 = wp::where(var_323, var_424, var_441);
        }
        var_443 = wp::where(var_153, var_321, var_418);
        var_444 = wp::where(var_153, var_238, var_419);
        var_445 = wp::where(var_153, var_182, var_420);
        var_446 = wp::where(var_153, var_184, var_421);
        var_447 = wp::where(var_153, var_220, var_422);
        var_448 = wp::where(var_153, var_233, var_423);
        var_449 = wp::where(var_153, var_313, var_442);
    }
}



extern "C" __global__ void _flex_passive_interp_38c71a67_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_nflex,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_flex_interp,
    wp::array_t<wp::vec_t<3, wp::int32>> var_flex_cellnum,
    wp::array_t<wp::int32> var_flex_nodeadr,
    wp::array_t<wp::int32> var_flex_stiffnessadr,
    wp::array_t<wp::int32> var_flex_nodebodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flex_node,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flex_node0,
    wp::array_t<wp::float32> var_flex_stiffness,
    wp::array_t<wp::float32> var_flex_damping,
    wp::array_t<wp::int32> var_flex_edgeequality,
    wp::array_t<bool> var_flex_centered,
    wp::array_t<wp::vec_t<4, wp::int32>> var_flex_cell_map,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexnode_xpos_in,
    bool var_dsbl_spring,
    bool var_dsbl_damper,
    wp::array_t<wp::vec_t<6, wp::float32>> var_flex_spring_body_force_out,
    wp::array_t<wp::vec_t<6, wp::float32>> var_flex_damper_body_force_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_displ_scratch_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vel_corot_scratch_out)
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
        wp::vec_t<4, wp::int32>* var_2;
        wp::vec_t<4, wp::int32> var_3;
        wp::vec_t<4, wp::int32> var_4;
        const wp::int32 var_5 = 0;
        wp::int32 var_6;
        const wp::int32 var_7 = 1;
        wp::int32 var_8;
        const wp::int32 var_9 = 2;
        wp::int32 var_10;
        const wp::int32 var_11 = 3;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        const wp::int32 var_16 = 0;
        bool var_17;
        const wp::int32 var_18 = 1;
        wp::int32 var_19;
        const wp::int32 var_20 = 1;
        wp::int32 var_21;
        wp::int32 var_22;
        const wp::int32 var_23 = 1;
        wp::int32 var_24;
        wp::int32 var_25;
        const wp::int32 var_26 = 3;
        wp::int32 var_27;
        wp::vec_t<3, wp::int32>* var_28;
        wp::vec_t<3, wp::int32> var_29;
        wp::vec_t<3, wp::int32> var_30;
        const wp::int32 var_31 = 1;
        wp::int32 var_32;
        const wp::int32 var_33 = 2;
        wp::int32 var_34;
        wp::int32* var_35;
        wp::int32 var_36;
        wp::int32 var_37;
        wp::int32 var_38;
        const wp::int32 var_39 = 1;
        wp::int32 var_40;
        wp::int32 var_41;
        const wp::int32 var_42 = 1;
        wp::int32 var_43;
        wp::int32* var_44;
        wp::int32 var_45;
        wp::int32 var_46;
        const wp::int32 var_47 = 0;
        bool var_48;
        wp::int32 var_49;
        wp::int32 var_50;
        wp::int32 var_51;
        wp::int32 var_52;
        wp::int32 var_53;
        wp::int32 var_54;
        wp::int32 var_55;
        wp::int32 var_56;
        wp::float32* var_57;
        const wp::float32 var_58 = 0.0;
        bool var_59;
        wp::float32 var_60;
        wp::quat_t<wp::float32> var_61;
        const wp::int32 var_62 = 0;
        wp::float32 var_63;
        wp::float32 var_64;
        const wp::int32 var_65 = 1;
        wp::float32 var_66;
        wp::float32 var_67;
        const wp::int32 var_68 = 2;
        wp::float32 var_69;
        wp::float32 var_70;
        const wp::int32 var_71 = 3;
        wp::float32 var_72;
        wp::quat_t<wp::float32> var_73;
        const wp::int32 var_74 = 0;
        wp::int32 var_75;
        const wp::int32 var_76 = 1;
        wp::int32 var_77;
        wp::range_t var_78;
        wp::int32 var_79;
        const wp::int32 var_80 = 1;
        wp::int32 var_81;
        wp::range_t var_82;
        wp::int32 var_83;
        const wp::int32 var_84 = 1;
        wp::int32 var_85;
        wp::range_t var_86;
        wp::int32 var_87;
        bool var_88;
        wp::int32 var_89;
        wp::int32 var_90;
        wp::int32 var_91;
        wp::int32 var_92;
        wp::int32 var_93;
        wp::int32 var_94;
        wp::int32 var_95;
        wp::int32 var_96;
        wp::int32 var_97;
        wp::int32 var_98;
        wp::int32 var_99;
        wp::int32 var_100;
        wp::vec_t<3, wp::float32>* var_101;
        wp::vec_t<3, wp::float32> var_102;
        wp::vec_t<3, wp::float32> var_103;
        bool var_104;
        wp::int32 var_105;
        wp::vec_t<3, wp::float32>* var_106;
        wp::vec_t<3, wp::float32> var_107;
        wp::vec_t<3, wp::float32> var_108;
        wp::vec_t<3, wp::float32> var_109;
        wp::vec_t<3, wp::float32> var_110;
        bool var_111;
        wp::int32 var_112;
        wp::int32* var_113;
        wp::int32 var_114;
        wp::int32 var_115;
        wp::vec_t<6, wp::float32>* var_116;
        wp::vec_t<6, wp::float32> var_117;
        wp::vec_t<6, wp::float32> var_118;
        wp::vec_t<3, wp::float32> var_119;
        wp::vec_t<3, wp::float32> var_120;
        wp::int32* var_121;
        wp::vec_t<3, wp::float32>* var_122;
        wp::int32 var_123;
        wp::vec_t<3, wp::float32> var_124;
        wp::vec_t<3, wp::float32> var_125;
        wp::vec_t<3, wp::float32> var_126;
        wp::vec_t<3, wp::float32> var_127;
        wp::vec_t<3, wp::float32> var_128;
        wp::vec_t<3, wp::float32> var_129;
        const wp::int32 var_130 = 1;
        wp::int32 var_131;
        wp::int32 var_132;
        const wp::int32 var_133 = 0;
        wp::int32 var_134;
        const wp::int32 var_135 = 1;
        wp::int32 var_136;
        wp::range_t var_137;
        wp::int32 var_138;
        const wp::int32 var_139 = 1;
        wp::int32 var_140;
        wp::range_t var_141;
        wp::int32 var_142;
        const wp::int32 var_143 = 1;
        wp::int32 var_144;
        wp::range_t var_145;
        wp::int32 var_146;
        bool var_147;
        wp::int32 var_148;
        wp::int32 var_149;
        wp::int32 var_150;
        wp::int32 var_151;
        wp::int32 var_152;
        wp::int32 var_153;
        wp::int32 var_154;
        wp::int32 var_155;
        wp::int32 var_156;
        wp::int32 var_157;
        wp::int32 var_158;
        wp::int32 var_159;
        wp::int32* var_160;
        wp::int32 var_161;
        wp::int32 var_162;
        const wp::float32 var_163 = 0.0;
        wp::vec_t<3, wp::float32> var_164;
        const wp::float32 var_165 = 0.0;
        wp::vec_t<3, wp::float32> var_166;
        const wp::int32 var_167 = 0;
        const wp::int32 var_168 = 3;
        wp::int32 var_169;
        wp::int32 var_170;
        const wp::float32 var_171 = 0.0;
        wp::float32 var_172;
        const wp::float32 var_173 = 0.0;
        wp::float32 var_174;
        wp::range_t var_175;
        wp::int32 var_176;
        const wp::int32 var_177 = 0;
        const wp::int32 var_178 = 3;
        wp::int32 var_179;
        wp::int32 var_180;
        wp::int32 var_181;
        wp::int32 var_182;
        wp::int32 var_183;
        wp::float32* var_184;
        wp::float32 var_185;
        wp::float32 var_186;
        bool var_187;
        wp::vec_t<3, wp::float32>* var_188;
        wp::float32 var_189;
        wp::vec_t<3, wp::float32> var_190;
        wp::float32 var_191;
        wp::float32 var_192;
        wp::float32 var_193;
        bool var_194;
        wp::vec_t<3, wp::float32>* var_195;
        wp::float32 var_196;
        wp::vec_t<3, wp::float32> var_197;
        wp::float32 var_198;
        wp::float32 var_199;
        wp::float32 var_200;
        const wp::int32 var_201 = 1;
        const wp::int32 var_202 = 3;
        wp::int32 var_203;
        wp::int32 var_204;
        wp::int32 var_205;
        wp::int32 var_206;
        wp::int32 var_207;
        wp::float32* var_208;
        wp::float32 var_209;
        wp::float32 var_210;
        bool var_211;
        wp::vec_t<3, wp::float32>* var_212;
        wp::float32 var_213;
        wp::vec_t<3, wp::float32> var_214;
        wp::float32 var_215;
        wp::float32 var_216;
        wp::float32 var_217;
        bool var_218;
        wp::vec_t<3, wp::float32>* var_219;
        wp::float32 var_220;
        wp::vec_t<3, wp::float32> var_221;
        wp::float32 var_222;
        wp::float32 var_223;
        wp::float32 var_224;
        const wp::int32 var_225 = 2;
        const wp::int32 var_226 = 3;
        wp::int32 var_227;
        wp::int32 var_228;
        wp::int32 var_229;
        wp::int32 var_230;
        wp::int32 var_231;
        wp::float32* var_232;
        wp::float32 var_233;
        wp::float32 var_234;
        bool var_235;
        wp::vec_t<3, wp::float32>* var_236;
        wp::float32 var_237;
        wp::vec_t<3, wp::float32> var_238;
        wp::float32 var_239;
        wp::float32 var_240;
        wp::float32 var_241;
        bool var_242;
        wp::vec_t<3, wp::float32>* var_243;
        wp::float32 var_244;
        wp::vec_t<3, wp::float32> var_245;
        wp::float32 var_246;
        wp::float32 var_247;
        wp::float32 var_248;
        const wp::int32 var_249 = 1;
        const wp::int32 var_250 = 3;
        wp::int32 var_251;
        wp::int32 var_252;
        const wp::float32 var_253 = 0.0;
        wp::float32 var_254;
        const wp::float32 var_255 = 0.0;
        wp::float32 var_256;
        wp::range_t var_257;
        wp::int32 var_258;
        const wp::int32 var_259 = 0;
        const wp::int32 var_260 = 3;
        wp::int32 var_261;
        wp::int32 var_262;
        wp::int32 var_263;
        wp::int32 var_264;
        wp::int32 var_265;
        wp::float32* var_266;
        wp::float32 var_267;
        wp::float32 var_268;
        bool var_269;
        wp::vec_t<3, wp::float32>* var_270;
        wp::float32 var_271;
        wp::vec_t<3, wp::float32> var_272;
        wp::float32 var_273;
        wp::float32 var_274;
        wp::float32 var_275;
        bool var_276;
        wp::vec_t<3, wp::float32>* var_277;
        wp::float32 var_278;
        wp::vec_t<3, wp::float32> var_279;
        wp::float32 var_280;
        wp::float32 var_281;
        wp::float32 var_282;
        const wp::int32 var_283 = 1;
        const wp::int32 var_284 = 3;
        wp::int32 var_285;
        wp::int32 var_286;
        wp::int32 var_287;
        wp::int32 var_288;
        wp::int32 var_289;
        wp::float32* var_290;
        wp::float32 var_291;
        wp::float32 var_292;
        bool var_293;
        wp::vec_t<3, wp::float32>* var_294;
        wp::float32 var_295;
        wp::vec_t<3, wp::float32> var_296;
        wp::float32 var_297;
        wp::float32 var_298;
        wp::float32 var_299;
        bool var_300;
        wp::vec_t<3, wp::float32>* var_301;
        wp::float32 var_302;
        wp::vec_t<3, wp::float32> var_303;
        wp::float32 var_304;
        wp::float32 var_305;
        wp::float32 var_306;
        const wp::int32 var_307 = 2;
        const wp::int32 var_308 = 3;
        wp::int32 var_309;
        wp::int32 var_310;
        wp::int32 var_311;
        wp::int32 var_312;
        wp::int32 var_313;
        wp::float32* var_314;
        wp::float32 var_315;
        wp::float32 var_316;
        bool var_317;
        wp::vec_t<3, wp::float32>* var_318;
        wp::float32 var_319;
        wp::vec_t<3, wp::float32> var_320;
        wp::float32 var_321;
        wp::float32 var_322;
        wp::float32 var_323;
        bool var_324;
        wp::vec_t<3, wp::float32>* var_325;
        wp::float32 var_326;
        wp::vec_t<3, wp::float32> var_327;
        wp::float32 var_328;
        wp::float32 var_329;
        wp::float32 var_330;
        const wp::int32 var_331 = 2;
        const wp::int32 var_332 = 3;
        wp::int32 var_333;
        wp::int32 var_334;
        const wp::float32 var_335 = 0.0;
        wp::float32 var_336;
        const wp::float32 var_337 = 0.0;
        wp::float32 var_338;
        wp::range_t var_339;
        wp::int32 var_340;
        const wp::int32 var_341 = 0;
        const wp::int32 var_342 = 3;
        wp::int32 var_343;
        wp::int32 var_344;
        wp::int32 var_345;
        wp::int32 var_346;
        wp::int32 var_347;
        wp::float32* var_348;
        wp::float32 var_349;
        wp::float32 var_350;
        bool var_351;
        wp::vec_t<3, wp::float32>* var_352;
        wp::float32 var_353;
        wp::vec_t<3, wp::float32> var_354;
        wp::float32 var_355;
        wp::float32 var_356;
        wp::float32 var_357;
        bool var_358;
        wp::vec_t<3, wp::float32>* var_359;
        wp::float32 var_360;
        wp::vec_t<3, wp::float32> var_361;
        wp::float32 var_362;
        wp::float32 var_363;
        wp::float32 var_364;
        const wp::int32 var_365 = 1;
        const wp::int32 var_366 = 3;
        wp::int32 var_367;
        wp::int32 var_368;
        wp::int32 var_369;
        wp::int32 var_370;
        wp::int32 var_371;
        wp::float32* var_372;
        wp::float32 var_373;
        wp::float32 var_374;
        bool var_375;
        wp::vec_t<3, wp::float32>* var_376;
        wp::float32 var_377;
        wp::vec_t<3, wp::float32> var_378;
        wp::float32 var_379;
        wp::float32 var_380;
        wp::float32 var_381;
        bool var_382;
        wp::vec_t<3, wp::float32>* var_383;
        wp::float32 var_384;
        wp::vec_t<3, wp::float32> var_385;
        wp::float32 var_386;
        wp::float32 var_387;
        wp::float32 var_388;
        const wp::int32 var_389 = 2;
        const wp::int32 var_390 = 3;
        wp::int32 var_391;
        wp::int32 var_392;
        wp::int32 var_393;
        wp::int32 var_394;
        wp::int32 var_395;
        wp::float32* var_396;
        wp::float32 var_397;
        wp::float32 var_398;
        bool var_399;
        wp::vec_t<3, wp::float32>* var_400;
        wp::float32 var_401;
        wp::vec_t<3, wp::float32> var_402;
        wp::float32 var_403;
        wp::float32 var_404;
        wp::float32 var_405;
        bool var_406;
        wp::vec_t<3, wp::float32>* var_407;
        wp::float32 var_408;
        wp::vec_t<3, wp::float32> var_409;
        wp::float32 var_410;
        wp::float32 var_411;
        wp::float32 var_412;
        wp::vec_t<3, wp::float32> var_413;
        wp::vec_t<3, wp::float32> var_414;
        wp::float32* var_415;
        wp::vec_t<3, wp::float32> var_416;
        wp::float32 var_417;
        wp::int32 var_418;
        wp::vec_t<3, wp::float32>* var_419;
        wp::vec_t<3, wp::float32> var_420;
        wp::vec_t<3, wp::float32> var_421;
        wp::vec_t<3, wp::float32>* var_422;
        wp::vec_t<3, wp::float32> var_423;
        wp::vec_t<3, wp::float32> var_424;
        wp::vec_t<3, wp::float32> var_425;
        bool var_426;
        wp::vec_t<3, wp::float32> var_427;
        wp::vec_t<3, wp::float32> var_428;
        wp::vec_t<6, wp::float32> var_429;
        wp::vec_t<6, wp::float32> var_430;
        bool var_431;
        wp::vec_t<3, wp::float32> var_432;
        wp::vec_t<3, wp::float32> var_433;
        wp::vec_t<6, wp::float32> var_434;
        wp::vec_t<6, wp::float32> var_435;
        const wp::int32 var_436 = 1;
        wp::int32 var_437;
        wp::int32 var_438;
        wp::int32 var_439;
        //---------
        // forward
        // def _flex_passive_interp(                                                              <L 822>
        // worldid, cellid = wp.tid()                                                             <L 853>
        builtin_tid2d(var_0, var_1);
        // mapping = flex_cell_map[cellid]                                                        <L 855>
        var_2 = wp::address(var_flex_cell_map, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // f = mapping[0]                                                                         <L 856>
        var_6 = wp::extract(var_3, var_5);
        // ci = mapping[1]                                                                        <L 857>
        var_8 = wp::extract(var_3, var_7);
        // cj = mapping[2]                                                                        <L 858>
        var_10 = wp::extract(var_3, var_9);
        // ck = mapping[3]                                                                        <L 859>
        var_12 = wp::extract(var_3, var_11);
        // order = flex_interp[f]                                                                 <L 861>
        var_13 = wp::address(var_flex_interp, var_6);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // if order <= 0:                                                                         <L 862>
        var_17 = (var_14 <= var_16);
        if (var_17) {
            // return                                                                             <L 863>
            continue;
        }
        // npc = (order + 1) * (order + 1) * (order + 1)                                          <L 865>
        var_19 = wp::add(var_14, var_18);
        var_21 = wp::add(var_14, var_20);
        var_22 = wp::mul(var_19, var_21);
        var_24 = wp::add(var_14, var_23);
        var_25 = wp::mul(var_22, var_24);
        // ndof_cell = 3 * npc                                                                    <L 866>
        var_27 = wp::mul(var_26, var_25);
        // cellnum = flex_cellnum[f]                                                              <L 868>
        var_28 = wp::address(var_flex_cellnum, var_6);
        var_30 = wp::load(var_28);
        var_29 = wp::copy(var_30);
        // cy = cellnum[1]                                                                        <L 869>
        var_32 = wp::extract(var_29, var_31);
        // cz = cellnum[2]                                                                        <L 870>
        var_34 = wp::extract(var_29, var_33);
        // nstart = flex_nodeadr[f]                                                               <L 871>
        var_35 = wp::address(var_flex_nodeadr, var_6);
        var_37 = wp::load(var_35);
        var_36 = wp::copy(var_37);
        // ny_g = cy * order + 1                                                                  <L 872>
        var_38 = wp::mul(var_32, var_14);
        var_40 = wp::add(var_38, var_39);
        // nz_g = cz * order + 1                                                                  <L 873>
        var_41 = wp::mul(var_34, var_14);
        var_43 = wp::add(var_41, var_42);
        // stiffness_adr_base = flex_stiffnessadr[f]                                              <L 876>
        var_44 = wp::address(var_flex_stiffnessadr, var_6);
        var_46 = wp::load(var_44);
        var_45 = wp::copy(var_46);
        // if stiffness_adr_base < 0:                                                             <L 877>
        var_48 = (var_45 < var_47);
        if (var_48) {
            // return                                                                             <L 878>
            continue;
        }
        // cell_idx = ci * cy * cz + cj * cz + ck                                                 <L 880>
        var_49 = wp::mul(var_8, var_32);
        var_50 = wp::mul(var_49, var_34);
        var_51 = wp::mul(var_10, var_34);
        var_52 = wp::add(var_50, var_51);
        var_53 = wp::add(var_52, var_12);
        // k_base = stiffness_adr_base + cell_idx * ndof_cell * ndof_cell                         <L 881>
        var_54 = wp::mul(var_53, var_27);
        var_55 = wp::mul(var_54, var_27);
        var_56 = wp::add(var_45, var_55);
        // if flex_stiffness[k_base] == 0.0:                                                      <L 884>
        var_57 = wp::address(var_flex_stiffness, var_56);
        var_60 = wp::load(var_57);
        var_59 = (var_60 == var_58);
        if (var_59) {
            // return                                                                             <L 885>
            continue;
        }
        // cell_quat = support.compute_interp_cell_quat(flexnode_xpos_in, order, ci, cj, ck, cy, cz, ny_g, nz_g, nstart, worldid)       <L 887>
        var_61 = compute_interp_cell_quat_0(var_flexnode_xpos_in, var_14, var_8, var_10, var_12, var_32, var_34, var_40, var_43, var_36, var_0);
        // cell_quat_inv = wp.quat(-cell_quat[0], -cell_quat[1], -cell_quat[2], cell_quat[3])       <L 890>
        var_63 = wp::extract(var_61, var_62);
        var_64 = wp::neg(var_63);
        var_66 = wp::extract(var_61, var_65);
        var_67 = wp::neg(var_66);
        var_69 = wp::extract(var_61, var_68);
        var_70 = wp::neg(var_69);
        var_72 = wp::extract(var_61, var_71);
        var_73 = wp::quat_t<wp::float32>(var_64, var_67, var_70, var_72);
        // idx_j = int(0)                                                                         <L 894>
        var_75 = wp::int(var_74);
        // for li_j in range(order + 1):                                                          <L 895>
        var_77 = wp::add(var_14, var_76);
        var_78 = wp::range(var_77);
        start_for_3:;
            if (iter_cmp(var_78) == 0) goto end_for_3;
            var_79 = wp::iter_next(var_78);
            // for lj_j in range(order + 1):                                                      <L 896>
            var_81 = wp::add(var_14, var_80);
            var_82 = wp::range(var_81);
            start_for_5:;
                if (iter_cmp(var_82) == 0) goto end_for_5;
                var_83 = wp::iter_next(var_82);
                // for lk_j in range(order + 1):                                                  <L 897>
                var_85 = wp::add(var_14, var_84);
                var_86 = wp::range(var_85);
                start_for_7:;
                    if (iter_cmp(var_86) == 0) goto end_for_7;
                    var_87 = wp::iter_next(var_86);
                    // if idx_j < npc:                                                            <L 898>
                    var_88 = (var_75 < var_25);
                    if (var_88) {
                        // gi_j = ci * order + li_j                                               <L 899>
                        var_89 = wp::mul(var_8, var_14);
                        var_90 = wp::add(var_89, var_79);
                        // gj_j = cj * order + lj_j                                               <L 900>
                        var_91 = wp::mul(var_10, var_14);
                        var_92 = wp::add(var_91, var_83);
                        // gk_j = ck * order + lk_j                                               <L 901>
                        var_93 = wp::mul(var_12, var_14);
                        var_94 = wp::add(var_93, var_87);
                        // gidx_j = gi_j * ny_g * nz_g + gj_j * nz_g + gk_j                       <L 902>
                        var_95 = wp::mul(var_90, var_40);
                        var_96 = wp::mul(var_95, var_43);
                        var_97 = wp::mul(var_92, var_43);
                        var_98 = wp::add(var_96, var_97);
                        var_99 = wp::add(var_98, var_94);
                        // xpos_j = flexnode_xpos_in[worldid, nstart + gidx_j]                    <L 904>
                        var_100 = wp::add(var_36, var_99);
                        var_101 = wp::address(var_flexnode_xpos_in, var_0, var_100);
                        var_103 = wp::load(var_101);
                        var_102 = wp::copy(var_103);
                        // if not dsbl_spring:                                                    <L 906>
                        var_104 = wp::unot(var_dsbl_spring);
                        if (var_104) {
                            // refpos_j = flex_node0[nstart + gidx_j]                             <L 907>
                            var_105 = wp::add(var_36, var_99);
                            var_106 = wp::address(var_flex_node0, var_105);
                            var_108 = wp::load(var_106);
                            var_107 = wp::copy(var_108);
                            // xrot_j = wp.quat_rotate(cell_quat_inv, xpos_j)                     <L 908>
                            var_109 = wp::quat_rotate(var_73, var_102);
                            // displ_scratch_out[worldid, cellid, idx_j] = xrot_j - refpos_j       <L 909>
                            var_110 = wp::sub(var_109, var_107);
                            wp::array_store(var_displ_scratch_out, var_0, var_1, var_75, var_110);
                        }
                        // if not dsbl_damper:                                                    <L 911>
                        var_111 = wp::unot(var_dsbl_damper);
                        if (var_111) {
                            // bodyid_j = flex_nodebodyid[nstart + gidx_j]                        <L 912>
                            var_112 = wp::add(var_36, var_99);
                            var_113 = wp::address(var_flex_nodebodyid, var_112);
                            var_115 = wp::load(var_113);
                            var_114 = wp::copy(var_115);
                            // cvel_j = cvel_in[worldid, bodyid_j]                                <L 913>
                            var_116 = wp::address(var_cvel_in, var_0, var_114);
                            var_118 = wp::load(var_116);
                            var_117 = wp::copy(var_118);
                            // omega_j = wp.spatial_top(cvel_j)                                   <L 914>
                            var_119 = wp::spatial_top(var_117);
                            // vcom_j = wp.spatial_bottom(cvel_j)                                 <L 915>
                            var_120 = wp::spatial_bottom(var_117);
                            // com_j = subtree_com_in[worldid, body_rootid[bodyid_j]]             <L 916>
                            var_121 = wp::address(var_body_rootid, var_114);
                            var_123 = wp::load(var_121);
                            var_122 = wp::address(var_subtree_com_in, var_0, var_123);
                            var_125 = wp::load(var_122);
                            var_124 = wp::copy(var_125);
                            // r_j = xpos_j - com_j                                               <L 917>
                            var_126 = wp::sub(var_102, var_124);
                            // vel_world_j = vcom_j + wp.cross(omega_j, r_j)                      <L 918>
                            var_127 = wp::cross(var_119, var_126);
                            var_128 = wp::add(var_120, var_127);
                            // vel_corot_scratch_out[worldid, cellid, idx_j] = wp.quat_rotate(cell_quat_inv, vel_world_j)       <L 919>
                            var_129 = wp::quat_rotate(var_73, var_128);
                            wp::array_store(var_vel_corot_scratch_out, var_0, var_1, var_75, var_129);
                        }
                        // idx_j += 1                                                             <L 921>
                        var_131 = wp::add(var_75, var_130);
                    }
                    var_132 = wp::where(var_88, var_131, var_75);
                    wp::assign(var_75, var_132);
                    goto start_for_7;
                end_for_7:;
                goto start_for_5;
            end_for_5:;
            goto start_for_3;
        end_for_3:;
        // idx_i = int(0)                                                                         <L 924>
        var_134 = wp::int(var_133);
        // for li_i in range(order + 1):                                                          <L 925>
        var_136 = wp::add(var_14, var_135);
        var_137 = wp::range(var_136);
        start_for_9:;
            if (iter_cmp(var_137) == 0) goto end_for_9;
            var_138 = wp::iter_next(var_137);
            // for lj_i in range(order + 1):                                                      <L 926>
            var_140 = wp::add(var_14, var_139);
            var_141 = wp::range(var_140);
            start_for_11:;
                if (iter_cmp(var_141) == 0) goto end_for_11;
                var_142 = wp::iter_next(var_141);
                // for lk_i in range(order + 1):                                                  <L 927>
                var_144 = wp::add(var_14, var_143);
                var_145 = wp::range(var_144);
                start_for_13:;
                    if (iter_cmp(var_145) == 0) goto end_for_13;
                    var_146 = wp::iter_next(var_145);
                    // if idx_i < npc:                                                            <L 928>
                    var_147 = (var_134 < var_25);
                    if (var_147) {
                        // gi_i = ci * order + li_i                                               <L 929>
                        var_148 = wp::mul(var_8, var_14);
                        var_149 = wp::add(var_148, var_138);
                        // gj_i = cj * order + lj_i                                               <L 930>
                        var_150 = wp::mul(var_10, var_14);
                        var_151 = wp::add(var_150, var_142);
                        // gk_i = ck * order + lk_i                                               <L 931>
                        var_152 = wp::mul(var_12, var_14);
                        var_153 = wp::add(var_152, var_146);
                        // gidx_i = gi_i * ny_g * nz_g + gj_i * nz_g + gk_i                       <L 932>
                        var_154 = wp::mul(var_149, var_40);
                        var_155 = wp::mul(var_154, var_43);
                        var_156 = wp::mul(var_151, var_43);
                        var_157 = wp::add(var_155, var_156);
                        var_158 = wp::add(var_157, var_153);
                        // bodyid_i = flex_nodebodyid[nstart + gidx_i]                            <L 933>
                        var_159 = wp::add(var_36, var_158);
                        var_160 = wp::address(var_flex_nodebodyid, var_159);
                        var_162 = wp::load(var_160);
                        var_161 = wp::copy(var_162);
                        // frc_spring = wp.vec3(0.0)                                              <L 935>
                        var_164 = wp::vec_t<3, wp::float32>(var_163);
                        // frc_damper = wp.vec3(0.0)                                              <L 936>
                        var_166 = wp::vec_t<3, wp::float32>(var_165);
                        // for comp_i in range(3):                                                <L 938>
                        // row = idx_i * 3 + comp_i                                               <L 939>
                        var_169 = wp::mul(var_134, var_168);
                        var_170 = wp::add(var_169, var_167);
                        // val_spring = float(0.0)                                                <L 940>
                        var_172 = wp::float(var_171);
                        // val_damper = float(0.0)                                                <L 941>
                        var_174 = wp::float(var_173);
                        // for idx_j in range(npc):                                               <L 943>
                        var_175 = wp::range(var_25);
                        start_for_15:;
                            if (iter_cmp(var_175) == 0) goto end_for_15;
                            var_176 = wp::iter_next(var_175);
                            // for comp_j in range(3):                                            <L 944>
                            // col = idx_j * 3 + comp_j                                           <L 945>
                            var_179 = wp::mul(var_176, var_178);
                            var_180 = wp::add(var_179, var_177);
                            // K_ij = flex_stiffness[k_base + row * ndof_cell + col]              <L 946>
                            var_181 = wp::mul(var_170, var_27);
                            var_182 = wp::add(var_56, var_181);
                            var_183 = wp::add(var_182, var_180);
                            var_184 = wp::address(var_flex_stiffness, var_183);
                            var_186 = wp::load(var_184);
                            var_185 = wp::copy(var_186);
                            // if not dsbl_spring:                                                <L 948>
                            var_187 = wp::unot(var_dsbl_spring);
                            if (var_187) {
                                // val_spring += K_ij * displ_scratch_out[worldid, cellid, idx_j][comp_j]       <L 949>
                                var_188 = wp::address(var_displ_scratch_out, var_0, var_1, var_176);
                                var_190 = wp::load(var_188);
                                var_189 = wp::extract(var_190, var_177);
                                var_191 = wp::mul(var_185, var_189);
                                var_192 = wp::add(var_172, var_191);
                            }
                            var_193 = wp::where(var_187, var_192, var_172);
                            // if not dsbl_damper:                                                <L 951>
                            var_194 = wp::unot(var_dsbl_damper);
                            if (var_194) {
                                // val_damper += K_ij * vel_corot_scratch_out[worldid, cellid, idx_j][comp_j]       <L 952>
                                var_195 = wp::address(var_vel_corot_scratch_out, var_0, var_1, var_176);
                                var_197 = wp::load(var_195);
                                var_196 = wp::extract(var_197, var_177);
                                var_198 = wp::mul(var_185, var_196);
                                var_199 = wp::add(var_174, var_198);
                            }
                            var_200 = wp::where(var_194, var_199, var_174);
                            // col = idx_j * 3 + comp_j                                           <L 945>
                            var_203 = wp::mul(var_176, var_202);
                            var_204 = wp::add(var_203, var_201);
                            // K_ij = flex_stiffness[k_base + row * ndof_cell + col]              <L 946>
                            var_205 = wp::mul(var_170, var_27);
                            var_206 = wp::add(var_56, var_205);
                            var_207 = wp::add(var_206, var_204);
                            var_208 = wp::address(var_flex_stiffness, var_207);
                            var_210 = wp::load(var_208);
                            var_209 = wp::copy(var_210);
                            // if not dsbl_spring:                                                <L 948>
                            var_211 = wp::unot(var_dsbl_spring);
                            if (var_211) {
                                // val_spring += K_ij * displ_scratch_out[worldid, cellid, idx_j][comp_j]       <L 949>
                                var_212 = wp::address(var_displ_scratch_out, var_0, var_1, var_176);
                                var_214 = wp::load(var_212);
                                var_213 = wp::extract(var_214, var_201);
                                var_215 = wp::mul(var_209, var_213);
                                var_216 = wp::add(var_193, var_215);
                            }
                            var_217 = wp::where(var_211, var_216, var_193);
                            // if not dsbl_damper:                                                <L 951>
                            var_218 = wp::unot(var_dsbl_damper);
                            if (var_218) {
                                // val_damper += K_ij * vel_corot_scratch_out[worldid, cellid, idx_j][comp_j]       <L 952>
                                var_219 = wp::address(var_vel_corot_scratch_out, var_0, var_1, var_176);
                                var_221 = wp::load(var_219);
                                var_220 = wp::extract(var_221, var_201);
                                var_222 = wp::mul(var_209, var_220);
                                var_223 = wp::add(var_200, var_222);
                            }
                            var_224 = wp::where(var_218, var_223, var_200);
                            // col = idx_j * 3 + comp_j                                           <L 945>
                            var_227 = wp::mul(var_176, var_226);
                            var_228 = wp::add(var_227, var_225);
                            // K_ij = flex_stiffness[k_base + row * ndof_cell + col]              <L 946>
                            var_229 = wp::mul(var_170, var_27);
                            var_230 = wp::add(var_56, var_229);
                            var_231 = wp::add(var_230, var_228);
                            var_232 = wp::address(var_flex_stiffness, var_231);
                            var_234 = wp::load(var_232);
                            var_233 = wp::copy(var_234);
                            // if not dsbl_spring:                                                <L 948>
                            var_235 = wp::unot(var_dsbl_spring);
                            if (var_235) {
                                // val_spring += K_ij * displ_scratch_out[worldid, cellid, idx_j][comp_j]       <L 949>
                                var_236 = wp::address(var_displ_scratch_out, var_0, var_1, var_176);
                                var_238 = wp::load(var_236);
                                var_237 = wp::extract(var_238, var_225);
                                var_239 = wp::mul(var_233, var_237);
                                var_240 = wp::add(var_217, var_239);
                            }
                            var_241 = wp::where(var_235, var_240, var_217);
                            // if not dsbl_damper:                                                <L 951>
                            var_242 = wp::unot(var_dsbl_damper);
                            if (var_242) {
                                // val_damper += K_ij * vel_corot_scratch_out[worldid, cellid, idx_j][comp_j]       <L 952>
                                var_243 = wp::address(var_vel_corot_scratch_out, var_0, var_1, var_176);
                                var_245 = wp::load(var_243);
                                var_244 = wp::extract(var_245, var_225);
                                var_246 = wp::mul(var_233, var_244);
                                var_247 = wp::add(var_224, var_246);
                            }
                            var_248 = wp::where(var_242, var_247, var_224);
                            wp::assign(var_172, var_241);
                            wp::assign(var_174, var_248);
                            goto start_for_15;
                        end_for_15:;
                        // frc_spring[comp_i] = val_spring                                        <L 954>
                        wp::assign_inplace(var_164, var_167, var_172);
                        // frc_damper[comp_i] = val_damper                                        <L 955>
                        wp::assign_inplace(var_166, var_167, var_174);
                        // row = idx_i * 3 + comp_i                                               <L 939>
                        var_251 = wp::mul(var_134, var_250);
                        var_252 = wp::add(var_251, var_249);
                        // val_spring = float(0.0)                                                <L 940>
                        var_254 = wp::float(var_253);
                        // val_damper = float(0.0)                                                <L 941>
                        var_256 = wp::float(var_255);
                        // for idx_j in range(npc):                                               <L 943>
                        var_257 = wp::range(var_25);
                        start_for_17:;
                            if (iter_cmp(var_257) == 0) goto end_for_17;
                            var_258 = wp::iter_next(var_257);
                            // for comp_j in range(3):                                            <L 944>
                            // col = idx_j * 3 + comp_j                                           <L 945>
                            var_261 = wp::mul(var_258, var_260);
                            var_262 = wp::add(var_261, var_259);
                            // K_ij = flex_stiffness[k_base + row * ndof_cell + col]              <L 946>
                            var_263 = wp::mul(var_252, var_27);
                            var_264 = wp::add(var_56, var_263);
                            var_265 = wp::add(var_264, var_262);
                            var_266 = wp::address(var_flex_stiffness, var_265);
                            var_268 = wp::load(var_266);
                            var_267 = wp::copy(var_268);
                            // if not dsbl_spring:                                                <L 948>
                            var_269 = wp::unot(var_dsbl_spring);
                            if (var_269) {
                                // val_spring += K_ij * displ_scratch_out[worldid, cellid, idx_j][comp_j]       <L 949>
                                var_270 = wp::address(var_displ_scratch_out, var_0, var_1, var_258);
                                var_272 = wp::load(var_270);
                                var_271 = wp::extract(var_272, var_259);
                                var_273 = wp::mul(var_267, var_271);
                                var_274 = wp::add(var_254, var_273);
                            }
                            var_275 = wp::where(var_269, var_274, var_254);
                            // if not dsbl_damper:                                                <L 951>
                            var_276 = wp::unot(var_dsbl_damper);
                            if (var_276) {
                                // val_damper += K_ij * vel_corot_scratch_out[worldid, cellid, idx_j][comp_j]       <L 952>
                                var_277 = wp::address(var_vel_corot_scratch_out, var_0, var_1, var_258);
                                var_279 = wp::load(var_277);
                                var_278 = wp::extract(var_279, var_259);
                                var_280 = wp::mul(var_267, var_278);
                                var_281 = wp::add(var_256, var_280);
                            }
                            var_282 = wp::where(var_276, var_281, var_256);
                            // col = idx_j * 3 + comp_j                                           <L 945>
                            var_285 = wp::mul(var_258, var_284);
                            var_286 = wp::add(var_285, var_283);
                            // K_ij = flex_stiffness[k_base + row * ndof_cell + col]              <L 946>
                            var_287 = wp::mul(var_252, var_27);
                            var_288 = wp::add(var_56, var_287);
                            var_289 = wp::add(var_288, var_286);
                            var_290 = wp::address(var_flex_stiffness, var_289);
                            var_292 = wp::load(var_290);
                            var_291 = wp::copy(var_292);
                            // if not dsbl_spring:                                                <L 948>
                            var_293 = wp::unot(var_dsbl_spring);
                            if (var_293) {
                                // val_spring += K_ij * displ_scratch_out[worldid, cellid, idx_j][comp_j]       <L 949>
                                var_294 = wp::address(var_displ_scratch_out, var_0, var_1, var_258);
                                var_296 = wp::load(var_294);
                                var_295 = wp::extract(var_296, var_283);
                                var_297 = wp::mul(var_291, var_295);
                                var_298 = wp::add(var_275, var_297);
                            }
                            var_299 = wp::where(var_293, var_298, var_275);
                            // if not dsbl_damper:                                                <L 951>
                            var_300 = wp::unot(var_dsbl_damper);
                            if (var_300) {
                                // val_damper += K_ij * vel_corot_scratch_out[worldid, cellid, idx_j][comp_j]       <L 952>
                                var_301 = wp::address(var_vel_corot_scratch_out, var_0, var_1, var_258);
                                var_303 = wp::load(var_301);
                                var_302 = wp::extract(var_303, var_283);
                                var_304 = wp::mul(var_291, var_302);
                                var_305 = wp::add(var_282, var_304);
                            }
                            var_306 = wp::where(var_300, var_305, var_282);
                            // col = idx_j * 3 + comp_j                                           <L 945>
                            var_309 = wp::mul(var_258, var_308);
                            var_310 = wp::add(var_309, var_307);
                            // K_ij = flex_stiffness[k_base + row * ndof_cell + col]              <L 946>
                            var_311 = wp::mul(var_252, var_27);
                            var_312 = wp::add(var_56, var_311);
                            var_313 = wp::add(var_312, var_310);
                            var_314 = wp::address(var_flex_stiffness, var_313);
                            var_316 = wp::load(var_314);
                            var_315 = wp::copy(var_316);
                            // if not dsbl_spring:                                                <L 948>
                            var_317 = wp::unot(var_dsbl_spring);
                            if (var_317) {
                                // val_spring += K_ij * displ_scratch_out[worldid, cellid, idx_j][comp_j]       <L 949>
                                var_318 = wp::address(var_displ_scratch_out, var_0, var_1, var_258);
                                var_320 = wp::load(var_318);
                                var_319 = wp::extract(var_320, var_307);
                                var_321 = wp::mul(var_315, var_319);
                                var_322 = wp::add(var_299, var_321);
                            }
                            var_323 = wp::where(var_317, var_322, var_299);
                            // if not dsbl_damper:                                                <L 951>
                            var_324 = wp::unot(var_dsbl_damper);
                            if (var_324) {
                                // val_damper += K_ij * vel_corot_scratch_out[worldid, cellid, idx_j][comp_j]       <L 952>
                                var_325 = wp::address(var_vel_corot_scratch_out, var_0, var_1, var_258);
                                var_327 = wp::load(var_325);
                                var_326 = wp::extract(var_327, var_307);
                                var_328 = wp::mul(var_315, var_326);
                                var_329 = wp::add(var_306, var_328);
                            }
                            var_330 = wp::where(var_324, var_329, var_306);
                            wp::assign(var_254, var_323);
                            wp::assign(var_256, var_330);
                            wp::assign(var_228, var_310);
                            wp::assign(var_233, var_315);
                            goto start_for_17;
                        end_for_17:;
                        // frc_spring[comp_i] = val_spring                                        <L 954>
                        wp::assign_inplace(var_164, var_249, var_254);
                        // frc_damper[comp_i] = val_damper                                        <L 955>
                        wp::assign_inplace(var_166, var_249, var_256);
                        // row = idx_i * 3 + comp_i                                               <L 939>
                        var_333 = wp::mul(var_134, var_332);
                        var_334 = wp::add(var_333, var_331);
                        // val_spring = float(0.0)                                                <L 940>
                        var_336 = wp::float(var_335);
                        // val_damper = float(0.0)                                                <L 941>
                        var_338 = wp::float(var_337);
                        // for idx_j in range(npc):                                               <L 943>
                        var_339 = wp::range(var_25);
                        start_for_19:;
                            if (iter_cmp(var_339) == 0) goto end_for_19;
                            var_340 = wp::iter_next(var_339);
                            // for comp_j in range(3):                                            <L 944>
                            // col = idx_j * 3 + comp_j                                           <L 945>
                            var_343 = wp::mul(var_340, var_342);
                            var_344 = wp::add(var_343, var_341);
                            // K_ij = flex_stiffness[k_base + row * ndof_cell + col]              <L 946>
                            var_345 = wp::mul(var_334, var_27);
                            var_346 = wp::add(var_56, var_345);
                            var_347 = wp::add(var_346, var_344);
                            var_348 = wp::address(var_flex_stiffness, var_347);
                            var_350 = wp::load(var_348);
                            var_349 = wp::copy(var_350);
                            // if not dsbl_spring:                                                <L 948>
                            var_351 = wp::unot(var_dsbl_spring);
                            if (var_351) {
                                // val_spring += K_ij * displ_scratch_out[worldid, cellid, idx_j][comp_j]       <L 949>
                                var_352 = wp::address(var_displ_scratch_out, var_0, var_1, var_340);
                                var_354 = wp::load(var_352);
                                var_353 = wp::extract(var_354, var_341);
                                var_355 = wp::mul(var_349, var_353);
                                var_356 = wp::add(var_336, var_355);
                            }
                            var_357 = wp::where(var_351, var_356, var_336);
                            // if not dsbl_damper:                                                <L 951>
                            var_358 = wp::unot(var_dsbl_damper);
                            if (var_358) {
                                // val_damper += K_ij * vel_corot_scratch_out[worldid, cellid, idx_j][comp_j]       <L 952>
                                var_359 = wp::address(var_vel_corot_scratch_out, var_0, var_1, var_340);
                                var_361 = wp::load(var_359);
                                var_360 = wp::extract(var_361, var_341);
                                var_362 = wp::mul(var_349, var_360);
                                var_363 = wp::add(var_338, var_362);
                            }
                            var_364 = wp::where(var_358, var_363, var_338);
                            // col = idx_j * 3 + comp_j                                           <L 945>
                            var_367 = wp::mul(var_340, var_366);
                            var_368 = wp::add(var_367, var_365);
                            // K_ij = flex_stiffness[k_base + row * ndof_cell + col]              <L 946>
                            var_369 = wp::mul(var_334, var_27);
                            var_370 = wp::add(var_56, var_369);
                            var_371 = wp::add(var_370, var_368);
                            var_372 = wp::address(var_flex_stiffness, var_371);
                            var_374 = wp::load(var_372);
                            var_373 = wp::copy(var_374);
                            // if not dsbl_spring:                                                <L 948>
                            var_375 = wp::unot(var_dsbl_spring);
                            if (var_375) {
                                // val_spring += K_ij * displ_scratch_out[worldid, cellid, idx_j][comp_j]       <L 949>
                                var_376 = wp::address(var_displ_scratch_out, var_0, var_1, var_340);
                                var_378 = wp::load(var_376);
                                var_377 = wp::extract(var_378, var_365);
                                var_379 = wp::mul(var_373, var_377);
                                var_380 = wp::add(var_357, var_379);
                            }
                            var_381 = wp::where(var_375, var_380, var_357);
                            // if not dsbl_damper:                                                <L 951>
                            var_382 = wp::unot(var_dsbl_damper);
                            if (var_382) {
                                // val_damper += K_ij * vel_corot_scratch_out[worldid, cellid, idx_j][comp_j]       <L 952>
                                var_383 = wp::address(var_vel_corot_scratch_out, var_0, var_1, var_340);
                                var_385 = wp::load(var_383);
                                var_384 = wp::extract(var_385, var_365);
                                var_386 = wp::mul(var_373, var_384);
                                var_387 = wp::add(var_364, var_386);
                            }
                            var_388 = wp::where(var_382, var_387, var_364);
                            // col = idx_j * 3 + comp_j                                           <L 945>
                            var_391 = wp::mul(var_340, var_390);
                            var_392 = wp::add(var_391, var_389);
                            // K_ij = flex_stiffness[k_base + row * ndof_cell + col]              <L 946>
                            var_393 = wp::mul(var_334, var_27);
                            var_394 = wp::add(var_56, var_393);
                            var_395 = wp::add(var_394, var_392);
                            var_396 = wp::address(var_flex_stiffness, var_395);
                            var_398 = wp::load(var_396);
                            var_397 = wp::copy(var_398);
                            // if not dsbl_spring:                                                <L 948>
                            var_399 = wp::unot(var_dsbl_spring);
                            if (var_399) {
                                // val_spring += K_ij * displ_scratch_out[worldid, cellid, idx_j][comp_j]       <L 949>
                                var_400 = wp::address(var_displ_scratch_out, var_0, var_1, var_340);
                                var_402 = wp::load(var_400);
                                var_401 = wp::extract(var_402, var_389);
                                var_403 = wp::mul(var_397, var_401);
                                var_404 = wp::add(var_381, var_403);
                            }
                            var_405 = wp::where(var_399, var_404, var_381);
                            // if not dsbl_damper:                                                <L 951>
                            var_406 = wp::unot(var_dsbl_damper);
                            if (var_406) {
                                // val_damper += K_ij * vel_corot_scratch_out[worldid, cellid, idx_j][comp_j]       <L 952>
                                var_407 = wp::address(var_vel_corot_scratch_out, var_0, var_1, var_340);
                                var_409 = wp::load(var_407);
                                var_408 = wp::extract(var_409, var_389);
                                var_410 = wp::mul(var_397, var_408);
                                var_411 = wp::add(var_388, var_410);
                            }
                            var_412 = wp::where(var_406, var_411, var_388);
                            wp::assign(var_336, var_405);
                            wp::assign(var_338, var_412);
                            wp::assign(var_228, var_392);
                            wp::assign(var_233, var_397);
                            goto start_for_19;
                        end_for_19:;
                        // frc_spring[comp_i] = val_spring                                        <L 954>
                        wp::assign_inplace(var_164, var_331, var_336);
                        // frc_damper[comp_i] = val_damper                                        <L 955>
                        wp::assign_inplace(var_166, var_331, var_338);
                        // frc_spring_world = wp.quat_rotate(cell_quat, frc_spring)               <L 958>
                        var_413 = wp::quat_rotate(var_61, var_164);
                        // frc_damper_world = wp.quat_rotate(cell_quat, frc_damper)               <L 959>
                        var_414 = wp::quat_rotate(var_61, var_166);
                        // frc_damper_world = frc_damper_world * flex_damping[f]                  <L 962>
                        var_415 = wp::address(var_flex_damping, var_6);
                        var_417 = wp::load(var_415);
                        var_416 = wp::mul(var_414, var_417);
                        // node_pos = flexnode_xpos_in[worldid, nstart + gidx_i]                  <L 965>
                        var_418 = wp::add(var_36, var_158);
                        var_419 = wp::address(var_flexnode_xpos_in, var_0, var_418);
                        var_421 = wp::load(var_419);
                        var_420 = wp::copy(var_421);
                        // body_xipos = xipos_in[worldid, bodyid_i]                               <L 966>
                        var_422 = wp::address(var_xipos_in, var_0, var_161);
                        var_424 = wp::load(var_422);
                        var_423 = wp::copy(var_424);
                        // offset = body_xipos - node_pos                                         <L 968>
                        var_425 = wp::sub(var_423, var_420);
                        // if not dsbl_spring:                                                    <L 969>
                        var_426 = wp::unot(var_dsbl_spring);
                        if (var_426) {
                            // spatial_frc_s = wp.spatial_vector(frc_spring_world, -wp.cross(offset, frc_spring_world))       <L 970>
                            var_427 = wp::cross(var_425, var_413);
                            var_428 = wp::neg(var_427);
                            var_429 = wp::vec_t<6, wp::float32>(var_413, var_428);
                            // wp.atomic_add(flex_spring_body_force_out, worldid, bodyid_i, spatial_frc_s)       <L 971>
                            var_430 = wp::atomic_add(var_flex_spring_body_force_out, var_0, var_161, var_429);
                        }
                        // if not dsbl_damper:                                                    <L 973>
                        var_431 = wp::unot(var_dsbl_damper);
                        if (var_431) {
                            // spatial_frc_d = wp.spatial_vector(frc_damper_world, -wp.cross(offset, frc_damper_world))       <L 974>
                            var_432 = wp::cross(var_425, var_416);
                            var_433 = wp::neg(var_432);
                            var_434 = wp::vec_t<6, wp::float32>(var_416, var_433);
                            // wp.atomic_add(flex_damper_body_force_out, worldid, bodyid_i, spatial_frc_d)       <L 975>
                            var_435 = wp::atomic_add(var_flex_damper_body_force_out, var_0, var_161, var_434);
                        }
                        // idx_i += 1                                                             <L 977>
                        var_437 = wp::add(var_134, var_436);
                    }
                    var_438 = wp::where(var_147, var_340, var_75);
                    var_439 = wp::where(var_147, var_437, var_134);
                    wp::assign(var_75, var_438);
                    wp::assign(var_134, var_439);
                    goto start_for_13;
                end_for_13:;
                goto start_for_11;
            end_for_11:;
            goto start_for_9;
        end_for_9:;
    }
}



extern "C" __global__ void _flex_elasticity_22c6ffab_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_nflex,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::array_t<wp::int32> var_flex_dim,
    wp::array_t<wp::int32> var_flex_vertadr,
    wp::array_t<wp::int32> var_flex_edgeadr,
    wp::array_t<wp::int32> var_flex_elemadr,
    wp::array_t<wp::int32> var_flex_elemnum,
    wp::array_t<wp::int32> var_flex_elemdataadr,
    wp::array_t<wp::int32> var_flex_stiffnessadr,
    wp::array_t<wp::int32> var_flex_elemedgeadr,
    wp::array_t<wp::int32> var_flex_vertbodyid,
    wp::array_t<wp::int32> var_flex_elem,
    wp::array_t<wp::int32> var_flex_elemedge,
    wp::array_t<wp::float32> var_flexedge_length0,
    wp::array_t<wp::float32> var_flex_stiffness,
    wp::array_t<wp::float32> var_flex_damping,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexvert_xpos_in,
    wp::array_t<wp::float32> var_flexedge_length_in,
    wp::array_t<wp::float32> var_flexedge_velocity_in,
    bool var_dsbl_damper,
    wp::array_t<wp::vec_t<6, wp::float32>> var_flex_spring_body_force_out)
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
        wp::range_t var_10;
        wp::int32 var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        bool var_15;
        const wp::int32 var_16 = 0;
        bool var_17;
        wp::int32* var_18;
        bool var_19;
        wp::int32 var_20;
        wp::int32 var_21;
        wp::int32* var_22;
        wp::int32 var_23;
        wp::int32 var_24;
        const wp::int32 var_25 = 0;
        bool var_26;
        wp::float32* var_27;
        const wp::float32 var_28 = 0.0;
        bool var_29;
        wp::float32 var_30;
        wp::int32* var_31;
        wp::int32 var_32;
        wp::int32 var_33;
        wp::int32* var_34;
        wp::int32 var_35;
        wp::int32 var_36;
        const wp::int32 var_37 = 1;
        wp::int32 var_38;
        const wp::int32 var_39 = 1;
        wp::int32 var_40;
        wp::int32 var_41;
        const wp::int32 var_42 = 2;
        wp::int32 var_43;
        const wp::int32 var_44 = 1;
        bool var_45;
        const wp::int32 var_46 = 0;
        const wp::int32 var_47 = 1;
        const wp::int32 var_48 = 0;
        const wp::int32 var_49 = 0;
        const wp::int32 var_50 = 0;
        const wp::int32 var_51 = 0;
        const wp::int32 var_52 = 0;
        const wp::int32 var_53 = 0;
        const wp::int32 var_54 = 0;
        const wp::int32 var_55 = 0;
        const wp::int32 var_56 = 0;
        const wp::int32 var_57 = 0;
        const wp::int32 var_58 = 6;
        const wp::int32 var_59 = 2;
        wp::tuple_t<wp::int32, wp::int32> var_60;
        wp::mat_t<6, 2, wp::int32> var_61;
        const wp::int32 var_62 = 3;
        bool var_63;
        const wp::int32 var_64 = 0;
        const wp::int32 var_65 = 1;
        const wp::int32 var_66 = 1;
        const wp::int32 var_67 = 2;
        const wp::int32 var_68 = 2;
        const wp::int32 var_69 = 0;
        const wp::int32 var_70 = 2;
        const wp::int32 var_71 = 3;
        const wp::int32 var_72 = 0;
        const wp::int32 var_73 = 3;
        const wp::int32 var_74 = 1;
        const wp::int32 var_75 = 3;
        const wp::int32 var_76 = 6;
        const wp::int32 var_77 = 2;
        wp::tuple_t<wp::int32, wp::int32> var_78;
        wp::mat_t<6, 2, wp::int32> var_79;
        const wp::int32 var_80 = 1;
        const wp::int32 var_81 = 2;
        const wp::int32 var_82 = 2;
        const wp::int32 var_83 = 0;
        const wp::int32 var_84 = 0;
        const wp::int32 var_85 = 1;
        const wp::int32 var_86 = 0;
        const wp::int32 var_87 = 0;
        const wp::int32 var_88 = 0;
        const wp::int32 var_89 = 0;
        const wp::int32 var_90 = 0;
        const wp::int32 var_91 = 0;
        const wp::int32 var_92 = 6;
        const wp::int32 var_93 = 2;
        wp::tuple_t<wp::int32, wp::int32> var_94;
        wp::mat_t<6, 2, wp::int32> var_95;
        wp::mat_t<6, 2, wp::int32> var_96;
        wp::mat_t<6, 2, wp::int32> var_97;
        bool var_98;
        const wp::float32 var_99 = 0.0;
        bool var_100;
        bool var_101;
        wp::float32* var_102;
        wp::float32 var_103;
        wp::float32 var_104;
        const wp::float32 var_105 = 0.0;
        wp::float32 var_106;
        wp::int32* var_107;
        const wp::int32 var_108 = 1;
        wp::int32 var_109;
        wp::int32 var_110;
        wp::int32 var_111;
        wp::int32 var_112;
        wp::int32* var_113;
        wp::int32 var_114;
        wp::int32 var_115;
        wp::int32* var_116;
        wp::int32 var_117;
        wp::int32 var_118;
        wp::int32 var_119;
        wp::int32* var_120;
        const wp::int32 var_121 = 0;
        bool var_122;
        wp::int32 var_123;
        const wp::float32 var_124 = 0.0;
        const wp::int32 var_125 = 6;
        const wp::int32 var_126 = 6;
        wp::tuple_t<wp::int32, wp::int32> var_127;
        wp::mat_t<6, 6, wp::float32> var_128;
        wp::range_t var_129;
        wp::int32 var_130;
        const wp::int32 var_131 = 0;
        wp::int32 var_132;
        wp::int32 var_133;
        wp::int32* var_134;
        wp::int32 var_135;
        wp::int32 var_136;
        const wp::int32 var_137 = 1;
        wp::int32 var_138;
        wp::int32 var_139;
        wp::int32* var_140;
        wp::int32 var_141;
        wp::int32 var_142;
        wp::int32 var_143;
        wp::vec_t<3, wp::float32>* var_144;
        wp::vec_t<3, wp::float32> var_145;
        wp::vec_t<3, wp::float32> var_146;
        wp::int32 var_147;
        wp::vec_t<3, wp::float32>* var_148;
        wp::vec_t<3, wp::float32> var_149;
        wp::vec_t<3, wp::float32> var_150;
        const wp::int32 var_151 = 0;
        wp::float32 var_152;
        wp::float32 var_153;
        wp::float32 var_154;
        const wp::int32 var_155 = 0;
        wp::int32 var_156;
        wp::float32 var_157;
        wp::float32 var_158;
        wp::float32 var_159;
        const wp::int32 var_160 = 3;
        wp::int32 var_161;
        const wp::int32 var_162 = 1;
        wp::float32 var_163;
        wp::float32 var_164;
        wp::float32 var_165;
        const wp::int32 var_166 = 0;
        wp::int32 var_167;
        wp::float32 var_168;
        wp::float32 var_169;
        wp::float32 var_170;
        const wp::int32 var_171 = 3;
        wp::int32 var_172;
        const wp::int32 var_173 = 2;
        wp::float32 var_174;
        wp::float32 var_175;
        wp::float32 var_176;
        const wp::int32 var_177 = 0;
        wp::int32 var_178;
        wp::float32 var_179;
        wp::float32 var_180;
        wp::float32 var_181;
        const wp::int32 var_182 = 3;
        wp::int32 var_183;
        const wp::float32 var_184 = 0.0;
        wp::vec_t<6, wp::float32> var_185;
        wp::range_t var_186;
        wp::int32 var_187;
        wp::int32* var_188;
        wp::int32 var_189;
        wp::int32 var_190;
        wp::int32 var_191;
        wp::int32 var_192;
        wp::int32* var_193;
        wp::int32 var_194;
        wp::int32 var_195;
        wp::int32* var_196;
        wp::int32 var_197;
        wp::int32 var_198;
        wp::float32* var_199;
        wp::float32 var_200;
        wp::float32 var_201;
        wp::int32* var_202;
        wp::int32 var_203;
        wp::int32 var_204;
        wp::float32* var_205;
        wp::float32 var_206;
        wp::float32 var_207;
        wp::int32* var_208;
        wp::int32 var_209;
        wp::int32 var_210;
        wp::float32* var_211;
        wp::float32 var_212;
        wp::float32 var_213;
        wp::float32 var_214;
        wp::float32 var_215;
        wp::float32 var_216;
        wp::float32 var_217;
        wp::float32 var_218;
        wp::float32 var_219;
        wp::float32 var_220;
        wp::float32 var_221;
        wp::float32 var_222;
        wp::float32 var_223;
        const wp::float32 var_224 = 0.0;
        const wp::int32 var_225 = 6;
        const wp::int32 var_226 = 6;
        wp::tuple_t<wp::int32, wp::int32> var_227;
        wp::mat_t<6, 6, wp::float32> var_228;
        const wp::int32 var_229 = 21;
        wp::int32 var_230;
        wp::int32 var_231;
        const wp::int32 var_232 = 0;
        wp::int32 var_233;
        wp::range_t var_234;
        wp::int32 var_235;
        wp::range_t var_236;
        wp::int32 var_237;
        wp::int32 var_238;
        wp::float32* var_239;
        wp::float32 var_240;
        wp::int32 var_241;
        wp::float32* var_242;
        wp::float32 var_243;
        const wp::int32 var_244 = 1;
        wp::int32 var_245;
        const wp::float32 var_246 = 0.0;
        const wp::int32 var_247 = 6;
        const wp::int32 var_248 = 3;
        wp::tuple_t<wp::int32, wp::int32> var_249;
        wp::mat_t<6, 3, wp::float32> var_250;
        wp::range_t var_251;
        wp::int32 var_252;
        wp::range_t var_253;
        wp::int32 var_254;
        const wp::int32 var_255 = 0;
        const wp::int32 var_256 = 0;
        wp::float32 var_257;
        const wp::int32 var_258 = 3;
        wp::int32 var_259;
        wp::int32 var_260;
        wp::float32 var_261;
        wp::float32 var_262;
        wp::float32 var_263;
        wp::float32 var_264;
        wp::int32 var_265;
        const wp::int32 var_266 = 1;
        wp::float32 var_267;
        const wp::int32 var_268 = 3;
        wp::int32 var_269;
        wp::int32 var_270;
        wp::float32 var_271;
        wp::float32 var_272;
        wp::float32 var_273;
        wp::float32 var_274;
        wp::int32 var_275;
        const wp::int32 var_276 = 2;
        wp::float32 var_277;
        const wp::int32 var_278 = 3;
        wp::int32 var_279;
        wp::int32 var_280;
        wp::float32 var_281;
        wp::float32 var_282;
        wp::float32 var_283;
        wp::float32 var_284;
        wp::int32 var_285;
        const wp::int32 var_286 = 1;
        const wp::int32 var_287 = 0;
        wp::float32 var_288;
        const wp::int32 var_289 = 3;
        wp::int32 var_290;
        wp::int32 var_291;
        wp::float32 var_292;
        wp::float32 var_293;
        wp::float32 var_294;
        wp::float32 var_295;
        wp::int32 var_296;
        const wp::int32 var_297 = 1;
        wp::float32 var_298;
        const wp::int32 var_299 = 3;
        wp::int32 var_300;
        wp::int32 var_301;
        wp::float32 var_302;
        wp::float32 var_303;
        wp::float32 var_304;
        wp::float32 var_305;
        wp::int32 var_306;
        const wp::int32 var_307 = 2;
        wp::float32 var_308;
        const wp::int32 var_309 = 3;
        wp::int32 var_310;
        wp::int32 var_311;
        wp::float32 var_312;
        wp::float32 var_313;
        wp::float32 var_314;
        wp::float32 var_315;
        wp::int32 var_316;
        wp::range_t var_317;
        wp::int32 var_318;
        wp::int32 var_319;
        wp::int32* var_320;
        wp::int32 var_321;
        wp::int32 var_322;
        wp::int32* var_323;
        wp::int32 var_324;
        wp::int32 var_325;
        wp::int32* var_326;
        wp::int32 var_327;
        wp::int32 var_328;
        wp::vec_t<3, wp::float32> var_329;
        wp::int32* var_330;
        wp::int32 var_331;
        wp::int32 var_332;
        wp::vec_t<3, wp::float32>* var_333;
        wp::vec_t<3, wp::float32> var_334;
        wp::vec_t<3, wp::float32> var_335;
        wp::vec_t<3, wp::float32>* var_336;
        wp::vec_t<3, wp::float32> var_337;
        wp::vec_t<3, wp::float32> var_338;
        wp::vec_t<3, wp::float32> var_339;
        wp::vec_t<3, wp::float32> var_340;
        wp::vec_t<3, wp::float32> var_341;
        wp::vec_t<6, wp::float32> var_342;
        wp::vec_t<6, wp::float32> var_343;
        //---------
        // forward
        // def _flex_elasticity(                                                                  <L 593>
        // worldid, elemid = wp.tid()                                                             <L 621>
        builtin_tid2d(var_0, var_1);
        // timestep = opt_timestep[worldid % opt_timestep.shape[0]]                               <L 622>
        var_2 = &(var_opt_timestep.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        var_7 = wp::address(var_opt_timestep, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // for i in range(nflex):                                                                 <L 624>
        var_10 = wp::range(var_nflex);
        start_for_0:;
            if (iter_cmp(var_10) == 0) goto end_for_0;
            var_11 = wp::iter_next(var_10);
            // locid = elemid - flex_elemadr[i]                                                   <L 625>
            var_12 = wp::address(var_flex_elemadr, var_11);
            var_14 = wp::load(var_12);
            var_13 = wp::sub(var_1, var_14);
            // if locid >= 0 and locid < flex_elemnum[i]:                                         <L 626>
            var_17 = (var_13 >= var_16);
            var_15 = var_17;
            if (var_15) {
                var_18 = wp::address(var_flex_elemnum, var_11);
                var_20 = wp::load(var_18);
                var_19 = (var_13 < var_20);
                var_15 = var_15 && var_19;
            }
            if (var_15) {
                // f = i                                                                          <L 627>
                var_21 = wp::copy(var_11);
                // break                                                                          <L 628>
                goto end_for_0;
            }
            goto start_for_0;
        end_for_0:;
        // stiffness_adr_base = flex_stiffnessadr[f]                                              <L 630>
        var_22 = wp::address(var_flex_stiffnessadr, var_21);
        var_24 = wp::load(var_22);
        var_23 = wp::copy(var_24);
        // if stiffness_adr_base < 0:                                                             <L 631>
        var_26 = (var_23 < var_25);
        if (var_26) {
            // return                                                                             <L 632>
            continue;
        }
        // if flex_stiffness[stiffness_adr_base] == 0.0:                                          <L 633>
        var_27 = wp::address(var_flex_stiffness, var_23);
        var_30 = wp::load(var_27);
        var_29 = (var_30 == var_28);
        if (var_29) {
            // return                                                                             <L 634>
            continue;
        }
        // local_elemid = elemid - flex_elemadr[f]                                                <L 636>
        var_31 = wp::address(var_flex_elemadr, var_21);
        var_33 = wp::load(var_31);
        var_32 = wp::sub(var_1, var_33);
        // dim = flex_dim[f]                                                                      <L 637>
        var_34 = wp::address(var_flex_dim, var_21);
        var_36 = wp::load(var_34);
        var_35 = wp::copy(var_36);
        // nvert = dim + 1                                                                        <L 638>
        var_38 = wp::add(var_35, var_37);
        // nedge = nvert * (nvert - 1) / 2                                                        <L 639>
        var_40 = wp::sub(var_38, var_39);
        var_41 = wp::mul(var_38, var_40);
        var_43 = wp::div(var_41, var_42);
        // edges = wp.where(                                                                      <L 640>
        // dim == 1,                                                                              <L 641>
        var_45 = (var_35 == var_44);
        // wp.matrix(0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, shape=(6, 2), dtype=int),                <L 642>
        var_60 = wp::tuple(var_58, var_59);
        var_61 = wp::mat_t<6, 2, wp::int32>({var_46, var_47, var_48, var_49, var_50, var_51, var_52, var_53, var_54, var_55, var_56, var_57});
        // wp.where(                                                                              <L 643>
        // dim == 3,                                                                              <L 644>
        var_63 = (var_35 == var_62);
        // wp.matrix(0, 1, 1, 2, 2, 0, 2, 3, 0, 3, 1, 3, shape=(6, 2), dtype=int),                <L 645>
        var_78 = wp::tuple(var_76, var_77);
        var_79 = wp::mat_t<6, 2, wp::int32>({var_64, var_65, var_66, var_67, var_68, var_69, var_70, var_71, var_72, var_73, var_74, var_75});
        // wp.matrix(1, 2, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, shape=(6, 2), dtype=int),                <L 646>
        var_94 = wp::tuple(var_92, var_93);
        var_95 = wp::mat_t<6, 2, wp::int32>({var_80, var_81, var_82, var_83, var_84, var_85, var_86, var_87, var_88, var_89, var_90, var_91});
        var_96 = wp::where(var_63, var_79, var_95);
        var_97 = wp::where(var_45, var_61, var_96);
        // if timestep > 0.0 and not dsbl_damper:                                                 <L 649>
        var_100 = (var_8 > var_99);
        var_98 = var_100;
        if (var_98) {
            var_101 = wp::unot(var_dsbl_damper);
            var_98 = var_98 && var_101;
        }
        if (var_98) {
            // kD = flex_damping[f] / timestep                                                    <L 650>
            var_102 = wp::address(var_flex_damping, var_21);
            var_104 = wp::load(var_102);
            var_103 = wp::div(var_104, var_8);
        }
        if (!var_98) {
            // kD = 0.0                                                                           <L 652>
        }
        var_106 = wp::where(var_98, var_103, var_105);
        // elem_data_adr = flex_elemdataadr[f] + local_elemid * (dim + 1)                         <L 654>
        var_107 = wp::address(var_flex_elemdataadr, var_21);
        var_109 = wp::add(var_35, var_108);
        var_110 = wp::mul(var_32, var_109);
        var_112 = wp::load(var_107);
        var_111 = wp::add(var_112, var_110);
        // vbase = flex_vertadr[f]                                                                <L 655>
        var_113 = wp::address(var_flex_vertadr, var_21);
        var_115 = wp::load(var_113);
        var_114 = wp::copy(var_115);
        // vert0_check = flex_elem[elem_data_adr]                                                 <L 658>
        var_116 = wp::address(var_flex_elem, var_111);
        var_118 = wp::load(var_116);
        var_117 = wp::copy(var_118);
        // if flex_vertbodyid[vbase + vert0_check] < 0:                                           <L 659>
        var_119 = wp::add(var_114, var_117);
        var_120 = wp::address(var_flex_vertbodyid, var_119);
        var_123 = wp::load(var_120);
        var_122 = (var_123 < var_121);
        if (var_122) {
            // return                                                                             <L 660>
            continue;
        }
        // gradient = wp.matrix(0.0, shape=(6, 6))                                                <L 661>
        var_127 = wp::tuple(var_125, var_126);
        var_128 = wp::mat_t<6, 6, wp::float32>(var_124);
        // for e in range(nedge):                                                                 <L 662>
        var_129 = wp::range(var_43);
        start_for_5:;
            if (iter_cmp(var_129) == 0) goto end_for_5;
            var_130 = wp::iter_next(var_129);
            // vert0 = flex_elem[elem_data_adr + edges[e, 0]]                                     <L 663>
            var_132 = wp::extract(var_97, var_130, var_131);
            var_133 = wp::add(var_111, var_132);
            var_134 = wp::address(var_flex_elem, var_133);
            var_136 = wp::load(var_134);
            var_135 = wp::copy(var_136);
            // vert1 = flex_elem[elem_data_adr + edges[e, 1]]                                     <L 664>
            var_138 = wp::extract(var_97, var_130, var_137);
            var_139 = wp::add(var_111, var_138);
            var_140 = wp::address(var_flex_elem, var_139);
            var_142 = wp::load(var_140);
            var_141 = wp::copy(var_142);
            // xpos0 = flexvert_xpos_in[worldid, vbase + vert0]                                   <L 665>
            var_143 = wp::add(var_114, var_135);
            var_144 = wp::address(var_flexvert_xpos_in, var_0, var_143);
            var_146 = wp::load(var_144);
            var_145 = wp::copy(var_146);
            // xpos1 = flexvert_xpos_in[worldid, vbase + vert1]                                   <L 666>
            var_147 = wp::add(var_114, var_141);
            var_148 = wp::address(var_flexvert_xpos_in, var_0, var_147);
            var_150 = wp::load(var_148);
            var_149 = wp::copy(var_150);
            // for i in range(3):                                                                 <L 667>
            // gradient[e, 0 + i] = xpos0[i] - xpos1[i]                                           <L 668>
            var_152 = wp::extract(var_145, var_151);
            var_153 = wp::extract(var_149, var_151);
            var_154 = wp::sub(var_152, var_153);
            var_156 = wp::add(var_155, var_151);
            wp::assign_inplace(var_128, var_130, var_156, var_154);
            // gradient[e, 3 + i] = xpos1[i] - xpos0[i]                                           <L 669>
            var_157 = wp::extract(var_149, var_151);
            var_158 = wp::extract(var_145, var_151);
            var_159 = wp::sub(var_157, var_158);
            var_161 = wp::add(var_160, var_151);
            wp::assign_inplace(var_128, var_130, var_161, var_159);
            // gradient[e, 0 + i] = xpos0[i] - xpos1[i]                                           <L 668>
            var_163 = wp::extract(var_145, var_162);
            var_164 = wp::extract(var_149, var_162);
            var_165 = wp::sub(var_163, var_164);
            var_167 = wp::add(var_166, var_162);
            wp::assign_inplace(var_128, var_130, var_167, var_165);
            // gradient[e, 3 + i] = xpos1[i] - xpos0[i]                                           <L 669>
            var_168 = wp::extract(var_149, var_162);
            var_169 = wp::extract(var_145, var_162);
            var_170 = wp::sub(var_168, var_169);
            var_172 = wp::add(var_171, var_162);
            wp::assign_inplace(var_128, var_130, var_172, var_170);
            // gradient[e, 0 + i] = xpos0[i] - xpos1[i]                                           <L 668>
            var_174 = wp::extract(var_145, var_173);
            var_175 = wp::extract(var_149, var_173);
            var_176 = wp::sub(var_174, var_175);
            var_178 = wp::add(var_177, var_173);
            wp::assign_inplace(var_128, var_130, var_178, var_176);
            // gradient[e, 3 + i] = xpos1[i] - xpos0[i]                                           <L 669>
            var_179 = wp::extract(var_149, var_173);
            var_180 = wp::extract(var_145, var_173);
            var_181 = wp::sub(var_179, var_180);
            var_183 = wp::add(var_182, var_173);
            wp::assign_inplace(var_128, var_130, var_183, var_181);
            goto start_for_5;
        end_for_5:;
        // elongation = wp.spatial_vectorf(0.0)                                                   <L 671>
        var_185 = wp::vec_t<6, wp::float32>(var_184);
        // for e in range(nedge):                                                                 <L 672>
        var_186 = wp::range(var_43);
        start_for_7:;
            if (iter_cmp(var_186) == 0) goto end_for_7;
            var_187 = wp::iter_next(var_186);
            // idx = flex_elemedge[flex_elemedgeadr[f] + local_elemid * nedge + e]                <L 673>
            var_188 = wp::address(var_flex_elemedgeadr, var_21);
            var_189 = wp::mul(var_32, var_43);
            var_191 = wp::load(var_188);
            var_190 = wp::add(var_191, var_189);
            var_192 = wp::add(var_190, var_187);
            var_193 = wp::address(var_flex_elemedge, var_192);
            var_195 = wp::load(var_193);
            var_194 = wp::copy(var_195);
            // vel = flexedge_velocity_in[worldid, flex_edgeadr[f] + idx]                         <L 674>
            var_196 = wp::address(var_flex_edgeadr, var_21);
            var_198 = wp::load(var_196);
            var_197 = wp::add(var_198, var_194);
            var_199 = wp::address(var_flexedge_velocity_in, var_0, var_197);
            var_201 = wp::load(var_199);
            var_200 = wp::copy(var_201);
            // deformed = flexedge_length_in[worldid, flex_edgeadr[f] + idx]                      <L 675>
            var_202 = wp::address(var_flex_edgeadr, var_21);
            var_204 = wp::load(var_202);
            var_203 = wp::add(var_204, var_194);
            var_205 = wp::address(var_flexedge_length_in, var_0, var_203);
            var_207 = wp::load(var_205);
            var_206 = wp::copy(var_207);
            // reference = flexedge_length0[flex_edgeadr[f] + idx]                                <L 676>
            var_208 = wp::address(var_flex_edgeadr, var_21);
            var_210 = wp::load(var_208);
            var_209 = wp::add(var_210, var_194);
            var_211 = wp::address(var_flexedge_length0, var_209);
            var_213 = wp::load(var_211);
            var_212 = wp::copy(var_213);
            // previous = deformed - vel * timestep                                               <L 677>
            var_214 = wp::mul(var_200, var_8);
            var_215 = wp::sub(var_206, var_214);
            // elongation[e] = deformed * deformed - reference * reference + (deformed * deformed - previous * previous) * kD       <L 678>
            var_216 = wp::mul(var_206, var_206);
            var_217 = wp::mul(var_212, var_212);
            var_218 = wp::sub(var_216, var_217);
            var_219 = wp::mul(var_206, var_206);
            var_220 = wp::mul(var_215, var_215);
            var_221 = wp::sub(var_219, var_220);
            var_222 = wp::mul(var_221, var_106);
            var_223 = wp::add(var_218, var_222);
            wp::assign_inplace(var_185, var_187, var_223);
            goto start_for_7;
        end_for_7:;
        // metric = wp.matrix(0.0, shape=(6, 6))                                                  <L 680>
        var_227 = wp::tuple(var_225, var_226);
        var_228 = wp::mat_t<6, 6, wp::float32>(var_224);
        // stiffness_size = 21                                                                    <L 681>
        // stiffness_adr = stiffness_adr_base + local_elemid * stiffness_size                     <L 682>
        var_230 = wp::mul(var_32, var_229);
        var_231 = wp::add(var_23, var_230);
        // id = int(0)                                                                            <L 683>
        var_233 = wp::int(var_232);
        // for ed1 in range(nedge):                                                               <L 684>
        var_234 = wp::range(var_43);
        start_for_9:;
            if (iter_cmp(var_234) == 0) goto end_for_9;
            var_235 = wp::iter_next(var_234);
            // for ed2 in range(ed1, nedge):                                                      <L 685>
            var_236 = wp::range(var_235, var_43);
            start_for_11:;
                if (iter_cmp(var_236) == 0) goto end_for_11;
                var_237 = wp::iter_next(var_236);
                // metric[ed1, ed2] = flex_stiffness[stiffness_adr + id]                          <L 686>
                var_238 = wp::add(var_231, var_233);
                var_239 = wp::address(var_flex_stiffness, var_238);
                var_240 = wp::load(var_239);
                wp::assign_inplace(var_228, var_235, var_237, var_240);
                // metric[ed2, ed1] = flex_stiffness[stiffness_adr + id]                          <L 687>
                var_241 = wp::add(var_231, var_233);
                var_242 = wp::address(var_flex_stiffness, var_241);
                var_243 = wp::load(var_242);
                wp::assign_inplace(var_228, var_237, var_235, var_243);
                // id += 1                                                                        <L 688>
                var_245 = wp::add(var_233, var_244);
                wp::assign(var_233, var_245);
                goto start_for_11;
            end_for_11:;
            goto start_for_9;
        end_for_9:;
        // force = wp.matrix(0.0, shape=(6, 3))                                                   <L 690>
        var_249 = wp::tuple(var_247, var_248);
        var_250 = wp::mat_t<6, 3, wp::float32>(var_246);
        // for ed1 in range(nedge):                                                               <L 691>
        var_251 = wp::range(var_43);
        start_for_13:;
            if (iter_cmp(var_251) == 0) goto end_for_13;
            var_252 = wp::iter_next(var_251);
            // for ed2 in range(nedge):                                                           <L 692>
            var_253 = wp::range(var_43);
            start_for_15:;
                if (iter_cmp(var_253) == 0) goto end_for_15;
                var_254 = wp::iter_next(var_253);
                // for i in range(2):                                                             <L 693>
                // for x in range(3):                                                             <L 694>
                // force[edges[ed2, i], x] -= elongation[ed1] * gradient[ed2, 3 * i + x] * metric[ed1, ed2]       <L 695>
                var_257 = wp::extract(var_185, var_252);
                var_259 = wp::mul(var_258, var_255);
                var_260 = wp::add(var_259, var_256);
                var_261 = wp::extract(var_128, var_254, var_260);
                var_262 = wp::mul(var_257, var_261);
                var_263 = wp::extract(var_228, var_252, var_254);
                var_264 = wp::mul(var_262, var_263);
                var_265 = wp::extract(var_97, var_254, var_255);
                wp::sub_inplace(var_250, var_265, var_256, var_264);
                var_267 = wp::extract(var_185, var_252);
                var_269 = wp::mul(var_268, var_255);
                var_270 = wp::add(var_269, var_266);
                var_271 = wp::extract(var_128, var_254, var_270);
                var_272 = wp::mul(var_267, var_271);
                var_273 = wp::extract(var_228, var_252, var_254);
                var_274 = wp::mul(var_272, var_273);
                var_275 = wp::extract(var_97, var_254, var_255);
                wp::sub_inplace(var_250, var_275, var_266, var_274);
                var_277 = wp::extract(var_185, var_252);
                var_279 = wp::mul(var_278, var_255);
                var_280 = wp::add(var_279, var_276);
                var_281 = wp::extract(var_128, var_254, var_280);
                var_282 = wp::mul(var_277, var_281);
                var_283 = wp::extract(var_228, var_252, var_254);
                var_284 = wp::mul(var_282, var_283);
                var_285 = wp::extract(var_97, var_254, var_255);
                wp::sub_inplace(var_250, var_285, var_276, var_284);
                // for x in range(3):                                                             <L 694>
                // force[edges[ed2, i], x] -= elongation[ed1] * gradient[ed2, 3 * i + x] * metric[ed1, ed2]       <L 695>
                var_288 = wp::extract(var_185, var_252);
                var_290 = wp::mul(var_289, var_286);
                var_291 = wp::add(var_290, var_287);
                var_292 = wp::extract(var_128, var_254, var_291);
                var_293 = wp::mul(var_288, var_292);
                var_294 = wp::extract(var_228, var_252, var_254);
                var_295 = wp::mul(var_293, var_294);
                var_296 = wp::extract(var_97, var_254, var_286);
                wp::sub_inplace(var_250, var_296, var_287, var_295);
                var_298 = wp::extract(var_185, var_252);
                var_300 = wp::mul(var_299, var_286);
                var_301 = wp::add(var_300, var_297);
                var_302 = wp::extract(var_128, var_254, var_301);
                var_303 = wp::mul(var_298, var_302);
                var_304 = wp::extract(var_228, var_252, var_254);
                var_305 = wp::mul(var_303, var_304);
                var_306 = wp::extract(var_97, var_254, var_286);
                wp::sub_inplace(var_250, var_306, var_297, var_305);
                var_308 = wp::extract(var_185, var_252);
                var_310 = wp::mul(var_309, var_286);
                var_311 = wp::add(var_310, var_307);
                var_312 = wp::extract(var_128, var_254, var_311);
                var_313 = wp::mul(var_308, var_312);
                var_314 = wp::extract(var_228, var_252, var_254);
                var_315 = wp::mul(var_313, var_314);
                var_316 = wp::extract(var_97, var_254, var_286);
                wp::sub_inplace(var_250, var_316, var_307, var_315);
                goto start_for_15;
            end_for_15:;
            wp::assign(var_237, var_254);
            goto start_for_13;
        end_for_13:;
        // for v in range(nvert):                                                                 <L 697>
        var_317 = wp::range(var_38);
        start_for_17:;
            if (iter_cmp(var_317) == 0) goto end_for_17;
            var_318 = wp::iter_next(var_317);
            // vert = flex_elem[elem_data_adr + v]                                                <L 698>
            var_319 = wp::add(var_111, var_318);
            var_320 = wp::address(var_flex_elem, var_319);
            var_322 = wp::load(var_320);
            var_321 = wp::copy(var_322);
            // bodyid = flex_vertbodyid[flex_vertadr[f] + vert]                                   <L 699>
            var_323 = wp::address(var_flex_vertadr, var_21);
            var_325 = wp::load(var_323);
            var_324 = wp::add(var_325, var_321);
            var_326 = wp::address(var_flex_vertbodyid, var_324);
            var_328 = wp::load(var_326);
            var_327 = wp::copy(var_328);
            // frc = force[v]                                                                     <L 701>
            var_329 = wp::extract(var_250, var_318);
            // node_pos = flexvert_xpos_in[worldid, flex_vertadr[f] + vert]                       <L 703>
            var_330 = wp::address(var_flex_vertadr, var_21);
            var_332 = wp::load(var_330);
            var_331 = wp::add(var_332, var_321);
            var_333 = wp::address(var_flexvert_xpos_in, var_0, var_331);
            var_335 = wp::load(var_333);
            var_334 = wp::copy(var_335);
            // body_xipos = xipos_in[worldid, bodyid]                                             <L 704>
            var_336 = wp::address(var_xipos_in, var_0, var_327);
            var_338 = wp::load(var_336);
            var_337 = wp::copy(var_338);
            // offset = body_xipos - node_pos                                                     <L 705>
            var_339 = wp::sub(var_337, var_334);
            // spatial_frc = wp.spatial_vector(frc, -wp.cross(offset, frc))                       <L 706>
            var_340 = wp::cross(var_339, var_329);
            var_341 = wp::neg(var_340);
            var_342 = wp::vec_t<6, wp::float32>(var_329, var_341);
            // wp.atomic_add(flex_spring_body_force_out, worldid, bodyid, spatial_frc)            <L 707>
            var_343 = wp::atomic_add(var_flex_spring_body_force_out, var_0, var_327, var_342);
            goto start_for_17;
        end_for_17:;
    }
}

