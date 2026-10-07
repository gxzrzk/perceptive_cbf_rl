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


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/math.py:120
static CUDA_CALLABLE wp::vec_t<6, wp::float32> inert_vec_0(
    wp::vec_t<10, wp::float32> var_i,
    wp::vec_t<6, wp::float32> var_v)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 0;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::int32 var_5 = 3;
    wp::float32 var_6;
    const wp::int32 var_7 = 1;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    const wp::int32 var_11 = 4;
    wp::float32 var_12;
    const wp::int32 var_13 = 2;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    const wp::int32 var_17 = 8;
    wp::float32 var_18;
    const wp::int32 var_19 = 4;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    const wp::int32 var_23 = 7;
    wp::float32 var_24;
    const wp::int32 var_25 = 5;
    wp::float32 var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    const wp::int32 var_29 = 3;
    wp::float32 var_30;
    const wp::int32 var_31 = 0;
    wp::float32 var_32;
    wp::float32 var_33;
    const wp::int32 var_34 = 1;
    wp::float32 var_35;
    const wp::int32 var_36 = 1;
    wp::float32 var_37;
    wp::float32 var_38;
    wp::float32 var_39;
    const wp::int32 var_40 = 5;
    wp::float32 var_41;
    const wp::int32 var_42 = 2;
    wp::float32 var_43;
    wp::float32 var_44;
    wp::float32 var_45;
    const wp::int32 var_46 = 8;
    wp::float32 var_47;
    const wp::int32 var_48 = 3;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::float32 var_51;
    const wp::int32 var_52 = 6;
    wp::float32 var_53;
    const wp::int32 var_54 = 5;
    wp::float32 var_55;
    wp::float32 var_56;
    wp::float32 var_57;
    const wp::int32 var_58 = 4;
    wp::float32 var_59;
    const wp::int32 var_60 = 0;
    wp::float32 var_61;
    wp::float32 var_62;
    const wp::int32 var_63 = 5;
    wp::float32 var_64;
    const wp::int32 var_65 = 1;
    wp::float32 var_66;
    wp::float32 var_67;
    wp::float32 var_68;
    const wp::int32 var_69 = 2;
    wp::float32 var_70;
    const wp::int32 var_71 = 2;
    wp::float32 var_72;
    wp::float32 var_73;
    wp::float32 var_74;
    const wp::int32 var_75 = 7;
    wp::float32 var_76;
    const wp::int32 var_77 = 3;
    wp::float32 var_78;
    wp::float32 var_79;
    wp::float32 var_80;
    const wp::int32 var_81 = 6;
    wp::float32 var_82;
    const wp::int32 var_83 = 4;
    wp::float32 var_84;
    wp::float32 var_85;
    wp::float32 var_86;
    const wp::int32 var_87 = 8;
    wp::float32 var_88;
    const wp::int32 var_89 = 1;
    wp::float32 var_90;
    wp::float32 var_91;
    const wp::int32 var_92 = 7;
    wp::float32 var_93;
    const wp::int32 var_94 = 2;
    wp::float32 var_95;
    wp::float32 var_96;
    wp::float32 var_97;
    const wp::int32 var_98 = 9;
    wp::float32 var_99;
    const wp::int32 var_100 = 3;
    wp::float32 var_101;
    wp::float32 var_102;
    wp::float32 var_103;
    const wp::int32 var_104 = 6;
    wp::float32 var_105;
    const wp::int32 var_106 = 2;
    wp::float32 var_107;
    wp::float32 var_108;
    const wp::int32 var_109 = 8;
    wp::float32 var_110;
    const wp::int32 var_111 = 0;
    wp::float32 var_112;
    wp::float32 var_113;
    wp::float32 var_114;
    const wp::int32 var_115 = 9;
    wp::float32 var_116;
    const wp::int32 var_117 = 4;
    wp::float32 var_118;
    wp::float32 var_119;
    wp::float32 var_120;
    const wp::int32 var_121 = 7;
    wp::float32 var_122;
    const wp::int32 var_123 = 0;
    wp::float32 var_124;
    wp::float32 var_125;
    const wp::int32 var_126 = 6;
    wp::float32 var_127;
    const wp::int32 var_128 = 1;
    wp::float32 var_129;
    wp::float32 var_130;
    wp::float32 var_131;
    const wp::int32 var_132 = 9;
    wp::float32 var_133;
    const wp::int32 var_134 = 5;
    wp::float32 var_135;
    wp::float32 var_136;
    wp::float32 var_137;
    wp::vec_t<6, wp::float32> var_138;
    //---------
    // forward
    // def inert_vec(i: types.vec10, v: wp.spatial_vector) -> wp.spatial_vector:              <L 121>
    // return wp.spatial_vector(                                                              <L 123>
    // i[0] * v[0] + i[3] * v[1] + i[4] * v[2] - i[8] * v[4] + i[7] * v[5],                   <L 124>
    var_1 = wp::extract(var_i, var_0);
    var_3 = wp::extract(var_v, var_2);
    var_4 = wp::mul(var_1, var_3);
    var_6 = wp::extract(var_i, var_5);
    var_8 = wp::extract(var_v, var_7);
    var_9 = wp::mul(var_6, var_8);
    var_10 = wp::add(var_4, var_9);
    var_12 = wp::extract(var_i, var_11);
    var_14 = wp::extract(var_v, var_13);
    var_15 = wp::mul(var_12, var_14);
    var_16 = wp::add(var_10, var_15);
    var_18 = wp::extract(var_i, var_17);
    var_20 = wp::extract(var_v, var_19);
    var_21 = wp::mul(var_18, var_20);
    var_22 = wp::sub(var_16, var_21);
    var_24 = wp::extract(var_i, var_23);
    var_26 = wp::extract(var_v, var_25);
    var_27 = wp::mul(var_24, var_26);
    var_28 = wp::add(var_22, var_27);
    // i[3] * v[0] + i[1] * v[1] + i[5] * v[2] + i[8] * v[3] - i[6] * v[5],                   <L 125>
    var_30 = wp::extract(var_i, var_29);
    var_32 = wp::extract(var_v, var_31);
    var_33 = wp::mul(var_30, var_32);
    var_35 = wp::extract(var_i, var_34);
    var_37 = wp::extract(var_v, var_36);
    var_38 = wp::mul(var_35, var_37);
    var_39 = wp::add(var_33, var_38);
    var_41 = wp::extract(var_i, var_40);
    var_43 = wp::extract(var_v, var_42);
    var_44 = wp::mul(var_41, var_43);
    var_45 = wp::add(var_39, var_44);
    var_47 = wp::extract(var_i, var_46);
    var_49 = wp::extract(var_v, var_48);
    var_50 = wp::mul(var_47, var_49);
    var_51 = wp::add(var_45, var_50);
    var_53 = wp::extract(var_i, var_52);
    var_55 = wp::extract(var_v, var_54);
    var_56 = wp::mul(var_53, var_55);
    var_57 = wp::sub(var_51, var_56);
    // i[4] * v[0] + i[5] * v[1] + i[2] * v[2] - i[7] * v[3] + i[6] * v[4],                   <L 126>
    var_59 = wp::extract(var_i, var_58);
    var_61 = wp::extract(var_v, var_60);
    var_62 = wp::mul(var_59, var_61);
    var_64 = wp::extract(var_i, var_63);
    var_66 = wp::extract(var_v, var_65);
    var_67 = wp::mul(var_64, var_66);
    var_68 = wp::add(var_62, var_67);
    var_70 = wp::extract(var_i, var_69);
    var_72 = wp::extract(var_v, var_71);
    var_73 = wp::mul(var_70, var_72);
    var_74 = wp::add(var_68, var_73);
    var_76 = wp::extract(var_i, var_75);
    var_78 = wp::extract(var_v, var_77);
    var_79 = wp::mul(var_76, var_78);
    var_80 = wp::sub(var_74, var_79);
    var_82 = wp::extract(var_i, var_81);
    var_84 = wp::extract(var_v, var_83);
    var_85 = wp::mul(var_82, var_84);
    var_86 = wp::add(var_80, var_85);
    // i[8] * v[1] - i[7] * v[2] + i[9] * v[3],                                               <L 127>
    var_88 = wp::extract(var_i, var_87);
    var_90 = wp::extract(var_v, var_89);
    var_91 = wp::mul(var_88, var_90);
    var_93 = wp::extract(var_i, var_92);
    var_95 = wp::extract(var_v, var_94);
    var_96 = wp::mul(var_93, var_95);
    var_97 = wp::sub(var_91, var_96);
    var_99 = wp::extract(var_i, var_98);
    var_101 = wp::extract(var_v, var_100);
    var_102 = wp::mul(var_99, var_101);
    var_103 = wp::add(var_97, var_102);
    // i[6] * v[2] - i[8] * v[0] + i[9] * v[4],                                               <L 128>
    var_105 = wp::extract(var_i, var_104);
    var_107 = wp::extract(var_v, var_106);
    var_108 = wp::mul(var_105, var_107);
    var_110 = wp::extract(var_i, var_109);
    var_112 = wp::extract(var_v, var_111);
    var_113 = wp::mul(var_110, var_112);
    var_114 = wp::sub(var_108, var_113);
    var_116 = wp::extract(var_i, var_115);
    var_118 = wp::extract(var_v, var_117);
    var_119 = wp::mul(var_116, var_118);
    var_120 = wp::add(var_114, var_119);
    // i[7] * v[0] - i[6] * v[1] + i[9] * v[5],                                               <L 129>
    var_122 = wp::extract(var_i, var_121);
    var_124 = wp::extract(var_v, var_123);
    var_125 = wp::mul(var_122, var_124);
    var_127 = wp::extract(var_i, var_126);
    var_129 = wp::extract(var_v, var_128);
    var_130 = wp::mul(var_127, var_129);
    var_131 = wp::sub(var_125, var_130);
    var_133 = wp::extract(var_i, var_132);
    var_135 = wp::extract(var_v, var_134);
    var_136 = wp::mul(var_133, var_135);
    var_137 = wp::add(var_131, var_136);
    var_138 = wp::vec_t<6, wp::float32>({var_28, var_57, var_86, var_103, var_120, var_137});
    return var_138;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/math.py:147
static CUDA_CALLABLE wp::vec_t<6, wp::float32> motion_cross_force_0(
    wp::vec_t<6, wp::float32> var_v,
    wp::vec_t<6, wp::float32> var_f)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    const wp::int32 var_4 = 2;
    wp::float32 var_5;
    wp::vec_t<3, wp::float32> var_6;
    const wp::int32 var_7 = 3;
    wp::float32 var_8;
    const wp::int32 var_9 = 4;
    wp::float32 var_10;
    const wp::int32 var_11 = 5;
    wp::float32 var_12;
    wp::vec_t<3, wp::float32> var_13;
    const wp::int32 var_14 = 0;
    wp::float32 var_15;
    const wp::int32 var_16 = 1;
    wp::float32 var_17;
    const wp::int32 var_18 = 2;
    wp::float32 var_19;
    wp::vec_t<3, wp::float32> var_20;
    const wp::int32 var_21 = 3;
    wp::float32 var_22;
    const wp::int32 var_23 = 4;
    wp::float32 var_24;
    const wp::int32 var_25 = 5;
    wp::float32 var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::vec_t<3, wp::float32> var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::vec_t<3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<6, wp::float32> var_32;
    //---------
    // forward
    // def motion_cross_force(v: wp.spatial_vector, f: wp.spatial_vector) -> wp.spatial_vector:       <L 148>
    // v0 = wp.vec3(v[0], v[1], v[2])                                                         <L 150>
    var_1 = wp::extract(var_v, var_0);
    var_3 = wp::extract(var_v, var_2);
    var_5 = wp::extract(var_v, var_4);
    var_6 = wp::vec_t<3, wp::float32>(var_1, var_3, var_5);
    // v1 = wp.vec3(v[3], v[4], v[5])                                                         <L 151>
    var_8 = wp::extract(var_v, var_7);
    var_10 = wp::extract(var_v, var_9);
    var_12 = wp::extract(var_v, var_11);
    var_13 = wp::vec_t<3, wp::float32>(var_8, var_10, var_12);
    // f0 = wp.vec3(f[0], f[1], f[2])                                                         <L 152>
    var_15 = wp::extract(var_f, var_14);
    var_17 = wp::extract(var_f, var_16);
    var_19 = wp::extract(var_f, var_18);
    var_20 = wp::vec_t<3, wp::float32>(var_15, var_17, var_19);
    // f1 = wp.vec3(f[3], f[4], f[5])                                                         <L 153>
    var_22 = wp::extract(var_f, var_21);
    var_24 = wp::extract(var_f, var_23);
    var_26 = wp::extract(var_f, var_25);
    var_27 = wp::vec_t<3, wp::float32>(var_22, var_24, var_26);
    // ang = wp.cross(v0, f0) + wp.cross(v1, f1)                                              <L 155>
    var_28 = wp::cross(var_6, var_20);
    var_29 = wp::cross(var_13, var_27);
    var_30 = wp::add(var_28, var_29);
    // vel = wp.cross(v0, f1)                                                                 <L 156>
    var_31 = wp::cross(var_6, var_27);
    // return wp.spatial_vector(ang, vel)                                                     <L 158>
    var_32 = wp::vec_t<6, wp::float32>(var_30, var_31);
    return var_32;
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


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/math.py:133
static CUDA_CALLABLE wp::vec_t<6, wp::float32> motion_cross_0(
    wp::vec_t<6, wp::float32> var_u,
    wp::vec_t<6, wp::float32> var_v)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    const wp::int32 var_4 = 2;
    wp::float32 var_5;
    wp::vec_t<3, wp::float32> var_6;
    const wp::int32 var_7 = 3;
    wp::float32 var_8;
    const wp::int32 var_9 = 4;
    wp::float32 var_10;
    const wp::int32 var_11 = 5;
    wp::float32 var_12;
    wp::vec_t<3, wp::float32> var_13;
    const wp::int32 var_14 = 0;
    wp::float32 var_15;
    const wp::int32 var_16 = 1;
    wp::float32 var_17;
    const wp::int32 var_18 = 2;
    wp::float32 var_19;
    wp::vec_t<3, wp::float32> var_20;
    const wp::int32 var_21 = 3;
    wp::float32 var_22;
    const wp::int32 var_23 = 4;
    wp::float32 var_24;
    const wp::int32 var_25 = 5;
    wp::float32 var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::vec_t<3, wp::float32> var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::vec_t<3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<6, wp::float32> var_32;
    //---------
    // forward
    // def motion_cross(u: wp.spatial_vector, v: wp.spatial_vector) -> wp.spatial_vector:       <L 134>
    // u0 = wp.vec3(u[0], u[1], u[2])                                                         <L 136>
    var_1 = wp::extract(var_u, var_0);
    var_3 = wp::extract(var_u, var_2);
    var_5 = wp::extract(var_u, var_4);
    var_6 = wp::vec_t<3, wp::float32>(var_1, var_3, var_5);
    // u1 = wp.vec3(u[3], u[4], u[5])                                                         <L 137>
    var_8 = wp::extract(var_u, var_7);
    var_10 = wp::extract(var_u, var_9);
    var_12 = wp::extract(var_u, var_11);
    var_13 = wp::vec_t<3, wp::float32>(var_8, var_10, var_12);
    // v0 = wp.vec3(v[0], v[1], v[2])                                                         <L 138>
    var_15 = wp::extract(var_v, var_14);
    var_17 = wp::extract(var_v, var_16);
    var_19 = wp::extract(var_v, var_18);
    var_20 = wp::vec_t<3, wp::float32>(var_15, var_17, var_19);
    // v1 = wp.vec3(v[3], v[4], v[5])                                                         <L 139>
    var_22 = wp::extract(var_v, var_21);
    var_24 = wp::extract(var_v, var_23);
    var_26 = wp::extract(var_v, var_25);
    var_27 = wp::vec_t<3, wp::float32>(var_22, var_24, var_26);
    // ang = wp.cross(u0, v0)                                                                 <L 141>
    var_28 = wp::cross(var_6, var_20);
    // vel = wp.cross(u1, v0) + wp.cross(u0, v1)                                              <L 142>
    var_29 = wp::cross(var_13, var_20);
    var_30 = wp::cross(var_6, var_27);
    var_31 = wp::add(var_29, var_30);
    // return wp.spatial_vector(ang, vel)                                                     <L 144>
    var_32 = wp::vec_t<6, wp::float32>(var_28, var_31);
    return var_32;
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


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/derivative.py:587
static CUDA_CALLABLE wp::float32 _deriv_ellipsoid_fluid_0(
    wp::int32 var_opt_integrator,
    wp::array_t<wp::int32> var_geom_type,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_size,
    wp::array_t<wp::float32> var_geom_fluid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::int32 var_worldid,
    wp::int32 var_bodyid,
    wp::int32 var_rootid,
    wp::int32 var_geomadr,
    wp::int32 var_geomnum,
    wp::vec_t<6, wp::float32> var_cdof_i,
    wp::vec_t<6, wp::float32> var_cdof_j,
    wp::vec_t<3, wp::float32> var_wind,
    wp::float32 var_density,
    wp::float32 var_viscosity)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 3;
    bool var_1;
    wp::vec_t<3, wp::float32>* var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<6, wp::float32>* var_5;
    wp::vec_t<6, wp::float32> var_6;
    wp::vec_t<6, wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::vec_t<3, wp::float32>* var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::vec_t<3, wp::float32> var_13;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    const wp::float32 var_16 = 0.0;
    wp::float32 var_17;
    const wp::int32 var_18 = 0;
    wp::float32 var_19;
    const wp::int32 var_20 = 1;
    wp::float32 var_21;
    const wp::int32 var_22 = 2;
    wp::float32 var_23;
    wp::vec_t<3, wp::float32> var_24;
    const wp::int32 var_25 = 3;
    wp::float32 var_26;
    const wp::int32 var_27 = 4;
    wp::float32 var_28;
    const wp::int32 var_29 = 5;
    wp::float32 var_30;
    wp::vec_t<3, wp::float32> var_31;
    const wp::int32 var_32 = 0;
    wp::float32 var_33;
    const wp::int32 var_34 = 1;
    wp::float32 var_35;
    const wp::int32 var_36 = 2;
    wp::float32 var_37;
    wp::vec_t<3, wp::float32> var_38;
    const wp::int32 var_39 = 3;
    wp::float32 var_40;
    const wp::int32 var_41 = 4;
    wp::float32 var_42;
    const wp::int32 var_43 = 5;
    wp::float32 var_44;
    wp::vec_t<3, wp::float32> var_45;
    wp::range_t var_46;
    wp::int32 var_47;
    wp::int32 var_48;
    const wp::int32 var_49 = 0;
    wp::float32* var_50;
    wp::float32 var_51;
    wp::float32 var_52;
    const wp::float32 var_53 = 0.0;
    bool var_54;
    wp::shape_t* var_55;
    const wp::int32 var_56 = 0;
    wp::int32 var_57;
    wp::shape_t var_58;
    wp::int32 var_59;
    wp::vec_t<3, wp::float32>* var_60;
    wp::vec_t<3, wp::float32> var_61;
    wp::vec_t<3, wp::float32> var_62;
    wp::int32* var_63;
    wp::vec_t<3, wp::float32> var_64;
    wp::int32 var_65;
    wp::mat_t<3, 3, wp::float32>* var_66;
    wp::mat_t<3, 3, wp::float32> var_67;
    wp::mat_t<3, 3, wp::float32> var_68;
    wp::mat_t<3, 3, wp::float32> var_69;
    wp::vec_t<3, wp::float32>* var_70;
    wp::vec_t<3, wp::float32> var_71;
    wp::vec_t<3, wp::float32> var_72;
    wp::vec_t<3, wp::float32> var_73;
    wp::vec_t<3, wp::float32> var_74;
    wp::vec_t<3, wp::float32> var_75;
    wp::vec_t<3, wp::float32> var_76;
    wp::vec_t<3, wp::float32> var_77;
    bool var_78;
    const wp::int32 var_79 = 0;
    wp::float32 var_80;
    const wp::float32 var_81 = 0.0;
    bool var_82;
    const wp::int32 var_83 = 1;
    wp::float32 var_84;
    const wp::float32 var_85 = 0.0;
    bool var_86;
    const wp::int32 var_87 = 2;
    wp::float32 var_88;
    const wp::float32 var_89 = 0.0;
    bool var_90;
    wp::vec_t<3, wp::float32> var_91;
    wp::vec_t<3, wp::float32> var_92;
    wp::vec_t<3, wp::float32> var_93;
    wp::vec_t<3, wp::float32> var_94;
    wp::vec_t<3, wp::float32> var_95;
    const wp::int32 var_96 = 1;
    wp::float32* var_97;
    wp::float32 var_98;
    wp::float32 var_99;
    const wp::int32 var_100 = 2;
    wp::float32* var_101;
    wp::float32 var_102;
    wp::float32 var_103;
    const wp::int32 var_104 = 3;
    wp::float32* var_105;
    wp::float32 var_106;
    wp::float32 var_107;
    const wp::int32 var_108 = 4;
    wp::float32* var_109;
    wp::float32 var_110;
    wp::float32 var_111;
    const wp::int32 var_112 = 5;
    wp::float32* var_113;
    wp::float32 var_114;
    wp::float32 var_115;
    const wp::int32 var_116 = 6;
    wp::float32* var_117;
    const wp::int32 var_118 = 7;
    wp::float32* var_119;
    const wp::int32 var_120 = 8;
    wp::float32* var_121;
    wp::vec_t<3, wp::float32> var_122;
    wp::float32 var_123;
    wp::float32 var_124;
    wp::float32 var_125;
    const wp::int32 var_126 = 9;
    wp::float32* var_127;
    const wp::int32 var_128 = 10;
    wp::float32* var_129;
    const wp::int32 var_130 = 11;
    wp::float32* var_131;
    wp::vec_t<3, wp::float32> var_132;
    wp::float32 var_133;
    wp::float32 var_134;
    wp::float32 var_135;
    const wp::float32 var_136 = 0.0;
    wp::mat_t<3, 3, wp::float32> var_137;
    const wp::float32 var_138 = 0.0;
    wp::mat_t<3, 3, wp::float32> var_139;
    const wp::float32 var_140 = 0.0;
    wp::mat_t<3, 3, wp::float32> var_141;
    const wp::float32 var_142 = 0.0;
    wp::mat_t<3, 3, wp::float32> var_143;
    const wp::float32 var_144 = 0.0;
    bool var_145;
    wp::vec_t<3, wp::float32> var_146;
    wp::vec_t<3, wp::float32> var_147;
    wp::vec_t<3, wp::float32> var_148;
    wp::vec_t<3, wp::float32> var_149;
    wp::mat_t<3, 3, wp::float32> var_150;
    wp::mat_t<3, 3, wp::float32> var_151;
    wp::mat_t<3, 3, wp::float32> var_152;
    wp::mat_t<3, 3, wp::float32> var_153;
    wp::mat_t<3, 3, wp::float32> var_154;
    wp::mat_t<3, 3, wp::float32> var_155;
    wp::mat_t<3, 3, wp::float32> var_156;
    wp::mat_t<3, 3, wp::float32> var_157;
    wp::mat_t<3, 3, wp::float32> var_158;
    wp::mat_t<3, 3, wp::float32> var_159;
    wp::mat_t<3, 3, wp::float32> var_160;
    wp::mat_t<3, 3, wp::float32> var_161;
    wp::mat_t<3, 3, wp::float32> var_162;
    wp::mat_t<3, 3, wp::float32> var_163;
    wp::mat_t<3, 3, wp::float32> var_164;
    wp::mat_t<3, 3, wp::float32> var_165;
    wp::mat_t<3, 3, wp::float32> var_166;
    wp::mat_t<3, 3, wp::float32> var_167;
    wp::mat_t<3, 3, wp::float32> var_168;
    wp::mat_t<3, 3, wp::float32> var_169;
    wp::mat_t<3, 3, wp::float32> var_170;
    wp::mat_t<3, 3, wp::float32> var_171;
    wp::mat_t<3, 3, wp::float32> var_172;
    const wp::float32 var_173 = 4.1887902047863905;
    const wp::int32 var_174 = 0;
    wp::float32 var_175;
    wp::float32 var_176;
    const wp::int32 var_177 = 1;
    wp::float32 var_178;
    wp::float32 var_179;
    const wp::int32 var_180 = 2;
    wp::float32 var_181;
    wp::float32 var_182;
    wp::float32 var_183;
    wp::float32 var_184;
    wp::mat_t<3, 3, wp::float32> var_185;
    wp::mat_t<3, 3, wp::float32> var_186;
    wp::mat_t<3, 3, wp::float32> var_187;
    wp::mat_t<3, 3, wp::float32> var_188;
    wp::mat_t<3, 3, wp::float32> var_189;
    wp::mat_t<3, 3, wp::float32> var_190;
    const wp::int32 var_191 = 1;
    wp::float32 var_192;
    const wp::int32 var_193 = 2;
    wp::float32 var_194;
    wp::float32 var_195;
    const wp::int32 var_196 = 1;
    wp::float32 var_197;
    const wp::int32 var_198 = 2;
    wp::float32 var_199;
    wp::float32 var_200;
    wp::float32 var_201;
    const wp::int32 var_202 = 2;
    wp::float32 var_203;
    const wp::int32 var_204 = 0;
    wp::float32 var_205;
    wp::float32 var_206;
    const wp::int32 var_207 = 2;
    wp::float32 var_208;
    const wp::int32 var_209 = 0;
    wp::float32 var_210;
    wp::float32 var_211;
    wp::float32 var_212;
    const wp::int32 var_213 = 0;
    wp::float32 var_214;
    const wp::int32 var_215 = 1;
    wp::float32 var_216;
    wp::float32 var_217;
    const wp::int32 var_218 = 0;
    wp::float32 var_219;
    const wp::int32 var_220 = 1;
    wp::float32 var_221;
    wp::float32 var_222;
    wp::float32 var_223;
    wp::float32 var_224;
    wp::float32 var_225;
    wp::float32 var_226;
    const wp::int32 var_227 = 0;
    wp::float32 var_228;
    const wp::int32 var_229 = 1;
    wp::float32 var_230;
    const wp::int32 var_231 = 2;
    wp::float32 var_232;
    wp::float32 var_233;
    wp::float32 var_234;
    wp::float32 var_235;
    wp::float32 var_236;
    wp::float32 var_237;
    wp::float32 var_238;
    wp::float32 var_239;
    wp::float32 var_240;
    wp::float32 var_241;
    wp::float32 var_242;
    wp::float32 var_243;
    wp::float32 var_244;
    wp::float32 var_245;
    wp::float32 var_246;
    wp::float32 var_247;
    wp::float32 var_248;
    wp::float32 var_249;
    wp::float32 var_250;
    const wp::float32 var_251 = 3.141592653589793;
    wp::float32 var_252;
    wp::float32 var_253;
    const wp::float32 var_254 = 1e-15;
    wp::float32 var_255;
    wp::float32 var_256;
    wp::float32 var_257;
    wp::float32 var_258;
    wp::float32 var_259;
    wp::float32 var_260;
    wp::float32 var_261;
    wp::float32 var_262;
    wp::float32 var_263;
    wp::float32 var_264;
    wp::float32 var_265;
    wp::float32 var_266;
    wp::float32 var_267;
    wp::float32 var_268;
    wp::float32 var_269;
    wp::float32 var_270;
    wp::float32 var_271;
    wp::float32 var_272;
    wp::float32 var_273;
    wp::float32 var_274;
    wp::float32 var_275;
    wp::float32 var_276;
    wp::float32 var_277;
    wp::float32 var_278;
    wp::float32 var_279;
    wp::float32 var_280;
    wp::float32 var_281;
    wp::vec_t<3, wp::float32> var_282;
    wp::mat_t<3, 3, wp::float32> var_283;
    const wp::float32 var_284 = 2.0;
    wp::float32 var_285;
    wp::mat_t<3, 3, wp::float32> var_286;
    wp::vec_t<3, wp::float32> var_287;
    wp::float32 var_288;
    wp::float32 var_289;
    wp::float32 var_290;
    wp::float32 var_291;
    wp::float32 var_292;
    wp::float32 var_293;
    wp::float32 var_294;
    wp::float32 var_295;
    wp::float32 var_296;
    wp::vec_t<3, wp::float32> var_297;
    wp::mat_t<3, 3, wp::float32> var_298;
    wp::mat_t<3, 3, wp::float32> var_299;
    wp::mat_t<3, 3, wp::float32> var_300;
    wp::mat_t<3, 3, wp::float32> var_301;
    wp::mat_t<3, 3, wp::float32> var_302;
    wp::vec_t<3, wp::float32> var_303;
    wp::mat_t<3, 3, wp::float32> var_304;
    wp::mat_t<3, 3, wp::float32> var_305;
    wp::mat_t<3, 3, wp::float32> var_306;
    wp::mat_t<3, 3, wp::float32> var_307;
    const wp::int32 var_308 = 0;
    wp::float32 var_309;
    const wp::int32 var_310 = 1;
    wp::float32 var_311;
    wp::float32 var_312;
    const wp::int32 var_313 = 2;
    wp::float32 var_314;
    wp::float32 var_315;
    const wp::int32 var_316 = 0;
    wp::float32 var_317;
    const wp::int32 var_318 = 1;
    wp::float32 var_319;
    wp::float32 var_320;
    const wp::int32 var_321 = 2;
    wp::float32 var_322;
    wp::float32 var_323;
    const wp::int32 var_324 = 0;
    wp::float32 var_325;
    const wp::int32 var_326 = 1;
    wp::float32 var_327;
    wp::float32 var_328;
    const wp::int32 var_329 = 2;
    wp::float32 var_330;
    wp::float32 var_331;
    wp::float32 var_332;
    wp::float32 var_333;
    const wp::float32 var_334 = 0.6666666666666666;
    const wp::int32 var_335 = 0;
    wp::float32 var_336;
    const wp::int32 var_337 = 1;
    wp::float32 var_338;
    wp::float32 var_339;
    const wp::int32 var_340 = 2;
    wp::float32 var_341;
    wp::float32 var_342;
    wp::float32 var_343;
    const wp::float32 var_344 = 3.141592653589793;
    wp::float32 var_345;
    wp::float32 var_346;
    const wp::float32 var_347 = 3.141592653589793;
    wp::float32 var_348;
    wp::float32 var_349;
    wp::float32 var_350;
    wp::float32 var_351;
    wp::float32 var_352;
    wp::float32 var_353;
    wp::float32 var_354;
    const wp::float32 var_355 = 1.0;
    wp::float32 var_356;
    wp::float32 var_357;
    const wp::float32 var_358 = 9.42477796076938;
    wp::float32 var_359;
    wp::float32 var_360;
    wp::float32 var_361;
    wp::float32 var_362;
    wp::float32 var_363;
    wp::float32 var_364;
    wp::float32 var_365;
    wp::float32 var_366;
    wp::float32 var_367;
    wp::float32 var_368;
    const wp::float32 var_369 = 3.141592653589793;
    wp::float32 var_370;
    wp::float32 var_371;
    wp::float32 var_372;
    wp::float32 var_373;
    wp::float32 var_374;
    wp::float32 var_375;
    wp::float32 var_376;
    wp::float32 var_377;
    wp::float32 var_378;
    wp::float32 var_379;
    wp::float32 var_380;
    wp::float32 var_381;
    wp::float32 var_382;
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
    wp::float32 var_395;
    wp::float32 var_396;
    wp::float32 var_397;
    wp::float32 var_398;
    wp::float32 var_399;
    wp::float32 var_400;
    wp::float32 var_401;
    wp::float32 var_402;
    wp::float32 var_403;
    wp::float32 var_404;
    wp::float32 var_405;
    wp::float32 var_406;
    wp::float32 var_407;
    wp::float32 var_408;
    wp::vec_t<3, wp::float32> var_409;
    wp::float32 var_410;
    wp::mat_t<3, 3, wp::float32> var_411;
    wp::vec_t<3, wp::float32> var_412;
    wp::mat_t<3, 3, wp::float32> var_413;
    wp::mat_t<3, 3, wp::float32> var_414;
    wp::float32 var_415;
    wp::float32 var_416;
    wp::mat_t<3, 3, wp::float32> var_417;
    wp::mat_t<3, 3, wp::float32> var_418;
    wp::mat_t<3, 3, wp::float32> var_419;
    wp::vec_t<3, wp::float32> var_420;
    wp::mat_t<3, 3, wp::float32> var_421;
    wp::mat_t<3, 3, wp::float32> var_422;
    wp::mat_t<3, 3, wp::float32> var_423;
    const wp::float32 var_424 = 3.141592653589793;
    wp::float32 var_425;
    wp::float32 var_426;
    wp::float32 var_427;
    const wp::float32 var_428 = 1.6755160819145563;
    wp::float32 var_429;
    wp::float32 var_430;
    wp::float32 var_431;
    wp::float32 var_432;
    wp::float32 var_433;
    const wp::int32 var_434 = 0;
    wp::float32 var_435;
    const wp::int32 var_436 = 1;
    wp::float32 var_437;
    const wp::int32 var_438 = 2;
    wp::float32 var_439;
    wp::vec_t<3, wp::float32> var_440;
    const wp::int32 var_441 = 0;
    wp::float32 var_442;
    wp::float32 var_443;
    const wp::int32 var_444 = 0;
    wp::float32 var_445;
    wp::float32 var_446;
    wp::float32 var_447;
    wp::float32 var_448;
    const wp::int32 var_449 = 1;
    wp::float32 var_450;
    wp::float32 var_451;
    const wp::int32 var_452 = 1;
    wp::float32 var_453;
    wp::float32 var_454;
    wp::float32 var_455;
    wp::float32 var_456;
    const wp::int32 var_457 = 2;
    wp::float32 var_458;
    wp::float32 var_459;
    const wp::int32 var_460 = 2;
    wp::float32 var_461;
    wp::float32 var_462;
    wp::float32 var_463;
    wp::float32 var_464;
    wp::vec_t<3, wp::float32> var_465;
    wp::vec_t<3, wp::float32> var_466;
    wp::float32 var_467;
    wp::float32 var_468;
    wp::float32 var_469;
    wp::float32 var_470;
    wp::vec_t<3, wp::float32> var_471;
    wp::vec_t<3, wp::float32> var_472;
    wp::vec_t<3, wp::float32> var_473;
    wp::float32 var_474;
    wp::float32 var_475;
    wp::float32 var_476;
    wp::mat_t<3, 3, wp::float32> var_477;
    wp::vec_t<3, wp::float32> var_478;
    wp::mat_t<3, 3, wp::float32> var_479;
    wp::mat_t<3, 3, wp::float32> var_480;
    wp::mat_t<3, 3, wp::float32> var_481;
    const wp::float32 var_482 = 0.5;
    wp::mat_t<3, 3, wp::float32> var_483;
    wp::mat_t<3, 3, wp::float32> var_484;
    wp::mat_t<3, 3, wp::float32> var_485;
    const wp::float32 var_486 = 0.5;
    wp::mat_t<3, 3, wp::float32> var_487;
    wp::mat_t<3, 3, wp::float32> var_488;
    wp::mat_t<3, 3, wp::float32> var_489;
    const wp::float32 var_490 = 0.5;
    wp::mat_t<3, 3, wp::float32> var_491;
    wp::mat_t<3, 3, wp::float32> var_492;
    wp::mat_t<3, 3, wp::float32> var_493;
    wp::mat_t<3, 3, wp::float32> var_494;
    wp::mat_t<3, 3, wp::float32> var_495;
    wp::mat_t<3, 3, wp::float32> var_496;
    wp::mat_t<3, 3, wp::float32> var_497;
    wp::mat_t<3, 3, wp::float32> var_498;
    wp::mat_t<3, 3, wp::float32> var_499;
    wp::vec_t<3, wp::float32> var_500;
    wp::vec_t<3, wp::float32> var_501;
    wp::vec_t<3, wp::float32> var_502;
    wp::vec_t<3, wp::float32> var_503;
    wp::vec_t<3, wp::float32> var_504;
    wp::vec_t<3, wp::float32> var_505;
    wp::vec_t<3, wp::float32> var_506;
    wp::vec_t<3, wp::float32> var_507;
    wp::vec_t<3, wp::float32> var_508;
    wp::vec_t<3, wp::float32> var_509;
    wp::vec_t<3, wp::float32> var_510;
    wp::vec_t<3, wp::float32> var_511;
    wp::vec_t<3, wp::float32> var_512;
    wp::vec_t<3, wp::float32> var_513;
    wp::vec_t<3, wp::float32> var_514;
    wp::float32 var_515;
    wp::float32 var_516;
    wp::float32 var_517;
    wp::float32 var_518;
    //---------
    // forward
    // def _deriv_ellipsoid_fluid(                                                            <L 588>
    // is_implicitfast = opt_integrator == IntegratorType.IMPLICITFAST                        <L 616>
    var_1 = (var_opt_integrator == var_0);
    // xipos = xipos_in[worldid, bodyid]                                                      <L 619>
    var_2 = wp::address(var_xipos_in, var_worldid, var_bodyid);
    var_4 = wp::load(var_2);
    var_3 = wp::copy(var_4);
    // cvel = cvel_in[worldid, bodyid]                                                        <L 620>
    var_5 = wp::address(var_cvel_in, var_worldid, var_bodyid);
    var_7 = wp::load(var_5);
    var_6 = wp::copy(var_7);
    // ang_global = wp.spatial_top(cvel)                                                      <L 621>
    var_8 = wp::spatial_top(var_6);
    // lin_global = wp.spatial_bottom(cvel)                                                   <L 622>
    var_9 = wp::spatial_bottom(var_6);
    // subtree_root = subtree_com_in[worldid, rootid]                                         <L 623>
    var_10 = wp::address(var_subtree_com_in, var_worldid, var_rootid);
    var_12 = wp::load(var_10);
    var_11 = wp::copy(var_12);
    // lin_com = lin_global - wp.cross(xipos - subtree_root, ang_global)                      <L 624>
    var_13 = wp::sub(var_3, var_11);
    var_14 = wp::cross(var_13, var_8);
    var_15 = wp::sub(var_9, var_14);
    // qderiv_contrib = float(0.0)                                                            <L 626>
    var_17 = wp::float(var_16);
    // cdof_ang_i = wp.vec3(cdof_i[0], cdof_i[1], cdof_i[2])                                  <L 628>
    var_19 = wp::extract(var_cdof_i, var_18);
    var_21 = wp::extract(var_cdof_i, var_20);
    var_23 = wp::extract(var_cdof_i, var_22);
    var_24 = wp::vec_t<3, wp::float32>(var_19, var_21, var_23);
    // cdof_lin_i = wp.vec3(cdof_i[3], cdof_i[4], cdof_i[5])                                  <L 629>
    var_26 = wp::extract(var_cdof_i, var_25);
    var_28 = wp::extract(var_cdof_i, var_27);
    var_30 = wp::extract(var_cdof_i, var_29);
    var_31 = wp::vec_t<3, wp::float32>(var_26, var_28, var_30);
    // cdof_ang_j = wp.vec3(cdof_j[0], cdof_j[1], cdof_j[2])                                  <L 630>
    var_33 = wp::extract(var_cdof_j, var_32);
    var_35 = wp::extract(var_cdof_j, var_34);
    var_37 = wp::extract(var_cdof_j, var_36);
    var_38 = wp::vec_t<3, wp::float32>(var_33, var_35, var_37);
    // cdof_lin_j = wp.vec3(cdof_j[3], cdof_j[4], cdof_j[5])                                  <L 631>
    var_40 = wp::extract(var_cdof_j, var_39);
    var_42 = wp::extract(var_cdof_j, var_41);
    var_44 = wp::extract(var_cdof_j, var_43);
    var_45 = wp::vec_t<3, wp::float32>(var_40, var_42, var_44);
    // for g in range(geomnum):                                                               <L 633>
    var_46 = wp::range(var_geomnum);
    start_for_0:;
        if (iter_cmp(var_46) == 0) goto end_for_0;
        var_47 = wp::iter_next(var_46);
        // geomid = geomadr + g                                                               <L 634>
        var_48 = wp::add(var_geomadr, var_47);
        // coef = geom_fluid[geomid, 0]                                                       <L 635>
        var_50 = wp::address(var_geom_fluid, var_48, var_49);
        var_52 = wp::load(var_50);
        var_51 = wp::copy(var_52);
        // if coef <= 0.0:                                                                    <L 636>
        var_54 = (var_51 <= var_53);
        if (var_54) {
            // continue                                                                       <L 637>
            goto start_for_0;
        }
        // size = geom_size[worldid % geom_size.shape[0], geomid]                             <L 639>
        var_55 = &(var_geom_size.shape);
        var_58 = wp::load(var_55);
        var_57 = wp::extract(var_58, var_56);
        var_59 = wp::mod(var_worldid, var_57);
        var_60 = wp::address(var_geom_size, var_59, var_48);
        var_62 = wp::load(var_60);
        var_61 = wp::copy(var_62);
        // semiaxes = geom_semiaxes(size, geom_type[geomid])                                  <L 640>
        var_63 = wp::address(var_geom_type, var_48);
        var_65 = wp::load(var_63);
        var_64 = geom_semiaxes_0(var_61, var_65);
        // geom_rot = geom_xmat_in[worldid, geomid]                                           <L 641>
        var_66 = wp::address(var_geom_xmat_in, var_worldid, var_48);
        var_68 = wp::load(var_66);
        var_67 = wp::copy(var_68);
        // geom_rotT = wp.transpose(geom_rot)                                                 <L 642>
        var_69 = wp::transpose(var_67);
        // geom_pos = geom_xpos_in[worldid, geomid]                                           <L 643>
        var_70 = wp::address(var_geom_xpos_in, var_worldid, var_48);
        var_72 = wp::load(var_70);
        var_71 = wp::copy(var_72);
        // lin_point = lin_com + wp.cross(ang_global, geom_pos - xipos)                       <L 646>
        var_73 = wp::sub(var_71, var_3);
        var_74 = wp::cross(var_8, var_73);
        var_75 = wp::add(var_15, var_74);
        // l_ang = geom_rotT @ ang_global                                                     <L 647>
        var_76 = wp::mul(var_69, var_8);
        // l_lin = geom_rotT @ lin_point                                                      <L 648>
        var_77 = wp::mul(var_69, var_75);
        // if wind[0] != 0.0 or wind[1] != 0.0 or wind[2] != 0.0:                             <L 650>
        var_80 = wp::extract(var_wind, var_79);
        var_82 = (var_80 != var_81);
        var_78 = var_82;
        if (!var_78) {
            var_84 = wp::extract(var_wind, var_83);
            var_86 = (var_84 != var_85);
            var_78 = var_78 || var_86;
        }
        if (!var_78) {
            var_88 = wp::extract(var_wind, var_87);
            var_90 = (var_88 != var_89);
            var_78 = var_78 || var_90;
        }
        if (var_78) {
            // l_lin -= geom_rotT @ wind                                                      <L 651>
            var_91 = wp::mul(var_69, var_wind);
            var_92 = wp::sub(var_77, var_91);
        }
        var_93 = wp::where(var_78, var_92, var_77);
        // ang_vel = l_ang                                                                    <L 653>
        var_94 = wp::copy(var_76);
        // lin_vel = l_lin                                                                    <L 654>
        var_95 = wp::copy(var_93);
        // blunt_drag_coef = geom_fluid[geomid, 1]                                            <L 657>
        var_97 = wp::address(var_geom_fluid, var_48, var_96);
        var_99 = wp::load(var_97);
        var_98 = wp::copy(var_99);
        // slender_drag_coef = geom_fluid[geomid, 2]                                          <L 658>
        var_101 = wp::address(var_geom_fluid, var_48, var_100);
        var_103 = wp::load(var_101);
        var_102 = wp::copy(var_103);
        // ang_drag_coef = geom_fluid[geomid, 3]                                              <L 659>
        var_105 = wp::address(var_geom_fluid, var_48, var_104);
        var_107 = wp::load(var_105);
        var_106 = wp::copy(var_107);
        // kutta_lift_coef = geom_fluid[geomid, 4]                                            <L 660>
        var_109 = wp::address(var_geom_fluid, var_48, var_108);
        var_111 = wp::load(var_109);
        var_110 = wp::copy(var_111);
        // magnus_lift_coef = geom_fluid[geomid, 5]                                           <L 661>
        var_113 = wp::address(var_geom_fluid, var_48, var_112);
        var_115 = wp::load(var_113);
        var_114 = wp::copy(var_115);
        // virtual_mass = wp.vec3(geom_fluid[geomid, 6], geom_fluid[geomid, 7], geom_fluid[geomid, 8])       <L 662>
        var_117 = wp::address(var_geom_fluid, var_48, var_116);
        var_119 = wp::address(var_geom_fluid, var_48, var_118);
        var_121 = wp::address(var_geom_fluid, var_48, var_120);
        var_123 = wp::load(var_117);
        var_124 = wp::load(var_119);
        var_125 = wp::load(var_121);
        var_122 = wp::vec_t<3, wp::float32>(var_123, var_124, var_125);
        // virtual_inertia = wp.vec3(geom_fluid[geomid, 9], geom_fluid[geomid, 10], geom_fluid[geomid, 11])       <L 663>
        var_127 = wp::address(var_geom_fluid, var_48, var_126);
        var_129 = wp::address(var_geom_fluid, var_48, var_128);
        var_131 = wp::address(var_geom_fluid, var_48, var_130);
        var_133 = wp::load(var_127);
        var_134 = wp::load(var_129);
        var_135 = wp::load(var_131);
        var_132 = wp::vec_t<3, wp::float32>(var_133, var_134, var_135);
        // B00 = wp.mat33(0.0)  # torque wrt ang_vel                                          <L 667>
        var_137 = wp::mat_t<3, 3, wp::float32>(var_136);
        // B01 = wp.mat33(0.0)  # torque wrt lin_vel                                          <L 668>
        var_139 = wp::mat_t<3, 3, wp::float32>(var_138);
        // B10 = wp.mat33(0.0)  # force wrt ang_vel                                           <L 669>
        var_141 = wp::mat_t<3, 3, wp::float32>(var_140);
        // B11 = wp.mat33(0.0)  # force wrt lin_vel                                           <L 670>
        var_143 = wp::mat_t<3, 3, wp::float32>(var_142);
        // if density > 0.0:                                                                  <L 672>
        var_145 = (var_density > var_144);
        if (var_145) {
            // density_vm = density * virtual_mass                                            <L 674>
            var_146 = wp::mul(var_density, var_122);
            // density_vi = density * virtual_inertia                                         <L 675>
            var_147 = wp::mul(var_density, var_132);
            // virtual_lin_mom = wp.cw_mul(density_vm, lin_vel)                               <L 676>
            var_148 = wp::cw_mul(var_146, var_95);
            // virtual_ang_mom = wp.cw_mul(density_vi, ang_vel)                               <L 677>
            var_149 = wp::cw_mul(var_147, var_94);
            // B00 += wp.skew(virtual_ang_mom) - wp.skew(ang_vel) @ wp.diag(density_vi)       <L 680>
            var_150 = wp::skew(var_149);
            var_151 = wp::skew(var_94);
            var_152 = wp::diag(var_147);
            var_153 = wp::mul(var_151, var_152);
            var_154 = wp::sub(var_150, var_153);
            var_155 = wp::add(var_137, var_154);
            // B01 += wp.skew(virtual_lin_mom) - wp.skew(lin_vel) @ wp.diag(density_vm)       <L 683>
            var_156 = wp::skew(var_148);
            var_157 = wp::skew(var_95);
            var_158 = wp::diag(var_146);
            var_159 = wp::mul(var_157, var_158);
            var_160 = wp::sub(var_156, var_159);
            var_161 = wp::add(var_139, var_160);
            // B10 += wp.skew(virtual_lin_mom)                                                <L 686>
            var_162 = wp::skew(var_148);
            var_163 = wp::add(var_141, var_162);
            // B11 += -wp.skew(ang_vel) @ wp.diag(density_vm)                                 <L 688>
            var_164 = wp::skew(var_94);
            var_165 = wp::neg(var_164);
            var_166 = wp::diag(var_146);
            var_167 = wp::mul(var_165, var_166);
            var_168 = wp::add(var_143, var_167);
        }
        var_169 = wp::where(var_145, var_155, var_137);
        var_170 = wp::where(var_145, var_161, var_139);
        var_171 = wp::where(var_145, var_163, var_141);
        var_172 = wp::where(var_145, var_168, var_143);
        // volume = wp.static(4.0 / 3.0 * wp.pi) * semiaxes[0] * semiaxes[1] * semiaxes[2]       <L 691>
        var_175 = wp::extract(var_64, var_174);
        var_176 = wp::mul(var_173, var_175);
        var_178 = wp::extract(var_64, var_177);
        var_179 = wp::mul(var_176, var_178);
        var_181 = wp::extract(var_64, var_180);
        var_182 = wp::mul(var_179, var_181);
        // magnus_coef = magnus_lift_coef * density * volume                                  <L 692>
        var_183 = wp::mul(var_114, var_density);
        var_184 = wp::mul(var_183, var_182);
        // B10 -= wp.skew(lin_vel) * magnus_coef                                              <L 693>
        var_185 = wp::skew(var_95);
        var_186 = wp::mul(var_185, var_184);
        var_187 = wp::sub(var_171, var_186);
        // B11 += wp.skew(ang_vel) * magnus_coef                                              <L 694>
        var_188 = wp::skew(var_94);
        var_189 = wp::mul(var_188, var_184);
        var_190 = wp::add(var_172, var_189);
        // a = (semiaxes[1] * semiaxes[2]) * (semiaxes[1] * semiaxes[2])                      <L 697>
        var_192 = wp::extract(var_64, var_191);
        var_194 = wp::extract(var_64, var_193);
        var_195 = wp::mul(var_192, var_194);
        var_197 = wp::extract(var_64, var_196);
        var_199 = wp::extract(var_64, var_198);
        var_200 = wp::mul(var_197, var_199);
        var_201 = wp::mul(var_195, var_200);
        // b = (semiaxes[2] * semiaxes[0]) * (semiaxes[2] * semiaxes[0])                      <L 698>
        var_203 = wp::extract(var_64, var_202);
        var_205 = wp::extract(var_64, var_204);
        var_206 = wp::mul(var_203, var_205);
        var_208 = wp::extract(var_64, var_207);
        var_210 = wp::extract(var_64, var_209);
        var_211 = wp::mul(var_208, var_210);
        var_212 = wp::mul(var_206, var_211);
        // c = (semiaxes[0] * semiaxes[1]) * (semiaxes[0] * semiaxes[1])                      <L 699>
        var_214 = wp::extract(var_64, var_213);
        var_216 = wp::extract(var_64, var_215);
        var_217 = wp::mul(var_214, var_216);
        var_219 = wp::extract(var_64, var_218);
        var_221 = wp::extract(var_64, var_220);
        var_222 = wp::mul(var_219, var_221);
        var_223 = wp::mul(var_217, var_222);
        // aa = a * a                                                                         <L 700>
        var_224 = wp::mul(var_201, var_201);
        // bb = b * b                                                                         <L 701>
        var_225 = wp::mul(var_212, var_212);
        // cc = c * c                                                                         <L 702>
        var_226 = wp::mul(var_223, var_223);
        // x = lin_vel[0]                                                                     <L 704>
        var_228 = wp::extract(var_95, var_227);
        // y = lin_vel[1]                                                                     <L 705>
        var_230 = wp::extract(var_95, var_229);
        // z = lin_vel[2]                                                                     <L 706>
        var_232 = wp::extract(var_95, var_231);
        // xx = x * x                                                                         <L 707>
        var_233 = wp::mul(var_228, var_228);
        // yy = y * y                                                                         <L 708>
        var_234 = wp::mul(var_230, var_230);
        // zz = z * z                                                                         <L 709>
        var_235 = wp::mul(var_232, var_232);
        // xy = x * y                                                                         <L 710>
        var_236 = wp::mul(var_228, var_230);
        // yz = y * z                                                                         <L 711>
        var_237 = wp::mul(var_230, var_232);
        // xz = x * z                                                                         <L 712>
        var_238 = wp::mul(var_228, var_232);
        // proj_denom = aa * xx + bb * yy + cc * zz                                           <L 714>
        var_239 = wp::mul(var_224, var_233);
        var_240 = wp::mul(var_225, var_234);
        var_241 = wp::add(var_239, var_240);
        var_242 = wp::mul(var_226, var_235);
        var_243 = wp::add(var_241, var_242);
        // proj_num = a * xx + b * yy + c * zz                                                <L 715>
        var_244 = wp::mul(var_201, var_233);
        var_245 = wp::mul(var_212, var_234);
        var_246 = wp::add(var_244, var_245);
        var_247 = wp::mul(var_223, var_235);
        var_248 = wp::add(var_246, var_247);
        // norm2 = xx + yy + zz                                                               <L 716>
        var_249 = wp::add(var_233, var_234);
        var_250 = wp::add(var_249, var_235);
        // df_denom = wp.pi * kutta_lift_coef * density / wp.max(MJ_MINVAL, wp.sqrt(proj_denom * proj_num * norm2))       <L 717>
        var_252 = wp::mul(var_251, var_110);
        var_253 = wp::mul(var_252, var_density);
        var_255 = wp::mul(var_243, var_248);
        var_256 = wp::mul(var_255, var_250);
        var_257 = wp::sqrt(var_256);
        var_258 = wp::max(var_254, var_257);
        var_259 = wp::div(var_253, var_258);
        // dfx_coef = yy * (a - b) + zz * (a - c)                                             <L 719>
        var_260 = wp::sub(var_201, var_212);
        var_261 = wp::mul(var_234, var_260);
        var_262 = wp::sub(var_201, var_223);
        var_263 = wp::mul(var_235, var_262);
        var_264 = wp::add(var_261, var_263);
        // dfy_coef = xx * (b - a) + zz * (b - c)                                             <L 720>
        var_265 = wp::sub(var_212, var_201);
        var_266 = wp::mul(var_233, var_265);
        var_267 = wp::sub(var_212, var_223);
        var_268 = wp::mul(var_235, var_267);
        var_269 = wp::add(var_266, var_268);
        // dfz_coef = xx * (c - a) + yy * (c - b)                                             <L 721>
        var_270 = wp::sub(var_223, var_201);
        var_271 = wp::mul(var_233, var_270);
        var_272 = wp::sub(var_223, var_212);
        var_273 = wp::mul(var_234, var_272);
        var_274 = wp::add(var_271, var_273);
        // proj_term = proj_num / wp.max(MJ_MINVAL, proj_denom)                               <L 722>
        var_275 = wp::max(var_254, var_243);
        var_276 = wp::div(var_248, var_275);
        // cos_term = proj_num / wp.max(MJ_MINVAL, norm2)                                     <L 723>
        var_277 = wp::max(var_254, var_250);
        var_278 = wp::div(var_248, var_277);
        // D = wp.skew(wp.vec3(b - c, c - a, a - b)) * (2.0 * proj_num)                       <L 725>
        var_279 = wp::sub(var_212, var_223);
        var_280 = wp::sub(var_223, var_201);
        var_281 = wp::sub(var_201, var_212);
        var_282 = wp::vec_t<3, wp::float32>(var_279, var_280, var_281);
        var_283 = wp::skew(var_282);
        var_285 = wp::mul(var_284, var_248);
        var_286 = wp::mul(var_283, var_285);
        // df_coef = wp.vec3(dfx_coef, dfy_coef, dfz_coef)                                    <L 727>
        var_287 = wp::vec_t<3, wp::float32>(var_264, var_269, var_274);
        // inner_term = wp.vec3(                                                              <L 728>
        // aa * proj_term - a + cos_term,                                                     <L 729>
        var_288 = wp::mul(var_224, var_276);
        var_289 = wp::sub(var_288, var_201);
        var_290 = wp::add(var_289, var_278);
        // bb * proj_term - b + cos_term,                                                     <L 730>
        var_291 = wp::mul(var_225, var_276);
        var_292 = wp::sub(var_291, var_212);
        var_293 = wp::add(var_292, var_278);
        // cc * proj_term - c + cos_term,                                                     <L 731>
        var_294 = wp::mul(var_226, var_276);
        var_295 = wp::sub(var_294, var_223);
        var_296 = wp::add(var_295, var_278);
        var_297 = wp::vec_t<3, wp::float32>(var_290, var_293, var_296);
        // D += wp.outer(df_coef, inner_term)                                                 <L 734>
        var_298 = wp::outer(var_287, var_297);
        var_299 = wp::add(var_286, var_298);
        // V = wp.diag(lin_vel)                                                               <L 736>
        var_300 = wp::diag(var_95);
        // D = V @ D @ V - wp.diag(df_coef * proj_num)                                        <L 737>
        var_301 = wp::mul(var_300, var_299);
        var_302 = wp::mul(var_301, var_300);
        var_303 = wp::mul(var_287, var_248);
        var_304 = wp::diag(var_303);
        var_305 = wp::sub(var_302, var_304);
        // D *= df_denom                                                                      <L 739>
        var_306 = wp::mul(var_305, var_259);
        // B11 += D                                                                           <L 740>
        var_307 = wp::add(var_190, var_306);
        // d_max = wp.max(wp.max(semiaxes[0], semiaxes[1]), semiaxes[2])                      <L 743>
        var_309 = wp::extract(var_64, var_308);
        var_311 = wp::extract(var_64, var_310);
        var_312 = wp::max(var_309, var_311);
        var_314 = wp::extract(var_64, var_313);
        var_315 = wp::max(var_312, var_314);
        // d_min = wp.min(wp.min(semiaxes[0], semiaxes[1]), semiaxes[2])                      <L 744>
        var_317 = wp::extract(var_64, var_316);
        var_319 = wp::extract(var_64, var_318);
        var_320 = wp::min(var_317, var_319);
        var_322 = wp::extract(var_64, var_321);
        var_323 = wp::min(var_320, var_322);
        // d_mid = semiaxes[0] + semiaxes[1] + semiaxes[2] - d_max - d_min                    <L 745>
        var_325 = wp::extract(var_64, var_324);
        var_327 = wp::extract(var_64, var_326);
        var_328 = wp::add(var_325, var_327);
        var_330 = wp::extract(var_64, var_329);
        var_331 = wp::add(var_328, var_330);
        var_332 = wp::sub(var_331, var_315);
        var_333 = wp::sub(var_332, var_323);
        // eq_sphere_D = wp.static(2.0 / 3.0) * (semiaxes[0] + semiaxes[1] + semiaxes[2])       <L 746>
        var_336 = wp::extract(var_64, var_335);
        var_338 = wp::extract(var_64, var_337);
        var_339 = wp::add(var_336, var_338);
        var_341 = wp::extract(var_64, var_340);
        var_342 = wp::add(var_339, var_341);
        var_343 = wp::mul(var_334, var_342);
        // A_max = wp.pi * d_max * d_mid                                                      <L 747>
        var_345 = wp::mul(var_344, var_315);
        var_346 = wp::mul(var_345, var_333);
        // A_proj = wp.pi * wp.sqrt(proj_denom / wp.max(MJ_MINVAL, proj_num))                 <L 749>
        var_348 = wp::max(var_254, var_248);
        var_349 = wp::div(var_243, var_348);
        var_350 = wp::sqrt(var_349);
        var_351 = wp::mul(var_347, var_350);
        // norm = wp.sqrt(xx + yy + zz)                                                       <L 751>
        var_352 = wp::add(var_233, var_234);
        var_353 = wp::add(var_352, var_235);
        var_354 = wp::sqrt(var_353);
        // inv_norm = 1.0 / wp.max(MJ_MINVAL, norm)                                           <L 752>
        var_356 = wp::max(var_254, var_354);
        var_357 = wp::div(var_355, var_356);
        // lin_coef = viscosity * wp.static(3.0 * wp.pi) * eq_sphere_D                        <L 754>
        var_359 = wp::mul(var_viscosity, var_358);
        var_360 = wp::mul(var_359, var_343);
        // quad_coef = density * (A_proj * blunt_drag_coef + slender_drag_coef * (A_max - A_proj))       <L 755>
        var_361 = wp::mul(var_351, var_98);
        var_362 = wp::sub(var_346, var_351);
        var_363 = wp::mul(var_102, var_362);
        var_364 = wp::add(var_361, var_363);
        var_365 = wp::mul(var_density, var_364);
        // Aproj_coef = density * norm * (blunt_drag_coef - slender_drag_coef)                <L 756>
        var_366 = wp::mul(var_density, var_354);
        var_367 = wp::sub(var_98, var_102);
        var_368 = wp::mul(var_366, var_367);
        // dA_coef = wp.pi / wp.max(MJ_MINVAL, wp.sqrt(proj_num * proj_num * proj_num * proj_denom))       <L 757>
        var_370 = wp::mul(var_248, var_248);
        var_371 = wp::mul(var_370, var_248);
        var_372 = wp::mul(var_371, var_243);
        var_373 = wp::sqrt(var_372);
        var_374 = wp::max(var_254, var_373);
        var_375 = wp::div(var_369, var_374);
        // dAproj_dv = wp.vec3(                                                               <L 759>
        // Aproj_coef * dA_coef * a * x * (b * yy * (a - b) + c * zz * (a - c)),              <L 760>
        var_376 = wp::mul(var_368, var_375);
        var_377 = wp::mul(var_376, var_201);
        var_378 = wp::mul(var_377, var_228);
        var_379 = wp::mul(var_212, var_234);
        var_380 = wp::sub(var_201, var_212);
        var_381 = wp::mul(var_379, var_380);
        var_382 = wp::mul(var_223, var_235);
        var_383 = wp::sub(var_201, var_223);
        var_384 = wp::mul(var_382, var_383);
        var_385 = wp::add(var_381, var_384);
        var_386 = wp::mul(var_378, var_385);
        // Aproj_coef * dA_coef * b * y * (a * xx * (b - a) + c * zz * (b - c)),              <L 761>
        var_387 = wp::mul(var_368, var_375);
        var_388 = wp::mul(var_387, var_212);
        var_389 = wp::mul(var_388, var_230);
        var_390 = wp::mul(var_201, var_233);
        var_391 = wp::sub(var_212, var_201);
        var_392 = wp::mul(var_390, var_391);
        var_393 = wp::mul(var_223, var_235);
        var_394 = wp::sub(var_212, var_223);
        var_395 = wp::mul(var_393, var_394);
        var_396 = wp::add(var_392, var_395);
        var_397 = wp::mul(var_389, var_396);
        // Aproj_coef * dA_coef * c * z * (a * xx * (c - a) + b * yy * (c - b)),              <L 762>
        var_398 = wp::mul(var_368, var_375);
        var_399 = wp::mul(var_398, var_223);
        var_400 = wp::mul(var_399, var_232);
        var_401 = wp::mul(var_201, var_233);
        var_402 = wp::sub(var_223, var_201);
        var_403 = wp::mul(var_401, var_402);
        var_404 = wp::mul(var_212, var_234);
        var_405 = wp::sub(var_223, var_212);
        var_406 = wp::mul(var_404, var_405);
        var_407 = wp::add(var_403, var_406);
        var_408 = wp::mul(var_400, var_407);
        var_409 = wp::vec_t<3, wp::float32>(var_386, var_397, var_408);
        // inner = wp.length_sq(lin_vel)                                                      <L 765>
        var_410 = wp::length_sq(var_95);
        // D = (wp.outer(lin_vel, lin_vel) + wp.diag(wp.vec3(inner))) * (-quad_coef * inv_norm)       <L 766>
        var_411 = wp::outer(var_95, var_95);
        var_412 = wp::vec_t<3, wp::float32>(var_410);
        var_413 = wp::diag(var_412);
        var_414 = wp::add(var_411, var_413);
        var_415 = wp::neg(var_365);
        var_416 = wp::mul(var_415, var_357);
        var_417 = wp::mul(var_414, var_416);
        // D -= wp.outer(lin_vel, dAproj_dv)                                                  <L 767>
        var_418 = wp::outer(var_95, var_409);
        var_419 = wp::sub(var_417, var_418);
        // D -= wp.diag(wp.vec3(lin_coef))                                                    <L 768>
        var_420 = wp::vec_t<3, wp::float32>(var_360);
        var_421 = wp::diag(var_420);
        var_422 = wp::sub(var_419, var_421);
        // B11 += D                                                                           <L 770>
        var_423 = wp::add(var_307, var_422);
        // lin_visc_torq_coef = wp.pi * eq_sphere_D * eq_sphere_D * eq_sphere_D               <L 773>
        var_425 = wp::mul(var_424, var_343);
        var_426 = wp::mul(var_425, var_343);
        var_427 = wp::mul(var_426, var_343);
        // I_max = wp.static(8.0 / 15.0 * wp.pi) * d_mid * d_max * d_max * d_max * d_max       <L 774>
        var_429 = wp::mul(var_428, var_333);
        var_430 = wp::mul(var_429, var_315);
        var_431 = wp::mul(var_430, var_315);
        var_432 = wp::mul(var_431, var_315);
        var_433 = wp::mul(var_432, var_315);
        // II = wp.vec3(                                                                      <L 775>
        // ellipsoid_max_moment(semiaxes, 0),                                                 <L 776>
        var_435 = ellipsoid_max_moment_0(var_64, var_434);
        // ellipsoid_max_moment(semiaxes, 1),                                                 <L 777>
        var_437 = ellipsoid_max_moment_0(var_64, var_436);
        // ellipsoid_max_moment(semiaxes, 2),                                                 <L 778>
        var_439 = ellipsoid_max_moment_0(var_64, var_438);
        var_440 = wp::vec_t<3, wp::float32>(var_435, var_437, var_439);
        // mom_coef = wp.vec3(                                                                <L 781>
        // ang_drag_coef * II[0] + slender_drag_coef * (I_max - II[0]),                       <L 782>
        var_442 = wp::extract(var_440, var_441);
        var_443 = wp::mul(var_106, var_442);
        var_445 = wp::extract(var_440, var_444);
        var_446 = wp::sub(var_433, var_445);
        var_447 = wp::mul(var_102, var_446);
        var_448 = wp::add(var_443, var_447);
        // ang_drag_coef * II[1] + slender_drag_coef * (I_max - II[1]),                       <L 783>
        var_450 = wp::extract(var_440, var_449);
        var_451 = wp::mul(var_106, var_450);
        var_453 = wp::extract(var_440, var_452);
        var_454 = wp::sub(var_433, var_453);
        var_455 = wp::mul(var_102, var_454);
        var_456 = wp::add(var_451, var_455);
        // ang_drag_coef * II[2] + slender_drag_coef * (I_max - II[2]),                       <L 784>
        var_458 = wp::extract(var_440, var_457);
        var_459 = wp::mul(var_106, var_458);
        var_461 = wp::extract(var_440, var_460);
        var_462 = wp::sub(var_433, var_461);
        var_463 = wp::mul(var_102, var_462);
        var_464 = wp::add(var_459, var_463);
        var_465 = wp::vec_t<3, wp::float32>(var_448, var_456, var_464);
        // mom_visc = wp.cw_mul(ang_vel, mom_coef)                                            <L 787>
        var_466 = wp::cw_mul(var_94, var_465);
        // norm_mom = wp.length(mom_visc)                                                     <L 788>
        var_467 = wp::length(var_466);
        // density_scaled = density / wp.max(MJ_MINVAL, norm_mom)                             <L 789>
        var_468 = wp::max(var_254, var_467);
        var_469 = wp::div(var_density, var_468);
        // mom_sq = -density_scaled * wp.cw_mul(wp.cw_mul(ang_vel, mom_coef), mom_coef)       <L 791>
        var_470 = wp::neg(var_469);
        var_471 = wp::cw_mul(var_94, var_465);
        var_472 = wp::cw_mul(var_471, var_465);
        var_473 = wp::mul(var_470, var_472);
        // torq_lin_coef = viscosity * lin_visc_torq_coef                                     <L 793>
        var_474 = wp::mul(var_viscosity, var_427);
        // diag_val = wp.dot(ang_vel, mom_sq) - torq_lin_coef                                 <L 794>
        var_475 = wp::dot(var_94, var_473);
        var_476 = wp::sub(var_475, var_474);
        // D = wp.outer(ang_vel, mom_sq) + wp.diag(wp.vec3(diag_val))                         <L 796>
        var_477 = wp::outer(var_94, var_473);
        var_478 = wp::vec_t<3, wp::float32>(var_476);
        var_479 = wp::diag(var_478);
        var_480 = wp::add(var_477, var_479);
        // B00 += D                                                                           <L 797>
        var_481 = wp::add(var_169, var_480);
        // if is_implicitfast:                                                                <L 800>
        if (var_1) {
            // B00 = 0.5 * (B00 + wp.transpose(B00))                                          <L 801>
            var_483 = wp::transpose(var_481);
            var_484 = wp::add(var_481, var_483);
            var_485 = wp::mul(var_482, var_484);
            // B11 = 0.5 * (B11 + wp.transpose(B11))                                          <L 802>
            var_487 = wp::transpose(var_423);
            var_488 = wp::add(var_423, var_487);
            var_489 = wp::mul(var_486, var_488);
            // B01_sym = 0.5 * (B01 + wp.transpose(B10))                                      <L 803>
            var_491 = wp::transpose(var_187);
            var_492 = wp::add(var_170, var_491);
            var_493 = wp::mul(var_490, var_492);
            // B01 = B01_sym                                                                  <L 804>
            var_494 = wp::copy(var_493);
            // B10 = wp.transpose(B01_sym)                                                    <L 805>
            var_495 = wp::transpose(var_493);
        }
        var_496 = wp::where(var_1, var_485, var_481);
        var_497 = wp::where(var_1, var_494, var_170);
        var_498 = wp::where(var_1, var_495, var_187);
        var_499 = wp::where(var_1, var_489, var_423);
        // offset = geom_pos - subtree_root                                                   <L 808>
        var_500 = wp::sub(var_71, var_11);
        // jac_p_i = cdof_lin_i + wp.cross(cdof_ang_i, offset)                                <L 810>
        var_501 = wp::cross(var_24, var_500);
        var_502 = wp::add(var_31, var_501);
        // la_i = geom_rotT @ cdof_ang_i                                                      <L 811>
        var_503 = wp::mul(var_69, var_24);
        // ll_i = geom_rotT @ jac_p_i                                                         <L 812>
        var_504 = wp::mul(var_69, var_502);
        // jac_p_j = cdof_lin_j + wp.cross(cdof_ang_j, offset)                                <L 814>
        var_505 = wp::cross(var_38, var_500);
        var_506 = wp::add(var_45, var_505);
        // la_j = geom_rotT @ cdof_ang_j                                                      <L 815>
        var_507 = wp::mul(var_69, var_38);
        // ll_j = geom_rotT @ jac_p_j                                                         <L 816>
        var_508 = wp::mul(var_69, var_506);
        // Bj_ang = B00 @ la_j + B01 @ ll_j                                                   <L 819>
        var_509 = wp::mul(var_496, var_507);
        var_510 = wp::mul(var_497, var_508);
        var_511 = wp::add(var_509, var_510);
        // Bj_lin = B10 @ la_j + B11 @ ll_j                                                   <L 820>
        var_512 = wp::mul(var_498, var_507);
        var_513 = wp::mul(var_499, var_508);
        var_514 = wp::add(var_512, var_513);
        // qderiv_contrib += wp.dot(la_i, Bj_ang) + wp.dot(ll_i, Bj_lin)                      <L 823>
        var_515 = wp::dot(var_503, var_511);
        var_516 = wp::dot(var_504, var_514);
        var_517 = wp::add(var_515, var_516);
        var_518 = wp::add(var_17, var_517);
        wp::assign(var_17, var_518);
        goto start_for_0;
    end_for_0:;
    // return qderiv_contrib                                                                  <L 825>
    return var_17;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/derivative.py:926
static CUDA_CALLABLE wp::mat_t<6, 6, wp::float32> _deriv_box_fluid_0(
    wp::int32 var_opt_integrator,
    wp::array_t<wp::float32> var_body_mass,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_inertia,
    wp::int32 var_worldid,
    wp::int32 var_bodyid,
    wp::vec_t<6, wp::float32> var_lvel,
    wp::float32 var_density,
    wp::float32 var_viscosity)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    wp::mat_t<6, 6, wp::float32> var_1;
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
    wp::vec_t<3, wp::float32>* var_15;
    wp::vec_t<3, wp::float32> var_16;
    wp::vec_t<3, wp::float32> var_17;
    const wp::float32 var_18 = 6.0;
    wp::float32 var_19;
    const wp::float32 var_20 = 1e-15;
    const wp::int32 var_21 = 1;
    wp::float32 var_22;
    const wp::int32 var_23 = 2;
    wp::float32 var_24;
    wp::float32 var_25;
    const wp::int32 var_26 = 0;
    wp::float32 var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::float32 var_30;
    wp::float32 var_31;
    const wp::int32 var_32 = 0;
    wp::float32 var_33;
    const wp::int32 var_34 = 2;
    wp::float32 var_35;
    wp::float32 var_36;
    const wp::int32 var_37 = 1;
    wp::float32 var_38;
    wp::float32 var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::float32 var_42;
    const wp::int32 var_43 = 0;
    wp::float32 var_44;
    const wp::int32 var_45 = 1;
    wp::float32 var_46;
    wp::float32 var_47;
    const wp::int32 var_48 = 2;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::float32 var_51;
    wp::float32 var_52;
    wp::float32 var_53;
    wp::vec_t<3, wp::float32> var_54;
    const wp::float32 var_55 = 0.0;
    bool var_56;
    const wp::int32 var_57 = 0;
    wp::float32 var_58;
    const wp::int32 var_59 = 1;
    wp::float32 var_60;
    wp::float32 var_61;
    const wp::int32 var_62 = 2;
    wp::float32 var_63;
    wp::float32 var_64;
    const wp::float32 var_65 = 0.3333333333333333;
    wp::float32 var_66;
    const wp::float32 var_67 = 3.141592653589793;
    const wp::float32 var_68 = -3.141592653589793;
    wp::float32 var_69;
    wp::float32 var_70;
    wp::float32 var_71;
    wp::float32 var_72;
    const wp::int32 var_73 = 0;
    const wp::int32 var_74 = 0;
    const wp::int32 var_75 = 1;
    const wp::int32 var_76 = 1;
    const wp::int32 var_77 = 2;
    const wp::int32 var_78 = 2;
    const wp::float32 var_79 = -9.42477796076938;
    wp::float32 var_80;
    wp::float32 var_81;
    const wp::int32 var_82 = 3;
    const wp::int32 var_83 = 3;
    const wp::int32 var_84 = 4;
    const wp::int32 var_85 = 4;
    const wp::int32 var_86 = 5;
    const wp::int32 var_87 = 5;
    const wp::float32 var_88 = 0.0;
    bool var_89;
    const wp::int32 var_90 = 1;
    wp::float32 var_91;
    const wp::int32 var_92 = 1;
    wp::float32 var_93;
    wp::float32 var_94;
    const wp::int32 var_95 = 1;
    wp::float32 var_96;
    wp::float32 var_97;
    const wp::int32 var_98 = 1;
    wp::float32 var_99;
    wp::float32 var_100;
    const wp::int32 var_101 = 2;
    wp::float32 var_102;
    const wp::int32 var_103 = 2;
    wp::float32 var_104;
    wp::float32 var_105;
    const wp::int32 var_106 = 2;
    wp::float32 var_107;
    wp::float32 var_108;
    const wp::int32 var_109 = 2;
    wp::float32 var_110;
    wp::float32 var_111;
    wp::float32 var_112;
    const wp::int32 var_113 = 0;
    wp::float32 var_114;
    const wp::int32 var_115 = 0;
    wp::float32 var_116;
    wp::float32 var_117;
    const wp::int32 var_118 = 0;
    wp::float32 var_119;
    wp::float32 var_120;
    const wp::int32 var_121 = 0;
    wp::float32 var_122;
    wp::float32 var_123;
    const wp::int32 var_124 = 2;
    wp::float32 var_125;
    const wp::int32 var_126 = 2;
    wp::float32 var_127;
    wp::float32 var_128;
    const wp::int32 var_129 = 2;
    wp::float32 var_130;
    wp::float32 var_131;
    const wp::int32 var_132 = 2;
    wp::float32 var_133;
    wp::float32 var_134;
    wp::float32 var_135;
    const wp::int32 var_136 = 0;
    wp::float32 var_137;
    const wp::int32 var_138 = 0;
    wp::float32 var_139;
    wp::float32 var_140;
    const wp::int32 var_141 = 0;
    wp::float32 var_142;
    wp::float32 var_143;
    const wp::int32 var_144 = 0;
    wp::float32 var_145;
    wp::float32 var_146;
    const wp::int32 var_147 = 1;
    wp::float32 var_148;
    const wp::int32 var_149 = 1;
    wp::float32 var_150;
    wp::float32 var_151;
    const wp::int32 var_152 = 1;
    wp::float32 var_153;
    wp::float32 var_154;
    const wp::int32 var_155 = 1;
    wp::float32 var_156;
    wp::float32 var_157;
    wp::float32 var_158;
    const wp::float32 var_159 = 0.03125;
    const wp::int32 var_160 = 0;
    wp::float32 var_161;
    wp::float32 var_162;
    wp::float32 var_163;
    const wp::int32 var_164 = 0;
    wp::float32 var_165;
    wp::float32 var_166;
    wp::float32 var_167;
    wp::float32 var_168;
    const wp::int32 var_169 = 0;
    const wp::int32 var_170 = 0;
    const wp::int32 var_171 = 1;
    wp::float32 var_172;
    wp::float32 var_173;
    wp::float32 var_174;
    const wp::int32 var_175 = 1;
    wp::float32 var_176;
    wp::float32 var_177;
    wp::float32 var_178;
    wp::float32 var_179;
    const wp::int32 var_180 = 1;
    const wp::int32 var_181 = 1;
    const wp::int32 var_182 = 2;
    wp::float32 var_183;
    wp::float32 var_184;
    wp::float32 var_185;
    const wp::int32 var_186 = 2;
    wp::float32 var_187;
    wp::float32 var_188;
    wp::float32 var_189;
    wp::float32 var_190;
    const wp::int32 var_191 = 2;
    const wp::int32 var_192 = 2;
    const wp::int32 var_193 = 1;
    wp::float32 var_194;
    wp::float32 var_195;
    const wp::int32 var_196 = 2;
    wp::float32 var_197;
    wp::float32 var_198;
    const wp::int32 var_199 = 3;
    wp::float32 var_200;
    wp::float32 var_201;
    wp::float32 var_202;
    const wp::int32 var_203 = 3;
    const wp::int32 var_204 = 3;
    const wp::int32 var_205 = 0;
    wp::float32 var_206;
    wp::float32 var_207;
    const wp::int32 var_208 = 2;
    wp::float32 var_209;
    wp::float32 var_210;
    const wp::int32 var_211 = 4;
    wp::float32 var_212;
    wp::float32 var_213;
    wp::float32 var_214;
    const wp::int32 var_215 = 4;
    const wp::int32 var_216 = 4;
    const wp::int32 var_217 = 0;
    wp::float32 var_218;
    wp::float32 var_219;
    const wp::int32 var_220 = 1;
    wp::float32 var_221;
    wp::float32 var_222;
    const wp::int32 var_223 = 5;
    wp::float32 var_224;
    wp::float32 var_225;
    wp::float32 var_226;
    const wp::int32 var_227 = 5;
    const wp::int32 var_228 = 5;
    const wp::int32 var_229 = 3;
    bool var_230;
    const wp::float32 var_231 = 0.5;
    wp::mat_t<6, 6, wp::float32> var_232;
    wp::mat_t<6, 6, wp::float32> var_233;
    wp::mat_t<6, 6, wp::float32> var_234;
    wp::mat_t<6, 6, wp::float32> var_235;
    //---------
    // forward
    // def _deriv_box_fluid(                                                                  <L 927>
    // B = wp.spatial_matrix(0.0)                                                             <L 939>
    var_1 = wp::mat_t<6, 6, wp::float32>(var_0);
    // mass = body_mass[worldid % body_mass.shape[0], bodyid]                                 <L 941>
    var_2 = &(var_body_mass.shape);
    var_5 = wp::load(var_2);
    var_4 = wp::extract(var_5, var_3);
    var_6 = wp::mod(var_worldid, var_4);
    var_7 = wp::address(var_body_mass, var_6, var_bodyid);
    var_9 = wp::load(var_7);
    var_8 = wp::copy(var_9);
    // inertia = body_inertia[worldid % body_inertia.shape[0], bodyid]                        <L 942>
    var_10 = &(var_body_inertia.shape);
    var_13 = wp::load(var_10);
    var_12 = wp::extract(var_13, var_11);
    var_14 = wp::mod(var_worldid, var_12);
    var_15 = wp::address(var_body_inertia, var_14, var_bodyid);
    var_17 = wp::load(var_15);
    var_16 = wp::copy(var_17);
    // scl = 6.0 / mass                                                                       <L 943>
    var_19 = wp::div(var_18, var_8);
    // box = wp.vec3(                                                                         <L 946>
    // wp.sqrt(wp.max(MJ_MINVAL, inertia[1] + inertia[2] - inertia[0]) * scl),                <L 947>
    var_22 = wp::extract(var_16, var_21);
    var_24 = wp::extract(var_16, var_23);
    var_25 = wp::add(var_22, var_24);
    var_27 = wp::extract(var_16, var_26);
    var_28 = wp::sub(var_25, var_27);
    var_29 = wp::max(var_20, var_28);
    var_30 = wp::mul(var_29, var_19);
    var_31 = wp::sqrt(var_30);
    // wp.sqrt(wp.max(MJ_MINVAL, inertia[0] + inertia[2] - inertia[1]) * scl),                <L 948>
    var_33 = wp::extract(var_16, var_32);
    var_35 = wp::extract(var_16, var_34);
    var_36 = wp::add(var_33, var_35);
    var_38 = wp::extract(var_16, var_37);
    var_39 = wp::sub(var_36, var_38);
    var_40 = wp::max(var_20, var_39);
    var_41 = wp::mul(var_40, var_19);
    var_42 = wp::sqrt(var_41);
    // wp.sqrt(wp.max(MJ_MINVAL, inertia[0] + inertia[1] - inertia[2]) * scl),                <L 949>
    var_44 = wp::extract(var_16, var_43);
    var_46 = wp::extract(var_16, var_45);
    var_47 = wp::add(var_44, var_46);
    var_49 = wp::extract(var_16, var_48);
    var_50 = wp::sub(var_47, var_49);
    var_51 = wp::max(var_20, var_50);
    var_52 = wp::mul(var_51, var_19);
    var_53 = wp::sqrt(var_52);
    var_54 = wp::vec_t<3, wp::float32>(var_31, var_42, var_53);
    // if viscosity > 0.0:                                                                    <L 953>
    var_56 = (var_viscosity > var_55);
    if (var_56) {
        // diam = (box[0] + box[1] + box[2]) * wp.static(1.0 / 3.0)                           <L 954>
        var_58 = wp::extract(var_54, var_57);
        var_60 = wp::extract(var_54, var_59);
        var_61 = wp::add(var_58, var_60);
        var_63 = wp::extract(var_54, var_62);
        var_64 = wp::add(var_61, var_63);
        var_66 = wp::mul(var_64, var_65);
        // visc_rot = -wp.pi * diam * diam * diam * viscosity                                 <L 957>
        var_69 = wp::mul(var_68, var_66);
        var_70 = wp::mul(var_69, var_66);
        var_71 = wp::mul(var_70, var_66);
        var_72 = wp::mul(var_71, var_viscosity);
        // B[0, 0] += visc_rot                                                                <L 958>
        wp::add_inplace(var_1, var_73, var_74, var_72);
        // B[1, 1] += visc_rot                                                                <L 959>
        wp::add_inplace(var_1, var_75, var_76, var_72);
        // B[2, 2] += visc_rot                                                                <L 960>
        wp::add_inplace(var_1, var_77, var_78, var_72);
        // visc_lin = wp.static(-3.0 * wp.pi) * diam * viscosity                              <L 963>
        var_80 = wp::mul(var_79, var_66);
        var_81 = wp::mul(var_80, var_viscosity);
        // B[3, 3] += visc_lin                                                                <L 964>
        wp::add_inplace(var_1, var_82, var_83, var_81);
        // B[4, 4] += visc_lin                                                                <L 965>
        wp::add_inplace(var_1, var_84, var_85, var_81);
        // B[5, 5] += visc_lin                                                                <L 966>
        wp::add_inplace(var_1, var_86, var_87, var_81);
    }
    // if density > 0.0:                                                                      <L 969>
    var_89 = (var_density > var_88);
    if (var_89) {
        // term0 = box[1] * box[1] * box[1] * box[1] + box[2] * box[2] * box[2] * box[2]       <L 970>
        var_91 = wp::extract(var_54, var_90);
        var_93 = wp::extract(var_54, var_92);
        var_94 = wp::mul(var_91, var_93);
        var_96 = wp::extract(var_54, var_95);
        var_97 = wp::mul(var_94, var_96);
        var_99 = wp::extract(var_54, var_98);
        var_100 = wp::mul(var_97, var_99);
        var_102 = wp::extract(var_54, var_101);
        var_104 = wp::extract(var_54, var_103);
        var_105 = wp::mul(var_102, var_104);
        var_107 = wp::extract(var_54, var_106);
        var_108 = wp::mul(var_105, var_107);
        var_110 = wp::extract(var_54, var_109);
        var_111 = wp::mul(var_108, var_110);
        var_112 = wp::add(var_100, var_111);
        // term1 = box[0] * box[0] * box[0] * box[0] + box[2] * box[2] * box[2] * box[2]       <L 971>
        var_114 = wp::extract(var_54, var_113);
        var_116 = wp::extract(var_54, var_115);
        var_117 = wp::mul(var_114, var_116);
        var_119 = wp::extract(var_54, var_118);
        var_120 = wp::mul(var_117, var_119);
        var_122 = wp::extract(var_54, var_121);
        var_123 = wp::mul(var_120, var_122);
        var_125 = wp::extract(var_54, var_124);
        var_127 = wp::extract(var_54, var_126);
        var_128 = wp::mul(var_125, var_127);
        var_130 = wp::extract(var_54, var_129);
        var_131 = wp::mul(var_128, var_130);
        var_133 = wp::extract(var_54, var_132);
        var_134 = wp::mul(var_131, var_133);
        var_135 = wp::add(var_123, var_134);
        // term2 = box[0] * box[0] * box[0] * box[0] + box[1] * box[1] * box[1] * box[1]       <L 972>
        var_137 = wp::extract(var_54, var_136);
        var_139 = wp::extract(var_54, var_138);
        var_140 = wp::mul(var_137, var_139);
        var_142 = wp::extract(var_54, var_141);
        var_143 = wp::mul(var_140, var_142);
        var_145 = wp::extract(var_54, var_144);
        var_146 = wp::mul(var_143, var_145);
        var_148 = wp::extract(var_54, var_147);
        var_150 = wp::extract(var_54, var_149);
        var_151 = wp::mul(var_148, var_150);
        var_153 = wp::extract(var_54, var_152);
        var_154 = wp::mul(var_151, var_153);
        var_156 = wp::extract(var_54, var_155);
        var_157 = wp::mul(var_154, var_156);
        var_158 = wp::add(var_146, var_157);
        // inv_32 = wp.static(1.0 / 32.0)                                                     <L 974>
        // B[0, 0] -= density * box[0] * term0 * wp.abs(lvel[0]) * inv_32                     <L 975>
        var_161 = wp::extract(var_54, var_160);
        var_162 = wp::mul(var_density, var_161);
        var_163 = wp::mul(var_162, var_112);
        var_165 = wp::extract(var_lvel, var_164);
        var_166 = wp::abs(var_165);
        var_167 = wp::mul(var_163, var_166);
        var_168 = wp::mul(var_167, var_159);
        wp::sub_inplace(var_1, var_169, var_170, var_168);
        // B[1, 1] -= density * box[1] * term1 * wp.abs(lvel[1]) * inv_32                     <L 976>
        var_172 = wp::extract(var_54, var_171);
        var_173 = wp::mul(var_density, var_172);
        var_174 = wp::mul(var_173, var_135);
        var_176 = wp::extract(var_lvel, var_175);
        var_177 = wp::abs(var_176);
        var_178 = wp::mul(var_174, var_177);
        var_179 = wp::mul(var_178, var_159);
        wp::sub_inplace(var_1, var_180, var_181, var_179);
        // B[2, 2] -= density * box[2] * term2 * wp.abs(lvel[2]) * inv_32                     <L 977>
        var_183 = wp::extract(var_54, var_182);
        var_184 = wp::mul(var_density, var_183);
        var_185 = wp::mul(var_184, var_158);
        var_187 = wp::extract(var_lvel, var_186);
        var_188 = wp::abs(var_187);
        var_189 = wp::mul(var_185, var_188);
        var_190 = wp::mul(var_189, var_159);
        wp::sub_inplace(var_1, var_191, var_192, var_190);
        // B[3, 3] -= density * box[1] * box[2] * wp.abs(lvel[3])                             <L 979>
        var_194 = wp::extract(var_54, var_193);
        var_195 = wp::mul(var_density, var_194);
        var_197 = wp::extract(var_54, var_196);
        var_198 = wp::mul(var_195, var_197);
        var_200 = wp::extract(var_lvel, var_199);
        var_201 = wp::abs(var_200);
        var_202 = wp::mul(var_198, var_201);
        wp::sub_inplace(var_1, var_203, var_204, var_202);
        // B[4, 4] -= density * box[0] * box[2] * wp.abs(lvel[4])                             <L 980>
        var_206 = wp::extract(var_54, var_205);
        var_207 = wp::mul(var_density, var_206);
        var_209 = wp::extract(var_54, var_208);
        var_210 = wp::mul(var_207, var_209);
        var_212 = wp::extract(var_lvel, var_211);
        var_213 = wp::abs(var_212);
        var_214 = wp::mul(var_210, var_213);
        wp::sub_inplace(var_1, var_215, var_216, var_214);
        // B[5, 5] -= density * box[0] * box[1] * wp.abs(lvel[5])                             <L 981>
        var_218 = wp::extract(var_54, var_217);
        var_219 = wp::mul(var_density, var_218);
        var_221 = wp::extract(var_54, var_220);
        var_222 = wp::mul(var_219, var_221);
        var_224 = wp::extract(var_lvel, var_223);
        var_225 = wp::abs(var_224);
        var_226 = wp::mul(var_222, var_225);
        wp::sub_inplace(var_1, var_227, var_228, var_226);
    }
    // if opt_integrator == IntegratorType.IMPLICITFAST:                                      <L 983>
    var_230 = (var_opt_integrator == var_229);
    if (var_230) {
        // B = 0.5 * (B + wp.transpose(B))                                                    <L 984>
        var_232 = wp::transpose(var_1);
        var_233 = wp::add(var_1, var_232);
        var_234 = wp::mul(var_231, var_233);
    }
    var_235 = wp::where(var_230, var_234, var_1);
    // return B                                                                               <L 986>
    return var_235;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/derivative.py:989
static CUDA_CALLABLE wp::vec_t<6, wp::float32> _get_jac_column_local_0(
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::vec_t<3, wp::float32> var_point_global,
    wp::int32 var_bodyid,
    wp::int32 var_dofid,
    wp::int32 var_worldid,
    wp::mat_t<3, 3, wp::float32> var_b_imat)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::vec_t<3, wp::float32>* var_1;
    wp::int32 var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<6, wp::float32>* var_5;
    wp::vec_t<6, wp::float32> var_6;
    wp::vec_t<6, wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::mat_t<3, 3, wp::float32> var_13;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::vec_t<6, wp::float32> var_16;
    //---------
    // forward
    // def _get_jac_column_local(                                                             <L 990>
    // offset = point_global - subtree_com_in[worldid, body_rootid[bodyid]]                   <L 1005>
    var_0 = wp::address(var_body_rootid, var_bodyid);
    var_2 = wp::load(var_0);
    var_1 = wp::address(var_subtree_com_in, var_worldid, var_2);
    var_4 = wp::load(var_1);
    var_3 = wp::sub(var_point_global, var_4);
    // cdof_val = cdof_in[worldid, dofid]                                                     <L 1006>
    var_5 = wp::address(var_cdof_in, var_worldid, var_dofid);
    var_7 = wp::load(var_5);
    var_6 = wp::copy(var_7);
    // cdof_ang = wp.spatial_top(cdof_val)                                                    <L 1007>
    var_8 = wp::spatial_top(var_6);
    // cdof_lin = wp.spatial_bottom(cdof_val)                                                 <L 1008>
    var_9 = wp::spatial_bottom(var_6);
    // jacp = cdof_lin + wp.cross(cdof_ang, offset)                                           <L 1010>
    var_10 = wp::cross(var_8, var_3);
    var_11 = wp::add(var_9, var_10);
    // jacr = cdof_ang                                                                        <L 1011>
    var_12 = wp::copy(var_8);
    // b_imat_T = wp.transpose(b_imat)                                                        <L 1013>
    var_13 = wp::transpose(var_b_imat);
    // jacp_loc = b_imat_T @ jacp                                                             <L 1014>
    var_14 = wp::mul(var_13, var_11);
    // jacr_loc = b_imat_T @ jacr                                                             <L 1015>
    var_15 = wp::mul(var_13, var_12);
    // return wp.spatial_vector(jacr_loc, jacp_loc)                                           <L 1016>
    var_16 = wp::vec_t<6, wp::float32>(var_15, var_14);
    return var_16;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/math.py:120
static CUDA_CALLABLE void adj_inert_vec_0(
    wp::vec_t<10, wp::float32> var_i,
    wp::vec_t<6, wp::float32> var_v,
    wp::vec_t<10, wp::float32> & adj_i,
    wp::vec_t<6, wp::float32> & adj_v,
    wp::vec_t<6, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/math.py:147
static CUDA_CALLABLE void adj_motion_cross_force_0(
    wp::vec_t<6, wp::float32> var_v,
    wp::vec_t<6, wp::float32> var_f,
    wp::vec_t<6, wp::float32> & adj_v,
    wp::vec_t<6, wp::float32> & adj_f,
    wp::vec_t<6, wp::float32> & adj_ret)
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


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/math.py:133
static CUDA_CALLABLE void adj_motion_cross_0(
    wp::vec_t<6, wp::float32> var_u,
    wp::vec_t<6, wp::float32> var_v,
    wp::vec_t<6, wp::float32> & adj_u,
    wp::vec_t<6, wp::float32> & adj_v,
    wp::vec_t<6, wp::float32> & adj_ret)
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


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/derivative.py:587
static CUDA_CALLABLE void adj__deriv_ellipsoid_fluid_0(
    wp::int32 var_opt_integrator,
    wp::array_t<wp::int32> var_geom_type,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_size,
    wp::array_t<wp::float32> var_geom_fluid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::int32 var_worldid,
    wp::int32 var_bodyid,
    wp::int32 var_rootid,
    wp::int32 var_geomadr,
    wp::int32 var_geomnum,
    wp::vec_t<6, wp::float32> var_cdof_i,
    wp::vec_t<6, wp::float32> var_cdof_j,
    wp::vec_t<3, wp::float32> var_wind,
    wp::float32 var_density,
    wp::float32 var_viscosity,
    wp::int32 & adj_opt_integrator,
    wp::array_t<wp::int32> & adj_geom_type,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_geom_size,
    wp::array_t<wp::float32> & adj_geom_fluid,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cvel_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_bodyid,
    wp::int32 & adj_rootid,
    wp::int32 & adj_geomadr,
    wp::int32 & adj_geomnum,
    wp::vec_t<6, wp::float32> & adj_cdof_i,
    wp::vec_t<6, wp::float32> & adj_cdof_j,
    wp::vec_t<3, wp::float32> & adj_wind,
    wp::float32 & adj_density,
    wp::float32 & adj_viscosity,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/derivative.py:926
static CUDA_CALLABLE void adj__deriv_box_fluid_0(
    wp::int32 var_opt_integrator,
    wp::array_t<wp::float32> var_body_mass,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_inertia,
    wp::int32 var_worldid,
    wp::int32 var_bodyid,
    wp::vec_t<6, wp::float32> var_lvel,
    wp::float32 var_density,
    wp::float32 var_viscosity,
    wp::int32 & adj_opt_integrator,
    wp::array_t<wp::float32> & adj_body_mass,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_body_inertia,
    wp::int32 & adj_worldid,
    wp::int32 & adj_bodyid,
    wp::vec_t<6, wp::float32> & adj_lvel,
    wp::float32 & adj_density,
    wp::float32 & adj_viscosity,
    wp::mat_t<6, 6, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/derivative.py:989
static CUDA_CALLABLE void adj__get_jac_column_local_0(
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::vec_t<3, wp::float32> var_point_global,
    wp::int32 var_bodyid,
    wp::int32 var_dofid,
    wp::int32 var_worldid,
    wp::mat_t<3, 3, wp::float32> var_b_imat,
    wp::array_t<wp::int32> & adj_body_parentid,
    wp::array_t<wp::int32> & adj_body_rootid,
    wp::array_t<wp::int32> & adj_dof_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cdof_in,
    wp::vec_t<3, wp::float32> & adj_point_global,
    wp::int32 & adj_bodyid,
    wp::int32 & adj_dofid,
    wp::int32 & adj_worldid,
    wp::mat_t<3, 3, wp::float32> & adj_b_imat,
    wp::vec_t<6, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void deriv_rne_cacc_cfrcbody_forward_7f6cd6a1_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::vec_t<10, wp::float32>> var_cinert_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_dot_in,
    wp::array_t<wp::int32> var_body_tree_,
    wp::array_t<wp::vec_t<6, wp::float32>> var_Dcvel_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_Dcdof_dot_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_Dcacc_out,
    wp::array_t<wp::vec_t<6, wp::float32>> var_Dcfrcbody_out)
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
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::slice_t var_15;
        const wp::int32 var_16 = 0;
        wp::array_t<wp::float32> var_17;
        wp::vec_t<6, wp::float32>* var_18;
        wp::vec_t<6, wp::float32> var_19;
        wp::vec_t<6, wp::float32> var_20;
        wp::int32 var_21;
        wp::range_t var_22;
        wp::int32 var_23;
        bool var_24;
        wp::vec_t<6, wp::float32>* var_25;
        wp::vec_t<6, wp::float32> var_26;
        wp::vec_t<6, wp::float32> var_27;
        wp::vec_t<6, wp::float32> var_28;
        wp::vec_t<6, wp::float32>* var_29;
        wp::vec_t<6, wp::float32> var_30;
        wp::vec_t<6, wp::float32> var_31;
        wp::float32* var_32;
        wp::vec_t<6, wp::float32> var_33;
        wp::float32 var_34;
        wp::vec_t<6, wp::float32> var_35;
        wp::vec_t<10, wp::float32>* var_36;
        wp::vec_t<10, wp::float32> var_37;
        wp::vec_t<10, wp::float32> var_38;
        wp::vec_t<6, wp::float32>* var_39;
        wp::vec_t<6, wp::float32> var_40;
        wp::vec_t<6, wp::float32> var_41;
        wp::vec_t<6, wp::float32>* var_42;
        wp::vec_t<6, wp::float32> var_43;
        wp::vec_t<6, wp::float32> var_44;
        wp::vec_t<6, wp::float32> var_45;
        wp::vec_t<6, wp::float32> var_46;
        wp::vec_t<6, wp::float32> var_47;
        wp::vec_t<6, wp::float32> var_48;
        wp::vec_t<6, wp::float32> var_49;
        wp::vec_t<6, wp::float32> var_50;
        wp::vec_t<6, wp::float32> var_51;
        //---------
        // forward
        // def deriv_rne_cacc_cfrcbody_forward(                                                   <L 406>
        // worldid, nodeid, dofid = wp.tid()                                                      <L 425>
        builtin_tid3d(var_0, var_1, var_2);
        // bodyid = body_tree_[nodeid]                                                            <L 426>
        var_3 = wp::address(var_body_tree_, var_1);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // dofadr = body_dofadr[bodyid]                                                           <L 427>
        var_6 = wp::address(var_body_dofadr, var_4);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // dofnum = body_dofnum[bodyid]                                                           <L 428>
        var_9 = wp::address(var_body_dofnum, var_4);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // pid = body_parentid[bodyid]                                                            <L 429>
        var_12 = wp::address(var_body_parentid, var_4);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // qvel = qvel_in[worldid]                                                                <L 431>
        var_15 = wp::slice_t(var_0, var_0, var_16);
        var_17 = wp::view(var_qvel_in, var_15);
        // dcacc = Dcacc_out[worldid, pid, dofid]                                                 <L 433>
        var_18 = wp::address(var_Dcacc_out, var_0, var_13, var_2);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // for j in range(dofadr, dofadr + dofnum):                                               <L 435>
        var_21 = wp::add(var_7, var_10);
        var_22 = wp::range(var_7, var_21);
        start_for_0:;
            if (iter_cmp(var_22) == 0) goto end_for_0;
            var_23 = wp::iter_next(var_22);
            // if j == dofid:                                                                     <L 437>
            var_24 = (var_23 == var_2);
            if (var_24) {
                // dcacc += cdof_dot_in[worldid, j]                                               <L 438>
                var_25 = wp::address(var_cdof_dot_in, var_0, var_23);
                var_27 = wp::load(var_25);
                var_26 = wp::add(var_19, var_27);
            }
            var_28 = wp::where(var_24, var_26, var_19);
            // dcdofdot = Dcdof_dot_in[worldid, j, dofid]                                         <L 441>
            var_29 = wp::address(var_Dcdof_dot_in, var_0, var_23, var_2);
            var_31 = wp::load(var_29);
            var_30 = wp::copy(var_31);
            // dcacc += dcdofdot * qvel[j]                                                        <L 442>
            var_32 = wp::address(var_17, var_23);
            var_34 = wp::load(var_32);
            var_33 = wp::mul(var_30, var_34);
            var_35 = wp::add(var_28, var_33);
            wp::assign(var_19, var_35);
            goto start_for_0;
        end_for_0:;
        // Dcacc_out[worldid, bodyid, dofid] = dcacc                                              <L 444>
        wp::array_store(var_Dcacc_out, var_0, var_4, var_2, var_19);
        // cinert = cinert_in[worldid, bodyid]                                                    <L 447>
        var_36 = wp::address(var_cinert_in, var_0, var_4);
        var_38 = wp::load(var_36);
        var_37 = wp::copy(var_38);
        // cvel = cvel_in[worldid, bodyid]                                                        <L 448>
        var_39 = wp::address(var_cvel_in, var_0, var_4);
        var_41 = wp::load(var_39);
        var_40 = wp::copy(var_41);
        // dcvel = Dcvel_in[worldid, bodyid, dofid]                                               <L 449>
        var_42 = wp::address(var_Dcvel_in, var_0, var_4, var_2);
        var_44 = wp::load(var_42);
        var_43 = wp::copy(var_44);
        // term1 = math.inert_vec(cinert, dcacc)                                                  <L 452>
        var_45 = inert_vec_0(var_37, var_19);
        // cinert_cvel = math.inert_vec(cinert, cvel)                                             <L 455>
        var_46 = inert_vec_0(var_37, var_40);
        // cinert_dcvel = math.inert_vec(cinert, dcvel)                                           <L 456>
        var_47 = inert_vec_0(var_37, var_43);
        // term2 = math.motion_cross_force(dcvel, cinert_cvel) + math.motion_cross_force(cvel, cinert_dcvel)       <L 457>
        var_48 = motion_cross_force_0(var_43, var_46);
        var_49 = motion_cross_force_0(var_40, var_47);
        var_50 = wp::add(var_48, var_49);
        // Dcfrcbody_out[worldid, bodyid, dofid] = term1 + term2                                  <L 459>
        var_51 = wp::add(var_45, var_50);
        wp::array_store(var_Dcfrcbody_out, var_0, var_4, var_2, var_51);
    }
}



extern "C" __global__ void _qderiv_tendon_damping_ec990995_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_ntendon,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::float32> var_tendon_damping,
    wp::array_t<wp::vec_t<2, wp::float32>> var_tendon_dampingpoly,
    wp::array_t<wp::int32> var_M_elemid,
    wp::array_t<wp::float32> var_ten_J_in,
    wp::array_t<wp::float32> var_ten_velocity_in,
    wp::array_t<wp::int32> var_Mi,
    wp::array_t<wp::int32> var_Mj,
    wp::array_t<wp::float32> var_qDeriv_out)
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
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        const wp::int32 var_11 = 0;
        bool var_12;
        const wp::float32 var_13 = 0.0;
        wp::float32 var_14;
        wp::shape_t* var_15;
        const wp::int32 var_16 = 0;
        wp::int32 var_17;
        wp::shape_t var_18;
        wp::int32 var_19;
        wp::range_t var_20;
        wp::int32 var_21;
        wp::float32* var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        wp::shape_t* var_25;
        const wp::int32 var_26 = 0;
        wp::int32 var_27;
        wp::shape_t var_28;
        wp::int32 var_29;
        wp::vec_t<2, wp::float32>* var_30;
        wp::vec_t<2, wp::float32> var_31;
        wp::vec_t<2, wp::float32> var_32;
        bool var_33;
        const wp::float32 var_34 = 0.0;
        bool var_35;
        const wp::int32 var_36 = 0;
        wp::float32 var_37;
        const wp::float32 var_38 = 0.0;
        bool var_39;
        const wp::int32 var_40 = 1;
        wp::float32 var_41;
        const wp::float32 var_42 = 0.0;
        bool var_43;
        wp::int32* var_44;
        wp::int32 var_45;
        wp::int32 var_46;
        wp::int32* var_47;
        wp::int32 var_48;
        wp::int32 var_49;
        const wp::float32 var_50 = 0.0;
        wp::float32 var_51;
        const wp::float32 var_52 = 0.0;
        wp::float32 var_53;
        wp::range_t var_54;
        wp::int32 var_55;
        bool var_56;
        const wp::float32 var_57 = 0.0;
        bool var_58;
        const wp::float32 var_59 = 0.0;
        bool var_60;
        wp::int32 var_61;
        wp::int32* var_62;
        wp::int32 var_63;
        wp::int32 var_64;
        bool var_65;
        wp::float32* var_66;
        wp::float32 var_67;
        wp::float32 var_68;
        wp::float32 var_69;
        bool var_70;
        wp::float32* var_71;
        wp::float32 var_72;
        wp::float32 var_73;
        wp::float32 var_74;
        wp::float32* var_75;
        wp::float32 var_76;
        wp::float32 var_77;
        wp::float32 var_78;
        const wp::int32 var_79 = 1;
        wp::float32 var_80;
        wp::float32 var_81;
        wp::float32 var_82;
        wp::shape_t* var_83;
        const wp::int32 var_84 = 0;
        wp::int32 var_85;
        wp::shape_t var_86;
        wp::int32 var_87;
        wp::float32* var_88;
        wp::float32 var_89;
        wp::float32 var_90;
        wp::float32 var_91;
        //---------
        // forward
        // def _qderiv_tendon_damping(                                                            <L 263>
        // worldid, elemid = wp.tid()                                                             <L 282>
        builtin_tid2d(var_0, var_1);
        // dofiid = Mi[elemid]                                                                    <L 283>
        var_2 = wp::address(var_Mi, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // dofjid = Mj[elemid]                                                                    <L 284>
        var_5 = wp::address(var_Mj, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // madr = M_elemid[dofiid, dofjid]                                                        <L 287>
        var_8 = wp::address(var_M_elemid, var_3, var_6);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // if madr < 0:                                                                           <L 288>
        var_12 = (var_9 < var_11);
        if (var_12) {
            // return                                                                             <L 289>
            continue;
        }
        // qderiv = float(0.0)                                                                    <L 291>
        var_14 = wp::float(var_13);
        // tendon_damping_id = worldid % tendon_damping.shape[0]                                  <L 292>
        var_15 = &(var_tendon_damping.shape);
        var_18 = wp::load(var_15);
        var_17 = wp::extract(var_18, var_16);
        var_19 = wp::mod(var_0, var_17);
        // for tenid in range(ntendon):                                                           <L 293>
        var_20 = wp::range(var_ntendon);
        start_for_1:;
            if (iter_cmp(var_20) == 0) goto end_for_1;
            var_21 = wp::iter_next(var_20);
            // damping = tendon_damping[tendon_damping_id, tenid]                                 <L 294>
            var_22 = wp::address(var_tendon_damping, var_19, var_21);
            var_24 = wp::load(var_22);
            var_23 = wp::copy(var_24);
            // dpoly = tendon_dampingpoly[worldid % tendon_dampingpoly.shape[0], tenid]           <L 295>
            var_25 = &(var_tendon_dampingpoly.shape);
            var_28 = wp::load(var_25);
            var_27 = wp::extract(var_28, var_26);
            var_29 = wp::mod(var_0, var_27);
            var_30 = wp::address(var_tendon_dampingpoly, var_29, var_21);
            var_32 = wp::load(var_30);
            var_31 = wp::copy(var_32);
            // if damping == 0.0 and dpoly[0] == 0.0 and dpoly[1] == 0.0:                         <L 296>
            var_35 = (var_23 == var_34);
            var_33 = var_35;
            if (var_33) {
                var_37 = wp::extract(var_31, var_36);
                var_39 = (var_37 == var_38);
                var_33 = var_33 && var_39;
            }
            if (var_33) {
                var_41 = wp::extract(var_31, var_40);
                var_43 = (var_41 == var_42);
                var_33 = var_33 && var_43;
            }
            if (var_33) {
                // continue                                                                       <L 297>
                goto start_for_1;
            }
            // rownnz = ten_J_rownnz[tenid]                                                       <L 299>
            var_44 = wp::address(var_ten_J_rownnz, var_21);
            var_46 = wp::load(var_44);
            var_45 = wp::copy(var_46);
            // rowadr = ten_J_rowadr[tenid]                                                       <L 300>
            var_47 = wp::address(var_ten_J_rowadr, var_21);
            var_49 = wp::load(var_47);
            var_48 = wp::copy(var_49);
            // Ji = float(0.0)                                                                    <L 301>
            var_51 = wp::float(var_50);
            // Jj = float(0.0)                                                                    <L 302>
            var_53 = wp::float(var_52);
            // for k in range(rownnz):                                                            <L 303>
            var_54 = wp::range(var_45);
            start_for_3:;
                if (iter_cmp(var_54) == 0) goto end_for_3;
                var_55 = wp::iter_next(var_54);
                // if Ji != 0.0 and Jj != 0.0:                                                    <L 304>
                var_58 = (var_51 != var_57);
                var_56 = var_58;
                if (var_56) {
                    var_60 = (var_53 != var_59);
                    var_56 = var_56 && var_60;
                }
                if (var_56) {
                    // break                                                                      <L 305>
                    goto end_for_3;
                }
                // sparseid = rowadr + k                                                          <L 306>
                var_61 = wp::add(var_48, var_55);
                // colind = ten_J_colind[sparseid]                                                <L 307>
                var_62 = wp::address(var_ten_J_colind, var_61);
                var_64 = wp::load(var_62);
                var_63 = wp::copy(var_64);
                // if colind == dofiid:                                                           <L 308>
                var_65 = (var_63 == var_3);
                if (var_65) {
                    // Ji = ten_J_in[worldid, sparseid]                                           <L 309>
                    var_66 = wp::address(var_ten_J_in, var_0, var_61);
                    var_68 = wp::load(var_66);
                    var_67 = wp::copy(var_68);
                }
                var_69 = wp::where(var_65, var_67, var_51);
                // if colind == dofjid:                                                           <L 310>
                var_70 = (var_63 == var_6);
                if (var_70) {
                    // Jj = ten_J_in[worldid, sparseid]                                           <L 311>
                    var_71 = wp::address(var_ten_J_in, var_0, var_61);
                    var_73 = wp::load(var_71);
                    var_72 = wp::copy(var_73);
                }
                var_74 = wp::where(var_70, var_72, var_53);
                wp::assign(var_51, var_69);
                wp::assign(var_53, var_74);
                goto start_for_3;
            end_for_3:;
            // v = ten_velocity_in[worldid, tenid]                                                <L 313>
            var_75 = wp::address(var_ten_velocity_in, var_0, var_21);
            var_77 = wp::load(var_75);
            var_76 = wp::copy(var_77);
            // qderiv -= Ji * Jj * util_misc._poly_force_deriv(damping, dpoly, v, 1)              <L 314>
            var_78 = wp::mul(var_51, var_53);
            var_80 = _poly_force_deriv_0(var_23, var_31, var_76, var_79);
            var_81 = wp::mul(var_78, var_80);
            var_82 = wp::sub(var_14, var_81);
            wp::assign(var_14, var_82);
            goto start_for_1;
        end_for_1:;
        // qderiv *= opt_timestep[worldid % opt_timestep.shape[0]]                                <L 316>
        var_83 = &(var_opt_timestep.shape);
        var_86 = wp::load(var_83);
        var_85 = wp::extract(var_86, var_84);
        var_87 = wp::mod(var_0, var_85);
        var_88 = wp::address(var_opt_timestep, var_87);
        var_90 = wp::load(var_88);
        var_89 = wp::mul(var_14, var_90);
        // qDeriv_out[worldid, madr] -= qderiv                                                    <L 318>
        var_91 = wp::atomic_sub(var_qDeriv_out, var_0, var_9, var_89);
    }
}



extern "C" __global__ void deriv_rne_cfrcbody_backward_32e4b713_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_tree_,
    wp::array_t<wp::vec_t<6, wp::float32>> var_Dcfrcbody_out)
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
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::vec_t<6, wp::float32>* var_9;
        wp::vec_t<6, wp::float32> var_10;
        wp::vec_t<6, wp::float32> var_11;
        wp::slice_t var_12;
        const wp::int32 var_13 = 0;
        wp::slice_t var_14;
        const wp::int32 var_15 = 0;
        wp::array_t<wp::vec_t<6, wp::float32>> var_16;
        wp::vec_t<6, wp::float32> var_17;
        //---------
        // forward
        // def deriv_rne_cfrcbody_backward(                                                       <L 463>
        // worldid, nodeid, dofid = wp.tid()                                                      <L 472>
        builtin_tid3d(var_0, var_1, var_2);
        // bodyid = body_tree_[nodeid]                                                            <L 473>
        var_3 = wp::address(var_body_tree_, var_1);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // pid = body_parentid[bodyid]                                                            <L 474>
        var_6 = wp::address(var_body_parentid, var_4);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // val = Dcfrcbody_out[worldid, bodyid, dofid]                                            <L 478>
        var_9 = wp::address(var_Dcfrcbody_out, var_0, var_4, var_2);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // wp.atomic_add(Dcfrcbody_out[worldid, pid], dofid, val)                                 <L 479>
        var_12 = wp::slice_t(var_0, var_0, var_13);
        var_14 = wp::slice_t(var_7, var_7, var_15);
        var_16 = wp::view(var_Dcfrcbody_out, var_12, var_14);
        var_17 = wp::atomic_add(var_16, var_2, var_10);
    }
}



extern "C" __global__ void _qderiv_actuator_passive_vel_cc1a6654_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::array_t<wp::int32> var_actuator_dyntype,
    wp::array_t<wp::int32> var_actuator_gaintype,
    wp::array_t<wp::int32> var_actuator_biastype,
    wp::array_t<wp::int32> var_actuator_actadr,
    wp::array_t<wp::int32> var_actuator_actnum,
    wp::array_t<bool> var_actuator_forcelimited,
    wp::array_t<bool> var_actuator_actlimited,
    wp::array_t<wp::vec_t<10, wp::float32>> var_actuator_dynprm,
    wp::array_t<wp::vec_t<10, wp::float32>> var_actuator_gainprm,
    wp::array_t<wp::vec_t<10, wp::float32>> var_actuator_biasprm,
    wp::array_t<bool> var_actuator_actearly,
    wp::array_t<wp::vec_t<2, wp::float32>> var_actuator_forcerange,
    wp::array_t<wp::vec_t<2, wp::float32>> var_actuator_actrange,
    wp::array_t<wp::float32> var_act_in,
    wp::array_t<wp::float32> var_ctrl_in,
    wp::array_t<wp::float32> var_act_dot_in,
    wp::array_t<wp::float32> var_actuator_force_in,
    wp::array_t<wp::float32> var_vel_out)
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
        const wp::float32 var_12 = 0.0;
        wp::float32 var_13;
        wp::int32* var_14;
        const wp::int32 var_15 = 1;
        bool var_16;
        wp::int32 var_17;
        wp::vec_t<10, wp::float32>* var_18;
        const wp::int32 var_19 = 2;
        wp::float32 var_20;
        wp::vec_t<10, wp::float32> var_21;
        wp::int32* var_22;
        const wp::int32 var_23 = 3;
        bool var_24;
        wp::int32 var_25;
        const wp::float32 var_26 = 0.0;
        wp::shape_t* var_27;
        const wp::int32 var_28 = 0;
        wp::int32 var_29;
        wp::shape_t var_30;
        wp::int32 var_31;
        wp::vec_t<10, wp::float32>* var_32;
        wp::vec_t<10, wp::float32> var_33;
        wp::vec_t<10, wp::float32> var_34;
        wp::vec_t<10, wp::float32>* var_35;
        wp::vec_t<10, wp::float32> var_36;
        wp::vec_t<10, wp::float32> var_37;
        const wp::int32 var_38 = 0;
        wp::float32 var_39;
        const wp::int32 var_40 = 8;
        wp::float32 var_41;
        wp::int32 var_42;
        const wp::float32 var_43 = 0.0;
        const wp::int32 var_44 = 1;
        bool var_45;
        const wp::int32 var_46 = 6;
        wp::float32 var_47;
        wp::float32 var_48;
        wp::float32 var_49;
        const wp::int32 var_50 = 2;
        bool var_51;
        const wp::int32 var_52 = 4;
        wp::float32 var_53;
        wp::float32 var_54;
        wp::float32 var_55;
        wp::float32 var_56;
        const wp::float32 var_57 = 0.0;
        bool var_58;
        const wp::float32 var_59 = 1e-15;
        const wp::int32 var_60 = 0;
        wp::float32 var_61;
        wp::float32 var_62;
        const wp::int32 var_63 = 1;
        wp::float32 var_64;
        const wp::float32 var_65 = 1.0;
        wp::shape_t* var_66;
        const wp::int32 var_67 = 0;
        wp::int32 var_68;
        wp::shape_t var_69;
        wp::int32 var_70;
        wp::float32* var_71;
        wp::float32 var_72;
        wp::float32 var_73;
        wp::float32 var_74;
        wp::float32 var_75;
        wp::float32 var_76;
        wp::float32 var_77;
        wp::float32 var_78;
        wp::float32 var_79;
        wp::float32 var_80;
        wp::float32 var_81;
        wp::float32 var_82;
        const wp::float32 var_83 = 0.0;
        bool var_84;
        const wp::int32 var_85 = 0;
        wp::float32 var_86;
        wp::float32 var_87;
        const wp::int32 var_88 = 1;
        wp::float32 var_89;
        wp::float32 var_90;
        wp::float32 var_91;
        wp::float32 var_92;
        wp::float32 var_93;
        wp::float32 var_94;
        wp::float32 var_95;
        wp::float32 var_96;
        wp::float32 var_97;
        wp::float32 var_98;
        const wp::int32 var_99 = 6;
        wp::float32 var_100;
        const wp::float32 var_101 = 0.0;
        bool var_102;
        wp::float32 var_103;
        wp::float32 var_104;
        wp::float32 var_105;
        wp::float32 var_106;
        const wp::float32 var_107 = 0.0;
        wp::float32 var_108;
        wp::float32 var_109;
        wp::float32 var_110;
        wp::int32* var_111;
        const wp::int32 var_112 = 1;
        bool var_113;
        wp::int32 var_114;
        wp::vec_t<10, wp::float32>* var_115;
        const wp::int32 var_116 = 2;
        wp::float32 var_117;
        wp::vec_t<10, wp::float32> var_118;
        wp::float32 var_119;
        wp::float32 var_120;
        wp::int32* var_121;
        const wp::int32 var_122 = 3;
        bool var_123;
        wp::int32 var_124;
        wp::shape_t* var_125;
        const wp::int32 var_126 = 0;
        wp::int32 var_127;
        wp::shape_t var_128;
        wp::int32 var_129;
        wp::vec_t<10, wp::float32>* var_130;
        wp::vec_t<10, wp::float32> var_131;
        wp::vec_t<10, wp::float32> var_132;
        const wp::int32 var_133 = 0;
        wp::float32 var_134;
        const wp::float32 var_135 = 0.0;
        bool var_136;
        wp::vec_t<10, wp::float32>* var_137;
        wp::vec_t<10, wp::float32> var_138;
        wp::vec_t<10, wp::float32> var_139;
        const wp::int32 var_140 = 0;
        wp::float32 var_141;
        const wp::int32 var_142 = 1;
        wp::float32 var_143;
        wp::vec_t<6, wp::int32> var_144;
        const wp::int32 var_145 = 2;
        wp::int32 var_146;
        const wp::int32 var_147 = 0;
        bool var_148;
        wp::int32* var_149;
        wp::int32 var_150;
        wp::int32 var_151;
        wp::float32* var_152;
        wp::float32 var_153;
        wp::float32 var_154;
        const wp::int32 var_155 = 2;
        wp::float32 var_156;
        const wp::int32 var_157 = 3;
        wp::float32 var_158;
        const wp::int32 var_159 = 4;
        wp::float32 var_160;
        const wp::float32 var_161 = 1.0;
        wp::float32 var_162;
        wp::float32 var_163;
        wp::float32 var_164;
        wp::float32 var_165;
        wp::float32 var_166;
        wp::float32 var_167;
        wp::float32 var_168;
        wp::float32 var_169;
        wp::float32 var_170;
        wp::float32 var_171;
        wp::float32 var_172;
        wp::float32 var_173;
        wp::vec_t<10, wp::float32> var_174;
        wp::float32 var_175;
        wp::float32 var_176;
        wp::float32 var_177;
        wp::vec_t<10, wp::float32> var_178;
        wp::vec_t<10, wp::float32> var_179;
        wp::float32 var_180;
        wp::float32 var_181;
        wp::float32 var_182;
        wp::float32 var_183;
        wp::vec_t<10, wp::float32> var_184;
        wp::vec_t<10, wp::float32> var_185;
        wp::float32 var_186;
        wp::float32 var_187;
        wp::float32 var_188;
        bool var_189;
        const wp::float32 var_190 = 0.0;
        bool var_191;
        const wp::float32 var_192 = 0.0;
        bool var_193;
        const wp::float32 var_194 = 0.0;
        bool* var_195;
        bool var_196;
        wp::float32* var_197;
        wp::float32 var_198;
        wp::float32 var_199;
        wp::shape_t* var_200;
        const wp::int32 var_201 = 0;
        wp::int32 var_202;
        wp::shape_t var_203;
        wp::int32 var_204;
        wp::vec_t<2, wp::float32>* var_205;
        wp::vec_t<2, wp::float32> var_206;
        wp::vec_t<2, wp::float32> var_207;
        bool var_208;
        const wp::int32 var_209 = 0;
        wp::float32 var_210;
        bool var_211;
        const wp::int32 var_212 = 1;
        wp::float32 var_213;
        bool var_214;
        const wp::float32 var_215 = 0.0;
        bool var_216;
        wp::float32 var_217;
        wp::int32* var_218;
        const wp::int32 var_219 = 0;
        bool var_220;
        wp::int32 var_221;
        const wp::float32 var_222 = 0.0;
        bool var_223;
        wp::int32* var_224;
        wp::int32* var_225;
        wp::int32 var_226;
        wp::int32 var_227;
        wp::int32 var_228;
        const wp::int32 var_229 = 1;
        wp::int32 var_230;
        bool* var_231;
        bool var_232;
        wp::shape_t* var_233;
        const wp::int32 var_234 = 0;
        wp::int32 var_235;
        wp::shape_t var_236;
        wp::int32 var_237;
        wp::float32* var_238;
        wp::int32* var_239;
        wp::shape_t* var_240;
        const wp::int32 var_241 = 0;
        wp::int32 var_242;
        wp::shape_t var_243;
        wp::int32 var_244;
        wp::vec_t<10, wp::float32>* var_245;
        wp::shape_t* var_246;
        const wp::int32 var_247 = 0;
        wp::int32 var_248;
        wp::shape_t var_249;
        wp::int32 var_250;
        wp::vec_t<2, wp::float32>* var_251;
        wp::float32* var_252;
        wp::float32* var_253;
        const wp::float32 var_254 = 1.0;
        bool* var_255;
        wp::float32 var_256;
        wp::float32 var_257;
        wp::int32 var_258;
        wp::vec_t<10, wp::float32> var_259;
        wp::vec_t<2, wp::float32> var_260;
        wp::float32 var_261;
        wp::float32 var_262;
        bool var_263;
        bool var_264;
        bool var_265;
        wp::float32* var_266;
        wp::float32 var_267;
        wp::float32 var_268;
        bool var_269;
        wp::float32 var_270;
        bool var_271;
        wp::float32 var_272;
        wp::float32 var_273;
        wp::float32 var_274;
        wp::float32 var_275;
        const wp::float32 var_276 = 0.0;
        bool var_277;
        wp::float32* var_278;
        wp::float32 var_279;
        wp::float32 var_280;
        wp::float32 var_281;
        wp::float32 var_282;
        wp::float32 var_283;
        //---------
        // forward
        // def _qderiv_actuator_passive_vel(                                                      <L 38>
        // worldid, actid = wp.tid()                                                              <L 62>
        builtin_tid2d(var_0, var_1);
        // actuator_gainprm_id = worldid % actuator_gainprm.shape[0]                              <L 64>
        var_2 = &(var_actuator_gainprm.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // actuator_biasprm_id = worldid % actuator_biasprm.shape[0]                              <L 65>
        var_7 = &(var_actuator_biasprm.shape);
        var_10 = wp::load(var_7);
        var_9 = wp::extract(var_10, var_8);
        var_11 = wp::mod(var_0, var_9);
        // bias = float(0.0)                                                                      <L 67>
        var_13 = wp::float(var_12);
        // if actuator_gaintype[actid] == GainType.AFFINE:                                        <L 69>
        var_14 = wp::address(var_actuator_gaintype, var_1);
        var_17 = wp::load(var_14);
        var_16 = (var_17 == var_15);
        if (var_16) {
            // gain = actuator_gainprm[actuator_gainprm_id, actid][2]                             <L 70>
            var_18 = wp::address(var_actuator_gainprm, var_6, var_1);
            var_21 = wp::load(var_18);
            var_20 = wp::extract(var_21, var_19);
        }
        if (!var_16) {
            // elif actuator_gaintype[actid] == GainType.DCMOTOR:                                 <L 71>
            var_22 = wp::address(var_actuator_gaintype, var_1);
            var_25 = wp::load(var_22);
            var_24 = (var_25 == var_23);
            if (var_24) {
                // gain = 0.0                                                                     <L 72>
                // dynprm = actuator_dynprm[worldid % actuator_dynprm.shape[0], actid]            <L 73>
                var_27 = &(var_actuator_dynprm.shape);
                var_30 = wp::load(var_27);
                var_29 = wp::extract(var_30, var_28);
                var_31 = wp::mod(var_0, var_29);
                var_32 = wp::address(var_actuator_dynprm, var_31, var_1);
                var_34 = wp::load(var_32);
                var_33 = wp::copy(var_34);
                // gainprm = actuator_gainprm[actuator_gainprm_id, actid]                         <L 74>
                var_35 = wp::address(var_actuator_gainprm, var_6, var_1);
                var_37 = wp::load(var_35);
                var_36 = wp::copy(var_37);
                // te = dynprm[0]                                                                 <L 75>
                var_39 = wp::extract(var_33, var_38);
                // input_mode = int(gainprm[8])                                                   <L 78>
                var_41 = wp::extract(var_36, var_40);
                var_42 = wp::int(var_41);
                // dVdw = 0.0                                                                     <L 79>
                // if input_mode == 1:                                                            <L 80>
                var_45 = (var_42 == var_44);
                if (var_45) {
                    // dVdw = -gainprm[6]  # position: -kd                                        <L 81>
                    var_47 = wp::extract(var_36, var_46);
                    var_48 = wp::neg(var_47);
                }
                var_49 = wp::where(var_45, var_48, var_43);
                if (!var_45) {
                    // elif input_mode == 2:                                                      <L 82>
                    var_51 = (var_42 == var_50);
                    if (var_51) {
                        // dVdw = -gainprm[4]  # velocity: -kp                                    <L 83>
                        var_53 = wp::extract(var_36, var_52);
                        var_54 = wp::neg(var_53);
                    }
                    var_55 = wp::where(var_51, var_54, var_49);
                }
                var_56 = wp::where(var_45, var_49, var_55);
                // if te > 0.0:                                                                   <L 85>
                var_58 = (var_39 > var_57);
                if (var_58) {
                    // R = wp.max(MJ_MINVAL, gainprm[0])                                          <L 88>
                    var_61 = wp::extract(var_36, var_60);
                    var_62 = wp::max(var_59, var_61);
                    // K = gainprm[1]                                                             <L 89>
                    var_64 = wp::extract(var_36, var_63);
                    // s = 1.0 - wp.exp(-opt_timestep[worldid % opt_timestep.shape[0]] / te)       <L 90>
                    var_66 = &(var_opt_timestep.shape);
                    var_69 = wp::load(var_66);
                    var_68 = wp::extract(var_69, var_67);
                    var_70 = wp::mod(var_0, var_68);
                    var_71 = wp::address(var_opt_timestep, var_70);
                    var_73 = wp::load(var_71);
                    var_72 = wp::neg(var_73);
                    var_74 = wp::div(var_72, var_39);
                    var_75 = wp::exp(var_74);
                    var_76 = wp::sub(var_65, var_75);
                    // bias += K * (dVdw - K) * s / R                                             <L 91>
                    var_77 = wp::sub(var_56, var_64);
                    var_78 = wp::mul(var_64, var_77);
                    var_79 = wp::mul(var_78, var_76);
                    var_80 = wp::div(var_79, var_62);
                    var_81 = wp::add(var_13, var_80);
                }
                var_82 = wp::where(var_58, var_81, var_13);
                if (!var_58) {
                    // elif dVdw != 0.0:                                                          <L 92>
                    var_84 = (var_56 != var_83);
                    if (var_84) {
                        // R = wp.max(MJ_MINVAL, gainprm[0])                                      <L 94>
                        var_86 = wp::extract(var_36, var_85);
                        var_87 = wp::max(var_59, var_86);
                        // K = gainprm[1]                                                         <L 95>
                        var_89 = wp::extract(var_36, var_88);
                        // bias += K * dVdw / R                                                   <L 96>
                        var_90 = wp::mul(var_89, var_56);
                        var_91 = wp::div(var_90, var_87);
                        var_92 = wp::add(var_82, var_91);
                    }
                    var_93 = wp::where(var_84, var_92, var_82);
                    var_94 = wp::where(var_84, var_87, var_62);
                    var_95 = wp::where(var_84, var_89, var_64);
                }
                var_96 = wp::where(var_58, var_82, var_93);
                var_97 = wp::where(var_58, var_62, var_94);
                var_98 = wp::where(var_58, var_64, var_95);
                // sigma1 = dynprm[6]                                                             <L 100>
                var_100 = wp::extract(var_33, var_99);
                // if sigma1 > 0.0:                                                               <L 101>
                var_102 = (var_100 > var_101);
                if (var_102) {
                    // bias -= sigma1                                                             <L 102>
                    var_103 = wp::sub(var_96, var_100);
                }
                var_104 = wp::where(var_102, var_103, var_96);
            }
            var_105 = wp::where(var_24, var_104, var_13);
            var_106 = wp::where(var_24, var_26, var_20);
            if (!var_24) {
                // gain = 0.0                                                                     <L 104>
            }
            var_108 = wp::where(var_24, var_106, var_107);
        }
        var_109 = wp::where(var_16, var_13, var_105);
        var_110 = wp::where(var_16, var_20, var_108);
        // if actuator_biastype[actid] == BiasType.AFFINE:                                        <L 106>
        var_111 = wp::address(var_actuator_biastype, var_1);
        var_114 = wp::load(var_111);
        var_113 = (var_114 == var_112);
        if (var_113) {
            // bias += actuator_biasprm[actuator_biasprm_id, actid][2]                            <L 107>
            var_115 = wp::address(var_actuator_biasprm, var_11, var_1);
            var_118 = wp::load(var_115);
            var_117 = wp::extract(var_118, var_116);
            var_119 = wp::add(var_109, var_117);
        }
        var_120 = wp::where(var_113, var_119, var_109);
        if (!var_113) {
            // elif actuator_biastype[actid] == BiasType.DCMOTOR:                                 <L 108>
            var_121 = wp::address(var_actuator_biastype, var_1);
            var_124 = wp::load(var_121);
            var_123 = (var_124 == var_122);
            if (var_123) {
                // dynprm = actuator_dynprm[worldid % actuator_dynprm.shape[0], actid]            <L 109>
                var_125 = &(var_actuator_dynprm.shape);
                var_128 = wp::load(var_125);
                var_127 = wp::extract(var_128, var_126);
                var_129 = wp::mod(var_0, var_127);
                var_130 = wp::address(var_actuator_dynprm, var_129, var_1);
                var_132 = wp::load(var_130);
                var_131 = wp::copy(var_132);
                // te = dynprm[0]                                                                 <L 110>
                var_134 = wp::extract(var_131, var_133);
                // if te <= 0.0:                                                                  <L 111>
                var_136 = (var_134 <= var_135);
                if (var_136) {
                    // gainprm = actuator_gainprm[actuator_gainprm_id, actid]                     <L 112>
                    var_137 = wp::address(var_actuator_gainprm, var_6, var_1);
                    var_139 = wp::load(var_137);
                    var_138 = wp::copy(var_139);
                    // R = gainprm[0]                                                             <L 113>
                    var_141 = wp::extract(var_138, var_140);
                    // K = gainprm[1]                                                             <L 114>
                    var_143 = wp::extract(var_138, var_142);
                    // slots = util_misc.dcmotor_slots(dynprm, gainprm)                           <L 116>
                    var_144 = dcmotor_slots_0(var_131, var_138);
                    // slot_Ta = slots[2]                                                         <L 117>
                    var_146 = wp::extract(var_144, var_145);
                    // if slot_Ta >= 0:                                                           <L 119>
                    var_148 = (var_146 >= var_147);
                    if (var_148) {
                        // adr = actuator_actadr[actid] + slot_Ta                                 <L 120>
                        var_149 = wp::address(var_actuator_actadr, var_1);
                        var_151 = wp::load(var_149);
                        var_150 = wp::add(var_151, var_146);
                        // T = act_in[worldid, adr]                                               <L 121>
                        var_152 = wp::address(var_act_in, var_0, var_150);
                        var_154 = wp::load(var_152);
                        var_153 = wp::copy(var_154);
                        // alpha = gainprm[2]                                                     <L 122>
                        var_156 = wp::extract(var_138, var_155);
                        // T0 = gainprm[3]                                                        <L 123>
                        var_158 = wp::extract(var_138, var_157);
                        // Ta = dynprm[4]                                                         <L 124>
                        var_160 = wp::extract(var_131, var_159);
                        // R *= 1.0 + alpha * (T + Ta - T0)                                       <L 125>
                        var_162 = wp::add(var_153, var_160);
                        var_163 = wp::sub(var_162, var_158);
                        var_164 = wp::mul(var_156, var_163);
                        var_165 = wp::add(var_161, var_164);
                        var_166 = wp::mul(var_141, var_165);
                    }
                    var_167 = wp::where(var_148, var_166, var_141);
                    // bias += -K * K / wp.max(MJ_MINVAL, R)                                      <L 127>
                    var_168 = wp::neg(var_143);
                    var_169 = wp::mul(var_168, var_143);
                    var_170 = wp::max(var_59, var_167);
                    var_171 = wp::div(var_169, var_170);
                    var_172 = wp::add(var_120, var_171);
                }
                var_173 = wp::where(var_136, var_172, var_120);
                var_174 = wp::where(var_136, var_138, var_36);
                var_175 = wp::where(var_136, var_167, var_97);
                var_176 = wp::where(var_136, var_143, var_98);
            }
            var_177 = wp::where(var_123, var_173, var_120);
            var_178 = wp::where(var_123, var_131, var_33);
            var_179 = wp::where(var_123, var_174, var_36);
            var_180 = wp::where(var_123, var_134, var_39);
            var_181 = wp::where(var_123, var_175, var_97);
            var_182 = wp::where(var_123, var_176, var_98);
        }
        var_183 = wp::where(var_113, var_120, var_177);
        var_184 = wp::where(var_113, var_33, var_178);
        var_185 = wp::where(var_113, var_36, var_179);
        var_186 = wp::where(var_113, var_39, var_180);
        var_187 = wp::where(var_113, var_97, var_181);
        var_188 = wp::where(var_113, var_98, var_182);
        // if bias == 0.0 and gain == 0.0:                                                        <L 129>
        var_191 = (var_183 == var_190);
        var_189 = var_191;
        if (var_189) {
            var_193 = (var_110 == var_192);
            var_189 = var_189 && var_193;
        }
        if (var_189) {
            // vel_out[worldid, actid] = 0.0                                                      <L 130>
            wp::array_store(var_vel_out, var_0, var_1, var_194);
            // return                                                                             <L 131>
            continue;
        }
        // if actuator_forcelimited[actid]:                                                       <L 134>
        var_195 = wp::address(var_actuator_forcelimited, var_1);
        var_196 = wp::load(var_195);
        if (var_196) {
            // force = actuator_force_in[worldid, actid]                                          <L 135>
            var_197 = wp::address(var_actuator_force_in, var_0, var_1);
            var_199 = wp::load(var_197);
            var_198 = wp::copy(var_199);
            // forcerange = actuator_forcerange[worldid % actuator_forcerange.shape[0], actid]       <L 136>
            var_200 = &(var_actuator_forcerange.shape);
            var_203 = wp::load(var_200);
            var_202 = wp::extract(var_203, var_201);
            var_204 = wp::mod(var_0, var_202);
            var_205 = wp::address(var_actuator_forcerange, var_204, var_1);
            var_207 = wp::load(var_205);
            var_206 = wp::copy(var_207);
            // if force <= forcerange[0] or force >= forcerange[1]:                               <L 137>
            var_210 = wp::extract(var_206, var_209);
            var_211 = (var_198 <= var_210);
            var_208 = var_211;
            if (!var_208) {
                var_213 = wp::extract(var_206, var_212);
                var_214 = (var_198 >= var_213);
                var_208 = var_208 || var_214;
            }
            if (var_208) {
                // vel_out[worldid, actid] = 0.0                                                  <L 138>
                wp::array_store(var_vel_out, var_0, var_1, var_215);
                // return                                                                         <L 139>
                continue;
            }
        }
        var_216 = wp::load(var_195);
        // vel = float(bias)                                                                      <L 141>
        var_217 = wp::float(var_183);
        // if actuator_dyntype[actid] != DynType.NONE:                                            <L 142>
        var_218 = wp::address(var_actuator_dyntype, var_1);
        var_221 = wp::load(var_218);
        var_220 = (var_221 != var_219);
        if (var_220) {
            // if gain != 0.0:                                                                    <L 143>
            var_223 = (var_110 != var_222);
            if (var_223) {
                // act_adr = actuator_actadr[actid] + actuator_actnum[actid] - 1                  <L 144>
                var_224 = wp::address(var_actuator_actadr, var_1);
                var_225 = wp::address(var_actuator_actnum, var_1);
                var_227 = wp::load(var_224);
                var_228 = wp::load(var_225);
                var_226 = wp::add(var_227, var_228);
                var_230 = wp::sub(var_226, var_229);
                // if actuator_actearly[actid]:                                                   <L 147>
                var_231 = wp::address(var_actuator_actearly, var_1);
                var_232 = wp::load(var_231);
                if (var_232) {
                    // act = next_act(                                                            <L 148>
                    // opt_timestep[worldid % opt_timestep.shape[0]],                             <L 149>
                    var_233 = &(var_opt_timestep.shape);
                    var_236 = wp::load(var_233);
                    var_235 = wp::extract(var_236, var_234);
                    var_237 = wp::mod(var_0, var_235);
                    var_238 = wp::address(var_opt_timestep, var_237);
                    // actuator_dyntype[actid],                                                   <L 150>
                    var_239 = wp::address(var_actuator_dyntype, var_1);
                    // actuator_dynprm[worldid % actuator_dynprm.shape[0], actid],                <L 151>
                    var_240 = &(var_actuator_dynprm.shape);
                    var_243 = wp::load(var_240);
                    var_242 = wp::extract(var_243, var_241);
                    var_244 = wp::mod(var_0, var_242);
                    var_245 = wp::address(var_actuator_dynprm, var_244, var_1);
                    // actuator_actrange[worldid % actuator_actrange.shape[0], actid],            <L 152>
                    var_246 = &(var_actuator_actrange.shape);
                    var_249 = wp::load(var_246);
                    var_248 = wp::extract(var_249, var_247);
                    var_250 = wp::mod(var_0, var_248);
                    var_251 = wp::address(var_actuator_actrange, var_250, var_1);
                    // act_in[worldid, act_adr],                                                  <L 153>
                    var_252 = wp::address(var_act_in, var_0, var_230);
                    // act_dot_in[worldid, act_adr],                                              <L 154>
                    var_253 = wp::address(var_act_dot_in, var_0, var_230);
                    // 1.0,                                                                       <L 155>
                    // actuator_actlimited[actid],                                                <L 156>
                    var_255 = wp::address(var_actuator_actlimited, var_1);
                    var_257 = wp::load(var_238);
                    var_258 = wp::load(var_239);
                    var_259 = wp::load(var_245);
                    var_260 = wp::load(var_251);
                    var_261 = wp::load(var_252);
                    var_262 = wp::load(var_253);
                    var_263 = wp::load(var_255);
                    var_256 = next_act_0(var_257, var_258, var_259, var_260, var_261, var_262, var_254, var_263);
                }
                var_264 = wp::load(var_231);
                var_265 = wp::load(var_231);
                if (!var_265) {
                    // act = act_in[worldid, act_adr]                                             <L 159>
                    var_266 = wp::address(var_act_in, var_0, var_230);
                    var_268 = wp::load(var_266);
                    var_267 = wp::copy(var_268);
                }
                var_269 = wp::load(var_231);
                var_271 = wp::load(var_231);
                var_270 = wp::where(var_271, var_256, var_267);
                // vel += gain * act                                                              <L 161>
                var_272 = wp::mul(var_110, var_270);
                var_273 = wp::add(var_217, var_272);
            }
            var_274 = wp::where(var_223, var_273, var_217);
        }
        var_275 = wp::where(var_220, var_274, var_217);
        if (!var_220) {
            // if gain != 0.0:                                                                    <L 163>
            var_277 = (var_110 != var_276);
            if (var_277) {
                // vel += gain * ctrl_in[worldid, actid]                                          <L 164>
                var_278 = wp::address(var_ctrl_in, var_0, var_1);
                var_280 = wp::load(var_278);
                var_279 = wp::mul(var_110, var_280);
                var_281 = wp::add(var_275, var_279);
            }
            var_282 = wp::where(var_277, var_281, var_275);
        }
        var_283 = wp::where(var_220, var_275, var_282);
        // vel_out[worldid, actid] = vel                                                          <L 166>
        wp::array_store(var_vel_out, var_0, var_1, var_283);
    }
}



extern "C" __global__ void deriv_rne_cvel_cdof_dot_8e31d9b1_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_jntnum,
    wp::array_t<wp::int32> var_body_jntadr,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_jnt_type,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::int32> var_body_tree_,
    wp::array_t<wp::vec_t<6, wp::float32>> var_Dcvel_out,
    wp::array_t<wp::vec_t<6, wp::float32>> var_Dcdof_dot_out)
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
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::int32* var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::slice_t var_18;
        const wp::int32 var_19 = 0;
        wp::array_t<wp::vec_t<6, wp::float32>> var_20;
        wp::vec_t<6, wp::float32>* var_21;
        wp::vec_t<6, wp::float32> var_22;
        wp::vec_t<6, wp::float32> var_23;
        const wp::int32 var_24 = 0;
        bool var_25;
        wp::int32 var_26;
        wp::int32 var_27;
        wp::range_t var_28;
        wp::int32 var_29;
        wp::int32* var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        const wp::int32 var_33 = 0;
        bool var_34;
        bool var_35;
        bool var_36;
        const wp::int32 var_37 = 3;
        wp::int32 var_38;
        bool var_39;
        wp::vec_t<6, wp::float32>* var_40;
        wp::vec_t<6, wp::float32> var_41;
        wp::vec_t<6, wp::float32> var_42;
        wp::vec_t<6, wp::float32> var_43;
        const wp::int32 var_44 = 3;
        wp::int32 var_45;
        wp::vec_t<6, wp::float32>* var_46;
        wp::vec_t<6, wp::float32> var_47;
        wp::vec_t<6, wp::float32> var_48;
        const wp::int32 var_49 = 3;
        wp::int32 var_50;
        const wp::int32 var_51 = 4;
        wp::int32 var_52;
        wp::vec_t<6, wp::float32>* var_53;
        wp::vec_t<6, wp::float32> var_54;
        wp::vec_t<6, wp::float32> var_55;
        const wp::int32 var_56 = 4;
        wp::int32 var_57;
        const wp::int32 var_58 = 5;
        wp::int32 var_59;
        wp::vec_t<6, wp::float32>* var_60;
        wp::vec_t<6, wp::float32> var_61;
        wp::vec_t<6, wp::float32> var_62;
        const wp::int32 var_63 = 5;
        wp::int32 var_64;
        bool var_65;
        const wp::int32 var_66 = 3;
        wp::int32 var_67;
        bool var_68;
        const wp::int32 var_69 = 6;
        wp::int32 var_70;
        bool var_71;
        wp::vec_t<6, wp::float32>* var_72;
        wp::vec_t<6, wp::float32> var_73;
        wp::vec_t<6, wp::float32> var_74;
        wp::vec_t<6, wp::float32> var_75;
        const wp::int32 var_76 = 6;
        wp::int32 var_77;
        wp::vec_t<6, wp::float32> var_78;
        wp::int32 var_79;
        const wp::int32 var_80 = 1;
        bool var_81;
        const wp::int32 var_82 = 0;
        wp::int32 var_83;
        wp::vec_t<6, wp::float32>* var_84;
        wp::vec_t<6, wp::float32> var_85;
        wp::vec_t<6, wp::float32> var_86;
        const wp::int32 var_87 = 0;
        wp::int32 var_88;
        const wp::int32 var_89 = 1;
        wp::int32 var_90;
        wp::vec_t<6, wp::float32>* var_91;
        wp::vec_t<6, wp::float32> var_92;
        wp::vec_t<6, wp::float32> var_93;
        const wp::int32 var_94 = 1;
        wp::int32 var_95;
        const wp::int32 var_96 = 2;
        wp::int32 var_97;
        wp::vec_t<6, wp::float32>* var_98;
        wp::vec_t<6, wp::float32> var_99;
        wp::vec_t<6, wp::float32> var_100;
        const wp::int32 var_101 = 2;
        wp::int32 var_102;
        bool var_103;
        bool var_104;
        const wp::int32 var_105 = 3;
        wp::int32 var_106;
        bool var_107;
        wp::vec_t<6, wp::float32>* var_108;
        wp::vec_t<6, wp::float32> var_109;
        wp::vec_t<6, wp::float32> var_110;
        wp::vec_t<6, wp::float32> var_111;
        const wp::int32 var_112 = 3;
        wp::int32 var_113;
        wp::vec_t<6, wp::float32> var_114;
        wp::int32 var_115;
        wp::vec_t<6, wp::float32>* var_116;
        wp::vec_t<6, wp::float32> var_117;
        wp::vec_t<6, wp::float32> var_118;
        bool var_119;
        wp::vec_t<6, wp::float32>* var_120;
        wp::vec_t<6, wp::float32> var_121;
        wp::vec_t<6, wp::float32> var_122;
        wp::vec_t<6, wp::float32> var_123;
        const wp::int32 var_124 = 1;
        wp::int32 var_125;
        wp::vec_t<6, wp::float32> var_126;
        wp::int32 var_127;
        wp::vec_t<6, wp::float32> var_128;
        wp::int32 var_129;
        //---------
        // forward
        // def deriv_rne_cvel_cdof_dot(                                                           <L 322>
        // worldid, nodeid, dofid = wp.tid()                                                      <L 345>
        builtin_tid3d(var_0, var_1, var_2);
        // bodyid = body_tree_[nodeid]                                                            <L 346>
        var_3 = wp::address(var_body_tree_, var_1);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // dofadr = body_dofadr[bodyid]                                                           <L 347>
        var_6 = wp::address(var_body_dofadr, var_4);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // jntid = body_jntadr[bodyid]                                                            <L 348>
        var_9 = wp::address(var_body_jntadr, var_4);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // jntnum = body_jntnum[bodyid]                                                           <L 349>
        var_12 = wp::address(var_body_jntnum, var_4);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // pid = body_parentid[bodyid]                                                            <L 350>
        var_15 = wp::address(var_body_parentid, var_4);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
        // cdof = cdof_in[worldid]                                                                <L 352>
        var_18 = wp::slice_t(var_0, var_0, var_19);
        var_20 = wp::view(var_cdof_in, var_18);
        // cvel_k = Dcvel_out[worldid, pid, dofid]                                                <L 355>
        var_21 = wp::address(var_Dcvel_out, var_0, var_16, var_2);
        var_23 = wp::load(var_21);
        var_22 = wp::copy(var_23);
        // if jntnum == 0:                                                                        <L 357>
        var_25 = (var_13 == var_24);
        if (var_25) {
            // Dcvel_out[worldid, bodyid, dofid] = cvel_k                                         <L 358>
            wp::array_store(var_Dcvel_out, var_0, var_4, var_2, var_22);
            // return                                                                             <L 359>
            continue;
        }
        // dof_i = dofadr                                                                         <L 361>
        var_26 = wp::copy(var_7);
        // for j in range(jntid, jntid + jntnum):                                                 <L 363>
        var_27 = wp::add(var_10, var_13);
        var_28 = wp::range(var_10, var_27);
        start_for_1:;
            if (iter_cmp(var_28) == 0) goto end_for_1;
            var_29 = wp::iter_next(var_28);
            // jnttype = jnt_type[j]                                                              <L 364>
            var_30 = wp::address(var_jnt_type, var_29);
            var_32 = wp::load(var_30);
            var_31 = wp::copy(var_32);
            // if jnttype == 0:  # FREE                                                           <L 366>
            var_34 = (var_31 == var_33);
            if (var_34) {
                // if dofid >= dof_i and dofid < dof_i + 3:                                       <L 368>
                var_36 = (var_2 >= var_26);
                var_35 = var_36;
                if (var_35) {
                    var_38 = wp::add(var_26, var_37);
                    var_39 = (var_2 < var_38);
                    var_35 = var_35 && var_39;
                }
                if (var_35) {
                    // cvel_k += cdof[dofid]                                                      <L 369>
                    var_40 = wp::address(var_20, var_2);
                    var_42 = wp::load(var_40);
                    var_41 = wp::add(var_22, var_42);
                }
                var_43 = wp::where(var_35, var_41, var_22);
                // Dcdof_dot_out[worldid, dof_i + 3, dofid] = math.motion_cross(cvel_k, cdof[dof_i + 3])       <L 375>
                var_45 = wp::add(var_26, var_44);
                var_46 = wp::address(var_20, var_45);
                var_48 = wp::load(var_46);
                var_47 = motion_cross_0(var_43, var_48);
                var_50 = wp::add(var_26, var_49);
                wp::array_store(var_Dcdof_dot_out, var_0, var_50, var_2, var_47);
                // Dcdof_dot_out[worldid, dof_i + 4, dofid] = math.motion_cross(cvel_k, cdof[dof_i + 4])       <L 376>
                var_52 = wp::add(var_26, var_51);
                var_53 = wp::address(var_20, var_52);
                var_55 = wp::load(var_53);
                var_54 = motion_cross_0(var_43, var_55);
                var_57 = wp::add(var_26, var_56);
                wp::array_store(var_Dcdof_dot_out, var_0, var_57, var_2, var_54);
                // Dcdof_dot_out[worldid, dof_i + 5, dofid] = math.motion_cross(cvel_k, cdof[dof_i + 5])       <L 377>
                var_59 = wp::add(var_26, var_58);
                var_60 = wp::address(var_20, var_59);
                var_62 = wp::load(var_60);
                var_61 = motion_cross_0(var_43, var_62);
                var_64 = wp::add(var_26, var_63);
                wp::array_store(var_Dcdof_dot_out, var_0, var_64, var_2, var_61);
                // if dofid >= dof_i + 3 and dofid < dof_i + 6:                                   <L 380>
                var_67 = wp::add(var_26, var_66);
                var_68 = (var_2 >= var_67);
                var_65 = var_68;
                if (var_65) {
                    var_70 = wp::add(var_26, var_69);
                    var_71 = (var_2 < var_70);
                    var_65 = var_65 && var_71;
                }
                if (var_65) {
                    // cvel_k += cdof[dofid]                                                      <L 381>
                    var_72 = wp::address(var_20, var_2);
                    var_74 = wp::load(var_72);
                    var_73 = wp::add(var_43, var_74);
                }
                var_75 = wp::where(var_65, var_73, var_43);
                // dof_i += 6                                                                     <L 383>
                var_77 = wp::add(var_26, var_76);
            }
            var_78 = wp::where(var_34, var_75, var_22);
            var_79 = wp::where(var_34, var_77, var_26);
            if (!var_34) {
                // elif jnttype == 1:  # BALL                                                     <L 385>
                var_81 = (var_31 == var_80);
                if (var_81) {
                    // Dcdof_dot_out[worldid, dof_i + 0, dofid] = math.motion_cross(cvel_k, cdof[dof_i + 0])       <L 386>
                    var_83 = wp::add(var_79, var_82);
                    var_84 = wp::address(var_20, var_83);
                    var_86 = wp::load(var_84);
                    var_85 = motion_cross_0(var_78, var_86);
                    var_88 = wp::add(var_79, var_87);
                    wp::array_store(var_Dcdof_dot_out, var_0, var_88, var_2, var_85);
                    // Dcdof_dot_out[worldid, dof_i + 1, dofid] = math.motion_cross(cvel_k, cdof[dof_i + 1])       <L 387>
                    var_90 = wp::add(var_79, var_89);
                    var_91 = wp::address(var_20, var_90);
                    var_93 = wp::load(var_91);
                    var_92 = motion_cross_0(var_78, var_93);
                    var_95 = wp::add(var_79, var_94);
                    wp::array_store(var_Dcdof_dot_out, var_0, var_95, var_2, var_92);
                    // Dcdof_dot_out[worldid, dof_i + 2, dofid] = math.motion_cross(cvel_k, cdof[dof_i + 2])       <L 388>
                    var_97 = wp::add(var_79, var_96);
                    var_98 = wp::address(var_20, var_97);
                    var_100 = wp::load(var_98);
                    var_99 = motion_cross_0(var_78, var_100);
                    var_102 = wp::add(var_79, var_101);
                    wp::array_store(var_Dcdof_dot_out, var_0, var_102, var_2, var_99);
                    // if dofid >= dof_i and dofid < dof_i + 3:                                   <L 390>
                    var_104 = (var_2 >= var_79);
                    var_103 = var_104;
                    if (var_103) {
                        var_106 = wp::add(var_79, var_105);
                        var_107 = (var_2 < var_106);
                        var_103 = var_103 && var_107;
                    }
                    if (var_103) {
                        // cvel_k += cdof[dofid]                                                  <L 391>
                        var_108 = wp::address(var_20, var_2);
                        var_110 = wp::load(var_108);
                        var_109 = wp::add(var_78, var_110);
                    }
                    var_111 = wp::where(var_103, var_109, var_78);
                    // dof_i += 3                                                                 <L 393>
                    var_113 = wp::add(var_79, var_112);
                }
                var_114 = wp::where(var_81, var_111, var_78);
                var_115 = wp::where(var_81, var_113, var_79);
                if (!var_81) {
                    // Dcdof_dot_out[worldid, dof_i, dofid] = math.motion_cross(cvel_k, cdof[dof_i])       <L 395>
                    var_116 = wp::address(var_20, var_115);
                    var_118 = wp::load(var_116);
                    var_117 = motion_cross_0(var_114, var_118);
                    wp::array_store(var_Dcdof_dot_out, var_0, var_115, var_2, var_117);
                    // if dofid == dof_i:                                                         <L 397>
                    var_119 = (var_2 == var_115);
                    if (var_119) {
                        // cvel_k += cdof[dof_i]                                                  <L 398>
                        var_120 = wp::address(var_20, var_115);
                        var_122 = wp::load(var_120);
                        var_121 = wp::add(var_114, var_122);
                    }
                    var_123 = wp::where(var_119, var_121, var_114);
                    // dof_i += 1                                                                 <L 400>
                    var_125 = wp::add(var_115, var_124);
                }
                var_126 = wp::where(var_81, var_114, var_123);
                var_127 = wp::where(var_81, var_115, var_125);
            }
            var_128 = wp::where(var_34, var_78, var_126);
            var_129 = wp::where(var_34, var_79, var_127);
            wp::assign(var_22, var_128);
            wp::assign(var_26, var_129);
            goto start_for_1;
        end_for_1:;
        // Dcvel_out[worldid, bodyid, dofid] = cvel_k                                             <L 402>
        wp::array_store(var_Dcvel_out, var_0, var_4, var_2, var_22);
    }
}



extern "C" __global__ void _qderiv_actuator_passive_actuation_sparse_9edf10e9_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_M_elemid,
    wp::array_t<wp::int32> var_moment_rownnz_in,
    wp::array_t<wp::int32> var_moment_rowadr_in,
    wp::array_t<wp::int32> var_moment_colind_in,
    wp::array_t<wp::float32> var_actuator_moment_in,
    wp::array_t<wp::float32> var_vel_in,
    wp::array_t<wp::float32> var_qDeriv_out)
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
        const wp::float32 var_5 = 0.0;
        bool var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::range_t var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::float32* var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        const wp::float32 var_19 = 0.0;
        bool var_20;
        wp::int32* var_21;
        wp::int32 var_22;
        wp::int32 var_23;
        const wp::int32 var_24 = 1;
        wp::int32 var_25;
        wp::range_t var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::float32* var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        const wp::float32 var_32 = 0.0;
        bool var_33;
        wp::int32* var_34;
        wp::int32 var_35;
        wp::int32 var_36;
        wp::int32* var_37;
        wp::int32 var_38;
        wp::int32 var_39;
        const wp::int32 var_40 = 0;
        bool var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::slice_t var_44;
        const wp::int32 var_45 = 0;
        wp::array_t<wp::float32> var_46;
        wp::float32 var_47;
        //---------
        // forward
        // def _qderiv_actuator_passive_actuation_sparse(                                         <L 178>
        // worldid, actid = wp.tid()                                                              <L 191>
        builtin_tid2d(var_0, var_1);
        // vel = vel_in[worldid, actid]                                                           <L 193>
        var_2 = wp::address(var_vel_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if vel == 0.0:                                                                         <L 194>
        var_6 = (var_3 == var_5);
        if (var_6) {
            // return                                                                             <L 195>
            continue;
        }
        // rownnz = moment_rownnz_in[worldid, actid]                                              <L 197>
        var_7 = wp::address(var_moment_rownnz_in, var_0, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // rowadr = moment_rowadr_in[worldid, actid]                                              <L 198>
        var_10 = wp::address(var_moment_rowadr_in, var_0, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // for i in range(rownnz):                                                                <L 200>
        var_13 = wp::range(var_8);
        start_for_1:;
            if (iter_cmp(var_13) == 0) goto end_for_1;
            var_14 = wp::iter_next(var_13);
            // rowadri = rowadr + i                                                               <L 201>
            var_15 = wp::add(var_11, var_14);
            // moment_i = actuator_moment_in[worldid, rowadri]                                    <L 202>
            var_16 = wp::address(var_actuator_moment_in, var_0, var_15);
            var_18 = wp::load(var_16);
            var_17 = wp::copy(var_18);
            // if moment_i == 0.0:                                                                <L 203>
            var_20 = (var_17 == var_19);
            if (var_20) {
                // continue                                                                       <L 204>
                goto start_for_1;
            }
            // dofi = moment_colind_in[worldid, rowadri]                                          <L 205>
            var_21 = wp::address(var_moment_colind_in, var_0, var_15);
            var_23 = wp::load(var_21);
            var_22 = wp::copy(var_23);
            // for j in range(i + 1):                                                             <L 207>
            var_25 = wp::add(var_14, var_24);
            var_26 = wp::range(var_25);
            start_for_3:;
                if (iter_cmp(var_26) == 0) goto end_for_3;
                var_27 = wp::iter_next(var_26);
                // rowadrj = rowadr + j                                                           <L 208>
                var_28 = wp::add(var_11, var_27);
                // moment_j = actuator_moment_in[worldid, rowadrj]                                <L 209>
                var_29 = wp::address(var_actuator_moment_in, var_0, var_28);
                var_31 = wp::load(var_29);
                var_30 = wp::copy(var_31);
                // if moment_j == 0.0:                                                            <L 210>
                var_33 = (var_30 == var_32);
                if (var_33) {
                    // continue                                                                   <L 211>
                    goto start_for_3;
                }
                // dofj = moment_colind_in[worldid, rowadrj]                                      <L 212>
                var_34 = wp::address(var_moment_colind_in, var_0, var_28);
                var_36 = wp::load(var_34);
                var_35 = wp::copy(var_36);
                // elemid = M_elemid[dofi, dofj]                                                  <L 214>
                var_37 = wp::address(var_M_elemid, var_22, var_35);
                var_39 = wp::load(var_37);
                var_38 = wp::copy(var_39);
                // if elemid >= 0:                                                                <L 215>
                var_41 = (var_38 >= var_40);
                if (var_41) {
                    // contrib = moment_i * moment_j * vel                                        <L 216>
                    var_42 = wp::mul(var_17, var_30);
                    var_43 = wp::mul(var_42, var_3);
                    // wp.atomic_add(qDeriv_out[worldid], elemid, contrib)                        <L 217>
                    var_44 = wp::slice_t(var_0, var_0, var_45);
                    var_46 = wp::view(var_qDeriv_out, var_44);
                    var_47 = wp::atomic_add(var_46, var_38, var_43);
                }
                goto start_for_3;
            end_for_3:;
            goto start_for_1;
        end_for_1:;
    }
}



extern "C" __global__ void _qderiv_ellipsoid_fluid_a8421048_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::array_t<wp::vec_t<3, wp::float32>> var_opt_wind,
    wp::array_t<wp::float32> var_opt_density,
    wp::array_t<wp::float32> var_opt_viscosity,
    wp::int32 var_opt_integrator,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_body_geomnum,
    wp::array_t<wp::int32> var_body_geomadr,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::int32> var_geom_type,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_size,
    wp::array_t<wp::float32> var_geom_fluid,
    wp::array_t<wp::int32> var_body_fluid_ellipsoid_adr,
    wp::array_t<wp::int32> var_body_isdofancestor,
    wp::array_t<wp::int32> var_M_elemid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::array_t<wp::int32> var_Mi,
    wp::array_t<wp::int32> var_Mj,
    wp::array_t<wp::float32> var_qDeriv_out)
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
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        const wp::int32 var_15 = 0;
        bool var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        const wp::int32 var_20 = 0;
        bool var_21;
        wp::int32* var_22;
        const wp::int32 var_23 = 0;
        bool var_24;
        wp::int32 var_25;
        wp::shape_t* var_26;
        const wp::int32 var_27 = 0;
        wp::int32 var_28;
        wp::shape_t var_29;
        wp::int32 var_30;
        wp::vec_t<3, wp::float32>* var_31;
        wp::vec_t<3, wp::float32> var_32;
        wp::vec_t<3, wp::float32> var_33;
        wp::shape_t* var_34;
        const wp::int32 var_35 = 0;
        wp::int32 var_36;
        wp::shape_t var_37;
        wp::int32 var_38;
        wp::float32* var_39;
        wp::float32 var_40;
        wp::float32 var_41;
        wp::shape_t* var_42;
        const wp::int32 var_43 = 0;
        wp::int32 var_44;
        wp::shape_t var_45;
        wp::int32 var_46;
        wp::float32* var_47;
        wp::float32 var_48;
        wp::float32 var_49;
        wp::shape_t* var_50;
        const wp::int32 var_51 = 0;
        wp::int32 var_52;
        wp::shape_t var_53;
        wp::int32 var_54;
        wp::float32* var_55;
        wp::float32 var_56;
        wp::float32 var_57;
        bool var_58;
        const wp::float32 var_59 = 0.0;
        bool var_60;
        const wp::float32 var_61 = 0.0;
        bool var_62;
        wp::vec_t<6, wp::float32>* var_63;
        wp::vec_t<6, wp::float32> var_64;
        wp::vec_t<6, wp::float32> var_65;
        wp::vec_t<6, wp::float32>* var_66;
        wp::vec_t<6, wp::float32> var_67;
        wp::vec_t<6, wp::float32> var_68;
        wp::int32* var_69;
        wp::int32* var_70;
        wp::int32* var_71;
        wp::float32 var_72;
        wp::int32 var_73;
        wp::int32 var_74;
        wp::int32 var_75;
        wp::float32 var_76;
        const wp::float32 var_77 = 0.0;
        bool var_78;
        wp::slice_t var_79;
        const wp::int32 var_80 = 0;
        wp::array_t<wp::float32> var_81;
        wp::float32 var_82;
        wp::float32 var_83;
        //---------
        // forward
        // def _qderiv_ellipsoid_fluid(                                                           <L 829>
        // worldid, fluid_idx, elemid = wp.tid()                                                  <L 866>
        builtin_tid3d(var_0, var_1, var_2);
        // bodyid = body_fluid_ellipsoid_adr[fluid_idx]                                           <L 868>
        var_3 = wp::address(var_body_fluid_ellipsoid_adr, var_1);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // dofiid = Mi[elemid]                                                                    <L 870>
        var_6 = wp::address(var_Mi, var_2);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // dofjid = Mj[elemid]                                                                    <L 871>
        var_9 = wp::address(var_Mj, var_2);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // madr = M_elemid[dofiid, dofjid]                                                        <L 873>
        var_12 = wp::address(var_M_elemid, var_7, var_10);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // if madr < 0:                                                                           <L 874>
        var_16 = (var_13 < var_15);
        if (var_16) {
            // return                                                                             <L 875>
            continue;
        }
        // bodyid_i = dof_bodyid[dofiid]                                                          <L 879>
        var_17 = wp::address(var_dof_bodyid, var_7);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // if bodyid_i == 0:                                                                      <L 881>
        var_21 = (var_18 == var_20);
        if (var_21) {
            // return                                                                             <L 882>
            continue;
        }
        // if body_isdofancestor[bodyid, dofiid] == 0:                                            <L 884>
        var_22 = wp::address(var_body_isdofancestor, var_4, var_7);
        var_25 = wp::load(var_22);
        var_24 = (var_25 == var_23);
        if (var_24) {
            // return                                                                             <L 885>
            continue;
        }
        // wind = opt_wind[worldid % opt_wind.shape[0]]                                           <L 887>
        var_26 = &(var_opt_wind.shape);
        var_29 = wp::load(var_26);
        var_28 = wp::extract(var_29, var_27);
        var_30 = wp::mod(var_0, var_28);
        var_31 = wp::address(var_opt_wind, var_30);
        var_33 = wp::load(var_31);
        var_32 = wp::copy(var_33);
        // density = opt_density[worldid % opt_density.shape[0]]                                  <L 888>
        var_34 = &(var_opt_density.shape);
        var_37 = wp::load(var_34);
        var_36 = wp::extract(var_37, var_35);
        var_38 = wp::mod(var_0, var_36);
        var_39 = wp::address(var_opt_density, var_38);
        var_41 = wp::load(var_39);
        var_40 = wp::copy(var_41);
        // viscosity = opt_viscosity[worldid % opt_viscosity.shape[0]]                            <L 889>
        var_42 = &(var_opt_viscosity.shape);
        var_45 = wp::load(var_42);
        var_44 = wp::extract(var_45, var_43);
        var_46 = wp::mod(var_0, var_44);
        var_47 = wp::address(var_opt_viscosity, var_46);
        var_49 = wp::load(var_47);
        var_48 = wp::copy(var_49);
        // timestep = opt_timestep[worldid % opt_timestep.shape[0]]                               <L 890>
        var_50 = &(var_opt_timestep.shape);
        var_53 = wp::load(var_50);
        var_52 = wp::extract(var_53, var_51);
        var_54 = wp::mod(var_0, var_52);
        var_55 = wp::address(var_opt_timestep, var_54);
        var_57 = wp::load(var_55);
        var_56 = wp::copy(var_57);
        // if density <= 0.0 and viscosity <= 0.0:                                                <L 892>
        var_60 = (var_40 <= var_59);
        var_58 = var_60;
        if (var_58) {
            var_62 = (var_48 <= var_61);
            var_58 = var_58 && var_62;
        }
        if (var_58) {
            // return                                                                             <L 893>
            continue;
        }
        // cdof_i = cdof_in[worldid, dofiid]                                                      <L 895>
        var_63 = wp::address(var_cdof_in, var_0, var_7);
        var_65 = wp::load(var_63);
        var_64 = wp::copy(var_65);
        // cdof_j = cdof_in[worldid, dofjid]                                                      <L 896>
        var_66 = wp::address(var_cdof_in, var_0, var_10);
        var_68 = wp::load(var_66);
        var_67 = wp::copy(var_68);
        // contrib = _deriv_ellipsoid_fluid(                                                      <L 898>
        // opt_integrator,                                                                        <L 899>
        // geom_type,                                                                             <L 900>
        // geom_size,                                                                             <L 901>
        // geom_fluid,                                                                            <L 902>
        // xipos_in,                                                                              <L 903>
        // geom_xpos_in,                                                                          <L 904>
        // geom_xmat_in,                                                                          <L 905>
        // subtree_com_in,                                                                        <L 906>
        // cvel_in,                                                                               <L 907>
        // worldid,                                                                               <L 908>
        // bodyid,                                                                                <L 909>
        // body_rootid[bodyid],                                                                   <L 910>
        var_69 = wp::address(var_body_rootid, var_4);
        // body_geomadr[bodyid],                                                                  <L 911>
        var_70 = wp::address(var_body_geomadr, var_4);
        // body_geomnum[bodyid],                                                                  <L 912>
        var_71 = wp::address(var_body_geomnum, var_4);
        // cdof_i,                                                                                <L 913>
        // cdof_j,                                                                                <L 914>
        // wind,                                                                                  <L 915>
        // density,                                                                               <L 916>
        // viscosity,                                                                             <L 917>
        var_73 = wp::load(var_69);
        var_74 = wp::load(var_70);
        var_75 = wp::load(var_71);
        var_72 = _deriv_ellipsoid_fluid_0(var_opt_integrator, var_geom_type, var_geom_size, var_geom_fluid, var_xipos_in, var_geom_xpos_in, var_geom_xmat_in, var_subtree_com_in, var_cvel_in, var_0, var_4, var_73, var_74, var_75, var_64, var_67, var_32, var_40, var_48);
        // contrib *= timestep                                                                    <L 920>
        var_76 = wp::mul(var_72, var_56);
        // if contrib != 0.0:                                                                     <L 922>
        var_78 = (var_76 != var_77);
        if (var_78) {
            // wp.atomic_add(qDeriv_out[worldid], madr, -contrib)                                 <L 923>
            var_79 = wp::slice_t(var_0, var_0, var_80);
            var_81 = wp::view(var_qDeriv_out, var_79);
            var_82 = wp::neg(var_76);
            var_83 = wp::atomic_add(var_81, var_13, var_82);
        }
    }
}



extern "C" __global__ void _qderiv_box_fluid_ab6c2235_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::array_t<wp::vec_t<3, wp::float32>> var_opt_wind,
    wp::array_t<wp::float32> var_opt_density,
    wp::array_t<wp::float32> var_opt_viscosity,
    wp::int32 var_opt_integrator,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::float32> var_body_mass,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_inertia,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::int32> var_body_fluid_box_adr,
    wp::array_t<wp::int32> var_body_isdofancestor,
    wp::array_t<wp::int32> var_M_elemid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_ximat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::array_t<wp::int32> var_Mi,
    wp::array_t<wp::int32> var_Mj,
    wp::array_t<wp::float32> var_qDeriv_out)
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
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        const wp::int32 var_15 = 0;
        bool var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        const wp::int32 var_20 = 0;
        bool var_21;
        wp::int32* var_22;
        const wp::int32 var_23 = 0;
        bool var_24;
        wp::int32 var_25;
        wp::shape_t* var_26;
        const wp::int32 var_27 = 0;
        wp::int32 var_28;
        wp::shape_t var_29;
        wp::int32 var_30;
        wp::vec_t<3, wp::float32>* var_31;
        wp::vec_t<3, wp::float32> var_32;
        wp::vec_t<3, wp::float32> var_33;
        wp::shape_t* var_34;
        const wp::int32 var_35 = 0;
        wp::int32 var_36;
        wp::shape_t var_37;
        wp::int32 var_38;
        wp::float32* var_39;
        wp::float32 var_40;
        wp::float32 var_41;
        wp::shape_t* var_42;
        const wp::int32 var_43 = 0;
        wp::int32 var_44;
        wp::shape_t var_45;
        wp::int32 var_46;
        wp::float32* var_47;
        wp::float32 var_48;
        wp::float32 var_49;
        wp::shape_t* var_50;
        const wp::int32 var_51 = 0;
        wp::int32 var_52;
        wp::shape_t var_53;
        wp::int32 var_54;
        wp::float32* var_55;
        wp::float32 var_56;
        wp::float32 var_57;
        bool var_58;
        const wp::float32 var_59 = 0.0;
        bool var_60;
        const wp::float32 var_61 = 0.0;
        bool var_62;
        wp::vec_t<3, wp::float32>* var_63;
        wp::vec_t<3, wp::float32> var_64;
        wp::vec_t<3, wp::float32> var_65;
        wp::mat_t<3, 3, wp::float32>* var_66;
        wp::mat_t<3, 3, wp::float32> var_67;
        wp::mat_t<3, 3, wp::float32> var_68;
        wp::int32* var_69;
        wp::vec_t<3, wp::float32>* var_70;
        wp::int32 var_71;
        wp::vec_t<3, wp::float32> var_72;
        wp::vec_t<3, wp::float32> var_73;
        wp::vec_t<6, wp::float32>* var_74;
        wp::vec_t<6, wp::float32> var_75;
        wp::vec_t<6, wp::float32> var_76;
        const wp::int32 var_77 = 0;
        wp::float32 var_78;
        const wp::int32 var_79 = 1;
        wp::float32 var_80;
        const wp::int32 var_81 = 2;
        wp::float32 var_82;
        wp::vec_t<3, wp::float32> var_83;
        const wp::int32 var_84 = 3;
        wp::float32 var_85;
        const wp::int32 var_86 = 4;
        wp::float32 var_87;
        const wp::int32 var_88 = 5;
        wp::float32 var_89;
        wp::vec_t<3, wp::float32> var_90;
        wp::vec_t<3, wp::float32> var_91;
        wp::vec_t<3, wp::float32> var_92;
        wp::vec_t<3, wp::float32> var_93;
        wp::mat_t<3, 3, wp::float32> var_94;
        wp::vec_t<3, wp::float32> var_95;
        wp::vec_t<3, wp::float32> var_96;
        wp::vec_t<3, wp::float32> var_97;
        wp::vec_t<3, wp::float32> var_98;
        wp::vec_t<6, wp::float32> var_99;
        wp::mat_t<6, 6, wp::float32> var_100;
        wp::vec_t<6, wp::float32> var_101;
        wp::vec_t<6, wp::float32> var_102;
        wp::vec_t<6, wp::float32> var_103;
        wp::float32 var_104;
        wp::float32 var_105;
        const wp::float32 var_106 = 0.0;
        bool var_107;
        wp::slice_t var_108;
        const wp::int32 var_109 = 0;
        wp::array_t<wp::float32> var_110;
        wp::float32 var_111;
        wp::float32 var_112;
        //---------
        // forward
        // def _qderiv_box_fluid(                                                                 <L 1020>
        // worldid, fluid_idx, elemid = wp.tid()                                                  <L 1047>
        builtin_tid3d(var_0, var_1, var_2);
        // bodyid = body_fluid_box_adr[fluid_idx]                                                 <L 1049>
        var_3 = wp::address(var_body_fluid_box_adr, var_1);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // dofiid = Mi[elemid]                                                                    <L 1051>
        var_6 = wp::address(var_Mi, var_2);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // dofjid = Mj[elemid]                                                                    <L 1052>
        var_9 = wp::address(var_Mj, var_2);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // madr = M_elemid[dofiid, dofjid]                                                        <L 1054>
        var_12 = wp::address(var_M_elemid, var_7, var_10);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // if madr < 0:                                                                           <L 1055>
        var_16 = (var_13 < var_15);
        if (var_16) {
            // return                                                                             <L 1056>
            continue;
        }
        // bodyid_i = dof_bodyid[dofiid]                                                          <L 1058>
        var_17 = wp::address(var_dof_bodyid, var_7);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // if bodyid_i == 0:                                                                      <L 1060>
        var_21 = (var_18 == var_20);
        if (var_21) {
            // return                                                                             <L 1061>
            continue;
        }
        // if body_isdofancestor[bodyid, dofiid] == 0:                                            <L 1063>
        var_22 = wp::address(var_body_isdofancestor, var_4, var_7);
        var_25 = wp::load(var_22);
        var_24 = (var_25 == var_23);
        if (var_24) {
            // return                                                                             <L 1064>
            continue;
        }
        // wind = opt_wind[worldid % opt_wind.shape[0]]                                           <L 1066>
        var_26 = &(var_opt_wind.shape);
        var_29 = wp::load(var_26);
        var_28 = wp::extract(var_29, var_27);
        var_30 = wp::mod(var_0, var_28);
        var_31 = wp::address(var_opt_wind, var_30);
        var_33 = wp::load(var_31);
        var_32 = wp::copy(var_33);
        // density = opt_density[worldid % opt_density.shape[0]]                                  <L 1067>
        var_34 = &(var_opt_density.shape);
        var_37 = wp::load(var_34);
        var_36 = wp::extract(var_37, var_35);
        var_38 = wp::mod(var_0, var_36);
        var_39 = wp::address(var_opt_density, var_38);
        var_41 = wp::load(var_39);
        var_40 = wp::copy(var_41);
        // viscosity = opt_viscosity[worldid % opt_viscosity.shape[0]]                            <L 1068>
        var_42 = &(var_opt_viscosity.shape);
        var_45 = wp::load(var_42);
        var_44 = wp::extract(var_45, var_43);
        var_46 = wp::mod(var_0, var_44);
        var_47 = wp::address(var_opt_viscosity, var_46);
        var_49 = wp::load(var_47);
        var_48 = wp::copy(var_49);
        // timestep = opt_timestep[worldid % opt_timestep.shape[0]]                               <L 1069>
        var_50 = &(var_opt_timestep.shape);
        var_53 = wp::load(var_50);
        var_52 = wp::extract(var_53, var_51);
        var_54 = wp::mod(var_0, var_52);
        var_55 = wp::address(var_opt_timestep, var_54);
        var_57 = wp::load(var_55);
        var_56 = wp::copy(var_57);
        // if density <= 0.0 and viscosity <= 0.0:                                                <L 1071>
        var_60 = (var_40 <= var_59);
        var_58 = var_60;
        if (var_58) {
            var_62 = (var_48 <= var_61);
            var_58 = var_58 && var_62;
        }
        if (var_58) {
            // return                                                                             <L 1072>
            continue;
        }
        // b_ipos = xipos_in[worldid, bodyid]                                                     <L 1075>
        var_63 = wp::address(var_xipos_in, var_0, var_4);
        var_65 = wp::load(var_63);
        var_64 = wp::copy(var_65);
        // b_imat = ximat_in[worldid, bodyid]                                                     <L 1076>
        var_66 = wp::address(var_ximat_in, var_0, var_4);
        var_68 = wp::load(var_66);
        var_67 = wp::copy(var_68);
        // subtree_root = subtree_com_in[worldid, body_rootid[bodyid]]                            <L 1077>
        var_69 = wp::address(var_body_rootid, var_4);
        var_71 = wp::load(var_69);
        var_70 = wp::address(var_subtree_com_in, var_0, var_71);
        var_73 = wp::load(var_70);
        var_72 = wp::copy(var_73);
        // vel_subtree = cvel_in[worldid, bodyid]                                                 <L 1079>
        var_74 = wp::address(var_cvel_in, var_0, var_4);
        var_76 = wp::load(var_74);
        var_75 = wp::copy(var_76);
        // v_subtree_ang = wp.vec3(vel_subtree[0], vel_subtree[1], vel_subtree[2])                <L 1080>
        var_78 = wp::extract(var_75, var_77);
        var_80 = wp::extract(var_75, var_79);
        var_82 = wp::extract(var_75, var_81);
        var_83 = wp::vec_t<3, wp::float32>(var_78, var_80, var_82);
        // v_subtree_lin = wp.vec3(vel_subtree[3], vel_subtree[4], vel_subtree[5])                <L 1081>
        var_85 = wp::extract(var_75, var_84);
        var_87 = wp::extract(var_75, var_86);
        var_89 = wp::extract(var_75, var_88);
        var_90 = wp::vec_t<3, wp::float32>(var_85, var_87, var_89);
        // lin_com = v_subtree_lin - wp.cross(b_ipos - subtree_root, v_subtree_ang)               <L 1083>
        var_91 = wp::sub(var_64, var_72);
        var_92 = wp::cross(var_91, var_83);
        var_93 = wp::sub(var_90, var_92);
        // b_imat_T = wp.transpose(b_imat)                                                        <L 1084>
        var_94 = wp::transpose(var_67);
        // v_local_ang = b_imat_T @ v_subtree_ang                                                 <L 1085>
        var_95 = wp::mul(var_94, var_83);
        // v_local_lin = b_imat_T @ lin_com                                                       <L 1086>
        var_96 = wp::mul(var_94, var_93);
        // wind_local = b_imat_T @ wind                                                           <L 1087>
        var_97 = wp::mul(var_94, var_32);
        // lvel = wp.spatial_vector(v_local_ang, v_local_lin - wind_local)                        <L 1089>
        var_98 = wp::sub(var_96, var_97);
        var_99 = wp::vec_t<6, wp::float32>(var_95, var_98);
        // B = _deriv_box_fluid(                                                                  <L 1091>
        // opt_integrator,                                                                        <L 1092>
        // body_mass,                                                                             <L 1093>
        // body_inertia,                                                                          <L 1094>
        // worldid,                                                                               <L 1095>
        // bodyid,                                                                                <L 1096>
        // lvel,                                                                                  <L 1097>
        // density,                                                                               <L 1098>
        // viscosity,                                                                             <L 1099>
        var_100 = _deriv_box_fluid_0(var_opt_integrator, var_body_mass, var_body_inertia, var_0, var_4, var_99, var_40, var_48);
        // J_i = _get_jac_column_local(                                                           <L 1103>
        // body_parentid, body_rootid, dof_bodyid, subtree_com_in, cdof_in, b_ipos, bodyid, dofiid, worldid, b_imat       <L 1104>
        var_101 = _get_jac_column_local_0(var_body_parentid, var_body_rootid, var_dof_bodyid, var_subtree_com_in, var_cdof_in, var_64, var_4, var_7, var_0, var_67);
        // J_j = _get_jac_column_local(                                                           <L 1106>
        // body_parentid, body_rootid, dof_bodyid, subtree_com_in, cdof_in, b_ipos, bodyid, dofjid, worldid, b_imat       <L 1107>
        var_102 = _get_jac_column_local_0(var_body_parentid, var_body_rootid, var_dof_bodyid, var_subtree_com_in, var_cdof_in, var_64, var_4, var_10, var_0, var_67);
        // contrib = wp.dot(J_i, B @ J_j) * timestep                                              <L 1110>
        var_103 = wp::mul(var_100, var_102);
        var_104 = wp::dot(var_101, var_103);
        var_105 = wp::mul(var_104, var_56);
        // if contrib != 0.0:                                                                     <L 1112>
        var_107 = (var_105 != var_106);
        if (var_107) {
            // wp.atomic_add(qDeriv_out[worldid], madr, -contrib)                                 <L 1113>
            var_108 = wp::slice_t(var_0, var_0, var_109);
            var_110 = wp::view(var_qDeriv_out, var_108);
            var_111 = wp::neg(var_105);
            var_112 = wp::atomic_add(var_110, var_13, var_111);
        }
    }
}



extern "C" __global__ void deriv_rne_body2jnt_sparse_485c0d3c_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::float32> var_timestep,
    wp::array_t<wp::int32> var_Di,
    wp::array_t<wp::int32> var_Dj,
    wp::array_t<wp::vec_t<6, wp::float32>> var_Dcfrcbody_in,
    bool var_flg_subtract,
    wp::array_t<wp::float32> var_qDeriv_out)
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
        wp::vec_t<6, wp::float32>* var_19;
        wp::vec_t<6, wp::float32> var_20;
        wp::vec_t<6, wp::float32> var_21;
        wp::vec_t<6, wp::float32>* var_22;
        wp::float32 var_23;
        wp::vec_t<6, wp::float32> var_24;
        wp::slice_t var_25;
        const wp::int32 var_26 = 0;
        wp::array_t<wp::float32> var_27;
        wp::float32 var_28;
        wp::float32 var_29;
        wp::slice_t var_30;
        const wp::int32 var_31 = 0;
        wp::array_t<wp::float32> var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        //---------
        // forward
        // def deriv_rne_body2jnt_sparse(                                                         <L 483>
        // worldid, elemid = wp.tid()                                                             <L 498>
        builtin_tid2d(var_0, var_1);
        // dt = timestep[worldid % timestep.shape[0]]                                             <L 499>
        var_2 = &(var_timestep.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        var_7 = wp::address(var_timestep, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // i = Di[elemid]                                                                         <L 501>
        var_10 = wp::address(var_Di, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // j = Dj[elemid]                                                                         <L 502>
        var_13 = wp::address(var_Dj, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // body_i = dof_bodyid[i]                                                                 <L 504>
        var_16 = wp::address(var_dof_bodyid, var_11);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // dcfrc = Dcfrcbody_in[worldid, body_i, j]                                               <L 505>
        var_19 = wp::address(var_Dcfrcbody_in, var_0, var_17, var_14);
        var_21 = wp::load(var_19);
        var_20 = wp::copy(var_21);
        // term = wp.dot(cdof_in[worldid, i], dcfrc)                                              <L 506>
        var_22 = wp::address(var_cdof_in, var_0, var_11);
        var_24 = wp::load(var_22);
        var_23 = wp::dot(var_24, var_20);
        // if flg_subtract:                                                                       <L 508>
        if (var_flg_subtract) {
            // wp.atomic_sub(qDeriv_out[worldid], elemid, dt * term)                              <L 509>
            var_25 = wp::slice_t(var_0, var_0, var_26);
            var_27 = wp::view(var_qDeriv_out, var_25);
            var_28 = wp::mul(var_8, var_23);
            var_29 = wp::atomic_sub(var_27, var_1, var_28);
        }
        if (!var_flg_subtract) {
            // wp.atomic_add(qDeriv_out[worldid], elemid, dt * term)                              <L 511>
            var_30 = wp::slice_t(var_0, var_0, var_31);
            var_32 = wp::view(var_qDeriv_out, var_30);
            var_33 = wp::mul(var_8, var_23);
            var_34 = wp::atomic_add(var_32, var_1, var_33);
        }
    }
}



extern "C" __global__ void _qderiv_actuator_passive_6cbfee2a_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::int32 var_opt_disableflags,
    wp::array_t<wp::float32> var_dof_damping,
    wp::array_t<wp::vec_t<2, wp::float32>> var_dof_dampingpoly,
    wp::array_t<wp::int32> var_M_elemid,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::float32> var_M_in,
    wp::array_t<wp::int32> var_Mi,
    wp::array_t<wp::int32> var_Mj,
    wp::array_t<wp::float32> var_qDeriv_in,
    wp::array_t<wp::float32> var_qDeriv_out)
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
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        const wp::int32 var_11 = 0;
        bool var_12;
        wp::float32* var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        bool var_16;
        const wp::int32 var_17 = 64;
        wp::int32 var_18;
        bool var_19;
        bool var_20;
        wp::shape_t* var_21;
        const wp::int32 var_22 = 0;
        wp::int32 var_23;
        wp::shape_t var_24;
        wp::int32 var_25;
        wp::float32* var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        wp::shape_t* var_29;
        const wp::int32 var_30 = 0;
        wp::int32 var_31;
        wp::shape_t var_32;
        wp::int32 var_33;
        wp::vec_t<2, wp::float32>* var_34;
        wp::vec_t<2, wp::float32> var_35;
        wp::vec_t<2, wp::float32> var_36;
        wp::float32* var_37;
        wp::float32 var_38;
        wp::float32 var_39;
        const wp::int32 var_40 = 1;
        wp::float32 var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::shape_t* var_44;
        const wp::int32 var_45 = 0;
        wp::int32 var_46;
        wp::shape_t var_47;
        wp::int32 var_48;
        wp::float32* var_49;
        wp::float32 var_50;
        wp::float32 var_51;
        wp::float32* var_52;
        wp::float32 var_53;
        wp::float32 var_54;
        //---------
        // forward
        // def _qderiv_actuator_passive(                                                          <L 221>
        // worldid, elemid = wp.tid()                                                             <L 238>
        builtin_tid2d(var_0, var_1);
        // dofiid = Mi[elemid]                                                                    <L 240>
        var_2 = wp::address(var_Mi, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // dofjid = Mj[elemid]                                                                    <L 241>
        var_5 = wp::address(var_Mj, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // madr = M_elemid[dofiid, dofjid]                                                        <L 244>
        var_8 = wp::address(var_M_elemid, var_3, var_6);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // if madr < 0:                                                                           <L 245>
        var_12 = (var_9 < var_11);
        if (var_12) {
            // return                                                                             <L 246>
            continue;
        }
        // qderiv = qDeriv_in[worldid, madr]                                                      <L 248>
        var_13 = wp::address(var_qDeriv_in, var_0, var_9);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // if not (opt_disableflags & DisableBit.DAMPER) and dofiid == dofjid:                    <L 250>
        var_18 = wp::bit_and(var_opt_disableflags, var_17);
        var_19 = wp::unot(var_18);
        var_16 = var_19;
        if (var_16) {
            var_20 = (var_3 == var_6);
            var_16 = var_16 && var_20;
        }
        if (var_16) {
            // damping = dof_damping[worldid % dof_damping.shape[0], dofiid]                      <L 251>
            var_21 = &(var_dof_damping.shape);
            var_24 = wp::load(var_21);
            var_23 = wp::extract(var_24, var_22);
            var_25 = wp::mod(var_0, var_23);
            var_26 = wp::address(var_dof_damping, var_25, var_3);
            var_28 = wp::load(var_26);
            var_27 = wp::copy(var_28);
            // dpoly = dof_dampingpoly[worldid % dof_dampingpoly.shape[0], dofiid]                <L 252>
            var_29 = &(var_dof_dampingpoly.shape);
            var_32 = wp::load(var_29);
            var_31 = wp::extract(var_32, var_30);
            var_33 = wp::mod(var_0, var_31);
            var_34 = wp::address(var_dof_dampingpoly, var_33, var_3);
            var_36 = wp::load(var_34);
            var_35 = wp::copy(var_36);
            // v = qvel_in[worldid, dofiid]                                                       <L 253>
            var_37 = wp::address(var_qvel_in, var_0, var_3);
            var_39 = wp::load(var_37);
            var_38 = wp::copy(var_39);
            // qderiv -= util_misc._poly_force_deriv(damping, dpoly, v, 1)                        <L 254>
            var_41 = _poly_force_deriv_0(var_27, var_35, var_38, var_40);
            var_42 = wp::sub(var_14, var_41);
        }
        var_43 = wp::where(var_16, var_42, var_14);
        // qderiv *= opt_timestep[worldid % opt_timestep.shape[0]]                                <L 256>
        var_44 = &(var_opt_timestep.shape);
        var_47 = wp::load(var_44);
        var_46 = wp::extract(var_47, var_45);
        var_48 = wp::mod(var_0, var_46);
        var_49 = wp::address(var_opt_timestep, var_48);
        var_51 = wp::load(var_49);
        var_50 = wp::mul(var_43, var_51);
        // qDeriv_out[worldid, madr] = M_in[worldid, madr] - qderiv                               <L 258>
        var_52 = wp::address(var_M_in, var_0, var_9);
        var_54 = wp::load(var_52);
        var_53 = wp::sub(var_54, var_50);
        wp::array_store(var_qDeriv_out, var_0, var_9, var_53);
    }
}

