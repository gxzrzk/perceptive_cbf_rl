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


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/math.py:115
static CUDA_CALLABLE wp::quat_t<wp::float32> quat_inv_0(
    wp::quat_t<wp::float32> var_quat)
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
    //---------
    // forward
    // def quat_inv(quat: wp.quat) -> wp.quat:                                                <L 116>
    // return wp.quat(quat[0], -quat[1], -quat[2], -quat[3])                                  <L 117>
    var_1 = wp::extract(var_quat, var_0);
    var_3 = wp::extract(var_quat, var_2);
    var_4 = wp::neg(var_3);
    var_6 = wp::extract(var_quat, var_5);
    var_7 = wp::neg(var_6);
    var_9 = wp::extract(var_quat, var_8);
    var_10 = wp::neg(var_9);
    var_11 = wp::quat_t<wp::float32>(var_1, var_4, var_7, var_10);
    return var_11;
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


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/render_util.py:66
static CUDA_CALLABLE wp::vec_t<3, wp::float32> compute_ray_0(
    wp::int32 var_projection,
    wp::float32 var_fovy,
    wp::vec_t<2, wp::float32> var_sensorsize,
    wp::vec_t<4, wp::float32> var_intrinsic,
    wp::int32 var_img_w,
    wp::int32 var_img_h,
    wp::int32 var_px,
    wp::int32 var_py,
    wp::float32 var_znear)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    bool var_1;
    const wp::float32 var_2 = 0.0;
    const wp::float32 var_3 = 0.0;
    const wp::float32 var_4 = -1.0;
    wp::vec_t<3, wp::float32> var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    const wp::int32 var_9 = 1;
    wp::float32 var_10;
    const wp::float32 var_11 = 0.0;
    bool var_12;
    const wp::int32 var_13 = 0;
    wp::float32 var_14;
    const wp::int32 var_15 = 1;
    wp::float32 var_16;
    const wp::int32 var_17 = 2;
    wp::float32 var_18;
    const wp::int32 var_19 = 3;
    wp::float32 var_20;
    const wp::int32 var_21 = 0;
    wp::float32 var_22;
    wp::float32 var_23;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::float32 var_26;
    bool var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    bool var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    wp::float32 var_35;
    wp::float32 var_36;
    const wp::float32 var_37 = 0.5;
    wp::float32 var_38;
    wp::float32 var_39;
    wp::float32 var_40;
    const wp::float32 var_41 = 0.5;
    wp::float32 var_42;
    wp::float32 var_43;
    wp::float32 var_44;
    const wp::float32 var_45 = 0.5;
    wp::float32 var_46;
    wp::float32 var_47;
    wp::float32 var_48;
    wp::float32 var_49;
    const wp::float32 var_50 = 0.5;
    wp::float32 var_51;
    wp::float32 var_52;
    wp::float32 var_53;
    wp::float32 var_54;
    const wp::float32 var_55 = 0.017453292519943295;
    wp::float32 var_56;
    const wp::float32 var_57 = 0.5;
    wp::float32 var_58;
    wp::float32 var_59;
    wp::float32 var_60;
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
    const wp::float32 var_71 = 0.5;
    wp::float32 var_72;
    wp::float32 var_73;
    wp::float32 var_74;
    wp::float32 var_75;
    const wp::float32 var_76 = 0.5;
    wp::float32 var_77;
    wp::float32 var_78;
    wp::float32 var_79;
    wp::float32 var_80;
    wp::float32 var_81;
    wp::float32 var_82;
    wp::float32 var_83;
    wp::float32 var_84;
    wp::float32 var_85;
    wp::float32 var_86;
    wp::vec_t<3, wp::float32> var_87;
    wp::vec_t<3, wp::float32> var_88;
    //---------
    // forward
    // def compute_ray(                                                                       <L 67>
    // if projection == ProjectionType.ORTHOGRAPHIC:                                          <L 84>
    var_1 = (var_projection == var_0);
    if (var_1) {
        // return wp.vec3(0.0, 0.0, -1.0)                                                     <L 85>
        var_5 = wp::vec_t<3, wp::float32>(var_2, var_3, var_4);
        return var_5;
    }
    // aspect = float(img_w) / float(img_h)                                                   <L 87>
    var_6 = wp::float(var_img_w);
    var_7 = wp::float(var_img_h);
    var_8 = wp::div(var_6, var_7);
    // sensor_h = sensorsize[1]                                                               <L 88>
    var_10 = wp::extract(var_sensorsize, var_9);
    // if sensor_h != 0.0:                                                                    <L 91>
    var_12 = (var_10 != var_11);
    if (var_12) {
        // fx = intrinsic[0]                                                                  <L 92>
        var_14 = wp::extract(var_intrinsic, var_13);
        // fy = intrinsic[1]                                                                  <L 93>
        var_16 = wp::extract(var_intrinsic, var_15);
        // cx = intrinsic[2]                                                                  <L 94>
        var_18 = wp::extract(var_intrinsic, var_17);
        // cy = intrinsic[3]                                                                  <L 95>
        var_20 = wp::extract(var_intrinsic, var_19);
        // sensor_w = sensorsize[0]                                                           <L 96>
        var_22 = wp::extract(var_sensorsize, var_21);
        // target_aspect = float(img_w) / float(img_h)                                        <L 98>
        var_23 = wp::float(var_img_w);
        var_24 = wp::float(var_img_h);
        var_25 = wp::div(var_23, var_24);
        // sensor_aspect = sensor_w / sensor_h                                                <L 99>
        var_26 = wp::div(var_22, var_10);
        // if target_aspect > sensor_aspect:                                                  <L 100>
        var_27 = (var_25 > var_26);
        if (var_27) {
            // sensor_h = sensor_w / target_aspect                                            <L 101>
            var_28 = wp::div(var_22, var_25);
        }
        var_29 = wp::where(var_27, var_28, var_10);
        if (!var_27) {
            // elif target_aspect < sensor_aspect:                                            <L 102>
            var_30 = (var_25 < var_26);
            if (var_30) {
                // sensor_w = sensor_h * target_aspect                                        <L 103>
                var_31 = wp::mul(var_29, var_25);
            }
            var_32 = wp::where(var_30, var_31, var_22);
        }
        var_33 = wp::where(var_27, var_22, var_32);
        // inv_fx_znear = znear / fx                                                          <L 105>
        var_34 = wp::div(var_znear, var_14);
        // inv_fy_znear = znear / fy                                                          <L 106>
        var_35 = wp::div(var_znear, var_16);
        // left = -inv_fx_znear * (sensor_w * 0.5 - cx)                                       <L 107>
        var_36 = wp::neg(var_34);
        var_38 = wp::mul(var_33, var_37);
        var_39 = wp::sub(var_38, var_18);
        var_40 = wp::mul(var_36, var_39);
        // right = inv_fx_znear * (sensor_w * 0.5 + cx)                                       <L 108>
        var_42 = wp::mul(var_33, var_41);
        var_43 = wp::add(var_42, var_18);
        var_44 = wp::mul(var_34, var_43);
        // top = inv_fy_znear * (sensor_h * 0.5 - cy)                                         <L 109>
        var_46 = wp::mul(var_29, var_45);
        var_47 = wp::sub(var_46, var_20);
        var_48 = wp::mul(var_35, var_47);
        // bottom = -inv_fy_znear * (sensor_h * 0.5 + cy)                                     <L 110>
        var_49 = wp::neg(var_35);
        var_51 = wp::mul(var_29, var_50);
        var_52 = wp::add(var_51, var_20);
        var_53 = wp::mul(var_49, var_52);
    }
    var_54 = wp::where(var_12, var_29, var_10);
    if (!var_12) {
        // fovy_rad = fovy * wp.static(wp.pi / 180.0)                                         <L 112>
        var_56 = wp::mul(var_fovy, var_55);
        // half_height = znear * wp.tan(0.5 * fovy_rad)                                       <L 113>
        var_58 = wp::mul(var_57, var_56);
        var_59 = wp::tan(var_58);
        var_60 = wp::mul(var_znear, var_59);
        // half_width = half_height * aspect                                                  <L 114>
        var_61 = wp::mul(var_60, var_8);
        // left = -half_width                                                                 <L 115>
        var_62 = wp::neg(var_61);
        // right = half_width                                                                 <L 116>
        var_63 = wp::copy(var_61);
        // top = half_height                                                                  <L 117>
        var_64 = wp::copy(var_60);
        // bottom = -half_height                                                              <L 118>
        var_65 = wp::neg(var_60);
    }
    var_66 = wp::where(var_12, var_40, var_62);
    var_67 = wp::where(var_12, var_44, var_63);
    var_68 = wp::where(var_12, var_48, var_64);
    var_69 = wp::where(var_12, var_53, var_65);
    // u = (float(px) + 0.5) / float(img_w)                                                   <L 120>
    var_70 = wp::float(var_px);
    var_72 = wp::add(var_70, var_71);
    var_73 = wp::float(var_img_w);
    var_74 = wp::div(var_72, var_73);
    // v = (float(py) + 0.5) / float(img_h)                                                   <L 121>
    var_75 = wp::float(var_py);
    var_77 = wp::add(var_75, var_76);
    var_78 = wp::float(var_img_h);
    var_79 = wp::div(var_77, var_78);
    // x = left + (right - left) * u                                                          <L 122>
    var_80 = wp::sub(var_67, var_66);
    var_81 = wp::mul(var_80, var_74);
    var_82 = wp::add(var_66, var_81);
    // y = top + (bottom - top) * v                                                           <L 123>
    var_83 = wp::sub(var_69, var_68);
    var_84 = wp::mul(var_83, var_79);
    var_85 = wp::add(var_68, var_84);
    // return wp.normalize(wp.vec3(x, y, -znear))                                             <L 125>
    var_86 = wp::neg(var_znear);
    var_87 = wp::vec_t<3, wp::float32>(var_82, var_85, var_86);
    var_88 = wp::normalize(var_87);
    return var_88;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/math.py:115
static CUDA_CALLABLE void adj_quat_inv_0(
    wp::quat_t<wp::float32> var_quat,
    wp::quat_t<wp::float32> & adj_quat,
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


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/render_util.py:66
static CUDA_CALLABLE void adj_compute_ray_0(
    wp::int32 var_projection,
    wp::float32 var_fovy,
    wp::vec_t<2, wp::float32> var_sensorsize,
    wp::vec_t<4, wp::float32> var_intrinsic,
    wp::int32 var_img_w,
    wp::int32 var_img_h,
    wp::int32 var_px,
    wp::int32 var_py,
    wp::float32 var_znear,
    wp::int32 & adj_projection,
    wp::float32 & adj_fovy,
    wp::vec_t<2, wp::float32> & adj_sensorsize,
    wp::vec_t<4, wp::float32> & adj_intrinsic,
    wp::int32 & adj_img_w,
    wp::int32 & adj_img_h,
    wp::int32 & adj_px,
    wp::int32 & adj_py,
    wp::float32 & adj_znear,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void _compute_body_A_diag_entry_d84c280f_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_nv,
    wp::int32 var_bodyid_target,
    wp::int32 var_row_idx,
    wp::array_t<wp::float32> var_body_jac_row_in,
    wp::array_t<wp::float32> var_result_vec_in,
    wp::array_t<wp::float32> var_body_A_diag_out)
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
        wp::shape_t* var_1;
        const wp::int32 var_2 = 0;
        wp::int32 var_3;
        wp::shape_t var_4;
        wp::int32 var_5;
        const wp::float32 var_6 = 0.0;
        wp::float32 var_7;
        wp::range_t var_8;
        wp::int32 var_9;
        wp::float32* var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        //---------
        // forward
        // def _compute_body_A_diag_entry(                                                        <L 3004>
        // worldid = wp.tid()                                                                     <L 3012>
        var_0 = builtin_tid1d();
        // body_A_diag_id = worldid % body_A_diag_out.shape[0]                                    <L 3013>
        var_1 = &(var_body_A_diag_out.shape);
        var_4 = wp::load(var_1);
        var_3 = wp::extract(var_4, var_2);
        var_5 = wp::mod(var_0, var_3);
        // dot_prod = float(0.0)                                                                  <L 3015>
        var_7 = wp::float(var_6);
        // for i in range(nv):                                                                    <L 3016>
        var_8 = wp::range(var_nv);
        start_for_0:;
            if (iter_cmp(var_8) == 0) goto end_for_0;
            var_9 = wp::iter_next(var_8);
            // dot_prod += body_jac_row_in[worldid, i] * result_vec_in[worldid, i]                <L 3017>
            var_10 = wp::address(var_body_jac_row_in, var_0, var_9);
            var_11 = wp::address(var_result_vec_in, var_0, var_9);
            var_13 = wp::load(var_10);
            var_14 = wp::load(var_11);
            var_12 = wp::mul(var_13, var_14);
            var_15 = wp::add(var_7, var_12);
            wp::assign(var_7, var_15);
            goto start_for_0;
        end_for_0:;
        // body_A_diag_out[body_A_diag_id, bodyid_target, row_idx] = dot_prod                     <L 3018>
        wp::array_store(var_body_A_diag_out, var_5, var_bodyid_target, var_row_idx, var_7);
    }
}



extern "C" __global__ void _finalize_body_invweight0_64175ec4_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_weldid,
    wp::array_t<wp::float32> var_body_A_diag_in,
    wp::array_t<wp::vec_t<2, wp::float32>> var_body_invweight0_out)
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
        bool var_12;
        const wp::int32 var_13 = 0;
        bool var_14;
        wp::int32* var_15;
        const wp::int32 var_16 = 0;
        bool var_17;
        wp::int32 var_18;
        const wp::float32 var_19 = 0.0;
        const wp::float32 var_20 = 0.0;
        wp::vec_t<2, wp::float32> var_21;
        const wp::float32 var_22 = 0.3333333333333333;
        const wp::int32 var_23 = 0;
        wp::float32* var_24;
        const wp::int32 var_25 = 1;
        wp::float32* var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        wp::float32 var_29;
        const wp::int32 var_30 = 2;
        wp::float32* var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        const wp::float32 var_35 = 0.3333333333333333;
        const wp::int32 var_36 = 3;
        wp::float32* var_37;
        const wp::int32 var_38 = 4;
        wp::float32* var_39;
        wp::float32 var_40;
        wp::float32 var_41;
        wp::float32 var_42;
        const wp::int32 var_43 = 5;
        wp::float32* var_44;
        wp::float32 var_45;
        wp::float32 var_46;
        wp::float32 var_47;
        bool var_48;
        const wp::float32 var_49 = 1e-15;
        bool var_50;
        const wp::float32 var_51 = 1e-15;
        bool var_52;
        wp::float32 var_53;
        wp::float32 var_54;
        bool var_55;
        const wp::float32 var_56 = 1e-15;
        bool var_57;
        const wp::float32 var_58 = 1e-15;
        bool var_59;
        wp::float32 var_60;
        wp::float32 var_61;
        wp::float32 var_62;
        wp::vec_t<2, wp::float32> var_63;
        //---------
        // forward
        // def _finalize_body_invweight0(                                                         <L 3022>
        // worldid, bodyid = wp.tid()                                                             <L 3027>
        builtin_tid2d(var_0, var_1);
        // body_invweight0_id = worldid % body_invweight0_out.shape[0]                            <L 3028>
        var_2 = &(var_body_invweight0_out.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // body_A_diag_id = worldid % body_A_diag_in.shape[0]                                     <L 3029>
        var_7 = &(var_body_A_diag_in.shape);
        var_10 = wp::load(var_7);
        var_9 = wp::extract(var_10, var_8);
        var_11 = wp::mod(var_0, var_9);
        // if bodyid == 0 or body_weldid[bodyid] == 0:                                            <L 3032>
        var_14 = (var_1 == var_13);
        var_12 = var_14;
        if (!var_12) {
            var_15 = wp::address(var_body_weldid, var_1);
            var_18 = wp::load(var_15);
            var_17 = (var_18 == var_16);
            var_12 = var_12 || var_17;
        }
        if (var_12) {
            // body_invweight0_out[body_invweight0_id, bodyid] = wp.vec2(0.0, 0.0)                <L 3033>
            var_21 = wp::vec_t<2, wp::float32>(var_19, var_20);
            wp::array_store(var_body_invweight0_out, var_6, var_1, var_21);
            // return                                                                             <L 3034>
            continue;
        }
        // inv_trans = wp.static(1.0 / 3.0) * (                                                   <L 3037>
        // body_A_diag_in[body_A_diag_id, bodyid, 0]                                              <L 3038>
        var_24 = wp::address(var_body_A_diag_in, var_11, var_1, var_23);
        // + body_A_diag_in[body_A_diag_id, bodyid, 1]                                            <L 3039>
        var_26 = wp::address(var_body_A_diag_in, var_11, var_1, var_25);
        var_28 = wp::load(var_24);
        var_29 = wp::load(var_26);
        var_27 = wp::add(var_28, var_29);
        // + body_A_diag_in[body_A_diag_id, bodyid, 2]                                            <L 3040>
        var_31 = wp::address(var_body_A_diag_in, var_11, var_1, var_30);
        var_33 = wp::load(var_31);
        var_32 = wp::add(var_27, var_33);
        var_34 = wp::mul(var_22, var_32);
        // inv_rot = wp.static(1.0 / 3.0) * (                                                     <L 3042>
        // body_A_diag_in[body_A_diag_id, bodyid, 3]                                              <L 3043>
        var_37 = wp::address(var_body_A_diag_in, var_11, var_1, var_36);
        // + body_A_diag_in[body_A_diag_id, bodyid, 4]                                            <L 3044>
        var_39 = wp::address(var_body_A_diag_in, var_11, var_1, var_38);
        var_41 = wp::load(var_37);
        var_42 = wp::load(var_39);
        var_40 = wp::add(var_41, var_42);
        // + body_A_diag_in[body_A_diag_id, bodyid, 5]                                            <L 3045>
        var_44 = wp::address(var_body_A_diag_in, var_11, var_1, var_43);
        var_46 = wp::load(var_44);
        var_45 = wp::add(var_40, var_46);
        var_47 = wp::mul(var_35, var_45);
        // if inv_trans < mujoco.mjMINVAL and inv_rot > mujoco.mjMINVAL:                          <L 3049>
        var_50 = (var_34 < var_49);
        var_48 = var_50;
        if (var_48) {
            var_52 = (var_47 > var_51);
            var_48 = var_48 && var_52;
        }
        if (var_48) {
            // inv_trans = inv_rot  # use rotation as fallback for translation                    <L 3050>
            var_53 = wp::copy(var_47);
        }
        var_54 = wp::where(var_48, var_53, var_34);
        if (!var_48) {
            // elif inv_rot < mujoco.mjMINVAL and inv_trans > mujoco.mjMINVAL:                    <L 3051>
            var_57 = (var_47 < var_56);
            var_55 = var_57;
            if (var_55) {
                var_59 = (var_54 > var_58);
                var_55 = var_55 && var_59;
            }
            if (var_55) {
                // inv_rot = inv_trans  # use translation as fallback for rotation                <L 3052>
                var_60 = wp::copy(var_54);
            }
            var_61 = wp::where(var_55, var_60, var_47);
        }
        var_62 = wp::where(var_48, var_47, var_61);
        // body_invweight0_out[body_invweight0_id, bodyid] = wp.vec2(inv_trans, inv_rot)          <L 3054>
        var_63 = wp::vec_t<2, wp::float32>(var_54, var_62);
        wp::array_store(var_body_invweight0_out, var_6, var_1, var_63);
    }
}



extern "C" __global__ void _compute_tendon_dot_product_be0bc0e6_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::int32 var_tenid_target,
    wp::array_t<wp::float32> var_ten_J_in,
    wp::array_t<wp::float32> var_result_vec_in,
    wp::array_t<wp::float32> var_tendon_invweight0_out)
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
        wp::shape_t* var_1;
        const wp::int32 var_2 = 0;
        wp::int32 var_3;
        wp::shape_t var_4;
        wp::int32 var_5;
        const wp::float32 var_6 = 0.0;
        wp::float32 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::range_t var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::float32* var_20;
        wp::float32* var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        wp::float32 var_25;
        //---------
        // forward
        // def _compute_tendon_dot_product(                                                       <L 3076>
        // worldid = wp.tid()                                                                     <L 3088>
        var_0 = builtin_tid1d();
        // tendon_invweight0_id = worldid % tendon_invweight0_out.shape[0]                        <L 3089>
        var_1 = &(var_tendon_invweight0_out.shape);
        var_4 = wp::load(var_1);
        var_3 = wp::extract(var_4, var_2);
        var_5 = wp::mod(var_0, var_3);
        // dot_prod = float(0.0)                                                                  <L 3090>
        var_7 = wp::float(var_6);
        // rownnz = ten_J_rownnz[tenid_target]                                                    <L 3092>
        var_8 = wp::address(var_ten_J_rownnz, var_tenid_target);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // rowadr = ten_J_rowadr[tenid_target]                                                    <L 3093>
        var_11 = wp::address(var_ten_J_rowadr, var_tenid_target);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // for i in range(rownnz):                                                                <L 3094>
        var_14 = wp::range(var_9);
        start_for_0:;
            if (iter_cmp(var_14) == 0) goto end_for_0;
            var_15 = wp::iter_next(var_14);
            // sparseid = rowadr + i                                                              <L 3095>
            var_16 = wp::add(var_12, var_15);
            // colind = ten_J_colind[sparseid]                                                    <L 3096>
            var_17 = wp::address(var_ten_J_colind, var_16);
            var_19 = wp::load(var_17);
            var_18 = wp::copy(var_19);
            // dot_prod += ten_J_in[worldid, sparseid] * result_vec_in[worldid, colind]           <L 3097>
            var_20 = wp::address(var_ten_J_in, var_0, var_16);
            var_21 = wp::address(var_result_vec_in, var_0, var_18);
            var_23 = wp::load(var_20);
            var_24 = wp::load(var_21);
            var_22 = wp::mul(var_23, var_24);
            var_25 = wp::add(var_7, var_22);
            wp::assign(var_7, var_25);
            goto start_for_0;
        end_for_0:;
        // tendon_invweight0_out[tendon_invweight0_id, tenid_target] = dot_prod                   <L 3099>
        wp::array_store(var_tendon_invweight0_out, var_5, var_tenid_target, var_7);
    }
}



extern "C" __global__ void _resolve_dampratio_9820ff1f_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_actuator_biastype,
    wp::array_t<wp::vec_t<10, wp::float32>> var_actuator_gainprm,
    wp::array_t<wp::int32> var_moment_rownnz_in,
    wp::array_t<wp::int32> var_moment_rowadr_in,
    wp::array_t<wp::int32> var_moment_colind_in,
    wp::array_t<wp::float32> var_actuator_moment_in,
    wp::array_t<wp::float32> var_dof_M0_in,
    wp::int32 var_nv,
    wp::array_t<wp::vec_t<10, wp::float32>> var_actuator_biasprm)
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
        const wp::int32 var_5 = 1;
        bool var_6;
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
        wp::vec_t<10, wp::float32>* var_17;
        const wp::int32 var_18 = 0;
        wp::float32 var_19;
        wp::vec_t<10, wp::float32> var_20;
        wp::vec_t<10, wp::float32>* var_21;
        wp::vec_t<10, wp::float32> var_22;
        wp::vec_t<10, wp::float32> var_23;
        const wp::int32 var_24 = 1;
        wp::float32 var_25;
        wp::float32 var_26;
        wp::float32 var_27;
        const wp::float32 var_28 = 1e-15;
        bool var_29;
        const wp::int32 var_30 = 2;
        wp::float32 var_31;
        const wp::float32 var_32 = 0.0;
        bool var_33;
        const wp::int32 var_34 = 2;
        wp::float32 var_35;
        const wp::float32 var_36 = 0.0;
        wp::float32 var_37;
        wp::int32* var_38;
        wp::int32 var_39;
        wp::int32 var_40;
        wp::int32* var_41;
        wp::int32 var_42;
        wp::int32 var_43;
        wp::range_t var_44;
        wp::int32 var_45;
        wp::int32 var_46;
        wp::int32* var_47;
        wp::int32 var_48;
        wp::int32 var_49;
        wp::float32* var_50;
        wp::float32 var_51;
        wp::float32 var_52;
        wp::float32 var_53;
        bool var_54;
        wp::float32* var_55;
        wp::float32 var_56;
        wp::float32 var_57;
        wp::float32 var_58;
        wp::float32 var_59;
        wp::float32 var_60;
        const wp::float32 var_61 = 2.0;
        wp::float32 var_62;
        wp::float32 var_63;
        wp::float32 var_64;
        wp::float32 var_65;
        wp::vec_t<10, wp::float32> var_66;
        wp::float32 var_67;
        const wp::int32 var_68 = 2;
        //---------
        // forward
        // def _resolve_dampratio(                                                                <L 3205>
        // worldid, actid = wp.tid()                                                              <L 3216>
        builtin_tid2d(var_0, var_1);
        // biastype = actuator_biastype[actid]                                                    <L 3217>
        var_2 = wp::address(var_actuator_biastype, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if biastype != BiasType.AFFINE:                                                        <L 3220>
        var_6 = (var_3 != var_5);
        if (var_6) {
            // return                                                                             <L 3221>
            continue;
        }
        // gainprm_id = worldid % actuator_gainprm.shape[0]                                       <L 3223>
        var_7 = &(var_actuator_gainprm.shape);
        var_10 = wp::load(var_7);
        var_9 = wp::extract(var_10, var_8);
        var_11 = wp::mod(var_0, var_9);
        // biasprm_id = worldid % actuator_biasprm.shape[0]                                       <L 3224>
        var_12 = &(var_actuator_biasprm.shape);
        var_15 = wp::load(var_12);
        var_14 = wp::extract(var_15, var_13);
        var_16 = wp::mod(var_0, var_14);
        // kp = actuator_gainprm[gainprm_id, actid][0]                                            <L 3225>
        var_17 = wp::address(var_actuator_gainprm, var_11, var_1);
        var_20 = wp::load(var_17);
        var_19 = wp::extract(var_20, var_18);
        // biasprm = actuator_biasprm[biasprm_id, actid]                                          <L 3227>
        var_21 = wp::address(var_actuator_biasprm, var_16, var_1);
        var_23 = wp::load(var_21);
        var_22 = wp::copy(var_23);
        // if wp.abs(kp + biasprm[1]) > MJ_MINVAL:                                                <L 3229>
        var_25 = wp::extract(var_22, var_24);
        var_26 = wp::add(var_19, var_25);
        var_27 = wp::abs(var_26);
        var_29 = (var_27 > var_28);
        if (var_29) {
            // return                                                                             <L 3230>
            continue;
        }
        // if biasprm[2] <= 0.0:                                                                  <L 3231>
        var_31 = wp::extract(var_22, var_30);
        var_33 = (var_31 <= var_32);
        if (var_33) {
            // return                                                                             <L 3232>
            continue;
        }
        // dampratio = biasprm[2]                                                                 <L 3234>
        var_35 = wp::extract(var_22, var_34);
        // mass = float(0.0)                                                                      <L 3237>
        var_37 = wp::float(var_36);
        // rownnz = moment_rownnz_in[worldid, actid]                                              <L 3238>
        var_38 = wp::address(var_moment_rownnz_in, var_0, var_1);
        var_40 = wp::load(var_38);
        var_39 = wp::copy(var_40);
        // rowadr = moment_rowadr_in[worldid, actid]                                              <L 3239>
        var_41 = wp::address(var_moment_rowadr_in, var_0, var_1);
        var_43 = wp::load(var_41);
        var_42 = wp::copy(var_43);
        // for k in range(rownnz):                                                                <L 3240>
        var_44 = wp::range(var_39);
        start_for_3:;
            if (iter_cmp(var_44) == 0) goto end_for_3;
            var_45 = wp::iter_next(var_44);
            // sparseid = rowadr + k                                                              <L 3241>
            var_46 = wp::add(var_42, var_45);
            // j = moment_colind_in[worldid, sparseid]                                            <L 3242>
            var_47 = wp::address(var_moment_colind_in, var_0, var_46);
            var_49 = wp::load(var_47);
            var_48 = wp::copy(var_49);
            // moment = actuator_moment_in[worldid, sparseid]                                     <L 3243>
            var_50 = wp::address(var_actuator_moment_in, var_0, var_46);
            var_52 = wp::load(var_50);
            var_51 = wp::copy(var_52);
            // if wp.abs(moment) > MJ_MINVAL:                                                     <L 3244>
            var_53 = wp::abs(var_51);
            var_54 = (var_53 > var_28);
            if (var_54) {
                // mass += dof_M0_in[worldid, j] / (moment * moment)                              <L 3245>
                var_55 = wp::address(var_dof_M0_in, var_0, var_48);
                var_56 = wp::mul(var_51, var_51);
                var_58 = wp::load(var_55);
                var_57 = wp::div(var_58, var_56);
                var_59 = wp::add(var_37, var_57);
            }
            var_60 = wp::where(var_54, var_59, var_37);
            wp::assign(var_37, var_60);
            goto start_for_3;
        end_for_3:;
        // damping = dampratio * 2.0 * wp.sqrt(kp * mass)                                         <L 3247>
        var_62 = wp::mul(var_35, var_61);
        var_63 = wp::mul(var_19, var_37);
        var_64 = wp::sqrt(var_63);
        var_65 = wp::mul(var_62, var_64);
        // new_biasprm = biasprm                                                                  <L 3250>
        var_66 = wp::copy(var_22);
        // new_biasprm[2] = -damping                                                              <L 3251>
        var_67 = wp::neg(var_65);
        wp::assign_inplace(var_66, var_68, var_67);
        // actuator_biasprm[biasprm_id, actid] = new_biasprm                                      <L 3252>
        wp::array_store(var_actuator_biasprm, var_16, var_1, var_66);
    }
}



extern "C" __global__ void _copy_tendon_jacobian_cf465fd5_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_tenid_target,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::float32> var_ten_J_in,
    wp::array_t<wp::float32> var_ten_J_vec_out)
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
        wp::shape_t* var_1;
        const wp::int32 var_2 = 2;
        wp::int32 var_3;
        wp::shape_t var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::range_t var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::float32* var_18;
        wp::float32 var_19;
        //---------
        // forward
        // def _copy_tendon_jacobian(                                                             <L 3058>
        // worldid = wp.tid()                                                                     <L 3066>
        var_0 = builtin_tid1d();
        // nv = ten_J_in.shape[2]                                                                 <L 3067>
        var_1 = &(var_ten_J_in.shape);
        var_4 = wp::load(var_1);
        var_3 = wp::extract(var_4, var_2);
        // rownnz = ten_J_rownnz[tenid_target]                                                    <L 3068>
        var_5 = wp::address(var_ten_J_rownnz, var_tenid_target);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // rowadr = ten_J_rowadr[tenid_target]                                                    <L 3069>
        var_8 = wp::address(var_ten_J_rowadr, var_tenid_target);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // for i in range(rownnz):                                                                <L 3070>
        var_11 = wp::range(var_6);
        start_for_0:;
            if (iter_cmp(var_11) == 0) goto end_for_0;
            var_12 = wp::iter_next(var_11);
            // colind = ten_J_colind[rowadr + i]                                                  <L 3071>
            var_13 = wp::add(var_9, var_12);
            var_14 = wp::address(var_ten_J_colind, var_13);
            var_16 = wp::load(var_14);
            var_15 = wp::copy(var_16);
            // ten_J_vec_out[worldid, colind] = ten_J_in[worldid, rowadr + i]                     <L 3072>
            var_17 = wp::add(var_9, var_12);
            var_18 = wp::address(var_ten_J_in, var_0, var_17);
            var_19 = wp::load(var_18);
            wp::array_store(var_ten_J_vec_out, var_0, var_15, var_19);
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void _accumulate_subtreemass_a3369753_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::float32> var_body_subtreemass_io,
    wp::array_t<wp::int32> var_body_tree_)
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
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        const wp::int32 var_13 = 0;
        bool var_14;
        wp::float32* var_15;
        wp::float32 var_16;
        wp::float32 var_17;
        //---------
        // forward
        // def _accumulate_subtreemass(                                                           <L 2728>
        // worldid, nodeid = wp.tid()                                                             <L 2733>
        builtin_tid2d(var_0, var_1);
        // body_subtreemass_id = worldid % body_subtreemass_io.shape[0]                           <L 2734>
        var_2 = &(var_body_subtreemass_io.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // bodyid = body_tree_[nodeid]                                                            <L 2735>
        var_7 = wp::address(var_body_tree_, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // parentid = body_parentid[bodyid]                                                       <L 2736>
        var_10 = wp::address(var_body_parentid, var_8);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // if bodyid != 0:                                                                        <L 2737>
        var_14 = (var_8 != var_13);
        if (var_14) {
            // wp.atomic_add(body_subtreemass_io, body_subtreemass_id, parentid, body_subtreemass_io[body_subtreemass_id, bodyid])       <L 2738>
            var_15 = wp::address(var_body_subtreemass_io, var_6, var_8);
            var_17 = wp::load(var_15);
            var_16 = wp::atomic_add(var_body_subtreemass_io, var_6, var_11, var_17);
        }
    }
}



extern "C" __global__ void _finalize_dof_invweight0_966ce0d5_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_dof_jntid,
    wp::array_t<wp::int32> var_jnt_type,
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::float32> var_dof_A_diag_in,
    wp::array_t<wp::float32> var_dof_invweight0_out)
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
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::int32* var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        const wp::int32 var_21 = 0;
        const wp::int32 var_22 = 0;
        wp::int32 var_23;
        bool var_24;
        const wp::int32 var_25 = 3;
        wp::int32 var_26;
        bool var_27;
        const wp::float32 var_28 = 0.3333333333333333;
        const wp::int32 var_29 = 0;
        wp::int32 var_30;
        wp::float32* var_31;
        const wp::int32 var_32 = 1;
        wp::int32 var_33;
        wp::float32* var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        wp::float32 var_37;
        const wp::int32 var_38 = 2;
        wp::int32 var_39;
        wp::float32* var_40;
        wp::float32 var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        const wp::float32 var_44 = 0.3333333333333333;
        const wp::int32 var_45 = 3;
        wp::int32 var_46;
        wp::float32* var_47;
        const wp::int32 var_48 = 4;
        wp::int32 var_49;
        wp::float32* var_50;
        wp::float32 var_51;
        wp::float32 var_52;
        wp::float32 var_53;
        const wp::int32 var_54 = 5;
        wp::int32 var_55;
        wp::float32* var_56;
        wp::float32 var_57;
        wp::float32 var_58;
        wp::float32 var_59;
        wp::float32 var_60;
        const wp::int32 var_61 = 1;
        const wp::int32 var_62 = 1;
        wp::int32 var_63;
        bool var_64;
        const wp::float32 var_65 = 0.3333333333333333;
        const wp::int32 var_66 = 0;
        wp::int32 var_67;
        wp::float32* var_68;
        const wp::int32 var_69 = 1;
        wp::int32 var_70;
        wp::float32* var_71;
        wp::float32 var_72;
        wp::float32 var_73;
        wp::float32 var_74;
        const wp::int32 var_75 = 2;
        wp::int32 var_76;
        wp::float32* var_77;
        wp::float32 var_78;
        wp::float32 var_79;
        wp::float32 var_80;
        wp::float32 var_81;
        wp::float32* var_82;
        wp::float32 var_83;
        wp::float32 var_84;
        //---------
        // forward
        // def _finalize_dof_invweight0(                                                          <L 2901>
        // worldid, dofid = wp.tid()                                                              <L 2908>
        builtin_tid2d(var_0, var_1);
        // dof_invweight0_id = worldid % dof_invweight0_out.shape[0]                              <L 2909>
        var_2 = &(var_dof_invweight0_out.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // dof_A_diag_id = worldid % dof_A_diag_in.shape[0]                                       <L 2910>
        var_7 = &(var_dof_A_diag_in.shape);
        var_10 = wp::load(var_7);
        var_9 = wp::extract(var_10, var_8);
        var_11 = wp::mod(var_0, var_9);
        // jntid = dof_jntid[dofid]                                                               <L 2912>
        var_12 = wp::address(var_dof_jntid, var_1);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // jtype = jnt_type[jntid]                                                                <L 2913>
        var_15 = wp::address(var_jnt_type, var_13);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
        // dofadr = jnt_dofadr[jntid]                                                             <L 2914>
        var_18 = wp::address(var_jnt_dofadr, var_13);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // if jtype == int(types.JointType.FREE.value):                                           <L 2916>
        var_23 = wp::int(var_22);
        var_24 = (var_16 == var_23);
        if (var_24) {
            // if dofid < dofadr + 3:                                                             <L 2918>
            var_26 = wp::add(var_19, var_25);
            var_27 = (var_1 < var_26);
            if (var_27) {
                // avg = wp.static(1.0 / 3.0) * (                                                 <L 2919>
                // dof_A_diag_in[dof_A_diag_id, dofadr + 0]                                       <L 2920>
                var_30 = wp::add(var_19, var_29);
                var_31 = wp::address(var_dof_A_diag_in, var_11, var_30);
                // + dof_A_diag_in[dof_A_diag_id, dofadr + 1]                                     <L 2921>
                var_33 = wp::add(var_19, var_32);
                var_34 = wp::address(var_dof_A_diag_in, var_11, var_33);
                var_36 = wp::load(var_31);
                var_37 = wp::load(var_34);
                var_35 = wp::add(var_36, var_37);
                // + dof_A_diag_in[dof_A_diag_id, dofadr + 2]                                     <L 2922>
                var_39 = wp::add(var_19, var_38);
                var_40 = wp::address(var_dof_A_diag_in, var_11, var_39);
                var_42 = wp::load(var_40);
                var_41 = wp::add(var_35, var_42);
                var_43 = wp::mul(var_28, var_41);
            }
            if (!var_27) {
                // avg = wp.static(1.0 / 3.0) * (                                                 <L 2925>
                // dof_A_diag_in[dof_A_diag_id, dofadr + 3]                                       <L 2926>
                var_46 = wp::add(var_19, var_45);
                var_47 = wp::address(var_dof_A_diag_in, var_11, var_46);
                // + dof_A_diag_in[dof_A_diag_id, dofadr + 4]                                     <L 2927>
                var_49 = wp::add(var_19, var_48);
                var_50 = wp::address(var_dof_A_diag_in, var_11, var_49);
                var_52 = wp::load(var_47);
                var_53 = wp::load(var_50);
                var_51 = wp::add(var_52, var_53);
                // + dof_A_diag_in[dof_A_diag_id, dofadr + 5]                                     <L 2928>
                var_55 = wp::add(var_19, var_54);
                var_56 = wp::address(var_dof_A_diag_in, var_11, var_55);
                var_58 = wp::load(var_56);
                var_57 = wp::add(var_51, var_58);
                var_59 = wp::mul(var_44, var_57);
            }
            var_60 = wp::where(var_27, var_43, var_59);
            // dof_invweight0_out[dof_invweight0_id, dofid] = avg                                 <L 2930>
            wp::array_store(var_dof_invweight0_out, var_6, var_1, var_60);
        }
        if (!var_24) {
            // elif jtype == int(types.JointType.BALL.value):                                     <L 2931>
            var_63 = wp::int(var_62);
            var_64 = (var_16 == var_63);
            if (var_64) {
                // avg = wp.static(1.0 / 3.0) * (                                                 <L 2933>
                // dof_A_diag_in[dof_A_diag_id, dofadr + 0]                                       <L 2934>
                var_67 = wp::add(var_19, var_66);
                var_68 = wp::address(var_dof_A_diag_in, var_11, var_67);
                // + dof_A_diag_in[dof_A_diag_id, dofadr + 1]                                     <L 2935>
                var_70 = wp::add(var_19, var_69);
                var_71 = wp::address(var_dof_A_diag_in, var_11, var_70);
                var_73 = wp::load(var_68);
                var_74 = wp::load(var_71);
                var_72 = wp::add(var_73, var_74);
                // + dof_A_diag_in[dof_A_diag_id, dofadr + 2]                                     <L 2936>
                var_76 = wp::add(var_19, var_75);
                var_77 = wp::address(var_dof_A_diag_in, var_11, var_76);
                var_79 = wp::load(var_77);
                var_78 = wp::add(var_72, var_79);
                var_80 = wp::mul(var_65, var_78);
                // dof_invweight0_out[dof_invweight0_id, dofid] = avg                             <L 2938>
                wp::array_store(var_dof_invweight0_out, var_6, var_1, var_80);
            }
            var_81 = wp::where(var_64, var_80, var_60);
            if (!var_64) {
                // dof_invweight0_out[dof_invweight0_id, dofid] = dof_A_diag_in[dof_A_diag_id, dofid]       <L 2941>
                var_82 = wp::address(var_dof_A_diag_in, var_11, var_1);
                var_83 = wp::load(var_82);
                wp::array_store(var_dof_invweight0_out, var_6, var_1, var_83);
            }
        }
        var_84 = wp::where(var_24, var_60, var_81);
    }
}



extern "C" __global__ void _compute_eq_data0_d31148f7_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_eq_type,
    wp::array_t<wp::int32> var_eq_obj1id,
    wp::array_t<wp::int32> var_eq_obj2id,
    wp::array_t<wp::int32> var_eq_objtype,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::quat_t<wp::float32>> var_xquat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::vec_t<11, wp::float32>> var_eq_data_out)
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
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::vec_t<11, wp::float32>* var_13;
        wp::vec_t<11, wp::float32> var_14;
        wp::vec_t<11, wp::float32> var_15;
        const wp::int32 var_16 = 0;
        const wp::int32 var_17 = 0;
        wp::int32 var_18;
        bool var_19;
        const wp::int32 var_20 = 1;
        const wp::int32 var_21 = 1;
        wp::int32 var_22;
        bool var_23;
        wp::int32* var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        wp::int32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        const wp::int32 var_30 = 0;
        wp::float32 var_31;
        const wp::int32 var_32 = 1;
        wp::float32 var_33;
        const wp::int32 var_34 = 2;
        wp::float32 var_35;
        wp::vec_t<3, wp::float32> var_36;
        wp::vec_t<3, wp::float32>* var_37;
        wp::mat_t<3, 3, wp::float32>* var_38;
        wp::vec_t<3, wp::float32> var_39;
        wp::mat_t<3, 3, wp::float32> var_40;
        wp::vec_t<3, wp::float32> var_41;
        wp::vec_t<3, wp::float32> var_42;
        wp::mat_t<3, 3, wp::float32>* var_43;
        wp::mat_t<3, 3, wp::float32> var_44;
        wp::mat_t<3, 3, wp::float32> var_45;
        wp::vec_t<3, wp::float32>* var_46;
        wp::vec_t<3, wp::float32> var_47;
        wp::vec_t<3, wp::float32> var_48;
        wp::vec_t<3, wp::float32> var_49;
        const wp::int32 var_50 = 0;
        wp::float32 var_51;
        const wp::int32 var_52 = 3;
        const wp::int32 var_53 = 1;
        wp::float32 var_54;
        const wp::int32 var_55 = 4;
        const wp::int32 var_56 = 2;
        wp::float32 var_57;
        const wp::int32 var_58 = 5;
        const wp::int32 var_59 = 6;
        const wp::int32 var_60 = 6;
        wp::int32 var_61;
        bool var_62;
        const wp::float32 var_63 = 0.0;
        wp::vec_t<11, wp::float32> var_64;
        const wp::int32 var_65 = 1;
        const wp::int32 var_66 = 1;
        wp::int32 var_67;
        bool var_68;
        const wp::int32 var_69 = 1;
        const wp::int32 var_70 = 1;
        wp::int32 var_71;
        bool var_72;
        const wp::int32 var_73 = 6;
        wp::float32 var_74;
        const wp::int32 var_75 = 7;
        wp::float32 var_76;
        const wp::int32 var_77 = 8;
        wp::float32 var_78;
        const wp::int32 var_79 = 9;
        wp::float32 var_80;
        wp::quat_t<wp::float32> var_81;
        wp::float32 var_82;
        const wp::float32 var_83 = 0.0;
        bool var_84;
        wp::quat_t<wp::float32> var_85;
        const wp::int32 var_86 = 0;
        wp::float32 var_87;
        const wp::int32 var_88 = 6;
        const wp::int32 var_89 = 1;
        wp::float32 var_90;
        const wp::int32 var_91 = 7;
        const wp::int32 var_92 = 2;
        wp::float32 var_93;
        const wp::int32 var_94 = 8;
        const wp::int32 var_95 = 3;
        wp::float32 var_96;
        const wp::int32 var_97 = 9;
        wp::quat_t<wp::float32> var_98;
        wp::int32* var_99;
        wp::int32 var_100;
        wp::int32 var_101;
        wp::int32* var_102;
        wp::int32 var_103;
        wp::int32 var_104;
        const wp::int32 var_105 = 0;
        wp::float32 var_106;
        const wp::int32 var_107 = 1;
        wp::float32 var_108;
        const wp::int32 var_109 = 2;
        wp::float32 var_110;
        wp::vec_t<3, wp::float32> var_111;
        wp::vec_t<3, wp::float32>* var_112;
        wp::mat_t<3, 3, wp::float32>* var_113;
        wp::vec_t<3, wp::float32> var_114;
        wp::mat_t<3, 3, wp::float32> var_115;
        wp::vec_t<3, wp::float32> var_116;
        wp::vec_t<3, wp::float32> var_117;
        wp::mat_t<3, 3, wp::float32>* var_118;
        wp::mat_t<3, 3, wp::float32> var_119;
        wp::mat_t<3, 3, wp::float32> var_120;
        wp::vec_t<3, wp::float32>* var_121;
        wp::vec_t<3, wp::float32> var_122;
        wp::vec_t<3, wp::float32> var_123;
        wp::vec_t<3, wp::float32> var_124;
        const wp::int32 var_125 = 0;
        wp::float32 var_126;
        const wp::int32 var_127 = 3;
        const wp::int32 var_128 = 1;
        wp::float32 var_129;
        const wp::int32 var_130 = 4;
        const wp::int32 var_131 = 2;
        wp::float32 var_132;
        const wp::int32 var_133 = 5;
        wp::quat_t<wp::float32>* var_134;
        wp::quat_t<wp::float32> var_135;
        wp::quat_t<wp::float32> var_136;
        wp::quat_t<wp::float32>* var_137;
        wp::quat_t<wp::float32> var_138;
        wp::quat_t<wp::float32> var_139;
        const wp::int32 var_140 = 0;
        wp::float32 var_141;
        const wp::int32 var_142 = 6;
        const wp::int32 var_143 = 1;
        wp::float32 var_144;
        const wp::int32 var_145 = 7;
        const wp::int32 var_146 = 2;
        wp::float32 var_147;
        const wp::int32 var_148 = 8;
        const wp::int32 var_149 = 3;
        wp::float32 var_150;
        const wp::int32 var_151 = 9;
        wp::int32 var_152;
        wp::int32 var_153;
        wp::vec_t<3, wp::float32> var_154;
        wp::vec_t<3, wp::float32> var_155;
        wp::vec_t<3, wp::float32> var_156;
        wp::int32 var_157;
        wp::int32 var_158;
        wp::vec_t<3, wp::float32> var_159;
        wp::vec_t<3, wp::float32> var_160;
        wp::vec_t<3, wp::float32> var_161;
        wp::int32 var_162;
        wp::int32 var_163;
        wp::vec_t<3, wp::float32> var_164;
        wp::vec_t<3, wp::float32> var_165;
        wp::vec_t<3, wp::float32> var_166;
        wp::int32 var_167;
        wp::int32 var_168;
        wp::vec_t<3, wp::float32> var_169;
        wp::vec_t<3, wp::float32> var_170;
        wp::vec_t<3, wp::float32> var_171;
        //---------
        // forward
        // def _compute_eq_data0(                                                                 <L 2762>
        // worldid, eqid = wp.tid()                                                               <L 2779>
        builtin_tid2d(var_0, var_1);
        // eq_data_id = worldid % eq_data_out.shape[0]                                            <L 2780>
        var_2 = &(var_eq_data_out.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // eqtype = eq_type[eqid]                                                                 <L 2782>
        var_7 = wp::address(var_eq_type, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // objtype = eq_objtype[eqid]                                                             <L 2783>
        var_10 = wp::address(var_eq_objtype, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // data = eq_data_out[eq_data_id, eqid]                                                   <L 2784>
        var_13 = wp::address(var_eq_data_out, var_6, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // if eqtype == int(types.EqType.CONNECT.value):                                          <L 2786>
        var_18 = wp::int(var_17);
        var_19 = (var_8 == var_18);
        if (var_19) {
            // if objtype == int(types.ObjType.BODY.value):                                       <L 2787>
            var_22 = wp::int(var_21);
            var_23 = (var_11 == var_22);
            if (var_23) {
                // obj1id = eq_obj1id[eqid]                                                       <L 2788>
                var_24 = wp::address(var_eq_obj1id, var_1);
                var_26 = wp::load(var_24);
                var_25 = wp::copy(var_26);
                // obj2id = eq_obj2id[eqid]                                                       <L 2789>
                var_27 = wp::address(var_eq_obj2id, var_1);
                var_29 = wp::load(var_27);
                var_28 = wp::copy(var_29);
                // anchor1 = wp.vec3(data[0], data[1], data[2])                                   <L 2792>
                var_31 = wp::extract(var_14, var_30);
                var_33 = wp::extract(var_14, var_32);
                var_35 = wp::extract(var_14, var_34);
                var_36 = wp::vec_t<3, wp::float32>(var_31, var_33, var_35);
                // pos = xpos_in[worldid, obj1id] + xmat_in[worldid, obj1id] @ anchor1            <L 2793>
                var_37 = wp::address(var_xpos_in, var_0, var_25);
                var_38 = wp::address(var_xmat_in, var_0, var_25);
                var_40 = wp::load(var_38);
                var_39 = wp::mul(var_40, var_36);
                var_42 = wp::load(var_37);
                var_41 = wp::add(var_42, var_39);
                // anchor2 = wp.transpose(xmat_in[worldid, obj2id]) @ (pos - xpos_in[worldid, obj2id])       <L 2796>
                var_43 = wp::address(var_xmat_in, var_0, var_28);
                var_45 = wp::load(var_43);
                var_44 = wp::transpose(var_45);
                var_46 = wp::address(var_xpos_in, var_0, var_28);
                var_48 = wp::load(var_46);
                var_47 = wp::sub(var_41, var_48);
                var_49 = wp::mul(var_44, var_47);
                // data[3] = anchor2[0]                                                           <L 2797>
                var_51 = wp::extract(var_49, var_50);
                wp::assign_inplace(var_14, var_52, var_51);
                // data[4] = anchor2[1]                                                           <L 2798>
                var_54 = wp::extract(var_49, var_53);
                wp::assign_inplace(var_14, var_55, var_54);
                // data[5] = anchor2[2]                                                           <L 2799>
                var_57 = wp::extract(var_49, var_56);
                wp::assign_inplace(var_14, var_58, var_57);
                // eq_data_out[eq_data_id, eqid] = data                                           <L 2800>
                wp::array_store(var_eq_data_out, var_6, var_1, var_14);
            }
            if (!var_23) {
                // elif objtype == int(types.ObjType.SITE.value):                                 <L 2801>
                var_61 = wp::int(var_60);
                var_62 = (var_11 == var_61);
                if (var_62) {
                    // eq_data_out[eq_data_id, eqid] = types.vec11(0.0)                           <L 2803>
                    var_64 = wp::vec_t<11, wp::float32>(var_63);
                    wp::array_store(var_eq_data_out, var_6, var_1, var_64);
                }
            }
        }
        if (!var_19) {
            // elif eqtype == int(types.EqType.WELD.value):                                       <L 2804>
            var_67 = wp::int(var_66);
            var_68 = (var_8 == var_67);
            if (var_68) {
                // if objtype == int(types.ObjType.BODY.value):                                   <L 2805>
                var_71 = wp::int(var_70);
                var_72 = (var_11 == var_71);
                if (var_72) {
                    // quat = wp.quat(data[6], data[7], data[8], data[9])                         <L 2806>
                    var_74 = wp::extract(var_14, var_73);
                    var_76 = wp::extract(var_14, var_75);
                    var_78 = wp::extract(var_14, var_77);
                    var_80 = wp::extract(var_14, var_79);
                    var_81 = wp::quat_t<wp::float32>(var_74, var_76, var_78, var_80);
                    // if wp.length_sq(quat) > 0.0:                                               <L 2807>
                    var_82 = wp::length_sq(var_81);
                    var_84 = (var_82 > var_83);
                    if (var_84) {
                        // quat = wp.normalize(quat)                                              <L 2809>
                        var_85 = wp::normalize(var_81);
                        // data[6] = quat[0]                                                      <L 2810>
                        var_87 = wp::extract(var_85, var_86);
                        wp::assign_inplace(var_14, var_88, var_87);
                        // data[7] = quat[1]                                                      <L 2811>
                        var_90 = wp::extract(var_85, var_89);
                        wp::assign_inplace(var_14, var_91, var_90);
                        // data[8] = quat[2]                                                      <L 2812>
                        var_93 = wp::extract(var_85, var_92);
                        wp::assign_inplace(var_14, var_94, var_93);
                        // data[9] = quat[3]                                                      <L 2813>
                        var_96 = wp::extract(var_85, var_95);
                        wp::assign_inplace(var_14, var_97, var_96);
                        // eq_data_out[eq_data_id, eqid] = data                                   <L 2814>
                        wp::array_store(var_eq_data_out, var_6, var_1, var_14);
                    }
                    var_98 = wp::where(var_84, var_85, var_81);
                    if (!var_84) {
                        // obj1id = eq_obj1id[eqid]                                               <L 2816>
                        var_99 = wp::address(var_eq_obj1id, var_1);
                        var_101 = wp::load(var_99);
                        var_100 = wp::copy(var_101);
                        // obj2id = eq_obj2id[eqid]                                               <L 2817>
                        var_102 = wp::address(var_eq_obj2id, var_1);
                        var_104 = wp::load(var_102);
                        var_103 = wp::copy(var_104);
                        // anchor2 = wp.vec3(data[0], data[1], data[2])                           <L 2820>
                        var_106 = wp::extract(var_14, var_105);
                        var_108 = wp::extract(var_14, var_107);
                        var_110 = wp::extract(var_14, var_109);
                        var_111 = wp::vec_t<3, wp::float32>(var_106, var_108, var_110);
                        // pos = xpos_in[worldid, obj2id] + xmat_in[worldid, obj2id] @ anchor2       <L 2821>
                        var_112 = wp::address(var_xpos_in, var_0, var_103);
                        var_113 = wp::address(var_xmat_in, var_0, var_103);
                        var_115 = wp::load(var_113);
                        var_114 = wp::mul(var_115, var_111);
                        var_117 = wp::load(var_112);
                        var_116 = wp::add(var_117, var_114);
                        // anchor1 = wp.transpose(xmat_in[worldid, obj1id]) @ (pos - xpos_in[worldid, obj1id])       <L 2824>
                        var_118 = wp::address(var_xmat_in, var_0, var_100);
                        var_120 = wp::load(var_118);
                        var_119 = wp::transpose(var_120);
                        var_121 = wp::address(var_xpos_in, var_0, var_100);
                        var_123 = wp::load(var_121);
                        var_122 = wp::sub(var_116, var_123);
                        var_124 = wp::mul(var_119, var_122);
                        // data[3] = anchor1[0]                                                   <L 2825>
                        var_126 = wp::extract(var_124, var_125);
                        wp::assign_inplace(var_14, var_127, var_126);
                        // data[4] = anchor1[1]                                                   <L 2826>
                        var_129 = wp::extract(var_124, var_128);
                        wp::assign_inplace(var_14, var_130, var_129);
                        // data[5] = anchor1[2]                                                   <L 2827>
                        var_132 = wp::extract(var_124, var_131);
                        wp::assign_inplace(var_14, var_133, var_132);
                        // relquat = mjmath.mul_quat(mjmath.quat_inv(xquat_in[worldid, obj1id]), xquat_in[worldid, obj2id])       <L 2830>
                        var_134 = wp::address(var_xquat_in, var_0, var_100);
                        var_136 = wp::load(var_134);
                        var_135 = quat_inv_0(var_136);
                        var_137 = wp::address(var_xquat_in, var_0, var_103);
                        var_139 = wp::load(var_137);
                        var_138 = mul_quat_0(var_135, var_139);
                        // data[6] = relquat[0]                                                   <L 2831>
                        var_141 = wp::extract(var_138, var_140);
                        wp::assign_inplace(var_14, var_142, var_141);
                        // data[7] = relquat[1]                                                   <L 2832>
                        var_144 = wp::extract(var_138, var_143);
                        wp::assign_inplace(var_14, var_145, var_144);
                        // data[8] = relquat[2]                                                   <L 2833>
                        var_147 = wp::extract(var_138, var_146);
                        wp::assign_inplace(var_14, var_148, var_147);
                        // data[9] = relquat[3]                                                   <L 2834>
                        var_150 = wp::extract(var_138, var_149);
                        wp::assign_inplace(var_14, var_151, var_150);
                        // eq_data_out[eq_data_id, eqid] = data                                   <L 2835>
                        wp::array_store(var_eq_data_out, var_6, var_1, var_14);
                    }
                    var_152 = wp::where(var_84, var_25, var_100);
                    var_153 = wp::where(var_84, var_28, var_103);
                    var_154 = wp::where(var_84, var_36, var_124);
                    var_155 = wp::where(var_84, var_41, var_116);
                    var_156 = wp::where(var_84, var_49, var_111);
                }
                var_157 = wp::where(var_72, var_152, var_25);
                var_158 = wp::where(var_72, var_153, var_28);
                var_159 = wp::where(var_72, var_154, var_36);
                var_160 = wp::where(var_72, var_155, var_41);
                var_161 = wp::where(var_72, var_156, var_49);
            }
            var_162 = wp::where(var_68, var_157, var_25);
            var_163 = wp::where(var_68, var_158, var_28);
            var_164 = wp::where(var_68, var_159, var_36);
            var_165 = wp::where(var_68, var_160, var_41);
            var_166 = wp::where(var_68, var_161, var_49);
        }
        var_167 = wp::where(var_19, var_25, var_162);
        var_168 = wp::where(var_19, var_28, var_163);
        var_169 = wp::where(var_19, var_36, var_164);
        var_170 = wp::where(var_19, var_41, var_165);
        var_171 = wp::where(var_19, var_49, var_166);
    }
}



extern "C" __global__ void _init_subtreemass_a34720d1_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_body_mass_in,
    wp::array_t<wp::float32> var_body_subtreemass_out)
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
        wp::float32* var_12;
        wp::float32 var_13;
        //---------
        // forward
        // def _init_subtreemass(                                                                 <L 2717>
        // worldid, bodyid = wp.tid()                                                             <L 2721>
        builtin_tid2d(var_0, var_1);
        // body_mass_id = worldid % body_mass_in.shape[0]                                         <L 2722>
        var_2 = &(var_body_mass_in.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // body_subtreemass_id = worldid % body_subtreemass_out.shape[0]                          <L 2723>
        var_7 = &(var_body_subtreemass_out.shape);
        var_10 = wp::load(var_7);
        var_9 = wp::extract(var_10, var_8);
        var_11 = wp::mod(var_0, var_9);
        // body_subtreemass_out[body_subtreemass_id, bodyid] = body_mass_in[body_mass_id, bodyid]       <L 2724>
        var_12 = wp::address(var_body_mass_in, var_6, var_1);
        var_13 = wp::load(var_12);
        wp::array_store(var_body_subtreemass_out, var_11, var_1, var_13);
    }
}



extern "C" __global__ void _compute_dof_M0_eaae0df3_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::float32> var_dof_armature,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::vec_t<10, wp::float32>> var_crb_in,
    wp::array_t<wp::float32> var_dof_M0_out)
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
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32 var_12;
        wp::vec_t<10, wp::float32>* var_13;
        wp::vec_t<6, wp::float32>* var_14;
        wp::vec_t<6, wp::float32> var_15;
        wp::vec_t<10, wp::float32> var_16;
        wp::vec_t<6, wp::float32> var_17;
        wp::vec_t<6, wp::float32>* var_18;
        wp::float32 var_19;
        wp::vec_t<6, wp::float32> var_20;
        wp::float32 var_21;
        //---------
        // forward
        // def _compute_dof_M0(                                                                   <L 3190>
        // worldid, dofid = wp.tid()                                                              <L 3197>
        builtin_tid2d(var_0, var_1);
        // bodyid = dof_bodyid[dofid]                                                             <L 3198>
        var_2 = wp::address(var_dof_bodyid, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // armature = dof_armature[worldid % dof_armature.shape[0], dofid]                        <L 3199>
        var_5 = &(var_dof_armature.shape);
        var_8 = wp::load(var_5);
        var_7 = wp::extract(var_8, var_6);
        var_9 = wp::mod(var_0, var_7);
        var_10 = wp::address(var_dof_armature, var_9, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // buf = mjmath.inert_vec(crb_in[worldid, bodyid], cdof_in[worldid, dofid])               <L 3200>
        var_13 = wp::address(var_crb_in, var_0, var_3);
        var_14 = wp::address(var_cdof_in, var_0, var_1);
        var_16 = wp::load(var_13);
        var_17 = wp::load(var_14);
        var_15 = inert_vec_0(var_16, var_17);
        // dof_M0_out[worldid, dofid] = armature + wp.dot(cdof_in[worldid, dofid], buf)           <L 3201>
        var_18 = wp::address(var_cdof_in, var_0, var_1);
        var_20 = wp::load(var_18);
        var_19 = wp::dot(var_20, var_15);
        var_21 = wp::add(var_11, var_19);
        wp::array_store(var_dof_M0_out, var_0, var_1, var_21);
    }
}



extern "C" __global__ void _copy_tendon_length0_5230a976_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_ten_length_in,
    wp::array_t<wp::float32> var_tendon_length0_out)
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
        //---------
        // forward
        // def _copy_tendon_length0(                                                              <L 2752>
        // worldid, tenid = wp.tid()                                                              <L 2756>
        builtin_tid2d(var_0, var_1);
        // tendon_length0_id = worldid % tendon_length0_out.shape[0]                              <L 2757>
        var_2 = &(var_tendon_length0_out.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // tendon_length0_out[tendon_length0_id, tenid] = ten_length_in[worldid, tenid]           <L 2758>
        var_7 = wp::address(var_ten_length_in, var_0, var_1);
        var_8 = wp::load(var_7);
        wp::array_store(var_tendon_length0_out, var_6, var_1, var_8);
    }
}



extern "C" __global__ void _set_length_range_a26be645_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_actuator_trntype,
    wp::array_t<wp::vec_t<2, wp::int32>> var_actuator_trnid,
    wp::array_t<wp::vec_t<6, wp::float32>> var_actuator_gear,
    wp::array_t<wp::int32> var_jnt_limited,
    wp::array_t<wp::vec_t<2, wp::float32>> var_jnt_range,
    wp::array_t<wp::int32> var_tendon_limited,
    wp::array_t<wp::vec_t<2, wp::float32>> var_tendon_range,
    wp::int32 var_ntendon,
    wp::array_t<wp::vec_t<2, wp::float32>> var_actuator_lengthrange_out)
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
        wp::vec_t<2, wp::int32>* var_5;
        const wp::int32 var_6 = 0;
        wp::int32 var_7;
        wp::vec_t<2, wp::int32> var_8;
        wp::shape_t* var_9;
        const wp::int32 var_10 = 0;
        wp::int32 var_11;
        wp::shape_t var_12;
        wp::int32 var_13;
        wp::vec_t<6, wp::float32>* var_14;
        const wp::int32 var_15 = 0;
        wp::float32 var_16;
        wp::vec_t<6, wp::float32> var_17;
        const wp::float32 var_18 = 0.0;
        const wp::float32 var_19 = 0.0;
        wp::vec_t<2, wp::float32> var_20;
        bool var_21;
        const wp::int32 var_22 = 0;
        bool var_23;
        const wp::int32 var_24 = 1;
        bool var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::shape_t* var_28;
        const wp::int32 var_29 = 0;
        wp::int32 var_30;
        wp::shape_t var_31;
        wp::int32 var_32;
        wp::vec_t<2, wp::float32>* var_33;
        wp::vec_t<2, wp::float32> var_34;
        wp::vec_t<2, wp::float32> var_35;
        const wp::float32 var_36 = 0.0;
        bool var_37;
        const wp::int32 var_38 = 0;
        wp::float32 var_39;
        wp::float32 var_40;
        const wp::int32 var_41 = 1;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::vec_t<2, wp::float32> var_44;
        wp::vec_t<2, wp::float32> var_45;
        const wp::int32 var_46 = 1;
        wp::float32 var_47;
        wp::float32 var_48;
        const wp::int32 var_49 = 0;
        wp::float32 var_50;
        wp::float32 var_51;
        wp::vec_t<2, wp::float32> var_52;
        wp::vec_t<2, wp::float32> var_53;
        wp::int32 var_54;
        wp::vec_t<2, wp::float32> var_55;
        wp::int32 var_56;
        wp::vec_t<2, wp::float32> var_57;
        const wp::int32 var_58 = 3;
        bool var_59;
        bool var_60;
        const wp::int32 var_61 = 0;
        bool var_62;
        wp::int32* var_63;
        wp::int32 var_64;
        wp::shape_t* var_65;
        const wp::int32 var_66 = 0;
        wp::int32 var_67;
        wp::shape_t var_68;
        wp::int32 var_69;
        wp::vec_t<2, wp::float32>* var_70;
        wp::vec_t<2, wp::float32> var_71;
        wp::vec_t<2, wp::float32> var_72;
        const wp::float32 var_73 = 0.0;
        bool var_74;
        const wp::int32 var_75 = 0;
        wp::float32 var_76;
        wp::float32 var_77;
        const wp::int32 var_78 = 1;
        wp::float32 var_79;
        wp::float32 var_80;
        wp::vec_t<2, wp::float32> var_81;
        wp::vec_t<2, wp::float32> var_82;
        const wp::int32 var_83 = 1;
        wp::float32 var_84;
        wp::float32 var_85;
        const wp::int32 var_86 = 0;
        wp::float32 var_87;
        wp::float32 var_88;
        wp::vec_t<2, wp::float32> var_89;
        wp::vec_t<2, wp::float32> var_90;
        wp::vec_t<2, wp::float32> var_91;
        wp::vec_t<2, wp::float32> var_92;
        wp::vec_t<2, wp::float32> var_93;
        wp::vec_t<2, wp::float32> var_94;
        wp::vec_t<2, wp::float32> var_95;
        wp::vec_t<2, wp::float32> var_96;
        //---------
        // forward
        // def _set_length_range(                                                                 <L 3256>
        // worldid, actid = wp.tid()                                                              <L 3267>
        builtin_tid2d(var_0, var_1);
        // trntype = actuator_trntype[actid]                                                      <L 3268>
        var_2 = wp::address(var_actuator_trntype, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // id0 = actuator_trnid[actid][0]                                                         <L 3269>
        var_5 = wp::address(var_actuator_trnid, var_1);
        var_8 = wp::load(var_5);
        var_7 = wp::extract(var_8, var_6);
        // gear0 = actuator_gear[worldid % actuator_gear.shape[0], actid][0]                      <L 3270>
        var_9 = &(var_actuator_gear.shape);
        var_12 = wp::load(var_9);
        var_11 = wp::extract(var_12, var_10);
        var_13 = wp::mod(var_0, var_11);
        var_14 = wp::address(var_actuator_gear, var_13, var_1);
        var_17 = wp::load(var_14);
        var_16 = wp::extract(var_17, var_15);
        // lr = wp.vec2(0.0, 0.0)                                                                 <L 3272>
        var_20 = wp::vec_t<2, wp::float32>(var_18, var_19);
        // if trntype == TrnType.JOINT or trntype == TrnType.JOINTINPARENT:                       <L 3274>
        var_23 = (var_3 == var_22);
        var_21 = var_23;
        if (!var_21) {
            var_25 = (var_3 == var_24);
            var_21 = var_21 || var_25;
        }
        if (var_21) {
            // if jnt_limited[id0]:                                                               <L 3275>
            var_26 = wp::address(var_jnt_limited, var_7);
            var_27 = wp::load(var_26);
            if (var_27) {
                // rng = jnt_range[worldid % jnt_range.shape[0], id0]                             <L 3276>
                var_28 = &(var_jnt_range.shape);
                var_31 = wp::load(var_28);
                var_30 = wp::extract(var_31, var_29);
                var_32 = wp::mod(var_0, var_30);
                var_33 = wp::address(var_jnt_range, var_32, var_7);
                var_35 = wp::load(var_33);
                var_34 = wp::copy(var_35);
                // if gear0 > 0.0:                                                                <L 3277>
                var_37 = (var_16 > var_36);
                if (var_37) {
                    // lr = wp.vec2(rng[0] * gear0, rng[1] * gear0)                               <L 3278>
                    var_39 = wp::extract(var_34, var_38);
                    var_40 = wp::mul(var_39, var_16);
                    var_42 = wp::extract(var_34, var_41);
                    var_43 = wp::mul(var_42, var_16);
                    var_44 = wp::vec_t<2, wp::float32>(var_40, var_43);
                }
                var_45 = wp::where(var_37, var_44, var_20);
                if (!var_37) {
                    // lr = wp.vec2(rng[1] * gear0, rng[0] * gear0)                               <L 3280>
                    var_47 = wp::extract(var_34, var_46);
                    var_48 = wp::mul(var_47, var_16);
                    var_50 = wp::extract(var_34, var_49);
                    var_51 = wp::mul(var_50, var_16);
                    var_52 = wp::vec_t<2, wp::float32>(var_48, var_51);
                }
                var_53 = wp::where(var_37, var_45, var_52);
            }
            var_54 = wp::load(var_26);
            var_56 = wp::load(var_26);
            var_55 = wp::where(var_56, var_53, var_20);
        }
        var_57 = wp::where(var_21, var_55, var_20);
        if (!var_21) {
            // elif trntype == TrnType.TENDON:                                                    <L 3281>
            var_59 = (var_3 == var_58);
            if (var_59) {
                // if ntendon > 0 and tendon_limited[id0]:                                        <L 3282>
                var_62 = (var_ntendon > var_61);
                var_60 = var_62;
                if (var_60) {
                    var_63 = wp::address(var_tendon_limited, var_7);
                    var_64 = wp::load(var_63);
                    var_60 = var_60 && var_64;
                }
                if (var_60) {
                    // rng = tendon_range[worldid % tendon_range.shape[0], id0]                   <L 3283>
                    var_65 = &(var_tendon_range.shape);
                    var_68 = wp::load(var_65);
                    var_67 = wp::extract(var_68, var_66);
                    var_69 = wp::mod(var_0, var_67);
                    var_70 = wp::address(var_tendon_range, var_69, var_7);
                    var_72 = wp::load(var_70);
                    var_71 = wp::copy(var_72);
                    // if gear0 > 0.0:                                                            <L 3284>
                    var_74 = (var_16 > var_73);
                    if (var_74) {
                        // lr = wp.vec2(rng[0] * gear0, rng[1] * gear0)                           <L 3285>
                        var_76 = wp::extract(var_71, var_75);
                        var_77 = wp::mul(var_76, var_16);
                        var_79 = wp::extract(var_71, var_78);
                        var_80 = wp::mul(var_79, var_16);
                        var_81 = wp::vec_t<2, wp::float32>(var_77, var_80);
                    }
                    var_82 = wp::where(var_74, var_81, var_57);
                    if (!var_74) {
                        // lr = wp.vec2(rng[1] * gear0, rng[0] * gear0)                           <L 3287>
                        var_84 = wp::extract(var_71, var_83);
                        var_85 = wp::mul(var_84, var_16);
                        var_87 = wp::extract(var_71, var_86);
                        var_88 = wp::mul(var_87, var_16);
                        var_89 = wp::vec_t<2, wp::float32>(var_85, var_88);
                    }
                    var_90 = wp::where(var_74, var_82, var_89);
                }
                var_91 = wp::where(var_60, var_90, var_57);
                var_92 = wp::where(var_60, var_71, var_34);
            }
            var_93 = wp::where(var_59, var_91, var_57);
            var_94 = wp::where(var_59, var_92, var_34);
        }
        var_95 = wp::where(var_21, var_57, var_93);
        var_96 = wp::where(var_21, var_34, var_94);
        // actuator_lengthrange_out[worldid, actid] = lr                                          <L 3289>
        wp::array_store(var_actuator_lengthrange_out, var_0, var_1, var_95);
    }
}



extern "C" __global__ void _set_unit_vector_f74e5cf9_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_dofid_target,
    wp::array_t<wp::float32> var_unit_vec_out)
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
        wp::shape_t* var_1;
        const wp::int32 var_2 = 1;
        wp::int32 var_3;
        wp::shape_t var_4;
        wp::range_t var_5;
        wp::int32 var_6;
        bool var_7;
        const wp::float32 var_8 = 1.0;
        const wp::float32 var_9 = 0.0;
        //---------
        // forward
        // def _set_unit_vector(                                                                  <L 2876>
        // worldid = wp.tid()                                                                     <L 2880>
        var_0 = builtin_tid1d();
        // nv = unit_vec_out.shape[1]                                                             <L 2881>
        var_1 = &(var_unit_vec_out.shape);
        var_4 = wp::load(var_1);
        var_3 = wp::extract(var_4, var_2);
        // for i in range(nv):                                                                    <L 2882>
        var_5 = wp::range(var_3);
        start_for_0:;
            if (iter_cmp(var_5) == 0) goto end_for_0;
            var_6 = wp::iter_next(var_5);
            // if i == dofid_target:                                                              <L 2883>
            var_7 = (var_6 == var_dofid_target);
            if (var_7) {
                // unit_vec_out[worldid, i] = 1.0                                                 <L 2884>
                wp::array_store(var_unit_vec_out, var_0, var_6, var_8);
            }
            if (!var_7) {
                // unit_vec_out[worldid, i] = 0.0                                                 <L 2886>
                wp::array_store(var_unit_vec_out, var_0, var_6, var_9);
            }
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void _compute_cam_pos0_5993818e_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_cam_bodyid,
    wp::array_t<wp::int32> var_cam_targetbodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cam_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_pos0_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_poscom0_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cam_mat0_out)
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
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::vec_t<3, wp::float32>* var_13;
        wp::vec_t<3, wp::float32> var_14;
        wp::vec_t<3, wp::float32> var_15;
        wp::vec_t<3, wp::float32>* var_16;
        wp::vec_t<3, wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        const wp::int32 var_19 = 0;
        bool var_20;
        wp::vec_t<3, wp::float32>* var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32>* var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::vec_t<3, wp::float32> var_26;
        wp::mat_t<3, 3, wp::float32>* var_27;
        wp::mat_t<3, 3, wp::float32> var_28;
        //---------
        // forward
        // def _compute_cam_pos0(                                                                 <L 3103>
        // worldid, camid = wp.tid()                                                              <L 3114>
        builtin_tid2d(var_0, var_1);
        // cam_pos0_id = worldid % cam_pos0_out.shape[0]                                          <L 3115>
        var_2 = &(var_cam_pos0_out.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // bodyid = cam_bodyid[camid]                                                             <L 3116>
        var_7 = wp::address(var_cam_bodyid, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // targetid = cam_targetbodyid[camid]                                                     <L 3117>
        var_10 = wp::address(var_cam_targetbodyid, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // cam_xpos = cam_xpos_in[worldid, camid]                                                 <L 3118>
        var_13 = wp::address(var_cam_xpos_in, var_0, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // cam_pos0_out[cam_pos0_id, camid] = cam_xpos - xpos_in[worldid, bodyid]                 <L 3120>
        var_16 = wp::address(var_xpos_in, var_0, var_8);
        var_18 = wp::load(var_16);
        var_17 = wp::sub(var_14, var_18);
        wp::array_store(var_cam_pos0_out, var_6, var_1, var_17);
        // if targetid >= 0:                                                                      <L 3121>
        var_20 = (var_11 >= var_19);
        if (var_20) {
            // cam_poscom0_out[cam_pos0_id, camid] = cam_xpos - subtree_com_in[worldid, targetid]       <L 3122>
            var_21 = wp::address(var_subtree_com_in, var_0, var_11);
            var_23 = wp::load(var_21);
            var_22 = wp::sub(var_14, var_23);
            wp::array_store(var_cam_poscom0_out, var_6, var_1, var_22);
        }
        if (!var_20) {
            // cam_poscom0_out[cam_pos0_id, camid] = cam_xpos - subtree_com_in[worldid, bodyid]       <L 3124>
            var_24 = wp::address(var_subtree_com_in, var_0, var_8);
            var_26 = wp::load(var_24);
            var_25 = wp::sub(var_14, var_26);
            wp::array_store(var_cam_poscom0_out, var_6, var_1, var_25);
        }
        // cam_mat0_out[cam_pos0_id, camid] = cam_xmat_in[worldid, camid]                         <L 3125>
        var_27 = wp::address(var_cam_xmat_in, var_0, var_1);
        var_28 = wp::load(var_27);
        wp::array_store(var_cam_mat0_out, var_6, var_1, var_28);
    }
}



extern "C" __global__ void _extract_dof_A_diag_917a07a1_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_dofid,
    wp::array_t<wp::float32> var_result_vec_in,
    wp::array_t<wp::float32> var_dof_A_diag_out)
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
        wp::shape_t* var_1;
        const wp::int32 var_2 = 0;
        wp::int32 var_3;
        wp::shape_t var_4;
        wp::int32 var_5;
        wp::float32* var_6;
        wp::float32 var_7;
        //---------
        // forward
        // def _extract_dof_A_diag(                                                               <L 2890>
        // worldid = wp.tid()                                                                     <L 2895>
        var_0 = builtin_tid1d();
        // dof_A_diag_id = worldid % dof_A_diag_out.shape[0]                                      <L 2896>
        var_1 = &(var_dof_A_diag_out.shape);
        var_4 = wp::load(var_1);
        var_3 = wp::extract(var_4, var_2);
        var_5 = wp::mod(var_0, var_3);
        // dof_A_diag_out[dof_A_diag_id, dofid] = result_vec_in[worldid, dofid]                   <L 2897>
        var_6 = wp::address(var_result_vec_in, var_0, var_dofid);
        var_7 = wp::load(var_6);
        wp::array_store(var_dof_A_diag_out, var_5, var_dofid, var_7);
    }
}



extern "C" __global__ void _compute_body_jac_row_6eb5882a_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_nv,
    wp::int32 var_bodyid_target,
    wp::int32 var_row_idx,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_dof_parentid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::float32> var_body_jac_row_out)
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
        wp::range_t var_1;
        wp::int32 var_2;
        const wp::float32 var_3 = 0.0;
        wp::int32 var_4;
        bool var_5;
        const wp::int32 var_6 = 0;
        bool var_7;
        wp::int32* var_8;
        const wp::int32 var_9 = 0;
        bool var_10;
        wp::int32 var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        const wp::int32 var_15 = 0;
        bool var_16;
        wp::vec_t<3, wp::float32>* var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::int32* var_20;
        wp::vec_t<3, wp::float32>* var_21;
        wp::int32 var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32> var_24;
        wp::int32* var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        const wp::int32 var_30 = 1;
        wp::int32 var_31;
        const wp::int32 var_32 = 0;
        bool var_33;
        wp::vec_t<6, wp::float32>* var_34;
        wp::vec_t<6, wp::float32> var_35;
        wp::vec_t<6, wp::float32> var_36;
        wp::vec_t<3, wp::float32> var_37;
        wp::vec_t<3, wp::float32> var_38;
        const wp::int32 var_39 = 3;
        bool var_40;
        wp::vec_t<3, wp::float32> var_41;
        const wp::int32 var_42 = 0;
        bool var_43;
        const wp::int32 var_44 = 0;
        wp::float32 var_45;
        const wp::int32 var_46 = 0;
        wp::float32 var_47;
        wp::float32 var_48;
        const wp::int32 var_49 = 1;
        bool var_50;
        const wp::int32 var_51 = 1;
        wp::float32 var_52;
        const wp::int32 var_53 = 1;
        wp::float32 var_54;
        wp::float32 var_55;
        const wp::int32 var_56 = 2;
        wp::float32 var_57;
        const wp::int32 var_58 = 2;
        wp::float32 var_59;
        wp::float32 var_60;
        const wp::int32 var_61 = 3;
        bool var_62;
        const wp::int32 var_63 = 0;
        wp::float32 var_64;
        const wp::int32 var_65 = 4;
        bool var_66;
        const wp::int32 var_67 = 1;
        wp::float32 var_68;
        const wp::int32 var_69 = 2;
        wp::float32 var_70;
        wp::int32* var_71;
        wp::int32 var_72;
        wp::int32 var_73;
        //---------
        // forward
        // def _compute_body_jac_row(                                                             <L 2945>
        // worldid = wp.tid()                                                                     <L 2959>
        var_0 = builtin_tid1d();
        // for i in range(nv):                                                                    <L 2961>
        var_1 = wp::range(var_nv);
        start_for_0:;
            if (iter_cmp(var_1) == 0) goto end_for_0;
            var_2 = wp::iter_next(var_1);
            // body_jac_row_out[worldid, i] = 0.0                                                 <L 2962>
            wp::array_store(var_body_jac_row_out, var_0, var_2, var_3);
            goto start_for_0;
        end_for_0:;
        // bodyid = bodyid_target                                                                 <L 2964>
        var_4 = wp::copy(var_bodyid_target);
        // while bodyid > 0 and body_dofnum[bodyid] == 0:                                         <L 2965>
        start_while_2:;
        var_7 = (var_4 > var_6);
        var_5 = var_7;
        if (var_5) {
            var_8 = wp::address(var_body_dofnum, var_4);
            var_11 = wp::load(var_8);
            var_10 = (var_11 == var_9);
            var_5 = var_5 && var_10;
        }
        if ((var_5) == false) goto end_while_2;
            // bodyid = body_parentid[bodyid]                                                     <L 2966>
            var_12 = wp::address(var_body_parentid, var_4);
            var_14 = wp::load(var_12);
            var_13 = wp::copy(var_14);
            wp::assign(var_4, var_13);
        goto start_while_2;
        end_while_2:;
        // if bodyid == 0:                                                                        <L 2968>
        var_16 = (var_4 == var_15);
        if (var_16) {
            // return                                                                             <L 2969>
            continue;
        }
        // point = xipos_in[worldid, bodyid_target]                                               <L 2972>
        var_17 = wp::address(var_xipos_in, var_0, var_bodyid_target);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // offset = point - subtree_com_in[worldid, body_rootid[bodyid_target]]                   <L 2973>
        var_20 = wp::address(var_body_rootid, var_bodyid_target);
        var_22 = wp::load(var_20);
        var_21 = wp::address(var_subtree_com_in, var_0, var_22);
        var_24 = wp::load(var_21);
        var_23 = wp::sub(var_18, var_24);
        // dofid = body_dofadr[bodyid] + body_dofnum[bodyid] - 1                                  <L 2976>
        var_25 = wp::address(var_body_dofadr, var_4);
        var_26 = wp::address(var_body_dofnum, var_4);
        var_28 = wp::load(var_25);
        var_29 = wp::load(var_26);
        var_27 = wp::add(var_28, var_29);
        var_31 = wp::sub(var_27, var_30);
        // while dofid >= 0:                                                                      <L 2979>
        start_while_5:;
        var_33 = (var_31 >= var_32);
        if ((var_33) == false) goto end_while_5;
            // cdof = cdof_in[worldid, dofid]                                                     <L 2980>
            var_34 = wp::address(var_cdof_in, var_0, var_31);
            var_36 = wp::load(var_34);
            var_35 = wp::copy(var_36);
            // cdof_ang = wp.spatial_top(cdof)                                                    <L 2981>
            var_37 = wp::spatial_top(var_35);
            // cdof_lin = wp.spatial_bottom(cdof)                                                 <L 2982>
            var_38 = wp::spatial_bottom(var_35);
            // if row_idx < 3:                                                                    <L 2984>
            var_40 = (var_row_idx < var_39);
            if (var_40) {
                // tmp = wp.cross(cdof_ang, offset)                                               <L 2985>
                var_41 = wp::cross(var_37, var_23);
                // if row_idx == 0:                                                               <L 2986>
                var_43 = (var_row_idx == var_42);
                if (var_43) {
                    // body_jac_row_out[worldid, dofid] = cdof_lin[0] + tmp[0]                    <L 2987>
                    var_45 = wp::extract(var_38, var_44);
                    var_47 = wp::extract(var_41, var_46);
                    var_48 = wp::add(var_45, var_47);
                    wp::array_store(var_body_jac_row_out, var_0, var_31, var_48);
                }
                if (!var_43) {
                    // elif row_idx == 1:                                                         <L 2988>
                    var_50 = (var_row_idx == var_49);
                    if (var_50) {
                        // body_jac_row_out[worldid, dofid] = cdof_lin[1] + tmp[1]                <L 2989>
                        var_52 = wp::extract(var_38, var_51);
                        var_54 = wp::extract(var_41, var_53);
                        var_55 = wp::add(var_52, var_54);
                        wp::array_store(var_body_jac_row_out, var_0, var_31, var_55);
                    }
                    if (!var_50) {
                        // body_jac_row_out[worldid, dofid] = cdof_lin[2] + tmp[2]                <L 2991>
                        var_57 = wp::extract(var_38, var_56);
                        var_59 = wp::extract(var_41, var_58);
                        var_60 = wp::add(var_57, var_59);
                        wp::array_store(var_body_jac_row_out, var_0, var_31, var_60);
                    }
                }
            }
            if (!var_40) {
                // if row_idx == 3:                                                               <L 2993>
                var_62 = (var_row_idx == var_61);
                if (var_62) {
                    // body_jac_row_out[worldid, dofid] = cdof_ang[0]                             <L 2994>
                    var_64 = wp::extract(var_37, var_63);
                    wp::array_store(var_body_jac_row_out, var_0, var_31, var_64);
                }
                if (!var_62) {
                    // elif row_idx == 4:                                                         <L 2995>
                    var_66 = (var_row_idx == var_65);
                    if (var_66) {
                        // body_jac_row_out[worldid, dofid] = cdof_ang[1]                         <L 2996>
                        var_68 = wp::extract(var_37, var_67);
                        wp::array_store(var_body_jac_row_out, var_0, var_31, var_68);
                    }
                    if (!var_66) {
                        // body_jac_row_out[worldid, dofid] = cdof_ang[2]                         <L 2998>
                        var_70 = wp::extract(var_37, var_69);
                        wp::array_store(var_body_jac_row_out, var_0, var_31, var_70);
                    }
                }
            }
            // dofid = dof_parentid[dofid]                                                        <L 3000>
            var_71 = wp::address(var_dof_parentid, var_31);
            var_73 = wp::load(var_71);
            var_72 = wp::copy(var_73);
            wp::assign(var_31, var_72);
        goto start_while_5;
        end_while_5:;
    }
}



extern "C" __global__ void _compute_meaninertia_af08211b_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_nv,
    wp::array_t<wp::int32> var_M_rownnz_in,
    wp::array_t<wp::int32> var_M_rowadr_in,
    wp::array_t<wp::float32> var_M_in,
    wp::array_t<wp::float32> var_meaninertia_out)
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
        bool var_2;
        const wp::float32 var_3 = 1.0;
        wp::shape_t* var_4;
        const wp::int32 var_5 = 0;
        wp::int32 var_6;
        wp::shape_t var_7;
        wp::int32 var_8;
        const wp::float32 var_9 = 0.0;
        wp::float32 var_10;
        wp::range_t var_11;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        const wp::int32 var_18 = 1;
        wp::int32 var_19;
        wp::float32* var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        wp::shape_t* var_25;
        const wp::int32 var_26 = 0;
        wp::int32 var_27;
        wp::shape_t var_28;
        wp::int32 var_29;
        //---------
        // forward
        // def _compute_meaninertia(                                                              <L 2852>
        // worldid = wp.tid()                                                                     <L 2860>
        var_0 = builtin_tid1d();
        // if nv == 0:                                                                            <L 2862>
        var_2 = (var_nv == var_1);
        if (var_2) {
            // meaninertia_out[worldid % meaninertia_out.shape[0]] = 1.0  # Default from MuJoCo       <L 2863>
            var_4 = &(var_meaninertia_out.shape);
            var_7 = wp::load(var_4);
            var_6 = wp::extract(var_7, var_5);
            var_8 = wp::mod(var_0, var_6);
            wp::array_store(var_meaninertia_out, var_8, var_3);
            // return                                                                             <L 2864>
            continue;
        }
        // total = float(0.0)                                                                     <L 2866>
        var_10 = wp::float(var_9);
        // for i in range(nv):                                                                    <L 2867>
        var_11 = wp::range(var_nv);
        start_for_1:;
            if (iter_cmp(var_11) == 0) goto end_for_1;
            var_12 = wp::iter_next(var_11);
            // madr = M_rowadr_in[i] + M_rownnz_in[i] - 1                                         <L 2869>
            var_13 = wp::address(var_M_rowadr_in, var_12);
            var_14 = wp::address(var_M_rownnz_in, var_12);
            var_16 = wp::load(var_13);
            var_17 = wp::load(var_14);
            var_15 = wp::add(var_16, var_17);
            var_19 = wp::sub(var_15, var_18);
            // total += M_in[worldid, madr]                                                       <L 2870>
            var_20 = wp::address(var_M_in, var_0, var_19);
            var_22 = wp::load(var_20);
            var_21 = wp::add(var_10, var_22);
            wp::assign(var_10, var_21);
            goto start_for_1;
        end_for_1:;
        // meaninertia_out[worldid % meaninertia_out.shape[0]] = total / float(nv)                <L 2872>
        var_23 = wp::float(var_nv);
        var_24 = wp::div(var_10, var_23);
        var_25 = &(var_meaninertia_out.shape);
        var_28 = wp::load(var_25);
        var_27 = wp::extract(var_28, var_26);
        var_29 = wp::mod(var_0, var_27);
        wp::array_store(var_meaninertia_out, var_29, var_24);
    }
}



extern "C" __global__ void _compute_light_pos0_a2f2ed64_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_light_bodyid,
    wp::array_t<wp::int32> var_light_targetbodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_xdir_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_pos0_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_poscom0_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_dir0_out)
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
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::vec_t<3, wp::float32>* var_13;
        wp::vec_t<3, wp::float32> var_14;
        wp::vec_t<3, wp::float32> var_15;
        wp::vec_t<3, wp::float32>* var_16;
        wp::vec_t<3, wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        const wp::int32 var_19 = 0;
        bool var_20;
        wp::vec_t<3, wp::float32>* var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32>* var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::vec_t<3, wp::float32> var_26;
        wp::vec_t<3, wp::float32>* var_27;
        wp::vec_t<3, wp::float32> var_28;
        //---------
        // forward
        // def _compute_light_pos0(                                                               <L 3129>
        // worldid, lightid = wp.tid()                                                            <L 3140>
        builtin_tid2d(var_0, var_1);
        // light_pos0_id = worldid % light_pos0_out.shape[0]                                      <L 3141>
        var_2 = &(var_light_pos0_out.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // bodyid = light_bodyid[lightid]                                                         <L 3142>
        var_7 = wp::address(var_light_bodyid, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // targetid = light_targetbodyid[lightid]                                                 <L 3143>
        var_10 = wp::address(var_light_targetbodyid, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // light_xpos = light_xpos_in[worldid, lightid]                                           <L 3144>
        var_13 = wp::address(var_light_xpos_in, var_0, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // light_pos0_out[light_pos0_id, lightid] = light_xpos - xpos_in[worldid, bodyid]         <L 3146>
        var_16 = wp::address(var_xpos_in, var_0, var_8);
        var_18 = wp::load(var_16);
        var_17 = wp::sub(var_14, var_18);
        wp::array_store(var_light_pos0_out, var_6, var_1, var_17);
        // if targetid >= 0:                                                                      <L 3147>
        var_20 = (var_11 >= var_19);
        if (var_20) {
            // light_poscom0_out[light_pos0_id, lightid] = light_xpos - subtree_com_in[worldid, targetid]       <L 3148>
            var_21 = wp::address(var_subtree_com_in, var_0, var_11);
            var_23 = wp::load(var_21);
            var_22 = wp::sub(var_14, var_23);
            wp::array_store(var_light_poscom0_out, var_6, var_1, var_22);
        }
        if (!var_20) {
            // light_poscom0_out[light_pos0_id, lightid] = light_xpos - subtree_com_in[worldid, bodyid]       <L 3150>
            var_24 = wp::address(var_subtree_com_in, var_0, var_8);
            var_26 = wp::load(var_24);
            var_25 = wp::sub(var_14, var_26);
            wp::array_store(var_light_poscom0_out, var_6, var_1, var_25);
        }
        // light_dir0_out[light_pos0_id, lightid] = light_xdir_in[worldid, lightid]               <L 3151>
        var_27 = wp::address(var_light_xdir_in, var_0, var_1);
        var_28 = wp::load(var_27);
        wp::array_store(var_light_dir0_out, var_6, var_1, var_28);
    }
}



extern "C" __global__ void _copy_actuator_moment_4af7079d_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_actid_target,
    wp::array_t<wp::int32> var_moment_rownnz_in,
    wp::array_t<wp::int32> var_moment_rowadr_in,
    wp::array_t<wp::int32> var_moment_colind_in,
    wp::array_t<wp::float32> var_actuator_moment_in,
    wp::array_t<wp::float32> var_act_moment_vec_out)
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
        wp::shape_t* var_1;
        const wp::int32 var_2 = 1;
        wp::int32 var_3;
        wp::shape_t var_4;
        wp::range_t var_5;
        wp::int32 var_6;
        const wp::float32 var_7 = 0.0;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::range_t var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::float32* var_20;
        wp::float32 var_21;
        //---------
        // forward
        // def _copy_actuator_moment(                                                             <L 3155>
        // worldid = wp.tid()                                                                     <L 3163>
        var_0 = builtin_tid1d();
        // nv = act_moment_vec_out.shape[1]                                                       <L 3164>
        var_1 = &(var_act_moment_vec_out.shape);
        var_4 = wp::load(var_1);
        var_3 = wp::extract(var_4, var_2);
        // for i in range(nv):                                                                    <L 3165>
        var_5 = wp::range(var_3);
        start_for_0:;
            if (iter_cmp(var_5) == 0) goto end_for_0;
            var_6 = wp::iter_next(var_5);
            // act_moment_vec_out[worldid, i] = 0.0                                               <L 3166>
            wp::array_store(var_act_moment_vec_out, var_0, var_6, var_7);
            goto start_for_0;
        end_for_0:;
        // rownnz = moment_rownnz_in[worldid, actid_target]                                       <L 3167>
        var_8 = wp::address(var_moment_rownnz_in, var_0, var_actid_target);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // rowadr = moment_rowadr_in[worldid, actid_target]                                       <L 3168>
        var_11 = wp::address(var_moment_rowadr_in, var_0, var_actid_target);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // for i in range(rownnz):                                                                <L 3169>
        var_14 = wp::range(var_9);
        start_for_2:;
            if (iter_cmp(var_14) == 0) goto end_for_2;
            var_15 = wp::iter_next(var_14);
            // sparseid = rowadr + i                                                              <L 3170>
            var_16 = wp::add(var_12, var_15);
            // col = moment_colind_in[worldid, sparseid]                                          <L 3171>
            var_17 = wp::address(var_moment_colind_in, var_0, var_16);
            var_19 = wp::load(var_17);
            var_18 = wp::copy(var_19);
            // act_moment_vec_out[worldid, col] = actuator_moment_in[worldid, sparseid]           <L 3172>
            var_20 = wp::address(var_actuator_moment_in, var_0, var_16);
            var_21 = wp::load(var_20);
            wp::array_store(var_act_moment_vec_out, var_0, var_18, var_21);
            goto start_for_2;
        end_for_2:;
    }
}



extern "C" __global__ void _copy_qpos0_to_qpos_32739b15_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_qpos0,
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
        //---------
        // forward
        // def _copy_qpos0_to_qpos(                                                               <L 2742>
        // worldid, i = wp.tid()                                                                  <L 2746>
        builtin_tid2d(var_0, var_1);
        // qpos0_id = worldid % qpos0.shape[0]                                                    <L 2747>
        var_2 = &(var_qpos0.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // qpos_out[worldid, i] = qpos0[qpos0_id, i]                                              <L 2748>
        var_7 = wp::address(var_qpos0, var_6, var_1);
        var_8 = wp::load(var_7);
        wp::array_store(var_qpos_out, var_0, var_1, var_8);
    }
}



extern "C" __global__ void _resolve_tendon_lengthspring_a988a292_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_ten_length_in,
    wp::array_t<wp::vec_t<2, wp::float32>> var_tendon_lengthspring_out)
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
        wp::vec_t<2, wp::float32>* var_7;
        wp::vec_t<2, wp::float32> var_8;
        wp::vec_t<2, wp::float32> var_9;
        bool var_10;
        const wp::int32 var_11 = 0;
        wp::float32 var_12;
        const wp::float32 var_13 = -1.0;
        bool var_14;
        const wp::int32 var_15 = 1;
        wp::float32 var_16;
        const wp::float32 var_17 = -1.0;
        bool var_18;
        wp::float32* var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::vec_t<2, wp::float32> var_22;
        //---------
        // forward
        // def _resolve_tendon_lengthspring(                                                      <L 2839>
        // worldid, tenid = wp.tid()                                                              <L 2843>
        builtin_tid2d(var_0, var_1);
        // tendon_lengthspring_id = worldid % tendon_lengthspring_out.shape[0]                    <L 2844>
        var_2 = &(var_tendon_lengthspring_out.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // val = tendon_lengthspring_out[tendon_lengthspring_id, tenid]                           <L 2845>
        var_7 = wp::address(var_tendon_lengthspring_out, var_6, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // if val[0] == -1.0 and val[1] == -1.0:                                                  <L 2846>
        var_12 = wp::extract(var_8, var_11);
        var_14 = (var_12 == var_13);
        var_10 = var_14;
        if (var_10) {
            var_16 = wp::extract(var_8, var_15);
            var_18 = (var_16 == var_17);
            var_10 = var_10 && var_18;
        }
        if (var_10) {
            // l = ten_length_in[worldid, tenid]                                                  <L 2847>
            var_19 = wp::address(var_ten_length_in, var_0, var_1);
            var_21 = wp::load(var_19);
            var_20 = wp::copy(var_21);
            // tendon_lengthspring_out[tendon_lengthspring_id, tenid] = wp.vec2(l, l)             <L 2848>
            var_22 = wp::vec_t<2, wp::float32>(var_20, var_20);
            wp::array_store(var_tendon_lengthspring_out, var_6, var_1, var_22);
        }
    }
}



extern "C" __global__ void _build_rays_5e1b397e_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_offset,
    wp::int32 var_img_w,
    wp::int32 var_img_h,
    wp::int32 var_projection,
    wp::float32 var_fovy,
    wp::vec_t<2, wp::float32> var_sensorsize,
    wp::vec_t<4, wp::float32> var_intrinsic,
    wp::float32 var_znear,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ray_out)
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
        wp::vec_t<3, wp::float32> var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        //---------
        // forward
        // def _build_rays(                                                                       <L 3835>
        // xid, yid = wp.tid()                                                                    <L 3848>
        builtin_tid2d(var_0, var_1);
        // ray_out[offset + xid + yid * img_w] = render_util.compute_ray(                         <L 3849>
        // projection, fovy, sensorsize, intrinsic, img_w, img_h, xid, yid, znear                 <L 3850>
        var_2 = compute_ray_0(var_projection, var_fovy, var_sensorsize, var_intrinsic, var_img_w, var_img_h, var_0, var_1, var_znear);
        // ray_out[offset + xid + yid * img_w] = render_util.compute_ray(                         <L 3849>
        var_3 = wp::add(var_offset, var_0);
        var_4 = wp::mul(var_1, var_img_w);
        var_5 = wp::add(var_3, var_4);
        wp::array_store(var_ray_out, var_5, var_2);
    }
}



extern "C" __global__ void _compute_actuator_acc0_a0c3fa27_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_actid_target,
    wp::int32 var_nv,
    wp::array_t<wp::float32> var_result_vec_in,
    wp::array_t<wp::float32> var_actuator_acc0_out)
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
        wp::float32 var_2;
        wp::range_t var_3;
        wp::int32 var_4;
        wp::float32* var_5;
        wp::float32* var_6;
        wp::float32 var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32 var_11;
        //---------
        // forward
        // def _compute_actuator_acc0(                                                            <L 3176>
        // worldid = wp.tid()                                                                     <L 3182>
        var_0 = builtin_tid1d();
        // norm_sq = float(0.0)                                                                   <L 3183>
        var_2 = wp::float(var_1);
        // for i in range(nv):                                                                    <L 3184>
        var_3 = wp::range(var_nv);
        start_for_0:;
            if (iter_cmp(var_3) == 0) goto end_for_0;
            var_4 = wp::iter_next(var_3);
            // norm_sq += result_vec_in[worldid, i] * result_vec_in[worldid, i]                   <L 3185>
            var_5 = wp::address(var_result_vec_in, var_0, var_4);
            var_6 = wp::address(var_result_vec_in, var_0, var_4);
            var_8 = wp::load(var_5);
            var_9 = wp::load(var_6);
            var_7 = wp::mul(var_8, var_9);
            var_10 = wp::add(var_2, var_7);
            wp::assign(var_2, var_10);
            goto start_for_0;
        end_for_0:;
        // actuator_acc0_out[worldid, actid_target] = wp.sqrt(norm_sq)                            <L 3186>
        var_11 = wp::sqrt(var_2);
        wp::array_store(var_actuator_acc0_out, var_0, var_actid_target, var_11);
    }
}

