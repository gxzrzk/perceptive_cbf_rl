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



extern "C" __global__ void _efc_contact_jac_sparse__locals__kernel_3a643de3_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_body_weldid,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::int32> var_dof_parentid,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::int32> var_body_isdofancestor,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::int32> var_contact_efc_address_in,
    wp::array_t<wp::int32> var_efc_J_rownnz_in,
    wp::array_t<wp::int32> var_efc_J_rowadr_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::array_t<wp::int32> var_condim_in,
    wp::array_t<wp::vec_t<2, wp::int32>> var_geom_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_pos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_frame_in,
    wp::array_t<wp::float32> var_friction_in,
    wp::array_t<wp::int32> var_worldid_in,
    wp::array_t<wp::int32> var_efc_J_colind_out,
    wp::array_t<wp::float32> var_efc_J_out,
    wp::array_t<wp::float32> var_efc_Jqvel_out)
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
        const wp::int32 var_2 = 0;
        wp::int32* var_3;
        bool var_4;
        wp::int32 var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        const wp::int32 var_9 = 0;
        bool var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::vec_t<2, wp::int32>* var_17;
        wp::vec_t<2, wp::int32> var_18;
        wp::vec_t<2, wp::int32> var_19;
        const wp::int32 var_20 = 0;
        wp::int32 var_21;
        wp::int32* var_22;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        const wp::int32 var_27 = 1;
        wp::int32 var_28;
        wp::int32* var_29;
        wp::int32* var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        wp::int32 var_33;
        wp::vec_t<3, wp::float32>* var_34;
        wp::vec_t<3, wp::float32> var_35;
        wp::vec_t<3, wp::float32> var_36;
        const bool var_37 = false;
        bool var_38;
        const wp::int32 var_39 = 0;
        wp::vec_t<3, wp::float32>* var_40;
        wp::vec_t<3, wp::float32> var_41;
        wp::vec_t<3, wp::float32> var_42;
        const wp::int32 var_43 = 1;
        bool var_44;
        const wp::int32 var_45 = 2;
        wp::int32 var_46;
        const wp::int32 var_47 = 1;
        wp::int32 var_48;
        const wp::int32 var_49 = 1;
        wp::int32 var_50;
        wp::float32* var_51;
        wp::float32 var_52;
        wp::float32 var_53;
        wp::int32* var_54;
        wp::int32* var_55;
        wp::int32 var_56;
        wp::int32 var_57;
        wp::int32 var_58;
        const wp::int32 var_59 = 1;
        wp::int32 var_60;
        wp::int32 var_61;
        wp::int32* var_62;
        wp::int32* var_63;
        wp::int32 var_64;
        wp::int32 var_65;
        wp::int32 var_66;
        const wp::int32 var_67 = 1;
        wp::int32 var_68;
        wp::int32 var_69;
        wp::int32 var_70;
        wp::int32* var_71;
        wp::int32 var_72;
        wp::int32 var_73;
        wp::int32* var_74;
        wp::int32 var_75;
        wp::int32 var_76;
        const wp::float32 var_77 = 0.0;
        wp::float32 var_78;
        const wp::int32 var_79 = 0;
        wp::int32 var_80;
        wp::int32 var_81;
        const bool var_82 = true;
        bool var_83;
        bool var_84;
        wp::vec_t<3, wp::float32> var_85;
        wp::vec_t<3, wp::float32> var_86;
        wp::vec_t<3, wp::float32> var_87;
        wp::vec_t<3, wp::float32> var_88;
        wp::vec_t<3, wp::float32> var_89;
        wp::vec_t<3, wp::float32> var_90;
        const bool var_91 = false;
        const wp::float32 var_92 = 0.0;
        wp::float32 var_93;
        const wp::float32 var_94 = 0.0;
        wp::float32 var_95;
        const wp::int32 var_96 = 0;
        wp::float32 var_97;
        wp::float32 var_98;
        wp::float32 var_99;
        wp::float32 var_100;
        const wp::int32 var_101 = 1;
        bool var_102;
        const wp::int32 var_103 = 3;
        bool var_104;
        wp::vec_t<3, wp::float32>* var_105;
        wp::float32 var_106;
        wp::vec_t<3, wp::float32> var_107;
        wp::float32 var_108;
        wp::float32 var_109;
        wp::float32 var_110;
        wp::float32 var_111;
        const wp::int32 var_112 = 3;
        wp::int32 var_113;
        wp::vec_t<3, wp::float32>* var_114;
        wp::float32 var_115;
        wp::vec_t<3, wp::float32> var_116;
        wp::float32 var_117;
        wp::float32 var_118;
        wp::float32 var_119;
        wp::float32 var_120;
        wp::float32 var_121;
        const wp::int32 var_122 = 1;
        wp::float32 var_123;
        wp::float32 var_124;
        wp::float32 var_125;
        wp::float32 var_126;
        const wp::int32 var_127 = 1;
        bool var_128;
        const wp::int32 var_129 = 3;
        bool var_130;
        wp::vec_t<3, wp::float32>* var_131;
        wp::float32 var_132;
        wp::vec_t<3, wp::float32> var_133;
        wp::float32 var_134;
        wp::float32 var_135;
        wp::float32 var_136;
        wp::float32 var_137;
        const wp::int32 var_138 = 3;
        wp::int32 var_139;
        wp::vec_t<3, wp::float32>* var_140;
        wp::float32 var_141;
        wp::vec_t<3, wp::float32> var_142;
        wp::float32 var_143;
        wp::float32 var_144;
        wp::float32 var_145;
        wp::float32 var_146;
        wp::float32 var_147;
        const wp::int32 var_148 = 2;
        wp::float32 var_149;
        wp::float32 var_150;
        wp::float32 var_151;
        wp::float32 var_152;
        const wp::int32 var_153 = 1;
        bool var_154;
        const wp::int32 var_155 = 3;
        bool var_156;
        wp::vec_t<3, wp::float32>* var_157;
        wp::float32 var_158;
        wp::vec_t<3, wp::float32> var_159;
        wp::float32 var_160;
        wp::float32 var_161;
        wp::float32 var_162;
        wp::float32 var_163;
        const wp::int32 var_164 = 3;
        wp::int32 var_165;
        wp::vec_t<3, wp::float32>* var_166;
        wp::float32 var_167;
        wp::vec_t<3, wp::float32> var_168;
        wp::float32 var_169;
        wp::float32 var_170;
        wp::float32 var_171;
        wp::float32 var_172;
        wp::float32 var_173;
        const wp::int32 var_174 = 1;
        bool var_175;
        const wp::int32 var_176 = 2;
        wp::int32 var_177;
        const wp::int32 var_178 = 0;
        bool var_179;
        wp::float32 var_180;
        wp::float32 var_181;
        wp::float32 var_182;
        wp::float32 var_183;
        wp::float32 var_184;
        wp::float32 var_185;
        wp::float32 var_186;
        wp::int32 var_187;
        const wp::int32 var_188 = 0;
        const wp::int32 var_189 = 0;
        const wp::int32 var_190 = 1;
        wp::int32 var_191;
        wp::float32* var_192;
        wp::float32 var_193;
        wp::float32 var_194;
        wp::float32 var_195;
        bool var_196;
        wp::int32* var_197;
        wp::int32 var_198;
        wp::int32 var_199;
        wp::int32 var_200;
        bool var_201;
        wp::int32* var_202;
        wp::int32 var_203;
        wp::int32 var_204;
        wp::int32 var_205;
        wp::int32 var_206;
        wp::int32 var_207;
        wp::int32 var_208;
        wp::int32 var_209;
        wp::int32 var_210;
        wp::float32 var_211;
        wp::int32 var_212;
        wp::int32 var_213;
        //---------
        // forward
        // def kernel(                                                                            <L 3096>
        // conid, dimid = wp.tid()                                                                <L 3127>
        builtin_tid2d(var_0, var_1);
        // if conid >= nacon_in[0]:                                                               <L 3129>
        var_3 = wp::address(var_nacon_in, var_2);
        var_5 = wp::load(var_3);
        var_4 = (var_0 >= var_5);
        if (var_4) {
            // return                                                                             <L 3130>
            continue;
        }
        // efcid = contact_efc_address_in[conid, dimid]                                           <L 3132>
        var_6 = wp::address(var_contact_efc_address_in, var_0, var_1);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // if efcid < 0:                                                                          <L 3133>
        var_10 = (var_7 < var_9);
        if (var_10) {
            // return                                                                             <L 3134>
            continue;
        }
        // worldid = worldid_in[conid]                                                            <L 3136>
        var_11 = wp::address(var_worldid_in, var_0);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // condim = condim_in[conid]                                                              <L 3137>
        var_14 = wp::address(var_condim_in, var_0);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // geom = geom_in[conid]                                                                  <L 3139>
        var_17 = wp::address(var_geom_in, var_0);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // body1 = body_weldid[geom_bodyid[geom[0]]]                                              <L 3140>
        var_21 = wp::extract(var_18, var_20);
        var_22 = wp::address(var_geom_bodyid, var_21);
        var_24 = wp::load(var_22);
        var_23 = wp::address(var_body_weldid, var_24);
        var_26 = wp::load(var_23);
        var_25 = wp::copy(var_26);
        // body2 = body_weldid[geom_bodyid[geom[1]]]                                              <L 3141>
        var_28 = wp::extract(var_18, var_27);
        var_29 = wp::address(var_geom_bodyid, var_28);
        var_31 = wp::load(var_29);
        var_30 = wp::address(var_body_weldid, var_31);
        var_33 = wp::load(var_30);
        var_32 = wp::copy(var_33);
        // con_pos = pos_in[conid]                                                                <L 3143>
        var_34 = wp::address(var_pos_in, var_0);
        var_36 = wp::load(var_34);
        var_35 = wp::copy(var_36);
        // if not wp.static(IS_ELLIPTIC):                                                         <L 3145>
        var_38 = wp::unot(var_37);
        if (var_38) {
            // frame_0 = frame_in[conid, 0]                                                       <L 3146>
            var_40 = wp::address(var_frame_in, var_0, var_39);
            var_42 = wp::load(var_40);
            var_41 = wp::copy(var_42);
            // if condim > 1:                                                                     <L 3147>
            var_44 = (var_15 > var_43);
            if (var_44) {
                // dimid2 = dimid / 2 + 1                                                         <L 3148>
                var_46 = wp::div(var_1, var_45);
                var_48 = wp::add(var_46, var_47);
                // frii = friction_in[conid, dimid2 - 1]                                          <L 3149>
                var_50 = wp::sub(var_48, var_49);
                var_51 = wp::address(var_friction_in, var_0, var_50);
                var_53 = wp::load(var_51);
                var_52 = wp::copy(var_53);
            }
        }
        // da1 = int(body_dofadr[body1] + body_dofnum[body1] - 1)                                 <L 3151>
        var_54 = wp::address(var_body_dofadr, var_25);
        var_55 = wp::address(var_body_dofnum, var_25);
        var_57 = wp::load(var_54);
        var_58 = wp::load(var_55);
        var_56 = wp::add(var_57, var_58);
        var_60 = wp::sub(var_56, var_59);
        var_61 = wp::int(var_60);
        // da2 = int(body_dofadr[body2] + body_dofnum[body2] - 1)                                 <L 3152>
        var_62 = wp::address(var_body_dofadr, var_32);
        var_63 = wp::address(var_body_dofnum, var_32);
        var_65 = wp::load(var_62);
        var_66 = wp::load(var_63);
        var_64 = wp::add(var_65, var_66);
        var_68 = wp::sub(var_64, var_67);
        var_69 = wp::int(var_68);
        // da = wp.max(da1, da2)                                                                  <L 3153>
        var_70 = wp::max(var_61, var_69);
        // rowadr = efc_J_rowadr_in[worldid, efcid]                                               <L 3155>
        var_71 = wp::address(var_efc_J_rowadr_in, var_12, var_7);
        var_73 = wp::load(var_71);
        var_72 = wp::copy(var_73);
        // rownnz = efc_J_rownnz_in[worldid, efcid]                                               <L 3156>
        var_74 = wp::address(var_efc_J_rownnz_in, var_12, var_7);
        var_76 = wp::load(var_74);
        var_75 = wp::copy(var_76);
        // Jqvel = float(0.0)                                                                     <L 3158>
        var_78 = wp::float(var_77);
        // nnz = int(0)                                                                           <L 3159>
        var_80 = wp::int(var_79);
        // dofid = int(da)                                                                        <L 3160>
        var_81 = wp::int(var_70);
        // while True:                                                                            <L 3162>
        start_while_2:;
        if ((var_82) == false) goto end_while_2;
            // if nnz >= rownnz:                                                                  <L 3163>
            var_83 = (var_80 >= var_75);
            if (var_83) {
                // break                                                                          <L 3164>
                goto end_while_2;
            }
            // if dofid == da:                                                                    <L 3166>
            var_84 = (var_81 == var_70);
            if (var_84) {
                // jac1p, jac1r = support.jac_dof(                                                <L 3167>
                // body_parentid,                                                                 <L 3168>
                // body_rootid,                                                                   <L 3169>
                // dof_bodyid,                                                                    <L 3170>
                // body_isdofancestor,                                                            <L 3171>
                // subtree_com_in,                                                                <L 3172>
                // cdof_in,                                                                       <L 3173>
                // con_pos,                                                                       <L 3174>
                // body1,                                                                         <L 3175>
                // dofid,                                                                         <L 3176>
                // worldid,                                                                       <L 3177>
                jac_dof_0(var_body_parentid, var_body_rootid, var_dof_bodyid, var_body_isdofancestor, var_subtree_com_in, var_cdof_in, var_35, var_25, var_81, var_12, var_85, var_86);
                // jac2p, jac2r = support.jac_dof(                                                <L 3179>
                // body_parentid,                                                                 <L 3180>
                // body_rootid,                                                                   <L 3181>
                // dof_bodyid,                                                                    <L 3182>
                // body_isdofancestor,                                                            <L 3183>
                // subtree_com_in,                                                                <L 3184>
                // cdof_in,                                                                       <L 3185>
                // con_pos,                                                                       <L 3186>
                // body2,                                                                         <L 3187>
                // dofid,                                                                         <L 3188>
                // worldid,                                                                       <L 3189>
                jac_dof_0(var_body_parentid, var_body_rootid, var_dof_bodyid, var_body_isdofancestor, var_subtree_com_in, var_cdof_in, var_35, var_32, var_81, var_12, var_87, var_88);
                // jacp_dif = jac2p - jac1p                                                       <L 3192>
                var_89 = wp::sub(var_87, var_85);
                // jacr_dif = jac2r - jac1r                                                       <L 3193>
                var_90 = wp::sub(var_88, var_86);
                // if wp.static(IS_ELLIPTIC):                                                     <L 3195>
                // J = float(0.0)                                                                 <L 3206>
                var_93 = wp::float(var_92);
                // Ji = float(0.0)                                                                <L 3207>
                var_95 = wp::float(var_94);
                // for xyz in range(3):                                                           <L 3209>
                // J += frame_0[xyz] * jacp_dif[xyz]                                              <L 3210>
                var_97 = wp::extract(var_41, var_96);
                var_98 = wp::extract(var_89, var_96);
                var_99 = wp::mul(var_97, var_98);
                var_100 = wp::add(var_93, var_99);
                // if condim > 1:                                                                 <L 3212>
                var_102 = (var_15 > var_101);
                if (var_102) {
                    // if dimid2 < 3:                                                             <L 3213>
                    var_104 = (var_48 < var_103);
                    if (var_104) {
                        // Ji += frame_in[conid, dimid2][xyz] * jacp_dif[xyz]                     <L 3214>
                        var_105 = wp::address(var_frame_in, var_0, var_48);
                        var_107 = wp::load(var_105);
                        var_106 = wp::extract(var_107, var_96);
                        var_108 = wp::extract(var_89, var_96);
                        var_109 = wp::mul(var_106, var_108);
                        var_110 = wp::add(var_95, var_109);
                    }
                    var_111 = wp::where(var_104, var_110, var_95);
                    if (!var_104) {
                        // Ji += frame_in[conid, dimid2 - 3][xyz] * jacr_dif[xyz]                 <L 3216>
                        var_113 = wp::sub(var_48, var_112);
                        var_114 = wp::address(var_frame_in, var_0, var_113);
                        var_116 = wp::load(var_114);
                        var_115 = wp::extract(var_116, var_96);
                        var_117 = wp::extract(var_90, var_96);
                        var_118 = wp::mul(var_115, var_117);
                        var_119 = wp::add(var_111, var_118);
                    }
                    var_120 = wp::where(var_104, var_111, var_119);
                }
                var_121 = wp::where(var_102, var_120, var_95);
                // J += frame_0[xyz] * jacp_dif[xyz]                                              <L 3210>
                var_123 = wp::extract(var_41, var_122);
                var_124 = wp::extract(var_89, var_122);
                var_125 = wp::mul(var_123, var_124);
                var_126 = wp::add(var_100, var_125);
                // if condim > 1:                                                                 <L 3212>
                var_128 = (var_15 > var_127);
                if (var_128) {
                    // if dimid2 < 3:                                                             <L 3213>
                    var_130 = (var_48 < var_129);
                    if (var_130) {
                        // Ji += frame_in[conid, dimid2][xyz] * jacp_dif[xyz]                     <L 3214>
                        var_131 = wp::address(var_frame_in, var_0, var_48);
                        var_133 = wp::load(var_131);
                        var_132 = wp::extract(var_133, var_122);
                        var_134 = wp::extract(var_89, var_122);
                        var_135 = wp::mul(var_132, var_134);
                        var_136 = wp::add(var_121, var_135);
                    }
                    var_137 = wp::where(var_130, var_136, var_121);
                    if (!var_130) {
                        // Ji += frame_in[conid, dimid2 - 3][xyz] * jacr_dif[xyz]                 <L 3216>
                        var_139 = wp::sub(var_48, var_138);
                        var_140 = wp::address(var_frame_in, var_0, var_139);
                        var_142 = wp::load(var_140);
                        var_141 = wp::extract(var_142, var_122);
                        var_143 = wp::extract(var_90, var_122);
                        var_144 = wp::mul(var_141, var_143);
                        var_145 = wp::add(var_137, var_144);
                    }
                    var_146 = wp::where(var_130, var_137, var_145);
                }
                var_147 = wp::where(var_128, var_146, var_121);
                // J += frame_0[xyz] * jacp_dif[xyz]                                              <L 3210>
                var_149 = wp::extract(var_41, var_148);
                var_150 = wp::extract(var_89, var_148);
                var_151 = wp::mul(var_149, var_150);
                var_152 = wp::add(var_126, var_151);
                // if condim > 1:                                                                 <L 3212>
                var_154 = (var_15 > var_153);
                if (var_154) {
                    // if dimid2 < 3:                                                             <L 3213>
                    var_156 = (var_48 < var_155);
                    if (var_156) {
                        // Ji += frame_in[conid, dimid2][xyz] * jacp_dif[xyz]                     <L 3214>
                        var_157 = wp::address(var_frame_in, var_0, var_48);
                        var_159 = wp::load(var_157);
                        var_158 = wp::extract(var_159, var_148);
                        var_160 = wp::extract(var_89, var_148);
                        var_161 = wp::mul(var_158, var_160);
                        var_162 = wp::add(var_147, var_161);
                    }
                    var_163 = wp::where(var_156, var_162, var_147);
                    if (!var_156) {
                        // Ji += frame_in[conid, dimid2 - 3][xyz] * jacr_dif[xyz]                 <L 3216>
                        var_165 = wp::sub(var_48, var_164);
                        var_166 = wp::address(var_frame_in, var_0, var_165);
                        var_168 = wp::load(var_166);
                        var_167 = wp::extract(var_168, var_148);
                        var_169 = wp::extract(var_90, var_148);
                        var_170 = wp::mul(var_167, var_169);
                        var_171 = wp::add(var_163, var_170);
                    }
                    var_172 = wp::where(var_156, var_163, var_171);
                }
                var_173 = wp::where(var_154, var_172, var_147);
                // if condim > 1:                                                                 <L 3218>
                var_175 = (var_15 > var_174);
                if (var_175) {
                    // if dimid % 2 == 0:                                                         <L 3219>
                    var_177 = wp::mod(var_1, var_176);
                    var_179 = (var_177 == var_178);
                    if (var_179) {
                        // J += Ji * frii                                                         <L 3220>
                        var_180 = wp::mul(var_173, var_52);
                        var_181 = wp::add(var_152, var_180);
                    }
                    var_182 = wp::where(var_179, var_181, var_152);
                    if (!var_179) {
                        // J -= Ji * frii                                                         <L 3222>
                        var_183 = wp::mul(var_173, var_52);
                        var_184 = wp::sub(var_182, var_183);
                    }
                    var_185 = wp::where(var_179, var_182, var_184);
                }
                var_186 = wp::where(var_175, var_185, var_152);
                // sparseid = rowadr + nnz                                                        <L 3224>
                var_187 = wp::add(var_72, var_80);
                // efc_J_colind_out[worldid, 0, sparseid] = dofid                                 <L 3225>
                wp::array_store(var_efc_J_colind_out, var_12, var_188, var_187, var_81);
                // efc_J_out[worldid, 0, sparseid] = J                                            <L 3226>
                wp::array_store(var_efc_J_out, var_12, var_189, var_187, var_186);
                // nnz += 1                                                                       <L 3227>
                var_191 = wp::add(var_80, var_190);
                // Jqvel += J * qvel_in[worldid, dofid]                                           <L 3228>
                var_192 = wp::address(var_qvel_in, var_12, var_81);
                var_194 = wp::load(var_192);
                var_193 = wp::mul(var_186, var_194);
                var_195 = wp::add(var_78, var_193);
                // if da1 == da:                                                                  <L 3231>
                var_196 = (var_61 == var_70);
                if (var_196) {
                    // da1 = dof_parentid[da1]                                                    <L 3232>
                    var_197 = wp::address(var_dof_parentid, var_61);
                    var_199 = wp::load(var_197);
                    var_198 = wp::copy(var_199);
                }
                var_200 = wp::where(var_196, var_198, var_61);
                // if da2 == da:                                                                  <L 3233>
                var_201 = (var_69 == var_70);
                if (var_201) {
                    // da2 = dof_parentid[da2]                                                    <L 3234>
                    var_202 = wp::address(var_dof_parentid, var_69);
                    var_204 = wp::load(var_202);
                    var_203 = wp::copy(var_204);
                }
                var_205 = wp::where(var_201, var_203, var_69);
                // da = wp.max(da1, da2)                                                          <L 3235>
                var_206 = wp::max(var_200, var_205);
                // dofid = da                                                                     <L 3236>
                var_207 = wp::copy(var_206);
            }
            var_208 = wp::where(var_84, var_200, var_61);
            var_209 = wp::where(var_84, var_205, var_69);
            var_210 = wp::where(var_84, var_206, var_70);
            var_211 = wp::where(var_84, var_195, var_78);
            var_212 = wp::where(var_84, var_191, var_80);
            var_213 = wp::where(var_84, var_207, var_81);
            wp::assign(var_61, var_208);
            wp::assign(var_69, var_209);
            wp::assign(var_70, var_210);
            wp::assign(var_78, var_211);
            wp::assign(var_80, var_212);
            wp::assign(var_81, var_213);
        goto start_while_2;
        end_while_2:;
        // efc_Jqvel_out[worldid, efcid] = Jqvel                                                  <L 3238>
        wp::array_store(var_efc_Jqvel_out, var_12, var_7, var_78);
    }
}

