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

extern "C" {
void potrf_16_16_1_120_32_1_1_5_x_x_0(wp::float32*, int*);
void trsm_16_16_1_120_32_0_1_5_0_1_1(wp::float32*, wp::float32*);
void trsm_16_1_1_120_32_1_1_5_0_1_0(wp::float32*, wp::float32*);
void trsm_16_1_1_120_32_0_1_5_0_1_1(wp::float32*, wp::float32*);
}

// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/block_cholesky.py:21
static CUDA_CALLABLE wp::vec_t<2, wp::float32> solve_search_sums_0(
    wp::float32 var_grad,
    wp::float32 var_solution)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::float32 var_1;
    wp::vec_t<2, wp::float32> var_2;
    //---------
    // forward
    // def solve_search_sums(grad: float, solution: float):                                   <L 22>
    // return wp.vec2(solution * solution, grad * solution)                                   <L 23>
    var_0 = wp::mul(var_solution, var_solution);
    var_1 = wp::mul(var_grad, var_solution);
    var_2 = wp::vec_t<2, wp::float32>(var_0, var_1);
    return var_2;
}

template<typename tile_solution_tile>

// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/block_cholesky.py:28
static CUDA_CALLABLE wp::vec_t<2, wp::float32> _create_newton_decrement_func__locals__newton_decrement_func_0(
    tile_solution_tile& var_solution_tile,
    wp::array_t<wp::float32> var_b,
    wp::array_t<wp::float32> var_search_out)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 353;
    const wp::int32 var_1 = 1;
    wp::tuple_t<wp::int32, wp::int32> var_2;
    const wp::int32 var_3 = 0;
    const wp::int32 var_4 = 0;
    wp::tuple_t<wp::int32, wp::int32> var_5;
    const bool var_6 = false;
    wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<353,1>>> var_7 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<353,1>>>{};
    const wp::int32 var_8 = 0;
    const wp::int32 var_9 = 0;
    const wp::int32 var_10 = 1;
    wp::tuple_t<wp::int32, wp::int32> var_11;
    const wp::int32 var_12 = 0;
    const wp::int32 var_13 = 0;
    wp::tuple_t<wp::int32, wp::int32> var_14;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<353,1>, wp::tile_stride_t<1,1>>, false> var_15 = nullptr;
    const wp::int32 var_16 = 0;
    const wp::int32 var_17 = 0;
    const wp::float32 var_18 = -1.0;
    wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<353,1>>> var_19 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<353,1>>>{};
    const bool var_20 = false;
    const wp::int32 var_21 = 0;
    const wp::int32 var_22 = 0;
    wp::tile_register_t<wp::vec_t<2, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<353,1>>> var_23 = wp::tile_register_t<wp::vec_t<2, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<353,1>>>{};
    wp::tile_shared_t<wp::vec_t<2, wp::float32>,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_24 = wp::tile_alloc_empty<wp::vec_t<2, wp::float32>,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
    const wp::int32 var_25 = 0;
    wp::vec_t<2, wp::float32> var_26;
    //---------
    // forward
    // def newton_decrement_func(                                                             <L 29>
    // grad_tile = wp.tile_load(b, shape=(vector_size_static, 1), offset=(0, 0), bounds_check=False)       <L 36>
    var_2 = wp::tuple(var_0, var_1);
    var_5 = wp::tuple(var_3, var_4);
    var_7 = wp::tile_load<wp::float32, false, false, 353, 1>(var_b, var_8, var_9);
    // active_solution = wp.tile_view(solution_tile, shape=(vector_size_static, 1), offset=(0, 0))       <L 37>
    var_11 = wp::tuple(var_0, var_10);
    var_14 = wp::tuple(var_12, var_13);
    var_15 = wp::tile_view<wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<353,1>, wp::tile_stride_t<1,1>>, false>>(var_solution_tile, var_16, var_17);
    // wp.tile_store(search_out, wp.tile_map(wp.mul, active_solution, -1.0), bounds_check=False)       <L 38>
    var_19 = wp::tile_binary_map(wp::mul, var_15, var_18);
    wp::tile_store<wp::float32, false, false>(var_search_out, var_21, var_22, var_19);
    // return wp.tile_reduce(wp.add, wp.tile_map(solve_search_sums, grad_tile, active_solution))[0]       <L 39>
    var_23 = wp::tile_binary_map(solve_search_sums_0, var_7, var_15);
    var_24 = wp::tile_reduce(wp::add, var_23);
    var_26 = wp::tile_extract(var_24, var_25);
    return var_26;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/block_cholesky.py:128
static CUDA_CALLABLE wp::vec_t<2, wp::float32> _create_blocked_cholesky_augmented_factorize_solve_func__locals__blocked_cholesky_augmented_factorize_solve_func_0(
    wp::array_t<wp::float32> var_A,
    wp::array_t<wp::float32> var_b,
    wp::int32 var_matrix_size,
    wp::array_t<wp::float32> var_U_out,
    wp::array_t<wp::float32> var_result_out)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 368;
    const wp::int32 var_1 = 1;
    wp::tuple_t<wp::int32, wp::int32> var_2;
    const wp::str var_3 = "shared";
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<368,1>, wp::tile_stride_t<1,1>>, true> var_4 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<368,1>,wp::tile_stride_t<1,1>,false>();
    const wp::int32 var_5 = 0;
    const wp::int32 var_6 = 16;
    wp::range_t var_7;
    wp::int32 var_8;
    const wp::int32 var_9 = 16;
    wp::int32 var_10;
    const wp::int32 var_11 = 1;
    wp::tuple_t<wp::int32, wp::int32> var_12;
    const wp::int32 var_13 = 0;
    wp::tuple_t<wp::int32, wp::int32> var_14;
    const wp::str var_15 = "shared";
    const bool var_16 = false;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,1>, wp::tile_stride_t<1,1>>, true> var_17 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,1>,wp::tile_stride_t<1,1>,false>();
    const wp::int32 var_18 = 0;
    wp::tuple_t<wp::int32, wp::int32> var_19;
    wp::tuple_t<wp::int32, wp::int32> var_20;
    const wp::str var_21 = "shared";
    const bool var_22 = false;
    const bool var_23 = true;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<16,1>>, true> var_24 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,16>,wp::tile_stride_t<16,1>,false>();
    const wp::int32 var_25 = 15;
    const wp::int32 var_26 = 1;
    wp::tuple_t<wp::int32, wp::int32> var_27;
    const wp::int32 var_28 = 0;
    const wp::int32 var_29 = 0;
    wp::tuple_t<wp::int32, wp::int32> var_30;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<15,1>, wp::tile_stride_t<1,1>>, false> var_31 = nullptr;
    const wp::int32 var_32 = 0;
    const wp::int32 var_33 = 0;
    bool var_34;
    const wp::int32 var_35 = 0;
    wp::tuple_t<wp::int32, wp::int32> var_36;
    const wp::int32 var_37 = 0;
    const wp::int32 var_38 = 15;
    const wp::float32 var_39 = 1e+30;
    const wp::int32 var_40 = 0;
    const wp::int32 var_41 = 16;
    wp::range_t var_42;
    wp::int32 var_43;
    wp::tuple_t<wp::int32, wp::int32> var_44;
    wp::tuple_t<wp::int32, wp::int32> var_45;
    const wp::str var_46 = "shared";
    const bool var_47 = false;
    const bool var_48 = true;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<16,1>>, true> var_49 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,16>,wp::tile_stride_t<16,1>,false>();
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<1,16>>, false> var_50 = nullptr;
    const wp::float32 var_51 = -1.0;
    const wp::int32 var_52 = 0;
    const wp::int32 var_53 = 0;
    const wp::int32 var_54 = 0;
    const wp::float32 var_55 = 1.0;
    const wp::str var_56 = "upper";
    const wp::int32 var_57 = 1;
    wp::tuple_t<wp::int32, wp::int32> var_58;
    const wp::int32 var_59 = 0;
    wp::tuple_t<wp::int32, wp::int32> var_60;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<15,1>, wp::tile_stride_t<16,1>>, false> var_61 = nullptr;
    const wp::int32 var_62 = 0;
    const wp::int32 var_63 = 15;
    bool var_64;
    const wp::int32 var_65 = 0;
    wp::tuple_t<wp::int32, wp::int32> var_66;
    const wp::int32 var_67 = 0;
    wp::tuple_t<wp::int32, wp::int32> var_68;
    const bool var_69 = false;
    const bool var_70 = true;
    const wp::int32 var_71 = 16;
    wp::range_t var_72;
    wp::int32 var_73;
    wp::tuple_t<wp::int32, wp::int32> var_74;
    wp::tuple_t<wp::int32, wp::int32> var_75;
    const wp::str var_76 = "shared";
    const bool var_77 = false;
    const bool var_78 = true;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<16,1>>, true> var_79 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,16>,wp::tile_stride_t<16,1>,false>();
    wp::int32 var_80;
    bool var_81;
    const wp::int32 var_82 = 0;
    wp::tuple_t<wp::int32, wp::int32> var_83;
    const wp::int32 var_84 = 0;
    const wp::int32 var_85 = 15;
    const wp::int32 var_86 = 0;
    const wp::int32 var_87 = 16;
    wp::range_t var_88;
    wp::int32 var_89;
    wp::tuple_t<wp::int32, wp::int32> var_90;
    wp::tuple_t<wp::int32, wp::int32> var_91;
    const wp::str var_92 = "shared";
    const bool var_93 = false;
    const bool var_94 = true;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<16,1>>, true> var_95 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,16>,wp::tile_stride_t<16,1>,false>();
    wp::tuple_t<wp::int32, wp::int32> var_96;
    wp::tuple_t<wp::int32, wp::int32> var_97;
    const wp::str var_98 = "shared";
    const bool var_99 = false;
    const bool var_100 = true;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<16,1>>, true> var_101 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,16>,wp::tile_stride_t<16,1>,false>();
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<1,16>>, false> var_102 = nullptr;
    const wp::float32 var_103 = -1.0;
    const wp::int32 var_104 = 0;
    const wp::int32 var_105 = 0;
    const wp::int32 var_106 = 0;
    const wp::float32 var_107 = 1.0;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<1,16>>, false> var_108 = nullptr;
    const wp::int32 var_109 = 1;
    wp::tuple_t<wp::int32, wp::int32> var_110;
    const wp::int32 var_111 = 0;
    wp::tuple_t<wp::int32, wp::int32> var_112;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,1>, wp::tile_stride_t<16,1>>, false> var_113 = nullptr;
    const wp::int32 var_114 = 0;
    const wp::int32 var_115 = 15;
    wp::int32 var_116;
    bool var_117;
    const wp::int32 var_118 = 0;
    wp::tuple_t<wp::int32, wp::int32> var_119;
    const wp::int32 var_120 = 0;
    wp::tuple_t<wp::int32, wp::int32> var_121;
    const bool var_122 = false;
    const bool var_123 = true;
    wp::int32 var_124;
    const wp::int32 var_125 = -16;
    const wp::int32 var_126 = -1;
    const wp::int32 var_127 = -16;
    wp::range_t var_128;
    wp::int32 var_129;
    wp::int32 var_130;
    const wp::int32 var_131 = 1;
    wp::tuple_t<wp::int32, wp::int32> var_132;
    const wp::int32 var_133 = 0;
    wp::tuple_t<wp::int32, wp::int32> var_134;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,1>, wp::tile_stride_t<1,1>>, false> var_135 = nullptr;
    const wp::int32 var_136 = 0;
    const wp::int32 var_137 = 16;
    wp::range_t var_138;
    wp::int32 var_139;
    wp::tuple_t<wp::int32, wp::int32> var_140;
    wp::tuple_t<wp::int32, wp::int32> var_141;
    const wp::str var_142 = "shared";
    const bool var_143 = false;
    const bool var_144 = true;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<16,1>>, true> var_145 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,16>,wp::tile_stride_t<16,1>,false>();
    const wp::int32 var_146 = 1;
    wp::tuple_t<wp::int32, wp::int32> var_147;
    const wp::int32 var_148 = 0;
    wp::tuple_t<wp::int32, wp::int32> var_149;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,1>, wp::tile_stride_t<1,1>>, false> var_150 = nullptr;
    const wp::int32 var_151 = 0;
    const wp::float32 var_152 = -1.0;
    const wp::int32 var_153 = 0;
    const wp::int32 var_154 = 0;
    const wp::int32 var_155 = 0;
    const wp::float32 var_156 = 1.0;
    wp::tuple_t<wp::int32, wp::int32> var_157;
    wp::tuple_t<wp::int32, wp::int32> var_158;
    const wp::str var_159 = "shared";
    const bool var_160 = false;
    const bool var_161 = true;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<16,1>>, true> var_162 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,16>,wp::tile_stride_t<16,1>,false>();
    const wp::float32 var_163 = 0.0;
    wp::vec_t<2, wp::float32> var_164;
    const bool var_165 = true;
    wp::vec_t<2, wp::float32> var_166;
    //---------
    // forward
    // def blocked_cholesky_augmented_factorize_solve_func(                                   <L 129>
    // rhs_tile = wp.tile_zeros(shape=(matrix_size_static, 1), dtype=float, storage="shared")       <L 139>
    var_2 = wp::tuple(var_0, var_1);
    var_4 = wp::tile_zeros<float, 368, 1>();
    // for k in range(0, matrix_size, block_size):                                            <L 141>
    var_7 = wp::range(var_5, var_matrix_size, var_6);
    start_for_0:;
        if (iter_cmp(var_7) == 0) goto end_for_0;
        var_8 = wp::iter_next(var_7);
        // end = k + block_size                                                               <L 142>
        var_10 = wp::add(var_8, var_9);
        // input_rhs = wp.tile_load(b, shape=(block_size, 1), offset=(k, 0), storage="shared", bounds_check=False)       <L 143>
        var_12 = wp::tuple(var_9, var_11);
        var_14 = wp::tuple(var_8, var_13);
        var_17 = wp::tile_load<wp::float32, false, false, 16, 1>(var_b, var_8, var_18);
        // A_kk_tile = wp.tile_load(                                                          <L 144>
        // A, shape=(block_size, block_size), offset=(k, k), storage="shared", bounds_check=False, aligned=True       <L 145>
        var_19 = wp::tuple(var_9, var_9);
        var_20 = wp::tuple(var_8, var_8);
        var_24 = wp::tile_load<wp::float32, false, true, 16, 16>(var_A, var_8, var_8);
        // input_diagonal_rhs = wp.tile_view(input_rhs, shape=(border_size, 1), offset=(0, 0))       <L 147>
        var_27 = wp::tuple(var_25, var_26);
        var_30 = wp::tuple(var_28, var_29);
        var_31 = wp::tile_view<wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<15,1>, wp::tile_stride_t<1,1>>, false>>(var_17, var_32, var_33);
        // if end == matrix_size:                                                             <L 148>
        var_34 = (var_10 == var_matrix_size);
        if (var_34) {
            // wp.tile_assign(A_kk_tile, input_diagonal_rhs, offset=(0, border_size))         <L 149>
            var_36 = wp::tuple(var_35, var_25);
            wp::tile_assign(var_24, var_31, var_37, var_38);
            // A_kk_tile[border_size, border_size] = 1.0e30                                   <L 151>
            wp::assign(var_24, var_25, var_25, var_39);
        }
        // for j in range(0, k, block_size):                                                  <L 153>
        var_42 = wp::range(var_40, var_8, var_41);
        start_for_2:;
            if (iter_cmp(var_42) == 0) goto end_for_2;
            var_43 = wp::iter_next(var_42);
            // U_block = wp.tile_load(                                                        <L 154>
            // U_out, shape=(block_size, block_size), offset=(j, k), storage="shared", bounds_check=False, aligned=True       <L 155>
            var_44 = wp::tuple(var_9, var_9);
            var_45 = wp::tuple(var_43, var_8);
            var_49 = wp::tile_load<wp::float32, false, true, 16, 16>(var_U_out, var_43, var_8);
            // wp.tile_matmul(wp.tile_transpose(U_block), U_block, A_kk_tile, alpha=-1.0)       <L 157>
            var_50 = wp::tile_transpose(var_49);
            wp::tile_matmul_acc(var_52, var_53, var_54, var_50, var_49, var_24, var_51, var_55);
            goto start_for_2;
        end_for_2:;
        // wp.tile_cholesky_inplace(A_kk_tile, fill_mode="upper")                             <L 159>
        wp::tile_cholesky_inplace<true>(potrf_16_16_1_120_32_1_1_5_x_x_0, var_24);
        // diagonal_border = wp.tile_view(A_kk_tile, shape=(border_size, 1), offset=(0, border_size))       <L 160>
        var_58 = wp::tuple(var_25, var_57);
        var_60 = wp::tuple(var_59, var_25);
        var_61 = wp::tile_view<wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<15,1>, wp::tile_stride_t<16,1>>, false>>(var_24, var_62, var_63);
        // if end == matrix_size:                                                             <L 161>
        var_64 = (var_10 == var_matrix_size);
        if (var_64) {
            // wp.tile_assign(rhs_tile, diagonal_border, offset=(k, 0))                       <L 162>
            var_66 = wp::tuple(var_8, var_65);
            wp::tile_assign(var_4, var_61, var_8, var_67);
        }
        // wp.tile_store(U_out, A_kk_tile, offset=(k, k), bounds_check=False, aligned=True)       <L 163>
        var_68 = wp::tuple(var_8, var_8);
        wp::tile_store<wp::float32, false, true>(var_U_out, var_8, var_8, var_24);
        // for i in range(end, matrix_size, block_size):                                      <L 165>
        var_72 = wp::range(var_10, var_matrix_size, var_71);
        start_for_4:;
            if (iter_cmp(var_72) == 0) goto end_for_4;
            var_73 = wp::iter_next(var_72);
            // A_ki_tile = wp.tile_load(                                                      <L 166>
            // A, shape=(block_size, block_size), offset=(k, i), storage="shared", bounds_check=False, aligned=True       <L 167>
            var_74 = wp::tuple(var_9, var_9);
            var_75 = wp::tuple(var_8, var_73);
            var_79 = wp::tile_load<wp::float32, false, true, 16, 16>(var_A, var_8, var_73);
            // if i + block_size == matrix_size:                                              <L 169>
            var_80 = wp::add(var_73, var_9);
            var_81 = (var_80 == var_matrix_size);
            if (var_81) {
                // wp.tile_assign(A_ki_tile, input_rhs, offset=(0, border_size))              <L 170>
                var_83 = wp::tuple(var_82, var_25);
                wp::tile_assign(var_79, var_17, var_84, var_85);
            }
            // for j in range(0, k, block_size):                                              <L 172>
            var_88 = wp::range(var_86, var_8, var_87);
            start_for_6:;
                if (iter_cmp(var_88) == 0) goto end_for_6;
                var_89 = wp::iter_next(var_88);
                // U_jk_tile = wp.tile_load(                                                  <L 173>
                // U_out, shape=(block_size, block_size), offset=(j, k), storage="shared", bounds_check=False, aligned=True       <L 174>
                var_90 = wp::tuple(var_9, var_9);
                var_91 = wp::tuple(var_89, var_8);
                var_95 = wp::tile_load<wp::float32, false, true, 16, 16>(var_U_out, var_89, var_8);
                // U_ji_tile = wp.tile_load(                                                  <L 176>
                // U_out, shape=(block_size, block_size), offset=(j, i), storage="shared", bounds_check=False, aligned=True       <L 177>
                var_96 = wp::tuple(var_9, var_9);
                var_97 = wp::tuple(var_89, var_73);
                var_101 = wp::tile_load<wp::float32, false, true, 16, 16>(var_U_out, var_89, var_73);
                // wp.tile_matmul(wp.tile_transpose(U_jk_tile), U_ji_tile, A_ki_tile, alpha=-1.0)       <L 179>
                var_102 = wp::tile_transpose(var_95);
                wp::tile_matmul_acc(var_104, var_105, var_106, var_102, var_101, var_79, var_103, var_107);
                goto start_for_6;
            end_for_6:;
            // wp.tile_lower_solve_inplace(wp.tile_transpose(A_kk_tile), A_ki_tile)           <L 181>
            var_108 = wp::tile_transpose(var_24);
            tile_lower_solve_inplace(trsm_16_16_1_120_32_0_1_5_0_1_1, var_108, var_79);
            // panel_border = wp.tile_view(A_ki_tile, shape=(block_size, 1), offset=(0, border_size))       <L 182>
            var_110 = wp::tuple(var_9, var_109);
            var_112 = wp::tuple(var_111, var_25);
            var_113 = wp::tile_view<wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,1>, wp::tile_stride_t<16,1>>, false>>(var_79, var_114, var_115);
            // if i + block_size == matrix_size:                                              <L 183>
            var_116 = wp::add(var_73, var_9);
            var_117 = (var_116 == var_matrix_size);
            if (var_117) {
                // wp.tile_assign(rhs_tile, panel_border, offset=(k, 0))                      <L 184>
                var_119 = wp::tuple(var_8, var_118);
                wp::tile_assign(var_4, var_113, var_8, var_120);
            }
            // wp.tile_store(U_out, A_ki_tile, offset=(k, i), bounds_check=False, aligned=True)       <L 185>
            var_121 = wp::tuple(var_8, var_73);
            wp::tile_store<wp::float32, false, true>(var_U_out, var_8, var_73, var_79);
            wp::assign(var_43, var_89);
            goto start_for_4;
        end_for_4:;
        goto start_for_0;
    end_for_0:;
    // for i in range(matrix_size - block_size, -1, -block_size):                             <L 187>
    var_124 = wp::sub(var_matrix_size, var_9);
    var_128 = wp::range(var_124, var_126, var_127);
    start_for_8:;
        if (iter_cmp(var_128) == 0) goto end_for_8;
        var_129 = wp::iter_next(var_128);
        // i_end = i + block_size                                                             <L 188>
        var_130 = wp::add(var_129, var_9);
        // tmp_tile = wp.tile_view(rhs_tile, shape=(block_size, 1), offset=(i, 0))            <L 189>
        var_132 = wp::tuple(var_9, var_131);
        var_134 = wp::tuple(var_129, var_133);
        var_135 = wp::tile_view<wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,1>, wp::tile_stride_t<1,1>>, false>>(var_4, var_129, var_136);
        // for j in range(i_end, matrix_size, block_size):                                    <L 190>
        var_138 = wp::range(var_130, var_matrix_size, var_137);
        start_for_10:;
            if (iter_cmp(var_138) == 0) goto end_for_10;
            var_139 = wp::iter_next(var_138);
            // U_tile = wp.tile_load(                                                         <L 191>
            // U_out, shape=(block_size, block_size), offset=(i, j), storage="shared", bounds_check=False, aligned=True       <L 192>
            var_140 = wp::tuple(var_9, var_9);
            var_141 = wp::tuple(var_129, var_139);
            var_145 = wp::tile_load<wp::float32, false, true, 16, 16>(var_U_out, var_129, var_139);
            // x_tile = wp.tile_view(rhs_tile, shape=(block_size, 1), offset=(j, 0))          <L 194>
            var_147 = wp::tuple(var_9, var_146);
            var_149 = wp::tuple(var_139, var_148);
            var_150 = wp::tile_view<wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,1>, wp::tile_stride_t<1,1>>, false>>(var_4, var_139, var_151);
            // wp.tile_matmul(U_tile, x_tile, tmp_tile, alpha=-1.0)                           <L 195>
            wp::tile_matmul_acc(var_153, var_154, var_155, var_145, var_150, var_135, var_152, var_156);
            goto start_for_10;
        end_for_10:;
        // U_tile = wp.tile_load(                                                             <L 197>
        // U_out, shape=(block_size, block_size), offset=(i, i), storage="shared", bounds_check=False, aligned=True       <L 198>
        var_157 = wp::tuple(var_9, var_9);
        var_158 = wp::tuple(var_129, var_129);
        var_162 = wp::tile_load<wp::float32, false, true, 16, 16>(var_U_out, var_129, var_129);
        // wp.tile_upper_solve_inplace(U_tile, tmp_tile)                                      <L 200>
        tile_upper_solve_inplace(trsm_16_1_1_120_32_1_1_5_0_1_0, var_162, var_135);
        wp::assign(var_43, var_139);
        goto start_for_8;
    end_for_8:;
    // sums = wp.vec2(0.0)                                                                    <L 202>
    var_164 = wp::vec_t<2, wp::float32>(var_163);
    // if wp.static(WITH_NEWTON_DECREMENT):                                                   <L 203>
    // sums = wp.static(_create_newton_decrement_func(matrix_size_static, vector_size_static))(rhs_tile, b, result_out)       <L 204>
    var_166 = _create_newton_decrement_func__locals__newton_decrement_func_0(var_4, var_b, var_result_out);
    // return sums                                                                            <L 208>
    return var_166;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/block_cholesky.py:222
static CUDA_CALLABLE wp::vec_t<2, wp::float32> _create_blocked_cholesky_solve_func__locals__blocked_cholesky_solve_func_0(
    wp::array_t<wp::float32> var_U,
    wp::array_t<wp::float32> var_b,
    wp::int32 var_matrix_size,
    wp::array_t<wp::float32> var_result_out)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 368;
    const wp::int32 var_1 = 1;
    wp::tuple_t<wp::int32, wp::int32> var_2;
    const wp::int32 var_3 = 0;
    const wp::int32 var_4 = 0;
    wp::tuple_t<wp::int32, wp::int32> var_5;
    const wp::str var_6 = "shared";
    const bool var_7 = false;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<368,1>, wp::tile_stride_t<1,1>>, true> var_8 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<368,1>,wp::tile_stride_t<1,1>,false>();
    const wp::int32 var_9 = 0;
    const wp::int32 var_10 = 0;
    const wp::int32 var_11 = 0;
    const wp::int32 var_12 = 16;
    wp::range_t var_13;
    wp::int32 var_14;
    const wp::int32 var_15 = 16;
    const wp::int32 var_16 = 1;
    wp::tuple_t<wp::int32, wp::int32> var_17;
    const wp::int32 var_18 = 0;
    wp::tuple_t<wp::int32, wp::int32> var_19;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,1>, wp::tile_stride_t<1,1>>, false> var_20 = nullptr;
    const wp::int32 var_21 = 0;
    const wp::int32 var_22 = 0;
    const wp::int32 var_23 = 16;
    wp::range_t var_24;
    wp::int32 var_25;
    wp::tuple_t<wp::int32, wp::int32> var_26;
    wp::tuple_t<wp::int32, wp::int32> var_27;
    const wp::str var_28 = "shared";
    const bool var_29 = false;
    const bool var_30 = true;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<16,1>>, true> var_31 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,16>,wp::tile_stride_t<16,1>,false>();
    const wp::int32 var_32 = 1;
    wp::tuple_t<wp::int32, wp::int32> var_33;
    const wp::int32 var_34 = 0;
    wp::tuple_t<wp::int32, wp::int32> var_35;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,1>, wp::tile_stride_t<1,1>>, false> var_36 = nullptr;
    const wp::int32 var_37 = 0;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<1,16>>, false> var_38 = nullptr;
    const wp::float32 var_39 = -1.0;
    const wp::int32 var_40 = 0;
    const wp::int32 var_41 = 0;
    const wp::int32 var_42 = 0;
    const wp::float32 var_43 = 1.0;
    wp::tuple_t<wp::int32, wp::int32> var_44;
    wp::tuple_t<wp::int32, wp::int32> var_45;
    const wp::str var_46 = "shared";
    const bool var_47 = false;
    const bool var_48 = true;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<16,1>>, true> var_49 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,16>,wp::tile_stride_t<16,1>,false>();
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<1,16>>, false> var_50 = nullptr;
    wp::int32 var_51;
    const wp::int32 var_52 = -16;
    const wp::int32 var_53 = -1;
    const wp::int32 var_54 = -16;
    wp::range_t var_55;
    wp::int32 var_56;
    wp::int32 var_57;
    const wp::int32 var_58 = 1;
    wp::tuple_t<wp::int32, wp::int32> var_59;
    const wp::int32 var_60 = 0;
    wp::tuple_t<wp::int32, wp::int32> var_61;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,1>, wp::tile_stride_t<1,1>>, false> var_62 = nullptr;
    const wp::int32 var_63 = 0;
    const wp::int32 var_64 = 16;
    wp::range_t var_65;
    wp::int32 var_66;
    wp::tuple_t<wp::int32, wp::int32> var_67;
    wp::tuple_t<wp::int32, wp::int32> var_68;
    const wp::str var_69 = "shared";
    const bool var_70 = false;
    const bool var_71 = true;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<16,1>>, true> var_72 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,16>,wp::tile_stride_t<16,1>,false>();
    const wp::int32 var_73 = 1;
    wp::tuple_t<wp::int32, wp::int32> var_74;
    const wp::int32 var_75 = 0;
    wp::tuple_t<wp::int32, wp::int32> var_76;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,1>, wp::tile_stride_t<1,1>>, false> var_77 = nullptr;
    const wp::int32 var_78 = 0;
    const wp::float32 var_79 = -1.0;
    const wp::int32 var_80 = 0;
    const wp::int32 var_81 = 0;
    const wp::int32 var_82 = 0;
    const wp::float32 var_83 = 1.0;
    wp::tuple_t<wp::int32, wp::int32> var_84;
    wp::tuple_t<wp::int32, wp::int32> var_85;
    const wp::str var_86 = "shared";
    const bool var_87 = false;
    const bool var_88 = true;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<16,1>>, true> var_89 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,16>,wp::tile_stride_t<16,1>,false>();
    const wp::float32 var_90 = 0.0;
    wp::vec_t<2, wp::float32> var_91;
    const bool var_92 = true;
    wp::vec_t<2, wp::float32> var_93;
    //---------
    // forward
    // def blocked_cholesky_solve_func(                                                       <L 223>
    // rhs_tile = wp.tile_load(b, shape=(matrix_size_static, 1), offset=(0, 0), storage="shared", bounds_check=False)       <L 236>
    var_2 = wp::tuple(var_0, var_1);
    var_5 = wp::tuple(var_3, var_4);
    var_8 = wp::tile_load<wp::float32, false, false, 368, 1>(var_b, var_9, var_10);
    // for i in range(0, matrix_size, block_size):                                            <L 239>
    var_13 = wp::range(var_11, var_matrix_size, var_12);
    start_for_0:;
        if (iter_cmp(var_13) == 0) goto end_for_0;
        var_14 = wp::iter_next(var_13);
        // rhs_view = wp.tile_view(rhs_tile, shape=(block_size, 1), offset=(i, 0))            <L 240>
        var_17 = wp::tuple(var_15, var_16);
        var_19 = wp::tuple(var_14, var_18);
        var_20 = wp::tile_view<wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,1>, wp::tile_stride_t<1,1>>, false>>(var_8, var_14, var_21);
        // for j in range(0, i, block_size):                                                  <L 241>
        var_24 = wp::range(var_22, var_14, var_23);
        start_for_2:;
            if (iter_cmp(var_24) == 0) goto end_for_2;
            var_25 = wp::iter_next(var_24);
            // U_block = wp.tile_load(                                                        <L 242>
            // U, shape=(block_size, block_size), offset=(j, i), storage="shared", bounds_check=False, aligned=True       <L 243>
            var_26 = wp::tuple(var_15, var_15);
            var_27 = wp::tuple(var_25, var_14);
            var_31 = wp::tile_load<wp::float32, false, true, 16, 16>(var_U, var_25, var_14);
            // y_block = wp.tile_view(rhs_tile, shape=(block_size, 1), offset=(j, 0))         <L 245>
            var_33 = wp::tuple(var_15, var_32);
            var_35 = wp::tuple(var_25, var_34);
            var_36 = wp::tile_view<wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,1>, wp::tile_stride_t<1,1>>, false>>(var_8, var_25, var_37);
            // wp.tile_matmul(wp.tile_transpose(U_block), y_block, rhs_view, alpha=-1.0)       <L 246>
            var_38 = wp::tile_transpose(var_31);
            wp::tile_matmul_acc(var_40, var_41, var_42, var_38, var_36, var_20, var_39, var_43);
            goto start_for_2;
        end_for_2:;
        // U_tile = wp.tile_load(                                                             <L 248>
        // U, shape=(block_size, block_size), offset=(i, i), storage="shared", bounds_check=False, aligned=True       <L 249>
        var_44 = wp::tuple(var_15, var_15);
        var_45 = wp::tuple(var_14, var_14);
        var_49 = wp::tile_load<wp::float32, false, true, 16, 16>(var_U, var_14, var_14);
        // wp.tile_lower_solve_inplace(wp.tile_transpose(U_tile), rhs_view)                   <L 251>
        var_50 = wp::tile_transpose(var_49);
        tile_lower_solve_inplace(trsm_16_1_1_120_32_0_1_5_0_1_1, var_50, var_20);
        goto start_for_0;
    end_for_0:;
    // for i in range(matrix_size - block_size, -1, -block_size):                             <L 254>
    var_51 = wp::sub(var_matrix_size, var_15);
    var_55 = wp::range(var_51, var_53, var_54);
    start_for_4:;
        if (iter_cmp(var_55) == 0) goto end_for_4;
        var_56 = wp::iter_next(var_55);
        // i_end = i + block_size                                                             <L 255>
        var_57 = wp::add(var_56, var_15);
        // tmp_tile = wp.tile_view(rhs_tile, shape=(block_size, 1), offset=(i, 0))            <L 256>
        var_59 = wp::tuple(var_15, var_58);
        var_61 = wp::tuple(var_56, var_60);
        var_62 = wp::tile_view<wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,1>, wp::tile_stride_t<1,1>>, false>>(var_8, var_56, var_63);
        // for j in range(i_end, matrix_size, block_size):                                    <L 257>
        var_65 = wp::range(var_57, var_matrix_size, var_64);
        start_for_6:;
            if (iter_cmp(var_65) == 0) goto end_for_6;
            var_66 = wp::iter_next(var_65);
            // U_tile = wp.tile_load(                                                         <L 258>
            // U, shape=(block_size, block_size), offset=(i, j), storage="shared", bounds_check=False, aligned=True       <L 259>
            var_67 = wp::tuple(var_15, var_15);
            var_68 = wp::tuple(var_56, var_66);
            var_72 = wp::tile_load<wp::float32, false, true, 16, 16>(var_U, var_56, var_66);
            // x_tile = wp.tile_view(rhs_tile, shape=(block_size, 1), offset=(j, 0))          <L 261>
            var_74 = wp::tuple(var_15, var_73);
            var_76 = wp::tuple(var_66, var_75);
            var_77 = wp::tile_view<wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,1>, wp::tile_stride_t<1,1>>, false>>(var_8, var_66, var_78);
            // wp.tile_matmul(U_tile, x_tile, tmp_tile, alpha=-1.0)                           <L 262>
            wp::tile_matmul_acc(var_80, var_81, var_82, var_72, var_77, var_62, var_79, var_83);
            wp::assign(var_49, var_72);
            goto start_for_6;
        end_for_6:;
        // U_tile = wp.tile_load(                                                             <L 263>
        // U, shape=(block_size, block_size), offset=(i, i), storage="shared", bounds_check=False, aligned=True       <L 264>
        var_84 = wp::tuple(var_15, var_15);
        var_85 = wp::tuple(var_56, var_56);
        var_89 = wp::tile_load<wp::float32, false, true, 16, 16>(var_U, var_56, var_56);
        // wp.tile_upper_solve_inplace(U_tile, tmp_tile)                                      <L 267>
        tile_upper_solve_inplace(trsm_16_1_1_120_32_1_1_5_0_1_0, var_89, var_62);
        wp::assign(var_25, var_66);
        wp::assign(var_49, var_89);
        goto start_for_4;
    end_for_4:;
    // sums = wp.vec2(0.0)                                                                    <L 269>
    var_91 = wp::vec_t<2, wp::float32>(var_90);
    // if wp.static(WITH_NEWTON_DECREMENT):                                                   <L 270>
    // sums = wp.static(_create_newton_decrement_func(matrix_size_static, vector_size_static))(rhs_tile, b, result_out)       <L 271>
    var_93 = _create_newton_decrement_func__locals__newton_decrement_func_0(var_8, var_b, var_result_out);
    // return sums                                                                            <L 275>
    return var_93;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/block_cholesky.py:21
static CUDA_CALLABLE void adj_solve_search_sums_0(
    wp::float32 var_grad,
    wp::float32 var_solution,
    wp::float32 & adj_grad,
    wp::float32 & adj_solution,
    wp::vec_t<2, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}

template<typename tile_solution_tile>

// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/block_cholesky.py:28
static CUDA_CALLABLE void adj__create_newton_decrement_func__locals__newton_decrement_func_0(
    tile_solution_tile& var_solution_tile,
    wp::array_t<wp::float32> var_b,
    wp::array_t<wp::float32> var_search_out,
    tile_solution_tile & adj_solution_tile,
    wp::array_t<wp::float32> & adj_b,
    wp::array_t<wp::float32> & adj_search_out,
    wp::vec_t<2, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/block_cholesky.py:128
static CUDA_CALLABLE void adj__create_blocked_cholesky_augmented_factorize_solve_func__locals__blocked_cholesky_augmented_factorize_solve_func_0(
    wp::array_t<wp::float32> var_A,
    wp::array_t<wp::float32> var_b,
    wp::int32 var_matrix_size,
    wp::array_t<wp::float32> var_U_out,
    wp::array_t<wp::float32> var_result_out,
    wp::array_t<wp::float32> & adj_A,
    wp::array_t<wp::float32> & adj_b,
    wp::int32 & adj_matrix_size,
    wp::array_t<wp::float32> & adj_U_out,
    wp::array_t<wp::float32> & adj_result_out,
    wp::vec_t<2, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/block_cholesky.py:222
static CUDA_CALLABLE void adj__create_blocked_cholesky_solve_func__locals__blocked_cholesky_solve_func_0(
    wp::array_t<wp::float32> var_U,
    wp::array_t<wp::float32> var_b,
    wp::int32 var_matrix_size,
    wp::array_t<wp::float32> var_result_out,
    wp::array_t<wp::float32> & adj_U,
    wp::array_t<wp::float32> & adj_b,
    wp::int32 & adj_matrix_size,
    wp::array_t<wp::float32> & adj_result_out,
    wp::vec_t<2, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void _update_gradient_cholesky_blocked_skip_unchanged__locals__kernel_6f388bbe_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_ctx_grad_in,
    wp::array_t<wp::float32> var_ctx_h_in,
    wp::array_t<wp::int32> var_quad_changed_count_in,
    wp::array_t<wp::int32> var_state_changed_count_in,
    wp::array_t<wp::float32> var_ctx_hfactor,
    wp::array_t<wp::float32> var_ctx_search_out,
    wp::array_t<wp::float32> var_ctx_search_dot_out,
    wp::array_t<wp::float32> var_ctx_newton_decrement_out)
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
        const wp::int32 var_1 = 16;
        bool* var_2;
        bool var_3;
        bool var_4;
        const bool var_5 = true;
        wp::int32* var_6;
        const wp::int32 var_7 = 0;
        bool var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        const wp::int32 var_11 = 0;
        bool var_12;
        wp::int32 var_13;
        wp::slice_t var_14;
        const wp::int32 var_15 = 0;
        wp::array_t<wp::float32> var_16;
        wp::slice_t var_17;
        const wp::int32 var_18 = 0;
        wp::array_t<wp::float32> var_19;
        const wp::int32 var_20 = 368;
        wp::slice_t var_21;
        const wp::int32 var_22 = 0;
        wp::array_t<wp::float32> var_23;
        wp::slice_t var_24;
        const wp::int32 var_25 = 0;
        wp::array_t<wp::float32> var_26;
        wp::vec_t<2, wp::float32> var_27;
        wp::slice_t var_28;
        const wp::int32 var_29 = 0;
        wp::array_t<wp::float32> var_30;
        wp::slice_t var_31;
        const wp::int32 var_32 = 0;
        wp::array_t<wp::float32> var_33;
        wp::slice_t var_34;
        const wp::int32 var_35 = 0;
        wp::array_t<wp::float32> var_36;
        wp::vec_t<2, wp::float32> var_37;
        wp::vec_t<2, wp::float32> var_38;
        const wp::int32 var_39 = 0;
        wp::float32 var_40;
        const wp::int32 var_41 = 1;
        wp::float32 var_42;
        //---------
        // forward
        // def kernel(                                                                            <L 2974>
        // worldid = wp.tid()                                                                     <L 2987>
        var_0 = builtin_tid1d();
        // TILE_SIZE = wp.static(tile_size)                                                       <L 2988>
        // if ctx_done_in[worldid]:                                                               <L 2990>
        var_2 = wp::address(var_ctx_done_in, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // return                                                                             <L 2991>
            continue;
        }
        var_4 = wp::load(var_2);
        // if wp.static(SKIP_NOFLIP):                                                             <L 2995>
        // if state_changed_count_in[worldid] == 0:                                               <L 2996>
        var_6 = wp::address(var_state_changed_count_in, var_0);
        var_9 = wp::load(var_6);
        var_8 = (var_9 == var_7);
        if (var_8) {
            // return                                                                             <L 2997>
            continue;
        }
        // if quad_changed_count_in[worldid] > 0:                                                 <L 2999>
        var_10 = wp::address(var_quad_changed_count_in, var_0);
        var_13 = wp::load(var_10);
        var_12 = (var_13 > var_11);
        if (var_12) {
            // sums = wp.static(create_blocked_cholesky_augmented_factorize_solve_newton_func(TILE_SIZE, matrix_size, vector_size))(       <L 3000>
            // ctx_h_in[worldid],                                                                 <L 3001>
            var_14 = wp::slice_t(var_0, var_0, var_15);
            var_16 = wp::view(var_ctx_h_in, var_14);
            // ctx_grad_in[worldid],                                                              <L 3002>
            var_17 = wp::slice_t(var_0, var_0, var_18);
            var_19 = wp::view(var_ctx_grad_in, var_17);
            // matrix_size,                                                                       <L 3003>
            // ctx_hfactor[worldid],                                                              <L 3004>
            var_21 = wp::slice_t(var_0, var_0, var_22);
            var_23 = wp::view(var_ctx_hfactor, var_21);
            // ctx_search_out[worldid],                                                           <L 3005>
            var_24 = wp::slice_t(var_0, var_0, var_25);
            var_26 = wp::view(var_ctx_search_out, var_24);
            var_27 = _create_blocked_cholesky_augmented_factorize_solve_func__locals__blocked_cholesky_augmented_factorize_solve_func_0(var_16, var_19, var_20, var_23, var_26);
        }
        if (!var_12) {
            // sums = wp.static(create_blocked_cholesky_solve_newton_func(TILE_SIZE, matrix_size, vector_size))(       <L 3008>
            // ctx_hfactor[worldid],                                                              <L 3009>
            var_28 = wp::slice_t(var_0, var_0, var_29);
            var_30 = wp::view(var_ctx_hfactor, var_28);
            // ctx_grad_in[worldid],                                                              <L 3010>
            var_31 = wp::slice_t(var_0, var_0, var_32);
            var_33 = wp::view(var_ctx_grad_in, var_31);
            // matrix_size,                                                                       <L 3011>
            // ctx_search_out[worldid],                                                           <L 3012>
            var_34 = wp::slice_t(var_0, var_0, var_35);
            var_36 = wp::view(var_ctx_search_out, var_34);
            var_37 = _create_blocked_cholesky_solve_func__locals__blocked_cholesky_solve_func_0(var_30, var_33, var_20, var_36);
        }
        var_38 = wp::where(var_12, var_27, var_37);
        // ctx_search_dot_out[worldid] = sums[0]                                                  <L 3015>
        var_40 = wp::extract(var_38, var_39);
        wp::array_store(var_ctx_search_dot_out, var_0, var_40);
        // ctx_newton_decrement_out[worldid] = sums[1]                                            <L 3016>
        var_42 = wp::extract(var_38, var_41);
        wp::array_store(var_ctx_newton_decrement_out, var_0, var_42);
    }
}

