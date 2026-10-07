#define WP_NO_BFLOAT16

#define WP_TILE_BLOCK_DIM 64
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


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/smooth.py:3071
static CUDA_CALLABLE void _small_cholesky_solve_0(
    wp::int32 var_block_size,
    wp::int32 var_worldid,
    wp::int32 var_factor_adr,
    wp::int32 var_start,
    wp::array_t<wp::float32> var_L_in,
    wp::array_t<wp::float32> var_y_in,
    wp::array_t<wp::float32> var_x_out)
{
    //---------
    // primal vars
    wp::range_t var_0;
    wp::int32 var_1;
    wp::int32 var_2;
    wp::float32* var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::range_t var_6;
    wp::int32 var_7;
    wp::int32 var_8;
    wp::int32 var_9;
    wp::int32 var_10;
    wp::float32* var_11;
    wp::int32 var_12;
    wp::float32* var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::int32 var_18;
    wp::int32 var_19;
    wp::int32 var_20;
    wp::float32* var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    wp::int32 var_24;
    wp::range_t var_25;
    wp::int32 var_26;
    const wp::int32 var_27 = 1;
    wp::int32 var_28;
    wp::int32 var_29;
    wp::int32 var_30;
    wp::float32* var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    const wp::int32 var_34 = 1;
    wp::int32 var_35;
    wp::range_t var_36;
    wp::int32 var_37;
    wp::int32 var_38;
    wp::int32 var_39;
    wp::int32 var_40;
    wp::float32* var_41;
    wp::int32 var_42;
    wp::float32* var_43;
    wp::float32 var_44;
    wp::float32 var_45;
    wp::float32 var_46;
    wp::float32 var_47;
    wp::int32 var_48;
    wp::int32 var_49;
    wp::int32 var_50;
    wp::float32* var_51;
    wp::float32 var_52;
    wp::float32 var_53;
    wp::int32 var_54;
    //---------
    // forward
    // def _small_cholesky_solve(                                                             <L 3072>
    // for i in range(block_size):                                                            <L 3083>
    var_0 = wp::range(var_block_size);
    start_for_0:;
        if (iter_cmp(var_0) == 0) goto end_for_0;
        var_1 = wp::iter_next(var_0);
        // value = y_in[worldid, start + i]                                                   <L 3084>
        var_2 = wp::add(var_start, var_1);
        var_3 = wp::address(var_y_in, var_worldid, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // for k in range(i):                                                                 <L 3085>
        var_6 = wp::range(var_1);
        start_for_2:;
            if (iter_cmp(var_6) == 0) goto end_for_2;
            var_7 = wp::iter_next(var_6);
            // value -= L_in[worldid, factor_adr + k * block_size + i] * x_out[worldid, start + k]       <L 3086>
            var_8 = wp::mul(var_7, var_block_size);
            var_9 = wp::add(var_factor_adr, var_8);
            var_10 = wp::add(var_9, var_1);
            var_11 = wp::address(var_L_in, var_worldid, var_10);
            var_12 = wp::add(var_start, var_7);
            var_13 = wp::address(var_x_out, var_worldid, var_12);
            var_15 = wp::load(var_11);
            var_16 = wp::load(var_13);
            var_14 = wp::mul(var_15, var_16);
            var_17 = wp::sub(var_4, var_14);
            wp::assign(var_4, var_17);
            goto start_for_2;
        end_for_2:;
        // x_out[worldid, start + i] = value / L_in[worldid, factor_adr + i * block_size + i]       <L 3087>
        var_18 = wp::mul(var_1, var_block_size);
        var_19 = wp::add(var_factor_adr, var_18);
        var_20 = wp::add(var_19, var_1);
        var_21 = wp::address(var_L_in, var_worldid, var_20);
        var_23 = wp::load(var_21);
        var_22 = wp::div(var_4, var_23);
        var_24 = wp::add(var_start, var_1);
        wp::array_store(var_x_out, var_worldid, var_24, var_22);
        goto start_for_0;
    end_for_0:;
    // for reverse_i in range(block_size):                                                    <L 3089>
    var_25 = wp::range(var_block_size);
    start_for_4:;
        if (iter_cmp(var_25) == 0) goto end_for_4;
        var_26 = wp::iter_next(var_25);
        // i = block_size - 1 - reverse_i                                                     <L 3090>
        var_28 = wp::sub(var_block_size, var_27);
        var_29 = wp::sub(var_28, var_26);
        // value = x_out[worldid, start + i]                                                  <L 3091>
        var_30 = wp::add(var_start, var_29);
        var_31 = wp::address(var_x_out, var_worldid, var_30);
        var_33 = wp::load(var_31);
        var_32 = wp::copy(var_33);
        // for k in range(i + 1, block_size):                                                 <L 3092>
        var_35 = wp::add(var_29, var_34);
        var_36 = wp::range(var_35, var_block_size);
        start_for_6:;
            if (iter_cmp(var_36) == 0) goto end_for_6;
            var_37 = wp::iter_next(var_36);
            // value -= L_in[worldid, factor_adr + i * block_size + k] * x_out[worldid, start + k]       <L 3093>
            var_38 = wp::mul(var_29, var_block_size);
            var_39 = wp::add(var_factor_adr, var_38);
            var_40 = wp::add(var_39, var_37);
            var_41 = wp::address(var_L_in, var_worldid, var_40);
            var_42 = wp::add(var_start, var_37);
            var_43 = wp::address(var_x_out, var_worldid, var_42);
            var_45 = wp::load(var_41);
            var_46 = wp::load(var_43);
            var_44 = wp::mul(var_45, var_46);
            var_47 = wp::sub(var_32, var_44);
            wp::assign(var_32, var_47);
            goto start_for_6;
        end_for_6:;
        // x_out[worldid, start + i] = value / L_in[worldid, factor_adr + i * block_size + i]       <L 3094>
        var_48 = wp::mul(var_29, var_block_size);
        var_49 = wp::add(var_factor_adr, var_48);
        var_50 = wp::add(var_49, var_29);
        var_51 = wp::address(var_L_in, var_worldid, var_50);
        var_53 = wp::load(var_51);
        var_52 = wp::div(var_32, var_53);
        var_54 = wp::add(var_start, var_29);
        wp::array_store(var_x_out, var_worldid, var_54, var_52);
        wp::assign(var_1, var_29);
        wp::assign(var_4, var_32);
        wp::assign(var_7, var_37);
        goto start_for_4;
    end_for_4:;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/smooth.py:3071
static CUDA_CALLABLE void adj__small_cholesky_solve_0(
    wp::int32 var_block_size,
    wp::int32 var_worldid,
    wp::int32 var_factor_adr,
    wp::int32 var_start,
    wp::array_t<wp::float32> var_L_in,
    wp::array_t<wp::float32> var_y_in,
    wp::array_t<wp::float32> var_x_out,
    wp::int32 & adj_block_size,
    wp::int32 & adj_worldid,
    wp::int32 & adj_factor_adr,
    wp::int32 & adj_start,
    wp::array_t<wp::float32> & adj_L_in,
    wp::array_t<wp::float32> & adj_y_in,
    wp::array_t<wp::float32> & adj_x_out)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void _small_cholesky_solve_block__locals__kernel_7916fd4b_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_qLD_block_adr,
    wp::array_t<wp::int32> var_block_dof,
    wp::array_t<wp::float32> var_D_in,
    wp::array_t<wp::float32> var_L_in,
    wp::array_t<wp::float32> var_y_in,
    wp::array_t<wp::float32> var_x_out)
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
        const wp::int32 var_5 = 6;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        const wp::int32 var_9 = -2;
        bool var_10;
        const wp::int32 var_11 = 0;
        wp::int32 var_12;
        wp::float32* var_13;
        wp::int32 var_14;
        wp::float32* var_15;
        wp::float32 var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        wp::int32 var_19;
        const wp::int32 var_20 = 1;
        wp::int32 var_21;
        wp::float32* var_22;
        wp::int32 var_23;
        wp::float32* var_24;
        wp::float32 var_25;
        wp::float32 var_26;
        wp::float32 var_27;
        wp::int32 var_28;
        const wp::int32 var_29 = 2;
        wp::int32 var_30;
        wp::float32* var_31;
        wp::int32 var_32;
        wp::float32* var_33;
        wp::float32 var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        wp::int32 var_37;
        const wp::int32 var_38 = 3;
        wp::int32 var_39;
        wp::float32* var_40;
        wp::int32 var_41;
        wp::float32* var_42;
        wp::float32 var_43;
        wp::float32 var_44;
        wp::float32 var_45;
        wp::int32 var_46;
        const wp::int32 var_47 = 4;
        wp::int32 var_48;
        wp::float32* var_49;
        wp::int32 var_50;
        wp::float32* var_51;
        wp::float32 var_52;
        wp::float32 var_53;
        wp::float32 var_54;
        wp::int32 var_55;
        const wp::int32 var_56 = 5;
        wp::int32 var_57;
        wp::float32* var_58;
        wp::int32 var_59;
        wp::float32* var_60;
        wp::float32 var_61;
        wp::float32 var_62;
        wp::float32 var_63;
        wp::int32 var_64;
        //---------
        // forward
        // def kernel(                                                                            <L 3100>
        // worldid, blk = wp.tid()                                                                <L 3111>
        builtin_tid2d(var_0, var_1);
        // start = block_dof[blk]                                                                 <L 3112>
        var_2 = wp::address(var_block_dof, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // size = wp.static(block_size)                                                           <L 3113>
        // factor_adr = qLD_block_adr[start]                                                      <L 3114>
        var_6 = wp::address(var_qLD_block_adr, var_3);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // if factor_adr == Q_LD_BLOCK_COMPACT:                                                   <L 3115>
        var_10 = (var_7 == var_9);
        if (var_10) {
            // for i in range(wp.static(block_size)):                                             <L 3116>
            // x_out[worldid, start + i] = D_in[worldid, start + i] * y_in[worldid, start + i]       <L 3117>
            var_12 = wp::add(var_3, var_11);
            var_13 = wp::address(var_D_in, var_0, var_12);
            var_14 = wp::add(var_3, var_11);
            var_15 = wp::address(var_y_in, var_0, var_14);
            var_17 = wp::load(var_13);
            var_18 = wp::load(var_15);
            var_16 = wp::mul(var_17, var_18);
            var_19 = wp::add(var_3, var_11);
            wp::array_store(var_x_out, var_0, var_19, var_16);
            var_21 = wp::add(var_3, var_20);
            var_22 = wp::address(var_D_in, var_0, var_21);
            var_23 = wp::add(var_3, var_20);
            var_24 = wp::address(var_y_in, var_0, var_23);
            var_26 = wp::load(var_22);
            var_27 = wp::load(var_24);
            var_25 = wp::mul(var_26, var_27);
            var_28 = wp::add(var_3, var_20);
            wp::array_store(var_x_out, var_0, var_28, var_25);
            var_30 = wp::add(var_3, var_29);
            var_31 = wp::address(var_D_in, var_0, var_30);
            var_32 = wp::add(var_3, var_29);
            var_33 = wp::address(var_y_in, var_0, var_32);
            var_35 = wp::load(var_31);
            var_36 = wp::load(var_33);
            var_34 = wp::mul(var_35, var_36);
            var_37 = wp::add(var_3, var_29);
            wp::array_store(var_x_out, var_0, var_37, var_34);
            var_39 = wp::add(var_3, var_38);
            var_40 = wp::address(var_D_in, var_0, var_39);
            var_41 = wp::add(var_3, var_38);
            var_42 = wp::address(var_y_in, var_0, var_41);
            var_44 = wp::load(var_40);
            var_45 = wp::load(var_42);
            var_43 = wp::mul(var_44, var_45);
            var_46 = wp::add(var_3, var_38);
            wp::array_store(var_x_out, var_0, var_46, var_43);
            var_48 = wp::add(var_3, var_47);
            var_49 = wp::address(var_D_in, var_0, var_48);
            var_50 = wp::add(var_3, var_47);
            var_51 = wp::address(var_y_in, var_0, var_50);
            var_53 = wp::load(var_49);
            var_54 = wp::load(var_51);
            var_52 = wp::mul(var_53, var_54);
            var_55 = wp::add(var_3, var_47);
            wp::array_store(var_x_out, var_0, var_55, var_52);
            var_57 = wp::add(var_3, var_56);
            var_58 = wp::address(var_D_in, var_0, var_57);
            var_59 = wp::add(var_3, var_56);
            var_60 = wp::address(var_y_in, var_0, var_59);
            var_62 = wp::load(var_58);
            var_63 = wp::load(var_60);
            var_61 = wp::mul(var_62, var_63);
            var_64 = wp::add(var_3, var_56);
            wp::array_store(var_x_out, var_0, var_64, var_61);
        }
        if (!var_10) {
            // _small_cholesky_solve(size, worldid, factor_adr, start, L_in, y_in, x_out)         <L 3119>
            _small_cholesky_solve_0(var_5, var_0, var_7, var_3, var_L_in, var_y_in, var_x_out);
        }
    }
}

