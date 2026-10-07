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

extern "C" {
void potrs_35_35_1_120_64_1_1_5_x_x_0(wp::float32*, wp::float32*);
}


extern "C" __global__ void _tile_cholesky_solve_block__locals__kernel_0f050262_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_qLD_block_adr,
    wp::array_t<wp::int32> var_block_dof,
    wp::array_t<wp::float32> var_L_in,
    wp::array_t<wp::float32> var_y,
    wp::array_t<wp::float32> var_x)
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
        wp::slice_t var_5;
        const wp::int32 var_6 = 0;
        wp::array_t<wp::float32> var_7;
        const wp::int32 var_8 = 1225;
        wp::tuple_t<wp::int32> var_9;
        wp::int32* var_10;
        wp::tuple_t<wp::int32> var_11;
        wp::int32 var_12;
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<1225>, wp::tile_stride_t<1>>, true> var_13 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<1225>,wp::tile_stride_t<1>,false>();
        wp::int32 var_14;
        const wp::int32 var_15 = 35;
        wp::tuple_t<wp::int32, wp::int32> var_16;
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<35,35>, wp::tile_stride_t<35,1>>, false> var_17 = nullptr;
        wp::slice_t var_18;
        const wp::int32 var_19 = 0;
        wp::array_t<wp::float32> var_20;
        wp::tuple_t<wp::int32> var_21;
        wp::tuple_t<wp::int32> var_22;
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<35>, wp::tile_stride_t<1>>, true> var_23 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<35>,wp::tile_stride_t<1>,false>();
        const wp::str var_24 = "upper";
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<35>, wp::tile_stride_t<1>>, true> var_25 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<35>,wp::tile_stride_t<1>,false>();
        wp::slice_t var_26;
        const wp::int32 var_27 = 0;
        wp::array_t<wp::float32> var_28;
        wp::tuple_t<wp::int32> var_29;
        //---------
        // forward
        // def kernel(                                                                            <L 3131>
        // worldid, blk = wp.tid()                                                                <L 3141>
        builtin_tid2d(var_0, var_1);
        // start = block_dof[blk]                                                                 <L 3142>
        var_2 = wp::address(var_block_dof, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // L = wp.tile_reshape(                                                                   <L 3144>
        // wp.tile_load(L_in[worldid], shape=(block_area,), offset=(qLD_block_adr[start],)), (block_size, block_size)       <L 3145>
        var_5 = wp::slice_t(var_0, var_0, var_6);
        var_7 = wp::view(var_L_in, var_5);
        var_9 = wp::tuple(var_8);
        var_10 = wp::address(var_qLD_block_adr, var_3);
        var_12 = wp::load(var_10);
        var_11 = wp::tuple(var_12);
        var_14 = wp::load(var_10);
        var_13 = wp::tile_load<wp::float32, true, false, 1225>(var_7, var_14);
        var_16 = wp::tuple(var_15, var_15);
        var_17 = wp::tile_reshape<wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<35,35>, wp::tile_stride_t<35,1>>, false>>(var_13);
        // rhs = wp.tile_load(y[worldid], shape=(block_size,), offset=(start,))                   <L 3147>
        var_18 = wp::slice_t(var_0, var_0, var_19);
        var_20 = wp::view(var_y, var_18);
        var_21 = wp::tuple(var_15);
        var_22 = wp::tuple(var_3);
        var_23 = wp::tile_load<wp::float32, true, false, 35>(var_20, var_3);
        // sol = wp.tile_cholesky_solve(L, rhs, fill_mode="upper")                                <L 3148>
        var_25 = wp::tile_cholesky_solve<true>(potrs_35_35_1_120_64_1_1_5_x_x_0, var_17, var_23, var_25);
        // wp.tile_store(x[worldid], sol, offset=(start,))                                        <L 3149>
        var_26 = wp::slice_t(var_0, var_0, var_27);
        var_28 = wp::view(var_x, var_26);
        var_29 = wp::tuple(var_3);
        wp::tile_store<wp::float32, true, false>(var_28, var_3, var_25);
    }
}

