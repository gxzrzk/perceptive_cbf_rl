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
void potrf_35_35_1_120_32_1_1_5_x_x_0(wp::float32*, int*);
}


extern "C" __global__ void _tile_cholesky_factorize_block__locals__kernel_f3565c16_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_qLD_block_adr,
    wp::array_t<wp::float32> var_M_in,
    wp::array_t<wp::int32> var_block_elemid,
    wp::array_t<wp::int32> var_block_dof,
    wp::array_t<wp::float32> var_L_out)
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
        const wp::int32 var_5 = 1225;
        wp::tuple_t<wp::int32> var_6;
        wp::int32 var_7;
        wp::tuple_t<wp::int32> var_8;
        const wp::str var_9 = "shared";
        wp::tile_shared_t<wp::int32,wp::tile_layout_strided_t<wp::tile_shape_t<1225>, wp::tile_stride_t<1>>, true> var_10 = wp::tile_alloc_empty<wp::int32,wp::tile_shape_t<1225>,wp::tile_stride_t<1>,false>();
        wp::slice_t var_11;
        const wp::int32 var_12 = 0;
        wp::array_t<wp::float32> var_13;
        wp::tuple_t<wp::int32> var_14;
        const wp::str var_15 = "shared";
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<1225>, wp::tile_stride_t<1>>, true> var_16 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<1225>,wp::tile_stride_t<1>,false>();
        const wp::int32 var_17 = 0;
        const wp::int32 var_18 = 0;
        const wp::int32 var_19 = 35;
        wp::tuple_t<wp::int32, wp::int32> var_20;
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<35,35>, wp::tile_stride_t<35,1>>, false> var_21 = nullptr;
        const wp::str var_22 = "upper";
        wp::slice_t var_23;
        const wp::int32 var_24 = 0;
        wp::array_t<wp::float32> var_25;
        wp::tuple_t<wp::int32> var_26;
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<1225>, wp::tile_stride_t<1>>, false> var_27 = nullptr;
        wp::int32* var_28;
        wp::tuple_t<wp::int32> var_29;
        wp::int32 var_30;
        wp::int32 var_31;
        //---------
        // forward
        // def kernel(                                                                            <L 1290>
        // worldid, blk = wp.tid()                                                                <L 1301>
        builtin_tid2d(var_0, var_1);
        // start = block_dof[blk]                                                                 <L 1302>
        var_2 = wp::address(var_block_dof, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // idx = wp.tile_load(block_elemid, shape=(block_area,), offset=(blk * block_area,), storage="shared")       <L 1304>
        var_6 = wp::tuple(var_5);
        var_7 = wp::mul(var_1, var_5);
        var_8 = wp::tuple(var_7);
        var_10 = wp::tile_load<wp::int32, true, false, 1225>(var_block_elemid, var_7);
        // block = wp.tile_load_indexed(M_in[worldid], idx, shape=(block_area,), storage="shared")       <L 1305>
        var_11 = wp::slice_t(var_0, var_0, var_12);
        var_13 = wp::view(var_M_in, var_11);
        var_14 = wp::tuple(var_5);
        var_16 = wp::tile_load_indexed<1225>(var_13, var_10, var_17, var_18);
        // L = wp.tile_reshape(block, (block_size, block_size))                                   <L 1307>
        var_20 = wp::tuple(var_19, var_19);
        var_21 = wp::tile_reshape<wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<35,35>, wp::tile_stride_t<35,1>>, false>>(var_16);
        // wp.tile_cholesky_inplace(L, fill_mode="upper")                                         <L 1308>
        wp::tile_cholesky_inplace<true>(potrf_35_35_1_120_32_1_1_5_x_x_0, var_21);
        // wp.tile_store(L_out[worldid], wp.tile_reshape(L, (block_area,)), offset=(qLD_block_adr[start],))       <L 1309>
        var_23 = wp::slice_t(var_0, var_0, var_24);
        var_25 = wp::view(var_L_out, var_23);
        var_26 = wp::tuple(var_5);
        var_27 = wp::tile_reshape<wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<1225>, wp::tile_stride_t<1>>, false>>(var_21);
        var_28 = wp::address(var_qLD_block_adr, var_3);
        var_30 = wp::load(var_28);
        var_29 = wp::tuple(var_30);
        var_31 = wp::load(var_28);
        wp::tile_store<wp::float32, true, false>(var_25, var_31, var_27);
    }
}

