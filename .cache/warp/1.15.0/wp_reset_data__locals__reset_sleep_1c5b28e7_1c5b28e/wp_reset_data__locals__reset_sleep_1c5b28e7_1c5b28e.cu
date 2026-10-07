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



extern "C" __global__ void reset_data__locals__reset_sleep_c522a10c_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_nv,
    wp::int32 var_nbody,
    wp::int32 var_ntree,
    wp::array_t<wp::int32> var_body_mocapid,
    wp::array_t<wp::int32> var_body_treeid,
    wp::int32 var_mj_minawake,
    wp::array_t<bool> var_reset_in,
    wp::array_t<wp::int32> var_tree_asleep_out,
    wp::array_t<wp::int32> var_tree_awake_out,
    wp::array_t<wp::int32> var_body_awake_out,
    wp::array_t<wp::int32> var_body_awake_ind_out,
    wp::array_t<wp::int32> var_dof_awake_ind_out)
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
        const bool var_2 = true;
        bool* var_3;
        bool var_4;
        bool var_5;
        bool var_6;
        const wp::int32 var_7 = 1;
        wp::int32 var_8;
        wp::int32 var_9;
        const wp::int32 var_10 = 1;
        bool var_11;
        wp::int32* var_12;
        const wp::int32 var_13 = 0;
        bool var_14;
        wp::int32 var_15;
        wp::int32* var_16;
        const wp::int32 var_17 = 0;
        bool var_18;
        wp::int32 var_19;
        const wp::int32 var_20 = 1;
        const wp::int32 var_21 = 1;
        wp::int32 var_22;
        const wp::int32 var_23 = -1;
        const wp::int32 var_24 = -1;
        wp::int32 var_25;
        const wp::int32 var_26 = 1;
        const wp::int32 var_27 = 1;
        wp::int32 var_28;
        bool var_29;
        //---------
        // forward
        // def reset_sleep(                                                                       <L 2569>
        // worldid, elemid = wp.tid()                                                             <L 2586>
        builtin_tid2d(var_0, var_1);
        // if wp.static(reset is not None):                                                       <L 2588>
        // if not reset_in[worldid]:                                                              <L 2589>
        var_3 = wp::address(var_reset_in, var_0);
        var_5 = wp::load(var_3);
        var_4 = wp::unot(var_5);
        if (var_4) {
            // return                                                                             <L 2590>
            continue;
        }
        // if elemid < ntree:                                                                     <L 2592>
        var_6 = (var_1 < var_ntree);
        if (var_6) {
            // tree_asleep_out[worldid, elemid] = -(1 + mj_minawake)                              <L 2593>
            var_8 = wp::add(var_7, var_mj_minawake);
            var_9 = wp::neg(var_8);
            wp::array_store(var_tree_asleep_out, var_0, var_1, var_9);
            // tree_awake_out[worldid, elemid] = 1                                                <L 2594>
            wp::array_store(var_tree_awake_out, var_0, var_1, var_10);
        }
        // if elemid < nbody:                                                                     <L 2596>
        var_11 = (var_1 < var_nbody);
        if (var_11) {
            // if body_treeid[elemid] < 0:                                                        <L 2597>
            var_12 = wp::address(var_body_treeid, var_1);
            var_15 = wp::load(var_12);
            var_14 = (var_15 < var_13);
            if (var_14) {
                // if body_mocapid[elemid] >= 0:                                                  <L 2598>
                var_16 = wp::address(var_body_mocapid, var_1);
                var_19 = wp::load(var_16);
                var_18 = (var_19 >= var_17);
                if (var_18) {
                    // body_awake_out[worldid, elemid] = int(types.SleepState.AWAKE)              <L 2599>
                    var_22 = wp::int(var_21);
                    wp::array_store(var_body_awake_out, var_0, var_1, var_22);
                }
                if (!var_18) {
                    // body_awake_out[worldid, elemid] = int(types.SleepState.STATIC)             <L 2601>
                    var_25 = wp::int(var_24);
                    wp::array_store(var_body_awake_out, var_0, var_1, var_25);
                }
            }
            if (!var_14) {
                // body_awake_out[worldid, elemid] = int(types.SleepState.AWAKE)                  <L 2603>
                var_28 = wp::int(var_27);
                wp::array_store(var_body_awake_out, var_0, var_1, var_28);
            }
            // body_awake_ind_out[worldid, elemid] = elemid                                       <L 2604>
            wp::array_store(var_body_awake_ind_out, var_0, var_1, var_1);
        }
        // if elemid < nv:                                                                        <L 2606>
        var_29 = (var_1 < var_nv);
        if (var_29) {
            // dof_awake_ind_out[worldid, elemid] = elemid                                        <L 2607>
            wp::array_store(var_dof_awake_ind_out, var_0, var_1, var_1);
        }
    }
}

