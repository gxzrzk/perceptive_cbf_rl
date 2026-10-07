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



extern "C" __global__ void _contact_sort__locals__contact_sort_9ed868d5_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_sensor_intprm,
    wp::array_t<wp::int32> var_sensor_contact_adr,
    wp::array_t<wp::int32> var_sensor_contact_nmatch_in,
    wp::array_t<wp::int32> var_sensor_contact_matchid_in,
    wp::array_t<wp::float32> var_sensor_contact_criteria_in,
    wp::array_t<wp::int32> var_sensor_contact_matchid_out)
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
        wp::int32 var_3;
        wp::int32* var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        const wp::int32 var_7 = 1;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        bool var_11;
        const wp::int32 var_12 = 0;
        bool var_13;
        const wp::int32 var_14 = 3;
        bool var_15;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        const wp::int32 var_19 = 1;
        bool var_20;
        wp::slice_t var_21;
        const wp::int32 var_22 = 0;
        wp::slice_t var_23;
        const wp::int32 var_24 = 0;
        wp::array_t<wp::float32> var_25;
        const wp::int32 var_26 = 256;
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<256>, wp::tile_stride_t<1>>, true> var_27 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<256>,wp::tile_stride_t<1>,false>();
        const wp::int32 var_28 = 0;
        wp::slice_t var_29;
        const wp::int32 var_30 = 0;
        wp::slice_t var_31;
        const wp::int32 var_32 = 0;
        wp::array_t<wp::int32> var_33;
        wp::tile_shared_t<wp::int32,wp::tile_layout_strided_t<wp::tile_shape_t<256>, wp::tile_stride_t<1>>, true> var_34 = wp::tile_alloc_empty<wp::int32,wp::tile_shape_t<256>,wp::tile_stride_t<1>,false>();
        const wp::int32 var_35 = 0;
        wp::slice_t var_36;
        const wp::int32 var_37 = 0;
        wp::slice_t var_38;
        const wp::int32 var_39 = 0;
        wp::array_t<wp::int32> var_40;
        const wp::int32 var_41 = 0;
        //---------
        // forward
        // def contact_sort(                                                                      <L 2472>
        // worldid, contactsensorid = wp.tid()                                                    <L 2483>
        builtin_tid2d(var_0, var_1);
        // worldid, contactsensorid = wp.tid()                                                    <L 2485>
        builtin_tid2d(var_2, var_3);
        // sensorid = sensor_contact_adr[contactsensorid]                                         <L 2486>
        var_4 = wp::address(var_sensor_contact_adr, var_3);
        var_6 = wp::load(var_4);
        var_5 = wp::copy(var_6);
        // reduce = sensor_intprm[sensorid, 1]                                                    <L 2488>
        var_8 = wp::address(var_sensor_intprm, var_5, var_7);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // if reduce == 0 or reduce == 3:  # none or netforce                                     <L 2489>
        var_13 = (var_9 == var_12);
        var_11 = var_13;
        if (!var_11) {
            var_15 = (var_9 == var_14);
            var_11 = var_11 || var_15;
        }
        if (var_11) {
            // return                                                                             <L 2490>
            continue;
        }
        // nmatch = sensor_contact_nmatch_in[worldid, contactsensorid]                            <L 2492>
        var_16 = wp::address(var_sensor_contact_nmatch_in, var_2, var_3);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // if nmatch <= 1:                                                                        <L 2495>
        var_20 = (var_17 <= var_19);
        if (var_20) {
            // return                                                                             <L 2496>
            continue;
        }
        // criteria_tile = wp.tile_load(sensor_contact_criteria_in[worldid, contactsensorid], shape=maxmatch)       <L 2498>
        var_21 = wp::slice_t(var_2, var_2, var_22);
        var_23 = wp::slice_t(var_3, var_3, var_24);
        var_25 = wp::view(var_sensor_contact_criteria_in, var_21, var_23);
        var_27 = wp::tile_load<wp::float32, true, false, 256>(var_25, var_28);
        // matchid_tile = wp.tile_load(sensor_contact_matchid_in[worldid, contactsensorid], shape=maxmatch)       <L 2499>
        var_29 = wp::slice_t(var_2, var_2, var_30);
        var_31 = wp::slice_t(var_3, var_3, var_32);
        var_33 = wp::view(var_sensor_contact_matchid_in, var_29, var_31);
        var_34 = wp::tile_load<wp::int32, true, false, 256>(var_33, var_35);
        // wp.tile_sort(criteria_tile, matchid_tile)                                              <L 2500>
        wp::tile_sort(var_27, var_34);
        // wp.tile_store(sensor_contact_matchid_out[worldid, contactsensorid], matchid_tile)       <L 2501>
        var_36 = wp::slice_t(var_2, var_2, var_37);
        var_38 = wp::slice_t(var_3, var_3, var_39);
        var_40 = wp::view(var_sensor_contact_matchid_out, var_36, var_38);
        wp::tile_store<wp::int32, true, false>(var_40, var_41, var_34);
    }
}

