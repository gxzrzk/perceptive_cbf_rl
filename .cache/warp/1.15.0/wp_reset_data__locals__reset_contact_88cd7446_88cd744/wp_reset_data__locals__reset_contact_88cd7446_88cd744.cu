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



extern "C" __global__ void reset_data__locals__reset_contact_328e86bd_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::int32> var_nacon_in,
    wp::array_t<bool> var_reset_in,
    wp::int32 var_nefcaddress,
    wp::array_t<wp::float32> var_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_out,
    wp::array_t<wp::float32> var_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_solimp_out,
    wp::array_t<wp::int32> var_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_flex_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_elem_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_vert_out,
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out)
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
        wp::int32* var_2;
        bool var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        const bool var_8 = true;
        const wp::int32 var_9 = 0;
        bool var_10;
        bool* var_11;
        bool var_12;
        bool var_13;
        const wp::float32 var_14 = 0.0;
        const wp::float32 var_15 = 0.0;
        wp::vec_t<3, wp::float32> var_16;
        const wp::float32 var_17 = 0.0;
        const wp::float32 var_18 = 0.0;
        const wp::float32 var_19 = 0.0;
        const wp::float32 var_20 = 0.0;
        const wp::float32 var_21 = 0.0;
        const wp::float32 var_22 = 0.0;
        const wp::float32 var_23 = 0.0;
        const wp::float32 var_24 = 0.0;
        const wp::float32 var_25 = 0.0;
        wp::mat_t<3, 3, wp::float32> var_26;
        const wp::float32 var_27 = 0.0;
        const wp::float32 var_28 = 0.0;
        const wp::float32 var_29 = 0.0;
        const wp::float32 var_30 = 0.0;
        const wp::float32 var_31 = 0.0;
        const wp::float32 var_32 = 0.0;
        wp::vec_t<5, wp::float32> var_33;
        const wp::float32 var_34 = 0.0;
        const wp::float32 var_35 = 0.0;
        wp::vec_t<2, wp::float32> var_36;
        const wp::float32 var_37 = 0.0;
        const wp::float32 var_38 = 0.0;
        wp::vec_t<2, wp::float32> var_39;
        const wp::float32 var_40 = 0.0;
        const wp::float32 var_41 = 0.0;
        const wp::float32 var_42 = 0.0;
        const wp::float32 var_43 = 0.0;
        const wp::float32 var_44 = 0.0;
        wp::vec_t<5, wp::float32> var_45;
        const wp::int32 var_46 = 0;
        const wp::int32 var_47 = 0;
        const wp::int32 var_48 = 0;
        wp::vec_t<2, wp::int32> var_49;
        wp::shape_t* var_50;
        const wp::int32 var_51 = 0;
        wp::int32 var_52;
        wp::shape_t var_53;
        const wp::int32 var_54 = 0;
        bool var_55;
        const wp::int32 var_56 = 0;
        const wp::int32 var_57 = 0;
        wp::vec_t<2, wp::int32> var_58;
        wp::shape_t* var_59;
        const wp::int32 var_60 = 0;
        wp::int32 var_61;
        wp::shape_t var_62;
        const wp::int32 var_63 = 0;
        bool var_64;
        const wp::int32 var_65 = 0;
        const wp::int32 var_66 = 0;
        wp::vec_t<2, wp::int32> var_67;
        wp::shape_t* var_68;
        const wp::int32 var_69 = 0;
        wp::int32 var_70;
        wp::shape_t var_71;
        const wp::int32 var_72 = 0;
        bool var_73;
        const wp::int32 var_74 = 0;
        const wp::int32 var_75 = 0;
        wp::vec_t<2, wp::int32> var_76;
        wp::range_t var_77;
        wp::int32 var_78;
        const wp::int32 var_79 = -1;
        const wp::int32 var_80 = 0;
        const wp::int32 var_81 = 0;
        const wp::int32 var_82 = 0;
        //---------
        // forward
        // def reset_contact(                                                                     <L 2510>
        // conid = wp.tid()                                                                       <L 2535>
        var_0 = builtin_tid1d();
        // if conid >= nacon_in[0]:                                                               <L 2537>
        var_2 = wp::address(var_nacon_in, var_1);
        var_4 = wp::load(var_2);
        var_3 = (var_0 >= var_4);
        if (var_3) {
            // return                                                                             <L 2538>
            continue;
        }
        // worldid = contact_worldid_out[conid]                                                   <L 2540>
        var_5 = wp::address(var_contact_worldid_out, var_0);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // if wp.static(reset is not None):                                                       <L 2541>
        // if worldid >= 0:                                                                       <L 2542>
        var_10 = (var_6 >= var_9);
        if (var_10) {
            // if not reset_in[worldid]:                                                          <L 2543>
            var_11 = wp::address(var_reset_in, var_6);
            var_13 = wp::load(var_11);
            var_12 = wp::unot(var_13);
            if (var_12) {
                // return                                                                         <L 2544>
                continue;
            }
        }
        // contact_dist_out[conid] = 0.0                                                          <L 2546>
        wp::array_store(var_contact_dist_out, var_0, var_14);
        // contact_pos_out[conid] = wp.vec3(0.0)                                                  <L 2547>
        var_16 = wp::vec_t<3, wp::float32>(var_15);
        wp::array_store(var_contact_pos_out, var_0, var_16);
        // contact_frame_out[conid] = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)       <L 2548>
        var_26 = wp::mat_t<3, 3, wp::float32>(var_17, var_18, var_19, var_20, var_21, var_22, var_23, var_24, var_25);
        wp::array_store(var_contact_frame_out, var_0, var_26);
        // contact_includemargin_out[conid] = 0.0                                                 <L 2549>
        wp::array_store(var_contact_includemargin_out, var_0, var_27);
        // contact_friction_out[conid] = types.vec5(0.0, 0.0, 0.0, 0.0, 0.0)                      <L 2550>
        var_33 = wp::vec_t<5, wp::float32>({var_28, var_29, var_30, var_31, var_32});
        wp::array_store(var_contact_friction_out, var_0, var_33);
        // contact_solref_out[conid] = wp.vec2(0.0, 0.0)                                          <L 2551>
        var_36 = wp::vec_t<2, wp::float32>(var_34, var_35);
        wp::array_store(var_contact_solref_out, var_0, var_36);
        // contact_solreffriction_out[conid] = wp.vec2(0.0, 0.0)                                  <L 2552>
        var_39 = wp::vec_t<2, wp::float32>(var_37, var_38);
        wp::array_store(var_contact_solreffriction_out, var_0, var_39);
        // contact_solimp_out[conid] = types.vec5(0.0, 0.0, 0.0, 0.0, 0.0)                        <L 2553>
        var_45 = wp::vec_t<5, wp::float32>({var_40, var_41, var_42, var_43, var_44});
        wp::array_store(var_contact_solimp_out, var_0, var_45);
        // contact_dim_out[conid] = 0                                                             <L 2554>
        wp::array_store(var_contact_dim_out, var_0, var_46);
        // contact_geom_out[conid] = wp.vec2i(0, 0)                                               <L 2555>
        var_49 = wp::vec_t<2, wp::int32>(var_47, var_48);
        wp::array_store(var_contact_geom_out, var_0, var_49);
        // if contact_flex_out.shape[0] > 0:                                                      <L 2556>
        var_50 = &(var_contact_flex_out.shape);
        var_53 = wp::load(var_50);
        var_52 = wp::extract(var_53, var_51);
        var_55 = (var_52 > var_54);
        if (var_55) {
            // contact_flex_out[conid] = wp.vec2i(0, 0)                                           <L 2557>
            var_58 = wp::vec_t<2, wp::int32>(var_56, var_57);
            wp::array_store(var_contact_flex_out, var_0, var_58);
        }
        // if contact_elem_out.shape[0] > 0:                                                      <L 2558>
        var_59 = &(var_contact_elem_out.shape);
        var_62 = wp::load(var_59);
        var_61 = wp::extract(var_62, var_60);
        var_64 = (var_61 > var_63);
        if (var_64) {
            // contact_elem_out[conid] = wp.vec2i(0, 0)                                           <L 2559>
            var_67 = wp::vec_t<2, wp::int32>(var_65, var_66);
            wp::array_store(var_contact_elem_out, var_0, var_67);
        }
        // if contact_vert_out.shape[0] > 0:                                                      <L 2560>
        var_68 = &(var_contact_vert_out.shape);
        var_71 = wp::load(var_68);
        var_70 = wp::extract(var_71, var_69);
        var_73 = (var_70 > var_72);
        if (var_73) {
            // contact_vert_out[conid] = wp.vec2i(0, 0)                                           <L 2561>
            var_76 = wp::vec_t<2, wp::int32>(var_74, var_75);
            wp::array_store(var_contact_vert_out, var_0, var_76);
        }
        // for i in range(nefcaddress):                                                           <L 2562>
        var_77 = wp::range(var_nefcaddress);
        start_for_2:;
            if (iter_cmp(var_77) == 0) goto end_for_2;
            var_78 = wp::iter_next(var_77);
            // contact_efc_address_out[conid, i] = -1                                             <L 2563>
            wp::array_store(var_contact_efc_address_out, var_0, var_78, var_79);
            goto start_for_2;
        end_for_2:;
        // contact_worldid_out[conid] = 0                                                         <L 2564>
        wp::array_store(var_contact_worldid_out, var_0, var_80);
        // contact_type_out[conid] = 0                                                            <L 2565>
        wp::array_store(var_contact_type_out, var_0, var_81);
        // contact_geomcollisionid_out[conid] = 0                                                 <L 2566>
        wp::array_store(var_contact_geomcollisionid_out, var_0, var_82);
    }
}

