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



extern "C" __global__ void _next_time_builder__locals___next_time_0a0438b4_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::float32> var_opt_timestep,
    bool var_is_sparse,
    wp::array_t<wp::int32> var_nefc_in,
    wp::array_t<wp::float32> var_time_in,
    wp::array_t<wp::int32> var_efc_J_rownnz_in,
    wp::array_t<wp::int32> var_efc_J_rowadr_in,
    wp::int32 var_nworld_in,
    wp::int32 var_naconmax_in,
    wp::int32 var_njmax_in,
    wp::int32 var_njmax_nnz_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::array_t<wp::int32> var_ncollision_in,
    wp::array_t<wp::float32> var_time_out,
    wp::array_t<wp::int32> var_overflow_out)
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
        wp::float32* var_1;
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        bool var_14;
        const bool var_15 = true;
        const wp::str var_16 = "nefc overflow - please increase njmax to %u\n";
        wp::int32* var_17;
        const wp::int32 var_18 = 1;
        wp::int32 var_19;
        wp::int32 var_20;
        bool var_21;
        const wp::int32 var_22 = 0;
        bool var_23;
        wp::int32 var_24;
        const wp::int32 var_25 = 1;
        wp::int32 var_26;
        wp::int32* var_27;
        wp::int32* var_28;
        wp::int32 var_29;
        wp::int32 var_30;
        wp::int32 var_31;
        bool var_32;
        const bool var_33 = true;
        const wp::str var_34 = "njmax_nnz overflow - please increase njmax_nnz to %u\n";
        wp::int32* var_35;
        const wp::int32 var_36 = 2;
        wp::int32 var_37;
        wp::int32 var_38;
        const wp::int32 var_39 = 0;
        wp::int32* var_40;
        wp::int32 var_41;
        wp::int32 var_42;
        bool var_43;
        bool var_44;
        const wp::int32 var_45 = 0;
        bool var_46;
        const bool var_47 = true;
        wp::float32 var_48;
        wp::float32 var_49;
        wp::float32 var_50;
        wp::float32 var_51;
        wp::int32 var_52;
        const wp::str var_53 = "broadphase overflow - please increase nconmax to %u or naconmax to %u\n";
        wp::int32* var_54;
        const wp::int32 var_55 = 4;
        wp::int32 var_56;
        wp::int32 var_57;
        const wp::int32 var_58 = 0;
        wp::int32* var_59;
        wp::int32 var_60;
        wp::int32 var_61;
        bool var_62;
        bool var_63;
        const wp::int32 var_64 = 0;
        bool var_65;
        const bool var_66 = true;
        wp::float32 var_67;
        wp::float32 var_68;
        wp::float32 var_69;
        wp::float32 var_70;
        wp::int32 var_71;
        const wp::str var_72 = "narrowphase overflow - please increase nconmax to %u or naconmax to %u\n";
        wp::int32 var_73;
        wp::int32* var_74;
        const wp::int32 var_75 = 8;
        wp::int32 var_76;
        wp::int32 var_77;
        wp::int32 var_78;
        //---------
        // forward
        // def _next_time(                                                                        <L 224>
        // worldid = wp.tid()                                                                     <L 243>
        var_0 = builtin_tid1d();
        // time_out[worldid] = time_in[worldid] + opt_timestep[worldid % opt_timestep.shape[0]]       <L 244>
        var_1 = wp::address(var_time_in, var_0);
        var_2 = &(var_opt_timestep.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        var_7 = wp::address(var_opt_timestep, var_6);
        var_9 = wp::load(var_1);
        var_10 = wp::load(var_7);
        var_8 = wp::add(var_9, var_10);
        wp::array_store(var_time_out, var_0, var_8);
        // nefc = nefc_in[worldid]                                                                <L 245>
        var_11 = wp::address(var_nefc_in, var_0);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // if nefc > njmax_in:                                                                    <L 247>
        var_14 = (var_12 > var_njmax_in);
        if (var_14) {
            // if wp.static(warn_overflow):                                                       <L 248>
            // wp.printf("nefc overflow - please increase njmax to %u\n", nefc)                   <L 249>
            printf(var_16, var_12);
            // overflow_out[worldid] = overflow_out[worldid] | OverflowType.NEFC                  <L 250>
            var_17 = wp::address(var_overflow_out, var_0);
            var_20 = wp::load(var_17);
            var_19 = wp::bit_or(var_20, var_18);
            wp::array_store(var_overflow_out, var_0, var_19);
        }
        if (!var_14) {
            // elif nefc > 0 and is_sparse:                                                       <L 251>
            var_23 = (var_12 > var_22);
            var_21 = var_23;
            if (var_21) {
                var_21 = var_21 && var_is_sparse;
            }
            if (var_21) {
                // efcid = wp.min(nefc, njmax_in) - 1                                             <L 252>
                var_24 = wp::min(var_12, var_njmax_in);
                var_26 = wp::sub(var_24, var_25);
                // efc_nnz = efc_J_rowadr_in[worldid, efcid] + efc_J_rownnz_in[worldid, efcid]       <L 253>
                var_27 = wp::address(var_efc_J_rowadr_in, var_0, var_26);
                var_28 = wp::address(var_efc_J_rownnz_in, var_0, var_26);
                var_30 = wp::load(var_27);
                var_31 = wp::load(var_28);
                var_29 = wp::add(var_30, var_31);
                // if efc_nnz > njmax_nnz_in:                                                     <L 254>
                var_32 = (var_29 > var_njmax_nnz_in);
                if (var_32) {
                    // if wp.static(warn_overflow):                                               <L 255>
                    // wp.printf("njmax_nnz overflow - please increase njmax_nnz to %u\n", efc_nnz)       <L 256>
                    printf(var_34, var_29);
                    // overflow_out[worldid] = overflow_out[worldid] | OverflowType.NJMAX_NNZ       <L 257>
                    var_35 = wp::address(var_overflow_out, var_0);
                    var_38 = wp::load(var_35);
                    var_37 = wp::bit_or(var_38, var_36);
                    wp::array_store(var_overflow_out, var_0, var_37);
                }
            }
        }
        // ncollision = ncollision_in[0]                                                          <L 259>
        var_40 = wp::address(var_ncollision_in, var_39);
        var_42 = wp::load(var_40);
        var_41 = wp::copy(var_42);
        // if ncollision > naconmax_in:                                                           <L 260>
        var_43 = (var_41 > var_naconmax_in);
        if (var_43) {
            // if worldid == 0 and wp.static(warn_overflow):                                      <L 261>
            var_46 = (var_0 == var_45);
            var_44 = var_46;
            if (var_44) {
                var_44 = var_44 && var_47;
            }
            if (var_44) {
                // nconmax = int(wp.ceil(float(ncollision) / float(nworld_in)))                   <L 262>
                var_48 = wp::float(var_41);
                var_49 = wp::float(var_nworld_in);
                var_50 = wp::div(var_48, var_49);
                var_51 = wp::ceil(var_50);
                var_52 = wp::int(var_51);
                // wp.printf("broadphase overflow - please increase nconmax to %u or naconmax to %u\n", nconmax, ncollision)       <L 263>
                printf(var_53, var_52, var_41);
            }
            // overflow_out[worldid] = overflow_out[worldid] | OverflowType.BROADPHASE            <L 264>
            var_54 = wp::address(var_overflow_out, var_0);
            var_57 = wp::load(var_54);
            var_56 = wp::bit_or(var_57, var_55);
            wp::array_store(var_overflow_out, var_0, var_56);
        }
        // nacon = nacon_in[0]                                                                    <L 266>
        var_59 = wp::address(var_nacon_in, var_58);
        var_61 = wp::load(var_59);
        var_60 = wp::copy(var_61);
        // if nacon > naconmax_in:                                                                <L 267>
        var_62 = (var_60 > var_naconmax_in);
        if (var_62) {
            // if worldid == 0 and wp.static(warn_overflow):                                      <L 268>
            var_65 = (var_0 == var_64);
            var_63 = var_65;
            if (var_63) {
                var_63 = var_63 && var_66;
            }
            if (var_63) {
                // nconmax = int(wp.ceil(float(nacon) / float(nworld_in)))                        <L 269>
                var_67 = wp::float(var_60);
                var_68 = wp::float(var_nworld_in);
                var_69 = wp::div(var_67, var_68);
                var_70 = wp::ceil(var_69);
                var_71 = wp::int(var_70);
                // wp.printf("narrowphase overflow - please increase nconmax to %u or naconmax to %u\n", nconmax, nacon)       <L 270>
                printf(var_72, var_71, var_60);
            }
            var_73 = wp::where(var_63, var_71, var_52);
            // overflow_out[worldid] = overflow_out[worldid] | OverflowType.NARROWPHASE           <L 271>
            var_74 = wp::address(var_overflow_out, var_0);
            var_77 = wp::load(var_74);
            var_76 = wp::bit_or(var_77, var_75);
            wp::array_store(var_overflow_out, var_0, var_76);
        }
        var_78 = wp::where(var_62, var_73, var_52);
    }
}

