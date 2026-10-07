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


struct Geom_3242f8a8
{
    wp::vec_t<3, wp::float32> pos;
    wp::mat_t<3, 3, wp::float32> rot;
    wp::vec_t<3, wp::float32> normal;
    wp::vec_t<3, wp::float32> size;
    wp::float32 margin;
    wp::mat_t<6, 3, wp::float32> hfprism;
    wp::int32 vertadr;
    wp::int32 vertnum;
    wp::array_t<wp::vec_t<3, wp::float32>> vert;
    wp::int32 graphadr;
    wp::array_t<wp::int32> graph;
    wp::int32 mesh_polynum;
    wp::int32 mesh_polyadr;
    wp::array_t<wp::vec_t<3, wp::float32>> mesh_polynormal;
    wp::array_t<wp::int32> mesh_polyvertadr;
    wp::array_t<wp::int32> mesh_polyvertnum;
    wp::array_t<wp::int32> mesh_polyvert;
    wp::array_t<wp::int32> mesh_polymapadr;
    wp::array_t<wp::int32> mesh_polymapnum;
    wp::array_t<wp::int32> mesh_polymap;
    wp::int32 index;


    Geom_3242f8a8() = default;
    CUDA_CALLABLE Geom_3242f8a8(wp::vec_t<3, wp::float32> const& pos,
    wp::mat_t<3, 3, wp::float32> const& rot = {},
    wp::vec_t<3, wp::float32> const& normal = {},
    wp::vec_t<3, wp::float32> const& size = {},
    wp::float32 const& margin = {},
    wp::mat_t<6, 3, wp::float32> const& hfprism = {},
    wp::int32 const& vertadr = {},
    wp::int32 const& vertnum = {},
    wp::array_t<wp::vec_t<3, wp::float32>> const& vert = {},
    wp::int32 const& graphadr = {},
    wp::array_t<wp::int32> const& graph = {},
    wp::int32 const& mesh_polynum = {},
    wp::int32 const& mesh_polyadr = {},
    wp::array_t<wp::vec_t<3, wp::float32>> const& mesh_polynormal = {},
    wp::array_t<wp::int32> const& mesh_polyvertadr = {},
    wp::array_t<wp::int32> const& mesh_polyvertnum = {},
    wp::array_t<wp::int32> const& mesh_polyvert = {},
    wp::array_t<wp::int32> const& mesh_polymapadr = {},
    wp::array_t<wp::int32> const& mesh_polymapnum = {},
    wp::array_t<wp::int32> const& mesh_polymap = {},
    wp::int32 const& index = {})
        : pos{pos}
        , rot{rot}
        , normal{normal}
        , size{size}
        , margin{margin}
        , hfprism{hfprism}
        , vertadr{vertadr}
        , vertnum{vertnum}
        , vert{vert}
        , graphadr{graphadr}
        , graph{graph}
        , mesh_polynum{mesh_polynum}
        , mesh_polyadr{mesh_polyadr}
        , mesh_polynormal{mesh_polynormal}
        , mesh_polyvertadr{mesh_polyvertadr}
        , mesh_polyvertnum{mesh_polyvertnum}
        , mesh_polyvert{mesh_polyvert}
        , mesh_polymapadr{mesh_polymapadr}
        , mesh_polymapnum{mesh_polymapnum}
        , mesh_polymap{mesh_polymap}
        , index{index}

    {
    }

    CUDA_CALLABLE Geom_3242f8a8& operator += (const Geom_3242f8a8& rhs)
    {    pos += rhs.pos;
    rot += rhs.rot;
    normal += rhs.normal;
    size += rhs.size;
    margin += rhs.margin;
    hfprism += rhs.hfprism;
    vertadr += rhs.vertadr;
    vertnum += rhs.vertnum;
    graphadr += rhs.graphadr;
    mesh_polynum += rhs.mesh_polynum;
    mesh_polyadr += rhs.mesh_polyadr;
    index += rhs.index;

        return *this;}

};

static CUDA_CALLABLE void adj_Geom_3242f8a8(wp::vec_t<3, wp::float32> const&,
    wp::mat_t<3, 3, wp::float32> const&,
    wp::vec_t<3, wp::float32> const&,
    wp::vec_t<3, wp::float32> const&,
    wp::float32 const&,
    wp::mat_t<6, 3, wp::float32> const&,
    wp::int32 const&,
    wp::int32 const&,
    wp::array_t<wp::vec_t<3, wp::float32>> const&,
    wp::int32 const&,
    wp::array_t<wp::int32> const&,
    wp::int32 const&,
    wp::int32 const&,
    wp::array_t<wp::vec_t<3, wp::float32>> const&,
    wp::array_t<wp::int32> const&,
    wp::array_t<wp::int32> const&,
    wp::array_t<wp::int32> const&,
    wp::array_t<wp::int32> const&,
    wp::array_t<wp::int32> const&,
    wp::array_t<wp::int32> const&,
    wp::int32 const&,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::mat_t<3, 3, wp::float32> & adj_rot,
    wp::vec_t<3, wp::float32> & adj_normal,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::float32 & adj_margin,
    wp::mat_t<6, 3, wp::float32> & adj_hfprism,
    wp::int32 & adj_vertadr,
    wp::int32 & adj_vertnum,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_vert,
    wp::int32 & adj_graphadr,
    wp::array_t<wp::int32> & adj_graph,
    wp::int32 & adj_mesh_polynum,
    wp::int32 & adj_mesh_polyadr,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_mesh_polynormal,
    wp::array_t<wp::int32> & adj_mesh_polyvertadr,
    wp::array_t<wp::int32> & adj_mesh_polyvertnum,
    wp::array_t<wp::int32> & adj_mesh_polyvert,
    wp::array_t<wp::int32> & adj_mesh_polymapadr,
    wp::array_t<wp::int32> & adj_mesh_polymapnum,
    wp::array_t<wp::int32> & adj_mesh_polymap,
    wp::int32 & adj_index,
    Geom_3242f8a8 & adj_ret)
{
    adj_pos += adj_ret.pos;
    adj_rot += adj_ret.rot;
    adj_normal += adj_ret.normal;
    adj_size += adj_ret.size;
    adj_margin += adj_ret.margin;
    adj_hfprism += adj_ret.hfprism;
    adj_vertadr += adj_ret.vertadr;
    adj_vertnum += adj_ret.vertnum;
    adj_vert = adj_ret.vert;
    adj_graphadr += adj_ret.graphadr;
    adj_graph = adj_ret.graph;
    adj_mesh_polynum += adj_ret.mesh_polynum;
    adj_mesh_polyadr += adj_ret.mesh_polyadr;
    adj_mesh_polynormal = adj_ret.mesh_polynormal;
    adj_mesh_polyvertadr = adj_ret.mesh_polyvertadr;
    adj_mesh_polyvertnum = adj_ret.mesh_polyvertnum;
    adj_mesh_polyvert = adj_ret.mesh_polyvert;
    adj_mesh_polymapadr = adj_ret.mesh_polymapadr;
    adj_mesh_polymapnum = adj_ret.mesh_polymapnum;
    adj_mesh_polymap = adj_ret.mesh_polymap;
    adj_index += adj_ret.index;
}

// Required when compiling adjoints.
CUDA_CALLABLE Geom_3242f8a8 add(const Geom_3242f8a8& a, const Geom_3242f8a8& b)
{
    return Geom_3242f8a8();
}

CUDA_CALLABLE void adj_atomic_add(Geom_3242f8a8* p, Geom_3242f8a8 t)
{
    wp::adj_atomic_add(&p->pos, t.pos);
    wp::adj_atomic_add(&p->rot, t.rot);
    wp::adj_atomic_add(&p->normal, t.normal);
    wp::adj_atomic_add(&p->size, t.size);
    wp::adj_atomic_add(&p->margin, t.margin);
    wp::adj_atomic_add(&p->hfprism, t.hfprism);
    wp::adj_atomic_add(&p->vertadr, t.vertadr);
    wp::adj_atomic_add(&p->vertnum, t.vertnum);
    wp::adj_atomic_add(&p->vert, t.vert);
    wp::adj_atomic_add(&p->graphadr, t.graphadr);
    wp::adj_atomic_add(&p->graph, t.graph);
    wp::adj_atomic_add(&p->mesh_polynum, t.mesh_polynum);
    wp::adj_atomic_add(&p->mesh_polyadr, t.mesh_polyadr);
    wp::adj_atomic_add(&p->mesh_polynormal, t.mesh_polynormal);
    wp::adj_atomic_add(&p->mesh_polyvertadr, t.mesh_polyvertadr);
    wp::adj_atomic_add(&p->mesh_polyvertnum, t.mesh_polyvertnum);
    wp::adj_atomic_add(&p->mesh_polyvert, t.mesh_polyvert);
    wp::adj_atomic_add(&p->mesh_polymapadr, t.mesh_polymapadr);
    wp::adj_atomic_add(&p->mesh_polymapnum, t.mesh_polymapnum);
    wp::adj_atomic_add(&p->mesh_polymap, t.mesh_polymap);
    wp::adj_atomic_add(&p->index, t.index);
}




// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_core.py:290
static CUDA_CALLABLE void contact_margin_gap_0(
    wp::array_t<wp::float32> var_geom_margin,
    wp::array_t<wp::float32> var_geom_gap,
    wp::array_t<wp::float32> var_pair_margin,
    wp::array_t<wp::float32> var_pair_gap,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::int32 var_pairid,
    wp::int32 var_worldid,
    wp::float32 & ret_0,
    wp::float32 & ret_1)
{
    //---------
    // primal vars
    const wp::int32 var_0 = -1;
    bool var_1;
    wp::shape_t* var_2;
    const wp::int32 var_3 = 0;
    wp::int32 var_4;
    wp::shape_t var_5;
    wp::int32 var_6;
    wp::float32* var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::shape_t* var_10;
    const wp::int32 var_11 = 0;
    wp::int32 var_12;
    wp::shape_t var_13;
    wp::int32 var_14;
    wp::float32* var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    const wp::int32 var_18 = 0;
    wp::int32 var_19;
    const wp::int32 var_20 = 1;
    wp::int32 var_21;
    wp::shape_t* var_22;
    const wp::int32 var_23 = 0;
    wp::int32 var_24;
    wp::shape_t var_25;
    wp::int32 var_26;
    wp::shape_t* var_27;
    const wp::int32 var_28 = 0;
    wp::int32 var_29;
    wp::shape_t var_30;
    wp::int32 var_31;
    wp::float32* var_32;
    wp::float32* var_33;
    wp::float32 var_34;
    wp::float32 var_35;
    wp::float32 var_36;
    wp::float32* var_37;
    wp::float32* var_38;
    wp::float32 var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::float32 var_42;
    wp::float32 var_43;
    //---------
    // forward
    // def contact_margin_gap(                                                                <L 291>
    // if pairid > -1:                                                                        <L 302>
    var_1 = (var_pairid > var_0);
    if (var_1) {
        // margin = pair_margin[worldid % pair_margin.shape[0], pairid]                       <L 303>
        var_2 = &(var_pair_margin.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_worldid, var_4);
        var_7 = wp::address(var_pair_margin, var_6, var_pairid);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // gap = pair_gap[worldid % pair_gap.shape[0], pairid]                                <L 304>
        var_10 = &(var_pair_gap.shape);
        var_13 = wp::load(var_10);
        var_12 = wp::extract(var_13, var_11);
        var_14 = wp::mod(var_worldid, var_12);
        var_15 = wp::address(var_pair_gap, var_14, var_pairid);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
    }
    if (!var_1) {
        // g1 = geoms[0]                                                                      <L 306>
        var_19 = wp::extract(var_geoms, var_18);
        // g2 = geoms[1]                                                                      <L 307>
        var_21 = wp::extract(var_geoms, var_20);
        // margin_id = worldid % geom_margin.shape[0]                                         <L 308>
        var_22 = &(var_geom_margin.shape);
        var_25 = wp::load(var_22);
        var_24 = wp::extract(var_25, var_23);
        var_26 = wp::mod(var_worldid, var_24);
        // gap_id = worldid % geom_gap.shape[0]                                               <L 309>
        var_27 = &(var_geom_gap.shape);
        var_30 = wp::load(var_27);
        var_29 = wp::extract(var_30, var_28);
        var_31 = wp::mod(var_worldid, var_29);
        // margin = geom_margin[margin_id, g1] + geom_margin[margin_id, g2]                   <L 310>
        var_32 = wp::address(var_geom_margin, var_26, var_19);
        var_33 = wp::address(var_geom_margin, var_26, var_21);
        var_35 = wp::load(var_32);
        var_36 = wp::load(var_33);
        var_34 = wp::add(var_35, var_36);
        // gap = geom_gap[gap_id, g1] + geom_gap[gap_id, g2]                                  <L 311>
        var_37 = wp::address(var_geom_gap, var_31, var_19);
        var_38 = wp::address(var_geom_gap, var_31, var_21);
        var_40 = wp::load(var_37);
        var_41 = wp::load(var_38);
        var_39 = wp::add(var_40, var_41);
    }
    var_42 = wp::where(var_1, var_8, var_34);
    var_43 = wp::where(var_1, var_16, var_39);
    // return margin, gap                                                                     <L 313>
    ret_0 = var_42;
    ret_1 = var_43;
    return;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/math.py:0
static CUDA_CALLABLE wp::float32 safe_div_0(
    wp::float32 var_x,
    wp::float32 var_y)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    bool var_1;
    const wp::float32 var_2 = 1e-15;
    const wp::float32 var_3 = 1e-15;
    wp::float32 var_4;
    wp::float32 var_5;
    //---------
    // forward
    // def safe_div(x: Any, y: Any) -> Any:                                                   <L 1>
    // return x / wp.where(y != 0.0, y, types.MJ_MINVAL)                                      <L 2>
    var_1 = (var_y != var_0);
    var_4 = wp::where(var_1, var_y, var_3);
    var_5 = wp::div(var_x, var_4);
    return var_5;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_core.py:316
static CUDA_CALLABLE void contact_material_params_0(
    wp::array_t<wp::int32> var_geom_condim,
    wp::array_t<wp::int32> var_geom_priority,
    wp::array_t<wp::float32> var_geom_solmix,
    wp::array_t<wp::vec_t<2, wp::float32>> var_geom_solref,
    wp::array_t<wp::vec_t<5, wp::float32>> var_geom_solimp,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_friction,
    wp::array_t<wp::int32> var_pair_dim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_pair_solref,
    wp::array_t<wp::vec_t<2, wp::float32>> var_pair_solreffriction,
    wp::array_t<wp::vec_t<5, wp::float32>> var_pair_solimp,
    wp::array_t<wp::vec_t<5, wp::float32>> var_pair_friction,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::int32 var_pairid,
    wp::int32 var_worldid,
    wp::int32 & ret_0,
    wp::vec_t<5, wp::float32> & ret_1,
    wp::vec_t<2, wp::float32> & ret_2,
    wp::vec_t<2, wp::float32> & ret_3,
    wp::vec_t<5, wp::float32> & ret_4)
{
    //---------
    // primal vars
    const wp::int32 var_0 = -1;
    bool var_1;
    wp::int32* var_2;
    wp::int32 var_3;
    wp::int32 var_4;
    wp::shape_t* var_5;
    const wp::int32 var_6 = 0;
    wp::int32 var_7;
    wp::shape_t var_8;
    wp::int32 var_9;
    wp::vec_t<5, wp::float32>* var_10;
    wp::vec_t<5, wp::float32> var_11;
    wp::vec_t<5, wp::float32> var_12;
    wp::shape_t* var_13;
    const wp::int32 var_14 = 0;
    wp::int32 var_15;
    wp::shape_t var_16;
    wp::int32 var_17;
    wp::vec_t<2, wp::float32>* var_18;
    wp::vec_t<2, wp::float32> var_19;
    wp::vec_t<2, wp::float32> var_20;
    wp::shape_t* var_21;
    const wp::int32 var_22 = 0;
    wp::int32 var_23;
    wp::shape_t var_24;
    wp::int32 var_25;
    wp::vec_t<2, wp::float32>* var_26;
    wp::vec_t<2, wp::float32> var_27;
    wp::vec_t<2, wp::float32> var_28;
    wp::shape_t* var_29;
    const wp::int32 var_30 = 0;
    wp::int32 var_31;
    wp::shape_t var_32;
    wp::int32 var_33;
    wp::vec_t<5, wp::float32>* var_34;
    wp::vec_t<5, wp::float32> var_35;
    wp::vec_t<5, wp::float32> var_36;
    const wp::int32 var_37 = 0;
    wp::int32 var_38;
    const wp::int32 var_39 = 1;
    wp::int32 var_40;
    wp::shape_t* var_41;
    const wp::int32 var_42 = 0;
    wp::int32 var_43;
    wp::shape_t var_44;
    wp::int32 var_45;
    wp::shape_t* var_46;
    const wp::int32 var_47 = 0;
    wp::int32 var_48;
    wp::shape_t var_49;
    wp::int32 var_50;
    wp::shape_t* var_51;
    const wp::int32 var_52 = 0;
    wp::int32 var_53;
    wp::shape_t var_54;
    wp::int32 var_55;
    wp::shape_t* var_56;
    const wp::int32 var_57 = 0;
    wp::int32 var_58;
    wp::shape_t var_59;
    wp::int32 var_60;
    wp::float32* var_61;
    wp::float32 var_62;
    wp::float32 var_63;
    wp::float32* var_64;
    wp::float32 var_65;
    wp::float32 var_66;
    wp::int32* var_67;
    wp::int32 var_68;
    wp::int32 var_69;
    wp::int32* var_70;
    wp::int32 var_71;
    wp::int32 var_72;
    wp::int32* var_73;
    wp::int32 var_74;
    wp::int32 var_75;
    wp::int32* var_76;
    wp::int32 var_77;
    wp::int32 var_78;
    bool var_79;
    const wp::float32 var_80 = 1.0;
    wp::int32 var_81;
    wp::vec_t<3, wp::float32>* var_82;
    wp::vec_t<3, wp::float32> var_83;
    wp::vec_t<3, wp::float32> var_84;
    wp::int32 var_85;
    bool var_86;
    const wp::float32 var_87 = 0.0;
    wp::int32 var_88;
    wp::vec_t<3, wp::float32>* var_89;
    wp::vec_t<3, wp::float32> var_90;
    wp::vec_t<3, wp::float32> var_91;
    wp::int32 var_92;
    wp::float32 var_93;
    wp::vec_t<3, wp::float32> var_94;
    wp::float32 var_95;
    wp::float32 var_96;
    bool var_97;
    const wp::float32 var_98 = 1e-15;
    bool var_99;
    bool var_100;
    const wp::float32 var_101 = 0.5;
    wp::float32 var_102;
    bool var_103;
    bool var_104;
    bool var_105;
    const wp::float32 var_106 = 0.0;
    wp::float32 var_107;
    bool var_108;
    bool var_109;
    bool var_110;
    const wp::float32 var_111 = 1.0;
    wp::float32 var_112;
    wp::int32 var_113;
    wp::vec_t<3, wp::float32>* var_114;
    wp::vec_t<3, wp::float32>* var_115;
    wp::vec_t<3, wp::float32> var_116;
    wp::vec_t<3, wp::float32> var_117;
    wp::vec_t<3, wp::float32> var_118;
    wp::int32 var_119;
    wp::float32 var_120;
    wp::vec_t<3, wp::float32> var_121;
    wp::int32 var_122;
    wp::float32 var_123;
    wp::vec_t<3, wp::float32> var_124;
    const wp::int32 var_125 = 0;
    wp::float32 var_126;
    const wp::int32 var_127 = 0;
    wp::float32 var_128;
    const wp::int32 var_129 = 1;
    wp::float32 var_130;
    const wp::int32 var_131 = 2;
    wp::float32 var_132;
    const wp::int32 var_133 = 2;
    wp::float32 var_134;
    wp::vec_t<5, wp::float32> var_135;
    bool var_136;
    wp::vec_t<2, wp::float32>* var_137;
    const wp::int32 var_138 = 0;
    wp::float32 var_139;
    wp::vec_t<2, wp::float32> var_140;
    const wp::float32 var_141 = 0.0;
    bool var_142;
    wp::vec_t<2, wp::float32>* var_143;
    const wp::int32 var_144 = 0;
    wp::float32 var_145;
    wp::vec_t<2, wp::float32> var_146;
    const wp::float32 var_147 = 0.0;
    bool var_148;
    wp::vec_t<2, wp::float32>* var_149;
    wp::vec_t<2, wp::float32> var_150;
    wp::vec_t<2, wp::float32> var_151;
    const wp::float32 var_152 = 1.0;
    wp::float32 var_153;
    wp::vec_t<2, wp::float32>* var_154;
    wp::vec_t<2, wp::float32> var_155;
    wp::vec_t<2, wp::float32> var_156;
    wp::vec_t<2, wp::float32> var_157;
    wp::vec_t<2, wp::float32> var_158;
    wp::vec_t<2, wp::float32>* var_159;
    wp::vec_t<2, wp::float32>* var_160;
    wp::vec_t<2, wp::float32> var_161;
    wp::vec_t<2, wp::float32> var_162;
    wp::vec_t<2, wp::float32> var_163;
    wp::vec_t<2, wp::float32> var_164;
    const wp::float32 var_165 = 0.0;
    const wp::float32 var_166 = 0.0;
    wp::vec_t<2, wp::float32> var_167;
    wp::vec_t<5, wp::float32>* var_168;
    wp::vec_t<5, wp::float32> var_169;
    wp::vec_t<5, wp::float32> var_170;
    const wp::float32 var_171 = 1.0;
    wp::float32 var_172;
    wp::vec_t<5, wp::float32>* var_173;
    wp::vec_t<5, wp::float32> var_174;
    wp::vec_t<5, wp::float32> var_175;
    wp::vec_t<5, wp::float32> var_176;
    wp::int32 var_177;
    wp::vec_t<5, wp::float32> var_178;
    wp::vec_t<2, wp::float32> var_179;
    wp::vec_t<2, wp::float32> var_180;
    wp::vec_t<5, wp::float32> var_181;
    const wp::float32 var_182 = 1e-05;
    const wp::int32 var_183 = 0;
    wp::float32 var_184;
    wp::float32 var_185;
    const wp::int32 var_186 = 1;
    wp::float32 var_187;
    wp::float32 var_188;
    const wp::int32 var_189 = 2;
    wp::float32 var_190;
    wp::float32 var_191;
    const wp::int32 var_192 = 3;
    wp::float32 var_193;
    wp::float32 var_194;
    const wp::int32 var_195 = 4;
    wp::float32 var_196;
    wp::float32 var_197;
    wp::vec_t<5, wp::float32> var_198;
    //---------
    // forward
    // def contact_material_params(                                                           <L 317>
    // if pairid > -1:                                                                        <L 335>
    var_1 = (var_pairid > var_0);
    if (var_1) {
        // condim = pair_dim[pairid]                                                          <L 336>
        var_2 = wp::address(var_pair_dim, var_pairid);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // friction = pair_friction[worldid % pair_friction.shape[0], pairid]                 <L 337>
        var_5 = &(var_pair_friction.shape);
        var_8 = wp::load(var_5);
        var_7 = wp::extract(var_8, var_6);
        var_9 = wp::mod(var_worldid, var_7);
        var_10 = wp::address(var_pair_friction, var_9, var_pairid);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // solref = pair_solref[worldid % pair_solref.shape[0], pairid]                       <L 338>
        var_13 = &(var_pair_solref.shape);
        var_16 = wp::load(var_13);
        var_15 = wp::extract(var_16, var_14);
        var_17 = wp::mod(var_worldid, var_15);
        var_18 = wp::address(var_pair_solref, var_17, var_pairid);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // solreffriction = pair_solreffriction[worldid % pair_solreffriction.shape[0], pairid]       <L 339>
        var_21 = &(var_pair_solreffriction.shape);
        var_24 = wp::load(var_21);
        var_23 = wp::extract(var_24, var_22);
        var_25 = wp::mod(var_worldid, var_23);
        var_26 = wp::address(var_pair_solreffriction, var_25, var_pairid);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // solimp = pair_solimp[worldid % pair_solimp.shape[0], pairid]                       <L 340>
        var_29 = &(var_pair_solimp.shape);
        var_32 = wp::load(var_29);
        var_31 = wp::extract(var_32, var_30);
        var_33 = wp::mod(var_worldid, var_31);
        var_34 = wp::address(var_pair_solimp, var_33, var_pairid);
        var_36 = wp::load(var_34);
        var_35 = wp::copy(var_36);
    }
    if (!var_1) {
        // g1 = geoms[0]                                                                      <L 342>
        var_38 = wp::extract(var_geoms, var_37);
        // g2 = geoms[1]                                                                      <L 343>
        var_40 = wp::extract(var_geoms, var_39);
        // solmix_id = worldid % geom_solmix.shape[0]                                         <L 344>
        var_41 = &(var_geom_solmix.shape);
        var_44 = wp::load(var_41);
        var_43 = wp::extract(var_44, var_42);
        var_45 = wp::mod(var_worldid, var_43);
        // friction_id = worldid % geom_friction.shape[0]                                     <L 345>
        var_46 = &(var_geom_friction.shape);
        var_49 = wp::load(var_46);
        var_48 = wp::extract(var_49, var_47);
        var_50 = wp::mod(var_worldid, var_48);
        // solref_id = worldid % geom_solref.shape[0]                                         <L 346>
        var_51 = &(var_geom_solref.shape);
        var_54 = wp::load(var_51);
        var_53 = wp::extract(var_54, var_52);
        var_55 = wp::mod(var_worldid, var_53);
        // solimp_id = worldid % geom_solimp.shape[0]                                         <L 347>
        var_56 = &(var_geom_solimp.shape);
        var_59 = wp::load(var_56);
        var_58 = wp::extract(var_59, var_57);
        var_60 = wp::mod(var_worldid, var_58);
        // solmix1 = geom_solmix[solmix_id, g1]                                               <L 349>
        var_61 = wp::address(var_geom_solmix, var_45, var_38);
        var_63 = wp::load(var_61);
        var_62 = wp::copy(var_63);
        // solmix2 = geom_solmix[solmix_id, g2]                                               <L 350>
        var_64 = wp::address(var_geom_solmix, var_45, var_40);
        var_66 = wp::load(var_64);
        var_65 = wp::copy(var_66);
        // condim1 = geom_condim[g1]                                                          <L 352>
        var_67 = wp::address(var_geom_condim, var_38);
        var_69 = wp::load(var_67);
        var_68 = wp::copy(var_69);
        // condim2 = geom_condim[g2]                                                          <L 353>
        var_70 = wp::address(var_geom_condim, var_40);
        var_72 = wp::load(var_70);
        var_71 = wp::copy(var_72);
        // p1 = geom_priority[g1]                                                             <L 356>
        var_73 = wp::address(var_geom_priority, var_38);
        var_75 = wp::load(var_73);
        var_74 = wp::copy(var_75);
        // p2 = geom_priority[g2]                                                             <L 357>
        var_76 = wp::address(var_geom_priority, var_40);
        var_78 = wp::load(var_76);
        var_77 = wp::copy(var_78);
        // if p1 > p2:                                                                        <L 359>
        var_79 = (var_74 > var_77);
        if (var_79) {
            // mix = 1.0                                                                      <L 360>
            // condim = condim1                                                               <L 361>
            var_81 = wp::copy(var_68);
            // max_geom_friction = geom_friction[friction_id, g1]                             <L 362>
            var_82 = wp::address(var_geom_friction, var_50, var_38);
            var_84 = wp::load(var_82);
            var_83 = wp::copy(var_84);
        }
        var_85 = wp::where(var_79, var_81, var_3);
        if (!var_79) {
            // elif p2 > p1:                                                                  <L 363>
            var_86 = (var_77 > var_74);
            if (var_86) {
                // mix = 0.0                                                                  <L 364>
                // condim = condim2                                                           <L 365>
                var_88 = wp::copy(var_71);
                // max_geom_friction = geom_friction[friction_id, g2]                         <L 366>
                var_89 = wp::address(var_geom_friction, var_50, var_40);
                var_91 = wp::load(var_89);
                var_90 = wp::copy(var_91);
            }
            var_92 = wp::where(var_86, var_88, var_85);
            var_93 = wp::where(var_86, var_87, var_80);
            var_94 = wp::where(var_86, var_90, var_83);
            if (!var_86) {
                // mix = safe_div(solmix1, solmix1 + solmix2)                                 <L 368>
                var_95 = wp::add(var_62, var_65);
                var_96 = safe_div_0(var_62, var_95);
                // mix = wp.where((solmix1 < MJ_MINVAL) and (solmix2 < MJ_MINVAL), 0.5, mix)       <L 369>
                var_99 = (var_62 < var_98);
                var_97 = var_99;
                if (var_97) {
                    var_100 = (var_65 < var_98);
                    var_97 = var_97 && var_100;
                }
                var_102 = wp::where(var_97, var_101, var_96);
                // mix = wp.where((solmix1 < MJ_MINVAL) and (solmix2 >= MJ_MINVAL), 0.0, mix)       <L 370>
                var_104 = (var_62 < var_98);
                var_103 = var_104;
                if (var_103) {
                    var_105 = (var_65 >= var_98);
                    var_103 = var_103 && var_105;
                }
                var_107 = wp::where(var_103, var_106, var_102);
                // mix = wp.where((solmix1 >= MJ_MINVAL) and (solmix2 < MJ_MINVAL), 1.0, mix)       <L 371>
                var_109 = (var_62 >= var_98);
                var_108 = var_109;
                if (var_108) {
                    var_110 = (var_65 < var_98);
                    var_108 = var_108 && var_110;
                }
                var_112 = wp::where(var_108, var_111, var_107);
                // condim = wp.max(condim1, condim2)                                          <L 372>
                var_113 = wp::max(var_68, var_71);
                // max_geom_friction = wp.max(geom_friction[friction_id, g1], geom_friction[friction_id, g2])       <L 373>
                var_114 = wp::address(var_geom_friction, var_50, var_38);
                var_115 = wp::address(var_geom_friction, var_50, var_40);
                var_117 = wp::load(var_114);
                var_118 = wp::load(var_115);
                var_116 = wp::max(var_117, var_118);
            }
            var_119 = wp::where(var_86, var_92, var_113);
            var_120 = wp::where(var_86, var_93, var_112);
            var_121 = wp::where(var_86, var_94, var_116);
        }
        var_122 = wp::where(var_79, var_85, var_119);
        var_123 = wp::where(var_79, var_80, var_120);
        var_124 = wp::where(var_79, var_83, var_121);
        // friction = vec5(                                                                   <L 375>
        // max_geom_friction[0],                                                              <L 376>
        var_126 = wp::extract(var_124, var_125);
        // max_geom_friction[0],                                                              <L 377>
        var_128 = wp::extract(var_124, var_127);
        // max_geom_friction[1],                                                              <L 378>
        var_130 = wp::extract(var_124, var_129);
        // max_geom_friction[2],                                                              <L 379>
        var_132 = wp::extract(var_124, var_131);
        // max_geom_friction[2],                                                              <L 380>
        var_134 = wp::extract(var_124, var_133);
        var_135 = wp::vec_t<5, wp::float32>({var_126, var_128, var_130, var_132, var_134});
        // if geom_solref[solref_id, g1][0] > 0.0 and geom_solref[solref_id, g2][0] > 0.0:       <L 383>
        var_137 = wp::address(var_geom_solref, var_55, var_38);
        var_140 = wp::load(var_137);
        var_139 = wp::extract(var_140, var_138);
        var_142 = (var_139 > var_141);
        var_136 = var_142;
        if (var_136) {
            var_143 = wp::address(var_geom_solref, var_55, var_40);
            var_146 = wp::load(var_143);
            var_145 = wp::extract(var_146, var_144);
            var_148 = (var_145 > var_147);
            var_136 = var_136 && var_148;
        }
        if (var_136) {
            // solref = mix * geom_solref[solref_id, g1] + (1.0 - mix) * geom_solref[solref_id, g2]       <L 384>
            var_149 = wp::address(var_geom_solref, var_55, var_38);
            var_151 = wp::load(var_149);
            var_150 = wp::mul(var_123, var_151);
            var_153 = wp::sub(var_152, var_123);
            var_154 = wp::address(var_geom_solref, var_55, var_40);
            var_156 = wp::load(var_154);
            var_155 = wp::mul(var_153, var_156);
            var_157 = wp::add(var_150, var_155);
        }
        var_158 = wp::where(var_136, var_157, var_19);
        if (!var_136) {
            // solref = wp.min(geom_solref[solref_id, g1], geom_solref[solref_id, g2])        <L 386>
            var_159 = wp::address(var_geom_solref, var_55, var_38);
            var_160 = wp::address(var_geom_solref, var_55, var_40);
            var_162 = wp::load(var_159);
            var_163 = wp::load(var_160);
            var_161 = wp::min(var_162, var_163);
        }
        var_164 = wp::where(var_136, var_158, var_161);
        // solreffriction = wp.vec2(0.0, 0.0)                                                 <L 388>
        var_167 = wp::vec_t<2, wp::float32>(var_165, var_166);
        // solimp = mix * geom_solimp[solimp_id, g1] + (1.0 - mix) * geom_solimp[solimp_id, g2]       <L 389>
        var_168 = wp::address(var_geom_solimp, var_60, var_38);
        var_170 = wp::load(var_168);
        var_169 = wp::mul(var_123, var_170);
        var_172 = wp::sub(var_171, var_123);
        var_173 = wp::address(var_geom_solimp, var_60, var_40);
        var_175 = wp::load(var_173);
        var_174 = wp::mul(var_172, var_175);
        var_176 = wp::add(var_169, var_174);
    }
    var_177 = wp::where(var_1, var_3, var_122);
    var_178 = wp::where(var_1, var_11, var_135);
    var_179 = wp::where(var_1, var_19, var_164);
    var_180 = wp::where(var_1, var_27, var_167);
    var_181 = wp::where(var_1, var_35, var_176);
    // friction = vec5(                                                                       <L 391>
    // wp.max(MJ_MINMU, friction[0]),                                                         <L 392>
    var_184 = wp::extract(var_178, var_183);
    var_185 = wp::max(var_182, var_184);
    // wp.max(MJ_MINMU, friction[1]),                                                         <L 393>
    var_187 = wp::extract(var_178, var_186);
    var_188 = wp::max(var_182, var_187);
    // wp.max(MJ_MINMU, friction[2]),                                                         <L 394>
    var_190 = wp::extract(var_178, var_189);
    var_191 = wp::max(var_182, var_190);
    // wp.max(MJ_MINMU, friction[3]),                                                         <L 395>
    var_193 = wp::extract(var_178, var_192);
    var_194 = wp::max(var_182, var_193);
    // wp.max(MJ_MINMU, friction[4]),                                                         <L 396>
    var_196 = wp::extract(var_178, var_195);
    var_197 = wp::max(var_182, var_196);
    var_198 = wp::vec_t<5, wp::float32>({var_185, var_188, var_191, var_194, var_197});
    // return condim, friction, solref, solreffriction, solimp                                <L 399>
    ret_0 = var_177;
    ret_1 = var_198;
    ret_2 = var_179;
    ret_3 = var_180;
    ret_4 = var_181;
    return;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_core.py:402
static CUDA_CALLABLE void contact_params_0(
    wp::array_t<wp::int32> var_geom_condim,
    wp::array_t<wp::int32> var_geom_priority,
    wp::array_t<wp::float32> var_geom_solmix,
    wp::array_t<wp::vec_t<2, wp::float32>> var_geom_solref,
    wp::array_t<wp::vec_t<5, wp::float32>> var_geom_solimp,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_friction,
    wp::array_t<wp::float32> var_geom_margin,
    wp::array_t<wp::float32> var_geom_gap,
    wp::array_t<wp::int32> var_pair_dim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_pair_solref,
    wp::array_t<wp::vec_t<2, wp::float32>> var_pair_solreffriction,
    wp::array_t<wp::vec_t<5, wp::float32>> var_pair_solimp,
    wp::array_t<wp::float32> var_pair_margin,
    wp::array_t<wp::float32> var_pair_gap,
    wp::array_t<wp::vec_t<5, wp::float32>> var_pair_friction,
    wp::array_t<wp::vec_t<2, wp::int32>> var_collision_pair_in,
    wp::array_t<wp::vec_t<2, wp::int32>> var_collision_pairid_in,
    wp::int32 var_cid,
    wp::int32 var_worldid,
    wp::vec_t<2, wp::int32> & ret_0,
    wp::float32 & ret_1,
    wp::float32 & ret_2,
    wp::int32 & ret_3,
    wp::vec_t<5, wp::float32> & ret_4,
    wp::vec_t<2, wp::float32> & ret_5,
    wp::vec_t<2, wp::float32> & ret_6,
    wp::vec_t<5, wp::float32> & ret_7)
{
    //---------
    // primal vars
    wp::vec_t<2, wp::int32>* var_0;
    wp::vec_t<2, wp::int32> var_1;
    wp::vec_t<2, wp::int32> var_2;
    wp::vec_t<2, wp::int32>* var_3;
    const wp::int32 var_4 = 0;
    wp::int32 var_5;
    wp::vec_t<2, wp::int32> var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    wp::int32 var_9;
    wp::vec_t<5, wp::float32> var_10;
    wp::vec_t<2, wp::float32> var_11;
    wp::vec_t<2, wp::float32> var_12;
    wp::vec_t<5, wp::float32> var_13;
    //---------
    // forward
    // def contact_params(                                                                    <L 403>
    // geoms = collision_pair_in[cid]                                                         <L 431>
    var_0 = wp::address(var_collision_pair_in, var_cid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    // pairid = collision_pairid_in[cid][0]                                                   <L 432>
    var_3 = wp::address(var_collision_pairid_in, var_cid);
    var_6 = wp::load(var_3);
    var_5 = wp::extract(var_6, var_4);
    // margin, gap = contact_margin_gap(geom_margin, geom_gap, pair_margin, pair_gap, geoms, pairid, worldid)       <L 437>
    contact_margin_gap_0(var_geom_margin, var_geom_gap, var_pair_margin, var_pair_gap, var_1, var_5, var_worldid, var_7, var_8);
    // condim, friction, solref, solreffriction, solimp = contact_material_params(            <L 438>
    // geom_condim,                                                                           <L 439>
    // geom_priority,                                                                         <L 440>
    // geom_solmix,                                                                           <L 441>
    // geom_solref,                                                                           <L 442>
    // geom_solimp,                                                                           <L 443>
    // geom_friction,                                                                         <L 444>
    // pair_dim,                                                                              <L 445>
    // pair_solref,                                                                           <L 446>
    // pair_solreffriction,                                                                   <L 447>
    // pair_solimp,                                                                           <L 448>
    // pair_friction,                                                                         <L 449>
    // geoms,                                                                                 <L 450>
    // pairid,                                                                                <L 451>
    // worldid,                                                                               <L 452>
    contact_material_params_0(var_geom_condim, var_geom_priority, var_geom_solmix, var_geom_solref, var_geom_solimp, var_geom_friction, var_pair_dim, var_pair_solref, var_pair_solreffriction, var_pair_solimp, var_pair_friction, var_1, var_5, var_worldid, var_9, var_10, var_11, var_12, var_13);
    // return geoms, margin, gap, condim, friction, solref, solreffriction, solimp            <L 455>
    ret_0 = var_1;
    ret_1 = var_7;
    ret_2 = var_8;
    ret_3 = var_9;
    ret_4 = var_10;
    ret_5 = var_11;
    ret_6 = var_12;
    ret_7 = var_13;
    return;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_core.py:65
static CUDA_CALLABLE void geom_collision_pair_from_types_0(
    wp::array_t<wp::int32> var_geom_dataid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_size,
    wp::array_t<wp::int32> var_mesh_vertadr,
    wp::array_t<wp::int32> var_mesh_vertnum,
    wp::array_t<wp::int32> var_mesh_graphadr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_mesh_vert,
    wp::array_t<wp::int32> var_mesh_graph,
    wp::array_t<wp::int32> var_mesh_polynum,
    wp::array_t<wp::int32> var_mesh_polyadr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_mesh_polynormal,
    wp::array_t<wp::int32> var_mesh_polyvertadr,
    wp::array_t<wp::int32> var_mesh_polyvertnum,
    wp::array_t<wp::int32> var_mesh_polyvert,
    wp::array_t<wp::int32> var_mesh_polymapadr,
    wp::array_t<wp::int32> var_mesh_polymapnum,
    wp::array_t<wp::int32> var_mesh_polymap,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::int32 var_geom_type1,
    wp::int32 var_geom_type2,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::int32 var_worldid,
    Geom_3242f8a8 & ret_0,
    Geom_3242f8a8 & ret_1)
{
    //---------
    // primal vars
    Geom_3242f8a8 var_0;
    Geom_3242f8a8 var_1;
    const wp::int32 var_2 = 0;
    wp::int32 var_3;
    const wp::int32 var_4 = 1;
    wp::int32 var_5;
    wp::vec_t<3, wp::float32>* var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::mat_t<3, 3, wp::float32>* var_9;
    wp::mat_t<3, 3, wp::float32> var_10;
    wp::mat_t<3, 3, wp::float32> var_11;
    wp::shape_t* var_12;
    const wp::int32 var_13 = 0;
    wp::int32 var_14;
    wp::shape_t var_15;
    wp::int32 var_16;
    wp::vec_t<3, wp::float32>* var_17;
    wp::vec_t<3, wp::float32> var_18;
    wp::vec_t<3, wp::float32> var_19;
    wp::mat_t<3, 3, wp::float32>* var_20;
    const wp::int32 var_21 = 0;
    const wp::int32 var_22 = 2;
    wp::float32 var_23;
    wp::mat_t<3, 3, wp::float32> var_24;
    wp::mat_t<3, 3, wp::float32>* var_25;
    const wp::int32 var_26 = 1;
    const wp::int32 var_27 = 2;
    wp::float32 var_28;
    wp::mat_t<3, 3, wp::float32> var_29;
    wp::mat_t<3, 3, wp::float32>* var_30;
    const wp::int32 var_31 = 2;
    const wp::int32 var_32 = 2;
    wp::float32 var_33;
    wp::mat_t<3, 3, wp::float32> var_34;
    wp::vec_t<3, wp::float32> var_35;
    wp::vec_t<3, wp::float32>* var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::mat_t<3, 3, wp::float32>* var_39;
    wp::mat_t<3, 3, wp::float32> var_40;
    wp::mat_t<3, 3, wp::float32> var_41;
    wp::shape_t* var_42;
    const wp::int32 var_43 = 0;
    wp::int32 var_44;
    wp::shape_t var_45;
    wp::int32 var_46;
    wp::vec_t<3, wp::float32>* var_47;
    wp::vec_t<3, wp::float32> var_48;
    wp::vec_t<3, wp::float32> var_49;
    wp::mat_t<3, 3, wp::float32>* var_50;
    const wp::int32 var_51 = 0;
    const wp::int32 var_52 = 2;
    wp::float32 var_53;
    wp::mat_t<3, 3, wp::float32> var_54;
    wp::mat_t<3, 3, wp::float32>* var_55;
    const wp::int32 var_56 = 1;
    const wp::int32 var_57 = 2;
    wp::float32 var_58;
    wp::mat_t<3, 3, wp::float32> var_59;
    wp::mat_t<3, 3, wp::float32>* var_60;
    const wp::int32 var_61 = 2;
    const wp::int32 var_62 = 2;
    wp::float32 var_63;
    wp::mat_t<3, 3, wp::float32> var_64;
    wp::vec_t<3, wp::float32> var_65;
    wp::shape_t* var_66;
    const wp::int32 var_67 = 0;
    wp::int32 var_68;
    wp::shape_t var_69;
    wp::int32 var_70;
    const wp::int32 var_71 = 7;
    bool var_72;
    wp::int32* var_73;
    wp::int32 var_74;
    wp::int32 var_75;
    const wp::int32 var_76 = 0;
    bool var_77;
    wp::int32* var_78;
    const wp::int32 var_79 = -1;
    wp::int32 var_80;
    wp::int32 var_81;
    const wp::int32 var_82 = 0;
    bool var_83;
    wp::int32* var_84;
    const wp::int32 var_85 = -1;
    wp::int32 var_86;
    wp::int32 var_87;
    const wp::int32 var_88 = 0;
    bool var_89;
    wp::int32* var_90;
    const wp::int32 var_91 = -1;
    wp::int32 var_92;
    wp::int32 var_93;
    const wp::int32 var_94 = 0;
    bool var_95;
    wp::int32* var_96;
    const wp::int32 var_97 = -1;
    wp::int32 var_98;
    wp::int32 var_99;
    const wp::int32 var_100 = 0;
    bool var_101;
    wp::int32* var_102;
    const wp::int32 var_103 = -1;
    wp::int32 var_104;
    wp::int32 var_105;
    const wp::int32 var_106 = 7;
    bool var_107;
    wp::int32* var_108;
    wp::int32 var_109;
    wp::int32 var_110;
    const wp::int32 var_111 = 0;
    bool var_112;
    wp::int32* var_113;
    const wp::int32 var_114 = -1;
    wp::int32 var_115;
    wp::int32 var_116;
    const wp::int32 var_117 = 0;
    bool var_118;
    wp::int32* var_119;
    const wp::int32 var_120 = -1;
    wp::int32 var_121;
    wp::int32 var_122;
    const wp::int32 var_123 = 0;
    bool var_124;
    wp::int32* var_125;
    const wp::int32 var_126 = -1;
    wp::int32 var_127;
    wp::int32 var_128;
    const wp::int32 var_129 = 0;
    bool var_130;
    wp::int32* var_131;
    const wp::int32 var_132 = -1;
    wp::int32 var_133;
    wp::int32 var_134;
    const wp::int32 var_135 = 0;
    bool var_136;
    wp::int32* var_137;
    const wp::int32 var_138 = -1;
    wp::int32 var_139;
    wp::int32 var_140;
    wp::int32 var_141;
    const wp::int32 var_142 = -1;
    const wp::float32 var_143 = 0.0;
    const wp::int32 var_144 = -1;
    const wp::float32 var_145 = 0.0;
    //---------
    // forward
    // def geom_collision_pair_from_types(                                                    <L 66>
    // geom1 = Geom()                                                                         <L 93>
    var_0 = Geom_3242f8a8();
    // geom2 = Geom()                                                                         <L 94>
    var_1 = Geom_3242f8a8();
    // g1 = geoms[0]                                                                          <L 96>
    var_3 = wp::extract(var_geoms, var_2);
    // g2 = geoms[1]                                                                          <L 97>
    var_5 = wp::extract(var_geoms, var_4);
    // geom1.pos = geom_xpos_in[worldid, g1]                                                  <L 99>
    var_6 = wp::address(var_geom_xpos_in, var_worldid, var_3);
    var_8 = wp::load(var_6);
    var_7 = wp::copy(var_8);
    var_0.pos = var_7;
    // geom1.rot = geom_xmat_in[worldid, g1]                                                  <L 100>
    var_9 = wp::address(var_geom_xmat_in, var_worldid, var_3);
    var_11 = wp::load(var_9);
    var_10 = wp::copy(var_11);
    var_0.rot = var_10;
    // geom1.size = geom_size[worldid % geom_size.shape[0], g1]                               <L 101>
    var_12 = &(var_geom_size.shape);
    var_15 = wp::load(var_12);
    var_14 = wp::extract(var_15, var_13);
    var_16 = wp::mod(var_worldid, var_14);
    var_17 = wp::address(var_geom_size, var_16, var_3);
    var_19 = wp::load(var_17);
    var_18 = wp::copy(var_19);
    var_0.size = var_18;
    // geom1.normal = wp.vec3(geom1.rot[0, 2], geom1.rot[1, 2], geom1.rot[2, 2])              <L 103>
    var_20 = &((var_0).rot);
    var_24 = wp::load(var_20);
    var_23 = wp::extract(var_24, var_21, var_22);
    var_25 = &((var_0).rot);
    var_29 = wp::load(var_25);
    var_28 = wp::extract(var_29, var_26, var_27);
    var_30 = &((var_0).rot);
    var_34 = wp::load(var_30);
    var_33 = wp::extract(var_34, var_31, var_32);
    var_35 = wp::vec_t<3, wp::float32>(var_23, var_28, var_33);
    var_0.normal = var_35;
    // geom2.pos = geom_xpos_in[worldid, g2]                                                  <L 105>
    var_36 = wp::address(var_geom_xpos_in, var_worldid, var_5);
    var_38 = wp::load(var_36);
    var_37 = wp::copy(var_38);
    var_1.pos = var_37;
    // geom2.rot = geom_xmat_in[worldid, g2]                                                  <L 106>
    var_39 = wp::address(var_geom_xmat_in, var_worldid, var_5);
    var_41 = wp::load(var_39);
    var_40 = wp::copy(var_41);
    var_1.rot = var_40;
    // geom2.size = geom_size[worldid % geom_size.shape[0], g2]                               <L 107>
    var_42 = &(var_geom_size.shape);
    var_45 = wp::load(var_42);
    var_44 = wp::extract(var_45, var_43);
    var_46 = wp::mod(var_worldid, var_44);
    var_47 = wp::address(var_geom_size, var_46, var_5);
    var_49 = wp::load(var_47);
    var_48 = wp::copy(var_49);
    var_1.size = var_48;
    // geom2.normal = wp.vec3(geom2.rot[0, 2], geom2.rot[1, 2], geom2.rot[2, 2])              <L 109>
    var_50 = &((var_1).rot);
    var_54 = wp::load(var_50);
    var_53 = wp::extract(var_54, var_51, var_52);
    var_55 = &((var_1).rot);
    var_59 = wp::load(var_55);
    var_58 = wp::extract(var_59, var_56, var_57);
    var_60 = &((var_1).rot);
    var_64 = wp::load(var_60);
    var_63 = wp::extract(var_64, var_61, var_62);
    var_65 = wp::vec_t<3, wp::float32>(var_53, var_58, var_63);
    var_1.normal = var_65;
    // dataid_setid = worldid % geom_dataid.shape[0]                                          <L 111>
    var_66 = &(var_geom_dataid.shape);
    var_69 = wp::load(var_66);
    var_68 = wp::extract(var_69, var_67);
    var_70 = wp::mod(var_worldid, var_68);
    // if geom_type1 == GeomType.MESH:                                                        <L 113>
    var_72 = (var_geom_type1 == var_71);
    if (var_72) {
        // dataid = geom_dataid[dataid_setid, g1]                                             <L 114>
        var_73 = wp::address(var_geom_dataid, var_70, var_3);
        var_75 = wp::load(var_73);
        var_74 = wp::copy(var_75);
        // geom1.vertadr = wp.where(dataid >= 0, mesh_vertadr[dataid], -1)                    <L 115>
        var_77 = (var_74 >= var_76);
        var_78 = wp::address(var_mesh_vertadr, var_74);
        var_81 = wp::load(var_78);
        var_80 = wp::where(var_77, var_81, var_79);
        var_0.vertadr = var_80;
        // geom1.vertnum = wp.where(dataid >= 0, mesh_vertnum[dataid], -1)                    <L 116>
        var_83 = (var_74 >= var_82);
        var_84 = wp::address(var_mesh_vertnum, var_74);
        var_87 = wp::load(var_84);
        var_86 = wp::where(var_83, var_87, var_85);
        var_0.vertnum = var_86;
        // geom1.graphadr = wp.where(dataid >= 0, mesh_graphadr[dataid], -1)                  <L 117>
        var_89 = (var_74 >= var_88);
        var_90 = wp::address(var_mesh_graphadr, var_74);
        var_93 = wp::load(var_90);
        var_92 = wp::where(var_89, var_93, var_91);
        var_0.graphadr = var_92;
        // geom1.mesh_polynum = wp.where(dataid >= 0, mesh_polynum[dataid], -1)               <L 118>
        var_95 = (var_74 >= var_94);
        var_96 = wp::address(var_mesh_polynum, var_74);
        var_99 = wp::load(var_96);
        var_98 = wp::where(var_95, var_99, var_97);
        var_0.mesh_polynum = var_98;
        // geom1.mesh_polyadr = wp.where(dataid >= 0, mesh_polyadr[dataid], -1)               <L 119>
        var_101 = (var_74 >= var_100);
        var_102 = wp::address(var_mesh_polyadr, var_74);
        var_105 = wp::load(var_102);
        var_104 = wp::where(var_101, var_105, var_103);
        var_0.mesh_polyadr = var_104;
        // geom1.vert = mesh_vert                                                             <L 121>
        var_0.vert = var_mesh_vert;
        // geom1.graph = mesh_graph                                                           <L 122>
        var_0.graph = var_mesh_graph;
        // geom1.mesh_polynormal = mesh_polynormal                                            <L 123>
        var_0.mesh_polynormal = var_mesh_polynormal;
        // geom1.mesh_polyvertadr = mesh_polyvertadr                                          <L 124>
        var_0.mesh_polyvertadr = var_mesh_polyvertadr;
        // geom1.mesh_polyvertnum = mesh_polyvertnum                                          <L 125>
        var_0.mesh_polyvertnum = var_mesh_polyvertnum;
        // geom1.mesh_polyvert = mesh_polyvert                                                <L 126>
        var_0.mesh_polyvert = var_mesh_polyvert;
        // geom1.mesh_polymapadr = mesh_polymapadr                                            <L 127>
        var_0.mesh_polymapadr = var_mesh_polymapadr;
        // geom1.mesh_polymapnum = mesh_polymapnum                                            <L 128>
        var_0.mesh_polymapnum = var_mesh_polymapnum;
        // geom1.mesh_polymap = mesh_polymap                                                  <L 129>
        var_0.mesh_polymap = var_mesh_polymap;
    }
    // if geom_type2 == GeomType.MESH:                                                        <L 131>
    var_107 = (var_geom_type2 == var_106);
    if (var_107) {
        // dataid = geom_dataid[dataid_setid, g2]                                             <L 132>
        var_108 = wp::address(var_geom_dataid, var_70, var_5);
        var_110 = wp::load(var_108);
        var_109 = wp::copy(var_110);
        // geom2.vertadr = wp.where(dataid >= 0, mesh_vertadr[dataid], -1)                    <L 133>
        var_112 = (var_109 >= var_111);
        var_113 = wp::address(var_mesh_vertadr, var_109);
        var_116 = wp::load(var_113);
        var_115 = wp::where(var_112, var_116, var_114);
        var_1.vertadr = var_115;
        // geom2.vertnum = wp.where(dataid >= 0, mesh_vertnum[dataid], -1)                    <L 134>
        var_118 = (var_109 >= var_117);
        var_119 = wp::address(var_mesh_vertnum, var_109);
        var_122 = wp::load(var_119);
        var_121 = wp::where(var_118, var_122, var_120);
        var_1.vertnum = var_121;
        // geom2.graphadr = wp.where(dataid >= 0, mesh_graphadr[dataid], -1)                  <L 135>
        var_124 = (var_109 >= var_123);
        var_125 = wp::address(var_mesh_graphadr, var_109);
        var_128 = wp::load(var_125);
        var_127 = wp::where(var_124, var_128, var_126);
        var_1.graphadr = var_127;
        // geom2.mesh_polynum = wp.where(dataid >= 0, mesh_polynum[dataid], -1)               <L 136>
        var_130 = (var_109 >= var_129);
        var_131 = wp::address(var_mesh_polynum, var_109);
        var_134 = wp::load(var_131);
        var_133 = wp::where(var_130, var_134, var_132);
        var_1.mesh_polynum = var_133;
        // geom2.mesh_polyadr = wp.where(dataid >= 0, mesh_polyadr[dataid], -1)               <L 137>
        var_136 = (var_109 >= var_135);
        var_137 = wp::address(var_mesh_polyadr, var_109);
        var_140 = wp::load(var_137);
        var_139 = wp::where(var_136, var_140, var_138);
        var_1.mesh_polyadr = var_139;
        // geom2.vert = mesh_vert                                                             <L 139>
        var_1.vert = var_mesh_vert;
        // geom2.graph = mesh_graph                                                           <L 140>
        var_1.graph = var_mesh_graph;
        // geom2.mesh_polynormal = mesh_polynormal                                            <L 141>
        var_1.mesh_polynormal = var_mesh_polynormal;
        // geom2.mesh_polyvertadr = mesh_polyvertadr                                          <L 142>
        var_1.mesh_polyvertadr = var_mesh_polyvertadr;
        // geom2.mesh_polyvertnum = mesh_polyvertnum                                          <L 143>
        var_1.mesh_polyvertnum = var_mesh_polyvertnum;
        // geom2.mesh_polyvert = mesh_polyvert                                                <L 144>
        var_1.mesh_polyvert = var_mesh_polyvert;
        // geom2.mesh_polymapadr = mesh_polymapadr                                            <L 145>
        var_1.mesh_polymapadr = var_mesh_polymapadr;
        // geom2.mesh_polymapnum = mesh_polymapnum                                            <L 146>
        var_1.mesh_polymapnum = var_mesh_polymapnum;
        // geom2.mesh_polymap = mesh_polymap                                                  <L 147>
        var_1.mesh_polymap = var_mesh_polymap;
    }
    var_141 = wp::where(var_107, var_109, var_74);
    // geom1.index = -1                                                                       <L 149>
    var_0.index = var_142;
    // geom1.margin = 0.0                                                                     <L 150>
    var_0.margin = var_143;
    // geom2.index = -1                                                                       <L 152>
    var_1.index = var_144;
    // geom2.margin = 0.0                                                                     <L 153>
    var_1.margin = var_145;
    // return geom1, geom2                                                                    <L 155>
    ret_0 = var_0;
    ret_1 = var_1;
    return;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_core.py:158
static CUDA_CALLABLE void geom_collision_pair_0(
    wp::array_t<wp::int32> var_geom_type,
    wp::array_t<wp::int32> var_geom_dataid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_size,
    wp::array_t<wp::int32> var_mesh_vertadr,
    wp::array_t<wp::int32> var_mesh_vertnum,
    wp::array_t<wp::int32> var_mesh_graphadr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_mesh_vert,
    wp::array_t<wp::int32> var_mesh_graph,
    wp::array_t<wp::int32> var_mesh_polynum,
    wp::array_t<wp::int32> var_mesh_polyadr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_mesh_polynormal,
    wp::array_t<wp::int32> var_mesh_polyvertadr,
    wp::array_t<wp::int32> var_mesh_polyvertnum,
    wp::array_t<wp::int32> var_mesh_polyvert,
    wp::array_t<wp::int32> var_mesh_polymapadr,
    wp::array_t<wp::int32> var_mesh_polymapnum,
    wp::array_t<wp::int32> var_mesh_polymap,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::int32 var_worldid,
    Geom_3242f8a8 & ret_0,
    Geom_3242f8a8 & ret_1)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::int32 var_1;
    wp::int32* var_2;
    wp::int32 var_3;
    wp::int32 var_4;
    const wp::int32 var_5 = 1;
    wp::int32 var_6;
    wp::int32* var_7;
    wp::int32 var_8;
    wp::int32 var_9;
    Geom_3242f8a8 var_10;
    Geom_3242f8a8 var_11;
    //---------
    // forward
    // def geom_collision_pair(                                                               <L 159>
    // geom_type1 = geom_type[geoms[0]]                                                       <L 185>
    var_1 = wp::extract(var_geoms, var_0);
    var_2 = wp::address(var_geom_type, var_1);
    var_4 = wp::load(var_2);
    var_3 = wp::copy(var_4);
    // geom_type2 = geom_type[geoms[1]]                                                       <L 186>
    var_6 = wp::extract(var_geoms, var_5);
    var_7 = wp::address(var_geom_type, var_6);
    var_9 = wp::load(var_7);
    var_8 = wp::copy(var_9);
    // return geom_collision_pair_from_types(                                                 <L 187>
    // geom_dataid,                                                                           <L 188>
    // geom_size,                                                                             <L 189>
    // mesh_vertadr,                                                                          <L 190>
    // mesh_vertnum,                                                                          <L 191>
    // mesh_graphadr,                                                                         <L 192>
    // mesh_vert,                                                                             <L 193>
    // mesh_graph,                                                                            <L 194>
    // mesh_polynum,                                                                          <L 195>
    // mesh_polyadr,                                                                          <L 196>
    // mesh_polynormal,                                                                       <L 197>
    // mesh_polyvertadr,                                                                      <L 198>
    // mesh_polyvertnum,                                                                      <L 199>
    // mesh_polyvert,                                                                         <L 200>
    // mesh_polymapadr,                                                                       <L 201>
    // mesh_polymapnum,                                                                       <L 202>
    // mesh_polymap,                                                                          <L 203>
    // geom_xpos_in,                                                                          <L 204>
    // geom_xmat_in,                                                                          <L 205>
    // geom_type1,                                                                            <L 206>
    // geom_type2,                                                                            <L 207>
    // geoms,                                                                                 <L 208>
    // worldid,                                                                               <L 209>
    geom_collision_pair_from_types_0(var_geom_dataid, var_geom_size, var_mesh_vertadr, var_mesh_vertnum, var_mesh_graphadr, var_mesh_vert, var_mesh_graph, var_mesh_polynum, var_mesh_polyadr, var_mesh_polynormal, var_mesh_polyvertadr, var_mesh_polyvertnum, var_mesh_polyvert, var_mesh_polymapadr, var_mesh_polymapnum, var_mesh_polymap, var_geom_xpos_in, var_geom_xmat_in, var_3, var_8, var_geoms, var_worldid, var_10, var_11);
    ret_0 = var_10;
    ret_1 = var_11;
    return;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_primitive_core.py:47
static CUDA_CALLABLE void plane_sphere_0(
    wp::vec_t<3, wp::float32> var_plane_normal,
    wp::vec_t<3, wp::float32> var_plane_pos,
    wp::vec_t<3, wp::float32> var_sphere_pos,
    wp::float32 var_sphere_radius,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    const wp::float32 var_3 = 0.5;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    //---------
    // forward
    // def plane_sphere(plane_normal: wp.vec3, plane_pos: wp.vec3, sphere_pos: wp.vec3, sphere_radius: float) -> Tuple[float, wp.vec3]:       <L 48>
    // dist = wp.dot(sphere_pos - plane_pos, plane_normal) - sphere_radius                    <L 50>
    var_0 = wp::sub(var_sphere_pos, var_plane_pos);
    var_1 = wp::dot(var_0, var_plane_normal);
    var_2 = wp::sub(var_1, var_sphere_radius);
    // pos = sphere_pos - plane_normal * (sphere_radius + 0.5 * dist)                         <L 51>
    var_4 = wp::mul(var_3, var_2);
    var_5 = wp::add(var_sphere_radius, var_4);
    var_6 = wp::mul(var_plane_normal, var_5);
    var_7 = wp::sub(var_sphere_pos, var_6);
    // return dist, pos                                                                       <L 52>
    ret_0 = var_2;
    ret_1 = var_7;
    return;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/math.py:202
static CUDA_CALLABLE void orthogonals_0(
    wp::vec_t<3, wp::float32> var_a,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    const wp::float32 var_1 = 1.0;
    const wp::float32 var_2 = 0.0;
    wp::vec_t<3, wp::float32> var_3;
    const wp::float32 var_4 = 0.0;
    const wp::float32 var_5 = 0.0;
    const wp::float32 var_6 = 1.0;
    wp::vec_t<3, wp::float32> var_7;
    bool var_8;
    const wp::float32 var_9 = -0.5;
    const wp::int32 var_10 = 1;
    wp::float32 var_11;
    bool var_12;
    const wp::int32 var_13 = 1;
    wp::float32 var_14;
    const wp::float32 var_15 = 0.5;
    bool var_16;
    wp::vec_t<3, wp::float32> var_17;
    wp::float32 var_18;
    wp::vec_t<3, wp::float32> var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32> var_21;
    wp::float32 var_22;
    const wp::float32 var_23 = 0.0;
    bool var_24;
    const wp::float32 var_25 = 0.0;
    const wp::float32 var_26 = 0.0;
    const wp::float32 var_27 = 0.0;
    wp::vec_t<3, wp::float32> var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::vec_t<3, wp::float32> var_30;
    //---------
    // forward
    // def orthogonals(a: wp.vec3):                                                           <L 203>
    // y = wp.vec3(0.0, 1.0, 0.0)                                                             <L 204>
    var_3 = wp::vec_t<3, wp::float32>(var_0, var_1, var_2);
    // z = wp.vec3(0.0, 0.0, 1.0)                                                             <L 205>
    var_7 = wp::vec_t<3, wp::float32>(var_4, var_5, var_6);
    // b = wp.where((-0.5 < a[1]) and (a[1] < 0.5), y, z)                                     <L 206>
    var_11 = wp::extract(var_a, var_10);
    var_12 = (var_9 < var_11);
    var_8 = var_12;
    if (var_8) {
        var_14 = wp::extract(var_a, var_13);
        var_16 = (var_14 < var_15);
        var_8 = var_8 && var_16;
    }
    var_17 = wp::where(var_8, var_3, var_7);
    // b = b - a * wp.dot(a, b)                                                               <L 207>
    var_18 = wp::dot(var_a, var_17);
    var_19 = wp::mul(var_a, var_18);
    var_20 = wp::sub(var_17, var_19);
    // b = wp.normalize(b)                                                                    <L 208>
    var_21 = wp::normalize(var_20);
    // if wp.length(a) == 0.0:                                                                <L 209>
    var_22 = wp::length(var_a);
    var_24 = (var_22 == var_23);
    if (var_24) {
        // b = wp.vec3(0.0, 0.0, 0.0)                                                         <L 210>
        var_28 = wp::vec_t<3, wp::float32>(var_25, var_26, var_27);
    }
    var_29 = wp::where(var_24, var_28, var_21);
    // c = wp.cross(a, b)                                                                     <L 211>
    var_30 = wp::cross(var_a, var_29);
    // return b, c                                                                            <L 213>
    ret_0 = var_29;
    ret_1 = var_30;
    return;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/math.py:246
static CUDA_CALLABLE wp::mat_t<3, 3, wp::float32> make_frame_0(
    wp::vec_t<3, wp::float32> var_a)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    const wp::int32 var_3 = 0;
    wp::float32 var_4;
    const wp::int32 var_5 = 1;
    wp::float32 var_6;
    const wp::int32 var_7 = 2;
    wp::float32 var_8;
    const wp::int32 var_9 = 0;
    wp::float32 var_10;
    const wp::int32 var_11 = 1;
    wp::float32 var_12;
    const wp::int32 var_13 = 2;
    wp::float32 var_14;
    const wp::int32 var_15 = 0;
    wp::float32 var_16;
    const wp::int32 var_17 = 1;
    wp::float32 var_18;
    const wp::int32 var_19 = 2;
    wp::float32 var_20;
    wp::mat_t<3, 3, wp::float32> var_21;
    //---------
    // forward
    // def make_frame(a: wp.vec3):                                                            <L 247>
    // a = wp.normalize(a)                                                                    <L 248>
    var_0 = wp::normalize(var_a);
    // b, c = orthogonals(a)                                                                  <L 249>
    orthogonals_0(var_0, var_1, var_2);
    // return wp.mat33(                                                                       <L 252>
    // a.x, a.y, a.z,                                                                         <L 253>
    var_4 = wp::extract(var_0, var_3);
    var_6 = wp::extract(var_0, var_5);
    var_8 = wp::extract(var_0, var_7);
    // b.x, b.y, b.z,                                                                         <L 254>
    var_10 = wp::extract(var_1, var_9);
    var_12 = wp::extract(var_1, var_11);
    var_14 = wp::extract(var_1, var_13);
    // c.x, c.y, c.z                                                                          <L 255>
    var_16 = wp::extract(var_2, var_15);
    var_18 = wp::extract(var_2, var_17);
    var_20 = wp::extract(var_2, var_19);
    var_21 = wp::mat_t<3, 3, wp::float32>(var_4, var_6, var_8, var_10, var_12, var_14, var_16, var_18, var_20);
    return var_21;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_core.py:213
static CUDA_CALLABLE wp::int32 write_contact_0(
    wp::int32 var_naconmax_in,
    wp::int32 var_id_,
    wp::float32 var_dist_in,
    wp::vec_t<3, wp::float32> var_pos_in,
    wp::mat_t<3, 3, wp::float32> var_frame_in,
    wp::float32 var_margin_in,
    wp::float32 var_gap_in,
    wp::int32 var_condim_in,
    wp::vec_t<5, wp::float32> var_friction_in,
    wp::vec_t<2, wp::float32> var_solref_in,
    wp::vec_t<2, wp::float32> var_solreffriction_in,
    wp::vec_t<5, wp::float32> var_solimp_in,
    wp::vec_t<2, wp::int32> var_geoms_in,
    wp::vec_t<2, wp::int32> var_pairid_in,
    wp::int32 var_worldid_in,
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
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::int32> var_nacon_out)
{
    //---------
    // primal vars
    bool var_0;
    wp::float32 var_1;
    bool var_2;
    bool var_3;
    bool var_4;
    const wp::int32 var_5 = 0;
    wp::int32 var_6;
    const wp::int32 var_7 = -2;
    bool var_8;
    bool var_9;
    const wp::int32 var_10 = 1;
    wp::int32 var_11;
    const wp::int32 var_12 = -1;
    bool var_13;
    const wp::int32 var_14 = 0;
    const wp::int32 var_15 = 0;
    bool var_16;
    const wp::int32 var_17 = 0;
    wp::int32 var_18;
    const wp::int32 var_19 = -1;
    bool var_20;
    const wp::int32 var_21 = 1;
    wp::int32 var_22;
    wp::int32 var_23;
    const wp::int32 var_24 = 1;
    wp::int32 var_25;
    const wp::int32 var_26 = 0;
    bool var_27;
    const wp::int32 var_28 = 2;
    wp::int32 var_29;
    wp::int32 var_30;
    const wp::int32 var_31 = 0;
    const wp::int32 var_32 = 1;
    wp::int32 var_33;
    bool var_34;
    wp::float32 var_35;
    wp::shape_t* var_36;
    const wp::int32 var_37 = 1;
    wp::int32 var_38;
    wp::shape_t var_39;
    wp::range_t var_40;
    wp::int32 var_41;
    const wp::int32 var_42 = -1;
    wp::int32 var_43;
    const wp::int32 var_44 = 0;
    //---------
    // forward
    // def write_contact(                                                                     <L 214>
    // active = dist_in < margin_in                                                           <L 253>
    var_0 = (var_dist_in < var_margin_in);
    // detected = dist_in < margin_in + gap_in                                                <L 254>
    var_1 = wp::add(var_margin_in, var_gap_in);
    var_2 = (var_dist_in < var_1);
    // if (pairid_in[0] == -2 or not detected) and pairid_in[1] == -1:                        <L 257>
    var_6 = wp::extract(var_pairid_in, var_5);
    var_8 = (var_6 == var_7);
    var_4 = var_8;
    if (!var_4) {
        var_9 = wp::unot(var_2);
        var_4 = var_4 || var_9;
    }
    var_3 = var_4;
    if (var_3) {
        var_11 = wp::extract(var_pairid_in, var_10);
        var_13 = (var_11 == var_12);
        var_3 = var_3 && var_13;
    }
    if (var_3) {
        // return 0                                                                           <L 258>
        return var_14;
    }
    // contact_type = 0                                                                       <L 260>
    // if pairid_in[0] >= -1 and detected:                                                    <L 262>
    var_18 = wp::extract(var_pairid_in, var_17);
    var_20 = (var_18 >= var_19);
    var_16 = var_20;
    if (var_16) {
        var_16 = var_16 && var_2;
    }
    if (var_16) {
        // contact_type |= ContactType.CONSTRAINT                                             <L 263>
        var_22 = wp::bit_or(var_15, var_21);
    }
    var_23 = wp::where(var_16, var_22, var_15);
    // if pairid_in[1] >= 0:                                                                  <L 265>
    var_25 = wp::extract(var_pairid_in, var_24);
    var_27 = (var_25 >= var_26);
    if (var_27) {
        // contact_type |= ContactType.SENSOR                                                 <L 266>
        var_29 = wp::bit_or(var_23, var_28);
    }
    var_30 = wp::where(var_27, var_29, var_23);
    // cid = wp.atomic_add(nacon_out, 0, 1)                                                   <L 268>
    var_33 = wp::atomic_add(var_nacon_out, var_31, var_32);
    // if cid < naconmax_in:                                                                  <L 269>
    var_34 = (var_33 < var_naconmax_in);
    if (var_34) {
        // contact_dist_out[cid] = dist_in                                                    <L 270>
        wp::array_store(var_contact_dist_out, var_33, var_dist_in);
        // contact_pos_out[cid] = pos_in                                                      <L 271>
        wp::array_store(var_contact_pos_out, var_33, var_pos_in);
        // contact_frame_out[cid] = frame_in                                                  <L 272>
        wp::array_store(var_contact_frame_out, var_33, var_frame_in);
        // contact_geom_out[cid] = geoms_in                                                   <L 273>
        wp::array_store(var_contact_geom_out, var_33, var_geoms_in);
        // contact_worldid_out[cid] = worldid_in                                              <L 274>
        wp::array_store(var_contact_worldid_out, var_33, var_worldid_in);
        // includemargin = margin_in                                                          <L 275>
        var_35 = wp::copy(var_margin_in);
        // contact_includemargin_out[cid] = includemargin                                     <L 276>
        wp::array_store(var_contact_includemargin_out, var_33, var_35);
        // contact_dim_out[cid] = condim_in                                                   <L 277>
        wp::array_store(var_contact_dim_out, var_33, var_condim_in);
        // contact_friction_out[cid] = friction_in                                            <L 278>
        wp::array_store(var_contact_friction_out, var_33, var_friction_in);
        // contact_solref_out[cid] = solref_in                                                <L 279>
        wp::array_store(var_contact_solref_out, var_33, var_solref_in);
        // contact_solreffriction_out[cid] = solreffriction_in                                <L 280>
        wp::array_store(var_contact_solreffriction_out, var_33, var_solreffriction_in);
        // contact_solimp_out[cid] = solimp_in                                                <L 281>
        wp::array_store(var_contact_solimp_out, var_33, var_solimp_in);
        // contact_type_out[cid] = contact_type                                               <L 282>
        wp::array_store(var_contact_type_out, var_33, var_30);
        // contact_geomcollisionid_out[cid] = id_                                             <L 283>
        wp::array_store(var_contact_geomcollisionid_out, var_33, var_id_);
        // for i in range(contact_efc_address_out.shape[1]):                                  <L 284>
        var_36 = &(var_contact_efc_address_out.shape);
        var_39 = wp::load(var_36);
        var_38 = wp::extract(var_39, var_37);
        var_40 = wp::range(var_38);
        start_for_1:;
            if (iter_cmp(var_40) == 0) goto end_for_1;
            var_41 = wp::iter_next(var_40);
            // contact_efc_address_out[cid, i] = -1                                           <L 285>
            wp::array_store(var_contact_efc_address_out, var_33, var_41, var_42);
            goto start_for_1;
        end_for_1:;
        // return int(active)                                                                 <L 286>
        var_43 = wp::int(var_0);
        return var_43;
    }
    // return 0                                                                               <L 287>
    return var_44;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_primitive.py:280
static CUDA_CALLABLE void plane_sphere_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_3242f8a8 var_plane,
    Geom_3242f8a8 var_sphere,
    wp::int32 var_worldid,
    wp::float32 var_margin,
    wp::float32 var_gap,
    wp::int32 var_condim,
    wp::vec_t<5, wp::float32> var_friction,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<2, wp::float32> var_solreffriction,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::vec_t<2, wp::int32> var_pairid,
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
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::int32> var_nacon_out)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32>* var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32>* var_3;
    wp::vec_t<3, wp::float32>* var_4;
    wp::vec_t<3, wp::float32>* var_5;
    const wp::int32 var_6 = 0;
    wp::float32 var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::float32 var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    const wp::int32 var_13 = 0;
    wp::mat_t<3, 3, wp::float32> var_14;
    wp::int32 var_15;
    //---------
    // forward
    // def plane_sphere_wrapper(                                                              <L 281>
    // normal = plane.normal                                                                  <L 315>
    var_0 = &((var_plane).normal);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    // dist, pos = plane_sphere(normal, plane.pos, sphere.pos, sphere.size[0])                <L 316>
    var_3 = &((var_plane).pos);
    var_4 = &((var_sphere).pos);
    var_5 = &((var_sphere).size);
    var_8 = wp::load(var_5);
    var_7 = wp::extract(var_8, var_6);
    var_11 = wp::load(var_3);
    var_12 = wp::load(var_4);
    plane_sphere_0(var_1, var_11, var_12, var_7, var_9, var_10);
    // write_contact(                                                                         <L 318>
    // naconmax_in,                                                                           <L 319>
    // 0,                                                                                     <L 320>
    // dist,                                                                                  <L 321>
    // pos,                                                                                   <L 322>
    // make_frame(normal),                                                                    <L 323>
    var_14 = make_frame_0(var_1);
    // margin,                                                                                <L 324>
    // gap,                                                                                   <L 325>
    // condim,                                                                                <L 326>
    // friction,                                                                              <L 327>
    // solref,                                                                                <L 328>
    // solreffriction,                                                                        <L 329>
    // solimp,                                                                                <L 330>
    // geoms,                                                                                 <L 331>
    // pairid,                                                                                <L 332>
    // worldid,                                                                               <L 333>
    // contact_dist_out,                                                                      <L 334>
    // contact_pos_out,                                                                       <L 335>
    // contact_frame_out,                                                                     <L 336>
    // contact_includemargin_out,                                                             <L 337>
    // contact_friction_out,                                                                  <L 338>
    // contact_solref_out,                                                                    <L 339>
    // contact_solreffriction_out,                                                            <L 340>
    // contact_solimp_out,                                                                    <L 341>
    // contact_dim_out,                                                                       <L 342>
    // contact_geom_out,                                                                      <L 343>
    // contact_efc_address_out,                                                               <L 344>
    // contact_worldid_out,                                                                   <L 345>
    // contact_type_out,                                                                      <L 346>
    // contact_geomcollisionid_out,                                                           <L 347>
    // nacon_out,                                                                             <L 348>
    var_15 = write_contact_0(var_naconmax_in, var_13, var_9, var_10, var_14, var_margin, var_gap, var_condim, var_friction, var_solref, var_solreffriction, var_solimp, var_geoms, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_nacon_out);
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/math.py:0
static CUDA_CALLABLE void normalize_with_norm_0(
    wp::vec_t<3, wp::float32> var_x,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::float32 & ret_1)
{
    //---------
    // primal vars
    wp::float32 var_0;
    const wp::float32 var_1 = 0.0;
    bool var_2;
    const wp::float32 var_3 = 0.0;
    wp::vec_t<3, wp::float32> var_4;
    //---------
    // forward
    // def normalize_with_norm(x: Any):                                                       <L 1>
    // norm = wp.length(x)                                                                    <L 2>
    var_0 = wp::length(var_x);
    // if norm == 0.0:                                                                        <L 3>
    var_2 = (var_0 == var_1);
    if (var_2) {
        // return x, 0.0                                                                      <L 4>
        ret_0 = var_x;
        ret_1 = var_3;
        return;
    }
    // return x / norm, norm                                                                  <L 5>
    var_4 = wp::div(var_x, var_0);
    ret_0 = var_4;
    ret_1 = var_0;
    return;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_primitive_core.py:252
static CUDA_CALLABLE void plane_capsule_0(
    wp::vec_t<3, wp::float32> var_plane_normal,
    wp::vec_t<3, wp::float32> var_plane_pos,
    wp::vec_t<3, wp::float32> var_capsule_pos,
    wp::vec_t<3, wp::float32> var_capsule_axis,
    wp::float32 var_capsule_radius,
    wp::float32 var_capsule_half_length,
    wp::vec_t<2, wp::float32> & ret_0,
    wp::mat_t<2, 3, wp::float32> & ret_1,
    wp::mat_t<3, 3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::float32 var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::float32 var_6;
    const wp::float32 var_7 = 0.5;
    bool var_8;
    bool var_9;
    const wp::float32 var_10 = -0.5;
    const wp::int32 var_11 = 1;
    wp::float32 var_12;
    bool var_13;
    const wp::int32 var_14 = 1;
    wp::float32 var_15;
    const wp::float32 var_16 = 0.5;
    bool var_17;
    const wp::float32 var_18 = 0.0;
    const wp::float32 var_19 = 1.0;
    const wp::float32 var_20 = 0.0;
    wp::vec_t<3, wp::float32> var_21;
    wp::vec_t<3, wp::float32> var_22;
    const wp::float32 var_23 = 0.0;
    const wp::float32 var_24 = 0.0;
    const wp::float32 var_25 = 1.0;
    wp::vec_t<3, wp::float32> var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::vec_t<3, wp::float32> var_28;
    wp::vec_t<3, wp::float32> var_29;
    const wp::int32 var_30 = 0;
    wp::float32 var_31;
    const wp::int32 var_32 = 1;
    wp::float32 var_33;
    const wp::int32 var_34 = 2;
    wp::float32 var_35;
    const wp::int32 var_36 = 0;
    wp::float32 var_37;
    const wp::int32 var_38 = 1;
    wp::float32 var_39;
    const wp::int32 var_40 = 2;
    wp::float32 var_41;
    const wp::int32 var_42 = 0;
    wp::float32 var_43;
    const wp::int32 var_44 = 1;
    wp::float32 var_45;
    const wp::int32 var_46 = 2;
    wp::float32 var_47;
    wp::mat_t<3, 3, wp::float32> var_48;
    wp::vec_t<3, wp::float32> var_49;
    wp::vec_t<3, wp::float32> var_50;
    wp::float32 var_51;
    wp::vec_t<3, wp::float32> var_52;
    wp::vec_t<3, wp::float32> var_53;
    wp::float32 var_54;
    wp::vec_t<3, wp::float32> var_55;
    wp::vec_t<2, wp::float32> var_56;
    const wp::int32 var_57 = 0;
    wp::float32 var_58;
    const wp::int32 var_59 = 1;
    wp::float32 var_60;
    const wp::int32 var_61 = 2;
    wp::float32 var_62;
    const wp::int32 var_63 = 0;
    wp::float32 var_64;
    const wp::int32 var_65 = 1;
    wp::float32 var_66;
    const wp::int32 var_67 = 2;
    wp::float32 var_68;
    wp::mat_t<2, 3, wp::float32> var_69;
    //---------
    // forward
    // def plane_capsule(                                                                     <L 253>
    // n = plane_normal                                                                       <L 277>
    var_0 = wp::copy(var_plane_normal);
    // axis = capsule_axis                                                                    <L 278>
    var_1 = wp::copy(var_capsule_axis);
    // b, b_norm = normalize_with_norm(axis - n * wp.dot(n, axis))                            <L 281>
    var_2 = wp::dot(var_0, var_1);
    var_3 = wp::mul(var_0, var_2);
    var_4 = wp::sub(var_1, var_3);
    normalize_with_norm_0(var_4, var_5, var_6);
    // if b_norm < 0.5:                                                                       <L 283>
    var_8 = (var_6 < var_7);
    if (var_8) {
        // if -0.5 < n[1] and n[1] < 0.5:                                                     <L 284>
        var_12 = wp::extract(var_0, var_11);
        var_13 = (var_10 < var_12);
        var_9 = var_13;
        if (var_9) {
            var_15 = wp::extract(var_0, var_14);
            var_17 = (var_15 < var_16);
            var_9 = var_9 && var_17;
        }
        if (var_9) {
            // b = wp.vec3(0.0, 1.0, 0.0)                                                     <L 285>
            var_21 = wp::vec_t<3, wp::float32>(var_18, var_19, var_20);
        }
        var_22 = wp::where(var_9, var_21, var_5);
        if (!var_9) {
            // b = wp.vec3(0.0, 0.0, 1.0)                                                     <L 287>
            var_26 = wp::vec_t<3, wp::float32>(var_23, var_24, var_25);
        }
        var_27 = wp::where(var_9, var_22, var_26);
    }
    var_28 = wp::where(var_8, var_27, var_5);
    // c = wp.cross(n, b)                                                                     <L 289>
    var_29 = wp::cross(var_0, var_28);
    // frame = wp.mat33(n[0], n[1], n[2], b[0], b[1], b[2], c[0], c[1], c[2])                 <L 290>
    var_31 = wp::extract(var_0, var_30);
    var_33 = wp::extract(var_0, var_32);
    var_35 = wp::extract(var_0, var_34);
    var_37 = wp::extract(var_28, var_36);
    var_39 = wp::extract(var_28, var_38);
    var_41 = wp::extract(var_28, var_40);
    var_43 = wp::extract(var_29, var_42);
    var_45 = wp::extract(var_29, var_44);
    var_47 = wp::extract(var_29, var_46);
    var_48 = wp::mat_t<3, 3, wp::float32>(var_31, var_33, var_35, var_37, var_39, var_41, var_43, var_45, var_47);
    // segment = axis * capsule_half_length                                                   <L 291>
    var_49 = wp::mul(var_1, var_capsule_half_length);
    // dist1, pos1 = plane_sphere(n, plane_pos, capsule_pos + segment, capsule_radius)        <L 294>
    var_50 = wp::add(var_capsule_pos, var_49);
    plane_sphere_0(var_0, var_plane_pos, var_50, var_capsule_radius, var_51, var_52);
    // dist2, pos2 = plane_sphere(n, plane_pos, capsule_pos - segment, capsule_radius)        <L 297>
    var_53 = wp::sub(var_capsule_pos, var_49);
    plane_sphere_0(var_0, var_plane_pos, var_53, var_capsule_radius, var_54, var_55);
    // dist = wp.vec2(dist1, dist2)                                                           <L 299>
    var_56 = wp::vec_t<2, wp::float32>(var_51, var_54);
    // pos = mat23f(pos1[0], pos1[1], pos1[2], pos2[0], pos2[1], pos2[2])                     <L 300>
    var_58 = wp::extract(var_52, var_57);
    var_60 = wp::extract(var_52, var_59);
    var_62 = wp::extract(var_52, var_61);
    var_64 = wp::extract(var_55, var_63);
    var_66 = wp::extract(var_55, var_65);
    var_68 = wp::extract(var_55, var_67);
    var_69 = wp::mat_t<2, 3, wp::float32>({var_58, var_60, var_62, var_64, var_66, var_68});
    // return dist, pos, frame                                                                <L 302>
    ret_0 = var_56;
    ret_1 = var_69;
    ret_2 = var_48;
    return;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_primitive.py:583
static CUDA_CALLABLE void plane_capsule_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_3242f8a8 var_plane,
    Geom_3242f8a8 var_cap,
    wp::int32 var_worldid,
    wp::float32 var_margin,
    wp::float32 var_gap,
    wp::int32 var_condim,
    wp::vec_t<5, wp::float32> var_friction,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<2, wp::float32> var_solreffriction,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::vec_t<2, wp::int32> var_pairid,
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
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::int32> var_nacon_out)
{
    //---------
    // primal vars
    wp::mat_t<3, 3, wp::float32>* var_0;
    const wp::int32 var_1 = 0;
    const wp::int32 var_2 = 2;
    wp::float32 var_3;
    wp::mat_t<3, 3, wp::float32> var_4;
    wp::mat_t<3, 3, wp::float32>* var_5;
    const wp::int32 var_6 = 1;
    const wp::int32 var_7 = 2;
    wp::float32 var_8;
    wp::mat_t<3, 3, wp::float32> var_9;
    wp::mat_t<3, 3, wp::float32>* var_10;
    const wp::int32 var_11 = 2;
    const wp::int32 var_12 = 2;
    wp::float32 var_13;
    wp::mat_t<3, 3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::vec_t<3, wp::float32>* var_16;
    wp::vec_t<3, wp::float32>* var_17;
    wp::vec_t<3, wp::float32>* var_18;
    wp::vec_t<3, wp::float32>* var_19;
    const wp::int32 var_20 = 0;
    wp::float32 var_21;
    wp::vec_t<3, wp::float32> var_22;
    wp::vec_t<3, wp::float32>* var_23;
    const wp::int32 var_24 = 1;
    wp::float32 var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::vec_t<2, wp::float32> var_27;
    wp::mat_t<2, 3, wp::float32> var_28;
    wp::mat_t<3, 3, wp::float32> var_29;
    wp::vec_t<3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<3, wp::float32> var_32;
    const wp::int32 var_33 = 0;
    wp::float32 var_34;
    wp::vec_t<3, wp::float32> var_35;
    wp::int32 var_36;
    const wp::int32 var_37 = 1;
    wp::float32 var_38;
    wp::vec_t<3, wp::float32> var_39;
    wp::int32 var_40;
    //---------
    // forward
    // def plane_capsule_wrapper(                                                             <L 584>
    // capsule_axis = wp.vec3(cap.rot[0, 2], cap.rot[1, 2], cap.rot[2, 2])                    <L 619>
    var_0 = &((var_cap).rot);
    var_4 = wp::load(var_0);
    var_3 = wp::extract(var_4, var_1, var_2);
    var_5 = &((var_cap).rot);
    var_9 = wp::load(var_5);
    var_8 = wp::extract(var_9, var_6, var_7);
    var_10 = &((var_cap).rot);
    var_14 = wp::load(var_10);
    var_13 = wp::extract(var_14, var_11, var_12);
    var_15 = wp::vec_t<3, wp::float32>(var_3, var_8, var_13);
    // dist, pos, frame = plane_capsule(                                                      <L 621>
    // plane.normal,                                                                          <L 622>
    var_16 = &((var_plane).normal);
    // plane.pos,                                                                             <L 623>
    var_17 = &((var_plane).pos);
    // cap.pos,                                                                               <L 624>
    var_18 = &((var_cap).pos);
    // capsule_axis,                                                                          <L 625>
    // cap.size[0],  # radius                                                                 <L 626>
    var_19 = &((var_cap).size);
    var_22 = wp::load(var_19);
    var_21 = wp::extract(var_22, var_20);
    // cap.size[1],  # half_length                                                            <L 627>
    var_23 = &((var_cap).size);
    var_26 = wp::load(var_23);
    var_25 = wp::extract(var_26, var_24);
    var_30 = wp::load(var_16);
    var_31 = wp::load(var_17);
    var_32 = wp::load(var_18);
    plane_capsule_0(var_30, var_31, var_32, var_15, var_21, var_25, var_27, var_28, var_29);
    // for i in range(2):                                                                     <L 630>
    // write_contact(                                                                         <L 631>
    // naconmax_in,                                                                           <L 632>
    // i,                                                                                     <L 633>
    // dist[i],                                                                               <L 634>
    var_34 = wp::extract(var_27, var_33);
    // pos[i],                                                                                <L 635>
    var_35 = wp::extract(var_28, var_33);
    // frame,                                                                                 <L 636>
    // margin,                                                                                <L 637>
    // gap,                                                                                   <L 638>
    // condim,                                                                                <L 639>
    // friction,                                                                              <L 640>
    // solref,                                                                                <L 641>
    // solreffriction,                                                                        <L 642>
    // solimp,                                                                                <L 643>
    // geoms,                                                                                 <L 644>
    // pairid,                                                                                <L 645>
    // worldid,                                                                               <L 646>
    // contact_dist_out,                                                                      <L 647>
    // contact_pos_out,                                                                       <L 648>
    // contact_frame_out,                                                                     <L 649>
    // contact_includemargin_out,                                                             <L 650>
    // contact_friction_out,                                                                  <L 651>
    // contact_solref_out,                                                                    <L 652>
    // contact_solreffriction_out,                                                            <L 653>
    // contact_solimp_out,                                                                    <L 654>
    // contact_dim_out,                                                                       <L 655>
    // contact_geom_out,                                                                      <L 656>
    // contact_efc_address_out,                                                               <L 657>
    // contact_worldid_out,                                                                   <L 658>
    // contact_type_out,                                                                      <L 659>
    // contact_geomcollisionid_out,                                                           <L 660>
    // nacon_out,                                                                             <L 661>
    var_36 = write_contact_0(var_naconmax_in, var_33, var_34, var_35, var_29, var_margin, var_gap, var_condim, var_friction, var_solref, var_solreffriction, var_solimp, var_geoms, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_nacon_out);
    // write_contact(                                                                         <L 631>
    // naconmax_in,                                                                           <L 632>
    // i,                                                                                     <L 633>
    // dist[i],                                                                               <L 634>
    var_38 = wp::extract(var_27, var_37);
    // pos[i],                                                                                <L 635>
    var_39 = wp::extract(var_28, var_37);
    // frame,                                                                                 <L 636>
    // margin,                                                                                <L 637>
    // gap,                                                                                   <L 638>
    // condim,                                                                                <L 639>
    // friction,                                                                              <L 640>
    // solref,                                                                                <L 641>
    // solreffriction,                                                                        <L 642>
    // solimp,                                                                                <L 643>
    // geoms,                                                                                 <L 644>
    // pairid,                                                                                <L 645>
    // worldid,                                                                               <L 646>
    // contact_dist_out,                                                                      <L 647>
    // contact_pos_out,                                                                       <L 648>
    // contact_frame_out,                                                                     <L 649>
    // contact_includemargin_out,                                                             <L 650>
    // contact_friction_out,                                                                  <L 651>
    // contact_solref_out,                                                                    <L 652>
    // contact_solreffriction_out,                                                            <L 653>
    // contact_solimp_out,                                                                    <L 654>
    // contact_dim_out,                                                                       <L 655>
    // contact_geom_out,                                                                      <L 656>
    // contact_efc_address_out,                                                               <L 657>
    // contact_worldid_out,                                                                   <L 658>
    // contact_type_out,                                                                      <L 659>
    // contact_geomcollisionid_out,                                                           <L 660>
    // nacon_out,                                                                             <L 661>
    var_40 = write_contact_0(var_naconmax_in, var_37, var_38, var_39, var_29, var_margin, var_gap, var_condim, var_friction, var_solref, var_solreffriction, var_solimp, var_geoms, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_nacon_out);
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_primitive_core.py:55
static CUDA_CALLABLE void sphere_sphere_0(
    wp::vec_t<3, wp::float32> var_pos1,
    wp::float32 var_radius1,
    wp::vec_t<3, wp::float32> var_pos2,
    wp::float32 var_radius2,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::float32 var_1;
    const wp::float32 var_2 = 0.0;
    bool var_3;
    const wp::float32 var_4 = 1.0;
    const wp::float32 var_5 = 0.0;
    const wp::float32 var_6 = 0.0;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    const wp::float32 var_12 = 0.5;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::vec_t<3, wp::float32> var_16;
    //---------
    // forward
    // def sphere_sphere(                                                                     <L 56>
    // dir = pos2 - pos1                                                                      <L 76>
    var_0 = wp::sub(var_pos2, var_pos1);
    // dist = wp.length(dir)                                                                  <L 77>
    var_1 = wp::length(var_0);
    // if dist == 0.0:                                                                        <L 78>
    var_3 = (var_1 == var_2);
    if (var_3) {
        // n = wp.vec3(1.0, 0.0, 0.0)                                                         <L 79>
        var_7 = wp::vec_t<3, wp::float32>(var_4, var_5, var_6);
    }
    if (!var_3) {
        // n = dir / dist                                                                     <L 81>
        var_8 = wp::div(var_0, var_1);
    }
    var_9 = wp::where(var_3, var_7, var_8);
    // dist = dist - (radius1 + radius2)                                                      <L 82>
    var_10 = wp::add(var_radius1, var_radius2);
    var_11 = wp::sub(var_1, var_10);
    // pos = pos1 + n * (radius1 + 0.5 * dist)                                                <L 83>
    var_13 = wp::mul(var_12, var_11);
    var_14 = wp::add(var_radius1, var_13);
    var_15 = wp::mul(var_9, var_14);
    var_16 = wp::add(var_pos1, var_15);
    // return dist, pos, n                                                                    <L 84>
    ret_0 = var_11;
    ret_1 = var_16;
    ret_2 = var_9;
    return;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_primitive.py:352
static CUDA_CALLABLE void sphere_sphere_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_3242f8a8 var_sphere1,
    Geom_3242f8a8 var_sphere2,
    wp::int32 var_worldid,
    wp::float32 var_margin,
    wp::float32 var_gap,
    wp::int32 var_condim,
    wp::vec_t<5, wp::float32> var_friction,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<2, wp::float32> var_solreffriction,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::vec_t<2, wp::int32> var_pairid,
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
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::int32> var_nacon_out)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32>* var_0;
    wp::vec_t<3, wp::float32>* var_1;
    const wp::int32 var_2 = 0;
    wp::float32 var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32>* var_5;
    wp::vec_t<3, wp::float32>* var_6;
    const wp::int32 var_7 = 0;
    wp::float32 var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::float32 var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::vec_t<3, wp::float32> var_13;
    wp::vec_t<3, wp::float32> var_14;
    const wp::int32 var_15 = 0;
    wp::mat_t<3, 3, wp::float32> var_16;
    wp::int32 var_17;
    //---------
    // forward
    // def sphere_sphere_wrapper(                                                             <L 353>
    // dist, pos, normal = sphere_sphere(sphere1.pos, sphere1.size[0], sphere2.pos, sphere2.size[0])       <L 387>
    var_0 = &((var_sphere1).pos);
    var_1 = &((var_sphere1).size);
    var_4 = wp::load(var_1);
    var_3 = wp::extract(var_4, var_2);
    var_5 = &((var_sphere2).pos);
    var_6 = &((var_sphere2).size);
    var_9 = wp::load(var_6);
    var_8 = wp::extract(var_9, var_7);
    var_13 = wp::load(var_0);
    var_14 = wp::load(var_5);
    sphere_sphere_0(var_13, var_3, var_14, var_8, var_10, var_11, var_12);
    // write_contact(                                                                         <L 389>
    // naconmax_in,                                                                           <L 390>
    // 0,                                                                                     <L 391>
    // dist,                                                                                  <L 392>
    // pos,                                                                                   <L 393>
    // make_frame(normal),                                                                    <L 394>
    var_16 = make_frame_0(var_12);
    // margin,                                                                                <L 395>
    // gap,                                                                                   <L 396>
    // condim,                                                                                <L 397>
    // friction,                                                                              <L 398>
    // solref,                                                                                <L 399>
    // solreffriction,                                                                        <L 400>
    // solimp,                                                                                <L 401>
    // geoms,                                                                                 <L 402>
    // pairid,                                                                                <L 403>
    // worldid,                                                                               <L 404>
    // contact_dist_out,                                                                      <L 405>
    // contact_pos_out,                                                                       <L 406>
    // contact_frame_out,                                                                     <L 407>
    // contact_includemargin_out,                                                             <L 408>
    // contact_friction_out,                                                                  <L 409>
    // contact_solref_out,                                                                    <L 410>
    // contact_solreffriction_out,                                                            <L 411>
    // contact_solimp_out,                                                                    <L 412>
    // contact_dim_out,                                                                       <L 413>
    // contact_geom_out,                                                                      <L 414>
    // contact_efc_address_out,                                                               <L 415>
    // contact_worldid_out,                                                                   <L 416>
    // contact_type_out,                                                                      <L 417>
    // contact_geomcollisionid_out,                                                           <L 418>
    // nacon_out,                                                                             <L 419>
    var_17 = write_contact_0(var_naconmax_in, var_15, var_10, var_11, var_16, var_margin, var_gap, var_condim, var_friction, var_solref, var_solreffriction, var_solimp, var_geoms, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_nacon_out);
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/math.py:268
static CUDA_CALLABLE wp::vec_t<3, wp::float32> closest_segment_point_0(
    wp::vec_t<3, wp::float32> var_a,
    wp::vec_t<3, wp::float32> var_b,
    wp::vec_t<3, wp::float32> var_pt)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    const wp::float32 var_4 = 1e-06;
    wp::float32 var_5;
    wp::float32 var_6;
    const wp::float32 var_7 = 0.0;
    const wp::float32 var_8 = 1.0;
    wp::float32 var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::vec_t<3, wp::float32> var_11;
    //---------
    // forward
    // def closest_segment_point(a: wp.vec3, b: wp.vec3, pt: wp.vec3) -> wp.vec3:             <L 269>
    // ab = b - a                                                                             <L 271>
    var_0 = wp::sub(var_b, var_a);
    // t = wp.dot(pt - a, ab) / (wp.dot(ab, ab) + 1e-6)                                       <L 272>
    var_1 = wp::sub(var_pt, var_a);
    var_2 = wp::dot(var_1, var_0);
    var_3 = wp::dot(var_0, var_0);
    var_5 = wp::add(var_3, var_4);
    var_6 = wp::div(var_2, var_5);
    // return a + wp.clamp(t, 0.0, 1.0) * ab                                                  <L 273>
    var_9 = wp::clamp(var_6, var_7, var_8);
    var_10 = wp::mul(var_9, var_0);
    var_11 = wp::add(var_a, var_10);
    return var_11;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_primitive_core.py:87
static CUDA_CALLABLE void sphere_capsule_0(
    wp::vec_t<3, wp::float32> var_sphere_pos,
    wp::float32 var_sphere_radius,
    wp::vec_t<3, wp::float32> var_capsule_pos,
    wp::vec_t<3, wp::float32> var_capsule_axis,
    wp::float32 var_capsule_radius,
    wp::float32 var_capsule_half_length,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::float32 var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32> var_6;
    //---------
    // forward
    // def sphere_capsule(                                                                    <L 88>
    // segment = capsule_axis * capsule_half_length                                           <L 113>
    var_0 = wp::mul(var_capsule_axis, var_capsule_half_length);
    // pt = closest_segment_point(capsule_pos - segment, capsule_pos + segment, sphere_pos)       <L 116>
    var_1 = wp::sub(var_capsule_pos, var_0);
    var_2 = wp::add(var_capsule_pos, var_0);
    var_3 = closest_segment_point_0(var_1, var_2, var_sphere_pos);
    // return sphere_sphere(sphere_pos, sphere_radius, pt, capsule_radius)                    <L 119>
    sphere_sphere_0(var_sphere_pos, var_sphere_radius, var_3, var_capsule_radius, var_4, var_5, var_6);
    ret_0 = var_4;
    ret_1 = var_5;
    ret_2 = var_6;
    return;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_primitive.py:423
static CUDA_CALLABLE void sphere_capsule_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_3242f8a8 var_sphere,
    Geom_3242f8a8 var_cap,
    wp::int32 var_worldid,
    wp::float32 var_margin,
    wp::float32 var_gap,
    wp::int32 var_condim,
    wp::vec_t<5, wp::float32> var_friction,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<2, wp::float32> var_solreffriction,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::vec_t<2, wp::int32> var_pairid,
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
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::int32> var_nacon_out)
{
    //---------
    // primal vars
    wp::mat_t<3, 3, wp::float32>* var_0;
    const wp::int32 var_1 = 0;
    const wp::int32 var_2 = 2;
    wp::float32 var_3;
    wp::mat_t<3, 3, wp::float32> var_4;
    wp::mat_t<3, 3, wp::float32>* var_5;
    const wp::int32 var_6 = 1;
    const wp::int32 var_7 = 2;
    wp::float32 var_8;
    wp::mat_t<3, 3, wp::float32> var_9;
    wp::mat_t<3, 3, wp::float32>* var_10;
    const wp::int32 var_11 = 2;
    const wp::int32 var_12 = 2;
    wp::float32 var_13;
    wp::mat_t<3, 3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::vec_t<3, wp::float32>* var_16;
    wp::vec_t<3, wp::float32>* var_17;
    const wp::int32 var_18 = 0;
    wp::float32 var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32>* var_21;
    wp::vec_t<3, wp::float32>* var_22;
    const wp::int32 var_23 = 0;
    wp::float32 var_24;
    wp::vec_t<3, wp::float32> var_25;
    wp::vec_t<3, wp::float32>* var_26;
    const wp::int32 var_27 = 1;
    wp::float32 var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::float32 var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::vec_t<3, wp::float32> var_33;
    wp::vec_t<3, wp::float32> var_34;
    const wp::int32 var_35 = 0;
    wp::mat_t<3, 3, wp::float32> var_36;
    wp::int32 var_37;
    //---------
    // forward
    // def sphere_capsule_wrapper(                                                            <L 424>
    // axis = wp.vec3(cap.rot[0, 2], cap.rot[1, 2], cap.rot[2, 2])                            <L 459>
    var_0 = &((var_cap).rot);
    var_4 = wp::load(var_0);
    var_3 = wp::extract(var_4, var_1, var_2);
    var_5 = &((var_cap).rot);
    var_9 = wp::load(var_5);
    var_8 = wp::extract(var_9, var_6, var_7);
    var_10 = &((var_cap).rot);
    var_14 = wp::load(var_10);
    var_13 = wp::extract(var_14, var_11, var_12);
    var_15 = wp::vec_t<3, wp::float32>(var_3, var_8, var_13);
    // dist, pos, normal = sphere_capsule(sphere.pos, sphere.size[0], cap.pos, axis, cap.size[0], cap.size[1])       <L 461>
    var_16 = &((var_sphere).pos);
    var_17 = &((var_sphere).size);
    var_20 = wp::load(var_17);
    var_19 = wp::extract(var_20, var_18);
    var_21 = &((var_cap).pos);
    var_22 = &((var_cap).size);
    var_25 = wp::load(var_22);
    var_24 = wp::extract(var_25, var_23);
    var_26 = &((var_cap).size);
    var_29 = wp::load(var_26);
    var_28 = wp::extract(var_29, var_27);
    var_33 = wp::load(var_16);
    var_34 = wp::load(var_21);
    sphere_capsule_0(var_33, var_19, var_34, var_15, var_24, var_28, var_30, var_31, var_32);
    // write_contact(                                                                         <L 463>
    // naconmax_in,                                                                           <L 464>
    // 0,                                                                                     <L 465>
    // dist,                                                                                  <L 466>
    // pos,                                                                                   <L 467>
    // make_frame(normal),                                                                    <L 468>
    var_36 = make_frame_0(var_32);
    // margin,                                                                                <L 469>
    // gap,                                                                                   <L 470>
    // condim,                                                                                <L 471>
    // friction,                                                                              <L 472>
    // solref,                                                                                <L 473>
    // solreffriction,                                                                        <L 474>
    // solimp,                                                                                <L 475>
    // geoms,                                                                                 <L 476>
    // pairid,                                                                                <L 477>
    // worldid,                                                                               <L 478>
    // contact_dist_out,                                                                      <L 479>
    // contact_pos_out,                                                                       <L 480>
    // contact_frame_out,                                                                     <L 481>
    // contact_includemargin_out,                                                             <L 482>
    // contact_friction_out,                                                                  <L 483>
    // contact_solref_out,                                                                    <L 484>
    // contact_solreffriction_out,                                                            <L 485>
    // contact_solimp_out,                                                                    <L 486>
    // contact_dim_out,                                                                       <L 487>
    // contact_geom_out,                                                                      <L 488>
    // contact_efc_address_out,                                                               <L 489>
    // contact_worldid_out,                                                                   <L 490>
    // contact_type_out,                                                                      <L 491>
    // contact_geomcollisionid_out,                                                           <L 492>
    // nacon_out,                                                                             <L 493>
    var_37 = write_contact_0(var_naconmax_in, var_35, var_30, var_31, var_36, var_margin, var_gap, var_condim, var_friction, var_solref, var_solreffriction, var_solimp, var_geoms, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_nacon_out);
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_primitive_core.py:1043
static CUDA_CALLABLE void sphere_box_0(
    wp::vec_t<3, wp::float32> var_sphere_pos,
    wp::float32 var_sphere_radius,
    wp::vec_t<3, wp::float32> var_box_pos,
    wp::mat_t<3, 3, wp::float32> var_box_rot,
    wp::vec_t<3, wp::float32> var_box_size,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::mat_t<3, 3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::float32 var_8;
    const wp::float32 var_9 = 1e-15;
    bool var_10;
    const wp::float32 var_11 = 2.0;
    const wp::int32 var_12 = 0;
    wp::float32 var_13;
    const wp::int32 var_14 = 1;
    wp::float32 var_15;
    wp::float32 var_16;
    const wp::int32 var_17 = 2;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    const wp::int32 var_21 = 0;
    wp::int32 var_22;
    const wp::int32 var_23 = 0;
    const wp::int32 var_24 = 2;
    wp::int32 var_25;
    const wp::float32 var_26 = 1.0;
    const wp::float32 var_27 = -1.0;
    wp::float32 var_28;
    const wp::int32 var_29 = 2;
    wp::int32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    const wp::int32 var_33 = 2;
    wp::int32 var_34;
    wp::float32 var_35;
    wp::float32 var_36;
    wp::float32 var_37;
    bool var_38;
    wp::float32 var_39;
    wp::int32 var_40;
    wp::float32 var_41;
    wp::int32 var_42;
    const wp::int32 var_43 = 1;
    const wp::int32 var_44 = 2;
    wp::int32 var_45;
    const wp::float32 var_46 = 1.0;
    const wp::float32 var_47 = -1.0;
    wp::float32 var_48;
    const wp::int32 var_49 = 2;
    wp::int32 var_50;
    wp::float32 var_51;
    wp::float32 var_52;
    const wp::int32 var_53 = 2;
    wp::int32 var_54;
    wp::float32 var_55;
    wp::float32 var_56;
    wp::float32 var_57;
    bool var_58;
    wp::float32 var_59;
    wp::int32 var_60;
    wp::float32 var_61;
    wp::int32 var_62;
    const wp::int32 var_63 = 2;
    const wp::int32 var_64 = 2;
    wp::int32 var_65;
    const wp::float32 var_66 = 1.0;
    const wp::float32 var_67 = -1.0;
    wp::float32 var_68;
    const wp::int32 var_69 = 2;
    wp::int32 var_70;
    wp::float32 var_71;
    wp::float32 var_72;
    const wp::int32 var_73 = 2;
    wp::int32 var_74;
    wp::float32 var_75;
    wp::float32 var_76;
    wp::float32 var_77;
    bool var_78;
    wp::float32 var_79;
    wp::int32 var_80;
    wp::float32 var_81;
    wp::int32 var_82;
    const wp::int32 var_83 = 3;
    const wp::int32 var_84 = 2;
    wp::int32 var_85;
    const wp::float32 var_86 = 1.0;
    const wp::float32 var_87 = -1.0;
    wp::float32 var_88;
    const wp::int32 var_89 = 2;
    wp::int32 var_90;
    wp::float32 var_91;
    wp::float32 var_92;
    const wp::int32 var_93 = 2;
    wp::int32 var_94;
    wp::float32 var_95;
    wp::float32 var_96;
    wp::float32 var_97;
    bool var_98;
    wp::float32 var_99;
    wp::int32 var_100;
    wp::float32 var_101;
    wp::int32 var_102;
    const wp::int32 var_103 = 4;
    const wp::int32 var_104 = 2;
    wp::int32 var_105;
    const wp::float32 var_106 = 1.0;
    const wp::float32 var_107 = -1.0;
    wp::float32 var_108;
    const wp::int32 var_109 = 2;
    wp::int32 var_110;
    wp::float32 var_111;
    wp::float32 var_112;
    const wp::int32 var_113 = 2;
    wp::int32 var_114;
    wp::float32 var_115;
    wp::float32 var_116;
    wp::float32 var_117;
    bool var_118;
    wp::float32 var_119;
    wp::int32 var_120;
    wp::float32 var_121;
    wp::int32 var_122;
    const wp::int32 var_123 = 5;
    const wp::int32 var_124 = 2;
    wp::int32 var_125;
    const wp::float32 var_126 = 1.0;
    const wp::float32 var_127 = -1.0;
    wp::float32 var_128;
    const wp::int32 var_129 = 2;
    wp::int32 var_130;
    wp::float32 var_131;
    wp::float32 var_132;
    const wp::int32 var_133 = 2;
    wp::int32 var_134;
    wp::float32 var_135;
    wp::float32 var_136;
    wp::float32 var_137;
    bool var_138;
    wp::float32 var_139;
    wp::int32 var_140;
    wp::float32 var_141;
    wp::int32 var_142;
    const wp::float32 var_143 = 0.0;
    wp::vec_t<3, wp::float32> var_144;
    const wp::int32 var_145 = 2;
    wp::int32 var_146;
    const wp::float32 var_147 = -1.0;
    const wp::float32 var_148 = 1.0;
    wp::float32 var_149;
    const wp::int32 var_150 = 2;
    wp::int32 var_151;
    wp::float32 var_152;
    wp::vec_t<3, wp::float32> var_153;
    const wp::float32 var_154 = 2.0;
    wp::vec_t<3, wp::float32> var_155;
    wp::vec_t<3, wp::float32> var_156;
    wp::vec_t<3, wp::float32> var_157;
    wp::float32 var_158;
    wp::float32 var_159;
    wp::vec_t<3, wp::float32> var_160;
    wp::vec_t<3, wp::float32> var_161;
    const wp::float32 var_162 = 0.5;
    wp::vec_t<3, wp::float32> var_163;
    wp::vec_t<3, wp::float32> var_164;
    wp::vec_t<3, wp::float32> var_165;
    wp::float32 var_166;
    wp::vec_t<3, wp::float32> var_167;
    wp::vec_t<3, wp::float32> var_168;
    wp::float32 var_169;
    wp::vec_t<3, wp::float32> var_170;
    wp::vec_t<3, wp::float32> var_171;
    //---------
    // forward
    // def sphere_box(                                                                        <L 1044>
    // center = wp.transpose(box_rot) @ (sphere_pos - box_pos)                                <L 1066>
    var_0 = wp::transpose(var_box_rot);
    var_1 = wp::sub(var_sphere_pos, var_box_pos);
    var_2 = wp::mul(var_0, var_1);
    // clamped = wp.max(-box_size, wp.min(box_size, center))                                  <L 1068>
    var_3 = wp::neg(var_box_size);
    var_4 = wp::min(var_box_size, var_2);
    var_5 = wp::max(var_3, var_4);
    // clamped_dir, dist = normalize_with_norm(clamped - center)                              <L 1069>
    var_6 = wp::sub(var_5, var_2);
    normalize_with_norm_0(var_6, var_7, var_8);
    // if dist <= MJ_MINVAL:                                                                  <L 1072>
    var_10 = (var_8 <= var_9);
    if (var_10) {
        // closest = 2.0 * (box_size[0] + box_size[1] + box_size[2])                          <L 1073>
        var_13 = wp::extract(var_box_size, var_12);
        var_15 = wp::extract(var_box_size, var_14);
        var_16 = wp::add(var_13, var_15);
        var_18 = wp::extract(var_box_size, var_17);
        var_19 = wp::add(var_16, var_18);
        var_20 = wp::mul(var_11, var_19);
        // k = wp.int32(0)                                                                    <L 1074>
        var_22 = wp::int32(var_21);
        // for i in range(6):                                                                 <L 1075>
        // face_dist = wp.abs(wp.where(i % 2, 1.0, -1.0) * box_size[i // 2] - center[i // 2])       <L 1076>
        var_25 = wp::mod(var_23, var_24);
        var_28 = wp::where(var_25, var_26, var_27);
        var_30 = wp::floordiv(var_23, var_29);
        var_31 = wp::extract(var_box_size, var_30);
        var_32 = wp::mul(var_28, var_31);
        var_34 = wp::floordiv(var_23, var_33);
        var_35 = wp::extract(var_2, var_34);
        var_36 = wp::sub(var_32, var_35);
        var_37 = wp::abs(var_36);
        // if closest > face_dist:                                                            <L 1077>
        var_38 = (var_20 > var_37);
        if (var_38) {
            // closest = face_dist                                                            <L 1078>
            var_39 = wp::copy(var_37);
            // k = i                                                                          <L 1079>
            var_40 = wp::copy(var_23);
        }
        var_41 = wp::where(var_38, var_39, var_20);
        var_42 = wp::where(var_38, var_40, var_22);
        // face_dist = wp.abs(wp.where(i % 2, 1.0, -1.0) * box_size[i // 2] - center[i // 2])       <L 1076>
        var_45 = wp::mod(var_43, var_44);
        var_48 = wp::where(var_45, var_46, var_47);
        var_50 = wp::floordiv(var_43, var_49);
        var_51 = wp::extract(var_box_size, var_50);
        var_52 = wp::mul(var_48, var_51);
        var_54 = wp::floordiv(var_43, var_53);
        var_55 = wp::extract(var_2, var_54);
        var_56 = wp::sub(var_52, var_55);
        var_57 = wp::abs(var_56);
        // if closest > face_dist:                                                            <L 1077>
        var_58 = (var_41 > var_57);
        if (var_58) {
            // closest = face_dist                                                            <L 1078>
            var_59 = wp::copy(var_57);
            // k = i                                                                          <L 1079>
            var_60 = wp::copy(var_43);
        }
        var_61 = wp::where(var_58, var_59, var_41);
        var_62 = wp::where(var_58, var_60, var_42);
        // face_dist = wp.abs(wp.where(i % 2, 1.0, -1.0) * box_size[i // 2] - center[i // 2])       <L 1076>
        var_65 = wp::mod(var_63, var_64);
        var_68 = wp::where(var_65, var_66, var_67);
        var_70 = wp::floordiv(var_63, var_69);
        var_71 = wp::extract(var_box_size, var_70);
        var_72 = wp::mul(var_68, var_71);
        var_74 = wp::floordiv(var_63, var_73);
        var_75 = wp::extract(var_2, var_74);
        var_76 = wp::sub(var_72, var_75);
        var_77 = wp::abs(var_76);
        // if closest > face_dist:                                                            <L 1077>
        var_78 = (var_61 > var_77);
        if (var_78) {
            // closest = face_dist                                                            <L 1078>
            var_79 = wp::copy(var_77);
            // k = i                                                                          <L 1079>
            var_80 = wp::copy(var_63);
        }
        var_81 = wp::where(var_78, var_79, var_61);
        var_82 = wp::where(var_78, var_80, var_62);
        // face_dist = wp.abs(wp.where(i % 2, 1.0, -1.0) * box_size[i // 2] - center[i // 2])       <L 1076>
        var_85 = wp::mod(var_83, var_84);
        var_88 = wp::where(var_85, var_86, var_87);
        var_90 = wp::floordiv(var_83, var_89);
        var_91 = wp::extract(var_box_size, var_90);
        var_92 = wp::mul(var_88, var_91);
        var_94 = wp::floordiv(var_83, var_93);
        var_95 = wp::extract(var_2, var_94);
        var_96 = wp::sub(var_92, var_95);
        var_97 = wp::abs(var_96);
        // if closest > face_dist:                                                            <L 1077>
        var_98 = (var_81 > var_97);
        if (var_98) {
            // closest = face_dist                                                            <L 1078>
            var_99 = wp::copy(var_97);
            // k = i                                                                          <L 1079>
            var_100 = wp::copy(var_83);
        }
        var_101 = wp::where(var_98, var_99, var_81);
        var_102 = wp::where(var_98, var_100, var_82);
        // face_dist = wp.abs(wp.where(i % 2, 1.0, -1.0) * box_size[i // 2] - center[i // 2])       <L 1076>
        var_105 = wp::mod(var_103, var_104);
        var_108 = wp::where(var_105, var_106, var_107);
        var_110 = wp::floordiv(var_103, var_109);
        var_111 = wp::extract(var_box_size, var_110);
        var_112 = wp::mul(var_108, var_111);
        var_114 = wp::floordiv(var_103, var_113);
        var_115 = wp::extract(var_2, var_114);
        var_116 = wp::sub(var_112, var_115);
        var_117 = wp::abs(var_116);
        // if closest > face_dist:                                                            <L 1077>
        var_118 = (var_101 > var_117);
        if (var_118) {
            // closest = face_dist                                                            <L 1078>
            var_119 = wp::copy(var_117);
            // k = i                                                                          <L 1079>
            var_120 = wp::copy(var_103);
        }
        var_121 = wp::where(var_118, var_119, var_101);
        var_122 = wp::where(var_118, var_120, var_102);
        // face_dist = wp.abs(wp.where(i % 2, 1.0, -1.0) * box_size[i // 2] - center[i // 2])       <L 1076>
        var_125 = wp::mod(var_123, var_124);
        var_128 = wp::where(var_125, var_126, var_127);
        var_130 = wp::floordiv(var_123, var_129);
        var_131 = wp::extract(var_box_size, var_130);
        var_132 = wp::mul(var_128, var_131);
        var_134 = wp::floordiv(var_123, var_133);
        var_135 = wp::extract(var_2, var_134);
        var_136 = wp::sub(var_132, var_135);
        var_137 = wp::abs(var_136);
        // if closest > face_dist:                                                            <L 1077>
        var_138 = (var_121 > var_137);
        if (var_138) {
            // closest = face_dist                                                            <L 1078>
            var_139 = wp::copy(var_137);
            // k = i                                                                          <L 1079>
            var_140 = wp::copy(var_123);
        }
        var_141 = wp::where(var_138, var_139, var_121);
        var_142 = wp::where(var_138, var_140, var_122);
        // nearest = wp.vec3(0.0)                                                             <L 1081>
        var_144 = wp::vec_t<3, wp::float32>(var_143);
        // nearest[k // 2] = wp.where(k % 2, -1.0, 1.0)                                       <L 1082>
        var_146 = wp::mod(var_142, var_145);
        var_149 = wp::where(var_146, var_147, var_148);
        var_151 = wp::floordiv(var_142, var_150);
        wp::assign_inplace(var_144, var_151, var_149);
        // pos = center + nearest * (sphere_radius - closest) / 2.0                           <L 1083>
        var_152 = wp::sub(var_sphere_radius, var_141);
        var_153 = wp::mul(var_144, var_152);
        var_155 = wp::div(var_153, var_154);
        var_156 = wp::add(var_2, var_155);
        // contact_normal = box_rot @ nearest                                                 <L 1084>
        var_157 = wp::mul(var_box_rot, var_144);
        // contact_distance = -closest - sphere_radius                                        <L 1085>
        var_158 = wp::neg(var_141);
        var_159 = wp::sub(var_158, var_sphere_radius);
    }
    if (!var_10) {
        // deepest = center + clamped_dir * sphere_radius                                     <L 1088>
        var_160 = wp::mul(var_7, var_sphere_radius);
        var_161 = wp::add(var_2, var_160);
        // pos = 0.5 * (clamped + deepest)                                                    <L 1089>
        var_163 = wp::add(var_5, var_161);
        var_164 = wp::mul(var_162, var_163);
        // contact_normal = box_rot @ clamped_dir                                             <L 1090>
        var_165 = wp::mul(var_box_rot, var_7);
        // contact_distance = dist - sphere_radius                                            <L 1091>
        var_166 = wp::sub(var_8, var_sphere_radius);
    }
    var_167 = wp::where(var_10, var_156, var_164);
    var_168 = wp::where(var_10, var_157, var_165);
    var_169 = wp::where(var_10, var_159, var_166);
    // contact_position = box_pos + box_rot @ pos                                             <L 1093>
    var_170 = wp::mul(var_box_rot, var_167);
    var_171 = wp::add(var_box_pos, var_170);
    // return contact_distance, contact_position, contact_normal                              <L 1095>
    ret_0 = var_169;
    ret_1 = var_171;
    ret_2 = var_168;
    return;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_primitive.py:1046
static CUDA_CALLABLE void sphere_box_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_3242f8a8 var_sphere,
    Geom_3242f8a8 var_box,
    wp::int32 var_worldid,
    wp::float32 var_margin,
    wp::float32 var_gap,
    wp::int32 var_condim,
    wp::vec_t<5, wp::float32> var_friction,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<2, wp::float32> var_solreffriction,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::vec_t<2, wp::int32> var_pairid,
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
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::int32> var_nacon_out)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32>* var_0;
    wp::vec_t<3, wp::float32>* var_1;
    const wp::int32 var_2 = 0;
    wp::float32 var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32>* var_5;
    wp::mat_t<3, 3, wp::float32>* var_6;
    wp::vec_t<3, wp::float32>* var_7;
    wp::float32 var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::mat_t<3, 3, wp::float32> var_13;
    wp::vec_t<3, wp::float32> var_14;
    const wp::int32 var_15 = 0;
    wp::mat_t<3, 3, wp::float32> var_16;
    wp::int32 var_17;
    //---------
    // forward
    // def sphere_box_wrapper(                                                                <L 1047>
    // dist, pos, normal = sphere_box(sphere.pos, sphere.size[0], box.pos, box.rot, box.size)       <L 1080>
    var_0 = &((var_sphere).pos);
    var_1 = &((var_sphere).size);
    var_4 = wp::load(var_1);
    var_3 = wp::extract(var_4, var_2);
    var_5 = &((var_box).pos);
    var_6 = &((var_box).rot);
    var_7 = &((var_box).size);
    var_11 = wp::load(var_0);
    var_12 = wp::load(var_5);
    var_13 = wp::load(var_6);
    var_14 = wp::load(var_7);
    sphere_box_0(var_11, var_3, var_12, var_13, var_14, var_8, var_9, var_10);
    // write_contact(                                                                         <L 1082>
    // naconmax_in,                                                                           <L 1083>
    // 0,                                                                                     <L 1084>
    // dist,                                                                                  <L 1085>
    // pos,                                                                                   <L 1086>
    // make_frame(normal),                                                                    <L 1087>
    var_16 = make_frame_0(var_10);
    // margin,                                                                                <L 1088>
    // gap,                                                                                   <L 1089>
    // condim,                                                                                <L 1090>
    // friction,                                                                              <L 1091>
    // solref,                                                                                <L 1092>
    // solreffriction,                                                                        <L 1093>
    // solimp,                                                                                <L 1094>
    // geoms,                                                                                 <L 1095>
    // pairid,                                                                                <L 1096>
    // worldid,                                                                               <L 1097>
    // contact_dist_out,                                                                      <L 1098>
    // contact_pos_out,                                                                       <L 1099>
    // contact_frame_out,                                                                     <L 1100>
    // contact_includemargin_out,                                                             <L 1101>
    // contact_friction_out,                                                                  <L 1102>
    // contact_solref_out,                                                                    <L 1103>
    // contact_solreffriction_out,                                                            <L 1104>
    // contact_solimp_out,                                                                    <L 1105>
    // contact_dim_out,                                                                       <L 1106>
    // contact_geom_out,                                                                      <L 1107>
    // contact_efc_address_out,                                                               <L 1108>
    // contact_worldid_out,                                                                   <L 1109>
    // contact_type_out,                                                                      <L 1110>
    // contact_geomcollisionid_out,                                                           <L 1111>
    // nacon_out,                                                                             <L 1112>
    var_17 = write_contact_0(var_naconmax_in, var_15, var_8, var_9, var_16, var_margin, var_gap, var_condim, var_friction, var_solref, var_solreffriction, var_solimp, var_geoms, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_nacon_out);
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_primitive_core.py:122
static CUDA_CALLABLE void capsule_capsule_0(
    wp::vec_t<3, wp::float32> var_cap1_pos,
    wp::vec_t<3, wp::float32> var_cap1_axis,
    wp::float32 var_cap1_radius,
    wp::float32 var_cap1_half_length,
    wp::vec_t<3, wp::float32> var_cap2_pos,
    wp::vec_t<3, wp::float32> var_cap2_axis,
    wp::float32 var_cap2_radius,
    wp::float32 var_cap2_half_length,
    wp::float32 var_margin,
    wp::vec_t<2, wp::float32> & ret_0,
    wp::mat_t<2, 3, wp::float32> & ret_1,
    wp::mat_t<2, 3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    const wp::float32 var_0 = INFINITY;
    const wp::float32 var_1 = INFINITY;
    const wp::float32 var_2 = INFINITY;
    const wp::float32 var_3 = INFINITY;
    wp::vec_t<2, wp::float32> var_4;
    wp::mat_t<2, 3, wp::float32> var_5;
    wp::mat_t<2, 3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    const wp::float32 var_21 = 1e-15;
    bool var_22;
    const wp::float32 var_23 = 1.0;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::float32 var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::float32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    const wp::float32 var_33 = 1.0;
    bool var_34;
    const wp::float32 var_35 = 1.0;
    wp::float32 var_36;
    wp::float32 var_37;
    wp::float32 var_38;
    wp::float32 var_39;
    const wp::float32 var_40 = -1.0;
    bool var_41;
    const wp::float32 var_42 = -1.0;
    wp::float32 var_43;
    wp::float32 var_44;
    wp::float32 var_45;
    wp::float32 var_46;
    wp::float32 var_47;
    wp::float32 var_48;
    const wp::float32 var_49 = 1.0;
    bool var_50;
    const wp::float32 var_51 = 1.0;
    wp::float32 var_52;
    wp::float32 var_53;
    const wp::float32 var_54 = -1.0;
    const wp::float32 var_55 = 1.0;
    wp::float32 var_56;
    wp::float32 var_57;
    wp::float32 var_58;
    const wp::float32 var_59 = -1.0;
    bool var_60;
    const wp::float32 var_61 = -1.0;
    wp::float32 var_62;
    wp::float32 var_63;
    const wp::float32 var_64 = -1.0;
    const wp::float32 var_65 = 1.0;
    wp::float32 var_66;
    wp::float32 var_67;
    wp::float32 var_68;
    wp::float32 var_69;
    wp::float32 var_70;
    wp::vec_t<3, wp::float32> var_71;
    wp::vec_t<3, wp::float32> var_72;
    wp::vec_t<3, wp::float32> var_73;
    wp::vec_t<3, wp::float32> var_74;
    wp::float32 var_75;
    wp::vec_t<3, wp::float32> var_76;
    wp::vec_t<3, wp::float32> var_77;
    bool var_78;
    const wp::int32 var_79 = 0;
    const wp::int32 var_80 = 0;
    const wp::int32 var_81 = 0;
    const wp::int32 var_82 = 0;
    wp::vec_t<3, wp::float32> var_83;
    wp::float32 var_84;
    wp::float32 var_85;
    const wp::float32 var_86 = -1.0;
    const wp::float32 var_87 = 1.0;
    wp::float32 var_88;
    wp::vec_t<3, wp::float32> var_89;
    wp::vec_t<3, wp::float32> var_90;
    wp::float32 var_91;
    wp::vec_t<3, wp::float32> var_92;
    wp::vec_t<3, wp::float32> var_93;
    bool var_94;
    const wp::int32 var_95 = 1;
    wp::int32 var_96;
    wp::int32 var_97;
    wp::vec_t<3, wp::float32> var_98;
    wp::float32 var_99;
    wp::float32 var_100;
    const wp::float32 var_101 = -1.0;
    const wp::float32 var_102 = 1.0;
    wp::float32 var_103;
    wp::vec_t<3, wp::float32> var_104;
    wp::vec_t<3, wp::float32> var_105;
    wp::float32 var_106;
    wp::vec_t<3, wp::float32> var_107;
    wp::vec_t<3, wp::float32> var_108;
    bool var_109;
    const wp::int32 var_110 = 1;
    wp::int32 var_111;
    wp::int32 var_112;
    const wp::int32 var_113 = 2;
    bool var_114;
    wp::vec_t<3, wp::float32> var_115;
    wp::float32 var_116;
    wp::float32 var_117;
    const wp::float32 var_118 = -1.0;
    const wp::float32 var_119 = 1.0;
    wp::float32 var_120;
    wp::vec_t<3, wp::float32> var_121;
    wp::vec_t<3, wp::float32> var_122;
    wp::float32 var_123;
    wp::vec_t<3, wp::float32> var_124;
    wp::vec_t<3, wp::float32> var_125;
    bool var_126;
    const wp::int32 var_127 = 1;
    wp::int32 var_128;
    wp::int32 var_129;
    wp::float32 var_130;
    wp::vec_t<3, wp::float32> var_131;
    wp::vec_t<3, wp::float32> var_132;
    wp::float32 var_133;
    wp::vec_t<3, wp::float32> var_134;
    wp::vec_t<3, wp::float32> var_135;
    wp::int32 var_136;
    const wp::int32 var_137 = 2;
    bool var_138;
    wp::vec_t<3, wp::float32> var_139;
    wp::float32 var_140;
    wp::float32 var_141;
    const wp::float32 var_142 = -1.0;
    const wp::float32 var_143 = 1.0;
    wp::float32 var_144;
    wp::vec_t<3, wp::float32> var_145;
    wp::vec_t<3, wp::float32> var_146;
    wp::float32 var_147;
    wp::vec_t<3, wp::float32> var_148;
    wp::vec_t<3, wp::float32> var_149;
    bool var_150;
    wp::float32 var_151;
    wp::vec_t<3, wp::float32> var_152;
    wp::vec_t<3, wp::float32> var_153;
    wp::float32 var_154;
    wp::vec_t<3, wp::float32> var_155;
    wp::vec_t<3, wp::float32> var_156;
    wp::float32 var_157;
    wp::float32 var_158;
    wp::vec_t<3, wp::float32> var_159;
    wp::vec_t<3, wp::float32> var_160;
    wp::float32 var_161;
    wp::vec_t<3, wp::float32> var_162;
    wp::vec_t<3, wp::float32> var_163;
    //---------
    // forward
    // def capsule_capsule(                                                                   <L 123>
    // contact_dist = wp.vec2(wp.inf, wp.inf)                                                 <L 153>
    var_4 = wp::vec_t<2, wp::float32>(var_1, var_3);
    // contact_pos = mat23f()                                                                 <L 154>
    var_5 = wp::mat_t<2, 3, wp::float32>();
    // contact_normal = mat23f()                                                              <L 155>
    var_6 = wp::mat_t<2, 3, wp::float32>();
    // axis1 = cap1_axis * cap1_half_length                                                   <L 158>
    var_7 = wp::mul(var_cap1_axis, var_cap1_half_length);
    // axis2 = cap2_axis * cap2_half_length                                                   <L 159>
    var_8 = wp::mul(var_cap2_axis, var_cap2_half_length);
    // dif = cap1_pos - cap2_pos                                                              <L 160>
    var_9 = wp::sub(var_cap1_pos, var_cap2_pos);
    // ma = wp.dot(axis1, axis1)                                                              <L 163>
    var_10 = wp::dot(var_7, var_7);
    // mb = -wp.dot(axis1, axis2)                                                             <L 164>
    var_11 = wp::dot(var_7, var_8);
    var_12 = wp::neg(var_11);
    // mc = wp.dot(axis2, axis2)                                                              <L 165>
    var_13 = wp::dot(var_8, var_8);
    // u = -wp.dot(axis1, dif)                                                                <L 166>
    var_14 = wp::dot(var_7, var_9);
    var_15 = wp::neg(var_14);
    // v = wp.dot(axis2, dif)                                                                 <L 167>
    var_16 = wp::dot(var_8, var_9);
    // det = ma * mc - mb * mb                                                                <L 168>
    var_17 = wp::mul(var_10, var_13);
    var_18 = wp::mul(var_12, var_12);
    var_19 = wp::sub(var_17, var_18);
    // if wp.abs(det) >= MJ_MINVAL:                                                           <L 171>
    var_20 = wp::abs(var_19);
    var_22 = (var_20 >= var_21);
    if (var_22) {
        // inv_det = 1.0 / det                                                                <L 172>
        var_24 = wp::div(var_23, var_19);
        // x1 = (mc * u - mb * v) * inv_det                                                   <L 173>
        var_25 = wp::mul(var_13, var_15);
        var_26 = wp::mul(var_12, var_16);
        var_27 = wp::sub(var_25, var_26);
        var_28 = wp::mul(var_27, var_24);
        // x2 = (ma * v - mb * u) * inv_det                                                   <L 174>
        var_29 = wp::mul(var_10, var_16);
        var_30 = wp::mul(var_12, var_15);
        var_31 = wp::sub(var_29, var_30);
        var_32 = wp::mul(var_31, var_24);
        // if x1 > 1.0:                                                                       <L 176>
        var_34 = (var_28 > var_33);
        if (var_34) {
            // x1 = 1.0                                                                       <L 177>
            // x2 = (v - mb) / mc                                                             <L 178>
            var_36 = wp::sub(var_16, var_12);
            var_37 = wp::div(var_36, var_13);
        }
        var_38 = wp::where(var_34, var_35, var_28);
        var_39 = wp::where(var_34, var_37, var_32);
        if (!var_34) {
            // elif x1 < -1.0:                                                                <L 179>
            var_41 = (var_38 < var_40);
            if (var_41) {
                // x1 = -1.0                                                                  <L 180>
                // x2 = (v + mb) / mc                                                         <L 181>
                var_43 = wp::add(var_16, var_12);
                var_44 = wp::div(var_43, var_13);
            }
            var_45 = wp::where(var_41, var_42, var_38);
            var_46 = wp::where(var_41, var_44, var_39);
        }
        var_47 = wp::where(var_34, var_38, var_45);
        var_48 = wp::where(var_34, var_39, var_46);
        // if x2 > 1.0:                                                                       <L 183>
        var_50 = (var_48 > var_49);
        if (var_50) {
            // x2 = 1.0                                                                       <L 184>
            // x1 = wp.clamp((u - mb) / ma, -1.0, 1.0)                                        <L 185>
            var_52 = wp::sub(var_15, var_12);
            var_53 = wp::div(var_52, var_10);
            var_56 = wp::clamp(var_53, var_54, var_55);
        }
        var_57 = wp::where(var_50, var_56, var_47);
        var_58 = wp::where(var_50, var_51, var_48);
        if (!var_50) {
            // elif x2 < -1.0:                                                                <L 186>
            var_60 = (var_58 < var_59);
            if (var_60) {
                // x2 = -1.0                                                                  <L 187>
                // x1 = wp.clamp((u + mb) / ma, -1.0, 1.0)                                    <L 188>
                var_62 = wp::add(var_15, var_12);
                var_63 = wp::div(var_62, var_10);
                var_66 = wp::clamp(var_63, var_64, var_65);
            }
            var_67 = wp::where(var_60, var_66, var_57);
            var_68 = wp::where(var_60, var_61, var_58);
        }
        var_69 = wp::where(var_50, var_57, var_67);
        var_70 = wp::where(var_50, var_58, var_68);
        // vec1 = cap1_pos + axis1 * x1                                                       <L 191>
        var_71 = wp::mul(var_7, var_69);
        var_72 = wp::add(var_cap1_pos, var_71);
        // vec2 = cap2_pos + axis2 * x2                                                       <L 192>
        var_73 = wp::mul(var_8, var_70);
        var_74 = wp::add(var_cap2_pos, var_73);
        // dist, pos, normal = sphere_sphere(vec1, cap1_radius, vec2, cap2_radius)            <L 194>
        sphere_sphere_0(var_72, var_cap1_radius, var_74, var_cap2_radius, var_75, var_76, var_77);
        // if dist <= margin:                                                                 <L 195>
        var_78 = (var_75 <= var_margin);
        if (var_78) {
            // contact_dist[0] = dist                                                         <L 196>
            wp::assign_inplace(var_4, var_79, var_75);
            // contact_pos[0] = pos                                                           <L 197>
            wp::assign_inplace(var_5, var_80, var_76);
            // contact_normal[0] = normal                                                     <L 198>
            wp::assign_inplace(var_6, var_81, var_77);
        }
    }
    if (!var_22) {
        // contact_count = 0                                                                  <L 202>
        // vec1 = cap1_pos + axis1                                                            <L 205>
        var_83 = wp::add(var_cap1_pos, var_7);
        // x2 = wp.clamp((v - mb) / mc, -1.0, 1.0)                                            <L 206>
        var_84 = wp::sub(var_16, var_12);
        var_85 = wp::div(var_84, var_13);
        var_88 = wp::clamp(var_85, var_86, var_87);
        // vec2 = cap2_pos + axis2 * x2                                                       <L 207>
        var_89 = wp::mul(var_8, var_88);
        var_90 = wp::add(var_cap2_pos, var_89);
        // dist, pos, normal = sphere_sphere(vec1, cap1_radius, vec2, cap2_radius)            <L 208>
        sphere_sphere_0(var_83, var_cap1_radius, var_90, var_cap2_radius, var_91, var_92, var_93);
        // if dist <= margin:                                                                 <L 209>
        var_94 = (var_91 <= var_margin);
        if (var_94) {
            // contact_dist[contact_count] = dist                                             <L 210>
            wp::assign_inplace(var_4, var_82, var_91);
            // contact_pos[contact_count] = pos                                               <L 211>
            wp::assign_inplace(var_5, var_82, var_92);
            // contact_normal[contact_count] = normal                                         <L 212>
            wp::assign_inplace(var_6, var_82, var_93);
            // contact_count += 1                                                             <L 213>
            var_96 = wp::add(var_82, var_95);
        }
        var_97 = wp::where(var_94, var_96, var_82);
        // vec1 = cap1_pos - axis1                                                            <L 216>
        var_98 = wp::sub(var_cap1_pos, var_7);
        // x2 = wp.clamp((v + mb) / mc, -1.0, 1.0)                                            <L 217>
        var_99 = wp::add(var_16, var_12);
        var_100 = wp::div(var_99, var_13);
        var_103 = wp::clamp(var_100, var_101, var_102);
        // vec2 = cap2_pos + axis2 * x2                                                       <L 218>
        var_104 = wp::mul(var_8, var_103);
        var_105 = wp::add(var_cap2_pos, var_104);
        // dist, pos, normal = sphere_sphere(vec1, cap1_radius, vec2, cap2_radius)            <L 219>
        sphere_sphere_0(var_98, var_cap1_radius, var_105, var_cap2_radius, var_106, var_107, var_108);
        // if dist <= margin:                                                                 <L 220>
        var_109 = (var_106 <= var_margin);
        if (var_109) {
            // contact_dist[contact_count] = dist                                             <L 221>
            wp::assign_inplace(var_4, var_97, var_106);
            // contact_pos[contact_count] = pos                                               <L 222>
            wp::assign_inplace(var_5, var_97, var_107);
            // contact_normal[contact_count] = normal                                         <L 223>
            wp::assign_inplace(var_6, var_97, var_108);
            // contact_count += 1                                                             <L 224>
            var_111 = wp::add(var_97, var_110);
        }
        var_112 = wp::where(var_109, var_111, var_97);
        // if contact_count < 2:                                                              <L 227>
        var_114 = (var_112 < var_113);
        if (var_114) {
            // vec2 = cap2_pos + axis2                                                        <L 228>
            var_115 = wp::add(var_cap2_pos, var_8);
            // x1 = wp.clamp((u - mb) / ma, -1.0, 1.0)                                        <L 229>
            var_116 = wp::sub(var_15, var_12);
            var_117 = wp::div(var_116, var_10);
            var_120 = wp::clamp(var_117, var_118, var_119);
            // vec1 = cap1_pos + axis1 * x1                                                   <L 230>
            var_121 = wp::mul(var_7, var_120);
            var_122 = wp::add(var_cap1_pos, var_121);
            // dist, pos, normal = sphere_sphere(vec1, cap1_radius, vec2, cap2_radius)        <L 231>
            sphere_sphere_0(var_122, var_cap1_radius, var_115, var_cap2_radius, var_123, var_124, var_125);
            // if dist <= margin:                                                             <L 232>
            var_126 = (var_123 <= var_margin);
            if (var_126) {
                // contact_dist[contact_count] = dist                                         <L 233>
                wp::assign_inplace(var_4, var_112, var_123);
                // contact_pos[contact_count] = pos                                           <L 234>
                wp::assign_inplace(var_5, var_112, var_124);
                // contact_normal[contact_count] = normal                                     <L 235>
                wp::assign_inplace(var_6, var_112, var_125);
                // contact_count += 1                                                         <L 236>
                var_128 = wp::add(var_112, var_127);
            }
            var_129 = wp::where(var_126, var_128, var_112);
        }
        var_130 = wp::where(var_114, var_120, var_69);
        var_131 = wp::where(var_114, var_122, var_98);
        var_132 = wp::where(var_114, var_115, var_105);
        var_133 = wp::where(var_114, var_123, var_106);
        var_134 = wp::where(var_114, var_124, var_107);
        var_135 = wp::where(var_114, var_125, var_108);
        var_136 = wp::where(var_114, var_129, var_112);
        // if contact_count < 2:                                                              <L 239>
        var_138 = (var_136 < var_137);
        if (var_138) {
            // vec2 = cap2_pos - axis2                                                        <L 240>
            var_139 = wp::sub(var_cap2_pos, var_8);
            // x1 = wp.clamp((u + mb) / ma, -1.0, 1.0)                                        <L 241>
            var_140 = wp::add(var_15, var_12);
            var_141 = wp::div(var_140, var_10);
            var_144 = wp::clamp(var_141, var_142, var_143);
            // vec1 = cap1_pos + axis1 * x1                                                   <L 242>
            var_145 = wp::mul(var_7, var_144);
            var_146 = wp::add(var_cap1_pos, var_145);
            // dist, pos, normal = sphere_sphere(vec1, cap1_radius, vec2, cap2_radius)        <L 243>
            sphere_sphere_0(var_146, var_cap1_radius, var_139, var_cap2_radius, var_147, var_148, var_149);
            // if dist <= margin:                                                             <L 244>
            var_150 = (var_147 <= var_margin);
            if (var_150) {
                // contact_dist[contact_count] = dist                                         <L 245>
                wp::assign_inplace(var_4, var_136, var_147);
                // contact_pos[contact_count] = pos                                           <L 246>
                wp::assign_inplace(var_5, var_136, var_148);
                // contact_normal[contact_count] = normal                                     <L 247>
                wp::assign_inplace(var_6, var_136, var_149);
            }
        }
        var_151 = wp::where(var_138, var_144, var_130);
        var_152 = wp::where(var_138, var_146, var_131);
        var_153 = wp::where(var_138, var_139, var_132);
        var_154 = wp::where(var_138, var_147, var_133);
        var_155 = wp::where(var_138, var_148, var_134);
        var_156 = wp::where(var_138, var_149, var_135);
    }
    var_157 = wp::where(var_22, var_69, var_151);
    var_158 = wp::where(var_22, var_70, var_103);
    var_159 = wp::where(var_22, var_72, var_152);
    var_160 = wp::where(var_22, var_74, var_153);
    var_161 = wp::where(var_22, var_75, var_154);
    var_162 = wp::where(var_22, var_76, var_155);
    var_163 = wp::where(var_22, var_77, var_156);
    // return contact_dist, contact_pos, contact_normal                                       <L 249>
    ret_0 = var_4;
    ret_1 = var_5;
    ret_2 = var_6;
    return;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_primitive.py:497
static CUDA_CALLABLE void capsule_capsule_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_3242f8a8 var_cap1,
    Geom_3242f8a8 var_cap2,
    wp::int32 var_worldid,
    wp::float32 var_margin,
    wp::float32 var_gap,
    wp::int32 var_condim,
    wp::vec_t<5, wp::float32> var_friction,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<2, wp::float32> var_solreffriction,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::vec_t<2, wp::int32> var_pairid,
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
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::int32> var_nacon_out)
{
    //---------
    // primal vars
    wp::mat_t<3, 3, wp::float32>* var_0;
    const wp::int32 var_1 = 0;
    const wp::int32 var_2 = 2;
    wp::float32 var_3;
    wp::mat_t<3, 3, wp::float32> var_4;
    wp::mat_t<3, 3, wp::float32>* var_5;
    const wp::int32 var_6 = 1;
    const wp::int32 var_7 = 2;
    wp::float32 var_8;
    wp::mat_t<3, 3, wp::float32> var_9;
    wp::mat_t<3, 3, wp::float32>* var_10;
    const wp::int32 var_11 = 2;
    const wp::int32 var_12 = 2;
    wp::float32 var_13;
    wp::mat_t<3, 3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::mat_t<3, 3, wp::float32>* var_16;
    const wp::int32 var_17 = 0;
    const wp::int32 var_18 = 2;
    wp::float32 var_19;
    wp::mat_t<3, 3, wp::float32> var_20;
    wp::mat_t<3, 3, wp::float32>* var_21;
    const wp::int32 var_22 = 1;
    const wp::int32 var_23 = 2;
    wp::float32 var_24;
    wp::mat_t<3, 3, wp::float32> var_25;
    wp::mat_t<3, 3, wp::float32>* var_26;
    const wp::int32 var_27 = 2;
    const wp::int32 var_28 = 2;
    wp::float32 var_29;
    wp::mat_t<3, 3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<3, wp::float32>* var_32;
    wp::vec_t<3, wp::float32>* var_33;
    const wp::int32 var_34 = 0;
    wp::float32 var_35;
    wp::vec_t<3, wp::float32> var_36;
    wp::vec_t<3, wp::float32>* var_37;
    const wp::int32 var_38 = 1;
    wp::float32 var_39;
    wp::vec_t<3, wp::float32> var_40;
    wp::vec_t<3, wp::float32>* var_41;
    wp::vec_t<3, wp::float32>* var_42;
    const wp::int32 var_43 = 0;
    wp::float32 var_44;
    wp::vec_t<3, wp::float32> var_45;
    wp::vec_t<3, wp::float32>* var_46;
    const wp::int32 var_47 = 1;
    wp::float32 var_48;
    wp::vec_t<3, wp::float32> var_49;
    wp::vec_t<2, wp::float32> var_50;
    wp::mat_t<2, 3, wp::float32> var_51;
    wp::mat_t<2, 3, wp::float32> var_52;
    wp::vec_t<3, wp::float32> var_53;
    wp::vec_t<3, wp::float32> var_54;
    const wp::int32 var_55 = 0;
    wp::float32 var_56;
    const wp::int32 var_57 = 0;
    wp::float32 var_58;
    const wp::int32 var_59 = 1;
    wp::float32 var_60;
    const wp::int32 var_61 = 2;
    wp::float32 var_62;
    wp::vec_t<3, wp::float32> var_63;
    const wp::int32 var_64 = 0;
    wp::float32 var_65;
    const wp::int32 var_66 = 1;
    wp::float32 var_67;
    const wp::int32 var_68 = 2;
    wp::float32 var_69;
    wp::vec_t<3, wp::float32> var_70;
    wp::mat_t<3, 3, wp::float32> var_71;
    wp::int32 var_72;
    const wp::int32 var_73 = 1;
    wp::float32 var_74;
    const wp::int32 var_75 = 0;
    wp::float32 var_76;
    const wp::int32 var_77 = 1;
    wp::float32 var_78;
    const wp::int32 var_79 = 2;
    wp::float32 var_80;
    wp::vec_t<3, wp::float32> var_81;
    const wp::int32 var_82 = 0;
    wp::float32 var_83;
    const wp::int32 var_84 = 1;
    wp::float32 var_85;
    const wp::int32 var_86 = 2;
    wp::float32 var_87;
    wp::vec_t<3, wp::float32> var_88;
    wp::mat_t<3, 3, wp::float32> var_89;
    wp::int32 var_90;
    //---------
    // forward
    // def capsule_capsule_wrapper(                                                           <L 498>
    // cap1_axis = wp.vec3(cap1.rot[0, 2], cap1.rot[1, 2], cap1.rot[2, 2])                    <L 533>
    var_0 = &((var_cap1).rot);
    var_4 = wp::load(var_0);
    var_3 = wp::extract(var_4, var_1, var_2);
    var_5 = &((var_cap1).rot);
    var_9 = wp::load(var_5);
    var_8 = wp::extract(var_9, var_6, var_7);
    var_10 = &((var_cap1).rot);
    var_14 = wp::load(var_10);
    var_13 = wp::extract(var_14, var_11, var_12);
    var_15 = wp::vec_t<3, wp::float32>(var_3, var_8, var_13);
    // cap2_axis = wp.vec3(cap2.rot[0, 2], cap2.rot[1, 2], cap2.rot[2, 2])                    <L 534>
    var_16 = &((var_cap2).rot);
    var_20 = wp::load(var_16);
    var_19 = wp::extract(var_20, var_17, var_18);
    var_21 = &((var_cap2).rot);
    var_25 = wp::load(var_21);
    var_24 = wp::extract(var_25, var_22, var_23);
    var_26 = &((var_cap2).rot);
    var_30 = wp::load(var_26);
    var_29 = wp::extract(var_30, var_27, var_28);
    var_31 = wp::vec_t<3, wp::float32>(var_19, var_24, var_29);
    // dist, pos, normal = capsule_capsule(                                                   <L 536>
    // cap1.pos,                                                                              <L 537>
    var_32 = &((var_cap1).pos);
    // cap1_axis,                                                                             <L 538>
    // cap1.size[0],  # radius1                                                               <L 539>
    var_33 = &((var_cap1).size);
    var_36 = wp::load(var_33);
    var_35 = wp::extract(var_36, var_34);
    // cap1.size[1],  # half_length1                                                          <L 540>
    var_37 = &((var_cap1).size);
    var_40 = wp::load(var_37);
    var_39 = wp::extract(var_40, var_38);
    // cap2.pos,                                                                              <L 541>
    var_41 = &((var_cap2).pos);
    // cap2_axis,                                                                             <L 542>
    // cap2.size[0],  # radius2                                                               <L 543>
    var_42 = &((var_cap2).size);
    var_45 = wp::load(var_42);
    var_44 = wp::extract(var_45, var_43);
    // cap2.size[1],  # half_length2                                                          <L 544>
    var_46 = &((var_cap2).size);
    var_49 = wp::load(var_46);
    var_48 = wp::extract(var_49, var_47);
    // margin,                                                                                <L 545>
    var_53 = wp::load(var_32);
    var_54 = wp::load(var_41);
    capsule_capsule_0(var_53, var_15, var_35, var_39, var_54, var_31, var_44, var_48, var_margin, var_50, var_51, var_52);
    // for i in range(2):                                                                     <L 548>
    // write_contact(                                                                         <L 549>
    // naconmax_in,                                                                           <L 550>
    // i,                                                                                     <L 551>
    // dist[i],                                                                               <L 552>
    var_56 = wp::extract(var_50, var_55);
    // wp.vec3(pos[i, 0], pos[i, 1], pos[i, 2]),                                              <L 553>
    var_58 = wp::extract(var_51, var_55, var_57);
    var_60 = wp::extract(var_51, var_55, var_59);
    var_62 = wp::extract(var_51, var_55, var_61);
    var_63 = wp::vec_t<3, wp::float32>(var_58, var_60, var_62);
    // make_frame(wp.vec3(normal[i, 0], normal[i, 1], normal[i, 2])),                         <L 554>
    var_65 = wp::extract(var_52, var_55, var_64);
    var_67 = wp::extract(var_52, var_55, var_66);
    var_69 = wp::extract(var_52, var_55, var_68);
    var_70 = wp::vec_t<3, wp::float32>(var_65, var_67, var_69);
    var_71 = make_frame_0(var_70);
    // margin,                                                                                <L 555>
    // gap,                                                                                   <L 556>
    // condim,                                                                                <L 557>
    // friction,                                                                              <L 558>
    // solref,                                                                                <L 559>
    // solreffriction,                                                                        <L 560>
    // solimp,                                                                                <L 561>
    // geoms,                                                                                 <L 562>
    // pairid,                                                                                <L 563>
    // worldid,                                                                               <L 564>
    // contact_dist_out,                                                                      <L 565>
    // contact_pos_out,                                                                       <L 566>
    // contact_frame_out,                                                                     <L 567>
    // contact_includemargin_out,                                                             <L 568>
    // contact_friction_out,                                                                  <L 569>
    // contact_solref_out,                                                                    <L 570>
    // contact_solreffriction_out,                                                            <L 571>
    // contact_solimp_out,                                                                    <L 572>
    // contact_dim_out,                                                                       <L 573>
    // contact_geom_out,                                                                      <L 574>
    // contact_efc_address_out,                                                               <L 575>
    // contact_worldid_out,                                                                   <L 576>
    // contact_type_out,                                                                      <L 577>
    // contact_geomcollisionid_out,                                                           <L 578>
    // nacon_out,                                                                             <L 579>
    var_72 = write_contact_0(var_naconmax_in, var_55, var_56, var_63, var_71, var_margin, var_gap, var_condim, var_friction, var_solref, var_solreffriction, var_solimp, var_geoms, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_nacon_out);
    // write_contact(                                                                         <L 549>
    // naconmax_in,                                                                           <L 550>
    // i,                                                                                     <L 551>
    // dist[i],                                                                               <L 552>
    var_74 = wp::extract(var_50, var_73);
    // wp.vec3(pos[i, 0], pos[i, 1], pos[i, 2]),                                              <L 553>
    var_76 = wp::extract(var_51, var_73, var_75);
    var_78 = wp::extract(var_51, var_73, var_77);
    var_80 = wp::extract(var_51, var_73, var_79);
    var_81 = wp::vec_t<3, wp::float32>(var_76, var_78, var_80);
    // make_frame(wp.vec3(normal[i, 0], normal[i, 1], normal[i, 2])),                         <L 554>
    var_83 = wp::extract(var_52, var_73, var_82);
    var_85 = wp::extract(var_52, var_73, var_84);
    var_87 = wp::extract(var_52, var_73, var_86);
    var_88 = wp::vec_t<3, wp::float32>(var_83, var_85, var_87);
    var_89 = make_frame_0(var_88);
    // margin,                                                                                <L 555>
    // gap,                                                                                   <L 556>
    // condim,                                                                                <L 557>
    // friction,                                                                              <L 558>
    // solref,                                                                                <L 559>
    // solreffriction,                                                                        <L 560>
    // solimp,                                                                                <L 561>
    // geoms,                                                                                 <L 562>
    // pairid,                                                                                <L 563>
    // worldid,                                                                               <L 564>
    // contact_dist_out,                                                                      <L 565>
    // contact_pos_out,                                                                       <L 566>
    // contact_frame_out,                                                                     <L 567>
    // contact_includemargin_out,                                                             <L 568>
    // contact_friction_out,                                                                  <L 569>
    // contact_solref_out,                                                                    <L 570>
    // contact_solreffriction_out,                                                            <L 571>
    // contact_solimp_out,                                                                    <L 572>
    // contact_dim_out,                                                                       <L 573>
    // contact_geom_out,                                                                      <L 574>
    // contact_efc_address_out,                                                               <L 575>
    // contact_worldid_out,                                                                   <L 576>
    // contact_type_out,                                                                      <L 577>
    // contact_geomcollisionid_out,                                                           <L 578>
    // nacon_out,                                                                             <L 579>
    var_90 = write_contact_0(var_naconmax_in, var_73, var_74, var_81, var_89, var_margin, var_gap, var_condim, var_friction, var_solref, var_solreffriction, var_solimp, var_geoms, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_nacon_out);
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_primitive_core.py:1098
static CUDA_CALLABLE void capsule_box_0(
    wp::vec_t<3, wp::float32> var_capsule_pos,
    wp::vec_t<3, wp::float32> var_capsule_axis,
    wp::float32 var_capsule_radius,
    wp::float32 var_capsule_half_length,
    wp::vec_t<3, wp::float32> var_box_pos,
    wp::mat_t<3, 3, wp::float32> var_box_rot,
    wp::vec_t<3, wp::float32> var_box_size,
    wp::vec_t<2, wp::float32> & ret_0,
    wp::mat_t<2, 3, wp::float32> & ret_1,
    wp::mat_t<2, 3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::mat_t<3, 3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    const wp::int32 var_5 = 0;
    wp::float32 var_6;
    const wp::float32 var_7 = 0.0;
    bool var_8;
    wp::int32 var_9;
    const wp::int32 var_10 = 2;
    const wp::int32 var_11 = 1;
    wp::float32 var_12;
    const wp::float32 var_13 = 0.0;
    bool var_14;
    wp::int32 var_15;
    wp::int32 var_16;
    wp::int32 var_17;
    const wp::int32 var_18 = 4;
    const wp::int32 var_19 = 2;
    wp::float32 var_20;
    const wp::float32 var_21 = 0.0;
    bool var_22;
    wp::int32 var_23;
    wp::int32 var_24;
    wp::int32 var_25;
    const wp::float32 var_26 = 1e+32;
    wp::float32 var_27;
    const wp::int32 var_28 = -12;
    wp::float32 var_29;
    const wp::int32 var_30 = -4;
    wp::int32 var_31;
    const wp::int32 var_32 = -12;
    wp::int32 var_33;
    const wp::int32 var_34 = -1;
    const wp::int32 var_35 = 2;
    const wp::int32 var_36 = 2;
    wp::range_t var_37;
    wp::int32 var_38;
    wp::float32 var_39;
    wp::vec_t<3, wp::float32> var_40;
    wp::vec_t<3, wp::float32> var_41;
    wp::vec_t<3, wp::float32> var_42;
    const wp::int32 var_43 = 0;
    wp::int32 var_44;
    const wp::int32 var_45 = -1;
    wp::int32 var_46;
    const wp::int32 var_47 = 0;
    wp::float32 var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    bool var_51;
    const wp::int32 var_52 = 1;
    wp::int32 var_53;
    wp::int32 var_54;
    wp::float32 var_55;
    wp::float32 var_56;
    wp::int32 var_57;
    wp::int32 var_58;
    wp::float32 var_59;
    wp::float32 var_60;
    bool var_61;
    const wp::int32 var_62 = 1;
    wp::int32 var_63;
    wp::int32 var_64;
    wp::float32 var_65;
    wp::int32 var_66;
    wp::int32 var_67;
    wp::int32 var_68;
    wp::int32 var_69;
    const wp::int32 var_70 = 1;
    wp::float32 var_71;
    wp::float32 var_72;
    wp::float32 var_73;
    bool var_74;
    const wp::int32 var_75 = 1;
    wp::int32 var_76;
    wp::int32 var_77;
    wp::float32 var_78;
    wp::float32 var_79;
    wp::int32 var_80;
    wp::int32 var_81;
    wp::float32 var_82;
    wp::float32 var_83;
    bool var_84;
    const wp::int32 var_85 = 1;
    wp::int32 var_86;
    wp::int32 var_87;
    wp::float32 var_88;
    wp::int32 var_89;
    wp::int32 var_90;
    wp::int32 var_91;
    wp::int32 var_92;
    const wp::int32 var_93 = 2;
    wp::float32 var_94;
    wp::float32 var_95;
    wp::float32 var_96;
    bool var_97;
    const wp::int32 var_98 = 1;
    wp::int32 var_99;
    wp::int32 var_100;
    wp::float32 var_101;
    wp::float32 var_102;
    wp::int32 var_103;
    wp::int32 var_104;
    wp::float32 var_105;
    wp::float32 var_106;
    bool var_107;
    const wp::int32 var_108 = 1;
    wp::int32 var_109;
    wp::int32 var_110;
    wp::float32 var_111;
    wp::int32 var_112;
    wp::int32 var_113;
    wp::int32 var_114;
    wp::int32 var_115;
    const wp::int32 var_116 = 1;
    bool var_117;
    wp::vec_t<3, wp::float32> var_118;
    wp::float32 var_119;
    bool var_120;
    wp::float32 var_121;
    wp::float32 var_122;
    const wp::int32 var_123 = -2;
    wp::int32 var_124;
    wp::int32 var_125;
    wp::float32 var_126;
    wp::float32 var_127;
    wp::int32 var_128;
    wp::int32 var_129;
    const wp::int32 var_130 = -123;
    wp::int32 var_131;
    const wp::int32 var_132 = -123;
    wp::int32 var_133;
    const wp::float32 var_134 = 0.0;
    wp::float32 var_135;
    const wp::int32 var_136 = 0;
    const wp::int32 var_137 = 3;
    wp::range_t var_138;
    wp::int32 var_139;
    const wp::int32 var_140 = 1;
    wp::int32 var_141;
    wp::int32 var_142;
    const wp::int32 var_143 = 0;
    bool var_144;
    const wp::int32 var_145 = -123;
    wp::int32 var_146;
    const wp::int32 var_147 = 1;
    wp::int32 var_148;
    const wp::float32 var_149 = 1.0;
    const wp::float32 var_150 = -1.0;
    wp::float32 var_151;
    const wp::int32 var_152 = 2;
    wp::int32 var_153;
    const wp::float32 var_154 = 1.0;
    const wp::float32 var_155 = -1.0;
    wp::float32 var_156;
    const wp::int32 var_157 = 4;
    wp::int32 var_158;
    const wp::float32 var_159 = 1.0;
    const wp::float32 var_160 = -1.0;
    wp::float32 var_161;
    wp::vec_t<3, wp::float32> var_162;
    wp::vec_t<3, wp::float32> var_163;
    const wp::float32 var_164 = 0.0;
    wp::vec_t<3, wp::float32> var_165;
    wp::float32 var_166;
    wp::float32 var_167;
    wp::float32 var_168;
    wp::float32 var_169;
    wp::float32 var_170;
    wp::float32 var_171;
    wp::float32 var_172;
    wp::float32 var_173;
    wp::float32 var_174;
    wp::float32 var_175;
    wp::float32 var_176;
    wp::float32 var_177;
    wp::float32 var_178;
    wp::float32 var_179;
    wp::float32 var_180;
    wp::float32 var_181;
    wp::float32 var_182;
    const wp::float32 var_183 = 1e-15;
    bool var_184;
    const wp::float32 var_185 = 1.0;
    wp::float32 var_186;
    wp::float32 var_187;
    wp::float32 var_188;
    wp::float32 var_189;
    wp::float32 var_190;
    wp::float32 var_191;
    wp::float32 var_192;
    wp::float32 var_193;
    wp::float32 var_194;
    wp::float32 var_195;
    wp::float32 var_196;
    const wp::int32 var_197 = 1;
    wp::int32 var_198;
    const wp::int32 var_199 = 1;
    wp::int32 var_200;
    const wp::int32 var_201 = 1;
    bool var_202;
    const wp::float32 var_203 = 1.0;
    const wp::int32 var_204 = 2;
    wp::float32 var_205;
    wp::float32 var_206;
    wp::float32 var_207;
    wp::float32 var_208;
    wp::int32 var_209;
    const wp::int32 var_210 = -1;
    bool var_211;
    const wp::float32 var_212 = -1.0;
    const wp::int32 var_213 = 0;
    wp::float32 var_214;
    wp::float32 var_215;
    wp::float32 var_216;
    wp::float32 var_217;
    wp::int32 var_218;
    wp::float32 var_219;
    wp::float32 var_220;
    wp::int32 var_221;
    const wp::float32 var_222 = 1.0;
    bool var_223;
    bool var_224;
    const wp::float32 var_225 = -1.0;
    bool var_226;
    const wp::float32 var_227 = 1.0;
    const wp::int32 var_228 = 2;
    wp::float32 var_229;
    wp::float32 var_230;
    wp::float32 var_231;
    wp::float32 var_232;
    wp::int32 var_233;
    const wp::float32 var_234 = -1.0;
    const wp::int32 var_235 = 0;
    wp::float32 var_236;
    wp::float32 var_237;
    wp::float32 var_238;
    wp::float32 var_239;
    wp::int32 var_240;
    const wp::int32 var_241 = 1;
    bool var_242;
    const wp::float32 var_243 = 1.0;
    const wp::int32 var_244 = 2;
    wp::float32 var_245;
    wp::int32 var_246;
    const wp::int32 var_247 = -1;
    bool var_248;
    const wp::float32 var_249 = -1.0;
    const wp::int32 var_250 = 0;
    wp::float32 var_251;
    wp::int32 var_252;
    wp::float32 var_253;
    wp::int32 var_254;
    wp::float32 var_255;
    wp::float32 var_256;
    wp::int32 var_257;
    wp::int32 var_258;
    wp::vec_t<3, wp::float32> var_259;
    wp::vec_t<3, wp::float32> var_260;
    wp::float32 var_261;
    wp::float32 var_262;
    const wp::int32 var_263 = 3;
    wp::int32 var_264;
    wp::int32 var_265;
    wp::float32 var_266;
    wp::float32 var_267;
    bool var_268;
    wp::float32 var_269;
    wp::float32 var_270;
    wp::float32 var_271;
    const wp::int32 var_272 = 6;
    wp::int32 var_273;
    const wp::int32 var_274 = 1;
    wp::int32 var_275;
    wp::int32 var_276;
    wp::int32 var_277;
    wp::int32 var_278;
    wp::int32 var_279;
    wp::float32 var_280;
    wp::float32 var_281;
    wp::int32 var_282;
    wp::int32 var_283;
    wp::int32 var_284;
    wp::float32 var_285;
    wp::int32 var_286;
    const wp::int32 var_287 = 1;
    const wp::int32 var_288 = 3;
    wp::range_t var_289;
    wp::int32 var_290;
    const wp::int32 var_291 = 1;
    wp::int32 var_292;
    wp::int32 var_293;
    const wp::int32 var_294 = 0;
    bool var_295;
    const wp::int32 var_296 = -123;
    wp::int32 var_297;
    const wp::int32 var_298 = 1;
    wp::int32 var_299;
    const wp::float32 var_300 = 1.0;
    const wp::float32 var_301 = -1.0;
    wp::float32 var_302;
    const wp::int32 var_303 = 2;
    wp::int32 var_304;
    const wp::float32 var_305 = 1.0;
    const wp::float32 var_306 = -1.0;
    wp::float32 var_307;
    const wp::int32 var_308 = 4;
    wp::int32 var_309;
    const wp::float32 var_310 = 1.0;
    const wp::float32 var_311 = -1.0;
    wp::float32 var_312;
    wp::vec_t<3, wp::float32> var_313;
    wp::vec_t<3, wp::float32> var_314;
    const wp::float32 var_315 = 0.0;
    wp::vec_t<3, wp::float32> var_316;
    wp::float32 var_317;
    wp::float32 var_318;
    wp::float32 var_319;
    wp::float32 var_320;
    wp::float32 var_321;
    wp::float32 var_322;
    wp::float32 var_323;
    wp::float32 var_324;
    wp::float32 var_325;
    wp::float32 var_326;
    wp::float32 var_327;
    wp::float32 var_328;
    wp::float32 var_329;
    wp::float32 var_330;
    wp::float32 var_331;
    wp::float32 var_332;
    wp::float32 var_333;
    bool var_334;
    wp::int32 var_335;
    wp::vec_t<3, wp::float32> var_336;
    wp::vec_t<3, wp::float32> var_337;
    wp::float32 var_338;
    wp::float32 var_339;
    wp::float32 var_340;
    wp::float32 var_341;
    wp::float32 var_342;
    wp::float32 var_343;
    const wp::float32 var_344 = 1.0;
    wp::float32 var_345;
    wp::float32 var_346;
    wp::float32 var_347;
    wp::float32 var_348;
    wp::float32 var_349;
    wp::float32 var_350;
    wp::float32 var_351;
    wp::float32 var_352;
    wp::float32 var_353;
    wp::float32 var_354;
    wp::float32 var_355;
    const wp::int32 var_356 = 1;
    wp::int32 var_357;
    const wp::int32 var_358 = 1;
    wp::int32 var_359;
    const wp::int32 var_360 = 1;
    bool var_361;
    const wp::float32 var_362 = 1.0;
    const wp::int32 var_363 = 2;
    wp::float32 var_364;
    wp::float32 var_365;
    wp::float32 var_366;
    wp::float32 var_367;
    wp::int32 var_368;
    const wp::int32 var_369 = -1;
    bool var_370;
    const wp::float32 var_371 = -1.0;
    const wp::int32 var_372 = 0;
    wp::float32 var_373;
    wp::float32 var_374;
    wp::float32 var_375;
    wp::float32 var_376;
    wp::int32 var_377;
    wp::float32 var_378;
    wp::float32 var_379;
    wp::int32 var_380;
    const wp::float32 var_381 = 1.0;
    bool var_382;
    bool var_383;
    const wp::float32 var_384 = -1.0;
    bool var_385;
    const wp::float32 var_386 = 1.0;
    const wp::int32 var_387 = 2;
    wp::float32 var_388;
    wp::float32 var_389;
    wp::float32 var_390;
    wp::float32 var_391;
    wp::int32 var_392;
    const wp::float32 var_393 = -1.0;
    const wp::int32 var_394 = 0;
    wp::float32 var_395;
    wp::float32 var_396;
    wp::float32 var_397;
    wp::float32 var_398;
    wp::int32 var_399;
    const wp::int32 var_400 = 1;
    bool var_401;
    const wp::float32 var_402 = 1.0;
    const wp::int32 var_403 = 2;
    wp::float32 var_404;
    wp::int32 var_405;
    const wp::int32 var_406 = -1;
    bool var_407;
    const wp::float32 var_408 = -1.0;
    const wp::int32 var_409 = 0;
    wp::float32 var_410;
    wp::int32 var_411;
    wp::float32 var_412;
    wp::int32 var_413;
    wp::float32 var_414;
    wp::float32 var_415;
    wp::int32 var_416;
    wp::int32 var_417;
    wp::vec_t<3, wp::float32> var_418;
    wp::vec_t<3, wp::float32> var_419;
    wp::float32 var_420;
    wp::float32 var_421;
    const wp::int32 var_422 = 3;
    wp::int32 var_423;
    wp::int32 var_424;
    wp::float32 var_425;
    wp::float32 var_426;
    bool var_427;
    wp::float32 var_428;
    wp::float32 var_429;
    wp::float32 var_430;
    const wp::int32 var_431 = 6;
    wp::int32 var_432;
    const wp::int32 var_433 = 1;
    wp::int32 var_434;
    wp::int32 var_435;
    wp::int32 var_436;
    wp::int32 var_437;
    wp::int32 var_438;
    wp::float32 var_439;
    wp::float32 var_440;
    wp::int32 var_441;
    wp::int32 var_442;
    wp::int32 var_443;
    wp::float32 var_444;
    wp::int32 var_445;
    const wp::int32 var_446 = 2;
    const wp::int32 var_447 = 3;
    wp::range_t var_448;
    wp::int32 var_449;
    const wp::int32 var_450 = 1;
    wp::int32 var_451;
    wp::int32 var_452;
    const wp::int32 var_453 = 0;
    bool var_454;
    const wp::int32 var_455 = -123;
    wp::int32 var_456;
    const wp::int32 var_457 = 1;
    wp::int32 var_458;
    const wp::float32 var_459 = 1.0;
    const wp::float32 var_460 = -1.0;
    wp::float32 var_461;
    const wp::int32 var_462 = 2;
    wp::int32 var_463;
    const wp::float32 var_464 = 1.0;
    const wp::float32 var_465 = -1.0;
    wp::float32 var_466;
    const wp::int32 var_467 = 4;
    wp::int32 var_468;
    const wp::float32 var_469 = 1.0;
    const wp::float32 var_470 = -1.0;
    wp::float32 var_471;
    wp::vec_t<3, wp::float32> var_472;
    wp::vec_t<3, wp::float32> var_473;
    const wp::float32 var_474 = 0.0;
    wp::vec_t<3, wp::float32> var_475;
    wp::float32 var_476;
    wp::float32 var_477;
    wp::float32 var_478;
    wp::float32 var_479;
    wp::float32 var_480;
    wp::float32 var_481;
    wp::float32 var_482;
    wp::float32 var_483;
    wp::float32 var_484;
    wp::float32 var_485;
    wp::float32 var_486;
    wp::float32 var_487;
    wp::float32 var_488;
    wp::float32 var_489;
    wp::float32 var_490;
    wp::float32 var_491;
    wp::float32 var_492;
    bool var_493;
    wp::int32 var_494;
    wp::vec_t<3, wp::float32> var_495;
    wp::vec_t<3, wp::float32> var_496;
    wp::float32 var_497;
    wp::float32 var_498;
    wp::float32 var_499;
    wp::float32 var_500;
    wp::float32 var_501;
    wp::float32 var_502;
    const wp::float32 var_503 = 1.0;
    wp::float32 var_504;
    wp::float32 var_505;
    wp::float32 var_506;
    wp::float32 var_507;
    wp::float32 var_508;
    wp::float32 var_509;
    wp::float32 var_510;
    wp::float32 var_511;
    wp::float32 var_512;
    wp::float32 var_513;
    wp::float32 var_514;
    const wp::int32 var_515 = 1;
    wp::int32 var_516;
    const wp::int32 var_517 = 1;
    wp::int32 var_518;
    const wp::int32 var_519 = 1;
    bool var_520;
    const wp::float32 var_521 = 1.0;
    const wp::int32 var_522 = 2;
    wp::float32 var_523;
    wp::float32 var_524;
    wp::float32 var_525;
    wp::float32 var_526;
    wp::int32 var_527;
    const wp::int32 var_528 = -1;
    bool var_529;
    const wp::float32 var_530 = -1.0;
    const wp::int32 var_531 = 0;
    wp::float32 var_532;
    wp::float32 var_533;
    wp::float32 var_534;
    wp::float32 var_535;
    wp::int32 var_536;
    wp::float32 var_537;
    wp::float32 var_538;
    wp::int32 var_539;
    const wp::float32 var_540 = 1.0;
    bool var_541;
    bool var_542;
    const wp::float32 var_543 = -1.0;
    bool var_544;
    const wp::float32 var_545 = 1.0;
    const wp::int32 var_546 = 2;
    wp::float32 var_547;
    wp::float32 var_548;
    wp::float32 var_549;
    wp::float32 var_550;
    wp::int32 var_551;
    const wp::float32 var_552 = -1.0;
    const wp::int32 var_553 = 0;
    wp::float32 var_554;
    wp::float32 var_555;
    wp::float32 var_556;
    wp::float32 var_557;
    wp::int32 var_558;
    const wp::int32 var_559 = 1;
    bool var_560;
    const wp::float32 var_561 = 1.0;
    const wp::int32 var_562 = 2;
    wp::float32 var_563;
    wp::int32 var_564;
    const wp::int32 var_565 = -1;
    bool var_566;
    const wp::float32 var_567 = -1.0;
    const wp::int32 var_568 = 0;
    wp::float32 var_569;
    wp::int32 var_570;
    wp::float32 var_571;
    wp::int32 var_572;
    wp::float32 var_573;
    wp::float32 var_574;
    wp::int32 var_575;
    wp::int32 var_576;
    wp::vec_t<3, wp::float32> var_577;
    wp::vec_t<3, wp::float32> var_578;
    wp::float32 var_579;
    wp::float32 var_580;
    const wp::int32 var_581 = 3;
    wp::int32 var_582;
    wp::int32 var_583;
    wp::float32 var_584;
    wp::float32 var_585;
    bool var_586;
    wp::float32 var_587;
    wp::float32 var_588;
    wp::float32 var_589;
    const wp::int32 var_590 = 6;
    wp::int32 var_591;
    const wp::int32 var_592 = 1;
    wp::int32 var_593;
    wp::int32 var_594;
    wp::int32 var_595;
    wp::int32 var_596;
    wp::int32 var_597;
    wp::float32 var_598;
    wp::float32 var_599;
    wp::int32 var_600;
    wp::int32 var_601;
    wp::int32 var_602;
    wp::float32 var_603;
    wp::int32 var_604;
    const wp::int32 var_605 = 3;
    const wp::int32 var_606 = 3;
    wp::range_t var_607;
    wp::int32 var_608;
    const wp::int32 var_609 = 1;
    wp::int32 var_610;
    wp::int32 var_611;
    const wp::int32 var_612 = 0;
    bool var_613;
    const wp::int32 var_614 = -123;
    wp::int32 var_615;
    const wp::int32 var_616 = 1;
    wp::int32 var_617;
    const wp::float32 var_618 = 1.0;
    const wp::float32 var_619 = -1.0;
    wp::float32 var_620;
    const wp::int32 var_621 = 2;
    wp::int32 var_622;
    const wp::float32 var_623 = 1.0;
    const wp::float32 var_624 = -1.0;
    wp::float32 var_625;
    const wp::int32 var_626 = 4;
    wp::int32 var_627;
    const wp::float32 var_628 = 1.0;
    const wp::float32 var_629 = -1.0;
    wp::float32 var_630;
    wp::vec_t<3, wp::float32> var_631;
    wp::vec_t<3, wp::float32> var_632;
    const wp::float32 var_633 = 0.0;
    wp::vec_t<3, wp::float32> var_634;
    wp::float32 var_635;
    wp::float32 var_636;
    wp::float32 var_637;
    wp::float32 var_638;
    wp::float32 var_639;
    wp::float32 var_640;
    wp::float32 var_641;
    wp::float32 var_642;
    wp::float32 var_643;
    wp::float32 var_644;
    wp::float32 var_645;
    wp::float32 var_646;
    wp::float32 var_647;
    wp::float32 var_648;
    wp::float32 var_649;
    wp::float32 var_650;
    wp::float32 var_651;
    bool var_652;
    wp::int32 var_653;
    wp::vec_t<3, wp::float32> var_654;
    wp::vec_t<3, wp::float32> var_655;
    wp::float32 var_656;
    wp::float32 var_657;
    wp::float32 var_658;
    wp::float32 var_659;
    wp::float32 var_660;
    wp::float32 var_661;
    const wp::float32 var_662 = 1.0;
    wp::float32 var_663;
    wp::float32 var_664;
    wp::float32 var_665;
    wp::float32 var_666;
    wp::float32 var_667;
    wp::float32 var_668;
    wp::float32 var_669;
    wp::float32 var_670;
    wp::float32 var_671;
    wp::float32 var_672;
    wp::float32 var_673;
    const wp::int32 var_674 = 1;
    wp::int32 var_675;
    const wp::int32 var_676 = 1;
    wp::int32 var_677;
    const wp::int32 var_678 = 1;
    bool var_679;
    const wp::float32 var_680 = 1.0;
    const wp::int32 var_681 = 2;
    wp::float32 var_682;
    wp::float32 var_683;
    wp::float32 var_684;
    wp::float32 var_685;
    wp::int32 var_686;
    const wp::int32 var_687 = -1;
    bool var_688;
    const wp::float32 var_689 = -1.0;
    const wp::int32 var_690 = 0;
    wp::float32 var_691;
    wp::float32 var_692;
    wp::float32 var_693;
    wp::float32 var_694;
    wp::int32 var_695;
    wp::float32 var_696;
    wp::float32 var_697;
    wp::int32 var_698;
    const wp::float32 var_699 = 1.0;
    bool var_700;
    bool var_701;
    const wp::float32 var_702 = -1.0;
    bool var_703;
    const wp::float32 var_704 = 1.0;
    const wp::int32 var_705 = 2;
    wp::float32 var_706;
    wp::float32 var_707;
    wp::float32 var_708;
    wp::float32 var_709;
    wp::int32 var_710;
    const wp::float32 var_711 = -1.0;
    const wp::int32 var_712 = 0;
    wp::float32 var_713;
    wp::float32 var_714;
    wp::float32 var_715;
    wp::float32 var_716;
    wp::int32 var_717;
    const wp::int32 var_718 = 1;
    bool var_719;
    const wp::float32 var_720 = 1.0;
    const wp::int32 var_721 = 2;
    wp::float32 var_722;
    wp::int32 var_723;
    const wp::int32 var_724 = -1;
    bool var_725;
    const wp::float32 var_726 = -1.0;
    const wp::int32 var_727 = 0;
    wp::float32 var_728;
    wp::int32 var_729;
    wp::float32 var_730;
    wp::int32 var_731;
    wp::float32 var_732;
    wp::float32 var_733;
    wp::int32 var_734;
    wp::int32 var_735;
    wp::vec_t<3, wp::float32> var_736;
    wp::vec_t<3, wp::float32> var_737;
    wp::float32 var_738;
    wp::float32 var_739;
    const wp::int32 var_740 = 3;
    wp::int32 var_741;
    wp::int32 var_742;
    wp::float32 var_743;
    wp::float32 var_744;
    bool var_745;
    wp::float32 var_746;
    wp::float32 var_747;
    wp::float32 var_748;
    const wp::int32 var_749 = 6;
    wp::int32 var_750;
    const wp::int32 var_751 = 1;
    wp::int32 var_752;
    wp::int32 var_753;
    wp::int32 var_754;
    wp::int32 var_755;
    wp::int32 var_756;
    wp::float32 var_757;
    wp::float32 var_758;
    wp::int32 var_759;
    wp::int32 var_760;
    wp::int32 var_761;
    wp::float32 var_762;
    wp::int32 var_763;
    const wp::int32 var_764 = 4;
    const wp::int32 var_765 = 3;
    wp::range_t var_766;
    wp::int32 var_767;
    const wp::int32 var_768 = 1;
    wp::int32 var_769;
    wp::int32 var_770;
    const wp::int32 var_771 = 0;
    bool var_772;
    const wp::int32 var_773 = -123;
    wp::int32 var_774;
    const wp::int32 var_775 = 1;
    wp::int32 var_776;
    const wp::float32 var_777 = 1.0;
    const wp::float32 var_778 = -1.0;
    wp::float32 var_779;
    const wp::int32 var_780 = 2;
    wp::int32 var_781;
    const wp::float32 var_782 = 1.0;
    const wp::float32 var_783 = -1.0;
    wp::float32 var_784;
    const wp::int32 var_785 = 4;
    wp::int32 var_786;
    const wp::float32 var_787 = 1.0;
    const wp::float32 var_788 = -1.0;
    wp::float32 var_789;
    wp::vec_t<3, wp::float32> var_790;
    wp::vec_t<3, wp::float32> var_791;
    const wp::float32 var_792 = 0.0;
    wp::vec_t<3, wp::float32> var_793;
    wp::float32 var_794;
    wp::float32 var_795;
    wp::float32 var_796;
    wp::float32 var_797;
    wp::float32 var_798;
    wp::float32 var_799;
    wp::float32 var_800;
    wp::float32 var_801;
    wp::float32 var_802;
    wp::float32 var_803;
    wp::float32 var_804;
    wp::float32 var_805;
    wp::float32 var_806;
    wp::float32 var_807;
    wp::float32 var_808;
    wp::float32 var_809;
    wp::float32 var_810;
    bool var_811;
    wp::int32 var_812;
    wp::vec_t<3, wp::float32> var_813;
    wp::vec_t<3, wp::float32> var_814;
    wp::float32 var_815;
    wp::float32 var_816;
    wp::float32 var_817;
    wp::float32 var_818;
    wp::float32 var_819;
    wp::float32 var_820;
    const wp::float32 var_821 = 1.0;
    wp::float32 var_822;
    wp::float32 var_823;
    wp::float32 var_824;
    wp::float32 var_825;
    wp::float32 var_826;
    wp::float32 var_827;
    wp::float32 var_828;
    wp::float32 var_829;
    wp::float32 var_830;
    wp::float32 var_831;
    wp::float32 var_832;
    const wp::int32 var_833 = 1;
    wp::int32 var_834;
    const wp::int32 var_835 = 1;
    wp::int32 var_836;
    const wp::int32 var_837 = 1;
    bool var_838;
    const wp::float32 var_839 = 1.0;
    const wp::int32 var_840 = 2;
    wp::float32 var_841;
    wp::float32 var_842;
    wp::float32 var_843;
    wp::float32 var_844;
    wp::int32 var_845;
    const wp::int32 var_846 = -1;
    bool var_847;
    const wp::float32 var_848 = -1.0;
    const wp::int32 var_849 = 0;
    wp::float32 var_850;
    wp::float32 var_851;
    wp::float32 var_852;
    wp::float32 var_853;
    wp::int32 var_854;
    wp::float32 var_855;
    wp::float32 var_856;
    wp::int32 var_857;
    const wp::float32 var_858 = 1.0;
    bool var_859;
    bool var_860;
    const wp::float32 var_861 = -1.0;
    bool var_862;
    const wp::float32 var_863 = 1.0;
    const wp::int32 var_864 = 2;
    wp::float32 var_865;
    wp::float32 var_866;
    wp::float32 var_867;
    wp::float32 var_868;
    wp::int32 var_869;
    const wp::float32 var_870 = -1.0;
    const wp::int32 var_871 = 0;
    wp::float32 var_872;
    wp::float32 var_873;
    wp::float32 var_874;
    wp::float32 var_875;
    wp::int32 var_876;
    const wp::int32 var_877 = 1;
    bool var_878;
    const wp::float32 var_879 = 1.0;
    const wp::int32 var_880 = 2;
    wp::float32 var_881;
    wp::int32 var_882;
    const wp::int32 var_883 = -1;
    bool var_884;
    const wp::float32 var_885 = -1.0;
    const wp::int32 var_886 = 0;
    wp::float32 var_887;
    wp::int32 var_888;
    wp::float32 var_889;
    wp::int32 var_890;
    wp::float32 var_891;
    wp::float32 var_892;
    wp::int32 var_893;
    wp::int32 var_894;
    wp::vec_t<3, wp::float32> var_895;
    wp::vec_t<3, wp::float32> var_896;
    wp::float32 var_897;
    wp::float32 var_898;
    const wp::int32 var_899 = 3;
    wp::int32 var_900;
    wp::int32 var_901;
    wp::float32 var_902;
    wp::float32 var_903;
    bool var_904;
    wp::float32 var_905;
    wp::float32 var_906;
    wp::float32 var_907;
    const wp::int32 var_908 = 6;
    wp::int32 var_909;
    const wp::int32 var_910 = 1;
    wp::int32 var_911;
    wp::int32 var_912;
    wp::int32 var_913;
    wp::int32 var_914;
    wp::int32 var_915;
    wp::float32 var_916;
    wp::float32 var_917;
    wp::int32 var_918;
    wp::int32 var_919;
    wp::int32 var_920;
    wp::float32 var_921;
    wp::int32 var_922;
    const wp::int32 var_923 = 5;
    const wp::int32 var_924 = 3;
    wp::range_t var_925;
    wp::int32 var_926;
    const wp::int32 var_927 = 1;
    wp::int32 var_928;
    wp::int32 var_929;
    const wp::int32 var_930 = 0;
    bool var_931;
    const wp::int32 var_932 = -123;
    wp::int32 var_933;
    const wp::int32 var_934 = 1;
    wp::int32 var_935;
    const wp::float32 var_936 = 1.0;
    const wp::float32 var_937 = -1.0;
    wp::float32 var_938;
    const wp::int32 var_939 = 2;
    wp::int32 var_940;
    const wp::float32 var_941 = 1.0;
    const wp::float32 var_942 = -1.0;
    wp::float32 var_943;
    const wp::int32 var_944 = 4;
    wp::int32 var_945;
    const wp::float32 var_946 = 1.0;
    const wp::float32 var_947 = -1.0;
    wp::float32 var_948;
    wp::vec_t<3, wp::float32> var_949;
    wp::vec_t<3, wp::float32> var_950;
    const wp::float32 var_951 = 0.0;
    wp::vec_t<3, wp::float32> var_952;
    wp::float32 var_953;
    wp::float32 var_954;
    wp::float32 var_955;
    wp::float32 var_956;
    wp::float32 var_957;
    wp::float32 var_958;
    wp::float32 var_959;
    wp::float32 var_960;
    wp::float32 var_961;
    wp::float32 var_962;
    wp::float32 var_963;
    wp::float32 var_964;
    wp::float32 var_965;
    wp::float32 var_966;
    wp::float32 var_967;
    wp::float32 var_968;
    wp::float32 var_969;
    bool var_970;
    wp::int32 var_971;
    wp::vec_t<3, wp::float32> var_972;
    wp::vec_t<3, wp::float32> var_973;
    wp::float32 var_974;
    wp::float32 var_975;
    wp::float32 var_976;
    wp::float32 var_977;
    wp::float32 var_978;
    wp::float32 var_979;
    const wp::float32 var_980 = 1.0;
    wp::float32 var_981;
    wp::float32 var_982;
    wp::float32 var_983;
    wp::float32 var_984;
    wp::float32 var_985;
    wp::float32 var_986;
    wp::float32 var_987;
    wp::float32 var_988;
    wp::float32 var_989;
    wp::float32 var_990;
    wp::float32 var_991;
    const wp::int32 var_992 = 1;
    wp::int32 var_993;
    const wp::int32 var_994 = 1;
    wp::int32 var_995;
    const wp::int32 var_996 = 1;
    bool var_997;
    const wp::float32 var_998 = 1.0;
    const wp::int32 var_999 = 2;
    wp::float32 var_1000;
    wp::float32 var_1001;
    wp::float32 var_1002;
    wp::float32 var_1003;
    wp::int32 var_1004;
    const wp::int32 var_1005 = -1;
    bool var_1006;
    const wp::float32 var_1007 = -1.0;
    const wp::int32 var_1008 = 0;
    wp::float32 var_1009;
    wp::float32 var_1010;
    wp::float32 var_1011;
    wp::float32 var_1012;
    wp::int32 var_1013;
    wp::float32 var_1014;
    wp::float32 var_1015;
    wp::int32 var_1016;
    const wp::float32 var_1017 = 1.0;
    bool var_1018;
    bool var_1019;
    const wp::float32 var_1020 = -1.0;
    bool var_1021;
    const wp::float32 var_1022 = 1.0;
    const wp::int32 var_1023 = 2;
    wp::float32 var_1024;
    wp::float32 var_1025;
    wp::float32 var_1026;
    wp::float32 var_1027;
    wp::int32 var_1028;
    const wp::float32 var_1029 = -1.0;
    const wp::int32 var_1030 = 0;
    wp::float32 var_1031;
    wp::float32 var_1032;
    wp::float32 var_1033;
    wp::float32 var_1034;
    wp::int32 var_1035;
    const wp::int32 var_1036 = 1;
    bool var_1037;
    const wp::float32 var_1038 = 1.0;
    const wp::int32 var_1039 = 2;
    wp::float32 var_1040;
    wp::int32 var_1041;
    const wp::int32 var_1042 = -1;
    bool var_1043;
    const wp::float32 var_1044 = -1.0;
    const wp::int32 var_1045 = 0;
    wp::float32 var_1046;
    wp::int32 var_1047;
    wp::float32 var_1048;
    wp::int32 var_1049;
    wp::float32 var_1050;
    wp::float32 var_1051;
    wp::int32 var_1052;
    wp::int32 var_1053;
    wp::vec_t<3, wp::float32> var_1054;
    wp::vec_t<3, wp::float32> var_1055;
    wp::float32 var_1056;
    wp::float32 var_1057;
    const wp::int32 var_1058 = 3;
    wp::int32 var_1059;
    wp::int32 var_1060;
    wp::float32 var_1061;
    wp::float32 var_1062;
    bool var_1063;
    wp::float32 var_1064;
    wp::float32 var_1065;
    wp::float32 var_1066;
    const wp::int32 var_1067 = 6;
    wp::int32 var_1068;
    const wp::int32 var_1069 = 1;
    wp::int32 var_1070;
    wp::int32 var_1071;
    wp::int32 var_1072;
    wp::int32 var_1073;
    wp::int32 var_1074;
    wp::float32 var_1075;
    wp::float32 var_1076;
    wp::int32 var_1077;
    wp::int32 var_1078;
    wp::int32 var_1079;
    wp::float32 var_1080;
    wp::int32 var_1081;
    const wp::int32 var_1082 = 6;
    const wp::int32 var_1083 = 3;
    wp::range_t var_1084;
    wp::int32 var_1085;
    const wp::int32 var_1086 = 1;
    wp::int32 var_1087;
    wp::int32 var_1088;
    const wp::int32 var_1089 = 0;
    bool var_1090;
    const wp::int32 var_1091 = -123;
    wp::int32 var_1092;
    const wp::int32 var_1093 = 1;
    wp::int32 var_1094;
    const wp::float32 var_1095 = 1.0;
    const wp::float32 var_1096 = -1.0;
    wp::float32 var_1097;
    const wp::int32 var_1098 = 2;
    wp::int32 var_1099;
    const wp::float32 var_1100 = 1.0;
    const wp::float32 var_1101 = -1.0;
    wp::float32 var_1102;
    const wp::int32 var_1103 = 4;
    wp::int32 var_1104;
    const wp::float32 var_1105 = 1.0;
    const wp::float32 var_1106 = -1.0;
    wp::float32 var_1107;
    wp::vec_t<3, wp::float32> var_1108;
    wp::vec_t<3, wp::float32> var_1109;
    const wp::float32 var_1110 = 0.0;
    wp::vec_t<3, wp::float32> var_1111;
    wp::float32 var_1112;
    wp::float32 var_1113;
    wp::float32 var_1114;
    wp::float32 var_1115;
    wp::float32 var_1116;
    wp::float32 var_1117;
    wp::float32 var_1118;
    wp::float32 var_1119;
    wp::float32 var_1120;
    wp::float32 var_1121;
    wp::float32 var_1122;
    wp::float32 var_1123;
    wp::float32 var_1124;
    wp::float32 var_1125;
    wp::float32 var_1126;
    wp::float32 var_1127;
    wp::float32 var_1128;
    bool var_1129;
    wp::int32 var_1130;
    wp::vec_t<3, wp::float32> var_1131;
    wp::vec_t<3, wp::float32> var_1132;
    wp::float32 var_1133;
    wp::float32 var_1134;
    wp::float32 var_1135;
    wp::float32 var_1136;
    wp::float32 var_1137;
    wp::float32 var_1138;
    const wp::float32 var_1139 = 1.0;
    wp::float32 var_1140;
    wp::float32 var_1141;
    wp::float32 var_1142;
    wp::float32 var_1143;
    wp::float32 var_1144;
    wp::float32 var_1145;
    wp::float32 var_1146;
    wp::float32 var_1147;
    wp::float32 var_1148;
    wp::float32 var_1149;
    wp::float32 var_1150;
    const wp::int32 var_1151 = 1;
    wp::int32 var_1152;
    const wp::int32 var_1153 = 1;
    wp::int32 var_1154;
    const wp::int32 var_1155 = 1;
    bool var_1156;
    const wp::float32 var_1157 = 1.0;
    const wp::int32 var_1158 = 2;
    wp::float32 var_1159;
    wp::float32 var_1160;
    wp::float32 var_1161;
    wp::float32 var_1162;
    wp::int32 var_1163;
    const wp::int32 var_1164 = -1;
    bool var_1165;
    const wp::float32 var_1166 = -1.0;
    const wp::int32 var_1167 = 0;
    wp::float32 var_1168;
    wp::float32 var_1169;
    wp::float32 var_1170;
    wp::float32 var_1171;
    wp::int32 var_1172;
    wp::float32 var_1173;
    wp::float32 var_1174;
    wp::int32 var_1175;
    const wp::float32 var_1176 = 1.0;
    bool var_1177;
    bool var_1178;
    const wp::float32 var_1179 = -1.0;
    bool var_1180;
    const wp::float32 var_1181 = 1.0;
    const wp::int32 var_1182 = 2;
    wp::float32 var_1183;
    wp::float32 var_1184;
    wp::float32 var_1185;
    wp::float32 var_1186;
    wp::int32 var_1187;
    const wp::float32 var_1188 = -1.0;
    const wp::int32 var_1189 = 0;
    wp::float32 var_1190;
    wp::float32 var_1191;
    wp::float32 var_1192;
    wp::float32 var_1193;
    wp::int32 var_1194;
    const wp::int32 var_1195 = 1;
    bool var_1196;
    const wp::float32 var_1197 = 1.0;
    const wp::int32 var_1198 = 2;
    wp::float32 var_1199;
    wp::int32 var_1200;
    const wp::int32 var_1201 = -1;
    bool var_1202;
    const wp::float32 var_1203 = -1.0;
    const wp::int32 var_1204 = 0;
    wp::float32 var_1205;
    wp::int32 var_1206;
    wp::float32 var_1207;
    wp::int32 var_1208;
    wp::float32 var_1209;
    wp::float32 var_1210;
    wp::int32 var_1211;
    wp::int32 var_1212;
    wp::vec_t<3, wp::float32> var_1213;
    wp::vec_t<3, wp::float32> var_1214;
    wp::float32 var_1215;
    wp::float32 var_1216;
    const wp::int32 var_1217 = 3;
    wp::int32 var_1218;
    wp::int32 var_1219;
    wp::float32 var_1220;
    wp::float32 var_1221;
    bool var_1222;
    wp::float32 var_1223;
    wp::float32 var_1224;
    wp::float32 var_1225;
    const wp::int32 var_1226 = 6;
    wp::int32 var_1227;
    const wp::int32 var_1228 = 1;
    wp::int32 var_1229;
    wp::int32 var_1230;
    wp::int32 var_1231;
    wp::int32 var_1232;
    wp::int32 var_1233;
    wp::float32 var_1234;
    wp::float32 var_1235;
    wp::int32 var_1236;
    wp::int32 var_1237;
    wp::int32 var_1238;
    wp::float32 var_1239;
    wp::int32 var_1240;
    const wp::int32 var_1241 = 7;
    const wp::int32 var_1242 = 3;
    wp::range_t var_1243;
    wp::int32 var_1244;
    const wp::int32 var_1245 = 1;
    wp::int32 var_1246;
    wp::int32 var_1247;
    const wp::int32 var_1248 = 0;
    bool var_1249;
    const wp::int32 var_1250 = -123;
    wp::int32 var_1251;
    const wp::int32 var_1252 = 1;
    wp::int32 var_1253;
    const wp::float32 var_1254 = 1.0;
    const wp::float32 var_1255 = -1.0;
    wp::float32 var_1256;
    const wp::int32 var_1257 = 2;
    wp::int32 var_1258;
    const wp::float32 var_1259 = 1.0;
    const wp::float32 var_1260 = -1.0;
    wp::float32 var_1261;
    const wp::int32 var_1262 = 4;
    wp::int32 var_1263;
    const wp::float32 var_1264 = 1.0;
    const wp::float32 var_1265 = -1.0;
    wp::float32 var_1266;
    wp::vec_t<3, wp::float32> var_1267;
    wp::vec_t<3, wp::float32> var_1268;
    const wp::float32 var_1269 = 0.0;
    wp::vec_t<3, wp::float32> var_1270;
    wp::float32 var_1271;
    wp::float32 var_1272;
    wp::float32 var_1273;
    wp::float32 var_1274;
    wp::float32 var_1275;
    wp::float32 var_1276;
    wp::float32 var_1277;
    wp::float32 var_1278;
    wp::float32 var_1279;
    wp::float32 var_1280;
    wp::float32 var_1281;
    wp::float32 var_1282;
    wp::float32 var_1283;
    wp::float32 var_1284;
    wp::float32 var_1285;
    wp::float32 var_1286;
    wp::float32 var_1287;
    bool var_1288;
    wp::int32 var_1289;
    wp::vec_t<3, wp::float32> var_1290;
    wp::vec_t<3, wp::float32> var_1291;
    wp::float32 var_1292;
    wp::float32 var_1293;
    wp::float32 var_1294;
    wp::float32 var_1295;
    wp::float32 var_1296;
    wp::float32 var_1297;
    const wp::float32 var_1298 = 1.0;
    wp::float32 var_1299;
    wp::float32 var_1300;
    wp::float32 var_1301;
    wp::float32 var_1302;
    wp::float32 var_1303;
    wp::float32 var_1304;
    wp::float32 var_1305;
    wp::float32 var_1306;
    wp::float32 var_1307;
    wp::float32 var_1308;
    wp::float32 var_1309;
    const wp::int32 var_1310 = 1;
    wp::int32 var_1311;
    const wp::int32 var_1312 = 1;
    wp::int32 var_1313;
    const wp::int32 var_1314 = 1;
    bool var_1315;
    const wp::float32 var_1316 = 1.0;
    const wp::int32 var_1317 = 2;
    wp::float32 var_1318;
    wp::float32 var_1319;
    wp::float32 var_1320;
    wp::float32 var_1321;
    wp::int32 var_1322;
    const wp::int32 var_1323 = -1;
    bool var_1324;
    const wp::float32 var_1325 = -1.0;
    const wp::int32 var_1326 = 0;
    wp::float32 var_1327;
    wp::float32 var_1328;
    wp::float32 var_1329;
    wp::float32 var_1330;
    wp::int32 var_1331;
    wp::float32 var_1332;
    wp::float32 var_1333;
    wp::int32 var_1334;
    const wp::float32 var_1335 = 1.0;
    bool var_1336;
    bool var_1337;
    const wp::float32 var_1338 = -1.0;
    bool var_1339;
    const wp::float32 var_1340 = 1.0;
    const wp::int32 var_1341 = 2;
    wp::float32 var_1342;
    wp::float32 var_1343;
    wp::float32 var_1344;
    wp::float32 var_1345;
    wp::int32 var_1346;
    const wp::float32 var_1347 = -1.0;
    const wp::int32 var_1348 = 0;
    wp::float32 var_1349;
    wp::float32 var_1350;
    wp::float32 var_1351;
    wp::float32 var_1352;
    wp::int32 var_1353;
    const wp::int32 var_1354 = 1;
    bool var_1355;
    const wp::float32 var_1356 = 1.0;
    const wp::int32 var_1357 = 2;
    wp::float32 var_1358;
    wp::int32 var_1359;
    const wp::int32 var_1360 = -1;
    bool var_1361;
    const wp::float32 var_1362 = -1.0;
    const wp::int32 var_1363 = 0;
    wp::float32 var_1364;
    wp::int32 var_1365;
    wp::float32 var_1366;
    wp::int32 var_1367;
    wp::float32 var_1368;
    wp::float32 var_1369;
    wp::int32 var_1370;
    wp::int32 var_1371;
    wp::vec_t<3, wp::float32> var_1372;
    wp::vec_t<3, wp::float32> var_1373;
    wp::float32 var_1374;
    wp::float32 var_1375;
    const wp::int32 var_1376 = 3;
    wp::int32 var_1377;
    wp::int32 var_1378;
    wp::float32 var_1379;
    wp::float32 var_1380;
    bool var_1381;
    wp::float32 var_1382;
    wp::float32 var_1383;
    wp::float32 var_1384;
    const wp::int32 var_1385 = 6;
    wp::int32 var_1386;
    const wp::int32 var_1387 = 1;
    wp::int32 var_1388;
    wp::int32 var_1389;
    wp::int32 var_1390;
    wp::int32 var_1391;
    wp::int32 var_1392;
    wp::float32 var_1393;
    wp::float32 var_1394;
    wp::int32 var_1395;
    wp::int32 var_1396;
    wp::int32 var_1397;
    wp::float32 var_1398;
    wp::int32 var_1399;
    const wp::float32 var_1400 = 0.0;
    wp::float32 var_1401;
    const wp::int32 var_1402 = 0;
    wp::float32 var_1403;
    const wp::int32 var_1404 = 1;
    wp::float32 var_1405;
    wp::vec_t<2, wp::float32> var_1406;
    const wp::int32 var_1407 = 0;
    wp::float32 var_1408;
    const wp::int32 var_1409 = 1;
    wp::float32 var_1410;
    wp::vec_t<2, wp::float32> var_1411;
    const wp::int32 var_1412 = 0;
    wp::float32 var_1413;
    const wp::int32 var_1414 = 1;
    wp::float32 var_1415;
    wp::vec_t<2, wp::float32> var_1416;
    const wp::float32 var_1417 = -4.0;
    wp::float32 var_1418;
    const wp::int32 var_1419 = 0;
    wp::float32 var_1420;
    const wp::int32 var_1421 = 1;
    wp::float32 var_1422;
    wp::float32 var_1423;
    const wp::int32 var_1424 = 1;
    wp::float32 var_1425;
    const wp::int32 var_1426 = 0;
    wp::float32 var_1427;
    wp::float32 var_1428;
    const wp::int32 var_1429 = 0;
    wp::float32 var_1430;
    const wp::int32 var_1431 = 1;
    wp::float32 var_1432;
    wp::float32 var_1433;
    const wp::int32 var_1434 = 1;
    wp::float32 var_1435;
    const wp::int32 var_1436 = 0;
    wp::float32 var_1437;
    wp::float32 var_1438;
    wp::float32 var_1439;
    const wp::int32 var_1440 = 0;
    bool var_1441;
    const wp::float32 var_1442 = -1.0;
    wp::float32 var_1443;
    wp::float32 var_1444;
    wp::float32 var_1445;
    wp::float32 var_1446;
    bool var_1447;
    wp::float32 var_1448;
    const wp::int32 var_1449 = 0;
    bool var_1450;
    bool var_1451;
    const wp::int32 var_1452 = 0;
    const wp::int32 var_1453 = 3;
    wp::int32 var_1454;
    wp::float32 var_1455;
    wp::float32 var_1456;
    bool var_1457;
    wp::float32 var_1458;
    const wp::int32 var_1459 = 0;
    bool var_1460;
    bool var_1461;
    const wp::int32 var_1462 = 1;
    const wp::int32 var_1463 = 2;
    wp::int32 var_1464;
    wp::float32 var_1465;
    wp::int32 var_1466;
    const wp::int32 var_1467 = -4;
    bool var_1468;
    const wp::float32 var_1469 = 10000000000.0;
    wp::vec_t<2, wp::float32> var_1470;
    wp::mat_t<2, 3, wp::float32> var_1471;
    wp::mat_t<2, 3, wp::float32> var_1472;
    bool var_1473;
    const wp::int32 var_1474 = 0;
    bool var_1475;
    const wp::int32 var_1476 = 3;
    wp::int32 var_1477;
    const wp::int32 var_1478 = 1;
    bool var_1479;
    wp::int32 var_1480;
    bool var_1481;
    const wp::int32 var_1482 = 0;
    bool var_1483;
    const wp::int32 var_1484 = 7;
    bool var_1485;
    bool var_1486;
    const wp::int32 var_1487 = 1;
    bool var_1488;
    const wp::int32 var_1489 = 2;
    bool var_1490;
    const wp::int32 var_1491 = 4;
    bool var_1492;
    const wp::int32 var_1493 = 1;
    const wp::int32 var_1494 = -1;
    const wp::int32 var_1495 = 7;
    wp::int32 var_1496;
    wp::int32 var_1497;
    wp::int32 var_1498;
    const wp::int32 var_1499 = 1;
    bool var_1500;
    const wp::int32 var_1501 = 0;
    const wp::int32 var_1502 = 1;
    const wp::int32 var_1503 = 2;
    const wp::int32 var_1504 = 2;
    bool var_1505;
    const wp::int32 var_1506 = 1;
    const wp::int32 var_1507 = 2;
    const wp::int32 var_1508 = 0;
    wp::int32 var_1509;
    wp::int32 var_1510;
    wp::int32 var_1511;
    const wp::int32 var_1512 = 4;
    bool var_1513;
    const wp::int32 var_1514 = 2;
    const wp::int32 var_1515 = 0;
    const wp::int32 var_1516 = 1;
    wp::int32 var_1517;
    wp::int32 var_1518;
    wp::int32 var_1519;
    wp::int32 var_1520;
    wp::int32 var_1521;
    wp::int32 var_1522;
    wp::int32 var_1523;
    wp::int32 var_1524;
    wp::int32 var_1525;
    wp::float32 var_1526;
    wp::float32 var_1527;
    wp::float32 var_1528;
    const wp::float32 var_1529 = 0.5;
    bool var_1530;
    const wp::float32 var_1531 = 2.0;
    wp::float32 var_1532;
    wp::float32 var_1533;
    wp::float32 var_1534;
    wp::float32 var_1535;
    wp::float32 var_1536;
    const wp::float32 var_1537 = 1.0;
    wp::float32 var_1538;
    wp::float32 var_1539;
    wp::float32 var_1540;
    wp::float32 var_1541;
    wp::float32 var_1542;
    const wp::float32 var_1543 = 2.0;
    wp::float32 var_1544;
    wp::float32 var_1545;
    wp::float32 var_1546;
    wp::float32 var_1547;
    wp::float32 var_1548;
    wp::float32 var_1549;
    wp::float32 var_1550;
    wp::float32 var_1551;
    wp::float32 var_1552;
    wp::float32 var_1553;
    const wp::float32 var_1554 = 1.0;
    wp::float32 var_1555;
    wp::float32 var_1556;
    wp::float32 var_1557;
    wp::float32 var_1558;
    wp::float32 var_1559;
    wp::float32 var_1560;
    wp::float32 var_1561;
    wp::float32 var_1562;
    wp::float32 var_1563;
    wp::float32 var_1564;
    wp::int32 var_1565;
    wp::float32 var_1566;
    wp::int32 var_1567;
    bool var_1568;
    const wp::int32 var_1569 = 0;
    bool var_1570;
    const wp::int32 var_1571 = 3;
    wp::int32 var_1572;
    const wp::int32 var_1573 = 1;
    bool var_1574;
    wp::int32 var_1575;
    const wp::int32 var_1576 = 7;
    const wp::int32 var_1577 = 1;
    wp::int32 var_1578;
    wp::int32 var_1579;
    wp::int32 var_1580;
    bool var_1581;
    const wp::int32 var_1582 = 1;
    bool var_1583;
    const wp::int32 var_1584 = 2;
    bool var_1585;
    const wp::int32 var_1586 = 4;
    bool var_1587;
    const wp::int32 var_1588 = 0;
    bool var_1589;
    const wp::int32 var_1590 = 1;
    const wp::int32 var_1591 = 2;
    wp::int32 var_1592;
    wp::int32 var_1593;
    const wp::int32 var_1594 = 1;
    bool var_1595;
    const wp::int32 var_1596 = 2;
    const wp::int32 var_1597 = 0;
    wp::int32 var_1598;
    wp::int32 var_1599;
    const wp::int32 var_1600 = 2;
    bool var_1601;
    const wp::int32 var_1602 = 0;
    const wp::int32 var_1603 = 1;
    wp::int32 var_1604;
    wp::int32 var_1605;
    wp::int32 var_1606;
    wp::float32 var_1607;
    wp::float32 var_1608;
    wp::float32 var_1609;
    wp::float32 var_1610;
    bool var_1611;
    wp::int32 var_1612;
    wp::int32 var_1613;
    const wp::int32 var_1614 = 3;
    wp::int32 var_1615;
    wp::int32 var_1616;
    const wp::int32 var_1617 = 1;
    wp::int32 var_1618;
    wp::int32 var_1619;
    const wp::int32 var_1620 = 1;
    const wp::float32 var_1621 = 1.0;
    wp::float32 var_1622;
    wp::float32 var_1623;
    wp::int32 var_1624;
    const wp::int32 var_1625 = -1;
    const wp::float32 var_1626 = 1.0;
    wp::float32 var_1627;
    wp::float32 var_1628;
    wp::int32 var_1629;
    const wp::float32 var_1630 = 2.0;
    wp::float32 var_1631;
    wp::float32 var_1632;
    wp::float32 var_1633;
    wp::float32 var_1634;
    wp::float32 var_1635;
    wp::float32 var_1636;
    const wp::int32 var_1637 = 1;
    wp::int32 var_1638;
    wp::int32 var_1639;
    const wp::int32 var_1640 = 0;
    bool var_1641;
    const wp::int32 var_1642 = 1;
    wp::int32 var_1643;
    wp::int32 var_1644;
    const wp::int32 var_1645 = 0;
    bool var_1646;
    bool var_1647;
    const wp::float32 var_1648 = 1.0;
    wp::float32 var_1649;
    const wp::float32 var_1650 = 1.0;
    wp::float32 var_1651;
    wp::float32 var_1652;
    wp::float32 var_1653;
    wp::float32 var_1654;
    wp::float32 var_1655;
    wp::float32 var_1656;
    wp::float32 var_1657;
    wp::float32 var_1658;
    wp::float32 var_1659;
    wp::float32 var_1660;
    wp::float32 var_1661;
    wp::int32 var_1662;
    wp::int32 var_1663;
    wp::int32 var_1664;
    wp::int32 var_1665;
    wp::float32 var_1666;
    wp::int32 var_1667;
    wp::int32 var_1668;
    wp::int32 var_1669;
    wp::int32 var_1670;
    wp::int32 var_1671;
    const wp::int32 var_1672 = 0;
    bool var_1673;
    const wp::int32 var_1674 = -1;
    bool var_1675;
    const wp::int32 var_1676 = -3;
    bool var_1677;
    const wp::int32 var_1678 = 1;
    const wp::int32 var_1679 = -1;
    wp::int32 var_1680;
    const wp::float32 var_1681 = 2.0;
    wp::float32 var_1682;
    wp::vec_t<3, wp::float32> var_1683;
    wp::vec_t<3, wp::float32> var_1684;
    const wp::int32 var_1685 = 0;
    bool var_1686;
    wp::float32 var_1687;
    wp::float32 var_1688;
    wp::float32 var_1689;
    wp::float32 var_1690;
    wp::float32 var_1691;
    wp::float32 var_1692;
    wp::float32 var_1693;
    bool var_1694;
    const wp::int32 var_1695 = 0;
    bool var_1696;
    bool var_1697;
    wp::float32 var_1698;
    wp::float32 var_1699;
    wp::float32 var_1700;
    wp::float32 var_1701;
    wp::float32 var_1702;
    wp::float32 var_1703;
    wp::float32 var_1704;
    bool var_1705;
    const wp::int32 var_1706 = 0;
    bool var_1707;
    bool var_1708;
    wp::float32 var_1709;
    wp::float32 var_1710;
    wp::float32 var_1711;
    wp::float32 var_1712;
    const wp::int32 var_1713 = 1;
    bool var_1714;
    wp::float32 var_1715;
    wp::float32 var_1716;
    wp::float32 var_1717;
    wp::float32 var_1718;
    wp::float32 var_1719;
    wp::float32 var_1720;
    wp::float32 var_1721;
    bool var_1722;
    const wp::int32 var_1723 = 0;
    bool var_1724;
    bool var_1725;
    wp::float32 var_1726;
    wp::float32 var_1727;
    wp::float32 var_1728;
    wp::float32 var_1729;
    wp::float32 var_1730;
    wp::float32 var_1731;
    wp::float32 var_1732;
    bool var_1733;
    const wp::int32 var_1734 = 0;
    bool var_1735;
    bool var_1736;
    wp::float32 var_1737;
    wp::float32 var_1738;
    wp::float32 var_1739;
    wp::float32 var_1740;
    wp::float32 var_1741;
    const wp::int32 var_1742 = 2;
    bool var_1743;
    wp::float32 var_1744;
    wp::float32 var_1745;
    wp::float32 var_1746;
    wp::float32 var_1747;
    wp::float32 var_1748;
    wp::float32 var_1749;
    wp::float32 var_1750;
    bool var_1751;
    const wp::int32 var_1752 = 0;
    bool var_1753;
    bool var_1754;
    wp::float32 var_1755;
    wp::float32 var_1756;
    wp::float32 var_1757;
    wp::float32 var_1758;
    wp::float32 var_1759;
    wp::float32 var_1760;
    wp::float32 var_1761;
    bool var_1762;
    const wp::int32 var_1763 = 0;
    bool var_1764;
    bool var_1765;
    wp::float32 var_1766;
    wp::float32 var_1767;
    wp::float32 var_1768;
    wp::float32 var_1769;
    wp::float32 var_1770;
    wp::float32 var_1771;
    wp::float32 var_1772;
    wp::int32 var_1773;
    wp::float32 var_1774;
    wp::int32 var_1775;
    wp::float32 var_1776;
    wp::int32 var_1777;
    wp::float32 var_1778;
    wp::int32 var_1779;
    wp::float32 var_1780;
    wp::int32 var_1781;
    wp::float32 var_1782;
    wp::int32 var_1783;
    wp::float32 var_1784;
    wp::int32 var_1785;
    wp::float32 var_1786;
    wp::int32 var_1787;
    wp::int32 var_1788;
    wp::int32 var_1789;
    wp::int32 var_1790;
    wp::int32 var_1791;
    wp::vec_t<3, wp::float32> var_1792;
    wp::vec_t<3, wp::float32> var_1793;
    wp::vec_t<3, wp::float32> var_1794;
    wp::vec_t<3, wp::float32> var_1795;
    wp::float32 var_1796;
    wp::vec_t<3, wp::float32> var_1797;
    wp::vec_t<3, wp::float32> var_1798;
    const wp::int32 var_1799 = -3;
    bool var_1800;
    wp::float32 var_1801;
    wp::vec_t<3, wp::float32> var_1802;
    wp::vec_t<3, wp::float32> var_1803;
    wp::vec_t<3, wp::float32> var_1804;
    wp::vec_t<3, wp::float32> var_1805;
    wp::float32 var_1806;
    wp::vec_t<3, wp::float32> var_1807;
    wp::vec_t<3, wp::float32> var_1808;
    wp::float32 var_1809;
    wp::vec_t<3, wp::float32> var_1810;
    wp::vec_t<3, wp::float32> var_1811;
    wp::float32 var_1812;
    wp::vec_t<3, wp::float32> var_1813;
    wp::vec_t<3, wp::float32> var_1814;
    wp::vec_t<2, wp::float32> var_1815;
    const wp::int32 var_1816 = 0;
    wp::float32 var_1817;
    const wp::int32 var_1818 = 1;
    wp::float32 var_1819;
    const wp::int32 var_1820 = 2;
    wp::float32 var_1821;
    const wp::int32 var_1822 = 0;
    wp::float32 var_1823;
    const wp::int32 var_1824 = 1;
    wp::float32 var_1825;
    const wp::int32 var_1826 = 2;
    wp::float32 var_1827;
    wp::mat_t<2, 3, wp::float32> var_1828;
    const wp::int32 var_1829 = 0;
    wp::float32 var_1830;
    const wp::int32 var_1831 = 1;
    wp::float32 var_1832;
    const wp::int32 var_1833 = 2;
    wp::float32 var_1834;
    const wp::int32 var_1835 = 0;
    wp::float32 var_1836;
    const wp::int32 var_1837 = 1;
    wp::float32 var_1838;
    const wp::int32 var_1839 = 2;
    wp::float32 var_1840;
    wp::mat_t<2, 3, wp::float32> var_1841;
    //---------
    // forward
    // def capsule_box(                                                                       <L 1099>
    // boxmatT = wp.transpose(box_rot)                                                        <L 1126>
    var_0 = wp::transpose(var_box_rot);
    // pos = boxmatT @ (capsule_pos - box_pos)                                                <L 1127>
    var_1 = wp::sub(var_capsule_pos, var_box_pos);
    var_2 = wp::mul(var_0, var_1);
    // axis = boxmatT @ capsule_axis                                                          <L 1128>
    var_3 = wp::mul(var_0, var_capsule_axis);
    // halfaxis = axis * capsule_half_length  # halfaxis is the capsule direction             <L 1129>
    var_4 = wp::mul(var_3, var_capsule_half_length);
    // axisdir = wp.int32(halfaxis[0] > 0.0) + 2 * wp.int32(halfaxis[1] > 0.0) + 4 * wp.int32(halfaxis[2] > 0.0)       <L 1130>
    var_6 = wp::extract(var_4, var_5);
    var_8 = (var_6 > var_7);
    var_9 = wp::int32(var_8);
    var_12 = wp::extract(var_4, var_11);
    var_14 = (var_12 > var_13);
    var_15 = wp::int32(var_14);
    var_16 = wp::mul(var_10, var_15);
    var_17 = wp::add(var_9, var_16);
    var_20 = wp::extract(var_4, var_19);
    var_22 = (var_20 > var_21);
    var_23 = wp::int32(var_22);
    var_24 = wp::mul(var_18, var_23);
    var_25 = wp::add(var_17, var_24);
    // bestdist = wp.float32(1.0e32)                                                          <L 1133>
    var_27 = wp::float32(var_26);
    // bestsegmentpos = wp.float32(-12)                                                       <L 1134>
    var_29 = wp::float32(var_28);
    // cltype = wp.int32(-4)                                                                  <L 1143>
    var_31 = wp::int32(var_30);
    // clface = wp.int32(-12)                                                                 <L 1148>
    var_33 = wp::int32(var_32);
    // for i in range(-1, 2, 2):                                                              <L 1151>
    var_37 = wp::range(var_34, var_35, var_36);
    start_for_0:;
        if (iter_cmp(var_37) == 0) goto end_for_0;
        var_38 = wp::iter_next(var_37);
        // axisTip = pos + wp.float32(i) * halfaxis                                           <L 1152>
        var_39 = wp::float32(var_38);
        var_40 = wp::mul(var_39, var_4);
        var_41 = wp::add(var_2, var_40);
        // boxPoint = wp.vec3(axisTip)                                                        <L 1153>
        var_42 = wp::vec_t<3, wp::float32>(var_41);
        // n_out = wp.int32(0)                                                                <L 1155>
        var_44 = wp::int32(var_43);
        // ax_out = wp.int32(-1)                                                              <L 1156>
        var_46 = wp::int32(var_45);
        // for j in range(3):                                                                 <L 1158>
        // if boxPoint[j] < -box_size[j]:                                                     <L 1159>
        var_48 = wp::extract(var_42, var_47);
        var_49 = wp::extract(var_box_size, var_47);
        var_50 = wp::neg(var_49);
        var_51 = (var_48 < var_50);
        if (var_51) {
            // n_out += 1                                                                     <L 1160>
            var_53 = wp::add(var_44, var_52);
            // ax_out = j                                                                     <L 1161>
            var_54 = wp::copy(var_47);
            // boxPoint[j] = -box_size[j]                                                     <L 1162>
            var_55 = wp::extract(var_box_size, var_47);
            var_56 = wp::neg(var_55);
            wp::assign_inplace(var_42, var_47, var_56);
        }
        var_57 = wp::where(var_51, var_53, var_44);
        var_58 = wp::where(var_51, var_54, var_46);
        if (!var_51) {
            // elif boxPoint[j] > box_size[j]:                                                <L 1163>
            var_59 = wp::extract(var_42, var_47);
            var_60 = wp::extract(var_box_size, var_47);
            var_61 = (var_59 > var_60);
            if (var_61) {
                // n_out += 1                                                                 <L 1164>
                var_63 = wp::add(var_57, var_62);
                // ax_out = j                                                                 <L 1165>
                var_64 = wp::copy(var_47);
                // boxPoint[j] = box_size[j]                                                  <L 1166>
                var_65 = wp::extract(var_box_size, var_47);
                wp::assign_inplace(var_42, var_47, var_65);
            }
            var_66 = wp::where(var_61, var_63, var_57);
            var_67 = wp::where(var_61, var_64, var_58);
        }
        var_68 = wp::where(var_51, var_57, var_66);
        var_69 = wp::where(var_51, var_58, var_67);
        // if boxPoint[j] < -box_size[j]:                                                     <L 1159>
        var_71 = wp::extract(var_42, var_70);
        var_72 = wp::extract(var_box_size, var_70);
        var_73 = wp::neg(var_72);
        var_74 = (var_71 < var_73);
        if (var_74) {
            // n_out += 1                                                                     <L 1160>
            var_76 = wp::add(var_68, var_75);
            // ax_out = j                                                                     <L 1161>
            var_77 = wp::copy(var_70);
            // boxPoint[j] = -box_size[j]                                                     <L 1162>
            var_78 = wp::extract(var_box_size, var_70);
            var_79 = wp::neg(var_78);
            wp::assign_inplace(var_42, var_70, var_79);
        }
        var_80 = wp::where(var_74, var_76, var_68);
        var_81 = wp::where(var_74, var_77, var_69);
        if (!var_74) {
            // elif boxPoint[j] > box_size[j]:                                                <L 1163>
            var_82 = wp::extract(var_42, var_70);
            var_83 = wp::extract(var_box_size, var_70);
            var_84 = (var_82 > var_83);
            if (var_84) {
                // n_out += 1                                                                 <L 1164>
                var_86 = wp::add(var_80, var_85);
                // ax_out = j                                                                 <L 1165>
                var_87 = wp::copy(var_70);
                // boxPoint[j] = box_size[j]                                                  <L 1166>
                var_88 = wp::extract(var_box_size, var_70);
                wp::assign_inplace(var_42, var_70, var_88);
            }
            var_89 = wp::where(var_84, var_86, var_80);
            var_90 = wp::where(var_84, var_87, var_81);
        }
        var_91 = wp::where(var_74, var_80, var_89);
        var_92 = wp::where(var_74, var_81, var_90);
        // if boxPoint[j] < -box_size[j]:                                                     <L 1159>
        var_94 = wp::extract(var_42, var_93);
        var_95 = wp::extract(var_box_size, var_93);
        var_96 = wp::neg(var_95);
        var_97 = (var_94 < var_96);
        if (var_97) {
            // n_out += 1                                                                     <L 1160>
            var_99 = wp::add(var_91, var_98);
            // ax_out = j                                                                     <L 1161>
            var_100 = wp::copy(var_93);
            // boxPoint[j] = -box_size[j]                                                     <L 1162>
            var_101 = wp::extract(var_box_size, var_93);
            var_102 = wp::neg(var_101);
            wp::assign_inplace(var_42, var_93, var_102);
        }
        var_103 = wp::where(var_97, var_99, var_91);
        var_104 = wp::where(var_97, var_100, var_92);
        if (!var_97) {
            // elif boxPoint[j] > box_size[j]:                                                <L 1163>
            var_105 = wp::extract(var_42, var_93);
            var_106 = wp::extract(var_box_size, var_93);
            var_107 = (var_105 > var_106);
            if (var_107) {
                // n_out += 1                                                                 <L 1164>
                var_109 = wp::add(var_103, var_108);
                // ax_out = j                                                                 <L 1165>
                var_110 = wp::copy(var_93);
                // boxPoint[j] = box_size[j]                                                  <L 1166>
                var_111 = wp::extract(var_box_size, var_93);
                wp::assign_inplace(var_42, var_93, var_111);
            }
            var_112 = wp::where(var_107, var_109, var_103);
            var_113 = wp::where(var_107, var_110, var_104);
        }
        var_114 = wp::where(var_97, var_103, var_112);
        var_115 = wp::where(var_97, var_104, var_113);
        // if n_out > 1:                                                                      <L 1168>
        var_117 = (var_114 > var_116);
        if (var_117) {
            // continue                                                                       <L 1169>
            goto start_for_0;
        }
        // dist = wp.length_sq(boxPoint - axisTip)                                            <L 1171>
        var_118 = wp::sub(var_42, var_41);
        var_119 = wp::length_sq(var_118);
        // if dist < bestdist:                                                                <L 1173>
        var_120 = (var_119 < var_27);
        if (var_120) {
            // bestdist = dist                                                                <L 1174>
            var_121 = wp::copy(var_119);
            // bestsegmentpos = wp.float32(i)                                                 <L 1175>
            var_122 = wp::float32(var_38);
            // cltype = -2 + i                                                                <L 1176>
            var_124 = wp::add(var_123, var_38);
            // clface = ax_out                                                                <L 1177>
            var_125 = wp::copy(var_115);
        }
        var_126 = wp::where(var_120, var_121, var_27);
        var_127 = wp::where(var_120, var_122, var_29);
        var_128 = wp::where(var_120, var_124, var_31);
        var_129 = wp::where(var_120, var_125, var_33);
        wp::assign(var_27, var_126);
        wp::assign(var_29, var_127);
        wp::assign(var_31, var_128);
        wp::assign(var_33, var_129);
        goto start_for_0;
    end_for_0:;
    // clcorner = wp.int32(-123)  # which corner is the closest                               <L 1180>
    var_131 = wp::int32(var_130);
    // cledge = wp.int32(-123)  # which axis                                                  <L 1181>
    var_133 = wp::int32(var_132);
    // bestboxpos = wp.float32(0.0)                                                           <L 1182>
    var_135 = wp::float32(var_134);
    // for i in range(8):                                                                     <L 1184>
    // for j in range(3):                                                                     <L 1185>
    var_138 = wp::range(var_137);
    start_for_2:;
        if (iter_cmp(var_138) == 0) goto end_for_2;
        var_139 = wp::iter_next(var_138);
        // if i & (1 << j) != 0:                                                              <L 1186>
        var_141 = wp::lshift(var_140, var_139);
        var_142 = wp::bit_and(var_136, var_141);
        var_144 = (var_142 != var_143);
        if (var_144) {
            // continue                                                                       <L 1187>
            goto start_for_2;
        }
        // c2 = wp.int32(-123)                                                                <L 1189>
        var_146 = wp::int32(var_145);
        // box_pt = wp.cw_mul(                                                                <L 1192>
        // wp.vec3(                                                                           <L 1193>
        // wp.where(i & 1, 1.0, -1.0),                                                        <L 1194>
        var_148 = wp::bit_and(var_136, var_147);
        var_151 = wp::where(var_148, var_149, var_150);
        // wp.where(i & 2, 1.0, -1.0),                                                        <L 1195>
        var_153 = wp::bit_and(var_136, var_152);
        var_156 = wp::where(var_153, var_154, var_155);
        // wp.where(i & 4, 1.0, -1.0),                                                        <L 1196>
        var_158 = wp::bit_and(var_136, var_157);
        var_161 = wp::where(var_158, var_159, var_160);
        var_162 = wp::vec_t<3, wp::float32>(var_151, var_156, var_161);
        // box_size,                                                                          <L 1198>
        var_163 = wp::cw_mul(var_162, var_box_size);
        // box_pt[j] = 0.0                                                                    <L 1200>
        wp::assign_inplace(var_163, var_139, var_164);
        // dif = box_pt - pos                                                                 <L 1203>
        var_165 = wp::sub(var_163, var_2);
        // u = -box_size[j] * dif[j]                                                          <L 1205>
        var_166 = wp::extract(var_box_size, var_139);
        var_167 = wp::neg(var_166);
        var_168 = wp::extract(var_165, var_139);
        var_169 = wp::mul(var_167, var_168);
        // v = wp.dot(halfaxis, dif)                                                          <L 1206>
        var_170 = wp::dot(var_4, var_165);
        // ma = box_size[j] * box_size[j]                                                     <L 1207>
        var_171 = wp::extract(var_box_size, var_139);
        var_172 = wp::extract(var_box_size, var_139);
        var_173 = wp::mul(var_171, var_172);
        // mb = -box_size[j] * halfaxis[j]                                                    <L 1208>
        var_174 = wp::extract(var_box_size, var_139);
        var_175 = wp::neg(var_174);
        var_176 = wp::extract(var_4, var_139);
        var_177 = wp::mul(var_175, var_176);
        // mc = capsule_half_length * capsule_half_length                                     <L 1209>
        var_178 = wp::mul(var_capsule_half_length, var_capsule_half_length);
        // det = ma * mc - mb * mb                                                            <L 1210>
        var_179 = wp::mul(var_173, var_178);
        var_180 = wp::mul(var_177, var_177);
        var_181 = wp::sub(var_179, var_180);
        // if wp.abs(det) < MJ_MINVAL:                                                        <L 1211>
        var_182 = wp::abs(var_181);
        var_184 = (var_182 < var_183);
        if (var_184) {
            // continue                                                                       <L 1212>
            goto start_for_2;
        }
        // idet = 1.0 / det                                                                   <L 1214>
        var_186 = wp::div(var_185, var_181);
        // x1 = wp.float32((mc * u - mb * v) * idet)                                          <L 1217>
        var_187 = wp::mul(var_178, var_169);
        var_188 = wp::mul(var_177, var_170);
        var_189 = wp::sub(var_187, var_188);
        var_190 = wp::mul(var_189, var_186);
        var_191 = wp::float32(var_190);
        // x2 = wp.float32((ma * v - mb * u) * idet)                                          <L 1218>
        var_192 = wp::mul(var_173, var_170);
        var_193 = wp::mul(var_177, var_169);
        var_194 = wp::sub(var_192, var_193);
        var_195 = wp::mul(var_194, var_186);
        var_196 = wp::float32(var_195);
        // s1 = wp.int32(1)                                                                   <L 1220>
        var_198 = wp::int32(var_197);
        // s2 = wp.int32(1)                                                                   <L 1221>
        var_200 = wp::int32(var_199);
        // if x1 > 1:                                                                         <L 1223>
        var_202 = (var_191 > var_201);
        if (var_202) {
            // x1 = 1.0                                                                       <L 1224>
            // s1 = 2                                                                         <L 1225>
            // x2 = safe_div(v - mb, mc)                                                      <L 1226>
            var_205 = wp::sub(var_170, var_177);
            var_206 = safe_div_0(var_205, var_178);
        }
        var_207 = wp::where(var_202, var_203, var_191);
        var_208 = wp::where(var_202, var_206, var_196);
        var_209 = wp::where(var_202, var_204, var_198);
        if (!var_202) {
            // elif x1 < -1:                                                                  <L 1227>
            var_211 = (var_207 < var_210);
            if (var_211) {
                // x1 = -1.0                                                                  <L 1228>
                // s1 = 0                                                                     <L 1229>
                // x2 = safe_div(v + mb, mc)                                                  <L 1230>
                var_214 = wp::add(var_170, var_177);
                var_215 = safe_div_0(var_214, var_178);
            }
            var_216 = wp::where(var_211, var_212, var_207);
            var_217 = wp::where(var_211, var_215, var_208);
            var_218 = wp::where(var_211, var_213, var_209);
        }
        var_219 = wp::where(var_202, var_207, var_216);
        var_220 = wp::where(var_202, var_208, var_217);
        var_221 = wp::where(var_202, var_209, var_218);
        // x2_over = x2 > 1.0                                                                 <L 1232>
        var_223 = (var_220 > var_222);
        // if x2_over or x2 < -1.0:                                                           <L 1233>
        var_224 = var_223;
        if (!var_224) {
            var_226 = (var_220 < var_225);
            var_224 = var_224 || var_226;
        }
        if (var_224) {
            // if x2_over:                                                                    <L 1234>
            if (var_223) {
                // x2 = 1.0                                                                   <L 1235>
                // s2 = 2                                                                     <L 1236>
                // x1 = safe_div(u - mb, ma)                                                  <L 1237>
                var_229 = wp::sub(var_169, var_177);
                var_230 = safe_div_0(var_229, var_173);
            }
            var_231 = wp::where(var_223, var_230, var_219);
            var_232 = wp::where(var_223, var_227, var_220);
            var_233 = wp::where(var_223, var_228, var_200);
            if (!var_223) {
                // x2 = -1.0                                                                  <L 1239>
                // s2 = 0                                                                     <L 1240>
                // x1 = safe_div(u + mb, ma)                                                  <L 1241>
                var_236 = wp::add(var_169, var_177);
                var_237 = safe_div_0(var_236, var_173);
            }
            var_238 = wp::where(var_223, var_231, var_237);
            var_239 = wp::where(var_223, var_232, var_234);
            var_240 = wp::where(var_223, var_233, var_235);
            // if x1 > 1:                                                                     <L 1243>
            var_242 = (var_238 > var_241);
            if (var_242) {
                // x1 = 1.0                                                                   <L 1244>
                // s1 = 2                                                                     <L 1245>
            }
            var_245 = wp::where(var_242, var_243, var_238);
            var_246 = wp::where(var_242, var_244, var_221);
            if (!var_242) {
                // elif x1 < -1:                                                              <L 1246>
                var_248 = (var_245 < var_247);
                if (var_248) {
                    // x1 = -1.0                                                              <L 1247>
                    // s1 = 0                                                                 <L 1248>
                }
                var_251 = wp::where(var_248, var_249, var_245);
                var_252 = wp::where(var_248, var_250, var_246);
            }
            var_253 = wp::where(var_242, var_245, var_251);
            var_254 = wp::where(var_242, var_246, var_252);
        }
        var_255 = wp::where(var_224, var_253, var_219);
        var_256 = wp::where(var_224, var_239, var_220);
        var_257 = wp::where(var_224, var_254, var_221);
        var_258 = wp::where(var_224, var_240, var_200);
        // dif -= halfaxis * x2                                                               <L 1250>
        var_259 = wp::mul(var_4, var_256);
        var_260 = wp::sub(var_165, var_259);
        // dif[j] += box_size[j] * x1                                                         <L 1251>
        var_261 = wp::extract(var_box_size, var_139);
        var_262 = wp::mul(var_261, var_255);
        wp::add_inplace(var_260, var_139, var_262);
        // ct = s1 * 3 + s2                                                                   <L 1254>
        var_264 = wp::mul(var_257, var_263);
        var_265 = wp::add(var_264, var_258);
        // dif_sq = wp.length_sq(dif)                                                         <L 1256>
        var_266 = wp::length_sq(var_260);
        // if dif_sq < bestdist - MJ_MINVAL:                                                  <L 1257>
        var_267 = wp::sub(var_27, var_183);
        var_268 = (var_266 < var_267);
        if (var_268) {
            // bestdist = dif_sq                                                              <L 1258>
            var_269 = wp::copy(var_266);
            // bestsegmentpos = x2                                                            <L 1259>
            var_270 = wp::copy(var_256);
            // bestboxpos = x1                                                                <L 1260>
            var_271 = wp::copy(var_255);
            // c2 = ct // 6                                                                   <L 1262>
            var_273 = wp::floordiv(var_265, var_272);
            // clcorner = i + (1 << j) * c2  # index of closest box corner                    <L 1264>
            var_275 = wp::lshift(var_274, var_139);
            var_276 = wp::mul(var_275, var_273);
            var_277 = wp::add(var_136, var_276);
            // cledge = j  # axis index of closest box edge                                   <L 1265>
            var_278 = wp::copy(var_139);
            // cltype = ct  # encoded collision configuration                                 <L 1266>
            var_279 = wp::copy(var_265);
        }
        var_280 = wp::where(var_268, var_269, var_27);
        var_281 = wp::where(var_268, var_270, var_29);
        var_282 = wp::where(var_268, var_279, var_31);
        var_283 = wp::where(var_268, var_277, var_131);
        var_284 = wp::where(var_268, var_278, var_133);
        var_285 = wp::where(var_268, var_271, var_135);
        var_286 = wp::where(var_268, var_273, var_146);
        wp::assign(var_27, var_280);
        wp::assign(var_29, var_281);
        wp::assign(var_31, var_282);
        wp::assign(var_131, var_283);
        wp::assign(var_133, var_284);
        wp::assign(var_135, var_285);
        goto start_for_2;
    end_for_2:;
    // for j in range(3):                                                                     <L 1185>
    var_289 = wp::range(var_288);
    start_for_4:;
        if (iter_cmp(var_289) == 0) goto end_for_4;
        var_290 = wp::iter_next(var_289);
        // if i & (1 << j) != 0:                                                              <L 1186>
        var_292 = wp::lshift(var_291, var_290);
        var_293 = wp::bit_and(var_287, var_292);
        var_295 = (var_293 != var_294);
        if (var_295) {
            // continue                                                                       <L 1187>
            goto start_for_4;
        }
        // c2 = wp.int32(-123)                                                                <L 1189>
        var_297 = wp::int32(var_296);
        // box_pt = wp.cw_mul(                                                                <L 1192>
        // wp.vec3(                                                                           <L 1193>
        // wp.where(i & 1, 1.0, -1.0),                                                        <L 1194>
        var_299 = wp::bit_and(var_287, var_298);
        var_302 = wp::where(var_299, var_300, var_301);
        // wp.where(i & 2, 1.0, -1.0),                                                        <L 1195>
        var_304 = wp::bit_and(var_287, var_303);
        var_307 = wp::where(var_304, var_305, var_306);
        // wp.where(i & 4, 1.0, -1.0),                                                        <L 1196>
        var_309 = wp::bit_and(var_287, var_308);
        var_312 = wp::where(var_309, var_310, var_311);
        var_313 = wp::vec_t<3, wp::float32>(var_302, var_307, var_312);
        // box_size,                                                                          <L 1198>
        var_314 = wp::cw_mul(var_313, var_box_size);
        // box_pt[j] = 0.0                                                                    <L 1200>
        wp::assign_inplace(var_314, var_290, var_315);
        // dif = box_pt - pos                                                                 <L 1203>
        var_316 = wp::sub(var_314, var_2);
        // u = -box_size[j] * dif[j]                                                          <L 1205>
        var_317 = wp::extract(var_box_size, var_290);
        var_318 = wp::neg(var_317);
        var_319 = wp::extract(var_316, var_290);
        var_320 = wp::mul(var_318, var_319);
        // v = wp.dot(halfaxis, dif)                                                          <L 1206>
        var_321 = wp::dot(var_4, var_316);
        // ma = box_size[j] * box_size[j]                                                     <L 1207>
        var_322 = wp::extract(var_box_size, var_290);
        var_323 = wp::extract(var_box_size, var_290);
        var_324 = wp::mul(var_322, var_323);
        // mb = -box_size[j] * halfaxis[j]                                                    <L 1208>
        var_325 = wp::extract(var_box_size, var_290);
        var_326 = wp::neg(var_325);
        var_327 = wp::extract(var_4, var_290);
        var_328 = wp::mul(var_326, var_327);
        // mc = capsule_half_length * capsule_half_length                                     <L 1209>
        var_329 = wp::mul(var_capsule_half_length, var_capsule_half_length);
        // det = ma * mc - mb * mb                                                            <L 1210>
        var_330 = wp::mul(var_324, var_329);
        var_331 = wp::mul(var_328, var_328);
        var_332 = wp::sub(var_330, var_331);
        // if wp.abs(det) < MJ_MINVAL:                                                        <L 1211>
        var_333 = wp::abs(var_332);
        var_334 = (var_333 < var_183);
        if (var_334) {
            // continue                                                                       <L 1212>
            wp::assign(var_286, var_297);
            wp::assign(var_163, var_314);
            wp::assign(var_260, var_316);
            wp::assign(var_169, var_320);
            wp::assign(var_170, var_321);
            wp::assign(var_173, var_324);
            wp::assign(var_177, var_328);
            wp::assign(var_178, var_329);
            wp::assign(var_181, var_332);
            goto start_for_4;
        }
        var_335 = wp::where(var_334, var_286, var_297);
        var_336 = wp::where(var_334, var_163, var_314);
        var_337 = wp::where(var_334, var_260, var_316);
        var_338 = wp::where(var_334, var_169, var_320);
        var_339 = wp::where(var_334, var_170, var_321);
        var_340 = wp::where(var_334, var_173, var_324);
        var_341 = wp::where(var_334, var_177, var_328);
        var_342 = wp::where(var_334, var_178, var_329);
        var_343 = wp::where(var_334, var_181, var_332);
        // idet = 1.0 / det                                                                   <L 1214>
        var_345 = wp::div(var_344, var_343);
        // x1 = wp.float32((mc * u - mb * v) * idet)                                          <L 1217>
        var_346 = wp::mul(var_342, var_338);
        var_347 = wp::mul(var_341, var_339);
        var_348 = wp::sub(var_346, var_347);
        var_349 = wp::mul(var_348, var_345);
        var_350 = wp::float32(var_349);
        // x2 = wp.float32((ma * v - mb * u) * idet)                                          <L 1218>
        var_351 = wp::mul(var_340, var_339);
        var_352 = wp::mul(var_341, var_338);
        var_353 = wp::sub(var_351, var_352);
        var_354 = wp::mul(var_353, var_345);
        var_355 = wp::float32(var_354);
        // s1 = wp.int32(1)                                                                   <L 1220>
        var_357 = wp::int32(var_356);
        // s2 = wp.int32(1)                                                                   <L 1221>
        var_359 = wp::int32(var_358);
        // if x1 > 1:                                                                         <L 1223>
        var_361 = (var_350 > var_360);
        if (var_361) {
            // x1 = 1.0                                                                       <L 1224>
            // s1 = 2                                                                         <L 1225>
            // x2 = safe_div(v - mb, mc)                                                      <L 1226>
            var_364 = wp::sub(var_339, var_341);
            var_365 = safe_div_0(var_364, var_342);
        }
        var_366 = wp::where(var_361, var_362, var_350);
        var_367 = wp::where(var_361, var_365, var_355);
        var_368 = wp::where(var_361, var_363, var_357);
        if (!var_361) {
            // elif x1 < -1:                                                                  <L 1227>
            var_370 = (var_366 < var_369);
            if (var_370) {
                // x1 = -1.0                                                                  <L 1228>
                // s1 = 0                                                                     <L 1229>
                // x2 = safe_div(v + mb, mc)                                                  <L 1230>
                var_373 = wp::add(var_339, var_341);
                var_374 = safe_div_0(var_373, var_342);
            }
            var_375 = wp::where(var_370, var_371, var_366);
            var_376 = wp::where(var_370, var_374, var_367);
            var_377 = wp::where(var_370, var_372, var_368);
        }
        var_378 = wp::where(var_361, var_366, var_375);
        var_379 = wp::where(var_361, var_367, var_376);
        var_380 = wp::where(var_361, var_368, var_377);
        // x2_over = x2 > 1.0                                                                 <L 1232>
        var_382 = (var_379 > var_381);
        // if x2_over or x2 < -1.0:                                                           <L 1233>
        var_383 = var_382;
        if (!var_383) {
            var_385 = (var_379 < var_384);
            var_383 = var_383 || var_385;
        }
        if (var_383) {
            // if x2_over:                                                                    <L 1234>
            if (var_382) {
                // x2 = 1.0                                                                   <L 1235>
                // s2 = 2                                                                     <L 1236>
                // x1 = safe_div(u - mb, ma)                                                  <L 1237>
                var_388 = wp::sub(var_338, var_341);
                var_389 = safe_div_0(var_388, var_340);
            }
            var_390 = wp::where(var_382, var_389, var_378);
            var_391 = wp::where(var_382, var_386, var_379);
            var_392 = wp::where(var_382, var_387, var_359);
            if (!var_382) {
                // x2 = -1.0                                                                  <L 1239>
                // s2 = 0                                                                     <L 1240>
                // x1 = safe_div(u + mb, ma)                                                  <L 1241>
                var_395 = wp::add(var_338, var_341);
                var_396 = safe_div_0(var_395, var_340);
            }
            var_397 = wp::where(var_382, var_390, var_396);
            var_398 = wp::where(var_382, var_391, var_393);
            var_399 = wp::where(var_382, var_392, var_394);
            // if x1 > 1:                                                                     <L 1243>
            var_401 = (var_397 > var_400);
            if (var_401) {
                // x1 = 1.0                                                                   <L 1244>
                // s1 = 2                                                                     <L 1245>
            }
            var_404 = wp::where(var_401, var_402, var_397);
            var_405 = wp::where(var_401, var_403, var_380);
            if (!var_401) {
                // elif x1 < -1:                                                              <L 1246>
                var_407 = (var_404 < var_406);
                if (var_407) {
                    // x1 = -1.0                                                              <L 1247>
                    // s1 = 0                                                                 <L 1248>
                }
                var_410 = wp::where(var_407, var_408, var_404);
                var_411 = wp::where(var_407, var_409, var_405);
            }
            var_412 = wp::where(var_401, var_404, var_410);
            var_413 = wp::where(var_401, var_405, var_411);
        }
        var_414 = wp::where(var_383, var_412, var_378);
        var_415 = wp::where(var_383, var_398, var_379);
        var_416 = wp::where(var_383, var_413, var_380);
        var_417 = wp::where(var_383, var_399, var_359);
        // dif -= halfaxis * x2                                                               <L 1250>
        var_418 = wp::mul(var_4, var_415);
        var_419 = wp::sub(var_337, var_418);
        // dif[j] += box_size[j] * x1                                                         <L 1251>
        var_420 = wp::extract(var_box_size, var_290);
        var_421 = wp::mul(var_420, var_414);
        wp::add_inplace(var_419, var_290, var_421);
        // ct = s1 * 3 + s2                                                                   <L 1254>
        var_423 = wp::mul(var_416, var_422);
        var_424 = wp::add(var_423, var_417);
        // dif_sq = wp.length_sq(dif)                                                         <L 1256>
        var_425 = wp::length_sq(var_419);
        // if dif_sq < bestdist - MJ_MINVAL:                                                  <L 1257>
        var_426 = wp::sub(var_27, var_183);
        var_427 = (var_425 < var_426);
        if (var_427) {
            // bestdist = dif_sq                                                              <L 1258>
            var_428 = wp::copy(var_425);
            // bestsegmentpos = x2                                                            <L 1259>
            var_429 = wp::copy(var_415);
            // bestboxpos = x1                                                                <L 1260>
            var_430 = wp::copy(var_414);
            // c2 = ct // 6                                                                   <L 1262>
            var_432 = wp::floordiv(var_424, var_431);
            // clcorner = i + (1 << j) * c2  # index of closest box corner                    <L 1264>
            var_434 = wp::lshift(var_433, var_290);
            var_435 = wp::mul(var_434, var_432);
            var_436 = wp::add(var_287, var_435);
            // cledge = j  # axis index of closest box edge                                   <L 1265>
            var_437 = wp::copy(var_290);
            // cltype = ct  # encoded collision configuration                                 <L 1266>
            var_438 = wp::copy(var_424);
        }
        var_439 = wp::where(var_427, var_428, var_27);
        var_440 = wp::where(var_427, var_429, var_29);
        var_441 = wp::where(var_427, var_438, var_31);
        var_442 = wp::where(var_427, var_436, var_131);
        var_443 = wp::where(var_427, var_437, var_133);
        var_444 = wp::where(var_427, var_430, var_135);
        var_445 = wp::where(var_427, var_432, var_335);
        wp::assign(var_27, var_439);
        wp::assign(var_29, var_440);
        wp::assign(var_31, var_441);
        wp::assign(var_131, var_442);
        wp::assign(var_133, var_443);
        wp::assign(var_135, var_444);
        wp::assign(var_286, var_445);
        wp::assign(var_163, var_336);
        wp::assign(var_260, var_419);
        wp::assign(var_169, var_338);
        wp::assign(var_170, var_339);
        wp::assign(var_173, var_340);
        wp::assign(var_177, var_341);
        wp::assign(var_178, var_342);
        wp::assign(var_181, var_343);
        wp::assign(var_186, var_345);
        wp::assign(var_255, var_414);
        wp::assign(var_256, var_415);
        wp::assign(var_257, var_416);
        wp::assign(var_258, var_417);
        wp::assign(var_223, var_382);
        wp::assign(var_265, var_424);
        wp::assign(var_266, var_425);
        goto start_for_4;
    end_for_4:;
    // for j in range(3):                                                                     <L 1185>
    var_448 = wp::range(var_447);
    start_for_6:;
        if (iter_cmp(var_448) == 0) goto end_for_6;
        var_449 = wp::iter_next(var_448);
        // if i & (1 << j) != 0:                                                              <L 1186>
        var_451 = wp::lshift(var_450, var_449);
        var_452 = wp::bit_and(var_446, var_451);
        var_454 = (var_452 != var_453);
        if (var_454) {
            // continue                                                                       <L 1187>
            goto start_for_6;
        }
        // c2 = wp.int32(-123)                                                                <L 1189>
        var_456 = wp::int32(var_455);
        // box_pt = wp.cw_mul(                                                                <L 1192>
        // wp.vec3(                                                                           <L 1193>
        // wp.where(i & 1, 1.0, -1.0),                                                        <L 1194>
        var_458 = wp::bit_and(var_446, var_457);
        var_461 = wp::where(var_458, var_459, var_460);
        // wp.where(i & 2, 1.0, -1.0),                                                        <L 1195>
        var_463 = wp::bit_and(var_446, var_462);
        var_466 = wp::where(var_463, var_464, var_465);
        // wp.where(i & 4, 1.0, -1.0),                                                        <L 1196>
        var_468 = wp::bit_and(var_446, var_467);
        var_471 = wp::where(var_468, var_469, var_470);
        var_472 = wp::vec_t<3, wp::float32>(var_461, var_466, var_471);
        // box_size,                                                                          <L 1198>
        var_473 = wp::cw_mul(var_472, var_box_size);
        // box_pt[j] = 0.0                                                                    <L 1200>
        wp::assign_inplace(var_473, var_449, var_474);
        // dif = box_pt - pos                                                                 <L 1203>
        var_475 = wp::sub(var_473, var_2);
        // u = -box_size[j] * dif[j]                                                          <L 1205>
        var_476 = wp::extract(var_box_size, var_449);
        var_477 = wp::neg(var_476);
        var_478 = wp::extract(var_475, var_449);
        var_479 = wp::mul(var_477, var_478);
        // v = wp.dot(halfaxis, dif)                                                          <L 1206>
        var_480 = wp::dot(var_4, var_475);
        // ma = box_size[j] * box_size[j]                                                     <L 1207>
        var_481 = wp::extract(var_box_size, var_449);
        var_482 = wp::extract(var_box_size, var_449);
        var_483 = wp::mul(var_481, var_482);
        // mb = -box_size[j] * halfaxis[j]                                                    <L 1208>
        var_484 = wp::extract(var_box_size, var_449);
        var_485 = wp::neg(var_484);
        var_486 = wp::extract(var_4, var_449);
        var_487 = wp::mul(var_485, var_486);
        // mc = capsule_half_length * capsule_half_length                                     <L 1209>
        var_488 = wp::mul(var_capsule_half_length, var_capsule_half_length);
        // det = ma * mc - mb * mb                                                            <L 1210>
        var_489 = wp::mul(var_483, var_488);
        var_490 = wp::mul(var_487, var_487);
        var_491 = wp::sub(var_489, var_490);
        // if wp.abs(det) < MJ_MINVAL:                                                        <L 1211>
        var_492 = wp::abs(var_491);
        var_493 = (var_492 < var_183);
        if (var_493) {
            // continue                                                                       <L 1212>
            wp::assign(var_286, var_456);
            wp::assign(var_163, var_473);
            wp::assign(var_260, var_475);
            wp::assign(var_169, var_479);
            wp::assign(var_170, var_480);
            wp::assign(var_173, var_483);
            wp::assign(var_177, var_487);
            wp::assign(var_178, var_488);
            wp::assign(var_181, var_491);
            goto start_for_6;
        }
        var_494 = wp::where(var_493, var_286, var_456);
        var_495 = wp::where(var_493, var_163, var_473);
        var_496 = wp::where(var_493, var_260, var_475);
        var_497 = wp::where(var_493, var_169, var_479);
        var_498 = wp::where(var_493, var_170, var_480);
        var_499 = wp::where(var_493, var_173, var_483);
        var_500 = wp::where(var_493, var_177, var_487);
        var_501 = wp::where(var_493, var_178, var_488);
        var_502 = wp::where(var_493, var_181, var_491);
        // idet = 1.0 / det                                                                   <L 1214>
        var_504 = wp::div(var_503, var_502);
        // x1 = wp.float32((mc * u - mb * v) * idet)                                          <L 1217>
        var_505 = wp::mul(var_501, var_497);
        var_506 = wp::mul(var_500, var_498);
        var_507 = wp::sub(var_505, var_506);
        var_508 = wp::mul(var_507, var_504);
        var_509 = wp::float32(var_508);
        // x2 = wp.float32((ma * v - mb * u) * idet)                                          <L 1218>
        var_510 = wp::mul(var_499, var_498);
        var_511 = wp::mul(var_500, var_497);
        var_512 = wp::sub(var_510, var_511);
        var_513 = wp::mul(var_512, var_504);
        var_514 = wp::float32(var_513);
        // s1 = wp.int32(1)                                                                   <L 1220>
        var_516 = wp::int32(var_515);
        // s2 = wp.int32(1)                                                                   <L 1221>
        var_518 = wp::int32(var_517);
        // if x1 > 1:                                                                         <L 1223>
        var_520 = (var_509 > var_519);
        if (var_520) {
            // x1 = 1.0                                                                       <L 1224>
            // s1 = 2                                                                         <L 1225>
            // x2 = safe_div(v - mb, mc)                                                      <L 1226>
            var_523 = wp::sub(var_498, var_500);
            var_524 = safe_div_0(var_523, var_501);
        }
        var_525 = wp::where(var_520, var_521, var_509);
        var_526 = wp::where(var_520, var_524, var_514);
        var_527 = wp::where(var_520, var_522, var_516);
        if (!var_520) {
            // elif x1 < -1:                                                                  <L 1227>
            var_529 = (var_525 < var_528);
            if (var_529) {
                // x1 = -1.0                                                                  <L 1228>
                // s1 = 0                                                                     <L 1229>
                // x2 = safe_div(v + mb, mc)                                                  <L 1230>
                var_532 = wp::add(var_498, var_500);
                var_533 = safe_div_0(var_532, var_501);
            }
            var_534 = wp::where(var_529, var_530, var_525);
            var_535 = wp::where(var_529, var_533, var_526);
            var_536 = wp::where(var_529, var_531, var_527);
        }
        var_537 = wp::where(var_520, var_525, var_534);
        var_538 = wp::where(var_520, var_526, var_535);
        var_539 = wp::where(var_520, var_527, var_536);
        // x2_over = x2 > 1.0                                                                 <L 1232>
        var_541 = (var_538 > var_540);
        // if x2_over or x2 < -1.0:                                                           <L 1233>
        var_542 = var_541;
        if (!var_542) {
            var_544 = (var_538 < var_543);
            var_542 = var_542 || var_544;
        }
        if (var_542) {
            // if x2_over:                                                                    <L 1234>
            if (var_541) {
                // x2 = 1.0                                                                   <L 1235>
                // s2 = 2                                                                     <L 1236>
                // x1 = safe_div(u - mb, ma)                                                  <L 1237>
                var_547 = wp::sub(var_497, var_500);
                var_548 = safe_div_0(var_547, var_499);
            }
            var_549 = wp::where(var_541, var_548, var_537);
            var_550 = wp::where(var_541, var_545, var_538);
            var_551 = wp::where(var_541, var_546, var_518);
            if (!var_541) {
                // x2 = -1.0                                                                  <L 1239>
                // s2 = 0                                                                     <L 1240>
                // x1 = safe_div(u + mb, ma)                                                  <L 1241>
                var_554 = wp::add(var_497, var_500);
                var_555 = safe_div_0(var_554, var_499);
            }
            var_556 = wp::where(var_541, var_549, var_555);
            var_557 = wp::where(var_541, var_550, var_552);
            var_558 = wp::where(var_541, var_551, var_553);
            // if x1 > 1:                                                                     <L 1243>
            var_560 = (var_556 > var_559);
            if (var_560) {
                // x1 = 1.0                                                                   <L 1244>
                // s1 = 2                                                                     <L 1245>
            }
            var_563 = wp::where(var_560, var_561, var_556);
            var_564 = wp::where(var_560, var_562, var_539);
            if (!var_560) {
                // elif x1 < -1:                                                              <L 1246>
                var_566 = (var_563 < var_565);
                if (var_566) {
                    // x1 = -1.0                                                              <L 1247>
                    // s1 = 0                                                                 <L 1248>
                }
                var_569 = wp::where(var_566, var_567, var_563);
                var_570 = wp::where(var_566, var_568, var_564);
            }
            var_571 = wp::where(var_560, var_563, var_569);
            var_572 = wp::where(var_560, var_564, var_570);
        }
        var_573 = wp::where(var_542, var_571, var_537);
        var_574 = wp::where(var_542, var_557, var_538);
        var_575 = wp::where(var_542, var_572, var_539);
        var_576 = wp::where(var_542, var_558, var_518);
        // dif -= halfaxis * x2                                                               <L 1250>
        var_577 = wp::mul(var_4, var_574);
        var_578 = wp::sub(var_496, var_577);
        // dif[j] += box_size[j] * x1                                                         <L 1251>
        var_579 = wp::extract(var_box_size, var_449);
        var_580 = wp::mul(var_579, var_573);
        wp::add_inplace(var_578, var_449, var_580);
        // ct = s1 * 3 + s2                                                                   <L 1254>
        var_582 = wp::mul(var_575, var_581);
        var_583 = wp::add(var_582, var_576);
        // dif_sq = wp.length_sq(dif)                                                         <L 1256>
        var_584 = wp::length_sq(var_578);
        // if dif_sq < bestdist - MJ_MINVAL:                                                  <L 1257>
        var_585 = wp::sub(var_27, var_183);
        var_586 = (var_584 < var_585);
        if (var_586) {
            // bestdist = dif_sq                                                              <L 1258>
            var_587 = wp::copy(var_584);
            // bestsegmentpos = x2                                                            <L 1259>
            var_588 = wp::copy(var_574);
            // bestboxpos = x1                                                                <L 1260>
            var_589 = wp::copy(var_573);
            // c2 = ct // 6                                                                   <L 1262>
            var_591 = wp::floordiv(var_583, var_590);
            // clcorner = i + (1 << j) * c2  # index of closest box corner                    <L 1264>
            var_593 = wp::lshift(var_592, var_449);
            var_594 = wp::mul(var_593, var_591);
            var_595 = wp::add(var_446, var_594);
            // cledge = j  # axis index of closest box edge                                   <L 1265>
            var_596 = wp::copy(var_449);
            // cltype = ct  # encoded collision configuration                                 <L 1266>
            var_597 = wp::copy(var_583);
        }
        var_598 = wp::where(var_586, var_587, var_27);
        var_599 = wp::where(var_586, var_588, var_29);
        var_600 = wp::where(var_586, var_597, var_31);
        var_601 = wp::where(var_586, var_595, var_131);
        var_602 = wp::where(var_586, var_596, var_133);
        var_603 = wp::where(var_586, var_589, var_135);
        var_604 = wp::where(var_586, var_591, var_494);
        wp::assign(var_27, var_598);
        wp::assign(var_29, var_599);
        wp::assign(var_31, var_600);
        wp::assign(var_131, var_601);
        wp::assign(var_133, var_602);
        wp::assign(var_135, var_603);
        wp::assign(var_286, var_604);
        wp::assign(var_163, var_495);
        wp::assign(var_260, var_578);
        wp::assign(var_169, var_497);
        wp::assign(var_170, var_498);
        wp::assign(var_173, var_499);
        wp::assign(var_177, var_500);
        wp::assign(var_178, var_501);
        wp::assign(var_181, var_502);
        wp::assign(var_186, var_504);
        wp::assign(var_255, var_573);
        wp::assign(var_256, var_574);
        wp::assign(var_257, var_575);
        wp::assign(var_258, var_576);
        wp::assign(var_223, var_541);
        wp::assign(var_265, var_583);
        wp::assign(var_266, var_584);
        goto start_for_6;
    end_for_6:;
    // for j in range(3):                                                                     <L 1185>
    var_607 = wp::range(var_606);
    start_for_8:;
        if (iter_cmp(var_607) == 0) goto end_for_8;
        var_608 = wp::iter_next(var_607);
        // if i & (1 << j) != 0:                                                              <L 1186>
        var_610 = wp::lshift(var_609, var_608);
        var_611 = wp::bit_and(var_605, var_610);
        var_613 = (var_611 != var_612);
        if (var_613) {
            // continue                                                                       <L 1187>
            goto start_for_8;
        }
        // c2 = wp.int32(-123)                                                                <L 1189>
        var_615 = wp::int32(var_614);
        // box_pt = wp.cw_mul(                                                                <L 1192>
        // wp.vec3(                                                                           <L 1193>
        // wp.where(i & 1, 1.0, -1.0),                                                        <L 1194>
        var_617 = wp::bit_and(var_605, var_616);
        var_620 = wp::where(var_617, var_618, var_619);
        // wp.where(i & 2, 1.0, -1.0),                                                        <L 1195>
        var_622 = wp::bit_and(var_605, var_621);
        var_625 = wp::where(var_622, var_623, var_624);
        // wp.where(i & 4, 1.0, -1.0),                                                        <L 1196>
        var_627 = wp::bit_and(var_605, var_626);
        var_630 = wp::where(var_627, var_628, var_629);
        var_631 = wp::vec_t<3, wp::float32>(var_620, var_625, var_630);
        // box_size,                                                                          <L 1198>
        var_632 = wp::cw_mul(var_631, var_box_size);
        // box_pt[j] = 0.0                                                                    <L 1200>
        wp::assign_inplace(var_632, var_608, var_633);
        // dif = box_pt - pos                                                                 <L 1203>
        var_634 = wp::sub(var_632, var_2);
        // u = -box_size[j] * dif[j]                                                          <L 1205>
        var_635 = wp::extract(var_box_size, var_608);
        var_636 = wp::neg(var_635);
        var_637 = wp::extract(var_634, var_608);
        var_638 = wp::mul(var_636, var_637);
        // v = wp.dot(halfaxis, dif)                                                          <L 1206>
        var_639 = wp::dot(var_4, var_634);
        // ma = box_size[j] * box_size[j]                                                     <L 1207>
        var_640 = wp::extract(var_box_size, var_608);
        var_641 = wp::extract(var_box_size, var_608);
        var_642 = wp::mul(var_640, var_641);
        // mb = -box_size[j] * halfaxis[j]                                                    <L 1208>
        var_643 = wp::extract(var_box_size, var_608);
        var_644 = wp::neg(var_643);
        var_645 = wp::extract(var_4, var_608);
        var_646 = wp::mul(var_644, var_645);
        // mc = capsule_half_length * capsule_half_length                                     <L 1209>
        var_647 = wp::mul(var_capsule_half_length, var_capsule_half_length);
        // det = ma * mc - mb * mb                                                            <L 1210>
        var_648 = wp::mul(var_642, var_647);
        var_649 = wp::mul(var_646, var_646);
        var_650 = wp::sub(var_648, var_649);
        // if wp.abs(det) < MJ_MINVAL:                                                        <L 1211>
        var_651 = wp::abs(var_650);
        var_652 = (var_651 < var_183);
        if (var_652) {
            // continue                                                                       <L 1212>
            wp::assign(var_286, var_615);
            wp::assign(var_163, var_632);
            wp::assign(var_260, var_634);
            wp::assign(var_169, var_638);
            wp::assign(var_170, var_639);
            wp::assign(var_173, var_642);
            wp::assign(var_177, var_646);
            wp::assign(var_178, var_647);
            wp::assign(var_181, var_650);
            goto start_for_8;
        }
        var_653 = wp::where(var_652, var_286, var_615);
        var_654 = wp::where(var_652, var_163, var_632);
        var_655 = wp::where(var_652, var_260, var_634);
        var_656 = wp::where(var_652, var_169, var_638);
        var_657 = wp::where(var_652, var_170, var_639);
        var_658 = wp::where(var_652, var_173, var_642);
        var_659 = wp::where(var_652, var_177, var_646);
        var_660 = wp::where(var_652, var_178, var_647);
        var_661 = wp::where(var_652, var_181, var_650);
        // idet = 1.0 / det                                                                   <L 1214>
        var_663 = wp::div(var_662, var_661);
        // x1 = wp.float32((mc * u - mb * v) * idet)                                          <L 1217>
        var_664 = wp::mul(var_660, var_656);
        var_665 = wp::mul(var_659, var_657);
        var_666 = wp::sub(var_664, var_665);
        var_667 = wp::mul(var_666, var_663);
        var_668 = wp::float32(var_667);
        // x2 = wp.float32((ma * v - mb * u) * idet)                                          <L 1218>
        var_669 = wp::mul(var_658, var_657);
        var_670 = wp::mul(var_659, var_656);
        var_671 = wp::sub(var_669, var_670);
        var_672 = wp::mul(var_671, var_663);
        var_673 = wp::float32(var_672);
        // s1 = wp.int32(1)                                                                   <L 1220>
        var_675 = wp::int32(var_674);
        // s2 = wp.int32(1)                                                                   <L 1221>
        var_677 = wp::int32(var_676);
        // if x1 > 1:                                                                         <L 1223>
        var_679 = (var_668 > var_678);
        if (var_679) {
            // x1 = 1.0                                                                       <L 1224>
            // s1 = 2                                                                         <L 1225>
            // x2 = safe_div(v - mb, mc)                                                      <L 1226>
            var_682 = wp::sub(var_657, var_659);
            var_683 = safe_div_0(var_682, var_660);
        }
        var_684 = wp::where(var_679, var_680, var_668);
        var_685 = wp::where(var_679, var_683, var_673);
        var_686 = wp::where(var_679, var_681, var_675);
        if (!var_679) {
            // elif x1 < -1:                                                                  <L 1227>
            var_688 = (var_684 < var_687);
            if (var_688) {
                // x1 = -1.0                                                                  <L 1228>
                // s1 = 0                                                                     <L 1229>
                // x2 = safe_div(v + mb, mc)                                                  <L 1230>
                var_691 = wp::add(var_657, var_659);
                var_692 = safe_div_0(var_691, var_660);
            }
            var_693 = wp::where(var_688, var_689, var_684);
            var_694 = wp::where(var_688, var_692, var_685);
            var_695 = wp::where(var_688, var_690, var_686);
        }
        var_696 = wp::where(var_679, var_684, var_693);
        var_697 = wp::where(var_679, var_685, var_694);
        var_698 = wp::where(var_679, var_686, var_695);
        // x2_over = x2 > 1.0                                                                 <L 1232>
        var_700 = (var_697 > var_699);
        // if x2_over or x2 < -1.0:                                                           <L 1233>
        var_701 = var_700;
        if (!var_701) {
            var_703 = (var_697 < var_702);
            var_701 = var_701 || var_703;
        }
        if (var_701) {
            // if x2_over:                                                                    <L 1234>
            if (var_700) {
                // x2 = 1.0                                                                   <L 1235>
                // s2 = 2                                                                     <L 1236>
                // x1 = safe_div(u - mb, ma)                                                  <L 1237>
                var_706 = wp::sub(var_656, var_659);
                var_707 = safe_div_0(var_706, var_658);
            }
            var_708 = wp::where(var_700, var_707, var_696);
            var_709 = wp::where(var_700, var_704, var_697);
            var_710 = wp::where(var_700, var_705, var_677);
            if (!var_700) {
                // x2 = -1.0                                                                  <L 1239>
                // s2 = 0                                                                     <L 1240>
                // x1 = safe_div(u + mb, ma)                                                  <L 1241>
                var_713 = wp::add(var_656, var_659);
                var_714 = safe_div_0(var_713, var_658);
            }
            var_715 = wp::where(var_700, var_708, var_714);
            var_716 = wp::where(var_700, var_709, var_711);
            var_717 = wp::where(var_700, var_710, var_712);
            // if x1 > 1:                                                                     <L 1243>
            var_719 = (var_715 > var_718);
            if (var_719) {
                // x1 = 1.0                                                                   <L 1244>
                // s1 = 2                                                                     <L 1245>
            }
            var_722 = wp::where(var_719, var_720, var_715);
            var_723 = wp::where(var_719, var_721, var_698);
            if (!var_719) {
                // elif x1 < -1:                                                              <L 1246>
                var_725 = (var_722 < var_724);
                if (var_725) {
                    // x1 = -1.0                                                              <L 1247>
                    // s1 = 0                                                                 <L 1248>
                }
                var_728 = wp::where(var_725, var_726, var_722);
                var_729 = wp::where(var_725, var_727, var_723);
            }
            var_730 = wp::where(var_719, var_722, var_728);
            var_731 = wp::where(var_719, var_723, var_729);
        }
        var_732 = wp::where(var_701, var_730, var_696);
        var_733 = wp::where(var_701, var_716, var_697);
        var_734 = wp::where(var_701, var_731, var_698);
        var_735 = wp::where(var_701, var_717, var_677);
        // dif -= halfaxis * x2                                                               <L 1250>
        var_736 = wp::mul(var_4, var_733);
        var_737 = wp::sub(var_655, var_736);
        // dif[j] += box_size[j] * x1                                                         <L 1251>
        var_738 = wp::extract(var_box_size, var_608);
        var_739 = wp::mul(var_738, var_732);
        wp::add_inplace(var_737, var_608, var_739);
        // ct = s1 * 3 + s2                                                                   <L 1254>
        var_741 = wp::mul(var_734, var_740);
        var_742 = wp::add(var_741, var_735);
        // dif_sq = wp.length_sq(dif)                                                         <L 1256>
        var_743 = wp::length_sq(var_737);
        // if dif_sq < bestdist - MJ_MINVAL:                                                  <L 1257>
        var_744 = wp::sub(var_27, var_183);
        var_745 = (var_743 < var_744);
        if (var_745) {
            // bestdist = dif_sq                                                              <L 1258>
            var_746 = wp::copy(var_743);
            // bestsegmentpos = x2                                                            <L 1259>
            var_747 = wp::copy(var_733);
            // bestboxpos = x1                                                                <L 1260>
            var_748 = wp::copy(var_732);
            // c2 = ct // 6                                                                   <L 1262>
            var_750 = wp::floordiv(var_742, var_749);
            // clcorner = i + (1 << j) * c2  # index of closest box corner                    <L 1264>
            var_752 = wp::lshift(var_751, var_608);
            var_753 = wp::mul(var_752, var_750);
            var_754 = wp::add(var_605, var_753);
            // cledge = j  # axis index of closest box edge                                   <L 1265>
            var_755 = wp::copy(var_608);
            // cltype = ct  # encoded collision configuration                                 <L 1266>
            var_756 = wp::copy(var_742);
        }
        var_757 = wp::where(var_745, var_746, var_27);
        var_758 = wp::where(var_745, var_747, var_29);
        var_759 = wp::where(var_745, var_756, var_31);
        var_760 = wp::where(var_745, var_754, var_131);
        var_761 = wp::where(var_745, var_755, var_133);
        var_762 = wp::where(var_745, var_748, var_135);
        var_763 = wp::where(var_745, var_750, var_653);
        wp::assign(var_27, var_757);
        wp::assign(var_29, var_758);
        wp::assign(var_31, var_759);
        wp::assign(var_131, var_760);
        wp::assign(var_133, var_761);
        wp::assign(var_135, var_762);
        wp::assign(var_286, var_763);
        wp::assign(var_163, var_654);
        wp::assign(var_260, var_737);
        wp::assign(var_169, var_656);
        wp::assign(var_170, var_657);
        wp::assign(var_173, var_658);
        wp::assign(var_177, var_659);
        wp::assign(var_178, var_660);
        wp::assign(var_181, var_661);
        wp::assign(var_186, var_663);
        wp::assign(var_255, var_732);
        wp::assign(var_256, var_733);
        wp::assign(var_257, var_734);
        wp::assign(var_258, var_735);
        wp::assign(var_223, var_700);
        wp::assign(var_265, var_742);
        wp::assign(var_266, var_743);
        goto start_for_8;
    end_for_8:;
    // for j in range(3):                                                                     <L 1185>
    var_766 = wp::range(var_765);
    start_for_10:;
        if (iter_cmp(var_766) == 0) goto end_for_10;
        var_767 = wp::iter_next(var_766);
        // if i & (1 << j) != 0:                                                              <L 1186>
        var_769 = wp::lshift(var_768, var_767);
        var_770 = wp::bit_and(var_764, var_769);
        var_772 = (var_770 != var_771);
        if (var_772) {
            // continue                                                                       <L 1187>
            goto start_for_10;
        }
        // c2 = wp.int32(-123)                                                                <L 1189>
        var_774 = wp::int32(var_773);
        // box_pt = wp.cw_mul(                                                                <L 1192>
        // wp.vec3(                                                                           <L 1193>
        // wp.where(i & 1, 1.0, -1.0),                                                        <L 1194>
        var_776 = wp::bit_and(var_764, var_775);
        var_779 = wp::where(var_776, var_777, var_778);
        // wp.where(i & 2, 1.0, -1.0),                                                        <L 1195>
        var_781 = wp::bit_and(var_764, var_780);
        var_784 = wp::where(var_781, var_782, var_783);
        // wp.where(i & 4, 1.0, -1.0),                                                        <L 1196>
        var_786 = wp::bit_and(var_764, var_785);
        var_789 = wp::where(var_786, var_787, var_788);
        var_790 = wp::vec_t<3, wp::float32>(var_779, var_784, var_789);
        // box_size,                                                                          <L 1198>
        var_791 = wp::cw_mul(var_790, var_box_size);
        // box_pt[j] = 0.0                                                                    <L 1200>
        wp::assign_inplace(var_791, var_767, var_792);
        // dif = box_pt - pos                                                                 <L 1203>
        var_793 = wp::sub(var_791, var_2);
        // u = -box_size[j] * dif[j]                                                          <L 1205>
        var_794 = wp::extract(var_box_size, var_767);
        var_795 = wp::neg(var_794);
        var_796 = wp::extract(var_793, var_767);
        var_797 = wp::mul(var_795, var_796);
        // v = wp.dot(halfaxis, dif)                                                          <L 1206>
        var_798 = wp::dot(var_4, var_793);
        // ma = box_size[j] * box_size[j]                                                     <L 1207>
        var_799 = wp::extract(var_box_size, var_767);
        var_800 = wp::extract(var_box_size, var_767);
        var_801 = wp::mul(var_799, var_800);
        // mb = -box_size[j] * halfaxis[j]                                                    <L 1208>
        var_802 = wp::extract(var_box_size, var_767);
        var_803 = wp::neg(var_802);
        var_804 = wp::extract(var_4, var_767);
        var_805 = wp::mul(var_803, var_804);
        // mc = capsule_half_length * capsule_half_length                                     <L 1209>
        var_806 = wp::mul(var_capsule_half_length, var_capsule_half_length);
        // det = ma * mc - mb * mb                                                            <L 1210>
        var_807 = wp::mul(var_801, var_806);
        var_808 = wp::mul(var_805, var_805);
        var_809 = wp::sub(var_807, var_808);
        // if wp.abs(det) < MJ_MINVAL:                                                        <L 1211>
        var_810 = wp::abs(var_809);
        var_811 = (var_810 < var_183);
        if (var_811) {
            // continue                                                                       <L 1212>
            wp::assign(var_286, var_774);
            wp::assign(var_163, var_791);
            wp::assign(var_260, var_793);
            wp::assign(var_169, var_797);
            wp::assign(var_170, var_798);
            wp::assign(var_173, var_801);
            wp::assign(var_177, var_805);
            wp::assign(var_178, var_806);
            wp::assign(var_181, var_809);
            goto start_for_10;
        }
        var_812 = wp::where(var_811, var_286, var_774);
        var_813 = wp::where(var_811, var_163, var_791);
        var_814 = wp::where(var_811, var_260, var_793);
        var_815 = wp::where(var_811, var_169, var_797);
        var_816 = wp::where(var_811, var_170, var_798);
        var_817 = wp::where(var_811, var_173, var_801);
        var_818 = wp::where(var_811, var_177, var_805);
        var_819 = wp::where(var_811, var_178, var_806);
        var_820 = wp::where(var_811, var_181, var_809);
        // idet = 1.0 / det                                                                   <L 1214>
        var_822 = wp::div(var_821, var_820);
        // x1 = wp.float32((mc * u - mb * v) * idet)                                          <L 1217>
        var_823 = wp::mul(var_819, var_815);
        var_824 = wp::mul(var_818, var_816);
        var_825 = wp::sub(var_823, var_824);
        var_826 = wp::mul(var_825, var_822);
        var_827 = wp::float32(var_826);
        // x2 = wp.float32((ma * v - mb * u) * idet)                                          <L 1218>
        var_828 = wp::mul(var_817, var_816);
        var_829 = wp::mul(var_818, var_815);
        var_830 = wp::sub(var_828, var_829);
        var_831 = wp::mul(var_830, var_822);
        var_832 = wp::float32(var_831);
        // s1 = wp.int32(1)                                                                   <L 1220>
        var_834 = wp::int32(var_833);
        // s2 = wp.int32(1)                                                                   <L 1221>
        var_836 = wp::int32(var_835);
        // if x1 > 1:                                                                         <L 1223>
        var_838 = (var_827 > var_837);
        if (var_838) {
            // x1 = 1.0                                                                       <L 1224>
            // s1 = 2                                                                         <L 1225>
            // x2 = safe_div(v - mb, mc)                                                      <L 1226>
            var_841 = wp::sub(var_816, var_818);
            var_842 = safe_div_0(var_841, var_819);
        }
        var_843 = wp::where(var_838, var_839, var_827);
        var_844 = wp::where(var_838, var_842, var_832);
        var_845 = wp::where(var_838, var_840, var_834);
        if (!var_838) {
            // elif x1 < -1:                                                                  <L 1227>
            var_847 = (var_843 < var_846);
            if (var_847) {
                // x1 = -1.0                                                                  <L 1228>
                // s1 = 0                                                                     <L 1229>
                // x2 = safe_div(v + mb, mc)                                                  <L 1230>
                var_850 = wp::add(var_816, var_818);
                var_851 = safe_div_0(var_850, var_819);
            }
            var_852 = wp::where(var_847, var_848, var_843);
            var_853 = wp::where(var_847, var_851, var_844);
            var_854 = wp::where(var_847, var_849, var_845);
        }
        var_855 = wp::where(var_838, var_843, var_852);
        var_856 = wp::where(var_838, var_844, var_853);
        var_857 = wp::where(var_838, var_845, var_854);
        // x2_over = x2 > 1.0                                                                 <L 1232>
        var_859 = (var_856 > var_858);
        // if x2_over or x2 < -1.0:                                                           <L 1233>
        var_860 = var_859;
        if (!var_860) {
            var_862 = (var_856 < var_861);
            var_860 = var_860 || var_862;
        }
        if (var_860) {
            // if x2_over:                                                                    <L 1234>
            if (var_859) {
                // x2 = 1.0                                                                   <L 1235>
                // s2 = 2                                                                     <L 1236>
                // x1 = safe_div(u - mb, ma)                                                  <L 1237>
                var_865 = wp::sub(var_815, var_818);
                var_866 = safe_div_0(var_865, var_817);
            }
            var_867 = wp::where(var_859, var_866, var_855);
            var_868 = wp::where(var_859, var_863, var_856);
            var_869 = wp::where(var_859, var_864, var_836);
            if (!var_859) {
                // x2 = -1.0                                                                  <L 1239>
                // s2 = 0                                                                     <L 1240>
                // x1 = safe_div(u + mb, ma)                                                  <L 1241>
                var_872 = wp::add(var_815, var_818);
                var_873 = safe_div_0(var_872, var_817);
            }
            var_874 = wp::where(var_859, var_867, var_873);
            var_875 = wp::where(var_859, var_868, var_870);
            var_876 = wp::where(var_859, var_869, var_871);
            // if x1 > 1:                                                                     <L 1243>
            var_878 = (var_874 > var_877);
            if (var_878) {
                // x1 = 1.0                                                                   <L 1244>
                // s1 = 2                                                                     <L 1245>
            }
            var_881 = wp::where(var_878, var_879, var_874);
            var_882 = wp::where(var_878, var_880, var_857);
            if (!var_878) {
                // elif x1 < -1:                                                              <L 1246>
                var_884 = (var_881 < var_883);
                if (var_884) {
                    // x1 = -1.0                                                              <L 1247>
                    // s1 = 0                                                                 <L 1248>
                }
                var_887 = wp::where(var_884, var_885, var_881);
                var_888 = wp::where(var_884, var_886, var_882);
            }
            var_889 = wp::where(var_878, var_881, var_887);
            var_890 = wp::where(var_878, var_882, var_888);
        }
        var_891 = wp::where(var_860, var_889, var_855);
        var_892 = wp::where(var_860, var_875, var_856);
        var_893 = wp::where(var_860, var_890, var_857);
        var_894 = wp::where(var_860, var_876, var_836);
        // dif -= halfaxis * x2                                                               <L 1250>
        var_895 = wp::mul(var_4, var_892);
        var_896 = wp::sub(var_814, var_895);
        // dif[j] += box_size[j] * x1                                                         <L 1251>
        var_897 = wp::extract(var_box_size, var_767);
        var_898 = wp::mul(var_897, var_891);
        wp::add_inplace(var_896, var_767, var_898);
        // ct = s1 * 3 + s2                                                                   <L 1254>
        var_900 = wp::mul(var_893, var_899);
        var_901 = wp::add(var_900, var_894);
        // dif_sq = wp.length_sq(dif)                                                         <L 1256>
        var_902 = wp::length_sq(var_896);
        // if dif_sq < bestdist - MJ_MINVAL:                                                  <L 1257>
        var_903 = wp::sub(var_27, var_183);
        var_904 = (var_902 < var_903);
        if (var_904) {
            // bestdist = dif_sq                                                              <L 1258>
            var_905 = wp::copy(var_902);
            // bestsegmentpos = x2                                                            <L 1259>
            var_906 = wp::copy(var_892);
            // bestboxpos = x1                                                                <L 1260>
            var_907 = wp::copy(var_891);
            // c2 = ct // 6                                                                   <L 1262>
            var_909 = wp::floordiv(var_901, var_908);
            // clcorner = i + (1 << j) * c2  # index of closest box corner                    <L 1264>
            var_911 = wp::lshift(var_910, var_767);
            var_912 = wp::mul(var_911, var_909);
            var_913 = wp::add(var_764, var_912);
            // cledge = j  # axis index of closest box edge                                   <L 1265>
            var_914 = wp::copy(var_767);
            // cltype = ct  # encoded collision configuration                                 <L 1266>
            var_915 = wp::copy(var_901);
        }
        var_916 = wp::where(var_904, var_905, var_27);
        var_917 = wp::where(var_904, var_906, var_29);
        var_918 = wp::where(var_904, var_915, var_31);
        var_919 = wp::where(var_904, var_913, var_131);
        var_920 = wp::where(var_904, var_914, var_133);
        var_921 = wp::where(var_904, var_907, var_135);
        var_922 = wp::where(var_904, var_909, var_812);
        wp::assign(var_27, var_916);
        wp::assign(var_29, var_917);
        wp::assign(var_31, var_918);
        wp::assign(var_131, var_919);
        wp::assign(var_133, var_920);
        wp::assign(var_135, var_921);
        wp::assign(var_286, var_922);
        wp::assign(var_163, var_813);
        wp::assign(var_260, var_896);
        wp::assign(var_169, var_815);
        wp::assign(var_170, var_816);
        wp::assign(var_173, var_817);
        wp::assign(var_177, var_818);
        wp::assign(var_178, var_819);
        wp::assign(var_181, var_820);
        wp::assign(var_186, var_822);
        wp::assign(var_255, var_891);
        wp::assign(var_256, var_892);
        wp::assign(var_257, var_893);
        wp::assign(var_258, var_894);
        wp::assign(var_223, var_859);
        wp::assign(var_265, var_901);
        wp::assign(var_266, var_902);
        goto start_for_10;
    end_for_10:;
    // for j in range(3):                                                                     <L 1185>
    var_925 = wp::range(var_924);
    start_for_12:;
        if (iter_cmp(var_925) == 0) goto end_for_12;
        var_926 = wp::iter_next(var_925);
        // if i & (1 << j) != 0:                                                              <L 1186>
        var_928 = wp::lshift(var_927, var_926);
        var_929 = wp::bit_and(var_923, var_928);
        var_931 = (var_929 != var_930);
        if (var_931) {
            // continue                                                                       <L 1187>
            goto start_for_12;
        }
        // c2 = wp.int32(-123)                                                                <L 1189>
        var_933 = wp::int32(var_932);
        // box_pt = wp.cw_mul(                                                                <L 1192>
        // wp.vec3(                                                                           <L 1193>
        // wp.where(i & 1, 1.0, -1.0),                                                        <L 1194>
        var_935 = wp::bit_and(var_923, var_934);
        var_938 = wp::where(var_935, var_936, var_937);
        // wp.where(i & 2, 1.0, -1.0),                                                        <L 1195>
        var_940 = wp::bit_and(var_923, var_939);
        var_943 = wp::where(var_940, var_941, var_942);
        // wp.where(i & 4, 1.0, -1.0),                                                        <L 1196>
        var_945 = wp::bit_and(var_923, var_944);
        var_948 = wp::where(var_945, var_946, var_947);
        var_949 = wp::vec_t<3, wp::float32>(var_938, var_943, var_948);
        // box_size,                                                                          <L 1198>
        var_950 = wp::cw_mul(var_949, var_box_size);
        // box_pt[j] = 0.0                                                                    <L 1200>
        wp::assign_inplace(var_950, var_926, var_951);
        // dif = box_pt - pos                                                                 <L 1203>
        var_952 = wp::sub(var_950, var_2);
        // u = -box_size[j] * dif[j]                                                          <L 1205>
        var_953 = wp::extract(var_box_size, var_926);
        var_954 = wp::neg(var_953);
        var_955 = wp::extract(var_952, var_926);
        var_956 = wp::mul(var_954, var_955);
        // v = wp.dot(halfaxis, dif)                                                          <L 1206>
        var_957 = wp::dot(var_4, var_952);
        // ma = box_size[j] * box_size[j]                                                     <L 1207>
        var_958 = wp::extract(var_box_size, var_926);
        var_959 = wp::extract(var_box_size, var_926);
        var_960 = wp::mul(var_958, var_959);
        // mb = -box_size[j] * halfaxis[j]                                                    <L 1208>
        var_961 = wp::extract(var_box_size, var_926);
        var_962 = wp::neg(var_961);
        var_963 = wp::extract(var_4, var_926);
        var_964 = wp::mul(var_962, var_963);
        // mc = capsule_half_length * capsule_half_length                                     <L 1209>
        var_965 = wp::mul(var_capsule_half_length, var_capsule_half_length);
        // det = ma * mc - mb * mb                                                            <L 1210>
        var_966 = wp::mul(var_960, var_965);
        var_967 = wp::mul(var_964, var_964);
        var_968 = wp::sub(var_966, var_967);
        // if wp.abs(det) < MJ_MINVAL:                                                        <L 1211>
        var_969 = wp::abs(var_968);
        var_970 = (var_969 < var_183);
        if (var_970) {
            // continue                                                                       <L 1212>
            wp::assign(var_286, var_933);
            wp::assign(var_163, var_950);
            wp::assign(var_260, var_952);
            wp::assign(var_169, var_956);
            wp::assign(var_170, var_957);
            wp::assign(var_173, var_960);
            wp::assign(var_177, var_964);
            wp::assign(var_178, var_965);
            wp::assign(var_181, var_968);
            goto start_for_12;
        }
        var_971 = wp::where(var_970, var_286, var_933);
        var_972 = wp::where(var_970, var_163, var_950);
        var_973 = wp::where(var_970, var_260, var_952);
        var_974 = wp::where(var_970, var_169, var_956);
        var_975 = wp::where(var_970, var_170, var_957);
        var_976 = wp::where(var_970, var_173, var_960);
        var_977 = wp::where(var_970, var_177, var_964);
        var_978 = wp::where(var_970, var_178, var_965);
        var_979 = wp::where(var_970, var_181, var_968);
        // idet = 1.0 / det                                                                   <L 1214>
        var_981 = wp::div(var_980, var_979);
        // x1 = wp.float32((mc * u - mb * v) * idet)                                          <L 1217>
        var_982 = wp::mul(var_978, var_974);
        var_983 = wp::mul(var_977, var_975);
        var_984 = wp::sub(var_982, var_983);
        var_985 = wp::mul(var_984, var_981);
        var_986 = wp::float32(var_985);
        // x2 = wp.float32((ma * v - mb * u) * idet)                                          <L 1218>
        var_987 = wp::mul(var_976, var_975);
        var_988 = wp::mul(var_977, var_974);
        var_989 = wp::sub(var_987, var_988);
        var_990 = wp::mul(var_989, var_981);
        var_991 = wp::float32(var_990);
        // s1 = wp.int32(1)                                                                   <L 1220>
        var_993 = wp::int32(var_992);
        // s2 = wp.int32(1)                                                                   <L 1221>
        var_995 = wp::int32(var_994);
        // if x1 > 1:                                                                         <L 1223>
        var_997 = (var_986 > var_996);
        if (var_997) {
            // x1 = 1.0                                                                       <L 1224>
            // s1 = 2                                                                         <L 1225>
            // x2 = safe_div(v - mb, mc)                                                      <L 1226>
            var_1000 = wp::sub(var_975, var_977);
            var_1001 = safe_div_0(var_1000, var_978);
        }
        var_1002 = wp::where(var_997, var_998, var_986);
        var_1003 = wp::where(var_997, var_1001, var_991);
        var_1004 = wp::where(var_997, var_999, var_993);
        if (!var_997) {
            // elif x1 < -1:                                                                  <L 1227>
            var_1006 = (var_1002 < var_1005);
            if (var_1006) {
                // x1 = -1.0                                                                  <L 1228>
                // s1 = 0                                                                     <L 1229>
                // x2 = safe_div(v + mb, mc)                                                  <L 1230>
                var_1009 = wp::add(var_975, var_977);
                var_1010 = safe_div_0(var_1009, var_978);
            }
            var_1011 = wp::where(var_1006, var_1007, var_1002);
            var_1012 = wp::where(var_1006, var_1010, var_1003);
            var_1013 = wp::where(var_1006, var_1008, var_1004);
        }
        var_1014 = wp::where(var_997, var_1002, var_1011);
        var_1015 = wp::where(var_997, var_1003, var_1012);
        var_1016 = wp::where(var_997, var_1004, var_1013);
        // x2_over = x2 > 1.0                                                                 <L 1232>
        var_1018 = (var_1015 > var_1017);
        // if x2_over or x2 < -1.0:                                                           <L 1233>
        var_1019 = var_1018;
        if (!var_1019) {
            var_1021 = (var_1015 < var_1020);
            var_1019 = var_1019 || var_1021;
        }
        if (var_1019) {
            // if x2_over:                                                                    <L 1234>
            if (var_1018) {
                // x2 = 1.0                                                                   <L 1235>
                // s2 = 2                                                                     <L 1236>
                // x1 = safe_div(u - mb, ma)                                                  <L 1237>
                var_1024 = wp::sub(var_974, var_977);
                var_1025 = safe_div_0(var_1024, var_976);
            }
            var_1026 = wp::where(var_1018, var_1025, var_1014);
            var_1027 = wp::where(var_1018, var_1022, var_1015);
            var_1028 = wp::where(var_1018, var_1023, var_995);
            if (!var_1018) {
                // x2 = -1.0                                                                  <L 1239>
                // s2 = 0                                                                     <L 1240>
                // x1 = safe_div(u + mb, ma)                                                  <L 1241>
                var_1031 = wp::add(var_974, var_977);
                var_1032 = safe_div_0(var_1031, var_976);
            }
            var_1033 = wp::where(var_1018, var_1026, var_1032);
            var_1034 = wp::where(var_1018, var_1027, var_1029);
            var_1035 = wp::where(var_1018, var_1028, var_1030);
            // if x1 > 1:                                                                     <L 1243>
            var_1037 = (var_1033 > var_1036);
            if (var_1037) {
                // x1 = 1.0                                                                   <L 1244>
                // s1 = 2                                                                     <L 1245>
            }
            var_1040 = wp::where(var_1037, var_1038, var_1033);
            var_1041 = wp::where(var_1037, var_1039, var_1016);
            if (!var_1037) {
                // elif x1 < -1:                                                              <L 1246>
                var_1043 = (var_1040 < var_1042);
                if (var_1043) {
                    // x1 = -1.0                                                              <L 1247>
                    // s1 = 0                                                                 <L 1248>
                }
                var_1046 = wp::where(var_1043, var_1044, var_1040);
                var_1047 = wp::where(var_1043, var_1045, var_1041);
            }
            var_1048 = wp::where(var_1037, var_1040, var_1046);
            var_1049 = wp::where(var_1037, var_1041, var_1047);
        }
        var_1050 = wp::where(var_1019, var_1048, var_1014);
        var_1051 = wp::where(var_1019, var_1034, var_1015);
        var_1052 = wp::where(var_1019, var_1049, var_1016);
        var_1053 = wp::where(var_1019, var_1035, var_995);
        // dif -= halfaxis * x2                                                               <L 1250>
        var_1054 = wp::mul(var_4, var_1051);
        var_1055 = wp::sub(var_973, var_1054);
        // dif[j] += box_size[j] * x1                                                         <L 1251>
        var_1056 = wp::extract(var_box_size, var_926);
        var_1057 = wp::mul(var_1056, var_1050);
        wp::add_inplace(var_1055, var_926, var_1057);
        // ct = s1 * 3 + s2                                                                   <L 1254>
        var_1059 = wp::mul(var_1052, var_1058);
        var_1060 = wp::add(var_1059, var_1053);
        // dif_sq = wp.length_sq(dif)                                                         <L 1256>
        var_1061 = wp::length_sq(var_1055);
        // if dif_sq < bestdist - MJ_MINVAL:                                                  <L 1257>
        var_1062 = wp::sub(var_27, var_183);
        var_1063 = (var_1061 < var_1062);
        if (var_1063) {
            // bestdist = dif_sq                                                              <L 1258>
            var_1064 = wp::copy(var_1061);
            // bestsegmentpos = x2                                                            <L 1259>
            var_1065 = wp::copy(var_1051);
            // bestboxpos = x1                                                                <L 1260>
            var_1066 = wp::copy(var_1050);
            // c2 = ct // 6                                                                   <L 1262>
            var_1068 = wp::floordiv(var_1060, var_1067);
            // clcorner = i + (1 << j) * c2  # index of closest box corner                    <L 1264>
            var_1070 = wp::lshift(var_1069, var_926);
            var_1071 = wp::mul(var_1070, var_1068);
            var_1072 = wp::add(var_923, var_1071);
            // cledge = j  # axis index of closest box edge                                   <L 1265>
            var_1073 = wp::copy(var_926);
            // cltype = ct  # encoded collision configuration                                 <L 1266>
            var_1074 = wp::copy(var_1060);
        }
        var_1075 = wp::where(var_1063, var_1064, var_27);
        var_1076 = wp::where(var_1063, var_1065, var_29);
        var_1077 = wp::where(var_1063, var_1074, var_31);
        var_1078 = wp::where(var_1063, var_1072, var_131);
        var_1079 = wp::where(var_1063, var_1073, var_133);
        var_1080 = wp::where(var_1063, var_1066, var_135);
        var_1081 = wp::where(var_1063, var_1068, var_971);
        wp::assign(var_27, var_1075);
        wp::assign(var_29, var_1076);
        wp::assign(var_31, var_1077);
        wp::assign(var_131, var_1078);
        wp::assign(var_133, var_1079);
        wp::assign(var_135, var_1080);
        wp::assign(var_286, var_1081);
        wp::assign(var_163, var_972);
        wp::assign(var_260, var_1055);
        wp::assign(var_169, var_974);
        wp::assign(var_170, var_975);
        wp::assign(var_173, var_976);
        wp::assign(var_177, var_977);
        wp::assign(var_178, var_978);
        wp::assign(var_181, var_979);
        wp::assign(var_186, var_981);
        wp::assign(var_255, var_1050);
        wp::assign(var_256, var_1051);
        wp::assign(var_257, var_1052);
        wp::assign(var_258, var_1053);
        wp::assign(var_223, var_1018);
        wp::assign(var_265, var_1060);
        wp::assign(var_266, var_1061);
        goto start_for_12;
    end_for_12:;
    // for j in range(3):                                                                     <L 1185>
    var_1084 = wp::range(var_1083);
    start_for_14:;
        if (iter_cmp(var_1084) == 0) goto end_for_14;
        var_1085 = wp::iter_next(var_1084);
        // if i & (1 << j) != 0:                                                              <L 1186>
        var_1087 = wp::lshift(var_1086, var_1085);
        var_1088 = wp::bit_and(var_1082, var_1087);
        var_1090 = (var_1088 != var_1089);
        if (var_1090) {
            // continue                                                                       <L 1187>
            goto start_for_14;
        }
        // c2 = wp.int32(-123)                                                                <L 1189>
        var_1092 = wp::int32(var_1091);
        // box_pt = wp.cw_mul(                                                                <L 1192>
        // wp.vec3(                                                                           <L 1193>
        // wp.where(i & 1, 1.0, -1.0),                                                        <L 1194>
        var_1094 = wp::bit_and(var_1082, var_1093);
        var_1097 = wp::where(var_1094, var_1095, var_1096);
        // wp.where(i & 2, 1.0, -1.0),                                                        <L 1195>
        var_1099 = wp::bit_and(var_1082, var_1098);
        var_1102 = wp::where(var_1099, var_1100, var_1101);
        // wp.where(i & 4, 1.0, -1.0),                                                        <L 1196>
        var_1104 = wp::bit_and(var_1082, var_1103);
        var_1107 = wp::where(var_1104, var_1105, var_1106);
        var_1108 = wp::vec_t<3, wp::float32>(var_1097, var_1102, var_1107);
        // box_size,                                                                          <L 1198>
        var_1109 = wp::cw_mul(var_1108, var_box_size);
        // box_pt[j] = 0.0                                                                    <L 1200>
        wp::assign_inplace(var_1109, var_1085, var_1110);
        // dif = box_pt - pos                                                                 <L 1203>
        var_1111 = wp::sub(var_1109, var_2);
        // u = -box_size[j] * dif[j]                                                          <L 1205>
        var_1112 = wp::extract(var_box_size, var_1085);
        var_1113 = wp::neg(var_1112);
        var_1114 = wp::extract(var_1111, var_1085);
        var_1115 = wp::mul(var_1113, var_1114);
        // v = wp.dot(halfaxis, dif)                                                          <L 1206>
        var_1116 = wp::dot(var_4, var_1111);
        // ma = box_size[j] * box_size[j]                                                     <L 1207>
        var_1117 = wp::extract(var_box_size, var_1085);
        var_1118 = wp::extract(var_box_size, var_1085);
        var_1119 = wp::mul(var_1117, var_1118);
        // mb = -box_size[j] * halfaxis[j]                                                    <L 1208>
        var_1120 = wp::extract(var_box_size, var_1085);
        var_1121 = wp::neg(var_1120);
        var_1122 = wp::extract(var_4, var_1085);
        var_1123 = wp::mul(var_1121, var_1122);
        // mc = capsule_half_length * capsule_half_length                                     <L 1209>
        var_1124 = wp::mul(var_capsule_half_length, var_capsule_half_length);
        // det = ma * mc - mb * mb                                                            <L 1210>
        var_1125 = wp::mul(var_1119, var_1124);
        var_1126 = wp::mul(var_1123, var_1123);
        var_1127 = wp::sub(var_1125, var_1126);
        // if wp.abs(det) < MJ_MINVAL:                                                        <L 1211>
        var_1128 = wp::abs(var_1127);
        var_1129 = (var_1128 < var_183);
        if (var_1129) {
            // continue                                                                       <L 1212>
            wp::assign(var_286, var_1092);
            wp::assign(var_163, var_1109);
            wp::assign(var_260, var_1111);
            wp::assign(var_169, var_1115);
            wp::assign(var_170, var_1116);
            wp::assign(var_173, var_1119);
            wp::assign(var_177, var_1123);
            wp::assign(var_178, var_1124);
            wp::assign(var_181, var_1127);
            goto start_for_14;
        }
        var_1130 = wp::where(var_1129, var_286, var_1092);
        var_1131 = wp::where(var_1129, var_163, var_1109);
        var_1132 = wp::where(var_1129, var_260, var_1111);
        var_1133 = wp::where(var_1129, var_169, var_1115);
        var_1134 = wp::where(var_1129, var_170, var_1116);
        var_1135 = wp::where(var_1129, var_173, var_1119);
        var_1136 = wp::where(var_1129, var_177, var_1123);
        var_1137 = wp::where(var_1129, var_178, var_1124);
        var_1138 = wp::where(var_1129, var_181, var_1127);
        // idet = 1.0 / det                                                                   <L 1214>
        var_1140 = wp::div(var_1139, var_1138);
        // x1 = wp.float32((mc * u - mb * v) * idet)                                          <L 1217>
        var_1141 = wp::mul(var_1137, var_1133);
        var_1142 = wp::mul(var_1136, var_1134);
        var_1143 = wp::sub(var_1141, var_1142);
        var_1144 = wp::mul(var_1143, var_1140);
        var_1145 = wp::float32(var_1144);
        // x2 = wp.float32((ma * v - mb * u) * idet)                                          <L 1218>
        var_1146 = wp::mul(var_1135, var_1134);
        var_1147 = wp::mul(var_1136, var_1133);
        var_1148 = wp::sub(var_1146, var_1147);
        var_1149 = wp::mul(var_1148, var_1140);
        var_1150 = wp::float32(var_1149);
        // s1 = wp.int32(1)                                                                   <L 1220>
        var_1152 = wp::int32(var_1151);
        // s2 = wp.int32(1)                                                                   <L 1221>
        var_1154 = wp::int32(var_1153);
        // if x1 > 1:                                                                         <L 1223>
        var_1156 = (var_1145 > var_1155);
        if (var_1156) {
            // x1 = 1.0                                                                       <L 1224>
            // s1 = 2                                                                         <L 1225>
            // x2 = safe_div(v - mb, mc)                                                      <L 1226>
            var_1159 = wp::sub(var_1134, var_1136);
            var_1160 = safe_div_0(var_1159, var_1137);
        }
        var_1161 = wp::where(var_1156, var_1157, var_1145);
        var_1162 = wp::where(var_1156, var_1160, var_1150);
        var_1163 = wp::where(var_1156, var_1158, var_1152);
        if (!var_1156) {
            // elif x1 < -1:                                                                  <L 1227>
            var_1165 = (var_1161 < var_1164);
            if (var_1165) {
                // x1 = -1.0                                                                  <L 1228>
                // s1 = 0                                                                     <L 1229>
                // x2 = safe_div(v + mb, mc)                                                  <L 1230>
                var_1168 = wp::add(var_1134, var_1136);
                var_1169 = safe_div_0(var_1168, var_1137);
            }
            var_1170 = wp::where(var_1165, var_1166, var_1161);
            var_1171 = wp::where(var_1165, var_1169, var_1162);
            var_1172 = wp::where(var_1165, var_1167, var_1163);
        }
        var_1173 = wp::where(var_1156, var_1161, var_1170);
        var_1174 = wp::where(var_1156, var_1162, var_1171);
        var_1175 = wp::where(var_1156, var_1163, var_1172);
        // x2_over = x2 > 1.0                                                                 <L 1232>
        var_1177 = (var_1174 > var_1176);
        // if x2_over or x2 < -1.0:                                                           <L 1233>
        var_1178 = var_1177;
        if (!var_1178) {
            var_1180 = (var_1174 < var_1179);
            var_1178 = var_1178 || var_1180;
        }
        if (var_1178) {
            // if x2_over:                                                                    <L 1234>
            if (var_1177) {
                // x2 = 1.0                                                                   <L 1235>
                // s2 = 2                                                                     <L 1236>
                // x1 = safe_div(u - mb, ma)                                                  <L 1237>
                var_1183 = wp::sub(var_1133, var_1136);
                var_1184 = safe_div_0(var_1183, var_1135);
            }
            var_1185 = wp::where(var_1177, var_1184, var_1173);
            var_1186 = wp::where(var_1177, var_1181, var_1174);
            var_1187 = wp::where(var_1177, var_1182, var_1154);
            if (!var_1177) {
                // x2 = -1.0                                                                  <L 1239>
                // s2 = 0                                                                     <L 1240>
                // x1 = safe_div(u + mb, ma)                                                  <L 1241>
                var_1190 = wp::add(var_1133, var_1136);
                var_1191 = safe_div_0(var_1190, var_1135);
            }
            var_1192 = wp::where(var_1177, var_1185, var_1191);
            var_1193 = wp::where(var_1177, var_1186, var_1188);
            var_1194 = wp::where(var_1177, var_1187, var_1189);
            // if x1 > 1:                                                                     <L 1243>
            var_1196 = (var_1192 > var_1195);
            if (var_1196) {
                // x1 = 1.0                                                                   <L 1244>
                // s1 = 2                                                                     <L 1245>
            }
            var_1199 = wp::where(var_1196, var_1197, var_1192);
            var_1200 = wp::where(var_1196, var_1198, var_1175);
            if (!var_1196) {
                // elif x1 < -1:                                                              <L 1246>
                var_1202 = (var_1199 < var_1201);
                if (var_1202) {
                    // x1 = -1.0                                                              <L 1247>
                    // s1 = 0                                                                 <L 1248>
                }
                var_1205 = wp::where(var_1202, var_1203, var_1199);
                var_1206 = wp::where(var_1202, var_1204, var_1200);
            }
            var_1207 = wp::where(var_1196, var_1199, var_1205);
            var_1208 = wp::where(var_1196, var_1200, var_1206);
        }
        var_1209 = wp::where(var_1178, var_1207, var_1173);
        var_1210 = wp::where(var_1178, var_1193, var_1174);
        var_1211 = wp::where(var_1178, var_1208, var_1175);
        var_1212 = wp::where(var_1178, var_1194, var_1154);
        // dif -= halfaxis * x2                                                               <L 1250>
        var_1213 = wp::mul(var_4, var_1210);
        var_1214 = wp::sub(var_1132, var_1213);
        // dif[j] += box_size[j] * x1                                                         <L 1251>
        var_1215 = wp::extract(var_box_size, var_1085);
        var_1216 = wp::mul(var_1215, var_1209);
        wp::add_inplace(var_1214, var_1085, var_1216);
        // ct = s1 * 3 + s2                                                                   <L 1254>
        var_1218 = wp::mul(var_1211, var_1217);
        var_1219 = wp::add(var_1218, var_1212);
        // dif_sq = wp.length_sq(dif)                                                         <L 1256>
        var_1220 = wp::length_sq(var_1214);
        // if dif_sq < bestdist - MJ_MINVAL:                                                  <L 1257>
        var_1221 = wp::sub(var_27, var_183);
        var_1222 = (var_1220 < var_1221);
        if (var_1222) {
            // bestdist = dif_sq                                                              <L 1258>
            var_1223 = wp::copy(var_1220);
            // bestsegmentpos = x2                                                            <L 1259>
            var_1224 = wp::copy(var_1210);
            // bestboxpos = x1                                                                <L 1260>
            var_1225 = wp::copy(var_1209);
            // c2 = ct // 6                                                                   <L 1262>
            var_1227 = wp::floordiv(var_1219, var_1226);
            // clcorner = i + (1 << j) * c2  # index of closest box corner                    <L 1264>
            var_1229 = wp::lshift(var_1228, var_1085);
            var_1230 = wp::mul(var_1229, var_1227);
            var_1231 = wp::add(var_1082, var_1230);
            // cledge = j  # axis index of closest box edge                                   <L 1265>
            var_1232 = wp::copy(var_1085);
            // cltype = ct  # encoded collision configuration                                 <L 1266>
            var_1233 = wp::copy(var_1219);
        }
        var_1234 = wp::where(var_1222, var_1223, var_27);
        var_1235 = wp::where(var_1222, var_1224, var_29);
        var_1236 = wp::where(var_1222, var_1233, var_31);
        var_1237 = wp::where(var_1222, var_1231, var_131);
        var_1238 = wp::where(var_1222, var_1232, var_133);
        var_1239 = wp::where(var_1222, var_1225, var_135);
        var_1240 = wp::where(var_1222, var_1227, var_1130);
        wp::assign(var_27, var_1234);
        wp::assign(var_29, var_1235);
        wp::assign(var_31, var_1236);
        wp::assign(var_131, var_1237);
        wp::assign(var_133, var_1238);
        wp::assign(var_135, var_1239);
        wp::assign(var_286, var_1240);
        wp::assign(var_163, var_1131);
        wp::assign(var_260, var_1214);
        wp::assign(var_169, var_1133);
        wp::assign(var_170, var_1134);
        wp::assign(var_173, var_1135);
        wp::assign(var_177, var_1136);
        wp::assign(var_178, var_1137);
        wp::assign(var_181, var_1138);
        wp::assign(var_186, var_1140);
        wp::assign(var_255, var_1209);
        wp::assign(var_256, var_1210);
        wp::assign(var_257, var_1211);
        wp::assign(var_258, var_1212);
        wp::assign(var_223, var_1177);
        wp::assign(var_265, var_1219);
        wp::assign(var_266, var_1220);
        goto start_for_14;
    end_for_14:;
    // for j in range(3):                                                                     <L 1185>
    var_1243 = wp::range(var_1242);
    start_for_16:;
        if (iter_cmp(var_1243) == 0) goto end_for_16;
        var_1244 = wp::iter_next(var_1243);
        // if i & (1 << j) != 0:                                                              <L 1186>
        var_1246 = wp::lshift(var_1245, var_1244);
        var_1247 = wp::bit_and(var_1241, var_1246);
        var_1249 = (var_1247 != var_1248);
        if (var_1249) {
            // continue                                                                       <L 1187>
            goto start_for_16;
        }
        // c2 = wp.int32(-123)                                                                <L 1189>
        var_1251 = wp::int32(var_1250);
        // box_pt = wp.cw_mul(                                                                <L 1192>
        // wp.vec3(                                                                           <L 1193>
        // wp.where(i & 1, 1.0, -1.0),                                                        <L 1194>
        var_1253 = wp::bit_and(var_1241, var_1252);
        var_1256 = wp::where(var_1253, var_1254, var_1255);
        // wp.where(i & 2, 1.0, -1.0),                                                        <L 1195>
        var_1258 = wp::bit_and(var_1241, var_1257);
        var_1261 = wp::where(var_1258, var_1259, var_1260);
        // wp.where(i & 4, 1.0, -1.0),                                                        <L 1196>
        var_1263 = wp::bit_and(var_1241, var_1262);
        var_1266 = wp::where(var_1263, var_1264, var_1265);
        var_1267 = wp::vec_t<3, wp::float32>(var_1256, var_1261, var_1266);
        // box_size,                                                                          <L 1198>
        var_1268 = wp::cw_mul(var_1267, var_box_size);
        // box_pt[j] = 0.0                                                                    <L 1200>
        wp::assign_inplace(var_1268, var_1244, var_1269);
        // dif = box_pt - pos                                                                 <L 1203>
        var_1270 = wp::sub(var_1268, var_2);
        // u = -box_size[j] * dif[j]                                                          <L 1205>
        var_1271 = wp::extract(var_box_size, var_1244);
        var_1272 = wp::neg(var_1271);
        var_1273 = wp::extract(var_1270, var_1244);
        var_1274 = wp::mul(var_1272, var_1273);
        // v = wp.dot(halfaxis, dif)                                                          <L 1206>
        var_1275 = wp::dot(var_4, var_1270);
        // ma = box_size[j] * box_size[j]                                                     <L 1207>
        var_1276 = wp::extract(var_box_size, var_1244);
        var_1277 = wp::extract(var_box_size, var_1244);
        var_1278 = wp::mul(var_1276, var_1277);
        // mb = -box_size[j] * halfaxis[j]                                                    <L 1208>
        var_1279 = wp::extract(var_box_size, var_1244);
        var_1280 = wp::neg(var_1279);
        var_1281 = wp::extract(var_4, var_1244);
        var_1282 = wp::mul(var_1280, var_1281);
        // mc = capsule_half_length * capsule_half_length                                     <L 1209>
        var_1283 = wp::mul(var_capsule_half_length, var_capsule_half_length);
        // det = ma * mc - mb * mb                                                            <L 1210>
        var_1284 = wp::mul(var_1278, var_1283);
        var_1285 = wp::mul(var_1282, var_1282);
        var_1286 = wp::sub(var_1284, var_1285);
        // if wp.abs(det) < MJ_MINVAL:                                                        <L 1211>
        var_1287 = wp::abs(var_1286);
        var_1288 = (var_1287 < var_183);
        if (var_1288) {
            // continue                                                                       <L 1212>
            wp::assign(var_286, var_1251);
            wp::assign(var_163, var_1268);
            wp::assign(var_260, var_1270);
            wp::assign(var_169, var_1274);
            wp::assign(var_170, var_1275);
            wp::assign(var_173, var_1278);
            wp::assign(var_177, var_1282);
            wp::assign(var_178, var_1283);
            wp::assign(var_181, var_1286);
            goto start_for_16;
        }
        var_1289 = wp::where(var_1288, var_286, var_1251);
        var_1290 = wp::where(var_1288, var_163, var_1268);
        var_1291 = wp::where(var_1288, var_260, var_1270);
        var_1292 = wp::where(var_1288, var_169, var_1274);
        var_1293 = wp::where(var_1288, var_170, var_1275);
        var_1294 = wp::where(var_1288, var_173, var_1278);
        var_1295 = wp::where(var_1288, var_177, var_1282);
        var_1296 = wp::where(var_1288, var_178, var_1283);
        var_1297 = wp::where(var_1288, var_181, var_1286);
        // idet = 1.0 / det                                                                   <L 1214>
        var_1299 = wp::div(var_1298, var_1297);
        // x1 = wp.float32((mc * u - mb * v) * idet)                                          <L 1217>
        var_1300 = wp::mul(var_1296, var_1292);
        var_1301 = wp::mul(var_1295, var_1293);
        var_1302 = wp::sub(var_1300, var_1301);
        var_1303 = wp::mul(var_1302, var_1299);
        var_1304 = wp::float32(var_1303);
        // x2 = wp.float32((ma * v - mb * u) * idet)                                          <L 1218>
        var_1305 = wp::mul(var_1294, var_1293);
        var_1306 = wp::mul(var_1295, var_1292);
        var_1307 = wp::sub(var_1305, var_1306);
        var_1308 = wp::mul(var_1307, var_1299);
        var_1309 = wp::float32(var_1308);
        // s1 = wp.int32(1)                                                                   <L 1220>
        var_1311 = wp::int32(var_1310);
        // s2 = wp.int32(1)                                                                   <L 1221>
        var_1313 = wp::int32(var_1312);
        // if x1 > 1:                                                                         <L 1223>
        var_1315 = (var_1304 > var_1314);
        if (var_1315) {
            // x1 = 1.0                                                                       <L 1224>
            // s1 = 2                                                                         <L 1225>
            // x2 = safe_div(v - mb, mc)                                                      <L 1226>
            var_1318 = wp::sub(var_1293, var_1295);
            var_1319 = safe_div_0(var_1318, var_1296);
        }
        var_1320 = wp::where(var_1315, var_1316, var_1304);
        var_1321 = wp::where(var_1315, var_1319, var_1309);
        var_1322 = wp::where(var_1315, var_1317, var_1311);
        if (!var_1315) {
            // elif x1 < -1:                                                                  <L 1227>
            var_1324 = (var_1320 < var_1323);
            if (var_1324) {
                // x1 = -1.0                                                                  <L 1228>
                // s1 = 0                                                                     <L 1229>
                // x2 = safe_div(v + mb, mc)                                                  <L 1230>
                var_1327 = wp::add(var_1293, var_1295);
                var_1328 = safe_div_0(var_1327, var_1296);
            }
            var_1329 = wp::where(var_1324, var_1325, var_1320);
            var_1330 = wp::where(var_1324, var_1328, var_1321);
            var_1331 = wp::where(var_1324, var_1326, var_1322);
        }
        var_1332 = wp::where(var_1315, var_1320, var_1329);
        var_1333 = wp::where(var_1315, var_1321, var_1330);
        var_1334 = wp::where(var_1315, var_1322, var_1331);
        // x2_over = x2 > 1.0                                                                 <L 1232>
        var_1336 = (var_1333 > var_1335);
        // if x2_over or x2 < -1.0:                                                           <L 1233>
        var_1337 = var_1336;
        if (!var_1337) {
            var_1339 = (var_1333 < var_1338);
            var_1337 = var_1337 || var_1339;
        }
        if (var_1337) {
            // if x2_over:                                                                    <L 1234>
            if (var_1336) {
                // x2 = 1.0                                                                   <L 1235>
                // s2 = 2                                                                     <L 1236>
                // x1 = safe_div(u - mb, ma)                                                  <L 1237>
                var_1342 = wp::sub(var_1292, var_1295);
                var_1343 = safe_div_0(var_1342, var_1294);
            }
            var_1344 = wp::where(var_1336, var_1343, var_1332);
            var_1345 = wp::where(var_1336, var_1340, var_1333);
            var_1346 = wp::where(var_1336, var_1341, var_1313);
            if (!var_1336) {
                // x2 = -1.0                                                                  <L 1239>
                // s2 = 0                                                                     <L 1240>
                // x1 = safe_div(u + mb, ma)                                                  <L 1241>
                var_1349 = wp::add(var_1292, var_1295);
                var_1350 = safe_div_0(var_1349, var_1294);
            }
            var_1351 = wp::where(var_1336, var_1344, var_1350);
            var_1352 = wp::where(var_1336, var_1345, var_1347);
            var_1353 = wp::where(var_1336, var_1346, var_1348);
            // if x1 > 1:                                                                     <L 1243>
            var_1355 = (var_1351 > var_1354);
            if (var_1355) {
                // x1 = 1.0                                                                   <L 1244>
                // s1 = 2                                                                     <L 1245>
            }
            var_1358 = wp::where(var_1355, var_1356, var_1351);
            var_1359 = wp::where(var_1355, var_1357, var_1334);
            if (!var_1355) {
                // elif x1 < -1:                                                              <L 1246>
                var_1361 = (var_1358 < var_1360);
                if (var_1361) {
                    // x1 = -1.0                                                              <L 1247>
                    // s1 = 0                                                                 <L 1248>
                }
                var_1364 = wp::where(var_1361, var_1362, var_1358);
                var_1365 = wp::where(var_1361, var_1363, var_1359);
            }
            var_1366 = wp::where(var_1355, var_1358, var_1364);
            var_1367 = wp::where(var_1355, var_1359, var_1365);
        }
        var_1368 = wp::where(var_1337, var_1366, var_1332);
        var_1369 = wp::where(var_1337, var_1352, var_1333);
        var_1370 = wp::where(var_1337, var_1367, var_1334);
        var_1371 = wp::where(var_1337, var_1353, var_1313);
        // dif -= halfaxis * x2                                                               <L 1250>
        var_1372 = wp::mul(var_4, var_1369);
        var_1373 = wp::sub(var_1291, var_1372);
        // dif[j] += box_size[j] * x1                                                         <L 1251>
        var_1374 = wp::extract(var_box_size, var_1244);
        var_1375 = wp::mul(var_1374, var_1368);
        wp::add_inplace(var_1373, var_1244, var_1375);
        // ct = s1 * 3 + s2                                                                   <L 1254>
        var_1377 = wp::mul(var_1370, var_1376);
        var_1378 = wp::add(var_1377, var_1371);
        // dif_sq = wp.length_sq(dif)                                                         <L 1256>
        var_1379 = wp::length_sq(var_1373);
        // if dif_sq < bestdist - MJ_MINVAL:                                                  <L 1257>
        var_1380 = wp::sub(var_27, var_183);
        var_1381 = (var_1379 < var_1380);
        if (var_1381) {
            // bestdist = dif_sq                                                              <L 1258>
            var_1382 = wp::copy(var_1379);
            // bestsegmentpos = x2                                                            <L 1259>
            var_1383 = wp::copy(var_1369);
            // bestboxpos = x1                                                                <L 1260>
            var_1384 = wp::copy(var_1368);
            // c2 = ct // 6                                                                   <L 1262>
            var_1386 = wp::floordiv(var_1378, var_1385);
            // clcorner = i + (1 << j) * c2  # index of closest box corner                    <L 1264>
            var_1388 = wp::lshift(var_1387, var_1244);
            var_1389 = wp::mul(var_1388, var_1386);
            var_1390 = wp::add(var_1241, var_1389);
            // cledge = j  # axis index of closest box edge                                   <L 1265>
            var_1391 = wp::copy(var_1244);
            // cltype = ct  # encoded collision configuration                                 <L 1266>
            var_1392 = wp::copy(var_1378);
        }
        var_1393 = wp::where(var_1381, var_1382, var_27);
        var_1394 = wp::where(var_1381, var_1383, var_29);
        var_1395 = wp::where(var_1381, var_1392, var_31);
        var_1396 = wp::where(var_1381, var_1390, var_131);
        var_1397 = wp::where(var_1381, var_1391, var_133);
        var_1398 = wp::where(var_1381, var_1384, var_135);
        var_1399 = wp::where(var_1381, var_1386, var_1289);
        wp::assign(var_27, var_1393);
        wp::assign(var_29, var_1394);
        wp::assign(var_31, var_1395);
        wp::assign(var_131, var_1396);
        wp::assign(var_133, var_1397);
        wp::assign(var_135, var_1398);
        wp::assign(var_286, var_1399);
        wp::assign(var_163, var_1290);
        wp::assign(var_260, var_1373);
        wp::assign(var_169, var_1292);
        wp::assign(var_170, var_1293);
        wp::assign(var_173, var_1294);
        wp::assign(var_177, var_1295);
        wp::assign(var_178, var_1296);
        wp::assign(var_181, var_1297);
        wp::assign(var_186, var_1299);
        wp::assign(var_255, var_1368);
        wp::assign(var_256, var_1369);
        wp::assign(var_257, var_1370);
        wp::assign(var_258, var_1371);
        wp::assign(var_223, var_1336);
        wp::assign(var_265, var_1378);
        wp::assign(var_266, var_1379);
        goto start_for_16;
    end_for_16:;
    // best = wp.float32(0.0)                                                                 <L 1268>
    var_1401 = wp::float32(var_1400);
    // p = wp.vec2(pos.x, pos.y)                                                              <L 1270>
    var_1403 = wp::extract(var_2, var_1402);
    var_1405 = wp::extract(var_2, var_1404);
    var_1406 = wp::vec_t<2, wp::float32>(var_1403, var_1405);
    // dd = wp.vec2(halfaxis.x, halfaxis.y)                                                   <L 1271>
    var_1408 = wp::extract(var_4, var_1407);
    var_1410 = wp::extract(var_4, var_1409);
    var_1411 = wp::vec_t<2, wp::float32>(var_1408, var_1410);
    // s = wp.vec2(box_size[0], box_size[1])                                                  <L 1272>
    var_1413 = wp::extract(var_box_size, var_1412);
    var_1415 = wp::extract(var_box_size, var_1414);
    var_1416 = wp::vec_t<2, wp::float32>(var_1413, var_1415);
    // secondpos = wp.float32(-4.0)                                                           <L 1273>
    var_1418 = wp::float32(var_1417);
    // uu = dd.x * s.y                                                                        <L 1275>
    var_1420 = wp::extract(var_1411, var_1419);
    var_1422 = wp::extract(var_1416, var_1421);
    var_1423 = wp::mul(var_1420, var_1422);
    // vv = dd.y * s.x                                                                        <L 1276>
    var_1425 = wp::extract(var_1411, var_1424);
    var_1427 = wp::extract(var_1416, var_1426);
    var_1428 = wp::mul(var_1425, var_1427);
    // w_neg = dd.x * p.y - dd.y * p.x < 0                                                    <L 1277>
    var_1430 = wp::extract(var_1411, var_1429);
    var_1432 = wp::extract(var_1406, var_1431);
    var_1433 = wp::mul(var_1430, var_1432);
    var_1435 = wp::extract(var_1411, var_1434);
    var_1437 = wp::extract(var_1406, var_1436);
    var_1438 = wp::mul(var_1435, var_1437);
    var_1439 = wp::sub(var_1433, var_1438);
    var_1441 = (var_1439 < var_1440);
    // best = wp.float32(-1.0)                                                                <L 1279>
    var_1443 = wp::float32(var_1442);
    // ee1 = uu - vv                                                                          <L 1281>
    var_1444 = wp::sub(var_1423, var_1428);
    // ee2 = uu + vv                                                                          <L 1282>
    var_1445 = wp::add(var_1423, var_1428);
    // if wp.abs(ee1) > best:                                                                 <L 1284>
    var_1446 = wp::abs(var_1444);
    var_1447 = (var_1446 > var_1443);
    if (var_1447) {
        // best = wp.abs(ee1)                                                                 <L 1285>
        var_1448 = wp::abs(var_1444);
        // c1 = wp.where((ee1 < 0) == w_neg, 0, 3)                                            <L 1286>
        var_1450 = (var_1444 < var_1449);
        var_1451 = (var_1450 == var_1441);
        var_1454 = wp::where(var_1451, var_1452, var_1453);
    }
    var_1455 = wp::where(var_1447, var_1448, var_1443);
    // if wp.abs(ee2) > best:                                                                 <L 1288>
    var_1456 = wp::abs(var_1445);
    var_1457 = (var_1456 > var_1455);
    if (var_1457) {
        // best = wp.abs(ee2)                                                                 <L 1289>
        var_1458 = wp::abs(var_1445);
        // c1 = wp.where((ee2 > 0) == w_neg, 1, 2)                                            <L 1290>
        var_1460 = (var_1445 > var_1459);
        var_1461 = (var_1460 == var_1441);
        var_1464 = wp::where(var_1461, var_1462, var_1463);
    }
    var_1465 = wp::where(var_1457, var_1458, var_1455);
    var_1466 = wp::where(var_1457, var_1464, var_1454);
    // if cltype == -4:  # invalid type                                                       <L 1292>
    var_1468 = (var_31 == var_1467);
    if (var_1468) {
        // return wp.vec2(MJ_MAXVAL), mat23f(), mat23f()                                      <L 1293>
        var_1470 = wp::vec_t<2, wp::float32>(var_1469);
        var_1471 = wp::mat_t<2, 3, wp::float32>();
        var_1472 = wp::mat_t<2, 3, wp::float32>();
        ret_0 = var_1470;
        ret_1 = var_1471;
        ret_2 = var_1472;
        return;
    }
    // if cltype >= 0 and cltype // 3 != 1:  # closest to a corner of the box                 <L 1295>
    var_1475 = (var_31 >= var_1474);
    var_1473 = var_1475;
    if (var_1473) {
        var_1477 = wp::floordiv(var_31, var_1476);
        var_1479 = (var_1477 != var_1478);
        var_1473 = var_1473 && var_1479;
    }
    if (var_1473) {
        // c1 = axisdir ^ clcorner                                                            <L 1296>
        var_1480 = wp::bit_xor(var_25, var_131);
        // if c1 != 0 and c1 != 7:  # create second contact point                             <L 1301>
        var_1483 = (var_1480 != var_1482);
        var_1481 = var_1483;
        if (var_1481) {
            var_1485 = (var_1480 != var_1484);
            var_1481 = var_1481 && var_1485;
        }
        if (var_1481) {
            // if c1 == 1 or c1 == 2 or c1 == 4:                                              <L 1302>
            var_1488 = (var_1480 == var_1487);
            var_1486 = var_1488;
            if (!var_1486) {
                var_1490 = (var_1480 == var_1489);
                var_1486 = var_1486 || var_1490;
            }
            if (!var_1486) {
                var_1492 = (var_1480 == var_1491);
                var_1486 = var_1486 || var_1492;
            }
            if (var_1486) {
                // mul = 1                                                                    <L 1303>
            }
            if (!var_1486) {
                // mul = -1                                                                   <L 1305>
                // c1 = 7 - c1                                                                <L 1306>
                var_1496 = wp::sub(var_1495, var_1480);
            }
            var_1497 = wp::where(var_1486, var_1480, var_1496);
            var_1498 = wp::where(var_1486, var_1493, var_1494);
            // if c1 == 1:                                                                    <L 1311>
            var_1500 = (var_1497 == var_1499);
            if (var_1500) {
                // ax = 0                                                                     <L 1312>
                // ax1 = 1                                                                    <L 1313>
                // ax2 = 2                                                                    <L 1314>
            }
            if (!var_1500) {
                // elif c1 == 2:                                                              <L 1315>
                var_1505 = (var_1497 == var_1504);
                if (var_1505) {
                    // ax = 1                                                                 <L 1316>
                    // ax1 = 2                                                                <L 1317>
                    // ax2 = 0                                                                <L 1318>
                }
                var_1509 = wp::where(var_1505, var_1506, var_1501);
                var_1510 = wp::where(var_1505, var_1507, var_1502);
                var_1511 = wp::where(var_1505, var_1508, var_1503);
                if (!var_1505) {
                    // elif c1 == 4:                                                          <L 1319>
                    var_1513 = (var_1497 == var_1512);
                    if (var_1513) {
                        // ax = 2                                                             <L 1320>
                        // ax1 = 0                                                            <L 1321>
                        // ax2 = 1                                                            <L 1322>
                    }
                    var_1517 = wp::where(var_1513, var_1514, var_1509);
                    var_1518 = wp::where(var_1513, var_1515, var_1510);
                    var_1519 = wp::where(var_1513, var_1516, var_1511);
                }
                var_1520 = wp::where(var_1505, var_1509, var_1517);
                var_1521 = wp::where(var_1505, var_1510, var_1518);
                var_1522 = wp::where(var_1505, var_1511, var_1519);
            }
            var_1523 = wp::where(var_1500, var_1501, var_1520);
            var_1524 = wp::where(var_1500, var_1502, var_1521);
            var_1525 = wp::where(var_1500, var_1503, var_1522);
            // if axis[ax] * axis[ax] > 0.5:  # second point along the edge of the box        <L 1324>
            var_1526 = wp::extract(var_3, var_1523);
            var_1527 = wp::extract(var_3, var_1523);
            var_1528 = wp::mul(var_1526, var_1527);
            var_1530 = (var_1528 > var_1529);
            if (var_1530) {
                // m = 2.0 * safe_div(box_size[ax], wp.abs(halfaxis[ax]))                     <L 1325>
                var_1532 = wp::extract(var_box_size, var_1523);
                var_1533 = wp::extract(var_4, var_1523);
                var_1534 = wp::abs(var_1533);
                var_1535 = safe_div_0(var_1532, var_1534);
                var_1536 = wp::mul(var_1531, var_1535);
                // secondpos = min(1.0 - wp.float32(mul) * bestsegmentpos, m)                 <L 1326>
                var_1538 = wp::float32(var_1498);
                var_1539 = wp::mul(var_1538, var_29);
                var_1540 = wp::sub(var_1537, var_1539);
                var_1541 = wp::min(var_1540, var_1536);
            }
            var_1542 = wp::where(var_1530, var_1541, var_1418);
            if (!var_1530) {
                // m = 2.0 * min(                                                             <L 1329>
                // safe_div(box_size[ax1], wp.abs(halfaxis[ax1])),                            <L 1330>
                var_1544 = wp::extract(var_box_size, var_1524);
                var_1545 = wp::extract(var_4, var_1524);
                var_1546 = wp::abs(var_1545);
                var_1547 = safe_div_0(var_1544, var_1546);
                // safe_div(box_size[ax2], wp.abs(halfaxis[ax2])),                            <L 1331>
                var_1548 = wp::extract(var_box_size, var_1525);
                var_1549 = wp::extract(var_4, var_1525);
                var_1550 = wp::abs(var_1549);
                var_1551 = safe_div_0(var_1548, var_1550);
                var_1552 = wp::min(var_1547, var_1551);
                var_1553 = wp::mul(var_1543, var_1552);
                // secondpos = -min(1.0 + wp.float32(mul) * bestsegmentpos, m)                <L 1333>
                var_1555 = wp::float32(var_1498);
                var_1556 = wp::mul(var_1555, var_29);
                var_1557 = wp::add(var_1554, var_1556);
                var_1558 = wp::min(var_1557, var_1553);
                var_1559 = wp::neg(var_1558);
            }
            var_1560 = wp::where(var_1530, var_1542, var_1559);
            var_1561 = wp::where(var_1530, var_1536, var_1553);
            // secondpos *= wp.float32(mul)                                                   <L 1334>
            var_1562 = wp::float32(var_1498);
            var_1563 = wp::mul(var_1560, var_1562);
        }
        var_1564 = wp::where(var_1481, var_1563, var_1418);
        var_1565 = wp::where(var_1481, var_1497, var_1480);
    }
    var_1566 = wp::where(var_1473, var_1564, var_1418);
    var_1567 = wp::where(var_1473, var_1565, var_1466);
    if (!var_1473) {
        // elif cltype >= 0 and cltype // 3 == 1:  # we are on box's edge                     <L 1336>
        var_1570 = (var_31 >= var_1569);
        var_1568 = var_1570;
        if (var_1568) {
            var_1572 = wp::floordiv(var_31, var_1571);
            var_1574 = (var_1572 == var_1573);
            var_1568 = var_1568 && var_1574;
        }
        if (var_1568) {
            // c1 = axisdir ^ clcorner                                                        <L 1341>
            var_1575 = wp::bit_xor(var_25, var_131);
            // c1 &= 7 - (1 << cledge)  # mask out edge axis to determine configuration       <L 1342>
            var_1578 = wp::lshift(var_1577, var_133);
            var_1579 = wp::sub(var_1576, var_1578);
            var_1580 = wp::bit_and(var_1575, var_1579);
            // if c1 == 1 or c1 == 2 or c1 == 4:  # create second contact point               <L 1344>
            var_1583 = (var_1580 == var_1582);
            var_1581 = var_1583;
            if (!var_1581) {
                var_1585 = (var_1580 == var_1584);
                var_1581 = var_1581 || var_1585;
            }
            if (!var_1581) {
                var_1587 = (var_1580 == var_1586);
                var_1581 = var_1581 || var_1587;
            }
            if (var_1581) {
                // if cledge == 0:                                                            <L 1345>
                var_1589 = (var_133 == var_1588);
                if (var_1589) {
                    // ax1 = 1                                                                <L 1346>
                    // ax2 = 2                                                                <L 1347>
                }
                var_1592 = wp::where(var_1589, var_1590, var_1524);
                var_1593 = wp::where(var_1589, var_1591, var_1525);
                // if cledge == 1:                                                            <L 1348>
                var_1595 = (var_133 == var_1594);
                if (var_1595) {
                    // ax1 = 2                                                                <L 1349>
                    // ax2 = 0                                                                <L 1350>
                }
                var_1598 = wp::where(var_1595, var_1596, var_1592);
                var_1599 = wp::where(var_1595, var_1597, var_1593);
                // if cledge == 2:                                                            <L 1351>
                var_1601 = (var_133 == var_1600);
                if (var_1601) {
                    // ax1 = 0                                                                <L 1352>
                    // ax2 = 1                                                                <L 1353>
                }
                var_1604 = wp::where(var_1601, var_1602, var_1598);
                var_1605 = wp::where(var_1601, var_1603, var_1599);
                // ax = cledge                                                                <L 1354>
                var_1606 = wp::copy(var_133);
                // if wp.abs(axis[ax1]) > wp.abs(axis[ax2]):                                  <L 1357>
                var_1607 = wp::extract(var_3, var_1604);
                var_1608 = wp::abs(var_1607);
                var_1609 = wp::extract(var_3, var_1605);
                var_1610 = wp::abs(var_1609);
                var_1611 = (var_1608 > var_1610);
                if (var_1611) {
                    // ax1 = ax2                                                              <L 1358>
                    var_1612 = wp::copy(var_1605);
                }
                var_1613 = wp::where(var_1611, var_1612, var_1604);
                // ax2 = 3 - ax - ax1                                                         <L 1359>
                var_1615 = wp::sub(var_1614, var_1606);
                var_1616 = wp::sub(var_1615, var_1613);
                // if c1 & (1 << ax2):                                                        <L 1362>
                var_1618 = wp::lshift(var_1617, var_1616);
                var_1619 = wp::bit_and(var_1580, var_1618);
                if (var_1619) {
                    // mul = 1                                                                <L 1363>
                    // secondpos = 1.0 - bestsegmentpos                                       <L 1364>
                    var_1622 = wp::sub(var_1621, var_29);
                }
                var_1623 = wp::where(var_1619, var_1622, var_1566);
                var_1624 = wp::where(var_1619, var_1620, var_1498);
                if (!var_1619) {
                    // mul = -1                                                               <L 1366>
                    // secondpos = 1.0 + bestsegmentpos                                       <L 1367>
                    var_1627 = wp::add(var_1626, var_29);
                }
                var_1628 = wp::where(var_1619, var_1623, var_1627);
                var_1629 = wp::where(var_1619, var_1624, var_1625);
                // e1 = 2.0 * safe_div(box_size[ax2], wp.abs(halfaxis[ax2]))                  <L 1372>
                var_1631 = wp::extract(var_box_size, var_1616);
                var_1632 = wp::extract(var_4, var_1616);
                var_1633 = wp::abs(var_1632);
                var_1634 = safe_div_0(var_1631, var_1633);
                var_1635 = wp::mul(var_1630, var_1634);
                // secondpos = min(e1, secondpos)                                             <L 1373>
                var_1636 = wp::min(var_1635, var_1628);
                // if ((axisdir & (1 << ax)) != 0) == ((c1 & (1 << ax2)) != 0):               <L 1375>
                var_1638 = wp::lshift(var_1637, var_1606);
                var_1639 = wp::bit_and(var_25, var_1638);
                var_1641 = (var_1639 != var_1640);
                var_1643 = wp::lshift(var_1642, var_1616);
                var_1644 = wp::bit_and(var_1580, var_1643);
                var_1646 = (var_1644 != var_1645);
                var_1647 = (var_1641 == var_1646);
                if (var_1647) {
                    // e2 = 1.0 - bestboxpos                                                  <L 1376>
                    var_1649 = wp::sub(var_1648, var_135);
                }
                if (!var_1647) {
                    // e2 = 1.0 + bestboxpos                                                  <L 1378>
                    var_1651 = wp::add(var_1650, var_135);
                }
                var_1652 = wp::where(var_1647, var_1649, var_1651);
                // e1 = box_size[ax] * safe_div(e2, wp.abs(halfaxis[ax]))                     <L 1380>
                var_1653 = wp::extract(var_box_size, var_1606);
                var_1654 = wp::extract(var_4, var_1606);
                var_1655 = wp::abs(var_1654);
                var_1656 = safe_div_0(var_1652, var_1655);
                var_1657 = wp::mul(var_1653, var_1656);
                // secondpos = min(e1, secondpos)                                             <L 1382>
                var_1658 = wp::min(var_1657, var_1636);
                // secondpos *= wp.float32(mul)                                               <L 1383>
                var_1659 = wp::float32(var_1629);
                var_1660 = wp::mul(var_1658, var_1659);
            }
            var_1661 = wp::where(var_1581, var_1660, var_1566);
            var_1662 = wp::where(var_1581, var_1629, var_1498);
            var_1663 = wp::where(var_1581, var_1606, var_1523);
            var_1664 = wp::where(var_1581, var_1613, var_1524);
            var_1665 = wp::where(var_1581, var_1616, var_1525);
        }
        var_1666 = wp::where(var_1568, var_1661, var_1566);
        var_1667 = wp::where(var_1568, var_1580, var_1567);
        var_1668 = wp::where(var_1568, var_1662, var_1498);
        var_1669 = wp::where(var_1568, var_1663, var_1523);
        var_1670 = wp::where(var_1568, var_1664, var_1524);
        var_1671 = wp::where(var_1568, var_1665, var_1525);
        if (!var_1568) {
            // elif cltype < 0:                                                               <L 1385>
            var_1673 = (var_31 < var_1672);
            if (var_1673) {
                // if clface != -1:  # create second contact point                            <L 1391>
                var_1675 = (var_33 != var_1674);
                if (var_1675) {
                    // mul = wp.where(cltype == -3, 1, -1)                                    <L 1392>
                    var_1677 = (var_31 == var_1676);
                    var_1680 = wp::where(var_1677, var_1678, var_1679);
                    // secondpos = 2.0                                                        <L 1393>
                    // tmp1 = pos - halfaxis * wp.float32(mul)                                <L 1395>
                    var_1682 = wp::float32(var_1680);
                    var_1683 = wp::mul(var_4, var_1682);
                    var_1684 = wp::sub(var_2, var_1683);
                    // for i in range(3):                                                     <L 1397>
                    // if i != clface:                                                        <L 1398>
                    var_1686 = (var_1685 != var_33);
                    if (var_1686) {
                        // ha_r = safe_div(wp.float32(mul), halfaxis[i])                      <L 1399>
                        var_1687 = wp::float32(var_1680);
                        var_1688 = wp::extract(var_4, var_1685);
                        var_1689 = safe_div_0(var_1687, var_1688);
                        // e1 = (box_size[i] - tmp1[i]) * ha_r                                <L 1400>
                        var_1690 = wp::extract(var_box_size, var_1685);
                        var_1691 = wp::extract(var_1684, var_1685);
                        var_1692 = wp::sub(var_1690, var_1691);
                        var_1693 = wp::mul(var_1692, var_1689);
                        // if 0 < e1 and e1 < secondpos:                                      <L 1401>
                        var_1696 = (var_1695 < var_1693);
                        var_1694 = var_1696;
                        if (var_1694) {
                            var_1697 = (var_1693 < var_1681);
                            var_1694 = var_1694 && var_1697;
                        }
                        if (var_1694) {
                            // secondpos = e1                                                 <L 1402>
                            var_1698 = wp::copy(var_1693);
                        }
                        var_1699 = wp::where(var_1694, var_1698, var_1681);
                        // e1 = (-box_size[i] - tmp1[i]) * ha_r                               <L 1404>
                        var_1700 = wp::extract(var_box_size, var_1685);
                        var_1701 = wp::neg(var_1700);
                        var_1702 = wp::extract(var_1684, var_1685);
                        var_1703 = wp::sub(var_1701, var_1702);
                        var_1704 = wp::mul(var_1703, var_1689);
                        // if 0 < e1 and e1 < secondpos:                                      <L 1405>
                        var_1707 = (var_1706 < var_1704);
                        var_1705 = var_1707;
                        if (var_1705) {
                            var_1708 = (var_1704 < var_1699);
                            var_1705 = var_1705 && var_1708;
                        }
                        if (var_1705) {
                            // secondpos = e1                                                 <L 1406>
                            var_1709 = wp::copy(var_1704);
                        }
                        var_1710 = wp::where(var_1705, var_1709, var_1699);
                    }
                    var_1711 = wp::where(var_1686, var_1710, var_1681);
                    var_1712 = wp::where(var_1686, var_1704, var_1657);
                    // if i != clface:                                                        <L 1398>
                    var_1714 = (var_1713 != var_33);
                    if (var_1714) {
                        // ha_r = safe_div(wp.float32(mul), halfaxis[i])                      <L 1399>
                        var_1715 = wp::float32(var_1680);
                        var_1716 = wp::extract(var_4, var_1713);
                        var_1717 = safe_div_0(var_1715, var_1716);
                        // e1 = (box_size[i] - tmp1[i]) * ha_r                                <L 1400>
                        var_1718 = wp::extract(var_box_size, var_1713);
                        var_1719 = wp::extract(var_1684, var_1713);
                        var_1720 = wp::sub(var_1718, var_1719);
                        var_1721 = wp::mul(var_1720, var_1717);
                        // if 0 < e1 and e1 < secondpos:                                      <L 1401>
                        var_1724 = (var_1723 < var_1721);
                        var_1722 = var_1724;
                        if (var_1722) {
                            var_1725 = (var_1721 < var_1711);
                            var_1722 = var_1722 && var_1725;
                        }
                        if (var_1722) {
                            // secondpos = e1                                                 <L 1402>
                            var_1726 = wp::copy(var_1721);
                        }
                        var_1727 = wp::where(var_1722, var_1726, var_1711);
                        // e1 = (-box_size[i] - tmp1[i]) * ha_r                               <L 1404>
                        var_1728 = wp::extract(var_box_size, var_1713);
                        var_1729 = wp::neg(var_1728);
                        var_1730 = wp::extract(var_1684, var_1713);
                        var_1731 = wp::sub(var_1729, var_1730);
                        var_1732 = wp::mul(var_1731, var_1717);
                        // if 0 < e1 and e1 < secondpos:                                      <L 1405>
                        var_1735 = (var_1734 < var_1732);
                        var_1733 = var_1735;
                        if (var_1733) {
                            var_1736 = (var_1732 < var_1727);
                            var_1733 = var_1733 && var_1736;
                        }
                        if (var_1733) {
                            // secondpos = e1                                                 <L 1406>
                            var_1737 = wp::copy(var_1732);
                        }
                        var_1738 = wp::where(var_1733, var_1737, var_1727);
                    }
                    var_1739 = wp::where(var_1714, var_1738, var_1711);
                    var_1740 = wp::where(var_1714, var_1732, var_1712);
                    var_1741 = wp::where(var_1714, var_1717, var_1689);
                    // if i != clface:                                                        <L 1398>
                    var_1743 = (var_1742 != var_33);
                    if (var_1743) {
                        // ha_r = safe_div(wp.float32(mul), halfaxis[i])                      <L 1399>
                        var_1744 = wp::float32(var_1680);
                        var_1745 = wp::extract(var_4, var_1742);
                        var_1746 = safe_div_0(var_1744, var_1745);
                        // e1 = (box_size[i] - tmp1[i]) * ha_r                                <L 1400>
                        var_1747 = wp::extract(var_box_size, var_1742);
                        var_1748 = wp::extract(var_1684, var_1742);
                        var_1749 = wp::sub(var_1747, var_1748);
                        var_1750 = wp::mul(var_1749, var_1746);
                        // if 0 < e1 and e1 < secondpos:                                      <L 1401>
                        var_1753 = (var_1752 < var_1750);
                        var_1751 = var_1753;
                        if (var_1751) {
                            var_1754 = (var_1750 < var_1739);
                            var_1751 = var_1751 && var_1754;
                        }
                        if (var_1751) {
                            // secondpos = e1                                                 <L 1402>
                            var_1755 = wp::copy(var_1750);
                        }
                        var_1756 = wp::where(var_1751, var_1755, var_1739);
                        // e1 = (-box_size[i] - tmp1[i]) * ha_r                               <L 1404>
                        var_1757 = wp::extract(var_box_size, var_1742);
                        var_1758 = wp::neg(var_1757);
                        var_1759 = wp::extract(var_1684, var_1742);
                        var_1760 = wp::sub(var_1758, var_1759);
                        var_1761 = wp::mul(var_1760, var_1746);
                        // if 0 < e1 and e1 < secondpos:                                      <L 1405>
                        var_1764 = (var_1763 < var_1761);
                        var_1762 = var_1764;
                        if (var_1762) {
                            var_1765 = (var_1761 < var_1756);
                            var_1762 = var_1762 && var_1765;
                        }
                        if (var_1762) {
                            // secondpos = e1                                                 <L 1406>
                            var_1766 = wp::copy(var_1761);
                        }
                        var_1767 = wp::where(var_1762, var_1766, var_1756);
                    }
                    var_1768 = wp::where(var_1743, var_1767, var_1739);
                    var_1769 = wp::where(var_1743, var_1761, var_1740);
                    var_1770 = wp::where(var_1743, var_1746, var_1741);
                    // secondpos *= wp.float32(mul)                                           <L 1408>
                    var_1771 = wp::float32(var_1680);
                    var_1772 = wp::mul(var_1768, var_1771);
                }
                var_1773 = wp::where(var_1675, var_1742, var_1241);
                var_1774 = wp::where(var_1675, var_1772, var_1666);
                var_1775 = wp::where(var_1675, var_1680, var_1668);
                var_1776 = wp::where(var_1675, var_1769, var_1657);
            }
            var_1777 = wp::where(var_1673, var_1773, var_1241);
            var_1778 = wp::where(var_1673, var_1774, var_1666);
            var_1779 = wp::where(var_1673, var_1775, var_1668);
            var_1780 = wp::where(var_1673, var_1776, var_1657);
        }
        var_1781 = wp::where(var_1568, var_1241, var_1777);
        var_1782 = wp::where(var_1568, var_1666, var_1778);
        var_1783 = wp::where(var_1568, var_1668, var_1779);
        var_1784 = wp::where(var_1568, var_1657, var_1780);
    }
    var_1785 = wp::where(var_1473, var_1241, var_1781);
    var_1786 = wp::where(var_1473, var_1566, var_1782);
    var_1787 = wp::where(var_1473, var_1567, var_1667);
    var_1788 = wp::where(var_1473, var_1498, var_1783);
    var_1789 = wp::where(var_1473, var_1523, var_1669);
    var_1790 = wp::where(var_1473, var_1524, var_1670);
    var_1791 = wp::where(var_1473, var_1525, var_1671);
    // s1_pos_l = pos + halfaxis * bestsegmentpos                                             <L 1411>
    var_1792 = wp::mul(var_4, var_29);
    var_1793 = wp::add(var_2, var_1792);
    // s1_pos_g = box_rot @ s1_pos_l + box_pos                                                <L 1412>
    var_1794 = wp::mul(var_box_rot, var_1793);
    var_1795 = wp::add(var_1794, var_box_pos);
    // dist1, pos1, normal1 = sphere_box(s1_pos_g, capsule_radius, box_pos, box_rot, box_size)       <L 1415>
    sphere_box_0(var_1795, var_capsule_radius, var_box_pos, var_box_rot, var_box_size, var_1796, var_1797, var_1798);
    // if secondpos > -3:  # secondpos was modified                                           <L 1417>
    var_1800 = (var_1786 > var_1799);
    if (var_1800) {
        // s2_pos_l = pos + halfaxis * (secondpos + bestsegmentpos)                           <L 1418>
        var_1801 = wp::add(var_1786, var_29);
        var_1802 = wp::mul(var_4, var_1801);
        var_1803 = wp::add(var_2, var_1802);
        // s2_pos_g = box_rot @ s2_pos_l + box_pos                                            <L 1419>
        var_1804 = wp::mul(var_box_rot, var_1803);
        var_1805 = wp::add(var_1804, var_box_pos);
        // dist2, pos2, normal2 = sphere_box(s2_pos_g, capsule_radius, box_pos, box_rot, box_size)       <L 1422>
        sphere_box_0(var_1805, var_capsule_radius, var_box_pos, var_box_rot, var_box_size, var_1806, var_1807, var_1808);
    }
    if (!var_1800) {
        // dist2 = MJ_MAXVAL                                                                  <L 1424>
        var_1809 = wp::copy(var_1469);
        // pos2 = wp.vec3()                                                                   <L 1425>
        var_1810 = wp::vec_t<3, wp::float32>();
        // normal2 = wp.vec3()                                                                <L 1426>
        var_1811 = wp::vec_t<3, wp::float32>();
    }
    var_1812 = wp::where(var_1800, var_1806, var_1809);
    var_1813 = wp::where(var_1800, var_1807, var_1810);
    var_1814 = wp::where(var_1800, var_1808, var_1811);
    // return (                                                                               <L 1428>
    // wp.vec2(dist1, dist2),                                                                 <L 1429>
    var_1815 = wp::vec_t<2, wp::float32>(var_1796, var_1812);
    // mat23f(pos1[0], pos1[1], pos1[2], pos2[0], pos2[1], pos2[2]),                          <L 1430>
    var_1817 = wp::extract(var_1797, var_1816);
    var_1819 = wp::extract(var_1797, var_1818);
    var_1821 = wp::extract(var_1797, var_1820);
    var_1823 = wp::extract(var_1813, var_1822);
    var_1825 = wp::extract(var_1813, var_1824);
    var_1827 = wp::extract(var_1813, var_1826);
    var_1828 = wp::mat_t<2, 3, wp::float32>({var_1817, var_1819, var_1821, var_1823, var_1825, var_1827});
    // mat23f(normal1[0], normal1[1], normal1[2], normal2[0], normal2[1], normal2[2]),        <L 1431>
    var_1830 = wp::extract(var_1798, var_1829);
    var_1832 = wp::extract(var_1798, var_1831);
    var_1834 = wp::extract(var_1798, var_1833);
    var_1836 = wp::extract(var_1814, var_1835);
    var_1838 = wp::extract(var_1814, var_1837);
    var_1840 = wp::extract(var_1814, var_1839);
    var_1841 = wp::mat_t<2, 3, wp::float32>({var_1830, var_1832, var_1834, var_1836, var_1838, var_1840});
    ret_0 = var_1815;
    ret_1 = var_1828;
    ret_2 = var_1841;
    return;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_primitive.py:1116
static CUDA_CALLABLE void capsule_box_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_3242f8a8 var_cap,
    Geom_3242f8a8 var_box,
    wp::int32 var_worldid,
    wp::float32 var_margin,
    wp::float32 var_gap,
    wp::int32 var_condim,
    wp::vec_t<5, wp::float32> var_friction,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<2, wp::float32> var_solreffriction,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::vec_t<2, wp::int32> var_pairid,
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
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::int32> var_nacon_out)
{
    //---------
    // primal vars
    wp::mat_t<3, 3, wp::float32>* var_0;
    const wp::int32 var_1 = 0;
    const wp::int32 var_2 = 2;
    wp::float32 var_3;
    wp::mat_t<3, 3, wp::float32> var_4;
    wp::mat_t<3, 3, wp::float32>* var_5;
    const wp::int32 var_6 = 1;
    const wp::int32 var_7 = 2;
    wp::float32 var_8;
    wp::mat_t<3, 3, wp::float32> var_9;
    wp::mat_t<3, 3, wp::float32>* var_10;
    const wp::int32 var_11 = 2;
    const wp::int32 var_12 = 2;
    wp::float32 var_13;
    wp::mat_t<3, 3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::vec_t<3, wp::float32>* var_16;
    wp::vec_t<3, wp::float32>* var_17;
    const wp::int32 var_18 = 0;
    wp::float32 var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32>* var_21;
    const wp::int32 var_22 = 1;
    wp::float32 var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::vec_t<3, wp::float32>* var_25;
    wp::mat_t<3, 3, wp::float32>* var_26;
    wp::vec_t<3, wp::float32>* var_27;
    wp::vec_t<2, wp::float32> var_28;
    wp::mat_t<2, 3, wp::float32> var_29;
    wp::mat_t<2, 3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::mat_t<3, 3, wp::float32> var_33;
    wp::vec_t<3, wp::float32> var_34;
    const wp::int32 var_35 = 0;
    wp::float32 var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::mat_t<3, 3, wp::float32> var_39;
    wp::int32 var_40;
    const wp::int32 var_41 = 1;
    wp::float32 var_42;
    wp::vec_t<3, wp::float32> var_43;
    wp::vec_t<3, wp::float32> var_44;
    wp::mat_t<3, 3, wp::float32> var_45;
    wp::int32 var_46;
    //---------
    // forward
    // def capsule_box_wrapper(                                                               <L 1117>
    // axis = wp.vec3(cap.rot[0, 2], cap.rot[1, 2], cap.rot[2, 2])                            <L 1152>
    var_0 = &((var_cap).rot);
    var_4 = wp::load(var_0);
    var_3 = wp::extract(var_4, var_1, var_2);
    var_5 = &((var_cap).rot);
    var_9 = wp::load(var_5);
    var_8 = wp::extract(var_9, var_6, var_7);
    var_10 = &((var_cap).rot);
    var_14 = wp::load(var_10);
    var_13 = wp::extract(var_14, var_11, var_12);
    var_15 = wp::vec_t<3, wp::float32>(var_3, var_8, var_13);
    // dist, pos, normal = capsule_box(                                                       <L 1155>
    // cap.pos,                                                                               <L 1156>
    var_16 = &((var_cap).pos);
    // axis,                                                                                  <L 1157>
    // cap.size[0],  # capsule radius                                                         <L 1158>
    var_17 = &((var_cap).size);
    var_20 = wp::load(var_17);
    var_19 = wp::extract(var_20, var_18);
    // cap.size[1],  # capsule half length                                                    <L 1159>
    var_21 = &((var_cap).size);
    var_24 = wp::load(var_21);
    var_23 = wp::extract(var_24, var_22);
    // box.pos,                                                                               <L 1160>
    var_25 = &((var_box).pos);
    // box.rot,                                                                               <L 1161>
    var_26 = &((var_box).rot);
    // box.size,                                                                              <L 1162>
    var_27 = &((var_box).size);
    var_31 = wp::load(var_16);
    var_32 = wp::load(var_25);
    var_33 = wp::load(var_26);
    var_34 = wp::load(var_27);
    capsule_box_0(var_31, var_15, var_19, var_23, var_32, var_33, var_34, var_28, var_29, var_30);
    // for i in range(2):                                                                     <L 1166>
    // write_contact(                                                                         <L 1167>
    // naconmax_in,                                                                           <L 1168>
    // i,                                                                                     <L 1169>
    // dist[i],                                                                               <L 1170>
    var_36 = wp::extract(var_28, var_35);
    // pos[i],                                                                                <L 1171>
    var_37 = wp::extract(var_29, var_35);
    // make_frame(normal[i]),                                                                 <L 1172>
    var_38 = wp::extract(var_30, var_35);
    var_39 = make_frame_0(var_38);
    // margin,                                                                                <L 1173>
    // gap,                                                                                   <L 1174>
    // condim,                                                                                <L 1175>
    // friction,                                                                              <L 1176>
    // solref,                                                                                <L 1177>
    // solreffriction,                                                                        <L 1178>
    // solimp,                                                                                <L 1179>
    // geoms,                                                                                 <L 1180>
    // pairid,                                                                                <L 1181>
    // worldid,                                                                               <L 1182>
    // contact_dist_out,                                                                      <L 1183>
    // contact_pos_out,                                                                       <L 1184>
    // contact_frame_out,                                                                     <L 1185>
    // contact_includemargin_out,                                                             <L 1186>
    // contact_friction_out,                                                                  <L 1187>
    // contact_solref_out,                                                                    <L 1188>
    // contact_solreffriction_out,                                                            <L 1189>
    // contact_solimp_out,                                                                    <L 1190>
    // contact_dim_out,                                                                       <L 1191>
    // contact_geom_out,                                                                      <L 1192>
    // contact_efc_address_out,                                                               <L 1193>
    // contact_worldid_out,                                                                   <L 1194>
    // contact_type_out,                                                                      <L 1195>
    // contact_geomcollisionid_out,                                                           <L 1196>
    // nacon_out,                                                                             <L 1197>
    var_40 = write_contact_0(var_naconmax_in, var_35, var_36, var_37, var_39, var_margin, var_gap, var_condim, var_friction, var_solref, var_solreffriction, var_solimp, var_geoms, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_nacon_out);
    // write_contact(                                                                         <L 1167>
    // naconmax_in,                                                                           <L 1168>
    // i,                                                                                     <L 1169>
    // dist[i],                                                                               <L 1170>
    var_42 = wp::extract(var_28, var_41);
    // pos[i],                                                                                <L 1171>
    var_43 = wp::extract(var_29, var_41);
    // make_frame(normal[i]),                                                                 <L 1172>
    var_44 = wp::extract(var_30, var_41);
    var_45 = make_frame_0(var_44);
    // margin,                                                                                <L 1173>
    // gap,                                                                                   <L 1174>
    // condim,                                                                                <L 1175>
    // friction,                                                                              <L 1176>
    // solref,                                                                                <L 1177>
    // solreffriction,                                                                        <L 1178>
    // solimp,                                                                                <L 1179>
    // geoms,                                                                                 <L 1180>
    // pairid,                                                                                <L 1181>
    // worldid,                                                                               <L 1182>
    // contact_dist_out,                                                                      <L 1183>
    // contact_pos_out,                                                                       <L 1184>
    // contact_frame_out,                                                                     <L 1185>
    // contact_includemargin_out,                                                             <L 1186>
    // contact_friction_out,                                                                  <L 1187>
    // contact_solref_out,                                                                    <L 1188>
    // contact_solreffriction_out,                                                            <L 1189>
    // contact_solimp_out,                                                                    <L 1190>
    // contact_dim_out,                                                                       <L 1191>
    // contact_geom_out,                                                                      <L 1192>
    // contact_efc_address_out,                                                               <L 1193>
    // contact_worldid_out,                                                                   <L 1194>
    // contact_type_out,                                                                      <L 1195>
    // contact_geomcollisionid_out,                                                           <L 1196>
    // nacon_out,                                                                             <L 1197>
    var_46 = write_contact_0(var_naconmax_in, var_41, var_42, var_43, var_45, var_margin, var_gap, var_condim, var_friction, var_solref, var_solreffriction, var_solimp, var_geoms, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_nacon_out);
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_core.py:290
static CUDA_CALLABLE void adj_contact_margin_gap_0(
    wp::array_t<wp::float32> var_geom_margin,
    wp::array_t<wp::float32> var_geom_gap,
    wp::array_t<wp::float32> var_pair_margin,
    wp::array_t<wp::float32> var_pair_gap,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::int32 var_pairid,
    wp::int32 var_worldid,
    wp::float32 & ret_0,
    wp::float32 & ret_1,
    wp::array_t<wp::float32> & adj_geom_margin,
    wp::array_t<wp::float32> & adj_geom_gap,
    wp::array_t<wp::float32> & adj_pair_margin,
    wp::array_t<wp::float32> & adj_pair_gap,
    wp::vec_t<2, wp::int32> & adj_geoms,
    wp::int32 & adj_pairid,
    wp::int32 & adj_worldid,
    wp::float32 & adj_ret_0,
    wp::float32 & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/math.py:0
static CUDA_CALLABLE void adj_safe_div_0(
    wp::float32 var_x,
    wp::float32 var_y,
    wp::float32 & adj_x,
    wp::float32 & adj_y,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_core.py:316
static CUDA_CALLABLE void adj_contact_material_params_0(
    wp::array_t<wp::int32> var_geom_condim,
    wp::array_t<wp::int32> var_geom_priority,
    wp::array_t<wp::float32> var_geom_solmix,
    wp::array_t<wp::vec_t<2, wp::float32>> var_geom_solref,
    wp::array_t<wp::vec_t<5, wp::float32>> var_geom_solimp,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_friction,
    wp::array_t<wp::int32> var_pair_dim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_pair_solref,
    wp::array_t<wp::vec_t<2, wp::float32>> var_pair_solreffriction,
    wp::array_t<wp::vec_t<5, wp::float32>> var_pair_solimp,
    wp::array_t<wp::vec_t<5, wp::float32>> var_pair_friction,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::int32 var_pairid,
    wp::int32 var_worldid,
    wp::int32 & ret_0,
    wp::vec_t<5, wp::float32> & ret_1,
    wp::vec_t<2, wp::float32> & ret_2,
    wp::vec_t<2, wp::float32> & ret_3,
    wp::vec_t<5, wp::float32> & ret_4,
    wp::array_t<wp::int32> & adj_geom_condim,
    wp::array_t<wp::int32> & adj_geom_priority,
    wp::array_t<wp::float32> & adj_geom_solmix,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_geom_solref,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_geom_solimp,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_geom_friction,
    wp::array_t<wp::int32> & adj_pair_dim,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_pair_solref,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_pair_solreffriction,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_pair_solimp,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_pair_friction,
    wp::vec_t<2, wp::int32> & adj_geoms,
    wp::int32 & adj_pairid,
    wp::int32 & adj_worldid,
    wp::int32 & adj_ret_0,
    wp::vec_t<5, wp::float32> & adj_ret_1,
    wp::vec_t<2, wp::float32> & adj_ret_2,
    wp::vec_t<2, wp::float32> & adj_ret_3,
    wp::vec_t<5, wp::float32> & adj_ret_4)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_core.py:402
static CUDA_CALLABLE void adj_contact_params_0(
    wp::array_t<wp::int32> var_geom_condim,
    wp::array_t<wp::int32> var_geom_priority,
    wp::array_t<wp::float32> var_geom_solmix,
    wp::array_t<wp::vec_t<2, wp::float32>> var_geom_solref,
    wp::array_t<wp::vec_t<5, wp::float32>> var_geom_solimp,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_friction,
    wp::array_t<wp::float32> var_geom_margin,
    wp::array_t<wp::float32> var_geom_gap,
    wp::array_t<wp::int32> var_pair_dim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_pair_solref,
    wp::array_t<wp::vec_t<2, wp::float32>> var_pair_solreffriction,
    wp::array_t<wp::vec_t<5, wp::float32>> var_pair_solimp,
    wp::array_t<wp::float32> var_pair_margin,
    wp::array_t<wp::float32> var_pair_gap,
    wp::array_t<wp::vec_t<5, wp::float32>> var_pair_friction,
    wp::array_t<wp::vec_t<2, wp::int32>> var_collision_pair_in,
    wp::array_t<wp::vec_t<2, wp::int32>> var_collision_pairid_in,
    wp::int32 var_cid,
    wp::int32 var_worldid,
    wp::vec_t<2, wp::int32> & ret_0,
    wp::float32 & ret_1,
    wp::float32 & ret_2,
    wp::int32 & ret_3,
    wp::vec_t<5, wp::float32> & ret_4,
    wp::vec_t<2, wp::float32> & ret_5,
    wp::vec_t<2, wp::float32> & ret_6,
    wp::vec_t<5, wp::float32> & ret_7,
    wp::array_t<wp::int32> & adj_geom_condim,
    wp::array_t<wp::int32> & adj_geom_priority,
    wp::array_t<wp::float32> & adj_geom_solmix,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_geom_solref,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_geom_solimp,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_geom_friction,
    wp::array_t<wp::float32> & adj_geom_margin,
    wp::array_t<wp::float32> & adj_geom_gap,
    wp::array_t<wp::int32> & adj_pair_dim,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_pair_solref,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_pair_solreffriction,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_pair_solimp,
    wp::array_t<wp::float32> & adj_pair_margin,
    wp::array_t<wp::float32> & adj_pair_gap,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_pair_friction,
    wp::array_t<wp::vec_t<2, wp::int32>> & adj_collision_pair_in,
    wp::array_t<wp::vec_t<2, wp::int32>> & adj_collision_pairid_in,
    wp::int32 & adj_cid,
    wp::int32 & adj_worldid,
    wp::vec_t<2, wp::int32> & adj_ret_0,
    wp::float32 & adj_ret_1,
    wp::float32 & adj_ret_2,
    wp::int32 & adj_ret_3,
    wp::vec_t<5, wp::float32> & adj_ret_4,
    wp::vec_t<2, wp::float32> & adj_ret_5,
    wp::vec_t<2, wp::float32> & adj_ret_6,
    wp::vec_t<5, wp::float32> & adj_ret_7)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_core.py:65
static CUDA_CALLABLE void adj_geom_collision_pair_from_types_0(
    wp::array_t<wp::int32> var_geom_dataid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_size,
    wp::array_t<wp::int32> var_mesh_vertadr,
    wp::array_t<wp::int32> var_mesh_vertnum,
    wp::array_t<wp::int32> var_mesh_graphadr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_mesh_vert,
    wp::array_t<wp::int32> var_mesh_graph,
    wp::array_t<wp::int32> var_mesh_polynum,
    wp::array_t<wp::int32> var_mesh_polyadr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_mesh_polynormal,
    wp::array_t<wp::int32> var_mesh_polyvertadr,
    wp::array_t<wp::int32> var_mesh_polyvertnum,
    wp::array_t<wp::int32> var_mesh_polyvert,
    wp::array_t<wp::int32> var_mesh_polymapadr,
    wp::array_t<wp::int32> var_mesh_polymapnum,
    wp::array_t<wp::int32> var_mesh_polymap,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::int32 var_geom_type1,
    wp::int32 var_geom_type2,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::int32 var_worldid,
    Geom_3242f8a8 & ret_0,
    Geom_3242f8a8 & ret_1,
    wp::array_t<wp::int32> & adj_geom_dataid,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_geom_size,
    wp::array_t<wp::int32> & adj_mesh_vertadr,
    wp::array_t<wp::int32> & adj_mesh_vertnum,
    wp::array_t<wp::int32> & adj_mesh_graphadr,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_mesh_vert,
    wp::array_t<wp::int32> & adj_mesh_graph,
    wp::array_t<wp::int32> & adj_mesh_polynum,
    wp::array_t<wp::int32> & adj_mesh_polyadr,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_mesh_polynormal,
    wp::array_t<wp::int32> & adj_mesh_polyvertadr,
    wp::array_t<wp::int32> & adj_mesh_polyvertnum,
    wp::array_t<wp::int32> & adj_mesh_polyvert,
    wp::array_t<wp::int32> & adj_mesh_polymapadr,
    wp::array_t<wp::int32> & adj_mesh_polymapnum,
    wp::array_t<wp::int32> & adj_mesh_polymap,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_geom_xmat_in,
    wp::int32 & adj_geom_type1,
    wp::int32 & adj_geom_type2,
    wp::vec_t<2, wp::int32> & adj_geoms,
    wp::int32 & adj_worldid,
    Geom_3242f8a8 & adj_ret_0,
    Geom_3242f8a8 & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_core.py:158
static CUDA_CALLABLE void adj_geom_collision_pair_0(
    wp::array_t<wp::int32> var_geom_type,
    wp::array_t<wp::int32> var_geom_dataid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_size,
    wp::array_t<wp::int32> var_mesh_vertadr,
    wp::array_t<wp::int32> var_mesh_vertnum,
    wp::array_t<wp::int32> var_mesh_graphadr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_mesh_vert,
    wp::array_t<wp::int32> var_mesh_graph,
    wp::array_t<wp::int32> var_mesh_polynum,
    wp::array_t<wp::int32> var_mesh_polyadr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_mesh_polynormal,
    wp::array_t<wp::int32> var_mesh_polyvertadr,
    wp::array_t<wp::int32> var_mesh_polyvertnum,
    wp::array_t<wp::int32> var_mesh_polyvert,
    wp::array_t<wp::int32> var_mesh_polymapadr,
    wp::array_t<wp::int32> var_mesh_polymapnum,
    wp::array_t<wp::int32> var_mesh_polymap,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::int32 var_worldid,
    Geom_3242f8a8 & ret_0,
    Geom_3242f8a8 & ret_1,
    wp::array_t<wp::int32> & adj_geom_type,
    wp::array_t<wp::int32> & adj_geom_dataid,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_geom_size,
    wp::array_t<wp::int32> & adj_mesh_vertadr,
    wp::array_t<wp::int32> & adj_mesh_vertnum,
    wp::array_t<wp::int32> & adj_mesh_graphadr,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_mesh_vert,
    wp::array_t<wp::int32> & adj_mesh_graph,
    wp::array_t<wp::int32> & adj_mesh_polynum,
    wp::array_t<wp::int32> & adj_mesh_polyadr,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_mesh_polynormal,
    wp::array_t<wp::int32> & adj_mesh_polyvertadr,
    wp::array_t<wp::int32> & adj_mesh_polyvertnum,
    wp::array_t<wp::int32> & adj_mesh_polyvert,
    wp::array_t<wp::int32> & adj_mesh_polymapadr,
    wp::array_t<wp::int32> & adj_mesh_polymapnum,
    wp::array_t<wp::int32> & adj_mesh_polymap,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_geom_xmat_in,
    wp::vec_t<2, wp::int32> & adj_geoms,
    wp::int32 & adj_worldid,
    Geom_3242f8a8 & adj_ret_0,
    Geom_3242f8a8 & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_primitive_core.py:47
static CUDA_CALLABLE void adj_plane_sphere_0(
    wp::vec_t<3, wp::float32> var_plane_normal,
    wp::vec_t<3, wp::float32> var_plane_pos,
    wp::vec_t<3, wp::float32> var_sphere_pos,
    wp::float32 var_sphere_radius,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & adj_plane_normal,
    wp::vec_t<3, wp::float32> & adj_plane_pos,
    wp::vec_t<3, wp::float32> & adj_sphere_pos,
    wp::float32 & adj_sphere_radius,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/math.py:202
static CUDA_CALLABLE void adj_orthogonals_0(
    wp::vec_t<3, wp::float32> var_a,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & adj_a,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/math.py:246
static CUDA_CALLABLE void adj_make_frame_0(
    wp::vec_t<3, wp::float32> var_a,
    wp::vec_t<3, wp::float32> & adj_a,
    wp::mat_t<3, 3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_core.py:213
static CUDA_CALLABLE void adj_write_contact_0(
    wp::int32 var_naconmax_in,
    wp::int32 var_id_,
    wp::float32 var_dist_in,
    wp::vec_t<3, wp::float32> var_pos_in,
    wp::mat_t<3, 3, wp::float32> var_frame_in,
    wp::float32 var_margin_in,
    wp::float32 var_gap_in,
    wp::int32 var_condim_in,
    wp::vec_t<5, wp::float32> var_friction_in,
    wp::vec_t<2, wp::float32> var_solref_in,
    wp::vec_t<2, wp::float32> var_solreffriction_in,
    wp::vec_t<5, wp::float32> var_solimp_in,
    wp::vec_t<2, wp::int32> var_geoms_in,
    wp::vec_t<2, wp::int32> var_pairid_in,
    wp::int32 var_worldid_in,
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
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::int32> var_nacon_out,
    wp::int32 & adj_naconmax_in,
    wp::int32 & adj_id_,
    wp::float32 & adj_dist_in,
    wp::vec_t<3, wp::float32> & adj_pos_in,
    wp::mat_t<3, 3, wp::float32> & adj_frame_in,
    wp::float32 & adj_margin_in,
    wp::float32 & adj_gap_in,
    wp::int32 & adj_condim_in,
    wp::vec_t<5, wp::float32> & adj_friction_in,
    wp::vec_t<2, wp::float32> & adj_solref_in,
    wp::vec_t<2, wp::float32> & adj_solreffriction_in,
    wp::vec_t<5, wp::float32> & adj_solimp_in,
    wp::vec_t<2, wp::int32> & adj_geoms_in,
    wp::vec_t<2, wp::int32> & adj_pairid_in,
    wp::int32 & adj_worldid_in,
    wp::array_t<wp::float32> & adj_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_contact_frame_out,
    wp::array_t<wp::float32> & adj_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_solimp_out,
    wp::array_t<wp::int32> & adj_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> & adj_contact_geom_out,
    wp::array_t<wp::int32> & adj_contact_efc_address_out,
    wp::array_t<wp::int32> & adj_contact_worldid_out,
    wp::array_t<wp::int32> & adj_contact_type_out,
    wp::array_t<wp::int32> & adj_contact_geomcollisionid_out,
    wp::array_t<wp::int32> & adj_nacon_out,
    wp::int32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_primitive.py:280
static CUDA_CALLABLE void adj_plane_sphere_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_3242f8a8 var_plane,
    Geom_3242f8a8 var_sphere,
    wp::int32 var_worldid,
    wp::float32 var_margin,
    wp::float32 var_gap,
    wp::int32 var_condim,
    wp::vec_t<5, wp::float32> var_friction,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<2, wp::float32> var_solreffriction,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::vec_t<2, wp::int32> var_pairid,
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
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::int32> var_nacon_out,
    wp::int32 & adj_naconmax_in,
    Geom_3242f8a8 & adj_plane,
    Geom_3242f8a8 & adj_sphere,
    wp::int32 & adj_worldid,
    wp::float32 & adj_margin,
    wp::float32 & adj_gap,
    wp::int32 & adj_condim,
    wp::vec_t<5, wp::float32> & adj_friction,
    wp::vec_t<2, wp::float32> & adj_solref,
    wp::vec_t<2, wp::float32> & adj_solreffriction,
    wp::vec_t<5, wp::float32> & adj_solimp,
    wp::vec_t<2, wp::int32> & adj_geoms,
    wp::vec_t<2, wp::int32> & adj_pairid,
    wp::array_t<wp::float32> & adj_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_contact_frame_out,
    wp::array_t<wp::float32> & adj_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_solimp_out,
    wp::array_t<wp::int32> & adj_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> & adj_contact_geom_out,
    wp::array_t<wp::int32> & adj_contact_efc_address_out,
    wp::array_t<wp::int32> & adj_contact_worldid_out,
    wp::array_t<wp::int32> & adj_contact_type_out,
    wp::array_t<wp::int32> & adj_contact_geomcollisionid_out,
    wp::array_t<wp::int32> & adj_nacon_out)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/math.py:0
static CUDA_CALLABLE void adj_normalize_with_norm_0(
    wp::vec_t<3, wp::float32> var_x,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::float32 & ret_1,
    wp::vec_t<3, wp::float32> & adj_x,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::float32 & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_primitive_core.py:252
static CUDA_CALLABLE void adj_plane_capsule_0(
    wp::vec_t<3, wp::float32> var_plane_normal,
    wp::vec_t<3, wp::float32> var_plane_pos,
    wp::vec_t<3, wp::float32> var_capsule_pos,
    wp::vec_t<3, wp::float32> var_capsule_axis,
    wp::float32 var_capsule_radius,
    wp::float32 var_capsule_half_length,
    wp::vec_t<2, wp::float32> & ret_0,
    wp::mat_t<2, 3, wp::float32> & ret_1,
    wp::mat_t<3, 3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & adj_plane_normal,
    wp::vec_t<3, wp::float32> & adj_plane_pos,
    wp::vec_t<3, wp::float32> & adj_capsule_pos,
    wp::vec_t<3, wp::float32> & adj_capsule_axis,
    wp::float32 & adj_capsule_radius,
    wp::float32 & adj_capsule_half_length,
    wp::vec_t<2, wp::float32> & adj_ret_0,
    wp::mat_t<2, 3, wp::float32> & adj_ret_1,
    wp::mat_t<3, 3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_primitive.py:583
static CUDA_CALLABLE void adj_plane_capsule_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_3242f8a8 var_plane,
    Geom_3242f8a8 var_cap,
    wp::int32 var_worldid,
    wp::float32 var_margin,
    wp::float32 var_gap,
    wp::int32 var_condim,
    wp::vec_t<5, wp::float32> var_friction,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<2, wp::float32> var_solreffriction,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::vec_t<2, wp::int32> var_pairid,
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
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::int32> var_nacon_out,
    wp::int32 & adj_naconmax_in,
    Geom_3242f8a8 & adj_plane,
    Geom_3242f8a8 & adj_cap,
    wp::int32 & adj_worldid,
    wp::float32 & adj_margin,
    wp::float32 & adj_gap,
    wp::int32 & adj_condim,
    wp::vec_t<5, wp::float32> & adj_friction,
    wp::vec_t<2, wp::float32> & adj_solref,
    wp::vec_t<2, wp::float32> & adj_solreffriction,
    wp::vec_t<5, wp::float32> & adj_solimp,
    wp::vec_t<2, wp::int32> & adj_geoms,
    wp::vec_t<2, wp::int32> & adj_pairid,
    wp::array_t<wp::float32> & adj_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_contact_frame_out,
    wp::array_t<wp::float32> & adj_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_solimp_out,
    wp::array_t<wp::int32> & adj_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> & adj_contact_geom_out,
    wp::array_t<wp::int32> & adj_contact_efc_address_out,
    wp::array_t<wp::int32> & adj_contact_worldid_out,
    wp::array_t<wp::int32> & adj_contact_type_out,
    wp::array_t<wp::int32> & adj_contact_geomcollisionid_out,
    wp::array_t<wp::int32> & adj_nacon_out)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_primitive_core.py:55
static CUDA_CALLABLE void adj_sphere_sphere_0(
    wp::vec_t<3, wp::float32> var_pos1,
    wp::float32 var_radius1,
    wp::vec_t<3, wp::float32> var_pos2,
    wp::float32 var_radius2,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & adj_pos1,
    wp::float32 & adj_radius1,
    wp::vec_t<3, wp::float32> & adj_pos2,
    wp::float32 & adj_radius2,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_primitive.py:352
static CUDA_CALLABLE void adj_sphere_sphere_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_3242f8a8 var_sphere1,
    Geom_3242f8a8 var_sphere2,
    wp::int32 var_worldid,
    wp::float32 var_margin,
    wp::float32 var_gap,
    wp::int32 var_condim,
    wp::vec_t<5, wp::float32> var_friction,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<2, wp::float32> var_solreffriction,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::vec_t<2, wp::int32> var_pairid,
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
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::int32> var_nacon_out,
    wp::int32 & adj_naconmax_in,
    Geom_3242f8a8 & adj_sphere1,
    Geom_3242f8a8 & adj_sphere2,
    wp::int32 & adj_worldid,
    wp::float32 & adj_margin,
    wp::float32 & adj_gap,
    wp::int32 & adj_condim,
    wp::vec_t<5, wp::float32> & adj_friction,
    wp::vec_t<2, wp::float32> & adj_solref,
    wp::vec_t<2, wp::float32> & adj_solreffriction,
    wp::vec_t<5, wp::float32> & adj_solimp,
    wp::vec_t<2, wp::int32> & adj_geoms,
    wp::vec_t<2, wp::int32> & adj_pairid,
    wp::array_t<wp::float32> & adj_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_contact_frame_out,
    wp::array_t<wp::float32> & adj_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_solimp_out,
    wp::array_t<wp::int32> & adj_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> & adj_contact_geom_out,
    wp::array_t<wp::int32> & adj_contact_efc_address_out,
    wp::array_t<wp::int32> & adj_contact_worldid_out,
    wp::array_t<wp::int32> & adj_contact_type_out,
    wp::array_t<wp::int32> & adj_contact_geomcollisionid_out,
    wp::array_t<wp::int32> & adj_nacon_out)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/math.py:268
static CUDA_CALLABLE void adj_closest_segment_point_0(
    wp::vec_t<3, wp::float32> var_a,
    wp::vec_t<3, wp::float32> var_b,
    wp::vec_t<3, wp::float32> var_pt,
    wp::vec_t<3, wp::float32> & adj_a,
    wp::vec_t<3, wp::float32> & adj_b,
    wp::vec_t<3, wp::float32> & adj_pt,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_primitive_core.py:87
static CUDA_CALLABLE void adj_sphere_capsule_0(
    wp::vec_t<3, wp::float32> var_sphere_pos,
    wp::float32 var_sphere_radius,
    wp::vec_t<3, wp::float32> var_capsule_pos,
    wp::vec_t<3, wp::float32> var_capsule_axis,
    wp::float32 var_capsule_radius,
    wp::float32 var_capsule_half_length,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & adj_sphere_pos,
    wp::float32 & adj_sphere_radius,
    wp::vec_t<3, wp::float32> & adj_capsule_pos,
    wp::vec_t<3, wp::float32> & adj_capsule_axis,
    wp::float32 & adj_capsule_radius,
    wp::float32 & adj_capsule_half_length,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_primitive.py:423
static CUDA_CALLABLE void adj_sphere_capsule_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_3242f8a8 var_sphere,
    Geom_3242f8a8 var_cap,
    wp::int32 var_worldid,
    wp::float32 var_margin,
    wp::float32 var_gap,
    wp::int32 var_condim,
    wp::vec_t<5, wp::float32> var_friction,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<2, wp::float32> var_solreffriction,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::vec_t<2, wp::int32> var_pairid,
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
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::int32> var_nacon_out,
    wp::int32 & adj_naconmax_in,
    Geom_3242f8a8 & adj_sphere,
    Geom_3242f8a8 & adj_cap,
    wp::int32 & adj_worldid,
    wp::float32 & adj_margin,
    wp::float32 & adj_gap,
    wp::int32 & adj_condim,
    wp::vec_t<5, wp::float32> & adj_friction,
    wp::vec_t<2, wp::float32> & adj_solref,
    wp::vec_t<2, wp::float32> & adj_solreffriction,
    wp::vec_t<5, wp::float32> & adj_solimp,
    wp::vec_t<2, wp::int32> & adj_geoms,
    wp::vec_t<2, wp::int32> & adj_pairid,
    wp::array_t<wp::float32> & adj_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_contact_frame_out,
    wp::array_t<wp::float32> & adj_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_solimp_out,
    wp::array_t<wp::int32> & adj_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> & adj_contact_geom_out,
    wp::array_t<wp::int32> & adj_contact_efc_address_out,
    wp::array_t<wp::int32> & adj_contact_worldid_out,
    wp::array_t<wp::int32> & adj_contact_type_out,
    wp::array_t<wp::int32> & adj_contact_geomcollisionid_out,
    wp::array_t<wp::int32> & adj_nacon_out)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_primitive_core.py:1043
static CUDA_CALLABLE void adj_sphere_box_0(
    wp::vec_t<3, wp::float32> var_sphere_pos,
    wp::float32 var_sphere_radius,
    wp::vec_t<3, wp::float32> var_box_pos,
    wp::mat_t<3, 3, wp::float32> var_box_rot,
    wp::vec_t<3, wp::float32> var_box_size,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & adj_sphere_pos,
    wp::float32 & adj_sphere_radius,
    wp::vec_t<3, wp::float32> & adj_box_pos,
    wp::mat_t<3, 3, wp::float32> & adj_box_rot,
    wp::vec_t<3, wp::float32> & adj_box_size,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_primitive.py:1046
static CUDA_CALLABLE void adj_sphere_box_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_3242f8a8 var_sphere,
    Geom_3242f8a8 var_box,
    wp::int32 var_worldid,
    wp::float32 var_margin,
    wp::float32 var_gap,
    wp::int32 var_condim,
    wp::vec_t<5, wp::float32> var_friction,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<2, wp::float32> var_solreffriction,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::vec_t<2, wp::int32> var_pairid,
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
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::int32> var_nacon_out,
    wp::int32 & adj_naconmax_in,
    Geom_3242f8a8 & adj_sphere,
    Geom_3242f8a8 & adj_box,
    wp::int32 & adj_worldid,
    wp::float32 & adj_margin,
    wp::float32 & adj_gap,
    wp::int32 & adj_condim,
    wp::vec_t<5, wp::float32> & adj_friction,
    wp::vec_t<2, wp::float32> & adj_solref,
    wp::vec_t<2, wp::float32> & adj_solreffriction,
    wp::vec_t<5, wp::float32> & adj_solimp,
    wp::vec_t<2, wp::int32> & adj_geoms,
    wp::vec_t<2, wp::int32> & adj_pairid,
    wp::array_t<wp::float32> & adj_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_contact_frame_out,
    wp::array_t<wp::float32> & adj_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_solimp_out,
    wp::array_t<wp::int32> & adj_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> & adj_contact_geom_out,
    wp::array_t<wp::int32> & adj_contact_efc_address_out,
    wp::array_t<wp::int32> & adj_contact_worldid_out,
    wp::array_t<wp::int32> & adj_contact_type_out,
    wp::array_t<wp::int32> & adj_contact_geomcollisionid_out,
    wp::array_t<wp::int32> & adj_nacon_out)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_primitive_core.py:122
static CUDA_CALLABLE void adj_capsule_capsule_0(
    wp::vec_t<3, wp::float32> var_cap1_pos,
    wp::vec_t<3, wp::float32> var_cap1_axis,
    wp::float32 var_cap1_radius,
    wp::float32 var_cap1_half_length,
    wp::vec_t<3, wp::float32> var_cap2_pos,
    wp::vec_t<3, wp::float32> var_cap2_axis,
    wp::float32 var_cap2_radius,
    wp::float32 var_cap2_half_length,
    wp::float32 var_margin,
    wp::vec_t<2, wp::float32> & ret_0,
    wp::mat_t<2, 3, wp::float32> & ret_1,
    wp::mat_t<2, 3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & adj_cap1_pos,
    wp::vec_t<3, wp::float32> & adj_cap1_axis,
    wp::float32 & adj_cap1_radius,
    wp::float32 & adj_cap1_half_length,
    wp::vec_t<3, wp::float32> & adj_cap2_pos,
    wp::vec_t<3, wp::float32> & adj_cap2_axis,
    wp::float32 & adj_cap2_radius,
    wp::float32 & adj_cap2_half_length,
    wp::float32 & adj_margin,
    wp::vec_t<2, wp::float32> & adj_ret_0,
    wp::mat_t<2, 3, wp::float32> & adj_ret_1,
    wp::mat_t<2, 3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_primitive.py:497
static CUDA_CALLABLE void adj_capsule_capsule_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_3242f8a8 var_cap1,
    Geom_3242f8a8 var_cap2,
    wp::int32 var_worldid,
    wp::float32 var_margin,
    wp::float32 var_gap,
    wp::int32 var_condim,
    wp::vec_t<5, wp::float32> var_friction,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<2, wp::float32> var_solreffriction,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::vec_t<2, wp::int32> var_pairid,
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
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::int32> var_nacon_out,
    wp::int32 & adj_naconmax_in,
    Geom_3242f8a8 & adj_cap1,
    Geom_3242f8a8 & adj_cap2,
    wp::int32 & adj_worldid,
    wp::float32 & adj_margin,
    wp::float32 & adj_gap,
    wp::int32 & adj_condim,
    wp::vec_t<5, wp::float32> & adj_friction,
    wp::vec_t<2, wp::float32> & adj_solref,
    wp::vec_t<2, wp::float32> & adj_solreffriction,
    wp::vec_t<5, wp::float32> & adj_solimp,
    wp::vec_t<2, wp::int32> & adj_geoms,
    wp::vec_t<2, wp::int32> & adj_pairid,
    wp::array_t<wp::float32> & adj_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_contact_frame_out,
    wp::array_t<wp::float32> & adj_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_solimp_out,
    wp::array_t<wp::int32> & adj_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> & adj_contact_geom_out,
    wp::array_t<wp::int32> & adj_contact_efc_address_out,
    wp::array_t<wp::int32> & adj_contact_worldid_out,
    wp::array_t<wp::int32> & adj_contact_type_out,
    wp::array_t<wp::int32> & adj_contact_geomcollisionid_out,
    wp::array_t<wp::int32> & adj_nacon_out)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_primitive_core.py:1098
static CUDA_CALLABLE void adj_capsule_box_0(
    wp::vec_t<3, wp::float32> var_capsule_pos,
    wp::vec_t<3, wp::float32> var_capsule_axis,
    wp::float32 var_capsule_radius,
    wp::float32 var_capsule_half_length,
    wp::vec_t<3, wp::float32> var_box_pos,
    wp::mat_t<3, 3, wp::float32> var_box_rot,
    wp::vec_t<3, wp::float32> var_box_size,
    wp::vec_t<2, wp::float32> & ret_0,
    wp::mat_t<2, 3, wp::float32> & ret_1,
    wp::mat_t<2, 3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & adj_capsule_pos,
    wp::vec_t<3, wp::float32> & adj_capsule_axis,
    wp::float32 & adj_capsule_radius,
    wp::float32 & adj_capsule_half_length,
    wp::vec_t<3, wp::float32> & adj_box_pos,
    wp::mat_t<3, 3, wp::float32> & adj_box_rot,
    wp::vec_t<3, wp::float32> & adj_box_size,
    wp::vec_t<2, wp::float32> & adj_ret_0,
    wp::mat_t<2, 3, wp::float32> & adj_ret_1,
    wp::mat_t<2, 3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_primitive.py:1116
static CUDA_CALLABLE void adj_capsule_box_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_3242f8a8 var_cap,
    Geom_3242f8a8 var_box,
    wp::int32 var_worldid,
    wp::float32 var_margin,
    wp::float32 var_gap,
    wp::int32 var_condim,
    wp::vec_t<5, wp::float32> var_friction,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<2, wp::float32> var_solreffriction,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::vec_t<2, wp::int32> var_pairid,
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
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::int32> var_nacon_out,
    wp::int32 & adj_naconmax_in,
    Geom_3242f8a8 & adj_cap,
    Geom_3242f8a8 & adj_box,
    wp::int32 & adj_worldid,
    wp::float32 & adj_margin,
    wp::float32 & adj_gap,
    wp::int32 & adj_condim,
    wp::vec_t<5, wp::float32> & adj_friction,
    wp::vec_t<2, wp::float32> & adj_solref,
    wp::vec_t<2, wp::float32> & adj_solreffriction,
    wp::vec_t<5, wp::float32> & adj_solimp,
    wp::vec_t<2, wp::int32> & adj_geoms,
    wp::vec_t<2, wp::int32> & adj_pairid,
    wp::array_t<wp::float32> & adj_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_contact_frame_out,
    wp::array_t<wp::float32> & adj_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_solimp_out,
    wp::array_t<wp::int32> & adj_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> & adj_contact_geom_out,
    wp::array_t<wp::int32> & adj_contact_efc_address_out,
    wp::array_t<wp::int32> & adj_contact_worldid_out,
    wp::array_t<wp::int32> & adj_contact_type_out,
    wp::array_t<wp::int32> & adj_contact_geomcollisionid_out,
    wp::array_t<wp::int32> & adj_nacon_out)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void _primitive_narrowphase__locals__primitive_narrowphase_198a21b9_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::int32> var_geom_type,
    wp::array_t<wp::int32> var_geom_condim,
    wp::array_t<wp::int32> var_geom_dataid,
    wp::array_t<wp::int32> var_geom_priority,
    wp::array_t<wp::float32> var_geom_solmix,
    wp::array_t<wp::vec_t<2, wp::float32>> var_geom_solref,
    wp::array_t<wp::vec_t<5, wp::float32>> var_geom_solimp,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_size,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_friction,
    wp::array_t<wp::float32> var_geom_margin,
    wp::array_t<wp::float32> var_geom_gap,
    wp::array_t<wp::int32> var_mesh_vertadr,
    wp::array_t<wp::int32> var_mesh_vertnum,
    wp::array_t<wp::int32> var_mesh_graphadr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_mesh_vert,
    wp::array_t<wp::int32> var_mesh_graph,
    wp::array_t<wp::int32> var_mesh_polynum,
    wp::array_t<wp::int32> var_mesh_polyadr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_mesh_polynormal,
    wp::array_t<wp::int32> var_mesh_polyvertadr,
    wp::array_t<wp::int32> var_mesh_polyvertnum,
    wp::array_t<wp::int32> var_mesh_polyvert,
    wp::array_t<wp::int32> var_mesh_polymapadr,
    wp::array_t<wp::int32> var_mesh_polymapnum,
    wp::array_t<wp::int32> var_mesh_polymap,
    wp::array_t<wp::int32> var_pair_dim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_pair_solref,
    wp::array_t<wp::vec_t<2, wp::float32>> var_pair_solreffriction,
    wp::array_t<wp::vec_t<5, wp::float32>> var_pair_solimp,
    wp::array_t<wp::float32> var_pair_margin,
    wp::array_t<wp::float32> var_pair_gap,
    wp::array_t<wp::vec_t<5, wp::float32>> var_pair_friction,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::int32 var_naconmax_in,
    wp::array_t<wp::int32> var_ncollision_in,
    wp::array_t<wp::vec_t<2, wp::int32>> var_collision_pair_in,
    wp::array_t<wp::vec_t<2, wp::int32>> var_collision_pairid_in,
    wp::array_t<wp::int32> var_collision_worldid_in,
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
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::int32> var_nacon_out)
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
        wp::vec_t<2, wp::int32>* var_5;
        wp::vec_t<2, wp::int32> var_6;
        wp::vec_t<2, wp::int32> var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::vec_t<2, wp::int32> var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::int32 var_14;
        wp::vec_t<5, wp::float32> var_15;
        wp::vec_t<2, wp::float32> var_16;
        wp::vec_t<2, wp::float32> var_17;
        wp::vec_t<5, wp::float32> var_18;
        Geom_3242f8a8 var_19;
        Geom_3242f8a8 var_20;
        const wp::int32 var_21 = 0;
        const wp::int32 var_22 = 0;
        const wp::int32 var_23 = 2;
        const wp::int32 var_24 = 0;
        wp::int32 var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        const wp::int32 var_29 = 1;
        wp::int32 var_30;
        wp::int32* var_31;
        wp::int32 var_32;
        wp::int32 var_33;
        bool var_34;
        bool var_35;
        bool var_36;
        wp::vec_t<2, wp::int32>* var_37;
        wp::vec_t<2, wp::int32> var_38;
        const wp::int32 var_39 = 1;
        const wp::int32 var_40 = 0;
        const wp::int32 var_41 = 3;
        const wp::int32 var_42 = 0;
        wp::int32 var_43;
        wp::int32* var_44;
        wp::int32 var_45;
        wp::int32 var_46;
        const wp::int32 var_47 = 1;
        wp::int32 var_48;
        wp::int32* var_49;
        wp::int32 var_50;
        wp::int32 var_51;
        bool var_52;
        bool var_53;
        bool var_54;
        wp::vec_t<2, wp::int32>* var_55;
        wp::vec_t<2, wp::int32> var_56;
        const wp::int32 var_57 = 2;
        const wp::int32 var_58 = 2;
        const wp::int32 var_59 = 2;
        const wp::int32 var_60 = 0;
        wp::int32 var_61;
        wp::int32* var_62;
        wp::int32 var_63;
        wp::int32 var_64;
        const wp::int32 var_65 = 1;
        wp::int32 var_66;
        wp::int32* var_67;
        wp::int32 var_68;
        wp::int32 var_69;
        bool var_70;
        bool var_71;
        bool var_72;
        wp::vec_t<2, wp::int32>* var_73;
        wp::vec_t<2, wp::int32> var_74;
        const wp::int32 var_75 = 3;
        const wp::int32 var_76 = 2;
        const wp::int32 var_77 = 3;
        const wp::int32 var_78 = 0;
        wp::int32 var_79;
        wp::int32* var_80;
        wp::int32 var_81;
        wp::int32 var_82;
        const wp::int32 var_83 = 1;
        wp::int32 var_84;
        wp::int32* var_85;
        wp::int32 var_86;
        wp::int32 var_87;
        bool var_88;
        bool var_89;
        bool var_90;
        wp::vec_t<2, wp::int32>* var_91;
        wp::vec_t<2, wp::int32> var_92;
        const wp::int32 var_93 = 4;
        const wp::int32 var_94 = 2;
        const wp::int32 var_95 = 6;
        const wp::int32 var_96 = 0;
        wp::int32 var_97;
        wp::int32* var_98;
        wp::int32 var_99;
        wp::int32 var_100;
        const wp::int32 var_101 = 1;
        wp::int32 var_102;
        wp::int32* var_103;
        wp::int32 var_104;
        wp::int32 var_105;
        bool var_106;
        bool var_107;
        bool var_108;
        wp::vec_t<2, wp::int32>* var_109;
        wp::vec_t<2, wp::int32> var_110;
        const wp::int32 var_111 = 5;
        const wp::int32 var_112 = 3;
        const wp::int32 var_113 = 3;
        const wp::int32 var_114 = 0;
        wp::int32 var_115;
        wp::int32* var_116;
        wp::int32 var_117;
        wp::int32 var_118;
        const wp::int32 var_119 = 1;
        wp::int32 var_120;
        wp::int32* var_121;
        wp::int32 var_122;
        wp::int32 var_123;
        bool var_124;
        bool var_125;
        bool var_126;
        wp::vec_t<2, wp::int32>* var_127;
        wp::vec_t<2, wp::int32> var_128;
        const wp::int32 var_129 = 6;
        const wp::int32 var_130 = 3;
        const wp::int32 var_131 = 6;
        const wp::int32 var_132 = 0;
        wp::int32 var_133;
        wp::int32* var_134;
        wp::int32 var_135;
        wp::int32 var_136;
        const wp::int32 var_137 = 1;
        wp::int32 var_138;
        wp::int32* var_139;
        wp::int32 var_140;
        wp::int32 var_141;
        bool var_142;
        bool var_143;
        bool var_144;
        wp::vec_t<2, wp::int32>* var_145;
        wp::vec_t<2, wp::int32> var_146;
        //---------
        // forward
        // def primitive_narrowphase(                                                             <L 1303>
        // tid = wp.tid()                                                                         <L 1363>
        var_0 = builtin_tid1d();
        // if tid >= ncollision_in[0]:                                                            <L 1365>
        var_2 = wp::address(var_ncollision_in, var_1);
        var_4 = wp::load(var_2);
        var_3 = (var_0 >= var_4);
        if (var_3) {
            // return                                                                             <L 1366>
            continue;
        }
        // geoms = collision_pair_in[tid]                                                         <L 1368>
        var_5 = wp::address(var_collision_pair_in, var_0);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // worldid = collision_worldid_in[tid]                                                    <L 1369>
        var_8 = wp::address(var_collision_worldid_in, var_0);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // _, margin, gap, condim, friction, solref, solreffriction, solimp = contact_params(       <L 1371>
        // geom_condim,                                                                           <L 1372>
        // geom_priority,                                                                         <L 1373>
        // geom_solmix,                                                                           <L 1374>
        // geom_solref,                                                                           <L 1375>
        // geom_solimp,                                                                           <L 1376>
        // geom_friction,                                                                         <L 1377>
        // geom_margin,                                                                           <L 1378>
        // geom_gap,                                                                              <L 1379>
        // pair_dim,                                                                              <L 1380>
        // pair_solref,                                                                           <L 1381>
        // pair_solreffriction,                                                                   <L 1382>
        // pair_solimp,                                                                           <L 1383>
        // pair_margin,                                                                           <L 1384>
        // pair_gap,                                                                              <L 1385>
        // pair_friction,                                                                         <L 1386>
        // collision_pair_in,                                                                     <L 1387>
        // collision_pairid_in,                                                                   <L 1388>
        // tid,                                                                                   <L 1389>
        // worldid,                                                                               <L 1390>
        contact_params_0(var_geom_condim, var_geom_priority, var_geom_solmix, var_geom_solref, var_geom_solimp, var_geom_friction, var_geom_margin, var_geom_gap, var_pair_dim, var_pair_solref, var_pair_solreffriction, var_pair_solimp, var_pair_margin, var_pair_gap, var_pair_friction, var_collision_pair_in, var_collision_pairid_in, var_0, var_9, var_11, var_12, var_13, var_14, var_15, var_16, var_17, var_18);
        // geom1, geom2 = geom_collision_pair(                                                    <L 1393>
        // geom_type,                                                                             <L 1394>
        // geom_dataid,                                                                           <L 1395>
        // geom_size,                                                                             <L 1396>
        // mesh_vertadr,                                                                          <L 1397>
        // mesh_vertnum,                                                                          <L 1398>
        // mesh_graphadr,                                                                         <L 1399>
        // mesh_vert,                                                                             <L 1400>
        // mesh_graph,                                                                            <L 1401>
        // mesh_polynum,                                                                          <L 1402>
        // mesh_polyadr,                                                                          <L 1403>
        // mesh_polynormal,                                                                       <L 1404>
        // mesh_polyvertadr,                                                                      <L 1405>
        // mesh_polyvertnum,                                                                      <L 1406>
        // mesh_polyvert,                                                                         <L 1407>
        // mesh_polymapadr,                                                                       <L 1408>
        // mesh_polymapnum,                                                                       <L 1409>
        // mesh_polymap,                                                                          <L 1410>
        // geom_xpos_in,                                                                          <L 1411>
        // geom_xmat_in,                                                                          <L 1412>
        // geoms,                                                                                 <L 1413>
        // worldid,                                                                               <L 1414>
        geom_collision_pair_0(var_geom_type, var_geom_dataid, var_geom_size, var_mesh_vertadr, var_mesh_vertnum, var_mesh_graphadr, var_mesh_vert, var_mesh_graph, var_mesh_polynum, var_mesh_polyadr, var_mesh_polynormal, var_mesh_polyvertadr, var_mesh_polyvertnum, var_mesh_polyvert, var_mesh_polymapadr, var_mesh_polymapnum, var_mesh_polymap, var_geom_xpos_in, var_geom_xmat_in, var_6, var_9, var_19, var_20);
        // for i in range(wp.static(len(primitive_collisions_func))):                             <L 1417>
        // collision_type1 = wp.static(primitive_collisions_types[i][0])                          <L 1418>
        // collision_type2 = wp.static(primitive_collisions_types[i][1])                          <L 1419>
        // type1 = geom_type[geoms[0]]                                                            <L 1420>
        var_25 = wp::extract(var_6, var_24);
        var_26 = wp::address(var_geom_type, var_25);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // type2 = geom_type[geoms[1]]                                                            <L 1421>
        var_30 = wp::extract(var_6, var_29);
        var_31 = wp::address(var_geom_type, var_30);
        var_33 = wp::load(var_31);
        var_32 = wp::copy(var_33);
        // if collision_type1 == type1 and collision_type2 == type2:                              <L 1422>
        var_35 = (var_22 == var_27);
        var_34 = var_35;
        if (var_34) {
            var_36 = (var_23 == var_32);
            var_34 = var_34 && var_36;
        }
        if (var_34) {
            // wp.static(primitive_collisions_func[i])(                                           <L 1423>
            // naconmax_in,                                                                       <L 1424>
            // geom1,                                                                             <L 1425>
            // geom2,                                                                             <L 1426>
            // worldid,                                                                           <L 1427>
            // margin,                                                                            <L 1428>
            // gap,                                                                               <L 1429>
            // condim,                                                                            <L 1430>
            // friction,                                                                          <L 1431>
            // solref,                                                                            <L 1432>
            // solreffriction,                                                                    <L 1433>
            // solimp,                                                                            <L 1434>
            // geoms,                                                                             <L 1435>
            // collision_pairid_in[tid],                                                          <L 1436>
            var_37 = wp::address(var_collision_pairid_in, var_0);
            // contact_dist_out,                                                                  <L 1437>
            // contact_pos_out,                                                                   <L 1438>
            // contact_frame_out,                                                                 <L 1439>
            // contact_includemargin_out,                                                         <L 1440>
            // contact_friction_out,                                                              <L 1441>
            // contact_solref_out,                                                                <L 1442>
            // contact_solreffriction_out,                                                        <L 1443>
            // contact_solimp_out,                                                                <L 1444>
            // contact_dim_out,                                                                   <L 1445>
            // contact_geom_out,                                                                  <L 1446>
            // contact_efc_address_out,                                                           <L 1447>
            // contact_worldid_out,                                                               <L 1448>
            // contact_type_out,                                                                  <L 1449>
            // contact_geomcollisionid_out,                                                       <L 1450>
            // nacon_out,                                                                         <L 1451>
            var_38 = wp::load(var_37);
            plane_sphere_wrapper_0(var_naconmax_in, var_19, var_20, var_9, var_12, var_13, var_14, var_15, var_16, var_17, var_18, var_6, var_38, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_nacon_out);
        }
        // collision_type1 = wp.static(primitive_collisions_types[i][0])                          <L 1418>
        // collision_type2 = wp.static(primitive_collisions_types[i][1])                          <L 1419>
        // type1 = geom_type[geoms[0]]                                                            <L 1420>
        var_43 = wp::extract(var_6, var_42);
        var_44 = wp::address(var_geom_type, var_43);
        var_46 = wp::load(var_44);
        var_45 = wp::copy(var_46);
        // type2 = geom_type[geoms[1]]                                                            <L 1421>
        var_48 = wp::extract(var_6, var_47);
        var_49 = wp::address(var_geom_type, var_48);
        var_51 = wp::load(var_49);
        var_50 = wp::copy(var_51);
        // if collision_type1 == type1 and collision_type2 == type2:                              <L 1422>
        var_53 = (var_40 == var_45);
        var_52 = var_53;
        if (var_52) {
            var_54 = (var_41 == var_50);
            var_52 = var_52 && var_54;
        }
        if (var_52) {
            // wp.static(primitive_collisions_func[i])(                                           <L 1423>
            // naconmax_in,                                                                       <L 1424>
            // geom1,                                                                             <L 1425>
            // geom2,                                                                             <L 1426>
            // worldid,                                                                           <L 1427>
            // margin,                                                                            <L 1428>
            // gap,                                                                               <L 1429>
            // condim,                                                                            <L 1430>
            // friction,                                                                          <L 1431>
            // solref,                                                                            <L 1432>
            // solreffriction,                                                                    <L 1433>
            // solimp,                                                                            <L 1434>
            // geoms,                                                                             <L 1435>
            // collision_pairid_in[tid],                                                          <L 1436>
            var_55 = wp::address(var_collision_pairid_in, var_0);
            // contact_dist_out,                                                                  <L 1437>
            // contact_pos_out,                                                                   <L 1438>
            // contact_frame_out,                                                                 <L 1439>
            // contact_includemargin_out,                                                         <L 1440>
            // contact_friction_out,                                                              <L 1441>
            // contact_solref_out,                                                                <L 1442>
            // contact_solreffriction_out,                                                        <L 1443>
            // contact_solimp_out,                                                                <L 1444>
            // contact_dim_out,                                                                   <L 1445>
            // contact_geom_out,                                                                  <L 1446>
            // contact_efc_address_out,                                                           <L 1447>
            // contact_worldid_out,                                                               <L 1448>
            // contact_type_out,                                                                  <L 1449>
            // contact_geomcollisionid_out,                                                       <L 1450>
            // nacon_out,                                                                         <L 1451>
            var_56 = wp::load(var_55);
            plane_capsule_wrapper_0(var_naconmax_in, var_19, var_20, var_9, var_12, var_13, var_14, var_15, var_16, var_17, var_18, var_6, var_56, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_nacon_out);
        }
        // collision_type1 = wp.static(primitive_collisions_types[i][0])                          <L 1418>
        // collision_type2 = wp.static(primitive_collisions_types[i][1])                          <L 1419>
        // type1 = geom_type[geoms[0]]                                                            <L 1420>
        var_61 = wp::extract(var_6, var_60);
        var_62 = wp::address(var_geom_type, var_61);
        var_64 = wp::load(var_62);
        var_63 = wp::copy(var_64);
        // type2 = geom_type[geoms[1]]                                                            <L 1421>
        var_66 = wp::extract(var_6, var_65);
        var_67 = wp::address(var_geom_type, var_66);
        var_69 = wp::load(var_67);
        var_68 = wp::copy(var_69);
        // if collision_type1 == type1 and collision_type2 == type2:                              <L 1422>
        var_71 = (var_58 == var_63);
        var_70 = var_71;
        if (var_70) {
            var_72 = (var_59 == var_68);
            var_70 = var_70 && var_72;
        }
        if (var_70) {
            // wp.static(primitive_collisions_func[i])(                                           <L 1423>
            // naconmax_in,                                                                       <L 1424>
            // geom1,                                                                             <L 1425>
            // geom2,                                                                             <L 1426>
            // worldid,                                                                           <L 1427>
            // margin,                                                                            <L 1428>
            // gap,                                                                               <L 1429>
            // condim,                                                                            <L 1430>
            // friction,                                                                          <L 1431>
            // solref,                                                                            <L 1432>
            // solreffriction,                                                                    <L 1433>
            // solimp,                                                                            <L 1434>
            // geoms,                                                                             <L 1435>
            // collision_pairid_in[tid],                                                          <L 1436>
            var_73 = wp::address(var_collision_pairid_in, var_0);
            // contact_dist_out,                                                                  <L 1437>
            // contact_pos_out,                                                                   <L 1438>
            // contact_frame_out,                                                                 <L 1439>
            // contact_includemargin_out,                                                         <L 1440>
            // contact_friction_out,                                                              <L 1441>
            // contact_solref_out,                                                                <L 1442>
            // contact_solreffriction_out,                                                        <L 1443>
            // contact_solimp_out,                                                                <L 1444>
            // contact_dim_out,                                                                   <L 1445>
            // contact_geom_out,                                                                  <L 1446>
            // contact_efc_address_out,                                                           <L 1447>
            // contact_worldid_out,                                                               <L 1448>
            // contact_type_out,                                                                  <L 1449>
            // contact_geomcollisionid_out,                                                       <L 1450>
            // nacon_out,                                                                         <L 1451>
            var_74 = wp::load(var_73);
            sphere_sphere_wrapper_0(var_naconmax_in, var_19, var_20, var_9, var_12, var_13, var_14, var_15, var_16, var_17, var_18, var_6, var_74, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_nacon_out);
        }
        // collision_type1 = wp.static(primitive_collisions_types[i][0])                          <L 1418>
        // collision_type2 = wp.static(primitive_collisions_types[i][1])                          <L 1419>
        // type1 = geom_type[geoms[0]]                                                            <L 1420>
        var_79 = wp::extract(var_6, var_78);
        var_80 = wp::address(var_geom_type, var_79);
        var_82 = wp::load(var_80);
        var_81 = wp::copy(var_82);
        // type2 = geom_type[geoms[1]]                                                            <L 1421>
        var_84 = wp::extract(var_6, var_83);
        var_85 = wp::address(var_geom_type, var_84);
        var_87 = wp::load(var_85);
        var_86 = wp::copy(var_87);
        // if collision_type1 == type1 and collision_type2 == type2:                              <L 1422>
        var_89 = (var_76 == var_81);
        var_88 = var_89;
        if (var_88) {
            var_90 = (var_77 == var_86);
            var_88 = var_88 && var_90;
        }
        if (var_88) {
            // wp.static(primitive_collisions_func[i])(                                           <L 1423>
            // naconmax_in,                                                                       <L 1424>
            // geom1,                                                                             <L 1425>
            // geom2,                                                                             <L 1426>
            // worldid,                                                                           <L 1427>
            // margin,                                                                            <L 1428>
            // gap,                                                                               <L 1429>
            // condim,                                                                            <L 1430>
            // friction,                                                                          <L 1431>
            // solref,                                                                            <L 1432>
            // solreffriction,                                                                    <L 1433>
            // solimp,                                                                            <L 1434>
            // geoms,                                                                             <L 1435>
            // collision_pairid_in[tid],                                                          <L 1436>
            var_91 = wp::address(var_collision_pairid_in, var_0);
            // contact_dist_out,                                                                  <L 1437>
            // contact_pos_out,                                                                   <L 1438>
            // contact_frame_out,                                                                 <L 1439>
            // contact_includemargin_out,                                                         <L 1440>
            // contact_friction_out,                                                              <L 1441>
            // contact_solref_out,                                                                <L 1442>
            // contact_solreffriction_out,                                                        <L 1443>
            // contact_solimp_out,                                                                <L 1444>
            // contact_dim_out,                                                                   <L 1445>
            // contact_geom_out,                                                                  <L 1446>
            // contact_efc_address_out,                                                           <L 1447>
            // contact_worldid_out,                                                               <L 1448>
            // contact_type_out,                                                                  <L 1449>
            // contact_geomcollisionid_out,                                                       <L 1450>
            // nacon_out,                                                                         <L 1451>
            var_92 = wp::load(var_91);
            sphere_capsule_wrapper_0(var_naconmax_in, var_19, var_20, var_9, var_12, var_13, var_14, var_15, var_16, var_17, var_18, var_6, var_92, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_nacon_out);
        }
        // collision_type1 = wp.static(primitive_collisions_types[i][0])                          <L 1418>
        // collision_type2 = wp.static(primitive_collisions_types[i][1])                          <L 1419>
        // type1 = geom_type[geoms[0]]                                                            <L 1420>
        var_97 = wp::extract(var_6, var_96);
        var_98 = wp::address(var_geom_type, var_97);
        var_100 = wp::load(var_98);
        var_99 = wp::copy(var_100);
        // type2 = geom_type[geoms[1]]                                                            <L 1421>
        var_102 = wp::extract(var_6, var_101);
        var_103 = wp::address(var_geom_type, var_102);
        var_105 = wp::load(var_103);
        var_104 = wp::copy(var_105);
        // if collision_type1 == type1 and collision_type2 == type2:                              <L 1422>
        var_107 = (var_94 == var_99);
        var_106 = var_107;
        if (var_106) {
            var_108 = (var_95 == var_104);
            var_106 = var_106 && var_108;
        }
        if (var_106) {
            // wp.static(primitive_collisions_func[i])(                                           <L 1423>
            // naconmax_in,                                                                       <L 1424>
            // geom1,                                                                             <L 1425>
            // geom2,                                                                             <L 1426>
            // worldid,                                                                           <L 1427>
            // margin,                                                                            <L 1428>
            // gap,                                                                               <L 1429>
            // condim,                                                                            <L 1430>
            // friction,                                                                          <L 1431>
            // solref,                                                                            <L 1432>
            // solreffriction,                                                                    <L 1433>
            // solimp,                                                                            <L 1434>
            // geoms,                                                                             <L 1435>
            // collision_pairid_in[tid],                                                          <L 1436>
            var_109 = wp::address(var_collision_pairid_in, var_0);
            // contact_dist_out,                                                                  <L 1437>
            // contact_pos_out,                                                                   <L 1438>
            // contact_frame_out,                                                                 <L 1439>
            // contact_includemargin_out,                                                         <L 1440>
            // contact_friction_out,                                                              <L 1441>
            // contact_solref_out,                                                                <L 1442>
            // contact_solreffriction_out,                                                        <L 1443>
            // contact_solimp_out,                                                                <L 1444>
            // contact_dim_out,                                                                   <L 1445>
            // contact_geom_out,                                                                  <L 1446>
            // contact_efc_address_out,                                                           <L 1447>
            // contact_worldid_out,                                                               <L 1448>
            // contact_type_out,                                                                  <L 1449>
            // contact_geomcollisionid_out,                                                       <L 1450>
            // nacon_out,                                                                         <L 1451>
            var_110 = wp::load(var_109);
            sphere_box_wrapper_0(var_naconmax_in, var_19, var_20, var_9, var_12, var_13, var_14, var_15, var_16, var_17, var_18, var_6, var_110, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_nacon_out);
        }
        // collision_type1 = wp.static(primitive_collisions_types[i][0])                          <L 1418>
        // collision_type2 = wp.static(primitive_collisions_types[i][1])                          <L 1419>
        // type1 = geom_type[geoms[0]]                                                            <L 1420>
        var_115 = wp::extract(var_6, var_114);
        var_116 = wp::address(var_geom_type, var_115);
        var_118 = wp::load(var_116);
        var_117 = wp::copy(var_118);
        // type2 = geom_type[geoms[1]]                                                            <L 1421>
        var_120 = wp::extract(var_6, var_119);
        var_121 = wp::address(var_geom_type, var_120);
        var_123 = wp::load(var_121);
        var_122 = wp::copy(var_123);
        // if collision_type1 == type1 and collision_type2 == type2:                              <L 1422>
        var_125 = (var_112 == var_117);
        var_124 = var_125;
        if (var_124) {
            var_126 = (var_113 == var_122);
            var_124 = var_124 && var_126;
        }
        if (var_124) {
            // wp.static(primitive_collisions_func[i])(                                           <L 1423>
            // naconmax_in,                                                                       <L 1424>
            // geom1,                                                                             <L 1425>
            // geom2,                                                                             <L 1426>
            // worldid,                                                                           <L 1427>
            // margin,                                                                            <L 1428>
            // gap,                                                                               <L 1429>
            // condim,                                                                            <L 1430>
            // friction,                                                                          <L 1431>
            // solref,                                                                            <L 1432>
            // solreffriction,                                                                    <L 1433>
            // solimp,                                                                            <L 1434>
            // geoms,                                                                             <L 1435>
            // collision_pairid_in[tid],                                                          <L 1436>
            var_127 = wp::address(var_collision_pairid_in, var_0);
            // contact_dist_out,                                                                  <L 1437>
            // contact_pos_out,                                                                   <L 1438>
            // contact_frame_out,                                                                 <L 1439>
            // contact_includemargin_out,                                                         <L 1440>
            // contact_friction_out,                                                              <L 1441>
            // contact_solref_out,                                                                <L 1442>
            // contact_solreffriction_out,                                                        <L 1443>
            // contact_solimp_out,                                                                <L 1444>
            // contact_dim_out,                                                                   <L 1445>
            // contact_geom_out,                                                                  <L 1446>
            // contact_efc_address_out,                                                           <L 1447>
            // contact_worldid_out,                                                               <L 1448>
            // contact_type_out,                                                                  <L 1449>
            // contact_geomcollisionid_out,                                                       <L 1450>
            // nacon_out,                                                                         <L 1451>
            var_128 = wp::load(var_127);
            capsule_capsule_wrapper_0(var_naconmax_in, var_19, var_20, var_9, var_12, var_13, var_14, var_15, var_16, var_17, var_18, var_6, var_128, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_nacon_out);
        }
        // collision_type1 = wp.static(primitive_collisions_types[i][0])                          <L 1418>
        // collision_type2 = wp.static(primitive_collisions_types[i][1])                          <L 1419>
        // type1 = geom_type[geoms[0]]                                                            <L 1420>
        var_133 = wp::extract(var_6, var_132);
        var_134 = wp::address(var_geom_type, var_133);
        var_136 = wp::load(var_134);
        var_135 = wp::copy(var_136);
        // type2 = geom_type[geoms[1]]                                                            <L 1421>
        var_138 = wp::extract(var_6, var_137);
        var_139 = wp::address(var_geom_type, var_138);
        var_141 = wp::load(var_139);
        var_140 = wp::copy(var_141);
        // if collision_type1 == type1 and collision_type2 == type2:                              <L 1422>
        var_143 = (var_130 == var_135);
        var_142 = var_143;
        if (var_142) {
            var_144 = (var_131 == var_140);
            var_142 = var_142 && var_144;
        }
        if (var_142) {
            // wp.static(primitive_collisions_func[i])(                                           <L 1423>
            // naconmax_in,                                                                       <L 1424>
            // geom1,                                                                             <L 1425>
            // geom2,                                                                             <L 1426>
            // worldid,                                                                           <L 1427>
            // margin,                                                                            <L 1428>
            // gap,                                                                               <L 1429>
            // condim,                                                                            <L 1430>
            // friction,                                                                          <L 1431>
            // solref,                                                                            <L 1432>
            // solreffriction,                                                                    <L 1433>
            // solimp,                                                                            <L 1434>
            // geoms,                                                                             <L 1435>
            // collision_pairid_in[tid],                                                          <L 1436>
            var_145 = wp::address(var_collision_pairid_in, var_0);
            // contact_dist_out,                                                                  <L 1437>
            // contact_pos_out,                                                                   <L 1438>
            // contact_frame_out,                                                                 <L 1439>
            // contact_includemargin_out,                                                         <L 1440>
            // contact_friction_out,                                                              <L 1441>
            // contact_solref_out,                                                                <L 1442>
            // contact_solreffriction_out,                                                        <L 1443>
            // contact_solimp_out,                                                                <L 1444>
            // contact_dim_out,                                                                   <L 1445>
            // contact_geom_out,                                                                  <L 1446>
            // contact_efc_address_out,                                                           <L 1447>
            // contact_worldid_out,                                                               <L 1448>
            // contact_type_out,                                                                  <L 1449>
            // contact_geomcollisionid_out,                                                       <L 1450>
            // nacon_out,                                                                         <L 1451>
            var_146 = wp::load(var_145);
            capsule_box_wrapper_0(var_naconmax_in, var_19, var_20, var_9, var_12, var_13, var_14, var_15, var_16, var_17, var_18, var_6, var_146, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_nacon_out);
        }
    }
}

