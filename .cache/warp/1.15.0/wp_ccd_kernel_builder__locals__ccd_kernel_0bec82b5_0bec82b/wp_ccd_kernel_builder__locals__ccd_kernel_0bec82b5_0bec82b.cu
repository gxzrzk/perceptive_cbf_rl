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




struct GJKResult_28609055
{
    wp::float32 dist;
    wp::vec_t<3, wp::float32> x1;
    wp::vec_t<3, wp::float32> x2;
    wp::int32 dim;
    wp::mat_t<4, 3, wp::float32> simplex;
    wp::mat_t<4, 3, wp::float32> simplex1;
    wp::mat_t<4, 3, wp::float32> simplex2;
    wp::vec_t<4, wp::int32> simplex_index1;
    wp::vec_t<4, wp::int32> simplex_index2;
    wp::int32 index1;
    wp::int32 index2;


    GJKResult_28609055() = default;
    CUDA_CALLABLE GJKResult_28609055(wp::float32 const& dist,
    wp::vec_t<3, wp::float32> const& x1 = {},
    wp::vec_t<3, wp::float32> const& x2 = {},
    wp::int32 const& dim = {},
    wp::mat_t<4, 3, wp::float32> const& simplex = {},
    wp::mat_t<4, 3, wp::float32> const& simplex1 = {},
    wp::mat_t<4, 3, wp::float32> const& simplex2 = {},
    wp::vec_t<4, wp::int32> const& simplex_index1 = {},
    wp::vec_t<4, wp::int32> const& simplex_index2 = {},
    wp::int32 const& index1 = {},
    wp::int32 const& index2 = {})
        : dist{dist}
        , x1{x1}
        , x2{x2}
        , dim{dim}
        , simplex{simplex}
        , simplex1{simplex1}
        , simplex2{simplex2}
        , simplex_index1{simplex_index1}
        , simplex_index2{simplex_index2}
        , index1{index1}
        , index2{index2}

    {
    }

    CUDA_CALLABLE GJKResult_28609055& operator += (const GJKResult_28609055& rhs)
    {    dist += rhs.dist;
    x1 += rhs.x1;
    x2 += rhs.x2;
    dim += rhs.dim;
    simplex += rhs.simplex;
    simplex1 += rhs.simplex1;
    simplex2 += rhs.simplex2;
    simplex_index1 += rhs.simplex_index1;
    simplex_index2 += rhs.simplex_index2;
    index1 += rhs.index1;
    index2 += rhs.index2;

        return *this;}

};

static CUDA_CALLABLE void adj_GJKResult_28609055(wp::float32 const&,
    wp::vec_t<3, wp::float32> const&,
    wp::vec_t<3, wp::float32> const&,
    wp::int32 const&,
    wp::mat_t<4, 3, wp::float32> const&,
    wp::mat_t<4, 3, wp::float32> const&,
    wp::mat_t<4, 3, wp::float32> const&,
    wp::vec_t<4, wp::int32> const&,
    wp::vec_t<4, wp::int32> const&,
    wp::int32 const&,
    wp::int32 const&,
    wp::float32 & adj_dist,
    wp::vec_t<3, wp::float32> & adj_x1,
    wp::vec_t<3, wp::float32> & adj_x2,
    wp::int32 & adj_dim,
    wp::mat_t<4, 3, wp::float32> & adj_simplex,
    wp::mat_t<4, 3, wp::float32> & adj_simplex1,
    wp::mat_t<4, 3, wp::float32> & adj_simplex2,
    wp::vec_t<4, wp::int32> & adj_simplex_index1,
    wp::vec_t<4, wp::int32> & adj_simplex_index2,
    wp::int32 & adj_index1,
    wp::int32 & adj_index2,
    GJKResult_28609055 & adj_ret)
{
    adj_dist += adj_ret.dist;
    adj_x1 += adj_ret.x1;
    adj_x2 += adj_ret.x2;
    adj_dim += adj_ret.dim;
    adj_simplex += adj_ret.simplex;
    adj_simplex1 += adj_ret.simplex1;
    adj_simplex2 += adj_ret.simplex2;
    adj_simplex_index1 += adj_ret.simplex_index1;
    adj_simplex_index2 += adj_ret.simplex_index2;
    adj_index1 += adj_ret.index1;
    adj_index2 += adj_ret.index2;
}

// Required when compiling adjoints.
CUDA_CALLABLE GJKResult_28609055 add(const GJKResult_28609055& a, const GJKResult_28609055& b)
{
    return GJKResult_28609055();
}

CUDA_CALLABLE void adj_atomic_add(GJKResult_28609055* p, GJKResult_28609055 t)
{
    wp::adj_atomic_add(&p->dist, t.dist);
    wp::adj_atomic_add(&p->x1, t.x1);
    wp::adj_atomic_add(&p->x2, t.x2);
    wp::adj_atomic_add(&p->dim, t.dim);
    wp::adj_atomic_add(&p->simplex, t.simplex);
    wp::adj_atomic_add(&p->simplex1, t.simplex1);
    wp::adj_atomic_add(&p->simplex2, t.simplex2);
    wp::adj_atomic_add(&p->simplex_index1, t.simplex_index1);
    wp::adj_atomic_add(&p->simplex_index2, t.simplex_index2);
    wp::adj_atomic_add(&p->index1, t.index1);
    wp::adj_atomic_add(&p->index2, t.index2);
}




struct SupportPoint_e82efc60
{
    wp::vec_t<3, wp::float32> point;
    wp::int32 cached_index;
    wp::int32 vertex_index;


    SupportPoint_e82efc60() = default;
    CUDA_CALLABLE SupportPoint_e82efc60(wp::vec_t<3, wp::float32> const& point,
    wp::int32 const& cached_index = {},
    wp::int32 const& vertex_index = {})
        : point{point}
        , cached_index{cached_index}
        , vertex_index{vertex_index}

    {
    }

    CUDA_CALLABLE SupportPoint_e82efc60& operator += (const SupportPoint_e82efc60& rhs)
    {    point += rhs.point;
    cached_index += rhs.cached_index;
    vertex_index += rhs.vertex_index;

        return *this;}

};

static CUDA_CALLABLE void adj_SupportPoint_e82efc60(wp::vec_t<3, wp::float32> const&,
    wp::int32 const&,
    wp::int32 const&,
    wp::vec_t<3, wp::float32> & adj_point,
    wp::int32 & adj_cached_index,
    wp::int32 & adj_vertex_index,
    SupportPoint_e82efc60 & adj_ret)
{
    adj_point += adj_ret.point;
    adj_cached_index += adj_ret.cached_index;
    adj_vertex_index += adj_ret.vertex_index;
}

// Required when compiling adjoints.
CUDA_CALLABLE SupportPoint_e82efc60 add(const SupportPoint_e82efc60& a, const SupportPoint_e82efc60& b)
{
    return SupportPoint_e82efc60();
}

CUDA_CALLABLE void adj_atomic_add(SupportPoint_e82efc60* p, SupportPoint_e82efc60 t)
{
    wp::adj_atomic_add(&p->point, t.point);
    wp::adj_atomic_add(&p->cached_index, t.cached_index);
    wp::adj_atomic_add(&p->vertex_index, t.vertex_index);
}




struct Polytope_10582b13
{
    wp::int32 status;
    wp::array_t<wp::vec_t<3, wp::float32>> vert;
    wp::array_t<wp::int32> vert_index;
    wp::int32 nvert;
    wp::vec_t<3, wp::float32> center;
    wp::array_t<wp::int32> face;
    wp::array_t<wp::vec_t<3, wp::float32>> face_pr;
    wp::array_t<wp::float32> face_norm2;
    wp::int32 nface;
    wp::array_t<wp::int32> horizon;
    wp::int32 nhorizon;


    Polytope_10582b13() = default;
    CUDA_CALLABLE Polytope_10582b13(wp::int32 const& status,
    wp::array_t<wp::vec_t<3, wp::float32>> const& vert = {},
    wp::array_t<wp::int32> const& vert_index = {},
    wp::int32 const& nvert = {},
    wp::vec_t<3, wp::float32> const& center = {},
    wp::array_t<wp::int32> const& face = {},
    wp::array_t<wp::vec_t<3, wp::float32>> const& face_pr = {},
    wp::array_t<wp::float32> const& face_norm2 = {},
    wp::int32 const& nface = {},
    wp::array_t<wp::int32> const& horizon = {},
    wp::int32 const& nhorizon = {})
        : status{status}
        , vert{vert}
        , vert_index{vert_index}
        , nvert{nvert}
        , center{center}
        , face{face}
        , face_pr{face_pr}
        , face_norm2{face_norm2}
        , nface{nface}
        , horizon{horizon}
        , nhorizon{nhorizon}

    {
    }

    CUDA_CALLABLE Polytope_10582b13& operator += (const Polytope_10582b13& rhs)
    {    status += rhs.status;
    nvert += rhs.nvert;
    center += rhs.center;
    nface += rhs.nface;
    nhorizon += rhs.nhorizon;

        return *this;}

};

static CUDA_CALLABLE void adj_Polytope_10582b13(wp::int32 const&,
    wp::array_t<wp::vec_t<3, wp::float32>> const&,
    wp::array_t<wp::int32> const&,
    wp::int32 const&,
    wp::vec_t<3, wp::float32> const&,
    wp::array_t<wp::int32> const&,
    wp::array_t<wp::vec_t<3, wp::float32>> const&,
    wp::array_t<wp::float32> const&,
    wp::int32 const&,
    wp::array_t<wp::int32> const&,
    wp::int32 const&,
    wp::int32 & adj_status,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_vert,
    wp::array_t<wp::int32> & adj_vert_index,
    wp::int32 & adj_nvert,
    wp::vec_t<3, wp::float32> & adj_center,
    wp::array_t<wp::int32> & adj_face,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_face_pr,
    wp::array_t<wp::float32> & adj_face_norm2,
    wp::int32 & adj_nface,
    wp::array_t<wp::int32> & adj_horizon,
    wp::int32 & adj_nhorizon,
    Polytope_10582b13 & adj_ret)
{
    adj_status += adj_ret.status;
    adj_vert = adj_ret.vert;
    adj_vert_index = adj_ret.vert_index;
    adj_nvert += adj_ret.nvert;
    adj_center += adj_ret.center;
    adj_face = adj_ret.face;
    adj_face_pr = adj_ret.face_pr;
    adj_face_norm2 = adj_ret.face_norm2;
    adj_nface += adj_ret.nface;
    adj_horizon = adj_ret.horizon;
    adj_nhorizon += adj_ret.nhorizon;
}

// Required when compiling adjoints.
CUDA_CALLABLE Polytope_10582b13 add(const Polytope_10582b13& a, const Polytope_10582b13& b)
{
    return Polytope_10582b13();
}

CUDA_CALLABLE void adj_atomic_add(Polytope_10582b13* p, Polytope_10582b13 t)
{
    wp::adj_atomic_add(&p->status, t.status);
    wp::adj_atomic_add(&p->vert, t.vert);
    wp::adj_atomic_add(&p->vert_index, t.vert_index);
    wp::adj_atomic_add(&p->nvert, t.nvert);
    wp::adj_atomic_add(&p->center, t.center);
    wp::adj_atomic_add(&p->face, t.face);
    wp::adj_atomic_add(&p->face_pr, t.face_pr);
    wp::adj_atomic_add(&p->face_norm2, t.face_norm2);
    wp::adj_atomic_add(&p->nface, t.nface);
    wp::adj_atomic_add(&p->horizon, t.horizon);
    wp::adj_atomic_add(&p->nhorizon, t.nhorizon);
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


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:107
static CUDA_CALLABLE bool _discrete_geoms_0(
    wp::int32 var_g1,
    wp::int32 var_g2)
{
    //---------
    // primal vars
    bool var_0;
    bool var_1;
    const wp::int32 var_2 = 7;
    bool var_3;
    const wp::int32 var_4 = 6;
    bool var_5;
    const wp::int32 var_6 = 1;
    bool var_7;
    bool var_8;
    const wp::int32 var_9 = 7;
    bool var_10;
    const wp::int32 var_11 = 6;
    bool var_12;
    const wp::int32 var_13 = 1;
    bool var_14;
    //---------
    // forward
    // def _discrete_geoms(g1: int, g2: int) -> bool:                                         <L 108>
    // return (g1 == GeomType.MESH or g1 == GeomType.BOX or g1 == GeomType.HFIELD) and (       <L 109>
    var_3 = (var_g1 == var_2);
    var_1 = var_3;
    if (!var_1) {
        var_5 = (var_g1 == var_4);
        var_1 = var_1 || var_5;
    }
    if (!var_1) {
        var_7 = (var_g1 == var_6);
        var_1 = var_1 || var_7;
    }
    var_0 = var_1;
    if (var_0) {
        // g2 == GeomType.MESH or g2 == GeomType.BOX or g2 == GeomType.HFIELD                 <L 110>
        var_10 = (var_g2 == var_9);
        var_8 = var_10;
        if (!var_8) {
            var_12 = (var_g2 == var_11);
            var_8 = var_8 || var_12;
        }
        if (!var_8) {
            var_14 = (var_g2 == var_13);
            var_8 = var_8 || var_14;
        }
        var_0 = var_0 && var_8;
    }
    return var_0;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:114
static CUDA_CALLABLE SupportPoint_e82efc60 support_0(
    Geom_3242f8a8 var_geom,
    wp::int32 var_geomtype,
    wp::vec_t<3, wp::float32> var_dir)
{
    //---------
    // primal vars
    SupportPoint_e82efc60 var_0;
    const wp::int32 var_1 = -1;
    const wp::int32 var_2 = -1;
    const wp::int32 var_3 = 2;
    bool var_4;
    wp::vec_t<3, wp::float32>* var_5;
    wp::vec_t<3, wp::float32>* var_6;
    const wp::int32 var_7 = 0;
    wp::float32 var_8;
    wp::vec_t<3, wp::float32> var_9;
    const wp::float32 var_10 = 0.5;
    wp::float32* var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::vec_t<3, wp::float32> var_16;
    wp::vec_t<3, wp::float32> var_17;
    wp::mat_t<3, 3, wp::float32>* var_18;
    wp::mat_t<3, 3, wp::float32> var_19;
    wp::mat_t<3, 3, wp::float32> var_20;
    wp::vec_t<3, wp::float32> var_21;
    const wp::int32 var_22 = 6;
    bool var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::vec_t<3, wp::float32>* var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::mat_t<3, 3, wp::float32>* var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::mat_t<3, 3, wp::float32> var_30;
    wp::vec_t<3, wp::float32>* var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::vec_t<3, wp::float32> var_33;
    const wp::int32 var_34 = 0;
    wp::float32 var_35;
    const wp::float32 var_36 = 0.0;
    bool var_37;
    const wp::int32 var_38 = 1;
    const wp::int32 var_39 = 0;
    wp::int32 var_40;
    const wp::int32 var_41 = 1;
    wp::float32 var_42;
    const wp::float32 var_43 = 0.0;
    bool var_44;
    const wp::int32 var_45 = 2;
    const wp::int32 var_46 = 0;
    wp::int32 var_47;
    wp::int32* var_48;
    wp::int32 var_49;
    wp::int32 var_50;
    const wp::int32 var_51 = 2;
    wp::float32 var_52;
    const wp::float32 var_53 = 0.0;
    bool var_54;
    const wp::int32 var_55 = 4;
    const wp::int32 var_56 = 0;
    wp::int32 var_57;
    wp::int32* var_58;
    wp::int32 var_59;
    wp::int32 var_60;
    const wp::int32 var_61 = 3;
    bool var_62;
    wp::vec_t<3, wp::float32>* var_63;
    const wp::int32 var_64 = 0;
    wp::float32 var_65;
    wp::vec_t<3, wp::float32> var_66;
    wp::vec_t<3, wp::float32> var_67;
    const wp::int32 var_68 = 2;
    wp::float32 var_69;
    wp::float32 var_70;
    wp::vec_t<3, wp::float32>* var_71;
    const wp::int32 var_72 = 1;
    wp::float32 var_73;
    wp::vec_t<3, wp::float32> var_74;
    wp::float32 var_75;
    const wp::int32 var_76 = 2;
    wp::mat_t<3, 3, wp::float32>* var_77;
    wp::vec_t<3, wp::float32> var_78;
    wp::mat_t<3, 3, wp::float32> var_79;
    wp::vec_t<3, wp::float32>* var_80;
    wp::vec_t<3, wp::float32> var_81;
    wp::vec_t<3, wp::float32> var_82;
    wp::vec_t<3, wp::float32> var_83;
    const wp::int32 var_84 = 4;
    bool var_85;
    wp::vec_t<3, wp::float32>* var_86;
    wp::vec_t<3, wp::float32> var_87;
    wp::vec_t<3, wp::float32> var_88;
    wp::vec_t<3, wp::float32> var_89;
    wp::vec_t<3, wp::float32>* var_90;
    wp::vec_t<3, wp::float32> var_91;
    wp::vec_t<3, wp::float32> var_92;
    wp::mat_t<3, 3, wp::float32>* var_93;
    wp::vec_t<3, wp::float32> var_94;
    wp::mat_t<3, 3, wp::float32> var_95;
    wp::vec_t<3, wp::float32>* var_96;
    wp::vec_t<3, wp::float32> var_97;
    wp::vec_t<3, wp::float32> var_98;
    wp::vec_t<3, wp::float32> var_99;
    const wp::int32 var_100 = 5;
    bool var_101;
    const wp::float32 var_102 = 0.0;
    const wp::float32 var_103 = 0.0;
    const wp::float32 var_104 = 0.0;
    wp::vec_t<3, wp::float32> var_105;
    const wp::int32 var_106 = 0;
    wp::float32 var_107;
    const wp::int32 var_108 = 0;
    wp::float32 var_109;
    wp::float32 var_110;
    const wp::int32 var_111 = 1;
    wp::float32 var_112;
    const wp::int32 var_113 = 1;
    wp::float32 var_114;
    wp::float32 var_115;
    wp::float32 var_116;
    wp::float32 var_117;
    const wp::float32 var_118 = 1e-15;
    bool var_119;
    wp::vec_t<3, wp::float32>* var_120;
    const wp::int32 var_121 = 0;
    wp::float32 var_122;
    wp::vec_t<3, wp::float32> var_123;
    wp::float32 var_124;
    const wp::int32 var_125 = 0;
    wp::float32 var_126;
    wp::float32 var_127;
    const wp::int32 var_128 = 0;
    const wp::int32 var_129 = 1;
    wp::float32 var_130;
    wp::float32 var_131;
    const wp::int32 var_132 = 1;
    const wp::int32 var_133 = 2;
    wp::float32 var_134;
    wp::float32 var_135;
    wp::vec_t<3, wp::float32>* var_136;
    const wp::int32 var_137 = 1;
    wp::float32 var_138;
    wp::vec_t<3, wp::float32> var_139;
    wp::float32 var_140;
    const wp::int32 var_141 = 2;
    wp::mat_t<3, 3, wp::float32>* var_142;
    wp::vec_t<3, wp::float32> var_143;
    wp::mat_t<3, 3, wp::float32> var_144;
    wp::vec_t<3, wp::float32>* var_145;
    wp::vec_t<3, wp::float32> var_146;
    wp::vec_t<3, wp::float32> var_147;
    wp::vec_t<3, wp::float32> var_148;
    const wp::int32 var_149 = 7;
    bool var_150;
    const wp::float32 var_151 = -1e+30;
    wp::float32 var_152;
    bool var_153;
    wp::int32* var_154;
    const wp::int32 var_155 = -1;
    bool var_156;
    wp::int32 var_157;
    wp::int32* var_158;
    const wp::int32 var_159 = 10;
    bool var_160;
    wp::int32 var_161;
    wp::int32* var_162;
    const wp::int32 var_163 = -1;
    bool var_164;
    wp::int32 var_165;
    wp::int32* var_166;
    wp::int32 var_167;
    wp::int32 var_168;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_169;
    wp::int32* var_170;
    wp::vec_t<3, wp::float32>* var_171;
    wp::array_t<wp::vec_t<3, wp::float32>> var_172;
    wp::int32 var_173;
    wp::float32 var_174;
    wp::vec_t<3, wp::float32> var_175;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_176;
    wp::int32* var_177;
    wp::vec_t<3, wp::float32>* var_178;
    wp::array_t<wp::vec_t<3, wp::float32>> var_179;
    wp::int32 var_180;
    wp::vec_t<3, wp::float32> var_181;
    wp::vec_t<3, wp::float32> var_182;
    wp::float32 var_183;
    wp::int32* var_184;
    wp::range_t var_185;
    wp::int32 var_186;
    wp::int32 var_187;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_188;
    wp::int32* var_189;
    wp::int32 var_190;
    wp::int32 var_191;
    wp::vec_t<3, wp::float32>* var_192;
    wp::array_t<wp::vec_t<3, wp::float32>> var_193;
    wp::vec_t<3, wp::float32> var_194;
    wp::vec_t<3, wp::float32> var_195;
    wp::float32 var_196;
    bool var_197;
    wp::float32 var_198;
    wp::int32* var_199;
    wp::int32 var_200;
    wp::int32 var_201;
    wp::float32 var_202;
    wp::int32* var_203;
    wp::int32* var_204;
    wp::int32 var_205;
    wp::int32 var_206;
    wp::int32 var_207;
    wp::float32 var_208;
    wp::array_t<wp::int32>* var_209;
    wp::int32* var_210;
    wp::int32* var_211;
    wp::array_t<wp::int32> var_212;
    wp::int32 var_213;
    wp::int32 var_214;
    wp::int32 var_215;
    wp::int32* var_216;
    const wp::int32 var_217 = 2;
    wp::int32 var_218;
    wp::int32 var_219;
    wp::int32* var_220;
    const wp::int32 var_221 = 2;
    wp::int32 var_222;
    wp::int32 var_223;
    wp::int32 var_224;
    wp::int32* var_225;
    const wp::int32 var_226 = 2;
    wp::int32 var_227;
    wp::int32 var_228;
    const wp::int32 var_229 = 2;
    wp::int32 var_230;
    wp::int32 var_231;
    const wp::int32 var_232 = -1;
    wp::int32 var_233;
    wp::int32* var_234;
    const wp::int32 var_235 = -1;
    bool var_236;
    wp::int32 var_237;
    wp::int32* var_238;
    const wp::int32 var_239 = 0;
    wp::int32 var_240;
    wp::int32 var_241;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_242;
    wp::int32* var_243;
    wp::array_t<wp::int32>* var_244;
    wp::int32 var_245;
    wp::int32* var_246;
    wp::array_t<wp::int32> var_247;
    wp::int32 var_248;
    wp::int32 var_249;
    wp::int32 var_250;
    wp::vec_t<3, wp::float32>* var_251;
    wp::array_t<wp::vec_t<3, wp::float32>> var_252;
    wp::float32 var_253;
    wp::vec_t<3, wp::float32> var_254;
    bool var_255;
    wp::int32 var_256;
    wp::array_t<wp::int32>* var_257;
    wp::int32 var_258;
    wp::int32* var_259;
    wp::array_t<wp::int32> var_260;
    wp::int32 var_261;
    wp::int32 var_262;
    wp::array_t<wp::int32>* var_263;
    wp::int32 var_264;
    wp::int32* var_265;
    wp::array_t<wp::int32> var_266;
    wp::int32 var_267;
    wp::int32 var_268;
    const wp::int32 var_269 = 0;
    bool var_270;
    wp::array_t<wp::int32>* var_271;
    wp::int32 var_272;
    wp::int32* var_273;
    wp::array_t<wp::int32> var_274;
    wp::int32 var_275;
    wp::int32 var_276;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_277;
    wp::int32* var_278;
    wp::int32 var_279;
    wp::int32 var_280;
    wp::vec_t<3, wp::float32>* var_281;
    wp::array_t<wp::vec_t<3, wp::float32>> var_282;
    wp::float32 var_283;
    wp::vec_t<3, wp::float32> var_284;
    bool var_285;
    wp::int32 var_286;
    bool var_287;
    wp::float32 var_288;
    const wp::int32 var_289 = 1;
    wp::int32 var_290;
    wp::array_t<wp::int32>* var_291;
    wp::int32 var_292;
    wp::int32* var_293;
    wp::array_t<wp::int32> var_294;
    wp::int32 var_295;
    wp::int32 var_296;
    wp::array_t<wp::int32>* var_297;
    wp::int32 var_298;
    wp::int32* var_299;
    wp::array_t<wp::int32> var_300;
    wp::int32 var_301;
    wp::int32 var_302;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_303;
    wp::int32* var_304;
    wp::int32* var_305;
    wp::int32 var_306;
    wp::int32 var_307;
    wp::int32 var_308;
    wp::vec_t<3, wp::float32>* var_309;
    wp::array_t<wp::vec_t<3, wp::float32>> var_310;
    wp::vec_t<3, wp::float32> var_311;
    wp::vec_t<3, wp::float32> var_312;
    wp::float32 var_313;
    wp::mat_t<3, 3, wp::float32>* var_314;
    wp::vec_t<3, wp::float32>* var_315;
    wp::vec_t<3, wp::float32> var_316;
    wp::mat_t<3, 3, wp::float32> var_317;
    wp::vec_t<3, wp::float32> var_318;
    wp::vec_t<3, wp::float32>* var_319;
    wp::vec_t<3, wp::float32> var_320;
    wp::vec_t<3, wp::float32> var_321;
    const wp::int32 var_322 = 1;
    bool var_323;
    wp::float32 var_324;
    const wp::int32 var_325 = 2;
    wp::float32 var_326;
    const wp::float32 var_327 = 0.0;
    bool var_328;
    const wp::int32 var_329 = -2;
    const wp::int32 var_330 = -3;
    wp::int32 var_331;
    const wp::int32 var_332 = 0;
    wp::mat_t<6, 3, wp::float32>* var_333;
    wp::vec_t<3, wp::float32> var_334;
    wp::mat_t<6, 3, wp::float32> var_335;
    wp::float32 var_336;
    bool var_337;
    wp::float32 var_338;
    wp::float32 var_339;
    const wp::int32 var_340 = 1;
    wp::mat_t<6, 3, wp::float32>* var_341;
    wp::vec_t<3, wp::float32> var_342;
    wp::mat_t<6, 3, wp::float32> var_343;
    wp::float32 var_344;
    bool var_345;
    wp::float32 var_346;
    wp::float32 var_347;
    const wp::int32 var_348 = 2;
    wp::mat_t<6, 3, wp::float32>* var_349;
    wp::vec_t<3, wp::float32> var_350;
    wp::mat_t<6, 3, wp::float32> var_351;
    wp::float32 var_352;
    bool var_353;
    wp::float32 var_354;
    wp::float32 var_355;
    const wp::int32 var_356 = 3;
    wp::mat_t<6, 3, wp::float32>* var_357;
    wp::vec_t<3, wp::float32> var_358;
    wp::mat_t<6, 3, wp::float32> var_359;
    wp::float32 var_360;
    bool var_361;
    wp::float32 var_362;
    wp::float32 var_363;
    const wp::int32 var_364 = 4;
    wp::mat_t<6, 3, wp::float32>* var_365;
    wp::vec_t<3, wp::float32> var_366;
    wp::mat_t<6, 3, wp::float32> var_367;
    wp::float32 var_368;
    bool var_369;
    wp::float32 var_370;
    wp::float32 var_371;
    const wp::int32 var_372 = 5;
    wp::mat_t<6, 3, wp::float32>* var_373;
    wp::vec_t<3, wp::float32> var_374;
    wp::mat_t<6, 3, wp::float32> var_375;
    wp::float32 var_376;
    bool var_377;
    wp::float32 var_378;
    wp::float32 var_379;
    wp::float32 var_380;
    wp::int32 var_381;
    wp::vec_t<3, wp::float32> var_382;
    wp::float32 var_383;
    const wp::int32 var_384 = 999;
    bool var_385;
    wp::mat_t<3, 3, wp::float32>* var_386;
    const wp::int32 var_387 = 0;
    wp::slice_t var_388;
    const wp::int32 var_389 = 0;
    const wp::int32 var_390 = 3;
    const wp::int32 var_391 = 1;
    wp::vec_t<3, wp::float32> var_392;
    wp::mat_t<3, 3, wp::float32> var_393;
    wp::mat_t<3, 3, wp::float32>* var_394;
    const wp::int32 var_395 = 1;
    wp::slice_t var_396;
    const wp::int32 var_397 = 0;
    const wp::int32 var_398 = 3;
    const wp::int32 var_399 = 1;
    wp::vec_t<3, wp::float32> var_400;
    wp::mat_t<3, 3, wp::float32> var_401;
    wp::mat_t<3, 3, wp::float32>* var_402;
    const wp::int32 var_403 = 2;
    wp::slice_t var_404;
    const wp::int32 var_405 = 0;
    const wp::int32 var_406 = 3;
    const wp::int32 var_407 = 1;
    wp::vec_t<3, wp::float32> var_408;
    wp::mat_t<3, 3, wp::float32> var_409;
    wp::float32 var_410;
    wp::float32 var_411;
    wp::float32 var_412;
    bool var_413;
    bool var_414;
    bool var_415;
    bool var_416;
    wp::float32 var_417;
    wp::int32 var_418;
    wp::vec_t<3, wp::float32> var_419;
    wp::float32 var_420;
    wp::vec_t<3, wp::float32> var_421;
    wp::vec_t<3, wp::float32> var_422;
    wp::vec_t<3, wp::float32> var_423;
    wp::float32* var_424;
    const wp::float32 var_425 = 0.0;
    bool var_426;
    wp::float32 var_427;
    const wp::float32 var_428 = 0.5;
    wp::float32* var_429;
    wp::float32 var_430;
    wp::float32 var_431;
    wp::vec_t<3, wp::float32> var_432;
    wp::vec_t<3, wp::float32>* var_433;
    wp::vec_t<3, wp::float32> var_434;
    wp::vec_t<3, wp::float32> var_435;
    //---------
    // forward
    // def support(geom: Geom, geomtype: int, dir: wp.vec3) -> SupportPoint:                  <L 115>
    // sp = SupportPoint()                                                                    <L 116>
    var_0 = SupportPoint_e82efc60();
    // sp.cached_index = -1                                                                   <L 117>
    var_0.cached_index = var_1;
    // sp.vertex_index = -1                                                                   <L 118>
    var_0.vertex_index = var_2;
    // if geomtype == GeomType.SPHERE:                                                        <L 119>
    var_4 = (var_geomtype == var_3);
    if (var_4) {
        // sp.point = geom.pos + (geom.size[0] + 0.5 * geom.margin) * dir                     <L 120>
        var_5 = &((var_geom).pos);
        var_6 = &((var_geom).size);
        var_9 = wp::load(var_6);
        var_8 = wp::extract(var_9, var_7);
        var_11 = &((var_geom).margin);
        var_13 = wp::load(var_11);
        var_12 = wp::mul(var_10, var_13);
        var_14 = wp::add(var_8, var_12);
        var_15 = wp::mul(var_14, var_dir);
        var_17 = wp::load(var_5);
        var_16 = wp::add(var_17, var_15);
        var_0.point = var_16;
        // return sp                                                                          <L 121>
        return var_0;
    }
    // local_dir = wp.transpose(geom.rot) @ dir                                               <L 123>
    var_18 = &((var_geom).rot);
    var_20 = wp::load(var_18);
    var_19 = wp::transpose(var_20);
    var_21 = wp::mul(var_19, var_dir);
    // if geomtype == GeomType.BOX:                                                           <L 124>
    var_23 = (var_geomtype == var_22);
    if (var_23) {
        // tmp = wp.sign(local_dir)                                                           <L 125>
        var_24 = wp::sign(var_21);
        // res = wp.cw_mul(tmp, geom.size)                                                    <L 126>
        var_25 = &((var_geom).size);
        var_27 = wp::load(var_25);
        var_26 = wp::cw_mul(var_24, var_27);
        // sp.point = geom.rot @ res + geom.pos                                               <L 127>
        var_28 = &((var_geom).rot);
        var_30 = wp::load(var_28);
        var_29 = wp::mul(var_30, var_26);
        var_31 = &((var_geom).pos);
        var_33 = wp::load(var_31);
        var_32 = wp::add(var_29, var_33);
        var_0.point = var_32;
        // sp.vertex_index = wp.where(tmp[0] > 0.0, 1, 0)                                     <L 128>
        var_35 = wp::extract(var_24, var_34);
        var_37 = (var_35 > var_36);
        var_40 = wp::where(var_37, var_38, var_39);
        var_0.vertex_index = var_40;
        // sp.vertex_index += wp.where(tmp[1] > 0.0, 2, 0)                                    <L 129>
        var_42 = wp::extract(var_24, var_41);
        var_44 = (var_42 > var_43);
        var_47 = wp::where(var_44, var_45, var_46);
        var_48 = &((var_0).vertex_index);
        var_50 = wp::load(var_48);
        var_49 = wp::add(var_50, var_47);
        var_0.vertex_index = var_49;
        // sp.vertex_index += wp.where(tmp[2] > 0.0, 4, 0)                                    <L 130>
        var_52 = wp::extract(var_24, var_51);
        var_54 = (var_52 > var_53);
        var_57 = wp::where(var_54, var_55, var_56);
        var_58 = &((var_0).vertex_index);
        var_60 = wp::load(var_58);
        var_59 = wp::add(var_60, var_57);
        var_0.vertex_index = var_59;
    }
    if (!var_23) {
        // elif geomtype == GeomType.CAPSULE:                                                 <L 131>
        var_62 = (var_geomtype == var_61);
        if (var_62) {
            // res = local_dir * geom.size[0]                                                 <L 132>
            var_63 = &((var_geom).size);
            var_66 = wp::load(var_63);
            var_65 = wp::extract(var_66, var_64);
            var_67 = wp::mul(var_21, var_65);
            // res[2] += wp.sign(local_dir[2]) * geom.size[1]                                 <L 134>
            var_69 = wp::extract(var_21, var_68);
            var_70 = wp::sign(var_69);
            var_71 = &((var_geom).size);
            var_74 = wp::load(var_71);
            var_73 = wp::extract(var_74, var_72);
            var_75 = wp::mul(var_70, var_73);
            wp::add_inplace(var_67, var_76, var_75);
            // sp.point = geom.rot @ res + geom.pos                                           <L 135>
            var_77 = &((var_geom).rot);
            var_79 = wp::load(var_77);
            var_78 = wp::mul(var_79, var_67);
            var_80 = &((var_geom).pos);
            var_82 = wp::load(var_80);
            var_81 = wp::add(var_78, var_82);
            var_0.point = var_81;
        }
        var_83 = wp::where(var_62, var_67, var_26);
        if (!var_62) {
            // elif geomtype == GeomType.ELLIPSOID:                                           <L 136>
            var_85 = (var_geomtype == var_84);
            if (var_85) {
                // res = wp.cw_mul(local_dir, geom.size)                                      <L 137>
                var_86 = &((var_geom).size);
                var_88 = wp::load(var_86);
                var_87 = wp::cw_mul(var_21, var_88);
                // res = wp.normalize(res)                                                    <L 138>
                var_89 = wp::normalize(var_87);
                // res = wp.cw_mul(res, geom.size)                                            <L 140>
                var_90 = &((var_geom).size);
                var_92 = wp::load(var_90);
                var_91 = wp::cw_mul(var_89, var_92);
                // sp.point = geom.rot @ res + geom.pos                                       <L 141>
                var_93 = &((var_geom).rot);
                var_95 = wp::load(var_93);
                var_94 = wp::mul(var_95, var_91);
                var_96 = &((var_geom).pos);
                var_98 = wp::load(var_96);
                var_97 = wp::add(var_94, var_98);
                var_0.point = var_97;
            }
            var_99 = wp::where(var_85, var_91, var_83);
            if (!var_85) {
                // elif geomtype == GeomType.CYLINDER:                                        <L 142>
                var_101 = (var_geomtype == var_100);
                if (var_101) {
                    // res = wp.vec3(0.0, 0.0, 0.0)                                           <L 143>
                    var_105 = wp::vec_t<3, wp::float32>(var_102, var_103, var_104);
                    // d = wp.sqrt(local_dir[0] * local_dir[0] + local_dir[1] * local_dir[1])       <L 145>
                    var_107 = wp::extract(var_21, var_106);
                    var_109 = wp::extract(var_21, var_108);
                    var_110 = wp::mul(var_107, var_109);
                    var_112 = wp::extract(var_21, var_111);
                    var_114 = wp::extract(var_21, var_113);
                    var_115 = wp::mul(var_112, var_114);
                    var_116 = wp::add(var_110, var_115);
                    var_117 = wp::sqrt(var_116);
                    // if d > MINVAL:                                                         <L 146>
                    var_119 = (var_117 > var_118);
                    if (var_119) {
                        // scl = geom.size[0] / d                                             <L 147>
                        var_120 = &((var_geom).size);
                        var_123 = wp::load(var_120);
                        var_122 = wp::extract(var_123, var_121);
                        var_124 = wp::div(var_122, var_117);
                        // res[0] = local_dir[0] * scl                                        <L 148>
                        var_126 = wp::extract(var_21, var_125);
                        var_127 = wp::mul(var_126, var_124);
                        wp::assign_inplace(var_105, var_128, var_127);
                        // res[1] = local_dir[1] * scl                                        <L 149>
                        var_130 = wp::extract(var_21, var_129);
                        var_131 = wp::mul(var_130, var_124);
                        wp::assign_inplace(var_105, var_132, var_131);
                    }
                    // res[2] = wp.sign(local_dir[2]) * geom.size[1]                          <L 151>
                    var_134 = wp::extract(var_21, var_133);
                    var_135 = wp::sign(var_134);
                    var_136 = &((var_geom).size);
                    var_139 = wp::load(var_136);
                    var_138 = wp::extract(var_139, var_137);
                    var_140 = wp::mul(var_135, var_138);
                    wp::assign_inplace(var_105, var_141, var_140);
                    // sp.point = geom.rot @ res + geom.pos                                   <L 152>
                    var_142 = &((var_geom).rot);
                    var_144 = wp::load(var_142);
                    var_143 = wp::mul(var_144, var_105);
                    var_145 = &((var_geom).pos);
                    var_147 = wp::load(var_145);
                    var_146 = wp::add(var_143, var_147);
                    var_0.point = var_146;
                }
                var_148 = wp::where(var_101, var_105, var_99);
                if (!var_101) {
                    // elif geomtype == GeomType.MESH:                                        <L 153>
                    var_150 = (var_geomtype == var_149);
                    if (var_150) {
                        // max_dist = float(FLOAT_MIN)                                        <L 154>
                        var_152 = wp::float(var_151);
                        // if geom.graphadr == -1 or geom.vertnum < 10:                       <L 155>
                        var_154 = &((var_geom).graphadr);
                        var_157 = wp::load(var_154);
                        var_156 = (var_157 == var_155);
                        var_153 = var_156;
                        if (!var_153) {
                            var_158 = &((var_geom).vertnum);
                            var_161 = wp::load(var_158);
                            var_160 = (var_161 < var_159);
                            var_153 = var_153 || var_160;
                        }
                        if (var_153) {
                            // if geom.index > -1:                                            <L 156>
                            var_162 = &((var_geom).index);
                            var_165 = wp::load(var_162);
                            var_164 = (var_165 > var_163);
                            if (var_164) {
                                // sp.cached_index = geom.index                               <L 157>
                                var_166 = &((var_geom).index);
                                var_168 = wp::load(var_166);
                                var_167 = wp::copy(var_168);
                                var_0.cached_index = var_167;
                                // max_dist = wp.dot(geom.vert[geom.index], local_dir)        <L 158>
                                var_169 = &((var_geom).vert);
                                var_170 = &((var_geom).index);
                                var_172 = wp::load(var_169);
                                var_173 = wp::load(var_170);
                                var_171 = wp::address(var_172, var_173);
                                var_175 = wp::load(var_171);
                                var_174 = wp::dot(var_175, var_21);
                                // sp.point = geom.vert[geom.index]                           <L 159>
                                var_176 = &((var_geom).vert);
                                var_177 = &((var_geom).index);
                                var_179 = wp::load(var_176);
                                var_180 = wp::load(var_177);
                                var_178 = wp::address(var_179, var_180);
                                var_182 = wp::load(var_178);
                                var_181 = wp::copy(var_182);
                                var_0.point = var_181;
                            }
                            var_183 = wp::where(var_164, var_174, var_152);
                            // for i in range(geom.vertnum):                                  <L 161>
                            var_184 = &((var_geom).vertnum);
                            var_186 = wp::load(var_184);
                            var_185 = wp::range(var_186);
                            start_for_1:;
                                if (iter_cmp(var_185) == 0) goto end_for_1;
                                var_187 = wp::iter_next(var_185);
                                // vert = geom.vert[geom.vertadr + i]                         <L 162>
                                var_188 = &((var_geom).vert);
                                var_189 = &((var_geom).vertadr);
                                var_191 = wp::load(var_189);
                                var_190 = wp::add(var_191, var_187);
                                var_193 = wp::load(var_188);
                                var_192 = wp::address(var_193, var_190);
                                var_195 = wp::load(var_192);
                                var_194 = wp::copy(var_195);
                                // dist = wp.dot(vert, local_dir)                             <L 163>
                                var_196 = wp::dot(var_194, var_21);
                                // if dist > max_dist:                                        <L 164>
                                var_197 = (var_196 > var_183);
                                if (var_197) {
                                    // max_dist = dist                                        <L 165>
                                    var_198 = wp::copy(var_196);
                                    // sp.point = vert                                        <L 166>
                                    var_0.point = var_194;
                                    // sp.cached_index = geom.vertadr + i                     <L 167>
                                    var_199 = &((var_geom).vertadr);
                                    var_201 = wp::load(var_199);
                                    var_200 = wp::add(var_201, var_187);
                                    var_0.cached_index = var_200;
                                }
                                var_202 = wp::where(var_197, var_198, var_183);
                                wp::assign(var_183, var_202);
                                goto start_for_1;
                            end_for_1:;
                            // sp.vertex_index = sp.cached_index - geom.vertadr               <L 168>
                            var_203 = &((var_0).cached_index);
                            var_204 = &((var_geom).vertadr);
                            var_206 = wp::load(var_203);
                            var_207 = wp::load(var_204);
                            var_205 = wp::sub(var_206, var_207);
                            var_0.vertex_index = var_205;
                        }
                        var_208 = wp::where(var_153, var_183, var_152);
                        if (!var_153) {
                            // numvert = geom.graph[geom.graphadr]                            <L 170>
                            var_209 = &((var_geom).graph);
                            var_210 = &((var_geom).graphadr);
                            var_212 = wp::load(var_209);
                            var_213 = wp::load(var_210);
                            var_211 = wp::address(var_212, var_213);
                            var_215 = wp::load(var_211);
                            var_214 = wp::copy(var_215);
                            // vert_edgeadr = geom.graphadr + 2                               <L 171>
                            var_216 = &((var_geom).graphadr);
                            var_219 = wp::load(var_216);
                            var_218 = wp::add(var_219, var_217);
                            // vert_globalid = geom.graphadr + 2 + numvert                    <L 172>
                            var_220 = &((var_geom).graphadr);
                            var_223 = wp::load(var_220);
                            var_222 = wp::add(var_223, var_221);
                            var_224 = wp::add(var_222, var_214);
                            // edge_localid = geom.graphadr + 2 + 2 * numvert                 <L 173>
                            var_225 = &((var_geom).graphadr);
                            var_228 = wp::load(var_225);
                            var_227 = wp::add(var_228, var_226);
                            var_230 = wp::mul(var_229, var_214);
                            var_231 = wp::add(var_227, var_230);
                            // prev = int(-1)                                                 <L 174>
                            var_233 = wp::int(var_232);
                            // imax = wp.where(geom.index > -1, geom.index, 0)                <L 175>
                            var_234 = &((var_geom).index);
                            var_237 = wp::load(var_234);
                            var_236 = (var_237 > var_235);
                            var_238 = &((var_geom).index);
                            var_241 = wp::load(var_238);
                            var_240 = wp::where(var_236, var_241, var_239);
                            // max_dist = wp.dot(local_dir, geom.vert[geom.vertadr + geom.graph[vert_globalid + imax]])       <L 176>
                            var_242 = &((var_geom).vert);
                            var_243 = &((var_geom).vertadr);
                            var_244 = &((var_geom).graph);
                            var_245 = wp::add(var_224, var_240);
                            var_247 = wp::load(var_244);
                            var_246 = wp::address(var_247, var_245);
                            var_249 = wp::load(var_243);
                            var_250 = wp::load(var_246);
                            var_248 = wp::add(var_249, var_250);
                            var_252 = wp::load(var_242);
                            var_251 = wp::address(var_252, var_248);
                            var_254 = wp::load(var_251);
                            var_253 = wp::dot(var_21, var_254);
                            // while imax != prev:                                            <L 179>
    start_while_3:;
                            var_255 = (var_240 != var_233);
    if ((var_255) == false) goto end_while_3;
                                // prev = imax                                                <L 180>
                                var_256 = wp::copy(var_240);
                                // i = geom.graph[vert_edgeadr + imax]                        <L 181>
                                var_257 = &((var_geom).graph);
                                var_258 = wp::add(var_218, var_240);
                                var_260 = wp::load(var_257);
                                var_259 = wp::address(var_260, var_258);
                                var_262 = wp::load(var_259);
                                var_261 = wp::copy(var_262);
                                // subidx = geom.graph[edge_localid + i]                      <L 182>
                                var_263 = &((var_geom).graph);
                                var_264 = wp::add(var_231, var_261);
                                var_266 = wp::load(var_263);
                                var_265 = wp::address(var_266, var_264);
                                var_268 = wp::load(var_265);
                                var_267 = wp::copy(var_268);
                                // while subidx >= 0:                                         <L 183>
    start_while_5:;
                                var_270 = (var_267 >= var_269);
    if ((var_270) == false) goto end_while_5;
                                    // idx = geom.graph[vert_globalid + subidx]               <L 184>
                                    var_271 = &((var_geom).graph);
                                    var_272 = wp::add(var_224, var_267);
                                    var_274 = wp::load(var_271);
                                    var_273 = wp::address(var_274, var_272);
                                    var_276 = wp::load(var_273);
                                    var_275 = wp::copy(var_276);
                                    // dist = wp.dot(local_dir, geom.vert[geom.vertadr + idx])       <L 185>
                                    var_277 = &((var_geom).vert);
                                    var_278 = &((var_geom).vertadr);
                                    var_280 = wp::load(var_278);
                                    var_279 = wp::add(var_280, var_275);
                                    var_282 = wp::load(var_277);
                                    var_281 = wp::address(var_282, var_279);
                                    var_284 = wp::load(var_281);
                                    var_283 = wp::dot(var_21, var_284);
                                    // imax = wp.where(dist > max_dist, subidx, imax)         <L 186>
                                    var_285 = (var_283 > var_253);
                                    var_286 = wp::where(var_285, var_267, var_240);
                                    // max_dist = wp.where(dist > max_dist, dist, max_dist)       <L 187>
                                    var_287 = (var_283 > var_253);
                                    var_288 = wp::where(var_287, var_283, var_253);
                                    // i += 1                                                 <L 188>
                                    var_290 = wp::add(var_261, var_289);
                                    // subidx = geom.graph[edge_localid + i]                  <L 189>
                                    var_291 = &((var_geom).graph);
                                    var_292 = wp::add(var_231, var_290);
                                    var_294 = wp::load(var_291);
                                    var_293 = wp::address(var_294, var_292);
                                    var_296 = wp::load(var_293);
                                    var_295 = wp::copy(var_296);
                                    wp::assign(var_253, var_288);
                                    wp::assign(var_261, var_290);
                                    wp::assign(var_196, var_283);
                                    wp::assign(var_240, var_286);
                                    wp::assign(var_267, var_295);
    goto start_while_5;
    end_while_5:;
                                wp::assign(var_187, var_261);
                                wp::assign(var_233, var_256);
    goto start_while_3;
    end_while_3:;
                            // sp.cached_index = imax                                         <L 191>
                            var_0.cached_index = var_240;
                            // sp.vertex_index = geom.graph[vert_globalid + imax]             <L 192>
                            var_297 = &((var_geom).graph);
                            var_298 = wp::add(var_224, var_240);
                            var_300 = wp::load(var_297);
                            var_299 = wp::address(var_300, var_298);
                            var_302 = wp::load(var_299);
                            var_301 = wp::copy(var_302);
                            var_0.vertex_index = var_301;
                            // sp.point = geom.vert[geom.vertadr + sp.vertex_index]           <L 193>
                            var_303 = &((var_geom).vert);
                            var_304 = &((var_geom).vertadr);
                            var_305 = &((var_0).vertex_index);
                            var_307 = wp::load(var_304);
                            var_308 = wp::load(var_305);
                            var_306 = wp::add(var_307, var_308);
                            var_310 = wp::load(var_303);
                            var_309 = wp::address(var_310, var_306);
                            var_312 = wp::load(var_309);
                            var_311 = wp::copy(var_312);
                            var_0.point = var_311;
                        }
                        var_313 = wp::where(var_153, var_208, var_253);
                        // sp.point = geom.rot @ sp.point + geom.pos                          <L 195>
                        var_314 = &((var_geom).rot);
                        var_315 = &((var_0).point);
                        var_317 = wp::load(var_314);
                        var_318 = wp::load(var_315);
                        var_316 = wp::mul(var_317, var_318);
                        var_319 = &((var_geom).pos);
                        var_321 = wp::load(var_319);
                        var_320 = wp::add(var_316, var_321);
                        var_0.point = var_320;
                    }
                    if (!var_150) {
                        // elif geomtype == GeomType.HFIELD:                                  <L 196>
                        var_323 = (var_geomtype == var_322);
                        if (var_323) {
                            // max_dist = float(FLOAT_MIN)                                    <L 197>
                            var_324 = wp::float(var_151);
                            // sp.vertex_index = wp.where(dir[2] < 0.0, -2, -3)               <L 199>
                            var_326 = wp::extract(var_dir, var_325);
                            var_328 = (var_326 < var_327);
                            var_331 = wp::where(var_328, var_329, var_330);
                            var_0.vertex_index = var_331;
                            // for i in range(6):                                             <L 200>
                            // vert = geom.hfprism[i]                                         <L 201>
                            var_333 = &((var_geom).hfprism);
                            var_335 = wp::load(var_333);
                            var_334 = wp::extract(var_335, var_332);
                            // dist = wp.dot(vert, dir)                                       <L 202>
                            var_336 = wp::dot(var_334, var_dir);
                            // if dist > max_dist:                                            <L 203>
                            var_337 = (var_336 > var_324);
                            if (var_337) {
                                // max_dist = dist                                            <L 204>
                                var_338 = wp::copy(var_336);
                                // sp.point = vert                                            <L 205>
                                var_0.point = var_334;
                            }
                            var_339 = wp::where(var_337, var_338, var_324);
                            // vert = geom.hfprism[i]                                         <L 201>
                            var_341 = &((var_geom).hfprism);
                            var_343 = wp::load(var_341);
                            var_342 = wp::extract(var_343, var_340);
                            // dist = wp.dot(vert, dir)                                       <L 202>
                            var_344 = wp::dot(var_342, var_dir);
                            // if dist > max_dist:                                            <L 203>
                            var_345 = (var_344 > var_339);
                            if (var_345) {
                                // max_dist = dist                                            <L 204>
                                var_346 = wp::copy(var_344);
                                // sp.point = vert                                            <L 205>
                                var_0.point = var_342;
                            }
                            var_347 = wp::where(var_345, var_346, var_339);
                            // vert = geom.hfprism[i]                                         <L 201>
                            var_349 = &((var_geom).hfprism);
                            var_351 = wp::load(var_349);
                            var_350 = wp::extract(var_351, var_348);
                            // dist = wp.dot(vert, dir)                                       <L 202>
                            var_352 = wp::dot(var_350, var_dir);
                            // if dist > max_dist:                                            <L 203>
                            var_353 = (var_352 > var_347);
                            if (var_353) {
                                // max_dist = dist                                            <L 204>
                                var_354 = wp::copy(var_352);
                                // sp.point = vert                                            <L 205>
                                var_0.point = var_350;
                            }
                            var_355 = wp::where(var_353, var_354, var_347);
                            // vert = geom.hfprism[i]                                         <L 201>
                            var_357 = &((var_geom).hfprism);
                            var_359 = wp::load(var_357);
                            var_358 = wp::extract(var_359, var_356);
                            // dist = wp.dot(vert, dir)                                       <L 202>
                            var_360 = wp::dot(var_358, var_dir);
                            // if dist > max_dist:                                            <L 203>
                            var_361 = (var_360 > var_355);
                            if (var_361) {
                                // max_dist = dist                                            <L 204>
                                var_362 = wp::copy(var_360);
                                // sp.point = vert                                            <L 205>
                                var_0.point = var_358;
                            }
                            var_363 = wp::where(var_361, var_362, var_355);
                            // vert = geom.hfprism[i]                                         <L 201>
                            var_365 = &((var_geom).hfprism);
                            var_367 = wp::load(var_365);
                            var_366 = wp::extract(var_367, var_364);
                            // dist = wp.dot(vert, dir)                                       <L 202>
                            var_368 = wp::dot(var_366, var_dir);
                            // if dist > max_dist:                                            <L 203>
                            var_369 = (var_368 > var_363);
                            if (var_369) {
                                // max_dist = dist                                            <L 204>
                                var_370 = wp::copy(var_368);
                                // sp.point = vert                                            <L 205>
                                var_0.point = var_366;
                            }
                            var_371 = wp::where(var_369, var_370, var_363);
                            // vert = geom.hfprism[i]                                         <L 201>
                            var_373 = &((var_geom).hfprism);
                            var_375 = wp::load(var_373);
                            var_374 = wp::extract(var_375, var_372);
                            // dist = wp.dot(vert, dir)                                       <L 202>
                            var_376 = wp::dot(var_374, var_dir);
                            // if dist > max_dist:                                            <L 203>
                            var_377 = (var_376 > var_371);
                            if (var_377) {
                                // max_dist = dist                                            <L 204>
                                var_378 = wp::copy(var_376);
                                // sp.point = vert                                            <L 205>
                                var_0.point = var_374;
                            }
                            var_379 = wp::where(var_377, var_378, var_371);
                        }
                        var_380 = wp::where(var_323, var_379, var_313);
                        var_381 = wp::where(var_323, var_372, var_187);
                        var_382 = wp::where(var_323, var_374, var_194);
                        var_383 = wp::where(var_323, var_376, var_196);
                        if (!var_323) {
                            // elif geomtype == GeomType.TRIANGLE:                            <L 206>
                            var_385 = (var_geomtype == var_384);
                            if (var_385) {
                                // t1 = geom.rot[0, :]                                        <L 207>
                                var_386 = &((var_geom).rot);
                                var_388 = wp::slice_t(var_389, var_390, var_391);
                                var_393 = wp::load(var_386);
                                var_392 = wp::extract<3>(var_393, var_387, var_388);
                                // t2 = geom.rot[1, :]                                        <L 208>
                                var_394 = &((var_geom).rot);
                                var_396 = wp::slice_t(var_397, var_398, var_399);
                                var_401 = wp::load(var_394);
                                var_400 = wp::extract<3>(var_401, var_395, var_396);
                                // t3 = geom.rot[2, :]                                        <L 209>
                                var_402 = &((var_geom).rot);
                                var_404 = wp::slice_t(var_405, var_406, var_407);
                                var_409 = wp::load(var_402);
                                var_408 = wp::extract<3>(var_409, var_403, var_404);
                                // d1 = wp.dot(t1, dir)                                       <L 210>
                                var_410 = wp::dot(var_392, var_dir);
                                // d2 = wp.dot(t2, dir)                                       <L 211>
                                var_411 = wp::dot(var_400, var_dir);
                                // d3 = wp.dot(t3, dir)                                       <L 212>
                                var_412 = wp::dot(var_408, var_dir);
                                // if d1 > d2 and d1 > d3:                                    <L 213>
                                var_414 = (var_410 > var_411);
                                var_413 = var_414;
                                if (var_413) {
                                    var_415 = (var_410 > var_412);
                                    var_413 = var_413 && var_415;
                                }
                                if (var_413) {
                                    // sp.point = t1                                          <L 214>
                                    var_0.point = var_392;
                                }
                                if (!var_413) {
                                    // elif d2 > d3:                                          <L 215>
                                    var_416 = (var_411 > var_412);
                                    if (var_416) {
                                        // sp.point = t2                                      <L 216>
                                        var_0.point = var_400;
                                    }
                                    if (!var_416) {
                                        // sp.point = t3                                      <L 218>
                                        var_0.point = var_408;
                                    }
                                }
                            }
                        }
                    }
                    var_417 = wp::where(var_150, var_313, var_380);
                    var_418 = wp::where(var_150, var_187, var_381);
                    var_419 = wp::where(var_150, var_194, var_382);
                    var_420 = wp::where(var_150, var_196, var_383);
                }
            }
            var_421 = wp::where(var_85, var_99, var_148);
        }
        var_422 = wp::where(var_62, var_83, var_421);
    }
    var_423 = wp::where(var_23, var_26, var_422);
    // if geom.margin > 0.0:                                                                  <L 220>
    var_424 = &((var_geom).margin);
    var_427 = wp::load(var_424);
    var_426 = (var_427 > var_425);
    if (var_426) {
        // sp.point += dir * (0.5 * geom.margin)                                              <L 221>
        var_429 = &((var_geom).margin);
        var_431 = wp::load(var_429);
        var_430 = wp::mul(var_428, var_431);
        var_432 = wp::mul(var_dir, var_430);
        var_433 = &((var_0).point);
        var_435 = wp::load(var_433);
        var_434 = wp::add(var_435, var_432);
        var_0.point = var_434;
    }
    // return sp                                                                              <L 222>
    return var_0;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:594
static CUDA_CALLABLE void _gjk_support_0(
    Geom_3242f8a8 var_geom1,
    Geom_3242f8a8 var_geom2,
    wp::int32 var_geomtype1,
    wp::int32 var_geomtype2,
    wp::vec_t<3, wp::float32> var_x_k,
    wp::float32 var_x_norm,
    wp::mat_t<4, 3, wp::float32> var_simplex,
    wp::int32 var_n,
    bool var_is_discrete,
    SupportPoint_e82efc60 & ret_0,
    SupportPoint_e82efc60 & ret_1)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    bool var_1;
    const wp::float32 var_2 = 0.0001;
    bool var_3;
    const wp::int32 var_4 = 2;
    bool var_5;
    const wp::int32 var_6 = 1;
    wp::vec_t<3, wp::float32> var_7;
    const wp::int32 var_8 = 0;
    wp::vec_t<3, wp::float32> var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::float32 var_11;
    const wp::float32 var_12 = 1e-30;
    bool var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::vec_t<3, wp::float32> var_16;
    wp::vec_t<3, wp::float32> var_17;
    wp::float32 var_18;
    const wp::float32 var_19 = 1e-15;
    bool var_20;
    wp::vec_t<3, wp::float32> var_21;
    wp::vec_t<3, wp::float32> var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::vec_t<3, wp::float32> var_24;
    const wp::int32 var_25 = 3;
    bool var_26;
    const wp::int32 var_27 = 1;
    wp::vec_t<3, wp::float32> var_28;
    const wp::int32 var_29 = 0;
    wp::vec_t<3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    const wp::int32 var_32 = 2;
    wp::vec_t<3, wp::float32> var_33;
    const wp::int32 var_34 = 0;
    wp::vec_t<3, wp::float32> var_35;
    wp::vec_t<3, wp::float32> var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::float32 var_38;
    bool var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::vec_t<3, wp::float32> var_42;
    wp::vec_t<3, wp::float32> var_43;
    wp::vec_t<3, wp::float32> var_44;
    wp::vec_t<3, wp::float32> var_45;
    wp::vec_t<3, wp::float32> var_46;
    wp::vec_t<3, wp::float32> var_47;
    wp::vec_t<3, wp::float32> var_48;
    SupportPoint_e82efc60 var_49;
    SupportPoint_e82efc60 var_50;
    //---------
    // forward
    // def _gjk_support(                                                                      <L 595>
    // dir_neg = x_k / x_norm                                                                 <L 607>
    var_0 = wp::div(var_x_k, var_x_norm);
    // if is_discrete and x_norm < 1e-4:                                                      <L 610>
    var_1 = var_is_discrete;
    if (var_1) {
        var_3 = (var_x_norm < var_2);
        var_1 = var_1 && var_3;
    }
    if (var_1) {
        // if n == 2:                                                                         <L 611>
        var_5 = (var_n == var_4);
        if (var_5) {
            // edge = simplex[1] - simplex[0]                                                 <L 612>
            var_7 = wp::extract(var_simplex, var_6);
            var_9 = wp::extract(var_simplex, var_8);
            var_10 = wp::sub(var_7, var_9);
            // edge_norm2 = wp.dot(edge, edge)                                                <L 613>
            var_11 = wp::dot(var_10, var_10);
            // if edge_norm2 > MINVAL2:                                                       <L 614>
            var_13 = (var_11 > var_12);
            if (var_13) {
                // proj = wp.dot(dir_neg, edge) / edge_norm2                                  <L 615>
                var_14 = wp::dot(var_0, var_10);
                var_15 = wp::div(var_14, var_11);
                // dir_neg = dir_neg - proj * edge                                            <L 616>
                var_16 = wp::mul(var_15, var_10);
                var_17 = wp::sub(var_0, var_16);
                // dir_norm = wp.length(dir_neg)                                              <L 617>
                var_18 = wp::length(var_17);
                // if dir_norm > MINVAL:                                                      <L 618>
                var_20 = (var_18 > var_19);
                if (var_20) {
                    // dir_neg = dir_neg / dir_norm                                           <L 619>
                    var_21 = wp::div(var_17, var_18);
                }
                var_22 = wp::where(var_20, var_21, var_17);
            }
            var_23 = wp::where(var_13, var_22, var_0);
        }
        var_24 = wp::where(var_5, var_23, var_0);
        if (!var_5) {
            // elif n == 3:                                                                   <L 620>
            var_26 = (var_n == var_25);
            if (var_26) {
                // e1 = simplex[1] - simplex[0]                                               <L 621>
                var_28 = wp::extract(var_simplex, var_27);
                var_30 = wp::extract(var_simplex, var_29);
                var_31 = wp::sub(var_28, var_30);
                // e2 = simplex[2] - simplex[0]                                               <L 622>
                var_33 = wp::extract(var_simplex, var_32);
                var_35 = wp::extract(var_simplex, var_34);
                var_36 = wp::sub(var_33, var_35);
                // normal = wp.cross(e1, e2)                                                  <L 623>
                var_37 = wp::cross(var_31, var_36);
                // normal_norm = wp.length(normal)                                            <L 624>
                var_38 = wp::length(var_37);
                // if normal_norm > MINVAL:                                                   <L 625>
                var_39 = (var_38 > var_19);
                if (var_39) {
                    // dir_neg = wp.sign(wp.dot(dir_neg, normal)) * normal / normal_norm       <L 626>
                    var_40 = wp::dot(var_24, var_37);
                    var_41 = wp::sign(var_40);
                    var_42 = wp::mul(var_41, var_37);
                    var_43 = wp::div(var_42, var_38);
                }
                var_44 = wp::where(var_39, var_43, var_24);
            }
            var_45 = wp::where(var_26, var_44, var_24);
        }
        var_46 = wp::where(var_5, var_24, var_45);
    }
    var_47 = wp::where(var_1, var_46, var_0);
    // sp1 = support(geom1, geomtype1, -dir_neg)                                              <L 628>
    var_48 = wp::neg(var_47);
    var_49 = support_0(var_geom1, var_geomtype1, var_48);
    // sp2 = support(geom2, geomtype2, dir_neg)                                               <L 629>
    var_50 = support_0(var_geom2, var_geomtype2, var_47);
    // return sp1, sp2                                                                        <L 630>
    ret_0 = var_49;
    ret_1 = var_50;
    return;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:292
static CUDA_CALLABLE wp::float32 _det3_0(
    wp::vec_t<3, wp::float32> var_v1,
    wp::vec_t<3, wp::float32> var_v2,
    wp::vec_t<3, wp::float32> var_v3)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::float32 var_1;
    //---------
    // forward
    // def _det3(v1: wp.vec3, v2: wp.vec3, v3: wp.vec3) -> float:                             <L 293>
    // return wp.dot(v1, wp.cross(v2, v3))                                                    <L 294>
    var_0 = wp::cross(var_v2, var_v3);
    var_1 = wp::dot(var_v1, var_0);
    return var_1;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:297
static CUDA_CALLABLE wp::int32 _same_sign_0(
    wp::float32 var_a,
    wp::float32 var_b)
{
    //---------
    // primal vars
    bool var_0;
    const wp::float32 var_1 = 0.0;
    bool var_2;
    const wp::float32 var_3 = 0.0;
    bool var_4;
    const wp::int32 var_5 = 1;
    bool var_6;
    const wp::float32 var_7 = 0.0;
    bool var_8;
    const wp::float32 var_9 = 0.0;
    bool var_10;
    const wp::int32 var_11 = -1;
    const wp::int32 var_12 = 0;
    //---------
    // forward
    // def _same_sign(a: float, b: float) -> int:                                             <L 298>
    // if a > 0.0 and b > 0.0:                                                                <L 299>
    var_2 = (var_a > var_1);
    var_0 = var_2;
    if (var_0) {
        var_4 = (var_b > var_3);
        var_0 = var_0 && var_4;
    }
    if (var_0) {
        // return 1                                                                           <L 300>
        return var_5;
    }
    // if a < 0.0 and b < 0.0:                                                                <L 301>
    var_8 = (var_a < var_7);
    var_6 = var_8;
    if (var_6) {
        var_10 = (var_b < var_9);
        var_6 = var_6 && var_10;
    }
    if (var_6) {
        // return -1                                                                          <L 302>
        return var_11;
    }
    // return 0                                                                               <L 303>
    return var_12;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:313
static CUDA_CALLABLE void _project_origin_plane_0(
    wp::vec_t<3, wp::float32> var_v1,
    wp::vec_t<3, wp::float32> var_v2,
    wp::vec_t<3, wp::float32> var_v3,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::int32 & ret_1)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    const wp::float32 var_8 = 0.0;
    bool var_9;
    const wp::int32 var_10 = 1;
    bool var_11;
    const wp::float32 var_12 = 0.0;
    bool var_13;
    const wp::float32 var_14 = 1e-15;
    bool var_15;
    wp::float32 var_16;
    wp::vec_t<3, wp::float32> var_17;
    const wp::int32 var_18 = 0;
    wp::vec_t<3, wp::float32> var_19;
    wp::float32 var_20;
    wp::float32 var_21;
    const wp::float32 var_22 = 0.0;
    bool var_23;
    const wp::int32 var_24 = 1;
    bool var_25;
    const wp::float32 var_26 = 0.0;
    bool var_27;
    bool var_28;
    wp::float32 var_29;
    wp::vec_t<3, wp::float32> var_30;
    const wp::int32 var_31 = 0;
    wp::vec_t<3, wp::float32> var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    wp::float32 var_35;
    wp::vec_t<3, wp::float32> var_36;
    const wp::int32 var_37 = 0;
    //---------
    // forward
    // def _project_origin_plane(v1: wp.vec3, v2: wp.vec3, v3: wp.vec3) -> Tuple[wp.vec3, int]:       <L 314>
    // z = wp.vec3(0.0)                                                                       <L 315>
    var_1 = wp::vec_t<3, wp::float32>(var_0);
    // diff21 = v2 - v1                                                                       <L 316>
    var_2 = wp::sub(var_v2, var_v1);
    // diff31 = v3 - v1                                                                       <L 317>
    var_3 = wp::sub(var_v3, var_v1);
    // diff32 = v3 - v2                                                                       <L 318>
    var_4 = wp::sub(var_v3, var_v2);
    // n = wp.cross(diff32, diff21)                                                           <L 321>
    var_5 = wp::cross(var_4, var_2);
    // nv = wp.dot(n, v2)                                                                     <L 322>
    var_6 = wp::dot(var_5, var_v2);
    // nn = wp.dot(n, n)                                                                      <L 323>
    var_7 = wp::dot(var_5, var_5);
    // if nn == 0.0:                                                                          <L 324>
    var_9 = (var_7 == var_8);
    if (var_9) {
        // return z, 1                                                                        <L 325>
        ret_0 = var_1;
        ret_1 = var_10;
        return;
    }
    // if nv != 0.0 and nn > MINVAL:                                                          <L 326>
    var_13 = (var_6 != var_12);
    var_11 = var_13;
    if (var_11) {
        var_15 = (var_7 > var_14);
        var_11 = var_11 && var_15;
    }
    if (var_11) {
        // return (nv / nn) * n, 0                                                            <L 327>
        var_16 = wp::div(var_6, var_7);
        var_17 = wp::mul(var_16, var_5);
        ret_0 = var_17;
        ret_1 = var_18;
        return;
    }
    // n = wp.cross(diff21, diff31)                                                           <L 330>
    var_19 = wp::cross(var_2, var_3);
    // nv = wp.dot(n, v1)                                                                     <L 331>
    var_20 = wp::dot(var_19, var_v1);
    // nn = wp.dot(n, n)                                                                      <L 332>
    var_21 = wp::dot(var_19, var_19);
    // if nn == 0.0:                                                                          <L 333>
    var_23 = (var_21 == var_22);
    if (var_23) {
        // return z, 1                                                                        <L 334>
        ret_0 = var_1;
        ret_1 = var_24;
        return;
    }
    // if nv != 0.0 and nn > MINVAL:                                                          <L 335>
    var_27 = (var_20 != var_26);
    var_25 = var_27;
    if (var_25) {
        var_28 = (var_21 > var_14);
        var_25 = var_25 && var_28;
    }
    if (var_25) {
        // return (nv / nn) * n, 0                                                            <L 336>
        var_29 = wp::div(var_20, var_21);
        var_30 = wp::mul(var_29, var_19);
        ret_0 = var_30;
        ret_1 = var_31;
        return;
    }
    // n = wp.cross(diff31, diff32)                                                           <L 339>
    var_32 = wp::cross(var_3, var_4);
    // nv = wp.dot(n, v3)                                                                     <L 340>
    var_33 = wp::dot(var_32, var_v3);
    // nn = wp.dot(n, n)                                                                      <L 341>
    var_34 = wp::dot(var_32, var_32);
    // return (nv / nn) * n, 0                                                                <L 342>
    var_35 = wp::div(var_33, var_34);
    var_36 = wp::mul(var_35, var_32);
    ret_0 = var_36;
    ret_1 = var_37;
    return;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:306
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _project_origin_line_0(
    wp::vec_t<3, wp::float32> var_v1,
    wp::vec_t<3, wp::float32> var_v2)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32> var_6;
    //---------
    // forward
    // def _project_origin_line(v1: wp.vec3, v2: wp.vec3) -> wp.vec3:                         <L 307>
    // diff = v2 - v1                                                                         <L 308>
    var_0 = wp::sub(var_v2, var_v1);
    // scl = -(wp.dot(v2, diff) / wp.dot(diff, diff))                                         <L 309>
    var_1 = wp::dot(var_v2, var_0);
    var_2 = wp::dot(var_0, var_0);
    var_3 = wp::div(var_1, var_2);
    var_4 = wp::neg(var_3);
    // return v2 + scl * diff                                                                 <L 310>
    var_5 = wp::mul(var_4, var_0);
    var_6 = wp::add(var_v2, var_5);
    return var_6;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:566
static CUDA_CALLABLE wp::vec_t<2, wp::float32> _S1D_0(
    wp::vec_t<3, wp::float32> var_s1,
    wp::vec_t<3, wp::float32> var_s2)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    const wp::int32 var_1 = 0;
    wp::float32 var_2;
    const wp::int32 var_3 = 0;
    wp::float32 var_4;
    wp::float32 var_5;
    const wp::int32 var_6 = 0;
    const wp::int32 var_7 = 1;
    wp::float32 var_8;
    const wp::int32 var_9 = 1;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    bool var_14;
    wp::float32 var_15;
    const wp::int32 var_16 = 1;
    wp::float32 var_17;
    wp::int32 var_18;
    const wp::int32 var_19 = 2;
    wp::float32 var_20;
    const wp::int32 var_21 = 2;
    wp::float32 var_22;
    wp::float32 var_23;
    wp::float32 var_24;
    wp::float32 var_25;
    bool var_26;
    wp::float32 var_27;
    const wp::int32 var_28 = 2;
    wp::float32 var_29;
    wp::int32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    wp::float32 var_35;
    wp::float32 var_36;
    bool var_37;
    wp::int32 var_38;
    wp::int32 var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::vec_t<2, wp::float32> var_42;
    const wp::float32 var_43 = 0.0;
    const wp::float32 var_44 = 1.0;
    wp::vec_t<2, wp::float32> var_45;
    //---------
    // forward
    // def _S1D(s1: wp.vec3, s2: wp.vec3) -> wp.vec2:                                         <L 567>
    // p_o = _project_origin_line(s1, s2)                                                     <L 569>
    var_0 = _project_origin_line_0(var_s1, var_s2);
    // mu_max = s1[0] - s2[0]                                                                 <L 572>
    var_2 = wp::extract(var_s1, var_1);
    var_4 = wp::extract(var_s2, var_3);
    var_5 = wp::sub(var_2, var_4);
    // index = 0                                                                              <L 573>
    // mu = s1[1] - s2[1]                                                                     <L 575>
    var_8 = wp::extract(var_s1, var_7);
    var_10 = wp::extract(var_s2, var_9);
    var_11 = wp::sub(var_8, var_10);
    // if wp.abs(mu) >= wp.abs(mu_max):                                                       <L 576>
    var_12 = wp::abs(var_11);
    var_13 = wp::abs(var_5);
    var_14 = (var_12 >= var_13);
    if (var_14) {
        // mu_max = mu                                                                        <L 577>
        var_15 = wp::copy(var_11);
        // index = 1                                                                          <L 578>
    }
    var_17 = wp::where(var_14, var_15, var_5);
    var_18 = wp::where(var_14, var_16, var_6);
    // mu = s1[2] - s2[2]                                                                     <L 580>
    var_20 = wp::extract(var_s1, var_19);
    var_22 = wp::extract(var_s2, var_21);
    var_23 = wp::sub(var_20, var_22);
    // if wp.abs(mu) >= wp.abs(mu_max):                                                       <L 581>
    var_24 = wp::abs(var_23);
    var_25 = wp::abs(var_17);
    var_26 = (var_24 >= var_25);
    if (var_26) {
        // mu_max = mu                                                                        <L 582>
        var_27 = wp::copy(var_23);
        // index = 2                                                                          <L 583>
    }
    var_29 = wp::where(var_26, var_27, var_17);
    var_30 = wp::where(var_26, var_28, var_18);
    // C1 = p_o[index] - s2[index]                                                            <L 585>
    var_31 = wp::extract(var_0, var_30);
    var_32 = wp::extract(var_s2, var_30);
    var_33 = wp::sub(var_31, var_32);
    // C2 = s1[index] - p_o[index]                                                            <L 586>
    var_34 = wp::extract(var_s1, var_30);
    var_35 = wp::extract(var_0, var_30);
    var_36 = wp::sub(var_34, var_35);
    // if _same_sign(mu_max, C1) and _same_sign(mu_max, C2):                                  <L 589>
    var_38 = _same_sign_0(var_29, var_33);
    var_37 = var_38;
    if (var_37) {
        var_39 = _same_sign_0(var_29, var_36);
        var_37 = var_37 && var_39;
    }
    if (var_37) {
        // return wp.vec2(C1 / mu_max, C2 / mu_max)                                           <L 590>
        var_40 = wp::div(var_33, var_29);
        var_41 = wp::div(var_36, var_29);
        var_42 = wp::vec_t<2, wp::float32>(var_40, var_41);
        return var_42;
    }
    // return wp.vec2(0.0, 1.0)                                                               <L 591>
    var_45 = wp::vec_t<2, wp::float32>(var_43, var_44);
    return var_45;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:421
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _S2D_0(
    wp::vec_t<3, wp::float32> var_s1,
    wp::vec_t<3, wp::float32> var_s2,
    wp::vec_t<3, wp::float32> var_s3)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::int32 var_1;
    wp::vec_t<2, wp::float32> var_2;
    const wp::int32 var_3 = 0;
    wp::float32 var_4;
    const wp::int32 var_5 = 1;
    wp::float32 var_6;
    const wp::float32 var_7 = 0.0;
    wp::vec_t<3, wp::float32> var_8;
    const wp::int32 var_9 = 1;
    wp::float32 var_10;
    const wp::int32 var_11 = 2;
    wp::float32 var_12;
    wp::float32 var_13;
    const wp::int32 var_14 = 2;
    wp::float32 var_15;
    const wp::int32 var_16 = 1;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    const wp::int32 var_20 = 1;
    wp::float32 var_21;
    const wp::int32 var_22 = 2;
    wp::float32 var_23;
    wp::float32 var_24;
    wp::float32 var_25;
    const wp::int32 var_26 = 2;
    wp::float32 var_27;
    const wp::int32 var_28 = 1;
    wp::float32 var_29;
    wp::float32 var_30;
    wp::float32 var_31;
    const wp::int32 var_32 = 1;
    wp::float32 var_33;
    const wp::int32 var_34 = 2;
    wp::float32 var_35;
    wp::float32 var_36;
    wp::float32 var_37;
    const wp::int32 var_38 = 2;
    wp::float32 var_39;
    const wp::int32 var_40 = 1;
    wp::float32 var_41;
    wp::float32 var_42;
    wp::float32 var_43;
    const wp::int32 var_44 = 0;
    wp::float32 var_45;
    const wp::int32 var_46 = 2;
    wp::float32 var_47;
    wp::float32 var_48;
    const wp::int32 var_49 = 2;
    wp::float32 var_50;
    const wp::int32 var_51 = 0;
    wp::float32 var_52;
    wp::float32 var_53;
    wp::float32 var_54;
    const wp::int32 var_55 = 0;
    wp::float32 var_56;
    const wp::int32 var_57 = 2;
    wp::float32 var_58;
    wp::float32 var_59;
    wp::float32 var_60;
    const wp::int32 var_61 = 2;
    wp::float32 var_62;
    const wp::int32 var_63 = 0;
    wp::float32 var_64;
    wp::float32 var_65;
    wp::float32 var_66;
    const wp::int32 var_67 = 0;
    wp::float32 var_68;
    const wp::int32 var_69 = 2;
    wp::float32 var_70;
    wp::float32 var_71;
    wp::float32 var_72;
    const wp::int32 var_73 = 2;
    wp::float32 var_74;
    const wp::int32 var_75 = 0;
    wp::float32 var_76;
    wp::float32 var_77;
    wp::float32 var_78;
    const wp::int32 var_79 = 0;
    wp::float32 var_80;
    const wp::int32 var_81 = 1;
    wp::float32 var_82;
    wp::float32 var_83;
    const wp::int32 var_84 = 1;
    wp::float32 var_85;
    const wp::int32 var_86 = 0;
    wp::float32 var_87;
    wp::float32 var_88;
    wp::float32 var_89;
    const wp::int32 var_90 = 0;
    wp::float32 var_91;
    const wp::int32 var_92 = 1;
    wp::float32 var_93;
    wp::float32 var_94;
    wp::float32 var_95;
    const wp::int32 var_96 = 1;
    wp::float32 var_97;
    const wp::int32 var_98 = 0;
    wp::float32 var_99;
    wp::float32 var_100;
    wp::float32 var_101;
    const wp::int32 var_102 = 0;
    wp::float32 var_103;
    const wp::int32 var_104 = 1;
    wp::float32 var_105;
    wp::float32 var_106;
    wp::float32 var_107;
    const wp::int32 var_108 = 1;
    wp::float32 var_109;
    const wp::int32 var_110 = 0;
    wp::float32 var_111;
    wp::float32 var_112;
    wp::float32 var_113;
    const wp::float32 var_114 = 0.0;
    const wp::float32 var_115 = 0.0;
    wp::vec_t<2, wp::float32> var_116;
    const wp::float32 var_117 = 0.0;
    wp::vec_t<2, wp::float32> var_118;
    const wp::float32 var_119 = 0.0;
    wp::vec_t<2, wp::float32> var_120;
    const wp::float32 var_121 = 0.0;
    wp::vec_t<2, wp::float32> var_122;
    wp::float32 var_123;
    wp::float32 var_124;
    wp::float32 var_125;
    bool var_126;
    bool var_127;
    bool var_128;
    wp::float32 var_129;
    const wp::int32 var_130 = 1;
    wp::float32 var_131;
    const wp::int32 var_132 = 0;
    const wp::int32 var_133 = 2;
    wp::float32 var_134;
    const wp::int32 var_135 = 1;
    const wp::int32 var_136 = 1;
    wp::float32 var_137;
    const wp::int32 var_138 = 0;
    const wp::int32 var_139 = 2;
    wp::float32 var_140;
    const wp::int32 var_141 = 1;
    const wp::int32 var_142 = 1;
    wp::float32 var_143;
    const wp::int32 var_144 = 0;
    const wp::int32 var_145 = 2;
    wp::float32 var_146;
    const wp::int32 var_147 = 1;
    const wp::int32 var_148 = 1;
    wp::float32 var_149;
    const wp::int32 var_150 = 0;
    const wp::int32 var_151 = 2;
    wp::float32 var_152;
    const wp::int32 var_153 = 1;
    wp::float32 var_154;
    bool var_155;
    wp::float32 var_156;
    const wp::int32 var_157 = 0;
    wp::float32 var_158;
    const wp::int32 var_159 = 0;
    const wp::int32 var_160 = 2;
    wp::float32 var_161;
    const wp::int32 var_162 = 1;
    const wp::int32 var_163 = 0;
    wp::float32 var_164;
    const wp::int32 var_165 = 0;
    const wp::int32 var_166 = 2;
    wp::float32 var_167;
    const wp::int32 var_168 = 1;
    const wp::int32 var_169 = 0;
    wp::float32 var_170;
    const wp::int32 var_171 = 0;
    const wp::int32 var_172 = 2;
    wp::float32 var_173;
    const wp::int32 var_174 = 1;
    const wp::int32 var_175 = 0;
    wp::float32 var_176;
    const wp::int32 var_177 = 0;
    const wp::int32 var_178 = 2;
    wp::float32 var_179;
    const wp::int32 var_180 = 1;
    wp::float32 var_181;
    wp::float32 var_182;
    const wp::int32 var_183 = 0;
    wp::float32 var_184;
    const wp::int32 var_185 = 0;
    const wp::int32 var_186 = 1;
    wp::float32 var_187;
    const wp::int32 var_188 = 1;
    const wp::int32 var_189 = 0;
    wp::float32 var_190;
    const wp::int32 var_191 = 0;
    const wp::int32 var_192 = 1;
    wp::float32 var_193;
    const wp::int32 var_194 = 1;
    const wp::int32 var_195 = 0;
    wp::float32 var_196;
    const wp::int32 var_197 = 0;
    const wp::int32 var_198 = 1;
    wp::float32 var_199;
    const wp::int32 var_200 = 1;
    const wp::int32 var_201 = 0;
    wp::float32 var_202;
    const wp::int32 var_203 = 0;
    const wp::int32 var_204 = 1;
    wp::float32 var_205;
    const wp::int32 var_206 = 1;
    wp::float32 var_207;
    wp::float32 var_208;
    const wp::int32 var_209 = 0;
    wp::float32 var_210;
    const wp::int32 var_211 = 1;
    wp::float32 var_212;
    wp::float32 var_213;
    const wp::int32 var_214 = 1;
    wp::float32 var_215;
    const wp::int32 var_216 = 0;
    wp::float32 var_217;
    wp::float32 var_218;
    wp::float32 var_219;
    const wp::int32 var_220 = 0;
    wp::float32 var_221;
    const wp::int32 var_222 = 1;
    wp::float32 var_223;
    wp::float32 var_224;
    wp::float32 var_225;
    const wp::int32 var_226 = 0;
    wp::float32 var_227;
    const wp::int32 var_228 = 1;
    wp::float32 var_229;
    wp::float32 var_230;
    wp::float32 var_231;
    const wp::int32 var_232 = 1;
    wp::float32 var_233;
    const wp::int32 var_234 = 0;
    wp::float32 var_235;
    wp::float32 var_236;
    wp::float32 var_237;
    const wp::int32 var_238 = 0;
    wp::float32 var_239;
    const wp::int32 var_240 = 1;
    wp::float32 var_241;
    wp::float32 var_242;
    wp::float32 var_243;
    const wp::int32 var_244 = 0;
    wp::float32 var_245;
    const wp::int32 var_246 = 1;
    wp::float32 var_247;
    wp::float32 var_248;
    const wp::int32 var_249 = 1;
    wp::float32 var_250;
    const wp::int32 var_251 = 0;
    wp::float32 var_252;
    wp::float32 var_253;
    wp::float32 var_254;
    const wp::int32 var_255 = 0;
    wp::float32 var_256;
    const wp::int32 var_257 = 1;
    wp::float32 var_258;
    wp::float32 var_259;
    wp::float32 var_260;
    const wp::int32 var_261 = 0;
    wp::float32 var_262;
    const wp::int32 var_263 = 1;
    wp::float32 var_264;
    wp::float32 var_265;
    wp::float32 var_266;
    const wp::int32 var_267 = 1;
    wp::float32 var_268;
    const wp::int32 var_269 = 0;
    wp::float32 var_270;
    wp::float32 var_271;
    wp::float32 var_272;
    const wp::int32 var_273 = 0;
    wp::float32 var_274;
    const wp::int32 var_275 = 1;
    wp::float32 var_276;
    wp::float32 var_277;
    wp::float32 var_278;
    const wp::int32 var_279 = 0;
    wp::float32 var_280;
    const wp::int32 var_281 = 1;
    wp::float32 var_282;
    wp::float32 var_283;
    const wp::int32 var_284 = 1;
    wp::float32 var_285;
    const wp::int32 var_286 = 0;
    wp::float32 var_287;
    wp::float32 var_288;
    wp::float32 var_289;
    const wp::int32 var_290 = 0;
    wp::float32 var_291;
    const wp::int32 var_292 = 1;
    wp::float32 var_293;
    wp::float32 var_294;
    wp::float32 var_295;
    const wp::int32 var_296 = 0;
    wp::float32 var_297;
    const wp::int32 var_298 = 1;
    wp::float32 var_299;
    wp::float32 var_300;
    wp::float32 var_301;
    const wp::int32 var_302 = 1;
    wp::float32 var_303;
    const wp::int32 var_304 = 0;
    wp::float32 var_305;
    wp::float32 var_306;
    wp::float32 var_307;
    const wp::int32 var_308 = 0;
    wp::float32 var_309;
    const wp::int32 var_310 = 1;
    wp::float32 var_311;
    wp::float32 var_312;
    wp::float32 var_313;
    wp::int32 var_314;
    wp::int32 var_315;
    wp::int32 var_316;
    bool var_317;
    wp::float32 var_318;
    wp::float32 var_319;
    wp::float32 var_320;
    wp::vec_t<3, wp::float32> var_321;
    const wp::float32 var_322 = 1e+30;
    wp::float32 var_323;
    const wp::float32 var_324 = 0.0;
    const wp::float32 var_325 = 0.0;
    const wp::float32 var_326 = 0.0;
    wp::vec_t<3, wp::float32> var_327;
    bool var_328;
    wp::vec_t<2, wp::float32> var_329;
    const wp::int32 var_330 = 0;
    wp::float32 var_331;
    wp::vec_t<3, wp::float32> var_332;
    const wp::int32 var_333 = 1;
    wp::float32 var_334;
    wp::vec_t<3, wp::float32> var_335;
    wp::vec_t<3, wp::float32> var_336;
    wp::float32 var_337;
    const wp::float32 var_338 = 0.0;
    const wp::int32 var_339 = 0;
    const wp::int32 var_340 = 0;
    wp::float32 var_341;
    const wp::int32 var_342 = 1;
    const wp::int32 var_343 = 1;
    wp::float32 var_344;
    const wp::int32 var_345 = 2;
    wp::float32 var_346;
    wp::float32 var_347;
    bool var_348;
    wp::vec_t<2, wp::float32> var_349;
    const wp::int32 var_350 = 0;
    wp::float32 var_351;
    wp::vec_t<3, wp::float32> var_352;
    const wp::int32 var_353 = 1;
    wp::float32 var_354;
    wp::vec_t<3, wp::float32> var_355;
    wp::vec_t<3, wp::float32> var_356;
    wp::float32 var_357;
    bool var_358;
    const wp::int32 var_359 = 0;
    wp::float32 var_360;
    const wp::int32 var_361 = 0;
    const wp::float32 var_362 = 0.0;
    const wp::int32 var_363 = 1;
    const wp::int32 var_364 = 1;
    wp::float32 var_365;
    const wp::int32 var_366 = 2;
    wp::float32 var_367;
    wp::float32 var_368;
    wp::float32 var_369;
    wp::vec_t<2, wp::float32> var_370;
    wp::vec_t<3, wp::float32> var_371;
    wp::float32 var_372;
    bool var_373;
    wp::vec_t<2, wp::float32> var_374;
    const wp::int32 var_375 = 0;
    wp::float32 var_376;
    wp::vec_t<3, wp::float32> var_377;
    const wp::int32 var_378 = 1;
    wp::float32 var_379;
    wp::vec_t<3, wp::float32> var_380;
    wp::vec_t<3, wp::float32> var_381;
    wp::float32 var_382;
    bool var_383;
    const wp::int32 var_384 = 0;
    wp::float32 var_385;
    const wp::int32 var_386 = 0;
    const wp::int32 var_387 = 1;
    wp::float32 var_388;
    const wp::int32 var_389 = 1;
    const wp::float32 var_390 = 0.0;
    const wp::int32 var_391 = 2;
    wp::vec_t<2, wp::float32> var_392;
    wp::vec_t<3, wp::float32> var_393;
    wp::float32 var_394;
    //---------
    // forward
    // def _S2D(s1: wp.vec3, s2: wp.vec3, s3: wp.vec3) -> wp.vec3:                            <L 422>
    // p_o, ret = _project_origin_plane(s1, s2, s3)                                           <L 424>
    _project_origin_plane_0(var_s1, var_s2, var_s3, var_0, var_1);
    // if ret:                                                                                <L 425>
    if (var_1) {
        // v = _S1D(s1, s2)                                                                   <L 426>
        var_2 = _S1D_0(var_s1, var_s2);
        // return wp.vec3(v[0], v[1], 0.0)                                                    <L 427>
        var_4 = wp::extract(var_2, var_3);
        var_6 = wp::extract(var_2, var_5);
        var_8 = wp::vec_t<3, wp::float32>(var_4, var_6, var_7);
        return var_8;
    }
    // M_14 = s2[1] * s3[2] - s2[2] * s3[1] - s1[1] * s3[2] + s1[2] * s3[1] + s1[1] * s2[2] - s1[2] * s2[1]       <L 434>
    var_10 = wp::extract(var_s2, var_9);
    var_12 = wp::extract(var_s3, var_11);
    var_13 = wp::mul(var_10, var_12);
    var_15 = wp::extract(var_s2, var_14);
    var_17 = wp::extract(var_s3, var_16);
    var_18 = wp::mul(var_15, var_17);
    var_19 = wp::sub(var_13, var_18);
    var_21 = wp::extract(var_s1, var_20);
    var_23 = wp::extract(var_s3, var_22);
    var_24 = wp::mul(var_21, var_23);
    var_25 = wp::sub(var_19, var_24);
    var_27 = wp::extract(var_s1, var_26);
    var_29 = wp::extract(var_s3, var_28);
    var_30 = wp::mul(var_27, var_29);
    var_31 = wp::add(var_25, var_30);
    var_33 = wp::extract(var_s1, var_32);
    var_35 = wp::extract(var_s2, var_34);
    var_36 = wp::mul(var_33, var_35);
    var_37 = wp::add(var_31, var_36);
    var_39 = wp::extract(var_s1, var_38);
    var_41 = wp::extract(var_s2, var_40);
    var_42 = wp::mul(var_39, var_41);
    var_43 = wp::sub(var_37, var_42);
    // M_24 = s2[0] * s3[2] - s2[2] * s3[0] - s1[0] * s3[2] + s1[2] * s3[0] + s1[0] * s2[2] - s1[2] * s2[0]       <L 435>
    var_45 = wp::extract(var_s2, var_44);
    var_47 = wp::extract(var_s3, var_46);
    var_48 = wp::mul(var_45, var_47);
    var_50 = wp::extract(var_s2, var_49);
    var_52 = wp::extract(var_s3, var_51);
    var_53 = wp::mul(var_50, var_52);
    var_54 = wp::sub(var_48, var_53);
    var_56 = wp::extract(var_s1, var_55);
    var_58 = wp::extract(var_s3, var_57);
    var_59 = wp::mul(var_56, var_58);
    var_60 = wp::sub(var_54, var_59);
    var_62 = wp::extract(var_s1, var_61);
    var_64 = wp::extract(var_s3, var_63);
    var_65 = wp::mul(var_62, var_64);
    var_66 = wp::add(var_60, var_65);
    var_68 = wp::extract(var_s1, var_67);
    var_70 = wp::extract(var_s2, var_69);
    var_71 = wp::mul(var_68, var_70);
    var_72 = wp::add(var_66, var_71);
    var_74 = wp::extract(var_s1, var_73);
    var_76 = wp::extract(var_s2, var_75);
    var_77 = wp::mul(var_74, var_76);
    var_78 = wp::sub(var_72, var_77);
    // M_34 = s2[0] * s3[1] - s2[1] * s3[0] - s1[0] * s3[1] + s1[1] * s3[0] + s1[0] * s2[1] - s1[1] * s2[0]       <L 436>
    var_80 = wp::extract(var_s2, var_79);
    var_82 = wp::extract(var_s3, var_81);
    var_83 = wp::mul(var_80, var_82);
    var_85 = wp::extract(var_s2, var_84);
    var_87 = wp::extract(var_s3, var_86);
    var_88 = wp::mul(var_85, var_87);
    var_89 = wp::sub(var_83, var_88);
    var_91 = wp::extract(var_s1, var_90);
    var_93 = wp::extract(var_s3, var_92);
    var_94 = wp::mul(var_91, var_93);
    var_95 = wp::sub(var_89, var_94);
    var_97 = wp::extract(var_s1, var_96);
    var_99 = wp::extract(var_s3, var_98);
    var_100 = wp::mul(var_97, var_99);
    var_101 = wp::add(var_95, var_100);
    var_103 = wp::extract(var_s1, var_102);
    var_105 = wp::extract(var_s2, var_104);
    var_106 = wp::mul(var_103, var_105);
    var_107 = wp::add(var_101, var_106);
    var_109 = wp::extract(var_s1, var_108);
    var_111 = wp::extract(var_s2, var_110);
    var_112 = wp::mul(var_109, var_111);
    var_113 = wp::sub(var_107, var_112);
    // M_max = 0.0                                                                            <L 439>
    // s1_2D = wp.vec2(0.0)                                                                   <L 440>
    var_116 = wp::vec_t<2, wp::float32>(var_115);
    // s2_2D = wp.vec2(0.0)                                                                   <L 441>
    var_118 = wp::vec_t<2, wp::float32>(var_117);
    // s3_2D = wp.vec2(0.0)                                                                   <L 442>
    var_120 = wp::vec_t<2, wp::float32>(var_119);
    // p_o_2D = wp.vec2(0.0)                                                                  <L 443>
    var_122 = wp::vec_t<2, wp::float32>(var_121);
    // mu1 = wp.abs(M_14)                                                                     <L 445>
    var_123 = wp::abs(var_43);
    // mu2 = wp.abs(M_24)                                                                     <L 446>
    var_124 = wp::abs(var_78);
    // mu3 = wp.abs(M_34)                                                                     <L 447>
    var_125 = wp::abs(var_113);
    // if mu1 >= mu2 and mu1 >= mu3:                                                          <L 449>
    var_127 = (var_123 >= var_124);
    var_126 = var_127;
    if (var_126) {
        var_128 = (var_123 >= var_125);
        var_126 = var_126 && var_128;
    }
    if (var_126) {
        // M_max = M_14                                                                       <L 450>
        var_129 = wp::copy(var_43);
        // s1_2D[0] = s1[1]                                                                   <L 451>
        var_131 = wp::extract(var_s1, var_130);
        wp::assign_inplace(var_116, var_132, var_131);
        // s1_2D[1] = s1[2]                                                                   <L 452>
        var_134 = wp::extract(var_s1, var_133);
        wp::assign_inplace(var_116, var_135, var_134);
        // s2_2D[0] = s2[1]                                                                   <L 454>
        var_137 = wp::extract(var_s2, var_136);
        wp::assign_inplace(var_118, var_138, var_137);
        // s2_2D[1] = s2[2]                                                                   <L 455>
        var_140 = wp::extract(var_s2, var_139);
        wp::assign_inplace(var_118, var_141, var_140);
        // s3_2D[0] = s3[1]                                                                   <L 457>
        var_143 = wp::extract(var_s3, var_142);
        wp::assign_inplace(var_120, var_144, var_143);
        // s3_2D[1] = s3[2]                                                                   <L 458>
        var_146 = wp::extract(var_s3, var_145);
        wp::assign_inplace(var_120, var_147, var_146);
        // p_o_2D[0] = p_o[1]                                                                 <L 460>
        var_149 = wp::extract(var_0, var_148);
        wp::assign_inplace(var_122, var_150, var_149);
        // p_o_2D[1] = p_o[2]                                                                 <L 461>
        var_152 = wp::extract(var_0, var_151);
        wp::assign_inplace(var_122, var_153, var_152);
    }
    var_154 = wp::where(var_126, var_129, var_114);
    if (!var_126) {
        // elif mu2 >= mu3:                                                                   <L 462>
        var_155 = (var_124 >= var_125);
        if (var_155) {
            // M_max = M_24                                                                   <L 463>
            var_156 = wp::copy(var_78);
            // s1_2D[0] = s1[0]                                                               <L 464>
            var_158 = wp::extract(var_s1, var_157);
            wp::assign_inplace(var_116, var_159, var_158);
            // s1_2D[1] = s1[2]                                                               <L 465>
            var_161 = wp::extract(var_s1, var_160);
            wp::assign_inplace(var_116, var_162, var_161);
            // s2_2D[0] = s2[0]                                                               <L 467>
            var_164 = wp::extract(var_s2, var_163);
            wp::assign_inplace(var_118, var_165, var_164);
            // s2_2D[1] = s2[2]                                                               <L 468>
            var_167 = wp::extract(var_s2, var_166);
            wp::assign_inplace(var_118, var_168, var_167);
            // s3_2D[0] = s3[0]                                                               <L 470>
            var_170 = wp::extract(var_s3, var_169);
            wp::assign_inplace(var_120, var_171, var_170);
            // s3_2D[1] = s3[2]                                                               <L 471>
            var_173 = wp::extract(var_s3, var_172);
            wp::assign_inplace(var_120, var_174, var_173);
            // p_o_2D[0] = p_o[0]                                                             <L 473>
            var_176 = wp::extract(var_0, var_175);
            wp::assign_inplace(var_122, var_177, var_176);
            // p_o_2D[1] = p_o[2]                                                             <L 474>
            var_179 = wp::extract(var_0, var_178);
            wp::assign_inplace(var_122, var_180, var_179);
        }
        var_181 = wp::where(var_155, var_156, var_154);
        if (!var_155) {
            // M_max = M_34                                                                   <L 476>
            var_182 = wp::copy(var_113);
            // s1_2D[0] = s1[0]                                                               <L 477>
            var_184 = wp::extract(var_s1, var_183);
            wp::assign_inplace(var_116, var_185, var_184);
            // s1_2D[1] = s1[1]                                                               <L 478>
            var_187 = wp::extract(var_s1, var_186);
            wp::assign_inplace(var_116, var_188, var_187);
            // s2_2D[0] = s2[0]                                                               <L 480>
            var_190 = wp::extract(var_s2, var_189);
            wp::assign_inplace(var_118, var_191, var_190);
            // s2_2D[1] = s2[1]                                                               <L 481>
            var_193 = wp::extract(var_s2, var_192);
            wp::assign_inplace(var_118, var_194, var_193);
            // s3_2D[0] = s3[0]                                                               <L 483>
            var_196 = wp::extract(var_s3, var_195);
            wp::assign_inplace(var_120, var_197, var_196);
            // s3_2D[1] = s3[1]                                                               <L 484>
            var_199 = wp::extract(var_s3, var_198);
            wp::assign_inplace(var_120, var_200, var_199);
            // p_o_2D[0] = p_o[0]                                                             <L 486>
            var_202 = wp::extract(var_0, var_201);
            wp::assign_inplace(var_122, var_203, var_202);
            // p_o_2D[1] = p_o[1]                                                             <L 487>
            var_205 = wp::extract(var_0, var_204);
            wp::assign_inplace(var_122, var_206, var_205);
        }
        var_207 = wp::where(var_155, var_181, var_182);
    }
    var_208 = wp::where(var_126, var_154, var_207);
    // C31 = (                                                                                <L 495>
    // p_o_2D[0] * s2_2D[1]                                                                   <L 496>
    var_210 = wp::extract(var_122, var_209);
    var_212 = wp::extract(var_118, var_211);
    var_213 = wp::mul(var_210, var_212);
    // + p_o_2D[1] * s3_2D[0]                                                                 <L 497>
    var_215 = wp::extract(var_122, var_214);
    var_217 = wp::extract(var_120, var_216);
    var_218 = wp::mul(var_215, var_217);
    var_219 = wp::add(var_213, var_218);
    // + s2_2D[0] * s3_2D[1]                                                                  <L 498>
    var_221 = wp::extract(var_118, var_220);
    var_223 = wp::extract(var_120, var_222);
    var_224 = wp::mul(var_221, var_223);
    var_225 = wp::add(var_219, var_224);
    // - p_o_2D[0] * s3_2D[1]                                                                 <L 499>
    var_227 = wp::extract(var_122, var_226);
    var_229 = wp::extract(var_120, var_228);
    var_230 = wp::mul(var_227, var_229);
    var_231 = wp::sub(var_225, var_230);
    // - p_o_2D[1] * s2_2D[0]                                                                 <L 500>
    var_233 = wp::extract(var_122, var_232);
    var_235 = wp::extract(var_118, var_234);
    var_236 = wp::mul(var_233, var_235);
    var_237 = wp::sub(var_231, var_236);
    // - s3_2D[0] * s2_2D[1]                                                                  <L 501>
    var_239 = wp::extract(var_120, var_238);
    var_241 = wp::extract(var_118, var_240);
    var_242 = wp::mul(var_239, var_241);
    var_243 = wp::sub(var_237, var_242);
    // C32 = (                                                                                <L 505>
    // p_o_2D[0] * s3_2D[1]                                                                   <L 506>
    var_245 = wp::extract(var_122, var_244);
    var_247 = wp::extract(var_120, var_246);
    var_248 = wp::mul(var_245, var_247);
    // + p_o_2D[1] * s1_2D[0]                                                                 <L 507>
    var_250 = wp::extract(var_122, var_249);
    var_252 = wp::extract(var_116, var_251);
    var_253 = wp::mul(var_250, var_252);
    var_254 = wp::add(var_248, var_253);
    // + s3_2D[0] * s1_2D[1]                                                                  <L 508>
    var_256 = wp::extract(var_120, var_255);
    var_258 = wp::extract(var_116, var_257);
    var_259 = wp::mul(var_256, var_258);
    var_260 = wp::add(var_254, var_259);
    // - p_o_2D[0] * s1_2D[1]                                                                 <L 509>
    var_262 = wp::extract(var_122, var_261);
    var_264 = wp::extract(var_116, var_263);
    var_265 = wp::mul(var_262, var_264);
    var_266 = wp::sub(var_260, var_265);
    // - p_o_2D[1] * s3_2D[0]                                                                 <L 510>
    var_268 = wp::extract(var_122, var_267);
    var_270 = wp::extract(var_120, var_269);
    var_271 = wp::mul(var_268, var_270);
    var_272 = wp::sub(var_266, var_271);
    // - s1_2D[0] * s3_2D[1]                                                                  <L 511>
    var_274 = wp::extract(var_116, var_273);
    var_276 = wp::extract(var_120, var_275);
    var_277 = wp::mul(var_274, var_276);
    var_278 = wp::sub(var_272, var_277);
    // C33 = (                                                                                <L 515>
    // p_o_2D[0] * s1_2D[1]                                                                   <L 516>
    var_280 = wp::extract(var_122, var_279);
    var_282 = wp::extract(var_116, var_281);
    var_283 = wp::mul(var_280, var_282);
    // + p_o_2D[1] * s2_2D[0]                                                                 <L 517>
    var_285 = wp::extract(var_122, var_284);
    var_287 = wp::extract(var_118, var_286);
    var_288 = wp::mul(var_285, var_287);
    var_289 = wp::add(var_283, var_288);
    // + s1_2D[0] * s2_2D[1]                                                                  <L 518>
    var_291 = wp::extract(var_116, var_290);
    var_293 = wp::extract(var_118, var_292);
    var_294 = wp::mul(var_291, var_293);
    var_295 = wp::add(var_289, var_294);
    // - p_o_2D[0] * s2_2D[1]                                                                 <L 519>
    var_297 = wp::extract(var_122, var_296);
    var_299 = wp::extract(var_118, var_298);
    var_300 = wp::mul(var_297, var_299);
    var_301 = wp::sub(var_295, var_300);
    // - p_o_2D[1] * s1_2D[0]                                                                 <L 520>
    var_303 = wp::extract(var_122, var_302);
    var_305 = wp::extract(var_116, var_304);
    var_306 = wp::mul(var_303, var_305);
    var_307 = wp::sub(var_301, var_306);
    // - s2_2D[0] * s1_2D[1]                                                                  <L 521>
    var_309 = wp::extract(var_118, var_308);
    var_311 = wp::extract(var_116, var_310);
    var_312 = wp::mul(var_309, var_311);
    var_313 = wp::sub(var_307, var_312);
    // comp1 = _same_sign(M_max, C31)                                                         <L 524>
    var_314 = _same_sign_0(var_208, var_243);
    // comp2 = _same_sign(M_max, C32)                                                         <L 525>
    var_315 = _same_sign_0(var_208, var_278);
    // comp3 = _same_sign(M_max, C33)                                                         <L 526>
    var_316 = _same_sign_0(var_208, var_313);
    // if comp1 and comp2 and comp3:                                                          <L 529>
    var_317 = var_314;
    if (var_317) {
        var_317 = var_317 && var_315;
    }
    if (var_317) {
        var_317 = var_317 && var_316;
    }
    if (var_317) {
        // return wp.vec3(C31 / M_max, C32 / M_max, C33 / M_max)                              <L 530>
        var_318 = wp::div(var_243, var_208);
        var_319 = wp::div(var_278, var_208);
        var_320 = wp::div(var_313, var_208);
        var_321 = wp::vec_t<3, wp::float32>(var_318, var_319, var_320);
        return var_321;
    }
    // dmin = FLOAT_MAX                                                                       <L 533>
    var_323 = wp::copy(var_322);
    // lmbda = wp.vec3(0.0, 0.0, 0.0)                                                         <L 534>
    var_327 = wp::vec_t<3, wp::float32>(var_324, var_325, var_326);
    // if not comp1:                                                                          <L 536>
    var_328 = wp::unot(var_314);
    if (var_328) {
        // sublmbda = _S1D(s2, s3)                                                            <L 537>
        var_329 = _S1D_0(var_s2, var_s3);
        // x = sublmbda[0] * s2 + sublmbda[1] * s3                                            <L 538>
        var_331 = wp::extract(var_329, var_330);
        var_332 = wp::mul(var_331, var_s2);
        var_334 = wp::extract(var_329, var_333);
        var_335 = wp::mul(var_334, var_s3);
        var_336 = wp::add(var_332, var_335);
        // d = wp.dot(x, x)                                                                   <L 539>
        var_337 = wp::dot(var_336, var_336);
        // lmbda[0] = 0.0                                                                     <L 540>
        wp::assign_inplace(var_327, var_339, var_338);
        // lmbda[1] = sublmbda[0]                                                             <L 541>
        var_341 = wp::extract(var_329, var_340);
        wp::assign_inplace(var_327, var_342, var_341);
        // lmbda[2] = sublmbda[1]                                                             <L 542>
        var_344 = wp::extract(var_329, var_343);
        wp::assign_inplace(var_327, var_345, var_344);
        // dmin = d                                                                           <L 543>
        var_346 = wp::copy(var_337);
    }
    var_347 = wp::where(var_328, var_346, var_323);
    // if not comp2:                                                                          <L 545>
    var_348 = wp::unot(var_315);
    if (var_348) {
        // sublmbda = _S1D(s1, s3)                                                            <L 546>
        var_349 = _S1D_0(var_s1, var_s3);
        // x = sublmbda[0] * s1 + sublmbda[1] * s3                                            <L 547>
        var_351 = wp::extract(var_349, var_350);
        var_352 = wp::mul(var_351, var_s1);
        var_354 = wp::extract(var_349, var_353);
        var_355 = wp::mul(var_354, var_s3);
        var_356 = wp::add(var_352, var_355);
        // d = wp.dot(x, x)                                                                   <L 548>
        var_357 = wp::dot(var_356, var_356);
        // if d < dmin:                                                                       <L 549>
        var_358 = (var_357 < var_347);
        if (var_358) {
            // lmbda[0] = sublmbda[0]                                                         <L 550>
            var_360 = wp::extract(var_349, var_359);
            wp::assign_inplace(var_327, var_361, var_360);
            // lmbda[1] = 0.0                                                                 <L 551>
            wp::assign_inplace(var_327, var_363, var_362);
            // lmbda[2] = sublmbda[1]                                                         <L 552>
            var_365 = wp::extract(var_349, var_364);
            wp::assign_inplace(var_327, var_366, var_365);
            // dmin = d                                                                       <L 553>
            var_367 = wp::copy(var_357);
        }
        var_368 = wp::where(var_358, var_367, var_347);
    }
    var_369 = wp::where(var_348, var_368, var_347);
    var_370 = wp::where(var_348, var_349, var_329);
    var_371 = wp::where(var_348, var_356, var_336);
    var_372 = wp::where(var_348, var_357, var_337);
    // if not comp3:                                                                          <L 555>
    var_373 = wp::unot(var_316);
    if (var_373) {
        // sublmbda = _S1D(s1, s2)                                                            <L 556>
        var_374 = _S1D_0(var_s1, var_s2);
        // x = sublmbda[0] * s1 + sublmbda[1] * s2                                            <L 557>
        var_376 = wp::extract(var_374, var_375);
        var_377 = wp::mul(var_376, var_s1);
        var_379 = wp::extract(var_374, var_378);
        var_380 = wp::mul(var_379, var_s2);
        var_381 = wp::add(var_377, var_380);
        // d = wp.dot(x, x)                                                                   <L 558>
        var_382 = wp::dot(var_381, var_381);
        // if d < dmin:                                                                       <L 559>
        var_383 = (var_382 < var_369);
        if (var_383) {
            // lmbda[0] = sublmbda[0]                                                         <L 560>
            var_385 = wp::extract(var_374, var_384);
            wp::assign_inplace(var_327, var_386, var_385);
            // lmbda[1] = sublmbda[1]                                                         <L 561>
            var_388 = wp::extract(var_374, var_387);
            wp::assign_inplace(var_327, var_389, var_388);
            // lmbda[2] = 0.0                                                                 <L 562>
            wp::assign_inplace(var_327, var_391, var_390);
        }
    }
    var_392 = wp::where(var_373, var_374, var_370);
    var_393 = wp::where(var_373, var_381, var_371);
    var_394 = wp::where(var_373, var_382, var_372);
    // return lmbda                                                                           <L 563>
    return var_327;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:345
static CUDA_CALLABLE wp::vec_t<4, wp::float32> _S3D_0(
    wp::vec_t<3, wp::float32> var_s1,
    wp::vec_t<3, wp::float32> var_s2,
    wp::vec_t<3, wp::float32> var_s3,
    wp::vec_t<3, wp::float32> var_s4)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    wp::int32 var_9;
    wp::int32 var_10;
    wp::int32 var_11;
    wp::int32 var_12;
    bool var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::vec_t<4, wp::float32> var_18;
    const wp::float32 var_19 = 0.0;
    const wp::float32 var_20 = 0.0;
    const wp::float32 var_21 = 0.0;
    const wp::float32 var_22 = 0.0;
    wp::vec_t<4, wp::float32> var_23;
    const wp::float32 var_24 = 1e+30;
    wp::float32 var_25;
    bool var_26;
    wp::vec_t<3, wp::float32> var_27;
    const wp::int32 var_28 = 0;
    wp::float32 var_29;
    wp::vec_t<3, wp::float32> var_30;
    const wp::int32 var_31 = 1;
    wp::float32 var_32;
    wp::vec_t<3, wp::float32> var_33;
    wp::vec_t<3, wp::float32> var_34;
    const wp::int32 var_35 = 2;
    wp::float32 var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::float32 var_39;
    const wp::float32 var_40 = 0.0;
    const wp::int32 var_41 = 0;
    const wp::int32 var_42 = 0;
    wp::float32 var_43;
    const wp::int32 var_44 = 1;
    const wp::int32 var_45 = 1;
    wp::float32 var_46;
    const wp::int32 var_47 = 2;
    const wp::int32 var_48 = 2;
    wp::float32 var_49;
    const wp::int32 var_50 = 3;
    wp::float32 var_51;
    wp::float32 var_52;
    bool var_53;
    wp::vec_t<3, wp::float32> var_54;
    const wp::int32 var_55 = 0;
    wp::float32 var_56;
    wp::vec_t<3, wp::float32> var_57;
    const wp::int32 var_58 = 1;
    wp::float32 var_59;
    wp::vec_t<3, wp::float32> var_60;
    wp::vec_t<3, wp::float32> var_61;
    const wp::int32 var_62 = 2;
    wp::float32 var_63;
    wp::vec_t<3, wp::float32> var_64;
    wp::vec_t<3, wp::float32> var_65;
    wp::float32 var_66;
    bool var_67;
    const wp::int32 var_68 = 0;
    wp::float32 var_69;
    const wp::int32 var_70 = 0;
    const wp::float32 var_71 = 0.0;
    const wp::int32 var_72 = 1;
    const wp::int32 var_73 = 1;
    wp::float32 var_74;
    const wp::int32 var_75 = 2;
    const wp::int32 var_76 = 2;
    wp::float32 var_77;
    const wp::int32 var_78 = 3;
    wp::float32 var_79;
    wp::float32 var_80;
    wp::float32 var_81;
    wp::vec_t<3, wp::float32> var_82;
    wp::vec_t<3, wp::float32> var_83;
    wp::float32 var_84;
    bool var_85;
    wp::vec_t<3, wp::float32> var_86;
    const wp::int32 var_87 = 0;
    wp::float32 var_88;
    wp::vec_t<3, wp::float32> var_89;
    const wp::int32 var_90 = 1;
    wp::float32 var_91;
    wp::vec_t<3, wp::float32> var_92;
    wp::vec_t<3, wp::float32> var_93;
    const wp::int32 var_94 = 2;
    wp::float32 var_95;
    wp::vec_t<3, wp::float32> var_96;
    wp::vec_t<3, wp::float32> var_97;
    wp::float32 var_98;
    bool var_99;
    const wp::int32 var_100 = 0;
    wp::float32 var_101;
    const wp::int32 var_102 = 0;
    const wp::int32 var_103 = 1;
    wp::float32 var_104;
    const wp::int32 var_105 = 1;
    const wp::float32 var_106 = 0.0;
    const wp::int32 var_107 = 2;
    const wp::int32 var_108 = 2;
    wp::float32 var_109;
    const wp::int32 var_110 = 3;
    wp::float32 var_111;
    wp::float32 var_112;
    wp::float32 var_113;
    wp::vec_t<3, wp::float32> var_114;
    wp::vec_t<3, wp::float32> var_115;
    wp::float32 var_116;
    bool var_117;
    wp::vec_t<3, wp::float32> var_118;
    const wp::int32 var_119 = 0;
    wp::float32 var_120;
    wp::vec_t<3, wp::float32> var_121;
    const wp::int32 var_122 = 1;
    wp::float32 var_123;
    wp::vec_t<3, wp::float32> var_124;
    wp::vec_t<3, wp::float32> var_125;
    const wp::int32 var_126 = 2;
    wp::float32 var_127;
    wp::vec_t<3, wp::float32> var_128;
    wp::vec_t<3, wp::float32> var_129;
    wp::float32 var_130;
    bool var_131;
    const wp::int32 var_132 = 0;
    wp::float32 var_133;
    const wp::int32 var_134 = 0;
    const wp::int32 var_135 = 1;
    wp::float32 var_136;
    const wp::int32 var_137 = 1;
    const wp::int32 var_138 = 2;
    wp::float32 var_139;
    const wp::int32 var_140 = 2;
    const wp::float32 var_141 = 0.0;
    const wp::int32 var_142 = 3;
    wp::vec_t<3, wp::float32> var_143;
    wp::vec_t<3, wp::float32> var_144;
    wp::float32 var_145;
    //---------
    // forward
    // def _S3D(s1: wp.vec3, s2: wp.vec3, s3: wp.vec3, s4: wp.vec3) -> wp.vec4:               <L 346>
    // C41 = -_det3(s2, s3, s4)                                                               <L 355>
    var_0 = _det3_0(var_s2, var_s3, var_s4);
    var_1 = wp::neg(var_0);
    // C42 = _det3(s1, s3, s4)                                                                <L 356>
    var_2 = _det3_0(var_s1, var_s3, var_s4);
    // C43 = -_det3(s1, s2, s4)                                                               <L 357>
    var_3 = _det3_0(var_s1, var_s2, var_s4);
    var_4 = wp::neg(var_3);
    // C44 = _det3(s1, s2, s3)                                                                <L 358>
    var_5 = _det3_0(var_s1, var_s2, var_s3);
    // m_det = C41 + C42 + C43 + C44                                                          <L 362>
    var_6 = wp::add(var_1, var_2);
    var_7 = wp::add(var_6, var_4);
    var_8 = wp::add(var_7, var_5);
    // comp1 = _same_sign(m_det, C41)                                                         <L 364>
    var_9 = _same_sign_0(var_8, var_1);
    // comp2 = _same_sign(m_det, C42)                                                         <L 365>
    var_10 = _same_sign_0(var_8, var_2);
    // comp3 = _same_sign(m_det, C43)                                                         <L 366>
    var_11 = _same_sign_0(var_8, var_4);
    // comp4 = _same_sign(m_det, C44)                                                         <L 367>
    var_12 = _same_sign_0(var_8, var_5);
    // if comp1 and comp2 and comp3 and comp4:                                                <L 370>
    var_13 = var_9;
    if (var_13) {
        var_13 = var_13 && var_10;
    }
    if (var_13) {
        var_13 = var_13 && var_11;
    }
    if (var_13) {
        var_13 = var_13 && var_12;
    }
    if (var_13) {
        // return wp.vec4(C41 / m_det, C42 / m_det, C43 / m_det, C44 / m_det)                 <L 371>
        var_14 = wp::div(var_1, var_8);
        var_15 = wp::div(var_2, var_8);
        var_16 = wp::div(var_4, var_8);
        var_17 = wp::div(var_5, var_8);
        var_18 = wp::vec_t<4, wp::float32>(var_14, var_15, var_16, var_17);
        return var_18;
    }
    // lmbda = wp.vec4(0.0, 0.0, 0.0, 0.0)                                                    <L 374>
    var_23 = wp::vec_t<4, wp::float32>(var_19, var_20, var_21, var_22);
    // dmin = FLOAT_MAX                                                                       <L 375>
    var_25 = wp::copy(var_24);
    // if not comp1:                                                                          <L 377>
    var_26 = wp::unot(var_9);
    if (var_26) {
        // sublmbda = _S2D(s2, s3, s4)                                                        <L 378>
        var_27 = _S2D_0(var_s2, var_s3, var_s4);
        // x = sublmbda[0] * s2 + sublmbda[1] * s3 + sublmbda[2] * s4                         <L 379>
        var_29 = wp::extract(var_27, var_28);
        var_30 = wp::mul(var_29, var_s2);
        var_32 = wp::extract(var_27, var_31);
        var_33 = wp::mul(var_32, var_s3);
        var_34 = wp::add(var_30, var_33);
        var_36 = wp::extract(var_27, var_35);
        var_37 = wp::mul(var_36, var_s4);
        var_38 = wp::add(var_34, var_37);
        // d = wp.dot(x, x)                                                                   <L 380>
        var_39 = wp::dot(var_38, var_38);
        // lmbda[0] = 0.0                                                                     <L 381>
        wp::assign_inplace(var_23, var_41, var_40);
        // lmbda[1] = sublmbda[0]                                                             <L 382>
        var_43 = wp::extract(var_27, var_42);
        wp::assign_inplace(var_23, var_44, var_43);
        // lmbda[2] = sublmbda[1]                                                             <L 383>
        var_46 = wp::extract(var_27, var_45);
        wp::assign_inplace(var_23, var_47, var_46);
        // lmbda[3] = sublmbda[2]                                                             <L 384>
        var_49 = wp::extract(var_27, var_48);
        wp::assign_inplace(var_23, var_50, var_49);
        // dmin = d                                                                           <L 385>
        var_51 = wp::copy(var_39);
    }
    var_52 = wp::where(var_26, var_51, var_25);
    // if not comp2:                                                                          <L 387>
    var_53 = wp::unot(var_10);
    if (var_53) {
        // sublmbda = _S2D(s1, s3, s4)                                                        <L 388>
        var_54 = _S2D_0(var_s1, var_s3, var_s4);
        // x = sublmbda[0] * s1 + sublmbda[1] * s3 + sublmbda[2] * s4                         <L 389>
        var_56 = wp::extract(var_54, var_55);
        var_57 = wp::mul(var_56, var_s1);
        var_59 = wp::extract(var_54, var_58);
        var_60 = wp::mul(var_59, var_s3);
        var_61 = wp::add(var_57, var_60);
        var_63 = wp::extract(var_54, var_62);
        var_64 = wp::mul(var_63, var_s4);
        var_65 = wp::add(var_61, var_64);
        // d = wp.dot(x, x)                                                                   <L 390>
        var_66 = wp::dot(var_65, var_65);
        // if d < dmin:                                                                       <L 391>
        var_67 = (var_66 < var_52);
        if (var_67) {
            // lmbda[0] = sublmbda[0]                                                         <L 392>
            var_69 = wp::extract(var_54, var_68);
            wp::assign_inplace(var_23, var_70, var_69);
            // lmbda[1] = 0.0                                                                 <L 393>
            wp::assign_inplace(var_23, var_72, var_71);
            // lmbda[2] = sublmbda[1]                                                         <L 394>
            var_74 = wp::extract(var_54, var_73);
            wp::assign_inplace(var_23, var_75, var_74);
            // lmbda[3] = sublmbda[2]                                                         <L 395>
            var_77 = wp::extract(var_54, var_76);
            wp::assign_inplace(var_23, var_78, var_77);
            // dmin = d                                                                       <L 396>
            var_79 = wp::copy(var_66);
        }
        var_80 = wp::where(var_67, var_79, var_52);
    }
    var_81 = wp::where(var_53, var_80, var_52);
    var_82 = wp::where(var_53, var_54, var_27);
    var_83 = wp::where(var_53, var_65, var_38);
    var_84 = wp::where(var_53, var_66, var_39);
    // if not comp3:                                                                          <L 398>
    var_85 = wp::unot(var_11);
    if (var_85) {
        // sublmbda = _S2D(s1, s2, s4)                                                        <L 399>
        var_86 = _S2D_0(var_s1, var_s2, var_s4);
        // x = sublmbda[0] * s1 + sublmbda[1] * s2 + sublmbda[2] * s4                         <L 400>
        var_88 = wp::extract(var_86, var_87);
        var_89 = wp::mul(var_88, var_s1);
        var_91 = wp::extract(var_86, var_90);
        var_92 = wp::mul(var_91, var_s2);
        var_93 = wp::add(var_89, var_92);
        var_95 = wp::extract(var_86, var_94);
        var_96 = wp::mul(var_95, var_s4);
        var_97 = wp::add(var_93, var_96);
        // d = wp.dot(x, x)                                                                   <L 401>
        var_98 = wp::dot(var_97, var_97);
        // if d < dmin:                                                                       <L 402>
        var_99 = (var_98 < var_81);
        if (var_99) {
            // lmbda[0] = sublmbda[0]                                                         <L 403>
            var_101 = wp::extract(var_86, var_100);
            wp::assign_inplace(var_23, var_102, var_101);
            // lmbda[1] = sublmbda[1]                                                         <L 404>
            var_104 = wp::extract(var_86, var_103);
            wp::assign_inplace(var_23, var_105, var_104);
            // lmbda[2] = 0.0                                                                 <L 405>
            wp::assign_inplace(var_23, var_107, var_106);
            // lmbda[3] = sublmbda[2]                                                         <L 406>
            var_109 = wp::extract(var_86, var_108);
            wp::assign_inplace(var_23, var_110, var_109);
            // dmin = d                                                                       <L 407>
            var_111 = wp::copy(var_98);
        }
        var_112 = wp::where(var_99, var_111, var_81);
    }
    var_113 = wp::where(var_85, var_112, var_81);
    var_114 = wp::where(var_85, var_86, var_82);
    var_115 = wp::where(var_85, var_97, var_83);
    var_116 = wp::where(var_85, var_98, var_84);
    // if not comp4:                                                                          <L 409>
    var_117 = wp::unot(var_12);
    if (var_117) {
        // sublmbda = _S2D(s1, s2, s3)                                                        <L 410>
        var_118 = _S2D_0(var_s1, var_s2, var_s3);
        // x = sublmbda[0] * s1 + sublmbda[1] * s2 + sublmbda[2] * s3                         <L 411>
        var_120 = wp::extract(var_118, var_119);
        var_121 = wp::mul(var_120, var_s1);
        var_123 = wp::extract(var_118, var_122);
        var_124 = wp::mul(var_123, var_s2);
        var_125 = wp::add(var_121, var_124);
        var_127 = wp::extract(var_118, var_126);
        var_128 = wp::mul(var_127, var_s3);
        var_129 = wp::add(var_125, var_128);
        // d = wp.dot(x, x)                                                                   <L 412>
        var_130 = wp::dot(var_129, var_129);
        // if d < dmin:                                                                       <L 413>
        var_131 = (var_130 < var_113);
        if (var_131) {
            // lmbda[0] = sublmbda[0]                                                         <L 414>
            var_133 = wp::extract(var_118, var_132);
            wp::assign_inplace(var_23, var_134, var_133);
            // lmbda[1] = sublmbda[1]                                                         <L 415>
            var_136 = wp::extract(var_118, var_135);
            wp::assign_inplace(var_23, var_137, var_136);
            // lmbda[2] = sublmbda[2]                                                         <L 416>
            var_139 = wp::extract(var_118, var_138);
            wp::assign_inplace(var_23, var_140, var_139);
            // lmbda[3] = 0.0                                                                 <L 417>
            wp::assign_inplace(var_23, var_142, var_141);
        }
    }
    var_143 = wp::where(var_117, var_118, var_114);
    var_144 = wp::where(var_117, var_129, var_115);
    var_145 = wp::where(var_117, var_130, var_116);
    // return lmbda                                                                           <L 418>
    return var_23;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:279
static CUDA_CALLABLE wp::vec_t<4, wp::float32> _subdistance_0(
    wp::int32 var_n,
    wp::mat_t<4, 3, wp::float32> var_simplex)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 4;
    bool var_1;
    const wp::int32 var_2 = 0;
    wp::vec_t<3, wp::float32> var_3;
    const wp::int32 var_4 = 1;
    wp::vec_t<3, wp::float32> var_5;
    const wp::int32 var_6 = 2;
    wp::vec_t<3, wp::float32> var_7;
    const wp::int32 var_8 = 3;
    wp::vec_t<3, wp::float32> var_9;
    wp::vec_t<4, wp::float32> var_10;
    const wp::int32 var_11 = 3;
    bool var_12;
    const wp::int32 var_13 = 0;
    wp::vec_t<3, wp::float32> var_14;
    const wp::int32 var_15 = 1;
    wp::vec_t<3, wp::float32> var_16;
    const wp::int32 var_17 = 2;
    wp::vec_t<3, wp::float32> var_18;
    wp::vec_t<3, wp::float32> var_19;
    const wp::int32 var_20 = 0;
    wp::float32 var_21;
    const wp::int32 var_22 = 1;
    wp::float32 var_23;
    const wp::int32 var_24 = 2;
    wp::float32 var_25;
    const wp::float32 var_26 = 0.0;
    wp::vec_t<4, wp::float32> var_27;
    const wp::int32 var_28 = 2;
    bool var_29;
    const wp::int32 var_30 = 0;
    wp::vec_t<3, wp::float32> var_31;
    const wp::int32 var_32 = 1;
    wp::vec_t<3, wp::float32> var_33;
    wp::vec_t<2, wp::float32> var_34;
    const wp::int32 var_35 = 0;
    wp::float32 var_36;
    const wp::int32 var_37 = 1;
    wp::float32 var_38;
    const wp::float32 var_39 = 0.0;
    const wp::float32 var_40 = 0.0;
    wp::vec_t<4, wp::float32> var_41;
    const wp::float32 var_42 = 1.0;
    const wp::float32 var_43 = 0.0;
    const wp::float32 var_44 = 0.0;
    const wp::float32 var_45 = 0.0;
    wp::vec_t<4, wp::float32> var_46;
    //---------
    // forward
    // def _subdistance(n: int, simplex: mat43) -> wp.vec4:                                   <L 280>
    // if n == 4:                                                                             <L 281>
    var_1 = (var_n == var_0);
    if (var_1) {
        // return _S3D(simplex[0], simplex[1], simplex[2], simplex[3])                        <L 282>
        var_3 = wp::extract(var_simplex, var_2);
        var_5 = wp::extract(var_simplex, var_4);
        var_7 = wp::extract(var_simplex, var_6);
        var_9 = wp::extract(var_simplex, var_8);
        var_10 = _S3D_0(var_3, var_5, var_7, var_9);
        return var_10;
    }
    // if n == 3:                                                                             <L 283>
    var_12 = (var_n == var_11);
    if (var_12) {
        // lmbda3 = _S2D(simplex[0], simplex[1], simplex[2])                                  <L 284>
        var_14 = wp::extract(var_simplex, var_13);
        var_16 = wp::extract(var_simplex, var_15);
        var_18 = wp::extract(var_simplex, var_17);
        var_19 = _S2D_0(var_14, var_16, var_18);
        // return wp.vec4(lmbda3[0], lmbda3[1], lmbda3[2], 0.0)                               <L 285>
        var_21 = wp::extract(var_19, var_20);
        var_23 = wp::extract(var_19, var_22);
        var_25 = wp::extract(var_19, var_24);
        var_27 = wp::vec_t<4, wp::float32>(var_21, var_23, var_25, var_26);
        return var_27;
    }
    // if n == 2:                                                                             <L 286>
    var_29 = (var_n == var_28);
    if (var_29) {
        // lmbda2 = _S1D(simplex[0], simplex[1])                                              <L 287>
        var_31 = wp::extract(var_simplex, var_30);
        var_33 = wp::extract(var_simplex, var_32);
        var_34 = _S1D_0(var_31, var_33);
        // return wp.vec4(lmbda2[0], lmbda2[1], 0.0, 0.0)                                     <L 288>
        var_36 = wp::extract(var_34, var_35);
        var_38 = wp::extract(var_34, var_37);
        var_41 = wp::vec_t<4, wp::float32>(var_36, var_38, var_39, var_40);
        return var_41;
    }
    // return wp.vec4(1.0, 0.0, 0.0, 0.0)                                                     <L 289>
    var_46 = wp::vec_t<4, wp::float32>(var_42, var_43, var_44, var_45);
    return var_46;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:268
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _linear_combine_0(
    wp::int32 var_n,
    wp::vec_t<4, wp::float32> var_scl,
    wp::mat_t<4, 3, wp::float32> var_mat)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    bool var_1;
    const wp::int32 var_2 = 0;
    wp::float32 var_3;
    const wp::int32 var_4 = 0;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32> var_6;
    const wp::int32 var_7 = 2;
    bool var_8;
    const wp::int32 var_9 = 0;
    wp::float32 var_10;
    const wp::int32 var_11 = 0;
    wp::vec_t<3, wp::float32> var_12;
    wp::vec_t<3, wp::float32> var_13;
    const wp::int32 var_14 = 1;
    wp::float32 var_15;
    const wp::int32 var_16 = 1;
    wp::vec_t<3, wp::float32> var_17;
    wp::vec_t<3, wp::float32> var_18;
    wp::vec_t<3, wp::float32> var_19;
    const wp::int32 var_20 = 3;
    bool var_21;
    const wp::int32 var_22 = 0;
    wp::float32 var_23;
    const wp::int32 var_24 = 0;
    wp::vec_t<3, wp::float32> var_25;
    wp::vec_t<3, wp::float32> var_26;
    const wp::int32 var_27 = 1;
    wp::float32 var_28;
    const wp::int32 var_29 = 1;
    wp::vec_t<3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<3, wp::float32> var_32;
    const wp::int32 var_33 = 2;
    wp::float32 var_34;
    const wp::int32 var_35 = 2;
    wp::vec_t<3, wp::float32> var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::vec_t<3, wp::float32> var_38;
    const wp::int32 var_39 = 0;
    wp::float32 var_40;
    const wp::int32 var_41 = 0;
    wp::vec_t<3, wp::float32> var_42;
    wp::vec_t<3, wp::float32> var_43;
    const wp::int32 var_44 = 1;
    wp::float32 var_45;
    const wp::int32 var_46 = 1;
    wp::vec_t<3, wp::float32> var_47;
    wp::vec_t<3, wp::float32> var_48;
    wp::vec_t<3, wp::float32> var_49;
    const wp::int32 var_50 = 2;
    wp::float32 var_51;
    const wp::int32 var_52 = 2;
    wp::vec_t<3, wp::float32> var_53;
    wp::vec_t<3, wp::float32> var_54;
    wp::vec_t<3, wp::float32> var_55;
    const wp::int32 var_56 = 3;
    wp::float32 var_57;
    const wp::int32 var_58 = 3;
    wp::vec_t<3, wp::float32> var_59;
    wp::vec_t<3, wp::float32> var_60;
    wp::vec_t<3, wp::float32> var_61;
    //---------
    // forward
    // def _linear_combine(n: int, scl: wp.vec4, mat: mat43) -> wp.vec3:                      <L 269>
    // if n == 1:                                                                             <L 270>
    var_1 = (var_n == var_0);
    if (var_1) {
        // return scl[0] * mat[0]                                                             <L 271>
        var_3 = wp::extract(var_scl, var_2);
        var_5 = wp::extract(var_mat, var_4);
        var_6 = wp::mul(var_3, var_5);
        return var_6;
    }
    // if n == 2:                                                                             <L 272>
    var_8 = (var_n == var_7);
    if (var_8) {
        // return scl[0] * mat[0] + scl[1] * mat[1]                                           <L 273>
        var_10 = wp::extract(var_scl, var_9);
        var_12 = wp::extract(var_mat, var_11);
        var_13 = wp::mul(var_10, var_12);
        var_15 = wp::extract(var_scl, var_14);
        var_17 = wp::extract(var_mat, var_16);
        var_18 = wp::mul(var_15, var_17);
        var_19 = wp::add(var_13, var_18);
        return var_19;
    }
    // if n == 3:                                                                             <L 274>
    var_21 = (var_n == var_20);
    if (var_21) {
        // return scl[0] * mat[0] + scl[1] * mat[1] + scl[2] * mat[2]                         <L 275>
        var_23 = wp::extract(var_scl, var_22);
        var_25 = wp::extract(var_mat, var_24);
        var_26 = wp::mul(var_23, var_25);
        var_28 = wp::extract(var_scl, var_27);
        var_30 = wp::extract(var_mat, var_29);
        var_31 = wp::mul(var_28, var_30);
        var_32 = wp::add(var_26, var_31);
        var_34 = wp::extract(var_scl, var_33);
        var_36 = wp::extract(var_mat, var_35);
        var_37 = wp::mul(var_34, var_36);
        var_38 = wp::add(var_32, var_37);
        return var_38;
    }
    // return scl[0] * mat[0] + scl[1] * mat[1] + scl[2] * mat[2] + scl[3] * mat[3]           <L 276>
    var_40 = wp::extract(var_scl, var_39);
    var_42 = wp::extract(var_mat, var_41);
    var_43 = wp::mul(var_40, var_42);
    var_45 = wp::extract(var_scl, var_44);
    var_47 = wp::extract(var_mat, var_46);
    var_48 = wp::mul(var_45, var_47);
    var_49 = wp::add(var_43, var_48);
    var_51 = wp::extract(var_scl, var_50);
    var_53 = wp::extract(var_mat, var_52);
    var_54 = wp::mul(var_51, var_53);
    var_55 = wp::add(var_49, var_54);
    var_57 = wp::extract(var_scl, var_56);
    var_59 = wp::extract(var_mat, var_58);
    var_60 = wp::mul(var_57, var_59);
    var_61 = wp::add(var_55, var_60);
    return var_61;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:633
static CUDA_CALLABLE GJKResult_28609055 gjk_0(
    wp::float32 var_tolerance,
    wp::int32 var_gjk_iterations,
    Geom_3242f8a8 var_geom1,
    Geom_3242f8a8 var_geom2,
    wp::vec_t<3, wp::float32> var_x1_0,
    wp::vec_t<3, wp::float32> var_x2_0,
    wp::int32 var_geomtype1,
    wp::int32 var_geomtype2,
    wp::float32 var_cutoff,
    bool var_is_discrete)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::mat_t<4, 3, wp::float32> var_1;
    wp::mat_t<4, 3, wp::float32> var_2;
    wp::mat_t<4, 3, wp::float32> var_3;
    wp::vec_t<4, wp::int32> var_4;
    wp::vec_t<4, wp::int32> var_5;
    const wp::int32 var_6 = 0;
    wp::int32 var_7;
    const wp::float32 var_8 = 1.0;
    const wp::float32 var_9 = 0.0;
    const wp::float32 var_10 = 0.0;
    const wp::float32 var_11 = 0.0;
    wp::vec_t<4, wp::float32> var_12;
    const wp::float32 var_13 = 0.0;
    const wp::float32 var_14 = 0.5;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    const wp::float32 var_18 = 1e-15;
    wp::float32 var_19;
    wp::float32 var_20;
    wp::vec_t<3, wp::float32> var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    const wp::float32 var_24 = 0.0;
    wp::float32 var_25;
    wp::range_t var_26;
    wp::int32 var_27;
    bool var_28;
    bool var_29;
    wp::float32 var_30;
    wp::float32 var_31;
    bool var_32;
    SupportPoint_e82efc60 var_33;
    SupportPoint_e82efc60 var_34;
    wp::vec_t<3, wp::float32>* var_35;
    wp::vec_t<3, wp::float32> var_36;
    wp::int32* var_37;
    wp::int32 var_38;
    wp::int32 var_39;
    wp::int32* var_40;
    wp::int32 var_41;
    wp::vec_t<3, wp::float32>* var_42;
    wp::vec_t<3, wp::float32> var_43;
    wp::int32* var_44;
    wp::int32 var_45;
    wp::int32 var_46;
    wp::int32* var_47;
    wp::int32 var_48;
    wp::vec_t<3, wp::float32> var_49;
    wp::vec_t<3, wp::float32> var_50;
    wp::vec_t<3, wp::float32> var_51;
    wp::vec_t<3, wp::float32> var_52;
    wp::vec_t<3, wp::float32> var_53;
    wp::float32 var_54;
    bool var_55;
    const wp::float32 var_56 = 0.0;
    bool var_57;
    wp::vec_t<3, wp::float32> var_58;
    wp::float32 var_59;
    const wp::float32 var_60 = 0.0;
    bool var_61;
    GJKResult_28609055 var_62;
    const wp::int32 var_63 = 0;
    const wp::float32 var_64 = 1e+30;
    wp::int32* var_65;
    wp::int32 var_66;
    wp::int32 var_67;
    wp::int32* var_68;
    wp::int32 var_69;
    wp::int32 var_70;
    bool var_71;
    wp::vec_t<3, wp::float32> var_72;
    wp::float32 var_73;
    bool var_74;
    wp::vec_t<3, wp::float32> var_75;
    wp::float32 var_76;
    const wp::float32 var_77 = 0.0;
    bool var_78;
    wp::float32 var_79;
    wp::float32 var_80;
    bool var_81;
    GJKResult_28609055 var_82;
    const wp::int32 var_83 = 0;
    wp::int32* var_84;
    wp::int32 var_85;
    wp::int32 var_86;
    wp::int32* var_87;
    wp::int32 var_88;
    wp::int32 var_89;
    GJKResult_28609055 var_90;
    GJKResult_28609055 var_91;
    GJKResult_28609055 var_92;
    const wp::int32 var_93 = 1;
    wp::int32 var_94;
    wp::vec_t<4, wp::float32> var_95;
    const wp::int32 var_96 = 0;
    wp::int32 var_97;
    const wp::int32 var_98 = 4;
    wp::range_t var_99;
    wp::int32 var_100;
    wp::float32 var_101;
    const wp::float32 var_102 = 0.0;
    bool var_103;
    wp::vec_t<3, wp::float32> var_104;
    wp::vec_t<3, wp::float32> var_105;
    wp::vec_t<3, wp::float32> var_106;
    wp::int32 var_107;
    wp::int32 var_108;
    wp::float32 var_109;
    const wp::int32 var_110 = 1;
    wp::int32 var_111;
    wp::int32 var_112;
    const wp::int32 var_113 = 1;
    bool var_114;
    wp::int32 var_115;
    wp::vec_t<4, wp::float32> var_116;
    const wp::int32 var_117 = 4;
    bool var_118;
    const wp::float32 var_119 = 0.0;
    wp::int32 var_120;
    wp::vec_t<4, wp::float32> var_121;
    wp::vec_t<3, wp::float32> var_122;
    wp::float32 var_123;
    wp::float32 var_124;
    wp::float32 var_125;
    GJKResult_28609055 var_126;
    const wp::int32 var_127 = 0;
    bool var_128;
    wp::vec_t<3, wp::float32> var_129;
    wp::vec_t<3, wp::float32> var_130;
    const wp::int32 var_131 = 0;
    bool var_132;
    wp::vec_t<3, wp::float32> var_133;
    wp::vec_t<3, wp::float32> var_134;
    wp::int32* var_135;
    wp::int32 var_136;
    wp::int32 var_137;
    wp::int32* var_138;
    wp::int32 var_139;
    wp::int32 var_140;
    //---------
    // forward
    // def gjk(                                                                               <L 634>
    // cutoff2 = cutoff * cutoff                                                              <L 648>
    var_0 = wp::mul(var_cutoff, var_cutoff);
    // simplex = mat43()                                                                      <L 649>
    var_1 = wp::mat_t<4, 3, wp::float32>();
    // simplex1 = mat43()                                                                     <L 650>
    var_2 = wp::mat_t<4, 3, wp::float32>();
    // simplex2 = mat43()                                                                     <L 651>
    var_3 = wp::mat_t<4, 3, wp::float32>();
    // simplex_index1 = wp.vec4i()                                                            <L 652>
    var_4 = wp::vec_t<4, wp::int32>();
    // simplex_index2 = wp.vec4i()                                                            <L 653>
    var_5 = wp::vec_t<4, wp::int32>();
    // n = int(0)                                                                             <L 654>
    var_7 = wp::int(var_6);
    // lmbda = wp.vec4(1.0, 0.0, 0.0, 0.0)  # barycentric coordinates                         <L 655>
    var_12 = wp::vec_t<4, wp::float32>(var_8, var_9, var_10, var_11);
    // epsilon = wp.where(is_discrete, 0.0, 0.5 * tolerance * tolerance)                      <L 660>
    var_15 = wp::mul(var_14, var_tolerance);
    var_16 = wp::mul(var_15, var_tolerance);
    var_17 = wp::where(var_is_discrete, var_13, var_16);
    // min_norm = wp.where(is_discrete, MINVAL, tolerance)                                    <L 661>
    var_19 = wp::where(var_is_discrete, var_18, var_tolerance);
    // min_tol = wp.where(is_discrete, MINVAL, tolerance)                                     <L 662>
    var_20 = wp::where(var_is_discrete, var_18, var_tolerance);
    // x_k = x1_0 - x2_0                                                                      <L 665>
    var_21 = wp::sub(var_x1_0, var_x2_0);
    // xnorm2 = wp.dot(x_k, x_k)                                                              <L 666>
    var_22 = wp::dot(var_21, var_21);
    // xnorm = wp.sqrt(xnorm2)                                                                <L 667>
    var_23 = wp::sqrt(var_22);
    // xnorm_prev = float(0.0)                                                                <L 668>
    var_25 = wp::float(var_24);
    // for _ in range(gjk_iterations):                                                        <L 670>
    var_26 = wp::range(var_gjk_iterations);
    start_for_0:;
        if (iter_cmp(var_26) == 0) goto end_for_0;
        var_27 = wp::iter_next(var_26);
        // if xnorm < min_norm or wp.abs(xnorm_prev - xnorm) < min_tol:                       <L 671>
        var_29 = (var_23 < var_19);
        var_28 = var_29;
        if (!var_28) {
            var_30 = wp::sub(var_25, var_23);
            var_31 = wp::abs(var_30);
            var_32 = (var_31 < var_20);
            var_28 = var_28 || var_32;
        }
        if (var_28) {
            // break                                                                          <L 672>
            goto end_for_0;
        }
        // sp1, sp2 = _gjk_support(geom1, geom2, geomtype1, geomtype2, x_k, xnorm, simplex, n, is_discrete)       <L 675>
        _gjk_support_0(var_geom1, var_geom2, var_geomtype1, var_geomtype2, var_21, var_23, var_1, var_7, var_is_discrete, var_33, var_34);
        // simplex1[n] = sp1.point                                                            <L 676>
        var_35 = &((var_33).point);
        var_36 = wp::load(var_35);
        wp::assign_inplace(var_2, var_7, var_36);
        // geom1.index = sp1.cached_index                                                     <L 677>
        var_37 = &((var_33).cached_index);
        var_39 = wp::load(var_37);
        var_38 = wp::copy(var_39);
        var_geom1.index = var_38;
        // simplex_index1[n] = sp1.vertex_index                                               <L 678>
        var_40 = &((var_33).vertex_index);
        var_41 = wp::load(var_40);
        wp::assign_inplace(var_4, var_7, var_41);
        // simplex2[n] = sp2.point                                                            <L 680>
        var_42 = &((var_34).point);
        var_43 = wp::load(var_42);
        wp::assign_inplace(var_3, var_7, var_43);
        // geom2.index = sp2.cached_index                                                     <L 681>
        var_44 = &((var_34).cached_index);
        var_46 = wp::load(var_44);
        var_45 = wp::copy(var_46);
        var_geom2.index = var_45;
        // simplex_index2[n] = sp2.vertex_index                                               <L 682>
        var_47 = &((var_34).vertex_index);
        var_48 = wp::load(var_47);
        wp::assign_inplace(var_5, var_7, var_48);
        // simplex[n] = simplex1[n] - simplex2[n]                                             <L 685>
        var_49 = wp::extract(var_2, var_7);
        var_50 = wp::extract(var_3, var_7);
        var_51 = wp::sub(var_49, var_50);
        wp::assign_inplace(var_1, var_7, var_51);
        // if wp.dot(x_k, x_k - simplex[n]) < epsilon:                                        <L 689>
        var_52 = wp::extract(var_1, var_7);
        var_53 = wp::sub(var_21, var_52);
        var_54 = wp::dot(var_21, var_53);
        var_55 = (var_54 < var_17);
        if (var_55) {
            // break                                                                          <L 690>
            goto end_for_0;
        }
        // if cutoff == 0.0:                                                                  <L 692>
        var_57 = (var_cutoff == var_56);
        if (var_57) {
            // if wp.dot(x_k, simplex[n]) > 0.0:                                              <L 693>
            var_58 = wp::extract(var_1, var_7);
            var_59 = wp::dot(var_21, var_58);
            var_61 = (var_59 > var_60);
            if (var_61) {
                // result = GJKResult()                                                       <L 694>
                var_62 = GJKResult_28609055();
                // result.dim = 0                                                             <L 695>
                var_62.dim = var_63;
                // result.dist = FLOAT_MAX                                                    <L 696>
                var_62.dist = var_64;
                // result.index1 = geom1.index                                                <L 697>
                var_65 = &((var_geom1).index);
                var_67 = wp::load(var_65);
                var_66 = wp::copy(var_67);
                var_62.index1 = var_66;
                // result.index2 = geom2.index                                                <L 698>
                var_68 = &((var_geom2).index);
                var_70 = wp::load(var_68);
                var_69 = wp::copy(var_70);
                var_62.index2 = var_69;
                // return result                                                              <L 699>
                return var_62;
            }
        }
        if (!var_57) {
            // elif cutoff < FLOAT_MAX:                                                       <L 700>
            var_71 = (var_cutoff < var_64);
            if (var_71) {
                // vs = wp.dot(x_k, simplex[n])                                               <L 701>
                var_72 = wp::extract(var_1, var_7);
                var_73 = wp::dot(var_21, var_72);
                // if wp.dot(x_k, simplex[n]) > 0.0 and (vs * vs / xnorm2) >= cutoff2:        <L 702>
                var_75 = wp::extract(var_1, var_7);
                var_76 = wp::dot(var_21, var_75);
                var_78 = (var_76 > var_77);
                var_74 = var_78;
                if (var_74) {
                    var_79 = wp::mul(var_73, var_73);
                    var_80 = wp::div(var_79, var_22);
                    var_81 = (var_80 >= var_0);
                    var_74 = var_74 && var_81;
                }
                if (var_74) {
                    // result = GJKResult()                                                   <L 703>
                    var_82 = GJKResult_28609055();
                    // result.dim = 0                                                         <L 704>
                    var_82.dim = var_83;
                    // result.dist = FLOAT_MAX                                                <L 705>
                    var_82.dist = var_64;
                    // result.index1 = geom1.index                                            <L 706>
                    var_84 = &((var_geom1).index);
                    var_86 = wp::load(var_84);
                    var_85 = wp::copy(var_86);
                    var_82.index1 = var_85;
                    // result.index2 = geom2.index                                            <L 707>
                    var_87 = &((var_geom2).index);
                    var_89 = wp::load(var_87);
                    var_88 = wp::copy(var_89);
                    var_82.index2 = var_88;
                    // return result                                                          <L 708>
                    return var_82;
                }
                var_90 = wp::where(var_74, var_82, var_62);
            }
            var_91 = wp::where(var_71, var_90, var_62);
        }
        var_92 = wp::where(var_57, var_62, var_91);
        // lmbda = _subdistance(n + 1, simplex)                                               <L 712>
        var_94 = wp::add(var_7, var_93);
        var_95 = _subdistance_0(var_94, var_1);
        // n = int(0)                                                                         <L 715>
        var_97 = wp::int(var_96);
        // for i in range(4):                                                                 <L 716>
        var_99 = wp::range(var_98);
        start_for_4:;
            if (iter_cmp(var_99) == 0) goto end_for_4;
            var_100 = wp::iter_next(var_99);
            // if lmbda[i] == 0.0:                                                            <L 717>
            var_101 = wp::extract(var_95, var_100);
            var_103 = (var_101 == var_102);
            if (var_103) {
                // continue                                                                   <L 718>
                goto start_for_4;
            }
            // simplex[n] = simplex[i]                                                        <L 720>
            var_104 = wp::extract(var_1, var_100);
            wp::assign_inplace(var_1, var_97, var_104);
            // simplex1[n] = simplex1[i]                                                      <L 721>
            var_105 = wp::extract(var_2, var_100);
            wp::assign_inplace(var_2, var_97, var_105);
            // simplex2[n] = simplex2[i]                                                      <L 722>
            var_106 = wp::extract(var_3, var_100);
            wp::assign_inplace(var_3, var_97, var_106);
            // simplex_index1[n] = simplex_index1[i]                                          <L 723>
            var_107 = wp::extract(var_4, var_100);
            wp::assign_inplace(var_4, var_97, var_107);
            // simplex_index2[n] = simplex_index2[i]                                          <L 724>
            var_108 = wp::extract(var_5, var_100);
            wp::assign_inplace(var_5, var_97, var_108);
            // lmbda[n] = lmbda[i]                                                            <L 725>
            var_109 = wp::extract(var_95, var_100);
            wp::assign_inplace(var_95, var_97, var_109);
            // n += int(1)                                                                    <L 726>
            var_111 = wp::int(var_110);
            var_112 = wp::add(var_97, var_111);
            wp::assign(var_97, var_112);
            goto start_for_4;
        end_for_4:;
        // if n < 1:                                                                          <L 729>
        var_114 = (var_97 < var_113);
        if (var_114) {
            // break                                                                          <L 730>
            wp::assign(var_7, var_97);
            wp::assign(var_12, var_95);
            goto end_for_0;
        }
        var_115 = wp::where(var_114, var_7, var_97);
        var_116 = wp::where(var_114, var_12, var_95);
        // if n == 4:                                                                         <L 733>
        var_118 = (var_115 == var_117);
        if (var_118) {
            // xnorm = 0.0                                                                    <L 734>
            // break                                                                          <L 735>
            wp::assign(var_7, var_115);
            wp::assign(var_12, var_116);
            wp::assign(var_23, var_119);
            goto end_for_0;
        }
        var_120 = wp::where(var_118, var_7, var_115);
        var_121 = wp::where(var_118, var_12, var_116);
        // x_k = _linear_combine(n, lmbda, simplex)                                           <L 738>
        var_122 = _linear_combine_0(var_120, var_121, var_1);
        // xnorm_prev = xnorm                                                                 <L 739>
        var_123 = wp::copy(var_23);
        // xnorm2 = wp.dot(x_k, x_k)                                                          <L 740>
        var_124 = wp::dot(var_122, var_122);
        // xnorm = wp.sqrt(xnorm2)                                                            <L 741>
        var_125 = wp::sqrt(var_124);
        wp::assign(var_7, var_120);
        wp::assign(var_12, var_121);
        wp::assign(var_21, var_122);
        wp::assign(var_22, var_124);
        wp::assign(var_23, var_125);
        wp::assign(var_25, var_123);
        goto start_for_0;
    end_for_0:;
    // result = GJKResult()                                                                   <L 743>
    var_126 = GJKResult_28609055();
    // result.x1 = wp.where(n == 0, x1_0, _linear_combine(n, lmbda, simplex1))                <L 748>
    var_128 = (var_7 == var_127);
    var_129 = _linear_combine_0(var_7, var_12, var_2);
    var_130 = wp::where(var_128, var_x1_0, var_129);
    var_126.x1 = var_130;
    // result.x2 = wp.where(n == 0, x2_0, _linear_combine(n, lmbda, simplex2))                <L 749>
    var_132 = (var_7 == var_131);
    var_133 = _linear_combine_0(var_7, var_12, var_3);
    var_134 = wp::where(var_132, var_x2_0, var_133);
    var_126.x2 = var_134;
    // result.dist = xnorm                                                                    <L 750>
    var_126.dist = var_23;
    // result.dim = n                                                                         <L 752>
    var_126.dim = var_7;
    // result.simplex1 = simplex1                                                             <L 753>
    var_126.simplex1 = var_2;
    // result.simplex2 = simplex2                                                             <L 754>
    var_126.simplex2 = var_3;
    // result.simplex_index1 = simplex_index1                                                 <L 755>
    var_126.simplex_index1 = var_4;
    // result.simplex_index2 = simplex_index2                                                 <L 756>
    var_126.simplex_index2 = var_5;
    // result.simplex = simplex                                                               <L 757>
    var_126.simplex = var_1;
    // result.index1 = geom1.index                                                            <L 758>
    var_135 = &((var_geom1).index);
    var_137 = wp::load(var_135);
    var_136 = wp::copy(var_137);
    var_126.index1 = var_136;
    // result.index2 = geom2.index                                                            <L 759>
    var_138 = &((var_geom2).index);
    var_140 = wp::load(var_138);
    var_139 = wp::copy(var_140);
    var_126.index2 = var_139;
    // return result                                                                          <L 760>
    return var_126;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:776
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _tri_affine_coord_0(
    wp::vec_t<3, wp::float32> var_v1,
    wp::vec_t<3, wp::float32> var_v2,
    wp::vec_t<3, wp::float32> var_v3,
    wp::vec_t<3, wp::float32> var_p)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    wp::float32 var_1;
    const wp::int32 var_2 = 2;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::int32 var_5 = 2;
    wp::float32 var_6;
    const wp::int32 var_7 = 1;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    const wp::int32 var_11 = 1;
    wp::float32 var_12;
    const wp::int32 var_13 = 2;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    const wp::int32 var_17 = 2;
    wp::float32 var_18;
    const wp::int32 var_19 = 1;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    const wp::int32 var_23 = 1;
    wp::float32 var_24;
    const wp::int32 var_25 = 2;
    wp::float32 var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    const wp::int32 var_29 = 2;
    wp::float32 var_30;
    const wp::int32 var_31 = 1;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    const wp::int32 var_35 = 0;
    wp::float32 var_36;
    const wp::int32 var_37 = 2;
    wp::float32 var_38;
    wp::float32 var_39;
    const wp::int32 var_40 = 2;
    wp::float32 var_41;
    const wp::int32 var_42 = 0;
    wp::float32 var_43;
    wp::float32 var_44;
    wp::float32 var_45;
    const wp::int32 var_46 = 0;
    wp::float32 var_47;
    const wp::int32 var_48 = 2;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::float32 var_51;
    const wp::int32 var_52 = 2;
    wp::float32 var_53;
    const wp::int32 var_54 = 0;
    wp::float32 var_55;
    wp::float32 var_56;
    wp::float32 var_57;
    const wp::int32 var_58 = 0;
    wp::float32 var_59;
    const wp::int32 var_60 = 2;
    wp::float32 var_61;
    wp::float32 var_62;
    wp::float32 var_63;
    const wp::int32 var_64 = 2;
    wp::float32 var_65;
    const wp::int32 var_66 = 0;
    wp::float32 var_67;
    wp::float32 var_68;
    wp::float32 var_69;
    const wp::int32 var_70 = 0;
    wp::float32 var_71;
    const wp::int32 var_72 = 1;
    wp::float32 var_73;
    wp::float32 var_74;
    const wp::int32 var_75 = 1;
    wp::float32 var_76;
    const wp::int32 var_77 = 0;
    wp::float32 var_78;
    wp::float32 var_79;
    wp::float32 var_80;
    const wp::int32 var_81 = 0;
    wp::float32 var_82;
    const wp::int32 var_83 = 1;
    wp::float32 var_84;
    wp::float32 var_85;
    wp::float32 var_86;
    const wp::int32 var_87 = 1;
    wp::float32 var_88;
    const wp::int32 var_89 = 0;
    wp::float32 var_90;
    wp::float32 var_91;
    wp::float32 var_92;
    const wp::int32 var_93 = 0;
    wp::float32 var_94;
    const wp::int32 var_95 = 1;
    wp::float32 var_96;
    wp::float32 var_97;
    wp::float32 var_98;
    const wp::int32 var_99 = 1;
    wp::float32 var_100;
    const wp::int32 var_101 = 0;
    wp::float32 var_102;
    wp::float32 var_103;
    wp::float32 var_104;
    const wp::float32 var_105 = 0.0;
    const wp::int32 var_106 = 0;
    const wp::int32 var_107 = 0;
    wp::float32 var_108;
    wp::float32 var_109;
    wp::float32 var_110;
    bool var_111;
    bool var_112;
    bool var_113;
    wp::float32 var_114;
    const wp::int32 var_115 = 1;
    const wp::int32 var_116 = 2;
    wp::float32 var_117;
    wp::int32 var_118;
    wp::int32 var_119;
    bool var_120;
    wp::float32 var_121;
    const wp::int32 var_122 = 0;
    const wp::int32 var_123 = 2;
    wp::float32 var_124;
    wp::int32 var_125;
    wp::int32 var_126;
    wp::float32 var_127;
    const wp::int32 var_128 = 0;
    const wp::int32 var_129 = 1;
    wp::float32 var_130;
    wp::int32 var_131;
    wp::int32 var_132;
    wp::float32 var_133;
    wp::int32 var_134;
    wp::int32 var_135;
    wp::float32 var_136;
    wp::float32 var_137;
    wp::float32 var_138;
    wp::float32 var_139;
    wp::float32 var_140;
    wp::float32 var_141;
    wp::float32 var_142;
    wp::float32 var_143;
    wp::float32 var_144;
    wp::float32 var_145;
    wp::float32 var_146;
    wp::float32 var_147;
    wp::float32 var_148;
    wp::float32 var_149;
    wp::float32 var_150;
    wp::float32 var_151;
    wp::float32 var_152;
    wp::float32 var_153;
    wp::float32 var_154;
    wp::float32 var_155;
    wp::float32 var_156;
    wp::float32 var_157;
    wp::float32 var_158;
    wp::float32 var_159;
    wp::float32 var_160;
    wp::float32 var_161;
    wp::float32 var_162;
    wp::float32 var_163;
    wp::float32 var_164;
    wp::float32 var_165;
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
    wp::float32 var_183;
    wp::float32 var_184;
    wp::float32 var_185;
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
    wp::float32 var_197;
    wp::float32 var_198;
    wp::float32 var_199;
    wp::float32 var_200;
    wp::float32 var_201;
    wp::float32 var_202;
    wp::float32 var_203;
    wp::float32 var_204;
    wp::float32 var_205;
    wp::float32 var_206;
    wp::float32 var_207;
    wp::vec_t<3, wp::float32> var_208;
    //---------
    // forward
    // def _tri_affine_coord(v1: wp.vec3, v2: wp.vec3, v3: wp.vec3, p: wp.vec3) -> wp.vec3:       <L 777>
    // M_14 = v2[1] * v3[2] - v2[2] * v3[1] - v1[1] * v3[2] + v1[2] * v3[1] + v1[1] * v2[2] - v1[2] * v2[1]       <L 779>
    var_1 = wp::extract(var_v2, var_0);
    var_3 = wp::extract(var_v3, var_2);
    var_4 = wp::mul(var_1, var_3);
    var_6 = wp::extract(var_v2, var_5);
    var_8 = wp::extract(var_v3, var_7);
    var_9 = wp::mul(var_6, var_8);
    var_10 = wp::sub(var_4, var_9);
    var_12 = wp::extract(var_v1, var_11);
    var_14 = wp::extract(var_v3, var_13);
    var_15 = wp::mul(var_12, var_14);
    var_16 = wp::sub(var_10, var_15);
    var_18 = wp::extract(var_v1, var_17);
    var_20 = wp::extract(var_v3, var_19);
    var_21 = wp::mul(var_18, var_20);
    var_22 = wp::add(var_16, var_21);
    var_24 = wp::extract(var_v1, var_23);
    var_26 = wp::extract(var_v2, var_25);
    var_27 = wp::mul(var_24, var_26);
    var_28 = wp::add(var_22, var_27);
    var_30 = wp::extract(var_v1, var_29);
    var_32 = wp::extract(var_v2, var_31);
    var_33 = wp::mul(var_30, var_32);
    var_34 = wp::sub(var_28, var_33);
    // M_24 = v2[0] * v3[2] - v2[2] * v3[0] - v1[0] * v3[2] + v1[2] * v3[0] + v1[0] * v2[2] - v1[2] * v2[0]       <L 780>
    var_36 = wp::extract(var_v2, var_35);
    var_38 = wp::extract(var_v3, var_37);
    var_39 = wp::mul(var_36, var_38);
    var_41 = wp::extract(var_v2, var_40);
    var_43 = wp::extract(var_v3, var_42);
    var_44 = wp::mul(var_41, var_43);
    var_45 = wp::sub(var_39, var_44);
    var_47 = wp::extract(var_v1, var_46);
    var_49 = wp::extract(var_v3, var_48);
    var_50 = wp::mul(var_47, var_49);
    var_51 = wp::sub(var_45, var_50);
    var_53 = wp::extract(var_v1, var_52);
    var_55 = wp::extract(var_v3, var_54);
    var_56 = wp::mul(var_53, var_55);
    var_57 = wp::add(var_51, var_56);
    var_59 = wp::extract(var_v1, var_58);
    var_61 = wp::extract(var_v2, var_60);
    var_62 = wp::mul(var_59, var_61);
    var_63 = wp::add(var_57, var_62);
    var_65 = wp::extract(var_v1, var_64);
    var_67 = wp::extract(var_v2, var_66);
    var_68 = wp::mul(var_65, var_67);
    var_69 = wp::sub(var_63, var_68);
    // M_34 = v2[0] * v3[1] - v2[1] * v3[0] - v1[0] * v3[1] + v1[1] * v3[0] + v1[0] * v2[1] - v1[1] * v2[0]       <L 781>
    var_71 = wp::extract(var_v2, var_70);
    var_73 = wp::extract(var_v3, var_72);
    var_74 = wp::mul(var_71, var_73);
    var_76 = wp::extract(var_v2, var_75);
    var_78 = wp::extract(var_v3, var_77);
    var_79 = wp::mul(var_76, var_78);
    var_80 = wp::sub(var_74, var_79);
    var_82 = wp::extract(var_v1, var_81);
    var_84 = wp::extract(var_v3, var_83);
    var_85 = wp::mul(var_82, var_84);
    var_86 = wp::sub(var_80, var_85);
    var_88 = wp::extract(var_v1, var_87);
    var_90 = wp::extract(var_v3, var_89);
    var_91 = wp::mul(var_88, var_90);
    var_92 = wp::add(var_86, var_91);
    var_94 = wp::extract(var_v1, var_93);
    var_96 = wp::extract(var_v2, var_95);
    var_97 = wp::mul(var_94, var_96);
    var_98 = wp::add(var_92, var_97);
    var_100 = wp::extract(var_v1, var_99);
    var_102 = wp::extract(var_v2, var_101);
    var_103 = wp::mul(var_100, var_102);
    var_104 = wp::sub(var_98, var_103);
    // M_max = 0.0                                                                            <L 785>
    // x = 0                                                                                  <L 786>
    // y = 0                                                                                  <L 787>
    // mu1 = wp.abs(M_14)                                                                     <L 789>
    var_108 = wp::abs(var_34);
    // mu2 = wp.abs(M_24)                                                                     <L 790>
    var_109 = wp::abs(var_69);
    // mu3 = wp.abs(M_34)                                                                     <L 791>
    var_110 = wp::abs(var_104);
    // if mu1 >= mu2 and mu1 >= mu3:                                                          <L 793>
    var_112 = (var_108 >= var_109);
    var_111 = var_112;
    if (var_111) {
        var_113 = (var_108 >= var_110);
        var_111 = var_111 && var_113;
    }
    if (var_111) {
        // M_max = M_14                                                                       <L 794>
        var_114 = wp::copy(var_34);
        // x = 1                                                                              <L 795>
        // y = 2                                                                              <L 796>
    }
    var_117 = wp::where(var_111, var_114, var_105);
    var_118 = wp::where(var_111, var_115, var_106);
    var_119 = wp::where(var_111, var_116, var_107);
    if (!var_111) {
        // elif mu2 >= mu3:                                                                   <L 797>
        var_120 = (var_109 >= var_110);
        if (var_120) {
            // M_max = M_24                                                                   <L 798>
            var_121 = wp::copy(var_69);
            // x = 0                                                                          <L 799>
            // y = 2                                                                          <L 800>
        }
        var_124 = wp::where(var_120, var_121, var_117);
        var_125 = wp::where(var_120, var_122, var_118);
        var_126 = wp::where(var_120, var_123, var_119);
        if (!var_120) {
            // M_max = M_34                                                                   <L 802>
            var_127 = wp::copy(var_104);
            // x = 0                                                                          <L 803>
            // y = 1                                                                          <L 804>
        }
        var_130 = wp::where(var_120, var_124, var_127);
        var_131 = wp::where(var_120, var_125, var_128);
        var_132 = wp::where(var_120, var_126, var_129);
    }
    var_133 = wp::where(var_111, var_117, var_130);
    var_134 = wp::where(var_111, var_118, var_131);
    var_135 = wp::where(var_111, var_119, var_132);
    // C31 = p[x] * v2[y] + p[y] * v3[x] + v2[x] * v3[y] - p[x] * v3[y] - p[y] * v2[x] - v3[x] * v2[y]       <L 807>
    var_136 = wp::extract(var_p, var_134);
    var_137 = wp::extract(var_v2, var_135);
    var_138 = wp::mul(var_136, var_137);
    var_139 = wp::extract(var_p, var_135);
    var_140 = wp::extract(var_v3, var_134);
    var_141 = wp::mul(var_139, var_140);
    var_142 = wp::add(var_138, var_141);
    var_143 = wp::extract(var_v2, var_134);
    var_144 = wp::extract(var_v3, var_135);
    var_145 = wp::mul(var_143, var_144);
    var_146 = wp::add(var_142, var_145);
    var_147 = wp::extract(var_p, var_134);
    var_148 = wp::extract(var_v3, var_135);
    var_149 = wp::mul(var_147, var_148);
    var_150 = wp::sub(var_146, var_149);
    var_151 = wp::extract(var_p, var_135);
    var_152 = wp::extract(var_v2, var_134);
    var_153 = wp::mul(var_151, var_152);
    var_154 = wp::sub(var_150, var_153);
    var_155 = wp::extract(var_v3, var_134);
    var_156 = wp::extract(var_v2, var_135);
    var_157 = wp::mul(var_155, var_156);
    var_158 = wp::sub(var_154, var_157);
    // C32 = p[x] * v3[y] + p[y] * v1[x] + v3[x] * v1[y] - p[x] * v1[y] - p[y] * v3[x] - v1[x] * v3[y]       <L 810>
    var_159 = wp::extract(var_p, var_134);
    var_160 = wp::extract(var_v3, var_135);
    var_161 = wp::mul(var_159, var_160);
    var_162 = wp::extract(var_p, var_135);
    var_163 = wp::extract(var_v1, var_134);
    var_164 = wp::mul(var_162, var_163);
    var_165 = wp::add(var_161, var_164);
    var_166 = wp::extract(var_v3, var_134);
    var_167 = wp::extract(var_v1, var_135);
    var_168 = wp::mul(var_166, var_167);
    var_169 = wp::add(var_165, var_168);
    var_170 = wp::extract(var_p, var_134);
    var_171 = wp::extract(var_v1, var_135);
    var_172 = wp::mul(var_170, var_171);
    var_173 = wp::sub(var_169, var_172);
    var_174 = wp::extract(var_p, var_135);
    var_175 = wp::extract(var_v3, var_134);
    var_176 = wp::mul(var_174, var_175);
    var_177 = wp::sub(var_173, var_176);
    var_178 = wp::extract(var_v1, var_134);
    var_179 = wp::extract(var_v3, var_135);
    var_180 = wp::mul(var_178, var_179);
    var_181 = wp::sub(var_177, var_180);
    // C33 = p[x] * v1[y] + p[y] * v2[x] + v1[x] * v2[y] - p[x] * v2[y] - p[y] * v1[x] - v2[x] * v1[y]       <L 813>
    var_182 = wp::extract(var_p, var_134);
    var_183 = wp::extract(var_v1, var_135);
    var_184 = wp::mul(var_182, var_183);
    var_185 = wp::extract(var_p, var_135);
    var_186 = wp::extract(var_v2, var_134);
    var_187 = wp::mul(var_185, var_186);
    var_188 = wp::add(var_184, var_187);
    var_189 = wp::extract(var_v1, var_134);
    var_190 = wp::extract(var_v2, var_135);
    var_191 = wp::mul(var_189, var_190);
    var_192 = wp::add(var_188, var_191);
    var_193 = wp::extract(var_p, var_134);
    var_194 = wp::extract(var_v2, var_135);
    var_195 = wp::mul(var_193, var_194);
    var_196 = wp::sub(var_192, var_195);
    var_197 = wp::extract(var_p, var_135);
    var_198 = wp::extract(var_v1, var_134);
    var_199 = wp::mul(var_197, var_198);
    var_200 = wp::sub(var_196, var_199);
    var_201 = wp::extract(var_v2, var_134);
    var_202 = wp::extract(var_v1, var_135);
    var_203 = wp::mul(var_201, var_202);
    var_204 = wp::sub(var_200, var_203);
    // return wp.vec3(C31 / M_max, C32 / M_max, C33 / M_max)                                  <L 816>
    var_205 = wp::div(var_158, var_133);
    var_206 = wp::div(var_181, var_133);
    var_207 = wp::div(var_204, var_133);
    var_208 = wp::vec_t<3, wp::float32>(var_205, var_206, var_207);
    return var_208;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/warp/_src/math.py:0
static CUDA_CALLABLE wp::float32 norm_l2_0(
    wp::vec_t<3, wp::float32> var_v)
{
    //---------
    // primal vars
    wp::float32 var_0;
    //---------
    // forward
    // def norm_l2(v: Any) -> float:                                                          <L 1>
    // return wp.length(v)                                                                    <L 12>
    var_0 = wp::length(var_v);
    return var_0;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:2291
static CUDA_CALLABLE void _inflate_0(
    GJKResult_28609055 var_result,
    Geom_3242f8a8 var_geom1,
    Geom_3242f8a8 var_geom2,
    wp::int32 var_geomtype1,
    wp::int32 var_geomtype2,
    wp::float32 var_margin1,
    wp::float32 var_margin2,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::float32* var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    wp::vec_t<3, wp::float32>* var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32>* var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    const wp::int32 var_9 = 1;
    bool var_10;
    wp::vec_t<4, wp::int32>* var_11;
    const wp::int32 var_12 = 0;
    wp::int32 var_13;
    wp::vec_t<4, wp::int32> var_14;
    const bool var_15 = false;
    bool var_16;
    wp::int32* var_17;
    wp::range_t var_18;
    wp::int32 var_19;
    wp::int32 var_20;
    wp::vec_t<4, wp::int32>* var_21;
    wp::int32 var_22;
    wp::vec_t<4, wp::int32> var_23;
    bool var_24;
    const bool var_25 = true;
    const wp::float32 var_26 = 0.0;
    const wp::float32 var_27 = 0.0;
    const wp::float32 var_28 = 1.0;
    wp::vec_t<3, wp::float32> var_29;
    SupportPoint_e82efc60 var_30;
    wp::vec_t<3, wp::float32>* var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::vec_t<3, wp::float32> var_33;
    wp::vec_t<3, wp::float32> var_34;
    wp::mat_t<6, 3, wp::float32>* var_35;
    const wp::int32 var_36 = 3;
    wp::vec_t<3, wp::float32> var_37;
    wp::mat_t<6, 3, wp::float32> var_38;
    wp::mat_t<6, 3, wp::float32>* var_39;
    const wp::int32 var_40 = 4;
    wp::vec_t<3, wp::float32> var_41;
    wp::mat_t<6, 3, wp::float32> var_42;
    wp::mat_t<6, 3, wp::float32>* var_43;
    const wp::int32 var_44 = 5;
    wp::vec_t<3, wp::float32> var_45;
    wp::mat_t<6, 3, wp::float32> var_46;
    wp::vec_t<3, wp::float32> var_47;
    bool var_48;
    const wp::int32 var_49 = 0;
    wp::float32 var_50;
    const wp::float32 var_51 = 0.0;
    bool var_52;
    const wp::int32 var_53 = 1;
    wp::float32 var_54;
    const wp::float32 var_55 = 0.0;
    bool var_56;
    const wp::int32 var_57 = 2;
    wp::float32 var_58;
    const wp::float32 var_59 = 0.0;
    bool var_60;
    const wp::int32 var_61 = 0;
    wp::float32 var_62;
    wp::vec_t<3, wp::float32> var_63;
    const wp::int32 var_64 = 1;
    wp::float32 var_65;
    wp::vec_t<3, wp::float32> var_66;
    wp::vec_t<3, wp::float32> var_67;
    const wp::int32 var_68 = 2;
    wp::float32 var_69;
    wp::vec_t<3, wp::float32> var_70;
    wp::vec_t<3, wp::float32> var_71;
    wp::vec_t<3, wp::float32> var_72;
    wp::vec_t<3, wp::float32> var_73;
    const wp::int32 var_74 = 1;
    wp::float32 var_75;
    const wp::float32 var_76 = 0.0;
    bool var_77;
    wp::vec_t<3, wp::float32> var_78;
    const wp::int32 var_79 = 0;
    wp::float32 var_80;
    const wp::float32 var_81 = 0.0;
    bool var_82;
    wp::vec_t<3, wp::float32> var_83;
    wp::vec_t<3, wp::float32> var_84;
    wp::float32 var_85;
    wp::vec_t<3, wp::float32> var_86;
    wp::vec_t<3, wp::float32> var_87;
    wp::vec_t<3, wp::float32> var_88;
    wp::vec_t<3, wp::float32> var_89;
    wp::float32 var_90;
    wp::float32 var_91;
    wp::float32 var_92;
    wp::vec_t<3, wp::float32> var_93;
    wp::vec_t<3, wp::float32> var_94;
    wp::float32 var_95;
    wp::vec_t<3, wp::float32> var_96;
    wp::vec_t<3, wp::float32> var_97;
    wp::vec_t<3, wp::float32> var_98;
    wp::vec_t<3, wp::float32> var_99;
    const wp::float32 var_100 = 0.0;
    bool var_101;
    wp::vec_t<3, wp::float32> var_102;
    wp::vec_t<3, wp::float32> var_103;
    wp::vec_t<3, wp::float32> var_104;
    const wp::float32 var_105 = 0.0;
    bool var_106;
    wp::vec_t<3, wp::float32> var_107;
    wp::vec_t<3, wp::float32> var_108;
    wp::vec_t<3, wp::float32> var_109;
    wp::float32 var_110;
    wp::float32 var_111;
    //---------
    // forward
    // def _inflate(                                                                          <L 2292>
    // dist = result.dist                                                                     <L 2295>
    var_0 = &((var_result).dist);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    // x1 = result.x1                                                                         <L 2296>
    var_3 = &((var_result).x1);
    var_5 = wp::load(var_3);
    var_4 = wp::copy(var_5);
    // x2 = result.x2                                                                         <L 2297>
    var_6 = &((var_result).x2);
    var_8 = wp::load(var_6);
    var_7 = wp::copy(var_8);
    // if geomtype1 == GeomType.HFIELD:                                                       <L 2299>
    var_10 = (var_geomtype1 == var_9);
    if (var_10) {
        // v = result.simplex_index1[0]                                                       <L 2300>
        var_11 = &((var_result).simplex_index1);
        var_14 = wp::load(var_11);
        var_13 = wp::extract(var_14, var_12);
        // is_side = bool(False)                                                              <L 2301>
        var_16 = bool(var_15);
        // for i in range(result.dim):                                                        <L 2302>
        var_17 = &((var_result).dim);
        var_19 = wp::load(var_17);
        var_18 = wp::range(var_19);
        start_for_0:;
            if (iter_cmp(var_18) == 0) goto end_for_0;
            var_20 = wp::iter_next(var_18);
            // if result.simplex_index1[i] != v:                                              <L 2303>
            var_21 = &((var_result).simplex_index1);
            var_23 = wp::load(var_21);
            var_22 = wp::extract(var_23, var_20);
            var_24 = (var_22 != var_13);
            if (var_24) {
                // is_side = True                                                             <L 2304>
                // break                                                                      <L 2305>
                wp::assign(var_16, var_25);
                goto end_for_0;
            }
            goto start_for_0;
        end_for_0:;
        // if is_side:                                                                        <L 2307>
        if (var_16) {
            // n = wp.vec3(0.0, 0.0, 1.0)                                                     <L 2308>
            var_29 = wp::vec_t<3, wp::float32>(var_26, var_27, var_28);
            // sp = support(geom2, geomtype2, x2)                                             <L 2309>
            var_30 = support_0(var_geom2, var_geomtype2, var_7);
            // x2 = sp.point - margin2 * n                                                    <L 2310>
            var_31 = &((var_30).point);
            var_32 = wp::mul(var_margin2, var_29);
            var_34 = wp::load(var_31);
            var_33 = wp::sub(var_34, var_32);
            // a = geom1.hfprism[3]                                                           <L 2313>
            var_35 = &((var_geom1).hfprism);
            var_38 = wp::load(var_35);
            var_37 = wp::extract(var_38, var_36);
            // b = geom1.hfprism[4]                                                           <L 2314>
            var_39 = &((var_geom1).hfprism);
            var_42 = wp::load(var_39);
            var_41 = wp::extract(var_42, var_40);
            // c = geom1.hfprism[5]                                                           <L 2315>
            var_43 = &((var_geom1).hfprism);
            var_46 = wp::load(var_43);
            var_45 = wp::extract(var_46, var_44);
            // coordinates = _tri_affine_coord(a, b, c, x2)                                   <L 2317>
            var_47 = _tri_affine_coord_0(var_37, var_41, var_45, var_33);
            // if coordinates[0] > 0.0 and coordinates[1] > 0.0 and coordinates[2] > 0.0:       <L 2318>
            var_50 = wp::extract(var_47, var_49);
            var_52 = (var_50 > var_51);
            var_48 = var_52;
            if (var_48) {
                var_54 = wp::extract(var_47, var_53);
                var_56 = (var_54 > var_55);
                var_48 = var_48 && var_56;
            }
            if (var_48) {
                var_58 = wp::extract(var_47, var_57);
                var_60 = (var_58 > var_59);
                var_48 = var_48 && var_60;
            }
            if (var_48) {
                // x1 = coordinates[0] * a + coordinates[1] * b + coordinates[2] * c          <L 2319>
                var_62 = wp::extract(var_47, var_61);
                var_63 = wp::mul(var_62, var_37);
                var_65 = wp::extract(var_47, var_64);
                var_66 = wp::mul(var_65, var_41);
                var_67 = wp::add(var_63, var_66);
                var_69 = wp::extract(var_47, var_68);
                var_70 = wp::mul(var_69, var_45);
                var_71 = wp::add(var_67, var_70);
            }
            var_72 = wp::where(var_48, var_71, var_4);
            if (!var_48) {
                // p = c                                                                      <L 2321>
                var_73 = wp::copy(var_45);
                // p = wp.where(coordinates[1] > 0.0, b, p)                                   <L 2322>
                var_75 = wp::extract(var_47, var_74);
                var_77 = (var_75 > var_76);
                var_78 = wp::where(var_77, var_41, var_73);
                // p = wp.where(coordinates[0] > 0.0, a, p)                                   <L 2323>
                var_80 = wp::extract(var_47, var_79);
                var_82 = (var_80 > var_81);
                var_83 = wp::where(var_82, var_37, var_78);
                // x1 = x2 - wp.dot(x2 - p, n) * n                                            <L 2324>
                var_84 = wp::sub(var_33, var_83);
                var_85 = wp::dot(var_84, var_29);
                var_86 = wp::mul(var_85, var_29);
                var_87 = wp::sub(var_33, var_86);
            }
            var_88 = wp::where(var_48, var_72, var_87);
            // dist = -wp.norm_l2(x1 - x2)                                                    <L 2325>
            var_89 = wp::sub(var_88, var_33);
            var_90 = norm_l2_0(var_89);
            var_91 = wp::neg(var_90);
            // return dist, x1, x2                                                            <L 2326>
            ret_0 = var_91;
            ret_1 = var_88;
            ret_2 = var_33;
            return;
        }
        var_92 = wp::where(var_16, var_91, var_1);
        var_93 = wp::where(var_16, var_88, var_4);
        var_94 = wp::where(var_16, var_33, var_7);
    }
    var_95 = wp::where(var_10, var_92, var_1);
    var_96 = wp::where(var_10, var_93, var_4);
    var_97 = wp::where(var_10, var_94, var_7);
    // n = wp.normalize(x2 - x1)                                                              <L 2328>
    var_98 = wp::sub(var_97, var_96);
    var_99 = wp::normalize(var_98);
    // if margin1 > 0.0:                                                                      <L 2329>
    var_101 = (var_margin1 > var_100);
    if (var_101) {
        // x1 += margin1 * n                                                                  <L 2330>
        var_102 = wp::mul(var_margin1, var_99);
        var_103 = wp::add(var_96, var_102);
    }
    var_104 = wp::where(var_101, var_103, var_96);
    // if margin2 > 0.0:                                                                      <L 2332>
    var_106 = (var_margin2 > var_105);
    if (var_106) {
        // x2 -= margin2 * n                                                                  <L 2333>
        var_107 = wp::mul(var_margin2, var_99);
        var_108 = wp::sub(var_97, var_107);
    }
    var_109 = wp::where(var_106, var_108, var_97);
    // dist -= margin1 + margin2                                                              <L 2334>
    var_110 = wp::add(var_margin1, var_margin2);
    var_111 = wp::sub(var_95, var_110);
    // return dist, x1, x2                                                                    <L 2335>
    ret_0 = var_111;
    ret_1 = var_104;
    ret_2 = var_109;
    return;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:2338
static CUDA_CALLABLE void gjk_phase_0(
    wp::float32 var_tolerance,
    wp::float32 var_cutoff,
    wp::int32 var_gjk_iterations,
    Geom_3242f8a8 var_geom1,
    Geom_3242f8a8 var_geom2,
    wp::int32 var_geomtype1,
    wp::int32 var_geomtype2,
    wp::vec_t<3, wp::float32> var_x_1,
    wp::vec_t<3, wp::float32> var_x_2,
    bool & ret_0,
    wp::float32 & ret_1,
    wp::int32 & ret_2,
    wp::vec_t<3, wp::float32> & ret_3,
    wp::vec_t<3, wp::float32> & ret_4,
    GJKResult_28609055 & ret_5,
    Geom_3242f8a8 & ret_6,
    Geom_3242f8a8 & ret_7)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    const wp::float32 var_1 = 0.0;
    const wp::float32 var_2 = 0.0;
    const wp::float32 var_3 = 0.0;
    GJKResult_28609055 var_4;
    bool var_5;
    bool var_6;
    bool var_7;
    wp::float32* var_8;
    const wp::float32 var_9 = 0.0;
    bool var_10;
    wp::float32 var_11;
    wp::float32* var_12;
    const wp::float32 var_13 = 0.0;
    bool var_14;
    wp::float32 var_15;
    bool var_16;
    const wp::int32 var_17 = 2;
    bool var_18;
    const wp::int32 var_19 = 3;
    bool var_20;
    wp::vec_t<3, wp::float32>* var_21;
    const wp::int32 var_22 = 0;
    wp::float32 var_23;
    wp::vec_t<3, wp::float32> var_24;
    const wp::float32 var_25 = 0.5;
    wp::float32* var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    const wp::float32 var_30 = 0.0;
    const wp::float32 var_31 = 0.0;
    wp::vec_t<3, wp::float32>* var_32;
    const wp::int32 var_33 = 1;
    wp::float32 var_34;
    wp::vec_t<3, wp::float32> var_35;
    wp::vec_t<3, wp::float32>* var_36;
    const wp::int32 var_37 = 2;
    wp::float32 var_38;
    wp::vec_t<3, wp::float32> var_39;
    wp::vec_t<3, wp::float32> var_40;
    wp::float32 var_41;
    wp::float32 var_42;
    bool var_43;
    const wp::int32 var_44 = 2;
    bool var_45;
    const wp::int32 var_46 = 3;
    bool var_47;
    wp::vec_t<3, wp::float32>* var_48;
    const wp::int32 var_49 = 0;
    wp::float32 var_50;
    wp::vec_t<3, wp::float32> var_51;
    const wp::float32 var_52 = 0.5;
    wp::float32* var_53;
    wp::float32 var_54;
    wp::float32 var_55;
    wp::float32 var_56;
    const wp::float32 var_57 = 0.0;
    const wp::float32 var_58 = 0.0;
    wp::vec_t<3, wp::float32>* var_59;
    const wp::int32 var_60 = 1;
    wp::float32 var_61;
    wp::vec_t<3, wp::float32> var_62;
    wp::vec_t<3, wp::float32>* var_63;
    const wp::int32 var_64 = 2;
    wp::float32 var_65;
    wp::vec_t<3, wp::float32> var_66;
    wp::vec_t<3, wp::float32> var_67;
    wp::float32 var_68;
    wp::float32 var_69;
    wp::float32 var_70;
    const wp::float32 var_71 = 0.0;
    bool var_72;
    wp::float32 var_73;
    wp::float32 var_74;
    GJKResult_28609055 var_75;
    wp::int32* var_76;
    wp::int32 var_77;
    wp::int32 var_78;
    wp::int32* var_79;
    wp::int32 var_80;
    wp::int32 var_81;
    wp::float32* var_82;
    bool var_83;
    wp::float32 var_84;
    wp::float32* var_85;
    const wp::float32 var_86 = 1e+30;
    bool var_87;
    wp::float32 var_88;
    const bool var_89 = false;
    wp::float32* var_90;
    const wp::int32 var_91 = 1;
    wp::vec_t<3, wp::float32>* var_92;
    wp::vec_t<3, wp::float32>* var_93;
    wp::float32 var_94;
    wp::float32 var_95;
    wp::vec_t<3, wp::float32> var_96;
    wp::vec_t<3, wp::float32> var_97;
    wp::vec_t<3, wp::float32> var_98;
    wp::vec_t<3, wp::float32> var_99;
    wp::float32 var_100;
    wp::vec_t<3, wp::float32> var_101;
    wp::vec_t<3, wp::float32> var_102;
    const bool var_103 = false;
    const wp::int32 var_104 = 1;
    wp::float32 var_105;
    wp::vec_t<3, wp::float32>* var_106;
    const wp::int32 var_107 = 1;
    wp::float32 var_108;
    wp::vec_t<3, wp::float32> var_109;
    wp::vec_t<3, wp::float32>* var_110;
    const wp::int32 var_111 = 2;
    wp::float32 var_112;
    wp::vec_t<3, wp::float32> var_113;
    wp::vec_t<3, wp::float32> var_114;
    wp::float32 var_115;
    wp::vec_t<3, wp::float32>* var_116;
    const wp::int32 var_117 = 1;
    wp::float32 var_118;
    wp::vec_t<3, wp::float32> var_119;
    wp::vec_t<3, wp::float32>* var_120;
    const wp::int32 var_121 = 2;
    wp::float32 var_122;
    wp::vec_t<3, wp::float32> var_123;
    wp::vec_t<3, wp::float32> var_124;
    wp::float32 var_125;
    wp::float32 var_126;
    wp::float32 var_127;
    GJKResult_28609055 var_128;
    wp::int32* var_129;
    wp::int32 var_130;
    wp::int32 var_131;
    wp::int32* var_132;
    wp::int32 var_133;
    wp::int32 var_134;
    bool var_135;
    wp::float32* var_136;
    bool var_137;
    wp::float32 var_138;
    wp::int32* var_139;
    const wp::int32 var_140 = 2;
    bool var_141;
    wp::int32 var_142;
    const bool var_143 = false;
    wp::float32* var_144;
    const wp::int32 var_145 = 1;
    wp::vec_t<3, wp::float32>* var_146;
    wp::vec_t<3, wp::float32>* var_147;
    wp::float32 var_148;
    wp::float32 var_149;
    wp::vec_t<3, wp::float32> var_150;
    wp::vec_t<3, wp::float32> var_151;
    wp::vec_t<3, wp::float32> var_152;
    wp::vec_t<3, wp::float32> var_153;
    const bool var_154 = true;
    wp::float32* var_155;
    const wp::int32 var_156 = 1;
    wp::vec_t<3, wp::float32>* var_157;
    wp::vec_t<3, wp::float32>* var_158;
    wp::float32 var_159;
    wp::float32 var_160;
    wp::vec_t<3, wp::float32> var_161;
    wp::vec_t<3, wp::float32> var_162;
    wp::vec_t<3, wp::float32> var_163;
    wp::vec_t<3, wp::float32> var_164;
    //---------
    // forward
    // def gjk_phase(                                                                         <L 2339>
    // full_margin1 = 0.0                                                                     <L 2352>
    // full_margin2 = 0.0                                                                     <L 2353>
    // size1 = 0.0                                                                            <L 2354>
    // size2 = 0.0                                                                            <L 2355>
    // empty = GJKResult()                                                                    <L 2356>
    var_4 = GJKResult_28609055();
    // is_discrete = _discrete_geoms(geomtype1, geomtype2) and (geom1.margin == 0.0 and geom2.margin == 0.0)       <L 2359>
    var_6 = _discrete_geoms_0(var_geomtype1, var_geomtype2);
    var_5 = var_6;
    if (var_5) {
        var_8 = &((var_geom1).margin);
        var_11 = wp::load(var_8);
        var_10 = (var_11 == var_9);
        var_7 = var_10;
        if (var_7) {
            var_12 = &((var_geom2).margin);
            var_15 = wp::load(var_12);
            var_14 = (var_15 == var_13);
            var_7 = var_7 && var_14;
        }
        var_5 = var_5 && var_7;
    }
    // if geomtype1 == GeomType.SPHERE or geomtype1 == GeomType.CAPSULE:                      <L 2362>
    var_18 = (var_geomtype1 == var_17);
    var_16 = var_18;
    if (!var_16) {
        var_20 = (var_geomtype1 == var_19);
        var_16 = var_16 || var_20;
    }
    if (var_16) {
        // size1 = geom1.size[0]                                                              <L 2363>
        var_21 = &((var_geom1).size);
        var_24 = wp::load(var_21);
        var_23 = wp::extract(var_24, var_22);
        // full_margin1 = size1 + 0.5 * geom1.margin                                          <L 2364>
        var_26 = &((var_geom1).margin);
        var_28 = wp::load(var_26);
        var_27 = wp::mul(var_25, var_28);
        var_29 = wp::add(var_23, var_27);
        // geom1.margin = 0.0                                                                 <L 2365>
        var_geom1.margin = var_30;
        // geom1.size = wp.vec3(0.0, geom1.size[1], geom1.size[2])                            <L 2366>
        var_32 = &((var_geom1).size);
        var_35 = wp::load(var_32);
        var_34 = wp::extract(var_35, var_33);
        var_36 = &((var_geom1).size);
        var_39 = wp::load(var_36);
        var_38 = wp::extract(var_39, var_37);
        var_40 = wp::vec_t<3, wp::float32>(var_31, var_34, var_38);
        var_geom1.size = var_40;
    }
    var_41 = wp::where(var_16, var_29, var_0);
    var_42 = wp::where(var_16, var_23, var_2);
    // if geomtype2 == GeomType.SPHERE or geomtype2 == GeomType.CAPSULE:                      <L 2368>
    var_45 = (var_geomtype2 == var_44);
    var_43 = var_45;
    if (!var_43) {
        var_47 = (var_geomtype2 == var_46);
        var_43 = var_43 || var_47;
    }
    if (var_43) {
        // size2 = geom2.size[0]                                                              <L 2369>
        var_48 = &((var_geom2).size);
        var_51 = wp::load(var_48);
        var_50 = wp::extract(var_51, var_49);
        // full_margin2 = size2 + 0.5 * geom2.margin                                          <L 2370>
        var_53 = &((var_geom2).margin);
        var_55 = wp::load(var_53);
        var_54 = wp::mul(var_52, var_55);
        var_56 = wp::add(var_50, var_54);
        // geom2.margin = 0.0                                                                 <L 2371>
        var_geom2.margin = var_57;
        // geom2.size = wp.vec3(0.0, geom2.size[1], geom2.size[2])                            <L 2372>
        var_59 = &((var_geom2).size);
        var_62 = wp::load(var_59);
        var_61 = wp::extract(var_62, var_60);
        var_63 = &((var_geom2).size);
        var_66 = wp::load(var_63);
        var_65 = wp::extract(var_66, var_64);
        var_67 = wp::vec_t<3, wp::float32>(var_58, var_61, var_65);
        var_geom2.size = var_67;
    }
    var_68 = wp::where(var_43, var_56, var_1);
    var_69 = wp::where(var_43, var_50, var_3);
    // if size1 + size2 > 0.0:                                                                <L 2374>
    var_70 = wp::add(var_42, var_69);
    var_72 = (var_70 > var_71);
    if (var_72) {
        // cutoff += full_margin1 + full_margin2                                              <L 2375>
        var_73 = wp::add(var_41, var_68);
        var_74 = wp::add(var_cutoff, var_73);
        // result = gjk(tolerance, gjk_iterations, geom1, geom2, x_1, x_2, geomtype1, geomtype2, cutoff, is_discrete)       <L 2376>
        var_75 = gjk_0(var_tolerance, var_gjk_iterations, var_geom1, var_geom2, var_x_1, var_x_2, var_geomtype1, var_geomtype2, var_74, var_5);
        // geom1.index = result.index1                                                        <L 2377>
        var_76 = &((var_75).index1);
        var_78 = wp::load(var_76);
        var_77 = wp::copy(var_78);
        var_geom1.index = var_77;
        // geom2.index = result.index2                                                        <L 2378>
        var_79 = &((var_75).index2);
        var_81 = wp::load(var_79);
        var_80 = wp::copy(var_81);
        var_geom2.index = var_80;
        // if result.dist > tolerance:                                                        <L 2381>
        var_82 = &((var_75).dist);
        var_84 = wp::load(var_82);
        var_83 = (var_84 > var_tolerance);
        if (var_83) {
            // if result.dist == FLOAT_MAX:                                                   <L 2382>
            var_85 = &((var_75).dist);
            var_88 = wp::load(var_85);
            var_87 = (var_88 == var_86);
            if (var_87) {
                // return False, result.dist, 1, result.x1, result.x2, empty, geom1, geom2       <L 2383>
                var_90 = &((var_75).dist);
                var_92 = &((var_75).x1);
                var_93 = &((var_75).x2);
                var_95 = wp::load(var_90);
                var_94 = wp::copy(var_95);
                var_97 = wp::load(var_92);
                var_96 = wp::copy(var_97);
                var_99 = wp::load(var_93);
                var_98 = wp::copy(var_99);
                ret_0 = var_89;
                ret_1 = var_94;
                ret_2 = var_91;
                ret_3 = var_96;
                ret_4 = var_98;
                ret_5 = var_4;
                ret_6 = var_geom1;
                ret_7 = var_geom2;
                return;
            }
            // dist, x1, x2 = _inflate(result, geom1, geom2, geomtype1, geomtype2, full_margin1, full_margin2)       <L 2384>
            _inflate_0(var_75, var_geom1, var_geom2, var_geomtype1, var_geomtype2, var_41, var_68, var_100, var_101, var_102);
            // return False, dist, 1, x1, x2, empty, geom1, geom2                             <L 2385>
            ret_0 = var_103;
            ret_1 = var_100;
            ret_2 = var_104;
            ret_3 = var_101;
            ret_4 = var_102;
            ret_5 = var_4;
            ret_6 = var_geom1;
            ret_7 = var_geom2;
            return;
        }
        // geom1.margin = full_margin1 - size1                                                <L 2388>
        var_105 = wp::sub(var_41, var_42);
        var_geom1.margin = var_105;
        // geom1.size = wp.vec3(size1, geom1.size[1], geom1.size[2])                          <L 2389>
        var_106 = &((var_geom1).size);
        var_109 = wp::load(var_106);
        var_108 = wp::extract(var_109, var_107);
        var_110 = &((var_geom1).size);
        var_113 = wp::load(var_110);
        var_112 = wp::extract(var_113, var_111);
        var_114 = wp::vec_t<3, wp::float32>(var_42, var_108, var_112);
        var_geom1.size = var_114;
        // geom2.margin = full_margin2 - size2                                                <L 2390>
        var_115 = wp::sub(var_68, var_69);
        var_geom2.margin = var_115;
        // geom2.size = wp.vec3(size2, geom2.size[1], geom2.size[2])                          <L 2391>
        var_116 = &((var_geom2).size);
        var_119 = wp::load(var_116);
        var_118 = wp::extract(var_119, var_117);
        var_120 = &((var_geom2).size);
        var_123 = wp::load(var_120);
        var_122 = wp::extract(var_123, var_121);
        var_124 = wp::vec_t<3, wp::float32>(var_69, var_118, var_122);
        var_geom2.size = var_124;
        // cutoff -= full_margin1 + full_margin2                                              <L 2392>
        var_125 = wp::add(var_41, var_68);
        var_126 = wp::sub(var_74, var_125);
    }
    var_127 = wp::where(var_72, var_126, var_cutoff);
    // result = gjk(tolerance, gjk_iterations, geom1, geom2, x_1, x_2, geomtype1, geomtype2, cutoff, is_discrete)       <L 2394>
    var_128 = gjk_0(var_tolerance, var_gjk_iterations, var_geom1, var_geom2, var_x_1, var_x_2, var_geomtype1, var_geomtype2, var_127, var_5);
    // geom1.index = result.index1                                                            <L 2395>
    var_129 = &((var_128).index1);
    var_131 = wp::load(var_129);
    var_130 = wp::copy(var_131);
    var_geom1.index = var_130;
    // geom2.index = result.index2                                                            <L 2396>
    var_132 = &((var_128).index2);
    var_134 = wp::load(var_132);
    var_133 = wp::copy(var_134);
    var_geom2.index = var_133;
    // if result.dist > tolerance or result.dim < 2:                                          <L 2399>
    var_136 = &((var_128).dist);
    var_138 = wp::load(var_136);
    var_137 = (var_138 > var_tolerance);
    var_135 = var_137;
    if (!var_135) {
        var_139 = &((var_128).dim);
        var_142 = wp::load(var_139);
        var_141 = (var_142 < var_140);
        var_135 = var_135 || var_141;
    }
    if (var_135) {
        // return False, result.dist, 1, result.x1, result.x2, empty, geom1, geom2            <L 2400>
        var_144 = &((var_128).dist);
        var_146 = &((var_128).x1);
        var_147 = &((var_128).x2);
        var_149 = wp::load(var_144);
        var_148 = wp::copy(var_149);
        var_151 = wp::load(var_146);
        var_150 = wp::copy(var_151);
        var_153 = wp::load(var_147);
        var_152 = wp::copy(var_153);
        ret_0 = var_143;
        ret_1 = var_148;
        ret_2 = var_145;
        ret_3 = var_150;
        ret_4 = var_152;
        ret_5 = var_4;
        ret_6 = var_geom1;
        ret_7 = var_geom2;
        return;
    }
    // return True, result.dist, 1, result.x1, result.x2, result, geom1, geom2                <L 2402>
    var_155 = &((var_128).dist);
    var_157 = &((var_128).x1);
    var_158 = &((var_128).x2);
    var_160 = wp::load(var_155);
    var_159 = wp::copy(var_160);
    var_162 = wp::load(var_157);
    var_161 = wp::copy(var_162);
    var_164 = wp::load(var_158);
    var_163 = wp::copy(var_164);
    ret_0 = var_154;
    ret_1 = var_159;
    ret_2 = var_156;
    ret_3 = var_161;
    ret_4 = var_163;
    ret_5 = var_128;
    ret_6 = var_geom1;
    ret_7 = var_geom2;
    return;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:875
static CUDA_CALLABLE wp::mat_t<3, 3, wp::float32> _rotmat_0(
    wp::vec_t<3, wp::float32> var_axis)
{
    //---------
    // primal vars
    wp::float32 var_0;
    const wp::int32 var_1 = 0;
    wp::float32 var_2;
    wp::float32 var_3;
    const wp::int32 var_4 = 1;
    wp::float32 var_5;
    wp::float32 var_6;
    const wp::int32 var_7 = 2;
    wp::float32 var_8;
    wp::float32 var_9;
    const wp::float32 var_10 = 0.86602540378;
    const wp::float32 var_11 = -0.5;
    wp::mat_t<3, 3, wp::float32> var_12;
    wp::float32 var_13;
    const wp::float32 var_14 = 1.0;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    const wp::int32 var_18 = 0;
    const wp::int32 var_19 = 0;
    wp::float32 var_20;
    const wp::float32 var_21 = 1.0;
    wp::float32 var_22;
    wp::float32 var_23;
    wp::float32 var_24;
    wp::float32 var_25;
    const wp::int32 var_26 = 0;
    const wp::int32 var_27 = 1;
    wp::float32 var_28;
    const wp::float32 var_29 = 1.0;
    wp::float32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    const wp::int32 var_34 = 0;
    const wp::int32 var_35 = 2;
    wp::float32 var_36;
    const wp::float32 var_37 = 1.0;
    wp::float32 var_38;
    wp::float32 var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    const wp::int32 var_42 = 1;
    const wp::int32 var_43 = 0;
    wp::float32 var_44;
    const wp::float32 var_45 = 1.0;
    wp::float32 var_46;
    wp::float32 var_47;
    wp::float32 var_48;
    const wp::int32 var_49 = 1;
    const wp::int32 var_50 = 1;
    wp::float32 var_51;
    const wp::float32 var_52 = 1.0;
    wp::float32 var_53;
    wp::float32 var_54;
    wp::float32 var_55;
    wp::float32 var_56;
    const wp::int32 var_57 = 1;
    const wp::int32 var_58 = 2;
    wp::float32 var_59;
    const wp::float32 var_60 = 1.0;
    wp::float32 var_61;
    wp::float32 var_62;
    wp::float32 var_63;
    wp::float32 var_64;
    const wp::int32 var_65 = 2;
    const wp::int32 var_66 = 0;
    wp::float32 var_67;
    const wp::float32 var_68 = 1.0;
    wp::float32 var_69;
    wp::float32 var_70;
    wp::float32 var_71;
    wp::float32 var_72;
    const wp::int32 var_73 = 2;
    const wp::int32 var_74 = 1;
    wp::float32 var_75;
    const wp::float32 var_76 = 1.0;
    wp::float32 var_77;
    wp::float32 var_78;
    wp::float32 var_79;
    const wp::int32 var_80 = 2;
    const wp::int32 var_81 = 2;
    //---------
    // forward
    // def _rotmat(axis: wp.vec3) -> wp.mat33:                                                <L 876>
    // n = wp.norm_l2(axis)                                                                   <L 877>
    var_0 = norm_l2_0(var_axis);
    // u1 = axis[0] / n                                                                       <L 878>
    var_2 = wp::extract(var_axis, var_1);
    var_3 = wp::div(var_2, var_0);
    // u2 = axis[1] / n                                                                       <L 879>
    var_5 = wp::extract(var_axis, var_4);
    var_6 = wp::div(var_5, var_0);
    // u3 = axis[2] / n                                                                       <L 880>
    var_8 = wp::extract(var_axis, var_7);
    var_9 = wp::div(var_8, var_0);
    // sin = 0.86602540378  # sin(120 deg)                                                    <L 882>
    // cos = -0.5  # cos(120 deg)                                                             <L 883>
    // R = wp.mat33()                                                                         <L 884>
    var_12 = wp::mat_t<3, 3, wp::float32>();
    // R[0, 0] = cos + u1 * u1 * (1.0 - cos)                                                  <L 885>
    var_13 = wp::mul(var_3, var_3);
    var_15 = wp::sub(var_14, var_11);
    var_16 = wp::mul(var_13, var_15);
    var_17 = wp::add(var_11, var_16);
    wp::assign_inplace(var_12, var_18, var_19, var_17);
    // R[0, 1] = u1 * u2 * (1.0 - cos) - u3 * sin                                             <L 886>
    var_20 = wp::mul(var_3, var_6);
    var_22 = wp::sub(var_21, var_11);
    var_23 = wp::mul(var_20, var_22);
    var_24 = wp::mul(var_9, var_10);
    var_25 = wp::sub(var_23, var_24);
    wp::assign_inplace(var_12, var_26, var_27, var_25);
    // R[0, 2] = u1 * u3 * (1.0 - cos) + u2 * sin                                             <L 887>
    var_28 = wp::mul(var_3, var_9);
    var_30 = wp::sub(var_29, var_11);
    var_31 = wp::mul(var_28, var_30);
    var_32 = wp::mul(var_6, var_10);
    var_33 = wp::add(var_31, var_32);
    wp::assign_inplace(var_12, var_34, var_35, var_33);
    // R[1, 0] = u2 * u1 * (1.0 - cos) + u3 * sin                                             <L 888>
    var_36 = wp::mul(var_6, var_3);
    var_38 = wp::sub(var_37, var_11);
    var_39 = wp::mul(var_36, var_38);
    var_40 = wp::mul(var_9, var_10);
    var_41 = wp::add(var_39, var_40);
    wp::assign_inplace(var_12, var_42, var_43, var_41);
    // R[1, 1] = cos + u2 * u2 * (1.0 - cos)                                                  <L 889>
    var_44 = wp::mul(var_6, var_6);
    var_46 = wp::sub(var_45, var_11);
    var_47 = wp::mul(var_44, var_46);
    var_48 = wp::add(var_11, var_47);
    wp::assign_inplace(var_12, var_49, var_50, var_48);
    // R[1, 2] = u2 * u3 * (1.0 - cos) - u1 * sin                                             <L 890>
    var_51 = wp::mul(var_6, var_9);
    var_53 = wp::sub(var_52, var_11);
    var_54 = wp::mul(var_51, var_53);
    var_55 = wp::mul(var_3, var_10);
    var_56 = wp::sub(var_54, var_55);
    wp::assign_inplace(var_12, var_57, var_58, var_56);
    // R[2, 0] = u1 * u3 * (1.0 - cos) - u2 * sin                                             <L 891>
    var_59 = wp::mul(var_3, var_9);
    var_61 = wp::sub(var_60, var_11);
    var_62 = wp::mul(var_59, var_61);
    var_63 = wp::mul(var_6, var_10);
    var_64 = wp::sub(var_62, var_63);
    wp::assign_inplace(var_12, var_65, var_66, var_64);
    // R[2, 1] = u2 * u3 * (1.0 - cos) + u1 * sin                                             <L 892>
    var_67 = wp::mul(var_6, var_9);
    var_69 = wp::sub(var_68, var_11);
    var_70 = wp::mul(var_67, var_69);
    var_71 = wp::mul(var_3, var_10);
    var_72 = wp::add(var_70, var_71);
    wp::assign_inplace(var_12, var_73, var_74, var_72);
    // R[2, 2] = cos + u3 * u3 * (1.0 - cos)                                                  <L 893>
    var_75 = wp::mul(var_9, var_9);
    var_77 = wp::sub(var_76, var_11);
    var_78 = wp::mul(var_75, var_77);
    var_79 = wp::add(var_11, var_78);
    wp::assign_inplace(var_12, var_80, var_81, var_79);
    // return R                                                                               <L 894>
    return var_12;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:251
static CUDA_CALLABLE void _epa_support_0(
    Polytope_10582b13 var_pt,
    wp::int32 var_idx,
    Geom_3242f8a8 var_geom1,
    Geom_3242f8a8 var_geom2,
    wp::int32 var_geom1_type,
    wp::int32 var_geom2_type,
    wp::vec_t<3, wp::float32> var_dir,
    wp::int32 & ret_0,
    wp::int32 & ret_1)
{
    //---------
    // primal vars
    SupportPoint_e82efc60 var_0;
    wp::vec_t<3, wp::float32>* var_1;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_2;
    const wp::int32 var_3 = 2;
    wp::int32 var_4;
    wp::array_t<wp::vec_t<3, wp::float32>> var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::int32* var_7;
    wp::array_t<wp::int32>* var_8;
    const wp::int32 var_9 = 2;
    wp::int32 var_10;
    wp::array_t<wp::int32> var_11;
    wp::int32 var_12;
    wp::int32* var_13;
    wp::int32 var_14;
    wp::int32 var_15;
    wp::vec_t<3, wp::float32> var_16;
    SupportPoint_e82efc60 var_17;
    wp::vec_t<3, wp::float32>* var_18;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_19;
    const wp::int32 var_20 = 2;
    wp::int32 var_21;
    const wp::int32 var_22 = 1;
    wp::int32 var_23;
    wp::array_t<wp::vec_t<3, wp::float32>> var_24;
    wp::vec_t<3, wp::float32> var_25;
    wp::int32* var_26;
    wp::array_t<wp::int32>* var_27;
    const wp::int32 var_28 = 2;
    wp::int32 var_29;
    const wp::int32 var_30 = 1;
    wp::int32 var_31;
    wp::array_t<wp::int32> var_32;
    wp::int32 var_33;
    wp::int32* var_34;
    wp::int32 var_35;
    wp::int32 var_36;
    //---------
    // forward
    // def _epa_support(                                                                      <L 252>
    // sp = support(geom1, geom1_type, dir)                                                   <L 255>
    var_0 = support_0(var_geom1, var_geom1_type, var_dir);
    // pt.vert[2 * idx] = sp.point                                                            <L 256>
    var_1 = &((var_0).point);
    var_2 = &((var_pt).vert);
    var_4 = wp::mul(var_3, var_idx);
    var_5 = wp::load(var_2);
    var_6 = wp::load(var_1);
    wp::array_store(var_5, var_4, var_6);
    // pt.vert_index[2 * idx] = sp.vertex_index                                               <L 257>
    var_7 = &((var_0).vertex_index);
    var_8 = &((var_pt).vert_index);
    var_10 = wp::mul(var_9, var_idx);
    var_11 = wp::load(var_8);
    var_12 = wp::load(var_7);
    wp::array_store(var_11, var_10, var_12);
    // index1 = sp.cached_index                                                               <L 258>
    var_13 = &((var_0).cached_index);
    var_15 = wp::load(var_13);
    var_14 = wp::copy(var_15);
    // sp = support(geom2, geom2_type, -dir)                                                  <L 260>
    var_16 = wp::neg(var_dir);
    var_17 = support_0(var_geom2, var_geom2_type, var_16);
    // pt.vert[2 * idx + 1] = sp.point                                                        <L 261>
    var_18 = &((var_17).point);
    var_19 = &((var_pt).vert);
    var_21 = wp::mul(var_20, var_idx);
    var_23 = wp::add(var_21, var_22);
    var_24 = wp::load(var_19);
    var_25 = wp::load(var_18);
    wp::array_store(var_24, var_23, var_25);
    // pt.vert_index[2 * idx + 1] = sp.vertex_index                                           <L 262>
    var_26 = &((var_17).vertex_index);
    var_27 = &((var_pt).vert_index);
    var_29 = wp::mul(var_28, var_idx);
    var_31 = wp::add(var_29, var_30);
    var_32 = wp::load(var_27);
    var_33 = wp::load(var_26);
    wp::array_store(var_32, var_31, var_33);
    // index2 = sp.cached_index                                                               <L 263>
    var_34 = &((var_17).cached_index);
    var_36 = wp::load(var_34);
    var_35 = wp::copy(var_36);
    // return index1, index2                                                                  <L 265>
    ret_0 = var_14;
    ret_1 = var_35;
    return;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:225
static CUDA_CALLABLE wp::float32 _attach_face_0(
    Polytope_10582b13 var_pt,
    wp::int32 var_idx,
    wp::int32 var_v1,
    wp::int32 var_v2,
    wp::int32 var_v3)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::array_t<wp::int32>* var_1;
    wp::shape_t* var_2;
    const wp::int32 var_3 = 0;
    wp::int32 var_4;
    wp::shape_t var_5;
    bool var_6;
    wp::int32 var_7;
    const wp::float32 var_8 = 0.0;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_9;
    const wp::int32 var_10 = 2;
    wp::int32 var_11;
    wp::vec_t<3, wp::float32>* var_12;
    wp::array_t<wp::vec_t<3, wp::float32>> var_13;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_14;
    const wp::int32 var_15 = 2;
    wp::int32 var_16;
    const wp::int32 var_17 = 1;
    wp::int32 var_18;
    wp::vec_t<3, wp::float32>* var_19;
    wp::array_t<wp::vec_t<3, wp::float32>> var_20;
    wp::vec_t<3, wp::float32> var_21;
    wp::vec_t<3, wp::float32> var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_24;
    const wp::int32 var_25 = 2;
    wp::int32 var_26;
    wp::vec_t<3, wp::float32>* var_27;
    wp::array_t<wp::vec_t<3, wp::float32>> var_28;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_29;
    const wp::int32 var_30 = 2;
    wp::int32 var_31;
    const wp::int32 var_32 = 1;
    wp::int32 var_33;
    wp::vec_t<3, wp::float32>* var_34;
    wp::array_t<wp::vec_t<3, wp::float32>> var_35;
    wp::vec_t<3, wp::float32> var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_39;
    const wp::int32 var_40 = 2;
    wp::int32 var_41;
    wp::vec_t<3, wp::float32>* var_42;
    wp::array_t<wp::vec_t<3, wp::float32>> var_43;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_44;
    const wp::int32 var_45 = 2;
    wp::int32 var_46;
    const wp::int32 var_47 = 1;
    wp::int32 var_48;
    wp::vec_t<3, wp::float32>* var_49;
    wp::array_t<wp::vec_t<3, wp::float32>> var_50;
    wp::vec_t<3, wp::float32> var_51;
    wp::vec_t<3, wp::float32> var_52;
    wp::vec_t<3, wp::float32> var_53;
    wp::vec_t<3, wp::float32> var_54;
    wp::int32 var_55;
    const wp::float32 var_56 = 0.0;
    wp::vec_t<3, wp::float32>* var_57;
    wp::vec_t<3, wp::float32> var_58;
    wp::vec_t<3, wp::float32> var_59;
    wp::float32 var_60;
    const wp::float32 var_61 = 0.0;
    bool var_62;
    wp::vec_t<3, wp::float32> var_63;
    wp::vec_t<3, wp::float32> var_64;
    const wp::int32 var_65 = 10;
    wp::int32 var_66;
    wp::int32 var_67;
    const wp::int32 var_68 = 20;
    wp::int32 var_69;
    wp::int32 var_70;
    wp::array_t<wp::int32>* var_71;
    wp::array_t<wp::int32> var_72;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_73;
    wp::array_t<wp::vec_t<3, wp::float32>> var_74;
    wp::float32 var_75;
    wp::array_t<wp::float32>* var_76;
    wp::array_t<wp::float32> var_77;
    wp::array_t<wp::float32>* var_78;
    wp::float32* var_79;
    wp::array_t<wp::float32> var_80;
    wp::float32 var_81;
    wp::float32 var_82;
    //---------
    // forward
    // def _attach_face(pt: Polytope, idx: int, v1: int, v2: int, v3: int) -> float:          <L 226>
    // if pt.nface == pt.face.shape[0]:                                                       <L 228>
    var_0 = &((var_pt).nface);
    var_1 = &((var_pt).face);
    var_2 = &(var_1->shape);
    var_5 = wp::load(var_2);
    var_4 = wp::extract(var_5, var_3);
    var_7 = wp::load(var_0);
    var_6 = (var_7 == var_4);
    if (var_6) {
        // return 0.0                                                                         <L 229>
        return var_8;
    }
    // p1 = pt.vert[2 * v1] - pt.vert[2 * v1 + 1]                                             <L 232>
    var_9 = &((var_pt).vert);
    var_11 = wp::mul(var_10, var_v1);
    var_13 = wp::load(var_9);
    var_12 = wp::address(var_13, var_11);
    var_14 = &((var_pt).vert);
    var_16 = wp::mul(var_15, var_v1);
    var_18 = wp::add(var_16, var_17);
    var_20 = wp::load(var_14);
    var_19 = wp::address(var_20, var_18);
    var_22 = wp::load(var_12);
    var_23 = wp::load(var_19);
    var_21 = wp::sub(var_22, var_23);
    // p2 = pt.vert[2 * v2] - pt.vert[2 * v2 + 1]                                             <L 233>
    var_24 = &((var_pt).vert);
    var_26 = wp::mul(var_25, var_v2);
    var_28 = wp::load(var_24);
    var_27 = wp::address(var_28, var_26);
    var_29 = &((var_pt).vert);
    var_31 = wp::mul(var_30, var_v2);
    var_33 = wp::add(var_31, var_32);
    var_35 = wp::load(var_29);
    var_34 = wp::address(var_35, var_33);
    var_37 = wp::load(var_27);
    var_38 = wp::load(var_34);
    var_36 = wp::sub(var_37, var_38);
    // p3 = pt.vert[2 * v3] - pt.vert[2 * v3 + 1]                                             <L 234>
    var_39 = &((var_pt).vert);
    var_41 = wp::mul(var_40, var_v3);
    var_43 = wp::load(var_39);
    var_42 = wp::address(var_43, var_41);
    var_44 = &((var_pt).vert);
    var_46 = wp::mul(var_45, var_v3);
    var_48 = wp::add(var_46, var_47);
    var_50 = wp::load(var_44);
    var_49 = wp::address(var_50, var_48);
    var_52 = wp::load(var_42);
    var_53 = wp::load(var_49);
    var_51 = wp::sub(var_52, var_53);
    // r, ret = _project_origin_plane(p3, p2, p1)                                             <L 235>
    _project_origin_plane_0(var_51, var_36, var_21, var_54, var_55);
    // if ret:                                                                                <L 236>
    if (var_55) {
        // return 0.0                                                                         <L 237>
        return var_56;
    }
    // if wp.dot(r, p1 - pt.center) < 0.0:                                                    <L 240>
    var_57 = &((var_pt).center);
    var_59 = wp::load(var_57);
    var_58 = wp::sub(var_21, var_59);
    var_60 = wp::dot(var_54, var_58);
    var_62 = (var_60 < var_61);
    if (var_62) {
        // r = -r                                                                             <L 241>
        var_63 = wp::neg(var_54);
    }
    var_64 = wp::where(var_62, var_63, var_54);
    // face = v1 + (v2 << 10) + (v3 << 20)                                                    <L 243>
    var_66 = wp::lshift(var_v2, var_65);
    var_67 = wp::add(var_v1, var_66);
    var_69 = wp::lshift(var_v3, var_68);
    var_70 = wp::add(var_67, var_69);
    // pt.face[idx] = face                                                                    <L 244>
    var_71 = &((var_pt).face);
    var_72 = wp::load(var_71);
    wp::array_store(var_72, var_idx, var_70);
    // pt.face_pr[idx] = r                                                                    <L 245>
    var_73 = &((var_pt).face_pr);
    var_74 = wp::load(var_73);
    wp::array_store(var_74, var_idx, var_64);
    // pt.face_norm2[idx] = wp.dot(r, r)                                                      <L 247>
    var_75 = wp::dot(var_64, var_64);
    var_76 = &((var_pt).face_norm2);
    var_77 = wp::load(var_76);
    wp::array_store(var_77, var_idx, var_75);
    // return pt.face_norm2[idx]                                                              <L 248>
    var_78 = &((var_pt).face_norm2);
    var_80 = wp::load(var_78);
    var_79 = wp::address(var_80, var_idx);
    var_82 = wp::load(var_79);
    var_81 = wp::copy(var_82);
    return var_81;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:836
static CUDA_CALLABLE GJKResult_28609055 _replace_simplex3_0(
    Polytope_10582b13 var_pt,
    wp::int32 var_v1,
    wp::int32 var_v2,
    wp::int32 var_v3)
{
    //---------
    // primal vars
    GJKResult_28609055 var_0;
    wp::mat_t<4, 3, wp::float32> var_1;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_2;
    const wp::int32 var_3 = 2;
    wp::int32 var_4;
    wp::vec_t<3, wp::float32>* var_5;
    wp::array_t<wp::vec_t<3, wp::float32>> var_6;
    const wp::int32 var_7 = 0;
    wp::vec_t<3, wp::float32> var_8;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_9;
    const wp::int32 var_10 = 2;
    wp::int32 var_11;
    wp::vec_t<3, wp::float32>* var_12;
    wp::array_t<wp::vec_t<3, wp::float32>> var_13;
    const wp::int32 var_14 = 1;
    wp::vec_t<3, wp::float32> var_15;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_16;
    const wp::int32 var_17 = 2;
    wp::int32 var_18;
    wp::vec_t<3, wp::float32>* var_19;
    wp::array_t<wp::vec_t<3, wp::float32>> var_20;
    const wp::int32 var_21 = 2;
    wp::vec_t<3, wp::float32> var_22;
    wp::mat_t<4, 3, wp::float32> var_23;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_24;
    const wp::int32 var_25 = 2;
    wp::int32 var_26;
    const wp::int32 var_27 = 1;
    wp::int32 var_28;
    wp::vec_t<3, wp::float32>* var_29;
    wp::array_t<wp::vec_t<3, wp::float32>> var_30;
    const wp::int32 var_31 = 0;
    wp::vec_t<3, wp::float32> var_32;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_33;
    const wp::int32 var_34 = 2;
    wp::int32 var_35;
    const wp::int32 var_36 = 1;
    wp::int32 var_37;
    wp::vec_t<3, wp::float32>* var_38;
    wp::array_t<wp::vec_t<3, wp::float32>> var_39;
    const wp::int32 var_40 = 1;
    wp::vec_t<3, wp::float32> var_41;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_42;
    const wp::int32 var_43 = 2;
    wp::int32 var_44;
    const wp::int32 var_45 = 1;
    wp::int32 var_46;
    wp::vec_t<3, wp::float32>* var_47;
    wp::array_t<wp::vec_t<3, wp::float32>> var_48;
    const wp::int32 var_49 = 2;
    wp::vec_t<3, wp::float32> var_50;
    wp::mat_t<4, 3, wp::float32> var_51;
    const wp::int32 var_52 = 0;
    wp::vec_t<3, wp::float32> var_53;
    const wp::int32 var_54 = 0;
    wp::vec_t<3, wp::float32> var_55;
    wp::vec_t<3, wp::float32> var_56;
    const wp::int32 var_57 = 0;
    const wp::int32 var_58 = 1;
    wp::vec_t<3, wp::float32> var_59;
    const wp::int32 var_60 = 1;
    wp::vec_t<3, wp::float32> var_61;
    wp::vec_t<3, wp::float32> var_62;
    const wp::int32 var_63 = 1;
    const wp::int32 var_64 = 2;
    wp::vec_t<3, wp::float32> var_65;
    const wp::int32 var_66 = 2;
    wp::vec_t<3, wp::float32> var_67;
    wp::vec_t<3, wp::float32> var_68;
    const wp::int32 var_69 = 2;
    wp::vec_t<4, wp::int32> var_70;
    wp::array_t<wp::int32>* var_71;
    const wp::int32 var_72 = 2;
    wp::int32 var_73;
    wp::int32* var_74;
    wp::array_t<wp::int32> var_75;
    const wp::int32 var_76 = 0;
    wp::int32 var_77;
    wp::array_t<wp::int32>* var_78;
    const wp::int32 var_79 = 2;
    wp::int32 var_80;
    wp::int32* var_81;
    wp::array_t<wp::int32> var_82;
    const wp::int32 var_83 = 1;
    wp::int32 var_84;
    wp::array_t<wp::int32>* var_85;
    const wp::int32 var_86 = 2;
    wp::int32 var_87;
    wp::int32* var_88;
    wp::array_t<wp::int32> var_89;
    const wp::int32 var_90 = 2;
    wp::int32 var_91;
    wp::vec_t<4, wp::int32> var_92;
    wp::array_t<wp::int32>* var_93;
    const wp::int32 var_94 = 2;
    wp::int32 var_95;
    const wp::int32 var_96 = 1;
    wp::int32 var_97;
    wp::int32* var_98;
    wp::array_t<wp::int32> var_99;
    const wp::int32 var_100 = 0;
    wp::int32 var_101;
    wp::array_t<wp::int32>* var_102;
    const wp::int32 var_103 = 2;
    wp::int32 var_104;
    const wp::int32 var_105 = 1;
    wp::int32 var_106;
    wp::int32* var_107;
    wp::array_t<wp::int32> var_108;
    const wp::int32 var_109 = 1;
    wp::int32 var_110;
    wp::array_t<wp::int32>* var_111;
    const wp::int32 var_112 = 2;
    wp::int32 var_113;
    const wp::int32 var_114 = 1;
    wp::int32 var_115;
    wp::int32* var_116;
    wp::array_t<wp::int32> var_117;
    const wp::int32 var_118 = 2;
    wp::int32 var_119;
    //---------
    // forward
    // def _replace_simplex3(pt: Polytope, v1: int, v2: int, v3: int) -> GJKResult:           <L 837>
    // result = GJKResult()                                                                   <L 838>
    var_0 = GJKResult_28609055();
    // simplex1 = mat43()                                                                     <L 841>
    var_1 = wp::mat_t<4, 3, wp::float32>();
    // simplex1[0] = pt.vert[2 * v1]                                                          <L 842>
    var_2 = &((var_pt).vert);
    var_4 = wp::mul(var_3, var_v1);
    var_6 = wp::load(var_2);
    var_5 = wp::address(var_6, var_4);
    var_8 = wp::load(var_5);
    wp::assign_inplace(var_1, var_7, var_8);
    // simplex1[1] = pt.vert[2 * v2]                                                          <L 843>
    var_9 = &((var_pt).vert);
    var_11 = wp::mul(var_10, var_v2);
    var_13 = wp::load(var_9);
    var_12 = wp::address(var_13, var_11);
    var_15 = wp::load(var_12);
    wp::assign_inplace(var_1, var_14, var_15);
    // simplex1[2] = pt.vert[2 * v3]                                                          <L 844>
    var_16 = &((var_pt).vert);
    var_18 = wp::mul(var_17, var_v3);
    var_20 = wp::load(var_16);
    var_19 = wp::address(var_20, var_18);
    var_22 = wp::load(var_19);
    wp::assign_inplace(var_1, var_21, var_22);
    // simplex2 = mat43()                                                                     <L 846>
    var_23 = wp::mat_t<4, 3, wp::float32>();
    // simplex2[0] = pt.vert[2 * v1 + 1]                                                      <L 847>
    var_24 = &((var_pt).vert);
    var_26 = wp::mul(var_25, var_v1);
    var_28 = wp::add(var_26, var_27);
    var_30 = wp::load(var_24);
    var_29 = wp::address(var_30, var_28);
    var_32 = wp::load(var_29);
    wp::assign_inplace(var_23, var_31, var_32);
    // simplex2[1] = pt.vert[2 * v2 + 1]                                                      <L 848>
    var_33 = &((var_pt).vert);
    var_35 = wp::mul(var_34, var_v2);
    var_37 = wp::add(var_35, var_36);
    var_39 = wp::load(var_33);
    var_38 = wp::address(var_39, var_37);
    var_41 = wp::load(var_38);
    wp::assign_inplace(var_23, var_40, var_41);
    // simplex2[2] = pt.vert[2 * v3 + 1]                                                      <L 849>
    var_42 = &((var_pt).vert);
    var_44 = wp::mul(var_43, var_v3);
    var_46 = wp::add(var_44, var_45);
    var_48 = wp::load(var_42);
    var_47 = wp::address(var_48, var_46);
    var_50 = wp::load(var_47);
    wp::assign_inplace(var_23, var_49, var_50);
    // simplex = mat43()                                                                      <L 851>
    var_51 = wp::mat_t<4, 3, wp::float32>();
    // simplex[0] = simplex1[0] - simplex2[0]                                                 <L 852>
    var_53 = wp::extract(var_1, var_52);
    var_55 = wp::extract(var_23, var_54);
    var_56 = wp::sub(var_53, var_55);
    wp::assign_inplace(var_51, var_57, var_56);
    // simplex[1] = simplex1[1] - simplex2[1]                                                 <L 853>
    var_59 = wp::extract(var_1, var_58);
    var_61 = wp::extract(var_23, var_60);
    var_62 = wp::sub(var_59, var_61);
    wp::assign_inplace(var_51, var_63, var_62);
    // simplex[2] = simplex1[2] - simplex2[2]                                                 <L 854>
    var_65 = wp::extract(var_1, var_64);
    var_67 = wp::extract(var_23, var_66);
    var_68 = wp::sub(var_65, var_67);
    wp::assign_inplace(var_51, var_69, var_68);
    // simplex_index1 = wp.vec4i()                                                            <L 856>
    var_70 = wp::vec_t<4, wp::int32>();
    // simplex_index1[0] = pt.vert_index[2 * v1]                                              <L 857>
    var_71 = &((var_pt).vert_index);
    var_73 = wp::mul(var_72, var_v1);
    var_75 = wp::load(var_71);
    var_74 = wp::address(var_75, var_73);
    var_77 = wp::load(var_74);
    wp::assign_inplace(var_70, var_76, var_77);
    // simplex_index1[1] = pt.vert_index[2 * v2]                                              <L 858>
    var_78 = &((var_pt).vert_index);
    var_80 = wp::mul(var_79, var_v2);
    var_82 = wp::load(var_78);
    var_81 = wp::address(var_82, var_80);
    var_84 = wp::load(var_81);
    wp::assign_inplace(var_70, var_83, var_84);
    // simplex_index1[2] = pt.vert_index[2 * v3]                                              <L 859>
    var_85 = &((var_pt).vert_index);
    var_87 = wp::mul(var_86, var_v3);
    var_89 = wp::load(var_85);
    var_88 = wp::address(var_89, var_87);
    var_91 = wp::load(var_88);
    wp::assign_inplace(var_70, var_90, var_91);
    // simplex_index2 = wp.vec4i()                                                            <L 861>
    var_92 = wp::vec_t<4, wp::int32>();
    // simplex_index2[0] = pt.vert_index[2 * v1 + 1]                                          <L 862>
    var_93 = &((var_pt).vert_index);
    var_95 = wp::mul(var_94, var_v1);
    var_97 = wp::add(var_95, var_96);
    var_99 = wp::load(var_93);
    var_98 = wp::address(var_99, var_97);
    var_101 = wp::load(var_98);
    wp::assign_inplace(var_92, var_100, var_101);
    // simplex_index2[1] = pt.vert_index[2 * v2 + 1]                                          <L 863>
    var_102 = &((var_pt).vert_index);
    var_104 = wp::mul(var_103, var_v2);
    var_106 = wp::add(var_104, var_105);
    var_108 = wp::load(var_102);
    var_107 = wp::address(var_108, var_106);
    var_110 = wp::load(var_107);
    wp::assign_inplace(var_92, var_109, var_110);
    // simplex_index2[2] = pt.vert_index[2 * v3 + 1]                                          <L 864>
    var_111 = &((var_pt).vert_index);
    var_113 = wp::mul(var_112, var_v3);
    var_115 = wp::add(var_113, var_114);
    var_117 = wp::load(var_111);
    var_116 = wp::address(var_117, var_115);
    var_119 = wp::load(var_116);
    wp::assign_inplace(var_92, var_118, var_119);
    // result.simplex = simplex                                                               <L 866>
    var_0.simplex = var_51;
    // result.simplex1 = simplex1                                                             <L 867>
    var_0.simplex1 = var_1;
    // result.simplex2 = simplex2                                                             <L 868>
    var_0.simplex2 = var_23;
    // result.simplex_index1 = simplex_index1                                                 <L 869>
    var_0.simplex_index1 = var_70;
    // result.simplex_index2 = simplex_index2                                                 <L 870>
    var_0.simplex_index2 = var_92;
    // return result                                                                          <L 872>
    return var_0;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:897
static CUDA_CALLABLE wp::int32 _ray_triangle_0(
    wp::vec_t<3, wp::float32> var_v1,
    wp::vec_t<3, wp::float32> var_v2,
    wp::vec_t<3, wp::float32> var_v3,
    wp::vec_t<3, wp::float32> var_v4,
    wp::vec_t<3, wp::float32> var_v5)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::float32 var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::float32 var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::float32 var_11;
    bool var_12;
    const wp::float32 var_13 = 0.0;
    bool var_14;
    const wp::float32 var_15 = 0.0;
    bool var_16;
    const wp::float32 var_17 = 0.0;
    bool var_18;
    const wp::int32 var_19 = 1;
    bool var_20;
    const wp::float32 var_21 = 0.0;
    bool var_22;
    const wp::float32 var_23 = 0.0;
    bool var_24;
    const wp::float32 var_25 = 0.0;
    bool var_26;
    const wp::int32 var_27 = -1;
    const wp::int32 var_28 = 0;
    //---------
    // forward
    // def _ray_triangle(v1: wp.vec3, v2: wp.vec3, v3: wp.vec3, v4: wp.vec3, v5: wp.vec3) -> int:       <L 898>
    // vol1 = _det3(v3 - v1, v4 - v1, v2 - v1)                                                <L 899>
    var_0 = wp::sub(var_v3, var_v1);
    var_1 = wp::sub(var_v4, var_v1);
    var_2 = wp::sub(var_v2, var_v1);
    var_3 = _det3_0(var_0, var_1, var_2);
    // vol2 = _det3(v4 - v1, v5 - v1, v2 - v1)                                                <L 900>
    var_4 = wp::sub(var_v4, var_v1);
    var_5 = wp::sub(var_v5, var_v1);
    var_6 = wp::sub(var_v2, var_v1);
    var_7 = _det3_0(var_4, var_5, var_6);
    // vol3 = _det3(v5 - v1, v3 - v1, v2 - v1)                                                <L 901>
    var_8 = wp::sub(var_v5, var_v1);
    var_9 = wp::sub(var_v3, var_v1);
    var_10 = wp::sub(var_v2, var_v1);
    var_11 = _det3_0(var_8, var_9, var_10);
    // if vol1 >= 0.0 and vol2 >= 0.0 and vol3 >= 0.0:                                        <L 903>
    var_14 = (var_3 >= var_13);
    var_12 = var_14;
    if (var_12) {
        var_16 = (var_7 >= var_15);
        var_12 = var_12 && var_16;
    }
    if (var_12) {
        var_18 = (var_11 >= var_17);
        var_12 = var_12 && var_18;
    }
    if (var_12) {
        // return 1                                                                           <L 904>
        return var_19;
    }
    // if vol1 <= 0.0 and vol2 <= 0.0 and vol3 <= 0.0:                                        <L 905>
    var_22 = (var_3 <= var_21);
    var_20 = var_22;
    if (var_20) {
        var_24 = (var_7 <= var_23);
        var_20 = var_20 && var_24;
    }
    if (var_20) {
        var_26 = (var_11 <= var_25);
        var_20 = var_20 && var_26;
    }
    if (var_20) {
        // return -1                                                                          <L 906>
        return var_27;
    }
    // return 0                                                                               <L 907>
    return var_28;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1011
static CUDA_CALLABLE void _polytope2_0(
    Polytope_10582b13 var_pt,
    wp::mat_t<4, 3, wp::float32> var_simplex,
    wp::mat_t<4, 3, wp::float32> var_simplex1,
    wp::mat_t<4, 3, wp::float32> var_simplex2,
    wp::vec_t<4, wp::int32> var_simplex_index1,
    wp::vec_t<4, wp::int32> var_simplex_index2,
    Geom_3242f8a8 var_geom1,
    Geom_3242f8a8 var_geom2,
    wp::int32 var_geomtype1,
    wp::int32 var_geomtype2,
    Polytope_10582b13 & ret_0,
    GJKResult_28609055 & ret_1)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    wp::vec_t<3, wp::float32> var_1;
    const wp::int32 var_2 = 0;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    const wp::float32 var_5 = 0.5;
    const wp::int32 var_6 = 0;
    wp::vec_t<3, wp::float32> var_7;
    const wp::int32 var_8 = 1;
    wp::vec_t<3, wp::float32> var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::vec_t<3, wp::float32> var_11;
    const wp::float32 var_12 = 1e+30;
    wp::float32 var_13;
    const wp::int32 var_14 = 0;
    const wp::int32 var_15 = 0;
    wp::float32 var_16;
    wp::float32 var_17;
    bool var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    wp::int32 var_21;
    wp::float32 var_22;
    wp::int32 var_23;
    const wp::int32 var_24 = 1;
    wp::float32 var_25;
    wp::float32 var_26;
    bool var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::int32 var_30;
    wp::float32 var_31;
    wp::int32 var_32;
    const wp::int32 var_33 = 2;
    wp::float32 var_34;
    wp::float32 var_35;
    bool var_36;
    wp::float32 var_37;
    wp::float32 var_38;
    wp::int32 var_39;
    wp::float32 var_40;
    wp::int32 var_41;
    const wp::float32 var_42 = 0.0;
    const wp::float32 var_43 = 0.0;
    const wp::float32 var_44 = 0.0;
    wp::vec_t<3, wp::float32> var_45;
    const wp::float32 var_46 = 1.0;
    wp::vec_t<3, wp::float32> var_47;
    wp::mat_t<3, 3, wp::float32> var_48;
    wp::vec_t<3, wp::float32> var_49;
    wp::vec_t<3, wp::float32> var_50;
    const wp::int32 var_51 = 0;
    wp::vec_t<3, wp::float32> var_52;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_53;
    const wp::int32 var_54 = 0;
    wp::array_t<wp::vec_t<3, wp::float32>> var_55;
    const wp::int32 var_56 = 0;
    wp::vec_t<3, wp::float32> var_57;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_58;
    const wp::int32 var_59 = 1;
    wp::array_t<wp::vec_t<3, wp::float32>> var_60;
    const wp::int32 var_61 = 1;
    wp::vec_t<3, wp::float32> var_62;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_63;
    const wp::int32 var_64 = 2;
    wp::array_t<wp::vec_t<3, wp::float32>> var_65;
    const wp::int32 var_66 = 1;
    wp::vec_t<3, wp::float32> var_67;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_68;
    const wp::int32 var_69 = 3;
    wp::array_t<wp::vec_t<3, wp::float32>> var_70;
    const wp::int32 var_71 = 0;
    wp::int32 var_72;
    wp::array_t<wp::int32>* var_73;
    const wp::int32 var_74 = 0;
    wp::array_t<wp::int32> var_75;
    const wp::int32 var_76 = 0;
    wp::int32 var_77;
    wp::array_t<wp::int32>* var_78;
    const wp::int32 var_79 = 1;
    wp::array_t<wp::int32> var_80;
    const wp::int32 var_81 = 1;
    wp::int32 var_82;
    wp::array_t<wp::int32>* var_83;
    const wp::int32 var_84 = 2;
    wp::array_t<wp::int32> var_85;
    const wp::int32 var_86 = 1;
    wp::int32 var_87;
    wp::array_t<wp::int32>* var_88;
    const wp::int32 var_89 = 3;
    wp::array_t<wp::int32> var_90;
    const wp::int32 var_91 = 2;
    wp::float32 var_92;
    wp::vec_t<3, wp::float32> var_93;
    wp::int32 var_94;
    wp::int32 var_95;
    const wp::int32 var_96 = 3;
    wp::float32 var_97;
    wp::vec_t<3, wp::float32> var_98;
    wp::int32 var_99;
    wp::int32 var_100;
    const wp::int32 var_101 = 4;
    wp::float32 var_102;
    wp::vec_t<3, wp::float32> var_103;
    wp::int32 var_104;
    wp::int32 var_105;
    const wp::int32 var_106 = 0;
    const wp::int32 var_107 = 0;
    const wp::int32 var_108 = 2;
    const wp::int32 var_109 = 3;
    wp::float32 var_110;
    const wp::float32 var_111 = 1e-10;
    bool var_112;
    const wp::int32 var_113 = -1;
    const wp::int32 var_114 = 0;
    const wp::int32 var_115 = 2;
    const wp::int32 var_116 = 3;
    GJKResult_28609055 var_117;
    const wp::int32 var_118 = 1;
    const wp::int32 var_119 = 0;
    const wp::int32 var_120 = 4;
    const wp::int32 var_121 = 2;
    wp::float32 var_122;
    bool var_123;
    const wp::int32 var_124 = -1;
    const wp::int32 var_125 = 0;
    const wp::int32 var_126 = 4;
    const wp::int32 var_127 = 2;
    GJKResult_28609055 var_128;
    const wp::int32 var_129 = 2;
    const wp::int32 var_130 = 0;
    const wp::int32 var_131 = 3;
    const wp::int32 var_132 = 4;
    wp::float32 var_133;
    bool var_134;
    const wp::int32 var_135 = -1;
    const wp::int32 var_136 = 0;
    const wp::int32 var_137 = 3;
    const wp::int32 var_138 = 4;
    GJKResult_28609055 var_139;
    const wp::int32 var_140 = 3;
    const wp::int32 var_141 = 1;
    const wp::int32 var_142 = 3;
    const wp::int32 var_143 = 2;
    wp::float32 var_144;
    bool var_145;
    const wp::int32 var_146 = -1;
    const wp::int32 var_147 = 1;
    const wp::int32 var_148 = 3;
    const wp::int32 var_149 = 2;
    GJKResult_28609055 var_150;
    const wp::int32 var_151 = 4;
    const wp::int32 var_152 = 1;
    const wp::int32 var_153 = 2;
    const wp::int32 var_154 = 4;
    wp::float32 var_155;
    bool var_156;
    const wp::int32 var_157 = -1;
    const wp::int32 var_158 = 1;
    const wp::int32 var_159 = 2;
    const wp::int32 var_160 = 4;
    GJKResult_28609055 var_161;
    const wp::int32 var_162 = 5;
    const wp::int32 var_163 = 1;
    const wp::int32 var_164 = 4;
    const wp::int32 var_165 = 3;
    wp::float32 var_166;
    bool var_167;
    const wp::int32 var_168 = -1;
    const wp::int32 var_169 = 1;
    const wp::int32 var_170 = 4;
    const wp::int32 var_171 = 3;
    GJKResult_28609055 var_172;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_173;
    const wp::int32 var_174 = 4;
    wp::vec_t<3, wp::float32>* var_175;
    wp::array_t<wp::vec_t<3, wp::float32>> var_176;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_177;
    const wp::int32 var_178 = 5;
    wp::vec_t<3, wp::float32>* var_179;
    wp::array_t<wp::vec_t<3, wp::float32>> var_180;
    wp::vec_t<3, wp::float32> var_181;
    wp::vec_t<3, wp::float32> var_182;
    wp::vec_t<3, wp::float32> var_183;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_184;
    const wp::int32 var_185 = 6;
    wp::vec_t<3, wp::float32>* var_186;
    wp::array_t<wp::vec_t<3, wp::float32>> var_187;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_188;
    const wp::int32 var_189 = 7;
    wp::vec_t<3, wp::float32>* var_190;
    wp::array_t<wp::vec_t<3, wp::float32>> var_191;
    wp::vec_t<3, wp::float32> var_192;
    wp::vec_t<3, wp::float32> var_193;
    wp::vec_t<3, wp::float32> var_194;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_195;
    const wp::int32 var_196 = 8;
    wp::vec_t<3, wp::float32>* var_197;
    wp::array_t<wp::vec_t<3, wp::float32>> var_198;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_199;
    const wp::int32 var_200 = 9;
    wp::vec_t<3, wp::float32>* var_201;
    wp::array_t<wp::vec_t<3, wp::float32>> var_202;
    wp::vec_t<3, wp::float32> var_203;
    wp::vec_t<3, wp::float32> var_204;
    wp::vec_t<3, wp::float32> var_205;
    const wp::int32 var_206 = 0;
    wp::vec_t<3, wp::float32> var_207;
    const wp::int32 var_208 = 1;
    wp::vec_t<3, wp::float32> var_209;
    wp::int32 var_210;
    bool var_211;
    const wp::int32 var_212 = 1;
    GJKResult_28609055 var_213;
    const wp::int32 var_214 = 5;
    const wp::int32 var_215 = 6;
    const wp::int32 var_216 = 0;
    GJKResult_28609055 var_217;
    //---------
    // forward
    // def _polytope2(                                                                        <L 1012>
    // diff = simplex[1] - simplex[0]                                                         <L 1026>
    var_1 = wp::extract(var_simplex, var_0);
    var_3 = wp::extract(var_simplex, var_2);
    var_4 = wp::sub(var_1, var_3);
    // pt.center = 0.5 * (simplex[0] + simplex[1])                                            <L 1029>
    var_7 = wp::extract(var_simplex, var_6);
    var_9 = wp::extract(var_simplex, var_8);
    var_10 = wp::add(var_7, var_9);
    var_11 = wp::mul(var_5, var_10);
    var_pt.center = var_11;
    // value = FLOAT_MAX                                                                      <L 1032>
    var_13 = wp::copy(var_12);
    // index = 0                                                                              <L 1033>
    // for i in range(3):                                                                     <L 1034>
    // if wp.abs(diff[i]) < value:                                                            <L 1035>
    var_16 = wp::extract(var_4, var_15);
    var_17 = wp::abs(var_16);
    var_18 = (var_17 < var_13);
    if (var_18) {
        // value = wp.abs(diff[i])                                                            <L 1036>
        var_19 = wp::extract(var_4, var_15);
        var_20 = wp::abs(var_19);
        // index = i                                                                          <L 1037>
        var_21 = wp::copy(var_15);
    }
    var_22 = wp::where(var_18, var_20, var_13);
    var_23 = wp::where(var_18, var_21, var_14);
    // if wp.abs(diff[i]) < value:                                                            <L 1035>
    var_25 = wp::extract(var_4, var_24);
    var_26 = wp::abs(var_25);
    var_27 = (var_26 < var_22);
    if (var_27) {
        // value = wp.abs(diff[i])                                                            <L 1036>
        var_28 = wp::extract(var_4, var_24);
        var_29 = wp::abs(var_28);
        // index = i                                                                          <L 1037>
        var_30 = wp::copy(var_24);
    }
    var_31 = wp::where(var_27, var_29, var_22);
    var_32 = wp::where(var_27, var_30, var_23);
    // if wp.abs(diff[i]) < value:                                                            <L 1035>
    var_34 = wp::extract(var_4, var_33);
    var_35 = wp::abs(var_34);
    var_36 = (var_35 < var_31);
    if (var_36) {
        // value = wp.abs(diff[i])                                                            <L 1036>
        var_37 = wp::extract(var_4, var_33);
        var_38 = wp::abs(var_37);
        // index = i                                                                          <L 1037>
        var_39 = wp::copy(var_33);
    }
    var_40 = wp::where(var_36, var_38, var_31);
    var_41 = wp::where(var_36, var_39, var_32);
    // e = wp.vec3(0.0, 0.0, 0.0)                                                             <L 1040>
    var_45 = wp::vec_t<3, wp::float32>(var_42, var_43, var_44);
    // e[index] = 1.0                                                                         <L 1041>
    wp::assign_inplace(var_45, var_41, var_46);
    // d1 = wp.cross(e, diff)                                                                 <L 1042>
    var_47 = wp::cross(var_45, var_4);
    // R = _rotmat(diff)                                                                      <L 1045>
    var_48 = _rotmat_0(var_4);
    // d2 = R @ d1                                                                            <L 1046>
    var_49 = wp::mul(var_48, var_47);
    // d3 = R @ d2                                                                            <L 1047>
    var_50 = wp::mul(var_48, var_49);
    // pt.vert[0] = simplex1[0]                                                               <L 1050>
    var_52 = wp::extract(var_simplex1, var_51);
    var_53 = &((var_pt).vert);
    var_55 = wp::load(var_53);
    wp::array_store(var_55, var_54, var_52);
    // pt.vert[1] = simplex2[0]                                                               <L 1051>
    var_57 = wp::extract(var_simplex2, var_56);
    var_58 = &((var_pt).vert);
    var_60 = wp::load(var_58);
    wp::array_store(var_60, var_59, var_57);
    // pt.vert[2] = simplex1[1]                                                               <L 1052>
    var_62 = wp::extract(var_simplex1, var_61);
    var_63 = &((var_pt).vert);
    var_65 = wp::load(var_63);
    wp::array_store(var_65, var_64, var_62);
    // pt.vert[3] = simplex2[1]                                                               <L 1053>
    var_67 = wp::extract(var_simplex2, var_66);
    var_68 = &((var_pt).vert);
    var_70 = wp::load(var_68);
    wp::array_store(var_70, var_69, var_67);
    // pt.vert_index[0] = simplex_index1[0]                                                   <L 1055>
    var_72 = wp::extract(var_simplex_index1, var_71);
    var_73 = &((var_pt).vert_index);
    var_75 = wp::load(var_73);
    wp::array_store(var_75, var_74, var_72);
    // pt.vert_index[1] = simplex_index2[0]                                                   <L 1056>
    var_77 = wp::extract(var_simplex_index2, var_76);
    var_78 = &((var_pt).vert_index);
    var_80 = wp::load(var_78);
    wp::array_store(var_80, var_79, var_77);
    // pt.vert_index[2] = simplex_index1[1]                                                   <L 1057>
    var_82 = wp::extract(var_simplex_index1, var_81);
    var_83 = &((var_pt).vert_index);
    var_85 = wp::load(var_83);
    wp::array_store(var_85, var_84, var_82);
    // pt.vert_index[3] = simplex_index2[1]                                                   <L 1058>
    var_87 = wp::extract(var_simplex_index2, var_86);
    var_88 = &((var_pt).vert_index);
    var_90 = wp::load(var_88);
    wp::array_store(var_90, var_89, var_87);
    // _epa_support(pt, 2, geom1, geom2, geomtype1, geomtype2, d1 / wp.norm_l2(d1))           <L 1060>
    var_92 = norm_l2_0(var_47);
    var_93 = wp::div(var_47, var_92);
    _epa_support_0(var_pt, var_91, var_geom1, var_geom2, var_geomtype1, var_geomtype2, var_93, var_94, var_95);
    // _epa_support(pt, 3, geom1, geom2, geomtype1, geomtype2, d2 / wp.norm_l2(d2))           <L 1061>
    var_97 = norm_l2_0(var_49);
    var_98 = wp::div(var_49, var_97);
    _epa_support_0(var_pt, var_96, var_geom1, var_geom2, var_geomtype1, var_geomtype2, var_98, var_99, var_100);
    // _epa_support(pt, 4, geom1, geom2, geomtype1, geomtype2, d3 / wp.norm_l2(d3))           <L 1062>
    var_102 = norm_l2_0(var_50);
    var_103 = wp::div(var_50, var_102);
    _epa_support_0(var_pt, var_101, var_geom1, var_geom2, var_geomtype1, var_geomtype2, var_103, var_104, var_105);
    // if _attach_face(pt, 0, 0, 2, 3) < MIN_DIST2:                                           <L 1065>
    var_110 = _attach_face_0(var_pt, var_106, var_107, var_108, var_109);
    var_112 = (var_110 < var_111);
    if (var_112) {
        // pt.status = -1                                                                     <L 1066>
        var_pt.status = var_113;
        // return pt, _replace_simplex3(pt, 0, 2, 3)                                          <L 1067>
        var_117 = _replace_simplex3_0(var_pt, var_114, var_115, var_116);
        ret_0 = var_pt;
        ret_1 = var_117;
        return;
    }
    // if _attach_face(pt, 1, 0, 4, 2) < MIN_DIST2:                                           <L 1069>
    var_122 = _attach_face_0(var_pt, var_118, var_119, var_120, var_121);
    var_123 = (var_122 < var_111);
    if (var_123) {
        // pt.status = -1                                                                     <L 1070>
        var_pt.status = var_124;
        // return pt, _replace_simplex3(pt, 0, 4, 2)                                          <L 1071>
        var_128 = _replace_simplex3_0(var_pt, var_125, var_126, var_127);
        ret_0 = var_pt;
        ret_1 = var_128;
        return;
    }
    // if _attach_face(pt, 2, 0, 3, 4) < MIN_DIST2:                                           <L 1073>
    var_133 = _attach_face_0(var_pt, var_129, var_130, var_131, var_132);
    var_134 = (var_133 < var_111);
    if (var_134) {
        // pt.status = -1                                                                     <L 1074>
        var_pt.status = var_135;
        // return pt, _replace_simplex3(pt, 0, 3, 4)                                          <L 1075>
        var_139 = _replace_simplex3_0(var_pt, var_136, var_137, var_138);
        ret_0 = var_pt;
        ret_1 = var_139;
        return;
    }
    // if _attach_face(pt, 3, 1, 3, 2) < MIN_DIST2:                                           <L 1077>
    var_144 = _attach_face_0(var_pt, var_140, var_141, var_142, var_143);
    var_145 = (var_144 < var_111);
    if (var_145) {
        // pt.status = -1                                                                     <L 1078>
        var_pt.status = var_146;
        // return pt, _replace_simplex3(pt, 1, 3, 2)                                          <L 1079>
        var_150 = _replace_simplex3_0(var_pt, var_147, var_148, var_149);
        ret_0 = var_pt;
        ret_1 = var_150;
        return;
    }
    // if _attach_face(pt, 4, 1, 2, 4) < MIN_DIST2:                                           <L 1081>
    var_155 = _attach_face_0(var_pt, var_151, var_152, var_153, var_154);
    var_156 = (var_155 < var_111);
    if (var_156) {
        // pt.status = -1                                                                     <L 1082>
        var_pt.status = var_157;
        // return pt, _replace_simplex3(pt, 1, 2, 4)                                          <L 1083>
        var_161 = _replace_simplex3_0(var_pt, var_158, var_159, var_160);
        ret_0 = var_pt;
        ret_1 = var_161;
        return;
    }
    // if _attach_face(pt, 5, 1, 4, 3) < MIN_DIST2:                                           <L 1085>
    var_166 = _attach_face_0(var_pt, var_162, var_163, var_164, var_165);
    var_167 = (var_166 < var_111);
    if (var_167) {
        // pt.status = -1                                                                     <L 1086>
        var_pt.status = var_168;
        // return pt, _replace_simplex3(pt, 1, 4, 3)                                          <L 1087>
        var_172 = _replace_simplex3_0(var_pt, var_169, var_170, var_171);
        ret_0 = var_pt;
        ret_1 = var_172;
        return;
    }
    // v2 = pt.vert[4] - pt.vert[5]                                                           <L 1090>
    var_173 = &((var_pt).vert);
    var_176 = wp::load(var_173);
    var_175 = wp::address(var_176, var_174);
    var_177 = &((var_pt).vert);
    var_180 = wp::load(var_177);
    var_179 = wp::address(var_180, var_178);
    var_182 = wp::load(var_175);
    var_183 = wp::load(var_179);
    var_181 = wp::sub(var_182, var_183);
    // v3 = pt.vert[6] - pt.vert[7]                                                           <L 1091>
    var_184 = &((var_pt).vert);
    var_187 = wp::load(var_184);
    var_186 = wp::address(var_187, var_185);
    var_188 = &((var_pt).vert);
    var_191 = wp::load(var_188);
    var_190 = wp::address(var_191, var_189);
    var_193 = wp::load(var_186);
    var_194 = wp::load(var_190);
    var_192 = wp::sub(var_193, var_194);
    // v4 = pt.vert[8] - pt.vert[9]                                                           <L 1092>
    var_195 = &((var_pt).vert);
    var_198 = wp::load(var_195);
    var_197 = wp::address(var_198, var_196);
    var_199 = &((var_pt).vert);
    var_202 = wp::load(var_199);
    var_201 = wp::address(var_202, var_200);
    var_204 = wp::load(var_197);
    var_205 = wp::load(var_201);
    var_203 = wp::sub(var_204, var_205);
    // if not _ray_triangle(simplex[0], simplex[1], v2, v3, v4):                              <L 1093>
    var_207 = wp::extract(var_simplex, var_206);
    var_209 = wp::extract(var_simplex, var_208);
    var_210 = _ray_triangle_0(var_207, var_209, var_181, var_192, var_203);
    var_211 = wp::unot(var_210);
    if (var_211) {
        // pt.status = 1                                                                      <L 1094>
        var_pt.status = var_212;
        // return pt, GJKResult()                                                             <L 1095>
        var_213 = GJKResult_28609055();
        ret_0 = var_pt;
        ret_1 = var_213;
        return;
    }
    // pt.nvert = 5                                                                           <L 1098>
    var_pt.nvert = var_214;
    // pt.nface = 6                                                                           <L 1099>
    var_pt.nface = var_215;
    // pt.status = 0                                                                          <L 1100>
    var_pt.status = var_216;
    // return pt, GJKResult()                                                                 <L 1101>
    var_217 = GJKResult_28609055();
    ret_0 = var_pt;
    ret_1 = var_217;
    return;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:763
static CUDA_CALLABLE bool _same_side_0(
    wp::vec_t<3, wp::float32> var_p0,
    wp::vec_t<3, wp::float32> var_p1,
    wp::vec_t<3, wp::float32> var_p2,
    wp::vec_t<3, wp::float32> var_p3)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::float32 var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::float32 var_6;
    bool var_7;
    bool var_8;
    const wp::float32 var_9 = 0.0;
    bool var_10;
    const wp::float32 var_11 = 0.0;
    bool var_12;
    bool var_13;
    const wp::float32 var_14 = 0.0;
    bool var_15;
    const wp::float32 var_16 = 0.0;
    bool var_17;
    //---------
    // forward
    // def _same_side(p0: wp.vec3, p1: wp.vec3, p2: wp.vec3, p3: wp.vec3) -> bool:            <L 764>
    // n = wp.cross(p1 - p0, p2 - p0)                                                         <L 765>
    var_0 = wp::sub(var_p1, var_p0);
    var_1 = wp::sub(var_p2, var_p0);
    var_2 = wp::cross(var_0, var_1);
    // dot1 = wp.dot(n, p3 - p0)                                                              <L 766>
    var_3 = wp::sub(var_p3, var_p0);
    var_4 = wp::dot(var_2, var_3);
    // dot2 = wp.dot(n, -p0)                                                                  <L 767>
    var_5 = wp::neg(var_p0);
    var_6 = wp::dot(var_2, var_5);
    // return (dot1 > 0.0 and dot2 > 0.0) or (dot1 < 0.0 and dot2 < 0.0)                      <L 768>
    var_10 = (var_4 > var_9);
    var_8 = var_10;
    if (var_8) {
        var_12 = (var_6 > var_11);
        var_8 = var_8 && var_12;
    }
    var_7 = var_8;
    if (!var_7) {
        var_15 = (var_4 < var_14);
        var_13 = var_15;
        if (var_13) {
            var_17 = (var_6 < var_16);
            var_13 = var_13 && var_17;
        }
        var_7 = var_7 || var_13;
    }
    return var_7;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:771
static CUDA_CALLABLE bool _test_tetra_0(
    wp::vec_t<3, wp::float32> var_p0,
    wp::vec_t<3, wp::float32> var_p1,
    wp::vec_t<3, wp::float32> var_p2,
    wp::vec_t<3, wp::float32> var_p3)
{
    //---------
    // primal vars
    bool var_0;
    bool var_1;
    bool var_2;
    bool var_3;
    bool var_4;
    //---------
    // forward
    // def _test_tetra(p0: wp.vec3, p1: wp.vec3, p2: wp.vec3, p3: wp.vec3) -> bool:           <L 772>
    // return _same_side(p0, p1, p2, p3) and _same_side(p1, p2, p3, p0) and _same_side(p2, p3, p0, p1) and _same_side(p3, p0, p1, p2)       <L 773>
    var_1 = _same_side_0(var_p0, var_p1, var_p2, var_p3);
    var_0 = var_1;
    if (var_0) {
        var_2 = _same_side_0(var_p1, var_p2, var_p3, var_p0);
        var_0 = var_0 && var_2;
    }
    if (var_0) {
        var_3 = _same_side_0(var_p2, var_p3, var_p0, var_p1);
        var_0 = var_0 && var_3;
    }
    if (var_0) {
        var_4 = _same_side_0(var_p3, var_p0, var_p1, var_p2);
        var_0 = var_0 && var_4;
    }
    return var_0;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1195
static CUDA_CALLABLE void _polytope4_0(
    Polytope_10582b13 var_pt,
    wp::mat_t<4, 3, wp::float32> var_simplex,
    wp::mat_t<4, 3, wp::float32> var_simplex1,
    wp::mat_t<4, 3, wp::float32> var_simplex2,
    wp::vec_t<4, wp::int32> var_simplex_index1,
    wp::vec_t<4, wp::int32> var_simplex_index2,
    Polytope_10582b13 & ret_0,
    GJKResult_28609055 & ret_1)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.25;
    const wp::int32 var_1 = 0;
    wp::vec_t<3, wp::float32> var_2;
    const wp::int32 var_3 = 1;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    const wp::int32 var_6 = 2;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    const wp::int32 var_9 = 3;
    wp::vec_t<3, wp::float32> var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    const wp::int32 var_13 = 0;
    wp::vec_t<3, wp::float32> var_14;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_15;
    const wp::int32 var_16 = 0;
    wp::array_t<wp::vec_t<3, wp::float32>> var_17;
    const wp::int32 var_18 = 0;
    wp::vec_t<3, wp::float32> var_19;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_20;
    const wp::int32 var_21 = 1;
    wp::array_t<wp::vec_t<3, wp::float32>> var_22;
    const wp::int32 var_23 = 1;
    wp::vec_t<3, wp::float32> var_24;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_25;
    const wp::int32 var_26 = 2;
    wp::array_t<wp::vec_t<3, wp::float32>> var_27;
    const wp::int32 var_28 = 1;
    wp::vec_t<3, wp::float32> var_29;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_30;
    const wp::int32 var_31 = 3;
    wp::array_t<wp::vec_t<3, wp::float32>> var_32;
    const wp::int32 var_33 = 2;
    wp::vec_t<3, wp::float32> var_34;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_35;
    const wp::int32 var_36 = 4;
    wp::array_t<wp::vec_t<3, wp::float32>> var_37;
    const wp::int32 var_38 = 2;
    wp::vec_t<3, wp::float32> var_39;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_40;
    const wp::int32 var_41 = 5;
    wp::array_t<wp::vec_t<3, wp::float32>> var_42;
    const wp::int32 var_43 = 3;
    wp::vec_t<3, wp::float32> var_44;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_45;
    const wp::int32 var_46 = 6;
    wp::array_t<wp::vec_t<3, wp::float32>> var_47;
    const wp::int32 var_48 = 3;
    wp::vec_t<3, wp::float32> var_49;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_50;
    const wp::int32 var_51 = 7;
    wp::array_t<wp::vec_t<3, wp::float32>> var_52;
    const wp::int32 var_53 = 0;
    wp::int32 var_54;
    wp::array_t<wp::int32>* var_55;
    const wp::int32 var_56 = 0;
    wp::array_t<wp::int32> var_57;
    const wp::int32 var_58 = 0;
    wp::int32 var_59;
    wp::array_t<wp::int32>* var_60;
    const wp::int32 var_61 = 1;
    wp::array_t<wp::int32> var_62;
    const wp::int32 var_63 = 1;
    wp::int32 var_64;
    wp::array_t<wp::int32>* var_65;
    const wp::int32 var_66 = 2;
    wp::array_t<wp::int32> var_67;
    const wp::int32 var_68 = 1;
    wp::int32 var_69;
    wp::array_t<wp::int32>* var_70;
    const wp::int32 var_71 = 3;
    wp::array_t<wp::int32> var_72;
    const wp::int32 var_73 = 2;
    wp::int32 var_74;
    wp::array_t<wp::int32>* var_75;
    const wp::int32 var_76 = 4;
    wp::array_t<wp::int32> var_77;
    const wp::int32 var_78 = 2;
    wp::int32 var_79;
    wp::array_t<wp::int32>* var_80;
    const wp::int32 var_81 = 5;
    wp::array_t<wp::int32> var_82;
    const wp::int32 var_83 = 3;
    wp::int32 var_84;
    wp::array_t<wp::int32>* var_85;
    const wp::int32 var_86 = 6;
    wp::array_t<wp::int32> var_87;
    const wp::int32 var_88 = 3;
    wp::int32 var_89;
    wp::array_t<wp::int32>* var_90;
    const wp::int32 var_91 = 7;
    wp::array_t<wp::int32> var_92;
    wp::vec_t<4, wp::float32> var_93;
    const wp::int32 var_94 = 0;
    wp::int32 var_95;
    const wp::int32 var_96 = 0;
    const wp::int32 var_97 = 0;
    const wp::int32 var_98 = 1;
    const wp::int32 var_99 = 2;
    wp::float32 var_100;
    const wp::int32 var_101 = 0;
    const wp::int32 var_102 = 0;
    wp::float32 var_103;
    const wp::float32 var_104 = 1e-17;
    bool var_105;
    const wp::int32 var_106 = -1;
    const wp::int32 var_107 = 0;
    const wp::int32 var_108 = 1;
    const wp::int32 var_109 = 2;
    GJKResult_28609055 var_110;
    const wp::int32 var_111 = 1;
    const wp::int32 var_112 = 0;
    const wp::int32 var_113 = 3;
    const wp::int32 var_114 = 1;
    wp::float32 var_115;
    const wp::int32 var_116 = 1;
    const wp::int32 var_117 = 1;
    wp::float32 var_118;
    bool var_119;
    const wp::int32 var_120 = -1;
    const wp::int32 var_121 = 0;
    const wp::int32 var_122 = 3;
    const wp::int32 var_123 = 1;
    GJKResult_28609055 var_124;
    const wp::int32 var_125 = 0;
    wp::float32 var_126;
    const wp::int32 var_127 = 1;
    wp::float32 var_128;
    bool var_129;
    const wp::int32 var_130 = 0;
    const wp::int32 var_131 = 1;
    wp::int32 var_132;
    const wp::int32 var_133 = 2;
    const wp::int32 var_134 = 0;
    const wp::int32 var_135 = 2;
    const wp::int32 var_136 = 3;
    wp::float32 var_137;
    const wp::int32 var_138 = 2;
    const wp::int32 var_139 = 2;
    wp::float32 var_140;
    bool var_141;
    const wp::int32 var_142 = -1;
    const wp::int32 var_143 = 0;
    const wp::int32 var_144 = 2;
    const wp::int32 var_145 = 3;
    GJKResult_28609055 var_146;
    const wp::int32 var_147 = 2;
    wp::float32 var_148;
    wp::float32 var_149;
    bool var_150;
    const wp::int32 var_151 = 2;
    wp::int32 var_152;
    const wp::int32 var_153 = 3;
    const wp::int32 var_154 = 3;
    const wp::int32 var_155 = 2;
    const wp::int32 var_156 = 1;
    wp::float32 var_157;
    const wp::int32 var_158 = 3;
    const wp::int32 var_159 = 3;
    wp::float32 var_160;
    bool var_161;
    const wp::int32 var_162 = -1;
    const wp::int32 var_163 = 3;
    const wp::int32 var_164 = 2;
    const wp::int32 var_165 = 1;
    GJKResult_28609055 var_166;
    const wp::int32 var_167 = 3;
    wp::float32 var_168;
    wp::float32 var_169;
    bool var_170;
    const wp::int32 var_171 = 3;
    wp::int32 var_172;
    const wp::int32 var_173 = 0;
    wp::vec_t<3, wp::float32> var_174;
    const wp::int32 var_175 = 1;
    wp::vec_t<3, wp::float32> var_176;
    const wp::int32 var_177 = 2;
    wp::vec_t<3, wp::float32> var_178;
    const wp::int32 var_179 = 3;
    wp::vec_t<3, wp::float32> var_180;
    bool var_181;
    bool var_182;
    wp::float32 var_183;
    const wp::float32 var_184 = 1e-15;
    bool var_185;
    const wp::int32 var_186 = 12;
    GJKResult_28609055 var_187;
    const wp::int32 var_188 = -1;
    const wp::int32 var_189 = 0;
    bool var_190;
    const wp::int32 var_191 = 0;
    const wp::int32 var_192 = 1;
    const wp::int32 var_193 = 2;
    GJKResult_28609055 var_194;
    const wp::int32 var_195 = 1;
    bool var_196;
    const wp::int32 var_197 = 0;
    const wp::int32 var_198 = 3;
    const wp::int32 var_199 = 1;
    GJKResult_28609055 var_200;
    const wp::int32 var_201 = 2;
    bool var_202;
    const wp::int32 var_203 = 0;
    const wp::int32 var_204 = 2;
    const wp::int32 var_205 = 3;
    GJKResult_28609055 var_206;
    const wp::int32 var_207 = 3;
    const wp::int32 var_208 = 2;
    const wp::int32 var_209 = 1;
    GJKResult_28609055 var_210;
    const wp::int32 var_211 = 4;
    const wp::int32 var_212 = 4;
    const wp::int32 var_213 = 0;
    GJKResult_28609055 var_214;
    //---------
    // forward
    // def _polytope4(                                                                        <L 1196>
    // pt.center = 0.25 * (simplex[0] + simplex[1] + simplex[2] + simplex[3])                 <L 1207>
    var_2 = wp::extract(var_simplex, var_1);
    var_4 = wp::extract(var_simplex, var_3);
    var_5 = wp::add(var_2, var_4);
    var_7 = wp::extract(var_simplex, var_6);
    var_8 = wp::add(var_5, var_7);
    var_10 = wp::extract(var_simplex, var_9);
    var_11 = wp::add(var_8, var_10);
    var_12 = wp::mul(var_0, var_11);
    var_pt.center = var_12;
    // pt.vert[0] = simplex1[0]                                                               <L 1209>
    var_14 = wp::extract(var_simplex1, var_13);
    var_15 = &((var_pt).vert);
    var_17 = wp::load(var_15);
    wp::array_store(var_17, var_16, var_14);
    // pt.vert[1] = simplex2[0]                                                               <L 1210>
    var_19 = wp::extract(var_simplex2, var_18);
    var_20 = &((var_pt).vert);
    var_22 = wp::load(var_20);
    wp::array_store(var_22, var_21, var_19);
    // pt.vert[2] = simplex1[1]                                                               <L 1211>
    var_24 = wp::extract(var_simplex1, var_23);
    var_25 = &((var_pt).vert);
    var_27 = wp::load(var_25);
    wp::array_store(var_27, var_26, var_24);
    // pt.vert[3] = simplex2[1]                                                               <L 1212>
    var_29 = wp::extract(var_simplex2, var_28);
    var_30 = &((var_pt).vert);
    var_32 = wp::load(var_30);
    wp::array_store(var_32, var_31, var_29);
    // pt.vert[4] = simplex1[2]                                                               <L 1213>
    var_34 = wp::extract(var_simplex1, var_33);
    var_35 = &((var_pt).vert);
    var_37 = wp::load(var_35);
    wp::array_store(var_37, var_36, var_34);
    // pt.vert[5] = simplex2[2]                                                               <L 1214>
    var_39 = wp::extract(var_simplex2, var_38);
    var_40 = &((var_pt).vert);
    var_42 = wp::load(var_40);
    wp::array_store(var_42, var_41, var_39);
    // pt.vert[6] = simplex1[3]                                                               <L 1215>
    var_44 = wp::extract(var_simplex1, var_43);
    var_45 = &((var_pt).vert);
    var_47 = wp::load(var_45);
    wp::array_store(var_47, var_46, var_44);
    // pt.vert[7] = simplex2[3]                                                               <L 1216>
    var_49 = wp::extract(var_simplex2, var_48);
    var_50 = &((var_pt).vert);
    var_52 = wp::load(var_50);
    wp::array_store(var_52, var_51, var_49);
    // pt.vert_index[0] = simplex_index1[0]                                                   <L 1218>
    var_54 = wp::extract(var_simplex_index1, var_53);
    var_55 = &((var_pt).vert_index);
    var_57 = wp::load(var_55);
    wp::array_store(var_57, var_56, var_54);
    // pt.vert_index[1] = simplex_index2[0]                                                   <L 1219>
    var_59 = wp::extract(var_simplex_index2, var_58);
    var_60 = &((var_pt).vert_index);
    var_62 = wp::load(var_60);
    wp::array_store(var_62, var_61, var_59);
    // pt.vert_index[2] = simplex_index1[1]                                                   <L 1220>
    var_64 = wp::extract(var_simplex_index1, var_63);
    var_65 = &((var_pt).vert_index);
    var_67 = wp::load(var_65);
    wp::array_store(var_67, var_66, var_64);
    // pt.vert_index[3] = simplex_index2[1]                                                   <L 1221>
    var_69 = wp::extract(var_simplex_index2, var_68);
    var_70 = &((var_pt).vert_index);
    var_72 = wp::load(var_70);
    wp::array_store(var_72, var_71, var_69);
    // pt.vert_index[4] = simplex_index1[2]                                                   <L 1222>
    var_74 = wp::extract(var_simplex_index1, var_73);
    var_75 = &((var_pt).vert_index);
    var_77 = wp::load(var_75);
    wp::array_store(var_77, var_76, var_74);
    // pt.vert_index[5] = simplex_index2[2]                                                   <L 1223>
    var_79 = wp::extract(var_simplex_index2, var_78);
    var_80 = &((var_pt).vert_index);
    var_82 = wp::load(var_80);
    wp::array_store(var_82, var_81, var_79);
    // pt.vert_index[6] = simplex_index1[3]                                                   <L 1224>
    var_84 = wp::extract(var_simplex_index1, var_83);
    var_85 = &((var_pt).vert_index);
    var_87 = wp::load(var_85);
    wp::array_store(var_87, var_86, var_84);
    // pt.vert_index[7] = simplex_index2[3]                                                   <L 1225>
    var_89 = wp::extract(var_simplex_index2, var_88);
    var_90 = &((var_pt).vert_index);
    var_92 = wp::load(var_90);
    wp::array_store(var_92, var_91, var_89);
    // dist = wp.vec4()                                                                       <L 1227>
    var_93 = wp::vec_t<4, wp::float32>();
    // idx = int(0)                                                                           <L 1228>
    var_95 = wp::int(var_94);
    // dist[0] = _attach_face(pt, 0, 0, 1, 2)                                                 <L 1231>
    var_100 = _attach_face_0(var_pt, var_96, var_97, var_98, var_99);
    wp::assign_inplace(var_93, var_101, var_100);
    // if dist[0] < MIN_DIST4:                                                                <L 1232>
    var_103 = wp::extract(var_93, var_102);
    var_105 = (var_103 < var_104);
    if (var_105) {
        // pt.status = -1                                                                     <L 1233>
        var_pt.status = var_106;
        // return pt, _replace_simplex3(pt, 0, 1, 2)                                          <L 1234>
        var_110 = _replace_simplex3_0(var_pt, var_107, var_108, var_109);
        ret_0 = var_pt;
        ret_1 = var_110;
        return;
    }
    // dist[1] = _attach_face(pt, 1, 0, 3, 1)                                                 <L 1236>
    var_115 = _attach_face_0(var_pt, var_111, var_112, var_113, var_114);
    wp::assign_inplace(var_93, var_116, var_115);
    // if dist[1] < MIN_DIST4:                                                                <L 1237>
    var_118 = wp::extract(var_93, var_117);
    var_119 = (var_118 < var_104);
    if (var_119) {
        // pt.status = -1                                                                     <L 1238>
        var_pt.status = var_120;
        // return pt, _replace_simplex3(pt, 0, 3, 1)                                          <L 1239>
        var_124 = _replace_simplex3_0(var_pt, var_121, var_122, var_123);
        ret_0 = var_pt;
        ret_1 = var_124;
        return;
    }
    // idx = wp.where(dist[0] < dist[1], 0, 1)                                                <L 1240>
    var_126 = wp::extract(var_93, var_125);
    var_128 = wp::extract(var_93, var_127);
    var_129 = (var_126 < var_128);
    var_132 = wp::where(var_129, var_130, var_131);
    // dist[2] = _attach_face(pt, 2, 0, 2, 3)                                                 <L 1242>
    var_137 = _attach_face_0(var_pt, var_133, var_134, var_135, var_136);
    wp::assign_inplace(var_93, var_138, var_137);
    // if dist[2] < MIN_DIST4:                                                                <L 1243>
    var_140 = wp::extract(var_93, var_139);
    var_141 = (var_140 < var_104);
    if (var_141) {
        // pt.status = -1                                                                     <L 1244>
        var_pt.status = var_142;
        // return pt, _replace_simplex3(pt, 0, 2, 3)                                          <L 1245>
        var_146 = _replace_simplex3_0(var_pt, var_143, var_144, var_145);
        ret_0 = var_pt;
        ret_1 = var_146;
        return;
    }
    // idx = wp.where(dist[2] < dist[idx], 2, idx)                                            <L 1246>
    var_148 = wp::extract(var_93, var_147);
    var_149 = wp::extract(var_93, var_132);
    var_150 = (var_148 < var_149);
    var_152 = wp::where(var_150, var_151, var_132);
    // dist[3] = _attach_face(pt, 3, 3, 2, 1)                                                 <L 1248>
    var_157 = _attach_face_0(var_pt, var_153, var_154, var_155, var_156);
    wp::assign_inplace(var_93, var_158, var_157);
    // if dist[3] < MIN_DIST4:                                                                <L 1249>
    var_160 = wp::extract(var_93, var_159);
    var_161 = (var_160 < var_104);
    if (var_161) {
        // pt.status = -1                                                                     <L 1250>
        var_pt.status = var_162;
        // return pt, _replace_simplex3(pt, 3, 2, 1)                                          <L 1251>
        var_166 = _replace_simplex3_0(var_pt, var_163, var_164, var_165);
        ret_0 = var_pt;
        ret_1 = var_166;
        return;
    }
    // idx = wp.where(dist[3] < dist[idx], 3, idx)                                            <L 1252>
    var_168 = wp::extract(var_93, var_167);
    var_169 = wp::extract(var_93, var_152);
    var_170 = (var_168 < var_169);
    var_172 = wp::where(var_170, var_171, var_152);
    // if not _test_tetra(simplex[0], simplex[1], simplex[2], simplex[3]):                    <L 1254>
    var_174 = wp::extract(var_simplex, var_173);
    var_176 = wp::extract(var_simplex, var_175);
    var_178 = wp::extract(var_simplex, var_177);
    var_180 = wp::extract(var_simplex, var_179);
    var_181 = _test_tetra_0(var_174, var_176, var_178, var_180);
    var_182 = wp::unot(var_181);
    if (var_182) {
        // if dist[idx] > MINVAL:                                                             <L 1255>
        var_183 = wp::extract(var_93, var_172);
        var_185 = (var_183 > var_184);
        if (var_185) {
            // pt.status = 12                                                                 <L 1256>
            var_pt.status = var_186;
            // return pt, GJKResult()                                                         <L 1257>
            var_187 = GJKResult_28609055();
            ret_0 = var_pt;
            ret_1 = var_187;
            return;
        }
        // pt.status = -1                                                                     <L 1260>
        var_pt.status = var_188;
        // if idx == 0:                                                                       <L 1261>
        var_190 = (var_172 == var_189);
        if (var_190) {
            // return pt, _replace_simplex3(pt, 0, 1, 2)                                      <L 1262>
            var_194 = _replace_simplex3_0(var_pt, var_191, var_192, var_193);
            ret_0 = var_pt;
            ret_1 = var_194;
            return;
        }
        if (!var_190) {
            // elif idx == 1:                                                                 <L 1263>
            var_196 = (var_172 == var_195);
            if (var_196) {
                // return pt, _replace_simplex3(pt, 0, 3, 1)                                  <L 1264>
                var_200 = _replace_simplex3_0(var_pt, var_197, var_198, var_199);
                ret_0 = var_pt;
                ret_1 = var_200;
                return;
            }
            if (!var_196) {
                // elif idx == 2:                                                             <L 1265>
                var_202 = (var_172 == var_201);
                if (var_202) {
                    // return pt, _replace_simplex3(pt, 0, 2, 3)                              <L 1266>
                    var_206 = _replace_simplex3_0(var_pt, var_203, var_204, var_205);
                    ret_0 = var_pt;
                    ret_1 = var_206;
                    return;
                }
                if (!var_202) {
                    // return pt, _replace_simplex3(pt, 3, 2, 1)                              <L 1268>
                    var_210 = _replace_simplex3_0(var_pt, var_207, var_208, var_209);
                    ret_0 = var_pt;
                    ret_1 = var_210;
                    return;
                }
            }
        }
    }
    // pt.nvert = 4                                                                           <L 1271>
    var_pt.nvert = var_211;
    // pt.nface = 4                                                                           <L 1272>
    var_pt.nface = var_212;
    // pt.status = 0                                                                          <L 1273>
    var_pt.status = var_213;
    // return pt, GJKResult()                                                                 <L 1274>
    var_214 = GJKResult_28609055();
    ret_0 = var_pt;
    ret_1 = var_214;
    return;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:819
static CUDA_CALLABLE bool _tri_point_intersect_0(
    wp::vec_t<3, wp::float32> var_v1,
    wp::vec_t<3, wp::float32> var_v2,
    wp::vec_t<3, wp::float32> var_v3,
    wp::vec_t<3, wp::float32> var_p)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    const wp::int32 var_1 = 0;
    wp::float32 var_2;
    const wp::int32 var_3 = 1;
    wp::float32 var_4;
    const wp::int32 var_5 = 2;
    wp::float32 var_6;
    bool var_7;
    const wp::float32 var_8 = 0.0;
    bool var_9;
    const wp::float32 var_10 = 0.0;
    bool var_11;
    const wp::float32 var_12 = 0.0;
    bool var_13;
    const bool var_14 = false;
    wp::vec_t<3, wp::float32> var_15;
    const wp::int32 var_16 = 0;
    wp::float32 var_17;
    wp::float32 var_18;
    const wp::int32 var_19 = 0;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    const wp::int32 var_23 = 0;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::float32 var_26;
    const wp::int32 var_27 = 0;
    const wp::int32 var_28 = 1;
    wp::float32 var_29;
    wp::float32 var_30;
    const wp::int32 var_31 = 1;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    const wp::int32 var_35 = 1;
    wp::float32 var_36;
    wp::float32 var_37;
    wp::float32 var_38;
    const wp::int32 var_39 = 1;
    const wp::int32 var_40 = 2;
    wp::float32 var_41;
    wp::float32 var_42;
    const wp::int32 var_43 = 2;
    wp::float32 var_44;
    wp::float32 var_45;
    wp::float32 var_46;
    const wp::int32 var_47 = 2;
    wp::float32 var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    const wp::int32 var_51 = 2;
    wp::vec_t<3, wp::float32> var_52;
    wp::float32 var_53;
    const wp::float32 var_54 = 1e-15;
    bool var_55;
    //---------
    // forward
    // def _tri_point_intersect(v1: wp.vec3, v2: wp.vec3, v3: wp.vec3, p: wp.vec3) -> bool:       <L 820>
    // coordinates = _tri_affine_coord(v1, v2, v3, p)                                         <L 821>
    var_0 = _tri_affine_coord_0(var_v1, var_v2, var_v3, var_p);
    // l1 = coordinates[0]                                                                    <L 822>
    var_2 = wp::extract(var_0, var_1);
    // l2 = coordinates[1]                                                                    <L 823>
    var_4 = wp::extract(var_0, var_3);
    // l3 = coordinates[2]                                                                    <L 824>
    var_6 = wp::extract(var_0, var_5);
    // if l1 < 0.0 or l2 < 0.0 or l3 < 0.0:                                                   <L 826>
    var_9 = (var_2 < var_8);
    var_7 = var_9;
    if (!var_7) {
        var_11 = (var_4 < var_10);
        var_7 = var_7 || var_11;
    }
    if (!var_7) {
        var_13 = (var_6 < var_12);
        var_7 = var_7 || var_13;
    }
    if (var_7) {
        // return False                                                                       <L 827>
        return var_14;
    }
    // pr = wp.vec3()                                                                         <L 829>
    var_15 = wp::vec_t<3, wp::float32>();
    // pr[0] = v1[0] * l1 + v2[0] * l2 + v3[0] * l3                                           <L 830>
    var_17 = wp::extract(var_v1, var_16);
    var_18 = wp::mul(var_17, var_2);
    var_20 = wp::extract(var_v2, var_19);
    var_21 = wp::mul(var_20, var_4);
    var_22 = wp::add(var_18, var_21);
    var_24 = wp::extract(var_v3, var_23);
    var_25 = wp::mul(var_24, var_6);
    var_26 = wp::add(var_22, var_25);
    wp::assign_inplace(var_15, var_27, var_26);
    // pr[1] = v1[1] * l1 + v2[1] * l2 + v3[1] * l3                                           <L 831>
    var_29 = wp::extract(var_v1, var_28);
    var_30 = wp::mul(var_29, var_2);
    var_32 = wp::extract(var_v2, var_31);
    var_33 = wp::mul(var_32, var_4);
    var_34 = wp::add(var_30, var_33);
    var_36 = wp::extract(var_v3, var_35);
    var_37 = wp::mul(var_36, var_6);
    var_38 = wp::add(var_34, var_37);
    wp::assign_inplace(var_15, var_39, var_38);
    // pr[2] = v1[2] * l1 + v2[2] * l2 + v3[2] * l3                                           <L 832>
    var_41 = wp::extract(var_v1, var_40);
    var_42 = wp::mul(var_41, var_2);
    var_44 = wp::extract(var_v2, var_43);
    var_45 = wp::mul(var_44, var_4);
    var_46 = wp::add(var_42, var_45);
    var_48 = wp::extract(var_v3, var_47);
    var_49 = wp::mul(var_48, var_6);
    var_50 = wp::add(var_46, var_49);
    wp::assign_inplace(var_15, var_51, var_50);
    // return wp.norm_l2(pr - p) < MINVAL                                                     <L 833>
    var_52 = wp::sub(var_15, var_p);
    var_53 = norm_l2_0(var_52);
    var_55 = (var_53 < var_54);
    return var_55;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1104
static CUDA_CALLABLE Polytope_10582b13 _polytope3_0(
    Polytope_10582b13 var_pt,
    wp::float32 var_dist,
    wp::mat_t<4, 3, wp::float32> var_simplex,
    wp::mat_t<4, 3, wp::float32> var_simplex1,
    wp::mat_t<4, 3, wp::float32> var_simplex2,
    wp::vec_t<4, wp::int32> var_simplex_index1,
    wp::vec_t<4, wp::int32> var_simplex_index2,
    Geom_3242f8a8 var_geom1,
    Geom_3242f8a8 var_geom2,
    wp::int32 var_geomtype1,
    wp::int32 var_geomtype2)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::vec_t<3, wp::float32> var_1;
    const wp::int32 var_2 = 1;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    const wp::int32 var_5 = 2;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    const wp::float32 var_8 = 0.3333333333333333;
    wp::vec_t<3, wp::float32> var_9;
    const wp::int32 var_10 = 1;
    wp::vec_t<3, wp::float32> var_11;
    const wp::int32 var_12 = 0;
    wp::vec_t<3, wp::float32> var_13;
    wp::vec_t<3, wp::float32> var_14;
    const wp::int32 var_15 = 2;
    wp::vec_t<3, wp::float32> var_16;
    const wp::int32 var_17 = 0;
    wp::vec_t<3, wp::float32> var_18;
    wp::vec_t<3, wp::float32> var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::float32 var_21;
    const wp::float32 var_22 = 1e-15;
    bool var_23;
    const wp::int32 var_24 = 2;
    const wp::int32 var_25 = 0;
    wp::vec_t<3, wp::float32> var_26;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_27;
    const wp::int32 var_28 = 0;
    wp::array_t<wp::vec_t<3, wp::float32>> var_29;
    const wp::int32 var_30 = 0;
    wp::vec_t<3, wp::float32> var_31;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_32;
    const wp::int32 var_33 = 1;
    wp::array_t<wp::vec_t<3, wp::float32>> var_34;
    const wp::int32 var_35 = 1;
    wp::vec_t<3, wp::float32> var_36;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_37;
    const wp::int32 var_38 = 2;
    wp::array_t<wp::vec_t<3, wp::float32>> var_39;
    const wp::int32 var_40 = 1;
    wp::vec_t<3, wp::float32> var_41;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_42;
    const wp::int32 var_43 = 3;
    wp::array_t<wp::vec_t<3, wp::float32>> var_44;
    const wp::int32 var_45 = 2;
    wp::vec_t<3, wp::float32> var_46;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_47;
    const wp::int32 var_48 = 4;
    wp::array_t<wp::vec_t<3, wp::float32>> var_49;
    const wp::int32 var_50 = 2;
    wp::vec_t<3, wp::float32> var_51;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_52;
    const wp::int32 var_53 = 5;
    wp::array_t<wp::vec_t<3, wp::float32>> var_54;
    const wp::int32 var_55 = 0;
    wp::int32 var_56;
    wp::array_t<wp::int32>* var_57;
    const wp::int32 var_58 = 0;
    wp::array_t<wp::int32> var_59;
    const wp::int32 var_60 = 0;
    wp::int32 var_61;
    wp::array_t<wp::int32>* var_62;
    const wp::int32 var_63 = 1;
    wp::array_t<wp::int32> var_64;
    const wp::int32 var_65 = 1;
    wp::int32 var_66;
    wp::array_t<wp::int32>* var_67;
    const wp::int32 var_68 = 2;
    wp::array_t<wp::int32> var_69;
    const wp::int32 var_70 = 1;
    wp::int32 var_71;
    wp::array_t<wp::int32>* var_72;
    const wp::int32 var_73 = 3;
    wp::array_t<wp::int32> var_74;
    const wp::int32 var_75 = 2;
    wp::int32 var_76;
    wp::array_t<wp::int32>* var_77;
    const wp::int32 var_78 = 4;
    wp::array_t<wp::int32> var_79;
    const wp::int32 var_80 = 2;
    wp::int32 var_81;
    wp::array_t<wp::int32>* var_82;
    const wp::int32 var_83 = 5;
    wp::array_t<wp::int32> var_84;
    const wp::int32 var_85 = 3;
    wp::vec_t<3, wp::float32> var_86;
    wp::int32 var_87;
    wp::int32 var_88;
    const wp::int32 var_89 = 4;
    wp::int32 var_90;
    wp::int32 var_91;
    const wp::int32 var_92 = 0;
    wp::vec_t<3, wp::float32> var_93;
    const wp::int32 var_94 = 1;
    wp::vec_t<3, wp::float32> var_95;
    const wp::int32 var_96 = 2;
    wp::vec_t<3, wp::float32> var_97;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_98;
    const wp::int32 var_99 = 6;
    wp::vec_t<3, wp::float32>* var_100;
    wp::array_t<wp::vec_t<3, wp::float32>> var_101;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_102;
    const wp::int32 var_103 = 7;
    wp::vec_t<3, wp::float32>* var_104;
    wp::array_t<wp::vec_t<3, wp::float32>> var_105;
    wp::vec_t<3, wp::float32> var_106;
    wp::vec_t<3, wp::float32> var_107;
    wp::vec_t<3, wp::float32> var_108;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_109;
    const wp::int32 var_110 = 8;
    wp::vec_t<3, wp::float32>* var_111;
    wp::array_t<wp::vec_t<3, wp::float32>> var_112;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_113;
    const wp::int32 var_114 = 9;
    wp::vec_t<3, wp::float32>* var_115;
    wp::array_t<wp::vec_t<3, wp::float32>> var_116;
    wp::vec_t<3, wp::float32> var_117;
    wp::vec_t<3, wp::float32> var_118;
    wp::vec_t<3, wp::float32> var_119;
    bool var_120;
    const wp::int32 var_121 = 3;
    bool var_122;
    const wp::int32 var_123 = 4;
    bool var_124;
    const wp::float32 var_125 = 1e-05;
    bool var_126;
    bool var_127;
    bool var_128;
    bool var_129;
    bool var_130;
    const wp::int32 var_131 = 5;
    const wp::int32 var_132 = 0;
    const wp::int32 var_133 = 4;
    const wp::int32 var_134 = 0;
    const wp::int32 var_135 = 1;
    wp::float32 var_136;
    const wp::float32 var_137 = 1e-10;
    bool var_138;
    const wp::int32 var_139 = 6;
    const wp::int32 var_140 = 1;
    const wp::int32 var_141 = 4;
    const wp::int32 var_142 = 2;
    const wp::int32 var_143 = 0;
    wp::float32 var_144;
    bool var_145;
    const wp::int32 var_146 = 7;
    const wp::int32 var_147 = 2;
    const wp::int32 var_148 = 4;
    const wp::int32 var_149 = 1;
    const wp::int32 var_150 = 2;
    wp::float32 var_151;
    bool var_152;
    const wp::int32 var_153 = 8;
    const wp::int32 var_154 = 3;
    const wp::int32 var_155 = 3;
    const wp::int32 var_156 = 1;
    const wp::int32 var_157 = 0;
    wp::float32 var_158;
    bool var_159;
    const wp::int32 var_160 = 9;
    const wp::int32 var_161 = 4;
    const wp::int32 var_162 = 3;
    const wp::int32 var_163 = 0;
    const wp::int32 var_164 = 2;
    wp::float32 var_165;
    bool var_166;
    const wp::int32 var_167 = 10;
    const wp::int32 var_168 = 5;
    const wp::int32 var_169 = 3;
    const wp::int32 var_170 = 2;
    const wp::int32 var_171 = 1;
    wp::float32 var_172;
    bool var_173;
    const wp::int32 var_174 = 11;
    const wp::int32 var_175 = 5;
    const wp::int32 var_176 = 6;
    const wp::int32 var_177 = 0;
    //---------
    // forward
    // def _polytope3(                                                                        <L 1105>
    // pt.center = (simplex[0] + simplex[1] + simplex[2]) * wp.static(1.0 / 3.0)              <L 1121>
    var_1 = wp::extract(var_simplex, var_0);
    var_3 = wp::extract(var_simplex, var_2);
    var_4 = wp::add(var_1, var_3);
    var_6 = wp::extract(var_simplex, var_5);
    var_7 = wp::add(var_4, var_6);
    var_9 = wp::mul(var_7, var_8);
    var_pt.center = var_9;
    // n = wp.cross(simplex[1] - simplex[0], simplex[2] - simplex[0])                         <L 1124>
    var_11 = wp::extract(var_simplex, var_10);
    var_13 = wp::extract(var_simplex, var_12);
    var_14 = wp::sub(var_11, var_13);
    var_16 = wp::extract(var_simplex, var_15);
    var_18 = wp::extract(var_simplex, var_17);
    var_19 = wp::sub(var_16, var_18);
    var_20 = wp::cross(var_14, var_19);
    // if wp.norm_l2(n) < MINVAL:                                                             <L 1125>
    var_21 = norm_l2_0(var_20);
    var_23 = (var_21 < var_22);
    if (var_23) {
        // pt.status = 2                                                                      <L 1126>
        var_pt.status = var_24;
        // return pt                                                                          <L 1127>
        return var_pt;
    }
    // pt.vert[0] = simplex1[0]                                                               <L 1129>
    var_26 = wp::extract(var_simplex1, var_25);
    var_27 = &((var_pt).vert);
    var_29 = wp::load(var_27);
    wp::array_store(var_29, var_28, var_26);
    // pt.vert[1] = simplex2[0]                                                               <L 1130>
    var_31 = wp::extract(var_simplex2, var_30);
    var_32 = &((var_pt).vert);
    var_34 = wp::load(var_32);
    wp::array_store(var_34, var_33, var_31);
    // pt.vert[2] = simplex1[1]                                                               <L 1131>
    var_36 = wp::extract(var_simplex1, var_35);
    var_37 = &((var_pt).vert);
    var_39 = wp::load(var_37);
    wp::array_store(var_39, var_38, var_36);
    // pt.vert[3] = simplex2[1]                                                               <L 1132>
    var_41 = wp::extract(var_simplex2, var_40);
    var_42 = &((var_pt).vert);
    var_44 = wp::load(var_42);
    wp::array_store(var_44, var_43, var_41);
    // pt.vert[4] = simplex1[2]                                                               <L 1133>
    var_46 = wp::extract(var_simplex1, var_45);
    var_47 = &((var_pt).vert);
    var_49 = wp::load(var_47);
    wp::array_store(var_49, var_48, var_46);
    // pt.vert[5] = simplex2[2]                                                               <L 1134>
    var_51 = wp::extract(var_simplex2, var_50);
    var_52 = &((var_pt).vert);
    var_54 = wp::load(var_52);
    wp::array_store(var_54, var_53, var_51);
    // pt.vert_index[0] = simplex_index1[0]                                                   <L 1136>
    var_56 = wp::extract(var_simplex_index1, var_55);
    var_57 = &((var_pt).vert_index);
    var_59 = wp::load(var_57);
    wp::array_store(var_59, var_58, var_56);
    // pt.vert_index[1] = simplex_index2[0]                                                   <L 1137>
    var_61 = wp::extract(var_simplex_index2, var_60);
    var_62 = &((var_pt).vert_index);
    var_64 = wp::load(var_62);
    wp::array_store(var_64, var_63, var_61);
    // pt.vert_index[2] = simplex_index1[1]                                                   <L 1138>
    var_66 = wp::extract(var_simplex_index1, var_65);
    var_67 = &((var_pt).vert_index);
    var_69 = wp::load(var_67);
    wp::array_store(var_69, var_68, var_66);
    // pt.vert_index[3] = simplex_index2[1]                                                   <L 1139>
    var_71 = wp::extract(var_simplex_index2, var_70);
    var_72 = &((var_pt).vert_index);
    var_74 = wp::load(var_72);
    wp::array_store(var_74, var_73, var_71);
    // pt.vert_index[4] = simplex_index1[2]                                                   <L 1140>
    var_76 = wp::extract(var_simplex_index1, var_75);
    var_77 = &((var_pt).vert_index);
    var_79 = wp::load(var_77);
    wp::array_store(var_79, var_78, var_76);
    // pt.vert_index[5] = simplex_index2[2]                                                   <L 1141>
    var_81 = wp::extract(var_simplex_index2, var_80);
    var_82 = &((var_pt).vert_index);
    var_84 = wp::load(var_82);
    wp::array_store(var_84, var_83, var_81);
    // _epa_support(pt, 3, geom1, geom2, geomtype1, geomtype2, -n)                            <L 1143>
    var_86 = wp::neg(var_20);
    _epa_support_0(var_pt, var_85, var_geom1, var_geom2, var_geomtype1, var_geomtype2, var_86, var_87, var_88);
    // _epa_support(pt, 4, geom1, geom2, geomtype1, geomtype2, n)                             <L 1144>
    _epa_support_0(var_pt, var_89, var_geom1, var_geom2, var_geomtype1, var_geomtype2, var_20, var_90, var_91);
    // v1 = simplex[0]                                                                        <L 1146>
    var_93 = wp::extract(var_simplex, var_92);
    // v2 = simplex[1]                                                                        <L 1147>
    var_95 = wp::extract(var_simplex, var_94);
    // v3 = simplex[2]                                                                        <L 1148>
    var_97 = wp::extract(var_simplex, var_96);
    // v4 = pt.vert[6] - pt.vert[7]                                                           <L 1149>
    var_98 = &((var_pt).vert);
    var_101 = wp::load(var_98);
    var_100 = wp::address(var_101, var_99);
    var_102 = &((var_pt).vert);
    var_105 = wp::load(var_102);
    var_104 = wp::address(var_105, var_103);
    var_107 = wp::load(var_100);
    var_108 = wp::load(var_104);
    var_106 = wp::sub(var_107, var_108);
    // v5 = pt.vert[8] - pt.vert[9]                                                           <L 1150>
    var_109 = &((var_pt).vert);
    var_112 = wp::load(var_109);
    var_111 = wp::address(var_112, var_110);
    var_113 = &((var_pt).vert);
    var_116 = wp::load(var_113);
    var_115 = wp::address(var_116, var_114);
    var_118 = wp::load(var_111);
    var_119 = wp::load(var_115);
    var_117 = wp::sub(var_118, var_119);
    // if _tri_point_intersect(v1, v2, v3, v4):                                               <L 1153>
    var_120 = _tri_point_intersect_0(var_93, var_95, var_97, var_106);
    if (var_120) {
        // pt.status = 3                                                                      <L 1154>
        var_pt.status = var_121;
        // return pt                                                                          <L 1155>
        return var_pt;
    }
    // if _tri_point_intersect(v1, v2, v3, v5):                                               <L 1158>
    var_122 = _tri_point_intersect_0(var_93, var_95, var_97, var_117);
    if (var_122) {
        // pt.status = 4                                                                      <L 1159>
        var_pt.status = var_123;
        // return pt                                                                          <L 1160>
        return var_pt;
    }
    // if dist > 1e-5 and not _test_tetra(v1, v2, v3, v4) and not _test_tetra(v1, v2, v3, v5):       <L 1164>
    var_126 = (var_dist > var_125);
    var_124 = var_126;
    if (var_124) {
        var_127 = _test_tetra_0(var_93, var_95, var_97, var_106);
        var_128 = wp::unot(var_127);
        var_124 = var_124 && var_128;
    }
    if (var_124) {
        var_129 = _test_tetra_0(var_93, var_95, var_97, var_117);
        var_130 = wp::unot(var_129);
        var_124 = var_124 && var_130;
    }
    if (var_124) {
        // pt.status = 5                                                                      <L 1165>
        var_pt.status = var_131;
        // return pt                                                                          <L 1166>
        return var_pt;
    }
    // if _attach_face(pt, 0, 4, 0, 1) < MIN_DIST3:                                           <L 1169>
    var_136 = _attach_face_0(var_pt, var_132, var_133, var_134, var_135);
    var_138 = (var_136 < var_137);
    if (var_138) {
        // pt.status = 6                                                                      <L 1170>
        var_pt.status = var_139;
        // return pt                                                                          <L 1171>
        return var_pt;
    }
    // if _attach_face(pt, 1, 4, 2, 0) < MIN_DIST3:                                           <L 1172>
    var_144 = _attach_face_0(var_pt, var_140, var_141, var_142, var_143);
    var_145 = (var_144 < var_137);
    if (var_145) {
        // pt.status = 7                                                                      <L 1173>
        var_pt.status = var_146;
        // return pt                                                                          <L 1174>
        return var_pt;
    }
    // if _attach_face(pt, 2, 4, 1, 2) < MIN_DIST3:                                           <L 1175>
    var_151 = _attach_face_0(var_pt, var_147, var_148, var_149, var_150);
    var_152 = (var_151 < var_137);
    if (var_152) {
        // pt.status = 8                                                                      <L 1176>
        var_pt.status = var_153;
        // return pt                                                                          <L 1177>
        return var_pt;
    }
    // if _attach_face(pt, 3, 3, 1, 0) < MIN_DIST3:                                           <L 1178>
    var_158 = _attach_face_0(var_pt, var_154, var_155, var_156, var_157);
    var_159 = (var_158 < var_137);
    if (var_159) {
        // pt.status = 9                                                                      <L 1179>
        var_pt.status = var_160;
        // return pt                                                                          <L 1180>
        return var_pt;
    }
    // if _attach_face(pt, 4, 3, 0, 2) < MIN_DIST3:                                           <L 1181>
    var_165 = _attach_face_0(var_pt, var_161, var_162, var_163, var_164);
    var_166 = (var_165 < var_137);
    if (var_166) {
        // pt.status = 10                                                                     <L 1182>
        var_pt.status = var_167;
        // return pt                                                                          <L 1183>
        return var_pt;
    }
    // if _attach_face(pt, 5, 3, 2, 1) < MIN_DIST3:                                           <L 1184>
    var_172 = _attach_face_0(var_pt, var_168, var_169, var_170, var_171);
    var_173 = (var_172 < var_137);
    if (var_173) {
        // pt.status = 11                                                                     <L 1185>
        var_pt.status = var_174;
        // return pt                                                                          <L 1186>
        return var_pt;
    }
    // pt.nvert = 5                                                                           <L 1189>
    var_pt.nvert = var_175;
    // pt.nface = 6                                                                           <L 1190>
    var_pt.nface = var_176;
    // pt.status = 0                                                                          <L 1191>
    var_pt.status = var_177;
    // return pt                                                                              <L 1192>
    return var_pt;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1301
static CUDA_CALLABLE bool _is_invalid_face_0(
    wp::int32 var_face)
{
    //---------
    // primal vars
    wp::uint32 var_0;
    const wp::uint32 var_1 = 3221225472u;
    wp::uint32 var_2;
    bool var_3;
    //---------
    // forward
    // def _is_invalid_face(face: int) -> bool:                                               <L 1302>
    // return bool(wp.uint32(face) & _FACE_INVALID_OR_DELETED_MASK)                           <L 1304>
    var_0 = wp::uint32(var_face);
    var_2 = wp::bit_and(var_0, var_1);
    var_3 = bool(var_2);
    return var_3;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1283
static CUDA_CALLABLE wp::int32 _delete_face_0(
    wp::int32 var_face)
{
    //---------
    // primal vars
    wp::uint32 var_0;
    const wp::uint32 var_1 = 2147483648u;
    wp::uint32 var_2;
    wp::int32 var_3;
    //---------
    // forward
    // def _delete_face(face: int) -> int:                                                    <L 1284>
    // return int(wp.uint32(face) | _FACE_DELETED_BIT)                                        <L 1286>
    var_0 = wp::uint32(var_face);
    var_2 = wp::bit_or(var_0, var_1);
    var_3 = wp::int(var_2);
    return var_3;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1277
static CUDA_CALLABLE wp::vec_t<3, wp::int32> _get_face_verts_0(
    wp::int32 var_face)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1023;
    wp::int32 var_1;
    const wp::int32 var_2 = 10;
    wp::int32 var_3;
    const wp::int32 var_4 = 1023;
    wp::int32 var_5;
    const wp::int32 var_6 = 20;
    wp::int32 var_7;
    const wp::int32 var_8 = 1023;
    wp::int32 var_9;
    wp::vec_t<3, wp::int32> var_10;
    //---------
    // forward
    // def _get_face_verts(face: int) -> wp.vec3i:                                            <L 1278>
    // return wp.vec3i(face & 0x3FF, face >> 10 & 0x3FF, face >> 20 & 0x3FF)                  <L 1280>
    var_1 = wp::bit_and(var_face, var_0);
    var_3 = wp::rshift(var_face, var_2);
    var_5 = wp::bit_and(var_3, var_4);
    var_7 = wp::rshift(var_face, var_6);
    var_9 = wp::bit_and(var_7, var_8);
    var_10 = wp::vec_t<3, wp::int32>(var_1, var_5, var_9);
    return var_10;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:915
static CUDA_CALLABLE wp::int32 _add_edge_0(
    Polytope_10582b13 var_pt,
    wp::int32 var_e1,
    wp::int32 var_e2)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::int32 var_1;
    wp::int32 var_2;
    const wp::int32 var_3 = 0;
    bool var_4;
    const wp::int32 var_5 = -1;
    wp::int32 var_6;
    const wp::int32 var_7 = 10;
    wp::int32 var_8;
    wp::int32 var_9;
    wp::int32 var_10;
    wp::range_t var_11;
    wp::int32 var_12;
    wp::array_t<wp::int32>* var_13;
    wp::int32* var_14;
    wp::array_t<wp::int32> var_15;
    bool var_16;
    wp::int32 var_17;
    wp::array_t<wp::int32>* var_18;
    const wp::int32 var_19 = 1;
    wp::int32 var_20;
    wp::int32* var_21;
    wp::array_t<wp::int32> var_22;
    wp::array_t<wp::int32>* var_23;
    wp::array_t<wp::int32> var_24;
    wp::int32 var_25;
    const wp::int32 var_26 = 1;
    wp::int32 var_27;
    wp::array_t<wp::int32>* var_28;
    wp::shape_t* var_29;
    const wp::int32 var_30 = 0;
    wp::int32 var_31;
    wp::shape_t var_32;
    bool var_33;
    const wp::int32 var_34 = -1;
    wp::array_t<wp::int32>* var_35;
    wp::array_t<wp::int32> var_36;
    const wp::int32 var_37 = 1;
    wp::int32 var_38;
    //---------
    // forward
    // def _add_edge(pt: Polytope, e1: int, e2: int) -> int:                                  <L 916>
    // n = pt.nhorizon                                                                        <L 917>
    var_0 = &((var_pt).nhorizon);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    // if n < 0:                                                                              <L 919>
    var_4 = (var_1 < var_3);
    if (var_4) {
        // return -1                                                                          <L 920>
        return var_5;
    }
    // edge = (wp.min(e1, e2) << 10) | wp.max(e1, e2)                                         <L 922>
    var_6 = wp::min(var_e1, var_e2);
    var_8 = wp::lshift(var_6, var_7);
    var_9 = wp::max(var_e1, var_e2);
    var_10 = wp::bit_or(var_8, var_9);
    // for i in range(n):                                                                     <L 924>
    var_11 = wp::range(var_1);
    start_for_1:;
        if (iter_cmp(var_11) == 0) goto end_for_1;
        var_12 = wp::iter_next(var_11);
        // if edge == pt.horizon[i]:                                                          <L 925>
        var_13 = &((var_pt).horizon);
        var_15 = wp::load(var_13);
        var_14 = wp::address(var_15, var_12);
        var_17 = wp::load(var_14);
        var_16 = (var_10 == var_17);
        if (var_16) {
            // pt.horizon[i] = pt.horizon[n - 1]                                              <L 926>
            var_18 = &((var_pt).horizon);
            var_20 = wp::sub(var_1, var_19);
            var_22 = wp::load(var_18);
            var_21 = wp::address(var_22, var_20);
            var_23 = &((var_pt).horizon);
            var_24 = wp::load(var_23);
            var_25 = wp::load(var_21);
            wp::array_store(var_24, var_12, var_25);
            // return n - 1                                                                   <L 927>
            var_27 = wp::sub(var_1, var_26);
            return var_27;
        }
        goto start_for_1;
    end_for_1:;
    // if n == pt.horizon.shape[0]:                                                           <L 930>
    var_28 = &((var_pt).horizon);
    var_29 = &(var_28->shape);
    var_32 = wp::load(var_29);
    var_31 = wp::extract(var_32, var_30);
    var_33 = (var_1 == var_31);
    if (var_33) {
        // return -1                                                                          <L 931>
        return var_34;
    }
    // pt.horizon[n] = edge                                                                   <L 933>
    var_35 = &((var_pt).horizon);
    var_36 = wp::load(var_35);
    wp::array_store(var_36, var_1, var_10);
    // return n + 1                                                                           <L 934>
    var_38 = wp::add(var_1, var_37);
    return var_38;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1289
static CUDA_CALLABLE bool _is_face_deleted_0(
    wp::int32 var_face)
{
    //---------
    // primal vars
    wp::uint32 var_0;
    const wp::uint32 var_1 = 2147483648u;
    wp::uint32 var_2;
    bool var_3;
    //---------
    // forward
    // def _is_face_deleted(face: int) -> bool:                                               <L 1290>
    // return bool(wp.uint32(face) & _FACE_DELETED_BIT)                                       <L 1292>
    var_0 = wp::uint32(var_face);
    var_2 = wp::bit_and(var_0, var_1);
    var_3 = bool(var_2);
    return var_3;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:910
static CUDA_CALLABLE wp::vec_t<2, wp::int32> _get_edge_0(
    wp::int32 var_edge)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1023;
    wp::int32 var_1;
    const wp::int32 var_2 = 10;
    wp::int32 var_3;
    const wp::int32 var_4 = 1023;
    wp::int32 var_5;
    wp::vec_t<2, wp::int32> var_6;
    //---------
    // forward
    // def _get_edge(edge: int) -> wp.vec2i:                                                  <L 911>
    // return wp.vec2i(edge & 0x3FF, (edge >> 10) & 0x3FF)                                    <L 912>
    var_1 = wp::bit_and(var_edge, var_0);
    var_3 = wp::rshift(var_edge, var_2);
    var_5 = wp::bit_and(var_3, var_4);
    var_6 = wp::vec_t<2, wp::int32>(var_1, var_5);
    return var_6;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1295
static CUDA_CALLABLE wp::int32 _invalidate_face_0(
    wp::int32 var_face)
{
    //---------
    // primal vars
    wp::uint32 var_0;
    const wp::uint32 var_1 = 1073741824u;
    wp::uint32 var_2;
    wp::int32 var_3;
    //---------
    // forward
    // def _invalidate_face(face: int) -> int:                                                <L 1296>
    // return int(wp.uint32(face) | _FACE_INVALID_BIT)                                        <L 1298>
    var_0 = wp::uint32(var_face);
    var_2 = wp::bit_or(var_0, var_1);
    var_3 = wp::int(var_2);
    return var_3;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:937
static CUDA_CALLABLE void _epa_witness_0(
    Polytope_10582b13 var_pt,
    Geom_3242f8a8 var_geom1,
    Geom_3242f8a8 var_geom2,
    wp::int32 var_geomtype1,
    wp::int32 var_geomtype2,
    wp::int32 var_face_idx,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::float32 & ret_2)
{
    //---------
    // primal vars
    wp::array_t<wp::int32>* var_0;
    wp::int32* var_1;
    wp::array_t<wp::int32> var_2;
    wp::vec_t<3, wp::int32> var_3;
    wp::int32 var_4;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_5;
    const wp::int32 var_6 = 2;
    const wp::int32 var_7 = 0;
    wp::int32 var_8;
    wp::int32 var_9;
    wp::vec_t<3, wp::float32>* var_10;
    wp::array_t<wp::vec_t<3, wp::float32>> var_11;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_12;
    const wp::int32 var_13 = 2;
    const wp::int32 var_14 = 0;
    wp::int32 var_15;
    wp::int32 var_16;
    const wp::int32 var_17 = 1;
    wp::int32 var_18;
    wp::vec_t<3, wp::float32>* var_19;
    wp::array_t<wp::vec_t<3, wp::float32>> var_20;
    wp::vec_t<3, wp::float32> var_21;
    wp::vec_t<3, wp::float32> var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_24;
    const wp::int32 var_25 = 2;
    const wp::int32 var_26 = 1;
    wp::int32 var_27;
    wp::int32 var_28;
    wp::vec_t<3, wp::float32>* var_29;
    wp::array_t<wp::vec_t<3, wp::float32>> var_30;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_31;
    const wp::int32 var_32 = 2;
    const wp::int32 var_33 = 1;
    wp::int32 var_34;
    wp::int32 var_35;
    const wp::int32 var_36 = 1;
    wp::int32 var_37;
    wp::vec_t<3, wp::float32>* var_38;
    wp::array_t<wp::vec_t<3, wp::float32>> var_39;
    wp::vec_t<3, wp::float32> var_40;
    wp::vec_t<3, wp::float32> var_41;
    wp::vec_t<3, wp::float32> var_42;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_43;
    const wp::int32 var_44 = 2;
    const wp::int32 var_45 = 2;
    wp::int32 var_46;
    wp::int32 var_47;
    wp::vec_t<3, wp::float32>* var_48;
    wp::array_t<wp::vec_t<3, wp::float32>> var_49;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_50;
    const wp::int32 var_51 = 2;
    const wp::int32 var_52 = 2;
    wp::int32 var_53;
    wp::int32 var_54;
    const wp::int32 var_55 = 1;
    wp::int32 var_56;
    wp::vec_t<3, wp::float32>* var_57;
    wp::array_t<wp::vec_t<3, wp::float32>> var_58;
    wp::vec_t<3, wp::float32> var_59;
    wp::vec_t<3, wp::float32> var_60;
    wp::vec_t<3, wp::float32> var_61;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_62;
    wp::vec_t<3, wp::float32>* var_63;
    wp::array_t<wp::vec_t<3, wp::float32>> var_64;
    wp::vec_t<3, wp::float32> var_65;
    wp::vec_t<3, wp::float32> var_66;
    const wp::int32 var_67 = 0;
    wp::float32 var_68;
    const wp::int32 var_69 = 1;
    wp::float32 var_70;
    const wp::int32 var_71 = 2;
    wp::float32 var_72;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_73;
    const wp::int32 var_74 = 2;
    const wp::int32 var_75 = 0;
    wp::int32 var_76;
    wp::int32 var_77;
    const wp::int32 var_78 = 1;
    wp::int32 var_79;
    wp::vec_t<3, wp::float32>* var_80;
    wp::array_t<wp::vec_t<3, wp::float32>> var_81;
    wp::vec_t<3, wp::float32> var_82;
    wp::vec_t<3, wp::float32> var_83;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_84;
    const wp::int32 var_85 = 2;
    const wp::int32 var_86 = 1;
    wp::int32 var_87;
    wp::int32 var_88;
    const wp::int32 var_89 = 1;
    wp::int32 var_90;
    wp::vec_t<3, wp::float32>* var_91;
    wp::array_t<wp::vec_t<3, wp::float32>> var_92;
    wp::vec_t<3, wp::float32> var_93;
    wp::vec_t<3, wp::float32> var_94;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_95;
    const wp::int32 var_96 = 2;
    const wp::int32 var_97 = 2;
    wp::int32 var_98;
    wp::int32 var_99;
    const wp::int32 var_100 = 1;
    wp::int32 var_101;
    wp::vec_t<3, wp::float32>* var_102;
    wp::array_t<wp::vec_t<3, wp::float32>> var_103;
    wp::vec_t<3, wp::float32> var_104;
    wp::vec_t<3, wp::float32> var_105;
    wp::vec_t<3, wp::float32> var_106;
    const wp::int32 var_107 = 0;
    wp::float32 var_108;
    wp::float32 var_109;
    const wp::int32 var_110 = 0;
    wp::float32 var_111;
    wp::float32 var_112;
    wp::float32 var_113;
    const wp::int32 var_114 = 0;
    wp::float32 var_115;
    wp::float32 var_116;
    wp::float32 var_117;
    const wp::int32 var_118 = 0;
    const wp::int32 var_119 = 1;
    wp::float32 var_120;
    wp::float32 var_121;
    const wp::int32 var_122 = 1;
    wp::float32 var_123;
    wp::float32 var_124;
    wp::float32 var_125;
    const wp::int32 var_126 = 1;
    wp::float32 var_127;
    wp::float32 var_128;
    wp::float32 var_129;
    const wp::int32 var_130 = 1;
    const wp::int32 var_131 = 2;
    wp::float32 var_132;
    wp::float32 var_133;
    const wp::int32 var_134 = 2;
    wp::float32 var_135;
    wp::float32 var_136;
    wp::float32 var_137;
    const wp::int32 var_138 = 2;
    wp::float32 var_139;
    wp::float32 var_140;
    wp::float32 var_141;
    const wp::int32 var_142 = 2;
    wp::array_t<wp::int32>* var_143;
    const wp::int32 var_144 = 2;
    const wp::int32 var_145 = 0;
    wp::int32 var_146;
    wp::int32 var_147;
    wp::int32* var_148;
    wp::array_t<wp::int32> var_149;
    wp::int32 var_150;
    wp::int32 var_151;
    wp::array_t<wp::int32>* var_152;
    const wp::int32 var_153 = 2;
    const wp::int32 var_154 = 1;
    wp::int32 var_155;
    wp::int32 var_156;
    wp::int32* var_157;
    wp::array_t<wp::int32> var_158;
    wp::int32 var_159;
    wp::int32 var_160;
    wp::array_t<wp::int32>* var_161;
    const wp::int32 var_162 = 2;
    const wp::int32 var_163 = 2;
    wp::int32 var_164;
    wp::int32 var_165;
    wp::int32* var_166;
    wp::array_t<wp::int32> var_167;
    wp::int32 var_168;
    wp::int32 var_169;
    bool var_170;
    const wp::int32 var_171 = 1;
    bool var_172;
    bool var_173;
    bool var_174;
    bool var_175;
    const wp::float32 var_176 = 0.0;
    const wp::float32 var_177 = 0.0;
    const wp::float32 var_178 = 1.0;
    wp::vec_t<3, wp::float32> var_179;
    wp::mat_t<6, 3, wp::float32>* var_180;
    const wp::int32 var_181 = 3;
    wp::vec_t<3, wp::float32> var_182;
    wp::mat_t<6, 3, wp::float32> var_183;
    wp::mat_t<6, 3, wp::float32>* var_184;
    const wp::int32 var_185 = 4;
    wp::vec_t<3, wp::float32> var_186;
    wp::mat_t<6, 3, wp::float32> var_187;
    wp::mat_t<6, 3, wp::float32>* var_188;
    const wp::int32 var_189 = 5;
    wp::vec_t<3, wp::float32> var_190;
    wp::mat_t<6, 3, wp::float32> var_191;
    bool var_192;
    const wp::int32 var_193 = 3;
    bool var_194;
    const wp::int32 var_195 = 2;
    bool var_196;
    wp::vec_t<3, wp::float32>* var_197;
    const wp::int32 var_198 = 0;
    wp::float32 var_199;
    wp::vec_t<3, wp::float32> var_200;
    wp::float32* var_201;
    wp::float32 var_202;
    wp::float32 var_203;
    const wp::float32 var_204 = 0.0;
    const wp::float32 var_205 = 0.0;
    wp::vec_t<3, wp::float32>* var_206;
    const wp::int32 var_207 = 1;
    wp::float32 var_208;
    wp::vec_t<3, wp::float32> var_209;
    wp::vec_t<3, wp::float32>* var_210;
    const wp::int32 var_211 = 2;
    wp::float32 var_212;
    wp::vec_t<3, wp::float32> var_213;
    wp::vec_t<3, wp::float32> var_214;
    SupportPoint_e82efc60 var_215;
    wp::vec_t<3, wp::float32>* var_216;
    const wp::float32 var_217 = 0.5;
    wp::float32 var_218;
    wp::float32 var_219;
    wp::vec_t<3, wp::float32> var_220;
    wp::vec_t<3, wp::float32> var_221;
    wp::vec_t<3, wp::float32> var_222;
    wp::vec_t<3, wp::float32>* var_223;
    const wp::int32 var_224 = 0;
    wp::float32* var_225;
    wp::vec_t<3, wp::float32> var_226;
    wp::vec_t<3, wp::float32> var_227;
    SupportPoint_e82efc60 var_228;
    wp::vec_t<3, wp::float32>* var_229;
    wp::vec_t<3, wp::float32> var_230;
    wp::vec_t<3, wp::float32> var_231;
    wp::vec_t<3, wp::float32> var_232;
    SupportPoint_e82efc60 var_233;
    wp::vec_t<3, wp::float32> var_234;
    bool var_235;
    const wp::int32 var_236 = 0;
    wp::float32 var_237;
    const wp::float32 var_238 = 0.0;
    bool var_239;
    const wp::int32 var_240 = 1;
    wp::float32 var_241;
    const wp::float32 var_242 = 0.0;
    bool var_243;
    const wp::int32 var_244 = 2;
    wp::float32 var_245;
    const wp::float32 var_246 = 0.0;
    bool var_247;
    const wp::int32 var_248 = 0;
    wp::float32 var_249;
    wp::vec_t<3, wp::float32> var_250;
    const wp::int32 var_251 = 1;
    wp::float32 var_252;
    wp::vec_t<3, wp::float32> var_253;
    wp::vec_t<3, wp::float32> var_254;
    const wp::int32 var_255 = 2;
    wp::float32 var_256;
    wp::vec_t<3, wp::float32> var_257;
    wp::vec_t<3, wp::float32> var_258;
    wp::vec_t<3, wp::float32> var_259;
    const wp::int32 var_260 = 1;
    wp::float32 var_261;
    const wp::int32 var_262 = 0;
    bool var_263;
    wp::vec_t<3, wp::float32> var_264;
    const wp::int32 var_265 = 0;
    wp::float32 var_266;
    const wp::int32 var_267 = 0;
    bool var_268;
    wp::vec_t<3, wp::float32> var_269;
    wp::vec_t<3, wp::float32> var_270;
    wp::float32 var_271;
    wp::vec_t<3, wp::float32> var_272;
    wp::vec_t<3, wp::float32> var_273;
    wp::vec_t<3, wp::float32> var_274;
    wp::vec_t<3, wp::float32> var_275;
    wp::float32 var_276;
    wp::float32 var_277;
    wp::vec_t<3, wp::float32> var_278;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_279;
    const wp::int32 var_280 = 2;
    const wp::int32 var_281 = 0;
    wp::int32 var_282;
    wp::int32 var_283;
    wp::vec_t<3, wp::float32>* var_284;
    wp::array_t<wp::vec_t<3, wp::float32>> var_285;
    wp::vec_t<3, wp::float32> var_286;
    wp::vec_t<3, wp::float32> var_287;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_288;
    const wp::int32 var_289 = 2;
    const wp::int32 var_290 = 1;
    wp::int32 var_291;
    wp::int32 var_292;
    wp::vec_t<3, wp::float32>* var_293;
    wp::array_t<wp::vec_t<3, wp::float32>> var_294;
    wp::vec_t<3, wp::float32> var_295;
    wp::vec_t<3, wp::float32> var_296;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_297;
    const wp::int32 var_298 = 2;
    const wp::int32 var_299 = 2;
    wp::int32 var_300;
    wp::int32 var_301;
    wp::vec_t<3, wp::float32>* var_302;
    wp::array_t<wp::vec_t<3, wp::float32>> var_303;
    wp::vec_t<3, wp::float32> var_304;
    wp::vec_t<3, wp::float32> var_305;
    wp::vec_t<3, wp::float32> var_306;
    const wp::int32 var_307 = 0;
    wp::float32 var_308;
    wp::float32 var_309;
    const wp::int32 var_310 = 0;
    wp::float32 var_311;
    wp::float32 var_312;
    wp::float32 var_313;
    const wp::int32 var_314 = 0;
    wp::float32 var_315;
    wp::float32 var_316;
    wp::float32 var_317;
    const wp::int32 var_318 = 0;
    const wp::int32 var_319 = 1;
    wp::float32 var_320;
    wp::float32 var_321;
    const wp::int32 var_322 = 1;
    wp::float32 var_323;
    wp::float32 var_324;
    wp::float32 var_325;
    const wp::int32 var_326 = 1;
    wp::float32 var_327;
    wp::float32 var_328;
    wp::float32 var_329;
    const wp::int32 var_330 = 1;
    const wp::int32 var_331 = 2;
    wp::float32 var_332;
    wp::float32 var_333;
    const wp::int32 var_334 = 2;
    wp::float32 var_335;
    wp::float32 var_336;
    wp::float32 var_337;
    const wp::int32 var_338 = 2;
    wp::float32 var_339;
    wp::float32 var_340;
    wp::float32 var_341;
    const wp::int32 var_342 = 2;
    wp::array_t<wp::float32>* var_343;
    wp::float32* var_344;
    wp::array_t<wp::float32> var_345;
    wp::float32 var_346;
    wp::float32 var_347;
    wp::float32 var_348;
    //---------
    // forward
    // def _epa_witness(                                                                      <L 938>
    // face = _get_face_verts(pt.face[face_idx])                                              <L 941>
    var_0 = &((var_pt).face);
    var_2 = wp::load(var_0);
    var_1 = wp::address(var_2, var_face_idx);
    var_4 = wp::load(var_1);
    var_3 = _get_face_verts_0(var_4);
    // v1 = pt.vert[2 * face[0]] - pt.vert[2 * face[0] + 1]                                   <L 943>
    var_5 = &((var_pt).vert);
    var_8 = wp::extract(var_3, var_7);
    var_9 = wp::mul(var_6, var_8);
    var_11 = wp::load(var_5);
    var_10 = wp::address(var_11, var_9);
    var_12 = &((var_pt).vert);
    var_15 = wp::extract(var_3, var_14);
    var_16 = wp::mul(var_13, var_15);
    var_18 = wp::add(var_16, var_17);
    var_20 = wp::load(var_12);
    var_19 = wp::address(var_20, var_18);
    var_22 = wp::load(var_10);
    var_23 = wp::load(var_19);
    var_21 = wp::sub(var_22, var_23);
    // v2 = pt.vert[2 * face[1]] - pt.vert[2 * face[1] + 1]                                   <L 944>
    var_24 = &((var_pt).vert);
    var_27 = wp::extract(var_3, var_26);
    var_28 = wp::mul(var_25, var_27);
    var_30 = wp::load(var_24);
    var_29 = wp::address(var_30, var_28);
    var_31 = &((var_pt).vert);
    var_34 = wp::extract(var_3, var_33);
    var_35 = wp::mul(var_32, var_34);
    var_37 = wp::add(var_35, var_36);
    var_39 = wp::load(var_31);
    var_38 = wp::address(var_39, var_37);
    var_41 = wp::load(var_29);
    var_42 = wp::load(var_38);
    var_40 = wp::sub(var_41, var_42);
    // v3 = pt.vert[2 * face[2]] - pt.vert[2 * face[2] + 1]                                   <L 945>
    var_43 = &((var_pt).vert);
    var_46 = wp::extract(var_3, var_45);
    var_47 = wp::mul(var_44, var_46);
    var_49 = wp::load(var_43);
    var_48 = wp::address(var_49, var_47);
    var_50 = &((var_pt).vert);
    var_53 = wp::extract(var_3, var_52);
    var_54 = wp::mul(var_51, var_53);
    var_56 = wp::add(var_54, var_55);
    var_58 = wp::load(var_50);
    var_57 = wp::address(var_58, var_56);
    var_60 = wp::load(var_48);
    var_61 = wp::load(var_57);
    var_59 = wp::sub(var_60, var_61);
    // coordinates = _tri_affine_coord(v1, v2, v3, pt.face_pr[face_idx])                      <L 947>
    var_62 = &((var_pt).face_pr);
    var_64 = wp::load(var_62);
    var_63 = wp::address(var_64, var_face_idx);
    var_66 = wp::load(var_63);
    var_65 = _tri_affine_coord_0(var_21, var_40, var_59, var_66);
    // l1 = coordinates[0]                                                                    <L 948>
    var_68 = wp::extract(var_65, var_67);
    // l2 = coordinates[1]                                                                    <L 949>
    var_70 = wp::extract(var_65, var_69);
    // l3 = coordinates[2]                                                                    <L 950>
    var_72 = wp::extract(var_65, var_71);
    // v1 = pt.vert[2 * face[0] + 1]                                                          <L 953>
    var_73 = &((var_pt).vert);
    var_76 = wp::extract(var_3, var_75);
    var_77 = wp::mul(var_74, var_76);
    var_79 = wp::add(var_77, var_78);
    var_81 = wp::load(var_73);
    var_80 = wp::address(var_81, var_79);
    var_83 = wp::load(var_80);
    var_82 = wp::copy(var_83);
    // v2 = pt.vert[2 * face[1] + 1]                                                          <L 954>
    var_84 = &((var_pt).vert);
    var_87 = wp::extract(var_3, var_86);
    var_88 = wp::mul(var_85, var_87);
    var_90 = wp::add(var_88, var_89);
    var_92 = wp::load(var_84);
    var_91 = wp::address(var_92, var_90);
    var_94 = wp::load(var_91);
    var_93 = wp::copy(var_94);
    // v3 = pt.vert[2 * face[2] + 1]                                                          <L 955>
    var_95 = &((var_pt).vert);
    var_98 = wp::extract(var_3, var_97);
    var_99 = wp::mul(var_96, var_98);
    var_101 = wp::add(var_99, var_100);
    var_103 = wp::load(var_95);
    var_102 = wp::address(var_103, var_101);
    var_105 = wp::load(var_102);
    var_104 = wp::copy(var_105);
    // x2 = wp.vec3()                                                                         <L 956>
    var_106 = wp::vec_t<3, wp::float32>();
    // x2[0] = v1[0] * l1 + v2[0] * l2 + v3[0] * l3                                           <L 957>
    var_108 = wp::extract(var_82, var_107);
    var_109 = wp::mul(var_108, var_68);
    var_111 = wp::extract(var_93, var_110);
    var_112 = wp::mul(var_111, var_70);
    var_113 = wp::add(var_109, var_112);
    var_115 = wp::extract(var_104, var_114);
    var_116 = wp::mul(var_115, var_72);
    var_117 = wp::add(var_113, var_116);
    wp::assign_inplace(var_106, var_118, var_117);
    // x2[1] = v1[1] * l1 + v2[1] * l2 + v3[1] * l3                                           <L 958>
    var_120 = wp::extract(var_82, var_119);
    var_121 = wp::mul(var_120, var_68);
    var_123 = wp::extract(var_93, var_122);
    var_124 = wp::mul(var_123, var_70);
    var_125 = wp::add(var_121, var_124);
    var_127 = wp::extract(var_104, var_126);
    var_128 = wp::mul(var_127, var_72);
    var_129 = wp::add(var_125, var_128);
    wp::assign_inplace(var_106, var_130, var_129);
    // x2[2] = v1[2] * l1 + v2[2] * l2 + v3[2] * l3                                           <L 959>
    var_132 = wp::extract(var_82, var_131);
    var_133 = wp::mul(var_132, var_68);
    var_135 = wp::extract(var_93, var_134);
    var_136 = wp::mul(var_135, var_70);
    var_137 = wp::add(var_133, var_136);
    var_139 = wp::extract(var_104, var_138);
    var_140 = wp::mul(var_139, var_72);
    var_141 = wp::add(var_137, var_140);
    wp::assign_inplace(var_106, var_142, var_141);
    // i1 = pt.vert_index[2 * face[0]]                                                        <L 962>
    var_143 = &((var_pt).vert_index);
    var_146 = wp::extract(var_3, var_145);
    var_147 = wp::mul(var_144, var_146);
    var_149 = wp::load(var_143);
    var_148 = wp::address(var_149, var_147);
    var_151 = wp::load(var_148);
    var_150 = wp::copy(var_151);
    // i2 = pt.vert_index[2 * face[1]]                                                        <L 963>
    var_152 = &((var_pt).vert_index);
    var_155 = wp::extract(var_3, var_154);
    var_156 = wp::mul(var_153, var_155);
    var_158 = wp::load(var_152);
    var_157 = wp::address(var_158, var_156);
    var_160 = wp::load(var_157);
    var_159 = wp::copy(var_160);
    // i3 = pt.vert_index[2 * face[2]]                                                        <L 964>
    var_161 = &((var_pt).vert_index);
    var_164 = wp::extract(var_3, var_163);
    var_165 = wp::mul(var_162, var_164);
    var_167 = wp::load(var_161);
    var_166 = wp::address(var_167, var_165);
    var_169 = wp::load(var_166);
    var_168 = wp::copy(var_169);
    // if geomtype1 == GeomType.HFIELD and (i1 != i2 or i1 != i3):                            <L 965>
    var_172 = (var_geomtype1 == var_171);
    var_170 = var_172;
    if (var_170) {
        var_174 = (var_150 != var_159);
        var_173 = var_174;
        if (!var_173) {
            var_175 = (var_150 != var_168);
            var_173 = var_173 || var_175;
        }
        var_170 = var_170 && var_173;
    }
    if (var_170) {
        // n = wp.vec3(0.0, 0.0, 1.0)                                                         <L 967>
        var_179 = wp::vec_t<3, wp::float32>(var_176, var_177, var_178);
        // a = geom1.hfprism[3]                                                               <L 970>
        var_180 = &((var_geom1).hfprism);
        var_183 = wp::load(var_180);
        var_182 = wp::extract(var_183, var_181);
        // b = geom1.hfprism[4]                                                               <L 971>
        var_184 = &((var_geom1).hfprism);
        var_187 = wp::load(var_184);
        var_186 = wp::extract(var_187, var_185);
        // c = geom1.hfprism[5]                                                               <L 972>
        var_188 = &((var_geom1).hfprism);
        var_191 = wp::load(var_188);
        var_190 = wp::extract(var_191, var_189);
        // if geomtype2 == GeomType.CAPSULE or geomtype2 == GeomType.SPHERE:                  <L 975>
        var_194 = (var_geomtype2 == var_193);
        var_192 = var_194;
        if (!var_192) {
            var_196 = (var_geomtype2 == var_195);
            var_192 = var_192 || var_196;
        }
        if (var_192) {
            // radius = geom2.size[0]                                                         <L 976>
            var_197 = &((var_geom2).size);
            var_200 = wp::load(var_197);
            var_199 = wp::extract(var_200, var_198);
            // margin = geom2.margin                                                          <L 977>
            var_201 = &((var_geom2).margin);
            var_203 = wp::load(var_201);
            var_202 = wp::copy(var_203);
            // geom2.margin = 0.0                                                             <L 978>
            var_geom2.margin = var_204;
            // geom2.size = wp.vec3(0.0, geom2.size[1], geom2.size[2])                        <L 979>
            var_206 = &((var_geom2).size);
            var_209 = wp::load(var_206);
            var_208 = wp::extract(var_209, var_207);
            var_210 = &((var_geom2).size);
            var_213 = wp::load(var_210);
            var_212 = wp::extract(var_213, var_211);
            var_214 = wp::vec_t<3, wp::float32>(var_205, var_208, var_212);
            var_geom2.size = var_214;
            // sp = support(geom2, geomtype2, x2)                                             <L 980>
            var_215 = support_0(var_geom2, var_geomtype2, var_106);
            // x2 = sp.point - (0.5 * margin + radius) * n                                    <L 981>
            var_216 = &((var_215).point);
            var_218 = wp::mul(var_217, var_202);
            var_219 = wp::add(var_218, var_199);
            var_220 = wp::mul(var_219, var_179);
            var_222 = wp::load(var_216);
            var_221 = wp::sub(var_222, var_220);
            // geom2.size[0] = radius                                                         <L 982>
            var_223 = &((var_geom2).size);
            var_225 = wp::indexref(var_223, var_224);
            wp::store(var_225, var_199);
            // geom2.margin = margin                                                          <L 983>
            var_geom2.margin = var_202;
        }
        var_226 = wp::where(var_192, var_221, var_106);
        if (!var_192) {
            // x2 = wp.normalize(x2)                                                          <L 985>
            var_227 = wp::normalize(var_226);
            // sp = support(geom2, geomtype2, x2)                                             <L 986>
            var_228 = support_0(var_geom2, var_geomtype2, var_227);
            // x2 = sp.point                                                                  <L 987>
            var_229 = &((var_228).point);
            var_231 = wp::load(var_229);
            var_230 = wp::copy(var_231);
        }
        var_232 = wp::where(var_192, var_226, var_230);
        var_233 = wp::where(var_192, var_215, var_228);
        // coordinates2 = _tri_affine_coord(a, b, c, x2)                                      <L 989>
        var_234 = _tri_affine_coord_0(var_182, var_186, var_190, var_232);
        // if coordinates2[0] > 0.0 and coordinates2[1] > 0.0 and coordinates2[2] > 0.0:       <L 990>
        var_237 = wp::extract(var_234, var_236);
        var_239 = (var_237 > var_238);
        var_235 = var_239;
        if (var_235) {
            var_241 = wp::extract(var_234, var_240);
            var_243 = (var_241 > var_242);
            var_235 = var_235 && var_243;
        }
        if (var_235) {
            var_245 = wp::extract(var_234, var_244);
            var_247 = (var_245 > var_246);
            var_235 = var_235 && var_247;
        }
        if (var_235) {
            // x1 = coordinates2[0] * a + coordinates2[1] * b + coordinates2[2] * c           <L 991>
            var_249 = wp::extract(var_234, var_248);
            var_250 = wp::mul(var_249, var_182);
            var_252 = wp::extract(var_234, var_251);
            var_253 = wp::mul(var_252, var_186);
            var_254 = wp::add(var_250, var_253);
            var_256 = wp::extract(var_234, var_255);
            var_257 = wp::mul(var_256, var_190);
            var_258 = wp::add(var_254, var_257);
        }
        if (!var_235) {
            // p = c                                                                          <L 993>
            var_259 = wp::copy(var_190);
            // p = wp.where(coordinates2[1] > 0, b, p)                                        <L 994>
            var_261 = wp::extract(var_234, var_260);
            var_263 = (var_261 > var_262);
            var_264 = wp::where(var_263, var_186, var_259);
            // p = wp.where(coordinates2[0] > 0, a, p)                                        <L 995>
            var_266 = wp::extract(var_234, var_265);
            var_268 = (var_266 > var_267);
            var_269 = wp::where(var_268, var_182, var_264);
            // x1 = x2 - wp.dot(x2 - p, n) * n                                                <L 996>
            var_270 = wp::sub(var_232, var_269);
            var_271 = wp::dot(var_270, var_179);
            var_272 = wp::mul(var_271, var_179);
            var_273 = wp::sub(var_232, var_272);
        }
        var_274 = wp::where(var_235, var_258, var_273);
        // return x1, x2, -wp.norm_l2(x1 - x2)                                                <L 997>
        var_275 = wp::sub(var_274, var_232);
        var_276 = norm_l2_0(var_275);
        var_277 = wp::neg(var_276);
        ret_0 = var_274;
        ret_1 = var_232;
        ret_2 = var_277;
        return;
    }
    var_278 = wp::where(var_170, var_232, var_106);
    // v1 = pt.vert[2 * face[0]]                                                              <L 1000>
    var_279 = &((var_pt).vert);
    var_282 = wp::extract(var_3, var_281);
    var_283 = wp::mul(var_280, var_282);
    var_285 = wp::load(var_279);
    var_284 = wp::address(var_285, var_283);
    var_287 = wp::load(var_284);
    var_286 = wp::copy(var_287);
    // v2 = pt.vert[2 * face[1]]                                                              <L 1001>
    var_288 = &((var_pt).vert);
    var_291 = wp::extract(var_3, var_290);
    var_292 = wp::mul(var_289, var_291);
    var_294 = wp::load(var_288);
    var_293 = wp::address(var_294, var_292);
    var_296 = wp::load(var_293);
    var_295 = wp::copy(var_296);
    // v3 = pt.vert[2 * face[2]]                                                              <L 1002>
    var_297 = &((var_pt).vert);
    var_300 = wp::extract(var_3, var_299);
    var_301 = wp::mul(var_298, var_300);
    var_303 = wp::load(var_297);
    var_302 = wp::address(var_303, var_301);
    var_305 = wp::load(var_302);
    var_304 = wp::copy(var_305);
    // x1 = wp.vec3()                                                                         <L 1003>
    var_306 = wp::vec_t<3, wp::float32>();
    // x1[0] = v1[0] * l1 + v2[0] * l2 + v3[0] * l3                                           <L 1004>
    var_308 = wp::extract(var_286, var_307);
    var_309 = wp::mul(var_308, var_68);
    var_311 = wp::extract(var_295, var_310);
    var_312 = wp::mul(var_311, var_70);
    var_313 = wp::add(var_309, var_312);
    var_315 = wp::extract(var_304, var_314);
    var_316 = wp::mul(var_315, var_72);
    var_317 = wp::add(var_313, var_316);
    wp::assign_inplace(var_306, var_318, var_317);
    // x1[1] = v1[1] * l1 + v2[1] * l2 + v3[1] * l3                                           <L 1005>
    var_320 = wp::extract(var_286, var_319);
    var_321 = wp::mul(var_320, var_68);
    var_323 = wp::extract(var_295, var_322);
    var_324 = wp::mul(var_323, var_70);
    var_325 = wp::add(var_321, var_324);
    var_327 = wp::extract(var_304, var_326);
    var_328 = wp::mul(var_327, var_72);
    var_329 = wp::add(var_325, var_328);
    wp::assign_inplace(var_306, var_330, var_329);
    // x1[2] = v1[2] * l1 + v2[2] * l2 + v3[2] * l3                                           <L 1006>
    var_332 = wp::extract(var_286, var_331);
    var_333 = wp::mul(var_332, var_68);
    var_335 = wp::extract(var_295, var_334);
    var_336 = wp::mul(var_335, var_70);
    var_337 = wp::add(var_333, var_336);
    var_339 = wp::extract(var_304, var_338);
    var_340 = wp::mul(var_339, var_72);
    var_341 = wp::add(var_337, var_340);
    wp::assign_inplace(var_306, var_342, var_341);
    // return x1, x2, -wp.sqrt(pt.face_norm2[face_idx])                                       <L 1008>
    var_343 = &((var_pt).face_norm2);
    var_345 = wp::load(var_343);
    var_344 = wp::address(var_345, var_face_idx);
    var_347 = wp::load(var_344);
    var_346 = wp::sqrt(var_347);
    var_348 = wp::neg(var_346);
    ret_0 = var_306;
    ret_1 = var_278;
    ret_2 = var_348;
    return;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1307
static CUDA_CALLABLE void _epa_0(
    wp::float32 var_tolerance,
    wp::int32 var_epa_iterations,
    Polytope_10582b13 var_pt,
    Geom_3242f8a8 var_geom1,
    Geom_3242f8a8 var_geom2,
    wp::int32 var_geomtype1,
    wp::int32 var_geomtype2,
    bool var_is_discrete,
    bool var_warn_overflow,
    wp::int32 var_worldid,
    wp::array_t<wp::int32> var_overflow_out,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::int32 & ret_3)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 1e+30;
    wp::float32 var_1;
    wp::float32 var_2;
    const wp::int32 var_3 = -1;
    wp::int32 var_4;
    const wp::int32 var_5 = -1;
    wp::int32 var_6;
    const wp::float32 var_7 = 1e-07;
    wp::float32 var_8;
    wp::int32* var_9;
    wp::int32 var_10;
    wp::int32 var_11;
    const wp::int32 var_12 = 1000;
    wp::int32 var_13;
    wp::range_t var_14;
    wp::int32 var_15;
    wp::int32 var_16;
    const wp::int32 var_17 = -1;
    wp::int32 var_18;
    wp::float32 var_19;
    wp::int32* var_20;
    wp::range_t var_21;
    wp::int32 var_22;
    wp::int32 var_23;
    bool var_24;
    wp::array_t<wp::int32>* var_25;
    wp::int32* var_26;
    wp::array_t<wp::int32> var_27;
    bool var_28;
    wp::int32 var_29;
    bool var_30;
    wp::array_t<wp::float32>* var_31;
    wp::float32* var_32;
    wp::array_t<wp::float32> var_33;
    bool var_34;
    wp::float32 var_35;
    wp::int32 var_36;
    wp::array_t<wp::float32>* var_37;
    wp::float32* var_38;
    wp::array_t<wp::float32> var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::int32 var_42;
    wp::float32 var_43;
    bool var_44;
    bool var_45;
    const wp::int32 var_46 = 0;
    bool var_47;
    wp::int32 var_48;
    wp::int32 var_49;
    wp::int32 var_50;
    const wp::float32 var_51 = 0.0;
    bool var_52;
    wp::int32 var_53;
    wp::int32 var_54;
    wp::float32 var_55;
    wp::int32* var_56;
    wp::int32 var_57;
    wp::int32 var_58;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_59;
    wp::vec_t<3, wp::float32>* var_60;
    wp::array_t<wp::vec_t<3, wp::float32>> var_61;
    wp::vec_t<3, wp::float32> var_62;
    wp::vec_t<3, wp::float32> var_63;
    wp::vec_t<3, wp::float32> var_64;
    wp::int32 var_65;
    wp::int32 var_66;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_67;
    const wp::int32 var_68 = 2;
    wp::int32 var_69;
    wp::vec_t<3, wp::float32>* var_70;
    wp::array_t<wp::vec_t<3, wp::float32>> var_71;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_72;
    const wp::int32 var_73 = 2;
    wp::int32 var_74;
    const wp::int32 var_75 = 1;
    wp::int32 var_76;
    wp::vec_t<3, wp::float32>* var_77;
    wp::array_t<wp::vec_t<3, wp::float32>> var_78;
    wp::vec_t<3, wp::float32> var_79;
    wp::vec_t<3, wp::float32> var_80;
    wp::vec_t<3, wp::float32> var_81;
    const wp::int32 var_82 = 1;
    wp::int32* var_83;
    wp::int32 var_84;
    wp::int32 var_85;
    wp::float32 var_86;
    wp::float32 var_87;
    bool var_88;
    wp::float32 var_89;
    wp::float32 var_90;
    wp::float32 var_91;
    wp::float32 var_92;
    wp::float32 var_93;
    bool var_94;
    wp::float32 var_95;
    wp::float32 var_96;
    wp::int32 var_97;
    wp::int32 var_98;
    const bool var_99 = false;
    bool var_100;
    wp::int32* var_101;
    const wp::int32 var_102 = 1;
    wp::int32 var_103;
    wp::int32 var_104;
    wp::range_t var_105;
    wp::int32 var_106;
    bool var_107;
    wp::array_t<wp::int32>* var_108;
    const wp::int32 var_109 = 2;
    wp::int32 var_110;
    wp::int32* var_111;
    wp::array_t<wp::int32> var_112;
    wp::array_t<wp::int32>* var_113;
    const wp::int32 var_114 = 2;
    wp::int32 var_115;
    wp::int32* var_116;
    wp::array_t<wp::int32> var_117;
    bool var_118;
    wp::int32 var_119;
    wp::int32 var_120;
    wp::array_t<wp::int32>* var_121;
    const wp::int32 var_122 = 2;
    wp::int32 var_123;
    const wp::int32 var_124 = 1;
    wp::int32 var_125;
    wp::int32* var_126;
    wp::array_t<wp::int32> var_127;
    wp::array_t<wp::int32>* var_128;
    const wp::int32 var_129 = 2;
    wp::int32 var_130;
    const wp::int32 var_131 = 1;
    wp::int32 var_132;
    wp::int32* var_133;
    wp::array_t<wp::int32> var_134;
    bool var_135;
    wp::int32 var_136;
    wp::int32 var_137;
    const bool var_138 = true;
    wp::float32 var_139;
    wp::float32 var_140;
    wp::int32 var_141;
    wp::int32 var_142;
    wp::float32 var_143;
    wp::float32 var_144;
    wp::int32 var_145;
    wp::int32 var_146;
    wp::int32 var_147;
    const wp::int32 var_148 = 1;
    wp::int32 var_149;
    wp::array_t<wp::int32>* var_150;
    wp::int32* var_151;
    wp::array_t<wp::int32> var_152;
    wp::int32 var_153;
    wp::int32 var_154;
    wp::array_t<wp::int32>* var_155;
    wp::array_t<wp::int32> var_156;
    wp::array_t<wp::int32>* var_157;
    wp::int32* var_158;
    wp::array_t<wp::int32> var_159;
    wp::vec_t<3, wp::int32> var_160;
    wp::int32 var_161;
    const wp::int32 var_162 = 0;
    wp::int32 var_163;
    const wp::int32 var_164 = 1;
    wp::int32 var_165;
    wp::int32 var_166;
    const wp::int32 var_167 = 1;
    wp::int32 var_168;
    const wp::int32 var_169 = 2;
    wp::int32 var_170;
    wp::int32 var_171;
    const wp::int32 var_172 = 2;
    wp::int32 var_173;
    const wp::int32 var_174 = 0;
    wp::int32 var_175;
    wp::int32 var_176;
    wp::int32* var_177;
    const wp::int32 var_178 = -1;
    bool var_179;
    wp::int32 var_180;
    const wp::str var_181 = "Warning: EPA horizon = %d isn't large enough.\n";
    wp::array_t<wp::int32>* var_182;
    wp::shape_t* var_183;
    const wp::int32 var_184 = 0;
    wp::int32 var_185;
    wp::shape_t var_186;
    const wp::int32 var_187 = 256;
    const wp::int32 var_188 = 256;
    wp::int32 var_189;
    const wp::int32 var_190 = -1;
    wp::float32 var_191;
    wp::float32 var_192;
    wp::int32 var_193;
    wp::int32 var_194;
    wp::int32 var_195;
    wp::int32* var_196;
    wp::range_t var_197;
    wp::int32 var_198;
    wp::int32 var_199;
    wp::array_t<wp::int32>* var_200;
    wp::int32* var_201;
    wp::array_t<wp::int32> var_202;
    bool var_203;
    wp::int32 var_204;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_205;
    wp::vec_t<3, wp::float32>* var_206;
    wp::array_t<wp::vec_t<3, wp::float32>> var_207;
    wp::float32 var_208;
    wp::vec_t<3, wp::float32> var_209;
    wp::array_t<wp::float32>* var_210;
    wp::float32* var_211;
    wp::array_t<wp::float32> var_212;
    wp::float32 var_213;
    wp::float32 var_214;
    const wp::float32 var_215 = 1e-10;
    bool var_216;
    wp::array_t<wp::int32>* var_217;
    wp::int32* var_218;
    wp::array_t<wp::int32> var_219;
    bool var_220;
    wp::int32 var_221;
    const wp::int32 var_222 = 1;
    wp::int32 var_223;
    wp::int32 var_224;
    wp::array_t<wp::int32>* var_225;
    wp::int32* var_226;
    wp::array_t<wp::int32> var_227;
    wp::int32 var_228;
    wp::int32 var_229;
    wp::array_t<wp::int32>* var_230;
    wp::array_t<wp::int32> var_231;
    wp::array_t<wp::int32>* var_232;
    wp::int32* var_233;
    wp::array_t<wp::int32> var_234;
    wp::vec_t<3, wp::int32> var_235;
    wp::int32 var_236;
    const wp::int32 var_237 = 0;
    wp::int32 var_238;
    const wp::int32 var_239 = 1;
    wp::int32 var_240;
    wp::int32 var_241;
    const wp::int32 var_242 = 1;
    wp::int32 var_243;
    const wp::int32 var_244 = 2;
    wp::int32 var_245;
    wp::int32 var_246;
    const wp::int32 var_247 = 2;
    wp::int32 var_248;
    const wp::int32 var_249 = 0;
    wp::int32 var_250;
    wp::int32 var_251;
    wp::int32* var_252;
    const wp::int32 var_253 = -1;
    bool var_254;
    wp::int32 var_255;
    const wp::str var_256 = "Warning: EPA horizon = %d isn't large enough.\n";
    wp::array_t<wp::int32>* var_257;
    wp::shape_t* var_258;
    const wp::int32 var_259 = 0;
    wp::int32 var_260;
    wp::shape_t var_261;
    const wp::int32 var_262 = 256;
    const wp::int32 var_263 = 256;
    wp::int32 var_264;
    const wp::int32 var_265 = -1;
    wp::int32 var_266;
    wp::vec_t<3, wp::int32> var_267;
    wp::int32 var_268;
    wp::vec_t<3, wp::int32> var_269;
    wp::int32* var_270;
    wp::range_t var_271;
    wp::int32 var_272;
    wp::int32 var_273;
    wp::array_t<wp::int32>* var_274;
    wp::int32* var_275;
    wp::array_t<wp::int32> var_276;
    wp::vec_t<2, wp::int32> var_277;
    wp::int32 var_278;
    wp::int32* var_279;
    const wp::int32 var_280 = 0;
    wp::int32 var_281;
    const wp::int32 var_282 = 1;
    wp::int32 var_283;
    wp::float32 var_284;
    wp::int32 var_285;
    const wp::float32 var_286 = 0.0;
    bool var_287;
    const wp::int32 var_288 = -1;
    const wp::int32 var_289 = 1;
    wp::int32* var_290;
    wp::int32 var_291;
    wp::int32 var_292;
    bool var_293;
    bool var_294;
    bool var_295;
    const wp::int32 var_296 = 1;
    wp::int32 var_297;
    wp::int32 var_298;
    wp::array_t<wp::int32>* var_299;
    wp::int32* var_300;
    const wp::int32 var_301 = 1;
    wp::int32 var_302;
    wp::int32 var_303;
    wp::int32* var_304;
    wp::array_t<wp::int32> var_305;
    wp::int32 var_306;
    wp::int32 var_307;
    wp::array_t<wp::int32>* var_308;
    wp::int32* var_309;
    const wp::int32 var_310 = 1;
    wp::int32 var_311;
    wp::int32 var_312;
    wp::array_t<wp::int32> var_313;
    bool var_314;
    const wp::int32 var_315 = 0;
    bool var_316;
    const wp::int32 var_317 = -1;
    bool var_318;
    wp::float32 var_319;
    wp::float32 var_320;
    wp::int32 var_321;
    wp::int32 var_322;
    wp::int32 var_323;
    const wp::int32 var_324 = 0;
    const wp::int32 var_325 = -1;
    bool var_326;
    wp::vec_t<3, wp::float32> var_327;
    wp::vec_t<3, wp::float32> var_328;
    wp::float32 var_329;
    const wp::float32 var_330 = 0.0;
    wp::vec_t<3, wp::float32> var_331;
    wp::vec_t<3, wp::float32> var_332;
    const wp::int32 var_333 = -1;
    //---------
    // forward
    // def _epa(                                                                              <L 1308>
    // upper = FLOAT_MAX                                                                      <L 1324>
    var_1 = wp::copy(var_0);
    // upper2 = FLOAT_MAX                                                                     <L 1325>
    var_2 = wp::copy(var_0);
    // idx = int(-1)                                                                          <L 1326>
    var_4 = wp::int(var_3);
    // pidx = int(-1)                                                                         <L 1327>
    var_6 = wp::int(var_5);
    // epsilon = wp.where(is_discrete, MIN_EPATOL, tolerance)                                 <L 1328>
    var_8 = wp::where(var_is_discrete, var_7, var_tolerance);
    // nvalid = pt.nface  # number of potential faces for expanding the polytope              <L 1329>
    var_9 = &((var_pt).nface);
    var_11 = wp::load(var_9);
    var_10 = wp::copy(var_11);
    // epa_iterations = wp.min(epa_iterations, 1000)                                          <L 1334>
    var_13 = wp::min(var_epa_iterations, var_12);
    // for _ in range(epa_iterations):                                                        <L 1335>
    var_14 = wp::range(var_13);
    start_for_0:;
        if (iter_cmp(var_14) == 0) goto end_for_0;
        var_15 = wp::iter_next(var_14);
        // pidx = idx                                                                         <L 1336>
        var_16 = wp::copy(var_4);
        // idx = int(-1)                                                                      <L 1337>
        var_18 = wp::int(var_17);
        // lower2 = float(FLOAT_MAX)                                                          <L 1338>
        var_19 = wp::float(var_0);
        // for i in range(pt.nface):                                                          <L 1341>
        var_20 = &((var_pt).nface);
        var_22 = wp::load(var_20);
        var_21 = wp::range(var_22);
        start_for_2:;
            if (iter_cmp(var_21) == 0) goto end_for_2;
            var_23 = wp::iter_next(var_21);
            // if not _is_invalid_face(pt.face[i]) and pt.face_norm2[i] < lower2:             <L 1342>
            var_25 = &((var_pt).face);
            var_27 = wp::load(var_25);
            var_26 = wp::address(var_27, var_23);
            var_29 = wp::load(var_26);
            var_28 = _is_invalid_face_0(var_29);
            var_30 = wp::unot(var_28);
            var_24 = var_30;
            if (var_24) {
                var_31 = &((var_pt).face_norm2);
                var_33 = wp::load(var_31);
                var_32 = wp::address(var_33, var_23);
                var_35 = wp::load(var_32);
                var_34 = (var_35 < var_19);
                var_24 = var_24 && var_34;
            }
            if (var_24) {
                // idx = i                                                                    <L 1343>
                var_36 = wp::copy(var_23);
                // lower2 = pt.face_norm2[i]                                                  <L 1344>
                var_37 = &((var_pt).face_norm2);
                var_39 = wp::load(var_37);
                var_38 = wp::address(var_39, var_23);
                var_41 = wp::load(var_38);
                var_40 = wp::copy(var_41);
            }
            var_42 = wp::where(var_24, var_36, var_18);
            var_43 = wp::where(var_24, var_40, var_19);
            wp::assign(var_18, var_42);
            wp::assign(var_19, var_43);
            goto start_for_2;
        end_for_2:;
        // if lower2 > upper2 or idx < 0:                                                     <L 1347>
        var_45 = (var_19 > var_2);
        var_44 = var_45;
        if (!var_44) {
            var_47 = (var_18 < var_46);
            var_44 = var_44 || var_47;
        }
        if (var_44) {
            // idx = pidx                                                                     <L 1348>
            var_48 = wp::copy(var_16);
            // break                                                                          <L 1349>
            wp::assign(var_4, var_48);
            wp::assign(var_6, var_16);
            goto end_for_0;
        }
        var_49 = wp::where(var_44, var_4, var_18);
        var_50 = wp::where(var_44, var_6, var_16);
        // if lower2 <= 0.0:                                                                  <L 1352>
        var_52 = (var_19 <= var_51);
        if (var_52) {
            // break                                                                          <L 1353>
            wp::assign(var_4, var_49);
            wp::assign(var_6, var_50);
            goto end_for_0;
        }
        var_53 = wp::where(var_52, var_4, var_49);
        var_54 = wp::where(var_52, var_6, var_50);
        // lower = wp.sqrt(lower2)                                                            <L 1356>
        var_55 = wp::sqrt(var_19);
        // wi = pt.nvert                                                                      <L 1357>
        var_56 = &((var_pt).nvert);
        var_58 = wp::load(var_56);
        var_57 = wp::copy(var_58);
        // face_pr = pt.face_pr[idx]                                                          <L 1358>
        var_59 = &((var_pt).face_pr);
        var_61 = wp::load(var_59);
        var_60 = wp::address(var_61, var_53);
        var_63 = wp::load(var_60);
        var_62 = wp::copy(var_63);
        // i1, i2 = _epa_support(pt, wi, geom1, geom2, geomtype1, geomtype2, face_pr / lower)       <L 1359>
        var_64 = wp::div(var_62, var_55);
        _epa_support_0(var_pt, var_57, var_geom1, var_geom2, var_geomtype1, var_geomtype2, var_64, var_65, var_66);
        // w = pt.vert[2 * wi] - pt.vert[2 * wi + 1]                                          <L 1360>
        var_67 = &((var_pt).vert);
        var_69 = wp::mul(var_68, var_57);
        var_71 = wp::load(var_67);
        var_70 = wp::address(var_71, var_69);
        var_72 = &((var_pt).vert);
        var_74 = wp::mul(var_73, var_57);
        var_76 = wp::add(var_74, var_75);
        var_78 = wp::load(var_72);
        var_77 = wp::address(var_78, var_76);
        var_80 = wp::load(var_70);
        var_81 = wp::load(var_77);
        var_79 = wp::sub(var_80, var_81);
        // geom1.index = i1                                                                   <L 1361>
        var_geom1.index = var_65;
        // geom2.index = i2                                                                   <L 1362>
        var_geom2.index = var_66;
        // pt.nvert += 1                                                                      <L 1363>
        var_83 = &((var_pt).nvert);
        var_85 = wp::load(var_83);
        var_84 = wp::add(var_85, var_82);
        var_pt.nvert = var_84;
        // upper_k = wp.dot(face_pr, w) / lower                                               <L 1366>
        var_86 = wp::dot(var_62, var_79);
        var_87 = wp::div(var_86, var_55);
        // if upper_k < upper:                                                                <L 1367>
        var_88 = (var_87 < var_1);
        if (var_88) {
            // upper = upper_k                                                                <L 1368>
            var_89 = wp::copy(var_87);
            // upper2 = upper * upper                                                         <L 1369>
            var_90 = wp::mul(var_89, var_89);
        }
        var_91 = wp::where(var_88, var_89, var_1);
        var_92 = wp::where(var_88, var_90, var_2);
        // if upper - lower < epsilon:                                                        <L 1371>
        var_93 = wp::sub(var_91, var_55);
        var_94 = (var_93 < var_8);
        if (var_94) {
            // break                                                                          <L 1372>
            wp::assign(var_1, var_91);
            wp::assign(var_2, var_92);
            wp::assign(var_4, var_53);
            wp::assign(var_6, var_54);
            goto end_for_0;
        }
        var_95 = wp::where(var_94, var_1, var_91);
        var_96 = wp::where(var_94, var_2, var_92);
        var_97 = wp::where(var_94, var_4, var_53);
        var_98 = wp::where(var_94, var_6, var_54);
        // if is_discrete:                                                                    <L 1375>
        if (var_is_discrete) {
            // found_repeated = bool(False)                                                   <L 1376>
            var_100 = bool(var_99);
            // for i in range(pt.nvert - 1):                                                  <L 1377>
            var_101 = &((var_pt).nvert);
            var_104 = wp::load(var_101);
            var_103 = wp::sub(var_104, var_102);
            var_105 = wp::range(var_103);
            start_for_4:;
                if (iter_cmp(var_105) == 0) goto end_for_4;
                var_106 = wp::iter_next(var_105);
                // if pt.vert_index[2 * i] == pt.vert_index[2 * wi] and pt.vert_index[2 * i + 1] == pt.vert_index[2 * wi + 1]:       <L 1378>
                var_108 = &((var_pt).vert_index);
                var_110 = wp::mul(var_109, var_106);
                var_112 = wp::load(var_108);
                var_111 = wp::address(var_112, var_110);
                var_113 = &((var_pt).vert_index);
                var_115 = wp::mul(var_114, var_57);
                var_117 = wp::load(var_113);
                var_116 = wp::address(var_117, var_115);
                var_119 = wp::load(var_111);
                var_120 = wp::load(var_116);
                var_118 = (var_119 == var_120);
                var_107 = var_118;
                if (var_107) {
                    var_121 = &((var_pt).vert_index);
                    var_123 = wp::mul(var_122, var_106);
                    var_125 = wp::add(var_123, var_124);
                    var_127 = wp::load(var_121);
                    var_126 = wp::address(var_127, var_125);
                    var_128 = &((var_pt).vert_index);
                    var_130 = wp::mul(var_129, var_57);
                    var_132 = wp::add(var_130, var_131);
                    var_134 = wp::load(var_128);
                    var_133 = wp::address(var_134, var_132);
                    var_136 = wp::load(var_126);
                    var_137 = wp::load(var_133);
                    var_135 = (var_136 == var_137);
                    var_107 = var_107 && var_135;
                }
                if (var_107) {
                    // found_repeated = True                                                  <L 1379>
                    // break                                                                  <L 1380>
                    wp::assign(var_100, var_138);
                    goto end_for_4;
                }
                goto start_for_4;
            end_for_4:;
            // if found_repeated:                                                             <L 1381>
            if (var_100) {
                // break                                                                      <L 1382>
                wp::assign(var_1, var_95);
                wp::assign(var_2, var_96);
                wp::assign(var_4, var_97);
                wp::assign(var_6, var_98);
                goto end_for_0;
            }
            var_139 = wp::where(var_100, var_1, var_95);
            var_140 = wp::where(var_100, var_2, var_96);
            var_141 = wp::where(var_100, var_4, var_97);
            var_142 = wp::where(var_100, var_6, var_98);
        }
        var_143 = wp::where(var_is_discrete, var_139, var_95);
        var_144 = wp::where(var_is_discrete, var_140, var_96);
        var_145 = wp::where(var_is_discrete, var_141, var_97);
        var_146 = wp::where(var_is_discrete, var_142, var_98);
        var_147 = wp::where(var_is_discrete, var_106, var_23);
        // nvalid -= 1                                                                        <L 1384>
        var_149 = wp::sub(var_10, var_148);
        // pt.face[idx] = _delete_face(pt.face[idx])                                          <L 1385>
        var_150 = &((var_pt).face);
        var_152 = wp::load(var_150);
        var_151 = wp::address(var_152, var_145);
        var_154 = wp::load(var_151);
        var_153 = _delete_face_0(var_154);
        var_155 = &((var_pt).face);
        var_156 = wp::load(var_155);
        wp::array_store(var_156, var_145, var_153);
        // face = _get_face_verts(pt.face[idx])                                               <L 1386>
        var_157 = &((var_pt).face);
        var_159 = wp::load(var_157);
        var_158 = wp::address(var_159, var_145);
        var_161 = wp::load(var_158);
        var_160 = _get_face_verts_0(var_161);
        // pt.nhorizon = _add_edge(pt, face[0], face[1])                                      <L 1387>
        var_163 = wp::extract(var_160, var_162);
        var_165 = wp::extract(var_160, var_164);
        var_166 = _add_edge_0(var_pt, var_163, var_165);
        var_pt.nhorizon = var_166;
        // pt.nhorizon = _add_edge(pt, face[1], face[2])                                      <L 1388>
        var_168 = wp::extract(var_160, var_167);
        var_170 = wp::extract(var_160, var_169);
        var_171 = _add_edge_0(var_pt, var_168, var_170);
        var_pt.nhorizon = var_171;
        // pt.nhorizon = _add_edge(pt, face[2], face[0])                                      <L 1389>
        var_173 = wp::extract(var_160, var_172);
        var_175 = wp::extract(var_160, var_174);
        var_176 = _add_edge_0(var_pt, var_173, var_175);
        var_pt.nhorizon = var_176;
        // if pt.nhorizon == -1:                                                              <L 1390>
        var_177 = &((var_pt).nhorizon);
        var_180 = wp::load(var_177);
        var_179 = (var_180 == var_178);
        if (var_179) {
            // if warn_overflow:                                                              <L 1391>
            if (var_warn_overflow) {
                // wp.printf("Warning: EPA horizon = %d isn't large enough.\n", pt.horizon.shape[0])       <L 1392>
                var_182 = &((var_pt).horizon);
                var_183 = &(var_182->shape);
                var_186 = wp::load(var_183);
                var_185 = wp::extract(var_186, var_184);
                printf(var_181, var_185);
            }
            // wp.atomic_or(overflow_out, worldid, OverflowType.EPA_HORIZON)                  <L 1393>
            var_189 = wp::atomic_or(var_overflow_out, var_worldid, var_188);
            // idx = -1                                                                       <L 1394>
            // break                                                                          <L 1395>
            wp::assign(var_1, var_143);
            wp::assign(var_2, var_144);
            wp::assign(var_4, var_190);
            wp::assign(var_6, var_146);
            wp::assign(var_10, var_149);
            goto end_for_0;
        }
        var_191 = wp::where(var_179, var_1, var_143);
        var_192 = wp::where(var_179, var_2, var_144);
        var_193 = wp::where(var_179, var_4, var_145);
        var_194 = wp::where(var_179, var_6, var_146);
        var_195 = wp::where(var_179, var_10, var_149);
        // for i in range(pt.nface):                                                          <L 1398>
        var_196 = &((var_pt).nface);
        var_198 = wp::load(var_196);
        var_197 = wp::range(var_198);
        start_for_6:;
            if (iter_cmp(var_197) == 0) goto end_for_6;
            var_199 = wp::iter_next(var_197);
            // if _is_face_deleted(pt.face[i]):                                               <L 1399>
            var_200 = &((var_pt).face);
            var_202 = wp::load(var_200);
            var_201 = wp::address(var_202, var_199);
            var_204 = wp::load(var_201);
            var_203 = _is_face_deleted_0(var_204);
            if (var_203) {
                // continue                                                                   <L 1400>
                goto start_for_6;
            }
            // if wp.dot(pt.face_pr[i], w) - pt.face_norm2[i] > 1e-10:                        <L 1402>
            var_205 = &((var_pt).face_pr);
            var_207 = wp::load(var_205);
            var_206 = wp::address(var_207, var_199);
            var_209 = wp::load(var_206);
            var_208 = wp::dot(var_209, var_79);
            var_210 = &((var_pt).face_norm2);
            var_212 = wp::load(var_210);
            var_211 = wp::address(var_212, var_199);
            var_214 = wp::load(var_211);
            var_213 = wp::sub(var_208, var_214);
            var_216 = (var_213 > var_215);
            if (var_216) {
                // nvalid = wp.where(_is_invalid_face(pt.face[i]), nvalid, nvalid - 1)        <L 1403>
                var_217 = &((var_pt).face);
                var_219 = wp::load(var_217);
                var_218 = wp::address(var_219, var_199);
                var_221 = wp::load(var_218);
                var_220 = _is_invalid_face_0(var_221);
                var_223 = wp::sub(var_195, var_222);
                var_224 = wp::where(var_220, var_195, var_223);
                // pt.face[i] = _delete_face(pt.face[i])                                      <L 1404>
                var_225 = &((var_pt).face);
                var_227 = wp::load(var_225);
                var_226 = wp::address(var_227, var_199);
                var_229 = wp::load(var_226);
                var_228 = _delete_face_0(var_229);
                var_230 = &((var_pt).face);
                var_231 = wp::load(var_230);
                wp::array_store(var_231, var_199, var_228);
                // face = _get_face_verts(pt.face[i])                                         <L 1405>
                var_232 = &((var_pt).face);
                var_234 = wp::load(var_232);
                var_233 = wp::address(var_234, var_199);
                var_236 = wp::load(var_233);
                var_235 = _get_face_verts_0(var_236);
                // pt.nhorizon = _add_edge(pt, face[0], face[1])                              <L 1406>
                var_238 = wp::extract(var_235, var_237);
                var_240 = wp::extract(var_235, var_239);
                var_241 = _add_edge_0(var_pt, var_238, var_240);
                var_pt.nhorizon = var_241;
                // pt.nhorizon = _add_edge(pt, face[1], face[2])                              <L 1407>
                var_243 = wp::extract(var_235, var_242);
                var_245 = wp::extract(var_235, var_244);
                var_246 = _add_edge_0(var_pt, var_243, var_245);
                var_pt.nhorizon = var_246;
                // pt.nhorizon = _add_edge(pt, face[2], face[0])                              <L 1408>
                var_248 = wp::extract(var_235, var_247);
                var_250 = wp::extract(var_235, var_249);
                var_251 = _add_edge_0(var_pt, var_248, var_250);
                var_pt.nhorizon = var_251;
                // if pt.nhorizon == -1:                                                      <L 1409>
                var_252 = &((var_pt).nhorizon);
                var_255 = wp::load(var_252);
                var_254 = (var_255 == var_253);
                if (var_254) {
                    // if warn_overflow:                                                      <L 1410>
                    if (var_warn_overflow) {
                        // wp.printf("Warning: EPA horizon = %d isn't large enough.\n", pt.horizon.shape[0])       <L 1411>
                        var_257 = &((var_pt).horizon);
                        var_258 = &(var_257->shape);
                        var_261 = wp::load(var_258);
                        var_260 = wp::extract(var_261, var_259);
                        printf(var_256, var_260);
                    }
                    // wp.atomic_or(overflow_out, worldid, OverflowType.EPA_HORIZON)          <L 1412>
                    var_264 = wp::atomic_or(var_overflow_out, var_worldid, var_263);
                    // idx = -1                                                               <L 1413>
                    // break                                                                  <L 1414>
                    wp::assign(var_193, var_265);
                    wp::assign(var_195, var_224);
                    wp::assign(var_160, var_235);
                    goto end_for_6;
                }
                var_266 = wp::where(var_254, var_195, var_224);
                var_267 = wp::where(var_254, var_160, var_235);
            }
            var_268 = wp::where(var_216, var_266, var_195);
            var_269 = wp::where(var_216, var_267, var_160);
            wp::assign(var_195, var_268);
            wp::assign(var_160, var_269);
            goto start_for_6;
        end_for_6:;
        // for i in range(pt.nhorizon):                                                       <L 1417>
        var_270 = &((var_pt).nhorizon);
        var_272 = wp::load(var_270);
        var_271 = wp::range(var_272);
        start_for_8:;
            if (iter_cmp(var_271) == 0) goto end_for_8;
            var_273 = wp::iter_next(var_271);
            // edge = _get_edge(pt.horizon[i])                                                <L 1418>
            var_274 = &((var_pt).horizon);
            var_276 = wp::load(var_274);
            var_275 = wp::address(var_276, var_273);
            var_278 = wp::load(var_275);
            var_277 = _get_edge_0(var_278);
            // dist2 = _attach_face(pt, pt.nface, wi, edge[0], edge[1])                       <L 1419>
            var_279 = &((var_pt).nface);
            var_281 = wp::extract(var_277, var_280);
            var_283 = wp::extract(var_277, var_282);
            var_285 = wp::load(var_279);
            var_284 = _attach_face_0(var_pt, var_285, var_57, var_281, var_283);
            // if dist2 == 0.0:                                                               <L 1420>
            var_287 = (var_284 == var_286);
            if (var_287) {
                // idx = -1                                                                   <L 1421>
                // break                                                                      <L 1422>
                wp::assign(var_193, var_288);
                goto end_for_8;
            }
            // pt.nface += 1                                                                  <L 1424>
            var_290 = &((var_pt).nface);
            var_292 = wp::load(var_290);
            var_291 = wp::add(var_292, var_289);
            var_pt.nface = var_291;
            // if dist2 >= lower2 and dist2 <= upper2:                                        <L 1426>
            var_294 = (var_284 >= var_19);
            var_293 = var_294;
            if (var_293) {
                var_295 = (var_284 <= var_192);
                var_293 = var_293 && var_295;
            }
            if (var_293) {
                // nvalid += 1                                                                <L 1427>
                var_297 = wp::add(var_195, var_296);
            }
            var_298 = wp::where(var_293, var_297, var_195);
            if (!var_293) {
                // pt.face[pt.nface - 1] = _invalidate_face(pt.face[pt.nface - 1])            <L 1429>
                var_299 = &((var_pt).face);
                var_300 = &((var_pt).nface);
                var_303 = wp::load(var_300);
                var_302 = wp::sub(var_303, var_301);
                var_305 = wp::load(var_299);
                var_304 = wp::address(var_305, var_302);
                var_307 = wp::load(var_304);
                var_306 = _invalidate_face_0(var_307);
                var_308 = &((var_pt).face);
                var_309 = &((var_pt).nface);
                var_312 = wp::load(var_309);
                var_311 = wp::sub(var_312, var_310);
                var_313 = wp::load(var_308);
                wp::array_store(var_313, var_311, var_306);
            }
            wp::assign(var_195, var_298);
            goto start_for_8;
        end_for_8:;
        // if nvalid == 0 or idx == -1:                                                       <L 1432>
        var_316 = (var_195 == var_315);
        var_314 = var_316;
        if (!var_314) {
            var_318 = (var_193 == var_317);
            var_314 = var_314 || var_318;
        }
        if (var_314) {
            // break                                                                          <L 1433>
            wp::assign(var_1, var_191);
            wp::assign(var_2, var_192);
            wp::assign(var_4, var_193);
            wp::assign(var_6, var_194);
            wp::assign(var_10, var_195);
            goto end_for_0;
        }
        var_319 = wp::where(var_314, var_1, var_191);
        var_320 = wp::where(var_314, var_2, var_192);
        var_321 = wp::where(var_314, var_4, var_193);
        var_322 = wp::where(var_314, var_6, var_194);
        var_323 = wp::where(var_314, var_10, var_195);
        // pt.nhorizon = 0                                                                    <L 1436>
        var_pt.nhorizon = var_324;
        wp::assign(var_1, var_319);
        wp::assign(var_2, var_320);
        wp::assign(var_4, var_321);
        wp::assign(var_6, var_322);
        wp::assign(var_10, var_323);
        goto start_for_0;
    end_for_0:;
    // if idx > -1:                                                                           <L 1439>
    var_326 = (var_4 > var_325);
    if (var_326) {
        // x1, x2, dist = _epa_witness(pt, geom1, geom2, geomtype1, geomtype2, idx)           <L 1440>
        _epa_witness_0(var_pt, var_geom1, var_geom2, var_geomtype1, var_geomtype2, var_4, var_327, var_328, var_329);
        // return dist, x1, x2, idx                                                           <L 1441>
        ret_0 = var_329;
        ret_1 = var_327;
        ret_2 = var_328;
        ret_3 = var_4;
        return;
    }
    // return 0.0, wp.vec3(), wp.vec3(), -1                                                   <L 1442>
    var_331 = wp::vec_t<3, wp::float32>();
    var_332 = wp::vec_t<3, wp::float32>();
    ret_0 = var_330;
    ret_1 = var_331;
    ret_2 = var_332;
    ret_3 = var_333;
    return;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:2405
static CUDA_CALLABLE void epa_phase_0(
    wp::float32 var_tolerance,
    wp::int32 var_epa_iterations,
    GJKResult_28609055 var_result,
    Geom_3242f8a8 var_geom1,
    Geom_3242f8a8 var_geom2,
    wp::int32 var_geomtype1,
    wp::int32 var_geomtype2,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vert,
    wp::array_t<wp::int32> var_vert_index,
    wp::array_t<wp::int32> var_face,
    wp::array_t<wp::vec_t<3, wp::float32>> var_face_pr,
    wp::array_t<wp::float32> var_face_norm2,
    wp::array_t<wp::int32> var_horizon,
    bool var_warn_overflow,
    wp::int32 var_worldid,
    wp::array_t<wp::int32> var_overflow_out,
    wp::float32 & ret_0,
    wp::int32 & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & ret_3,
    wp::int32 & ret_4)
{
    //---------
    // primal vars
    Polytope_10582b13 var_0;
    const wp::int32 var_1 = 0;
    const wp::int32 var_2 = 0;
    const wp::int32 var_3 = 0;
    wp::int32* var_4;
    const wp::int32 var_5 = 2;
    bool var_6;
    wp::int32 var_7;
    wp::mat_t<4, 3, wp::float32>* var_8;
    wp::mat_t<4, 3, wp::float32>* var_9;
    wp::mat_t<4, 3, wp::float32>* var_10;
    wp::vec_t<4, wp::int32>* var_11;
    wp::vec_t<4, wp::int32>* var_12;
    Polytope_10582b13 var_13;
    GJKResult_28609055 var_14;
    wp::mat_t<4, 3, wp::float32> var_15;
    wp::mat_t<4, 3, wp::float32> var_16;
    wp::mat_t<4, 3, wp::float32> var_17;
    wp::vec_t<4, wp::int32> var_18;
    wp::vec_t<4, wp::int32> var_19;
    wp::int32* var_20;
    const wp::int32 var_21 = -1;
    bool var_22;
    wp::int32 var_23;
    wp::mat_t<4, 3, wp::float32>* var_24;
    wp::mat_t<4, 3, wp::float32> var_25;
    wp::mat_t<4, 3, wp::float32> var_26;
    wp::mat_t<4, 3, wp::float32>* var_27;
    wp::mat_t<4, 3, wp::float32> var_28;
    wp::mat_t<4, 3, wp::float32> var_29;
    wp::mat_t<4, 3, wp::float32>* var_30;
    wp::mat_t<4, 3, wp::float32> var_31;
    wp::mat_t<4, 3, wp::float32> var_32;
    wp::vec_t<4, wp::int32>* var_33;
    wp::vec_t<4, wp::int32> var_34;
    wp::vec_t<4, wp::int32> var_35;
    wp::vec_t<4, wp::int32>* var_36;
    wp::vec_t<4, wp::int32> var_37;
    wp::vec_t<4, wp::int32> var_38;
    const wp::int32 var_39 = 3;
    Polytope_10582b13 var_40;
    wp::int32* var_41;
    const wp::int32 var_42 = 4;
    bool var_43;
    wp::int32 var_44;
    wp::mat_t<4, 3, wp::float32>* var_45;
    wp::mat_t<4, 3, wp::float32>* var_46;
    wp::mat_t<4, 3, wp::float32>* var_47;
    wp::vec_t<4, wp::int32>* var_48;
    wp::vec_t<4, wp::int32>* var_49;
    Polytope_10582b13 var_50;
    GJKResult_28609055 var_51;
    wp::mat_t<4, 3, wp::float32> var_52;
    wp::mat_t<4, 3, wp::float32> var_53;
    wp::mat_t<4, 3, wp::float32> var_54;
    wp::vec_t<4, wp::int32> var_55;
    wp::vec_t<4, wp::int32> var_56;
    wp::int32* var_57;
    const wp::int32 var_58 = -1;
    bool var_59;
    wp::int32 var_60;
    wp::mat_t<4, 3, wp::float32>* var_61;
    wp::mat_t<4, 3, wp::float32> var_62;
    wp::mat_t<4, 3, wp::float32> var_63;
    wp::mat_t<4, 3, wp::float32>* var_64;
    wp::mat_t<4, 3, wp::float32> var_65;
    wp::mat_t<4, 3, wp::float32> var_66;
    wp::mat_t<4, 3, wp::float32>* var_67;
    wp::mat_t<4, 3, wp::float32> var_68;
    wp::mat_t<4, 3, wp::float32> var_69;
    wp::vec_t<4, wp::int32>* var_70;
    wp::vec_t<4, wp::int32> var_71;
    wp::vec_t<4, wp::int32> var_72;
    wp::vec_t<4, wp::int32>* var_73;
    wp::vec_t<4, wp::int32> var_74;
    wp::vec_t<4, wp::int32> var_75;
    const wp::int32 var_76 = 3;
    Polytope_10582b13 var_77;
    GJKResult_28609055 var_78;
    Polytope_10582b13 var_79;
    GJKResult_28609055 var_80;
    wp::int32* var_81;
    const wp::int32 var_82 = 3;
    bool var_83;
    wp::int32 var_84;
    wp::float32* var_85;
    wp::mat_t<4, 3, wp::float32>* var_86;
    wp::mat_t<4, 3, wp::float32>* var_87;
    wp::mat_t<4, 3, wp::float32>* var_88;
    wp::vec_t<4, wp::int32>* var_89;
    wp::vec_t<4, wp::int32>* var_90;
    Polytope_10582b13 var_91;
    wp::float32 var_92;
    wp::mat_t<4, 3, wp::float32> var_93;
    wp::mat_t<4, 3, wp::float32> var_94;
    wp::mat_t<4, 3, wp::float32> var_95;
    wp::vec_t<4, wp::int32> var_96;
    wp::vec_t<4, wp::int32> var_97;
    Polytope_10582b13 var_98;
    wp::int32* var_99;
    wp::int32 var_100;
    wp::float32* var_101;
    const wp::int32 var_102 = 1;
    wp::vec_t<3, wp::float32>* var_103;
    wp::vec_t<3, wp::float32>* var_104;
    const wp::int32 var_105 = -1;
    wp::float32 var_106;
    wp::float32 var_107;
    wp::vec_t<3, wp::float32> var_108;
    wp::vec_t<3, wp::float32> var_109;
    wp::vec_t<3, wp::float32> var_110;
    wp::vec_t<3, wp::float32> var_111;
    wp::int32 var_112;
    bool var_113;
    bool var_114;
    bool var_115;
    wp::float32* var_116;
    const wp::float32 var_117 = 0.0;
    bool var_118;
    wp::float32 var_119;
    wp::float32* var_120;
    const wp::float32 var_121 = 0.0;
    bool var_122;
    wp::float32 var_123;
    wp::float32 var_124;
    wp::vec_t<3, wp::float32> var_125;
    wp::vec_t<3, wp::float32> var_126;
    wp::int32 var_127;
    const wp::int32 var_128 = -1;
    bool var_129;
    const wp::float32 var_130 = 1e+30;
    const wp::int32 var_131 = 0;
    wp::vec_t<3, wp::float32> var_132;
    wp::vec_t<3, wp::float32> var_133;
    const wp::int32 var_134 = -1;
    bool var_135;
    wp::float32* var_136;
    const wp::float32 var_137 = 0.0;
    bool var_138;
    wp::float32 var_139;
    wp::float32* var_140;
    const wp::float32 var_141 = 0.0;
    bool var_142;
    wp::float32 var_143;
    const wp::int32 var_144 = -1;
    wp::int32 var_145;
    bool var_146;
    bool var_147;
    const wp::int32 var_148 = 6;
    bool var_149;
    const wp::int32 var_150 = 7;
    bool var_151;
    bool var_152;
    const wp::int32 var_153 = 6;
    bool var_154;
    const wp::int32 var_155 = 7;
    bool var_156;
    const wp::int32 var_157 = -1;
    wp::int32 var_158;
    const wp::int32 var_159 = 1;
    //---------
    // forward
    // def epa_phase(                                                                         <L 2406>
    // pt = Polytope()                                                                        <L 2427>
    var_0 = Polytope_10582b13();
    // pt.nface = 0                                                                           <L 2428>
    var_0.nface = var_1;
    // pt.nvert = 0                                                                           <L 2429>
    var_0.nvert = var_2;
    // pt.nhorizon = 0                                                                        <L 2430>
    var_0.nhorizon = var_3;
    // pt.vert = vert                                                                         <L 2431>
    var_0.vert = var_vert;
    // pt.vert_index = vert_index                                                             <L 2432>
    var_0.vert_index = var_vert_index;
    // pt.face = face                                                                         <L 2433>
    var_0.face = var_face;
    // pt.face_pr = face_pr                                                                   <L 2434>
    var_0.face_pr = var_face_pr;
    // pt.face_norm2 = face_norm2                                                             <L 2435>
    var_0.face_norm2 = var_face_norm2;
    // pt.horizon = horizon                                                                   <L 2436>
    var_0.horizon = var_horizon;
    // if result.dim == 2:                                                                    <L 2438>
    var_4 = &((var_result).dim);
    var_7 = wp::load(var_4);
    var_6 = (var_7 == var_5);
    if (var_6) {
        // pt, new_result = _polytope2(                                                       <L 2439>
        // pt,                                                                                <L 2440>
        // result.simplex,                                                                    <L 2441>
        var_8 = &((var_result).simplex);
        // result.simplex1,                                                                   <L 2442>
        var_9 = &((var_result).simplex1);
        // result.simplex2,                                                                   <L 2443>
        var_10 = &((var_result).simplex2);
        // result.simplex_index1,                                                             <L 2444>
        var_11 = &((var_result).simplex_index1);
        // result.simplex_index2,                                                             <L 2445>
        var_12 = &((var_result).simplex_index2);
        // geom1,                                                                             <L 2446>
        // geom2,                                                                             <L 2447>
        // geomtype1,                                                                         <L 2448>
        // geomtype2,                                                                         <L 2449>
        var_15 = wp::load(var_8);
        var_16 = wp::load(var_9);
        var_17 = wp::load(var_10);
        var_18 = wp::load(var_11);
        var_19 = wp::load(var_12);
        _polytope2_0(var_0, var_15, var_16, var_17, var_18, var_19, var_geom1, var_geom2, var_geomtype1, var_geomtype2, var_13, var_14);
        // if pt.status == -1:                                                                <L 2451>
        var_20 = &((var_13).status);
        var_23 = wp::load(var_20);
        var_22 = (var_23 == var_21);
        if (var_22) {
            // result.simplex = new_result.simplex                                            <L 2452>
            var_24 = &((var_14).simplex);
            var_26 = wp::load(var_24);
            var_25 = wp::copy(var_26);
            var_result.simplex = var_25;
            // result.simplex1 = new_result.simplex1                                          <L 2453>
            var_27 = &((var_14).simplex1);
            var_29 = wp::load(var_27);
            var_28 = wp::copy(var_29);
            var_result.simplex1 = var_28;
            // result.simplex2 = new_result.simplex2                                          <L 2454>
            var_30 = &((var_14).simplex2);
            var_32 = wp::load(var_30);
            var_31 = wp::copy(var_32);
            var_result.simplex2 = var_31;
            // result.simplex_index1 = new_result.simplex_index1                              <L 2455>
            var_33 = &((var_14).simplex_index1);
            var_35 = wp::load(var_33);
            var_34 = wp::copy(var_35);
            var_result.simplex_index1 = var_34;
            // result.simplex_index2 = new_result.simplex_index2                              <L 2456>
            var_36 = &((var_14).simplex_index2);
            var_38 = wp::load(var_36);
            var_37 = wp::copy(var_38);
            var_result.simplex_index2 = var_37;
            // result.dim = 3                                                                 <L 2457>
            var_result.dim = var_39;
        }
    }
    var_40 = wp::where(var_6, var_13, var_0);
    if (!var_6) {
        // elif result.dim == 4:                                                              <L 2458>
        var_41 = &((var_result).dim);
        var_44 = wp::load(var_41);
        var_43 = (var_44 == var_42);
        if (var_43) {
            // pt, new_result = _polytope4(                                                   <L 2459>
            // pt,                                                                            <L 2460>
            // result.simplex,                                                                <L 2461>
            var_45 = &((var_result).simplex);
            // result.simplex1,                                                               <L 2462>
            var_46 = &((var_result).simplex1);
            // result.simplex2,                                                               <L 2463>
            var_47 = &((var_result).simplex2);
            // result.simplex_index1,                                                         <L 2464>
            var_48 = &((var_result).simplex_index1);
            // result.simplex_index2,                                                         <L 2465>
            var_49 = &((var_result).simplex_index2);
            var_52 = wp::load(var_45);
            var_53 = wp::load(var_46);
            var_54 = wp::load(var_47);
            var_55 = wp::load(var_48);
            var_56 = wp::load(var_49);
            _polytope4_0(var_40, var_52, var_53, var_54, var_55, var_56, var_50, var_51);
            // if pt.status == -1:                                                            <L 2467>
            var_57 = &((var_50).status);
            var_60 = wp::load(var_57);
            var_59 = (var_60 == var_58);
            if (var_59) {
                // result.simplex = new_result.simplex                                        <L 2468>
                var_61 = &((var_51).simplex);
                var_63 = wp::load(var_61);
                var_62 = wp::copy(var_63);
                var_result.simplex = var_62;
                // result.simplex1 = new_result.simplex1                                      <L 2469>
                var_64 = &((var_51).simplex1);
                var_66 = wp::load(var_64);
                var_65 = wp::copy(var_66);
                var_result.simplex1 = var_65;
                // result.simplex2 = new_result.simplex2                                      <L 2470>
                var_67 = &((var_51).simplex2);
                var_69 = wp::load(var_67);
                var_68 = wp::copy(var_69);
                var_result.simplex2 = var_68;
                // result.simplex_index1 = new_result.simplex_index1                          <L 2471>
                var_70 = &((var_51).simplex_index1);
                var_72 = wp::load(var_70);
                var_71 = wp::copy(var_72);
                var_result.simplex_index1 = var_71;
                // result.simplex_index2 = new_result.simplex_index2                          <L 2472>
                var_73 = &((var_51).simplex_index2);
                var_75 = wp::load(var_73);
                var_74 = wp::copy(var_75);
                var_result.simplex_index2 = var_74;
                // result.dim = 3                                                             <L 2473>
                var_result.dim = var_76;
            }
        }
        var_77 = wp::where(var_43, var_50, var_40);
        var_78 = wp::where(var_43, var_51, var_14);
    }
    var_79 = wp::where(var_6, var_40, var_77);
    var_80 = wp::where(var_6, var_14, var_78);
    // if result.dim == 3:                                                                    <L 2476>
    var_81 = &((var_result).dim);
    var_84 = wp::load(var_81);
    var_83 = (var_84 == var_82);
    if (var_83) {
        // pt = _polytope3(                                                                   <L 2477>
        // pt,                                                                                <L 2478>
        // result.dist,                                                                       <L 2479>
        var_85 = &((var_result).dist);
        // result.simplex,                                                                    <L 2480>
        var_86 = &((var_result).simplex);
        // result.simplex1,                                                                   <L 2481>
        var_87 = &((var_result).simplex1);
        // result.simplex2,                                                                   <L 2482>
        var_88 = &((var_result).simplex2);
        // result.simplex_index1,                                                             <L 2483>
        var_89 = &((var_result).simplex_index1);
        // result.simplex_index2,                                                             <L 2484>
        var_90 = &((var_result).simplex_index2);
        // geom1,                                                                             <L 2485>
        // geom2,                                                                             <L 2486>
        // geomtype1,                                                                         <L 2487>
        // geomtype2,                                                                         <L 2488>
        var_92 = wp::load(var_85);
        var_93 = wp::load(var_86);
        var_94 = wp::load(var_87);
        var_95 = wp::load(var_88);
        var_96 = wp::load(var_89);
        var_97 = wp::load(var_90);
        var_91 = _polytope3_0(var_79, var_92, var_93, var_94, var_95, var_96, var_97, var_geom1, var_geom2, var_geomtype1, var_geomtype2);
    }
    var_98 = wp::where(var_83, var_91, var_79);
    // if pt.status:                                                                          <L 2492>
    var_99 = &((var_98).status);
    var_100 = wp::load(var_99);
    if (var_100) {
        // return result.dist, 1, result.x1, result.x2, -1                                    <L 2493>
        var_101 = &((var_result).dist);
        var_103 = &((var_result).x1);
        var_104 = &((var_result).x2);
        var_107 = wp::load(var_101);
        var_106 = wp::copy(var_107);
        var_109 = wp::load(var_103);
        var_108 = wp::copy(var_109);
        var_111 = wp::load(var_104);
        var_110 = wp::copy(var_111);
        ret_0 = var_106;
        ret_1 = var_102;
        ret_2 = var_108;
        ret_3 = var_110;
        ret_4 = var_105;
        return;
    }
    var_112 = wp::load(var_99);
    // is_discrete = _discrete_geoms(geomtype1, geomtype2) and (geom1.margin == 0.0 and geom2.margin == 0.0)       <L 2495>
    var_114 = _discrete_geoms_0(var_geomtype1, var_geomtype2);
    var_113 = var_114;
    if (var_113) {
        var_116 = &((var_geom1).margin);
        var_119 = wp::load(var_116);
        var_118 = (var_119 == var_117);
        var_115 = var_118;
        if (var_115) {
            var_120 = &((var_geom2).margin);
            var_123 = wp::load(var_120);
            var_122 = (var_123 == var_121);
            var_115 = var_115 && var_122;
        }
        var_113 = var_113 && var_115;
    }
    // dist, x1, x2, idx = _epa(                                                              <L 2496>
    // tolerance, epa_iterations, pt, geom1, geom2, geomtype1, geomtype2, is_discrete, warn_overflow, worldid, overflow_out       <L 2497>
    _epa_0(var_tolerance, var_epa_iterations, var_98, var_geom1, var_geom2, var_geomtype1, var_geomtype2, var_113, var_warn_overflow, var_worldid, var_overflow_out, var_124, var_125, var_126, var_127);
    // if idx == -1:                                                                          <L 2499>
    var_129 = (var_127 == var_128);
    if (var_129) {
        // return FLOAT_MAX, 0, wp.vec3(), wp.vec3(), -1                                      <L 2500>
        var_132 = wp::vec_t<3, wp::float32>();
        var_133 = wp::vec_t<3, wp::float32>();
        ret_0 = var_130;
        ret_1 = var_131;
        ret_2 = var_132;
        ret_3 = var_133;
        ret_4 = var_134;
        return;
    }
    // if geom1.margin != 0.0 or geom2.margin != 0.0:                                         <L 2503>
    var_136 = &((var_geom1).margin);
    var_139 = wp::load(var_136);
    var_138 = (var_139 != var_137);
    var_135 = var_138;
    if (!var_135) {
        var_140 = &((var_geom2).margin);
        var_143 = wp::load(var_140);
        var_142 = (var_143 != var_141);
        var_135 = var_135 || var_142;
    }
    if (var_135) {
        // idx = -1                                                                           <L 2504>
    }
    var_145 = wp::where(var_135, var_144, var_127);
    // if (geomtype1 != GeomType.BOX and geomtype1 != GeomType.MESH) or (geomtype2 != GeomType.BOX and geomtype2 != GeomType.MESH):       <L 2507>
    var_149 = (var_geomtype1 != var_148);
    var_147 = var_149;
    if (var_147) {
        var_151 = (var_geomtype1 != var_150);
        var_147 = var_147 && var_151;
    }
    var_146 = var_147;
    if (!var_146) {
        var_154 = (var_geomtype2 != var_153);
        var_152 = var_154;
        if (var_152) {
            var_156 = (var_geomtype2 != var_155);
            var_152 = var_152 && var_156;
        }
        var_146 = var_146 || var_152;
    }
    if (var_146) {
        // idx = -1                                                                           <L 2508>
    }
    var_158 = wp::where(var_146, var_157, var_145);
    // return dist, 1, x1, x2, idx                                                            <L 2510>
    ret_0 = var_124;
    ret_1 = var_159;
    ret_2 = var_125;
    ret_3 = var_126;
    ret_4 = var_158;
    return;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1491
static CUDA_CALLABLE void _feature_dim_0(
    wp::vec_t<3, wp::int32> var_face,
    wp::array_t<wp::int32> var_vert_index,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vert,
    wp::int32 var_offset,
    wp::int32 & ret_0,
    wp::vec_t<3, wp::int32> & ret_1,
    wp::mat_t<3, 3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 2;
    const wp::int32 var_1 = 0;
    wp::int32 var_2;
    wp::int32 var_3;
    wp::int32 var_4;
    wp::int32* var_5;
    wp::int32 var_6;
    wp::int32 var_7;
    const wp::int32 var_8 = 2;
    const wp::int32 var_9 = 1;
    wp::int32 var_10;
    wp::int32 var_11;
    wp::int32 var_12;
    wp::int32* var_13;
    wp::int32 var_14;
    wp::int32 var_15;
    const wp::int32 var_16 = 2;
    const wp::int32 var_17 = 2;
    wp::int32 var_18;
    wp::int32 var_19;
    wp::int32 var_20;
    wp::int32* var_21;
    wp::int32 var_22;
    wp::int32 var_23;
    wp::vec_t<3, wp::int32> var_24;
    wp::mat_t<3, 3, wp::float32> var_25;
    const wp::int32 var_26 = 2;
    const wp::int32 var_27 = 0;
    wp::int32 var_28;
    wp::int32 var_29;
    wp::int32 var_30;
    wp::vec_t<3, wp::float32>* var_31;
    const wp::int32 var_32 = 0;
    wp::vec_t<3, wp::float32> var_33;
    const wp::int32 var_34 = 2;
    const wp::int32 var_35 = 1;
    wp::int32 var_36;
    wp::int32 var_37;
    wp::int32 var_38;
    wp::vec_t<3, wp::float32>* var_39;
    const wp::int32 var_40 = 1;
    wp::vec_t<3, wp::float32> var_41;
    const wp::int32 var_42 = 2;
    const wp::int32 var_43 = 2;
    wp::int32 var_44;
    wp::int32 var_45;
    wp::int32 var_46;
    wp::vec_t<3, wp::float32>* var_47;
    const wp::int32 var_48 = 2;
    wp::vec_t<3, wp::float32> var_49;
    bool var_50;
    bool var_51;
    bool var_52;
    bool var_53;
    const wp::int32 var_54 = 2;
    const wp::int32 var_55 = 3;
    wp::int32 var_56;
    const wp::int32 var_57 = 1;
    const wp::int32 var_58 = 2;
    const wp::int32 var_59 = 2;
    wp::int32 var_60;
    wp::int32 var_61;
    wp::int32 var_62;
    wp::vec_t<3, wp::float32>* var_63;
    const wp::int32 var_64 = 1;
    wp::vec_t<3, wp::float32> var_65;
    bool var_66;
    const wp::int32 var_67 = 2;
    const wp::int32 var_68 = 1;
    wp::int32 var_69;
    //---------
    // forward
    // def _feature_dim(                                                                      <L 1492>
    // v1i = vert_index[2 * face[0] + offset]                                                 <L 1495>
    var_2 = wp::extract(var_face, var_1);
    var_3 = wp::mul(var_0, var_2);
    var_4 = wp::add(var_3, var_offset);
    var_5 = wp::address(var_vert_index, var_4);
    var_7 = wp::load(var_5);
    var_6 = wp::copy(var_7);
    // v2i = vert_index[2 * face[1] + offset]                                                 <L 1496>
    var_10 = wp::extract(var_face, var_9);
    var_11 = wp::mul(var_8, var_10);
    var_12 = wp::add(var_11, var_offset);
    var_13 = wp::address(var_vert_index, var_12);
    var_15 = wp::load(var_13);
    var_14 = wp::copy(var_15);
    // v3i = vert_index[2 * face[2] + offset]                                                 <L 1497>
    var_18 = wp::extract(var_face, var_17);
    var_19 = wp::mul(var_16, var_18);
    var_20 = wp::add(var_19, var_offset);
    var_21 = wp::address(var_vert_index, var_20);
    var_23 = wp::load(var_21);
    var_22 = wp::copy(var_23);
    // feature_index = wp.vec3i(v1i, v2i, v3i)                                                <L 1499>
    var_24 = wp::vec_t<3, wp::int32>(var_6, var_14, var_22);
    // feature_vert = wp.mat33()                                                              <L 1500>
    var_25 = wp::mat_t<3, 3, wp::float32>();
    // feature_vert[0] = vert[2 * face[0] + offset]                                           <L 1501>
    var_28 = wp::extract(var_face, var_27);
    var_29 = wp::mul(var_26, var_28);
    var_30 = wp::add(var_29, var_offset);
    var_31 = wp::address(var_vert, var_30);
    var_33 = wp::load(var_31);
    wp::assign_inplace(var_25, var_32, var_33);
    // feature_vert[1] = vert[2 * face[1] + offset]                                           <L 1502>
    var_36 = wp::extract(var_face, var_35);
    var_37 = wp::mul(var_34, var_36);
    var_38 = wp::add(var_37, var_offset);
    var_39 = wp::address(var_vert, var_38);
    var_41 = wp::load(var_39);
    wp::assign_inplace(var_25, var_40, var_41);
    // feature_vert[2] = vert[2 * face[2] + offset]                                           <L 1503>
    var_44 = wp::extract(var_face, var_43);
    var_45 = wp::mul(var_42, var_44);
    var_46 = wp::add(var_45, var_offset);
    var_47 = wp::address(var_vert, var_46);
    var_49 = wp::load(var_47);
    wp::assign_inplace(var_25, var_48, var_49);
    // if v1i != v2i:                                                                         <L 1505>
    var_50 = (var_6 != var_14);
    if (var_50) {
        // dim = wp.where(v3i == v1i or v3i == v2i, 2, 3)                                     <L 1506>
        var_52 = (var_22 == var_6);
        var_51 = var_52;
        if (!var_51) {
            var_53 = (var_22 == var_14);
            var_51 = var_51 || var_53;
        }
        var_56 = wp::where(var_51, var_54, var_55);
        // return dim, feature_index, feature_vert                                            <L 1507>
        ret_0 = var_56;
        ret_1 = var_24;
        ret_2 = var_25;
        return;
    }
    // feature_index[1] = v3i                                                                 <L 1509>
    wp::assign_inplace(var_24, var_57, var_22);
    // feature_vert[1] = vert[2 * face[2] + offset]                                           <L 1510>
    var_60 = wp::extract(var_face, var_59);
    var_61 = wp::mul(var_58, var_60);
    var_62 = wp::add(var_61, var_offset);
    var_63 = wp::address(var_vert, var_62);
    var_65 = wp::load(var_63);
    wp::assign_inplace(var_25, var_64, var_65);
    // dim = wp.where(v1i != v3i, 2, 1)                                                       <L 1512>
    var_66 = (var_6 != var_22);
    var_69 = wp::where(var_66, var_67, var_68);
    // return dim, feature_index, feature_vert                                                <L 1513>
    ret_0 = var_69;
    ret_1 = var_24;
    ret_2 = var_25;
    return;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1691
static CUDA_CALLABLE wp::int32 _box_normals2_0(
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_n,
    wp::array_t<wp::vec_t<3, wp::float32>> var_normal_out,
    wp::array_t<wp::int32> var_index_out)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 1.0;
    const wp::float32 var_1 = 0.0;
    const wp::float32 var_2 = 0.0;
    const wp::float32 var_3 = -1.0;
    const wp::float32 var_4 = 0.0;
    const wp::float32 var_5 = 0.0;
    const wp::float32 var_6 = 0.0;
    const wp::float32 var_7 = 1.0;
    const wp::float32 var_8 = 0.0;
    const wp::float32 var_9 = 0.0;
    const wp::float32 var_10 = -1.0;
    const wp::float32 var_11 = 0.0;
    const wp::float32 var_12 = 0.0;
    const wp::float32 var_13 = 0.0;
    const wp::float32 var_14 = 1.0;
    const wp::float32 var_15 = 0.0;
    const wp::float32 var_16 = 0.0;
    const wp::float32 var_17 = -1.0;
    wp::mat_t<6, 3, wp::float32> var_18;
    const wp::int32 var_19 = 0;
    const wp::int32 var_20 = 0;
    wp::float32 var_21;
    const wp::int32 var_22 = 0;
    wp::float32 var_23;
    wp::float32 var_24;
    const wp::int32 var_25 = 1;
    const wp::int32 var_26 = 0;
    wp::float32 var_27;
    const wp::int32 var_28 = 1;
    wp::float32 var_29;
    wp::float32 var_30;
    wp::float32 var_31;
    const wp::int32 var_32 = 2;
    const wp::int32 var_33 = 0;
    wp::float32 var_34;
    const wp::int32 var_35 = 2;
    wp::float32 var_36;
    wp::float32 var_37;
    wp::float32 var_38;
    const wp::int32 var_39 = 0;
    const wp::int32 var_40 = 1;
    wp::float32 var_41;
    const wp::int32 var_42 = 0;
    wp::float32 var_43;
    wp::float32 var_44;
    const wp::int32 var_45 = 1;
    const wp::int32 var_46 = 1;
    wp::float32 var_47;
    const wp::int32 var_48 = 1;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::float32 var_51;
    const wp::int32 var_52 = 2;
    const wp::int32 var_53 = 1;
    wp::float32 var_54;
    const wp::int32 var_55 = 2;
    wp::float32 var_56;
    wp::float32 var_57;
    wp::float32 var_58;
    const wp::int32 var_59 = 0;
    const wp::int32 var_60 = 2;
    wp::float32 var_61;
    const wp::int32 var_62 = 0;
    wp::float32 var_63;
    wp::float32 var_64;
    const wp::int32 var_65 = 1;
    const wp::int32 var_66 = 2;
    wp::float32 var_67;
    const wp::int32 var_68 = 1;
    wp::float32 var_69;
    wp::float32 var_70;
    wp::float32 var_71;
    const wp::int32 var_72 = 2;
    const wp::int32 var_73 = 2;
    wp::float32 var_74;
    const wp::int32 var_75 = 2;
    wp::float32 var_76;
    wp::float32 var_77;
    wp::float32 var_78;
    wp::vec_t<3, wp::float32> var_79;
    wp::vec_t<3, wp::float32> var_80;
    const wp::int32 var_81 = 0;
    wp::vec_t<3, wp::float32> var_82;
    wp::float32 var_83;
    const wp::float32 var_84 = 0.999998720000273;
    bool var_85;
    wp::vec_t<3, wp::float32> var_86;
    wp::vec_t<3, wp::float32> var_87;
    const wp::int32 var_88 = 0;
    const wp::int32 var_89 = 0;
    const wp::int32 var_90 = 1;
    const wp::int32 var_91 = 1;
    wp::vec_t<3, wp::float32> var_92;
    wp::float32 var_93;
    bool var_94;
    wp::vec_t<3, wp::float32> var_95;
    wp::vec_t<3, wp::float32> var_96;
    const wp::int32 var_97 = 0;
    const wp::int32 var_98 = 0;
    const wp::int32 var_99 = 1;
    const wp::int32 var_100 = 2;
    wp::vec_t<3, wp::float32> var_101;
    wp::float32 var_102;
    bool var_103;
    wp::vec_t<3, wp::float32> var_104;
    wp::vec_t<3, wp::float32> var_105;
    const wp::int32 var_106 = 0;
    const wp::int32 var_107 = 0;
    const wp::int32 var_108 = 1;
    const wp::int32 var_109 = 3;
    wp::vec_t<3, wp::float32> var_110;
    wp::float32 var_111;
    bool var_112;
    wp::vec_t<3, wp::float32> var_113;
    wp::vec_t<3, wp::float32> var_114;
    const wp::int32 var_115 = 0;
    const wp::int32 var_116 = 0;
    const wp::int32 var_117 = 1;
    const wp::int32 var_118 = 4;
    wp::vec_t<3, wp::float32> var_119;
    wp::float32 var_120;
    bool var_121;
    wp::vec_t<3, wp::float32> var_122;
    wp::vec_t<3, wp::float32> var_123;
    const wp::int32 var_124 = 0;
    const wp::int32 var_125 = 0;
    const wp::int32 var_126 = 1;
    const wp::int32 var_127 = 5;
    wp::vec_t<3, wp::float32> var_128;
    wp::float32 var_129;
    bool var_130;
    wp::vec_t<3, wp::float32> var_131;
    wp::vec_t<3, wp::float32> var_132;
    const wp::int32 var_133 = 0;
    const wp::int32 var_134 = 0;
    const wp::int32 var_135 = 1;
    const wp::int32 var_136 = 0;
    //---------
    // forward
    // def _box_normals2(                                                                     <L 1692>
    // face_normals = mat63(1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0)       <L 1701>
    var_18 = wp::mat_t<6, 3, wp::float32>({var_0, var_1, var_2, var_3, var_4, var_5, var_6, var_7, var_8, var_9, var_10, var_11, var_12, var_13, var_14, var_15, var_16, var_17});
    // local_n = wp.normalize(                                                                <L 1705>
    // wp.vec3(                                                                               <L 1706>
    // mat[0][0] * n[0] + mat[1][0] * n[1] + mat[2][0] * n[2],                                <L 1707>
    var_21 = wp::extract(var_mat, var_19, var_20);
    var_23 = wp::extract(var_n, var_22);
    var_24 = wp::mul(var_21, var_23);
    var_27 = wp::extract(var_mat, var_25, var_26);
    var_29 = wp::extract(var_n, var_28);
    var_30 = wp::mul(var_27, var_29);
    var_31 = wp::add(var_24, var_30);
    var_34 = wp::extract(var_mat, var_32, var_33);
    var_36 = wp::extract(var_n, var_35);
    var_37 = wp::mul(var_34, var_36);
    var_38 = wp::add(var_31, var_37);
    // mat[0][1] * n[0] + mat[1][1] * n[1] + mat[2][1] * n[2],                                <L 1708>
    var_41 = wp::extract(var_mat, var_39, var_40);
    var_43 = wp::extract(var_n, var_42);
    var_44 = wp::mul(var_41, var_43);
    var_47 = wp::extract(var_mat, var_45, var_46);
    var_49 = wp::extract(var_n, var_48);
    var_50 = wp::mul(var_47, var_49);
    var_51 = wp::add(var_44, var_50);
    var_54 = wp::extract(var_mat, var_52, var_53);
    var_56 = wp::extract(var_n, var_55);
    var_57 = wp::mul(var_54, var_56);
    var_58 = wp::add(var_51, var_57);
    // mat[0][2] * n[0] + mat[1][2] * n[1] + mat[2][2] * n[2],                                <L 1709>
    var_61 = wp::extract(var_mat, var_59, var_60);
    var_63 = wp::extract(var_n, var_62);
    var_64 = wp::mul(var_61, var_63);
    var_67 = wp::extract(var_mat, var_65, var_66);
    var_69 = wp::extract(var_n, var_68);
    var_70 = wp::mul(var_67, var_69);
    var_71 = wp::add(var_64, var_70);
    var_74 = wp::extract(var_mat, var_72, var_73);
    var_76 = wp::extract(var_n, var_75);
    var_77 = wp::mul(var_74, var_76);
    var_78 = wp::add(var_71, var_77);
    var_79 = wp::vec_t<3, wp::float32>(var_38, var_58, var_78);
    var_80 = wp::normalize(var_79);
    // for i in range(6):                                                                     <L 1714>
    // if wp.dot(local_n, face_normals[i]) > FACE_TOL:                                        <L 1715>
    var_82 = wp::extract(var_18, var_81);
    var_83 = wp::dot(var_80, var_82);
    var_85 = (var_83 > var_84);
    if (var_85) {
        // normal_out[0] = mat @ face_normals[i]                                              <L 1716>
        var_86 = wp::extract(var_18, var_81);
        var_87 = wp::mul(var_mat, var_86);
        wp::array_store(var_normal_out, var_88, var_87);
        // index_out[0] = i                                                                   <L 1717>
        wp::array_store(var_index_out, var_89, var_81);
        // return 1                                                                           <L 1718>
        return var_90;
    }
    // if wp.dot(local_n, face_normals[i]) > FACE_TOL:                                        <L 1715>
    var_92 = wp::extract(var_18, var_91);
    var_93 = wp::dot(var_80, var_92);
    var_94 = (var_93 > var_84);
    if (var_94) {
        // normal_out[0] = mat @ face_normals[i]                                              <L 1716>
        var_95 = wp::extract(var_18, var_91);
        var_96 = wp::mul(var_mat, var_95);
        wp::array_store(var_normal_out, var_97, var_96);
        // index_out[0] = i                                                                   <L 1717>
        wp::array_store(var_index_out, var_98, var_91);
        // return 1                                                                           <L 1718>
        return var_99;
    }
    // if wp.dot(local_n, face_normals[i]) > FACE_TOL:                                        <L 1715>
    var_101 = wp::extract(var_18, var_100);
    var_102 = wp::dot(var_80, var_101);
    var_103 = (var_102 > var_84);
    if (var_103) {
        // normal_out[0] = mat @ face_normals[i]                                              <L 1716>
        var_104 = wp::extract(var_18, var_100);
        var_105 = wp::mul(var_mat, var_104);
        wp::array_store(var_normal_out, var_106, var_105);
        // index_out[0] = i                                                                   <L 1717>
        wp::array_store(var_index_out, var_107, var_100);
        // return 1                                                                           <L 1718>
        return var_108;
    }
    // if wp.dot(local_n, face_normals[i]) > FACE_TOL:                                        <L 1715>
    var_110 = wp::extract(var_18, var_109);
    var_111 = wp::dot(var_80, var_110);
    var_112 = (var_111 > var_84);
    if (var_112) {
        // normal_out[0] = mat @ face_normals[i]                                              <L 1716>
        var_113 = wp::extract(var_18, var_109);
        var_114 = wp::mul(var_mat, var_113);
        wp::array_store(var_normal_out, var_115, var_114);
        // index_out[0] = i                                                                   <L 1717>
        wp::array_store(var_index_out, var_116, var_109);
        // return 1                                                                           <L 1718>
        return var_117;
    }
    // if wp.dot(local_n, face_normals[i]) > FACE_TOL:                                        <L 1715>
    var_119 = wp::extract(var_18, var_118);
    var_120 = wp::dot(var_80, var_119);
    var_121 = (var_120 > var_84);
    if (var_121) {
        // normal_out[0] = mat @ face_normals[i]                                              <L 1716>
        var_122 = wp::extract(var_18, var_118);
        var_123 = wp::mul(var_mat, var_122);
        wp::array_store(var_normal_out, var_124, var_123);
        // index_out[0] = i                                                                   <L 1717>
        wp::array_store(var_index_out, var_125, var_118);
        // return 1                                                                           <L 1718>
        return var_126;
    }
    // if wp.dot(local_n, face_normals[i]) > FACE_TOL:                                        <L 1715>
    var_128 = wp::extract(var_18, var_127);
    var_129 = wp::dot(var_80, var_128);
    var_130 = (var_129 > var_84);
    if (var_130) {
        // normal_out[0] = mat @ face_normals[i]                                              <L 1716>
        var_131 = wp::extract(var_18, var_127);
        var_132 = wp::mul(var_mat, var_131);
        wp::array_store(var_normal_out, var_133, var_132);
        // index_out[0] = i                                                                   <L 1717>
        wp::array_store(var_index_out, var_134, var_127);
        // return 1                                                                           <L 1718>
        return var_135;
    }
    // return 0                                                                               <L 1720>
    return var_136;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1724
static CUDA_CALLABLE wp::int32 _box_normals_0(
    wp::int32 var_feature_dim,
    wp::vec_t<3, wp::int32> var_feature_index,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_dir,
    wp::array_t<wp::vec_t<3, wp::float32>> var_normal_out,
    wp::array_t<wp::int32> var_index_out)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::int32 var_1;
    const wp::int32 var_2 = 1;
    wp::int32 var_3;
    const wp::int32 var_4 = 2;
    wp::int32 var_5;
    const wp::int32 var_6 = 3;
    bool var_7;
    const wp::int32 var_8 = 0;
    bool var_9;
    const wp::int32 var_10 = 1;
    wp::int32 var_11;
    const wp::int32 var_12 = 1;
    wp::int32 var_13;
    const wp::int32 var_14 = 1;
    wp::int32 var_15;
    wp::float32 var_16;
    bool var_17;
    const wp::int32 var_18 = 1;
    wp::int32 var_19;
    bool var_20;
    const wp::int32 var_21 = 1;
    wp::int32 var_22;
    bool var_23;
    const wp::int32 var_24 = 1;
    wp::int32 var_25;
    bool var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    bool var_29;
    const wp::int32 var_30 = 2;
    wp::int32 var_31;
    const wp::int32 var_32 = 2;
    wp::int32 var_33;
    const wp::int32 var_34 = 2;
    wp::int32 var_35;
    wp::float32 var_36;
    bool var_37;
    const wp::int32 var_38 = 2;
    wp::int32 var_39;
    bool var_40;
    const wp::int32 var_41 = 2;
    wp::int32 var_42;
    bool var_43;
    const wp::int32 var_44 = 2;
    wp::int32 var_45;
    bool var_46;
    wp::float32 var_47;
    wp::float32 var_48;
    bool var_49;
    const wp::int32 var_50 = 4;
    wp::int32 var_51;
    const wp::int32 var_52 = 4;
    wp::int32 var_53;
    const wp::int32 var_54 = 4;
    wp::int32 var_55;
    wp::float32 var_56;
    bool var_57;
    const wp::int32 var_58 = 4;
    wp::int32 var_59;
    bool var_60;
    const wp::int32 var_61 = 4;
    wp::int32 var_62;
    bool var_63;
    const wp::int32 var_64 = 4;
    wp::int32 var_65;
    bool var_66;
    wp::float32 var_67;
    wp::float32 var_68;
    wp::vec_t<3, wp::float32> var_69;
    wp::vec_t<3, wp::float32> var_70;
    const wp::int32 var_71 = 0;
    wp::float32 var_72;
    wp::float32 var_73;
    const wp::float32 var_74 = 0.0;
    bool var_75;
    const wp::int32 var_76 = 0;
    const wp::int32 var_77 = 1;
    wp::int32 var_78;
    wp::int32 var_79;
    const wp::float32 var_80 = 0.0;
    bool var_81;
    const wp::int32 var_82 = 2;
    const wp::int32 var_83 = 1;
    wp::int32 var_84;
    wp::int32 var_85;
    const wp::float32 var_86 = 0.0;
    bool var_87;
    const wp::int32 var_88 = 4;
    const wp::int32 var_89 = 1;
    wp::int32 var_90;
    wp::int32 var_91;
    const wp::float32 var_92 = -1.0;
    bool var_93;
    const wp::int32 var_94 = 0;
    wp::int32* var_95;
    const wp::int32 var_96 = 1;
    wp::int32 var_97;
    wp::int32 var_98;
    const wp::int32 var_99 = 0;
    const wp::int32 var_100 = 1;
    bool var_101;
    const wp::int32 var_102 = 1;
    wp::int32 var_103;
    const wp::int32 var_104 = 2;
    bool var_105;
    const wp::int32 var_106 = 0;
    bool var_107;
    const wp::int32 var_108 = 1;
    wp::int32 var_109;
    const wp::int32 var_110 = 1;
    wp::int32 var_111;
    wp::float32 var_112;
    bool var_113;
    const wp::int32 var_114 = 1;
    wp::int32 var_115;
    bool var_116;
    const wp::int32 var_117 = 1;
    wp::int32 var_118;
    bool var_119;
    wp::float32 var_120;
    wp::float32 var_121;
    bool var_122;
    const wp::int32 var_123 = 2;
    wp::int32 var_124;
    const wp::int32 var_125 = 2;
    wp::int32 var_126;
    wp::float32 var_127;
    bool var_128;
    const wp::int32 var_129 = 2;
    wp::int32 var_130;
    bool var_131;
    const wp::int32 var_132 = 2;
    wp::int32 var_133;
    bool var_134;
    wp::float32 var_135;
    wp::float32 var_136;
    bool var_137;
    const wp::int32 var_138 = 4;
    wp::int32 var_139;
    const wp::int32 var_140 = 4;
    wp::int32 var_141;
    wp::float32 var_142;
    bool var_143;
    const wp::int32 var_144 = 4;
    wp::int32 var_145;
    bool var_146;
    const wp::int32 var_147 = 4;
    wp::int32 var_148;
    bool var_149;
    wp::float32 var_150;
    wp::float32 var_151;
    const wp::float32 var_152 = 0.0;
    bool var_153;
    const wp::float32 var_154 = 0.0;
    const wp::float32 var_155 = 0.0;
    wp::vec_t<3, wp::float32> var_156;
    wp::vec_t<3, wp::float32> var_157;
    const wp::float32 var_158 = 0.0;
    bool var_159;
    const wp::int32 var_160 = 0;
    const wp::int32 var_161 = 1;
    wp::int32 var_162;
    const wp::int32 var_163 = 1;
    wp::int32 var_164;
    wp::int32 var_165;
    const wp::float32 var_166 = 0.0;
    bool var_167;
    const wp::float32 var_168 = 0.0;
    const wp::float32 var_169 = 0.0;
    wp::vec_t<3, wp::float32> var_170;
    wp::vec_t<3, wp::float32> var_171;
    const wp::float32 var_172 = 0.0;
    bool var_173;
    const wp::int32 var_174 = 2;
    const wp::int32 var_175 = 3;
    wp::int32 var_176;
    const wp::int32 var_177 = 1;
    wp::int32 var_178;
    wp::int32 var_179;
    const wp::float32 var_180 = 0.0;
    bool var_181;
    const wp::float32 var_182 = 0.0;
    const wp::float32 var_183 = 0.0;
    wp::vec_t<3, wp::float32> var_184;
    wp::vec_t<3, wp::float32> var_185;
    const wp::float32 var_186 = 0.0;
    bool var_187;
    const wp::int32 var_188 = 4;
    const wp::int32 var_189 = 5;
    wp::int32 var_190;
    const wp::int32 var_191 = 1;
    wp::int32 var_192;
    wp::int32 var_193;
    bool var_194;
    const wp::int32 var_195 = 1;
    bool var_196;
    const wp::int32 var_197 = 2;
    bool var_198;
    wp::int32 var_199;
    wp::int32 var_200;
    wp::float32 var_201;
    wp::float32 var_202;
    wp::float32 var_203;
    const wp::int32 var_204 = 1;
    bool var_205;
    const wp::int32 var_206 = 1;
    wp::int32 var_207;
    const wp::float32 var_208 = 1.0;
    const wp::float32 var_209 = -1.0;
    wp::float32 var_210;
    const wp::int32 var_211 = 2;
    wp::int32 var_212;
    const wp::float32 var_213 = 1.0;
    const wp::float32 var_214 = -1.0;
    wp::float32 var_215;
    const wp::int32 var_216 = 4;
    wp::int32 var_217;
    const wp::float32 var_218 = 1.0;
    const wp::float32 var_219 = -1.0;
    wp::float32 var_220;
    const wp::float32 var_221 = 0.0;
    const wp::float32 var_222 = 0.0;
    wp::vec_t<3, wp::float32> var_223;
    wp::vec_t<3, wp::float32> var_224;
    const wp::int32 var_225 = 0;
    const wp::float32 var_226 = 0.0;
    const wp::float32 var_227 = 0.0;
    wp::vec_t<3, wp::float32> var_228;
    wp::vec_t<3, wp::float32> var_229;
    const wp::int32 var_230 = 1;
    const wp::float32 var_231 = 0.0;
    const wp::float32 var_232 = 0.0;
    wp::vec_t<3, wp::float32> var_233;
    wp::vec_t<3, wp::float32> var_234;
    const wp::int32 var_235 = 2;
    const wp::float32 var_236 = 0.0;
    bool var_237;
    const wp::int32 var_238 = 0;
    const wp::int32 var_239 = 1;
    wp::int32 var_240;
    const wp::int32 var_241 = 0;
    const wp::float32 var_242 = 0.0;
    bool var_243;
    const wp::int32 var_244 = 2;
    const wp::int32 var_245 = 3;
    wp::int32 var_246;
    const wp::int32 var_247 = 1;
    const wp::float32 var_248 = 0.0;
    bool var_249;
    const wp::int32 var_250 = 4;
    const wp::int32 var_251 = 5;
    wp::int32 var_252;
    const wp::int32 var_253 = 2;
    const wp::int32 var_254 = 3;
    wp::float32 var_255;
    wp::float32 var_256;
    wp::float32 var_257;
    const wp::int32 var_258 = 0;
    //---------
    // forward
    // def _box_normals(                                                                      <L 1725>
    // v1 = feature_index[0]                                                                  <L 1735>
    var_1 = wp::extract(var_feature_index, var_0);
    // v2 = feature_index[1]                                                                  <L 1736>
    var_3 = wp::extract(var_feature_index, var_2);
    // v3 = feature_index[2]                                                                  <L 1737>
    var_5 = wp::extract(var_feature_index, var_4);
    // if feature_dim == 3:                                                                   <L 1739>
    var_7 = (var_feature_dim == var_6);
    if (var_7) {
        // c = 0                                                                              <L 1740>
        // x = float((v1 & 1) and (v2 & 1) and (v3 & 1)) - float(not (v1 & 1) and not (v2 & 1) and not (v3 & 1))       <L 1741>
        var_11 = wp::bit_and(var_1, var_10);
        var_9 = var_11;
        if (var_9) {
            var_13 = wp::bit_and(var_3, var_12);
            var_9 = var_9 && var_13;
        }
        if (var_9) {
            var_15 = wp::bit_and(var_5, var_14);
            var_9 = var_9 && var_15;
        }
        var_16 = wp::float(var_9);
        var_19 = wp::bit_and(var_1, var_18);
        var_20 = wp::unot(var_19);
        var_17 = var_20;
        if (var_17) {
            var_22 = wp::bit_and(var_3, var_21);
            var_23 = wp::unot(var_22);
            var_17 = var_17 && var_23;
        }
        if (var_17) {
            var_25 = wp::bit_and(var_5, var_24);
            var_26 = wp::unot(var_25);
            var_17 = var_17 && var_26;
        }
        var_27 = wp::float(var_17);
        var_28 = wp::sub(var_16, var_27);
        // y = float((v1 & 2) and (v2 & 2) and (v3 & 2)) - float(not (v1 & 2) and not (v2 & 2) and not (v3 & 2))       <L 1742>
        var_31 = wp::bit_and(var_1, var_30);
        var_29 = var_31;
        if (var_29) {
            var_33 = wp::bit_and(var_3, var_32);
            var_29 = var_29 && var_33;
        }
        if (var_29) {
            var_35 = wp::bit_and(var_5, var_34);
            var_29 = var_29 && var_35;
        }
        var_36 = wp::float(var_29);
        var_39 = wp::bit_and(var_1, var_38);
        var_40 = wp::unot(var_39);
        var_37 = var_40;
        if (var_37) {
            var_42 = wp::bit_and(var_3, var_41);
            var_43 = wp::unot(var_42);
            var_37 = var_37 && var_43;
        }
        if (var_37) {
            var_45 = wp::bit_and(var_5, var_44);
            var_46 = wp::unot(var_45);
            var_37 = var_37 && var_46;
        }
        var_47 = wp::float(var_37);
        var_48 = wp::sub(var_36, var_47);
        // z = float((v1 & 4) and (v2 & 4) and (v3 & 4)) - float(not (v1 & 4) and not (v2 & 4) and not (v3 & 4))       <L 1743>
        var_51 = wp::bit_and(var_1, var_50);
        var_49 = var_51;
        if (var_49) {
            var_53 = wp::bit_and(var_3, var_52);
            var_49 = var_49 && var_53;
        }
        if (var_49) {
            var_55 = wp::bit_and(var_5, var_54);
            var_49 = var_49 && var_55;
        }
        var_56 = wp::float(var_49);
        var_59 = wp::bit_and(var_1, var_58);
        var_60 = wp::unot(var_59);
        var_57 = var_60;
        if (var_57) {
            var_62 = wp::bit_and(var_3, var_61);
            var_63 = wp::unot(var_62);
            var_57 = var_57 && var_63;
        }
        if (var_57) {
            var_65 = wp::bit_and(var_5, var_64);
            var_66 = wp::unot(var_65);
            var_57 = var_57 && var_66;
        }
        var_67 = wp::float(var_57);
        var_68 = wp::sub(var_56, var_67);
        // normal_out[0] = mat @ wp.vec3(x, y, z)                                             <L 1744>
        var_69 = wp::vec_t<3, wp::float32>(var_28, var_48, var_68);
        var_70 = wp::mul(var_mat, var_69);
        wp::array_store(var_normal_out, var_71, var_70);
        // sgn = x + y + z                                                                    <L 1745>
        var_72 = wp::add(var_28, var_48);
        var_73 = wp::add(var_72, var_68);
        // if x != 0.0:                                                                       <L 1746>
        var_75 = (var_28 != var_74);
        if (var_75) {
            // index_out[c] = 0                                                               <L 1747>
            wp::array_store(var_index_out, var_8, var_76);
            // c += 1                                                                         <L 1748>
            var_78 = wp::add(var_8, var_77);
        }
        var_79 = wp::where(var_75, var_78, var_8);
        // if y != 0.0:                                                                       <L 1749>
        var_81 = (var_48 != var_80);
        if (var_81) {
            // index_out[c] = 2                                                               <L 1750>
            wp::array_store(var_index_out, var_79, var_82);
            // c += 1                                                                         <L 1751>
            var_84 = wp::add(var_79, var_83);
        }
        var_85 = wp::where(var_81, var_84, var_79);
        // if z != 0.0:                                                                       <L 1752>
        var_87 = (var_68 != var_86);
        if (var_87) {
            // index_out[c] = 4                                                               <L 1753>
            wp::array_store(var_index_out, var_85, var_88);
            // c += 1                                                                         <L 1754>
            var_90 = wp::add(var_85, var_89);
        }
        var_91 = wp::where(var_87, var_90, var_85);
        // if sgn == -1.0:                                                                    <L 1755>
        var_93 = (var_73 == var_92);
        if (var_93) {
            // index_out[0] = index_out[0] + 1                                                <L 1756>
            var_95 = wp::address(var_index_out, var_94);
            var_98 = wp::load(var_95);
            var_97 = wp::add(var_98, var_96);
            wp::array_store(var_index_out, var_99, var_97);
        }
        // if c == 1:                                                                         <L 1757>
        var_101 = (var_91 == var_100);
        if (var_101) {
            // return 1                                                                       <L 1758>
            return var_102;
        }
        // return _box_normals2(mat, dir, normal_out, index_out)                              <L 1759>
        var_103 = _box_normals2_0(var_mat, var_dir, var_normal_out, var_index_out);
        return var_103;
    }
    // if feature_dim == 2:                                                                   <L 1760>
    var_105 = (var_feature_dim == var_104);
    if (var_105) {
        // c = 0                                                                              <L 1761>
        // x = float((v1 & 1) and (v2 & 1)) - float(not (v1 & 1) and not (v2 & 1))            <L 1762>
        var_109 = wp::bit_and(var_1, var_108);
        var_107 = var_109;
        if (var_107) {
            var_111 = wp::bit_and(var_3, var_110);
            var_107 = var_107 && var_111;
        }
        var_112 = wp::float(var_107);
        var_115 = wp::bit_and(var_1, var_114);
        var_116 = wp::unot(var_115);
        var_113 = var_116;
        if (var_113) {
            var_118 = wp::bit_and(var_3, var_117);
            var_119 = wp::unot(var_118);
            var_113 = var_113 && var_119;
        }
        var_120 = wp::float(var_113);
        var_121 = wp::sub(var_112, var_120);
        // y = float((v1 & 2) and (v2 & 2)) - float(not (v1 & 2) and not (v2 & 2))            <L 1763>
        var_124 = wp::bit_and(var_1, var_123);
        var_122 = var_124;
        if (var_122) {
            var_126 = wp::bit_and(var_3, var_125);
            var_122 = var_122 && var_126;
        }
        var_127 = wp::float(var_122);
        var_130 = wp::bit_and(var_1, var_129);
        var_131 = wp::unot(var_130);
        var_128 = var_131;
        if (var_128) {
            var_133 = wp::bit_and(var_3, var_132);
            var_134 = wp::unot(var_133);
            var_128 = var_128 && var_134;
        }
        var_135 = wp::float(var_128);
        var_136 = wp::sub(var_127, var_135);
        // z = float((v1 & 4) and (v2 & 4)) - float(not (v1 & 4) and not (v2 & 4))            <L 1764>
        var_139 = wp::bit_and(var_1, var_138);
        var_137 = var_139;
        if (var_137) {
            var_141 = wp::bit_and(var_3, var_140);
            var_137 = var_137 && var_141;
        }
        var_142 = wp::float(var_137);
        var_145 = wp::bit_and(var_1, var_144);
        var_146 = wp::unot(var_145);
        var_143 = var_146;
        if (var_143) {
            var_148 = wp::bit_and(var_3, var_147);
            var_149 = wp::unot(var_148);
            var_143 = var_143 && var_149;
        }
        var_150 = wp::float(var_143);
        var_151 = wp::sub(var_142, var_150);
        // if x != 0.0:                                                                       <L 1765>
        var_153 = (var_121 != var_152);
        if (var_153) {
            // normal_out[c] = mat @ wp.vec3(x, 0.0, 0.0)                                     <L 1766>
            var_156 = wp::vec_t<3, wp::float32>(var_121, var_154, var_155);
            var_157 = wp::mul(var_mat, var_156);
            wp::array_store(var_normal_out, var_106, var_157);
            // index_out[c] = wp.where(x > 0.0, 0, 1)                                         <L 1767>
            var_159 = (var_121 > var_158);
            var_162 = wp::where(var_159, var_160, var_161);
            wp::array_store(var_index_out, var_106, var_162);
            // c += 1                                                                         <L 1768>
            var_164 = wp::add(var_106, var_163);
        }
        var_165 = wp::where(var_153, var_164, var_106);
        // if y != 0.0:                                                                       <L 1769>
        var_167 = (var_136 != var_166);
        if (var_167) {
            // normal_out[c] = mat @ wp.vec3(0.0, y, 0.0)                                     <L 1770>
            var_170 = wp::vec_t<3, wp::float32>(var_168, var_136, var_169);
            var_171 = wp::mul(var_mat, var_170);
            wp::array_store(var_normal_out, var_165, var_171);
            // index_out[c] = wp.where(y > 0.0, 2, 3)                                         <L 1771>
            var_173 = (var_136 > var_172);
            var_176 = wp::where(var_173, var_174, var_175);
            wp::array_store(var_index_out, var_165, var_176);
            // c += 1                                                                         <L 1772>
            var_178 = wp::add(var_165, var_177);
        }
        var_179 = wp::where(var_167, var_178, var_165);
        // if z != 0.0:                                                                       <L 1773>
        var_181 = (var_151 != var_180);
        if (var_181) {
            // normal_out[c] = mat @ wp.vec3(0.0, 0.0, z)                                     <L 1774>
            var_184 = wp::vec_t<3, wp::float32>(var_182, var_183, var_151);
            var_185 = wp::mul(var_mat, var_184);
            wp::array_store(var_normal_out, var_179, var_185);
            // index_out[c] = wp.where(z > 0.0, 4, 5)                                         <L 1775>
            var_187 = (var_151 > var_186);
            var_190 = wp::where(var_187, var_188, var_189);
            wp::array_store(var_index_out, var_179, var_190);
            // c += 1                                                                         <L 1776>
            var_192 = wp::add(var_179, var_191);
        }
        var_193 = wp::where(var_181, var_192, var_179);
        // if c == 1 or c == 2:                                                               <L 1779>
        var_196 = (var_193 == var_195);
        var_194 = var_196;
        if (!var_194) {
            var_198 = (var_193 == var_197);
            var_194 = var_194 || var_198;
        }
        if (var_194) {
            // return c                                                                       <L 1780>
            return var_193;
        }
        // return _box_normals2(mat, dir, normal_out, index_out)                              <L 1781>
        var_199 = _box_normals2_0(var_mat, var_dir, var_normal_out, var_index_out);
        return var_199;
    }
    var_200 = wp::where(var_105, var_193, var_91);
    var_201 = wp::where(var_105, var_121, var_28);
    var_202 = wp::where(var_105, var_136, var_48);
    var_203 = wp::where(var_105, var_151, var_68);
    // if feature_dim == 1:                                                                   <L 1783>
    var_205 = (var_feature_dim == var_204);
    if (var_205) {
        // x = wp.where(v1 & 1, 1.0, -1.0)                                                    <L 1784>
        var_207 = wp::bit_and(var_1, var_206);
        var_210 = wp::where(var_207, var_208, var_209);
        // y = wp.where(v1 & 2, 1.0, -1.0)                                                    <L 1785>
        var_212 = wp::bit_and(var_1, var_211);
        var_215 = wp::where(var_212, var_213, var_214);
        // z = wp.where(v1 & 4, 1.0, -1.0)                                                    <L 1786>
        var_217 = wp::bit_and(var_1, var_216);
        var_220 = wp::where(var_217, var_218, var_219);
        // normal_out[0] = mat @ wp.vec3(x, 0.0, 0.0)                                         <L 1787>
        var_223 = wp::vec_t<3, wp::float32>(var_210, var_221, var_222);
        var_224 = wp::mul(var_mat, var_223);
        wp::array_store(var_normal_out, var_225, var_224);
        // normal_out[1] = mat @ wp.vec3(0.0, y, 0.0)                                         <L 1788>
        var_228 = wp::vec_t<3, wp::float32>(var_226, var_215, var_227);
        var_229 = wp::mul(var_mat, var_228);
        wp::array_store(var_normal_out, var_230, var_229);
        // normal_out[2] = mat @ wp.vec3(0.0, 0.0, z)                                         <L 1789>
        var_233 = wp::vec_t<3, wp::float32>(var_231, var_232, var_220);
        var_234 = wp::mul(var_mat, var_233);
        wp::array_store(var_normal_out, var_235, var_234);
        // index_out[0] = wp.where(x > 0.0, 0, 1)                                             <L 1790>
        var_237 = (var_210 > var_236);
        var_240 = wp::where(var_237, var_238, var_239);
        wp::array_store(var_index_out, var_241, var_240);
        // index_out[1] = wp.where(y > 0.0, 2, 3)                                             <L 1791>
        var_243 = (var_215 > var_242);
        var_246 = wp::where(var_243, var_244, var_245);
        wp::array_store(var_index_out, var_247, var_246);
        // index_out[2] = wp.where(z > 0.0, 4, 5)                                             <L 1792>
        var_249 = (var_220 > var_248);
        var_252 = wp::where(var_249, var_250, var_251);
        wp::array_store(var_index_out, var_253, var_252);
        // return 3                                                                           <L 1793>
        return var_254;
    }
    var_255 = wp::where(var_205, var_210, var_201);
    var_256 = wp::where(var_205, var_215, var_202);
    var_257 = wp::where(var_205, var_220, var_203);
    // return 0                                                                               <L 1794>
    return var_258;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1544
static CUDA_CALLABLE void _intersect1_0(
    wp::array_t<wp::int32> var_a1,
    wp::array_t<wp::int32> var_a2,
    wp::int32 var_start1,
    wp::int32 var_start2,
    wp::int32 var_len1,
    wp::int32 var_len2,
    wp::int32 & ret_0,
    wp::vec_t<2, wp::int32> & ret_1)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::int32 var_1;
    wp::vec_t<2, wp::int32> var_2;
    wp::int32 var_3;
    wp::range_t var_4;
    wp::int32 var_5;
    wp::int32 var_6;
    wp::range_t var_7;
    wp::int32 var_8;
    wp::int32* var_9;
    wp::int32* var_10;
    bool var_11;
    wp::int32 var_12;
    wp::int32 var_13;
    wp::int32* var_14;
    wp::int32 var_15;
    const wp::int32 var_16 = 1;
    wp::int32 var_17;
    const wp::int32 var_18 = 2;
    bool var_19;
    const wp::int32 var_20 = 2;
    wp::int32 var_21;
    //---------
    // forward
    // def _intersect1(a1: wp.array[int], a2: wp.array[int], start1: int, start2: int, len1: int, len2: int) -> Tuple[int, wp.vec2i]:       <L 1545>
    // count = int(0)                                                                         <L 1546>
    var_1 = wp::int(var_0);
    // res = wp.vec2i()                                                                       <L 1547>
    var_2 = wp::vec_t<2, wp::int32>();
    // for i in range(start1, start1 + len1):                                                 <L 1548>
    var_3 = wp::add(var_start1, var_len1);
    var_4 = wp::range(var_start1, var_3);
    start_for_0:;
        if (iter_cmp(var_4) == 0) goto end_for_0;
        var_5 = wp::iter_next(var_4);
        // for j in range(start2, start2 + len2):                                             <L 1549>
        var_6 = wp::add(var_start2, var_len2);
        var_7 = wp::range(var_start2, var_6);
        start_for_2:;
            if (iter_cmp(var_7) == 0) goto end_for_2;
            var_8 = wp::iter_next(var_7);
            // if a1[i] == a2[j]:                                                             <L 1550>
            var_9 = wp::address(var_a1, var_5);
            var_10 = wp::address(var_a2, var_8);
            var_12 = wp::load(var_9);
            var_13 = wp::load(var_10);
            var_11 = (var_12 == var_13);
            if (var_11) {
                // res[count] = a1[i]                                                         <L 1551>
                var_14 = wp::address(var_a1, var_5);
                var_15 = wp::load(var_14);
                wp::assign_inplace(var_2, var_1, var_15);
                // count += 1                                                                 <L 1552>
                var_17 = wp::add(var_1, var_16);
                // if count == 2:                                                             <L 1553>
                var_19 = (var_17 == var_18);
                if (var_19) {
                    // return 2, res                                                          <L 1554>
                    ret_0 = var_20;
                    ret_1 = var_2;
                    return;
                }
            }
            var_21 = wp::where(var_11, var_17, var_1);
            wp::assign(var_1, var_21);
            goto start_for_2;
        end_for_2:;
        goto start_for_0;
    end_for_0:;
    // return count, res                                                                      <L 1555>
    ret_0 = var_1;
    ret_1 = var_2;
    return;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1558
static CUDA_CALLABLE void _intersect2_0(
    wp::vec_t<2, wp::int32> var_a1,
    wp::array_t<wp::int32> var_a2,
    wp::int32 var_start2,
    wp::int32 var_len1,
    wp::int32 var_len2,
    wp::int32 & ret_0,
    wp::vec_t<2, wp::int32> & ret_1)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::int32 var_1;
    wp::vec_t<2, wp::int32> var_2;
    wp::range_t var_3;
    wp::int32 var_4;
    wp::int32 var_5;
    wp::range_t var_6;
    wp::int32 var_7;
    wp::int32 var_8;
    wp::int32* var_9;
    bool var_10;
    wp::int32 var_11;
    wp::int32 var_12;
    const wp::int32 var_13 = 1;
    wp::int32 var_14;
    const wp::int32 var_15 = 2;
    bool var_16;
    const wp::int32 var_17 = 2;
    wp::int32 var_18;
    //---------
    // forward
    // def _intersect2(a1: wp.vec2i, a2: wp.array[int], start2: int, len1: int, len2: int) -> Tuple[int, wp.vec2i]:       <L 1559>
    // count = int(0)                                                                         <L 1560>
    var_1 = wp::int(var_0);
    // res = wp.vec2i()                                                                       <L 1561>
    var_2 = wp::vec_t<2, wp::int32>();
    // for i in range(len1):                                                                  <L 1562>
    var_3 = wp::range(var_len1);
    start_for_0:;
        if (iter_cmp(var_3) == 0) goto end_for_0;
        var_4 = wp::iter_next(var_3);
        // for j in range(start2, start2 + len2):                                             <L 1563>
        var_5 = wp::add(var_start2, var_len2);
        var_6 = wp::range(var_start2, var_5);
        start_for_2:;
            if (iter_cmp(var_6) == 0) goto end_for_2;
            var_7 = wp::iter_next(var_6);
            // if a1[i] == a2[j]:                                                             <L 1564>
            var_8 = wp::extract(var_a1, var_4);
            var_9 = wp::address(var_a2, var_7);
            var_11 = wp::load(var_9);
            var_10 = (var_8 == var_11);
            if (var_10) {
                // res[count] = a1[i]                                                         <L 1565>
                var_12 = wp::extract(var_a1, var_4);
                wp::assign_inplace(var_2, var_1, var_12);
                // count += 1                                                                 <L 1566>
                var_14 = wp::add(var_1, var_13);
                // if count == 2:                                                             <L 1567>
                var_16 = (var_14 == var_15);
                if (var_16) {
                    // return 2, res                                                          <L 1568>
                    ret_0 = var_17;
                    ret_1 = var_2;
                    return;
                }
            }
            var_18 = wp::where(var_10, var_14, var_1);
            wp::assign(var_1, var_18);
            goto start_for_2;
        end_for_2:;
        goto start_for_0;
    end_for_0:;
    // return count, res                                                                      <L 1569>
    ret_0 = var_1;
    ret_1 = var_2;
    return;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1573
static CUDA_CALLABLE wp::int32 _mesh_normals_0(
    wp::int32 var_feature_dim,
    wp::vec_t<3, wp::int32> var_feature_index,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::int32 var_vertadr,
    wp::int32 var_polyadr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_polynormal,
    wp::array_t<wp::int32> var_polymapadr,
    wp::array_t<wp::int32> var_polymapnum,
    wp::array_t<wp::int32> var_polymap,
    wp::array_t<wp::vec_t<3, wp::float32>> var_normals_out,
    wp::array_t<wp::int32> var_indices_out)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::int32 var_1;
    const wp::int32 var_2 = 1;
    wp::int32 var_3;
    const wp::int32 var_4 = 2;
    wp::int32 var_5;
    const wp::int32 var_6 = 3;
    bool var_7;
    wp::int32 var_8;
    wp::int32* var_9;
    wp::int32 var_10;
    wp::int32 var_11;
    wp::int32 var_12;
    wp::int32* var_13;
    wp::int32 var_14;
    wp::int32 var_15;
    wp::int32 var_16;
    wp::int32* var_17;
    wp::int32 var_18;
    wp::int32 var_19;
    wp::int32 var_20;
    wp::int32* var_21;
    wp::int32 var_22;
    wp::int32 var_23;
    wp::int32 var_24;
    wp::int32* var_25;
    wp::int32 var_26;
    wp::int32 var_27;
    wp::int32 var_28;
    wp::int32* var_29;
    wp::int32 var_30;
    wp::int32 var_31;
    wp::vec_t<2, wp::int32> var_32;
    wp::int32 var_33;
    wp::vec_t<2, wp::int32> var_34;
    const wp::int32 var_35 = 0;
    bool var_36;
    const wp::int32 var_37 = 0;
    wp::int32 var_38;
    wp::vec_t<2, wp::int32> var_39;
    const wp::int32 var_40 = 0;
    bool var_41;
    const wp::int32 var_42 = 0;
    const wp::int32 var_43 = 0;
    wp::int32 var_44;
    wp::int32 var_45;
    wp::vec_t<3, wp::float32>* var_46;
    wp::vec_t<3, wp::float32> var_47;
    wp::vec_t<3, wp::float32> var_48;
    const wp::int32 var_49 = 0;
    const wp::int32 var_50 = 0;
    wp::int32 var_51;
    const wp::int32 var_52 = 0;
    const wp::int32 var_53 = 1;
    const wp::int32 var_54 = 2;
    bool var_55;
    wp::int32 var_56;
    wp::int32* var_57;
    wp::int32 var_58;
    wp::int32 var_59;
    wp::int32 var_60;
    wp::int32* var_61;
    wp::int32 var_62;
    wp::int32 var_63;
    wp::int32 var_64;
    wp::int32* var_65;
    wp::int32 var_66;
    wp::int32 var_67;
    wp::int32 var_68;
    wp::int32* var_69;
    wp::int32 var_70;
    wp::int32 var_71;
    wp::int32 var_72;
    wp::vec_t<2, wp::int32> var_73;
    const wp::int32 var_74 = 0;
    bool var_75;
    const wp::int32 var_76 = 0;
    wp::range_t var_77;
    wp::int32 var_78;
    wp::int32 var_79;
    wp::int32 var_80;
    wp::vec_t<3, wp::float32>* var_81;
    wp::vec_t<3, wp::float32> var_82;
    wp::vec_t<3, wp::float32> var_83;
    wp::int32 var_84;
    wp::int32 var_85;
    wp::int32 var_86;
    wp::int32 var_87;
    wp::int32 var_88;
    wp::int32 var_89;
    wp::vec_t<2, wp::int32> var_90;
    const wp::int32 var_91 = 1;
    bool var_92;
    wp::int32 var_93;
    wp::int32* var_94;
    wp::int32 var_95;
    wp::int32 var_96;
    wp::int32 var_97;
    wp::int32* var_98;
    wp::int32 var_99;
    wp::int32 var_100;
    wp::range_t var_101;
    wp::int32 var_102;
    wp::int32 var_103;
    wp::int32* var_104;
    wp::int32 var_105;
    wp::int32 var_106;
    wp::int32 var_107;
    wp::vec_t<3, wp::float32>* var_108;
    wp::vec_t<3, wp::float32> var_109;
    wp::vec_t<3, wp::float32> var_110;
    wp::int32 var_111;
    wp::int32 var_112;
    wp::int32 var_113;
    const wp::int32 var_114 = 0;
    //---------
    // forward
    // def _mesh_normals(                                                                     <L 1574>
    // v1 = feature_index[0]                                                                  <L 1589>
    var_1 = wp::extract(var_feature_index, var_0);
    // v2 = feature_index[1]                                                                  <L 1590>
    var_3 = wp::extract(var_feature_index, var_2);
    // v3 = feature_index[2]                                                                  <L 1591>
    var_5 = wp::extract(var_feature_index, var_4);
    // if feature_dim == 3:                                                                   <L 1592>
    var_7 = (var_feature_dim == var_6);
    if (var_7) {
        // v1_adr = polymapadr[vertadr + v1]                                                  <L 1593>
        var_8 = wp::add(var_vertadr, var_1);
        var_9 = wp::address(var_polymapadr, var_8);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // v1_num = polymapnum[vertadr + v1]                                                  <L 1594>
        var_12 = wp::add(var_vertadr, var_1);
        var_13 = wp::address(var_polymapnum, var_12);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // v2_adr = polymapadr[vertadr + v2]                                                  <L 1596>
        var_16 = wp::add(var_vertadr, var_3);
        var_17 = wp::address(var_polymapadr, var_16);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // v2_num = polymapnum[vertadr + v2]                                                  <L 1597>
        var_20 = wp::add(var_vertadr, var_3);
        var_21 = wp::address(var_polymapnum, var_20);
        var_23 = wp::load(var_21);
        var_22 = wp::copy(var_23);
        // v3_adr = polymapadr[vertadr + v3]                                                  <L 1599>
        var_24 = wp::add(var_vertadr, var_5);
        var_25 = wp::address(var_polymapadr, var_24);
        var_27 = wp::load(var_25);
        var_26 = wp::copy(var_27);
        // v3_num = polymapnum[vertadr + v3]                                                  <L 1600>
        var_28 = wp::add(var_vertadr, var_5);
        var_29 = wp::address(var_polymapnum, var_28);
        var_31 = wp::load(var_29);
        var_30 = wp::copy(var_31);
        // faceset = wp.vec2i()                                                               <L 1602>
        var_32 = wp::vec_t<2, wp::int32>();
        // n, edgeset = _intersect1(polymap, polymap, v1_adr, v2_adr, v1_num, v2_num)         <L 1603>
        _intersect1_0(var_polymap, var_polymap, var_10, var_18, var_14, var_22, var_33, var_34);
        // if n == 0:                                                                         <L 1604>
        var_36 = (var_33 == var_35);
        if (var_36) {
            // return 0                                                                       <L 1605>
            return var_37;
        }
        // n, faceset = _intersect2(edgeset, polymap, v3_adr, n, v3_num)                      <L 1606>
        _intersect2_0(var_34, var_polymap, var_26, var_33, var_30, var_38, var_39);
        // if n == 0:                                                                         <L 1607>
        var_41 = (var_38 == var_40);
        if (var_41) {
            // return 0                                                                       <L 1608>
            return var_42;
        }
        // normals_out[0] = mat @ polynormal[polyadr + faceset[0]]                            <L 1611>
        var_44 = wp::extract(var_39, var_43);
        var_45 = wp::add(var_polyadr, var_44);
        var_46 = wp::address(var_polynormal, var_45);
        var_48 = wp::load(var_46);
        var_47 = wp::mul(var_mat, var_48);
        wp::array_store(var_normals_out, var_49, var_47);
        // indices_out[0] = faceset[0]                                                        <L 1612>
        var_51 = wp::extract(var_39, var_50);
        wp::array_store(var_indices_out, var_52, var_51);
        // return 1                                                                           <L 1613>
        return var_53;
    }
    // if feature_dim == 2:                                                                   <L 1615>
    var_55 = (var_feature_dim == var_54);
    if (var_55) {
        // v1_adr = polymapadr[vertadr + v1]                                                  <L 1616>
        var_56 = wp::add(var_vertadr, var_1);
        var_57 = wp::address(var_polymapadr, var_56);
        var_59 = wp::load(var_57);
        var_58 = wp::copy(var_59);
        // v1_num = polymapnum[vertadr + v1]                                                  <L 1617>
        var_60 = wp::add(var_vertadr, var_1);
        var_61 = wp::address(var_polymapnum, var_60);
        var_63 = wp::load(var_61);
        var_62 = wp::copy(var_63);
        // v2_adr = polymapadr[vertadr + v2]                                                  <L 1619>
        var_64 = wp::add(var_vertadr, var_3);
        var_65 = wp::address(var_polymapadr, var_64);
        var_67 = wp::load(var_65);
        var_66 = wp::copy(var_67);
        // v2_num = polymapnum[vertadr + v2]                                                  <L 1620>
        var_68 = wp::add(var_vertadr, var_3);
        var_69 = wp::address(var_polymapnum, var_68);
        var_71 = wp::load(var_69);
        var_70 = wp::copy(var_71);
        // n, edgeset = _intersect1(polymap, polymap, v1_adr, v2_adr, v1_num, v2_num)         <L 1623>
        _intersect1_0(var_polymap, var_polymap, var_58, var_66, var_62, var_70, var_72, var_73);
        // if n == 0:                                                                         <L 1624>
        var_75 = (var_72 == var_74);
        if (var_75) {
            // return 0                                                                       <L 1625>
            return var_76;
        }
        // for i in range(n):                                                                 <L 1626>
        var_77 = wp::range(var_72);
        start_for_4:;
            if (iter_cmp(var_77) == 0) goto end_for_4;
            var_78 = wp::iter_next(var_77);
            // normals_out[i] = mat @ polynormal[polyadr + edgeset[i]]                        <L 1627>
            var_79 = wp::extract(var_73, var_78);
            var_80 = wp::add(var_polyadr, var_79);
            var_81 = wp::address(var_polynormal, var_80);
            var_83 = wp::load(var_81);
            var_82 = wp::mul(var_mat, var_83);
            wp::array_store(var_normals_out, var_78, var_82);
            // indices_out[i] = edgeset[i]                                                    <L 1628>
            var_84 = wp::extract(var_73, var_78);
            wp::array_store(var_indices_out, var_78, var_84);
            goto start_for_4;
        end_for_4:;
        // return n                                                                           <L 1629>
        return var_72;
    }
    var_85 = wp::where(var_55, var_58, var_10);
    var_86 = wp::where(var_55, var_62, var_14);
    var_87 = wp::where(var_55, var_66, var_18);
    var_88 = wp::where(var_55, var_70, var_22);
    var_89 = wp::where(var_55, var_72, var_38);
    var_90 = wp::where(var_55, var_73, var_34);
    // if feature_dim == 1:                                                                   <L 1631>
    var_92 = (var_feature_dim == var_91);
    if (var_92) {
        // v1_adr = polymapadr[vertadr + v1]                                                  <L 1632>
        var_93 = wp::add(var_vertadr, var_1);
        var_94 = wp::address(var_polymapadr, var_93);
        var_96 = wp::load(var_94);
        var_95 = wp::copy(var_96);
        // v1_num = polymapnum[vertadr + v1]                                                  <L 1633>
        var_97 = wp::add(var_vertadr, var_1);
        var_98 = wp::address(var_polymapnum, var_97);
        var_100 = wp::load(var_98);
        var_99 = wp::copy(var_100);
        // for i in range(v1_num):                                                            <L 1634>
        var_101 = wp::range(var_99);
        start_for_7:;
            if (iter_cmp(var_101) == 0) goto end_for_7;
            var_102 = wp::iter_next(var_101);
            // index = polymap[v1_adr + i]                                                    <L 1635>
            var_103 = wp::add(var_95, var_102);
            var_104 = wp::address(var_polymap, var_103);
            var_106 = wp::load(var_104);
            var_105 = wp::copy(var_106);
            // normals_out[i] = mat @ polynormal[polyadr + index]                             <L 1636>
            var_107 = wp::add(var_polyadr, var_105);
            var_108 = wp::address(var_polynormal, var_107);
            var_110 = wp::load(var_108);
            var_109 = wp::mul(var_mat, var_110);
            wp::array_store(var_normals_out, var_102, var_109);
            // indices_out[i] = index                                                         <L 1637>
            wp::array_store(var_indices_out, var_102, var_105);
            goto start_for_7;
        end_for_7:;
        // return v1_num                                                                      <L 1638>
        return var_99;
    }
    var_111 = wp::where(var_92, var_95, var_85);
    var_112 = wp::where(var_92, var_99, var_86);
    var_113 = wp::where(var_92, var_102, var_78);
    // return 0                                                                               <L 1639>
    return var_114;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1517
static CUDA_CALLABLE void _aligned_faces_0(
    wp::array_t<wp::vec_t<3, wp::float32>> var_vert1,
    wp::int32 var_len1,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vert2,
    wp::int32 var_len2,
    wp::int32 & ret_0,
    wp::vec_t<2, wp::int32> & ret_1)
{
    //---------
    // primal vars
    wp::vec_t<2, wp::int32> var_0;
    wp::range_t var_1;
    wp::int32 var_2;
    wp::range_t var_3;
    wp::int32 var_4;
    wp::vec_t<3, wp::float32>* var_5;
    wp::vec_t<3, wp::float32>* var_6;
    wp::float32 var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    const wp::float32 var_10 = 0.999998720000273;
    const wp::float32 var_11 = -0.999998720000273;
    bool var_12;
    const wp::int32 var_13 = 0;
    const wp::int32 var_14 = 1;
    const wp::int32 var_15 = 1;
    const wp::int32 var_16 = 0;
    //---------
    // forward
    // def _aligned_faces(vert1: wp.array[wp.vec3], len1: int, vert2: wp.array[wp.vec3], len2: int) -> Tuple[int, wp.vec2i]:       <L 1518>
    // res = wp.vec2i()                                                                       <L 1519>
    var_0 = wp::vec_t<2, wp::int32>();
    // for i in range(len1):                                                                  <L 1520>
    var_1 = wp::range(var_len1);
    start_for_0:;
        if (iter_cmp(var_1) == 0) goto end_for_0;
        var_2 = wp::iter_next(var_1);
        // for j in range(len2):                                                              <L 1521>
        var_3 = wp::range(var_len2);
        start_for_2:;
            if (iter_cmp(var_3) == 0) goto end_for_2;
            var_4 = wp::iter_next(var_3);
            // if wp.dot(vert1[i], vert2[j]) < -FACE_TOL:                                     <L 1522>
            var_5 = wp::address(var_vert1, var_2);
            var_6 = wp::address(var_vert2, var_4);
            var_8 = wp::load(var_5);
            var_9 = wp::load(var_6);
            var_7 = wp::dot(var_8, var_9);
            var_12 = (var_7 < var_11);
            if (var_12) {
                // res[0] = i                                                                 <L 1523>
                wp::assign_inplace(var_0, var_13, var_2);
                // res[1] = j                                                                 <L 1524>
                wp::assign_inplace(var_0, var_14, var_4);
                // return 1, res                                                              <L 1525>
                ret_0 = var_15;
                ret_1 = var_0;
                return;
            }
            goto start_for_2;
        end_for_2:;
        goto start_for_0;
    end_for_0:;
    // return 0, res                                                                          <L 1526>
    ret_0 = var_16;
    ret_1 = var_0;
    return;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1798
static CUDA_CALLABLE wp::int32 _box_edge_normals_0(
    wp::int32 var_dim,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_pos,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> var_v1,
    wp::vec_t<3, wp::float32> var_v2,
    wp::int32 var_v1i,
    wp::array_t<wp::vec_t<3, wp::float32>> var_normal_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_endvert_out)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 2;
    bool var_1;
    const wp::int32 var_2 = 0;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    const wp::int32 var_5 = 0;
    const wp::int32 var_6 = 1;
    const wp::int32 var_7 = 1;
    bool var_8;
    const wp::int32 var_9 = 1;
    wp::int32 var_10;
    const wp::int32 var_11 = 0;
    wp::float32 var_12;
    const wp::int32 var_13 = 0;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    const wp::int32 var_17 = 2;
    wp::int32 var_18;
    const wp::int32 var_19 = 1;
    wp::float32 var_20;
    const wp::int32 var_21 = 1;
    wp::float32 var_22;
    wp::float32 var_23;
    wp::float32 var_24;
    const wp::int32 var_25 = 4;
    wp::int32 var_26;
    const wp::int32 var_27 = 2;
    wp::float32 var_28;
    const wp::int32 var_29 = 2;
    wp::float32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::vec_t<3, wp::float32> var_34;
    wp::vec_t<3, wp::float32> var_35;
    wp::vec_t<3, wp::float32> var_36;
    const wp::int32 var_37 = 0;
    const wp::int32 var_38 = 0;
    wp::vec_t<3, wp::float32>* var_39;
    wp::vec_t<3, wp::float32> var_40;
    wp::vec_t<3, wp::float32> var_41;
    wp::vec_t<3, wp::float32> var_42;
    const wp::int32 var_43 = 0;
    wp::float32 var_44;
    wp::vec_t<3, wp::float32> var_45;
    wp::vec_t<3, wp::float32> var_46;
    wp::vec_t<3, wp::float32> var_47;
    const wp::int32 var_48 = 1;
    const wp::int32 var_49 = 1;
    wp::vec_t<3, wp::float32>* var_50;
    wp::vec_t<3, wp::float32> var_51;
    wp::vec_t<3, wp::float32> var_52;
    wp::vec_t<3, wp::float32> var_53;
    const wp::int32 var_54 = 1;
    wp::float32 var_55;
    wp::vec_t<3, wp::float32> var_56;
    wp::vec_t<3, wp::float32> var_57;
    wp::vec_t<3, wp::float32> var_58;
    const wp::int32 var_59 = 2;
    const wp::int32 var_60 = 2;
    wp::vec_t<3, wp::float32>* var_61;
    wp::vec_t<3, wp::float32> var_62;
    wp::vec_t<3, wp::float32> var_63;
    wp::vec_t<3, wp::float32> var_64;
    const wp::int32 var_65 = 2;
    const wp::int32 var_66 = 3;
    const wp::int32 var_67 = 0;
    //---------
    // forward
    // def _box_edge_normals(                                                                 <L 1799>
    // if dim == 2:                                                                           <L 1812>
    var_1 = (var_dim == var_0);
    if (var_1) {
        // endvert_out[0] = v2                                                                <L 1813>
        wp::array_store(var_endvert_out, var_2, var_v2);
        // normal_out[0] = wp.normalize(v2 - v1)                                              <L 1814>
        var_3 = wp::sub(var_v2, var_v1);
        var_4 = wp::normalize(var_3);
        wp::array_store(var_normal_out, var_5, var_4);
        // return 1                                                                           <L 1815>
        return var_6;
    }
    // if dim == 1:                                                                           <L 1818>
    var_8 = (var_dim == var_7);
    if (var_8) {
        // x = wp.where(v1i & 1, size[0], -size[0])                                           <L 1819>
        var_10 = wp::bit_and(var_v1i, var_9);
        var_12 = wp::extract(var_size, var_11);
        var_14 = wp::extract(var_size, var_13);
        var_15 = wp::neg(var_14);
        var_16 = wp::where(var_10, var_12, var_15);
        // y = wp.where(v1i & 2, size[1], -size[1])                                           <L 1820>
        var_18 = wp::bit_and(var_v1i, var_17);
        var_20 = wp::extract(var_size, var_19);
        var_22 = wp::extract(var_size, var_21);
        var_23 = wp::neg(var_22);
        var_24 = wp::where(var_18, var_20, var_23);
        // z = wp.where(v1i & 4, size[2], -size[2])                                           <L 1821>
        var_26 = wp::bit_and(var_v1i, var_25);
        var_28 = wp::extract(var_size, var_27);
        var_30 = wp::extract(var_size, var_29);
        var_31 = wp::neg(var_30);
        var_32 = wp::where(var_26, var_28, var_31);
        // endvert_out[0] = mat @ wp.vec3(-x, y, z) + pos                                     <L 1823>
        var_33 = wp::neg(var_16);
        var_34 = wp::vec_t<3, wp::float32>(var_33, var_24, var_32);
        var_35 = wp::mul(var_mat, var_34);
        var_36 = wp::add(var_35, var_pos);
        wp::array_store(var_endvert_out, var_37, var_36);
        // normal_out[0] = wp.normalize(endvert_out[0] - v1)                                  <L 1824>
        var_39 = wp::address(var_endvert_out, var_38);
        var_41 = wp::load(var_39);
        var_40 = wp::sub(var_41, var_v1);
        var_42 = wp::normalize(var_40);
        wp::array_store(var_normal_out, var_43, var_42);
        // endvert_out[1] = mat @ wp.vec3(x, -y, z) + pos                                     <L 1826>
        var_44 = wp::neg(var_24);
        var_45 = wp::vec_t<3, wp::float32>(var_16, var_44, var_32);
        var_46 = wp::mul(var_mat, var_45);
        var_47 = wp::add(var_46, var_pos);
        wp::array_store(var_endvert_out, var_48, var_47);
        // normal_out[1] = wp.normalize(endvert_out[1] - v1)                                  <L 1827>
        var_50 = wp::address(var_endvert_out, var_49);
        var_52 = wp::load(var_50);
        var_51 = wp::sub(var_52, var_v1);
        var_53 = wp::normalize(var_51);
        wp::array_store(var_normal_out, var_54, var_53);
        // endvert_out[2] = mat @ wp.vec3(x, y, -z) + pos                                     <L 1829>
        var_55 = wp::neg(var_32);
        var_56 = wp::vec_t<3, wp::float32>(var_16, var_24, var_55);
        var_57 = wp::mul(var_mat, var_56);
        var_58 = wp::add(var_57, var_pos);
        wp::array_store(var_endvert_out, var_59, var_58);
        // normal_out[2] = wp.normalize(endvert_out[2] - v1)                                  <L 1830>
        var_61 = wp::address(var_endvert_out, var_60);
        var_63 = wp::load(var_61);
        var_62 = wp::sub(var_63, var_v1);
        var_64 = wp::normalize(var_62);
        wp::array_store(var_normal_out, var_65, var_64);
        // return 3                                                                           <L 1831>
        return var_66;
    }
    // return 0                                                                               <L 1832>
    return var_67;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1643
static CUDA_CALLABLE wp::int32 _mesh_edge_normals_0(
    wp::int32 var_dim,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_pos,
    wp::int32 var_vertadr,
    wp::int32 var_polyadr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vert,
    wp::array_t<wp::int32> var_polyvertadr,
    wp::array_t<wp::int32> var_polyvertnum,
    wp::array_t<wp::int32> var_polyvert,
    wp::array_t<wp::int32> var_polymapadr,
    wp::array_t<wp::int32> var_polymapnum,
    wp::array_t<wp::int32> var_polymap,
    wp::vec_t<3, wp::float32> var_v1,
    wp::vec_t<3, wp::float32> var_v2,
    wp::int32 var_v1i,
    wp::array_t<wp::vec_t<3, wp::float32>> var_normals_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_endverts_out)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 2;
    bool var_1;
    const wp::int32 var_2 = 0;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    const wp::int32 var_5 = 0;
    const wp::int32 var_6 = 1;
    const wp::int32 var_7 = 1;
    bool var_8;
    wp::int32 var_9;
    wp::int32* var_10;
    wp::int32 var_11;
    wp::int32 var_12;
    wp::int32 var_13;
    wp::int32* var_14;
    wp::int32 var_15;
    wp::int32 var_16;
    wp::range_t var_17;
    wp::int32 var_18;
    wp::int32 var_19;
    wp::int32* var_20;
    wp::int32 var_21;
    wp::int32 var_22;
    wp::int32 var_23;
    wp::int32* var_24;
    wp::int32 var_25;
    wp::int32 var_26;
    wp::int32 var_27;
    wp::int32* var_28;
    wp::int32 var_29;
    wp::int32 var_30;
    wp::range_t var_31;
    wp::int32 var_32;
    wp::int32 var_33;
    wp::int32* var_34;
    bool var_35;
    wp::int32 var_36;
    const wp::int32 var_37 = 0;
    bool var_38;
    const wp::int32 var_39 = 1;
    wp::int32 var_40;
    const wp::int32 var_41 = 1;
    wp::int32 var_42;
    wp::int32 var_43;
    wp::int32 var_44;
    wp::int32* var_45;
    wp::int32 var_46;
    wp::int32 var_47;
    wp::vec_t<3, wp::float32>* var_48;
    wp::vec_t<3, wp::float32> var_49;
    wp::vec_t<3, wp::float32> var_50;
    wp::vec_t<3, wp::float32> var_51;
    wp::vec_t<3, wp::float32>* var_52;
    wp::vec_t<3, wp::float32> var_53;
    wp::vec_t<3, wp::float32> var_54;
    wp::vec_t<3, wp::float32> var_55;
    const wp::int32 var_56 = 0;
    //---------
    // forward
    // def _mesh_edge_normals(                                                                <L 1644>
    // if dim == 2:                                                                           <L 1666>
    var_1 = (var_dim == var_0);
    if (var_1) {
        // endverts_out[0] = v2                                                               <L 1667>
        wp::array_store(var_endverts_out, var_2, var_v2);
        // normals_out[0] = wp.normalize(v2 - v1)                                             <L 1668>
        var_3 = wp::sub(var_v2, var_v1);
        var_4 = wp::normalize(var_3);
        wp::array_store(var_normals_out, var_5, var_4);
        // return 1                                                                           <L 1669>
        return var_6;
    }
    // if dim == 1:                                                                           <L 1671>
    var_8 = (var_dim == var_7);
    if (var_8) {
        // v1_adr = polymapadr[vertadr + v1i]                                                 <L 1672>
        var_9 = wp::add(var_vertadr, var_v1i);
        var_10 = wp::address(var_polymapadr, var_9);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // v1_num = polymapnum[vertadr + v1i]                                                 <L 1673>
        var_13 = wp::add(var_vertadr, var_v1i);
        var_14 = wp::address(var_polymapnum, var_13);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // for i in range(v1_num):                                                            <L 1676>
        var_17 = wp::range(var_15);
        start_for_1:;
            if (iter_cmp(var_17) == 0) goto end_for_1;
            var_18 = wp::iter_next(var_17);
            // idx = polymap[v1_adr + i]                                                      <L 1677>
            var_19 = wp::add(var_11, var_18);
            var_20 = wp::address(var_polymap, var_19);
            var_22 = wp::load(var_20);
            var_21 = wp::copy(var_22);
            // adr = polyvertadr[polyadr + idx]                                               <L 1678>
            var_23 = wp::add(var_polyadr, var_21);
            var_24 = wp::address(var_polyvertadr, var_23);
            var_26 = wp::load(var_24);
            var_25 = wp::copy(var_26);
            // nvert = polyvertnum[polyadr + idx]                                             <L 1679>
            var_27 = wp::add(var_polyadr, var_21);
            var_28 = wp::address(var_polyvertnum, var_27);
            var_30 = wp::load(var_28);
            var_29 = wp::copy(var_30);
            // for j in range(nvert):                                                         <L 1681>
            var_31 = wp::range(var_29);
            start_for_3:;
                if (iter_cmp(var_31) == 0) goto end_for_3;
                var_32 = wp::iter_next(var_31);
                // if polyvert[adr + j] == v1i:                                               <L 1682>
                var_33 = wp::add(var_25, var_32);
                var_34 = wp::address(var_polyvert, var_33);
                var_36 = wp::load(var_34);
                var_35 = (var_36 == var_v1i);
                if (var_35) {
                    // k = wp.where(j == 0, nvert - 1, j - 1)                                 <L 1683>
                    var_38 = (var_32 == var_37);
                    var_40 = wp::sub(var_29, var_39);
                    var_42 = wp::sub(var_32, var_41);
                    var_43 = wp::where(var_38, var_40, var_42);
                    // endverts_out[i] = mat @ vert[vertadr + polyvert[adr + k]] + pos        <L 1684>
                    var_44 = wp::add(var_25, var_43);
                    var_45 = wp::address(var_polyvert, var_44);
                    var_47 = wp::load(var_45);
                    var_46 = wp::add(var_vertadr, var_47);
                    var_48 = wp::address(var_vert, var_46);
                    var_50 = wp::load(var_48);
                    var_49 = wp::mul(var_mat, var_50);
                    var_51 = wp::add(var_49, var_pos);
                    wp::array_store(var_endverts_out, var_18, var_51);
                    // normals_out[i] = wp.normalize(endverts_out[i] - v1)                    <L 1685>
                    var_52 = wp::address(var_endverts_out, var_18);
                    var_54 = wp::load(var_52);
                    var_53 = wp::sub(var_54, var_v1);
                    var_55 = wp::normalize(var_53);
                    wp::array_store(var_normals_out, var_18, var_55);
                }
                goto start_for_3;
            end_for_3:;
            goto start_for_1;
        end_for_1:;
        // return v1_num                                                                      <L 1686>
        return var_15;
    }
    // return 0                                                                               <L 1687>
    return var_56;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1531
static CUDA_CALLABLE void _aligned_face_edge_0(
    wp::array_t<wp::vec_t<3, wp::float32>> var_edge,
    wp::int32 var_nedge,
    wp::array_t<wp::vec_t<3, wp::float32>> var_face,
    wp::int32 var_nface,
    wp::int32 & ret_0,
    wp::vec_t<2, wp::int32> & ret_1)
{
    //---------
    // primal vars
    wp::vec_t<2, wp::int32> var_0;
    wp::range_t var_1;
    wp::int32 var_2;
    wp::range_t var_3;
    wp::int32 var_4;
    wp::vec_t<3, wp::float32>* var_5;
    wp::vec_t<3, wp::float32>* var_6;
    wp::float32 var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::float32 var_10;
    const wp::float32 var_11 = 0.0015999993173334208;
    bool var_12;
    const wp::int32 var_13 = 0;
    const wp::int32 var_14 = 1;
    const wp::int32 var_15 = 1;
    const wp::int32 var_16 = 0;
    //---------
    // forward
    // def _aligned_face_edge(edge: wp.array[wp.vec3], nedge: int, face: wp.array[wp.vec3], nface: int) -> Tuple[int, wp.vec2i]:       <L 1532>
    // res = wp.vec2i()                                                                       <L 1533>
    var_0 = wp::vec_t<2, wp::int32>();
    // for i in range(nface):                                                                 <L 1534>
    var_1 = wp::range(var_nface);
    start_for_0:;
        if (iter_cmp(var_1) == 0) goto end_for_0;
        var_2 = wp::iter_next(var_1);
        // for j in range(nedge):                                                             <L 1535>
        var_3 = wp::range(var_nedge);
        start_for_2:;
            if (iter_cmp(var_3) == 0) goto end_for_2;
            var_4 = wp::iter_next(var_3);
            // if wp.abs(wp.dot(edge[j], face[i])) < EDGE_TOL:                                <L 1536>
            var_5 = wp::address(var_edge, var_4);
            var_6 = wp::address(var_face, var_2);
            var_8 = wp::load(var_5);
            var_9 = wp::load(var_6);
            var_7 = wp::dot(var_8, var_9);
            var_10 = wp::abs(var_7);
            var_12 = (var_10 < var_11);
            if (var_12) {
                // res[0] = j                                                                 <L 1537>
                wp::assign_inplace(var_0, var_13, var_4);
                // res[1] = i                                                                 <L 1538>
                wp::assign_inplace(var_0, var_14, var_2);
                // return 1, res                                                              <L 1539>
                ret_0 = var_15;
                ret_1 = var_0;
                return;
            }
            goto start_for_2;
        end_for_2:;
        goto start_for_0;
    end_for_0:;
    // return 0, res                                                                          <L 1540>
    ret_0 = var_16;
    ret_1 = var_0;
    return;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:2047
static CUDA_CALLABLE wp::int32 _set_edge_0(
    wp::array_t<wp::vec_t<3, wp::float32>> var_vert1,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vert2,
    wp::int32 var_start,
    wp::int32 var_end,
    wp::int32 var_offset,
    wp::array_t<wp::vec_t<3, wp::float32>> var_face_out)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 2;
    wp::int32 var_1;
    wp::int32 var_2;
    wp::vec_t<3, wp::float32>* var_3;
    const wp::int32 var_4 = 0;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32>* var_6;
    const wp::int32 var_7 = 1;
    wp::vec_t<3, wp::float32> var_8;
    const wp::int32 var_9 = 2;
    //---------
    // forward
    // def _set_edge(                                                                         <L 2048>
    // face_out[0] = vert1[2 * start + offset]                                                <L 2058>
    var_1 = wp::mul(var_0, var_start);
    var_2 = wp::add(var_1, var_offset);
    var_3 = wp::address(var_vert1, var_2);
    var_5 = wp::load(var_3);
    wp::array_store(var_face_out, var_4, var_5);
    // face_out[1] = vert2[end]                                                               <L 2059>
    var_6 = wp::address(var_vert2, var_end);
    var_8 = wp::load(var_6);
    wp::array_store(var_face_out, var_7, var_8);
    // return 2                                                                               <L 2060>
    return var_9;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1836
static CUDA_CALLABLE wp::int32 _box_face_0(
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_pos,
    wp::vec_t<3, wp::float32> var_size,
    wp::int32 var_idx,
    wp::array_t<wp::vec_t<3, wp::float32>> var_face_out)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    bool var_1;
    const wp::int32 var_2 = 0;
    wp::float32 var_3;
    const wp::int32 var_4 = 1;
    wp::float32 var_5;
    const wp::int32 var_6 = 2;
    wp::float32 var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::vec_t<3, wp::float32> var_10;
    const wp::int32 var_11 = 0;
    const wp::int32 var_12 = 0;
    wp::float32 var_13;
    const wp::int32 var_14 = 1;
    wp::float32 var_15;
    const wp::int32 var_16 = 2;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::vec_t<3, wp::float32> var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32> var_21;
    const wp::int32 var_22 = 1;
    const wp::int32 var_23 = 0;
    wp::float32 var_24;
    const wp::int32 var_25 = 1;
    wp::float32 var_26;
    wp::float32 var_27;
    const wp::int32 var_28 = 2;
    wp::float32 var_29;
    wp::float32 var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::vec_t<3, wp::float32> var_33;
    const wp::int32 var_34 = 2;
    const wp::int32 var_35 = 0;
    wp::float32 var_36;
    const wp::int32 var_37 = 1;
    wp::float32 var_38;
    wp::float32 var_39;
    const wp::int32 var_40 = 2;
    wp::float32 var_41;
    wp::vec_t<3, wp::float32> var_42;
    wp::vec_t<3, wp::float32> var_43;
    wp::vec_t<3, wp::float32> var_44;
    const wp::int32 var_45 = 3;
    const wp::int32 var_46 = 4;
    const wp::int32 var_47 = 1;
    bool var_48;
    const wp::int32 var_49 = 0;
    wp::float32 var_50;
    wp::float32 var_51;
    const wp::int32 var_52 = 1;
    wp::float32 var_53;
    const wp::int32 var_54 = 2;
    wp::float32 var_55;
    wp::float32 var_56;
    wp::vec_t<3, wp::float32> var_57;
    wp::vec_t<3, wp::float32> var_58;
    wp::vec_t<3, wp::float32> var_59;
    const wp::int32 var_60 = 0;
    const wp::int32 var_61 = 0;
    wp::float32 var_62;
    wp::float32 var_63;
    const wp::int32 var_64 = 1;
    wp::float32 var_65;
    const wp::int32 var_66 = 2;
    wp::float32 var_67;
    wp::vec_t<3, wp::float32> var_68;
    wp::vec_t<3, wp::float32> var_69;
    wp::vec_t<3, wp::float32> var_70;
    const wp::int32 var_71 = 1;
    const wp::int32 var_72 = 0;
    wp::float32 var_73;
    wp::float32 var_74;
    const wp::int32 var_75 = 1;
    wp::float32 var_76;
    wp::float32 var_77;
    const wp::int32 var_78 = 2;
    wp::float32 var_79;
    wp::vec_t<3, wp::float32> var_80;
    wp::vec_t<3, wp::float32> var_81;
    wp::vec_t<3, wp::float32> var_82;
    const wp::int32 var_83 = 2;
    const wp::int32 var_84 = 0;
    wp::float32 var_85;
    wp::float32 var_86;
    const wp::int32 var_87 = 1;
    wp::float32 var_88;
    wp::float32 var_89;
    const wp::int32 var_90 = 2;
    wp::float32 var_91;
    wp::float32 var_92;
    wp::vec_t<3, wp::float32> var_93;
    wp::vec_t<3, wp::float32> var_94;
    wp::vec_t<3, wp::float32> var_95;
    const wp::int32 var_96 = 3;
    const wp::int32 var_97 = 4;
    const wp::int32 var_98 = 2;
    bool var_99;
    const wp::int32 var_100 = 0;
    wp::float32 var_101;
    wp::float32 var_102;
    const wp::int32 var_103 = 1;
    wp::float32 var_104;
    const wp::int32 var_105 = 2;
    wp::float32 var_106;
    wp::float32 var_107;
    wp::vec_t<3, wp::float32> var_108;
    wp::vec_t<3, wp::float32> var_109;
    wp::vec_t<3, wp::float32> var_110;
    const wp::int32 var_111 = 0;
    const wp::int32 var_112 = 0;
    wp::float32 var_113;
    const wp::int32 var_114 = 1;
    wp::float32 var_115;
    const wp::int32 var_116 = 2;
    wp::float32 var_117;
    wp::float32 var_118;
    wp::vec_t<3, wp::float32> var_119;
    wp::vec_t<3, wp::float32> var_120;
    wp::vec_t<3, wp::float32> var_121;
    const wp::int32 var_122 = 1;
    const wp::int32 var_123 = 0;
    wp::float32 var_124;
    const wp::int32 var_125 = 1;
    wp::float32 var_126;
    const wp::int32 var_127 = 2;
    wp::float32 var_128;
    wp::vec_t<3, wp::float32> var_129;
    wp::vec_t<3, wp::float32> var_130;
    wp::vec_t<3, wp::float32> var_131;
    const wp::int32 var_132 = 2;
    const wp::int32 var_133 = 0;
    wp::float32 var_134;
    wp::float32 var_135;
    const wp::int32 var_136 = 1;
    wp::float32 var_137;
    const wp::int32 var_138 = 2;
    wp::float32 var_139;
    wp::vec_t<3, wp::float32> var_140;
    wp::vec_t<3, wp::float32> var_141;
    wp::vec_t<3, wp::float32> var_142;
    const wp::int32 var_143 = 3;
    const wp::int32 var_144 = 4;
    const wp::int32 var_145 = 3;
    bool var_146;
    const wp::int32 var_147 = 0;
    wp::float32 var_148;
    wp::float32 var_149;
    const wp::int32 var_150 = 1;
    wp::float32 var_151;
    wp::float32 var_152;
    const wp::int32 var_153 = 2;
    wp::float32 var_154;
    wp::vec_t<3, wp::float32> var_155;
    wp::vec_t<3, wp::float32> var_156;
    wp::vec_t<3, wp::float32> var_157;
    const wp::int32 var_158 = 0;
    const wp::int32 var_159 = 0;
    wp::float32 var_160;
    const wp::int32 var_161 = 1;
    wp::float32 var_162;
    wp::float32 var_163;
    const wp::int32 var_164 = 2;
    wp::float32 var_165;
    wp::vec_t<3, wp::float32> var_166;
    wp::vec_t<3, wp::float32> var_167;
    wp::vec_t<3, wp::float32> var_168;
    const wp::int32 var_169 = 1;
    const wp::int32 var_170 = 0;
    wp::float32 var_171;
    const wp::int32 var_172 = 1;
    wp::float32 var_173;
    wp::float32 var_174;
    const wp::int32 var_175 = 2;
    wp::float32 var_176;
    wp::float32 var_177;
    wp::vec_t<3, wp::float32> var_178;
    wp::vec_t<3, wp::float32> var_179;
    wp::vec_t<3, wp::float32> var_180;
    const wp::int32 var_181 = 2;
    const wp::int32 var_182 = 0;
    wp::float32 var_183;
    wp::float32 var_184;
    const wp::int32 var_185 = 1;
    wp::float32 var_186;
    wp::float32 var_187;
    const wp::int32 var_188 = 2;
    wp::float32 var_189;
    wp::float32 var_190;
    wp::vec_t<3, wp::float32> var_191;
    wp::vec_t<3, wp::float32> var_192;
    wp::vec_t<3, wp::float32> var_193;
    const wp::int32 var_194 = 3;
    const wp::int32 var_195 = 4;
    const wp::int32 var_196 = 4;
    bool var_197;
    const wp::int32 var_198 = 0;
    wp::float32 var_199;
    wp::float32 var_200;
    const wp::int32 var_201 = 1;
    wp::float32 var_202;
    const wp::int32 var_203 = 2;
    wp::float32 var_204;
    wp::vec_t<3, wp::float32> var_205;
    wp::vec_t<3, wp::float32> var_206;
    wp::vec_t<3, wp::float32> var_207;
    const wp::int32 var_208 = 0;
    const wp::int32 var_209 = 0;
    wp::float32 var_210;
    const wp::int32 var_211 = 1;
    wp::float32 var_212;
    const wp::int32 var_213 = 2;
    wp::float32 var_214;
    wp::vec_t<3, wp::float32> var_215;
    wp::vec_t<3, wp::float32> var_216;
    wp::vec_t<3, wp::float32> var_217;
    const wp::int32 var_218 = 1;
    const wp::int32 var_219 = 0;
    wp::float32 var_220;
    const wp::int32 var_221 = 1;
    wp::float32 var_222;
    wp::float32 var_223;
    const wp::int32 var_224 = 2;
    wp::float32 var_225;
    wp::vec_t<3, wp::float32> var_226;
    wp::vec_t<3, wp::float32> var_227;
    wp::vec_t<3, wp::float32> var_228;
    const wp::int32 var_229 = 2;
    const wp::int32 var_230 = 0;
    wp::float32 var_231;
    wp::float32 var_232;
    const wp::int32 var_233 = 1;
    wp::float32 var_234;
    wp::float32 var_235;
    const wp::int32 var_236 = 2;
    wp::float32 var_237;
    wp::vec_t<3, wp::float32> var_238;
    wp::vec_t<3, wp::float32> var_239;
    wp::vec_t<3, wp::float32> var_240;
    const wp::int32 var_241 = 3;
    const wp::int32 var_242 = 4;
    const wp::int32 var_243 = 5;
    bool var_244;
    const wp::int32 var_245 = 0;
    wp::float32 var_246;
    const wp::int32 var_247 = 1;
    wp::float32 var_248;
    const wp::int32 var_249 = 2;
    wp::float32 var_250;
    wp::float32 var_251;
    wp::vec_t<3, wp::float32> var_252;
    wp::vec_t<3, wp::float32> var_253;
    wp::vec_t<3, wp::float32> var_254;
    const wp::int32 var_255 = 0;
    const wp::int32 var_256 = 0;
    wp::float32 var_257;
    wp::float32 var_258;
    const wp::int32 var_259 = 1;
    wp::float32 var_260;
    const wp::int32 var_261 = 2;
    wp::float32 var_262;
    wp::float32 var_263;
    wp::vec_t<3, wp::float32> var_264;
    wp::vec_t<3, wp::float32> var_265;
    wp::vec_t<3, wp::float32> var_266;
    const wp::int32 var_267 = 1;
    const wp::int32 var_268 = 0;
    wp::float32 var_269;
    wp::float32 var_270;
    const wp::int32 var_271 = 1;
    wp::float32 var_272;
    wp::float32 var_273;
    const wp::int32 var_274 = 2;
    wp::float32 var_275;
    wp::float32 var_276;
    wp::vec_t<3, wp::float32> var_277;
    wp::vec_t<3, wp::float32> var_278;
    wp::vec_t<3, wp::float32> var_279;
    const wp::int32 var_280 = 2;
    const wp::int32 var_281 = 0;
    wp::float32 var_282;
    const wp::int32 var_283 = 1;
    wp::float32 var_284;
    wp::float32 var_285;
    const wp::int32 var_286 = 2;
    wp::float32 var_287;
    wp::float32 var_288;
    wp::vec_t<3, wp::float32> var_289;
    wp::vec_t<3, wp::float32> var_290;
    wp::vec_t<3, wp::float32> var_291;
    const wp::int32 var_292 = 3;
    const wp::int32 var_293 = 4;
    const wp::int32 var_294 = 0;
    //---------
    // forward
    // def _box_face(mat: wp.mat33, pos: wp.vec3, size: wp.vec3, idx: int, face_out: wp.array[wp.vec3]) -> int:       <L 1837>
    // if idx == 0:  # right                                                                  <L 1839>
    var_1 = (var_idx == var_0);
    if (var_1) {
        // face_out[0] = mat @ wp.vec3(size[0], size[1], size[2]) + pos                       <L 1840>
        var_3 = wp::extract(var_size, var_2);
        var_5 = wp::extract(var_size, var_4);
        var_7 = wp::extract(var_size, var_6);
        var_8 = wp::vec_t<3, wp::float32>(var_3, var_5, var_7);
        var_9 = wp::mul(var_mat, var_8);
        var_10 = wp::add(var_9, var_pos);
        wp::array_store(var_face_out, var_11, var_10);
        // face_out[1] = mat @ wp.vec3(size[0], size[1], -size[2]) + pos                      <L 1841>
        var_13 = wp::extract(var_size, var_12);
        var_15 = wp::extract(var_size, var_14);
        var_17 = wp::extract(var_size, var_16);
        var_18 = wp::neg(var_17);
        var_19 = wp::vec_t<3, wp::float32>(var_13, var_15, var_18);
        var_20 = wp::mul(var_mat, var_19);
        var_21 = wp::add(var_20, var_pos);
        wp::array_store(var_face_out, var_22, var_21);
        // face_out[2] = mat @ wp.vec3(size[0], -size[1], -size[2]) + pos                     <L 1842>
        var_24 = wp::extract(var_size, var_23);
        var_26 = wp::extract(var_size, var_25);
        var_27 = wp::neg(var_26);
        var_29 = wp::extract(var_size, var_28);
        var_30 = wp::neg(var_29);
        var_31 = wp::vec_t<3, wp::float32>(var_24, var_27, var_30);
        var_32 = wp::mul(var_mat, var_31);
        var_33 = wp::add(var_32, var_pos);
        wp::array_store(var_face_out, var_34, var_33);
        // face_out[3] = mat @ wp.vec3(size[0], -size[1], size[2]) + pos                      <L 1843>
        var_36 = wp::extract(var_size, var_35);
        var_38 = wp::extract(var_size, var_37);
        var_39 = wp::neg(var_38);
        var_41 = wp::extract(var_size, var_40);
        var_42 = wp::vec_t<3, wp::float32>(var_36, var_39, var_41);
        var_43 = wp::mul(var_mat, var_42);
        var_44 = wp::add(var_43, var_pos);
        wp::array_store(var_face_out, var_45, var_44);
        // return 4                                                                           <L 1844>
        return var_46;
    }
    // if idx == 1:  # left                                                                   <L 1845>
    var_48 = (var_idx == var_47);
    if (var_48) {
        // face_out[0] = mat @ wp.vec3(-size[0], size[1], -size[2]) + pos                     <L 1846>
        var_50 = wp::extract(var_size, var_49);
        var_51 = wp::neg(var_50);
        var_53 = wp::extract(var_size, var_52);
        var_55 = wp::extract(var_size, var_54);
        var_56 = wp::neg(var_55);
        var_57 = wp::vec_t<3, wp::float32>(var_51, var_53, var_56);
        var_58 = wp::mul(var_mat, var_57);
        var_59 = wp::add(var_58, var_pos);
        wp::array_store(var_face_out, var_60, var_59);
        // face_out[1] = mat @ wp.vec3(-size[0], size[1], size[2]) + pos                      <L 1847>
        var_62 = wp::extract(var_size, var_61);
        var_63 = wp::neg(var_62);
        var_65 = wp::extract(var_size, var_64);
        var_67 = wp::extract(var_size, var_66);
        var_68 = wp::vec_t<3, wp::float32>(var_63, var_65, var_67);
        var_69 = wp::mul(var_mat, var_68);
        var_70 = wp::add(var_69, var_pos);
        wp::array_store(var_face_out, var_71, var_70);
        // face_out[2] = mat @ wp.vec3(-size[0], -size[1], size[2]) + pos                     <L 1848>
        var_73 = wp::extract(var_size, var_72);
        var_74 = wp::neg(var_73);
        var_76 = wp::extract(var_size, var_75);
        var_77 = wp::neg(var_76);
        var_79 = wp::extract(var_size, var_78);
        var_80 = wp::vec_t<3, wp::float32>(var_74, var_77, var_79);
        var_81 = wp::mul(var_mat, var_80);
        var_82 = wp::add(var_81, var_pos);
        wp::array_store(var_face_out, var_83, var_82);
        // face_out[3] = mat @ wp.vec3(-size[0], -size[1], -size[2]) + pos                    <L 1849>
        var_85 = wp::extract(var_size, var_84);
        var_86 = wp::neg(var_85);
        var_88 = wp::extract(var_size, var_87);
        var_89 = wp::neg(var_88);
        var_91 = wp::extract(var_size, var_90);
        var_92 = wp::neg(var_91);
        var_93 = wp::vec_t<3, wp::float32>(var_86, var_89, var_92);
        var_94 = wp::mul(var_mat, var_93);
        var_95 = wp::add(var_94, var_pos);
        wp::array_store(var_face_out, var_96, var_95);
        // return 4                                                                           <L 1850>
        return var_97;
    }
    // if idx == 2:  # top                                                                    <L 1851>
    var_99 = (var_idx == var_98);
    if (var_99) {
        // face_out[0] = mat @ wp.vec3(-size[0], size[1], -size[2]) + pos                     <L 1852>
        var_101 = wp::extract(var_size, var_100);
        var_102 = wp::neg(var_101);
        var_104 = wp::extract(var_size, var_103);
        var_106 = wp::extract(var_size, var_105);
        var_107 = wp::neg(var_106);
        var_108 = wp::vec_t<3, wp::float32>(var_102, var_104, var_107);
        var_109 = wp::mul(var_mat, var_108);
        var_110 = wp::add(var_109, var_pos);
        wp::array_store(var_face_out, var_111, var_110);
        // face_out[1] = mat @ wp.vec3(size[0], size[1], -size[2]) + pos                      <L 1853>
        var_113 = wp::extract(var_size, var_112);
        var_115 = wp::extract(var_size, var_114);
        var_117 = wp::extract(var_size, var_116);
        var_118 = wp::neg(var_117);
        var_119 = wp::vec_t<3, wp::float32>(var_113, var_115, var_118);
        var_120 = wp::mul(var_mat, var_119);
        var_121 = wp::add(var_120, var_pos);
        wp::array_store(var_face_out, var_122, var_121);
        // face_out[2] = mat @ wp.vec3(size[0], size[1], size[2]) + pos                       <L 1854>
        var_124 = wp::extract(var_size, var_123);
        var_126 = wp::extract(var_size, var_125);
        var_128 = wp::extract(var_size, var_127);
        var_129 = wp::vec_t<3, wp::float32>(var_124, var_126, var_128);
        var_130 = wp::mul(var_mat, var_129);
        var_131 = wp::add(var_130, var_pos);
        wp::array_store(var_face_out, var_132, var_131);
        // face_out[3] = mat @ wp.vec3(-size[0], size[1], size[2]) + pos                      <L 1855>
        var_134 = wp::extract(var_size, var_133);
        var_135 = wp::neg(var_134);
        var_137 = wp::extract(var_size, var_136);
        var_139 = wp::extract(var_size, var_138);
        var_140 = wp::vec_t<3, wp::float32>(var_135, var_137, var_139);
        var_141 = wp::mul(var_mat, var_140);
        var_142 = wp::add(var_141, var_pos);
        wp::array_store(var_face_out, var_143, var_142);
        // return 4                                                                           <L 1856>
        return var_144;
    }
    // if idx == 3:  # bottom                                                                 <L 1857>
    var_146 = (var_idx == var_145);
    if (var_146) {
        // face_out[0] = mat @ wp.vec3(-size[0], -size[1], size[2]) + pos                     <L 1858>
        var_148 = wp::extract(var_size, var_147);
        var_149 = wp::neg(var_148);
        var_151 = wp::extract(var_size, var_150);
        var_152 = wp::neg(var_151);
        var_154 = wp::extract(var_size, var_153);
        var_155 = wp::vec_t<3, wp::float32>(var_149, var_152, var_154);
        var_156 = wp::mul(var_mat, var_155);
        var_157 = wp::add(var_156, var_pos);
        wp::array_store(var_face_out, var_158, var_157);
        // face_out[1] = mat @ wp.vec3(size[0], -size[1], size[2]) + pos                      <L 1859>
        var_160 = wp::extract(var_size, var_159);
        var_162 = wp::extract(var_size, var_161);
        var_163 = wp::neg(var_162);
        var_165 = wp::extract(var_size, var_164);
        var_166 = wp::vec_t<3, wp::float32>(var_160, var_163, var_165);
        var_167 = wp::mul(var_mat, var_166);
        var_168 = wp::add(var_167, var_pos);
        wp::array_store(var_face_out, var_169, var_168);
        // face_out[2] = mat @ wp.vec3(size[0], -size[1], -size[2]) + pos                     <L 1860>
        var_171 = wp::extract(var_size, var_170);
        var_173 = wp::extract(var_size, var_172);
        var_174 = wp::neg(var_173);
        var_176 = wp::extract(var_size, var_175);
        var_177 = wp::neg(var_176);
        var_178 = wp::vec_t<3, wp::float32>(var_171, var_174, var_177);
        var_179 = wp::mul(var_mat, var_178);
        var_180 = wp::add(var_179, var_pos);
        wp::array_store(var_face_out, var_181, var_180);
        // face_out[3] = mat @ wp.vec3(-size[0], -size[1], -size[2]) + pos                    <L 1861>
        var_183 = wp::extract(var_size, var_182);
        var_184 = wp::neg(var_183);
        var_186 = wp::extract(var_size, var_185);
        var_187 = wp::neg(var_186);
        var_189 = wp::extract(var_size, var_188);
        var_190 = wp::neg(var_189);
        var_191 = wp::vec_t<3, wp::float32>(var_184, var_187, var_190);
        var_192 = wp::mul(var_mat, var_191);
        var_193 = wp::add(var_192, var_pos);
        wp::array_store(var_face_out, var_194, var_193);
        // return 4                                                                           <L 1862>
        return var_195;
    }
    // if idx == 4:  # front                                                                  <L 1863>
    var_197 = (var_idx == var_196);
    if (var_197) {
        // face_out[0] = mat @ wp.vec3(-size[0], size[1], size[2]) + pos                      <L 1864>
        var_199 = wp::extract(var_size, var_198);
        var_200 = wp::neg(var_199);
        var_202 = wp::extract(var_size, var_201);
        var_204 = wp::extract(var_size, var_203);
        var_205 = wp::vec_t<3, wp::float32>(var_200, var_202, var_204);
        var_206 = wp::mul(var_mat, var_205);
        var_207 = wp::add(var_206, var_pos);
        wp::array_store(var_face_out, var_208, var_207);
        // face_out[1] = mat @ wp.vec3(size[0], size[1], size[2]) + pos                       <L 1865>
        var_210 = wp::extract(var_size, var_209);
        var_212 = wp::extract(var_size, var_211);
        var_214 = wp::extract(var_size, var_213);
        var_215 = wp::vec_t<3, wp::float32>(var_210, var_212, var_214);
        var_216 = wp::mul(var_mat, var_215);
        var_217 = wp::add(var_216, var_pos);
        wp::array_store(var_face_out, var_218, var_217);
        // face_out[2] = mat @ wp.vec3(size[0], -size[1], size[2]) + pos                      <L 1866>
        var_220 = wp::extract(var_size, var_219);
        var_222 = wp::extract(var_size, var_221);
        var_223 = wp::neg(var_222);
        var_225 = wp::extract(var_size, var_224);
        var_226 = wp::vec_t<3, wp::float32>(var_220, var_223, var_225);
        var_227 = wp::mul(var_mat, var_226);
        var_228 = wp::add(var_227, var_pos);
        wp::array_store(var_face_out, var_229, var_228);
        // face_out[3] = mat @ wp.vec3(-size[0], -size[1], size[2]) + pos                     <L 1867>
        var_231 = wp::extract(var_size, var_230);
        var_232 = wp::neg(var_231);
        var_234 = wp::extract(var_size, var_233);
        var_235 = wp::neg(var_234);
        var_237 = wp::extract(var_size, var_236);
        var_238 = wp::vec_t<3, wp::float32>(var_232, var_235, var_237);
        var_239 = wp::mul(var_mat, var_238);
        var_240 = wp::add(var_239, var_pos);
        wp::array_store(var_face_out, var_241, var_240);
        // return 4                                                                           <L 1868>
        return var_242;
    }
    // if idx == 5:  # back                                                                   <L 1869>
    var_244 = (var_idx == var_243);
    if (var_244) {
        // face_out[0] = mat @ wp.vec3(size[0], size[1], -size[2]) + pos                      <L 1870>
        var_246 = wp::extract(var_size, var_245);
        var_248 = wp::extract(var_size, var_247);
        var_250 = wp::extract(var_size, var_249);
        var_251 = wp::neg(var_250);
        var_252 = wp::vec_t<3, wp::float32>(var_246, var_248, var_251);
        var_253 = wp::mul(var_mat, var_252);
        var_254 = wp::add(var_253, var_pos);
        wp::array_store(var_face_out, var_255, var_254);
        // face_out[1] = mat @ wp.vec3(-size[0], size[1], -size[2]) + pos                     <L 1871>
        var_257 = wp::extract(var_size, var_256);
        var_258 = wp::neg(var_257);
        var_260 = wp::extract(var_size, var_259);
        var_262 = wp::extract(var_size, var_261);
        var_263 = wp::neg(var_262);
        var_264 = wp::vec_t<3, wp::float32>(var_258, var_260, var_263);
        var_265 = wp::mul(var_mat, var_264);
        var_266 = wp::add(var_265, var_pos);
        wp::array_store(var_face_out, var_267, var_266);
        // face_out[2] = mat @ wp.vec3(-size[0], -size[1], -size[2]) + pos                    <L 1872>
        var_269 = wp::extract(var_size, var_268);
        var_270 = wp::neg(var_269);
        var_272 = wp::extract(var_size, var_271);
        var_273 = wp::neg(var_272);
        var_275 = wp::extract(var_size, var_274);
        var_276 = wp::neg(var_275);
        var_277 = wp::vec_t<3, wp::float32>(var_270, var_273, var_276);
        var_278 = wp::mul(var_mat, var_277);
        var_279 = wp::add(var_278, var_pos);
        wp::array_store(var_face_out, var_280, var_279);
        // face_out[3] = mat @ wp.vec3(size[0], -size[1], -size[2]) + pos                     <L 1873>
        var_282 = wp::extract(var_size, var_281);
        var_284 = wp::extract(var_size, var_283);
        var_285 = wp::neg(var_284);
        var_287 = wp::extract(var_size, var_286);
        var_288 = wp::neg(var_287);
        var_289 = wp::vec_t<3, wp::float32>(var_282, var_285, var_288);
        var_290 = wp::mul(var_mat, var_289);
        var_291 = wp::add(var_290, var_pos);
        wp::array_store(var_face_out, var_292, var_291);
        // return 4                                                                           <L 1874>
        return var_293;
    }
    // return 0                                                                               <L 1875>
    return var_294;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1879
static CUDA_CALLABLE wp::int32 _mesh_face_0(
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_pos,
    wp::int32 var_vertadr,
    wp::int32 var_polyadr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vert,
    wp::array_t<wp::int32> var_polyvertadr,
    wp::array_t<wp::int32> var_polyvertnum,
    wp::array_t<wp::int32> var_polyvert,
    wp::int32 var_idx,
    wp::array_t<wp::vec_t<3, wp::float32>> var_face_out)
{
    //---------
    // primal vars
    wp::int32 var_0;
    wp::int32* var_1;
    wp::int32 var_2;
    wp::int32 var_3;
    const wp::int32 var_4 = 0;
    wp::int32 var_5;
    wp::int32 var_6;
    wp::int32* var_7;
    wp::int32 var_8;
    wp::int32 var_9;
    const wp::int32 var_10 = 1;
    wp::int32 var_11;
    const wp::int32 var_12 = -1;
    const wp::int32 var_13 = -1;
    wp::range_t var_14;
    wp::int32 var_15;
    wp::int32 var_16;
    wp::int32* var_17;
    wp::int32 var_18;
    wp::int32 var_19;
    wp::vec_t<3, wp::float32>* var_20;
    wp::vec_t<3, wp::float32> var_21;
    wp::vec_t<3, wp::float32> var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::vec_t<3, wp::float32> var_24;
    const wp::int32 var_25 = 1;
    wp::int32 var_26;
    //---------
    // forward
    // def _mesh_face(                                                                        <L 1880>
    // adr = polyvertadr[polyadr + idx]                                                       <L 1894>
    var_0 = wp::add(var_polyadr, var_idx);
    var_1 = wp::address(var_polyvertadr, var_0);
    var_3 = wp::load(var_1);
    var_2 = wp::copy(var_3);
    // j = int(0)                                                                             <L 1895>
    var_5 = wp::int(var_4);
    // nvert = polyvertnum[polyadr + idx]                                                     <L 1896>
    var_6 = wp::add(var_polyadr, var_idx);
    var_7 = wp::address(var_polyvertnum, var_6);
    var_9 = wp::load(var_7);
    var_8 = wp::copy(var_9);
    // for i in range(nvert - 1, -1, -1):                                                     <L 1897>
    var_11 = wp::sub(var_8, var_10);
    var_14 = wp::range(var_11, var_12, var_13);
    start_for_0:;
        if (iter_cmp(var_14) == 0) goto end_for_0;
        var_15 = wp::iter_next(var_14);
        // v = vert[vertadr + polyvert[adr + i]]                                              <L 1898>
        var_16 = wp::add(var_2, var_15);
        var_17 = wp::address(var_polyvert, var_16);
        var_19 = wp::load(var_17);
        var_18 = wp::add(var_vertadr, var_19);
        var_20 = wp::address(var_vert, var_18);
        var_22 = wp::load(var_20);
        var_21 = wp::copy(var_22);
        // face_out[j] = mat @ v + pos                                                        <L 1899>
        var_23 = wp::mul(var_mat, var_21);
        var_24 = wp::add(var_23, var_pos);
        wp::array_store(var_face_out, var_5, var_24);
        // j += 1                                                                             <L 1900>
        var_26 = wp::add(var_5, var_25);
        wp::assign(var_5, var_26);
        goto start_for_0;
    end_for_0:;
    // return nvert                                                                           <L 1901>
    return var_8;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1904
static CUDA_CALLABLE void _plane_normal_0(
    wp::vec_t<3, wp::float32> var_v1,
    wp::vec_t<3, wp::float32> var_v2,
    wp::vec_t<3, wp::float32> var_n,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::float32 var_4;
    //---------
    // forward
    // def _plane_normal(v1: wp.vec3, v2: wp.vec3, n: wp.vec3) -> Tuple[float, wp.vec3]:       <L 1905>
    // v3 = v1 + n                                                                            <L 1906>
    var_0 = wp::add(var_v1, var_n);
    // res = wp.cross(v2 - v1, v3 - v1)                                                       <L 1907>
    var_1 = wp::sub(var_v2, var_v1);
    var_2 = wp::sub(var_0, var_v1);
    var_3 = wp::cross(var_1, var_2);
    // return wp.dot(res, v1), res                                                            <L 1908>
    var_4 = wp::dot(var_3, var_v1);
    ret_0 = var_4;
    ret_1 = var_3;
    return;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1911
static CUDA_CALLABLE bool _halfspace_0(
    wp::vec_t<3, wp::float32> var_a,
    wp::vec_t<3, wp::float32> var_n,
    wp::vec_t<3, wp::float32> var_p)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::float32 var_1;
    const wp::float32 var_2 = -1e-10;
    bool var_3;
    //---------
    // forward
    // def _halfspace(a: wp.vec3, n: wp.vec3, p: wp.vec3) -> bool:                            <L 1912>
    // return wp.dot(p - a, n) > -1e-10                                                       <L 1913>
    var_0 = wp::sub(var_p, var_a);
    var_1 = wp::dot(var_0, var_n);
    var_3 = (var_1 > var_2);
    return var_3;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1916
static CUDA_CALLABLE wp::float32 _plane_intersect_0(
    wp::vec_t<3, wp::float32> var_pn,
    wp::float32 var_pd,
    wp::vec_t<3, wp::float32> var_a,
    wp::vec_t<3, wp::float32> var_b)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    const wp::float32 var_3 = 1e-10;
    bool var_4;
    const wp::float32 var_5 = 1e+30;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    //---------
    // forward
    // def _plane_intersect(pn: wp.vec3, pd: float, a: wp.vec3, b: wp.vec3) -> float:         <L 1917>
    // dot = wp.dot(pn, b - a)                                                                <L 1919>
    var_0 = wp::sub(var_b, var_a);
    var_1 = wp::dot(var_pn, var_0);
    // if wp.abs(dot) < 1e-10:                                                                <L 1922>
    var_2 = wp::abs(var_1);
    var_4 = (var_2 < var_3);
    if (var_4) {
        // return FLOAT_MAX                                                                   <L 1923>
        return var_5;
    }
    // return (pd - wp.dot(pn, a)) / dot                                                      <L 1925>
    var_6 = wp::dot(var_pn, var_a);
    var_7 = wp::sub(var_pd, var_6);
    var_8 = wp::div(var_7, var_1);
    return var_8;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1445
static CUDA_CALLABLE wp::float32 _area4_0(
    wp::vec_t<3, wp::float32> var_a,
    wp::vec_t<3, wp::float32> var_b,
    wp::vec_t<3, wp::float32> var_c,
    wp::vec_t<3, wp::float32> var_d)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.5;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    //---------
    // forward
    // def _area4(a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3) -> float:                   <L 1446>
    // return 0.5 * wp.norm_l2(wp.cross(a - d, d - b) + wp.cross(b - c, c - a))               <L 1448>
    var_1 = wp::sub(var_a, var_d);
    var_2 = wp::sub(var_d, var_b);
    var_3 = wp::cross(var_1, var_2);
    var_4 = wp::sub(var_b, var_c);
    var_5 = wp::sub(var_c, var_a);
    var_6 = wp::cross(var_4, var_5);
    var_7 = wp::add(var_3, var_6);
    var_8 = norm_l2_0(var_7);
    var_9 = wp::mul(var_0, var_8);
    return var_9;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1451
static CUDA_CALLABLE wp::vec_t<4, wp::int32> _polygon_quad_0(
    wp::array_t<wp::vec_t<3, wp::float32>> var_polygon,
    wp::int32 var_npolygon)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    wp::int32 var_1;
    const wp::int32 var_2 = 2;
    wp::int32 var_3;
    const wp::int32 var_4 = 3;
    wp::int32 var_5;
    const wp::int32 var_6 = 0;
    wp::vec_t<4, wp::int32> var_7;
    const wp::int32 var_8 = 0;
    wp::vec_t<3, wp::float32>* var_9;
    wp::vec_t<3, wp::float32>* var_10;
    wp::vec_t<3, wp::float32>* var_11;
    wp::vec_t<3, wp::float32>* var_12;
    wp::float32 var_13;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::vec_t<3, wp::float32> var_16;
    wp::vec_t<3, wp::float32> var_17;
    wp::range_t var_18;
    wp::int32 var_19;
    const bool var_20 = true;
    wp::vec_t<3, wp::float32>* var_21;
    wp::vec_t<3, wp::float32>* var_22;
    wp::vec_t<3, wp::float32>* var_23;
    const wp::int32 var_24 = 1;
    wp::int32 var_25;
    wp::int32 var_26;
    wp::vec_t<3, wp::float32>* var_27;
    wp::float32 var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::vec_t<3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<3, wp::float32> var_32;
    bool var_33;
    wp::float32 var_34;
    const wp::int32 var_35 = 1;
    wp::int32 var_36;
    wp::int32 var_37;
    wp::vec_t<4, wp::int32> var_38;
    const bool var_39 = true;
    wp::vec_t<3, wp::float32>* var_40;
    wp::vec_t<3, wp::float32>* var_41;
    const wp::int32 var_42 = 1;
    wp::int32 var_43;
    wp::int32 var_44;
    wp::vec_t<3, wp::float32>* var_45;
    wp::vec_t<3, wp::float32>* var_46;
    wp::float32 var_47;
    wp::vec_t<3, wp::float32> var_48;
    wp::vec_t<3, wp::float32> var_49;
    wp::vec_t<3, wp::float32> var_50;
    wp::vec_t<3, wp::float32> var_51;
    bool var_52;
    wp::float32 var_53;
    wp::float32 var_54;
    const wp::int32 var_55 = 1;
    wp::int32 var_56;
    wp::int32 var_57;
    wp::vec_t<4, wp::int32> var_58;
    const bool var_59 = true;
    wp::vec_t<3, wp::float32>* var_60;
    const wp::int32 var_61 = 1;
    wp::int32 var_62;
    wp::int32 var_63;
    wp::vec_t<3, wp::float32>* var_64;
    wp::vec_t<3, wp::float32>* var_65;
    wp::vec_t<3, wp::float32>* var_66;
    wp::float32 var_67;
    wp::vec_t<3, wp::float32> var_68;
    wp::vec_t<3, wp::float32> var_69;
    wp::vec_t<3, wp::float32> var_70;
    wp::vec_t<3, wp::float32> var_71;
    bool var_72;
    wp::float32 var_73;
    wp::float32 var_74;
    const wp::int32 var_75 = 1;
    wp::int32 var_76;
    wp::int32 var_77;
    wp::vec_t<4, wp::int32> var_78;
    bool var_79;
    const wp::int32 var_80 = 1;
    wp::int32 var_81;
    wp::int32 var_82;
    bool var_83;
    const wp::int32 var_84 = 1;
    wp::int32 var_85;
    wp::int32 var_86;
    bool var_87;
    const wp::int32 var_88 = 1;
    wp::int32 var_89;
    wp::int32 var_90;
    wp::int32 var_91;
    wp::int32 var_92;
    wp::int32 var_93;
    wp::int32 var_94;
    wp::int32 var_95;
    wp::int32 var_96;
    //---------
    // forward
    // def _polygon_quad(polygon: wp.array[wp.vec3], npolygon: int) -> wp.vec4i:              <L 1452>
    // b = int(1)                                                                             <L 1454>
    var_1 = wp::int(var_0);
    // c = int(2)                                                                             <L 1455>
    var_3 = wp::int(var_2);
    // d = int(3)                                                                             <L 1456>
    var_5 = wp::int(var_4);
    // res = wp.vec4i(0, b, c, d)                                                             <L 1457>
    var_7 = wp::vec_t<4, wp::int32>(var_6, var_1, var_3, var_5);
    // m = _area4(polygon[0], polygon[b], polygon[c], polygon[d])                             <L 1458>
    var_9 = wp::address(var_polygon, var_8);
    var_10 = wp::address(var_polygon, var_1);
    var_11 = wp::address(var_polygon, var_3);
    var_12 = wp::address(var_polygon, var_5);
    var_14 = wp::load(var_9);
    var_15 = wp::load(var_10);
    var_16 = wp::load(var_11);
    var_17 = wp::load(var_12);
    var_13 = _area4_0(var_14, var_15, var_16, var_17);
    // for a in range(npolygon):                                                              <L 1459>
    var_18 = wp::range(var_npolygon);
    start_for_0:;
        if (iter_cmp(var_18) == 0) goto end_for_0;
        var_19 = wp::iter_next(var_18);
        // while True:                                                                        <L 1460>
    start_while_2:;
    if ((var_20) == false) goto end_while_2;
            // m_next = _area4(polygon[a], polygon[b], polygon[c], polygon[(d + 1) % npolygon])       <L 1461>
            var_21 = wp::address(var_polygon, var_19);
            var_22 = wp::address(var_polygon, var_1);
            var_23 = wp::address(var_polygon, var_3);
            var_25 = wp::add(var_5, var_24);
            var_26 = wp::mod(var_25, var_npolygon);
            var_27 = wp::address(var_polygon, var_26);
            var_29 = wp::load(var_21);
            var_30 = wp::load(var_22);
            var_31 = wp::load(var_23);
            var_32 = wp::load(var_27);
            var_28 = _area4_0(var_29, var_30, var_31, var_32);
            // if m_next <= m:                                                                <L 1462>
            var_33 = (var_28 <= var_13);
            if (var_33) {
                // break                                                                      <L 1463>
                goto end_while_2;
            }
            // m = m_next                                                                     <L 1464>
            var_34 = wp::copy(var_28);
            // d = (d + 1) % npolygon                                                         <L 1465>
            var_36 = wp::add(var_5, var_35);
            var_37 = wp::mod(var_36, var_npolygon);
            // res = wp.vec4i(a, b, c, d)                                                     <L 1466>
            var_38 = wp::vec_t<4, wp::int32>(var_19, var_1, var_3, var_37);
            // while True:                                                                    <L 1467>
    start_while_4:;
    if ((var_39) == false) goto end_while_4;
                // m_next = _area4(polygon[a], polygon[b], polygon[(c + 1) % npolygon], polygon[d])       <L 1468>
                var_40 = wp::address(var_polygon, var_19);
                var_41 = wp::address(var_polygon, var_1);
                var_43 = wp::add(var_3, var_42);
                var_44 = wp::mod(var_43, var_npolygon);
                var_45 = wp::address(var_polygon, var_44);
                var_46 = wp::address(var_polygon, var_37);
                var_48 = wp::load(var_40);
                var_49 = wp::load(var_41);
                var_50 = wp::load(var_45);
                var_51 = wp::load(var_46);
                var_47 = _area4_0(var_48, var_49, var_50, var_51);
                // if m_next <= m:                                                            <L 1469>
                var_52 = (var_47 <= var_34);
                if (var_52) {
                    // break                                                                  <L 1470>
                    wp::assign(var_28, var_47);
                    goto end_while_4;
                }
                var_53 = wp::where(var_52, var_28, var_47);
                // m = m_next                                                                 <L 1471>
                var_54 = wp::copy(var_53);
                // c = (c + 1) % npolygon                                                     <L 1472>
                var_56 = wp::add(var_3, var_55);
                var_57 = wp::mod(var_56, var_npolygon);
                // res = wp.vec4i(a, b, c, d)                                                 <L 1473>
                var_58 = wp::vec_t<4, wp::int32>(var_19, var_1, var_57, var_37);
                wp::assign(var_3, var_57);
                wp::assign(var_38, var_58);
                wp::assign(var_34, var_54);
                wp::assign(var_28, var_53);
    goto start_while_4;
    end_while_4:;
            // while True:                                                                    <L 1474>
    start_while_6:;
    if ((var_59) == false) goto end_while_6;
                // m_next = _area4(polygon[a], polygon[(b + 1) % npolygon], polygon[c], polygon[d])       <L 1475>
                var_60 = wp::address(var_polygon, var_19);
                var_62 = wp::add(var_1, var_61);
                var_63 = wp::mod(var_62, var_npolygon);
                var_64 = wp::address(var_polygon, var_63);
                var_65 = wp::address(var_polygon, var_3);
                var_66 = wp::address(var_polygon, var_37);
                var_68 = wp::load(var_60);
                var_69 = wp::load(var_64);
                var_70 = wp::load(var_65);
                var_71 = wp::load(var_66);
                var_67 = _area4_0(var_68, var_69, var_70, var_71);
                // if m_next <= m:                                                            <L 1476>
                var_72 = (var_67 <= var_34);
                if (var_72) {
                    // break                                                                  <L 1477>
                    wp::assign(var_28, var_67);
                    goto end_while_6;
                }
                var_73 = wp::where(var_72, var_28, var_67);
                // m = m_next                                                                 <L 1478>
                var_74 = wp::copy(var_73);
                // b = (b + 1) % npolygon                                                     <L 1479>
                var_76 = wp::add(var_1, var_75);
                var_77 = wp::mod(var_76, var_npolygon);
                // res = wp.vec4i(a, b, c, d)                                                 <L 1480>
                var_78 = wp::vec_t<4, wp::int32>(var_19, var_77, var_3, var_37);
                wp::assign(var_1, var_77);
                wp::assign(var_38, var_78);
                wp::assign(var_34, var_74);
                wp::assign(var_28, var_73);
    goto start_while_6;
    end_while_6:;
            wp::assign(var_5, var_37);
            wp::assign(var_7, var_38);
            wp::assign(var_13, var_34);
    goto start_while_2;
    end_while_2:;
        // if b == a:                                                                         <L 1481>
        var_79 = (var_1 == var_19);
        if (var_79) {
            // b = (b + 1) % npolygon                                                         <L 1482>
            var_81 = wp::add(var_1, var_80);
            var_82 = wp::mod(var_81, var_npolygon);
            // if c == b:                                                                     <L 1483>
            var_83 = (var_3 == var_82);
            if (var_83) {
                // c = (c + 1) % npolygon                                                     <L 1484>
                var_85 = wp::add(var_3, var_84);
                var_86 = wp::mod(var_85, var_npolygon);
                // if d == c:                                                                 <L 1485>
                var_87 = (var_5 == var_86);
                if (var_87) {
                    // d = (d + 1) % npolygon                                                 <L 1486>
                    var_89 = wp::add(var_5, var_88);
                    var_90 = wp::mod(var_89, var_npolygon);
                }
                var_91 = wp::where(var_87, var_90, var_5);
            }
            var_92 = wp::where(var_83, var_86, var_3);
            var_93 = wp::where(var_83, var_91, var_5);
        }
        var_94 = wp::where(var_79, var_82, var_1);
        var_95 = wp::where(var_79, var_92, var_3);
        var_96 = wp::where(var_79, var_93, var_5);
        wp::assign(var_1, var_94);
        wp::assign(var_3, var_95);
        wp::assign(var_5, var_96);
        goto start_for_0;
    end_for_0:;
    // return res                                                                             <L 1487>
    return var_7;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1929
static CUDA_CALLABLE void _polygon_clip_0(
    wp::array_t<wp::vec_t<3, wp::float32>> var_plane_normal,
    wp::array_t<wp::float32> var_plane_dist,
    wp::array_t<wp::vec_t<3, wp::float32>> var_face1,
    wp::int32 var_nface1,
    wp::array_t<wp::vec_t<3, wp::float32>> var_face2,
    wp::int32 var_nface2,
    wp::vec_t<3, wp::float32> var_n,
    wp::vec_t<3, wp::float32> var_dir,
    wp::array_t<wp::vec_t<3, wp::float32>> var_polygon_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_clipped_out,
    wp::int32 & ret_0,
    wp::mat_t<4, 3, wp::float32> & ret_1,
    wp::mat_t<4, 3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::mat_t<4, 3, wp::float32> var_0;
    wp::mat_t<4, 3, wp::float32> var_1;
    const wp::int32 var_2 = 3;
    bool var_3;
    const wp::int32 var_4 = 0;
    wp::array_t<wp::vec_t<3, wp::float32>> var_5;
    wp::array_t<wp::float32> var_6;
    const wp::int32 var_7 = 1;
    wp::int32 var_8;
    wp::range_t var_9;
    wp::int32 var_10;
    wp::vec_t<3, wp::float32>* var_11;
    const wp::int32 var_12 = 1;
    wp::int32 var_13;
    wp::vec_t<3, wp::float32>* var_14;
    wp::float32 var_15;
    wp::vec_t<3, wp::float32> var_16;
    wp::vec_t<3, wp::float32> var_17;
    wp::vec_t<3, wp::float32> var_18;
    const wp::int32 var_19 = 1;
    wp::int32 var_20;
    wp::vec_t<3, wp::float32>* var_21;
    const wp::int32 var_22 = 0;
    wp::vec_t<3, wp::float32>* var_23;
    wp::float32 var_24;
    wp::vec_t<3, wp::float32> var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::vec_t<3, wp::float32> var_27;
    const wp::int32 var_28 = 1;
    wp::int32 var_29;
    const wp::int32 var_30 = 1;
    wp::int32 var_31;
    wp::int32 var_32;
    const wp::int32 var_33 = 0;
    wp::int32 var_34;
    wp::range_t var_35;
    wp::int32 var_36;
    wp::vec_t<3, wp::float32>* var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::range_t var_39;
    wp::int32 var_40;
    wp::range_t var_41;
    wp::int32 var_42;
    wp::vec_t<3, wp::float32>* var_43;
    wp::vec_t<3, wp::float32> var_44;
    wp::vec_t<3, wp::float32> var_45;
    const wp::int32 var_46 = 1;
    wp::int32 var_47;
    wp::int32 var_48;
    wp::vec_t<3, wp::float32>* var_49;
    wp::vec_t<3, wp::float32> var_50;
    wp::vec_t<3, wp::float32> var_51;
    wp::vec_t<3, wp::float32>* var_52;
    wp::vec_t<3, wp::float32>* var_53;
    bool var_54;
    wp::vec_t<3, wp::float32> var_55;
    wp::vec_t<3, wp::float32> var_56;
    wp::vec_t<3, wp::float32>* var_57;
    wp::vec_t<3, wp::float32>* var_58;
    bool var_59;
    wp::vec_t<3, wp::float32> var_60;
    wp::vec_t<3, wp::float32> var_61;
    bool var_62;
    bool var_63;
    bool var_64;
    bool var_65;
    const wp::int32 var_66 = 1;
    wp::int32 var_67;
    wp::vec_t<3, wp::float32>* var_68;
    wp::float32* var_69;
    wp::float32 var_70;
    wp::vec_t<3, wp::float32> var_71;
    wp::float32 var_72;
    bool var_73;
    const wp::float32 var_74 = 3e-07;
    const wp::float32 var_75 = -3e-07;
    bool var_76;
    const wp::float32 var_77 = 1.0;
    wp::float32 var_78;
    bool var_79;
    const wp::float32 var_80 = 0.0;
    const wp::float32 var_81 = 1.0;
    wp::float32 var_82;
    wp::vec_t<3, wp::float32> var_83;
    wp::vec_t<3, wp::float32> var_84;
    wp::vec_t<3, wp::float32> var_85;
    const wp::int32 var_86 = 1;
    wp::int32 var_87;
    wp::int32 var_88;
    wp::float32 var_89;
    const wp::int32 var_90 = 1;
    wp::int32 var_91;
    wp::int32 var_92;
    wp::array_t<wp::vec_t<3, wp::float32>> var_93;
    wp::array_t<wp::vec_t<3, wp::float32>> var_94;
    wp::array_t<wp::vec_t<3, wp::float32>> var_95;
    wp::int32 var_96;
    const wp::int32 var_97 = 0;
    const wp::int32 var_98 = 1;
    bool var_99;
    const wp::int32 var_100 = 0;
    bool var_101;
    const wp::int32 var_102 = 2;
    bool var_103;
    const wp::int32 var_104 = 2;
    bool var_105;
    const wp::int32 var_106 = 0;
    wp::int32 var_107;
    const wp::int32 var_108 = 1;
    wp::int32 var_109;
    const wp::float32 var_110 = 0.0;
    wp::float32 var_111;
    wp::range_t var_112;
    wp::int32 var_113;
    wp::vec_t<3, wp::float32>* var_114;
    wp::vec_t<3, wp::float32> var_115;
    wp::vec_t<3, wp::float32> var_116;
    const wp::int32 var_117 = 1;
    wp::int32 var_118;
    wp::range_t var_119;
    wp::int32 var_120;
    wp::vec_t<3, wp::float32>* var_121;
    wp::vec_t<3, wp::float32> var_122;
    wp::vec_t<3, wp::float32> var_123;
    wp::float32 var_124;
    bool var_125;
    wp::float32 var_126;
    wp::int32 var_127;
    wp::int32 var_128;
    wp::int32 var_129;
    wp::int32 var_130;
    wp::float32 var_131;
    wp::vec_t<3, wp::float32>* var_132;
    const wp::int32 var_133 = 0;
    wp::vec_t<3, wp::float32> var_134;
    const wp::int32 var_135 = 0;
    wp::vec_t<3, wp::float32> var_136;
    wp::vec_t<3, wp::float32> var_137;
    const wp::int32 var_138 = 0;
    wp::vec_t<3, wp::float32>* var_139;
    const wp::int32 var_140 = 1;
    wp::vec_t<3, wp::float32> var_141;
    const wp::int32 var_142 = 1;
    wp::vec_t<3, wp::float32> var_143;
    wp::vec_t<3, wp::float32> var_144;
    const wp::int32 var_145 = 1;
    const wp::int32 var_146 = 2;
    wp::int32 var_147;
    const wp::int32 var_148 = 4;
    bool var_149;
    wp::vec_t<4, wp::int32> var_150;
    const wp::int32 var_151 = 0;
    wp::int32 var_152;
    wp::vec_t<3, wp::float32>* var_153;
    wp::vec_t<3, wp::float32> var_154;
    wp::vec_t<3, wp::float32> var_155;
    wp::vec_t<3, wp::float32> var_156;
    const wp::int32 var_157 = 1;
    wp::int32 var_158;
    wp::vec_t<3, wp::float32>* var_159;
    wp::vec_t<3, wp::float32> var_160;
    wp::vec_t<3, wp::float32> var_161;
    wp::vec_t<3, wp::float32> var_162;
    const wp::int32 var_163 = 2;
    wp::int32 var_164;
    wp::vec_t<3, wp::float32>* var_165;
    wp::vec_t<3, wp::float32> var_166;
    wp::vec_t<3, wp::float32> var_167;
    wp::vec_t<3, wp::float32> var_168;
    const wp::int32 var_169 = 3;
    wp::int32 var_170;
    wp::vec_t<3, wp::float32>* var_171;
    wp::vec_t<3, wp::float32> var_172;
    wp::vec_t<3, wp::float32> var_173;
    wp::vec_t<3, wp::float32> var_174;
    const wp::int32 var_175 = 4;
    wp::int32 var_176;
    wp::range_t var_177;
    wp::int32 var_178;
    wp::vec_t<3, wp::float32>* var_179;
    wp::vec_t<3, wp::float32> var_180;
    wp::vec_t<3, wp::float32> var_181;
    wp::vec_t<3, wp::float32> var_182;
    //---------
    // forward
    // def _polygon_clip(                                                                     <L 1930>
    // witness1 = mat43()                                                                     <L 1944>
    var_0 = wp::mat_t<4, 3, wp::float32>();
    // witness2 = mat43()                                                                     <L 1945>
    var_1 = wp::mat_t<4, 3, wp::float32>();
    // if nface1 < 3:                                                                         <L 1948>
    var_3 = (var_nface1 < var_2);
    if (var_3) {
        // return 0, witness1, witness2                                                       <L 1949>
        ret_0 = var_4;
        ret_1 = var_0;
        ret_2 = var_1;
        return;
    }
    // pn = plane_normal                                                                      <L 1952>
    var_5 = wp::copy(var_plane_normal);
    // pd = plane_dist                                                                        <L 1953>
    var_6 = wp::copy(var_plane_dist);
    // for i in range(nface1 - 1):                                                            <L 1954>
    var_8 = wp::sub(var_nface1, var_7);
    var_9 = wp::range(var_8);
    start_for_1:;
        if (iter_cmp(var_9) == 0) goto end_for_1;
        var_10 = wp::iter_next(var_9);
        // pdi, pni = _plane_normal(face1[i], face1[i + 1], n)                                <L 1955>
        var_11 = wp::address(var_face1, var_10);
        var_13 = wp::add(var_10, var_12);
        var_14 = wp::address(var_face1, var_13);
        var_17 = wp::load(var_11);
        var_18 = wp::load(var_14);
        _plane_normal_0(var_17, var_18, var_n, var_15, var_16);
        // pd[i] = pdi                                                                        <L 1956>
        wp::array_store(var_6, var_10, var_15);
        // pn[i] = pni                                                                        <L 1957>
        wp::array_store(var_5, var_10, var_16);
        goto start_for_1;
    end_for_1:;
    // pdi, pni = _plane_normal(face1[nface1 - 1], face1[0], n)                               <L 1958>
    var_20 = wp::sub(var_nface1, var_19);
    var_21 = wp::address(var_face1, var_20);
    var_23 = wp::address(var_face1, var_22);
    var_26 = wp::load(var_21);
    var_27 = wp::load(var_23);
    _plane_normal_0(var_26, var_27, var_n, var_24, var_25);
    // pd[nface1 - 1] = pdi                                                                   <L 1959>
    var_29 = wp::sub(var_nface1, var_28);
    wp::array_store(var_6, var_29, var_24);
    // pn[nface1 - 1] = pni                                                                   <L 1960>
    var_31 = wp::sub(var_nface1, var_30);
    wp::array_store(var_5, var_31, var_25);
    // npolygon = nface2                                                                      <L 1963>
    var_32 = wp::copy(var_nface2);
    // nclipped = int(0)                                                                      <L 1964>
    var_34 = wp::int(var_33);
    // for i in range(nface2):                                                                <L 1966>
    var_35 = wp::range(var_nface2);
    start_for_3:;
        if (iter_cmp(var_35) == 0) goto end_for_3;
        var_36 = wp::iter_next(var_35);
        // polygon_out[i] = face2[i]                                                          <L 1967>
        var_37 = wp::address(var_face2, var_36);
        var_38 = wp::load(var_37);
        wp::array_store(var_polygon_out, var_36, var_38);
        goto start_for_3;
    end_for_3:;
    // for e in range(nface1):                                                                <L 1970>
    var_39 = wp::range(var_nface1);
    start_for_5:;
        if (iter_cmp(var_39) == 0) goto end_for_5;
        var_40 = wp::iter_next(var_39);
        // for i in range(npolygon):                                                          <L 1971>
        var_41 = wp::range(var_32);
        start_for_7:;
            if (iter_cmp(var_41) == 0) goto end_for_7;
            var_42 = wp::iter_next(var_41);
            // P = polygon_out[i]                                                             <L 1973>
            var_43 = wp::address(var_polygon_out, var_42);
            var_45 = wp::load(var_43);
            var_44 = wp::copy(var_45);
            // Q = polygon_out[(i + 1) % npolygon]                                            <L 1974>
            var_47 = wp::add(var_42, var_46);
            var_48 = wp::mod(var_47, var_32);
            var_49 = wp::address(var_polygon_out, var_48);
            var_51 = wp::load(var_49);
            var_50 = wp::copy(var_51);
            // inside1 = _halfspace(face1[e], pn[e], P)                                       <L 1977>
            var_52 = wp::address(var_face1, var_40);
            var_53 = wp::address(var_5, var_40);
            var_55 = wp::load(var_52);
            var_56 = wp::load(var_53);
            var_54 = _halfspace_0(var_55, var_56, var_44);
            // inside2 = _halfspace(face1[e], pn[e], Q)                                       <L 1978>
            var_57 = wp::address(var_face1, var_40);
            var_58 = wp::address(var_5, var_40);
            var_60 = wp::load(var_57);
            var_61 = wp::load(var_58);
            var_59 = _halfspace_0(var_60, var_61, var_50);
            // if not inside1 and not inside2:                                                <L 1981>
            var_63 = wp::unot(var_54);
            var_62 = var_63;
            if (var_62) {
                var_64 = wp::unot(var_59);
                var_62 = var_62 && var_64;
            }
            if (var_62) {
                // continue                                                                   <L 1982>
                goto start_for_7;
            }
            // if inside1 and inside2:                                                        <L 1985>
            var_65 = var_54;
            if (var_65) {
                var_65 = var_65 && var_59;
            }
            if (var_65) {
                // clipped_out[nclipped] = Q                                                  <L 1986>
                wp::array_store(var_clipped_out, var_34, var_50);
                // nclipped += 1                                                              <L 1987>
                var_67 = wp::add(var_34, var_66);
                // continue                                                                   <L 1988>
                wp::assign(var_34, var_67);
                goto start_for_7;
            }
            // t = _plane_intersect(pn[e], pd[e], P, Q)                                       <L 1991>
            var_68 = wp::address(var_5, var_40);
            var_69 = wp::address(var_6, var_40);
            var_71 = wp::load(var_68);
            var_72 = wp::load(var_69);
            var_70 = _plane_intersect_0(var_71, var_72, var_44, var_50);
            // if t > -INTERSECT_TOL and t < 1.0 + INTERSECT_TOL:                             <L 1992>
            var_76 = (var_70 > var_75);
            var_73 = var_76;
            if (var_73) {
                var_78 = wp::add(var_77, var_74);
                var_79 = (var_70 < var_78);
                var_73 = var_73 && var_79;
            }
            if (var_73) {
                // t = wp.clamp(t, 0.0, 1.0)                                                  <L 1993>
                var_82 = wp::clamp(var_70, var_80, var_81);
                // clipped_out[nclipped] = P + t * (Q - P)                                    <L 1994>
                var_83 = wp::sub(var_50, var_44);
                var_84 = wp::mul(var_82, var_83);
                var_85 = wp::add(var_44, var_84);
                wp::array_store(var_clipped_out, var_34, var_85);
                // nclipped += 1                                                              <L 1995>
                var_87 = wp::add(var_34, var_86);
            }
            var_88 = wp::where(var_73, var_87, var_34);
            var_89 = wp::where(var_73, var_82, var_70);
            // if inside2:                                                                    <L 1998>
            if (var_59) {
                // clipped_out[nclipped] = Q                                                  <L 1999>
                wp::array_store(var_clipped_out, var_88, var_50);
                // nclipped += 1                                                              <L 2000>
                var_91 = wp::add(var_88, var_90);
            }
            var_92 = wp::where(var_59, var_91, var_88);
            wp::assign(var_34, var_92);
            goto start_for_7;
        end_for_7:;
        // tmp = polygon_out                                                                  <L 2003>
        var_93 = wp::copy(var_polygon_out);
        // polygon_out = clipped_out                                                          <L 2004>
        var_94 = wp::copy(var_clipped_out);
        // clipped_out = tmp                                                                  <L 2005>
        var_95 = wp::copy(var_93);
        // npolygon = nclipped                                                                <L 2006>
        var_96 = wp::copy(var_34);
        // nclipped = 0                                                                       <L 2007>
        wp::assign(var_polygon_out, var_94);
        wp::assign(var_clipped_out, var_95);
        wp::assign(var_36, var_42);
        wp::assign(var_32, var_96);
        wp::assign(var_34, var_97);
        goto start_for_5;
    end_for_5:;
    // if npolygon < 1:                                                                       <L 2009>
    var_99 = (var_32 < var_98);
    if (var_99) {
        // return 0, witness1, witness2                                                       <L 2010>
        ret_0 = var_100;
        ret_1 = var_0;
        ret_2 = var_1;
        return;
    }
    // if nface2 == 2 and npolygon > 2:                                                       <L 2013>
    var_103 = (var_nface2 == var_102);
    var_101 = var_103;
    if (var_101) {
        var_105 = (var_32 > var_104);
        var_101 = var_101 && var_105;
    }
    if (var_101) {
        // best1 = int(0)                                                                     <L 2014>
        var_107 = wp::int(var_106);
        // best2 = int(1)                                                                     <L 2015>
        var_109 = wp::int(var_108);
        // max_d = float(0.0)                                                                 <L 2016>
        var_111 = wp::float(var_110);
        // for i in range(npolygon):                                                          <L 2017>
        var_112 = wp::range(var_32);
        start_for_10:;
            if (iter_cmp(var_112) == 0) goto end_for_10;
            var_113 = wp::iter_next(var_112);
            // polygon_out_i = polygon_out[i]                                                 <L 2018>
            var_114 = wp::address(var_polygon_out, var_113);
            var_116 = wp::load(var_114);
            var_115 = wp::copy(var_116);
            // for j in range(i + 1, npolygon):                                               <L 2019>
            var_118 = wp::add(var_113, var_117);
            var_119 = wp::range(var_118, var_32);
            start_for_12:;
                if (iter_cmp(var_119) == 0) goto end_for_12;
                var_120 = wp::iter_next(var_119);
                // diff = polygon_out[j] - polygon_out_i                                      <L 2020>
                var_121 = wp::address(var_polygon_out, var_120);
                var_123 = wp::load(var_121);
                var_122 = wp::sub(var_123, var_115);
                // d2 = wp.dot(diff, diff)                                                    <L 2021>
                var_124 = wp::dot(var_122, var_122);
                // if d2 > max_d:                                                             <L 2022>
                var_125 = (var_124 > var_111);
                if (var_125) {
                    // max_d = d2                                                             <L 2023>
                    var_126 = wp::copy(var_124);
                    // best1 = i                                                              <L 2024>
                    var_127 = wp::copy(var_113);
                    // best2 = j                                                              <L 2025>
                    var_128 = wp::copy(var_120);
                }
                var_129 = wp::where(var_125, var_127, var_107);
                var_130 = wp::where(var_125, var_128, var_109);
                var_131 = wp::where(var_125, var_126, var_111);
                wp::assign(var_107, var_129);
                wp::assign(var_109, var_130);
                wp::assign(var_111, var_131);
                goto start_for_12;
            end_for_12:;
            goto start_for_10;
        end_for_10:;
        // witness2[0] = polygon_out[best1]                                                   <L 2027>
        var_132 = wp::address(var_polygon_out, var_107);
        var_134 = wp::load(var_132);
        wp::assign_inplace(var_1, var_133, var_134);
        // witness1[0] = witness2[0] - dir                                                    <L 2028>
        var_136 = wp::extract(var_1, var_135);
        var_137 = wp::sub(var_136, var_dir);
        wp::assign_inplace(var_0, var_138, var_137);
        // witness2[1] = polygon_out[best2]                                                   <L 2029>
        var_139 = wp::address(var_polygon_out, var_109);
        var_141 = wp::load(var_139);
        wp::assign_inplace(var_1, var_140, var_141);
        // witness1[1] = witness2[1] - dir                                                    <L 2030>
        var_143 = wp::extract(var_1, var_142);
        var_144 = wp::sub(var_143, var_dir);
        wp::assign_inplace(var_0, var_145, var_144);
        // return 2, witness1, witness2                                                       <L 2031>
        ret_0 = var_146;
        ret_1 = var_0;
        ret_2 = var_1;
        return;
    }
    var_147 = wp::where(var_101, var_113, var_36);
    // if npolygon > 4:                                                                       <L 2033>
    var_149 = (var_32 > var_148);
    if (var_149) {
        // quad = _polygon_quad(polygon_out, npolygon)                                        <L 2034>
        var_150 = _polygon_quad_0(var_polygon_out, var_32);
        // for i in range(4):                                                                 <L 2035>
        // witness2[i] = polygon_out[quad[i]]                                                 <L 2036>
        var_152 = wp::extract(var_150, var_151);
        var_153 = wp::address(var_polygon_out, var_152);
        var_154 = wp::load(var_153);
        wp::assign_inplace(var_1, var_151, var_154);
        // witness1[i] = witness2[i] - dir                                                    <L 2037>
        var_155 = wp::extract(var_1, var_151);
        var_156 = wp::sub(var_155, var_dir);
        wp::assign_inplace(var_0, var_151, var_156);
        // witness2[i] = polygon_out[quad[i]]                                                 <L 2036>
        var_158 = wp::extract(var_150, var_157);
        var_159 = wp::address(var_polygon_out, var_158);
        var_160 = wp::load(var_159);
        wp::assign_inplace(var_1, var_157, var_160);
        // witness1[i] = witness2[i] - dir                                                    <L 2037>
        var_161 = wp::extract(var_1, var_157);
        var_162 = wp::sub(var_161, var_dir);
        wp::assign_inplace(var_0, var_157, var_162);
        // witness2[i] = polygon_out[quad[i]]                                                 <L 2036>
        var_164 = wp::extract(var_150, var_163);
        var_165 = wp::address(var_polygon_out, var_164);
        var_166 = wp::load(var_165);
        wp::assign_inplace(var_1, var_163, var_166);
        // witness1[i] = witness2[i] - dir                                                    <L 2037>
        var_167 = wp::extract(var_1, var_163);
        var_168 = wp::sub(var_167, var_dir);
        wp::assign_inplace(var_0, var_163, var_168);
        // witness2[i] = polygon_out[quad[i]]                                                 <L 2036>
        var_170 = wp::extract(var_150, var_169);
        var_171 = wp::address(var_polygon_out, var_170);
        var_172 = wp::load(var_171);
        wp::assign_inplace(var_1, var_169, var_172);
        // witness1[i] = witness2[i] - dir                                                    <L 2037>
        var_173 = wp::extract(var_1, var_169);
        var_174 = wp::sub(var_173, var_dir);
        wp::assign_inplace(var_0, var_169, var_174);
        // return 4, witness1, witness2                                                       <L 2038>
        ret_0 = var_175;
        ret_1 = var_0;
        ret_2 = var_1;
        return;
    }
    var_176 = wp::where(var_149, var_169, var_147);
    // for i in range(npolygon):                                                              <L 2041>
    var_177 = wp::range(var_32);
    start_for_16:;
        if (iter_cmp(var_177) == 0) goto end_for_16;
        var_178 = wp::iter_next(var_177);
        // witness2[i] = polygon_out[i]                                                       <L 2042>
        var_179 = wp::address(var_polygon_out, var_178);
        var_180 = wp::load(var_179);
        wp::assign_inplace(var_1, var_178, var_180);
        // witness1[i] = witness2[i] - dir                                                    <L 2043>
        var_181 = wp::extract(var_1, var_178);
        var_182 = wp::sub(var_181, var_dir);
        wp::assign_inplace(var_0, var_178, var_182);
        goto start_for_16;
    end_for_16:;
    // return npolygon, witness1, witness2                                                    <L 2044>
    ret_0 = var_32;
    ret_1 = var_0;
    ret_2 = var_1;
    return;
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:2064
static CUDA_CALLABLE void multicontact_0(
    wp::array_t<wp::vec_t<3, wp::float32>> var_polygon,
    wp::array_t<wp::vec_t<3, wp::float32>> var_clipped,
    wp::array_t<wp::vec_t<3, wp::float32>> var_plane_normal,
    wp::array_t<wp::float32> var_plane_dist,
    wp::array_t<wp::int32> var_idx1,
    wp::array_t<wp::int32> var_idx2,
    wp::array_t<wp::vec_t<3, wp::float32>> var_n1,
    wp::array_t<wp::vec_t<3, wp::float32>> var_n2,
    wp::array_t<wp::vec_t<3, wp::float32>> var_endvert,
    wp::array_t<wp::vec_t<3, wp::float32>> var_face1,
    wp::array_t<wp::vec_t<3, wp::float32>> var_face2,
    wp::array_t<wp::vec_t<3, wp::float32>> var_epa_vert,
    wp::array_t<wp::int32> var_epa_vert_index,
    wp::int32 var_epa_face,
    wp::vec_t<3, wp::float32> var_x1,
    wp::vec_t<3, wp::float32> var_x2,
    Geom_3242f8a8 var_geom1,
    Geom_3242f8a8 var_geom2,
    wp::int32 var_geomtype1,
    wp::int32 var_geomtype2,
    wp::int32 & ret_0,
    wp::mat_t<4, 3, wp::float32> & ret_1,
    wp::mat_t<4, 3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::mat_t<4, 3, wp::float32> var_0;
    wp::mat_t<4, 3, wp::float32> var_1;
    const wp::int32 var_2 = 0;
    const wp::int32 var_3 = 0;
    const wp::int32 var_4 = 7;
    bool var_5;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_6;
    wp::array_t<wp::vec_t<3, wp::float32>> var_7;
    wp::array_t<wp::vec_t<3, wp::float32>> var_8;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_9;
    wp::array_t<wp::vec_t<3, wp::float32>> var_10;
    wp::array_t<wp::vec_t<3, wp::float32>> var_11;
    wp::array_t<wp::int32>* var_12;
    wp::array_t<wp::int32> var_13;
    wp::array_t<wp::int32> var_14;
    wp::array_t<wp::int32>* var_15;
    wp::array_t<wp::int32> var_16;
    wp::array_t<wp::int32> var_17;
    wp::array_t<wp::int32>* var_18;
    wp::array_t<wp::int32> var_19;
    wp::array_t<wp::int32> var_20;
    wp::array_t<wp::int32>* var_21;
    wp::array_t<wp::int32> var_22;
    wp::array_t<wp::int32> var_23;
    wp::array_t<wp::int32>* var_24;
    wp::array_t<wp::int32> var_25;
    wp::array_t<wp::int32> var_26;
    wp::array_t<wp::int32>* var_27;
    wp::array_t<wp::int32> var_28;
    wp::array_t<wp::int32> var_29;
    const wp::int32 var_30 = 7;
    bool var_31;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_32;
    wp::array_t<wp::vec_t<3, wp::float32>> var_33;
    wp::array_t<wp::vec_t<3, wp::float32>> var_34;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_35;
    wp::array_t<wp::vec_t<3, wp::float32>> var_36;
    wp::array_t<wp::vec_t<3, wp::float32>> var_37;
    wp::array_t<wp::int32>* var_38;
    wp::array_t<wp::int32> var_39;
    wp::array_t<wp::int32> var_40;
    wp::array_t<wp::int32>* var_41;
    wp::array_t<wp::int32> var_42;
    wp::array_t<wp::int32> var_43;
    wp::array_t<wp::int32>* var_44;
    wp::array_t<wp::int32> var_45;
    wp::array_t<wp::int32> var_46;
    wp::array_t<wp::int32>* var_47;
    wp::array_t<wp::int32> var_48;
    wp::array_t<wp::int32> var_49;
    wp::array_t<wp::int32>* var_50;
    wp::array_t<wp::int32> var_51;
    wp::array_t<wp::int32> var_52;
    wp::array_t<wp::int32>* var_53;
    wp::array_t<wp::int32> var_54;
    wp::array_t<wp::int32> var_55;
    wp::array_t<wp::vec_t<3, wp::float32>> var_56;
    wp::array_t<wp::vec_t<3, wp::float32>> var_57;
    wp::array_t<wp::int32> var_58;
    wp::array_t<wp::int32> var_59;
    wp::array_t<wp::int32> var_60;
    wp::array_t<wp::int32> var_61;
    wp::array_t<wp::int32> var_62;
    wp::array_t<wp::int32> var_63;
    wp::array_t<wp::vec_t<3, wp::float32>> var_64;
    wp::array_t<wp::vec_t<3, wp::float32>> var_65;
    wp::array_t<wp::int32> var_66;
    wp::array_t<wp::int32> var_67;
    wp::array_t<wp::int32> var_68;
    wp::array_t<wp::int32> var_69;
    wp::array_t<wp::int32> var_70;
    wp::array_t<wp::int32> var_71;
    wp::vec_t<3, wp::int32> var_72;
    const wp::int32 var_73 = 0;
    wp::int32 var_74;
    wp::vec_t<3, wp::int32> var_75;
    wp::mat_t<3, 3, wp::float32> var_76;
    const wp::int32 var_77 = 1;
    wp::int32 var_78;
    wp::vec_t<3, wp::int32> var_79;
    wp::mat_t<3, 3, wp::float32> var_80;
    wp::vec_t<3, wp::float32> var_81;
    wp::vec_t<3, wp::float32> var_82;
    const wp::int32 var_83 = 6;
    bool var_84;
    wp::mat_t<3, 3, wp::float32>* var_85;
    wp::int32 var_86;
    wp::mat_t<3, 3, wp::float32> var_87;
    const wp::int32 var_88 = 7;
    bool var_89;
    wp::mat_t<3, 3, wp::float32>* var_90;
    wp::int32* var_91;
    wp::int32* var_92;
    wp::int32 var_93;
    wp::mat_t<3, 3, wp::float32> var_94;
    wp::int32 var_95;
    wp::int32 var_96;
    wp::int32 var_97;
    wp::int32 var_98;
    const wp::int32 var_99 = 6;
    bool var_100;
    wp::mat_t<3, 3, wp::float32>* var_101;
    wp::int32 var_102;
    wp::mat_t<3, 3, wp::float32> var_103;
    const wp::int32 var_104 = 7;
    bool var_105;
    wp::mat_t<3, 3, wp::float32>* var_106;
    wp::int32* var_107;
    wp::int32* var_108;
    wp::int32 var_109;
    wp::mat_t<3, 3, wp::float32> var_110;
    wp::int32 var_111;
    wp::int32 var_112;
    wp::int32 var_113;
    wp::int32 var_114;
    const wp::int32 var_115 = 0;
    const wp::int32 var_116 = 0;
    wp::int32 var_117;
    wp::vec_t<2, wp::int32> var_118;
    bool var_119;
    bool var_120;
    const wp::int32 var_121 = 3;
    bool var_122;
    bool var_123;
    const wp::int32 var_124 = 0;
    const wp::int32 var_125 = 6;
    bool var_126;
    wp::mat_t<3, 3, wp::float32>* var_127;
    wp::vec_t<3, wp::float32>* var_128;
    wp::vec_t<3, wp::float32>* var_129;
    const wp::int32 var_130 = 0;
    wp::vec_t<3, wp::float32> var_131;
    const wp::int32 var_132 = 1;
    wp::vec_t<3, wp::float32> var_133;
    const wp::int32 var_134 = 0;
    wp::int32 var_135;
    wp::int32 var_136;
    wp::mat_t<3, 3, wp::float32> var_137;
    wp::vec_t<3, wp::float32> var_138;
    wp::vec_t<3, wp::float32> var_139;
    wp::int32 var_140;
    const wp::int32 var_141 = 7;
    bool var_142;
    wp::mat_t<3, 3, wp::float32>* var_143;
    wp::vec_t<3, wp::float32>* var_144;
    wp::int32* var_145;
    wp::int32* var_146;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_147;
    const wp::int32 var_148 = 0;
    wp::vec_t<3, wp::float32> var_149;
    const wp::int32 var_150 = 1;
    wp::vec_t<3, wp::float32> var_151;
    const wp::int32 var_152 = 0;
    wp::int32 var_153;
    wp::int32 var_154;
    wp::mat_t<3, 3, wp::float32> var_155;
    wp::vec_t<3, wp::float32> var_156;
    wp::int32 var_157;
    wp::int32 var_158;
    wp::array_t<wp::vec_t<3, wp::float32>> var_159;
    wp::int32 var_160;
    wp::int32 var_161;
    wp::int32 var_162;
    wp::vec_t<2, wp::int32> var_163;
    bool var_164;
    const wp::int32 var_165 = 1;
    const wp::int32 var_166 = 1;
    wp::int32 var_167;
    wp::int32 var_168;
    wp::int32 var_169;
    wp::vec_t<2, wp::int32> var_170;
    const wp::int32 var_171 = 3;
    bool var_172;
    const wp::int32 var_173 = 0;
    const wp::int32 var_174 = 6;
    bool var_175;
    wp::mat_t<3, 3, wp::float32>* var_176;
    wp::vec_t<3, wp::float32>* var_177;
    wp::vec_t<3, wp::float32>* var_178;
    const wp::int32 var_179 = 0;
    wp::vec_t<3, wp::float32> var_180;
    const wp::int32 var_181 = 1;
    wp::vec_t<3, wp::float32> var_182;
    const wp::int32 var_183 = 0;
    wp::int32 var_184;
    wp::int32 var_185;
    wp::mat_t<3, 3, wp::float32> var_186;
    wp::vec_t<3, wp::float32> var_187;
    wp::vec_t<3, wp::float32> var_188;
    wp::int32 var_189;
    const wp::int32 var_190 = 7;
    bool var_191;
    wp::mat_t<3, 3, wp::float32>* var_192;
    wp::vec_t<3, wp::float32>* var_193;
    wp::int32* var_194;
    wp::int32* var_195;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_196;
    const wp::int32 var_197 = 0;
    wp::vec_t<3, wp::float32> var_198;
    const wp::int32 var_199 = 1;
    wp::vec_t<3, wp::float32> var_200;
    const wp::int32 var_201 = 0;
    wp::int32 var_202;
    wp::int32 var_203;
    wp::mat_t<3, 3, wp::float32> var_204;
    wp::vec_t<3, wp::float32> var_205;
    wp::int32 var_206;
    wp::int32 var_207;
    wp::array_t<wp::vec_t<3, wp::float32>> var_208;
    wp::int32 var_209;
    wp::int32 var_210;
    wp::int32 var_211;
    wp::vec_t<2, wp::int32> var_212;
    bool var_213;
    const wp::int32 var_214 = 1;
    const wp::int32 var_215 = 1;
    wp::int32 var_216;
    wp::int32 var_217;
    wp::int32 var_218;
    wp::vec_t<2, wp::int32> var_219;
    const wp::int32 var_220 = 1;
    wp::int32 var_221;
    wp::int32 var_222;
    wp::int32 var_223;
    wp::vec_t<2, wp::int32> var_224;
    wp::int32 var_225;
    wp::int32 var_226;
    wp::int32 var_227;
    wp::int32 var_228;
    wp::int32 var_229;
    wp::vec_t<2, wp::int32> var_230;
    const wp::int32 var_231 = 0;
    wp::int32 var_232;
    const wp::int32 var_233 = 1;
    wp::int32 var_234;
    const wp::int32 var_235 = 0;
    wp::int32 var_236;
    const wp::int32 var_237 = 0;
    wp::int32 var_238;
    wp::int32 var_239;
    wp::int32* var_240;
    wp::int32* var_241;
    wp::int32 var_242;
    wp::int32 var_243;
    wp::int32 var_244;
    const wp::int32 var_245 = 6;
    bool var_246;
    wp::mat_t<3, 3, wp::float32>* var_247;
    wp::vec_t<3, wp::float32>* var_248;
    wp::vec_t<3, wp::float32>* var_249;
    wp::int32 var_250;
    wp::mat_t<3, 3, wp::float32> var_251;
    wp::vec_t<3, wp::float32> var_252;
    wp::vec_t<3, wp::float32> var_253;
    wp::int32 var_254;
    const wp::int32 var_255 = 7;
    bool var_256;
    wp::mat_t<3, 3, wp::float32>* var_257;
    wp::vec_t<3, wp::float32>* var_258;
    wp::int32* var_259;
    wp::int32* var_260;
    wp::int32 var_261;
    wp::mat_t<3, 3, wp::float32> var_262;
    wp::vec_t<3, wp::float32> var_263;
    wp::int32 var_264;
    wp::int32 var_265;
    wp::int32 var_266;
    wp::int32 var_267;
    wp::int32 var_268;
    const wp::int32 var_269 = 0;
    wp::int32 var_270;
    const wp::int32 var_271 = 1;
    wp::int32 var_272;
    wp::int32 var_273;
    const wp::int32 var_274 = 6;
    bool var_275;
    wp::mat_t<3, 3, wp::float32>* var_276;
    wp::vec_t<3, wp::float32>* var_277;
    wp::vec_t<3, wp::float32>* var_278;
    wp::int32* var_279;
    wp::int32 var_280;
    wp::mat_t<3, 3, wp::float32> var_281;
    wp::vec_t<3, wp::float32> var_282;
    wp::vec_t<3, wp::float32> var_283;
    wp::int32 var_284;
    wp::int32 var_285;
    const wp::int32 var_286 = 7;
    bool var_287;
    wp::mat_t<3, 3, wp::float32>* var_288;
    wp::vec_t<3, wp::float32>* var_289;
    wp::int32* var_290;
    wp::int32* var_291;
    wp::int32* var_292;
    wp::int32 var_293;
    wp::mat_t<3, 3, wp::float32> var_294;
    wp::vec_t<3, wp::float32> var_295;
    wp::int32 var_296;
    wp::int32 var_297;
    wp::int32 var_298;
    wp::int32 var_299;
    wp::int32 var_300;
    wp::int32 var_301;
    wp::float32 var_302;
    wp::float32 var_303;
    wp::vec_t<3, wp::float32>* var_304;
    wp::vec_t<3, wp::float32> var_305;
    wp::vec_t<3, wp::float32> var_306;
    wp::vec_t<3, wp::float32>* var_307;
    wp::int32 var_308;
    wp::mat_t<4, 3, wp::float32> var_309;
    wp::mat_t<4, 3, wp::float32> var_310;
    wp::vec_t<3, wp::float32> var_311;
    wp::float32 var_312;
    wp::float32 var_313;
    wp::vec_t<3, wp::float32>* var_314;
    wp::vec_t<3, wp::float32> var_315;
    wp::vec_t<3, wp::float32> var_316;
    wp::vec_t<3, wp::float32>* var_317;
    wp::int32 var_318;
    wp::mat_t<4, 3, wp::float32> var_319;
    wp::mat_t<4, 3, wp::float32> var_320;
    wp::vec_t<3, wp::float32> var_321;
    wp::vec_t<3, wp::float32> var_322;
    wp::float32 var_323;
    wp::vec_t<3, wp::float32>* var_324;
    wp::vec_t<3, wp::float32> var_325;
    wp::vec_t<3, wp::float32> var_326;
    wp::vec_t<3, wp::float32>* var_327;
    wp::int32 var_328;
    wp::mat_t<4, 3, wp::float32> var_329;
    wp::mat_t<4, 3, wp::float32> var_330;
    wp::vec_t<3, wp::float32> var_331;
    //---------
    // forward
    // def multicontact(                                                                      <L 2065>
    // witness1 = mat43()                                                                     <L 2088>
    var_0 = wp::mat_t<4, 3, wp::float32>();
    // witness2 = mat43()                                                                     <L 2089>
    var_1 = wp::mat_t<4, 3, wp::float32>();
    // witness1[0] = x1                                                                       <L 2090>
    wp::assign_inplace(var_0, var_2, var_x1);
    // witness2[0] = x2                                                                       <L 2091>
    wp::assign_inplace(var_1, var_3, var_x2);
    // if geomtype1 == GeomType.MESH:                                                         <L 2093>
    var_5 = (var_geomtype1 == var_4);
    if (var_5) {
        // vert = geom1.vert                                                                  <L 2094>
        var_6 = &((var_geom1).vert);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // polynormal = geom1.mesh_polynormal                                                 <L 2095>
        var_9 = &((var_geom1).mesh_polynormal);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // polyvertadr = geom1.mesh_polyvertadr                                               <L 2096>
        var_12 = &((var_geom1).mesh_polyvertadr);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // polyvertnum = geom1.mesh_polyvertnum                                               <L 2097>
        var_15 = &((var_geom1).mesh_polyvertnum);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
        // polyvert = geom1.mesh_polyvert                                                     <L 2098>
        var_18 = &((var_geom1).mesh_polyvert);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // polymapadr = geom1.mesh_polymapadr                                                 <L 2099>
        var_21 = &((var_geom1).mesh_polymapadr);
        var_23 = wp::load(var_21);
        var_22 = wp::copy(var_23);
        // polymapnum = geom1.mesh_polymapnum                                                 <L 2100>
        var_24 = &((var_geom1).mesh_polymapnum);
        var_26 = wp::load(var_24);
        var_25 = wp::copy(var_26);
        // polymap = geom1.mesh_polymap                                                       <L 2101>
        var_27 = &((var_geom1).mesh_polymap);
        var_29 = wp::load(var_27);
        var_28 = wp::copy(var_29);
    }
    if (!var_5) {
        // elif geomtype2 == GeomType.MESH:                                                   <L 2102>
        var_31 = (var_geomtype2 == var_30);
        if (var_31) {
            // vert = geom2.vert                                                              <L 2103>
            var_32 = &((var_geom2).vert);
            var_34 = wp::load(var_32);
            var_33 = wp::copy(var_34);
            // polynormal = geom2.mesh_polynormal                                             <L 2104>
            var_35 = &((var_geom2).mesh_polynormal);
            var_37 = wp::load(var_35);
            var_36 = wp::copy(var_37);
            // polyvertadr = geom2.mesh_polyvertadr                                           <L 2105>
            var_38 = &((var_geom2).mesh_polyvertadr);
            var_40 = wp::load(var_38);
            var_39 = wp::copy(var_40);
            // polyvertnum = geom2.mesh_polyvertnum                                           <L 2106>
            var_41 = &((var_geom2).mesh_polyvertnum);
            var_43 = wp::load(var_41);
            var_42 = wp::copy(var_43);
            // polyvert = geom2.mesh_polyvert                                                 <L 2107>
            var_44 = &((var_geom2).mesh_polyvert);
            var_46 = wp::load(var_44);
            var_45 = wp::copy(var_46);
            // polymapadr = geom2.mesh_polymapadr                                             <L 2108>
            var_47 = &((var_geom2).mesh_polymapadr);
            var_49 = wp::load(var_47);
            var_48 = wp::copy(var_49);
            // polymapnum = geom2.mesh_polymapnum                                             <L 2109>
            var_50 = &((var_geom2).mesh_polymapnum);
            var_52 = wp::load(var_50);
            var_51 = wp::copy(var_52);
            // polymap = geom2.mesh_polymap                                                   <L 2110>
            var_53 = &((var_geom2).mesh_polymap);
            var_55 = wp::load(var_53);
            var_54 = wp::copy(var_55);
        }
        var_56 = wp::where(var_31, var_33, var_7);
        var_57 = wp::where(var_31, var_36, var_10);
        var_58 = wp::where(var_31, var_39, var_13);
        var_59 = wp::where(var_31, var_42, var_16);
        var_60 = wp::where(var_31, var_45, var_19);
        var_61 = wp::where(var_31, var_48, var_22);
        var_62 = wp::where(var_31, var_51, var_25);
        var_63 = wp::where(var_31, var_54, var_28);
    }
    var_64 = wp::where(var_5, var_7, var_56);
    var_65 = wp::where(var_5, var_10, var_57);
    var_66 = wp::where(var_5, var_13, var_58);
    var_67 = wp::where(var_5, var_16, var_59);
    var_68 = wp::where(var_5, var_19, var_60);
    var_69 = wp::where(var_5, var_22, var_61);
    var_70 = wp::where(var_5, var_25, var_62);
    var_71 = wp::where(var_5, var_28, var_63);
    // face = _get_face_verts(epa_face)                                                       <L 2112>
    var_72 = _get_face_verts_0(var_epa_face);
    // nface1, feature_index1, feature_vertex1 = _feature_dim(face, epa_vert_index, epa_vert, 0)       <L 2115>
    _feature_dim_0(var_72, var_epa_vert_index, var_epa_vert, var_73, var_74, var_75, var_76);
    // nface2, feature_index2, feature_vertex2 = _feature_dim(face, epa_vert_index, epa_vert, 1)       <L 2116>
    _feature_dim_0(var_72, var_epa_vert_index, var_epa_vert, var_77, var_78, var_79, var_80);
    // dir = x2 - x1                                                                          <L 2118>
    var_81 = wp::sub(var_x2, var_x1);
    // dir_neg = -dir                                                                         <L 2119>
    var_82 = wp::neg(var_81);
    // if geomtype1 == GeomType.BOX:                                                          <L 2122>
    var_84 = (var_geomtype1 == var_83);
    if (var_84) {
        // nnorms1 = _box_normals(nface1, feature_index1, geom1.rot, dir_neg, n1, idx1)       <L 2123>
        var_85 = &((var_geom1).rot);
        var_87 = wp::load(var_85);
        var_86 = _box_normals_0(var_74, var_75, var_87, var_82, var_n1, var_idx1);
    }
    if (!var_84) {
        // elif geomtype1 == GeomType.MESH:                                                   <L 2124>
        var_89 = (var_geomtype1 == var_88);
        if (var_89) {
            // nnorms1 = _mesh_normals(                                                       <L 2125>
            // nface1,                                                                        <L 2126>
            // feature_index1,                                                                <L 2127>
            // geom1.rot,                                                                     <L 2128>
            var_90 = &((var_geom1).rot);
            // geom1.vertadr,                                                                 <L 2129>
            var_91 = &((var_geom1).vertadr);
            // geom1.mesh_polyadr,                                                            <L 2130>
            var_92 = &((var_geom1).mesh_polyadr);
            // polynormal,                                                                    <L 2131>
            // polymapadr,                                                                    <L 2132>
            // polymapnum,                                                                    <L 2133>
            // polymap,                                                                       <L 2134>
            // n1,                                                                            <L 2135>
            // idx1,                                                                          <L 2136>
            var_94 = wp::load(var_90);
            var_95 = wp::load(var_91);
            var_96 = wp::load(var_92);
            var_93 = _mesh_normals_0(var_74, var_75, var_94, var_95, var_96, var_65, var_69, var_70, var_71, var_n1, var_idx1);
        }
        var_97 = wp::where(var_89, var_93, var_86);
    }
    var_98 = wp::where(var_84, var_86, var_97);
    // if geomtype2 == GeomType.BOX:                                                          <L 2138>
    var_100 = (var_geomtype2 == var_99);
    if (var_100) {
        // nnorms2 = _box_normals(nface2, feature_index2, geom2.rot, dir, n2, idx2)           <L 2139>
        var_101 = &((var_geom2).rot);
        var_103 = wp::load(var_101);
        var_102 = _box_normals_0(var_78, var_79, var_103, var_81, var_n2, var_idx2);
    }
    if (!var_100) {
        // elif geomtype2 == GeomType.MESH:                                                   <L 2140>
        var_105 = (var_geomtype2 == var_104);
        if (var_105) {
            // nnorms2 = _mesh_normals(                                                       <L 2141>
            // nface2,                                                                        <L 2142>
            // feature_index2,                                                                <L 2143>
            // geom2.rot,                                                                     <L 2144>
            var_106 = &((var_geom2).rot);
            // geom2.vertadr,                                                                 <L 2145>
            var_107 = &((var_geom2).vertadr);
            // geom2.mesh_polyadr,                                                            <L 2146>
            var_108 = &((var_geom2).mesh_polyadr);
            // polynormal,                                                                    <L 2147>
            // polymapadr,                                                                    <L 2148>
            // polymapnum,                                                                    <L 2149>
            // polymap,                                                                       <L 2150>
            // n2,                                                                            <L 2151>
            // idx2,                                                                          <L 2152>
            var_110 = wp::load(var_106);
            var_111 = wp::load(var_107);
            var_112 = wp::load(var_108);
            var_109 = _mesh_normals_0(var_78, var_79, var_110, var_111, var_112, var_65, var_69, var_70, var_71, var_n2, var_idx2);
        }
        var_113 = wp::where(var_105, var_109, var_102);
    }
    var_114 = wp::where(var_100, var_102, var_113);
    // is_edge_contact_geom1 = 0                                                              <L 2156>
    // is_edge_contact_geom2 = 0                                                              <L 2157>
    // nres, res = _aligned_faces(n1, nnorms1, n2, nnorms2)                                   <L 2158>
    _aligned_faces_0(var_n1, var_98, var_n2, var_114, var_117, var_118);
    // if not nres:                                                                           <L 2159>
    var_119 = wp::unot(var_117);
    if (var_119) {
        // if nface1 < 3 and nface1 <= nface2:                                                <L 2161>
        var_122 = (var_74 < var_121);
        var_120 = var_122;
        if (var_120) {
            var_123 = (var_74 <= var_78);
            var_120 = var_120 && var_123;
        }
        if (var_120) {
            // nnorms1 = 0                                                                    <L 2162>
            // if geomtype1 == GeomType.BOX:                                                  <L 2163>
            var_126 = (var_geomtype1 == var_125);
            if (var_126) {
                // nnorms1 = _box_edge_normals(                                               <L 2164>
                // nface1, geom1.rot, geom1.pos, geom1.size, feature_vertex1[0], feature_vertex1[1], feature_index1[0], n1, endvert       <L 2165>
                var_127 = &((var_geom1).rot);
                var_128 = &((var_geom1).pos);
                var_129 = &((var_geom1).size);
                var_131 = wp::extract(var_76, var_130);
                var_133 = wp::extract(var_76, var_132);
                var_135 = wp::extract(var_75, var_134);
                var_137 = wp::load(var_127);
                var_138 = wp::load(var_128);
                var_139 = wp::load(var_129);
                var_136 = _box_edge_normals_0(var_74, var_137, var_138, var_139, var_131, var_133, var_135, var_n1, var_endvert);
            }
            var_140 = wp::where(var_126, var_136, var_124);
            if (!var_126) {
                // elif geomtype1 == GeomType.MESH:                                           <L 2167>
                var_142 = (var_geomtype1 == var_141);
                if (var_142) {
                    // nnorms1 = _mesh_edge_normals(                                          <L 2168>
                    // nface1,                                                                <L 2169>
                    // geom1.rot,                                                             <L 2170>
                    var_143 = &((var_geom1).rot);
                    // geom1.pos,                                                             <L 2171>
                    var_144 = &((var_geom1).pos);
                    // geom1.vertadr,                                                         <L 2172>
                    var_145 = &((var_geom1).vertadr);
                    // geom1.mesh_polyadr,                                                    <L 2173>
                    var_146 = &((var_geom1).mesh_polyadr);
                    // geom1.vert,                                                            <L 2174>
                    var_147 = &((var_geom1).vert);
                    // polyvertadr,                                                           <L 2175>
                    // polyvertnum,                                                           <L 2176>
                    // polyvert,                                                              <L 2177>
                    // polymapadr,                                                            <L 2178>
                    // polymapnum,                                                            <L 2179>
                    // polymap,                                                               <L 2180>
                    // feature_vertex1[0],                                                    <L 2181>
                    var_149 = wp::extract(var_76, var_148);
                    // feature_vertex1[1],                                                    <L 2182>
                    var_151 = wp::extract(var_76, var_150);
                    // feature_index1[0],                                                     <L 2183>
                    var_153 = wp::extract(var_75, var_152);
                    // n1,                                                                    <L 2184>
                    // endvert,                                                               <L 2185>
                    var_155 = wp::load(var_143);
                    var_156 = wp::load(var_144);
                    var_157 = wp::load(var_145);
                    var_158 = wp::load(var_146);
                    var_159 = wp::load(var_147);
                    var_154 = _mesh_edge_normals_0(var_74, var_155, var_156, var_157, var_158, var_159, var_66, var_67, var_68, var_69, var_70, var_71, var_149, var_151, var_153, var_n1, var_endvert);
                }
                var_160 = wp::where(var_142, var_154, var_140);
            }
            var_161 = wp::where(var_126, var_140, var_160);
            // nres, res = _aligned_face_edge(n1, nnorms1, n2, nnorms2)                       <L 2187>
            _aligned_face_edge_0(var_n1, var_161, var_n2, var_114, var_162, var_163);
            // if not nres:                                                                   <L 2188>
            var_164 = wp::unot(var_162);
            if (var_164) {
                // return 1, witness1, witness2                                               <L 2189>
                ret_0 = var_165;
                ret_1 = var_0;
                ret_2 = var_1;
                return;
            }
            // is_edge_contact_geom1 = 1                                                      <L 2190>
        }
        var_167 = wp::where(var_120, var_161, var_98);
        var_168 = wp::where(var_120, var_166, var_115);
        var_169 = wp::where(var_120, var_162, var_117);
        var_170 = wp::where(var_120, var_163, var_118);
        if (!var_120) {
            // elif nface2 < 3:                                                               <L 2193>
            var_172 = (var_78 < var_171);
            if (var_172) {
                // nnorms2 = 0                                                                <L 2194>
                // if geomtype2 == GeomType.BOX:                                              <L 2195>
                var_175 = (var_geomtype2 == var_174);
                if (var_175) {
                    // nnorms2 = _box_edge_normals(                                           <L 2196>
                    // nface2, geom2.rot, geom2.pos, geom2.size, feature_vertex2[0], feature_vertex2[1], feature_index2[0], n2, endvert       <L 2197>
                    var_176 = &((var_geom2).rot);
                    var_177 = &((var_geom2).pos);
                    var_178 = &((var_geom2).size);
                    var_180 = wp::extract(var_80, var_179);
                    var_182 = wp::extract(var_80, var_181);
                    var_184 = wp::extract(var_79, var_183);
                    var_186 = wp::load(var_176);
                    var_187 = wp::load(var_177);
                    var_188 = wp::load(var_178);
                    var_185 = _box_edge_normals_0(var_78, var_186, var_187, var_188, var_180, var_182, var_184, var_n2, var_endvert);
                }
                var_189 = wp::where(var_175, var_185, var_173);
                if (!var_175) {
                    // elif geomtype2 == GeomType.MESH:                                       <L 2199>
                    var_191 = (var_geomtype2 == var_190);
                    if (var_191) {
                        // nnorms2 = _mesh_edge_normals(                                      <L 2200>
                        // nface2,                                                            <L 2201>
                        // geom2.rot,                                                         <L 2202>
                        var_192 = &((var_geom2).rot);
                        // geom2.pos,                                                         <L 2203>
                        var_193 = &((var_geom2).pos);
                        // geom2.vertadr,                                                     <L 2204>
                        var_194 = &((var_geom2).vertadr);
                        // geom2.mesh_polyadr,                                                <L 2205>
                        var_195 = &((var_geom2).mesh_polyadr);
                        // geom2.vert,                                                        <L 2206>
                        var_196 = &((var_geom2).vert);
                        // polyvertadr,                                                       <L 2207>
                        // polyvertnum,                                                       <L 2208>
                        // polyvert,                                                          <L 2209>
                        // polymapadr,                                                        <L 2210>
                        // polymapnum,                                                        <L 2211>
                        // polymap,                                                           <L 2212>
                        // feature_vertex2[0],                                                <L 2213>
                        var_198 = wp::extract(var_80, var_197);
                        // feature_vertex2[1],                                                <L 2214>
                        var_200 = wp::extract(var_80, var_199);
                        // feature_index2[0],                                                 <L 2215>
                        var_202 = wp::extract(var_79, var_201);
                        // n2,                                                                <L 2216>
                        // endvert,                                                           <L 2217>
                        var_204 = wp::load(var_192);
                        var_205 = wp::load(var_193);
                        var_206 = wp::load(var_194);
                        var_207 = wp::load(var_195);
                        var_208 = wp::load(var_196);
                        var_203 = _mesh_edge_normals_0(var_78, var_204, var_205, var_206, var_207, var_208, var_66, var_67, var_68, var_69, var_70, var_71, var_198, var_200, var_202, var_n2, var_endvert);
                    }
                    var_209 = wp::where(var_191, var_203, var_189);
                }
                var_210 = wp::where(var_175, var_189, var_209);
                // nres, res = _aligned_face_edge(n2, nnorms2, n1, nnorms1)                   <L 2219>
                _aligned_face_edge_0(var_n2, var_210, var_n1, var_167, var_211, var_212);
                // if not nres:                                                               <L 2220>
                var_213 = wp::unot(var_211);
                if (var_213) {
                    // return 1, witness1, witness2                                           <L 2221>
                    ret_0 = var_214;
                    ret_1 = var_0;
                    ret_2 = var_1;
                    return;
                }
                // is_edge_contact_geom2 = 1                                                  <L 2222>
            }
            var_216 = wp::where(var_172, var_210, var_114);
            var_217 = wp::where(var_172, var_215, var_116);
            var_218 = wp::where(var_172, var_211, var_169);
            var_219 = wp::where(var_172, var_212, var_170);
            if (!var_172) {
                // return 1, witness1, witness2                                               <L 2225>
                ret_0 = var_220;
                ret_1 = var_0;
                ret_2 = var_1;
                return;
            }
        }
        var_221 = wp::where(var_120, var_114, var_216);
        var_222 = wp::where(var_120, var_116, var_217);
        var_223 = wp::where(var_120, var_169, var_218);
        var_224 = wp::where(var_120, var_170, var_219);
    }
    var_225 = wp::where(var_119, var_167, var_98);
    var_226 = wp::where(var_119, var_221, var_114);
    var_227 = wp::where(var_119, var_168, var_115);
    var_228 = wp::where(var_119, var_222, var_116);
    var_229 = wp::where(var_119, var_223, var_117);
    var_230 = wp::where(var_119, var_224, var_118);
    // i = res[0]                                                                             <L 2227>
    var_232 = wp::extract(var_230, var_231);
    // j = res[1]                                                                             <L 2228>
    var_234 = wp::extract(var_230, var_233);
    // if is_edge_contact_geom1:                                                              <L 2231>
    if (var_227) {
        // nface1 = _set_edge(epa_vert, endvert, face[0], i, 0, face1)                        <L 2232>
        var_236 = wp::extract(var_72, var_235);
        var_238 = _set_edge_0(var_epa_vert, var_endvert, var_236, var_232, var_237, var_face1);
    }
    var_239 = wp::where(var_227, var_238, var_74);
    if (!var_227) {
        // ind = wp.where(is_edge_contact_geom2, idx1[j], idx1[i])                            <L 2234>
        var_240 = wp::address(var_idx1, var_234);
        var_241 = wp::address(var_idx1, var_232);
        var_243 = wp::load(var_240);
        var_244 = wp::load(var_241);
        var_242 = wp::where(var_228, var_243, var_244);
        // if geomtype1 == GeomType.BOX:                                                      <L 2235>
        var_246 = (var_geomtype1 == var_245);
        if (var_246) {
            // nface1 = _box_face(geom1.rot, geom1.pos, geom1.size, ind, face1)               <L 2236>
            var_247 = &((var_geom1).rot);
            var_248 = &((var_geom1).pos);
            var_249 = &((var_geom1).size);
            var_251 = wp::load(var_247);
            var_252 = wp::load(var_248);
            var_253 = wp::load(var_249);
            var_250 = _box_face_0(var_251, var_252, var_253, var_242, var_face1);
        }
        var_254 = wp::where(var_246, var_250, var_239);
        if (!var_246) {
            // elif geomtype1 == GeomType.MESH:                                               <L 2237>
            var_256 = (var_geomtype1 == var_255);
            if (var_256) {
                // nface1 = _mesh_face(                                                       <L 2238>
                // geom1.rot,                                                                 <L 2239>
                var_257 = &((var_geom1).rot);
                // geom1.pos,                                                                 <L 2240>
                var_258 = &((var_geom1).pos);
                // geom1.vertadr,                                                             <L 2241>
                var_259 = &((var_geom1).vertadr);
                // geom1.mesh_polyadr,                                                        <L 2242>
                var_260 = &((var_geom1).mesh_polyadr);
                // vert,                                                                      <L 2243>
                // polyvertadr,                                                               <L 2244>
                // polyvertnum,                                                               <L 2245>
                // polyvert,                                                                  <L 2246>
                // ind,                                                                       <L 2247>
                // face1,                                                                     <L 2248>
                var_262 = wp::load(var_257);
                var_263 = wp::load(var_258);
                var_264 = wp::load(var_259);
                var_265 = wp::load(var_260);
                var_261 = _mesh_face_0(var_262, var_263, var_264, var_265, var_64, var_66, var_67, var_68, var_242, var_face1);
            }
            var_266 = wp::where(var_256, var_261, var_254);
        }
        var_267 = wp::where(var_246, var_254, var_266);
    }
    var_268 = wp::where(var_227, var_239, var_267);
    // if is_edge_contact_geom2:                                                              <L 2252>
    if (var_228) {
        // nface2 = _set_edge(epa_vert, endvert, face[0], i, 1, face2)                        <L 2253>
        var_270 = wp::extract(var_72, var_269);
        var_272 = _set_edge_0(var_epa_vert, var_endvert, var_270, var_232, var_271, var_face2);
    }
    var_273 = wp::where(var_228, var_272, var_78);
    if (!var_228) {
        // if geomtype2 == GeomType.BOX:                                                      <L 2255>
        var_275 = (var_geomtype2 == var_274);
        if (var_275) {
            // nface2 = _box_face(geom2.rot, geom2.pos, geom2.size, idx2[j], face2)           <L 2256>
            var_276 = &((var_geom2).rot);
            var_277 = &((var_geom2).pos);
            var_278 = &((var_geom2).size);
            var_279 = wp::address(var_idx2, var_234);
            var_281 = wp::load(var_276);
            var_282 = wp::load(var_277);
            var_283 = wp::load(var_278);
            var_284 = wp::load(var_279);
            var_280 = _box_face_0(var_281, var_282, var_283, var_284, var_face2);
        }
        var_285 = wp::where(var_275, var_280, var_273);
        if (!var_275) {
            // elif geomtype2 == GeomType.MESH:                                               <L 2257>
            var_287 = (var_geomtype2 == var_286);
            if (var_287) {
                // nface2 = _mesh_face(                                                       <L 2258>
                // geom2.rot,                                                                 <L 2259>
                var_288 = &((var_geom2).rot);
                // geom2.pos,                                                                 <L 2260>
                var_289 = &((var_geom2).pos);
                // geom2.vertadr,                                                             <L 2261>
                var_290 = &((var_geom2).vertadr);
                // geom2.mesh_polyadr,                                                        <L 2262>
                var_291 = &((var_geom2).mesh_polyadr);
                // vert,                                                                      <L 2263>
                // polyvertadr,                                                               <L 2264>
                // polyvertnum,                                                               <L 2265>
                // polyvert,                                                                  <L 2266>
                // idx2[j],                                                                   <L 2267>
                var_292 = wp::address(var_idx2, var_234);
                // face2,                                                                     <L 2268>
                var_294 = wp::load(var_288);
                var_295 = wp::load(var_289);
                var_296 = wp::load(var_290);
                var_297 = wp::load(var_291);
                var_298 = wp::load(var_292);
                var_293 = _mesh_face_0(var_294, var_295, var_296, var_297, var_64, var_66, var_67, var_68, var_298, var_face2);
            }
            var_299 = wp::where(var_287, var_293, var_285);
        }
        var_300 = wp::where(var_275, var_285, var_299);
    }
    var_301 = wp::where(var_228, var_273, var_300);
    // if is_edge_contact_geom1:                                                              <L 2272>
    if (var_227) {
        // approx_dir = -wp.norm_l2(dir) * n2[j]                                              <L 2273>
        var_302 = norm_l2_0(var_81);
        var_303 = wp::neg(var_302);
        var_304 = wp::address(var_n2, var_234);
        var_306 = wp::load(var_304);
        var_305 = wp::mul(var_303, var_306);
        // nclipped, clipped1, clipped2 = _polygon_clip(                                      <L 2274>
        // plane_normal, plane_dist, face2, nface2, face1, nface1, n2[j], approx_dir, polygon, clipped       <L 2275>
        var_307 = wp::address(var_n2, var_234);
        var_311 = wp::load(var_307);
        _polygon_clip_0(var_plane_normal, var_plane_dist, var_face2, var_301, var_face1, var_268, var_311, var_305, var_polygon, var_clipped, var_308, var_309, var_310);
        // return nclipped, clipped2, clipped1                                                <L 2278>
        ret_0 = var_308;
        ret_1 = var_310;
        ret_2 = var_309;
        return;
    }
    // if is_edge_contact_geom2:                                                              <L 2281>
    if (var_228) {
        // approx_dir = -wp.norm_l2(dir) * n1[j]                                              <L 2282>
        var_312 = norm_l2_0(var_81);
        var_313 = wp::neg(var_312);
        var_314 = wp::address(var_n1, var_234);
        var_316 = wp::load(var_314);
        var_315 = wp::mul(var_313, var_316);
        // return _polygon_clip(plane_normal, plane_dist, face1, nface1, face2, nface2, n1[j], approx_dir, polygon, clipped)       <L 2283>
        var_317 = wp::address(var_n1, var_234);
        var_321 = wp::load(var_317);
        _polygon_clip_0(var_plane_normal, var_plane_dist, var_face1, var_268, var_face2, var_301, var_321, var_315, var_polygon, var_clipped, var_318, var_319, var_320);
        ret_0 = var_318;
        ret_1 = var_319;
        ret_2 = var_320;
        return;
    }
    var_322 = wp::where(var_228, var_315, var_305);
    // approx_dir = wp.norm_l2(dir) * n2[j]                                                   <L 2286>
    var_323 = norm_l2_0(var_81);
    var_324 = wp::address(var_n2, var_234);
    var_326 = wp::load(var_324);
    var_325 = wp::mul(var_323, var_326);
    // return _polygon_clip(plane_normal, plane_dist, face1, nface1, face2, nface2, n1[i], approx_dir, polygon, clipped)       <L 2288>
    var_327 = wp::address(var_n1, var_232);
    var_331 = wp::load(var_327);
    _polygon_clip_0(var_plane_normal, var_plane_dist, var_face1, var_268, var_face2, var_301, var_331, var_325, var_polygon, var_clipped, var_328, var_329, var_330);
    ret_0 = var_328;
    ret_1 = var_329;
    ret_2 = var_330;
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


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_convex.py:733
static CUDA_CALLABLE void ccd_kernel_builder__locals__eval_ccd_write_contact_0(
    wp::array_t<wp::float32> var_opt_ccd_tolerance,
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
    wp::int32 var_naconmax_in,
    wp::int32 var_naccdmax_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_epa_vert_in,
    wp::array_t<wp::int32> var_epa_vert_index_in,
    wp::array_t<wp::int32> var_epa_face_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_epa_pr_in,
    wp::array_t<wp::float32> var_epa_norm2_in,
    wp::array_t<wp::int32> var_epa_horizon_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_polygon_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_clipped_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_pnormal_in,
    wp::array_t<wp::float32> var_multiccd_pdist_in,
    wp::array_t<wp::int32> var_multiccd_idx1_in,
    wp::array_t<wp::int32> var_multiccd_idx2_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_n1_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_n2_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_endvert_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_face1_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_face2_in,
    Geom_3242f8a8 var_geom1,
    Geom_3242f8a8 var_geom2,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::int32 var_worldid,
    wp::array_t<wp::int32> var_nccd_in,
    wp::float32 var_margin,
    wp::float32 var_gap,
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
    wp::array_t<wp::int32> var_overflow_out)
{
    //---------
    // primal vars
    wp::shape_t* var_0;
    const wp::int32 var_1 = 0;
    wp::int32 var_2;
    wp::shape_t var_3;
    wp::int32 var_4;
    wp::float32* var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    const wp::int32 var_8 = 1;
    wp::int32 var_9;
    const wp::int32 var_10 = 0;
    bool var_11;
    const wp::float32 var_12 = 1e+32;
    wp::float32 var_13;
    wp::float32 var_14;
    const wp::int32 var_15 = 50;
    const wp::int32 var_16 = 6;
    const wp::int32 var_17 = 6;
    wp::vec_t<3, wp::float32>* var_18;
    wp::vec_t<3, wp::float32>* var_19;
    bool var_20;
    wp::float32 var_21;
    wp::int32 var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::vec_t<3, wp::float32> var_24;
    GJKResult_28609055 var_25;
    Geom_3242f8a8 var_26;
    Geom_3242f8a8 var_27;
    wp::vec_t<3, wp::float32> var_28;
    wp::vec_t<3, wp::float32> var_29;
    const wp::int32 var_30 = -1;
    wp::int32 var_31;
    const wp::int32 var_32 = -1;
    wp::int32 var_33;
    const wp::int32 var_34 = 51;
    const wp::int32 var_35 = 1;
    wp::int32 var_36;
    bool var_37;
    const bool var_38 = true;
    const wp::str var_39 = "CCD overflow - please increase naccdmax to %u\n";
    const wp::int32 var_40 = 16;
    const wp::int32 var_41 = 16;
    wp::int32 var_42;
    const wp::int32 var_43 = 16;
    wp::slice_t var_44;
    const wp::int32 var_45 = 0;
    wp::array_t<wp::vec_t<3, wp::float32>> var_46;
    wp::slice_t var_47;
    const wp::int32 var_48 = 0;
    wp::array_t<wp::int32> var_49;
    wp::slice_t var_50;
    const wp::int32 var_51 = 0;
    wp::array_t<wp::int32> var_52;
    wp::slice_t var_53;
    const wp::int32 var_54 = 0;
    wp::array_t<wp::vec_t<3, wp::float32>> var_55;
    wp::slice_t var_56;
    const wp::int32 var_57 = 0;
    wp::array_t<wp::float32> var_58;
    wp::slice_t var_59;
    const wp::int32 var_60 = 0;
    wp::array_t<wp::int32> var_61;
    const bool var_62 = true;
    wp::float32 var_63;
    wp::int32 var_64;
    wp::vec_t<3, wp::float32> var_65;
    wp::vec_t<3, wp::float32> var_66;
    wp::int32 var_67;
    wp::float32 var_68;
    wp::int32 var_69;
    wp::vec_t<3, wp::float32> var_70;
    wp::vec_t<3, wp::float32> var_71;
    wp::int32 var_72;
    wp::int32 var_73;
    bool var_74;
    bool var_75;
    bool var_76;
    wp::float32 var_77;
    wp::mat_t<4, 3, wp::float32> var_78;
    wp::mat_t<4, 3, wp::float32> var_79;
    const wp::int32 var_80 = 0;
    const wp::int32 var_81 = 0;
    const bool var_82 = true;
    const bool var_83 = false;
    const bool var_84 = false;
    const wp::int32 var_85 = -1;
    bool var_86;
    wp::slice_t var_87;
    const wp::int32 var_88 = 0;
    wp::array_t<wp::vec_t<3, wp::float32>> var_89;
    wp::slice_t var_90;
    const wp::int32 var_91 = 0;
    wp::array_t<wp::vec_t<3, wp::float32>> var_92;
    wp::slice_t var_93;
    const wp::int32 var_94 = 0;
    wp::array_t<wp::vec_t<3, wp::float32>> var_95;
    wp::slice_t var_96;
    const wp::int32 var_97 = 0;
    wp::array_t<wp::float32> var_98;
    wp::slice_t var_99;
    const wp::int32 var_100 = 0;
    wp::array_t<wp::int32> var_101;
    wp::slice_t var_102;
    const wp::int32 var_103 = 0;
    wp::array_t<wp::int32> var_104;
    wp::slice_t var_105;
    const wp::int32 var_106 = 0;
    wp::array_t<wp::vec_t<3, wp::float32>> var_107;
    wp::slice_t var_108;
    const wp::int32 var_109 = 0;
    wp::array_t<wp::vec_t<3, wp::float32>> var_110;
    wp::slice_t var_111;
    const wp::int32 var_112 = 0;
    wp::array_t<wp::vec_t<3, wp::float32>> var_113;
    wp::slice_t var_114;
    const wp::int32 var_115 = 0;
    wp::array_t<wp::vec_t<3, wp::float32>> var_116;
    wp::slice_t var_117;
    const wp::int32 var_118 = 0;
    wp::array_t<wp::vec_t<3, wp::float32>> var_119;
    wp::slice_t var_120;
    const wp::int32 var_121 = 0;
    wp::array_t<wp::vec_t<3, wp::float32>> var_122;
    wp::slice_t var_123;
    const wp::int32 var_124 = 0;
    wp::array_t<wp::int32> var_125;
    wp::int32* var_126;
    wp::int32 var_127;
    wp::mat_t<4, 3, wp::float32> var_128;
    wp::mat_t<4, 3, wp::float32> var_129;
    wp::int32 var_130;
    wp::int32 var_131;
    wp::mat_t<4, 3, wp::float32> var_132;
    wp::mat_t<4, 3, wp::float32> var_133;
    const wp::int32 var_134 = 0;
    wp::int32 var_135;
    wp::int32 var_136;
    wp::vec_t<5, wp::float32> var_137;
    wp::vec_t<2, wp::float32> var_138;
    wp::vec_t<2, wp::float32> var_139;
    wp::vec_t<5, wp::float32> var_140;
    const wp::int32 var_141 = 0;
    wp::vec_t<3, wp::float32> var_142;
    const wp::int32 var_143 = 0;
    wp::vec_t<3, wp::float32> var_144;
    wp::vec_t<3, wp::float32> var_145;
    wp::mat_t<3, 3, wp::float32> var_146;
    const wp::float32 var_147 = -1.0;
    wp::mat_t<3, 3, wp::float32> var_148;
    const wp::int32 var_149 = 1;
    wp::int32 var_150;
    const wp::int32 var_151 = 0;
    wp::int32 var_152;
    wp::vec_t<2, wp::int32> var_153;
    wp::vec_t<2, wp::int32> var_154;
    wp::mat_t<3, 3, wp::float32> var_155;
    wp::range_t var_156;
    wp::int32 var_157;
    const wp::float32 var_158 = 0.5;
    wp::vec_t<3, wp::float32> var_159;
    wp::vec_t<3, wp::float32> var_160;
    wp::vec_t<3, wp::float32> var_161;
    wp::vec_t<3, wp::float32> var_162;
    wp::int32 var_163;
    //---------
    // forward
    // def eval_ccd_write_contact(                                                            <L 734>
    // geom1.margin = margin                                                                  <L 795>
    var_geom1.margin = var_margin;
    // geom2.margin = margin                                                                  <L 796>
    var_geom2.margin = var_margin;
    // tolerance = opt_ccd_tolerance[worldid % opt_ccd_tolerance.shape[0]]                    <L 797>
    var_0 = &(var_opt_ccd_tolerance.shape);
    var_3 = wp::load(var_0);
    var_2 = wp::extract(var_3, var_1);
    var_4 = wp::mod(var_worldid, var_2);
    var_5 = wp::address(var_opt_ccd_tolerance, var_4);
    var_7 = wp::load(var_5);
    var_6 = wp::copy(var_7);
    // is_collision_sensor = pairid[1] >= 0                                                   <L 798>
    var_9 = wp::extract(var_pairid, var_8);
    var_11 = (var_9 >= var_10);
    // if is_collision_sensor:                                                                <L 799>
    if (var_11) {
        // cutoff = 1.0e32                                                                    <L 800>
    }
    if (!var_11) {
        // cutoff = gap                                                                       <L 802>
        var_13 = wp::copy(var_gap);
    }
    var_14 = wp::where(var_11, var_12, var_13);
    // needs_epa, dist, ncollision, w1, w2, gjk_result, geom1, geom2 = gjk_phase(             <L 803>
    // tolerance,                                                                             <L 804>
    // cutoff,                                                                                <L 805>
    // gjk_iterations,                                                                        <L 806>
    // geom1,                                                                                 <L 807>
    // geom2,                                                                                 <L 808>
    // geomtype1,                                                                             <L 809>
    // geomtype2,                                                                             <L 810>
    // geom1.pos,                                                                             <L 811>
    var_18 = &((var_geom1).pos);
    // geom2.pos,                                                                             <L 812>
    var_19 = &((var_geom2).pos);
    var_28 = wp::load(var_18);
    var_29 = wp::load(var_19);
    gjk_phase_0(var_6, var_14, var_15, var_geom1, var_geom2, var_16, var_17, var_28, var_29, var_20, var_21, var_22, var_23, var_24, var_25, var_26, var_27);
    // ccdid = int(-1)                                                                        <L 815>
    var_31 = wp::int(var_30);
    // multiccd_idx = int(-1)                                                                 <L 816>
    var_33 = wp::int(var_32);
    // if needs_epa:                                                                          <L 818>
    if (var_20) {
        // ccdid = wp.atomic_add(nccd_in, geomgeomid, 1)                                      <L 819>
        var_36 = wp::atomic_add(var_nccd_in, var_34, var_35);
        // if ccdid >= naccdmax_in:                                                           <L 820>
        var_37 = (var_36 >= var_naccdmax_in);
        if (var_37) {
            // if wp.static(warn_overflow):                                                   <L 821>
            // wp.printf("CCD overflow - please increase naccdmax to %u\n", ccdid)            <L 822>
            printf(var_39, var_36);
            // wp.atomic_or(overflow_out, worldid, OverflowType.CCD)                          <L 823>
            var_42 = wp::atomic_or(var_overflow_out, var_worldid, var_41);
            // return                                                                         <L 824>
            return;
        }
        // dist, ncollision, w1, w2, multiccd_idx = epa_phase(                                <L 825>
        // tolerance,                                                                         <L 826>
        // epa_iterations,                                                                    <L 827>
        // gjk_result,                                                                        <L 828>
        // geom1,                                                                             <L 829>
        // geom2,                                                                             <L 830>
        // geomtype1,                                                                         <L 831>
        // geomtype2,                                                                         <L 832>
        // epa_vert_in[ccdid],                                                                <L 833>
        var_44 = wp::slice_t(var_36, var_36, var_45);
        var_46 = wp::view(var_epa_vert_in, var_44);
        // epa_vert_index_in[ccdid],                                                          <L 834>
        var_47 = wp::slice_t(var_36, var_36, var_48);
        var_49 = wp::view(var_epa_vert_index_in, var_47);
        // epa_face_in[ccdid],                                                                <L 835>
        var_50 = wp::slice_t(var_36, var_36, var_51);
        var_52 = wp::view(var_epa_face_in, var_50);
        // epa_pr_in[ccdid],                                                                  <L 836>
        var_53 = wp::slice_t(var_36, var_36, var_54);
        var_55 = wp::view(var_epa_pr_in, var_53);
        // epa_norm2_in[ccdid],                                                               <L 837>
        var_56 = wp::slice_t(var_36, var_36, var_57);
        var_58 = wp::view(var_epa_norm2_in, var_56);
        // epa_horizon_in[ccdid],                                                             <L 838>
        var_59 = wp::slice_t(var_36, var_36, var_60);
        var_61 = wp::view(var_epa_horizon_in, var_59);
        // wp.static(warn_overflow),                                                          <L 839>
        // worldid,                                                                           <L 840>
        // overflow_out,                                                                      <L 841>
        epa_phase_0(var_6, var_43, var_25, var_26, var_27, var_16, var_17, var_46, var_49, var_52, var_55, var_58, var_61, var_62, var_worldid, var_overflow_out, var_63, var_64, var_65, var_66, var_67);
    }
    var_68 = wp::where(var_20, var_63, var_21);
    var_69 = wp::where(var_20, var_64, var_22);
    var_70 = wp::where(var_20, var_65, var_23);
    var_71 = wp::where(var_20, var_66, var_24);
    var_72 = wp::where(var_20, var_36, var_31);
    var_73 = wp::where(var_20, var_67, var_33);
    // if dist >= gap and not is_collision_sensor:                                            <L 844>
    var_75 = (var_68 >= var_gap);
    var_74 = var_75;
    if (var_74) {
        var_76 = wp::unot(var_11);
        var_74 = var_74 && var_76;
    }
    if (var_74) {
        // return                                                                             <L 845>
        return;
    }
    // dist += margin                                                                         <L 852>
    var_77 = wp::add(var_68, var_margin);
    // witness1 = mat43()                                                                     <L 854>
    var_78 = wp::mat_t<4, 3, wp::float32>();
    // witness2 = mat43()                                                                     <L 855>
    var_79 = wp::mat_t<4, 3, wp::float32>();
    // witness1[0] = w1                                                                       <L 856>
    wp::assign_inplace(var_78, var_80, var_70);
    // witness2[0] = w2                                                                       <L 857>
    wp::assign_inplace(var_79, var_81, var_71);
    // if wp.static(                                                                          <L 859>
    // if wp.static(geomtype1 == GeomType.MESH):                                              <L 864>
    // if wp.static(geomtype2 == GeomType.MESH):                                              <L 869>
    // if multiccd_idx > -1:                                                                  <L 874>
    var_86 = (var_73 > var_85);
    if (var_86) {
        // ncollision, witness1, witness2 = multicontact(                                     <L 875>
        // multiccd_polygon_in[ccdid],                                                        <L 876>
        var_87 = wp::slice_t(var_72, var_72, var_88);
        var_89 = wp::view(var_multiccd_polygon_in, var_87);
        // multiccd_clipped_in[ccdid],                                                        <L 877>
        var_90 = wp::slice_t(var_72, var_72, var_91);
        var_92 = wp::view(var_multiccd_clipped_in, var_90);
        // multiccd_pnormal_in[ccdid],                                                        <L 878>
        var_93 = wp::slice_t(var_72, var_72, var_94);
        var_95 = wp::view(var_multiccd_pnormal_in, var_93);
        // multiccd_pdist_in[ccdid],                                                          <L 879>
        var_96 = wp::slice_t(var_72, var_72, var_97);
        var_98 = wp::view(var_multiccd_pdist_in, var_96);
        // multiccd_idx1_in[ccdid],                                                           <L 880>
        var_99 = wp::slice_t(var_72, var_72, var_100);
        var_101 = wp::view(var_multiccd_idx1_in, var_99);
        // multiccd_idx2_in[ccdid],                                                           <L 881>
        var_102 = wp::slice_t(var_72, var_72, var_103);
        var_104 = wp::view(var_multiccd_idx2_in, var_102);
        // multiccd_n1_in[ccdid],                                                             <L 882>
        var_105 = wp::slice_t(var_72, var_72, var_106);
        var_107 = wp::view(var_multiccd_n1_in, var_105);
        // multiccd_n2_in[ccdid],                                                             <L 883>
        var_108 = wp::slice_t(var_72, var_72, var_109);
        var_110 = wp::view(var_multiccd_n2_in, var_108);
        // multiccd_endvert_in[ccdid],                                                        <L 884>
        var_111 = wp::slice_t(var_72, var_72, var_112);
        var_113 = wp::view(var_multiccd_endvert_in, var_111);
        // multiccd_face1_in[ccdid],                                                          <L 885>
        var_114 = wp::slice_t(var_72, var_72, var_115);
        var_116 = wp::view(var_multiccd_face1_in, var_114);
        // multiccd_face2_in[ccdid],                                                          <L 886>
        var_117 = wp::slice_t(var_72, var_72, var_118);
        var_119 = wp::view(var_multiccd_face2_in, var_117);
        // epa_vert_in[ccdid],                                                                <L 887>
        var_120 = wp::slice_t(var_72, var_72, var_121);
        var_122 = wp::view(var_epa_vert_in, var_120);
        // epa_vert_index_in[ccdid],                                                          <L 888>
        var_123 = wp::slice_t(var_72, var_72, var_124);
        var_125 = wp::view(var_epa_vert_index_in, var_123);
        // epa_face_in[ccdid, multiccd_idx],                                                  <L 889>
        var_126 = wp::address(var_epa_face_in, var_72, var_73);
        // w1,                                                                                <L 890>
        // w2,                                                                                <L 891>
        // geom1,                                                                             <L 892>
        // geom2,                                                                             <L 893>
        // geomtype1,                                                                         <L 894>
        // geomtype2,                                                                         <L 895>
        var_130 = wp::load(var_126);
        multicontact_0(var_89, var_92, var_95, var_98, var_101, var_104, var_107, var_110, var_113, var_116, var_119, var_122, var_125, var_130, var_70, var_71, var_26, var_27, var_16, var_17, var_127, var_128, var_129);
    }
    var_131 = wp::where(var_86, var_127, var_69);
    var_132 = wp::where(var_86, var_128, var_78);
    var_133 = wp::where(var_86, var_129, var_79);
    // condim, friction, solref, solreffriction, solimp = contact_material_params(            <L 898>
    // geom_condim,                                                                           <L 899>
    // geom_priority,                                                                         <L 900>
    // geom_solmix,                                                                           <L 901>
    // geom_solref,                                                                           <L 902>
    // geom_solimp,                                                                           <L 903>
    // geom_friction,                                                                         <L 904>
    // pair_dim,                                                                              <L 905>
    // pair_solref,                                                                           <L 906>
    // pair_solreffriction,                                                                   <L 907>
    // pair_solimp,                                                                           <L 908>
    // pair_friction,                                                                         <L 909>
    // geoms,                                                                                 <L 910>
    // pairid[0],                                                                             <L 911>
    var_135 = wp::extract(var_pairid, var_134);
    // worldid,                                                                               <L 912>
    contact_material_params_0(var_geom_condim, var_geom_priority, var_geom_solmix, var_geom_solref, var_geom_solimp, var_geom_friction, var_pair_dim, var_pair_solref, var_pair_solreffriction, var_pair_solimp, var_pair_friction, var_geoms, var_135, var_worldid, var_136, var_137, var_138, var_139, var_140);
    // frame = make_frame(witness1[0] - witness2[0])                                          <L 915>
    var_142 = wp::extract(var_132, var_141);
    var_144 = wp::extract(var_133, var_143);
    var_145 = wp::sub(var_142, var_144);
    var_146 = make_frame_0(var_145);
    // if is_collision_sensor:                                                                <L 916>
    if (var_11) {
        // frame *= -1.0                                                                      <L 917>
        var_148 = wp::mul(var_146, var_147);
        // geoms = wp.vec2i(geoms[1], geoms[0])                                               <L 918>
        var_150 = wp::extract(var_geoms, var_149);
        var_152 = wp::extract(var_geoms, var_151);
        var_153 = wp::vec_t<2, wp::int32>(var_150, var_152);
    }
    var_154 = wp::where(var_11, var_153, var_geoms);
    var_155 = wp::where(var_11, var_148, var_146);
    // for i in range(ncollision):                                                            <L 920>
    var_156 = wp::range(var_131);
    start_for_2:;
        if (iter_cmp(var_156) == 0) goto end_for_2;
        var_157 = wp::iter_next(var_156);
        // write_contact(                                                                     <L 921>
        // naconmax_in,                                                                       <L 922>
        // i,                                                                                 <L 923>
        // dist,                                                                              <L 924>
        // 0.5 * (witness1[i] + witness2[i]),                                                 <L 925>
        var_159 = wp::extract(var_132, var_157);
        var_160 = wp::extract(var_133, var_157);
        var_161 = wp::add(var_159, var_160);
        var_162 = wp::mul(var_158, var_161);
        // frame,                                                                             <L 926>
        // margin,                                                                            <L 927>
        // gap,                                                                               <L 928>
        // condim,                                                                            <L 929>
        // friction,                                                                          <L 930>
        // solref,                                                                            <L 931>
        // solreffriction,                                                                    <L 932>
        // solimp,                                                                            <L 933>
        // geoms,                                                                             <L 934>
        // pairid,                                                                            <L 935>
        // worldid,                                                                           <L 936>
        // contact_dist_out,                                                                  <L 937>
        // contact_pos_out,                                                                   <L 938>
        // contact_frame_out,                                                                 <L 939>
        // contact_includemargin_out,                                                         <L 940>
        // contact_friction_out,                                                              <L 941>
        // contact_solref_out,                                                                <L 942>
        // contact_solreffriction_out,                                                        <L 943>
        // contact_solimp_out,                                                                <L 944>
        // contact_dim_out,                                                                   <L 945>
        // contact_geom_out,                                                                  <L 946>
        // contact_efc_address_out,                                                           <L 947>
        // contact_worldid_out,                                                               <L 948>
        // contact_type_out,                                                                  <L 949>
        // contact_geomcollisionid_out,                                                       <L 950>
        // nacon_out,                                                                         <L 951>
        var_163 = write_contact_0(var_naconmax_in, var_157, var_77, var_162, var_155, var_margin, var_gap, var_136, var_137, var_138, var_139, var_140, var_154, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_nacon_out);
        goto start_for_2;
    end_for_2:;
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


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:107
static CUDA_CALLABLE void adj__discrete_geoms_0(
    wp::int32 var_g1,
    wp::int32 var_g2,
    wp::int32 & adj_g1,
    wp::int32 & adj_g2,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:114
static CUDA_CALLABLE void adj_support_0(
    Geom_3242f8a8 var_geom,
    wp::int32 var_geomtype,
    wp::vec_t<3, wp::float32> var_dir,
    Geom_3242f8a8 & adj_geom,
    wp::int32 & adj_geomtype,
    wp::vec_t<3, wp::float32> & adj_dir,
    SupportPoint_e82efc60 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:594
static CUDA_CALLABLE void adj__gjk_support_0(
    Geom_3242f8a8 var_geom1,
    Geom_3242f8a8 var_geom2,
    wp::int32 var_geomtype1,
    wp::int32 var_geomtype2,
    wp::vec_t<3, wp::float32> var_x_k,
    wp::float32 var_x_norm,
    wp::mat_t<4, 3, wp::float32> var_simplex,
    wp::int32 var_n,
    bool var_is_discrete,
    SupportPoint_e82efc60 & ret_0,
    SupportPoint_e82efc60 & ret_1,
    Geom_3242f8a8 & adj_geom1,
    Geom_3242f8a8 & adj_geom2,
    wp::int32 & adj_geomtype1,
    wp::int32 & adj_geomtype2,
    wp::vec_t<3, wp::float32> & adj_x_k,
    wp::float32 & adj_x_norm,
    wp::mat_t<4, 3, wp::float32> & adj_simplex,
    wp::int32 & adj_n,
    bool & adj_is_discrete,
    SupportPoint_e82efc60 & adj_ret_0,
    SupportPoint_e82efc60 & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:292
static CUDA_CALLABLE void adj__det3_0(
    wp::vec_t<3, wp::float32> var_v1,
    wp::vec_t<3, wp::float32> var_v2,
    wp::vec_t<3, wp::float32> var_v3,
    wp::vec_t<3, wp::float32> & adj_v1,
    wp::vec_t<3, wp::float32> & adj_v2,
    wp::vec_t<3, wp::float32> & adj_v3,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:297
static CUDA_CALLABLE void adj__same_sign_0(
    wp::float32 var_a,
    wp::float32 var_b,
    wp::float32 & adj_a,
    wp::float32 & adj_b,
    wp::int32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:313
static CUDA_CALLABLE void adj__project_origin_plane_0(
    wp::vec_t<3, wp::float32> var_v1,
    wp::vec_t<3, wp::float32> var_v2,
    wp::vec_t<3, wp::float32> var_v3,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::int32 & ret_1,
    wp::vec_t<3, wp::float32> & adj_v1,
    wp::vec_t<3, wp::float32> & adj_v2,
    wp::vec_t<3, wp::float32> & adj_v3,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::int32 & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:306
static CUDA_CALLABLE void adj__project_origin_line_0(
    wp::vec_t<3, wp::float32> var_v1,
    wp::vec_t<3, wp::float32> var_v2,
    wp::vec_t<3, wp::float32> & adj_v1,
    wp::vec_t<3, wp::float32> & adj_v2,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:566
static CUDA_CALLABLE void adj__S1D_0(
    wp::vec_t<3, wp::float32> var_s1,
    wp::vec_t<3, wp::float32> var_s2,
    wp::vec_t<3, wp::float32> & adj_s1,
    wp::vec_t<3, wp::float32> & adj_s2,
    wp::vec_t<2, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:421
static CUDA_CALLABLE void adj__S2D_0(
    wp::vec_t<3, wp::float32> var_s1,
    wp::vec_t<3, wp::float32> var_s2,
    wp::vec_t<3, wp::float32> var_s3,
    wp::vec_t<3, wp::float32> & adj_s1,
    wp::vec_t<3, wp::float32> & adj_s2,
    wp::vec_t<3, wp::float32> & adj_s3,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:345
static CUDA_CALLABLE void adj__S3D_0(
    wp::vec_t<3, wp::float32> var_s1,
    wp::vec_t<3, wp::float32> var_s2,
    wp::vec_t<3, wp::float32> var_s3,
    wp::vec_t<3, wp::float32> var_s4,
    wp::vec_t<3, wp::float32> & adj_s1,
    wp::vec_t<3, wp::float32> & adj_s2,
    wp::vec_t<3, wp::float32> & adj_s3,
    wp::vec_t<3, wp::float32> & adj_s4,
    wp::vec_t<4, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:279
static CUDA_CALLABLE void adj__subdistance_0(
    wp::int32 var_n,
    wp::mat_t<4, 3, wp::float32> var_simplex,
    wp::int32 & adj_n,
    wp::mat_t<4, 3, wp::float32> & adj_simplex,
    wp::vec_t<4, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:268
static CUDA_CALLABLE void adj__linear_combine_0(
    wp::int32 var_n,
    wp::vec_t<4, wp::float32> var_scl,
    wp::mat_t<4, 3, wp::float32> var_mat,
    wp::int32 & adj_n,
    wp::vec_t<4, wp::float32> & adj_scl,
    wp::mat_t<4, 3, wp::float32> & adj_mat,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:633
static CUDA_CALLABLE void adj_gjk_0(
    wp::float32 var_tolerance,
    wp::int32 var_gjk_iterations,
    Geom_3242f8a8 var_geom1,
    Geom_3242f8a8 var_geom2,
    wp::vec_t<3, wp::float32> var_x1_0,
    wp::vec_t<3, wp::float32> var_x2_0,
    wp::int32 var_geomtype1,
    wp::int32 var_geomtype2,
    wp::float32 var_cutoff,
    bool var_is_discrete,
    wp::float32 & adj_tolerance,
    wp::int32 & adj_gjk_iterations,
    Geom_3242f8a8 & adj_geom1,
    Geom_3242f8a8 & adj_geom2,
    wp::vec_t<3, wp::float32> & adj_x1_0,
    wp::vec_t<3, wp::float32> & adj_x2_0,
    wp::int32 & adj_geomtype1,
    wp::int32 & adj_geomtype2,
    wp::float32 & adj_cutoff,
    bool & adj_is_discrete,
    GJKResult_28609055 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:776
static CUDA_CALLABLE void adj__tri_affine_coord_0(
    wp::vec_t<3, wp::float32> var_v1,
    wp::vec_t<3, wp::float32> var_v2,
    wp::vec_t<3, wp::float32> var_v3,
    wp::vec_t<3, wp::float32> var_p,
    wp::vec_t<3, wp::float32> & adj_v1,
    wp::vec_t<3, wp::float32> & adj_v2,
    wp::vec_t<3, wp::float32> & adj_v3,
    wp::vec_t<3, wp::float32> & adj_p,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/warp/_src/math.py:0
static CUDA_CALLABLE void adj_norm_l2_0(
    wp::vec_t<3, wp::float32> var_v,
    wp::vec_t<3, wp::float32> & adj_v,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:2291
static CUDA_CALLABLE void adj__inflate_0(
    GJKResult_28609055 var_result,
    Geom_3242f8a8 var_geom1,
    Geom_3242f8a8 var_geom2,
    wp::int32 var_geomtype1,
    wp::int32 var_geomtype2,
    wp::float32 var_margin1,
    wp::float32 var_margin2,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    GJKResult_28609055 & adj_result,
    Geom_3242f8a8 & adj_geom1,
    Geom_3242f8a8 & adj_geom2,
    wp::int32 & adj_geomtype1,
    wp::int32 & adj_geomtype2,
    wp::float32 & adj_margin1,
    wp::float32 & adj_margin2,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:2338
static CUDA_CALLABLE void adj_gjk_phase_0(
    wp::float32 var_tolerance,
    wp::float32 var_cutoff,
    wp::int32 var_gjk_iterations,
    Geom_3242f8a8 var_geom1,
    Geom_3242f8a8 var_geom2,
    wp::int32 var_geomtype1,
    wp::int32 var_geomtype2,
    wp::vec_t<3, wp::float32> var_x_1,
    wp::vec_t<3, wp::float32> var_x_2,
    bool & ret_0,
    wp::float32 & ret_1,
    wp::int32 & ret_2,
    wp::vec_t<3, wp::float32> & ret_3,
    wp::vec_t<3, wp::float32> & ret_4,
    GJKResult_28609055 & ret_5,
    Geom_3242f8a8 & ret_6,
    Geom_3242f8a8 & ret_7,
    wp::float32 & adj_tolerance,
    wp::float32 & adj_cutoff,
    wp::int32 & adj_gjk_iterations,
    Geom_3242f8a8 & adj_geom1,
    Geom_3242f8a8 & adj_geom2,
    wp::int32 & adj_geomtype1,
    wp::int32 & adj_geomtype2,
    wp::vec_t<3, wp::float32> & adj_x_1,
    wp::vec_t<3, wp::float32> & adj_x_2,
    bool & adj_ret_0,
    wp::float32 & adj_ret_1,
    wp::int32 & adj_ret_2,
    wp::vec_t<3, wp::float32> & adj_ret_3,
    wp::vec_t<3, wp::float32> & adj_ret_4,
    GJKResult_28609055 & adj_ret_5,
    Geom_3242f8a8 & adj_ret_6,
    Geom_3242f8a8 & adj_ret_7)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:875
static CUDA_CALLABLE void adj__rotmat_0(
    wp::vec_t<3, wp::float32> var_axis,
    wp::vec_t<3, wp::float32> & adj_axis,
    wp::mat_t<3, 3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:251
static CUDA_CALLABLE void adj__epa_support_0(
    Polytope_10582b13 var_pt,
    wp::int32 var_idx,
    Geom_3242f8a8 var_geom1,
    Geom_3242f8a8 var_geom2,
    wp::int32 var_geom1_type,
    wp::int32 var_geom2_type,
    wp::vec_t<3, wp::float32> var_dir,
    wp::int32 & ret_0,
    wp::int32 & ret_1,
    Polytope_10582b13 & adj_pt,
    wp::int32 & adj_idx,
    Geom_3242f8a8 & adj_geom1,
    Geom_3242f8a8 & adj_geom2,
    wp::int32 & adj_geom1_type,
    wp::int32 & adj_geom2_type,
    wp::vec_t<3, wp::float32> & adj_dir,
    wp::int32 & adj_ret_0,
    wp::int32 & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:225
static CUDA_CALLABLE void adj__attach_face_0(
    Polytope_10582b13 var_pt,
    wp::int32 var_idx,
    wp::int32 var_v1,
    wp::int32 var_v2,
    wp::int32 var_v3,
    Polytope_10582b13 & adj_pt,
    wp::int32 & adj_idx,
    wp::int32 & adj_v1,
    wp::int32 & adj_v2,
    wp::int32 & adj_v3,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:836
static CUDA_CALLABLE void adj__replace_simplex3_0(
    Polytope_10582b13 var_pt,
    wp::int32 var_v1,
    wp::int32 var_v2,
    wp::int32 var_v3,
    Polytope_10582b13 & adj_pt,
    wp::int32 & adj_v1,
    wp::int32 & adj_v2,
    wp::int32 & adj_v3,
    GJKResult_28609055 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:897
static CUDA_CALLABLE void adj__ray_triangle_0(
    wp::vec_t<3, wp::float32> var_v1,
    wp::vec_t<3, wp::float32> var_v2,
    wp::vec_t<3, wp::float32> var_v3,
    wp::vec_t<3, wp::float32> var_v4,
    wp::vec_t<3, wp::float32> var_v5,
    wp::vec_t<3, wp::float32> & adj_v1,
    wp::vec_t<3, wp::float32> & adj_v2,
    wp::vec_t<3, wp::float32> & adj_v3,
    wp::vec_t<3, wp::float32> & adj_v4,
    wp::vec_t<3, wp::float32> & adj_v5,
    wp::int32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1011
static CUDA_CALLABLE void adj__polytope2_0(
    Polytope_10582b13 var_pt,
    wp::mat_t<4, 3, wp::float32> var_simplex,
    wp::mat_t<4, 3, wp::float32> var_simplex1,
    wp::mat_t<4, 3, wp::float32> var_simplex2,
    wp::vec_t<4, wp::int32> var_simplex_index1,
    wp::vec_t<4, wp::int32> var_simplex_index2,
    Geom_3242f8a8 var_geom1,
    Geom_3242f8a8 var_geom2,
    wp::int32 var_geomtype1,
    wp::int32 var_geomtype2,
    Polytope_10582b13 & ret_0,
    GJKResult_28609055 & ret_1,
    Polytope_10582b13 & adj_pt,
    wp::mat_t<4, 3, wp::float32> & adj_simplex,
    wp::mat_t<4, 3, wp::float32> & adj_simplex1,
    wp::mat_t<4, 3, wp::float32> & adj_simplex2,
    wp::vec_t<4, wp::int32> & adj_simplex_index1,
    wp::vec_t<4, wp::int32> & adj_simplex_index2,
    Geom_3242f8a8 & adj_geom1,
    Geom_3242f8a8 & adj_geom2,
    wp::int32 & adj_geomtype1,
    wp::int32 & adj_geomtype2,
    Polytope_10582b13 & adj_ret_0,
    GJKResult_28609055 & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:763
static CUDA_CALLABLE void adj__same_side_0(
    wp::vec_t<3, wp::float32> var_p0,
    wp::vec_t<3, wp::float32> var_p1,
    wp::vec_t<3, wp::float32> var_p2,
    wp::vec_t<3, wp::float32> var_p3,
    wp::vec_t<3, wp::float32> & adj_p0,
    wp::vec_t<3, wp::float32> & adj_p1,
    wp::vec_t<3, wp::float32> & adj_p2,
    wp::vec_t<3, wp::float32> & adj_p3,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:771
static CUDA_CALLABLE void adj__test_tetra_0(
    wp::vec_t<3, wp::float32> var_p0,
    wp::vec_t<3, wp::float32> var_p1,
    wp::vec_t<3, wp::float32> var_p2,
    wp::vec_t<3, wp::float32> var_p3,
    wp::vec_t<3, wp::float32> & adj_p0,
    wp::vec_t<3, wp::float32> & adj_p1,
    wp::vec_t<3, wp::float32> & adj_p2,
    wp::vec_t<3, wp::float32> & adj_p3,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1195
static CUDA_CALLABLE void adj__polytope4_0(
    Polytope_10582b13 var_pt,
    wp::mat_t<4, 3, wp::float32> var_simplex,
    wp::mat_t<4, 3, wp::float32> var_simplex1,
    wp::mat_t<4, 3, wp::float32> var_simplex2,
    wp::vec_t<4, wp::int32> var_simplex_index1,
    wp::vec_t<4, wp::int32> var_simplex_index2,
    Polytope_10582b13 & ret_0,
    GJKResult_28609055 & ret_1,
    Polytope_10582b13 & adj_pt,
    wp::mat_t<4, 3, wp::float32> & adj_simplex,
    wp::mat_t<4, 3, wp::float32> & adj_simplex1,
    wp::mat_t<4, 3, wp::float32> & adj_simplex2,
    wp::vec_t<4, wp::int32> & adj_simplex_index1,
    wp::vec_t<4, wp::int32> & adj_simplex_index2,
    Polytope_10582b13 & adj_ret_0,
    GJKResult_28609055 & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:819
static CUDA_CALLABLE void adj__tri_point_intersect_0(
    wp::vec_t<3, wp::float32> var_v1,
    wp::vec_t<3, wp::float32> var_v2,
    wp::vec_t<3, wp::float32> var_v3,
    wp::vec_t<3, wp::float32> var_p,
    wp::vec_t<3, wp::float32> & adj_v1,
    wp::vec_t<3, wp::float32> & adj_v2,
    wp::vec_t<3, wp::float32> & adj_v3,
    wp::vec_t<3, wp::float32> & adj_p,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1104
static CUDA_CALLABLE void adj__polytope3_0(
    Polytope_10582b13 var_pt,
    wp::float32 var_dist,
    wp::mat_t<4, 3, wp::float32> var_simplex,
    wp::mat_t<4, 3, wp::float32> var_simplex1,
    wp::mat_t<4, 3, wp::float32> var_simplex2,
    wp::vec_t<4, wp::int32> var_simplex_index1,
    wp::vec_t<4, wp::int32> var_simplex_index2,
    Geom_3242f8a8 var_geom1,
    Geom_3242f8a8 var_geom2,
    wp::int32 var_geomtype1,
    wp::int32 var_geomtype2,
    Polytope_10582b13 & adj_pt,
    wp::float32 & adj_dist,
    wp::mat_t<4, 3, wp::float32> & adj_simplex,
    wp::mat_t<4, 3, wp::float32> & adj_simplex1,
    wp::mat_t<4, 3, wp::float32> & adj_simplex2,
    wp::vec_t<4, wp::int32> & adj_simplex_index1,
    wp::vec_t<4, wp::int32> & adj_simplex_index2,
    Geom_3242f8a8 & adj_geom1,
    Geom_3242f8a8 & adj_geom2,
    wp::int32 & adj_geomtype1,
    wp::int32 & adj_geomtype2,
    Polytope_10582b13 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1301
static CUDA_CALLABLE void adj__is_invalid_face_0(
    wp::int32 var_face,
    wp::int32 & adj_face,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1283
static CUDA_CALLABLE void adj__delete_face_0(
    wp::int32 var_face,
    wp::int32 & adj_face,
    wp::int32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1277
static CUDA_CALLABLE void adj__get_face_verts_0(
    wp::int32 var_face,
    wp::int32 & adj_face,
    wp::vec_t<3, wp::int32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:915
static CUDA_CALLABLE void adj__add_edge_0(
    Polytope_10582b13 var_pt,
    wp::int32 var_e1,
    wp::int32 var_e2,
    Polytope_10582b13 & adj_pt,
    wp::int32 & adj_e1,
    wp::int32 & adj_e2,
    wp::int32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1289
static CUDA_CALLABLE void adj__is_face_deleted_0(
    wp::int32 var_face,
    wp::int32 & adj_face,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:910
static CUDA_CALLABLE void adj__get_edge_0(
    wp::int32 var_edge,
    wp::int32 & adj_edge,
    wp::vec_t<2, wp::int32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1295
static CUDA_CALLABLE void adj__invalidate_face_0(
    wp::int32 var_face,
    wp::int32 & adj_face,
    wp::int32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:937
static CUDA_CALLABLE void adj__epa_witness_0(
    Polytope_10582b13 var_pt,
    Geom_3242f8a8 var_geom1,
    Geom_3242f8a8 var_geom2,
    wp::int32 var_geomtype1,
    wp::int32 var_geomtype2,
    wp::int32 var_face_idx,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::float32 & ret_2,
    Polytope_10582b13 & adj_pt,
    Geom_3242f8a8 & adj_geom1,
    Geom_3242f8a8 & adj_geom2,
    wp::int32 & adj_geomtype1,
    wp::int32 & adj_geomtype2,
    wp::int32 & adj_face_idx,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::float32 & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1307
static CUDA_CALLABLE void adj__epa_0(
    wp::float32 var_tolerance,
    wp::int32 var_epa_iterations,
    Polytope_10582b13 var_pt,
    Geom_3242f8a8 var_geom1,
    Geom_3242f8a8 var_geom2,
    wp::int32 var_geomtype1,
    wp::int32 var_geomtype2,
    bool var_is_discrete,
    bool var_warn_overflow,
    wp::int32 var_worldid,
    wp::array_t<wp::int32> var_overflow_out,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::int32 & ret_3,
    wp::float32 & adj_tolerance,
    wp::int32 & adj_epa_iterations,
    Polytope_10582b13 & adj_pt,
    Geom_3242f8a8 & adj_geom1,
    Geom_3242f8a8 & adj_geom2,
    wp::int32 & adj_geomtype1,
    wp::int32 & adj_geomtype2,
    bool & adj_is_discrete,
    bool & adj_warn_overflow,
    wp::int32 & adj_worldid,
    wp::array_t<wp::int32> & adj_overflow_out,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2,
    wp::int32 & adj_ret_3)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:2405
static CUDA_CALLABLE void adj_epa_phase_0(
    wp::float32 var_tolerance,
    wp::int32 var_epa_iterations,
    GJKResult_28609055 var_result,
    Geom_3242f8a8 var_geom1,
    Geom_3242f8a8 var_geom2,
    wp::int32 var_geomtype1,
    wp::int32 var_geomtype2,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vert,
    wp::array_t<wp::int32> var_vert_index,
    wp::array_t<wp::int32> var_face,
    wp::array_t<wp::vec_t<3, wp::float32>> var_face_pr,
    wp::array_t<wp::float32> var_face_norm2,
    wp::array_t<wp::int32> var_horizon,
    bool var_warn_overflow,
    wp::int32 var_worldid,
    wp::array_t<wp::int32> var_overflow_out,
    wp::float32 & ret_0,
    wp::int32 & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & ret_3,
    wp::int32 & ret_4,
    wp::float32 & adj_tolerance,
    wp::int32 & adj_epa_iterations,
    GJKResult_28609055 & adj_result,
    Geom_3242f8a8 & adj_geom1,
    Geom_3242f8a8 & adj_geom2,
    wp::int32 & adj_geomtype1,
    wp::int32 & adj_geomtype2,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_vert,
    wp::array_t<wp::int32> & adj_vert_index,
    wp::array_t<wp::int32> & adj_face,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_face_pr,
    wp::array_t<wp::float32> & adj_face_norm2,
    wp::array_t<wp::int32> & adj_horizon,
    bool & adj_warn_overflow,
    wp::int32 & adj_worldid,
    wp::array_t<wp::int32> & adj_overflow_out,
    wp::float32 & adj_ret_0,
    wp::int32 & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2,
    wp::vec_t<3, wp::float32> & adj_ret_3,
    wp::int32 & adj_ret_4)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1491
static CUDA_CALLABLE void adj__feature_dim_0(
    wp::vec_t<3, wp::int32> var_face,
    wp::array_t<wp::int32> var_vert_index,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vert,
    wp::int32 var_offset,
    wp::int32 & ret_0,
    wp::vec_t<3, wp::int32> & ret_1,
    wp::mat_t<3, 3, wp::float32> & ret_2,
    wp::vec_t<3, wp::int32> & adj_face,
    wp::array_t<wp::int32> & adj_vert_index,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_vert,
    wp::int32 & adj_offset,
    wp::int32 & adj_ret_0,
    wp::vec_t<3, wp::int32> & adj_ret_1,
    wp::mat_t<3, 3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1691
static CUDA_CALLABLE void adj__box_normals2_0(
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_n,
    wp::array_t<wp::vec_t<3, wp::float32>> var_normal_out,
    wp::array_t<wp::int32> var_index_out,
    wp::mat_t<3, 3, wp::float32> & adj_mat,
    wp::vec_t<3, wp::float32> & adj_n,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_normal_out,
    wp::array_t<wp::int32> & adj_index_out,
    wp::int32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1724
static CUDA_CALLABLE void adj__box_normals_0(
    wp::int32 var_feature_dim,
    wp::vec_t<3, wp::int32> var_feature_index,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_dir,
    wp::array_t<wp::vec_t<3, wp::float32>> var_normal_out,
    wp::array_t<wp::int32> var_index_out,
    wp::int32 & adj_feature_dim,
    wp::vec_t<3, wp::int32> & adj_feature_index,
    wp::mat_t<3, 3, wp::float32> & adj_mat,
    wp::vec_t<3, wp::float32> & adj_dir,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_normal_out,
    wp::array_t<wp::int32> & adj_index_out,
    wp::int32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1544
static CUDA_CALLABLE void adj__intersect1_0(
    wp::array_t<wp::int32> var_a1,
    wp::array_t<wp::int32> var_a2,
    wp::int32 var_start1,
    wp::int32 var_start2,
    wp::int32 var_len1,
    wp::int32 var_len2,
    wp::int32 & ret_0,
    wp::vec_t<2, wp::int32> & ret_1,
    wp::array_t<wp::int32> & adj_a1,
    wp::array_t<wp::int32> & adj_a2,
    wp::int32 & adj_start1,
    wp::int32 & adj_start2,
    wp::int32 & adj_len1,
    wp::int32 & adj_len2,
    wp::int32 & adj_ret_0,
    wp::vec_t<2, wp::int32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1558
static CUDA_CALLABLE void adj__intersect2_0(
    wp::vec_t<2, wp::int32> var_a1,
    wp::array_t<wp::int32> var_a2,
    wp::int32 var_start2,
    wp::int32 var_len1,
    wp::int32 var_len2,
    wp::int32 & ret_0,
    wp::vec_t<2, wp::int32> & ret_1,
    wp::vec_t<2, wp::int32> & adj_a1,
    wp::array_t<wp::int32> & adj_a2,
    wp::int32 & adj_start2,
    wp::int32 & adj_len1,
    wp::int32 & adj_len2,
    wp::int32 & adj_ret_0,
    wp::vec_t<2, wp::int32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1573
static CUDA_CALLABLE void adj__mesh_normals_0(
    wp::int32 var_feature_dim,
    wp::vec_t<3, wp::int32> var_feature_index,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::int32 var_vertadr,
    wp::int32 var_polyadr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_polynormal,
    wp::array_t<wp::int32> var_polymapadr,
    wp::array_t<wp::int32> var_polymapnum,
    wp::array_t<wp::int32> var_polymap,
    wp::array_t<wp::vec_t<3, wp::float32>> var_normals_out,
    wp::array_t<wp::int32> var_indices_out,
    wp::int32 & adj_feature_dim,
    wp::vec_t<3, wp::int32> & adj_feature_index,
    wp::mat_t<3, 3, wp::float32> & adj_mat,
    wp::int32 & adj_vertadr,
    wp::int32 & adj_polyadr,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_polynormal,
    wp::array_t<wp::int32> & adj_polymapadr,
    wp::array_t<wp::int32> & adj_polymapnum,
    wp::array_t<wp::int32> & adj_polymap,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_normals_out,
    wp::array_t<wp::int32> & adj_indices_out,
    wp::int32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1517
static CUDA_CALLABLE void adj__aligned_faces_0(
    wp::array_t<wp::vec_t<3, wp::float32>> var_vert1,
    wp::int32 var_len1,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vert2,
    wp::int32 var_len2,
    wp::int32 & ret_0,
    wp::vec_t<2, wp::int32> & ret_1,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_vert1,
    wp::int32 & adj_len1,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_vert2,
    wp::int32 & adj_len2,
    wp::int32 & adj_ret_0,
    wp::vec_t<2, wp::int32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1798
static CUDA_CALLABLE void adj__box_edge_normals_0(
    wp::int32 var_dim,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_pos,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> var_v1,
    wp::vec_t<3, wp::float32> var_v2,
    wp::int32 var_v1i,
    wp::array_t<wp::vec_t<3, wp::float32>> var_normal_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_endvert_out,
    wp::int32 & adj_dim,
    wp::mat_t<3, 3, wp::float32> & adj_mat,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::vec_t<3, wp::float32> & adj_v1,
    wp::vec_t<3, wp::float32> & adj_v2,
    wp::int32 & adj_v1i,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_normal_out,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_endvert_out,
    wp::int32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1643
static CUDA_CALLABLE void adj__mesh_edge_normals_0(
    wp::int32 var_dim,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_pos,
    wp::int32 var_vertadr,
    wp::int32 var_polyadr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vert,
    wp::array_t<wp::int32> var_polyvertadr,
    wp::array_t<wp::int32> var_polyvertnum,
    wp::array_t<wp::int32> var_polyvert,
    wp::array_t<wp::int32> var_polymapadr,
    wp::array_t<wp::int32> var_polymapnum,
    wp::array_t<wp::int32> var_polymap,
    wp::vec_t<3, wp::float32> var_v1,
    wp::vec_t<3, wp::float32> var_v2,
    wp::int32 var_v1i,
    wp::array_t<wp::vec_t<3, wp::float32>> var_normals_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_endverts_out,
    wp::int32 & adj_dim,
    wp::mat_t<3, 3, wp::float32> & adj_mat,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::int32 & adj_vertadr,
    wp::int32 & adj_polyadr,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_vert,
    wp::array_t<wp::int32> & adj_polyvertadr,
    wp::array_t<wp::int32> & adj_polyvertnum,
    wp::array_t<wp::int32> & adj_polyvert,
    wp::array_t<wp::int32> & adj_polymapadr,
    wp::array_t<wp::int32> & adj_polymapnum,
    wp::array_t<wp::int32> & adj_polymap,
    wp::vec_t<3, wp::float32> & adj_v1,
    wp::vec_t<3, wp::float32> & adj_v2,
    wp::int32 & adj_v1i,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_normals_out,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_endverts_out,
    wp::int32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1531
static CUDA_CALLABLE void adj__aligned_face_edge_0(
    wp::array_t<wp::vec_t<3, wp::float32>> var_edge,
    wp::int32 var_nedge,
    wp::array_t<wp::vec_t<3, wp::float32>> var_face,
    wp::int32 var_nface,
    wp::int32 & ret_0,
    wp::vec_t<2, wp::int32> & ret_1,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_edge,
    wp::int32 & adj_nedge,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_face,
    wp::int32 & adj_nface,
    wp::int32 & adj_ret_0,
    wp::vec_t<2, wp::int32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:2047
static CUDA_CALLABLE void adj__set_edge_0(
    wp::array_t<wp::vec_t<3, wp::float32>> var_vert1,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vert2,
    wp::int32 var_start,
    wp::int32 var_end,
    wp::int32 var_offset,
    wp::array_t<wp::vec_t<3, wp::float32>> var_face_out,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_vert1,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_vert2,
    wp::int32 & adj_start,
    wp::int32 & adj_end,
    wp::int32 & adj_offset,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_face_out,
    wp::int32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1836
static CUDA_CALLABLE void adj__box_face_0(
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_pos,
    wp::vec_t<3, wp::float32> var_size,
    wp::int32 var_idx,
    wp::array_t<wp::vec_t<3, wp::float32>> var_face_out,
    wp::mat_t<3, 3, wp::float32> & adj_mat,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::int32 & adj_idx,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_face_out,
    wp::int32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1879
static CUDA_CALLABLE void adj__mesh_face_0(
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_pos,
    wp::int32 var_vertadr,
    wp::int32 var_polyadr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vert,
    wp::array_t<wp::int32> var_polyvertadr,
    wp::array_t<wp::int32> var_polyvertnum,
    wp::array_t<wp::int32> var_polyvert,
    wp::int32 var_idx,
    wp::array_t<wp::vec_t<3, wp::float32>> var_face_out,
    wp::mat_t<3, 3, wp::float32> & adj_mat,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::int32 & adj_vertadr,
    wp::int32 & adj_polyadr,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_vert,
    wp::array_t<wp::int32> & adj_polyvertadr,
    wp::array_t<wp::int32> & adj_polyvertnum,
    wp::array_t<wp::int32> & adj_polyvert,
    wp::int32 & adj_idx,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_face_out,
    wp::int32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1904
static CUDA_CALLABLE void adj__plane_normal_0(
    wp::vec_t<3, wp::float32> var_v1,
    wp::vec_t<3, wp::float32> var_v2,
    wp::vec_t<3, wp::float32> var_n,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & adj_v1,
    wp::vec_t<3, wp::float32> & adj_v2,
    wp::vec_t<3, wp::float32> & adj_n,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1911
static CUDA_CALLABLE void adj__halfspace_0(
    wp::vec_t<3, wp::float32> var_a,
    wp::vec_t<3, wp::float32> var_n,
    wp::vec_t<3, wp::float32> var_p,
    wp::vec_t<3, wp::float32> & adj_a,
    wp::vec_t<3, wp::float32> & adj_n,
    wp::vec_t<3, wp::float32> & adj_p,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1916
static CUDA_CALLABLE void adj__plane_intersect_0(
    wp::vec_t<3, wp::float32> var_pn,
    wp::float32 var_pd,
    wp::vec_t<3, wp::float32> var_a,
    wp::vec_t<3, wp::float32> var_b,
    wp::vec_t<3, wp::float32> & adj_pn,
    wp::float32 & adj_pd,
    wp::vec_t<3, wp::float32> & adj_a,
    wp::vec_t<3, wp::float32> & adj_b,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1445
static CUDA_CALLABLE void adj__area4_0(
    wp::vec_t<3, wp::float32> var_a,
    wp::vec_t<3, wp::float32> var_b,
    wp::vec_t<3, wp::float32> var_c,
    wp::vec_t<3, wp::float32> var_d,
    wp::vec_t<3, wp::float32> & adj_a,
    wp::vec_t<3, wp::float32> & adj_b,
    wp::vec_t<3, wp::float32> & adj_c,
    wp::vec_t<3, wp::float32> & adj_d,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1451
static CUDA_CALLABLE void adj__polygon_quad_0(
    wp::array_t<wp::vec_t<3, wp::float32>> var_polygon,
    wp::int32 var_npolygon,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_polygon,
    wp::int32 & adj_npolygon,
    wp::vec_t<4, wp::int32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:1929
static CUDA_CALLABLE void adj__polygon_clip_0(
    wp::array_t<wp::vec_t<3, wp::float32>> var_plane_normal,
    wp::array_t<wp::float32> var_plane_dist,
    wp::array_t<wp::vec_t<3, wp::float32>> var_face1,
    wp::int32 var_nface1,
    wp::array_t<wp::vec_t<3, wp::float32>> var_face2,
    wp::int32 var_nface2,
    wp::vec_t<3, wp::float32> var_n,
    wp::vec_t<3, wp::float32> var_dir,
    wp::array_t<wp::vec_t<3, wp::float32>> var_polygon_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_clipped_out,
    wp::int32 & ret_0,
    wp::mat_t<4, 3, wp::float32> & ret_1,
    wp::mat_t<4, 3, wp::float32> & ret_2,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_plane_normal,
    wp::array_t<wp::float32> & adj_plane_dist,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_face1,
    wp::int32 & adj_nface1,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_face2,
    wp::int32 & adj_nface2,
    wp::vec_t<3, wp::float32> & adj_n,
    wp::vec_t<3, wp::float32> & adj_dir,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_polygon_out,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_clipped_out,
    wp::int32 & adj_ret_0,
    wp::mat_t<4, 3, wp::float32> & adj_ret_1,
    wp::mat_t<4, 3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_gjk.py:2064
static CUDA_CALLABLE void adj_multicontact_0(
    wp::array_t<wp::vec_t<3, wp::float32>> var_polygon,
    wp::array_t<wp::vec_t<3, wp::float32>> var_clipped,
    wp::array_t<wp::vec_t<3, wp::float32>> var_plane_normal,
    wp::array_t<wp::float32> var_plane_dist,
    wp::array_t<wp::int32> var_idx1,
    wp::array_t<wp::int32> var_idx2,
    wp::array_t<wp::vec_t<3, wp::float32>> var_n1,
    wp::array_t<wp::vec_t<3, wp::float32>> var_n2,
    wp::array_t<wp::vec_t<3, wp::float32>> var_endvert,
    wp::array_t<wp::vec_t<3, wp::float32>> var_face1,
    wp::array_t<wp::vec_t<3, wp::float32>> var_face2,
    wp::array_t<wp::vec_t<3, wp::float32>> var_epa_vert,
    wp::array_t<wp::int32> var_epa_vert_index,
    wp::int32 var_epa_face,
    wp::vec_t<3, wp::float32> var_x1,
    wp::vec_t<3, wp::float32> var_x2,
    Geom_3242f8a8 var_geom1,
    Geom_3242f8a8 var_geom2,
    wp::int32 var_geomtype1,
    wp::int32 var_geomtype2,
    wp::int32 & ret_0,
    wp::mat_t<4, 3, wp::float32> & ret_1,
    wp::mat_t<4, 3, wp::float32> & ret_2,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_polygon,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_clipped,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_plane_normal,
    wp::array_t<wp::float32> & adj_plane_dist,
    wp::array_t<wp::int32> & adj_idx1,
    wp::array_t<wp::int32> & adj_idx2,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_n1,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_n2,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_endvert,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_face1,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_face2,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_epa_vert,
    wp::array_t<wp::int32> & adj_epa_vert_index,
    wp::int32 & adj_epa_face,
    wp::vec_t<3, wp::float32> & adj_x1,
    wp::vec_t<3, wp::float32> & adj_x2,
    Geom_3242f8a8 & adj_geom1,
    Geom_3242f8a8 & adj_geom2,
    wp::int32 & adj_geomtype1,
    wp::int32 & adj_geomtype2,
    wp::int32 & adj_ret_0,
    wp::mat_t<4, 3, wp::float32> & adj_ret_1,
    wp::mat_t<4, 3, wp::float32> & adj_ret_2)
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


// /root/autodl-tmp/zxg/perceptive_cbf_rl/.venv/lib/python3.11/site-packages/mujoco_warp/_src/collision_convex.py:733
static CUDA_CALLABLE void adj_ccd_kernel_builder__locals__eval_ccd_write_contact_0(
    wp::array_t<wp::float32> var_opt_ccd_tolerance,
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
    wp::int32 var_naconmax_in,
    wp::int32 var_naccdmax_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_epa_vert_in,
    wp::array_t<wp::int32> var_epa_vert_index_in,
    wp::array_t<wp::int32> var_epa_face_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_epa_pr_in,
    wp::array_t<wp::float32> var_epa_norm2_in,
    wp::array_t<wp::int32> var_epa_horizon_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_polygon_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_clipped_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_pnormal_in,
    wp::array_t<wp::float32> var_multiccd_pdist_in,
    wp::array_t<wp::int32> var_multiccd_idx1_in,
    wp::array_t<wp::int32> var_multiccd_idx2_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_n1_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_n2_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_endvert_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_face1_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_face2_in,
    Geom_3242f8a8 var_geom1,
    Geom_3242f8a8 var_geom2,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::int32 var_worldid,
    wp::array_t<wp::int32> var_nccd_in,
    wp::float32 var_margin,
    wp::float32 var_gap,
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
    wp::array_t<wp::int32> var_overflow_out,
    wp::array_t<wp::float32> & adj_opt_ccd_tolerance,
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
    wp::int32 & adj_naconmax_in,
    wp::int32 & adj_naccdmax_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_epa_vert_in,
    wp::array_t<wp::int32> & adj_epa_vert_index_in,
    wp::array_t<wp::int32> & adj_epa_face_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_epa_pr_in,
    wp::array_t<wp::float32> & adj_epa_norm2_in,
    wp::array_t<wp::int32> & adj_epa_horizon_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_multiccd_polygon_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_multiccd_clipped_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_multiccd_pnormal_in,
    wp::array_t<wp::float32> & adj_multiccd_pdist_in,
    wp::array_t<wp::int32> & adj_multiccd_idx1_in,
    wp::array_t<wp::int32> & adj_multiccd_idx2_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_multiccd_n1_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_multiccd_n2_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_multiccd_endvert_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_multiccd_face1_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_multiccd_face2_in,
    Geom_3242f8a8 & adj_geom1,
    Geom_3242f8a8 & adj_geom2,
    wp::vec_t<2, wp::int32> & adj_geoms,
    wp::int32 & adj_worldid,
    wp::array_t<wp::int32> & adj_nccd_in,
    wp::float32 & adj_margin,
    wp::float32 & adj_gap,
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
    wp::array_t<wp::int32> & adj_nacon_out,
    wp::array_t<wp::int32> & adj_overflow_out)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __launch_bounds__(64, 8) __global__ void ccd_kernel_builder__locals__ccd_kernel_7b4725dd_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::float32> var_opt_ccd_tolerance,
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
    wp::int32 var_naccdmax_in,
    wp::array_t<wp::int32> var_ncollision_in,
    wp::int32 var_grid_stride_in,
    wp::array_t<wp::vec_t<2, wp::int32>> var_collision_pair_in,
    wp::array_t<wp::vec_t<2, wp::int32>> var_collision_pairid_in,
    wp::array_t<wp::int32> var_collision_worldid_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_epa_vert_in,
    wp::array_t<wp::int32> var_epa_vert_index_in,
    wp::array_t<wp::int32> var_epa_face_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_epa_pr_in,
    wp::array_t<wp::float32> var_epa_norm2_in,
    wp::array_t<wp::int32> var_epa_horizon_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_polygon_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_clipped_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_pnormal_in,
    wp::array_t<wp::float32> var_multiccd_pdist_in,
    wp::array_t<wp::int32> var_multiccd_idx1_in,
    wp::array_t<wp::int32> var_multiccd_idx2_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_n1_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_n2_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_endvert_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_face1_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_face2_in,
    wp::array_t<wp::int32> var_nccd_in,
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
        const wp::int32 var_1 = 0;
        wp::int32* var_2;
        wp::range_t var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::vec_t<2, wp::int32>* var_6;
        wp::vec_t<2, wp::int32> var_7;
        wp::vec_t<2, wp::int32> var_8;
        const wp::int32 var_9 = 0;
        wp::int32 var_10;
        const wp::int32 var_11 = 1;
        wp::int32 var_12;
        bool var_13;
        wp::int32* var_14;
        const wp::int32 var_15 = 6;
        bool var_16;
        wp::int32 var_17;
        wp::int32* var_18;
        const wp::int32 var_19 = 6;
        bool var_20;
        wp::int32 var_21;
        wp::int32* var_22;
        wp::int32 var_23;
        wp::int32 var_24;
        wp::vec_t<2, wp::int32>* var_25;
        wp::vec_t<2, wp::int32> var_26;
        wp::vec_t<2, wp::int32> var_27;
        const wp::int32 var_28 = 0;
        wp::int32 var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        Geom_3242f8a8 var_32;
        Geom_3242f8a8 var_33;
        //---------
        // forward
        // def ccd_kernel(                                                                        <L 956>
        // tid = wp.tid()                                                                         <L 1039>
        var_0 = builtin_tid1d();
        // for collisionid in range(tid, ncollision_in[0], grid_stride_in):                       <L 1040>
        var_2 = wp::address(var_ncollision_in, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::range(var_0, var_4, var_grid_stride_in);
        start_for_0:;
            if (iter_cmp(var_3) == 0) goto end_for_0;
            var_5 = wp::iter_next(var_3);
            // geoms = collision_pair_in[collisionid]                                             <L 1041>
            var_6 = wp::address(var_collision_pair_in, var_5);
            var_8 = wp::load(var_6);
            var_7 = wp::copy(var_8);
            // g1 = geoms[0]                                                                      <L 1042>
            var_10 = wp::extract(var_7, var_9);
            // g2 = geoms[1]                                                                      <L 1043>
            var_12 = wp::extract(var_7, var_11);
            // if geom_type[g1] != geomtype1 or geom_type[g2] != geomtype2:                       <L 1045>
            var_14 = wp::address(var_geom_type, var_10);
            var_17 = wp::load(var_14);
            var_16 = (var_17 != var_15);
            var_13 = var_16;
            if (!var_13) {
                var_18 = wp::address(var_geom_type, var_12);
                var_21 = wp::load(var_18);
                var_20 = (var_21 != var_19);
                var_13 = var_13 || var_20;
            }
            if (var_13) {
                // continue                                                                       <L 1046>
                goto start_for_0;
            }
            // worldid = collision_worldid_in[collisionid]                                        <L 1048>
            var_22 = wp::address(var_collision_worldid_in, var_5);
            var_24 = wp::load(var_22);
            var_23 = wp::copy(var_24);
            // pairid = collision_pairid_in[collisionid]                                          <L 1049>
            var_25 = wp::address(var_collision_pairid_in, var_5);
            var_27 = wp::load(var_25);
            var_26 = wp::copy(var_27);
            // margin, gap = contact_margin_gap(                                                  <L 1050>
            // geom_margin,                                                                       <L 1051>
            // geom_gap,                                                                          <L 1052>
            // pair_margin,                                                                       <L 1053>
            // pair_gap,                                                                          <L 1054>
            // geoms,                                                                             <L 1055>
            // pairid[0],                                                                         <L 1056>
            var_29 = wp::extract(var_26, var_28);
            // worldid,                                                                           <L 1057>
            contact_margin_gap_0(var_geom_margin, var_geom_gap, var_pair_margin, var_pair_gap, var_7, var_29, var_23, var_30, var_31);
            // geom1, geom2 = geom_collision_pair_from_types(                                     <L 1060>
            // geom_dataid,                                                                       <L 1061>
            // geom_size,                                                                         <L 1062>
            // mesh_vertadr,                                                                      <L 1063>
            // mesh_vertnum,                                                                      <L 1064>
            // mesh_graphadr,                                                                     <L 1065>
            // mesh_vert,                                                                         <L 1066>
            // mesh_graph,                                                                        <L 1067>
            // mesh_polynum,                                                                      <L 1068>
            // mesh_polyadr,                                                                      <L 1069>
            // mesh_polynormal,                                                                   <L 1070>
            // mesh_polyvertadr,                                                                  <L 1071>
            // mesh_polyvertnum,                                                                  <L 1072>
            // mesh_polyvert,                                                                     <L 1073>
            // mesh_polymapadr,                                                                   <L 1074>
            // mesh_polymapnum,                                                                   <L 1075>
            // mesh_polymap,                                                                      <L 1076>
            // geom_xpos_in,                                                                      <L 1077>
            // geom_xmat_in,                                                                      <L 1078>
            // geomtype1,                                                                         <L 1079>
            // geomtype2,                                                                         <L 1080>
            // geoms,                                                                             <L 1081>
            // worldid,                                                                           <L 1082>
            geom_collision_pair_from_types_0(var_geom_dataid, var_geom_size, var_mesh_vertadr, var_mesh_vertnum, var_mesh_graphadr, var_mesh_vert, var_mesh_graph, var_mesh_polynum, var_mesh_polyadr, var_mesh_polynormal, var_mesh_polyvertadr, var_mesh_polyvertnum, var_mesh_polyvert, var_mesh_polymapadr, var_mesh_polymapnum, var_mesh_polymap, var_geom_xpos_in, var_geom_xmat_in, var_15, var_19, var_7, var_23, var_32, var_33);
            // eval_ccd_write_contact(                                                            <L 1085>
            // opt_ccd_tolerance,                                                                 <L 1086>
            // geom_condim,                                                                       <L 1087>
            // geom_priority,                                                                     <L 1088>
            // geom_solmix,                                                                       <L 1089>
            // geom_solref,                                                                       <L 1090>
            // geom_solimp,                                                                       <L 1091>
            // geom_friction,                                                                     <L 1092>
            // pair_dim,                                                                          <L 1093>
            // pair_solref,                                                                       <L 1094>
            // pair_solreffriction,                                                               <L 1095>
            // pair_solimp,                                                                       <L 1096>
            // pair_friction,                                                                     <L 1097>
            // naconmax_in,                                                                       <L 1098>
            // naccdmax_in,                                                                       <L 1099>
            // epa_vert_in,                                                                       <L 1100>
            // epa_vert_index_in,                                                                 <L 1101>
            // epa_face_in,                                                                       <L 1102>
            // epa_pr_in,                                                                         <L 1103>
            // epa_norm2_in,                                                                      <L 1104>
            // epa_horizon_in,                                                                    <L 1105>
            // multiccd_polygon_in,                                                               <L 1106>
            // multiccd_clipped_in,                                                               <L 1107>
            // multiccd_pnormal_in,                                                               <L 1108>
            // multiccd_pdist_in,                                                                 <L 1109>
            // multiccd_idx1_in,                                                                  <L 1110>
            // multiccd_idx2_in,                                                                  <L 1111>
            // multiccd_n1_in,                                                                    <L 1112>
            // multiccd_n2_in,                                                                    <L 1113>
            // multiccd_endvert_in,                                                               <L 1114>
            // multiccd_face1_in,                                                                 <L 1115>
            // multiccd_face2_in,                                                                 <L 1116>
            // geom1,                                                                             <L 1117>
            // geom2,                                                                             <L 1118>
            // geoms,                                                                             <L 1119>
            // worldid,                                                                           <L 1120>
            // nccd_in,                                                                           <L 1121>
            // margin,                                                                            <L 1122>
            // gap,                                                                               <L 1123>
            // pairid,                                                                            <L 1124>
            // contact_dist_out,                                                                  <L 1125>
            // contact_pos_out,                                                                   <L 1126>
            // contact_frame_out,                                                                 <L 1127>
            // contact_includemargin_out,                                                         <L 1128>
            // contact_friction_out,                                                              <L 1129>
            // contact_solref_out,                                                                <L 1130>
            // contact_solreffriction_out,                                                        <L 1131>
            // contact_solimp_out,                                                                <L 1132>
            // contact_dim_out,                                                                   <L 1133>
            // contact_geom_out,                                                                  <L 1134>
            // contact_efc_address_out,                                                           <L 1135>
            // contact_worldid_out,                                                               <L 1136>
            // contact_type_out,                                                                  <L 1137>
            // contact_geomcollisionid_out,                                                       <L 1138>
            // nacon_out,                                                                         <L 1139>
            // overflow_out,                                                                      <L 1140>
            ccd_kernel_builder__locals__eval_ccd_write_contact_0(var_opt_ccd_tolerance, var_geom_condim, var_geom_priority, var_geom_solmix, var_geom_solref, var_geom_solimp, var_geom_friction, var_pair_dim, var_pair_solref, var_pair_solreffriction, var_pair_solimp, var_pair_friction, var_naconmax_in, var_naccdmax_in, var_epa_vert_in, var_epa_vert_index_in, var_epa_face_in, var_epa_pr_in, var_epa_norm2_in, var_epa_horizon_in, var_multiccd_polygon_in, var_multiccd_clipped_in, var_multiccd_pnormal_in, var_multiccd_pdist_in, var_multiccd_idx1_in, var_multiccd_idx2_in, var_multiccd_n1_in, var_multiccd_n2_in, var_multiccd_endvert_in, var_multiccd_face1_in, var_multiccd_face2_in, var_32, var_33, var_7, var_23, var_nccd_in, var_30, var_31, var_26, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_nacon_out, var_overflow_out);
            goto start_for_0;
        end_for_0:;
    }
}

