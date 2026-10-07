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



extern "C" __global__ void _small_cholesky_factorize_block__locals__kernel_740481bd_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_M_rowadr,
    wp::array_t<wp::int32> var_qLD_block_adr,
    wp::array_t<wp::float32> var_M_in,
    wp::array_t<wp::int32> var_block_dof,
    wp::array_t<wp::float32> var_D_out,
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
        const wp::int32 var_5 = 6;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        const wp::int32 var_12 = -2;
        bool var_13;
        const wp::int32 var_14 = 0;
        const wp::float32 var_15 = 1.0;
        wp::int32 var_16;
        wp::float32* var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::int32 var_20;
        const wp::int32 var_21 = 1;
        const wp::float32 var_22 = 1.0;
        wp::int32 var_23;
        wp::float32* var_24;
        wp::float32 var_25;
        wp::float32 var_26;
        wp::int32 var_27;
        const wp::int32 var_28 = 2;
        const wp::float32 var_29 = 1.0;
        wp::int32 var_30;
        wp::float32* var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        wp::int32 var_34;
        const wp::int32 var_35 = 3;
        const wp::float32 var_36 = 1.0;
        wp::int32 var_37;
        wp::float32* var_38;
        wp::float32 var_39;
        wp::float32 var_40;
        wp::int32 var_41;
        const wp::int32 var_42 = 4;
        const wp::float32 var_43 = 1.0;
        wp::int32 var_44;
        wp::float32* var_45;
        wp::float32 var_46;
        wp::float32 var_47;
        wp::int32 var_48;
        const wp::int32 var_49 = 5;
        const wp::float32 var_50 = 1.0;
        wp::int32 var_51;
        wp::float32* var_52;
        wp::float32 var_53;
        wp::float32 var_54;
        wp::int32 var_55;
        const wp::int32 var_56 = 0;
        const wp::int32 var_57 = 1;
        wp::int32 var_58;
        wp::int32 var_59;
        const wp::int32 var_60 = 2;
        wp::int32 var_61;
        wp::int32 var_62;
        wp::int32 var_63;
        wp::float32* var_64;
        wp::float32 var_65;
        wp::float32 var_66;
        wp::float32 var_67;
        wp::int32 var_68;
        wp::int32 var_69;
        wp::int32 var_70;
        const wp::float32 var_71 = 1.0;
        wp::float32 var_72;
        const wp::int32 var_73 = 1;
        wp::int32 var_74;
        const wp::int32 var_75 = 6;
        wp::range_t var_76;
        wp::int32 var_77;
        const wp::int32 var_78 = 1;
        wp::int32 var_79;
        wp::int32 var_80;
        const wp::int32 var_81 = 2;
        wp::int32 var_82;
        wp::int32 var_83;
        wp::int32 var_84;
        wp::float32* var_85;
        wp::float32 var_86;
        wp::float32 var_87;
        wp::float32 var_88;
        wp::int32 var_89;
        wp::int32 var_90;
        wp::int32 var_91;
        const wp::int32 var_92 = 1;
        const wp::int32 var_93 = 1;
        wp::int32 var_94;
        wp::int32 var_95;
        const wp::int32 var_96 = 2;
        wp::int32 var_97;
        wp::int32 var_98;
        wp::int32 var_99;
        wp::float32* var_100;
        wp::float32 var_101;
        wp::float32 var_102;
        const wp::int32 var_103 = 0;
        wp::int32 var_104;
        wp::int32 var_105;
        wp::int32 var_106;
        wp::float32* var_107;
        wp::float32 var_108;
        wp::float32 var_109;
        wp::float32 var_110;
        wp::float32 var_111;
        wp::float32 var_112;
        wp::int32 var_113;
        wp::int32 var_114;
        wp::int32 var_115;
        const wp::float32 var_116 = 1.0;
        wp::float32 var_117;
        const wp::int32 var_118 = 1;
        wp::int32 var_119;
        const wp::int32 var_120 = 6;
        wp::range_t var_121;
        wp::int32 var_122;
        const wp::int32 var_123 = 1;
        wp::int32 var_124;
        wp::int32 var_125;
        const wp::int32 var_126 = 2;
        wp::int32 var_127;
        wp::int32 var_128;
        wp::int32 var_129;
        wp::float32* var_130;
        wp::float32 var_131;
        wp::float32 var_132;
        const wp::int32 var_133 = 0;
        wp::int32 var_134;
        wp::int32 var_135;
        wp::int32 var_136;
        wp::float32* var_137;
        wp::int32 var_138;
        wp::int32 var_139;
        wp::int32 var_140;
        wp::float32* var_141;
        wp::float32 var_142;
        wp::float32 var_143;
        wp::float32 var_144;
        wp::float32 var_145;
        wp::float32 var_146;
        wp::int32 var_147;
        wp::int32 var_148;
        wp::int32 var_149;
        const wp::int32 var_150 = 2;
        const wp::int32 var_151 = 1;
        wp::int32 var_152;
        wp::int32 var_153;
        const wp::int32 var_154 = 2;
        wp::int32 var_155;
        wp::int32 var_156;
        wp::int32 var_157;
        wp::float32* var_158;
        wp::float32 var_159;
        wp::float32 var_160;
        const wp::int32 var_161 = 0;
        wp::int32 var_162;
        wp::int32 var_163;
        wp::int32 var_164;
        wp::float32* var_165;
        wp::float32 var_166;
        wp::float32 var_167;
        wp::float32 var_168;
        wp::float32 var_169;
        const wp::int32 var_170 = 1;
        wp::int32 var_171;
        wp::int32 var_172;
        wp::int32 var_173;
        wp::float32* var_174;
        wp::float32 var_175;
        wp::float32 var_176;
        wp::float32 var_177;
        wp::float32 var_178;
        wp::float32 var_179;
        wp::int32 var_180;
        wp::int32 var_181;
        wp::int32 var_182;
        const wp::float32 var_183 = 1.0;
        wp::float32 var_184;
        const wp::int32 var_185 = 1;
        wp::int32 var_186;
        const wp::int32 var_187 = 6;
        wp::range_t var_188;
        wp::int32 var_189;
        const wp::int32 var_190 = 1;
        wp::int32 var_191;
        wp::int32 var_192;
        const wp::int32 var_193 = 2;
        wp::int32 var_194;
        wp::int32 var_195;
        wp::int32 var_196;
        wp::float32* var_197;
        wp::float32 var_198;
        wp::float32 var_199;
        const wp::int32 var_200 = 0;
        wp::int32 var_201;
        wp::int32 var_202;
        wp::int32 var_203;
        wp::float32* var_204;
        wp::int32 var_205;
        wp::int32 var_206;
        wp::int32 var_207;
        wp::float32* var_208;
        wp::float32 var_209;
        wp::float32 var_210;
        wp::float32 var_211;
        wp::float32 var_212;
        const wp::int32 var_213 = 1;
        wp::int32 var_214;
        wp::int32 var_215;
        wp::int32 var_216;
        wp::float32* var_217;
        wp::int32 var_218;
        wp::int32 var_219;
        wp::int32 var_220;
        wp::float32* var_221;
        wp::float32 var_222;
        wp::float32 var_223;
        wp::float32 var_224;
        wp::float32 var_225;
        wp::float32 var_226;
        wp::int32 var_227;
        wp::int32 var_228;
        wp::int32 var_229;
        const wp::int32 var_230 = 3;
        const wp::int32 var_231 = 1;
        wp::int32 var_232;
        wp::int32 var_233;
        const wp::int32 var_234 = 2;
        wp::int32 var_235;
        wp::int32 var_236;
        wp::int32 var_237;
        wp::float32* var_238;
        wp::float32 var_239;
        wp::float32 var_240;
        const wp::int32 var_241 = 0;
        wp::int32 var_242;
        wp::int32 var_243;
        wp::int32 var_244;
        wp::float32* var_245;
        wp::float32 var_246;
        wp::float32 var_247;
        wp::float32 var_248;
        wp::float32 var_249;
        const wp::int32 var_250 = 1;
        wp::int32 var_251;
        wp::int32 var_252;
        wp::int32 var_253;
        wp::float32* var_254;
        wp::float32 var_255;
        wp::float32 var_256;
        wp::float32 var_257;
        wp::float32 var_258;
        const wp::int32 var_259 = 2;
        wp::int32 var_260;
        wp::int32 var_261;
        wp::int32 var_262;
        wp::float32* var_263;
        wp::float32 var_264;
        wp::float32 var_265;
        wp::float32 var_266;
        wp::float32 var_267;
        wp::float32 var_268;
        wp::int32 var_269;
        wp::int32 var_270;
        wp::int32 var_271;
        const wp::float32 var_272 = 1.0;
        wp::float32 var_273;
        const wp::int32 var_274 = 1;
        wp::int32 var_275;
        const wp::int32 var_276 = 6;
        wp::range_t var_277;
        wp::int32 var_278;
        const wp::int32 var_279 = 1;
        wp::int32 var_280;
        wp::int32 var_281;
        const wp::int32 var_282 = 2;
        wp::int32 var_283;
        wp::int32 var_284;
        wp::int32 var_285;
        wp::float32* var_286;
        wp::float32 var_287;
        wp::float32 var_288;
        const wp::int32 var_289 = 0;
        wp::int32 var_290;
        wp::int32 var_291;
        wp::int32 var_292;
        wp::float32* var_293;
        wp::int32 var_294;
        wp::int32 var_295;
        wp::int32 var_296;
        wp::float32* var_297;
        wp::float32 var_298;
        wp::float32 var_299;
        wp::float32 var_300;
        wp::float32 var_301;
        const wp::int32 var_302 = 1;
        wp::int32 var_303;
        wp::int32 var_304;
        wp::int32 var_305;
        wp::float32* var_306;
        wp::int32 var_307;
        wp::int32 var_308;
        wp::int32 var_309;
        wp::float32* var_310;
        wp::float32 var_311;
        wp::float32 var_312;
        wp::float32 var_313;
        wp::float32 var_314;
        const wp::int32 var_315 = 2;
        wp::int32 var_316;
        wp::int32 var_317;
        wp::int32 var_318;
        wp::float32* var_319;
        wp::int32 var_320;
        wp::int32 var_321;
        wp::int32 var_322;
        wp::float32* var_323;
        wp::float32 var_324;
        wp::float32 var_325;
        wp::float32 var_326;
        wp::float32 var_327;
        wp::float32 var_328;
        wp::int32 var_329;
        wp::int32 var_330;
        wp::int32 var_331;
        const wp::int32 var_332 = 4;
        const wp::int32 var_333 = 1;
        wp::int32 var_334;
        wp::int32 var_335;
        const wp::int32 var_336 = 2;
        wp::int32 var_337;
        wp::int32 var_338;
        wp::int32 var_339;
        wp::float32* var_340;
        wp::float32 var_341;
        wp::float32 var_342;
        const wp::int32 var_343 = 0;
        wp::int32 var_344;
        wp::int32 var_345;
        wp::int32 var_346;
        wp::float32* var_347;
        wp::float32 var_348;
        wp::float32 var_349;
        wp::float32 var_350;
        wp::float32 var_351;
        const wp::int32 var_352 = 1;
        wp::int32 var_353;
        wp::int32 var_354;
        wp::int32 var_355;
        wp::float32* var_356;
        wp::float32 var_357;
        wp::float32 var_358;
        wp::float32 var_359;
        wp::float32 var_360;
        const wp::int32 var_361 = 2;
        wp::int32 var_362;
        wp::int32 var_363;
        wp::int32 var_364;
        wp::float32* var_365;
        wp::float32 var_366;
        wp::float32 var_367;
        wp::float32 var_368;
        wp::float32 var_369;
        const wp::int32 var_370 = 3;
        wp::int32 var_371;
        wp::int32 var_372;
        wp::int32 var_373;
        wp::float32* var_374;
        wp::float32 var_375;
        wp::float32 var_376;
        wp::float32 var_377;
        wp::float32 var_378;
        wp::float32 var_379;
        wp::int32 var_380;
        wp::int32 var_381;
        wp::int32 var_382;
        const wp::float32 var_383 = 1.0;
        wp::float32 var_384;
        const wp::int32 var_385 = 1;
        wp::int32 var_386;
        const wp::int32 var_387 = 6;
        wp::range_t var_388;
        wp::int32 var_389;
        const wp::int32 var_390 = 1;
        wp::int32 var_391;
        wp::int32 var_392;
        const wp::int32 var_393 = 2;
        wp::int32 var_394;
        wp::int32 var_395;
        wp::int32 var_396;
        wp::float32* var_397;
        wp::float32 var_398;
        wp::float32 var_399;
        const wp::int32 var_400 = 0;
        wp::int32 var_401;
        wp::int32 var_402;
        wp::int32 var_403;
        wp::float32* var_404;
        wp::int32 var_405;
        wp::int32 var_406;
        wp::int32 var_407;
        wp::float32* var_408;
        wp::float32 var_409;
        wp::float32 var_410;
        wp::float32 var_411;
        wp::float32 var_412;
        const wp::int32 var_413 = 1;
        wp::int32 var_414;
        wp::int32 var_415;
        wp::int32 var_416;
        wp::float32* var_417;
        wp::int32 var_418;
        wp::int32 var_419;
        wp::int32 var_420;
        wp::float32* var_421;
        wp::float32 var_422;
        wp::float32 var_423;
        wp::float32 var_424;
        wp::float32 var_425;
        const wp::int32 var_426 = 2;
        wp::int32 var_427;
        wp::int32 var_428;
        wp::int32 var_429;
        wp::float32* var_430;
        wp::int32 var_431;
        wp::int32 var_432;
        wp::int32 var_433;
        wp::float32* var_434;
        wp::float32 var_435;
        wp::float32 var_436;
        wp::float32 var_437;
        wp::float32 var_438;
        const wp::int32 var_439 = 3;
        wp::int32 var_440;
        wp::int32 var_441;
        wp::int32 var_442;
        wp::float32* var_443;
        wp::int32 var_444;
        wp::int32 var_445;
        wp::int32 var_446;
        wp::float32* var_447;
        wp::float32 var_448;
        wp::float32 var_449;
        wp::float32 var_450;
        wp::float32 var_451;
        wp::float32 var_452;
        wp::int32 var_453;
        wp::int32 var_454;
        wp::int32 var_455;
        const wp::int32 var_456 = 5;
        const wp::int32 var_457 = 1;
        wp::int32 var_458;
        wp::int32 var_459;
        const wp::int32 var_460 = 2;
        wp::int32 var_461;
        wp::int32 var_462;
        wp::int32 var_463;
        wp::float32* var_464;
        wp::float32 var_465;
        wp::float32 var_466;
        const wp::int32 var_467 = 0;
        wp::int32 var_468;
        wp::int32 var_469;
        wp::int32 var_470;
        wp::float32* var_471;
        wp::float32 var_472;
        wp::float32 var_473;
        wp::float32 var_474;
        wp::float32 var_475;
        const wp::int32 var_476 = 1;
        wp::int32 var_477;
        wp::int32 var_478;
        wp::int32 var_479;
        wp::float32* var_480;
        wp::float32 var_481;
        wp::float32 var_482;
        wp::float32 var_483;
        wp::float32 var_484;
        const wp::int32 var_485 = 2;
        wp::int32 var_486;
        wp::int32 var_487;
        wp::int32 var_488;
        wp::float32* var_489;
        wp::float32 var_490;
        wp::float32 var_491;
        wp::float32 var_492;
        wp::float32 var_493;
        const wp::int32 var_494 = 3;
        wp::int32 var_495;
        wp::int32 var_496;
        wp::int32 var_497;
        wp::float32* var_498;
        wp::float32 var_499;
        wp::float32 var_500;
        wp::float32 var_501;
        wp::float32 var_502;
        const wp::int32 var_503 = 4;
        wp::int32 var_504;
        wp::int32 var_505;
        wp::int32 var_506;
        wp::float32* var_507;
        wp::float32 var_508;
        wp::float32 var_509;
        wp::float32 var_510;
        wp::float32 var_511;
        wp::float32 var_512;
        wp::int32 var_513;
        wp::int32 var_514;
        wp::int32 var_515;
        const wp::float32 var_516 = 1.0;
        wp::float32 var_517;
        const wp::int32 var_518 = 1;
        wp::int32 var_519;
        const wp::int32 var_520 = 6;
        wp::range_t var_521;
        wp::int32 var_522;
        const wp::int32 var_523 = 1;
        wp::int32 var_524;
        wp::int32 var_525;
        const wp::int32 var_526 = 2;
        wp::int32 var_527;
        wp::int32 var_528;
        wp::int32 var_529;
        wp::float32* var_530;
        wp::float32 var_531;
        wp::float32 var_532;
        const wp::int32 var_533 = 0;
        wp::int32 var_534;
        wp::int32 var_535;
        wp::int32 var_536;
        wp::float32* var_537;
        wp::int32 var_538;
        wp::int32 var_539;
        wp::int32 var_540;
        wp::float32* var_541;
        wp::float32 var_542;
        wp::float32 var_543;
        wp::float32 var_544;
        wp::float32 var_545;
        const wp::int32 var_546 = 1;
        wp::int32 var_547;
        wp::int32 var_548;
        wp::int32 var_549;
        wp::float32* var_550;
        wp::int32 var_551;
        wp::int32 var_552;
        wp::int32 var_553;
        wp::float32* var_554;
        wp::float32 var_555;
        wp::float32 var_556;
        wp::float32 var_557;
        wp::float32 var_558;
        const wp::int32 var_559 = 2;
        wp::int32 var_560;
        wp::int32 var_561;
        wp::int32 var_562;
        wp::float32* var_563;
        wp::int32 var_564;
        wp::int32 var_565;
        wp::int32 var_566;
        wp::float32* var_567;
        wp::float32 var_568;
        wp::float32 var_569;
        wp::float32 var_570;
        wp::float32 var_571;
        const wp::int32 var_572 = 3;
        wp::int32 var_573;
        wp::int32 var_574;
        wp::int32 var_575;
        wp::float32* var_576;
        wp::int32 var_577;
        wp::int32 var_578;
        wp::int32 var_579;
        wp::float32* var_580;
        wp::float32 var_581;
        wp::float32 var_582;
        wp::float32 var_583;
        wp::float32 var_584;
        const wp::int32 var_585 = 4;
        wp::int32 var_586;
        wp::int32 var_587;
        wp::int32 var_588;
        wp::float32* var_589;
        wp::int32 var_590;
        wp::int32 var_591;
        wp::int32 var_592;
        wp::float32* var_593;
        wp::float32 var_594;
        wp::float32 var_595;
        wp::float32 var_596;
        wp::float32 var_597;
        wp::float32 var_598;
        wp::int32 var_599;
        wp::int32 var_600;
        wp::int32 var_601;
        wp::int32 var_602;
        //---------
        // forward
        // def kernel(                                                                            <L 1238>
        // worldid, blk = wp.tid()                                                                <L 1250>
        builtin_tid2d(var_0, var_1);
        // start = block_dof[blk]                                                                 <L 1251>
        var_2 = wp::address(var_block_dof, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // size = wp.static(block_size)                                                           <L 1252>
        // matrix_adr = M_rowadr[start]                                                           <L 1253>
        var_6 = wp::address(var_M_rowadr, var_3);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // factor_adr = qLD_block_adr[start]                                                      <L 1255>
        var_9 = wp::address(var_qLD_block_adr, var_3);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // if factor_adr == Q_LD_BLOCK_COMPACT:                                                   <L 1256>
        var_13 = (var_10 == var_12);
        if (var_13) {
            // for i in range(wp.static(block_size)):                                             <L 1257>
            // D_out[worldid, start + i] = 1.0 / M_in[worldid, matrix_adr + i]                    <L 1258>
            var_16 = wp::add(var_7, var_14);
            var_17 = wp::address(var_M_in, var_0, var_16);
            var_19 = wp::load(var_17);
            var_18 = wp::div(var_15, var_19);
            var_20 = wp::add(var_3, var_14);
            wp::array_store(var_D_out, var_0, var_20, var_18);
            var_23 = wp::add(var_7, var_21);
            var_24 = wp::address(var_M_in, var_0, var_23);
            var_26 = wp::load(var_24);
            var_25 = wp::div(var_22, var_26);
            var_27 = wp::add(var_3, var_21);
            wp::array_store(var_D_out, var_0, var_27, var_25);
            var_30 = wp::add(var_7, var_28);
            var_31 = wp::address(var_M_in, var_0, var_30);
            var_33 = wp::load(var_31);
            var_32 = wp::div(var_29, var_33);
            var_34 = wp::add(var_3, var_28);
            wp::array_store(var_D_out, var_0, var_34, var_32);
            var_37 = wp::add(var_7, var_35);
            var_38 = wp::address(var_M_in, var_0, var_37);
            var_40 = wp::load(var_38);
            var_39 = wp::div(var_36, var_40);
            var_41 = wp::add(var_3, var_35);
            wp::array_store(var_D_out, var_0, var_41, var_39);
            var_44 = wp::add(var_7, var_42);
            var_45 = wp::address(var_M_in, var_0, var_44);
            var_47 = wp::load(var_45);
            var_46 = wp::div(var_43, var_47);
            var_48 = wp::add(var_3, var_42);
            wp::array_store(var_D_out, var_0, var_48, var_46);
            var_51 = wp::add(var_7, var_49);
            var_52 = wp::address(var_M_in, var_0, var_51);
            var_54 = wp::load(var_52);
            var_53 = wp::div(var_50, var_54);
            var_55 = wp::add(var_3, var_49);
            wp::array_store(var_D_out, var_0, var_55, var_53);
        }
        if (!var_13) {
            // for i in range(wp.static(block_size)):                                             <L 1260>
            // value = M_in[worldid, matrix_adr + i * (i + 1) // 2 + i]                           <L 1261>
            var_58 = wp::add(var_56, var_57);
            var_59 = wp::mul(var_56, var_58);
            var_61 = wp::floordiv(var_59, var_60);
            var_62 = wp::add(var_7, var_61);
            var_63 = wp::add(var_62, var_56);
            var_64 = wp::address(var_M_in, var_0, var_63);
            var_66 = wp::load(var_64);
            var_65 = wp::copy(var_66);
            // for k in range(i):                                                                 <L 1262>
            // diagonal_value = wp.sqrt(value)                                                    <L 1266>
            var_67 = wp::sqrt(var_65);
            // L_out[worldid, factor_adr + i * size + i] = diagonal_value                         <L 1267>
            var_68 = wp::mul(var_56, var_5);
            var_69 = wp::add(var_10, var_68);
            var_70 = wp::add(var_69, var_56);
            wp::array_store(var_L_out, var_0, var_70, var_67);
            // diagonal_inv = 1.0 / diagonal_value                                                <L 1268>
            var_72 = wp::div(var_71, var_67);
            // for j in range(i + 1, size):                                                       <L 1270>
            var_74 = wp::add(var_56, var_73);
            var_76 = wp::range(var_74, var_75);
            start_for_0:;
                if (iter_cmp(var_76) == 0) goto end_for_0;
                var_77 = wp::iter_next(var_76);
                // value = M_in[worldid, matrix_adr + j * (j + 1) // 2 + i]                       <L 1271>
                var_79 = wp::add(var_77, var_78);
                var_80 = wp::mul(var_77, var_79);
                var_82 = wp::floordiv(var_80, var_81);
                var_83 = wp::add(var_7, var_82);
                var_84 = wp::add(var_83, var_56);
                var_85 = wp::address(var_M_in, var_0, var_84);
                var_87 = wp::load(var_85);
                var_86 = wp::copy(var_87);
                // for k in range(i):                                                             <L 1272>
                // L_out[worldid, factor_adr + i * size + j] = value * diagonal_inv               <L 1274>
                var_88 = wp::mul(var_86, var_72);
                var_89 = wp::mul(var_56, var_5);
                var_90 = wp::add(var_10, var_89);
                var_91 = wp::add(var_90, var_77);
                wp::array_store(var_L_out, var_0, var_91, var_88);
                wp::assign(var_65, var_86);
                goto start_for_0;
            end_for_0:;
            // value = M_in[worldid, matrix_adr + i * (i + 1) // 2 + i]                           <L 1261>
            var_94 = wp::add(var_92, var_93);
            var_95 = wp::mul(var_92, var_94);
            var_97 = wp::floordiv(var_95, var_96);
            var_98 = wp::add(var_7, var_97);
            var_99 = wp::add(var_98, var_92);
            var_100 = wp::address(var_M_in, var_0, var_99);
            var_102 = wp::load(var_100);
            var_101 = wp::copy(var_102);
            // for k in range(i):                                                                 <L 1262>
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 1263>
            var_104 = wp::mul(var_103, var_5);
            var_105 = wp::add(var_10, var_104);
            var_106 = wp::add(var_105, var_92);
            var_107 = wp::address(var_L_out, var_0, var_106);
            var_109 = wp::load(var_107);
            var_108 = wp::copy(var_109);
            // value -= factor * factor                                                           <L 1264>
            var_110 = wp::mul(var_108, var_108);
            var_111 = wp::sub(var_101, var_110);
            // diagonal_value = wp.sqrt(value)                                                    <L 1266>
            var_112 = wp::sqrt(var_111);
            // L_out[worldid, factor_adr + i * size + i] = diagonal_value                         <L 1267>
            var_113 = wp::mul(var_92, var_5);
            var_114 = wp::add(var_10, var_113);
            var_115 = wp::add(var_114, var_92);
            wp::array_store(var_L_out, var_0, var_115, var_112);
            // diagonal_inv = 1.0 / diagonal_value                                                <L 1268>
            var_117 = wp::div(var_116, var_112);
            // for j in range(i + 1, size):                                                       <L 1270>
            var_119 = wp::add(var_92, var_118);
            var_121 = wp::range(var_119, var_120);
            start_for_2:;
                if (iter_cmp(var_121) == 0) goto end_for_2;
                var_122 = wp::iter_next(var_121);
                // value = M_in[worldid, matrix_adr + j * (j + 1) // 2 + i]                       <L 1271>
                var_124 = wp::add(var_122, var_123);
                var_125 = wp::mul(var_122, var_124);
                var_127 = wp::floordiv(var_125, var_126);
                var_128 = wp::add(var_7, var_127);
                var_129 = wp::add(var_128, var_92);
                var_130 = wp::address(var_M_in, var_0, var_129);
                var_132 = wp::load(var_130);
                var_131 = wp::copy(var_132);
                // for k in range(i):                                                             <L 1272>
                // value -= L_out[worldid, factor_adr + k * size + i] * L_out[worldid, factor_adr + k * size + j]       <L 1273>
                var_134 = wp::mul(var_133, var_5);
                var_135 = wp::add(var_10, var_134);
                var_136 = wp::add(var_135, var_92);
                var_137 = wp::address(var_L_out, var_0, var_136);
                var_138 = wp::mul(var_133, var_5);
                var_139 = wp::add(var_10, var_138);
                var_140 = wp::add(var_139, var_122);
                var_141 = wp::address(var_L_out, var_0, var_140);
                var_143 = wp::load(var_137);
                var_144 = wp::load(var_141);
                var_142 = wp::mul(var_143, var_144);
                var_145 = wp::sub(var_131, var_142);
                // L_out[worldid, factor_adr + i * size + j] = value * diagonal_inv               <L 1274>
                var_146 = wp::mul(var_145, var_117);
                var_147 = wp::mul(var_92, var_5);
                var_148 = wp::add(var_10, var_147);
                var_149 = wp::add(var_148, var_122);
                wp::array_store(var_L_out, var_0, var_149, var_146);
                wp::assign(var_111, var_145);
                goto start_for_2;
            end_for_2:;
            // value = M_in[worldid, matrix_adr + i * (i + 1) // 2 + i]                           <L 1261>
            var_152 = wp::add(var_150, var_151);
            var_153 = wp::mul(var_150, var_152);
            var_155 = wp::floordiv(var_153, var_154);
            var_156 = wp::add(var_7, var_155);
            var_157 = wp::add(var_156, var_150);
            var_158 = wp::address(var_M_in, var_0, var_157);
            var_160 = wp::load(var_158);
            var_159 = wp::copy(var_160);
            // for k in range(i):                                                                 <L 1262>
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 1263>
            var_162 = wp::mul(var_161, var_5);
            var_163 = wp::add(var_10, var_162);
            var_164 = wp::add(var_163, var_150);
            var_165 = wp::address(var_L_out, var_0, var_164);
            var_167 = wp::load(var_165);
            var_166 = wp::copy(var_167);
            // value -= factor * factor                                                           <L 1264>
            var_168 = wp::mul(var_166, var_166);
            var_169 = wp::sub(var_159, var_168);
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 1263>
            var_171 = wp::mul(var_170, var_5);
            var_172 = wp::add(var_10, var_171);
            var_173 = wp::add(var_172, var_150);
            var_174 = wp::address(var_L_out, var_0, var_173);
            var_176 = wp::load(var_174);
            var_175 = wp::copy(var_176);
            // value -= factor * factor                                                           <L 1264>
            var_177 = wp::mul(var_175, var_175);
            var_178 = wp::sub(var_169, var_177);
            // diagonal_value = wp.sqrt(value)                                                    <L 1266>
            var_179 = wp::sqrt(var_178);
            // L_out[worldid, factor_adr + i * size + i] = diagonal_value                         <L 1267>
            var_180 = wp::mul(var_150, var_5);
            var_181 = wp::add(var_10, var_180);
            var_182 = wp::add(var_181, var_150);
            wp::array_store(var_L_out, var_0, var_182, var_179);
            // diagonal_inv = 1.0 / diagonal_value                                                <L 1268>
            var_184 = wp::div(var_183, var_179);
            // for j in range(i + 1, size):                                                       <L 1270>
            var_186 = wp::add(var_150, var_185);
            var_188 = wp::range(var_186, var_187);
            start_for_4:;
                if (iter_cmp(var_188) == 0) goto end_for_4;
                var_189 = wp::iter_next(var_188);
                // value = M_in[worldid, matrix_adr + j * (j + 1) // 2 + i]                       <L 1271>
                var_191 = wp::add(var_189, var_190);
                var_192 = wp::mul(var_189, var_191);
                var_194 = wp::floordiv(var_192, var_193);
                var_195 = wp::add(var_7, var_194);
                var_196 = wp::add(var_195, var_150);
                var_197 = wp::address(var_M_in, var_0, var_196);
                var_199 = wp::load(var_197);
                var_198 = wp::copy(var_199);
                // for k in range(i):                                                             <L 1272>
                // value -= L_out[worldid, factor_adr + k * size + i] * L_out[worldid, factor_adr + k * size + j]       <L 1273>
                var_201 = wp::mul(var_200, var_5);
                var_202 = wp::add(var_10, var_201);
                var_203 = wp::add(var_202, var_150);
                var_204 = wp::address(var_L_out, var_0, var_203);
                var_205 = wp::mul(var_200, var_5);
                var_206 = wp::add(var_10, var_205);
                var_207 = wp::add(var_206, var_189);
                var_208 = wp::address(var_L_out, var_0, var_207);
                var_210 = wp::load(var_204);
                var_211 = wp::load(var_208);
                var_209 = wp::mul(var_210, var_211);
                var_212 = wp::sub(var_198, var_209);
                var_214 = wp::mul(var_213, var_5);
                var_215 = wp::add(var_10, var_214);
                var_216 = wp::add(var_215, var_150);
                var_217 = wp::address(var_L_out, var_0, var_216);
                var_218 = wp::mul(var_213, var_5);
                var_219 = wp::add(var_10, var_218);
                var_220 = wp::add(var_219, var_189);
                var_221 = wp::address(var_L_out, var_0, var_220);
                var_223 = wp::load(var_217);
                var_224 = wp::load(var_221);
                var_222 = wp::mul(var_223, var_224);
                var_225 = wp::sub(var_212, var_222);
                // L_out[worldid, factor_adr + i * size + j] = value * diagonal_inv               <L 1274>
                var_226 = wp::mul(var_225, var_184);
                var_227 = wp::mul(var_150, var_5);
                var_228 = wp::add(var_10, var_227);
                var_229 = wp::add(var_228, var_189);
                wp::array_store(var_L_out, var_0, var_229, var_226);
                wp::assign(var_178, var_225);
                goto start_for_4;
            end_for_4:;
            // value = M_in[worldid, matrix_adr + i * (i + 1) // 2 + i]                           <L 1261>
            var_232 = wp::add(var_230, var_231);
            var_233 = wp::mul(var_230, var_232);
            var_235 = wp::floordiv(var_233, var_234);
            var_236 = wp::add(var_7, var_235);
            var_237 = wp::add(var_236, var_230);
            var_238 = wp::address(var_M_in, var_0, var_237);
            var_240 = wp::load(var_238);
            var_239 = wp::copy(var_240);
            // for k in range(i):                                                                 <L 1262>
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 1263>
            var_242 = wp::mul(var_241, var_5);
            var_243 = wp::add(var_10, var_242);
            var_244 = wp::add(var_243, var_230);
            var_245 = wp::address(var_L_out, var_0, var_244);
            var_247 = wp::load(var_245);
            var_246 = wp::copy(var_247);
            // value -= factor * factor                                                           <L 1264>
            var_248 = wp::mul(var_246, var_246);
            var_249 = wp::sub(var_239, var_248);
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 1263>
            var_251 = wp::mul(var_250, var_5);
            var_252 = wp::add(var_10, var_251);
            var_253 = wp::add(var_252, var_230);
            var_254 = wp::address(var_L_out, var_0, var_253);
            var_256 = wp::load(var_254);
            var_255 = wp::copy(var_256);
            // value -= factor * factor                                                           <L 1264>
            var_257 = wp::mul(var_255, var_255);
            var_258 = wp::sub(var_249, var_257);
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 1263>
            var_260 = wp::mul(var_259, var_5);
            var_261 = wp::add(var_10, var_260);
            var_262 = wp::add(var_261, var_230);
            var_263 = wp::address(var_L_out, var_0, var_262);
            var_265 = wp::load(var_263);
            var_264 = wp::copy(var_265);
            // value -= factor * factor                                                           <L 1264>
            var_266 = wp::mul(var_264, var_264);
            var_267 = wp::sub(var_258, var_266);
            // diagonal_value = wp.sqrt(value)                                                    <L 1266>
            var_268 = wp::sqrt(var_267);
            // L_out[worldid, factor_adr + i * size + i] = diagonal_value                         <L 1267>
            var_269 = wp::mul(var_230, var_5);
            var_270 = wp::add(var_10, var_269);
            var_271 = wp::add(var_270, var_230);
            wp::array_store(var_L_out, var_0, var_271, var_268);
            // diagonal_inv = 1.0 / diagonal_value                                                <L 1268>
            var_273 = wp::div(var_272, var_268);
            // for j in range(i + 1, size):                                                       <L 1270>
            var_275 = wp::add(var_230, var_274);
            var_277 = wp::range(var_275, var_276);
            start_for_6:;
                if (iter_cmp(var_277) == 0) goto end_for_6;
                var_278 = wp::iter_next(var_277);
                // value = M_in[worldid, matrix_adr + j * (j + 1) // 2 + i]                       <L 1271>
                var_280 = wp::add(var_278, var_279);
                var_281 = wp::mul(var_278, var_280);
                var_283 = wp::floordiv(var_281, var_282);
                var_284 = wp::add(var_7, var_283);
                var_285 = wp::add(var_284, var_230);
                var_286 = wp::address(var_M_in, var_0, var_285);
                var_288 = wp::load(var_286);
                var_287 = wp::copy(var_288);
                // for k in range(i):                                                             <L 1272>
                // value -= L_out[worldid, factor_adr + k * size + i] * L_out[worldid, factor_adr + k * size + j]       <L 1273>
                var_290 = wp::mul(var_289, var_5);
                var_291 = wp::add(var_10, var_290);
                var_292 = wp::add(var_291, var_230);
                var_293 = wp::address(var_L_out, var_0, var_292);
                var_294 = wp::mul(var_289, var_5);
                var_295 = wp::add(var_10, var_294);
                var_296 = wp::add(var_295, var_278);
                var_297 = wp::address(var_L_out, var_0, var_296);
                var_299 = wp::load(var_293);
                var_300 = wp::load(var_297);
                var_298 = wp::mul(var_299, var_300);
                var_301 = wp::sub(var_287, var_298);
                var_303 = wp::mul(var_302, var_5);
                var_304 = wp::add(var_10, var_303);
                var_305 = wp::add(var_304, var_230);
                var_306 = wp::address(var_L_out, var_0, var_305);
                var_307 = wp::mul(var_302, var_5);
                var_308 = wp::add(var_10, var_307);
                var_309 = wp::add(var_308, var_278);
                var_310 = wp::address(var_L_out, var_0, var_309);
                var_312 = wp::load(var_306);
                var_313 = wp::load(var_310);
                var_311 = wp::mul(var_312, var_313);
                var_314 = wp::sub(var_301, var_311);
                var_316 = wp::mul(var_315, var_5);
                var_317 = wp::add(var_10, var_316);
                var_318 = wp::add(var_317, var_230);
                var_319 = wp::address(var_L_out, var_0, var_318);
                var_320 = wp::mul(var_315, var_5);
                var_321 = wp::add(var_10, var_320);
                var_322 = wp::add(var_321, var_278);
                var_323 = wp::address(var_L_out, var_0, var_322);
                var_325 = wp::load(var_319);
                var_326 = wp::load(var_323);
                var_324 = wp::mul(var_325, var_326);
                var_327 = wp::sub(var_314, var_324);
                // L_out[worldid, factor_adr + i * size + j] = value * diagonal_inv               <L 1274>
                var_328 = wp::mul(var_327, var_273);
                var_329 = wp::mul(var_230, var_5);
                var_330 = wp::add(var_10, var_329);
                var_331 = wp::add(var_330, var_278);
                wp::array_store(var_L_out, var_0, var_331, var_328);
                wp::assign(var_267, var_327);
                goto start_for_6;
            end_for_6:;
            // value = M_in[worldid, matrix_adr + i * (i + 1) // 2 + i]                           <L 1261>
            var_334 = wp::add(var_332, var_333);
            var_335 = wp::mul(var_332, var_334);
            var_337 = wp::floordiv(var_335, var_336);
            var_338 = wp::add(var_7, var_337);
            var_339 = wp::add(var_338, var_332);
            var_340 = wp::address(var_M_in, var_0, var_339);
            var_342 = wp::load(var_340);
            var_341 = wp::copy(var_342);
            // for k in range(i):                                                                 <L 1262>
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 1263>
            var_344 = wp::mul(var_343, var_5);
            var_345 = wp::add(var_10, var_344);
            var_346 = wp::add(var_345, var_332);
            var_347 = wp::address(var_L_out, var_0, var_346);
            var_349 = wp::load(var_347);
            var_348 = wp::copy(var_349);
            // value -= factor * factor                                                           <L 1264>
            var_350 = wp::mul(var_348, var_348);
            var_351 = wp::sub(var_341, var_350);
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 1263>
            var_353 = wp::mul(var_352, var_5);
            var_354 = wp::add(var_10, var_353);
            var_355 = wp::add(var_354, var_332);
            var_356 = wp::address(var_L_out, var_0, var_355);
            var_358 = wp::load(var_356);
            var_357 = wp::copy(var_358);
            // value -= factor * factor                                                           <L 1264>
            var_359 = wp::mul(var_357, var_357);
            var_360 = wp::sub(var_351, var_359);
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 1263>
            var_362 = wp::mul(var_361, var_5);
            var_363 = wp::add(var_10, var_362);
            var_364 = wp::add(var_363, var_332);
            var_365 = wp::address(var_L_out, var_0, var_364);
            var_367 = wp::load(var_365);
            var_366 = wp::copy(var_367);
            // value -= factor * factor                                                           <L 1264>
            var_368 = wp::mul(var_366, var_366);
            var_369 = wp::sub(var_360, var_368);
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 1263>
            var_371 = wp::mul(var_370, var_5);
            var_372 = wp::add(var_10, var_371);
            var_373 = wp::add(var_372, var_332);
            var_374 = wp::address(var_L_out, var_0, var_373);
            var_376 = wp::load(var_374);
            var_375 = wp::copy(var_376);
            // value -= factor * factor                                                           <L 1264>
            var_377 = wp::mul(var_375, var_375);
            var_378 = wp::sub(var_369, var_377);
            // diagonal_value = wp.sqrt(value)                                                    <L 1266>
            var_379 = wp::sqrt(var_378);
            // L_out[worldid, factor_adr + i * size + i] = diagonal_value                         <L 1267>
            var_380 = wp::mul(var_332, var_5);
            var_381 = wp::add(var_10, var_380);
            var_382 = wp::add(var_381, var_332);
            wp::array_store(var_L_out, var_0, var_382, var_379);
            // diagonal_inv = 1.0 / diagonal_value                                                <L 1268>
            var_384 = wp::div(var_383, var_379);
            // for j in range(i + 1, size):                                                       <L 1270>
            var_386 = wp::add(var_332, var_385);
            var_388 = wp::range(var_386, var_387);
            start_for_8:;
                if (iter_cmp(var_388) == 0) goto end_for_8;
                var_389 = wp::iter_next(var_388);
                // value = M_in[worldid, matrix_adr + j * (j + 1) // 2 + i]                       <L 1271>
                var_391 = wp::add(var_389, var_390);
                var_392 = wp::mul(var_389, var_391);
                var_394 = wp::floordiv(var_392, var_393);
                var_395 = wp::add(var_7, var_394);
                var_396 = wp::add(var_395, var_332);
                var_397 = wp::address(var_M_in, var_0, var_396);
                var_399 = wp::load(var_397);
                var_398 = wp::copy(var_399);
                // for k in range(i):                                                             <L 1272>
                // value -= L_out[worldid, factor_adr + k * size + i] * L_out[worldid, factor_adr + k * size + j]       <L 1273>
                var_401 = wp::mul(var_400, var_5);
                var_402 = wp::add(var_10, var_401);
                var_403 = wp::add(var_402, var_332);
                var_404 = wp::address(var_L_out, var_0, var_403);
                var_405 = wp::mul(var_400, var_5);
                var_406 = wp::add(var_10, var_405);
                var_407 = wp::add(var_406, var_389);
                var_408 = wp::address(var_L_out, var_0, var_407);
                var_410 = wp::load(var_404);
                var_411 = wp::load(var_408);
                var_409 = wp::mul(var_410, var_411);
                var_412 = wp::sub(var_398, var_409);
                var_414 = wp::mul(var_413, var_5);
                var_415 = wp::add(var_10, var_414);
                var_416 = wp::add(var_415, var_332);
                var_417 = wp::address(var_L_out, var_0, var_416);
                var_418 = wp::mul(var_413, var_5);
                var_419 = wp::add(var_10, var_418);
                var_420 = wp::add(var_419, var_389);
                var_421 = wp::address(var_L_out, var_0, var_420);
                var_423 = wp::load(var_417);
                var_424 = wp::load(var_421);
                var_422 = wp::mul(var_423, var_424);
                var_425 = wp::sub(var_412, var_422);
                var_427 = wp::mul(var_426, var_5);
                var_428 = wp::add(var_10, var_427);
                var_429 = wp::add(var_428, var_332);
                var_430 = wp::address(var_L_out, var_0, var_429);
                var_431 = wp::mul(var_426, var_5);
                var_432 = wp::add(var_10, var_431);
                var_433 = wp::add(var_432, var_389);
                var_434 = wp::address(var_L_out, var_0, var_433);
                var_436 = wp::load(var_430);
                var_437 = wp::load(var_434);
                var_435 = wp::mul(var_436, var_437);
                var_438 = wp::sub(var_425, var_435);
                var_440 = wp::mul(var_439, var_5);
                var_441 = wp::add(var_10, var_440);
                var_442 = wp::add(var_441, var_332);
                var_443 = wp::address(var_L_out, var_0, var_442);
                var_444 = wp::mul(var_439, var_5);
                var_445 = wp::add(var_10, var_444);
                var_446 = wp::add(var_445, var_389);
                var_447 = wp::address(var_L_out, var_0, var_446);
                var_449 = wp::load(var_443);
                var_450 = wp::load(var_447);
                var_448 = wp::mul(var_449, var_450);
                var_451 = wp::sub(var_438, var_448);
                // L_out[worldid, factor_adr + i * size + j] = value * diagonal_inv               <L 1274>
                var_452 = wp::mul(var_451, var_384);
                var_453 = wp::mul(var_332, var_5);
                var_454 = wp::add(var_10, var_453);
                var_455 = wp::add(var_454, var_389);
                wp::array_store(var_L_out, var_0, var_455, var_452);
                wp::assign(var_378, var_451);
                goto start_for_8;
            end_for_8:;
            // value = M_in[worldid, matrix_adr + i * (i + 1) // 2 + i]                           <L 1261>
            var_458 = wp::add(var_456, var_457);
            var_459 = wp::mul(var_456, var_458);
            var_461 = wp::floordiv(var_459, var_460);
            var_462 = wp::add(var_7, var_461);
            var_463 = wp::add(var_462, var_456);
            var_464 = wp::address(var_M_in, var_0, var_463);
            var_466 = wp::load(var_464);
            var_465 = wp::copy(var_466);
            // for k in range(i):                                                                 <L 1262>
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 1263>
            var_468 = wp::mul(var_467, var_5);
            var_469 = wp::add(var_10, var_468);
            var_470 = wp::add(var_469, var_456);
            var_471 = wp::address(var_L_out, var_0, var_470);
            var_473 = wp::load(var_471);
            var_472 = wp::copy(var_473);
            // value -= factor * factor                                                           <L 1264>
            var_474 = wp::mul(var_472, var_472);
            var_475 = wp::sub(var_465, var_474);
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 1263>
            var_477 = wp::mul(var_476, var_5);
            var_478 = wp::add(var_10, var_477);
            var_479 = wp::add(var_478, var_456);
            var_480 = wp::address(var_L_out, var_0, var_479);
            var_482 = wp::load(var_480);
            var_481 = wp::copy(var_482);
            // value -= factor * factor                                                           <L 1264>
            var_483 = wp::mul(var_481, var_481);
            var_484 = wp::sub(var_475, var_483);
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 1263>
            var_486 = wp::mul(var_485, var_5);
            var_487 = wp::add(var_10, var_486);
            var_488 = wp::add(var_487, var_456);
            var_489 = wp::address(var_L_out, var_0, var_488);
            var_491 = wp::load(var_489);
            var_490 = wp::copy(var_491);
            // value -= factor * factor                                                           <L 1264>
            var_492 = wp::mul(var_490, var_490);
            var_493 = wp::sub(var_484, var_492);
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 1263>
            var_495 = wp::mul(var_494, var_5);
            var_496 = wp::add(var_10, var_495);
            var_497 = wp::add(var_496, var_456);
            var_498 = wp::address(var_L_out, var_0, var_497);
            var_500 = wp::load(var_498);
            var_499 = wp::copy(var_500);
            // value -= factor * factor                                                           <L 1264>
            var_501 = wp::mul(var_499, var_499);
            var_502 = wp::sub(var_493, var_501);
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 1263>
            var_504 = wp::mul(var_503, var_5);
            var_505 = wp::add(var_10, var_504);
            var_506 = wp::add(var_505, var_456);
            var_507 = wp::address(var_L_out, var_0, var_506);
            var_509 = wp::load(var_507);
            var_508 = wp::copy(var_509);
            // value -= factor * factor                                                           <L 1264>
            var_510 = wp::mul(var_508, var_508);
            var_511 = wp::sub(var_502, var_510);
            // diagonal_value = wp.sqrt(value)                                                    <L 1266>
            var_512 = wp::sqrt(var_511);
            // L_out[worldid, factor_adr + i * size + i] = diagonal_value                         <L 1267>
            var_513 = wp::mul(var_456, var_5);
            var_514 = wp::add(var_10, var_513);
            var_515 = wp::add(var_514, var_456);
            wp::array_store(var_L_out, var_0, var_515, var_512);
            // diagonal_inv = 1.0 / diagonal_value                                                <L 1268>
            var_517 = wp::div(var_516, var_512);
            // for j in range(i + 1, size):                                                       <L 1270>
            var_519 = wp::add(var_456, var_518);
            var_521 = wp::range(var_519, var_520);
            start_for_10:;
                if (iter_cmp(var_521) == 0) goto end_for_10;
                var_522 = wp::iter_next(var_521);
                // value = M_in[worldid, matrix_adr + j * (j + 1) // 2 + i]                       <L 1271>
                var_524 = wp::add(var_522, var_523);
                var_525 = wp::mul(var_522, var_524);
                var_527 = wp::floordiv(var_525, var_526);
                var_528 = wp::add(var_7, var_527);
                var_529 = wp::add(var_528, var_456);
                var_530 = wp::address(var_M_in, var_0, var_529);
                var_532 = wp::load(var_530);
                var_531 = wp::copy(var_532);
                // for k in range(i):                                                             <L 1272>
                // value -= L_out[worldid, factor_adr + k * size + i] * L_out[worldid, factor_adr + k * size + j]       <L 1273>
                var_534 = wp::mul(var_533, var_5);
                var_535 = wp::add(var_10, var_534);
                var_536 = wp::add(var_535, var_456);
                var_537 = wp::address(var_L_out, var_0, var_536);
                var_538 = wp::mul(var_533, var_5);
                var_539 = wp::add(var_10, var_538);
                var_540 = wp::add(var_539, var_522);
                var_541 = wp::address(var_L_out, var_0, var_540);
                var_543 = wp::load(var_537);
                var_544 = wp::load(var_541);
                var_542 = wp::mul(var_543, var_544);
                var_545 = wp::sub(var_531, var_542);
                var_547 = wp::mul(var_546, var_5);
                var_548 = wp::add(var_10, var_547);
                var_549 = wp::add(var_548, var_456);
                var_550 = wp::address(var_L_out, var_0, var_549);
                var_551 = wp::mul(var_546, var_5);
                var_552 = wp::add(var_10, var_551);
                var_553 = wp::add(var_552, var_522);
                var_554 = wp::address(var_L_out, var_0, var_553);
                var_556 = wp::load(var_550);
                var_557 = wp::load(var_554);
                var_555 = wp::mul(var_556, var_557);
                var_558 = wp::sub(var_545, var_555);
                var_560 = wp::mul(var_559, var_5);
                var_561 = wp::add(var_10, var_560);
                var_562 = wp::add(var_561, var_456);
                var_563 = wp::address(var_L_out, var_0, var_562);
                var_564 = wp::mul(var_559, var_5);
                var_565 = wp::add(var_10, var_564);
                var_566 = wp::add(var_565, var_522);
                var_567 = wp::address(var_L_out, var_0, var_566);
                var_569 = wp::load(var_563);
                var_570 = wp::load(var_567);
                var_568 = wp::mul(var_569, var_570);
                var_571 = wp::sub(var_558, var_568);
                var_573 = wp::mul(var_572, var_5);
                var_574 = wp::add(var_10, var_573);
                var_575 = wp::add(var_574, var_456);
                var_576 = wp::address(var_L_out, var_0, var_575);
                var_577 = wp::mul(var_572, var_5);
                var_578 = wp::add(var_10, var_577);
                var_579 = wp::add(var_578, var_522);
                var_580 = wp::address(var_L_out, var_0, var_579);
                var_582 = wp::load(var_576);
                var_583 = wp::load(var_580);
                var_581 = wp::mul(var_582, var_583);
                var_584 = wp::sub(var_571, var_581);
                var_586 = wp::mul(var_585, var_5);
                var_587 = wp::add(var_10, var_586);
                var_588 = wp::add(var_587, var_456);
                var_589 = wp::address(var_L_out, var_0, var_588);
                var_590 = wp::mul(var_585, var_5);
                var_591 = wp::add(var_10, var_590);
                var_592 = wp::add(var_591, var_522);
                var_593 = wp::address(var_L_out, var_0, var_592);
                var_595 = wp::load(var_589);
                var_596 = wp::load(var_593);
                var_594 = wp::mul(var_595, var_596);
                var_597 = wp::sub(var_584, var_594);
                // L_out[worldid, factor_adr + i * size + j] = value * diagonal_inv               <L 1274>
                var_598 = wp::mul(var_597, var_517);
                var_599 = wp::mul(var_456, var_5);
                var_600 = wp::add(var_10, var_599);
                var_601 = wp::add(var_600, var_522);
                wp::array_store(var_L_out, var_0, var_601, var_598);
                wp::assign(var_511, var_597);
                goto start_for_10;
            end_for_10:;
        }
        var_602 = wp::where(var_13, var_49, var_456);
    }
}

