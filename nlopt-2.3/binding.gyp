{'targets': [{
    'target_name': 'nlopt'
  , 'variables': {
        'nloptversion': '2.3'
    }
  , 'type': 'static_library'
		# Overcomes an issue with the linker and thin .a files on SmartOS
  , 'standalone_static_library': 1
  , 'defines': [
        'SNAPPY=1'
    ]
  , 'include_dirs': [
        './'
      , './stogo/'
      , './util/'
      , './direct/'
      , './cdirect/'
      , './praxis/'
      , './luksan/'
      , './crs/'
      , './mlsl/'
      , './mma/'
      , './cobyla/'
      , './newuoa/'
      , './neldermead/'
      , './auglag/'
      , './bobyqa/'
      , './isres/'
      , './slsqp/'
      , './api/'
    ]
  , 'sources': [
      './config.h',
      './direct/DIRect.c',
      './direct/direct_wrap.c',
      './direct/DIRserial.c',
      './direct/DIRsubrout.c',
      './direct/direct-internal.h',
      './direct/direct.h',
      './cdirect/cdirect.c',
      './cdirect/hybrid.c',
      './cdirect/cdirect.h',
      './praxis/praxis.c',
      './praxis/praxis.h',
      './luksan/plis.c',
      './luksan/plip.c',
      './luksan/pnet.c',
      './luksan/mssubs.c',
      './luksan/pssubs.c',
      './luksan/luksan.h',
      './crs/crs.c',
      './crs/crs.h',
      './mlsl/mlsl.c',
      './mlsl/mlsl.h',
      './mma/mma.c',
      './mma/mma.h',
      './mma/ccsa_quadratic.c',
      './cobyla/cobyla.c',
      './cobyla/cobyla.h',
      './newuoa/newuoa.c',
      './newuoa/newuoa.h',
      './',
      './neldermead/nldrmd.c',
      './neldermead/neldermead.h',
      './neldermead/sbplx.c',
      './auglag/auglag.c',
      './auglag/auglag.h',
      './bobyqa/bobyqa.c',
      './bobyqa/bobyqa.h',
      './isres/isres.c',
      './isres/isres.h',
      './slsqp/slsqp.c',
      './slsqp/slsqp.h',
      './api/general.c',
      './api/options.c',
      './api/optimize.c',
      './api/deprecated.c',
      './api/nlopt-internal.h',
      './api/nlopt.h',
      './api/f77api.c',
      './api/f77funcs.h',
      './api/f77funcs_.h',
      './api/nlopt.hpp',
      './api/nlopt-in.hpp',
      './util/mt19937ar.c',
      './util/sobolseq.c',
      './util/soboldata.h',
      './util/timer.c',
      './util/stop.c',
      './util/nlopt-util.h',
      './util/redblack.c',
      './util/redblack.h',
      './util/qsort_r.c',
      './util/rescale.c',
      './stogo/global.cc',
      './stogo/linalg.cc',
      './stogo/local.cc',
      './stogo/stogo.cc',
      './stogo/tools.cc',
      './stogo/global.h',
      './stogo/linalg.h',
      './stogo/local.h',
      './stogo/stogo_config.h',
      './stogo/stogo.h',
      './stogo/tools.h'
    ]
}]}