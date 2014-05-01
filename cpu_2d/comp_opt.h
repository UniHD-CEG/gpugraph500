/*
 *  Optimisations that several compiler could apply.
 */


#if defined( __INTEL_COMPILER )
    #define assume_aligned(typ, var, alg) __assume_aligned(var, alg)
#elif defined( __GNUC__ )
    #if ((__GNUC__ > 4) ||(__GNUC__ == 4 && __GNUC_MINOR__ >= 7 ))
        #define assume_aligned(typ, var, alg) var = (typ) __builtin_assume_aligned(var, alg)
    #else
        #define assume_aligned(typ, var, alg)
    #endif
    #if !((__GNUC__ > 4) ||(__GNUC__ == 4 && __GNUC_MINOR__ >= 7 ))
        #define __restrict__
    #endif
#else
    #define assume_aligned(typ, var, alg)
#endif
