/*
Copyright (c) 2003-2010, Mark Borgerding

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * Neither the author nor the names of any contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/* kiss_fft.h
   defines kiss_fft_scalar as either short or a float type
   and defines
   typedef struct { kiss_fft_scalar r; kiss_fft_scalar i; }kiss_fft_cpx; */
#include "kiss_fft.h"
#include <limits.h>

#ifdef USE_SIMD
#if USE_SIMD == 4
#define kiss_fft_scalar kiss_fft_scalar_sse
#define kiss_fft_cpx kiss_fft_cpx_sse
#elif USE_SIMD == 8
#define kiss_fft_scalar kiss_fft_scalar_avx
#define kiss_fft_cpx kiss_fft_cpx_avx
#endif
#else
#define kiss_fft_scalar kiss_fft_scalar_c
#define kiss_fft_cpx kiss_fft_cpx_c
#endif


#define MAXFACTORS 32
/* e.g. an fft of length 128 has 4 factors
 as far as kissfft is concerned
 4*4*4*2
 */

struct kiss_fft_state{
    int nfft;
    int inverse;
    int factors[2*MAXFACTORS];
    kiss_fft_cpx twiddles[1];
};

/*
  Explanation of macros dealing with complex math:

   C_MUL(m,a,b)         : m = a*b
   C_FIXDIV( c , div )  : if a fixed point impl., c /= div. noop otherwise
   C_SUB( res, a,b)     : res = a - b
   C_SUBFROM( res , a)  : res -= a
   C_ADDTO( res , a)    : res += a
 * */
#ifdef FIXED_POINT
#if (FIXED_POINT==32)
# define FRACBITS 31
# define SAMPPROD int64_t
#define SAMP_MAX 2147483647
#else
# define FRACBITS 15
# define SAMPPROD int32_t
#define SAMP_MAX 32767
#endif

#define SAMP_MIN -SAMP_MAX

#if defined(CHECK_OVERFLOW)
#  define CHECK_OVERFLOW_OP(a,op,b)  \
	if ( (SAMPPROD)(a) op (SAMPPROD)(b) > SAMP_MAX || (SAMPPROD)(a) op (SAMPPROD)(b) < SAMP_MIN ) { \
		fprintf(stderr,"WARNING:overflow @ " __FILE__ "(%d): (%d " #op" %d) = %ld\n",__LINE__,(a),(b),(SAMPPROD)(a) op (SAMPPROD)(b) );  }
#endif


#ifdef USE_SIMD
#if USE_SIMD == 4
static inline __m128i s_mul_q31_v4si(__m128i a, __m128i b)
{
    __m128i tmpa, tmpb, tmpc;

    tmpa = _mm_srli_si128(a, 4);
    tmpb = _mm_srli_si128(b, 4);

    tmpc = _mm_mul_epi32(a, b);
    tmpa = _mm_mul_epi32(tmpa, tmpb);

    tmpc = _mm_srli_epi64(_mm_add_epi64(tmpc, _mm_set1_epi64x(1 << 30)), 31);
    tmpa = _mm_slli_epi64(_mm_add_epi64(tmpa, _mm_set1_epi64x(1 << 30)), 1);

    return _mm_blend_epi16(tmpc, tmpa, 204);
}
#endif
#if USE_SIMD == 8
static inline __m256i s_mul_q31_v8si(__m256i a, __m256i b)
{
    __m256i tmpa, tmpb, tmpc;

    tmpa = _mm256_srli_si256(a, 4);
    tmpb = _mm256_srli_si256(b, 4);

    tmpc = _mm256_mul_epi32(a, b);
    tmpa = _mm256_mul_epi32(tmpa, tmpb);

    tmpc = _mm256_srli_epi64(_mm256_add_epi64(tmpc, _mm256_set1_epi64x(1 << 30)), 31);
    tmpa = _mm256_slli_epi64(_mm256_add_epi64(tmpa, _mm256_set1_epi64x(1 << 30)), 1);

    return _mm256_blend_epi32(tmpc, tmpa, 170);
}
#endif
#if USE_SIMD == 4
#   define S_MUL(a,b) s_mul_q31_v4si(a, b)

#   define C_MUL(m,a,b) \
      do{ __m128i a2r, b2r, a2i, b2i;                                                         \
          __m128i tmpdiff1, tmpdiff2;                                                         \
          __m128i tmpsum1, tmpsum2;                                                           \
                                                                                              \
          a2r = _mm_srli_si128((a).r, 4);                                                     \
          b2r = _mm_srli_si128((b).r, 4);                                                     \
          a2i = _mm_srli_si128((a).i, 4);                                                     \
          b2i = _mm_srli_si128((b).i, 4);                                                     \
                                                                                              \
          tmpdiff1 = _mm_sub_epi64(_mm_mul_epi32((a).r, (b).r), _mm_mul_epi32((a).i, (b).i)); \
          tmpdiff2 = _mm_sub_epi64(_mm_mul_epi32(a2r, b2r), _mm_mul_epi32(a2i, b2i));         \
                                                                                              \
          tmpsum1 = _mm_add_epi64(_mm_mul_epi32((a).r, (b).i), _mm_mul_epi32((a).i, (b).r));  \
          tmpsum2 = _mm_add_epi64(_mm_mul_epi32(a2r, b2i), _mm_mul_epi32(a2i, b2r));          \
                                                                                              \
          tmpdiff1 = _mm_srli_epi64(_mm_add_epi64(tmpdiff1, _mm_set1_epi64x(1<<30)), 31);     \
          tmpdiff2 = _mm_slli_epi64(_mm_add_epi64(tmpdiff2, _mm_set1_epi64x(1<<30)), 1);      \
                                                                                              \
          tmpsum1 = _mm_srli_epi64(_mm_add_epi64(tmpsum1, _mm_set1_epi64x(1<<30)), 31);       \
          tmpsum2 = _mm_slli_epi64(_mm_add_epi64(tmpsum2, _mm_set1_epi64x(1<<30)), 1);        \
                                                                                              \
          (m).r = _mm_blend_epi16(tmpdiff1, tmpdiff2, 204);                                   \
          (m).i = _mm_blend_epi16(tmpsum1, tmpsum2, 204);                             }while(0)

#   define DIVSCALAR(x,k) \
	(x) = S_MUL( x, _mm_set1_epi32(SAMP_MAX/k) )

/* x = (x + 1 - (negative ? 1 : 0)) >> 1 */
#   define DIVSCALARBY2(x) \
        (x) = _mm_srai_epi32(_mm_add_epi32((x), _mm_sub_epi32(_mm_set1_epi32(1), _mm_srli_epi32((x), FRACBITS))), 1)

/* x = (x + 2 - (negative ? 1 : 0)) >> 2 */
#   define DIVSCALARBY4(x) \
        (x) = _mm_srai_epi32(_mm_add_epi32((x), _mm_sub_epi32(_mm_set1_epi32(2), _mm_srli_epi32((x), FRACBITS))), 2)

#elif USE_SIMD == 8
#   define S_MUL(a,b) s_mul_q31_v8si(a, b)

#   define C_MUL(m,a,b) \
      do{ __m256i a2r, b2r, a2i, b2i;                                                                  \
          __m256i tmpdiff1, tmpdiff2;                                                                  \
          __m256i tmpsum1, tmpsum2;                                                                    \
                                                                                                       \
          a2r = _mm256_srli_si256((a).r, 4);                                                           \
          b2r = _mm256_srli_si256((b).r, 4);                                                           \
          a2i = _mm256_srli_si256((a).i, 4);                                                           \
          b2i = _mm256_srli_si256((b).i, 4);                                                           \
                                                                                                       \
          tmpdiff1 = _mm256_sub_epi64(_mm256_mul_epi32((a).r, (b).r), _mm256_mul_epi32((a).i, (b).i)); \
          tmpdiff2 = _mm256_sub_epi64(_mm256_mul_epi32(a2r, b2r), _mm256_mul_epi32(a2i, b2i));         \
                                                                                                       \
          tmpsum1 = _mm256_add_epi64(_mm256_mul_epi32((a).r, (b).i), _mm256_mul_epi32((a).i, (b).r));  \
          tmpsum2 = _mm256_add_epi64(_mm256_mul_epi32(a2r, b2i), _mm256_mul_epi32(a2i, b2r));          \
                                                                                                       \
          tmpdiff1 = _mm256_srli_epi64(_mm256_add_epi64(tmpdiff1, _mm256_set1_epi64x(1<<30)), 31);     \
          tmpdiff2 = _mm256_slli_epi64(_mm256_add_epi64(tmpdiff2, _mm256_set1_epi64x(1<<30)), 1);      \
                                                                                                       \
          tmpsum1 = _mm256_srli_epi64(_mm256_add_epi64(tmpsum1, _mm256_set1_epi64x(1<<30)), 31);       \
          tmpsum2 = _mm256_slli_epi64(_mm256_add_epi64(tmpsum2, _mm256_set1_epi64x(1<<30)), 1);        \
                                                                                                       \
          (m).r = _mm256_blend_epi32(tmpdiff1, tmpdiff2, 170);                                         \
          (m).i = _mm256_blend_epi32(tmpsum1, tmpsum2, 170);                                   }while(0)

#   define DIVSCALAR(x,k) \
	(x) = S_MUL( x, _mm256_set1_epi32(SAMP_MAX/k) )

/* x = (x + 1 - (negative ? 1 : 0)) >> 1 */
#   define DIVSCALARBY2(x) \
        (x) = _mm256_srai_epi32(_mm256_add_epi32((x), _mm256_sub_epi32(_mm256_set1_epi32(1), _mm256_srli_epi32((x), FRACBITS))), 1)

/* x = (x + 2 - (negative ? 1 : 0)) >> 2 */
#   define DIVSCALARBY4(x) \
        (x) = _mm256_srai_epi32(_mm256_add_epi32((x), _mm256_sub_epi32(_mm256_set1_epi32(2), _mm256_srli_epi32((x), FRACBITS))), 2)
#endif

#else /* not USE_SIMD */
#   define smul(a,b) ( (SAMPPROD)(a)*(b) )
#   define sround( x )  (kiss_fft_scalar)( ( (x) + (1<<(FRACBITS-1)) ) >> FRACBITS )

#   define S_MUL(a,b) sround( smul(a,b) )

#   define C_MUL(m,a,b) \
      do{ (m).r = sround( smul((a).r,(b).r) - smul((a).i,(b).i) ); \
          (m).i = sround( smul((a).r,(b).i) + smul((a).i,(b).r) ); }while(0)

#   define DIVSCALAR(x,k) \
	(x) = sround( smul(  x, SAMP_MAX/k ) )

/* x = (x + 1 - (negative ? 1 : 0)) >> 1 */
#   define DIVSCALARBY2(x) \
        (x) = (((x) + 1 - (kiss_fft_scalar)((kiss_fft_uscalar)(x) >> FRACBITS)) >> 1)

/* x = (x + 2 - (negative ? 1 : 0)) >> 2 */
#   define DIVSCALARBY4(x) \
        (x) = (((x) + 2 - (kiss_fft_scalar)((kiss_fft_uscalar)(x) >> FRACBITS)) >> 2)
#endif /* USE_SIMD */

#   define C_FIXDIV(c,div) \
	do {    DIVSCALAR( (c).r , div);  \
		DIVSCALAR( (c).i  , div); }while (0)

#   define C_FIXDIVBY2(c) \
	do {    DIVSCALARBY2( (c).r );  \
		DIVSCALARBY2( (c).i ); }while (0)

#   define C_FIXDIVBY4(c) \
	do {    DIVSCALARBY4( (c).r );  \
		DIVSCALARBY4( (c).i ); }while (0)

#ifdef USE_SIMD
#if USE_SIMD == 4
#   define C_MULBYSCALAR( c, s ) \
    do{ (c).r =  s_mul_q31_v4si( (c).r , s ) ;\
        (c).i =  s_mul_q31_v4si( (c).i , s ) ; }while(0)
#elif USE_SIMD == 8
#   define C_MULBYSCALAR( c, s ) \
    do{ (c).r =  s_mul_q31_v8si( (c).r , s ) ;\
        (c).i =  s_mul_q31_v8si( (c).i , s ) ; }while(0)
#endif
#else /* not USE_SIMD */
#   define C_MULBYSCALAR( c, s ) \
    do{ (c).r =  sround( smul( (c).r , s ) ) ;\
        (c).i =  sround( smul( (c).i , s ) ) ; }while(0)
#endif /* USE_SIMD */

#else  /* not FIXED_POINT*/

#   define S_MUL(a,b) ( (a)*(b) )
#define C_MUL(m,a,b) \
    do{ (m).r = (a).r*(b).r - (a).i*(b).i;\
        (m).i = (a).r*(b).i + (a).i*(b).r; }while(0)
#   define C_FIXDIV(c,div) /* NOOP */
#   define C_MULBYSCALAR( c, s ) \
    do{ (c).r *= (s);\
        (c).i *= (s); }while(0)
#endif

#ifndef CHECK_OVERFLOW_OP
#  define CHECK_OVERFLOW_OP(a,op,b) /* noop */
#endif

#ifdef USE_SIMD
#if USE_SIMD == 4
#define S_ADD(a,b) _mm_add_epi32((a), (b))
#define S_SUB(a,b) _mm_sub_epi32((a), (b))
#define S_NEGATE(a) _mm_sub_epi32(_mm_setzero_si128(), (a))
#elif USE_SIMD == 8
#define S_ADD(a,b) _mm256_add_epi32((a), (b))
#define S_SUB(a,b) _mm256_sub_epi32((a), (b))
#define S_NEGATE(a) _mm256_sub_epi32(_mm256_setzero_si256(), (a))
#endif
#else
#define S_ADD(a,b) ((a) + (b))
#define S_SUB(a,b) ((a) - (b))
#define S_NEGATE(a) (-(a))
#endif

#define S_ADD3(a,b,c) S_ADD(S_ADD((a), (b)), (c))
#define S_ADDTO(res , a)\
    do { \
        (res) = S_ADD((res), (a)); \
    }while(0)
#define S_SUBFROM(res , a)\
    do { \
        (res) = S_SUB((res), (a)); \
    }while(0)
#define S_ADD2TO(res , a, b)\
    do { \
        (res) = S_ADD3((res), (a), (b)); \
    }while(0)

#ifdef USE_SIMD
#define C_ADD(res, a,b) \
    do { \
        (res).r = S_ADD((a).r, (b).r); \
        (res).i = S_ADD((a).i, (b).i); \
    }while(0)
#define C_SUB(res, a,b)\
    do { \
        (res).r = S_SUB((a).r, (b).r); \
        (res).i = S_SUB((a).i, (b).i); \
    }while(0)
#define C_ADDTO(res , a)\
    do { \
        (res).r = S_ADD((res).r, (a).r); \
        (res).i = S_ADD((res).i, (a).i); \
    }while(0)
#define C_SUBFROM(res , a)\
    do {\
        (res).r = S_SUB((res).r, (a).r); \
        (res).i = S_SUB((res).i, (a).i); \
    }while(0)

#else /* not USE_SIMD */

#define  C_ADD( res, a,b)\
    do { \
	    CHECK_OVERFLOW_OP((a).r,+,(b).r)\
	    CHECK_OVERFLOW_OP((a).i,+,(b).i)\
	    (res).r=(a).r+(b).r;  (res).i=(a).i+(b).i; \
    }while(0)
#define  C_SUB( res, a,b)\
    do { \
	    CHECK_OVERFLOW_OP((a).r,-,(b).r)\
	    CHECK_OVERFLOW_OP((a).i,-,(b).i)\
	    (res).r=(a).r-(b).r;  (res).i=(a).i-(b).i; \
    }while(0)
#define C_ADDTO( res , a)\
    do { \
	    CHECK_OVERFLOW_OP((res).r,+,(a).r)\
	    CHECK_OVERFLOW_OP((res).i,+,(a).i)\
	    (res).r += (a).r;  (res).i += (a).i;\
    }while(0)

#define C_SUBFROM( res , a)\
    do {\
	    CHECK_OVERFLOW_OP((res).r,-,(a).r)\
	    CHECK_OVERFLOW_OP((res).i,-,(a).i)\
	    (res).r -= (a).r;  (res).i -= (a).i; \
    }while(0)
#endif /* USE_SIMD */


#ifdef FIXED_POINT
#  define KISS_FFT_COS(phase)  floor(.5+SAMP_MAX * cos (phase))
#  define KISS_FFT_SIN(phase)  floor(.5+SAMP_MAX * sin (phase))
#ifdef USE_SIMD
#if USE_SIMD == 4
#  define HALF_OF(x) _mm_srai_epi32((x), 1)
#elif USE_SIMD == 8
#  define HALF_OF(x) _mm256_srai_epi32((x), 1)
#endif
#else
#  define HALF_OF(x) ((x)>>1)
#endif
#elif defined(USE_SIMD)
#  define KISS_FFT_COS(phase) _mm_set1_ps( cos(phase) )
#  define KISS_FFT_SIN(phase) _mm_set1_ps( sin(phase) )
#  define HALF_OF(x) ((x)*_mm_set1_ps(.5))
#else
#  define KISS_FFT_COS(phase) (kiss_fft_scalar) cos(phase)
#  define KISS_FFT_SIN(phase) (kiss_fft_scalar) sin(phase)
#  define HALF_OF(x) ((x)*.5)
#endif

#if defined(USE_SIMD) && defined(FIXED_POINT)
# if USE_SIMD == 4
#  define  kf_cexp(x,phase) \
    do{ \
        (x)->r = _mm_set1_epi32(KISS_FFT_COS(phase));\
        (x)->i = _mm_set1_epi32(KISS_FFT_SIN(phase));\
    }while(0)
# elif USE_SIMD == 8
#  define  kf_cexp(x,phase) \
    do{ \
        (x)->r = _mm256_set1_epi32(KISS_FFT_COS(phase));\
        (x)->i = _mm256_set1_epi32(KISS_FFT_SIN(phase));\
    }while(0)
# endif
#else
# define  kf_cexp(x,phase) \
	do{ \
		(x)->r = KISS_FFT_COS(phase);\
		(x)->i = KISS_FFT_SIN(phase);\
	}while(0)
#endif


/* a debugging function */
#define pcpx(c)\
    fprintf(stderr,"%g + %gi\n",(double)((c)->r),(double)((c)->i) )


#ifdef KISS_FFT_USE_ALLOCA
// define this to allow use of alloca instead of malloc for temporary buffers
// Temporary buffers are used in two case:
// 1. FFT sizes that have "bad" factors. i.e. not 2,3 and 5
// 2. "in-place" FFTs.  Notice the quotes, since kissfft does not really do an in-place transform.
#include <alloca.h>
#define  KISS_FFT_TMP_ALLOC(nbytes) alloca(nbytes)
#define  KISS_FFT_TMP_FREE(ptr)
#else
#define  KISS_FFT_TMP_ALLOC(nbytes) KISS_FFT_MALLOC(nbytes)
#define  KISS_FFT_TMP_FREE(ptr) KISS_FFT_FREE(ptr)
#endif
