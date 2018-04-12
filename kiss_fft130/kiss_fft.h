#ifndef KISS_FFT_H
#define KISS_FFT_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 ATTENTION!
 If you would like a :
 -- a utility that will handle the caching of fft objects
 -- real-only (no imaginary time component ) FFT
 -- a multi-dimensional FFT
 -- a command-line utility to perform ffts
 -- a command-line utility to perform fast-convolution filtering

 Then see kfc.h kiss_fftr.h kiss_fftnd.h fftutil.c kiss_fastfir.c
  in the tools/ directory.
*/

#include <xmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#define kiss_fft_scalar_sse __m128i
#define kiss_fft_scalar_avx __m256i
#define KISS_FFT_MALLOC(nbytes) _mm_malloc(nbytes,32)
#define KISS_FFT_FREE _mm_free


#ifdef FIXED_POINT
#include <sys/types.h>
# if (FIXED_POINT == 32)
#  ifndef kiss_fft_scalar_c
#   define kiss_fft_scalar_c int32_t
#   define kiss_fft_uscalar uint32_t
#  endif
# else
#  define kiss_fft_scalar int16_t
#   define kiss_fft_uscalar uint16_t
# endif
#else
# ifndef kiss_fft_scalar
/*  default is float */
#   define kiss_fft_scalar float
# endif
#endif

typedef struct {
    kiss_fft_scalar_c r;
    kiss_fft_scalar_c i;
}kiss_fft_cpx_c;

typedef struct {
    kiss_fft_scalar_sse r;
    kiss_fft_scalar_sse i;
}kiss_fft_cpx_sse;

typedef struct {
    kiss_fft_scalar_avx r;
    kiss_fft_scalar_avx i;
}kiss_fft_cpx_avx;

typedef struct kiss_fft_state* kiss_fft_cfg;

/*
 *  kiss_fft_alloc
 *
 *  Initialize a FFT (or IFFT) algorithm's cfg/state buffer.
 *
 *  typical usage:      kiss_fft_cfg mycfg=kiss_fft_alloc(1024,0,NULL,NULL);
 *
 *  The return value from fft_alloc is a cfg buffer used internally
 *  by the fft routine or NULL.
 *
 *  If lenmem is NULL, then kiss_fft_alloc will allocate a cfg buffer using malloc.
 *  The returned value should be free()d when done to avoid memory leaks.
 *
 *  The state can be placed in a user supplied buffer 'mem':
 *  If lenmem is not NULL and mem is not NULL and *lenmem is large enough,
 *      then the function places the cfg in mem and the size used in *lenmem
 *      and returns mem.
 *
 *  If lenmem is not NULL and ( mem is NULL or *lenmem is not large enough),
 *      then the function returns NULL and places the minimum cfg
 *      buffer size in *lenmem.
 * */

kiss_fft_cfg kiss_fft_alloc_c(int nfft,int inverse_fft,void * mem,size_t * lenmem);
kiss_fft_cfg kiss_fft_alloc_sse(int nfft, int inverse_fft, void * mem, size_t * lenmem);
kiss_fft_cfg kiss_fft_alloc_avx(int nfft, int inverse_fft, void * mem, size_t * lenmem);

/*
 * kiss_fft(cfg,in_out_buf)
 *
 * Perform an FFT on a complex input buffer.
 * for a forward FFT,
 * fin should be  f[0] , f[1] , ... ,f[nfft-1]
 * fout will be   F[0] , F[1] , ... ,F[nfft-1]
 * Note that each element is complex and can be accessed like
    f[k].r and f[k].i
 * */
void kiss_fft_c(kiss_fft_cfg cfg,const kiss_fft_cpx_c *fin,kiss_fft_cpx_c *fout);
void kiss_fft_sse(kiss_fft_cfg cfg, const kiss_fft_cpx_sse *fin, kiss_fft_cpx_sse *fout);
void kiss_fft_avx(kiss_fft_cfg cfg, const kiss_fft_cpx_avx *fin, kiss_fft_cpx_avx *fout);

/*
 A more generic version of the above function. It reads its input from every Nth sample.
 * */
void kiss_fft_stride_c(kiss_fft_cfg cfg,const kiss_fft_cpx_c *fin,kiss_fft_cpx_c *fout,int fin_stride);
void kiss_fft_stride_sse(kiss_fft_cfg cfg, const kiss_fft_cpx_sse *fin, kiss_fft_cpx_sse *fout, int fin_stride);
void kiss_fft_stride_avx(kiss_fft_cfg cfg, const kiss_fft_cpx_avx *fin, kiss_fft_cpx_avx *fout, int fin_stride);

/* If kiss_fft_alloc allocated a buffer, it is one contiguous
   buffer and can be simply free()d when no longer needed*/
#define kiss_fft_free KISS_FFT_FREE

/*
 Cleans up some memory that gets managed internally. Not necessary to call, but it might clean up
 your compiler output to call this before you exit.
*/
void kiss_fft_cleanup_c(void);
void kiss_fft_cleanup_sse(void);
void kiss_fft_cleanup_avx(void);


/*
 * Returns the smallest integer k, such that k>=n and k has only "fast" factors (2,3,5)
 */
int kiss_fft_next_fast_size_c(int n);
int kiss_fft_next_fast_size_sse(int n);
int kiss_fft_next_fast_size_avx(int n);

/* for real ffts, we need an even size */
#define kiss_fftr_next_fast_size_real(n) \
        (kiss_fft_next_fast_size_c( ((n)+1)>>1)<<1)
#define kiss_fftr_next_fast_size_real_sse(n) \
        (kiss_fft_next_fast_size_sse( ((n)+1)>>1)<<1)
#define kiss_fftr_next_fast_size_real_avx(n) \
        (kiss_fft_next_fast_size_avx( ((n)+1)>>1)<<1)

#ifdef __cplusplus
}
#endif

#endif
