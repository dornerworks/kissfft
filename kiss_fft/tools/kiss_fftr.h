#ifndef KISS_FTR_H
#define KISS_FTR_H

#include "kiss_fft.h"
#ifdef __cplusplus
extern "C" {
#endif


/*

 Real optimized version can save about 45% cpu time vs. complex fft of a real seq.



 */

typedef struct kiss_fftr_state *kiss_fftr_cfg;


kiss_fftr_cfg kiss_fftr_alloc_c(int nfft,int inverse_fft,void * mem, size_t * lenmem);
kiss_fftr_cfg kiss_fftr_alloc_sse(int nfft, int inverse_fft, void * mem, size_t * lenmem);
kiss_fftr_cfg kiss_fftr_alloc_avx(int nfft, int inverse_fft, void * mem, size_t * lenmem);
/*
 nfft must be even

 If you don't care to allocate space, use mem = lenmem = NULL
*/


void kiss_fftr_c(kiss_fftr_cfg cfg,const kiss_fft_scalar_c *timedata,kiss_fft_cpx_c *freqdata);
void kiss_fftr_sse(kiss_fftr_cfg cfg, const kiss_fft_scalar_sse *timedata, kiss_fft_cpx_sse *freqdata);
void kiss_fftr_avx(kiss_fftr_cfg cfg, const kiss_fft_scalar_avx *timedata, kiss_fft_cpx_avx *freqdata);
/*
 input timedata has nfft scalar points
 output freqdata has nfft/2+1 complex points
*/

void kiss_fftri_c(kiss_fftr_cfg cfg,const kiss_fft_cpx_c *freqdata,kiss_fft_scalar_c *timedata);
void kiss_fftri_sse(kiss_fftr_cfg cfg, const kiss_fft_cpx_sse *freqdata, kiss_fft_scalar_sse *timedata);
void kiss_fftri_avx(kiss_fftr_cfg cfg, const kiss_fft_cpx_avx *freqdata, kiss_fft_scalar_avx *timedata);
/*
 input freqdata has  nfft/2+1 complex points
 output timedata has nfft scalar points
*/

#define kiss_fftr_free KISS_FFT_FREE

#ifdef __cplusplus
}
#endif
#endif
