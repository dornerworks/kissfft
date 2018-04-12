/**
 * \file run_ffts.c
 *
 * \brief Runs FFTs using FFTW and several forms of the KISS FFT library
 * \details See README
 *
 * \copyright Copyright (c) 2018, DornerWorks, Ltd. 
 * \license See LICENSE for full licensing and copying information
 */


/******************************************************************************
 *                                                                 Inclusions *
 ******************************************************************************/
#include "run_ffts.h"

#include "kiss_fft.h"
#include "kiss_fftr.h"
#include <fftw3.h>

#include <string.h>


/******************************************************************************
 *                                                                    Defines *
 ******************************************************************************/
#define NUM_RUNS (16384)
#define Q31_ONE (2147483648u)

#define FFTW_REAL (0)
#define FFTW_IMAG (1)
/* Per link, FFTW out buffer must be N/2+1
 * http://www.fftw.org/fftw3_doc/One_002dDimensional-DFTs-of-Real-Data.html */
#define FFTR_OUT_LEN_FFTW (FFTR_OUT_LEN + 1)
/* These flags were chosen to minimize FFTW runtime; to be fairest, planning
 * time shouldn't be counted... */
#define FFTW_PLAN_FFT   FFTW_MEASURE
#define FFTW_PLAN_FFTR  FFTW_ESTIMATE


/******************************************************************************
 *                                                                      Types *
 ******************************************************************************/

/* FFT -- SSE */
typedef kiss_fft_cpx_sse fft_input_sse_t[FFT_LEN];
typedef fft_input_sse_t *pfft_input_sse_t;
typedef kiss_fft_cpx_sse fft_output_sse_t[FFT_LEN];
typedef fft_output_sse_t *pfft_output_sse_t;

/* FFTR -- SSE */
typedef kiss_fft_scalar_sse fftr_input_sse_t[FFTR_LEN];
typedef fftr_input_sse_t *pfftr_input_sse_t;
typedef kiss_fft_cpx_sse fftr_output_sse_t[FFTR_OUT_LEN];
typedef fftr_output_sse_t *pfftr_output_sse_t;

/* FFT -- AVX */
typedef kiss_fft_cpx_avx fft_input_avx_t[FFT_LEN];
typedef fft_input_avx_t *pfft_input_avx_t;
typedef kiss_fft_cpx_avx fft_output_avx_t[FFT_LEN];
typedef fft_output_avx_t *pfft_output_avx_t;

/* FFTR -- AVX */
typedef kiss_fft_scalar_avx fftr_input_avx_t[FFTR_LEN];
typedef fftr_input_avx_t *pfftr_input_avx_t;
typedef kiss_fft_cpx_avx fftr_output_avx_t[FFTR_OUT_LEN];
typedef fftr_output_avx_t *pfftr_output_avx_t;


/******************************************************************************
 *                                                      Function declarations *
 ******************************************************************************/

/* SSE Transpositions */
void KissFftSsePreprocTranspose(pfft_input_t orig, pfft_input_sse_t transposed);
void KissFftSsePostprocTranspose(pfft_output_sse_t transposed, pfft_output_t final);
void KissFftrSsePreprocTranspose(pfftr_input_t orig, pfftr_input_sse_t transposed);
void KissFftrSsePostprocTranspose(pfftr_output_sse_t transposed, pfftr_output_t final);

/* AVX Transpositions */
void KissFftAvxPreprocTranspose(pfft_input_t orig, pfft_input_avx_t transposed);
void KissFftAvxPostprocTranspose(pfft_output_avx_t transposed, pfft_output_t final);
void KissFftrAvxPreprocTranspose(pfftr_input_t orig, pfftr_input_avx_t transposed);
void KissFftrAvxPostprocTranspose(pfftr_output_avx_t transposed, pfftr_output_t final);


/******************************************************************************
 *                                                         External functions *
 ******************************************************************************/

/* FFT */
pfft_output_t getOutputFftwFft(pfft_input_t input)
{
   static fft_output_t output;
   memset(&output, 0, sizeof(output));

   fftw_complex *in = fftw_malloc(sizeof(fftw_complex) * FFT_LEN);
   fftw_complex *out = fftw_malloc(sizeof(fftw_complex) * FFT_LEN);
   fftw_plan plan = fftw_plan_dft_1d(FFT_LEN, in, out, FFTW_FORWARD, FFTW_PLAN_FFT);

   /* translate input -- prescale to account for normalization */
   for (int32_t i = 0; i < FFT_LEN; i++)
   {
      in[i][FFTW_REAL] = (double)(*input)[i].r / (double)(FFT_LEN);
      in[i][FFTW_IMAG] = (double)(*input)[i].i / (double)(FFT_LEN);
   }

   for (int32_t i = 0; i < NUM_RUNS; i++)
   {
      fftw_execute(plan);
   }

   /* translate output */
   for (int32_t i = 0; i < FFT_LEN; i++)
   {
      output[i].r = (kiss_fft_scalar)out[i][FFTW_REAL];
      output[i].i = (kiss_fft_scalar)out[i][FFTW_IMAG];
   }

   fftw_destroy_plan(plan);
   fftw_free(in);
   fftw_free(out);

   return &output;
}

pfft_output_t getOutputKissFftPlain(pfft_input_t input)
{
   static fft_output_t output;
   memset(&output, 0, sizeof(output));

   kiss_fft_cfg cfg = kiss_fft_alloc_c(FFT_LEN, 0, NULL, NULL);

   for (int32_t i = 0; i < NUM_RUNS; i++)
   {
      kiss_fft_c(cfg, (kiss_fft_cpx *)input, output);
   }

   kiss_fft_free(cfg);

   return &output;
}

pfft_output_t getOutputKissFftSse(pfft_input_t input)
{
   SSE_ALIGNED fft_input_sse_t transposedInput;
   SSE_ALIGNED fft_output_sse_t transposedOutput;
   static SSE_ALIGNED fft_output_t output;
   memset(&output, 0, sizeof(output));

   kiss_fft_cfg cfg = kiss_fft_alloc_sse(FFT_LEN, 0, NULL, NULL);

   /* Does 4 independent FFTs in parallel */
   for (int32_t i = 0; i < NUM_RUNS / 4; i++)
   {
      KissFftSsePreprocTranspose(input, &transposedInput);
      kiss_fft_sse(cfg,
                   (kiss_fft_cpx_sse *)transposedInput,
                   (kiss_fft_cpx_sse *)transposedOutput);
      KissFftSsePostprocTranspose(&transposedOutput, &output);
   }

   kiss_fft_free(cfg);

   return &output;
}

pfft_output_t getOutputKissFftAvx(pfft_input_t input)
{
   AVX_ALIGNED fft_input_avx_t transposedInput;
   AVX_ALIGNED fft_output_avx_t transposedOutput;
   static AVX_ALIGNED fft_output_t output;
   memset(&output, 0, sizeof(output));

   kiss_fft_cfg cfg = kiss_fft_alloc_avx(FFT_LEN, 0, NULL, NULL);

   /* Does 8 independent FFTs in parallel */
   for (int32_t i = 0; i < NUM_RUNS / 8; i++)
   {
      KissFftAvxPreprocTranspose(input, &transposedInput);
      kiss_fft_avx(cfg,
                   (kiss_fft_cpx_avx *)transposedInput,
                   (kiss_fft_cpx_avx *)transposedOutput);
      KissFftAvxPostprocTranspose(&transposedOutput, &output);
   }

   kiss_fft_free(cfg);

   return &output;
}


/* FFTR */
pfftr_output_t getOutputFftwFftr(pfftr_input_t input)
{
   static fftr_output_t output;
   memset(&output, 0, sizeof(output));

   double *in = fftw_malloc(sizeof(double) * FFTR_LEN);
   fftw_complex *out = fftw_malloc(sizeof(fftw_complex) * FFTR_OUT_LEN_FFTW);
   fftw_plan plan = fftw_plan_dft_r2c_1d(FFTR_LEN, in, out, FFTW_PLAN_FFTR);

   /* translate input -- prescale to account for normalization */
   for (int32_t i = 0; i < FFTR_LEN; i++)
   {
      in[i] = (double)(*input)[i] / (double)(FFTR_LEN);
   }

   for (int32_t i = 0; i < NUM_RUNS; i++)
   {
      fftw_execute(plan);
   }

   /* translate output */
   for (int32_t i = 0; i < FFTR_OUT_LEN; i++)
   {
      output[i].r = (kiss_fft_scalar)out[i][FFTW_REAL];
      output[i].i = (kiss_fft_scalar)out[i][FFTW_IMAG];
   }

   fftw_destroy_plan(plan);
   fftw_free(in);
   fftw_free(out);

   return &output;
}

pfftr_output_t getOutputKissFftrPlain(pfftr_input_t input)
{
   static fftr_output_t output;
   memset(&output, 0, sizeof(output));

   kiss_fftr_cfg cfg = kiss_fftr_alloc_c(FFTR_LEN, 0, NULL, NULL);

   for (int32_t i = 0; i < NUM_RUNS; i++)
   {
      kiss_fftr_c(cfg, (kiss_fft_scalar *)input, output);
   }

   kiss_fftr_free(cfg);

   return &output;
}

pfftr_output_t getOutputKissFftrSse(pfftr_input_t input)
{
   SSE_ALIGNED fftr_input_sse_t transposedInput;
   SSE_ALIGNED fftr_output_sse_t transposedOutput;
   static SSE_ALIGNED fftr_output_t output;
   memset(&output, 0, sizeof(output));

   kiss_fftr_cfg cfg = kiss_fftr_alloc_sse(FFTR_LEN, 0, NULL, NULL);

   /* Does 4 independent FFTs in parallel */
   for (int32_t i = 0; i < NUM_RUNS / 4; i++)
   {
      KissFftrSsePreprocTranspose(input, &transposedInput);
      kiss_fftr_sse(cfg,
                    (kiss_fft_scalar_sse *)transposedInput,
                    (kiss_fft_cpx_sse *)transposedOutput);
      KissFftrSsePostprocTranspose(&transposedOutput, &output);
   }

   kiss_fftr_free(cfg);

   return &output;
}

pfftr_output_t getOutputKissFftrAvx(pfftr_input_t input)
{
   AVX_ALIGNED fftr_input_avx_t transposedInput;
   AVX_ALIGNED fftr_output_avx_t transposedOutput;
   static AVX_ALIGNED fftr_output_t output;
   memset(&output, 0, sizeof(output));

   kiss_fftr_cfg cfg = kiss_fftr_alloc_avx(FFTR_LEN, 0, NULL, NULL);

   /* Does 8 independent FFTs in parallel */
   for (int32_t i = 0; i < NUM_RUNS / 8; i++)
   {
      KissFftrAvxPreprocTranspose(input, &transposedInput);
      kiss_fftr_avx(cfg,
                    (kiss_fft_scalar_avx *)transposedInput,
                    (kiss_fft_cpx_avx *)transposedOutput);
      KissFftrAvxPostprocTranspose(&transposedOutput, &output);
   }

   kiss_fftr_free(cfg);

   return &output;
}


/******************************************************************************
 *                                                         Internal functions *
 ******************************************************************************/

/*
 * For transpose functions--note that these can be vectorized as well; they are kept 
 * in this form (1) for understandability and (2) because I don't want to spend the 
 * time optimizing a demo 
 *  
 * ALSO note that if SIMD usage is taken into account from the beginning, these kind 
 * of transforms may be unneccesary (though only one format would be supported).
 */ 

/* SSE */
/* FFT */
void KissFftSsePreprocTranspose(pfft_input_t orig, pfft_input_sse_t transposed)
{
   for (uint32_t run_index = 0; run_index < FFT_LEN; run_index++)
   {
      for (uint32_t fft_run = 0; fft_run < 4; fft_run++)
      {
         ((kiss_fft_scalar_c *)(&(*transposed)[run_index].r))[fft_run] = (*orig)[run_index].r;
         ((kiss_fft_scalar_c *)(&(*transposed)[run_index].i))[fft_run] = (*orig)[run_index].i;
      }
   }
}

void KissFftSsePostprocTranspose(pfft_output_sse_t transposed, pfft_output_t final)
{
   for (uint32_t run_index = 0; run_index < FFT_LEN; run_index++)
   {
      for (uint32_t fft_run = 0; fft_run < 4; fft_run++)
      {
         (*final)[run_index].r = ((kiss_fft_scalar_c *)(&(*transposed)[run_index].r))[fft_run];
         (*final)[run_index].i = ((kiss_fft_scalar_c *)(&(*transposed)[run_index].i))[fft_run];
      }
   }
}

/* FFTR */
void KissFftrSsePreprocTranspose(pfftr_input_t orig, pfftr_input_sse_t transposed)
{
   for (uint32_t run_index = 0; run_index < FFTR_LEN; run_index++)
   {
      for (uint32_t fft_run = 0; fft_run < 4; fft_run++)
      {
         ((kiss_fft_scalar_c *)(&(*transposed)[run_index]))[fft_run] = (*orig)[run_index];
      }
   }
}

void KissFftrSsePostprocTranspose(pfftr_output_sse_t transposed, pfftr_output_t final)
{
   for (uint32_t run_index = 0; run_index < FFTR_OUT_LEN; run_index++)
   {
      for (uint32_t fft_run = 0; fft_run < 4; fft_run++)
      {
         (*final)[run_index].r = ((kiss_fft_scalar_c *)(&(*transposed)[run_index].r))[fft_run];
         (*final)[run_index].i = ((kiss_fft_scalar_c *)(&(*transposed)[run_index].i))[fft_run];
      }
   }
}

/* AVX */
/* FFT */
void KissFftAvxPreprocTranspose(pfft_input_t orig, pfft_input_avx_t transposed)
{
   for (uint32_t run_index = 0; run_index < FFT_LEN; run_index++)
   {
      for (uint32_t fft_run = 0; fft_run < 8; fft_run++)
      {
         ((kiss_fft_scalar_c *)(&(*transposed)[run_index].r))[fft_run] = (*orig)[run_index].r;
         ((kiss_fft_scalar_c *)(&(*transposed)[run_index].i))[fft_run] = (*orig)[run_index].i;
      }
   }
}

void KissFftAvxPostprocTranspose(pfft_output_avx_t transposed, pfft_output_t final)
{
   for (uint32_t run_index = 0; run_index < FFT_LEN; run_index++)
   {
      for (uint32_t fft_run = 0; fft_run < 8; fft_run++)
      {
         (*final)[run_index].r = ((kiss_fft_scalar_c *)(&(*transposed)[run_index].r))[fft_run];
         (*final)[run_index].i = ((kiss_fft_scalar_c *)(&(*transposed)[run_index].i))[fft_run];
      }
   }
}

/* FFTR */
void KissFftrAvxPreprocTranspose(pfftr_input_t orig, pfftr_input_avx_t transposed)
{
   for (uint32_t run_index = 0; run_index < FFTR_LEN; run_index++)
   {
      for (uint32_t fft_run = 0; fft_run < 8; fft_run++)
      {
         ((kiss_fft_scalar_c *)(&(*transposed)[run_index]))[fft_run] = (*orig)[run_index];
      }
   }
}

void KissFftrAvxPostprocTranspose(pfftr_output_avx_t transposed, pfftr_output_t final)
{
   for (uint32_t run_index = 0; run_index < FFTR_OUT_LEN; run_index++)
   {
      for (uint32_t fft_run = 0; fft_run < 8; fft_run++)
      {
         (*final)[run_index].r = ((kiss_fft_scalar_c *)(&(*transposed)[run_index].r))[fft_run];
         (*final)[run_index].i = ((kiss_fft_scalar_c *)(&(*transposed)[run_index].i))[fft_run];
      }
   }
}

