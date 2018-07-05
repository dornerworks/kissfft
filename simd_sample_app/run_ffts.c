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
#define FFTW_REAL (0)
#define FFTW_IMAG (1)

#define FFTW_PLAN_FFT   FFTW_MEASURE
#define FFTW_PLAN_FFTR  FFTW_MEASURE


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
void KissFftSsePreprocTranspose(input_array_t orig, uint32_t array_index, pfft_input_sse_t transposed);
void KissFftSsePostprocTranspose(pfft_output_sse_t transposed, pfft_output_t final);
void KissFftrSsePreprocTranspose(inputr_array_t orig, uint32_t array_index, pfftr_input_sse_t transposed);
void KissFftrSsePostprocTranspose(pfftr_output_sse_t transposed, pfftr_output_t final);

/* AVX Transpositions */
void KissFftAvxPreprocTranspose(input_array_t orig, uint32_t array_index, pfft_input_avx_t transposed);
void KissFftAvxPostprocTranspose(pfft_output_avx_t transposed, pfft_output_t final);
void KissFftrAvxPreprocTranspose(inputr_array_t orig, uint32_t array_index, pfftr_input_avx_t transposed);
void KissFftrAvxPostprocTranspose(pfftr_output_avx_t transposed, pfftr_output_t final);

/******************************************************************************
 *                                                           Global Variables *
 ******************************************************************************/

static fftw_complex *in_fft;
static fftw_complex *out_fft;

static double *in_fftr;
static fftw_complex *out_fftr;

static fftw_plan fft_plan;
static fftw_plan fftr_plan;


/******************************************************************************
 *                                                         External functions *
 ******************************************************************************/

/* Prepping for FFTW; not counted in timing, for fairness */
void prepFftwFftPlan(void)
{
   in_fft = fftw_malloc(sizeof(fftw_complex) * FFT_LEN);
   out_fft = fftw_malloc(sizeof(fftw_complex) * FFT_LEN);

   fft_plan = fftw_plan_dft_1d(FFT_LEN, in_fft, out_fft, FFTW_FORWARD, FFTW_PLAN_FFT);
}

void prepFftwFftrPlan(void)
{
   in_fftr = fftw_malloc(sizeof(double) * FFTR_LEN);
   out_fftr = fftw_malloc(sizeof(fftw_complex) * FFTR_OUT_LEN_FFTW);

   fftr_plan = fftw_plan_dft_r2c_1d(FFT_LEN, in_fftr, out_fftr, FFTW_PLAN_FFTR);
}

void destroyFftPlan(void)
{
   fftw_destroy_plan(fft_plan);
   fftw_free(in_fft);
   fftw_free(out_fft);
}

void destroyFftrPlan(void)
{

   fftw_destroy_plan(fftr_plan);
   fftw_free(in_fftr);
   fftw_free(out_fftr);
}

/* FFT */
pfft_output_t getOutputFftwFft(input_array_t input)
{
   static fft_output_t output;
   memset(&output, 0, sizeof(output));

   for (int32_t i = 0; i < NUM_RUNS; i++)
   {
      /* The different number sets */
      for (uint32_t j = 0; j < NUM_FFT_SETS; j++)
      {
         /* translate input to floating point -- prescale to account for normalization */
         for (int32_t k = 0; k < FFT_LEN; k++)
         {
            in_fft[k][FFTW_REAL] = (double)input.input_array[j][k].r / (double)(FFT_LEN);
            in_fft[k][FFTW_IMAG] = (double)input.input_array[j][k].i / (double)(FFT_LEN);
         }

         fftw_execute(fft_plan);

         /* translate output */
         for (int32_t k = 0; k < FFT_LEN; k++)
         {
            output[k].r = (kiss_fft_scalar)out_fft[k][FFTW_REAL];
            output[k].i = (kiss_fft_scalar)out_fft[k][FFTW_IMAG];
         }
      }
   }

   /* returns the last number set of the last run as a representative dataset */
   return &output;
}

pfft_output_t getOutputKissFftPlain(input_array_t input)
{
   static fft_output_t output;
   memset(&output, 0, sizeof(output));

   kiss_fft_cfg cfg = kiss_fft_alloc_c(FFT_LEN, 0, NULL, NULL);

   for (int32_t i = 0; i < NUM_RUNS; i++)
   {
      for (uint32_t j = 0; j < NUM_FFT_SETS; j++)
      {
         kiss_fft_c(cfg, (kiss_fft_cpx *)input.input_array[j], output);
      }
   }
   kiss_fft_free(cfg);

   /* returns the last number set of the last run as a representative dataset */
   return &output;
}

pfft_output_t getOutputKissFftSse(input_array_t input)
{
   SSE_ALIGNED fft_input_sse_t transposedInput;
   SSE_ALIGNED fft_output_sse_t transposedOutput;
   static SSE_ALIGNED fft_output_t output;
   memset(&output, 0, sizeof(output));

   kiss_fft_cfg cfg = kiss_fft_alloc_sse(FFT_LEN, 0, NULL, NULL);

   for (int32_t i = 0; i < NUM_RUNS; i++)
   {
      /* Does 4 FFTs in parallel, so divided by 4 */
      for (uint32_t j = 0; j < NUM_FFT_SETS/4; j++)
      {
         KissFftSsePreprocTranspose(input, j*4, &transposedInput);
         kiss_fft_sse(cfg,
                      (kiss_fft_cpx_sse *)transposedInput,
                      (kiss_fft_cpx_sse *)transposedOutput);
         KissFftSsePostprocTranspose(&transposedOutput, &output);
      }
   }
   kiss_fft_free(cfg);

   /* returns the last number set of the last run as a representative dataset */
   return &output;
}

pfft_output_t getOutputKissFftAvx(input_array_t input)
{
   AVX_ALIGNED fft_input_avx_t transposedInput;
   AVX_ALIGNED fft_output_avx_t transposedOutput;
   static AVX_ALIGNED fft_output_t output;
   memset(&output, 0, sizeof(output));

   kiss_fft_cfg cfg = kiss_fft_alloc_avx(FFT_LEN, 0, NULL, NULL);

   for (int32_t i = 0; i < NUM_RUNS; i++)
   {
      /* Does 8 FFTs in parallel, so divided by 8 */
      for (uint32_t j = 0; j < NUM_FFT_SETS/8; j++)
      {
         KissFftAvxPreprocTranspose(input, j*8, &transposedInput);
         kiss_fft_avx(cfg,
                      (kiss_fft_cpx_avx *)transposedInput,
                      (kiss_fft_cpx_avx *)transposedOutput);
         KissFftAvxPostprocTranspose(&transposedOutput, &output);
      }
   }
   kiss_fft_free(cfg);

   /* returns the last number set of the last run as a representative dataset */
   return &output;
}


/* FFTR */
pfftr_output_t getOutputFftwFftr(inputr_array_t input)
{
   static fftr_output_t output;
   memset(&output, 0, sizeof(output));

   for (int32_t i = 0; i < NUM_RUNS; i++)
   {
      /* The different number sets */
      for (uint32_t j = 0; j < NUM_FFT_SETS; j++)
      {
         /* translate input to floating point -- prescale to account for normalization */
         for (int32_t k = 0; k < FFTR_LEN; k++)
         {
            in_fftr[k] = (double)input.input_array[j][k] / (double)(FFTR_LEN);
         }

         fftw_execute(fftr_plan);

         /* translate output */
         for (int32_t k = 0; k < FFTR_OUT_LEN; k++)
         {
            output[k].r = (kiss_fft_scalar)out_fftr[k][FFTW_REAL];
            output[k].i = (kiss_fft_scalar)out_fftr[k][FFTW_IMAG];
         }
      }
   }

   /* returns the last number set of the last run as a representative dataset */
   return &output;
}

pfftr_output_t getOutputKissFftrPlain(inputr_array_t input)
{
   static fftr_output_t output;
   memset(&output, 0, sizeof(output));

   kiss_fftr_cfg cfg = kiss_fftr_alloc_c(FFTR_LEN, 0, NULL, NULL);

   for (int32_t i = 0; i < NUM_RUNS; i++)
   {
      for (uint32_t j = 0; j < NUM_FFT_SETS; j++)
      {
         kiss_fftr_c(cfg, (kiss_fft_scalar *)input.input_array[j], output);
      }
   }
   kiss_fftr_free(cfg);

   /* returns the last number set of the last run as a representative dataset */
   return &output;
}

pfftr_output_t getOutputKissFftrSse(inputr_array_t input)
{
   SSE_ALIGNED fftr_input_sse_t transposedInput;
   SSE_ALIGNED fftr_output_sse_t transposedOutput;
   static SSE_ALIGNED fftr_output_t output;
   memset(&output, 0, sizeof(output));

   kiss_fftr_cfg cfg = kiss_fftr_alloc_sse(FFTR_LEN, 0, NULL, NULL);

   for (int32_t i = 0; i < NUM_RUNS; i++)
   {
      /* Does 4 FFTs in parallel, so divided by 4 */
      for (uint32_t j = 0; j < NUM_FFT_SETS/4; j++)
      {
         KissFftrSsePreprocTranspose(input, j*4, &transposedInput);
         kiss_fftr_sse(cfg,
                       (kiss_fft_scalar_sse *)transposedInput,
                       (kiss_fft_cpx_sse *)transposedOutput);
         KissFftrSsePostprocTranspose(&transposedOutput, &output);
      }
   }
   kiss_fftr_free(cfg);

   /* returns the last number set of the last run as a representative dataset */
   return &output;
}

pfftr_output_t getOutputKissFftrAvx(inputr_array_t input)
{
   AVX_ALIGNED fftr_input_avx_t transposedInput;
   AVX_ALIGNED fftr_output_avx_t transposedOutput;
   static AVX_ALIGNED fftr_output_t output;
   memset(&output, 0, sizeof(output));

   kiss_fftr_cfg cfg = kiss_fftr_alloc_avx(FFTR_LEN, 0, NULL, NULL);

   for (int32_t i = 0; i < NUM_RUNS; i++)
   {
      /* Does 8 FFTs in parallel, so divided by 8 */
      for (uint32_t j = 0; j < NUM_FFT_SETS/8; j++)
      {
         KissFftrAvxPreprocTranspose(input, j*8, &transposedInput);
         kiss_fftr_avx(cfg,
                       (kiss_fft_scalar_avx *)transposedInput,
                       (kiss_fft_cpx_avx *)transposedOutput);
         KissFftrAvxPostprocTranspose(&transposedOutput, &output);
      }
   }
   kiss_fftr_free(cfg);

   /* returns the last number set of the last run as a representative dataset */
   return &output;
}


/******************************************************************************
 *                                                         Internal functions *
 ******************************************************************************/

/*
 * Note that if SIMD usage is taken into account from the beginning, these kind
 * of transforms may be unneccesary (though only one format would be supported).
 */

/* SSE */
/* FFT */
void KissFftSsePreprocTranspose(input_array_t orig, uint32_t array_index, pfft_input_sse_t transposed)
{
   for (uint32_t run_index = 0; run_index < FFT_LEN; run_index++)
   {
      (*transposed)[run_index].r = _mm_set_epi32(orig.input_array[array_index][run_index].r,
                                             orig.input_array[array_index + 1][run_index].r,
                                             orig.input_array[array_index + 2][run_index].r,
                                             orig.input_array[array_index + 3][run_index].r);
      (*transposed)[run_index].i = _mm_set_epi32(orig.input_array[array_index][run_index].i,
                                             orig.input_array[array_index + 1][run_index].i,
                                             orig.input_array[array_index + 2][run_index].i,
                                             orig.input_array[array_index + 3][run_index].i);
   }
}

void KissFftSsePostprocTranspose(pfft_output_sse_t transposed, pfft_output_t final)
{
   for (uint32_t run_index = 0; run_index < FFT_LEN; run_index++)
   {
      /* Take the last value from the __m128i for comparison */
      (*final)[run_index].r = _mm_extract_epi32((*transposed)[run_index].r, 0);
      (*final)[run_index].i = _mm_extract_epi32((*transposed)[run_index].i, 0);
   }
}

/* FFTR */
void KissFftrSsePreprocTranspose(inputr_array_t orig, uint32_t array_index, pfftr_input_sse_t transposed)
{
   for (uint32_t run_index = 0; run_index < FFTR_LEN; run_index++)
   {
      (*transposed)[run_index] = _mm_set_epi32(orig.input_array[array_index][run_index],
                                           orig.input_array[array_index + 1][run_index],
                                           orig.input_array[array_index + 2][run_index],
                                           orig.input_array[array_index + 3][run_index]);
   }
}

void KissFftrSsePostprocTranspose(pfftr_output_sse_t transposed, pfftr_output_t final)
{
   for (uint32_t run_index = 0; run_index < FFTR_OUT_LEN; run_index++)
   {
      /* Take the last value from the __m128i for comparison */
      (*final)[run_index].r = _mm_extract_epi32((*transposed)[run_index].r, 0);
      (*final)[run_index].i = _mm_extract_epi32((*transposed)[run_index].i, 0);
   }
}

/* AVX */
/* FFT */
void KissFftAvxPreprocTranspose(input_array_t orig, uint32_t array_index, pfft_input_avx_t transposed)
{
   for (uint32_t run_index = 0; run_index < FFT_LEN; run_index++)
   {
      (*transposed)[run_index].r = _mm256_set_epi32(orig.input_array[array_index][run_index].r,
                                                orig.input_array[array_index + 1][run_index].r,
                                                orig.input_array[array_index + 2][run_index].r,
                                                orig.input_array[array_index + 3][run_index].r,
                                                orig.input_array[array_index + 4][run_index].r,
                                                orig.input_array[array_index + 5][run_index].r,
                                                orig.input_array[array_index + 6][run_index].r,
                                                orig.input_array[array_index + 7][run_index].r);
      (*transposed)[run_index].i = _mm256_set_epi32(orig.input_array[array_index][run_index].i,
                                                orig.input_array[array_index + 1][run_index].i,
                                                orig.input_array[array_index + 2][run_index].i,
                                                orig.input_array[array_index + 3][run_index].i,
                                                orig.input_array[array_index + 4][run_index].i,
                                                orig.input_array[array_index + 5][run_index].i,
                                                orig.input_array[array_index + 6][run_index].i,
                                                orig.input_array[array_index + 7][run_index].i);
   }
}

void KissFftAvxPostprocTranspose(pfft_output_avx_t transposed, pfft_output_t final)
{
   for (uint32_t run_index = 0; run_index < FFT_LEN; run_index++)
   {
      /* Take the last value from the __m256i for comparison */
      (*final)[run_index].r = _mm256_extract_epi32((*transposed)[run_index].r, 0);
      (*final)[run_index].i = _mm256_extract_epi32((*transposed)[run_index].i, 0);
   }
}

/* FFTR */
void KissFftrAvxPreprocTranspose(inputr_array_t orig, uint32_t array_index, pfftr_input_avx_t transposed)
{
   for (uint32_t run_index = 0; run_index < FFTR_LEN; run_index++)
   {
      (*transposed)[run_index] = _mm256_set_epi32(orig.input_array[array_index][run_index],
                                              orig.input_array[array_index + 1][run_index],
                                              orig.input_array[array_index + 2][run_index],
                                              orig.input_array[array_index + 3][run_index],
                                              orig.input_array[array_index + 4][run_index],
                                              orig.input_array[array_index + 5][run_index],
                                              orig.input_array[array_index + 6][run_index],
                                              orig.input_array[array_index + 7][run_index]);
   }
}

void KissFftrAvxPostprocTranspose(pfftr_output_avx_t transposed, pfftr_output_t final)
{
   for (uint32_t run_index = 0; run_index < FFTR_OUT_LEN; run_index++)
   {
      /* Take the last value from the __m256i for comparison */
      (*final)[run_index].r = _mm256_extract_epi32((*transposed)[run_index].r, 0);
      (*final)[run_index].i = _mm256_extract_epi32((*transposed)[run_index].i, 0);
   }
}
