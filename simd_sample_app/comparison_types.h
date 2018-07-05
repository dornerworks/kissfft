/**
 * \file comparison_types.h
 *
 * \brief Common types for FFT comparisons
 * \details See README
 *
 * \copyright Copyright (c) 2018, DornerWorks, Ltd.
 * \license See LICENSE for full licensing and copying information
 */

#ifndef COMPARISON_TYPES_H
#define COMPARISON_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************
 *                                                                 Inclusions *
 ******************************************************************************/
#include "kiss_fft.h"

#include <stdint.h>


/******************************************************************************
 *                                                                    Defines *
 ******************************************************************************/
/* The number of runs to do for the timing test */
#define NUM_RUNS (8192)
/* The number of number sets to run FFT tests on.
 * This must be a multiple of 8 in order to not break everything */
#define NUM_FFT_SETS (8)

#define FFT_LEN (1024)
#define FFTR_LEN (1024)
#define FFTR_OUT_LEN (FFTR_LEN / 2)
/* Per link, FFTW out buffer must be N/2+1
 * http://www.fftw.org/fftw3_doc/One_002dDimensional-DFTs-of-Real-Data.html */
#define FFTR_OUT_LEN_FFTW (FFTR_OUT_LEN + 1)

#define NUM_FFT_COMPARISONS (3)
#define ALLOWABLE_ERROR (0)
#define MAX_RESULTS_NAME_LEN (16)

#define SSE_ALIGNED __attribute__((aligned(16)))
#define AVX_ALIGNED __attribute__((aligned(32)))
#define SIMD_ALIGNED AVX_ALIGNED


/******************************************************************************
 *                                                                      Types *
 ******************************************************************************/
typedef kiss_fft_cpx_c kiss_fft_cpx;
typedef kiss_fft_scalar_c kiss_fft_scalar;

/* FFT */
typedef kiss_fft_cpx fft_input_t[FFT_LEN];
typedef fft_input_t *pfft_input_t;
typedef kiss_fft_cpx fft_output_t[FFT_LEN];
typedef fft_output_t *pfft_output_t;

/* FFTR */
typedef kiss_fft_scalar fftr_input_t[FFTR_LEN];
typedef fftr_input_t *pfftr_input_t;
typedef kiss_fft_cpx fftr_output_t[FFTR_OUT_LEN];
typedef fftr_output_t *pfftr_output_t;

/* Other */
typedef enum verboseness_e
{
   NOT_VERBOSE = 0,
   VERBOSE
} verboseness_t;

typedef struct results_s
{
   union
   {
      pfft_output_t pfft;
      pfftr_output_t pfftr;
   } output;
   char name[MAX_RESULTS_NAME_LEN];
   uint32_t runtime_ns;
   verboseness_t verbose;
} results_t;

typedef struct input_array_s
{
   SIMD_ALIGNED fft_input_t input_array[NUM_FFT_SETS];
} input_array_t;

typedef struct inputr_array_s
{
   SIMD_ALIGNED fftr_input_t input_array[NUM_FFT_SETS];
} inputr_array_t;


/******************************************************************************
 *                                                                        EOF *
 ******************************************************************************/
#ifdef __cplusplus
}
#endif
#endif /* header guard */
