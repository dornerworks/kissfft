/**
 * \file run_ffts.h
 *
 * \brief Runs FFTs using FFTW and several forms of the KISS FFT library
 * \details See README
 *
 * \copyright Copyright (c) 2018, DornerWorks, Ltd.
 * \license See LICENSE for full licensing and copying information
 */

#ifndef RUN_FFTS_H
#define RUN_FFTS_H

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************
 *                                                                 Inclusions *
 ******************************************************************************/
#include "comparison_types.h"
#include <fftw3.h>


/******************************************************************************
 *                                                      Function declarations *
 ******************************************************************************/

/* FFT */
fftw_plan getFftwFftPlan(fftw_complex *in, fftw_complex *out);
fftw_plan getFftwFftrPlan(double *in, fftw_complex *out);
pfft_output_t getOutputFftwFft(pfft_input_t input, fftw_plan plan, fftw_complex *in, fftw_complex *out);
pfft_output_t getOutputKissFftPlain(pfft_input_t input);
pfft_output_t getOutputKissFftSse(pfft_input_t input);
pfft_output_t getOutputKissFftAvx(pfft_input_t input);

/* FFTR */
pfftr_output_t getOutputFftwFftr(pfftr_input_t input, fftw_plan plan, double *in, fftw_complex *out);
pfftr_output_t getOutputKissFftrPlain(pfftr_input_t input);
pfftr_output_t getOutputKissFftrSse(pfftr_input_t input);
pfftr_output_t getOutputKissFftrAvx(pfftr_input_t input);


/******************************************************************************
 *                                                                        EOF *
 ******************************************************************************/
#ifdef __cplusplus
}
#endif
#endif /* header guard */
