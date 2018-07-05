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


/******************************************************************************
 *                                                      Function declarations *
 ******************************************************************************/
/* FFTW stuff */
void prepFftwFftPlan(void);
void prepFftwFftrPlan(void);
void destroyFftPlan(void);
void destroyFftrPlan(void);

/* FFT */
pfft_output_t getOutputFftwFft(input_array_t input);
pfft_output_t getOutputKissFftPlain(input_array_t input);
pfft_output_t getOutputKissFftSse(input_array_t input);
pfft_output_t getOutputKissFftAvx(input_array_t input);

/* FFTR */
pfftr_output_t getOutputFftwFftr(inputr_array_t input);
pfftr_output_t getOutputKissFftrPlain(inputr_array_t input);
pfftr_output_t getOutputKissFftrSse(inputr_array_t input);
pfftr_output_t getOutputKissFftrAvx(inputr_array_t input);


/******************************************************************************
 *                                                                        EOF *
 ******************************************************************************/
#ifdef __cplusplus
}
#endif
#endif /* header guard */
