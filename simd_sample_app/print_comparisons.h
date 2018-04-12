/**
 * \file print_comparisons.h
 *
 * \brief Compares and formats messages describing each FFT result
 * \details See README
 *
 * \copyright Copyright (c) 2018, DornerWorks, Ltd. 
 * \license See LICENSE for full licensing and copying information
 */

#ifndef PRINT_COMPARISONS_H
#define PRINT_COMPARISONS_H

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
void printFftComparisons(results_t master_results, results_t other_results[NUM_FFT_COMPARISONS]);
void printFftrComparisons(results_t master_results, results_t other_results[NUM_FFT_COMPARISONS]);


/******************************************************************************
 *                                                                        EOF *
 ******************************************************************************/
#ifdef __cplusplus
}
#endif
#endif /* header guard */

